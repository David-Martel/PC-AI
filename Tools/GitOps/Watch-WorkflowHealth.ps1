#requires -Version 7.0
<#
.SYNOPSIS
Inspect workflow failures and expose unavailable GitHub metadata explicitly.
.DESCRIPTION
Inspect current-month consolidated usage for the repository owner. Billing access
or schema failures remain unavailable; workflow availability is also reported
separately. Usage does not expose an Actions allowance, so quota status is Unknown.
#>
[CmdletBinding()]
param([string]$Repo, [ValidateRange(1, 100)][int]$Limit = 15, [switch]$Quiet,
    [string]$RepoRoot = (Join-Path $PSScriptRoot '../..'), [string]$OutputDirectory,
    [ValidateRange(1, 120)][int]$RequestTimeoutSeconds = 15)
. (Join-Path $PSScriptRoot 'Invoke-GitOpsMonitors.ps1') -LibraryOnly -Repo $Repo -RepoRoot $RepoRoot -OutputDirectory $OutputDirectory -RequestTimeoutSeconds $RequestTimeoutSeconds
$ErrorActionPreference = 'Stop'
$RepoRoot = (Resolve-Path -LiteralPath $RepoRoot).Path
if (-not $OutputDirectory) { $OutputDirectory = Join-Path $RepoRoot 'Reports/gitops' }
if (-not $Repo) {
    $identity = Get-GitOpsResponse -Operation repository -Arguments @('repo', 'view', '--json', 'nameWithOwner') -WorkingDirectory $RepoRoot -TimeoutSeconds $RequestTimeoutSeconds -Shape Object
    if ($identity.status -eq 'Available') { $Repo = $identity.value.nameWithOwner }
}
if ($Repo -notmatch '^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$') { Write-Warning 'Repository identity is unavailable or invalid.'; exit 2 }
$problems = [Collections.Generic.List[string]]::new()
$inspections = [Collections.Generic.List[object]]::new()
$runs = Get-GitOpsResponse -Operation runs -Arguments @('run', 'list', '-R', $Repo, '--limit', "$Limit", '--json', 'databaseId,status,conclusion,workflowName,event,createdAt') -WorkingDirectory $RepoRoot -TimeoutSeconds $RequestTimeoutSeconds -Shape Array
$inspections.Add($runs)
if ($runs.status -eq 'Available') {
    foreach ($run in $runs.value) {
        if (-not $run.status -or -not $run.workflowName -or
            ($run.status -eq 'completed' -and [string]::IsNullOrWhiteSpace($run.conclusion))) {
            $runs.status = 'Unavailable'; $runs.reason = 'Run metadata is incomplete.'; break
        }
        if ($run.conclusion -in @('startup_failure', 'failure', 'action_required')) { $problems.Add("$($run.conclusion): $($run.workflowName) (run $($run.databaseId))") }
    }
}
$workflows = Get-GitOpsResponse -Operation workflows -Arguments @('api', "repos/$Repo/actions/workflows") -WorkingDirectory $RepoRoot -TimeoutSeconds $RequestTimeoutSeconds -Shape Object
$inspections.Add($workflows)
if ($workflows.status -eq 'Available') {
    if ($workflows.value.workflows -isnot [array]) { $workflows.status = 'Unavailable'; $workflows.reason = 'Workflow metadata is incomplete.' }
    else {
        foreach ($workflow in $workflows.value.workflows) {
            if (-not $workflow.state -or -not $workflow.name) { $workflows.status = 'Unavailable'; $workflows.reason = 'Workflow metadata is incomplete.'; break }
            if ($workflow.state -like 'disabled*') { $problems.Add("workflow disabled: $($workflow.name)") }
        }
    }
}
$billingMonth = [datetime]::UtcNow
$repoOwner = $Repo.Split('/')[0]
# The retired product billing API exposed an allowance; consolidated usage does not.
# Resolve the repository's billing owner without assuming it is the authenticated user.
$billingOwner = Get-GitOpsResponse -Operation billing-owner -Arguments @('api', "users/$repoOwner") -WorkingDirectory $RepoRoot -TimeoutSeconds $RequestTimeoutSeconds -Shape Object
$inspections.Add($billingOwner)
$billing = [pscustomobject]@{ operation = 'billing'; status = 'Unavailable'; exit_code = $null; reason = 'Billing owner is unavailable or invalid.'; value = $null }
if ($billingOwner.status -eq 'Available') {
    $ownerNames = @($billingOwner.value.PSObject.Properties.Name)
    if ($ownerNames -cnotcontains 'login' -or $ownerNames -cnotcontains 'type' -or
        $billingOwner.value.login -isnot [string] -or $billingOwner.value.login -ine $repoOwner -or
        $billingOwner.value.type -isnot [string] -or $billingOwner.value.type -cnotin @('User', 'Organization')) {
        $billingOwner.status = 'Unavailable'; $billingOwner.reason = 'Billing owner metadata is incomplete or invalid.'
    }
    else {
        $ownerRoute = if ($billingOwner.value.type -ceq 'Organization') { 'organizations' } else { 'users' }
        $endpoint = "$ownerRoute/$repoOwner/settings/billing/usage?year=$($billingMonth.Year)&month=$($billingMonth.Month)"
        $billing = Get-GitOpsResponse -Operation billing -Arguments @('api', $endpoint) -WorkingDirectory $RepoRoot -TimeoutSeconds $RequestTimeoutSeconds -Shape Object
        if ($billing.status -eq 'Available') {
            # Validate the schema without publishing private quantities or charges.
            try {
                $names = @($billing.value.PSObject.Properties.Name)
                if ($names -cnotcontains 'usageItems' -or $billing.value.usageItems -isnot [array]) { throw 'Invalid usage array.' }
                if ($names -ccontains 'timePeriod') {
                    $period = $billing.value.timePeriod
                    if ($period -isnot [pscustomobject] -or @($period.PSObject.Properties.Name) -cnotcontains 'year' -or
                        @($period.PSObject.Properties.Name) -cnotcontains 'month' -or
                        ($period.year -isnot [long] -and $period.year -isnot [int]) -or
                        ($period.month -isnot [long] -and $period.month -isnot [int]) -or
                        $period.year -ne $billingMonth.Year -or $period.month -ne $billingMonth.Month) { throw 'Invalid report period.' }
                }
                foreach ($item in $billing.value.usageItems) {
                    if ($item -isnot [pscustomobject]) { throw 'Invalid usage item.' }
                    $itemNames = @($item.PSObject.Properties.Name)
                    foreach ($field in @('date', 'product', 'sku', 'unitType')) {
                        if ($itemNames -cnotcontains $field -or $item.$field -isnot [string] -or [string]::IsNullOrWhiteSpace($item.$field)) { throw 'Invalid usage text.' }
                    }
                    $usageDate = [datetime]::MinValue
                    if (-not [datetime]::TryParseExact($item.date, 'yyyy-MM-dd', [Globalization.CultureInfo]::InvariantCulture,
                        [Globalization.DateTimeStyles]::None, [ref]$usageDate) -or
                        $usageDate.Year -ne $billingMonth.Year -or $usageDate.Month -ne $billingMonth.Month) { throw 'Invalid usage date.' }
                    foreach ($field in @('quantity', 'pricePerUnit', 'grossAmount', 'discountAmount', 'netAmount')) {
                        if ($itemNames -cnotcontains $field -or ($item.$field -isnot [long] -and $item.$field -isnot [int] -and
                            $item.$field -isnot [double] -and $item.$field -isnot [decimal])) { throw 'Invalid usage number.' }
                        $number = [double]$item.$field
                        if ([double]::IsNaN($number) -or [double]::IsInfinity($number)) { throw 'Nonfinite usage number.' }
                    }
                }
            }
            catch { $billing.status = 'Unavailable'; $billing.reason = 'Billing usage metadata is incomplete or invalid.' }
        }
    }
}
$inspections.Add($billing)
$available = @($inspections | Where-Object status -EQ Unavailable).Count -eq 0
$snap = [ordered]@{
    repo = $Repo; checked_utc = [datetime]::UtcNow.ToString('o'); inspection_status = if ($available) { 'Available' } else { 'Unavailable' }
    problem_count = $problems.Count; problems = @($problems)
    last_runs   = if ($runs.status -eq 'Available') { @($runs.value | Select-Object -First 5 | ForEach-Object { @{ wf = $_.workflowName; status = $_.status; conclusion = $_.conclusion } }) } else { @() }
    workflow_inspection_status = if ($runs.status -eq 'Available' -and $workflows.status -eq 'Available') { 'Available' } else { 'Unavailable' }
    billing     = @{ status = $billing.status; year = $billingMonth.Year; month = $billingMonth.Month; quota_status = 'Unknown' }
    inspections = @($inspections | Select-Object operation, status, exit_code, reason)
}
Write-GitOpsReport -Report $snap -Directory $OutputDirectory -Name ('workflow-health-' + $Repo.Replace('/', '_') + '.json')
if (-not $Quiet) { "workflow-health $Repo : inspection=$($snap.inspection_status) problems=$($problems.Count)" }
$snap | ConvertTo-Json -Depth 8 -Compress
exit $(if (-not $available) { 2 } elseif ($problems.Count -gt 0) { 1 } else { 0 })
