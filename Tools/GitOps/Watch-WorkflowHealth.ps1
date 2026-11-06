#requires -Version 7.0
<#
.SYNOPSIS
Inspect workflow failures and expose unavailable GitHub metadata explicitly.
.DESCRIPTION
Workflow availability reflects validated run and workflow-list metadata separately
from billing. Unavailable billing retains unknown quota and null metrics. Successful
billing projects only finite JSON numeric scalars. The legacy billing endpoint is unchanged.
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
$billing = Get-GitOpsResponse -Operation billing -Arguments @('api', '/user/settings/billing/actions') -WorkingDirectory $RepoRoot -TimeoutSeconds $RequestTimeoutSeconds -Shape Object
$inspections.Add($billing)
if ($billing.status -eq 'Available') {
    $used = $billing.value.total_minutes_used
    $included = $billing.value.included_minutes
    $usedIsNumber = $used -is [int] -or $used -is [long] -or $used -is [double]
    $includedIsNumber = $included -is [int] -or $included -is [long] -or $included -is [double]
    # JSON strings, Booleans and containers must not coerce into known quota values.
    if (-not $usedIsNumber -or -not $includedIsNumber -or
        [double]::IsNaN([double]$used) -or [double]::IsInfinity([double]$used) -or
        [double]::IsNaN([double]$included) -or [double]::IsInfinity([double]$included)) {
        $billing.status = 'Unavailable'; $billing.reason = 'Billing metadata is incomplete.'
    }
    elseif ($billing.value.included_minutes -gt 0 -and $billing.value.total_minutes_used -ge $billing.value.included_minutes) { $problems.Add("Actions minutes exhausted: $($billing.value.total_minutes_used)/$($billing.value.included_minutes)") }
}
$workflowAvailable = $runs.status -eq 'Available' -and $workflows.status -eq 'Available'
$available = @($inspections | Where-Object status -EQ Unavailable).Count -eq 0
$snap = [ordered]@{
    repo = $Repo; checked_utc = [datetime]::UtcNow.ToString('o'); inspection_status = if ($available) { 'Available' } else { 'Unavailable' }
    workflow_inspection_status = if ($workflowAvailable) { 'Available' } else { 'Unavailable' }
    problem_count = $problems.Count; problems = @($problems)
    last_runs   = if ($runs.status -eq 'Available') { @($runs.value | Select-Object -First 5 | ForEach-Object { @{ wf = $_.workflowName; status = $_.status; conclusion = $_.conclusion } }) } else { @() }
    billing     = if ($billing.status -eq 'Available') { @{ status = $billing.status; used = $billing.value.total_minutes_used; included = $billing.value.included_minutes } } else { @{ status = $billing.status; quota_status = 'Unknown'; used = $null; included = $null } }
    inspections = @($inspections | Select-Object operation, status, exit_code, reason)
}
Write-GitOpsReport -Report $snap -Directory $OutputDirectory -Name ('workflow-health-' + $Repo.Replace('/', '_') + '.json')
if (-not $Quiet) { "workflow-health $Repo : inspection=$($snap.inspection_status) problems=$($problems.Count)" }
$snap | ConvertTo-Json -Depth 8 -Compress
exit $(if (-not $available) { 2 } elseif ($problems.Count -gt 0) { 1 } else { 0 })
