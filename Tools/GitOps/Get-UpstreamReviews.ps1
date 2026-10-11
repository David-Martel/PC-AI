#requires -Version 7.0
<#
.SYNOPSIS
Collect upstream findings while retaining per-source inspection failures.
#>
[CmdletBinding()]
param([string]$Repo, [switch]$Quiet,
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
$items = [Collections.Generic.List[object]]::new()
$inspections = [Collections.Generic.List[object]]::new()
$dependabot = Get-GitOpsResponse -Operation dependabot -Arguments @('api', "repos/$Repo/dependabot/alerts?state=open&per_page=100", '--paginate', '--slurp') -WorkingDirectory $RepoRoot -TimeoutSeconds $RequestTimeoutSeconds -Shape Array
$inspections.Add($dependabot)
if ($dependabot.status -eq 'Available') {
    foreach ($page in $dependabot.value) {
        if ($page -isnot [array]) { $dependabot.status = 'Unavailable'; $dependabot.reason = 'Dependabot page metadata is incomplete.'; break }
        foreach ($alert in $page) {
            if (-not $alert.security_advisory.summary -or -not $alert.html_url) { $dependabot.status = 'Unavailable'; $dependabot.reason = 'Dependabot metadata is incomplete.'; break }
            $items.Add(@{ source = 'dependabot'; severity = $alert.security_advisory.severity; title = $alert.security_advisory.summary; ref = $alert.html_url })
        }
    }
}
$scanning = Get-GitOpsResponse -Operation code-scanning -Arguments @('api', "repos/$Repo/code-scanning/alerts?state=open&per_page=100") -WorkingDirectory $RepoRoot -TimeoutSeconds $RequestTimeoutSeconds -Shape Array
$inspections.Add($scanning)
if ($scanning.status -eq 'Available') {
    foreach ($alert in $scanning.value) {
        if (-not $alert.rule -or -not $alert.html_url) { $scanning.status = 'Unavailable'; $scanning.reason = 'Code scanning metadata is incomplete.'; break }
        $items.Add(@{ source = 'code-scanning'; severity = $alert.rule.security_severity_level ?? $alert.rule.severity; title = $alert.rule.description; ref = $alert.html_url })
    }
}
$pulls = Get-GitOpsResponse -Operation pulls -Arguments @('api', "repos/$Repo/pulls?state=open&per_page=50") -WorkingDirectory $RepoRoot -TimeoutSeconds $RequestTimeoutSeconds -Shape Array
$inspections.Add($pulls)
if ($pulls.status -eq 'Available') {
    foreach ($pull in $pulls.value) {
        if (($pull.number -isnot [long] -and $pull.number -isnot [int]) -or $pull.number -le 0) { $pulls.status = 'Unavailable'; $pulls.reason = 'Pull request metadata is incomplete.'; break }
        $reviews = Get-GitOpsResponse -Operation "reviews:$($pull.number)" -Arguments @('api', "repos/$Repo/pulls/$($pull.number)/reviews") -WorkingDirectory $RepoRoot -TimeoutSeconds $RequestTimeoutSeconds -Shape Array
        $inspections.Add($reviews)
        if ($reviews.status -eq 'Available') {
            foreach ($review in $reviews.value) {
                if (-not $review.user.login -or -not $review.state) { $reviews.status = 'Unavailable'; $reviews.reason = 'Review metadata is incomplete.'; break }
                if ($review.user.login -match 'copilot|gemini|jules|bot') { $items.Add(@{ source = "pr-review:$($review.user.login)"; severity = 'review'; title = "PR #$($pull.number) $($review.state)"; ref = $review.html_url }) }
            }
        }
    }
}
$available = @($inspections | Where-Object status -EQ Unavailable).Count -eq 0
$bySeverity = $items | Group-Object { $_.severity } | ForEach-Object { "$($_.Name)=$($_.Count)" }
$snap = [ordered]@{ repo = $Repo; checked_utc = [datetime]::UtcNow.ToString('o'); inspection_status = if ($available) { 'Available' } else { 'Unavailable' }; total = $items.Count; by_severity = $bySeverity -join ' '; items = @($items); inspections = @($inspections | Select-Object operation, status, exit_code, reason) }
Write-GitOpsReport -Report $snap -Directory $OutputDirectory -Name ('upstream-reviews-' + $Repo.Replace('/', '_') + '.json')
if (-not $Quiet) { "upstream-reviews $Repo : inspection=$($snap.inspection_status) items=$($items.Count)" }
$snap | ConvertTo-Json -Depth 8 -Compress
exit $(if (-not $available) { 2 } elseif ($items.Count -gt 0) { 1 } else { 0 })
