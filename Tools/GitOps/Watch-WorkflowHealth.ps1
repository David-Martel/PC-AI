#requires -Version 7.0
<#
.SYNOPSIS
    Detect SILENT GitHub Actions failures for a repo: billing/quota blocks (runs never start),
    startup_failure, disabled workflows, and recent run failures. The "CI looks green" illusion
    comes from runs that never ran — this surfaces them.

.DESCRIPTION
    Read-only. Writes a JSON snapshot to Reports/gitops/ and returns a summary object. Intended to be
    fired non-blocking from a git hook (see Install-GitOpsHooks.ps1) so agents working with git/gh can
    read the result. Posts a high-priority agent-bus note only when problems are found.

.PARAMETER Repo   owner/name (default: derived from the current repo via gh).
.PARAMETER Limit  recent runs to inspect (default 15).
.PARAMETER Quiet  suppress console output (hook mode).
#>
[CmdletBinding()]
param([string] $Repo, [int] $Limit = 15, [switch] $Quiet)

$ErrorActionPreference = 'Continue'
if (-not $Repo) { $Repo = (gh repo view --json nameWithOwner --jq .nameWithOwner 2>$null) }
if (-not $Repo) { Write-Error 'cannot determine repo'; exit 2 }
function Say($m, $c = 'Gray') { if (-not $Quiet) { Write-Host $m -ForegroundColor $c } }

$problems = [System.Collections.Generic.List[string]]::new()

# 1. Recent runs: failures + startup_failure (the silent one)
$runs = gh run list -R $Repo --limit $Limit --json databaseId,status,conclusion,workflowName,event,createdAt 2>$null | ConvertFrom-Json
foreach ($r in $runs) {
    if ($r.conclusion -eq 'startup_failure') { $problems.Add("startup_failure: $($r.workflowName) (run $($r.databaseId)) — often billing/permissions, no failure email") }
    elseif ($r.conclusion -eq 'failure') { $problems.Add("failure: $($r.workflowName) (run $($r.databaseId))") }
    elseif ($r.conclusion -eq 'action_required') { $problems.Add("action_required: $($r.workflowName)") }
}

# 2. Disabled workflows (a configured workflow silently turned off)
$wf = gh api "repos/$Repo/actions/workflows" --jq '.workflows[] | select(.state|startswith("disabled")) | .name' 2>$null
foreach ($w in $wf) { $problems.Add("workflow disabled: $w") }

# 3. Billing / minutes (account-level; quota exhaustion makes runs never start)
$billing = gh api /user/settings/billing/actions 2>$null | ConvertFrom-Json
if ($billing -and $billing.total_minutes_used -ge $billing.included_minutes -and $billing.included_minutes -gt 0) {
    $problems.Add("Actions minutes exhausted: $($billing.total_minutes_used)/$($billing.included_minutes) — paid runs may be blocked")
}

$snap = [ordered]@{
    repo = $Repo; checked_utc = (Get-Date).ToString('o')
    problem_count = $problems.Count; problems = $problems
    last_runs = @($runs | Select-Object -First 5 | ForEach-Object { @{ wf = $_.workflowName; status = $_.status; conclusion = $_.conclusion } })
    billing = if ($billing) { @{ used = $billing.total_minutes_used; included = $billing.included_minutes } } else { $null }
}
$root = Resolve-Path (Join-Path $PSScriptRoot '..\..') | Select-Object -ExpandProperty Path
$dir = Join-Path $root 'Reports\gitops'; New-Item -ItemType Directory $dir -Force | Out-Null
$out = Join-Path $dir ("workflow-health-{0}.json" -f ($Repo -replace '[\\/]', '_'))
$snap | ConvertTo-Json -Depth 6 | Set-Content $out -Encoding UTF8

Say "workflow-health $Repo : $($problems.Count) problem(s) -> $out" ($problems.Count ? 'Yellow' : 'Green')
$problems | ForEach-Object { Say "  ! $_" 'Red' }
$snap | ConvertTo-Json -Depth 6 -Compress
exit ($problems.Count ? 1 : 0)
