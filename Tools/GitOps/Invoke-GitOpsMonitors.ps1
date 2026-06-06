#requires -Version 7.0
<#
.SYNOPSIS
    Run the GitOps monitors (workflow-health + upstream-reviews) for a repo and publish the result
    where agents can read it: a combined JSON under Reports/gitops/ and (best-effort) the agent-bus.
    Built to be launched DETACHED from a git hook so it never blocks commit/push (directive 3).

.PARAMETER Repo          owner/name (default: derived via gh in the repo dir).
.PARAMETER DelaySeconds  wait before querying, so post-push CI has time to register (default 20).
.PARAMETER NoDelay       skip the delay (for on-commit use).
.PARAMETER BusUrl        agent-bus HTTP ingest (default http://localhost:8400/messages).
#>
[CmdletBinding()]
param(
    [string] $Repo,
    [int]    $DelaySeconds = 20,
    [switch] $NoDelay,
    [string] $BusUrl = 'http://localhost:8400/messages'
)
$ErrorActionPreference = 'Continue'
if (-not $NoDelay -and $DelaySeconds -gt 0) { Start-Sleep -Seconds $DelaySeconds }
if (-not $Repo) { $Repo = (gh repo view --json nameWithOwner --jq .nameWithOwner 2>$null) }
if (-not $Repo) { exit 2 }

$here = $PSScriptRoot
$wh = & "$here\Watch-WorkflowHealth.ps1" -Repo $Repo -Quiet | Select-Object -Last 1 | ConvertFrom-Json
$ur = & "$here\Get-UpstreamReviews.ps1" -Repo $Repo -Quiet | Select-Object -Last 1 | ConvertFrom-Json

$combined = [ordered]@{
    repo = $Repo; ts_utc = (Get-Date).ToString('o')
    workflow_problems = $wh.problem_count
    upstream_items    = $ur.total
    detail = @{ workflow = $wh; upstream = $ur }
}
$root = Resolve-Path (Join-Path $here '..\..') | Select-Object -ExpandProperty Path
$dir = Join-Path $root 'Reports\gitops'; New-Item -ItemType Directory $dir -Force | Out-Null
$combined | ConvertTo-Json -Depth 8 | Set-Content (Join-Path $dir ("monitors-latest-{0}.json" -f ($Repo -replace '[\\/]', '_'))) -Encoding UTF8

# Best-effort publish to the agent-bus so live agents capture it (never throw if the bus is down).
if (($wh.problem_count -gt 0) -or ($ur.total -gt 0)) {
    try {
        $body = @{
            sender = 'gitops-monitor'; recipient = 'broadcast'; topic = 'status'; schema = 'status'
            tags = @("repo:$($Repo -replace '.*/','')")
            body = "GitOps monitor [$Repo]: $($wh.problem_count) workflow problem(s), $($ur.total) upstream review item(s). See Reports/gitops/monitors-latest-*.json. Workflow: $((($wh.problems) -join '; ')). Upstream: $($ur.by_severity)."
        } | ConvertTo-Json -Depth 5
        Invoke-RestMethod -Uri $BusUrl -Method Post -Body $body -ContentType 'application/json' -TimeoutSec 4 | Out-Null
    } catch { }
}
"gitops-monitors $Repo : wf=$($wh.problem_count) upstream=$($ur.total)"
