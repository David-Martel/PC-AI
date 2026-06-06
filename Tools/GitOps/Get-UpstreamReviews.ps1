#requires -Version 7.0
<#
.SYNOPSIS
    Harvest upstream automated-review findings for a repo into one actionable ledger: Dependabot
    alerts, code-scanning alerts, and bot PR reviews (Copilot / gemini-code-assist / Jules). These
    normally pile up where nobody looks; this pulls them into Reports/gitops/ and the agent-bus.

.DESCRIPTION
    Read-only. Endpoints that aren't enabled (e.g. code scanning) are skipped gracefully. Designed to
    be fired non-blocking from a git hook so agents working the repo see outstanding review items.

.PARAMETER Repo   owner/name (default: derived via gh).
.PARAMETER Quiet  suppress console output (hook mode).
#>
[CmdletBinding()]
param([string] $Repo, [switch] $Quiet)

$ErrorActionPreference = 'Continue'
if (-not $Repo) { $Repo = (gh repo view --json nameWithOwner --jq .nameWithOwner 2>$null) }
if (-not $Repo) { Write-Error 'cannot determine repo'; exit 2 }
function Say($m, $c = 'Gray') { if (-not $Quiet) { Write-Host $m -ForegroundColor $c } }

$items = [System.Collections.Generic.List[object]]::new()

# 1. Dependabot alerts (open)
$dep = gh api "repos/$Repo/dependabot/alerts?state=open&per_page=100" --paginate 2>$null | ConvertFrom-Json
foreach ($a in @($dep)) {
    $items.Add(@{ source = 'dependabot'; severity = $a.security_advisory.severity; title = $a.security_advisory.summary; ref = $a.html_url })
}
# 2. Code scanning alerts (open) — 404s if not enabled
$cs = gh api "repos/$Repo/code-scanning/alerts?state=open&per_page=100" 2>$null | ConvertFrom-Json
foreach ($a in @($cs)) {
    if ($a.rule) { $items.Add(@{ source = 'code-scanning'; severity = $a.rule.security_severity_level ?? $a.rule.severity; title = $a.rule.description; ref = $a.html_url }) }
}
# 3. Bot reviews on open PRs (Copilot / Gemini / Jules)
$prs = gh api "repos/$Repo/pulls?state=open&per_page=50" --jq '.[].number' 2>$null
foreach ($n in @($prs)) {
    $revs = gh api "repos/$Repo/pulls/$n/reviews" 2>$null | ConvertFrom-Json
    foreach ($r in @($revs)) {
        if ($r.user.login -match 'copilot|gemini|jules|bot') {
            $items.Add(@{ source = "pr-review:$($r.user.login)"; severity = 'review'; title = "PR #$n $($r.state)"; ref = $r.html_url })
        }
    }
}

$bySev = $items | Group-Object { $_.severity } | ForEach-Object { "$($_.Name)=$($_.Count)" }
$snap = [ordered]@{ repo = $Repo; checked_utc = (Get-Date).ToString('o'); total = $items.Count; by_severity = ($bySev -join ' '); items = $items }
$root = Resolve-Path (Join-Path $PSScriptRoot '..\..') | Select-Object -ExpandProperty Path
$dir = Join-Path $root 'Reports\gitops'; New-Item -ItemType Directory $dir -Force | Out-Null
$out = Join-Path $dir ("upstream-reviews-{0}.json" -f ($Repo -replace '[\\/]', '_'))
$snap | ConvertTo-Json -Depth 6 | Set-Content $out -Encoding UTF8

Say "upstream-reviews $Repo : $($items.Count) item(s) [$($bySev -join ' ')] -> $out" ($items.Count ? 'Yellow' : 'Green')
$snap | ConvertTo-Json -Depth 6 -Compress
exit ($items.Count ? 1 : 0)
