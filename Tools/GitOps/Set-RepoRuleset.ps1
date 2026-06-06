#requires -Version 7.0
<#
.SYNOPSIS
    Idempotently apply the standard David-Martel branch-protection ruleset to one or many repos.

.DESCRIPTION
    Profile chosen 2026-06-06 ("strict PR-only, self-mergeable"):
      - required_signatures         (SSH/GPG signed commits enforced)
      - pull_request                (no direct push to default branch; 0 required approvals so the
                                     owner can self-merge after checks; conversation resolution on)
      - non_fast_forward + deletion (block force-push and branch deletion)
      - bypass: GitHub Actions app (id 15368) so CI auto-commit workflows (changelog/format) keep
                working; NO human/owner bypass.

    Idempotent: if a ruleset named $RulesetName already exists on a repo it is UPDATED (PUT), else
    CREATED (POST). Per-repo results are logged. Use -DryRun to preview without changing anything.

.PARAMETER Owner       GitHub owner (default David-Martel)
.PARAMETER Repos       Explicit repo names. If omitted with -All, enumerates eligible repos.
.PARAMETER All         Apply to all eligible repos (ADMIN, not archived, not fork).
.PARAMETER RulesetName Ruleset name (default dtm-default-protection)
.PARAMETER RequireStatusChecks  Optional list of required status-check contexts (repo-specific).
.PARAMETER DryRun      Preview only.
.PARAMETER LogDir      Where to write the run log (default Reports/gitops/).

.EXAMPLE
    .\Set-RepoRuleset.ps1 -Repos PC-AI -DryRun
.EXAMPLE
    .\Set-RepoRuleset.ps1 -All        # flip every eligible repo
#>
[CmdletBinding()]
param(
    [string]   $Owner = 'David-Martel',
    [string[]] $Repos,
    [switch]   $All,
    [string]   $RulesetName = 'dtm-default-protection',
    [string[]] $RequireStatusChecks,
    [switch]   $DryRun,
    [string]   $LogDir
)

$ErrorActionPreference = 'Stop'
$GH_ACTIONS_APP_ID = 15368   # gh api /apps/github-actions => id

if (-not $LogDir) {
    $repoRoot = Resolve-Path (Join-Path $PSScriptRoot '..\..') | Select-Object -ExpandProperty Path
    $LogDir = Join-Path $repoRoot 'Reports\gitops'
}
New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
$stamp = (Get-Date).ToString('yyyyMMdd-HHmmss')
$log = Join-Path $LogDir "ruleset-apply-$stamp.jsonl"

function New-RulesetBody {
    param([string[]] $Checks)
    $rules = [System.Collections.Generic.List[object]]::new()
    $rules.Add(@{ type = 'deletion' })
    $rules.Add(@{ type = 'non_fast_forward' })
    $rules.Add(@{ type = 'required_signatures' })
    $rules.Add(@{ type = 'pull_request'; parameters = @{
        required_approving_review_count   = 0
        dismiss_stale_reviews_on_push     = $false
        require_code_owner_review         = $false
        require_last_push_approval        = $false
        required_review_thread_resolution = $true
    } })
    if ($Checks) {
        $rules.Add(@{ type = 'required_status_checks'; parameters = @{
            strict_required_status_checks_policy = $true
            required_status_checks = @($Checks | ForEach-Object { @{ context = $_ } })
        } })
    }
    return @{
        name        = $RulesetName
        target      = 'branch'
        enforcement = 'active'
        conditions  = @{ ref_name = @{ include = @('~DEFAULT_BRANCH'); exclude = @() } }
        bypass_actors = @(
            @{ actor_id = $GH_ACTIONS_APP_ID; actor_type = 'Integration'; bypass_mode = 'always' }
        )
        rules = $rules
    }
}

function Get-EligibleRepos {
    Write-Host "Enumerating eligible repos for $Owner ..." -ForegroundColor Cyan
    $all = gh repo list $Owner --limit 400 --json name,isArchived,isFork,viewerPermission | ConvertFrom-Json
    $all | Where-Object { $_.viewerPermission -eq 'ADMIN' -and -not $_.isArchived -and -not $_.isFork } |
        Select-Object -ExpandProperty name
}

function Set-OneRepoRuleset {
    param([string] $Repo)
    $result = [ordered]@{ repo = $Repo; action = $null; status = $null; ruleset_id = $null; error = $null }
    try {
        $existing = gh api "repos/$Owner/$Repo/rulesets" 2>$null | ConvertFrom-Json
        $match = $existing | Where-Object { $_.name -eq $RulesetName } | Select-Object -First 1
        $body = New-RulesetBody -Checks $RequireStatusChecks | ConvertTo-Json -Depth 10
        $tmp = New-TemporaryFile
        $body | Set-Content $tmp -Encoding utf8
        if ($DryRun) {
            $result.action = if ($match) { 'would-update' } else { 'would-create' }
            $result.status = 'dry-run'
        }
        elseif ($match) {
            $r = gh api "repos/$Owner/$Repo/rulesets/$($match.id)" -X PUT --input $tmp | ConvertFrom-Json
            $result.action = 'updated'; $result.status = 'ok'; $result.ruleset_id = $r.id
        }
        else {
            $r = gh api "repos/$Owner/$Repo/rulesets" -X POST --input $tmp | ConvertFrom-Json
            $result.action = 'created'; $result.status = 'ok'; $result.ruleset_id = $r.id
        }
        Remove-Item $tmp -Force -ErrorAction SilentlyContinue
    }
    catch {
        $result.status = 'error'; $result.error = "$_"
    }
    ($result | ConvertTo-Json -Compress) | Add-Content -Path $log -Encoding utf8
    $color = switch ($result.status) { 'ok' { 'Green' } 'dry-run' { 'Yellow' } default { 'Red' } }
    Write-Host ("  {0,-32} {1,-12} {2}" -f $Repo, $result.action, ($result.error ?? $result.status)) -ForegroundColor $color
    return $result
}

# --- main ---
$targets = if ($Repos) { $Repos } elseif ($All) { Get-EligibleRepos } else { throw 'Specify -Repos <names> or -All' }
Write-Host "Applying ruleset '$RulesetName' to $($targets.Count) repo(s)$(if($DryRun){' [DRY RUN]'})" -ForegroundColor Cyan
Write-Host "Log: $log`n" -ForegroundColor DarkGray
$results = foreach ($r in $targets) { Set-OneRepoRuleset -Repo $r }
$ok = ($results | Where-Object status -in 'ok', 'dry-run').Count
$err = ($results | Where-Object status -eq 'error').Count
Write-Host "`nDone: $ok ok, $err error(s). Log: $log" -ForegroundColor Cyan
if ($err) { $results | Where-Object status -eq 'error' | ForEach-Object { Write-Host "  FAIL $($_.repo): $($_.error)" -ForegroundColor Red } }
