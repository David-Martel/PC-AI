#requires -Version 7.0
<#
.SYNOPSIS
    Install a NON-BLOCKING git pre-push hook that fires the GitOps monitors (workflow-health +
    upstream-reviews) detached, so a push is never delayed but the results are captured for agents.

.DESCRIPTION
    For repos that use lefthook (a lefthook.yml is present) this prints the pre-push snippet to add
    instead of fighting lefthook over .git/hooks. For plain repos it writes .git/hooks/pre-push.
    The hook launches Invoke-GitOpsMonitors.ps1 in the background and returns immediately (exit 0).

.PARAMETER RepoPath  target repo (default: current dir).
.PARAMETER ToolsRoot path to Tools/GitOps (default: this script's folder).
.PARAMETER Force     overwrite an existing .git/hooks/pre-push.
#>
[CmdletBinding()]
param([string] $RepoPath = (Get-Location).Path, [string] $ToolsRoot = $PSScriptRoot, [switch] $Force)

$gitDir = Join-Path $RepoPath '.git'
if (-not (Test-Path $gitDir)) { Write-Error "not a git repo: $RepoPath"; exit 2 }
$monitor = (Resolve-Path (Join-Path $ToolsRoot 'Invoke-GitOpsMonitors.ps1')).Path

if (Test-Path (Join-Path $RepoPath 'lefthook.yml')) {
    Write-Host "lefthook.yml detected — add this to lefthook.yml instead of installing a raw hook:" -ForegroundColor Yellow
    @"
pre-push:
  commands:
    gitops-monitors:
      run: pwsh -NoProfile -Command "Start-Process pwsh -ArgumentList '-NoProfile','-File','{root}/Tools/GitOps/Invoke-GitOpsMonitors.ps1' -WindowStyle Hidden"
"@ | Write-Host -ForegroundColor Cyan
    return
}

$hook = Join-Path $gitDir 'hooks\pre-push'
if ((Test-Path $hook) -and -not $Force) { Write-Error "pre-push exists (use -Force): $hook"; exit 1 }
New-Item -ItemType Directory (Split-Path $hook) -Force | Out-Null
# POSIX sh (Git for Windows runs hooks via sh). '&' backgrounds; exit 0 returns immediately = non-blocking.
$mon = $monitor -replace '\\', '/'
@"
#!/bin/sh
# GitOps non-blocking monitor (auto-installed). Fires CI-health + upstream-review harvest detached.
pwsh -NoProfile -WindowStyle Hidden -File "$mon" >/dev/null 2>&1 &
exit 0
"@ | Set-Content $hook -Encoding ascii -NoNewline
Write-Host "installed non-blocking pre-push hook -> $hook" -ForegroundColor Green
