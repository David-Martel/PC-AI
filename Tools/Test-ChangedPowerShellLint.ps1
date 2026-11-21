#Requires -Version 7.0
<#
.SYNOPSIS
    Runs PSScriptAnalyzer on only the PowerShell files this branch changed.

.DESCRIPTION
    `Build.ps1 -Component lint -LintProfile powershell` analyzes every .ps1, .psm1 and .psd1 file
    (about 850) one at a time and takes 8 to 12 minutes. PSScriptAnalyzer keeps static state, so
    it cannot be fanned out across runspaces in one process, and splitting across eight processes
    measured only 1.4x faster (701 s to 492 s on a 16-thread laptop). Analyzing just the changed
    files takes seconds and is what a developer needs before a push; the full pass stays in CI and
    in the Full tier of Tests/Invoke-PcaiCi.ps1.

    Changed means: committed on this branch since the merge base with -BaseRef, plus staged,
    modified and untracked files. Files under bin, obj, target, .git, .pcai, packages, artifacts,
    output and node_modules are skipped, matching Build.ps1.

    The tool reports how many files it analyzed so a run that silently saw nothing is visible.

.PARAMETER RepoRoot
    Repository to inspect. Defaults to the repository containing this script.

.PARAMETER BaseRef
    Ref to diff against. Default origin/main.

.PARAMETER SettingsPath
    PSScriptAnalyzer settings file. Default PSScriptAnalyzerSettings.psd1 in RepoRoot.

.EXAMPLE
    pwsh Tools/Test-ChangedPowerShellLint.ps1
#>
[CmdletBinding()]
param(
    [string]$RepoRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$BaseRef = 'origin/main',
    [string]$SettingsPath
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if (-not $SettingsPath) { $SettingsPath = Join-Path $RepoRoot 'PSScriptAnalyzerSettings.psd1' }
if (-not (Test-Path -LiteralPath $SettingsPath)) { throw "Analyzer settings not found: $SettingsPath" }
if (-not (Get-Module -ListAvailable PSScriptAnalyzer)) {
    # CI installs it; a local run that cannot analyze must not pass.
    throw 'PSScriptAnalyzer is not installed; run Setup-DevEnvironment.ps1 or Install-Module PSScriptAnalyzer.'
}
Import-Module PSScriptAnalyzer -ErrorAction Stop

$mergeBase = (git -C $RepoRoot merge-base $BaseRef HEAD 2>$null)
if ($LASTEXITCODE -ne 0 -or -not $mergeBase) { throw "Cannot find a merge base between '$BaseRef' and HEAD in $RepoRoot" }

$committed = @(git -C $RepoRoot diff --name-only --diff-filter=ACMR "$mergeBase" HEAD)
$working = @(git -C $RepoRoot diff --name-only --diff-filter=ACMR HEAD)
$untracked = @(git -C $RepoRoot ls-files --others --exclude-standard)
$exclude = '(^|/)(target|bin|obj|node_modules|\.git|\.pcai|packages|artifacts|output)/'
$files = @($committed + $working + $untracked |
    Where-Object { $_ -and $_ -match '\.(ps1|psm1|psd1)$' -and $_ -notmatch $exclude } |
    Sort-Object -Unique |
    ForEach-Object { Join-Path $RepoRoot $_ } |
    Where-Object { Test-Path -LiteralPath $_ })

if ($files.Count -eq 0) {
    Write-Host "no changed PowerShell files against $BaseRef"
    return
}

$issues = foreach ($file in $files) {
    Invoke-ScriptAnalyzer -Path $file -Settings $SettingsPath -Severity Warning, Error
}
$issues = @($issues)
Write-Host "analyzed $($files.Count) changed PowerShell file(s) against $BaseRef"
if ($issues.Count -gt 0) {
    foreach ($issue in $issues) {
        Write-Host ("{0}:{1}:{2} [{3}] {4}" -f $issue.ScriptPath, $issue.Line, $issue.Column, $issue.RuleName, $issue.Message)
    }
    Write-Error "PSScriptAnalyzer reported $($issues.Count) issue(s) in changed files" -ErrorAction Continue
    exit 1
}
