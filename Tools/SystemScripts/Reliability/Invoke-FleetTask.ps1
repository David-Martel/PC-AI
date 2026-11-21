#Requires -Version 5.1
<#
.SYNOPSIS
    Wraps a scheduled task's command so every run lands in the fleet failure ledger.

.DESCRIPTION
    Runs -FilePath with -ArgumentList under a timeout, records start/end/exit code, and scans the
    output (plus the bytes the tool appended to -LogPath during this run) for failure lines. A run
    that exits 0 but prints "[ERROR]", "STALE" or "integrity check did not pass" is recorded as
    failed and this script exits 1, so Task Scheduler stops reporting 0x0 for it. The first failed
    run opens an incident (agent-bus alert); the next clean run resolves it.
    Docs: docs/reliability/fleet-ledger.md.

.PARAMETER Name
    The task's full path, e.g. '\fleet-reliability-test\demo'. It is the incident key, and the
    sweeper's task-result check uses the same key, so one failure opens one incident.

.PARAMETER PolicyPath
    Host policy (Discord recipients and floor). Default: the fleet-reliability home's policy.json.

.PARAMETER DryRun
    Print what would run and be recorded; run nothing and write nothing. `--DryRun` also works.

.PARAMETER Help
    Print this help and exit. `-h`, `-?` and `--help` also work.

.EXAMPLE
    pwsh -NoProfile -Command "& 'Invoke-FleetTask.ps1' -Name '\backups\pull' -FilePath pwsh.exe -ArgumentList '-NoProfile','-File','C:\jobs\pull.ps1' -LogPath C:\logs\pull.log; exit $LASTEXITCODE"
    Use -Command, not -File: `pwsh -File` cannot pass an array to -ArgumentList (a quoted list is refused).
#>
[CmdletBinding()]
param(
    [string]$Name,
    [string]$FilePath,
    [string[]]$ArgumentList = @(),
    [string]$LogPath,
    [string[]]$FailurePattern,
    [int]$TimeoutMinutes = 120,
    [ValidateSet('low', 'medium', 'high', 'critical')][string]$Severity = 'medium',
    [string]$PolicyPath,
    [switch]$DryRun,
    [Alias('h', '?')]
    [switch]$Help,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$CliArgs
)
$CliArgs = @($CliArgs | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
if ($CliArgs -contains '--help') { $Help = $true; $CliArgs = @($CliArgs | Where-Object { $_ -ne '--help' }) }
if ($CliArgs -contains '--DryRun') { $DryRun = $true; $CliArgs = @($CliArgs | Where-Object { $_ -ne '--DryRun' }) }
if ($Help) {
    $m = [regex]::Match((Get-Content -LiteralPath $PSCommandPath -Raw), '(?s)<#\s*(.*?)\s*#>')
    Write-Output $m.Groups[1].Value.Trim()
    exit 0
}
if ($CliArgs.Count -gt 0) { Write-Error "Unknown argument(s): $($CliArgs -join ', ')"; exit 2 }
if (-not $Name -or -not $FilePath) { Write-Error '-Name and -FilePath are required (see --help)'; exit 2 }
# `pwsh -File` does not build arrays: -ArgumentList '/c','x' arrives as the single string "'/c','x'" and the job
# would run with garbage arguments. Refuse it loudly; use the -Command form shown in the help.
if (@($ArgumentList).Count -eq 1 -and $ArgumentList[0] -match "^'.*','.*'$") {
    Write-Error "-ArgumentList arrived as one string ($($ArgumentList[0])): pwsh -File cannot pass arrays. Use: pwsh -Command `"& '<this script>' -Name ... -FilePath ... -ArgumentList 'a','b'; exit `$LASTEXITCODE`""
    exit 2
}

$ErrorActionPreference = 'Stop'
try {
    Import-Module (Join-Path $PSScriptRoot '..\..\..\Modules\PC-AI.Reliability\PC-AI.Reliability.psd1') -Force
    $p = @{ Name = $Name; FilePath = $FilePath; ArgumentList = $ArgumentList; TimeoutMinutes = $TimeoutMinutes; Severity = $Severity; DryRun = $DryRun }
    if ($LogPath) { $p.LogPath = $LogPath }
    if ($FailurePattern) { $p.FailurePattern = $FailurePattern }
    if ($PolicyPath) { $p.PolicyPath = $PolicyPath }
    $code = Invoke-FleetTask @p
    exit $code
}
catch {
    Write-Error "fleet task wrapper failed before the task could be recorded: $($_.Exception.Message)"
    exit 3
}
