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

.EXAMPLE
    pwsh -NoProfile -File Invoke-FleetTask.ps1 -Name '\backups\pull' -FilePath pwsh.exe -ArgumentList '-NoProfile','-File','C:\jobs\pull.ps1' -LogPath C:\logs\pull.log
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$Name,
    [Parameter(Mandatory)][string]$FilePath,
    [string[]]$ArgumentList = @(),
    [string]$LogPath,
    [string[]]$FailurePattern,
    [int]$TimeoutMinutes = 120,
    [ValidateSet('low', 'medium', 'high', 'critical')][string]$Severity = 'medium'
)
$ErrorActionPreference = 'Stop'
try {
    Import-Module (Join-Path $PSScriptRoot '..\..\..\Modules\PC-AI.Reliability\PC-AI.Reliability.psd1') -Force
    $p = @{ Name = $Name; FilePath = $FilePath; ArgumentList = $ArgumentList; TimeoutMinutes = $TimeoutMinutes; Severity = $Severity }
    if ($LogPath) { $p.LogPath = $LogPath }
    if ($FailurePattern) { $p.FailurePattern = $FailurePattern }
    $code = Invoke-FleetTask @p
    exit $code
}
catch {
    Write-Error "fleet task wrapper failed before the task could be recorded: $($_.Exception.Message)"
    exit 3
}
