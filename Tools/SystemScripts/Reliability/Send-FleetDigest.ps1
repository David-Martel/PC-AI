#Requires -Version 5.1
<#
.SYNOPSIS
    Posts the daily fleet health digest to agent-bus (and Discord when configured).

.DESCRIPTION
    With -Fleet (or policy digest.fleet) it summarises every host from the agent-bus fleet-failures
    topic; otherwise this host from its own ledger. Docs: docs/reliability/fleet-ledger.md.

.EXAMPLE
    pwsh -NoProfile -File Send-FleetDigest.ps1 -Fleet
#>
[CmdletBinding()]
param([string]$PolicyPath, [int]$SinceHours = 24, [switch]$Fleet, [switch]$NoSend)
$ErrorActionPreference = 'Stop'
try {
    Import-Module (Join-Path $PSScriptRoot '..\..\..\Modules\PC-AI.Reliability\PC-AI.Reliability.psd1') -Force
    $null = Set-FleetNormalPriority   # tasks start BelowNormal; a starved monitor goes quiet under load
    $p = @{ SinceHours = $SinceHours; Fleet = $Fleet; NoSend = $NoSend }
    if ($PolicyPath) { $p.PolicyPath = $PolicyPath }
    $d = Send-FleetDigest @p
    Write-Output $d.Message
    exit 0
}
catch {
    Write-Error "fleet digest failed: $($_.Exception.Message)"
    exit 2
}
