#Requires -Version 7.0
<#
.SYNOPSIS
    Plans settings on a Windows Thunderbolt peer using the maintained bridge.
.DESCRIPTION
    Run in PowerShell 7 (pwsh) on the intended Windows peer. This compatibility
    entrypoint defaults to a plan with the legacy peer address 172.31.240.2/30.
    It never deletes addresses, enables WinRM or changes the network category.
    Apply delegates through Initialize-ThunderboltLink to the central optimizer
    and its shared IPv4 identity/conflict/preservation guard. Security requests
    require separately owned host policies and are refused on Apply. No Linux peer
    profile or remote transport is selected automatically.
.EXAMPLE
    ./Bootstrap-ThunderboltPeerRemote.ps1 -InterfaceAlias 'Ethernet 11' -DryRun
.EXAMPLE
    ./Bootstrap-ThunderboltPeerRemote.ps1 -InterfaceAlias 'Ethernet 11' -MetricOnly -Apply -WhatIf
#>
[CmdletBinding(SupportsShouldProcess, PositionalBinding = $false)]
param(
    [string]$InterfaceAlias,
    [string]$IPv4Address,
    [ValidateRange(8, 30)][int]$PrefixLength = 30,
    [ValidateRange(1, 9999)][int]$InterfaceMetric = 15,
    [ValidateRange(1280, 65535)][int]$MtuBytes = 62000,
    [switch]$SetPrivateProfile,
    [switch]$EnablePsRemoting,
    [switch]$MetricOnly,
    [switch]$Apply,
    [switch]$DryRun,
    [Alias('h', 'help')][switch]$ShowHelp,
    [Parameter(ValueFromRemainingArguments)][string[]]$CliArgs
)

# Keep explicitly bound options, including common WhatIf/Confirm controls.
# Validation, planning and ShouldProcess decisions belong to the one bridge.
$forwarded = @{}
foreach ($name in $PSBoundParameters.Keys) {
    $forwarded[$name] = $PSBoundParameters[$name]
}
$forwarded.CompatibilityRole = 'WindowsPeer'
& (Join-Path $PSScriptRoot 'Initialize-ThunderboltLink.ps1') @forwarded
