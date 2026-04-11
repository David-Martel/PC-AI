#Requires -Version 7.0
<#
.SYNOPSIS
    Operator entrypoint for Thunderbolt / USB4 peer networking in PC-AI.

.DESCRIPTION
    Wraps the PC-AI.Drivers Thunderbolt functions behind a single script with three
    modes:

      - Status   : discover Thunderbolt / USB4 adapters and peer candidates
      - Connect  : connect to a Windows peer over WinRM
      - Optimize : build or apply a conservative tuning plan
#>
[Diagnostics.CodeAnalysis.SuppressMessageAttribute('PSAvoidUsingPlainTextForPassword', 'Password',
    Justification = 'Operator CLI tool; accepts plain password from interactive invocation and wraps into SecureString for the WinRM call immediately. No persistence.')]
[Diagnostics.CodeAnalysis.SuppressMessageAttribute('PSAvoidUsingConvertToSecureStringWithPlainText', '',
    Justification = 'Operator CLI bridge: user passes -Password on the command line, we wrap to SecureString for the downstream cmdlet call only. Refactoring to a SecureString param would require an interactive Read-Host -AsSecureString fallback which blocks automated use.')]
[CmdletBinding(SupportsShouldProcess)]
param(
    [Parameter()]
    [ValidateSet('Status', 'Connect', 'Optimize')]
    [string]$Mode = 'Status',

    [Parameter()]
    [string]$InterfaceAlias,

    [Parameter()]
    [string]$ComputerName,

    [Parameter()]
    [string]$Address,

    [Parameter()]
    [string]$MicrosoftAccountEmail = 'davidmartel07@gmail.com',

    [Parameter()]
    [string]$Password,

    [Parameter()]
    [int]$InterfaceMetric = 15,

    [Parameter()]
    [int]$MtuBytes = 62000,

    [Parameter()]
    [string]$IPv4Address,

    [Parameter()]
    [int]$PrefixLength = 30,

    [Parameter()]
    [switch]$ProbeWinRM,

    [Parameter()]
    [switch]$Apply
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
$driversManifest = Join-Path $repoRoot 'Modules\PC-AI.Drivers\PC-AI.Drivers.psd1'
Import-Module $driversManifest -Force

$securePassword = $null
if ($Password) {
    $securePassword = ConvertTo-SecureString -String $Password -AsPlainText -Force
}

switch ($Mode) {
    'Status' {
        Get-ThunderboltNetworkStatus -InterfaceAlias $InterfaceAlias -ProbeWinRM:$ProbeWinRM
    }
    'Connect' {
        Connect-ThunderboltPeer `
            -InterfaceAlias $InterfaceAlias `
            -ComputerName $ComputerName `
            -Address $Address `
            -MicrosoftAccountEmail $MicrosoftAccountEmail `
            -Password $securePassword
    }
    'Optimize' {
        Set-ThunderboltNetworkOptimization `
            -InterfaceAlias $InterfaceAlias `
            -InterfaceMetric $InterfaceMetric `
            -MtuBytes $MtuBytes `
            -IPv4Address $IPv4Address `
            -PrefixLength $PrefixLength `
            -Apply:$Apply `
            -WhatIf:$WhatIfPreference
    }
}
