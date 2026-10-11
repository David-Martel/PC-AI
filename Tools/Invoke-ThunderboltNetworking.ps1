#Requires -Version 7.0
<#
.SYNOPSIS
    Operator entrypoint for Thunderbolt / USB4 peer networking in PC-AI.

.DESCRIPTION
    Wraps the PC-AI.Drivers Thunderbolt functions behind a single script with four
    modes:

      - Status   : discover Thunderbolt / USB4 adapters and peer candidates
      - Connect  : connect to a Windows peer over WinRM
      - Optimize : build or apply a conservative tuning plan
      - LinuxPeer: inspect, prepare, configure or benchmark a Linux SSH peer
#>
[Diagnostics.CodeAnalysis.SuppressMessageAttribute('PSAvoidUsingPlainTextForPassword', 'Password',
    Justification = 'Operator CLI tool; accepts plain password from interactive invocation and wraps into SecureString for the WinRM call immediately. No persistence.')]
[Diagnostics.CodeAnalysis.SuppressMessageAttribute('PSAvoidUsingConvertToSecureStringWithPlainText', '',
    Justification = 'Operator CLI bridge: user passes -Password on the command line, we wrap to SecureString for the downstream cmdlet call only. Refactoring to a SecureString param would require an interactive Read-Host -AsSecureString fallback which blocks automated use.')]
[CmdletBinding(SupportsShouldProcess)]
param(
    [Parameter()]
    [ValidateSet('Status', 'Connect', 'Optimize', 'LinuxPeer')]
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
    [switch]$Apply,

    [ValidateSet('Status', 'Prepare', 'Configure', 'Benchmark')]
    [string]$Action = 'Status',

    [string]$Peer = 'millylaptop1',

    [string]$ConfigPath = (Join-Path $PSScriptRoot '../Config/thunderbolt-peers.json'),

    [string]$LinuxInterface,

    [switch]$DryRun,

    [ValidateRange(1, 60)][int]$DurationSeconds = 10,

    [ValidateRange(1024, 65535)][int]$Port = 5201,

    [string]$IperfPath = 'iperf3',

    [Alias('h', 'help')][switch]$ShowHelp
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if ($ShowHelp) { Get-Help $PSCommandPath -Detailed; return }
if ($Mode -eq 'LinuxPeer') {
    & (Join-Path $PSScriptRoot 'SystemScripts/Networking/Invoke-ThunderboltLinuxPeer.ps1') `
        -Action $Action -Peer $Peer -ConfigPath $ConfigPath -InterfaceAlias $InterfaceAlias `
        -LinuxInterface $LinuxInterface -Apply:$Apply -DryRun:$DryRun `
        -DurationSeconds $DurationSeconds -Port $Port -IperfPath $IperfPath -WhatIf:$WhatIfPreference
    return
}

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
            -Apply:($Apply -and -not $DryRun) `
            -WhatIf:$WhatIfPreference
    }
}
