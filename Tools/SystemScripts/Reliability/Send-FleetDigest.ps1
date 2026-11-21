#Requires -Version 5.1
<#
.SYNOPSIS
    Posts the daily fleet health digest to agent-bus (and Discord when configured).

.DESCRIPTION
    With -Fleet (or policy digest.fleet) it summarises every host from the agent-bus fleet-failures
    topic; otherwise this host from its own ledger. Docs: docs/reliability/fleet-ledger.md.

.PARAMETER DryRun
    Build and print the digest without writing the ledger or sending anything. `--DryRun` also works.

.PARAMETER Help
    Print this help and exit. `-h`, `-?` and `--help` also work.

.EXAMPLE
    pwsh -NoProfile -File Send-FleetDigest.ps1 -Fleet -DryRun
#>
[CmdletBinding()]
param(
    [string]$PolicyPath,
    [int]$SinceHours = 24,
    [switch]$Fleet,
    [switch]$NoSend,
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

$ErrorActionPreference = 'Stop'
try {
    Import-Module (Join-Path $PSScriptRoot '..\..\..\Modules\PC-AI.Reliability\PC-AI.Reliability.psd1') -Force
    $null = Set-FleetNormalPriority   # tasks start BelowNormal; a starved monitor goes quiet under load
    $p = @{ SinceHours = $SinceHours; Fleet = $Fleet; NoSend = $NoSend; DryRun = $DryRun }
    if ($PolicyPath) { $p.PolicyPath = $PolicyPath }
    $d = Send-FleetDigest @p
    if ($DryRun) { Write-Output 'DRY RUN: digest not recorded or sent' }
    Write-Output $d.Message
    exit 0
}
catch {
    Write-Error "fleet digest failed: $($_.Exception.Message)"
    exit 2
}
