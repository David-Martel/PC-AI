#Requires -Version 7.0
[CmdletBinding()]
param(
    [Parameter()]
    [string]$ComputerName,

    [Parameter()]
    [switch]$IncludeRawCommands
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
$driversManifest = Join-Path $repoRoot 'Modules\PC-AI.Drivers\PC-AI.Drivers.psd1'
Import-Module $driversManifest -Force

Get-NetworkDiscoverySnapshot -ComputerName $ComputerName -IncludeRawCommands:$IncludeRawCommands
