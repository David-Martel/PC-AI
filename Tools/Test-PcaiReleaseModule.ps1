[CmdletBinding()]
param(
    [string]$ReleaseModulePath = (Join-Path $PSScriptRoot '..\Release\PowerShell\PC-AI\PC-AI.psd1')
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if (-not (Test-Path -LiteralPath $ReleaseModulePath)) {
    throw "Release module manifest not found: $ReleaseModulePath"
}

Import-Module -Name $ReleaseModulePath -Force -ErrorAction Stop

$requiredCommands = @(
    'Get-PcaiReleaseInfo',
    'Import-PcaiComponentModules',
    'Get-RustToolStatus',
    'Find-FilesFast',
    'Get-ProcessPerformance',
    'Get-WSLStatus'
)

$missing = @()
foreach ($commandName in $requiredCommands) {
    if (-not (Get-Command -Name $commandName -ErrorAction SilentlyContinue)) {
        $missing += $commandName
    }
}

[PSCustomObject]@{
    ReleaseModulePath = $ReleaseModulePath
    RequiredCommands = $requiredCommands.Count
    MissingCommands = $missing
    Passed = ($missing.Count -eq 0)
}
