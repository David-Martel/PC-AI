[CmdletBinding()]
param(
    [string]$ReleaseModulePath = (Join-Path $PSScriptRoot '..\Release\PowerShell\PC-AI\PC-AI.psd1')
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if (-not (Test-Path -LiteralPath $ReleaseModulePath)) {
    throw "Release module manifest not found: $ReleaseModulePath"
}

$package = @(Import-Module -Name $ReleaseModulePath -Force -PassThru -ErrorAction Stop)
if ($package.Count -ne 1) {
    throw "Expected one imported release module from: $ReleaseModulePath"
}

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
    if (-not $package[0].ExportedFunctions.ContainsKey($commandName)) {
        $missing += $commandName
    }
}

[PSCustomObject]@{
    ReleaseModulePath = $ReleaseModulePath
    RequiredCommands = $requiredCommands.Count
    MissingCommands = $missing
    Passed = ($missing.Count -eq 0)
}
