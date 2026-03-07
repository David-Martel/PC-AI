#Requires -Version 5.1
[CmdletBinding()]
param(
    [string]$RepoRoot = (Split-Path -Parent $PSScriptRoot),
    [string[]]$ModuleName = @('CargoTools', 'PC-AI.Acceleration')
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

. (Join-Path $PSScriptRoot 'PcaiModuleBootstrap.ps1')

$userModulePath = [Environment]::GetEnvironmentVariable('PSModulePath', 'User')
$machineModulePath = [Environment]::GetEnvironmentVariable('PSModulePath', 'Machine')
$stableInstallRoot = Get-PcaiStableModuleInstallRoot

$moduleStatus = foreach ($name in $ModuleName) {
    $trace = Get-PcaiModuleResolutionTrace -ModuleName $name -RepoRoot $RepoRoot -InstallRoot $stableInstallRoot
    $resolved = $trace | Where-Object { $_.Exists } | Select-Object -First 1
    $loaded = Get-Module -Name $name | Select-Object -First 1

    [PSCustomObject]@{
        ModuleName      = $name
        ResolvedPath    = $resolved.Path
        ResolvedSource  = $resolved.Source
        LoadedPath      = if ($loaded) { $loaded.Path } else { $null }
        StableInstalled = Test-Path -LiteralPath (Join-Path $stableInstallRoot $name)
        Trace           = $trace
    }
}

[PSCustomObject]@{
    StableInstallRoot = $stableInstallRoot
    ProcessPSModulePath = $env:PSModulePath
    UserPSModulePath  = $userModulePath
    MachinePSModulePath = $machineModulePath
    ProcessNeedsRestart = -not ($env:PSModulePath -like "*$stableInstallRoot*")
    Modules           = $moduleStatus
}
