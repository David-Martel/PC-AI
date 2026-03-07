#Requires -Version 5.1
<#
.SYNOPSIS
Install PC_AI development modules into a stable local PowerShell module root.

.DESCRIPTION
Publishes repo-backed PC_AI modules into a non-OneDrive module root so module
resolution is deterministic across shells and build scripts. Repo-local modules
are linked by junction when possible for fast iteration; external modules such
as CargoTools are copied by default when their source lives under OneDrive.
#>
[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [string]$RepoRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$InstallRoot = (Join-Path $env:LOCALAPPDATA 'PowerShell\Modules'),
    [ValidateSet('Auto', 'Junction', 'Copy')]
    [string]$Mode = 'Auto',
    [bool]$UpdatePSModulePath = $true,
    [ValidateSet('Process', 'User', 'Machine')]
    [string]$PSModulePathScope = 'User',
    [switch]$IncludeCargoTools
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

. (Join-Path $PSScriptRoot 'PcaiModuleBootstrap.ps1')

function Test-IsOneDrivePath {
    param([string]$Path)

    if ([string]::IsNullOrWhiteSpace($Path)) { return $false }
    return $Path -like '*\OneDrive\*'
}

function Resolve-InstallMode {
    param(
        [string]$RequestedMode,
        [string]$SourcePath,
        [string]$RepoRoot
    )

    if ($RequestedMode -ne 'Auto') {
        return $RequestedMode
    }

    if ((Test-IsOneDrivePath -Path $SourcePath) -or -not ($SourcePath -like "$RepoRoot*")) {
        return 'Copy'
    }

    return 'Junction'
}

function Set-PreferredPsModulePath {
    param(
        [string]$PreferredRoot,
        [ValidateSet('Process', 'User', 'Machine')]
        [string]$Scope
    )

    $currentValue = if ($Scope -eq 'Process') {
        $env:PSModulePath
    } else {
        [Environment]::GetEnvironmentVariable('PSModulePath', $Scope)
    }

    $pathParts = @($currentValue -split ';' | Where-Object { $_ -and $_.Trim() })
    $dedupedParts = New-Object System.Collections.Generic.List[string]
    $seen = New-Object 'System.Collections.Generic.HashSet[string]' ([System.StringComparer]::OrdinalIgnoreCase)

    foreach ($part in @($PreferredRoot) + $pathParts) {
        $normalized = try {
            [System.IO.Path]::GetFullPath($part)
        } catch {
            $part
        }

        if ($seen.Add($normalized)) {
            $dedupedParts.Add($part)
        }
    }

    $newValue = if ($Scope -eq 'User') {
        $PreferredRoot
    } else {
        $dedupedParts -join ';'
    }
    if ($Scope -eq 'Process') {
        $env:PSModulePath = $newValue
    } else {
        [Environment]::SetEnvironmentVariable('PSModulePath', $newValue, $Scope)
        $env:PSModulePath = $newValue
    }
}

function Install-ModuleDirectory {
    param(
        [Parameter(Mandatory)]
        [string]$Name,
        [Parameter(Mandatory)]
        [string]$SourcePath,
        [Parameter(Mandatory)]
        [string]$DestinationRoot,
        [ValidateSet('Junction', 'Copy')]
        [string]$InstallMode
    )

    $destinationPath = Join-Path $DestinationRoot $Name
    if (Test-Path -LiteralPath $destinationPath) {
        Remove-Item -LiteralPath $destinationPath -Recurse -Force
    }

    if ($InstallMode -eq 'Junction') {
        New-Item -ItemType Junction -Path $destinationPath -Target $SourcePath | Out-Null
    } else {
        New-Item -ItemType Directory -Path $destinationPath -Force | Out-Null
        Get-ChildItem -LiteralPath $SourcePath -Force | ForEach-Object {
            Copy-Item -LiteralPath $_.FullName -Destination $destinationPath -Recurse -Force
        }
    }

    return $destinationPath
}

$RepoRoot = [System.IO.Path]::GetFullPath($RepoRoot)
$InstallRoot = Get-PcaiStableModuleInstallRoot -InstallRoot $InstallRoot
New-Item -ItemType Directory -Path $InstallRoot -Force | Out-Null

$moduleSources = New-Object System.Collections.Generic.List[object]
$sourceModulesRoot = Join-Path $RepoRoot 'Modules'
foreach ($moduleDir in (Get-ChildItem -LiteralPath $sourceModulesRoot -Directory | Sort-Object Name)) {
    $manifestPath = Join-Path $moduleDir.FullName "$($moduleDir.Name).psd1"
    $moduleScriptPath = Join-Path $moduleDir.FullName "$($moduleDir.Name).psm1"
    if ((Test-Path -LiteralPath $manifestPath) -or (Test-Path -LiteralPath $moduleScriptPath)) {
        $moduleSources.Add([PSCustomObject]@{
            Name       = $moduleDir.Name
            SourcePath = $moduleDir.FullName
            SourceType = 'repo-module'
        })
    }
}

if ($IncludeCargoTools) {
    $cargoToolsManifest = $null
    $cargoSourceCandidates = @(
        (Join-Path (Join-Path $env:USERPROFILE 'OneDrive\Documents\PowerShell\Modules\CargoTools') 'CargoTools.psd1')
    )

    foreach ($candidate in $cargoSourceCandidates) {
        if (Test-Path -LiteralPath $candidate) {
            $cargoToolsManifest = $candidate
            break
        }
    }

    if (-not $cargoToolsManifest) {
        $availableCargoTools = Get-Module -ListAvailable -Name 'CargoTools' | Sort-Object Version -Descending
        foreach ($availableModule in $availableCargoTools) {
            if ($availableModule.Path -and ($availableModule.Path -notlike "$InstallRoot*")) {
                $cargoToolsManifest = $availableModule.Path
                break
            }
        }
    }

    if (-not $cargoToolsManifest) {
        $cargoToolsManifest = Resolve-PcaiModuleManifestPath -ModuleName 'CargoTools' -RepoRoot $RepoRoot -InstallRoot $InstallRoot
    }

    if ($cargoToolsManifest) {
        $moduleSources.Add([PSCustomObject]@{
            Name       = 'CargoTools'
            SourcePath = (Split-Path -Parent $cargoToolsManifest)
            SourceType = 'external-module'
        })
    } else {
        Write-Warning 'CargoTools source not found; skipping CargoTools install.'
    }
}

$results = foreach ($module in $moduleSources) {
    $installMode = Resolve-InstallMode -RequestedMode $Mode -SourcePath $module.SourcePath -RepoRoot $RepoRoot
    $targetPath = Join-Path $InstallRoot $module.Name

    if (-not $PSCmdlet.ShouldProcess($targetPath, "Install $($module.Name) using $installMode")) {
        continue
    }

    $installedPath = Install-ModuleDirectory -Name $module.Name -SourcePath $module.SourcePath -DestinationRoot $InstallRoot -InstallMode $installMode
    [PSCustomObject]@{
        Name         = $module.Name
        SourcePath   = $module.SourcePath
        InstalledPath = $installedPath
        Mode         = $installMode
        SourceType   = $module.SourceType
    }
}

if ($UpdatePSModulePath) {
    Set-PreferredPsModulePath -PreferredRoot $InstallRoot -Scope $PSModulePathScope
}

$results
