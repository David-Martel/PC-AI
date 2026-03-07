Set-StrictMode -Version Latest

function Get-PcaiStableModuleInstallRoot {
    [CmdletBinding()]
    param(
        [string]$InstallRoot
    )

    $resolvedRoot = if ($InstallRoot) {
        $InstallRoot
    } elseif ($env:PCAI_MODULE_INSTALL_ROOT) {
        $env:PCAI_MODULE_INSTALL_ROOT
    } else {
        Join-Path $env:LOCALAPPDATA 'PowerShell\Modules'
    }

    return [System.IO.Path]::GetFullPath($resolvedRoot)
}

function Get-PcaiModuleResolutionTrace {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$ModuleName,
        [string]$RepoRoot = (Split-Path -Parent $PSScriptRoot),
        [string]$InstallRoot,
        [string[]]$AdditionalRoots,
        [string]$ExplicitManifest
    )

    $items = New-Object System.Collections.Generic.List[object]
    $seen = New-Object 'System.Collections.Generic.HashSet[string]' ([System.StringComparer]::OrdinalIgnoreCase)

    function Get-EnvironmentValue {
        param([string]$Name)

        return [Environment]::GetEnvironmentVariable($Name)
    }

    function Add-Trace {
        param(
            [string]$Path,
            [string]$Source
        )

        if ([string]::IsNullOrWhiteSpace($Path)) { return }

        $candidatePath = if ([System.IO.Path]::IsPathRooted($Path)) {
            [System.IO.Path]::GetFullPath($Path)
        } else {
            $Path
        }

        if (-not $seen.Add($candidatePath)) { return }

        $items.Add([PSCustomObject]@{
            ModuleName = $ModuleName
            Source     = $Source
            Path       = $candidatePath
            Exists     = (Test-Path -LiteralPath $candidatePath)
        })
    }

    if ($ExplicitManifest) {
        Add-Trace -Path $ExplicitManifest -Source 'explicit-manifest'
    }

    switch ($ModuleName) {
        'CargoTools' {
            foreach ($envName in @('CARGOTOOLS_MANIFEST', 'CARGOTOOLS_MODULE_MANIFEST')) {
                $envValue = Get-EnvironmentValue -Name $envName
                if ($envValue) {
                    Add-Trace -Path $envValue -Source "env:$envName"
                }
            }

            foreach ($envName in @('CARGOTOOLS_SOURCE_ROOT', 'CARGOTOOLS_MODULE_ROOT')) {
                $envValue = Get-EnvironmentValue -Name $envName
                if ($envValue) {
                    Add-Trace -Path (Join-Path $envValue "$ModuleName.psd1") -Source "env:$envName"
                }
            }
        }
        'PC-AI.Acceleration' {
            foreach ($envName in @('PCAI_ACCELERATION_MANIFEST', 'PCAI_ACCELERATION_MODULE_MANIFEST')) {
                $envValue = Get-EnvironmentValue -Name $envName
                if ($envValue) {
                    Add-Trace -Path $envValue -Source "env:$envName"
                }
            }

            foreach ($envName in @('PCAI_ACCELERATION_MODULE_ROOT')) {
                $envValue = Get-EnvironmentValue -Name $envName
                if ($envValue) {
                    Add-Trace -Path (Join-Path $envValue "$ModuleName.psd1") -Source "env:$envName"
                }
            }
        }
    }

    foreach ($root in @($AdditionalRoots | Where-Object { $_ })) {
        Add-Trace -Path (Join-Path $root "$ModuleName.psd1") -Source 'additional-root'
        Add-Trace -Path (Join-Path $root $ModuleName "$ModuleName.psd1") -Source 'additional-root'
    }

    if ($RepoRoot) {
        Add-Trace -Path (Join-Path $RepoRoot "Modules\$ModuleName\$ModuleName.psd1") -Source 'repo-source'
        Add-Trace -Path (Join-Path $RepoRoot "Release\PowerShell\PC-AI\Modules\$ModuleName\$ModuleName.psd1") -Source 'repo-release'
        Add-Trace -Path (Join-Path $RepoRoot "Release\PowerShell\PC-AI\$ModuleName.psd1") -Source 'repo-release-root'
    }

    $stableInstallRoot = Get-PcaiStableModuleInstallRoot -InstallRoot $InstallRoot
    Add-Trace -Path (Join-Path $stableInstallRoot "$ModuleName\$ModuleName.psd1") -Source 'stable-install-root'

    $oneDriveUserModules = Join-Path $env:USERPROFILE 'OneDrive\Documents\PowerShell\Modules'
    Add-Trace -Path (Join-Path $oneDriveUserModules "$ModuleName\$ModuleName.psd1") -Source 'onedrive-user-modules'

    $loadedModule = Get-Module -Name $ModuleName | Select-Object -First 1
    if ($loadedModule -and $loadedModule.Path) {
        Add-Trace -Path $loadedModule.Path -Source 'loaded-module'
    }

    foreach ($availableModule in (Get-Module -ListAvailable -Name $ModuleName | Sort-Object Version -Descending)) {
        if ($availableModule.Path) {
            Add-Trace -Path $availableModule.Path -Source 'psmodulepath'
        }
    }

    return $items.ToArray()
}

function Resolve-PcaiModuleManifestPath {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$ModuleName,
        [string]$RepoRoot = (Split-Path -Parent $PSScriptRoot),
        [string]$InstallRoot,
        [string[]]$AdditionalRoots,
        [string]$ExplicitManifest
    )

    $trace = Get-PcaiModuleResolutionTrace -ModuleName $ModuleName -RepoRoot $RepoRoot -InstallRoot $InstallRoot -AdditionalRoots $AdditionalRoots -ExplicitManifest $ExplicitManifest
    $match = $trace | Where-Object { $_.Exists } | Select-Object -First 1
    if ($match) {
        return $match.Path
    }

    return $null
}

function Import-PcaiResolvedModule {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$ModuleName,
        [string]$RepoRoot = (Split-Path -Parent $PSScriptRoot),
        [string]$InstallRoot,
        [string[]]$AdditionalRoots,
        [string]$ExplicitManifest,
        [switch]$Force
    )

    $loaded = Get-Module -Name $ModuleName | Select-Object -First 1
    if ($loaded -and -not $Force) {
        return $loaded
    }

    $manifestPath = Resolve-PcaiModuleManifestPath -ModuleName $ModuleName -RepoRoot $RepoRoot -InstallRoot $InstallRoot -AdditionalRoots $AdditionalRoots -ExplicitManifest $ExplicitManifest
    if (-not $manifestPath) {
        return $null
    }

    return Import-Module -Name $manifestPath -PassThru -Force:$Force.IsPresent -ErrorAction Stop
}
