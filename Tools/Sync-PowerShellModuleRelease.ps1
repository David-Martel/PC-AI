#Requires -Version 7.0

[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [string]$RepoRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$ModuleName = 'PC-AI',
    [string]$ReleaseRoot = (Join-Path (Split-Path -Parent $PSScriptRoot) 'Release\PowerShell'),
    [string]$ReleaseScriptPath = (Join-Path $PSScriptRoot 'New-PcaiPowerShellRelease.ps1'),
    [string]$ValidationScriptPath = (Join-Path $PSScriptRoot 'Test-PcaiReleaseModule.ps1'),
    [string[]]$DestinationRoots,
    [ValidateSet('Auto', 'Always', 'Never')]
    [string]$BuildMode = 'Auto',
    [switch]$SkipValidation,
    [string]$Trigger = 'manual',
    [switch]$Quiet
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

. (Join-Path $PSScriptRoot 'PcaiModuleBootstrap.ps1')

function Write-ReleaseSyncMessage {
    param(
        [Parameter(Mandatory)]
        [string]$Message,
        [string]$Color = 'DarkGray'
    )

    if (-not $Quiet) {
        Write-Host "[release-sync] $Message" -ForegroundColor $Color
    }
}

function Get-PathTreeLastWriteTimeUtc {
    param(
        [Parameter(Mandatory)]
        [string]$Path
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        return [datetime]::MinValue
    }

    $item = Get-Item -LiteralPath $Path -Force
    $latest = $item.LastWriteTimeUtc
    if ($item.PSIsContainer) {
        foreach ($child in Get-ChildItem -LiteralPath $Path -Force -Recurse -ErrorAction SilentlyContinue) {
            if ($child.LastWriteTimeUtc -gt $latest) {
                $latest = $child.LastWriteTimeUtc
            }
        }
    }

    return $latest
}

function Test-ReleaseRefreshRequired {
    param(
        [Parameter(Mandatory)]
        [string]$ReleaseModuleRoot,
        [Parameter(Mandatory)]
        [string[]]$SourceInputs
    )

    if (-not (Test-Path -LiteralPath $ReleaseModuleRoot)) {
        return $true
    }

    $releaseTime = Get-PathTreeLastWriteTimeUtc -Path $ReleaseModuleRoot
    foreach ($inputPath in $SourceInputs) {
        if (-not $inputPath) {
            continue
        }

        $inputTime = Get-PathTreeLastWriteTimeUtc -Path $inputPath
        if ($inputTime -gt $releaseTime) {
            return $true
        }
    }

    return $false
}

function Get-DefaultDestinationRoots {
    $roots = New-Object System.Collections.Generic.List[string]

    $oneDriveModulesRoot = Join-Path $env:USERPROFILE 'OneDrive\Documents\PowerShell\Modules'
    if (Test-Path -LiteralPath (Split-Path -Parent $oneDriveModulesRoot)) {
        $roots.Add($oneDriveModulesRoot)
    }

    $documentsModulesRoot = Join-Path $env:USERPROFILE 'Documents\PowerShell\Modules'
    $roots.Add($documentsModulesRoot)

    $stableInstallRoot = Get-PcaiStableModuleInstallRoot
    if ($stableInstallRoot) {
        $roots.Add($stableInstallRoot)
    }

    return $roots.ToArray() | Select-Object -Unique
}

function Invoke-RobocopyMirror {
    param(
        [Parameter(Mandatory)]
        [string]$SourcePath,
        [Parameter(Mandatory)]
        [string]$DestinationPath
    )

    $destinationParent = Split-Path -Parent $DestinationPath
    if ($destinationParent) {
        New-Item -ItemType Directory -Path $destinationParent -Force | Out-Null
    }

    $arguments = @(
        $SourcePath,
        $DestinationPath,
        '/MIR',
        '/R:1',
        '/W:1',
        '/NFL',
        '/NDL',
        '/NJH',
        '/NJS',
        '/NP',
        '/XJ'
    )

    & robocopy @arguments | Out-Null
    $exitCode = $LASTEXITCODE
    if ($exitCode -gt 7) {
        throw "robocopy failed for '$DestinationPath' with exit code $exitCode"
    }

    return $exitCode
}

$RepoRoot = [System.IO.Path]::GetFullPath($RepoRoot)
$ReleaseRoot = [System.IO.Path]::GetFullPath($ReleaseRoot)
$ReleaseModuleRoot = Join-Path $ReleaseRoot $ModuleName
$ReleaseManifestPath = Join-Path $ReleaseModuleRoot "$ModuleName.psd1"

$sourceInputs = @(
    (Join-Path $RepoRoot 'Modules'),
    (Join-Path $RepoRoot 'Config'),
    (Join-Path $RepoRoot 'LICENSE'),
    $ReleaseScriptPath,
    (Join-Path $PSScriptRoot 'Get-BuildVersion.ps1')
)

$shouldBuild = switch ($BuildMode) {
    'Always' { $true }
    'Never' { $false }
    default { Test-ReleaseRefreshRequired -ReleaseModuleRoot $ReleaseModuleRoot -SourceInputs $sourceInputs }
}

$buildResult = $null
if ($shouldBuild) {
    if (-not (Test-Path -LiteralPath $ReleaseScriptPath)) {
        throw "Release script not found: $ReleaseScriptPath"
    }

    Write-ReleaseSyncMessage "Refreshing $ModuleName release bundle ($Trigger)" 'Cyan'
    $buildResult = & $ReleaseScriptPath -Clean
}

if (-not (Test-Path -LiteralPath $ReleaseManifestPath)) {
    throw "Release manifest not found: $ReleaseManifestPath"
}

$validationResult = $null
if (-not $SkipValidation -and (Test-Path -LiteralPath $ValidationScriptPath)) {
    Write-ReleaseSyncMessage "Validating $ModuleName release bundle" 'Cyan'
    $validationResult = & $ValidationScriptPath -ReleaseModulePath $ReleaseManifestPath
    if (-not $validationResult.Passed) {
        $missingCommands = @($validationResult.MissingCommands) -join ', '
        throw "Release validation failed. Missing commands: $missingCommands"
    }
}

$commit = $null
try {
    $commit = (git -C $RepoRoot rev-parse --short HEAD 2>$null).Trim()
} catch {
    $commit = $null
}

$resolvedDestinations = if ($DestinationRoots -and $DestinationRoots.Count -gt 0) {
    $DestinationRoots | Where-Object { $_ } | Select-Object -Unique
} else {
    Get-DefaultDestinationRoots
}

$syncResults = New-Object System.Collections.Generic.List[object]
foreach ($destinationRoot in $resolvedDestinations) {
    $destinationRootFull = [System.IO.Path]::GetFullPath($destinationRoot)
    $destinationPath = Join-Path $destinationRootFull $ModuleName

    if ($destinationPath -eq $ReleaseModuleRoot) {
        continue
    }

    $releaseTime = Get-PathTreeLastWriteTimeUtc -Path $ReleaseModuleRoot
    $destinationTime = Get-PathTreeLastWriteTimeUtc -Path $destinationPath
    $needsSync = $destinationTime -lt $releaseTime

    if ($needsSync -and $PSCmdlet.ShouldProcess($destinationPath, "Mirror $ModuleName release bundle")) {
        Write-ReleaseSyncMessage "Mirroring release to $destinationPath" 'Green'
        $null = Invoke-RobocopyMirror -SourcePath $ReleaseModuleRoot -DestinationPath $destinationPath
    } elseif (-not $needsSync) {
        Write-ReleaseSyncMessage "Destination already current: $destinationPath"
    }

    $syncResults.Add([pscustomobject]@{
        DestinationRoot = $destinationRootFull
        DestinationPath = $destinationPath
        Updated = $needsSync
    })
}

[pscustomobject]@{
    RepoRoot = $RepoRoot
    ModuleName = $ModuleName
    Trigger = $Trigger
    Commit = $commit
    BuiltRelease = [bool]$shouldBuild
    BuildResult = $buildResult
    ValidationResult = $validationResult
    ReleaseModuleRoot = $ReleaseModuleRoot
    SyncResults = $syncResults.ToArray()
}
