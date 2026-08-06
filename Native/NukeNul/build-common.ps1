Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Import-NukeNulCargoTools {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$RepoRoot
    )

    if (Get-Command Get-BuildVersionInfo -ErrorAction SilentlyContinue) {
        return $true
    }

    foreach ($candidate in @(
        'CargoTools',
        'C:\Users\david\Documents\PowerShell\Modules\CargoTools\CargoTools.psd1',
        'C:\Users\david\OneDrive\Documents\PowerShell\Modules\CargoTools\CargoTools.psd1'
    )) {
        try {
            if ($candidate -eq 'CargoTools') {
                Import-Module CargoTools -ErrorAction Stop | Out-Null
                break
            }

            if (Test-Path -LiteralPath $candidate) {
                Import-Module $candidate -ErrorAction Stop | Out-Null
                break
            }
        } catch {
        }
    }

    return [bool](Get-Command Get-BuildVersionInfo -ErrorAction SilentlyContinue)
}

function Get-NukeNulBuildVersionInfo {
    [CmdletBinding()]
    [OutputType([pscustomobject])]
    param(
        [Parameter(Mandatory)]
        [string]$RepoRoot,

        [string]$DefaultVersion = '0.1.0'
    )

    if (Import-NukeNulCargoTools -RepoRoot $RepoRoot) {
        return Get-BuildVersionInfo -RepoRoot $RepoRoot -DefaultVersion $DefaultVersion
    }

    return [pscustomobject]@{
        Version = $DefaultVersion
        SemVer = $DefaultVersion
        AssemblyVersion = "$DefaultVersion.0"
        FileVersion = "$DefaultVersion.0"
        InformationalVersion = $DefaultVersion
        ReleaseTag = "v$DefaultVersion"
        GitDescribe = ''
        GitHash = 'unknown'
        GitHashShort = 'unknown'
        GitBranch = 'unknown'
        CommitsSinceTag = 0
        Timestamp = [DateTime]::UtcNow.ToString('yyyy-MM-ddTHH:mm:ssZ')
        TimestampUnix = [int][double]::Parse((Get-Date -UFormat %s))
        IsDirty = $false
        BuildType = 'dev'
        RepoRoot = $RepoRoot
    }
}

function Set-NukeNulBuildEnvironment {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [pscustomobject]$VersionInfo
    )

    if (Get-Command Set-BuildVersionEnvironment -ErrorAction SilentlyContinue) {
        Set-BuildVersionEnvironment -VersionInfo $VersionInfo -Prefixes @('BUILD', 'NUKENUL') | Out-Null
    } else {
        $env:BUILD_VERSION = $VersionInfo.Version
        $env:BUILD_SEMVER = $VersionInfo.SemVer
        $env:BUILD_ASSEMBLY_VERSION = $VersionInfo.AssemblyVersion
        $env:BUILD_FILE_VERSION = $VersionInfo.FileVersion
        $env:BUILD_INFORMATIONAL_VERSION = $VersionInfo.InformationalVersion
        $env:BUILD_RELEASE_TAG = $VersionInfo.ReleaseTag
        $env:NUKENUL_VERSION = $VersionInfo.Version
        $env:NUKENUL_SEMVER = $VersionInfo.SemVer
        $env:NUKENUL_RELEASE_TAG = $VersionInfo.ReleaseTag
    }
}

function Resolve-NukeNulCargoOutputDirectory {
    [CmdletBinding()]
    [OutputType([string])]
    param(
        [Parameter(Mandatory)]
        [string]$ProjectDir,

        [ValidateSet('Debug', 'Release')]
        [string]$Configuration = 'Release'
    )

    if (Get-Command Resolve-CargoTargetDirectory -ErrorAction SilentlyContinue) {
        return Resolve-CargoTargetDirectory -ProjectDir $ProjectDir -ManifestPath (Join-Path $ProjectDir 'Cargo.toml') -Configuration $Configuration
    }

    $profile = if ($Configuration -eq 'Debug') { 'debug' } else { 'release' }
    return Join-Path $ProjectDir "target\$profile"
}

function Publish-NukeNulArtifact {
    [CmdletBinding()]
    [OutputType([pscustomobject])]
    param(
        [Parameter(Mandatory)]
        [string]$SourcePath,

        [Parameter(Mandatory)]
        [string]$DestinationDirectory,

        [string]$DestinationFileName,

        [pscustomobject]$VersionInfo,

        [string]$ArtifactKind = 'binary'
    )

    if (Get-Command Publish-BuildArtifact -ErrorAction SilentlyContinue) {
        return Publish-BuildArtifact -SourcePath $SourcePath -DestinationDirectory $DestinationDirectory -DestinationFileName $DestinationFileName -VersionInfo $VersionInfo -ArtifactKind $ArtifactKind
    }

    if (-not (Test-Path -LiteralPath $DestinationDirectory)) {
        New-Item -ItemType Directory -Path $DestinationDirectory -Force | Out-Null
    }

    if (-not $DestinationFileName) {
        $DestinationFileName = [System.IO.Path]::GetFileName($SourcePath)
    }

    $destinationPath = Join-Path $DestinationDirectory $DestinationFileName
    $resolvedSourcePath = (Resolve-Path -LiteralPath $SourcePath).Path
    $resolvedDestinationPath = [System.IO.Path]::GetFullPath($destinationPath)
    if (-not [System.StringComparer]::OrdinalIgnoreCase.Equals($resolvedSourcePath, $resolvedDestinationPath)) {
        Copy-Item -LiteralPath $SourcePath -Destination $destinationPath -Force
    }

    return [pscustomobject]@{
        DestinationPath = $destinationPath
        ManifestPath = $null
        FileName = [System.IO.Path]::GetFileName($destinationPath)
    }
}
