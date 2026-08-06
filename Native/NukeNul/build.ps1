#Requires -Version 5.1

[CmdletBinding()]
param(
    [ValidateSet('Debug', 'Release')]
    [string]$Configuration = 'Release',

    [switch]$Clean,
    [switch]$SkipRust,
    [switch]$SkipCSharp
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Import-NukeNulCargoTools {
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
            } elseif (Test-Path -LiteralPath $candidate) {
                Import-Module $candidate -ErrorAction Stop | Out-Null
            } else {
                continue
            }

            if (Get-Command Get-BuildVersionInfo -ErrorAction SilentlyContinue) {
                return $true
            }
        } catch {
        }
    }

    return $false
}

function Get-NukeNulVersionInfo {
    if (Import-NukeNulCargoTools) {
        $versionInfo = Get-BuildVersionInfo -RepoRoot $PSScriptRoot -DefaultVersion '0.1.0'
        Set-BuildVersionEnvironment -VersionInfo $versionInfo -Prefixes @('BUILD', 'NUKENUL') | Out-Null
        return $versionInfo
    }

    return [pscustomobject]@{
        Version = '0.1.0'
        SemVer = '0.1.0'
        AssemblyVersion = '0.1.0.0'
        FileVersion = '0.1.0.0'
        InformationalVersion = '0.1.0 (v0.1.0)'
        ReleaseTag = 'v0.1.0'
        GitHashShort = 'unknown'
    }
}

$projectRoot = $PSScriptRoot
$rustRoot = Join-Path $projectRoot 'nuker_core'
$dotnetProject = Join-Path $projectRoot 'NukeNul.csproj'
$publishRoot = Join-Path $projectRoot "bin\$Configuration\net8.0\win-x64"
$versionInfo = Get-NukeNulVersionInfo

if ($Clean) {
    if (-not $SkipRust -and (Test-Path -LiteralPath $rustRoot)) {
        Push-Location $rustRoot
        try {
            cargo clean
        } finally {
            Pop-Location
        }
    }

    if (-not $SkipCSharp -and (Test-Path -LiteralPath $dotnetProject)) {
        & dotnet clean $dotnetProject -c $Configuration
        if ($LASTEXITCODE -ne 0) {
            throw "dotnet clean failed with exit code $LASTEXITCODE"
        }
    }
}

if (-not $SkipRust) {
    $profile = if ($Configuration -eq 'Release') { 'release' } else { 'debug' }
    Push-Location $rustRoot
    try {
        & (Join-Path $rustRoot 'build.ps1') -Profile $profile -Copy
        if ($LASTEXITCODE -ne 0) {
            throw "nuker_core build failed with exit code $LASTEXITCODE"
        }
    } finally {
        Pop-Location
    }
}

if (-not $SkipCSharp) {
    $dotnetArgs = @(
        'publish',
        $dotnetProject,
        '-c', $Configuration,
        '-r', 'win-x64',
        '--self-contained', 'false',
        '-p:PublishSingleFile=false',
        '-o', $publishRoot,
        "-p:Version=$($versionInfo.SemVer)",
        "-p:AssemblyVersion=$($versionInfo.AssemblyVersion)",
        "-p:FileVersion=$($versionInfo.FileVersion)",
        "-p:InformationalVersion=$($versionInfo.InformationalVersion)"
    )

    & dotnet @dotnetArgs
    if ($LASTEXITCODE -ne 0) {
        throw "dotnet publish failed with exit code $LASTEXITCODE"
    }

    $nukerCoreDll = Join-Path $projectRoot 'nuker_core.dll'
    if (Test-Path -LiteralPath $nukerCoreDll) {
        if (Get-Command Publish-BuildArtifact -ErrorAction SilentlyContinue) {
            Publish-BuildArtifact -SourcePath $nukerCoreDll -DestinationDirectory $publishRoot -DestinationFileName 'nuker_core.dll' -VersionInfo $versionInfo -ArtifactKind 'native-rust' | Out-Null
        } else {
            Copy-Item -LiteralPath $nukerCoreDll -Destination (Join-Path $publishRoot 'nuker_core.dll') -Force
        }
    }

    $exePath = Join-Path $publishRoot 'NukeNul.exe'
    if ((Test-Path -LiteralPath $exePath) -and (Get-Command Publish-BuildArtifact -ErrorAction SilentlyContinue)) {
        Publish-BuildArtifact -SourcePath $exePath -DestinationDirectory $publishRoot -DestinationFileName 'NukeNul.exe' -VersionInfo $versionInfo -ArtifactKind 'managed-dotnet' | Out-Null
    }
}

Write-Host "NukeNul build complete: $publishRoot" -ForegroundColor Green
