#Requires -Version 5.1
<#
.SYNOPSIS
    Unified build orchestrator for PC_AI native components.

.DESCRIPTION
    Consolidates all build processes into a single entry point with well-defined
    output folders, clear progress messaging, and artifact manifests.

    Output Structure:
        .pcai/
        └── build/
            ├── artifacts/           # Final distributable binaries
            │   ├── pcai-llamacpp/   # llamacpp backend binaries
            │   ├── pcai-mistralrs/  # mistralrs backend binaries
            │   └── manifest.json    # Build manifest with hashes
            ├── logs/                # Build logs (timestamped)
            └── packages/            # Release packages (ZIPs)

.PARAMETER Component
    Which component(s) to build:
    - inference: pcai-inference (llamacpp + mistralrs backends)
    - llamacpp: pcai-inference llamacpp backend only
    - mistralrs: pcai-inference mistralrs backend only
    - functiongemma: FunctionGemma router runtime
    - all: All components (default)

.PARAMETER Configuration
    Build configuration: Debug or Release (default: Release)

.PARAMETER EnableCuda
    Enable CUDA GPU acceleration for supported backends.

.PARAMETER Clean
    Clean all build artifacts before building.

.PARAMETER Package
    Create distributable ZIP packages after building.

.PARAMETER SkipTests
    Skip running tests after build.

.PARAMETER Verbose
    Show detailed build output.

.EXAMPLE
    .\Build.ps1 -Component inference -EnableCuda
    Build pcai-inference with CUDA support.

.EXAMPLE
    .\Build.ps1 -Clean -Package
    Clean build all components and create release packages.

.EXAMPLE
    .\Build.ps1 -Component llamacpp -Configuration Debug
    Debug build of llamacpp backend only.
#>

[CmdletBinding()]
param(
    [ValidateSet('inference', 'llamacpp', 'mistralrs', 'functiongemma', 'tui', 'all')]
    [string]$Component = 'all',

    [ValidateSet('Debug', 'Release')]
    [string]$Configuration = 'Release',

    [switch]$EnableCuda,
    [switch]$Clean,
    [switch]$Package,
    [switch]$SkipTests,
    [switch]$Quiet
)

$ErrorActionPreference = 'Stop'
$script:StartTime = Get-Date
$script:ProjectRoot = $PSScriptRoot
$script:ArtifactsRoot = if ($env:PCAI_ARTIFACTS_ROOT) {
    $env:PCAI_ARTIFACTS_ROOT
} else {
    Join-Path $script:ProjectRoot '.pcai'
}
$script:BuildRoot = Join-Path $script:ArtifactsRoot 'build'
$script:BuildArtifactsDir = Join-Path $script:BuildRoot 'artifacts'
$script:BuildLogsDir = Join-Path $script:BuildRoot 'logs'
$script:BuildPackagesDir = Join-Path $script:BuildRoot 'packages'
$script:QuietMode = $Quiet.IsPresent

#region Output Formatting

function Write-BuildHeader {
    param([string]$Message)
    if ($script:QuietMode) { return }
    $line = '=' * 70
    Write-Host "`n$line" -ForegroundColor Cyan
    Write-Host " $Message" -ForegroundColor Cyan
    Write-Host "$line" -ForegroundColor Cyan
}

function Write-BuildPhase {
    param([string]$Phase, [string]$Description)
    if ($script:QuietMode) { return }
    $elapsed = (Get-Date) - $script:StartTime
    Write-Host "`n[$($elapsed.ToString('mm\:ss'))] " -ForegroundColor DarkGray -NoNewline
    Write-Host "PHASE: $Phase" -ForegroundColor Yellow
    if ($Description) {
        Write-Host "        $Description" -ForegroundColor DarkGray
    }
}

function Write-BuildStep {
    param([string]$Step, [string]$Status = 'running')
    if ($script:QuietMode -and $Status -notin @('warning', 'error')) { return }
    $symbol = switch ($Status) {
        'running' { '[..]' }
        'success' { '[OK]' }
        'warning' { '[!!]' }
        'error' { '[XX]' }
        'skip' { '[--]' }
        default { '[..]' }
    }
    $color = switch ($Status) {
        'running' { 'White' }
        'success' { 'Green' }
        'warning' { 'Yellow' }
        'error' { 'Red' }
        'skip' { 'DarkGray' }
        default { 'White' }
    }
    Write-Host "  $symbol " -ForegroundColor $color -NoNewline
    Write-Host $Step
}

function Write-BuildResult {
    param(
        [string]$Component,
        [bool]$Success,
        [TimeSpan]$Duration,
        [string[]]$Artifacts
    )
    $status = if ($Success) { 'SUCCESS' } else { 'FAILED' }
    $color = if ($Success) { 'Green' } else { 'Red' }

    if ($script:QuietMode) { return }

    Write-Host "`n  $Component build: " -NoNewline
    Write-Host $status -ForegroundColor $color -NoNewline
    Write-Host " ($($Duration.ToString('mm\:ss')))"

    if ($Artifacts -and $Artifacts.Count -gt 0) {
        Write-Host '  Artifacts:' -ForegroundColor DarkGray
        foreach ($artifact in $Artifacts) {
            Write-Host "    - $artifact" -ForegroundColor DarkGray
        }
    }
}

function Write-BuildSummary {
    param(
        [hashtable]$Results,
        [string]$ManifestPath
    )
    $elapsed = (Get-Date) - $script:StartTime
    $successCount = ($Results.Values | Where-Object { $_.Success }).Count
    $totalCount = $Results.Count

    Write-BuildHeader 'BUILD SUMMARY'

    Write-Host "`n  Total Time: $($elapsed.ToString('hh\:mm\:ss'))" -ForegroundColor Cyan
    Write-Host "  Components: $successCount / $totalCount succeeded" -ForegroundColor $(if ($successCount -eq $totalCount) { 'Green' } else { 'Yellow' })

    Write-Host "`n  Results:" -ForegroundColor White
    foreach ($name in $Results.Keys | Sort-Object) {
        $result = $Results[$name]
        $status = if ($result.Success) { 'OK' } else { 'FAILED' }
        $color = if ($result.Success) { 'Green' } else { 'Red' }
        Write-Host "    [$status] " -ForegroundColor $color -NoNewline
        Write-Host "$name ($($result.Duration.ToString('mm\:ss')))"
    }

    if ($ManifestPath -and (Test-Path $ManifestPath)) {
        Write-Host "`n  Manifest: $ManifestPath" -ForegroundColor DarkGray
    }

    if (Test-Path $script:BuildArtifactsDir) {
        Write-Host "  Artifacts: $script:BuildArtifactsDir" -ForegroundColor DarkGray
    }
    if (Test-Path $script:BuildLogsDir) {
        Write-Host "  Logs: $script:BuildLogsDir" -ForegroundColor DarkGray
    }
    if (Test-Path $script:BuildPackagesDir) {
        Write-Host "  Packages: $script:BuildPackagesDir" -ForegroundColor DarkGray
    }

    Write-Host ''
}

#endregion

#region Version Information

function Initialize-BuildVersion {
    Write-BuildPhase 'Version' 'Extracting git metadata'

    $versionScript = Join-Path $script:ProjectRoot 'Tools\Get-BuildVersion.ps1'
    if (Test-Path $versionScript) {
        $script:VersionInfo = & $versionScript -SetEnv -Quiet
        Write-BuildStep "Version: $($script:VersionInfo.Version)" 'success'
        Write-BuildStep "Git: $($script:VersionInfo.GitHashShort) ($($script:VersionInfo.GitBranch))" 'success'
        Write-BuildStep "Type: $($script:VersionInfo.BuildType)" 'success'
        return $script:VersionInfo
    } else {
        Write-BuildStep 'Version script not found, using defaults' 'warning'
        $env:PCAI_VERSION = '0.1.0+unknown'
        $env:PCAI_BUILD_VERSION = '0.1.0+unknown'
        return $null
    }
}

#endregion

#region Directory Setup

function Initialize-BuildDirectories {
    Write-BuildPhase 'Initialize' 'Setting up build directory structure'

    $dirs = @(
        $script:BuildRoot,
        $script:BuildArtifactsDir,
        (Join-Path $script:BuildArtifactsDir 'pcai-llamacpp'),
        (Join-Path $script:BuildArtifactsDir 'pcai-mistralrs'),
        (Join-Path $script:BuildArtifactsDir 'functiongemma'),
        (Join-Path $script:BuildArtifactsDir 'pcai-chattui'),
        $script:BuildLogsDir,
        $script:BuildPackagesDir
    )

    foreach ($path in $dirs) {
        if (-not (Test-Path $path)) {
            New-Item -ItemType Directory -Path $path -Force | Out-Null
            Write-BuildStep "Created $(Resolve-Path -LiteralPath $path)" 'success'
        }
    }

    return $true
}

function Clear-BuildArtifacts {
    Write-BuildPhase 'Clean' 'Removing previous build artifacts'

    $dirsToClean = @(
        $script:BuildArtifactsDir,
        $script:BuildLogsDir,
        $script:BuildPackagesDir
    )

    foreach ($path in $dirsToClean) {
        if (Test-Path $path) {
            Remove-Item $path -Recurse -Force -ErrorAction SilentlyContinue
            Write-BuildStep "Cleaned $path" 'success'
        }
    }

    # Clean Rust target directories if requested
    $rustTargets = @(
        'Native\pcai_core\target',
        'Deploy\rust-functiongemma-runtime\target',
        'Deploy\rust-functiongemma-train\target'
    )

    foreach ($target in $rustTargets) {
        $path = Join-Path $script:ProjectRoot $target
        if (Test-Path $path) {
            Write-BuildStep "Cleaning $target..." 'running'
            Remove-Item $path -Recurse -Force -ErrorAction SilentlyContinue
            Write-BuildStep "Cleaned $target" 'success'
        }
    }

    return $true
}

#endregion

#region Build Functions

function Invoke-InferenceBuild {
    param(
        [string]$Backend,
        [string]$Configuration,
        [bool]$EnableCuda
    )

    $componentStart = Get-Date
    $buildScript = Join-Path $script:ProjectRoot 'Native\pcai_core\pcai_inference\Invoke-PcaiBuild.ps1'

    if (-not (Test-Path $buildScript)) {
        Write-BuildStep "Build script not found: $buildScript" 'error'
        return @{ Success = $false; Duration = (Get-Date) - $componentStart; Artifacts = @() }
    }

    Write-BuildStep "Building pcai-inference ($Backend)..." 'running'

    # Prepare log file
    $timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    $logDir = $script:BuildLogsDir
    if (-not (Test-Path $logDir)) {
        New-Item -ItemType Directory -Path $logDir -Force | Out-Null
    }
    $logFile = Join-Path $logDir "build_${Backend}_$timestamp.log"

    # Build arguments
    $buildArgs = @{
        Backend       = $Backend
        Configuration = $Configuration
    }
    if ($EnableCuda) { $buildArgs['EnableCuda'] = $true }

    try {
        # Capture output to log file while showing progress
        $output = & $buildScript @buildArgs 2>&1 | Tee-Object -FilePath $logFile
        $success = $LASTEXITCODE -eq 0

        # Collect artifacts
        $artifacts = @()
        $targetDir = Join-Path $script:ProjectRoot "Native\pcai_core\pcai_inference\target\$($Configuration.ToLower())"

        $binName = if ($Backend -eq 'llamacpp') { 'pcai-llamacpp' } else { 'pcai-mistralrs' }
        $exePath = Join-Path $targetDir "$binName.exe"
        $dllPath = Join-Path $targetDir 'pcai_inference.dll'
        $libDllPath = Join-Path $targetDir 'pcai_inference_lib.dll'

        # Copy to artifacts directory
        $artifactDir = Join-Path $script:BuildArtifactsDir $binName
        if (-not (Test-Path $artifactDir)) {
            New-Item -ItemType Directory -Path $artifactDir -Force | Out-Null
        }

        if (Test-Path $exePath) {
            Copy-Item $exePath -Destination $artifactDir -Force
            $artifacts += "$binName.exe"
        }
        if (Test-Path $dllPath) {
            Copy-Item $dllPath -Destination $artifactDir -Force
            $artifacts += 'pcai_inference.dll'
        }
        if (Test-Path $libDllPath) {
            Copy-Item $libDllPath -Destination $artifactDir -Force
            $artifacts += 'pcai_inference_lib.dll'
        }

        $status = if ($success) { 'success' } else { 'error' }
        Write-BuildStep "pcai-inference ($Backend) build complete" $status

        return @{
            Success   = $success
            Duration  = (Get-Date) - $componentStart
            Artifacts = $artifacts
            LogFile   = $logFile
        }
    } catch {
        Write-BuildStep "pcai-inference ($Backend) build failed: $($_.Exception.Message)" 'error'
        return @{
            Success   = $false
            Duration  = (Get-Date) - $componentStart
            Artifacts = @()
            Error     = $_.Exception.Message
        }
    }
}

function Invoke-FunctionGemmaBuild {
    param([string]$Configuration)

    $componentStart = Get-Date
    $projectDir = Join-Path $script:ProjectRoot 'Deploy\rust-functiongemma-runtime'

    if (-not (Test-Path $projectDir)) {
        Write-BuildStep 'FunctionGemma project not found' 'warning'
        return @{ Success = $false; Duration = (Get-Date) - $componentStart; Artifacts = @() }
    }

    Write-BuildStep 'Building FunctionGemma runtime...' 'running'

    $timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    $logDir = $script:BuildLogsDir
    if (-not (Test-Path $logDir)) {
        New-Item -ItemType Directory -Path $logDir -Force | Out-Null
    }
    $logFile = Join-Path $logDir "build_functiongemma_$timestamp.log"

    try {
        Push-Location $projectDir

        $cargoArgs = @('build')
        if ($Configuration -eq 'Release') { $cargoArgs += '--release' }

        $output = & cargo @cargoArgs 2>&1 | Tee-Object -FilePath $logFile
        $success = $LASTEXITCODE -eq 0

        Pop-Location

        # Collect artifacts
        $artifacts = @()
        $configDir = if ($Configuration -eq 'Release') { 'release' } else { 'debug' }
        $targetDir = Join-Path $projectDir "target\$configDir"

        $artifactDir = Join-Path $script:BuildArtifactsDir 'functiongemma'
        if (-not (Test-Path $artifactDir)) {
            New-Item -ItemType Directory -Path $artifactDir -Force | Out-Null
        }

        # Copy binaries
        Get-ChildItem $targetDir -Filter '*.exe' -ErrorAction SilentlyContinue | ForEach-Object {
            Copy-Item $_.FullName -Destination $artifactDir -Force
            $artifacts += $_.Name
        }
        Get-ChildItem $targetDir -Filter '*.dll' -ErrorAction SilentlyContinue | ForEach-Object {
            Copy-Item $_.FullName -Destination $artifactDir -Force
            $artifacts += $_.Name
        }

        $status = if ($success) { 'success' } else { 'error' }
        Write-BuildStep 'FunctionGemma build complete' $status

        return @{
            Success   = $success
            Duration  = (Get-Date) - $componentStart
            Artifacts = $artifacts
            LogFile   = $logFile
        }
    } catch {
        Pop-Location -ErrorAction SilentlyContinue
        Write-BuildStep "FunctionGemma build failed: $($_.Exception.Message)" 'error'
        return @{
            Success   = $false
            Duration  = (Get-Date) - $componentStart
            Artifacts = @()
            Error     = $_.Exception.Message
        }
    }
}

function Invoke-TuiBuild {
    param([string]$Configuration)

    $componentStart = Get-Date
    $projectPath = Join-Path $script:ProjectRoot 'Native\PcaiChatTui\PcaiChatTui.csproj'

    if (-not (Test-Path $projectPath)) {
        Write-BuildStep "PcaiChatTui project not found: $projectPath" 'warning'
        return @{ Success = $false; Duration = (Get-Date) - $componentStart; Artifacts = @() }
    }

    Write-BuildStep 'Building PcaiChatTui...' 'running'

    $timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    $logDir = $script:BuildLogsDir
    if (-not (Test-Path $logDir)) {
        New-Item -ItemType Directory -Path $logDir -Force | Out-Null
    }
    $logFile = Join-Path $logDir "build_pcai_chattui_$timestamp.log"

    $publishRoot = Join-Path $script:ProjectRoot ".pcai\build\tmp\pcai-chattui-$($Configuration.ToLower())"

    try {
        $dotnetArgs = @(
            'publish',
            $projectPath,
            '-c', $Configuration,
            '-r', 'win-x64',
            '--self-contained', 'false',
            '-p:PublishSingleFile=false',
            '-o', $publishRoot
        )

        $output = & dotnet @dotnetArgs 2>&1 | Tee-Object -FilePath $logFile
        $success = $LASTEXITCODE -eq 0

        $artifacts = @()
        $artifactDir = Join-Path $script:BuildArtifactsDir 'pcai-chattui'
        if (-not (Test-Path $artifactDir)) {
            New-Item -ItemType Directory -Path $artifactDir -Force | Out-Null
        }

        if ($success -and (Test-Path $publishRoot)) {
            $filesToStage = Get-ChildItem $publishRoot -File -ErrorAction SilentlyContinue | Where-Object {
                $_.Extension -in @('.exe', '.dll', '.json', '.deps.json', '.runtimeconfig.json')
            }

            foreach ($file in $filesToStage) {
                Copy-Item $file.FullName -Destination $artifactDir -Force
                $artifacts += $file.Name
            }
        }

        $status = if ($success) { 'success' } else { 'error' }
        Write-BuildStep 'PcaiChatTui build complete' $status

        return @{
            Success   = $success
            Duration  = (Get-Date) - $componentStart
            Artifacts = $artifacts
            LogFile   = $logFile
        }
    } catch {
        Write-BuildStep "PcaiChatTui build failed: $($_.Exception.Message)" 'error'
        return @{
            Success   = $false
            Duration  = (Get-Date) - $componentStart
            Artifacts = @()
            Error     = $_.Exception.Message
        }
    }
}

#endregion

#region Manifest Generation

function New-BuildManifest {
    param(
        [hashtable]$Results,
        [string]$Configuration,
        [bool]$EnableCuda
    )

    Write-BuildPhase 'Manifest' 'Generating build manifest'

    $artifactsDir = $script:BuildArtifactsDir
    $manifestPath = Join-Path $artifactsDir 'manifest.json'

    $artifacts = @()

    # Collect all artifacts with hashes
    Get-ChildItem $artifactsDir -Recurse -File | Where-Object { $_.Name -ne 'manifest.json' } | ForEach-Object {
        $relativePath = $_.FullName.Substring($artifactsDir.Length + 1)
        $hash = (Get-FileHash $_.FullName -Algorithm SHA256).Hash
        $artifacts += @{
            path     = $relativePath
            size     = $_.Length
            sha256   = $hash
            modified = $_.LastWriteTimeUtc.ToString('o')
        }
    }

    # Get version info from environment or git
    $version = if ($env:PCAI_VERSION) { $env:PCAI_VERSION } else { '0.1.0+unknown' }
    $semver = if ($env:PCAI_SEMVER) { $env:PCAI_SEMVER } else { '0.1.0' }

    $fallbackGitCommit = ((git rev-parse HEAD 2>$null) -replace '\s', '')
    if (-not $fallbackGitCommit) { $fallbackGitCommit = 'unknown' }
    $fallbackGitCommitShort = if ($fallbackGitCommit.Length -ge 7) {
        $fallbackGitCommit.Substring(0, 7)
    } else {
        $fallbackGitCommit
    }

    $manifest = @{
        manifestVersion    = '2.0'
        pcaiVersion        = $version
        semver             = $semver
        buildTime          = (Get-Date).ToUniversalTime().ToString('o')
        buildTimestampUnix = [int][double]::Parse((Get-Date -UFormat %s))
        configuration      = $Configuration
        cuda               = $EnableCuda
        platform           = 'win-x64'
        gitCommit          = if ($env:PCAI_GIT_HASH) { $env:PCAI_GIT_HASH } else { $fallbackGitCommit }
        gitCommitShort     = if ($env:PCAI_GIT_HASH_SHORT) { $env:PCAI_GIT_HASH_SHORT } else { $fallbackGitCommitShort }
        gitBranch          = if ($env:PCAI_GIT_BRANCH) { $env:PCAI_GIT_BRANCH } else { (git rev-parse --abbrev-ref HEAD 2>$null) -replace '\s', '' }
        gitTag             = $env:PCAI_GIT_TAG
        buildType          = if ($env:PCAI_BUILD_TYPE) { $env:PCAI_BUILD_TYPE } else { 'dev' }
        components         = @{}
        artifacts          = $artifacts
    }

    foreach ($name in $Results.Keys) {
        $result = $Results[$name]
        $manifest.components[$name] = @{
            success   = $result.Success
            duration  = $result.Duration.TotalSeconds
            artifacts = $result.Artifacts
        }
    }

    $manifest | ConvertTo-Json -Depth 10 | Out-File $manifestPath -Encoding UTF8
    Write-BuildStep 'Created manifest.json' 'success'

    return $manifestPath
}

#endregion

#region Package Creation

function New-ReleasePackages {
    param([string]$Configuration, [bool]$EnableCuda)

    Write-BuildPhase 'Package' 'Creating release packages'

    $artifactsDir = $script:BuildArtifactsDir
    $packagesDir = $script:BuildPackagesDir
    $variant = if ($EnableCuda) { 'cuda' } else { 'cpu' }

    $packages = @()

    # Package each component
    $components = @('pcai-llamacpp', 'pcai-mistralrs', 'functiongemma', 'pcai-chattui')

    foreach ($component in $components) {
        $componentDir = Join-Path $artifactsDir $component
        if (-not (Test-Path $componentDir) -or (Get-ChildItem $componentDir).Count -eq 0) {
            continue
        }

        $packageName = "$component-$variant-win64.zip"
        $packagePath = Join-Path $packagesDir $packageName

        Write-BuildStep "Creating $packageName..." 'running'

        Compress-Archive -Path "$componentDir\*" -DestinationPath $packagePath -Force
        $packages += $packagePath

        Write-BuildStep "Created $packageName" 'success'
    }

    return $packages
}

#endregion

#region Main Execution

Write-BuildHeader 'PC_AI Build System'
Write-Host "  Component:     $Component" -ForegroundColor White
Write-Host "  Configuration: $Configuration" -ForegroundColor White
Write-Host "  CUDA:          $(if ($EnableCuda) { 'Enabled' } else { 'Disabled' })" -ForegroundColor White
Write-Host "  Clean:         $(if ($Clean) { 'Yes' } else { 'No' })" -ForegroundColor White
Write-Host "  Package:       $(if ($Package) { 'Yes' } else { 'No' })" -ForegroundColor White
Write-Host "  Artifacts Root: $script:ArtifactsRoot" -ForegroundColor White

# Clean if requested
if ($Clean) {
    Clear-BuildArtifacts
}

# Initialize directories
Initialize-BuildDirectories

# Initialize version information (sets PCAI_* environment variables)
$versionInfo = Initialize-BuildVersion

# Determine what to build
$buildTargets = switch ($Component) {
    'inference' { @('llamacpp', 'mistralrs') }
    'llamacpp' { @('llamacpp') }
    'mistralrs' { @('mistralrs') }
    'functiongemma' { @('functiongemma') }
    'tui' { @('tui') }
    'all' { @('llamacpp', 'mistralrs', 'functiongemma', 'tui') }
}

$results = @{}

# Build inference backends
Write-BuildPhase 'Build' 'Compiling native components'

foreach ($target in $buildTargets) {
    if ($target -eq 'functiongemma') {
        $result = Invoke-FunctionGemmaBuild -Configuration $Configuration
        $results['functiongemma'] = $result
        Write-BuildResult 'FunctionGemma' $result.Success $result.Duration $result.Artifacts
    } elseif ($target -eq 'tui') {
        $result = Invoke-TuiBuild -Configuration $Configuration
        $results['pcai-chattui'] = $result
        Write-BuildResult 'pcai-chattui' $result.Success $result.Duration $result.Artifacts
    } else {
        $result = Invoke-InferenceBuild -Backend $target -Configuration $Configuration -EnableCuda $EnableCuda
        $results["pcai-$target"] = $result
        Write-BuildResult "pcai-$target" $result.Success $result.Duration $result.Artifacts
    }
}

if ($SkipTests) {
    Write-BuildStep 'Post-build tests skipped by request' 'skip'
} else {
    Write-BuildStep 'No post-build test phase configured in Build.ps1 (use Tests/.pester.ps1 or Native/build.ps1 -Test)' 'warning'
}

# Generate manifest
$manifestPath = New-BuildManifest -Results $results -Configuration $Configuration -EnableCuda $EnableCuda

# Create packages if requested
if ($Package) {
    $packages = New-ReleasePackages -Configuration $Configuration -EnableCuda $EnableCuda
}

# Summary
Write-BuildSummary -Results $results -ManifestPath $manifestPath

# Exit with appropriate code
$failedCount = ($results.Values | Where-Object { -not $_.Success }).Count
exit $failedCount

#endregion
