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
    - functiongemma: FunctionGemma runtime + train crates
    - functiongemma-router-data: Generate FunctionGemma router training dataset
    - functiongemma-token-cache: Build FunctionGemma token cache artifacts
    - functiongemma-train: Run FunctionGemma training via rust-functiongemma-train
    - functiongemma-eval: Run FunctionGemma evaluation harness via Rust CLI
    - tui: PcaiChatTui .NET chat terminal UI
    - pcainative: PcaiNative .NET interop wrapper
    - servicehost: PcaiServiceHost .NET host binary
    - nukenul: Hybrid Rust/C# Native/NukeNul utility
    - native: .NET native toolchain (pcainative + servicehost + tui)
    - all: All components (default)

.PARAMETER Configuration
    Build configuration: Debug or Release (default: Release)

.PARAMETER EnableCuda
    Enable CUDA GPU acceleration for supported backends.

.PARAMETER Clean
    Clean all build artifacts before building.

.PARAMETER Package
    Create distributable ZIP packages after building.

.PARAMETER RunTests
    Run component-aligned post-build tests.

.PARAMETER SkipTests
    Skip running tests after build even if -RunTests is provided.

.PARAMETER Deploy
    Create a deployable aggregate bundle in addition to per-component packages.

.PARAMETER CargoTools
    CargoTools integration mode:
    - auto: use CargoTools when available (default)
    - enabled: require CargoTools and fail if unavailable
    - disabled: do not import/use CargoTools defaults

.PARAMETER CargoPreflight
    Enable CargoTools preflight checks for Rust builds.

.PARAMETER CargoPreflightMode
    CargoTools preflight mode: check, clippy, fmt, all.

.PARAMETER SyncCargoDefaults
    Persist CargoTools default Cargo/rustfmt/clippy settings to user config files.
    Disabled by default to avoid mutating developer workstation state implicitly.

.PARAMETER FunctionGemmaArgs
    Optional passthrough arguments for FunctionGemma operation components:
    functiongemma-router-data, functiongemma-token-cache, functiongemma-train, functiongemma-eval.

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

.EXAMPLE
    .\Build.ps1 -Component functiongemma-router-data
    Generate FunctionGemma router dataset using unified orchestration defaults.

.EXAMPLE
    .\Build.ps1 -Component functiongemma-train
    Run FunctionGemma training using default config and dataset paths.

.EXAMPLE
    .\Build.ps1 -Component nukenul
    Build Native/NukeNul hybrid Rust + C# utility.
#>

[CmdletBinding()]
param(
    [ValidateSet('inference', 'llamacpp', 'mistralrs', 'functiongemma', 'functiongemma-router-data', 'functiongemma-token-cache', 'functiongemma-train', 'functiongemma-eval', 'tui', 'pcainative', 'servicehost', 'nukenul', 'native', 'all')]
    [string]$Component = 'all',

    [ValidateSet('Debug', 'Release')]
    [string]$Configuration = 'Release',

    [switch]$EnableCuda,
    [switch]$Clean,
    [switch]$Package,
    [switch]$RunTests,
    [switch]$SkipTests,
    [switch]$Deploy,
    [ValidateSet('auto', 'enabled', 'disabled')]
    [string]$CargoTools = 'auto',
    [switch]$CargoPreflight,
    [ValidateSet('check', 'clippy', 'fmt', 'all')]
    [string]$CargoPreflightMode = 'check',
    [switch]$SyncCargoDefaults,
    [string[]]$FunctionGemmaArgs = @(),
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
$script:BuildDeployDir = Join-Path $script:BuildRoot 'deploy'
$script:QuietMode = $Quiet.IsPresent
$script:CargoToolsEnabled = $false
$script:RustBuildWrapper = Join-Path $script:ProjectRoot 'Tools\Invoke-RustBuild.ps1'

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
    if (Test-Path $script:BuildDeployDir) {
        Write-Host "  Deploy: $script:BuildDeployDir" -ForegroundColor DarkGray
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

#region Toolchain Setup

function Resolve-PowerShellExecutable {
    $pwshCmd = Get-Command pwsh -ErrorAction SilentlyContinue
    if ($pwshCmd) { return $pwshCmd.Source }

    $powershellCmd = Get-Command powershell -ErrorAction SilentlyContinue
    if ($powershellCmd) { return $powershellCmd.Source }

    throw 'Neither pwsh nor powershell is available on PATH.'
}

function Initialize-CargoToolsDefaults {
    Write-BuildPhase 'Toolchain' 'Initializing CargoTools defaults and build environment'

    if ($CargoTools -eq 'disabled') {
        Write-BuildStep 'CargoTools integration disabled by request' 'skip'
        return $false
    }

    $cargoToolsModule = Get-Module -ListAvailable CargoTools
    if (-not $cargoToolsModule) {
        if ($CargoTools -eq 'enabled') {
            Write-BuildStep 'CargoTools module required but not installed' 'error'
            throw 'CargoTools integration was explicitly enabled but the module was not found.'
        }
        Write-BuildStep 'CargoTools module not found; using direct tooling defaults' 'warning'
        return $false
    }

    if (-not (Get-Module CargoTools)) {
        Import-Module CargoTools -ErrorAction Stop
    }

    $cacheRoot = if ($env:PCAI_CACHE_ROOT) {
        $env:PCAI_CACHE_ROOT
    } elseif (Test-Path 'T:\RustCache') {
        'T:\RustCache'
    } else {
        Join-Path $script:ArtifactsRoot 'cache'
    }

    try {
        $callerTargetDir = $env:CARGO_TARGET_DIR

        if ($SyncCargoDefaults) {
            Initialize-RustDefaults -Scope Cargo | Out-Null
            Write-BuildStep 'CargoTools defaults synchronized to user Cargo config' 'success'
        } else {
            Write-BuildStep 'CargoTools defaults not persisted (use -SyncCargoDefaults to write config files)' 'skip'
        }
        Initialize-CargoEnv -CacheRoot $cacheRoot
        if ($callerTargetDir) {
            $env:CARGO_TARGET_DIR = $callerTargetDir
            Write-BuildStep "Preserved caller CARGO_TARGET_DIR: $callerTargetDir" 'success'
        }

        if ($CargoPreflight) {
            $env:CARGO_PREFLIGHT = '1'
            $env:CARGO_PREFLIGHT_MODE = $CargoPreflightMode
            Write-BuildStep "Cargo preflight enabled ($CargoPreflightMode)" 'success'
        }

        $script:CargoToolsEnabled = $true
        Write-BuildStep "CargoTools initialized (cache root: $cacheRoot)" 'success'
        if ($env:RUSTC_WRAPPER) {
            Write-BuildStep "Rust compiler wrapper: $($env:RUSTC_WRAPPER)" 'success'
        }

        if (Get-Command Start-SccacheServer -Module CargoTools -ErrorAction SilentlyContinue) {
            Start-SccacheServer | Out-Null
            Write-BuildStep 'sccache server initialized via CargoTools' 'success'
        }
    } catch {
        if ($CargoTools -eq 'enabled') {
            Write-BuildStep "CargoTools initialization failed: $($_.Exception.Message)" 'error'
            throw
        }
        Write-BuildStep "CargoTools initialization failed; continuing without module defaults: $($_.Exception.Message)" 'warning'
        $script:CargoToolsEnabled = $false
    }

    return $script:CargoToolsEnabled
}

function Invoke-RustBuildCommand {
    param(
        [Parameter(Mandatory)]
        [string]$Path,
        [Parameter(Mandatory)]
        [string[]]$CargoArgs,
        [Parameter(Mandatory)]
        [string]$LogFile
    )

    if (-not (Test-Path $Path)) {
        throw "Rust project path not found: $Path"
    }

    if (Test-Path $script:RustBuildWrapper) {
        $shellExe = Resolve-PowerShellExecutable
        $wrapperArgs = @('-NoProfile', '-File', $script:RustBuildWrapper, '-Path', $Path, '-CargoArgs') + $CargoArgs

        if ($env:CARGO_USE_LLD -eq '1') {
            $wrapperArgs += '-UseLld'
        }
        if ($env:PCAI_DISABLE_CACHE -eq '1') {
            $wrapperArgs += '-DisableCache'
        }
        if ($CargoPreflight) {
            $wrapperArgs += @('-Preflight', '-PreflightMode', $CargoPreflightMode, '-PreflightBlocking')
        }

        & $shellExe @wrapperArgs 2>&1 | Tee-Object -FilePath $LogFile | Out-Null
        return ($LASTEXITCODE -eq 0)
    }

    Push-Location $Path
    try {
        if ($CargoPreflight) {
            $preflightSets = switch ($CargoPreflightMode) {
                'check' { @(@('check')) }
                'clippy' { @(@('clippy', '--all-targets', '--', '-D', 'warnings')) }
                'fmt' { @(@('fmt', '--', '--check')) }
                'all' {
                    @(
                        @('check'),
                        @('clippy', '--all-targets', '--', '-D', 'warnings'),
                        @('fmt', '--', '--check')
                    )
                }
            }

            foreach ($preflightArgs in $preflightSets) {
                & cargo @preflightArgs 2>&1 | Tee-Object -FilePath $LogFile | Out-Null
                if ($LASTEXITCODE -ne 0) {
                    return $false
                }
            }
        }

        & cargo @CargoArgs 2>&1 | Tee-Object -FilePath $LogFile | Out-Null
        return ($LASTEXITCODE -eq 0)
    } finally {
        Pop-Location
    }
}

function Resolve-CargoOutputDirectory {
    param(
        [Parameter(Mandatory)]
        [string]$ProjectDir,
        [Parameter(Mandatory)]
        [string]$Configuration
    )

    $configDir = $Configuration.ToLower()
    if ($env:CARGO_TARGET_DIR) {
        $sharedTargetDir = Join-Path $env:CARGO_TARGET_DIR $configDir
        if (Test-Path $sharedTargetDir) {
            return $sharedTargetDir
        }
    }

    return Join-Path $ProjectDir "target\$configDir"
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
        (Join-Path $script:BuildArtifactsDir 'pcai-native'),
        (Join-Path $script:BuildArtifactsDir 'pcai-servicehost'),
        (Join-Path $script:BuildArtifactsDir 'nukenul'),
        $script:BuildLogsDir,
        $script:BuildPackagesDir,
        $script:BuildDeployDir
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
        $script:BuildPackagesDir,
        $script:BuildDeployDir
    )

    foreach ($path in $dirsToClean) {
        if (Test-Path $path) {
            Remove-Item $path -Recurse -Force -ErrorAction SilentlyContinue
            Write-BuildStep "Cleaned $path" 'success'
        }
    }

    # Clean Rust target directories if requested
    $rustTargets = @(
        'Native\pcai_core\pcai_inference\target',
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
}

#endregion

#region Preflight Checks

function Invoke-AstGrepCheck {
    Write-BuildPhase 'Security' 'Running AST-Grep security checks and auto-fixes'

    # Run YAML auto-formatter first
    $yamlFormatScript = Join-Path $script:ProjectRoot 'Tools\Invoke-YamlFormat.ps1'
    if (Test-Path $yamlFormatScript) {
        Write-BuildStep 'Formatting AST-Grep YAML rules...' 'running'
        & $yamlFormatScript | Out-Null
        if ($LASTEXITCODE -ne 0) {
            Write-BuildStep 'YAML formatting failed' 'warning'
        } else {
            Write-BuildStep 'YAML rules formatted' 'success'
        }
    }

    if (-not (Get-Command sg -ErrorAction SilentlyContinue)) {
        Write-BuildStep 'ast-grep (sg) not found, skipping security checks' 'warning'
        return $true
    }

    # Auto-fix structural issues first
    Write-BuildStep 'Applying AST-Grep auto-fixes (--update-all)...' 'running'
    $fixOutput = sg scan --update-all 2>&1
    Write-BuildStep 'AST-Grep auto-fixes applied' 'success'

    # Run final scan to check for errors/hardcoded endpoints
    Write-BuildStep 'Scanning for network security violations...' 'running'
    $output = sg scan 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-BuildStep 'AST-Grep found hardcoded endpoint violations!' 'error'
        Write-Error ($output -join "`n")
        throw 'AST-Grep security check failed.'
    }

    Write-BuildStep 'AST-Grep security checks passed' 'success'
    return $true
}

#endregion

#region Component Builders

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
    if ($env:PCAI_DISABLE_CACHE -eq '1') { $buildArgs['DisableCache'] = $true }
    if ($env:CARGO_USE_LLD -eq '1') { $buildArgs['UseLld'] = $true }
    if ($CargoPreflight) {
        $buildArgs['Preflight'] = $true
        $buildArgs['PreflightMode'] = $CargoPreflightMode
        $buildArgs['PreflightBlocking'] = $true
    }

    try {
        # Capture output to log file while showing progress
        $output = & $buildScript @buildArgs 2>&1 | Tee-Object -FilePath $logFile
        $success = $LASTEXITCODE -eq 0

        # Collect artifacts
        $artifacts = @()
        $inferenceProjectDir = Join-Path $script:ProjectRoot 'Native\pcai_core\pcai_inference'
        $targetDir = Resolve-CargoOutputDirectory -ProjectDir $inferenceProjectDir -Configuration $Configuration

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
    $runtimeDir = Join-Path $script:ProjectRoot 'Deploy\rust-functiongemma-runtime'
    $trainDir = Join-Path $script:ProjectRoot 'Deploy\rust-functiongemma-train'
    $projectDirs = @($runtimeDir, $trainDir)

    if ($projectDirs | Where-Object { -not (Test-Path $_) }) {
        Write-BuildStep 'FunctionGemma project directories not found' 'warning'
        return @{ Success = $false; Duration = (Get-Date) - $componentStart; Artifacts = @() }
    }

    Write-BuildStep 'Building FunctionGemma runtime + train crates...' 'running'

    $timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    $logDir = $script:BuildLogsDir
    if (-not (Test-Path $logDir)) {
        New-Item -ItemType Directory -Path $logDir -Force | Out-Null
    }
    $logFile = Join-Path $logDir "build_functiongemma_$timestamp.log"

    try {
        $cargoArgs = @('build')
        if ($Configuration -eq 'Release') { $cargoArgs += '--release' }

        $success = $true
        foreach ($projectDir in $projectDirs) {
            $projectSuccess = Invoke-RustBuildCommand -Path $projectDir -CargoArgs $cargoArgs -LogFile $logFile
            if (-not $projectSuccess) {
                $success = $false
                break
            }
        }

        # Collect artifacts
        $artifacts = @()
        $configDir = if ($Configuration -eq 'Release') { 'release' } else { 'debug' }

        $artifactDir = Join-Path $script:BuildArtifactsDir 'functiongemma'
        if (-not (Test-Path $artifactDir)) {
            New-Item -ItemType Directory -Path $artifactDir -Force | Out-Null
        }

        # Copy binaries from both crates
        foreach ($projectDir in $projectDirs) {
            $targetDir = Resolve-CargoOutputDirectory -ProjectDir $projectDir -Configuration $Configuration
            if (-not (Test-Path $targetDir)) { continue }

            Get-ChildItem $targetDir -Filter '*.exe' -ErrorAction SilentlyContinue | ForEach-Object {
                Copy-Item $_.FullName -Destination $artifactDir -Force
                $artifacts += $_.Name
            }
            Get-ChildItem $targetDir -Filter '*.dll' -ErrorAction SilentlyContinue | ForEach-Object {
                Copy-Item $_.FullName -Destination $artifactDir -Force
                $artifacts += $_.Name
            }
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
        Write-BuildStep "FunctionGemma build failed: $($_.Exception.Message)" 'error'
        return @{
            Success   = $false
            Duration  = (Get-Date) - $componentStart
            Artifacts = @()
            Error     = $_.Exception.Message
        }
    }
}

function Invoke-FunctionGemmaOperation {
    param(
        [Parameter(Mandatory)]
        [ValidateSet('router-data', 'token-cache', 'train', 'eval')]
        [string]$Operation,
        [string[]]$OperationArgs = @()
    )

    $componentStart = Get-Date
    $trainDir = Join-Path $script:ProjectRoot 'Deploy\rust-functiongemma-train'
    if (-not (Test-Path $trainDir)) {
        Write-BuildStep 'FunctionGemma train crate not found' 'warning'
        return @{ Success = $false; Duration = (Get-Date) - $componentStart; Artifacts = @() }
    }

    $timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    $logFile = Join-Path $script:BuildLogsDir "build_functiongemma_${Operation}_$timestamp.log"
    $opLabel = switch ($Operation) {
        'router-data' { 'FunctionGemma router dataset generation' }
        'token-cache' { 'FunctionGemma token cache generation' }
        'train' { 'FunctionGemma training run' }
        'eval' { 'FunctionGemma evaluation run' }
    }
    Write-BuildStep "$opLabel..." 'running'

    $repoRoot = $script:ProjectRoot
    $defaultRouterArgs = @(
        '--tools', (Join-Path $repoRoot 'Config\pcai-tools.json'),
        '--output', (Join-Path $repoRoot 'Deploy\rust-functiongemma-train\data\rust_router_train.jsonl'),
        '--diagnose-prompt', (Join-Path $repoRoot 'Deploy\rust-functiongemma-train\examples\router_diagnose_prompt.md'),
        '--chat-prompt', (Join-Path $repoRoot 'Deploy\rust-functiongemma-train\examples\router_chat_prompt.md'),
        '--scenarios', (Join-Path $repoRoot 'Deploy\rust-functiongemma-train\examples\scenarios.json'),
        '--test-vectors', (Join-Path $repoRoot 'Deploy\rust-functiongemma-train\data\test_vectors.json'),
        '--max-cases', '24'
    )
    $defaultCacheArgs = @(
        '--input', (Join-Path $repoRoot 'Deploy\rust-functiongemma-train\data\rust_router_train.jsonl'),
        '--tokenizer', (Join-Path $repoRoot 'Models\functiongemma-270m-it\tokenizer.json'),
        '--output-dir', (Join-Path $repoRoot 'output\functiongemma-token-cache')
    )
    $defaultEvalArgs = @(
        '--config', (Join-Path $repoRoot 'Config\pcai-functiongemma.json'),
        'eval',
        '--model-path', (Join-Path $repoRoot 'Models\functiongemma-270m-it'),
        '--test-data', (Join-Path $repoRoot 'Deploy\rust-functiongemma-train\data\rust_router_train.jsonl'),
        '--lora-r', '16',
        '--max-new-tokens', '64',
        '--metrics-output', (Join-Path $repoRoot 'Reports\functiongemma_eval_metrics.json'),
        '--quiet'
    )
    $defaultTrainArgs = @(
        '--config', (Join-Path $repoRoot 'Config\pcai-functiongemma.json'),
        'train',
        '--model-path', (Join-Path $repoRoot 'Models\functiongemma-270m-it'),
        '--train-data', (Join-Path $repoRoot 'Deploy\rust-functiongemma-train\data\rust_router_train.jsonl'),
        '--output', (Join-Path $repoRoot 'output\functiongemma-lora'),
        '--epochs', '1',
        '--lr', '1e-5',
        '--lora-r', '16',
        '--lora-alpha', '32',
        '--batch-size', '1',
        '--grad-accum', '4'
    )

    $cargoArgs = switch ($Operation) {
        'router-data' {
            if ($OperationArgs.Count -gt 0) {
                @('run', '--', 'prepare-router') + $OperationArgs
            } else {
                @('run', '--', 'prepare-router') + $defaultRouterArgs
            }
        }
        'token-cache' {
            if ($OperationArgs.Count -gt 0) {
                @('run', '--', 'prepare-cache') + $OperationArgs
            } else {
                $chatTemplate = Join-Path $repoRoot 'Models\functiongemma-270m-it\chat_template.jinja'
                if (Test-Path $chatTemplate) {
                    @('run', '--', 'prepare-cache') + $defaultCacheArgs + @('--chat-template', $chatTemplate)
                } else {
                    @('run', '--', 'prepare-cache') + $defaultCacheArgs
                }
            }
        }
        'eval' {
            if ($OperationArgs.Count -gt 0) {
                @('run', '--') + $OperationArgs
            } else {
                @('run', '--') + $defaultEvalArgs
            }
        }
        'train' {
            if ($OperationArgs.Count -gt 0) {
                @('run', '--') + $OperationArgs
            } else {
                @('run', '--') + $defaultTrainArgs
            }
        }
    }

    try {
        $success = Invoke-RustBuildCommand -Path $trainDir -CargoArgs $cargoArgs -LogFile $logFile
        if ($success) {
            Write-BuildStep "$opLabel complete" 'success'
        } else {
            Write-BuildStep "$opLabel failed" 'error'
        }

        return @{
            Success   = $success
            Duration  = (Get-Date) - $componentStart
            Artifacts = @()
            LogFile   = $logFile
        }
    } catch {
        Write-BuildStep "$opLabel failed: $($_.Exception.Message)" 'error'
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

function Invoke-DotnetComponentBuild {
    param(
        [Parameter(Mandatory)]
        [string]$ComponentName,
        [Parameter(Mandatory)]
        [string]$ProjectRelativePath,
        [Parameter(Mandatory)]
        [string]$Configuration
    )

    $componentStart = Get-Date
    $projectPath = Join-Path $script:ProjectRoot $ProjectRelativePath

    if (-not (Test-Path $projectPath)) {
        Write-BuildStep "$ComponentName project not found: $projectPath" 'warning'
        return @{ Success = $false; Duration = (Get-Date) - $componentStart; Artifacts = @() }
    }

    if (-not (Get-Command dotnet -ErrorAction SilentlyContinue)) {
        Write-BuildStep "dotnet not found; cannot build $ComponentName" 'error'
        return @{ Success = $false; Duration = (Get-Date) - $componentStart; Artifacts = @() }
    }

    Write-BuildStep "Building $ComponentName..." 'running'

    $timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    $logFile = Join-Path $script:BuildLogsDir "build_${ComponentName}_$timestamp.log"
    $publishRoot = Join-Path $script:BuildRoot "tmp\$ComponentName-$($Configuration.ToLower())"

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

        & dotnet @dotnetArgs 2>&1 | Tee-Object -FilePath $logFile | Out-Null
        $success = $LASTEXITCODE -eq 0

        $artifacts = @()
        $artifactDir = Join-Path $script:BuildArtifactsDir $ComponentName
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
        Write-BuildStep "$ComponentName build complete" $status

        return @{
            Success   = $success
            Duration  = (Get-Date) - $componentStart
            Artifacts = $artifacts
            LogFile   = $logFile
        }
    } catch {
        Write-BuildStep "$ComponentName build failed: $($_.Exception.Message)" 'error'
        return @{
            Success   = $false
            Duration  = (Get-Date) - $componentStart
            Artifacts = @()
            Error     = $_.Exception.Message
        }
    }
}

function Invoke-NukeNulBuild {
    param([string]$Configuration)

    $componentStart = Get-Date
    $nukeRoot = Join-Path $script:ProjectRoot 'Native\NukeNul'
    $rustDir = Join-Path $nukeRoot 'nuker_core'
    $projectPath = Join-Path $nukeRoot 'NukeNul.csproj'
    $timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    $logFile = Join-Path $script:BuildLogsDir "build_nukenul_$timestamp.log"
    $artifactDir = Join-Path $script:BuildArtifactsDir 'nukenul'

    if (-not (Test-Path $rustDir) -or -not (Test-Path $projectPath)) {
        Write-BuildStep 'NukeNul project structure not found' 'warning'
        return @{ Success = $false; Duration = (Get-Date) - $componentStart; Artifacts = @() }
    }

    if (-not (Test-Path $artifactDir)) {
        New-Item -ItemType Directory -Path $artifactDir -Force | Out-Null
    }

    Write-BuildStep 'Building NukeNul (Rust DLL + .NET host)...' 'running'
    try {
        $cargoArgs = @('build')
        if ($Configuration -eq 'Release') { $cargoArgs += '--release' }

        $rustOk = Invoke-RustBuildCommand -Path $rustDir -CargoArgs $cargoArgs -LogFile $logFile
        if (-not $rustOk) {
            return @{
                Success   = $false
                Duration  = (Get-Date) - $componentStart
                Artifacts = @()
                LogFile   = $logFile
            }
        }

        $rustTarget = Resolve-CargoOutputDirectory -ProjectDir $rustDir -Configuration $Configuration
        $rustDll = Join-Path $rustTarget 'nuker_core.dll'
        if (Test-Path $rustDll) {
            Copy-Item $rustDll -Destination $nukeRoot -Force
            Copy-Item $rustDll -Destination $artifactDir -Force
        }

        $dotnetResult = Invoke-DotnetComponentBuild -ComponentName 'nukenul' -ProjectRelativePath 'Native\NukeNul\NukeNul.csproj' -Configuration $Configuration
        $artifacts = @($dotnetResult.Artifacts)
        if (Test-Path $rustDll) {
            $artifacts += 'nuker_core.dll'
        }

        return @{
            Success   = $dotnetResult.Success
            Duration  = (Get-Date) - $componentStart
            Artifacts = $artifacts
            LogFile   = $logFile
        }
    } catch {
        Write-BuildStep "NukeNul build failed: $($_.Exception.Message)" 'error'
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
    $components = @('pcai-llamacpp', 'pcai-mistralrs', 'functiongemma', 'pcai-chattui', 'pcai-native', 'pcai-servicehost', 'nukenul')

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

#region Test and Deploy

function Invoke-PostBuildTests {
    param(
        [string[]]$BuildTargets,
        [string]$Configuration
    )

    Write-BuildPhase 'Test' 'Running post-build test suites'

    $testFailures = @()
    $timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'

    $runInferenceTests = ($BuildTargets | Where-Object { $_ -in @('llamacpp', 'mistralrs') }).Count -gt 0
    if ($runInferenceTests) {
        $inferenceProject = Join-Path $script:ProjectRoot 'Native\pcai_core\pcai_inference'
        $baseArgs = @('test', '--no-default-features', '--features')

        $featuresToTest = @()
        if ($BuildTargets -contains 'llamacpp') { $featuresToTest += 'server,ffi,llamacpp' }
        if ($BuildTargets -contains 'mistralrs') { $featuresToTest += 'server,ffi,mistralrs-backend' }
        if ($featuresToTest.Count -eq 0) { $featuresToTest += 'server,ffi' }

        foreach ($featureSet in $featuresToTest | Select-Object -Unique) {
            $logFile = Join-Path $script:BuildLogsDir "test_inference_$timestamp.log"
            Write-BuildStep "Testing pcai-inference ($featureSet)..." 'running'
            $ok = Invoke-RustBuildCommand -Path $inferenceProject -CargoArgs ($baseArgs + @($featureSet, '--lib')) -LogFile $logFile
            if (-not $ok) {
                $testFailures += "pcai-inference:$featureSet"
                Write-BuildStep "pcai-inference tests failed ($featureSet)" 'error'
            } else {
                Write-BuildStep "pcai-inference tests passed ($featureSet)" 'success'
            }
        }
    }

    if ($BuildTargets -contains 'functiongemma') {
        $runtimeDir = Join-Path $script:ProjectRoot 'Deploy\rust-functiongemma-runtime'
        $trainDir = Join-Path $script:ProjectRoot 'Deploy\rust-functiongemma-train'

        foreach ($path in @($runtimeDir, $trainDir)) {
            $name = Split-Path $path -Leaf
            $logFile = Join-Path $script:BuildLogsDir "test_${name}_$timestamp.log"
            Write-BuildStep "Testing $name..." 'running'
            $ok = Invoke-RustBuildCommand -Path $path -CargoArgs @('test') -LogFile $logFile
            if (-not $ok) {
                $testFailures += $name
                Write-BuildStep "$name tests failed" 'error'
            } else {
                Write-BuildStep "$name tests passed" 'success'
            }
        }
    }

    return @{
        Success = ($testFailures.Count -eq 0)
        Failed  = $testFailures
    }
}

function New-DeployBundle {
    param(
        [string]$Configuration,
        [bool]$EnableCuda
    )

    Write-BuildPhase 'Deploy' 'Creating deploy-ready aggregate bundle'

    $variant = if ($EnableCuda) { 'cuda' } else { 'cpu' }
    $version = if ($env:PCAI_VERSION) { $env:PCAI_VERSION } else { '0.1.0+unknown' }
    $safeVersion = ($version -replace '[^A-Za-z0-9\.\-\+_]', '_') -replace '\+', '_'
    $bundleName = "pc-ai-bundle-$safeVersion-$variant-win64"
    $bundleRoot = Join-Path $script:BuildDeployDir $bundleName
    $bundleZip = Join-Path $script:BuildPackagesDir "$bundleName.zip"

    if (Test-Path $bundleRoot) {
        Remove-Item $bundleRoot -Recurse -Force -ErrorAction SilentlyContinue
    }
    New-Item -ItemType Directory -Path $bundleRoot -Force | Out-Null

    Copy-Item -Path (Join-Path $script:BuildArtifactsDir '*') -Destination $bundleRoot -Recurse -Force
    if (Test-Path (Join-Path $script:BuildArtifactsDir 'manifest.json')) {
        Copy-Item -Path (Join-Path $script:BuildArtifactsDir 'manifest.json') -Destination $bundleRoot -Force
    }

    if (Test-Path $bundleZip) {
        Remove-Item $bundleZip -Force -ErrorAction SilentlyContinue
    }
    Compress-Archive -Path (Join-Path $bundleRoot '*') -DestinationPath $bundleZip -Force
    Write-BuildStep "Created deploy bundle: $bundleZip" 'success'

    return $bundleZip
}

#endregion

#region Main Execution

Write-BuildHeader 'PC_AI Build System'
Write-Host "  Component:     $Component" -ForegroundColor White
Write-Host "  Configuration: $Configuration" -ForegroundColor White
Write-Host "  CUDA:          $(if ($EnableCuda) { 'Enabled' } else { 'Disabled' })" -ForegroundColor White
Write-Host "  Clean:         $(if ($Clean) { 'Yes' } else { 'No' })" -ForegroundColor White
Write-Host "  Package:       $(if ($Package) { 'Yes' } else { 'No' })" -ForegroundColor White
Write-Host "  RunTests:      $(if ($RunTests) { 'Yes' } else { 'No' })" -ForegroundColor White
Write-Host "  Deploy:        $(if ($Deploy) { 'Yes' } else { 'No' })" -ForegroundColor White
Write-Host "  CargoTools:    $CargoTools" -ForegroundColor White
Write-Host "  Preflight:     $(if ($CargoPreflight) { $CargoPreflightMode } else { 'Disabled' })" -ForegroundColor White
Write-Host "  SyncDefaults:  $(if ($SyncCargoDefaults) { 'Yes' } else { 'No' })" -ForegroundColor White
Write-Host "  FG Args:       $(if ($FunctionGemmaArgs.Count -gt 0) { ($FunctionGemmaArgs -join ' ') } else { '(none)' })" -ForegroundColor White
Write-Host "  Artifacts Root: $script:ArtifactsRoot" -ForegroundColor White

# Clean if requested
if ($Clean) {
    Clear-BuildArtifacts
}

# Preflight checks
Invoke-AstGrepCheck

# Initialize directories
Initialize-BuildDirectories

# Initialize version information (sets PCAI_* environment variables)
$versionInfo = Initialize-BuildVersion

# Initialize CargoTools defaults / wrappers
Initialize-CargoToolsDefaults | Out-Null

# Deploy implies package creation
if ($Deploy) {
    $Package = $true
}

# Determine what to build
$buildTargets = switch ($Component) {
    'inference' { @('llamacpp', 'mistralrs') }
    'llamacpp' { @('llamacpp') }
    'mistralrs' { @('mistralrs') }
    'functiongemma' { @('functiongemma') }
    'functiongemma-router-data' { @('functiongemma-router-data') }
    'functiongemma-token-cache' { @('functiongemma-token-cache') }
    'functiongemma-train' { @('functiongemma-train') }
    'functiongemma-eval' { @('functiongemma-eval') }
    'tui' { @('tui') }
    'pcainative' { @('pcainative') }
    'servicehost' { @('servicehost') }
    'nukenul' { @('nukenul') }
    'native' { @('pcainative', 'servicehost', 'tui', 'nukenul') }
    'all' { @('llamacpp', 'mistralrs', 'functiongemma', 'pcainative', 'servicehost', 'tui', 'nukenul') }
}

$results = @{}

# Build inference backends
Write-BuildPhase 'Build' 'Compiling native components'

foreach ($target in $buildTargets) {
    if ($target -eq 'functiongemma') {
        $result = Invoke-FunctionGemmaBuild -Configuration $Configuration
        $results['functiongemma'] = $result
        Write-BuildResult 'FunctionGemma' $result.Success $result.Duration $result.Artifacts
    } elseif ($target -eq 'functiongemma-router-data') {
        $result = Invoke-FunctionGemmaOperation -Operation 'router-data' -OperationArgs $FunctionGemmaArgs
        $results['functiongemma-router-data'] = $result
        Write-BuildResult 'functiongemma-router-data' $result.Success $result.Duration $result.Artifacts
    } elseif ($target -eq 'functiongemma-token-cache') {
        $result = Invoke-FunctionGemmaOperation -Operation 'token-cache' -OperationArgs $FunctionGemmaArgs
        $results['functiongemma-token-cache'] = $result
        Write-BuildResult 'functiongemma-token-cache' $result.Success $result.Duration $result.Artifacts
    } elseif ($target -eq 'functiongemma-train') {
        $result = Invoke-FunctionGemmaOperation -Operation 'train' -OperationArgs $FunctionGemmaArgs
        $results['functiongemma-train'] = $result
        Write-BuildResult 'functiongemma-train' $result.Success $result.Duration $result.Artifacts
    } elseif ($target -eq 'functiongemma-eval') {
        $result = Invoke-FunctionGemmaOperation -Operation 'eval' -OperationArgs $FunctionGemmaArgs
        $results['functiongemma-eval'] = $result
        Write-BuildResult 'functiongemma-eval' $result.Success $result.Duration $result.Artifacts
    } elseif ($target -eq 'tui') {
        $result = Invoke-TuiBuild -Configuration $Configuration
        $results['pcai-chattui'] = $result
        Write-BuildResult 'pcai-chattui' $result.Success $result.Duration $result.Artifacts
    } elseif ($target -eq 'pcainative') {
        $result = Invoke-DotnetComponentBuild -ComponentName 'pcai-native' -ProjectRelativePath 'Native\PcaiNative\PcaiNative.csproj' -Configuration $Configuration
        $results['pcai-native'] = $result
        Write-BuildResult 'pcai-native' $result.Success $result.Duration $result.Artifacts
    } elseif ($target -eq 'servicehost') {
        $result = Invoke-DotnetComponentBuild -ComponentName 'pcai-servicehost' -ProjectRelativePath 'Native\PcaiServiceHost\PcaiServiceHost.csproj' -Configuration $Configuration
        $results['pcai-servicehost'] = $result
        Write-BuildResult 'pcai-servicehost' $result.Success $result.Duration $result.Artifacts
    } elseif ($target -eq 'nukenul') {
        $result = Invoke-NukeNulBuild -Configuration $Configuration
        $results['nukenul'] = $result
        Write-BuildResult 'nukenul' $result.Success $result.Duration $result.Artifacts
    } else {
        $result = Invoke-InferenceBuild -Backend $target -Configuration $Configuration -EnableCuda $EnableCuda
        $results["pcai-$target"] = $result
        Write-BuildResult "pcai-$target" $result.Success $result.Duration $result.Artifacts
    }
}

if ($SkipTests) {
    Write-BuildStep 'Post-build tests skipped by request' 'skip'
} elseif ($RunTests) {
    $testResult = Invoke-PostBuildTests -BuildTargets $buildTargets -Configuration $Configuration
    if (-not $testResult.Success) {
        foreach ($failedSuite in $testResult.Failed) {
            Write-BuildStep "Test suite failed: $failedSuite" 'error'
        }
    }
} else {
    Write-BuildStep 'Post-build tests not requested. Use -RunTests to execute component-aligned tests.' 'warning'
}

# Generate manifest
$manifestPath = New-BuildManifest -Results $results -Configuration $Configuration -EnableCuda $EnableCuda

# Create packages if requested
if ($Package) {
    $packages = New-ReleasePackages -Configuration $Configuration -EnableCuda $EnableCuda
}

if ($Deploy) {
    $deployBundle = New-DeployBundle -Configuration $Configuration -EnableCuda $EnableCuda
}

# Summary
Write-BuildSummary -Results $results -ManifestPath $manifestPath

# Exit with appropriate code
$failedCount = ($results.Values | Where-Object { -not $_.Success }).Count
if ($RunTests -and -not $SkipTests -and $testResult -and -not $testResult.Success) {
    $failedCount += $testResult.Failed.Count
}
exit $failedCount

#endregion
