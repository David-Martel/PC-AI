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
    - media: pcai-media Janus-Pro media agent (FFI DLL + HTTP server)
    - tui: PcaiChatTui .NET chat terminal UI
    - pcainative: PcaiNative .NET interop wrapper
    - servicehost: PcaiServiceHost .NET host binary
    - nukenul: Hybrid Rust/C# NukeNul utility (external repo preferred)
    - lint: Run repository lint checks only
    - format: Run formatting checks only
    - fix: Apply automated formatting/fixes where supported
    - deps: Refresh and validate Rust dependency graph/cache state
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

.PARAMETER LintProfile
    Lint/format profile selector used by lint/format/fix components:
    all, rust-check, rust-clippy, rust-fmt, rust-inference-check,
    rust-inference-clippy, rust-inference-fmt, dotnet-format, powershell, docs, astgrep, toml.

.PARAMETER DependencyStrategy
    Rust dependency lock behavior:
    - locked: pass --locked to cargo commands (default, cache-friendly)
    - frozen: pass --frozen to cargo commands
    - update: allow dependency updates (no lock flag)

.PARAMETER AutoFix
    Apply auto-fixes where supported (ast-grep update, formatters, dotnet format write mode).

.PARAMETER SkipQualityGate
    Skip pre-build ast-grep quality gate for non-lint components.

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
    Build the NukeNul hybrid Rust + C# utility from `NUKENUL_ROOT` or `C:\codedev\nukenul`.

.EXAMPLE
    .\Build.ps1 -Component lint -LintProfile all
    Run repository lint checks only.

.EXAMPLE
    .\Build.ps1 -Component format -LintProfile rust-fmt
    Run Rust formatting checks only.

.EXAMPLE
    .\Build.ps1 -Component fix -LintProfile all -AutoFix
    Apply automated code/document fixes where toolchains support write mode.
#>

[CmdletBinding()]
param(
    [ValidateSet('inference', 'llamacpp', 'mistralrs', 'functiongemma', 'functiongemma-router-data', 'functiongemma-token-cache', 'functiongemma-train', 'functiongemma-eval', 'media', 'tui', 'pcainative', 'servicehost', 'nukenul', 'lint', 'format', 'fix', 'deps', 'native', 'all')]
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
    [ValidateSet('all', 'rust-check', 'rust-clippy', 'rust-fmt', 'rust-inference-check', 'rust-inference-clippy', 'rust-inference-fmt', 'dotnet-format', 'powershell', 'docs', 'astgrep', 'toml', 'rag-quality')]
    [string]$LintProfile = 'all',
    [ValidateSet('locked', 'frozen', 'update')]
    [string]$DependencyStrategy = 'locked',
    [switch]$AutoFix,
    [switch]$SkipQualityGate,
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
$script:DependencyStrategy = $DependencyStrategy
$script:PcaiModuleBootstrap = Join-Path $script:ProjectRoot 'Tools\PcaiModuleBootstrap.ps1'
$script:NukeNulRootCandidates = @(
    $env:NUKENUL_ROOT,
    'C:\codedev\nukenul',
    (Join-Path $script:ProjectRoot 'Native\NukeNul')
) | Where-Object { $_ }

if (Test-Path -LiteralPath $script:PcaiModuleBootstrap) {
    . $script:PcaiModuleBootstrap
}

function Resolve-NukeNulProjectRoot {
    foreach ($candidate in $script:NukeNulRootCandidates) {
        if (-not $candidate) { continue }

        $resolved = try {
            if (Test-Path -LiteralPath $candidate) {
                (Resolve-Path -LiteralPath $candidate -ErrorAction Stop).Path
            } else {
                [System.IO.Path]::GetFullPath($candidate)
            }
        } catch {
            continue
        }

        if (Test-Path -LiteralPath (Join-Path $resolved 'NukeNul.csproj')) {
            return $resolved
        }
    }

    return $null
}

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

function Format-InvariantString {
    param(
        [Parameter(Mandatory)]
        [string]$Template,
        [Parameter(Mandatory)]
        [object[]]$Args
    )

    return [string]::Format([System.Globalization.CultureInfo]::InvariantCulture, $Template, $Args)
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

    if ($Artifacts -and @($Artifacts).Count -gt 0) {
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
    $successCount = @($Results.Values | Where-Object { $_.Success }).Count
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

    $cargoToolsManifest = if (Get-Command Resolve-PcaiModuleManifestPath -ErrorAction SilentlyContinue) {
        Resolve-PcaiModuleManifestPath -ModuleName 'CargoTools' -RepoRoot $script:ProjectRoot
    } else {
        $null
    }

    if (-not $cargoToolsManifest) {
        if ($CargoTools -eq 'enabled') {
            Write-BuildStep 'CargoTools module required but not installed' 'error'
            throw 'CargoTools integration was explicitly enabled but the module was not found.'
        }
        Write-BuildStep 'CargoTools module not found; using direct tooling defaults' 'warning'
        return $false
    }

    if (-not (Get-Module CargoTools)) {
        Import-Module -Name $cargoToolsManifest -ErrorAction Stop | Out-Null
    }

    $cacheRoot = if ($env:PCAI_CACHE_ROOT) {
        $env:PCAI_CACHE_ROOT
    } elseif (Get-Command Resolve-CacheRoot -Module CargoTools -ErrorAction SilentlyContinue) {
        Resolve-CacheRoot
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
        Write-BuildStep "CargoTools manifest: $cargoToolsManifest" 'success'
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

function Add-CargoDependencyFlags {
    param(
        [Parameter(Mandatory)]
        [string[]]$CargoArgs
    )

    if (-not $CargoArgs -or $CargoArgs.Count -eq 0) {
        return $CargoArgs
    }

    if ($CargoArgs -contains '--locked' -or $CargoArgs -contains '--frozen') {
        return $CargoArgs
    }

    if ($script:DependencyStrategy -eq 'locked') {
        return $CargoArgs + @('--locked')
    }
    if ($script:DependencyStrategy -eq 'frozen') {
        return $CargoArgs + @('--frozen')
    }

    return $CargoArgs
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

    $effectiveCargoArgs = Add-CargoDependencyFlags -CargoArgs $CargoArgs

    if (Test-Path $script:RustBuildWrapper) {
        $shellExe = Resolve-PowerShellExecutable
        $wrapperArgs = @('-NoProfile', '-File', $script:RustBuildWrapper, '-Path', $Path, '-CargoArgs') + $effectiveCargoArgs

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
                $effectivePreflightArgs = Add-CargoDependencyFlags -CargoArgs $preflightArgs
                & cargo @effectivePreflightArgs 2>&1 | Tee-Object -FilePath $LogFile | Out-Null
                if ($LASTEXITCODE -ne 0) {
                    return $false
                }
            }
        }

        & cargo @effectiveCargoArgs 2>&1 | Tee-Object -FilePath $LogFile | Out-Null
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

    if (Get-Command Resolve-CargoTargetDirectory -ErrorAction SilentlyContinue) {
        $manifestPath = Join-Path $ProjectDir 'Cargo.toml'
        return Resolve-CargoTargetDirectory -ProjectDir $ProjectDir -ManifestPath $manifestPath -Configuration $Configuration
    }

    $configDir = $Configuration.ToLower()
    if ($env:CARGO_TARGET_DIR) {
        $sharedTargetDir = Join-Path $env:CARGO_TARGET_DIR $configDir
        if (Test-Path $sharedTargetDir) {
            return $sharedTargetDir
        }
    }

    return Join-Path $ProjectDir "target\$configDir"
}

function Publish-StagedArtifact {
    param(
        [Parameter(Mandatory)]
        [string]$SourcePath,
        [Parameter(Mandatory)]
        [string]$DestinationDirectory,
        [string]$DestinationFileName,
        [string]$ArtifactKind = 'binary'
    )

    if (-not (Test-Path -LiteralPath $SourcePath -PathType Leaf)) {
        return $null
    }

    if (Get-Command Publish-BuildArtifact -ErrorAction SilentlyContinue) {
        return Publish-BuildArtifact -SourcePath $SourcePath -DestinationDirectory $DestinationDirectory -DestinationFileName $DestinationFileName -VersionInfo $script:VersionInfo -ArtifactKind $ArtifactKind
    }

    if (-not (Test-Path -LiteralPath $DestinationDirectory)) {
        New-Item -ItemType Directory -Path $DestinationDirectory -Force | Out-Null
    }

    if (-not $DestinationFileName) {
        $DestinationFileName = [System.IO.Path]::GetFileName($SourcePath)
    }

    $destinationPath = Join-Path $DestinationDirectory $DestinationFileName
    Copy-Item -LiteralPath $SourcePath -Destination $destinationPath -Force
    return [pscustomobject]@{
        DestinationPath = $destinationPath
        ManifestPath = $null
        FileName = [System.IO.Path]::GetFileName($destinationPath)
    }
}

function ConvertTo-SafePathLabel {
    param(
        [Parameter(Mandatory)]
        [string]$Value
    )

    $safe = $Value -replace '[^A-Za-z0-9._-]', '_'
    $safe = $safe.Trim('_')
    if ([string]::IsNullOrWhiteSpace($safe)) {
        return 'unknown'
    }

    return $safe
}

function Publish-PcaiNativeBundle {
    param(
        [Parameter(Mandatory)]
        [string]$PublishRoot,
        [Parameter(Mandatory)]
        [string]$Configuration
    )

    $repoBinRoot = Join-Path $script:ProjectRoot 'bin'
    $bundleParent = Join-Path $repoBinRoot 'native-bundles'
    if (-not (Test-Path -LiteralPath $bundleParent)) {
        New-Item -ItemType Directory -Path $bundleParent -Force | Out-Null
    }

    $versionLabel = if ($script:VersionInfo) {
        if ($script:VersionInfo.ReleaseTag) {
            $script:VersionInfo.ReleaseTag
        } elseif ($script:VersionInfo.InformationalVersion) {
            $script:VersionInfo.InformationalVersion
        } elseif ($script:VersionInfo.SemVer) {
            $script:VersionInfo.SemVer
        } else {
            $script:VersionInfo.FileVersion
        }
    } else {
        'unknown'
    }

    $bundleName = '{0}-{1}' -f (ConvertTo-SafePathLabel -Value $versionLabel), (Get-Date -Format 'yyyyMMdd-HHmmss')
    $bundleRoot = Join-Path $bundleParent $bundleName
    if (-not (Test-Path -LiteralPath $bundleRoot)) {
        New-Item -ItemType Directory -Path $bundleRoot -Force | Out-Null
    }

    $coreProjectDir = Join-Path $script:ProjectRoot 'Native\\pcai_core'
    $coreTargetDir = Resolve-CargoOutputDirectory -ProjectDir $coreProjectDir -Configuration $Configuration

    $publishedEntries = [System.Collections.Generic.List[object]]::new()
    $bundleSpecs = @(
        @{ Name = 'PcaiNative.dll'; Source = (Join-Path $PublishRoot 'PcaiNative.dll'); Kind = 'managed-dotnet'; Required = $true },
        @{ Name = 'PcaiNative.deps.json'; Source = (Join-Path $PublishRoot 'PcaiNative.deps.json'); Kind = 'managed-dotnet'; Required = $true },
        @{ Name = 'PcaiNative.pdb'; Source = (Join-Path $PublishRoot 'PcaiNative.pdb'); Kind = 'managed-dotnet'; Required = $false },
        @{ Name = 'PcaiNative.xml'; Source = (Join-Path $PublishRoot 'PcaiNative.xml'); Kind = 'managed-dotnet'; Required = $false },
        @{ Name = 'pcai_core_lib.dll'; Source = (Join-Path $coreTargetDir 'pcai_core_lib.dll'); Fallback = (Join-Path $PublishRoot 'pcai_core_lib.dll'); Kind = 'native-rust'; Required = $true },
        @{ Name = 'pcai_core_lib.pdb'; Source = (Join-Path $coreTargetDir 'pcai_core_lib.pdb'); Fallback = (Join-Path $PublishRoot 'pcai_core_lib.pdb'); Kind = 'native-rust'; Required = $false },
        @{ Name = 'pcai_inference.dll'; Source = (Join-Path $coreTargetDir 'pcai_inference.dll'); Fallback = (Join-Path $PublishRoot 'pcai_inference.dll'); Kind = 'native-rust'; Required = $false }
    )

    foreach ($spec in $bundleSpecs) {
        $sourcePath = $spec.Source
        if ((-not $sourcePath -or -not (Test-Path -LiteralPath $sourcePath -PathType Leaf)) -and $spec.ContainsKey('Fallback')) {
            $fallback = $spec.Fallback
            if ($fallback -and (Test-Path -LiteralPath $fallback -PathType Leaf)) {
                $sourcePath = $fallback
            }
        }

        if (-not $sourcePath -or -not (Test-Path -LiteralPath $sourcePath -PathType Leaf)) {
            if ($spec.Required) {
                throw "Required native bundle asset not found: $($spec.Name)"
            }

            continue
        }

        $published = Publish-StagedArtifact -SourcePath $sourcePath -DestinationDirectory $bundleRoot -DestinationFileName $spec.Name -ArtifactKind $spec.Kind
        if ($published) {
            $publishedEntries.Add([pscustomobject]@{
                Name = $spec.Name
                SourcePath = $sourcePath
                DestinationPath = $published.DestinationPath
                Sha256 = (Get-FileHash -LiteralPath $published.DestinationPath -Algorithm SHA256).Hash
            })
        }
    }

    $bundleManifest = [ordered]@{
        bundleName = $bundleName
        bundleRoot = $bundleRoot
        createdAt = (Get-Date).ToUniversalTime().ToString('o')
        configuration = $Configuration
        releaseTag = if ($script:VersionInfo) { $script:VersionInfo.ReleaseTag } else { $null }
        semVer = if ($script:VersionInfo) { $script:VersionInfo.SemVer } else { $null }
        informationalVersion = if ($script:VersionInfo) { $script:VersionInfo.InformationalVersion } else { $null }
        cargoTargetDir = $coreTargetDir
        files = @($publishedEntries)
    }

    $manifestPath = Join-Path $bundleRoot 'bundle.buildinfo.json'
    $bundleManifest | ConvertTo-Json -Depth 6 | Out-File -FilePath $manifestPath -Encoding utf8

    return [pscustomobject]@{
        BundleRoot = $bundleRoot
        BundleName = $bundleName
        ManifestPath = $manifestPath
        Files = @($publishedEntries)
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
        (Join-Path $script:BuildArtifactsDir 'pcai-native'),
        (Join-Path $script:BuildArtifactsDir 'pcai-servicehost'),
        (Join-Path $script:BuildArtifactsDir 'pcai-media'),
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

#region Quality Checks

function Get-QualityTargets {
    param(
        [ValidateSet('all', 'inference')]
        [string]$Scope = 'all'
    )

    if ($Scope -eq 'inference') {
        return @(
            @{
                Name = 'pcai-inference'
                Path = Join-Path $script:ProjectRoot 'Native\pcai_core\pcai_inference'
            }
        ) | Where-Object { Test-Path $_.Path }
    }

    return @(
        @{
            Name = 'pcai-core'
            Path = Join-Path $script:ProjectRoot 'Native\pcai_core'
        },
        @{
            Name = 'functiongemma-workspace'
            Path = Join-Path $script:ProjectRoot 'Deploy\rust-functiongemma'
        },
        @{
            Name = 'nukenul-core'
            Path = if (Resolve-NukeNulProjectRoot) { Join-Path (Resolve-NukeNulProjectRoot) 'nuker_core' } else { $null }
        }
    ) | Where-Object { Test-Path $_.Path }
}

function Invoke-PrettierQuality {
    param(
        [switch]$WriteMode
    )

    if (-not (Get-Command npx -ErrorAction SilentlyContinue)) {
        Write-BuildStep 'npx not found; skipping JSON/YAML/Markdown/TOML formatting checks' 'warning'
        return $true
    }

    $configPath = Join-Path $script:ProjectRoot '.prettierrc.toml'
    $ignorePath = Join-Path $script:ProjectRoot '.prettierignore'
    $glob = '**/*.{json,yml,yaml,md,toml}'

    $args = @('prettier')
    if (Test-Path $configPath) { $args += @('--config', $configPath) }
    if (Test-Path $ignorePath) { $args += @('--ignore-path', $ignorePath) }
    if ($WriteMode) {
        $args += @('--write', $glob)
        Write-BuildStep 'Running Prettier write pass (JSON/YAML/Markdown/TOML)...' 'running'
    } else {
        $args += @('--check', $glob)
        Write-BuildStep 'Running Prettier check (JSON/YAML/Markdown/TOML)...' 'running'
    }

    Push-Location $script:ProjectRoot
    try {
        & npx @args 2>&1 | Out-Null
        if ($LASTEXITCODE -ne 0) {
            Write-BuildStep 'Prettier quality pass failed' 'error'
            return $false
        }
    } finally {
        Pop-Location
    }

    Write-BuildStep 'Prettier quality pass complete' 'success'
    return $true
}

function Invoke-NativeCommandCapture {
    param(
        [Parameter(Mandatory)]
        [string]$FilePath,
        [string[]]$ArgumentList = @(),
        [string]$WorkingDirectory = $script:ProjectRoot
    )

    $commandInfo = Get-Command $FilePath -ErrorAction Stop
    $resolvedPath = if ($commandInfo.Path) { $commandInfo.Path } else { $FilePath }
    $launchFile = $resolvedPath
    $launchArgs = $ArgumentList
    if ([System.IO.Path]::GetExtension($resolvedPath).ToLowerInvariant() -eq '.ps1') {
        $launchFile = Resolve-PowerShellExecutable
        $launchArgs = @('-NoProfile', '-File', $resolvedPath) + $ArgumentList
    }

    $stdoutFile = New-TemporaryFile
    $stderrFile = New-TemporaryFile
    try {
        $process = Start-Process `
            -FilePath $launchFile `
            -ArgumentList $launchArgs `
            -WorkingDirectory $WorkingDirectory `
            -NoNewWindow `
            -Wait `
            -PassThru `
            -RedirectStandardOutput $stdoutFile.FullName `
            -RedirectStandardError $stderrFile.FullName

        $stdoutLines = @()
        $stderrLines = @()
        if (Test-Path $stdoutFile.FullName) {
            $stdoutLines = Get-Content -Path $stdoutFile.FullName -ErrorAction SilentlyContinue
        }
        if (Test-Path $stderrFile.FullName) {
            $stderrLines = Get-Content -Path $stderrFile.FullName -ErrorAction SilentlyContinue
        }

        return @{
            ExitCode = $process.ExitCode
            StdOut   = $stdoutLines
            StdErr   = $stderrLines
        }
    } finally {
        Remove-Item $stdoutFile.FullName -Force -ErrorAction SilentlyContinue
        Remove-Item $stderrFile.FullName -Force -ErrorAction SilentlyContinue
    }
}

function Get-AstGrepRuleDirectories {
    $defaultRulesDir = Join-Path $script:ProjectRoot 'rules'
    $configPath = Join-Path $script:ProjectRoot 'sgconfig.yml'
    if (-not (Test-Path $configPath)) {
        if (Test-Path $defaultRulesDir) { return @($defaultRulesDir) }
        return @()
    }

    $ruleDirs = @()
    $inRuleDirsSection = $false
    $lines = Get-Content -Path $configPath -ErrorAction SilentlyContinue
    foreach ($line in $lines) {
        $trimmed = $line.Trim()
        if ($trimmed -match '^ruleDirs\s*:') {
            $inRuleDirsSection = $true
            continue
        }

        if (-not $inRuleDirsSection) {
            continue
        }

        if ($trimmed -match '^-\s*(.+)$') {
            $raw = $matches[1].Trim().Trim("'`"")
            $candidate = if ([System.IO.Path]::IsPathRooted($raw)) {
                $raw
            } else {
                Join-Path $script:ProjectRoot $raw
            }
            if (Test-Path $candidate) {
                $ruleDirs += (Resolve-Path -Path $candidate).Path
            }
            continue
        }

        if ($trimmed -match '^[A-Za-z0-9_]+\s*:') {
            break
        }
    }

    if ($ruleDirs.Count -eq 0 -and (Test-Path $defaultRulesDir)) {
        return @($defaultRulesDir)
    }

    return @($ruleDirs | Select-Object -Unique)
}

function Invoke-AstGrepCheck {
    param(
        [switch]$EnableAutoFix
    )

    if (-not (Get-Command sg -ErrorAction SilentlyContinue)) {
        Write-BuildStep 'ast-grep (sg) not found; skipping ast-grep checks' 'warning'
        return $true
    }

    $scanArgs = @('scan')
    $configPath = Join-Path $script:ProjectRoot 'sgconfig.yml'
    if (Test-Path $configPath) {
        $scanArgs += @('--config', $configPath)
    }

    $isDiffMode = $false
    $diffFiles = @()
    if (-not $EnableAutoFix) {
        if (Get-Command git -ErrorAction SilentlyContinue) {
            $gitOutput = Invoke-NativeCommandCapture -FilePath 'git' -ArgumentList @('diff', '--name-only', 'HEAD') -WorkingDirectory $script:ProjectRoot
            if ($gitOutput.ExitCode -eq 0) {
                $isDiffMode = $true
                foreach ($file in $gitOutput.StdOut) {
                    if (-not [string]::IsNullOrWhiteSpace($file) -and (Test-Path (Join-Path $script:ProjectRoot $file) -PathType Leaf)) {
                        $diffFiles += $file
                    }
                }
            }
        }

        if ($isDiffMode) {
            if ($diffFiles.Count -eq 0) {
                Write-BuildStep 'AST-grep fast-path: no modified files to scan' 'success'
                return $true
            }
            $scanArgs += $diffFiles
        }
    }

    $astGrepParsePattern = 'Cannot parse rule|not a valid ast-grep rule|Fail to parse yaml as Rule|Rule contains invalid pattern matcher|`rule` is not configured correctly|Invalid config|No such file'

    $rules = @()
    $ruleDirs = Get-AstGrepRuleDirectories
    foreach ($ruleDir in $ruleDirs) {
        $rules += Get-ChildItem -Path $ruleDir -Filter '*.yml' -File -Recurse -ErrorAction SilentlyContinue
    }
    if ($rules.Count -gt 0) {
        $rules = @($rules | Sort-Object -Property FullName -Unique)
        if ($rules.Count -gt 0) {
            Write-BuildStep "Validating $($rules.Count) AST-grep rule file(s)..." 'running'
            $dummyFile = New-TemporaryFile
            try {
                'validation_probe' | Set-Content -Path $dummyFile.FullName -Encoding UTF8
                foreach ($rule in $rules) {
                    $validationResult = Invoke-NativeCommandCapture -FilePath 'sg' -ArgumentList @('scan', '--rule', $rule.FullName, $dummyFile.FullName) -WorkingDirectory $script:ProjectRoot
                    $validationText = (($validationResult.StdOut + $validationResult.StdErr) -join "`n")
                    if ($validationResult.ExitCode -gt 1 -or ($validationText -match $astGrepParsePattern)) {
                        Write-BuildStep "AST-grep rule validation failed: $($rule.Name)" 'error'
                        return $false
                    }
                }
            } finally {
                Remove-Item $dummyFile.FullName -Force -ErrorAction SilentlyContinue
            }
        }
    }

    if ($EnableAutoFix) {
        Write-BuildStep 'Applying AST-grep auto-fixes (--update-all)...' 'running'
        $fixResult = Invoke-NativeCommandCapture -FilePath 'sg' -ArgumentList ($scanArgs + @('--update-all')) -WorkingDirectory $script:ProjectRoot
        if ($fixResult.ExitCode -ne 0) {
            Write-BuildStep 'AST-grep auto-fix pass failed' 'error'
            return $false
        }
        Write-BuildStep 'AST-grep auto-fix pass complete' 'success'
    }

    Write-BuildStep 'Running AST-grep scan...' 'running'
    $scanResult = Invoke-NativeCommandCapture -FilePath 'sg' -ArgumentList ($scanArgs + @('--json=stream')) -WorkingDirectory $script:ProjectRoot
    $scanOutput = $scanResult.StdOut
    $scanErrors = ($scanResult.StdErr -join "`n")
    if ($scanResult.ExitCode -gt 1 -or ($scanErrors -match $astGrepParsePattern)) {
        Write-BuildStep "AST-grep reported a parser/config error: $($scanResult.StdErr[0])" 'error'
        return $false
    }

    $findingsByRule = @{}
    $severityTotals = @{
        error   = 0
        warning = 0
        info    = 0
        hint    = 0
        other   = 0
    }

    foreach ($line in $scanOutput) {
        if ([string]::IsNullOrWhiteSpace($line)) {
            continue
        }

        $lineText = [string]$line
        if ($lineText -match '^Error:' -or $lineText -match '^Help:' -or $lineText -match 'Caused by') {
            Write-BuildStep "AST-grep reported a parser/config error: $lineText" 'error'
            return $false
        }

        try {
            $parsed = $lineText | ConvertFrom-Json -ErrorAction Stop
        } catch {
            continue
        }

        if (-not $parsed.ruleId) {
            continue
        }

        $ruleId = [string]$parsed.ruleId
        $severity = [string]$parsed.severity
        $message = [string]$parsed.message
        if ([string]::IsNullOrWhiteSpace($severity)) {
            $severity = 'other'
        }

        if (-not $findingsByRule.ContainsKey($ruleId)) {
            $findingsByRule[$ruleId] = [ordered]@{
                Count    = 0
                Severity = $severity
                Message  = $message
            }
        }

        $findingsByRule[$ruleId].Count += 1
        $sevKey = $severity.ToLowerInvariant()
        if ($severityTotals.ContainsKey($sevKey)) {
            $severityTotals[$sevKey] += 1
        } else {
            $severityTotals.other += 1
        }
    }

    if ($findingsByRule.Count -eq 0) {
        if ($scanResult.ExitCode -eq 0) {
            Write-BuildStep 'AST-grep checks passed (no findings)' 'success'
            return $true
        }

        $fallbackMessage = if ($scanResult.StdErr.Count -gt 0) { $scanResult.StdErr[0] } else { 'AST-grep scan failed without structured findings.' }
        Write-BuildStep "AST-grep scan failed: $fallbackMessage" 'error'
        return $false
    }

    $sortedRules = $findingsByRule.GetEnumerator() | Sort-Object { $_.Value.Count } -Descending
    $topRules = $sortedRules | Select-Object -First 10
    foreach ($rule in $topRules) {
        $summary = Format-InvariantString -Template '{0} ({1}) x{2}: {3}' -Args @($rule.Key, $rule.Value.Severity, $rule.Value.Count, $rule.Value.Message)
        Write-BuildStep $summary 'warning'
    }
    if ($findingsByRule.Count -gt 10) {
        Write-BuildStep "$($findingsByRule.Count - 10) additional AST-grep rule(s) omitted from summary" 'warning'
    }

    $errorCount = $severityTotals.error
    $warningCount = $severityTotals.warning
    $infoCount = $severityTotals.info + $severityTotals.hint + $severityTotals.other
    Write-BuildStep (Format-InvariantString -Template 'AST-grep totals: errors={0} warnings={1} info={2}' -Args @($errorCount, $warningCount, $infoCount)) 'running'

    if ($errorCount -gt 0) {
        Write-BuildStep 'AST-grep scan blocked the pipeline due to error-level findings' 'error'
        return $false
    }

    Write-BuildStep 'AST-grep scan completed with non-blocking findings only' 'warning'
    return $true
}

function Invoke-RustQuality {
    param(
        [ValidateSet('check', 'clippy', 'fmt')]
        [string]$Mode,
        [switch]$WriteMode,
        [ValidateSet('all', 'inference')]
        [string]$TargetScope = 'all'
    )

    $targets = Get-QualityTargets -Scope $TargetScope
    if (-not $targets -or $targets.Count -eq 0) {
        Write-BuildStep 'No Rust quality targets found' 'warning'
        return $true
    }

    $success = $true
    foreach ($target in $targets) {
        $timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
        $logFile = Join-Path $script:BuildLogsDir "quality_rust_${Mode}_$($target.Name)_$timestamp.log"
        $cargoArgs = switch ($Mode) {
            'check' { @('check', '--workspace', '--all-targets') }
            'clippy' { @('clippy', '--workspace', '--all-targets', '--', '-D', 'warnings') }
            'fmt' {
                if ($WriteMode) { @('fmt', '--all') } else { @('fmt', '--all', '--', '--check') }
            }
        }

        Write-BuildStep "Rust $Mode ($($target.Name))..." 'running'
        $ok = Invoke-RustBuildCommand -Path $target.Path -CargoArgs $cargoArgs -LogFile $logFile
        if (-not $ok) {
            $success = $false
            Write-BuildStep "Rust $Mode failed for $($target.Name)" 'error'
        } else {
            Write-BuildStep "Rust $Mode passed for $($target.Name)" 'success'
        }
    }

    return $success
}

function Invoke-RustInferenceQuality {
    param(
        [ValidateSet('check', 'clippy', 'fmt')]
        [string]$Mode,
        [switch]$WriteMode
    )

    $inferencePath = Join-Path $script:ProjectRoot 'Native\pcai_core\pcai_inference'
    if (-not (Test-Path $inferencePath)) {
        Write-BuildStep 'pcai_inference crate not found for targeted quality checks' 'warning'
        return $true
    }

    $timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    $commands = @()
    switch ($Mode) {
        'check' {
            $commands += , @('check', '--lib', '--no-default-features')
            $commands += , @('check', '--lib', '--features', 'server')
        }
        'clippy' {
            $commands += , @('clippy', '--no-default-features', '--features', 'server,ffi', '--', '-D', 'warnings')
        }
        'fmt' {
            if ($WriteMode) {
                $commands += , @('fmt', '--all')
            } else {
                $commands += , @('fmt', '--all', '--', '--check')
            }
        }
    }

    $allOk = $true
    $index = 0
    foreach ($cargoArgs in $commands) {
        $index += 1
        $logFile = Join-Path $script:BuildLogsDir "quality_inference_${Mode}_${index}_$timestamp.log"
        Write-BuildStep "pcai_inference $Mode command $index..." 'running'
        $ok = Invoke-RustBuildCommand -Path $inferencePath -CargoArgs $cargoArgs -LogFile $logFile
        if (-not $ok) {
            $allOk = $false
            Write-BuildStep "pcai_inference $Mode command $index failed" 'error'
        } else {
            Write-BuildStep "pcai_inference $Mode command $index passed" 'success'
        }
    }

    return $allOk
}

function Invoke-DotnetFormatQuality {
    param(
        [switch]$WriteMode
    )

    if (-not (Get-Command dotnet -ErrorAction SilentlyContinue)) {
        Write-BuildStep 'dotnet not found; skipping C# format checks' 'warning'
        return $true
    }

    $projects = @(
        'Native\PcaiChatTui\PcaiChatTui.csproj',
        'Native\PcaiNative\PcaiNative.csproj',
        'Native\PcaiServiceHost\PcaiServiceHost.csproj'
    ) | ForEach-Object { Join-Path $script:ProjectRoot $_ } | Where-Object { Test-Path $_ }

    $nukeRoot = Resolve-NukeNulProjectRoot
    if ($nukeRoot) {
        $projects += (Join-Path $nukeRoot 'NukeNul.csproj')
    }

    if (-not $projects -or $projects.Count -eq 0) {
        Write-BuildStep 'No C# project files found for dotnet format' 'warning'
        return $true
    }

    $allOk = $true
    foreach ($project in $projects) {
        $name = Split-Path $project -Leaf
        $timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
        $logFile = Join-Path $script:BuildLogsDir "quality_dotnet_${name}_$timestamp.log"
        $args = @('format', $project, '--no-restore', '--verbosity', 'minimal')
        if (-not $WriteMode) { $args += '--verify-no-changes' }

        Write-BuildStep "dotnet format ($name)..." 'running'
        & dotnet @args 2>&1 | Tee-Object -FilePath $logFile | Out-Null
        if ($LASTEXITCODE -ne 0) {
            $allOk = $false
            Write-BuildStep "dotnet format failed for $name" 'error'
        } else {
            Write-BuildStep "dotnet format passed for $name" 'success'
        }
    }

    return $allOk
}

function Invoke-PowerShellLiteralPathCheck {
    if (-not (Get-Command rg -ErrorAction SilentlyContinue)) {
        Write-BuildStep 'rg not found; skipping PowerShell literal path guard' 'warning'
        return $true
    }

    $pattern = '(?i)[A-Z]:\\Users\\[^\\]+\\PC_AI'
    $rgArgs = @(
        '--line-number',
        '--with-filename',
        '--pcre2',
        '--glob', '*.ps1',
        '--glob', '*.psm1',
        '--glob', '*.psd1',
        '--glob', '!**/target/**',
        '--glob', '!**/bin/**',
        '--glob', '!**/obj/**',
        '--glob', '!**/node_modules/**',
        '--glob', '!**/.git/**',
        '--glob', '!**/.pcai/**',
        '--glob', '!**/packages/**',
        '--glob', '!**/artifacts/**',
        '--glob', '!**/output/**',
        $pattern,
        $script:ProjectRoot
    )

    $scanResult = Invoke-NativeCommandCapture -FilePath 'rg' -ArgumentList $rgArgs -WorkingDirectory $script:ProjectRoot
    if ($scanResult.ExitCode -gt 1) {
        $reason = if ($scanResult.StdErr.Count -gt 0) { $scanResult.StdErr[0] } else { 'unknown rg error' }
        Write-BuildStep "PowerShell literal path guard failed: $reason" 'error'
        return $false
    }

    $matches = @($scanResult.StdOut | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
    if ($matches.Count -gt 0) {
        Write-BuildStep "PowerShell literal path guard found $($matches.Count) hardcoded repo path hit(s)" 'error'
        $matches | Select-Object -First 20 | ForEach-Object {
            Write-BuildStep $_ 'error'
        }
        if ($matches.Count -gt 20) {
            Write-BuildStep "$($matches.Count - 20) additional hardcoded path hit(s) omitted from summary" 'warning'
        }
        return $false
    }

    Write-BuildStep 'PowerShell literal path guard passed' 'success'
    return $true
}

function Invoke-PowerShellQuality {
    param(
        [switch]$WriteMode
    )

    if (-not (Get-Module -ListAvailable PSScriptAnalyzer)) {
        Write-BuildStep 'PSScriptAnalyzer module not found; skipping PowerShell lint checks' 'warning'
        return $true
    }

    Import-Module PSScriptAnalyzer -ErrorAction Stop
    $settingsPath = Join-Path $script:ProjectRoot 'PSScriptAnalyzerSettings.psd1'
    $excludeRegex = '\\(target|bin|obj|node_modules|\.git|\.pcai|packages|artifacts|output)\\'
    $psFiles = Get-ChildItem -Path $script:ProjectRoot -Recurse -File -ErrorAction SilentlyContinue | Where-Object {
        $_.Extension -in @('.ps1', '.psm1', '.psd1') -and $_.FullName -notmatch $excludeRegex
    }

    if (-not $psFiles -or $psFiles.Count -eq 0) {
        Write-BuildStep 'No PowerShell files found for quality checks' 'warning'
        return $true
    }

    if ($WriteMode) {
        Write-BuildStep 'Running PowerShell formatter pass...' 'running'
        foreach ($file in $psFiles) {
            try {
                $formatted = Invoke-Formatter -Path $file.FullName -Settings $settingsPath
                Set-Content -Path $file.FullName -Value $formatted -Encoding UTF8
            } catch {
                Write-BuildStep "Formatter failed for $($file.FullName): $($_.Exception.Message)" 'error'
                return $false
            }
        }
    }

    $syntaxErrors = @()
    foreach ($file in $psFiles) {
        if (-not (Test-Path $file.FullName)) {
            Write-BuildStep "PowerShell file disappeared during syntax scan; skipping $($file.FullName)" 'warning'
            continue
        }

        try {
            $tokens = $null
            $parseErrors = $null
            [System.Management.Automation.Language.Parser]::ParseFile($file.FullName, [ref]$tokens, [ref]$parseErrors) | Out-Null
            if ($parseErrors) {
                foreach ($parseError in $parseErrors) {
                    $extent = $parseError.Extent
                    $syntaxErrors += Format-InvariantString -Template '{0}:{1}:{2} {3}' -Args @($file.FullName, $extent.StartLineNumber, $extent.StartColumnNumber, $parseError.Message)
                }
            }
        } catch {
            Write-BuildStep "PowerShell syntax scan failed for $($file.FullName): $($_.Exception.Message)" 'error'
            return $false
        }
    }

    if ($syntaxErrors.Count -gt 0) {
        Write-BuildStep "PowerShell syntax check reported $($syntaxErrors.Count) parse issue(s)" 'error'
        $syntaxErrors | Select-Object -First 10 | ForEach-Object { Write-BuildStep $_ 'error' }
        if ($syntaxErrors.Count -gt 10) {
            Write-BuildStep "$($syntaxErrors.Count - 10) additional parse issue(s) omitted from summary" 'warning'
        }
        return $false
    }

    $issues = @()
    foreach ($file in $psFiles) {
        if (-not (Test-Path $file.FullName)) {
            Write-BuildStep "PowerShell file disappeared during analyzer scan; skipping $($file.FullName)" 'warning'
            continue
        }

        try {
            $issues += Invoke-ScriptAnalyzer -Path $file.FullName -Settings $settingsPath -Severity Warning, Error
        } catch {
            Write-BuildStep "PowerShell analyzer failed for $($file.FullName): $($_.Exception.Message)" 'error'
            return $false
        }
    }

    if ($issues.Count -gt 0) {
        Write-BuildStep "PowerShell analyzer reported $($issues.Count) issue(s)" 'error'
        $issues | Select-Object -First 10 | ForEach-Object {
            $issuePath = if ($_ -and $_.ScriptPath) { $_.ScriptPath } else { '(unknown)' }
            $issueLine = if ($_ -and $null -ne $_.Line) { [int]$_.Line } else { 0 }
            $issueColumn = if ($_ -and $null -ne $_.Column) { [int]$_.Column } else { 0 }
            $issueRule = if ($_ -and $_.RuleName) { $_.RuleName } else { 'UnknownRule' }
            $issueMessage = if ($_ -and $_.Message) { $_.Message } else { 'No analyzer message provided.' }
            Write-BuildStep "${issuePath}:${issueLine}:${issueColumn} [$issueRule] $issueMessage" 'error'
        }
        if ($issues.Count -gt 10) {
            Write-BuildStep "$($issues.Count - 10) additional analyzer issue(s) omitted from summary" 'warning'
        }
        return $false
    }

    if (-not (Invoke-PowerShellLiteralPathCheck)) {
        return $false
    }

    Write-BuildStep 'PowerShell analyzer checks passed' 'success'
    return $true
}

function Invoke-TomlQuality {
    param(
        [switch]$WriteMode
    )

    if (Get-Command taplo -ErrorAction SilentlyContinue) {
        $taploConfig = Join-Path $script:ProjectRoot 'taplo.toml'
        $args = @('fmt')
        if (Test-Path $taploConfig) { $args += @('--config', $taploConfig) }
        if (-not $WriteMode) { $args += '--check' }
        $args += '.'

        Push-Location $script:ProjectRoot
        try {
            & taplo @args 2>&1 | Out-Null
            if ($LASTEXITCODE -ne 0) {
                Write-BuildStep 'taplo TOML quality pass failed' 'error'
                return $false
            }
            Write-BuildStep 'taplo TOML quality pass complete' 'success'
            return $true
        } finally {
            Pop-Location
        }
    }

    Write-BuildStep 'taplo not found; skipping TOML format checks' 'warning'
    return $true
}

function Invoke-RustDependencyRefresh {
    Write-BuildPhase 'Dependencies' 'Refreshing Rust dependency graph and warming local cache'

    $targets = Get-QualityTargets
    if (-not $targets -or $targets.Count -eq 0) {
        Write-BuildStep 'No Rust targets found for dependency refresh' 'warning'
        return $true
    }

    $ok = $true
    foreach ($target in $targets) {
        $timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
        $logFile = Join-Path $script:BuildLogsDir "deps_$($target.Name)_$timestamp.log"

        if ($script:DependencyStrategy -eq 'update') {
            Write-BuildStep "Updating lockfile in $($target.Name)..." 'running'
            Push-Location $target.Path
            try {
                & cargo update -w 2>&1 | Tee-Object -FilePath $logFile | Out-Null
                if ($LASTEXITCODE -ne 0) {
                    $ok = $false
                    Write-BuildStep "cargo update failed for $($target.Name)" 'error'
                    continue
                }
            } finally {
                Pop-Location
            }
        }

        Write-BuildStep "Fetching crates in $($target.Name)..." 'running'
        $fetchArgs = Add-CargoDependencyFlags -CargoArgs @('fetch')
        Push-Location $target.Path
        try {
            & cargo @fetchArgs 2>&1 | Tee-Object -FilePath $logFile | Out-Null
            if ($LASTEXITCODE -ne 0) {
                $ok = $false
                Write-BuildStep "cargo fetch failed for $($target.Name)" 'error'
            } else {
                Write-BuildStep "Dependency refresh passed for $($target.Name)" 'success'
            }
        } finally {
            Pop-Location
        }
    }

    return $ok
}

function Invoke-RagQuality {
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        Write-BuildStep 'uv not found; skipping RAG semantic overlap checks' 'warning'
        return $true
    }

    $clusterScript = Join-Path $script:ProjectRoot 'Deploy\rag-database\cluster_rag.py'
    if (-not (Test-Path $clusterScript)) {
        return $true
    }

    Write-BuildStep 'Running RAG cluster deduplication crawler...' 'running'
    $dbUrl = if ($env:PCAI_DB_URL) { $env:PCAI_DB_URL } else { 'postgresql://pcai_user:changeme@localhost/semantic_rag' }

    $ragArgs = @('run', $clusterScript, '--db', $dbUrl, '--ci-mode')
    $ragResult = Invoke-NativeCommandCapture -FilePath 'uv' -ArgumentList $ragArgs -WorkingDirectory $script:ProjectRoot

    if ($ragResult.ExitCode -eq 2) {
        $msg = if ($ragResult.StdOut.Count -gt 0) { $ragResult.StdOut[-1] } else { 'Semantic overlap detected.' }
        Write-BuildStep "RAG cluster deduplication failed: $msg" 'error'
        return $false
    } elseif ($ragResult.ExitCode -ne 0) {
        $msg = if ($ragResult.StdErr.Count -gt 0) { $ragResult.StdErr[0] } else { 'Execution failed.' }
        Write-BuildStep "RAG cluster deduplication error: $msg" 'warning'
        return $true
    }

    Write-BuildStep 'RAG cluster deduplication passed (no semantic overlap violations)' 'success'
    return $true
}

function Invoke-QualityPipeline {
    param(
        [ValidateSet('lint', 'format', 'fix')]
        [string]$Mode,
        [ValidateSet('all', 'rust-check', 'rust-clippy', 'rust-fmt', 'rust-inference-check', 'rust-inference-clippy', 'rust-inference-fmt', 'dotnet-format', 'powershell', 'docs', 'astgrep', 'toml', 'rag-quality')]
        [string]$Profile
    )

    Write-BuildPhase 'Quality' "Running quality pipeline ($Mode / $Profile)"
    $fixMode = $AutoFix -or ($Mode -eq 'fix')
    $ok = $true

    switch ($Profile) {
        'rust-check' { return (Invoke-RustQuality -Mode 'check') }
        'rust-clippy' { return (Invoke-RustQuality -Mode 'clippy') }
        'rust-fmt' { return (Invoke-RustQuality -Mode 'fmt' -WriteMode:$fixMode) }
        'rust-inference-check' { return (Invoke-RustInferenceQuality -Mode 'check') }
        'rust-inference-clippy' { return (Invoke-RustInferenceQuality -Mode 'clippy') }
        'rust-inference-fmt' { return (Invoke-RustInferenceQuality -Mode 'fmt' -WriteMode:$fixMode) }
        'dotnet-format' { return (Invoke-DotnetFormatQuality -WriteMode:$fixMode) }
        'powershell' { return (Invoke-PowerShellQuality -WriteMode:$fixMode) }
        'docs' { return (Invoke-PrettierQuality -WriteMode:$fixMode) }
        'astgrep' { return (Invoke-AstGrepCheck -EnableAutoFix:$fixMode) }
        'toml' { return (Invoke-TomlQuality -WriteMode:$fixMode) }
        'rag-quality' { return (Invoke-RagQuality) }
        default {
            $profiles = @()
            if ($Mode -eq 'lint') {
                $profiles = @('astgrep', 'rag-quality', 'rust-check', 'rust-clippy', 'rust-fmt', 'dotnet-format', 'powershell', 'docs', 'toml')
            } elseif ($Mode -eq 'format') {
                $profiles = @('rust-fmt', 'dotnet-format', 'powershell', 'docs', 'toml')
            } else {
                $profiles = @('astgrep', 'rag-quality', 'rust-fmt', 'dotnet-format', 'powershell', 'docs', 'toml')
            }

            Write-BuildStep "Launching $($profiles.Count) parallel quality gates concurrently..." 'running'

            $jobs = @()
            $pwshExe = (Get-Process -Id $PID).Path
            if (-not $pwshExe) { $pwshExe = 'pwsh' }

            $buildScriptPath = Join-Path $script:ProjectRoot 'Build.ps1'

            # Consolidate concurrent outputs into NDJSON
            $qaLogFile = Join-Path $script:BuildLogsDir 'BUILD_QA_LOGS.jsonl'
            if (Test-Path $qaLogFile) { Remove-Item $qaLogFile -Force -ErrorAction SilentlyContinue }

            foreach ($p in $profiles) {
                $jobArgs = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $buildScriptPath, '-Component', 'lint', '-LintProfile', $p)
                if ($AutoFix -or $Mode -eq 'fix') { $jobArgs += '-AutoFix' }
                if ($SkipQualityGate) { $jobArgs += '-SkipQualityGate' }

                $outLog = Join-Path $script:BuildLogsDir "qa_job_$($p)_out.log"
                $errLog = Join-Path $script:BuildLogsDir "qa_job_$($p)_err.log"

                $proc = Start-Process -FilePath $pwshExe -ArgumentList $jobArgs -NoNewWindow -PassThru -RedirectStandardOutput $outLog -RedirectStandardError $errLog
                $jobs += [PSCustomObject]@{ Profile = $p; Process = $proc; OutLog = $outLog; ErrLog = $errLog }
            }

            foreach ($job in $jobs) {
                $job.Process.WaitForExit()
                $exitCode = $job.Process.ExitCode
                $status = if ($exitCode -eq 0) { 'success' } else { 'error' }
                Write-BuildStep "Quality Gate Pipeline: $($job.Profile) exited with code $exitCode" $status

                # Append raw outputs to the centralized QA tracking JSONL container securely
                [PSCustomObject]@{
                    Profile  = $job.Profile
                    ExitCode = $exitCode
                    Output   = if (Test-Path $job.OutLog) { Get-Content $job.OutLog -Raw } else { '' }
                    Error    = if (Test-Path $job.ErrLog) { Get-Content $job.ErrLog -Raw } else { '' }
                } | ConvertTo-Json -Compress | Out-File -FilePath $qaLogFile -Append -Encoding UTF8

                if ($exitCode -ne 0) { $ok = $false }
            }
        }
    }

    return $ok
}

function Test-QualityProfileRequiresRust {
    param(
        [ValidateSet('lint', 'format', 'fix')]
        [string]$Mode,
        [ValidateSet('all', 'rust-check', 'rust-clippy', 'rust-fmt', 'rust-inference-check', 'rust-inference-clippy', 'rust-inference-fmt', 'dotnet-format', 'powershell', 'docs', 'astgrep', 'toml', 'rag-quality')]
        [string]$Profile
    )

    if ($Profile -in @('rust-check', 'rust-clippy', 'rust-fmt', 'rust-inference-check', 'rust-inference-clippy', 'rust-inference-fmt')) {
        return $true
    }

    if ($Profile -eq 'all') {
        return ($Mode -in @('lint', 'format', 'fix'))
    }

    return $false
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
            # Compatibility alias: some build paths emit only pcai_inference_lib.dll.
            # Stage a pcai_inference.dll copy for PcaiNative P/Invoke resolution.
            if (-not (Test-Path $dllPath)) {
                Copy-Item $libDllPath -Destination (Join-Path $artifactDir 'pcai_inference.dll') -Force
                $artifacts += 'pcai_inference.dll'
            }
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
        [string]$ProjectPath,
        [string]$ProjectRelativePath,
        [Parameter(Mandatory)]
        [string]$Configuration
    )

    $componentStart = Get-Date
    if (-not $ProjectPath) {
        if (-not $ProjectRelativePath) {
            throw 'ProjectPath or ProjectRelativePath is required.'
        }
        $projectPath = Join-Path $script:ProjectRoot $ProjectRelativePath
    }

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
        if ($script:VersionInfo) {
            $dotnetArgs += @(
                "-p:Version=$($script:VersionInfo.SemVer)",
                "-p:AssemblyVersion=$($script:VersionInfo.AssemblyVersion)",
                "-p:FileVersion=$($script:VersionInfo.FileVersion)",
                "-p:InformationalVersion=$($script:VersionInfo.InformationalVersion)"
            )
        }

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
                $published = Publish-StagedArtifact -SourcePath $file.FullName -DestinationDirectory $artifactDir -DestinationFileName $file.Name -ArtifactKind 'managed-dotnet'
                if ($published) {
                    $artifacts += $file.Name
                }
            }

            if ($ComponentName -eq 'pcai-native') {
                $nativeBundle = Publish-PcaiNativeBundle -PublishRoot $publishRoot -Configuration $Configuration
                if ($nativeBundle) {
                    $artifacts += ("native-bundles/{0}" -f $nativeBundle.BundleName)
                }
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

function Invoke-MediaBuild {
    param(
        [string]$Configuration,
        [bool]$EnableCuda
    )

    $componentStart = Get-Date
    $mediaRoot = Join-Path $script:ProjectRoot 'Native\pcai_core'
    $timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    $logFile = Join-Path $script:BuildLogsDir "build_media_$timestamp.log"
    $artifactDir = Join-Path $script:BuildArtifactsDir 'pcai-media'

    if (-not (Test-Path $artifactDir)) {
        New-Item -ItemType Directory -Path $artifactDir -Force | Out-Null
    }

    Write-BuildStep 'Building pcai-media (Janus-Pro media agent)...' 'running'
    try {
        $cargoArgs = @('build')
        if ($Configuration -eq 'Release') { $cargoArgs += '--release' }
        $cargoArgs += @('--package', 'pcai-media', '--package', 'pcai-media-server')

        $featureList = @('pcai-media/upscale')
        if ($EnableCuda) {
            $featureList += 'pcai-media/cuda'
            $featureList += 'pcai-media/flash-attn'
        }
        $cargoArgs += @('--features', ($featureList -join ','))

        $rustOk = Invoke-RustBuildCommand -Path $mediaRoot -CargoArgs $cargoArgs -LogFile $logFile
        if (-not $rustOk) {
            return @{
                Success   = $false
                Duration  = (Get-Date) - $componentStart
                Artifacts = @()
                LogFile   = $logFile
            }
        }

        $targetDir = Resolve-CargoOutputDirectory -ProjectDir $mediaRoot -Configuration $Configuration
        $artifacts = @()

        foreach ($file in @('pcai_media.dll', 'pcai_media.dll.lib', 'pcai_media.pdb', 'pcai-media.exe', 'pcai-media.pdb')) {
            $src = Join-Path $targetDir $file
            if (Test-Path $src) {
                Publish-StagedArtifact -SourcePath $src -DestinationDirectory $artifactDir -DestinationFileName $file -ArtifactKind 'native-rust' | Out-Null
                $artifacts += $file
            }
        }

        return @{
            Success   = $true
            Duration  = (Get-Date) - $componentStart
            Artifacts = $artifacts
            LogFile   = $logFile
        }
    } catch {
        Write-BuildStep "pcai-media build failed: $($_.Exception.Message)" 'error'
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
    $nukeRoot = Resolve-NukeNulProjectRoot
    if (-not $nukeRoot) {
        Write-BuildStep 'NukeNul project root not found. Set NUKENUL_ROOT or install the external repo at C:\codedev\nukenul' 'warning'
        return @{ Success = $false; Duration = (Get-Date) - $componentStart; Artifacts = @() }
    }
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
            Publish-StagedArtifact -SourcePath $rustDll -DestinationDirectory $nukeRoot -DestinationFileName 'nuker_core.dll' -ArtifactKind 'native-rust' | Out-Null
            Publish-StagedArtifact -SourcePath $rustDll -DestinationDirectory $artifactDir -DestinationFileName 'nuker_core.dll' -ArtifactKind 'native-rust' | Out-Null
        }

        $dotnetResult = Invoke-DotnetComponentBuild -ComponentName 'nukenul' -ProjectPath $projectPath -Configuration $Configuration
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
        releaseTag         = if ($env:PCAI_RELEASE_TAG) { $env:PCAI_RELEASE_TAG } elseif ($script:VersionInfo) { $script:VersionInfo.ReleaseTag } else { '' }
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
Write-Host "  LintProfile:   $LintProfile" -ForegroundColor White
Write-Host "  DepStrategy:   $DependencyStrategy" -ForegroundColor White
Write-Host "  AutoFix:       $(if ($AutoFix) { 'Yes' } else { 'No' })" -ForegroundColor White
Write-Host "  SkipQGate:     $(if ($SkipQualityGate) { 'Yes' } else { 'No' })" -ForegroundColor White
Write-Host "  FG Args:       $(if ($FunctionGemmaArgs.Count -gt 0) { ($FunctionGemmaArgs -join ' ') } else { '(none)' })" -ForegroundColor White
Write-Host "  Artifacts Root: $script:ArtifactsRoot" -ForegroundColor White

# Clean if requested
if ($Clean) {
    Clear-BuildArtifacts
}

# Initialize directories
Initialize-BuildDirectories

# Quality-only components
if ($Component -in @('lint', 'format', 'fix')) {
    $mode = if ($Component -eq 'lint') { 'lint' } elseif ($Component -eq 'format') { 'format' } else { 'fix' }
    $requiresRustToolchain = Test-QualityProfileRequiresRust -Mode $mode -Profile $LintProfile
    if ($requiresRustToolchain) {
        $versionInfo = Initialize-BuildVersion
        Initialize-CargoToolsDefaults | Out-Null
    } else {
        Write-BuildStep 'Skipping Rust toolchain initialization for non-Rust quality profile' 'skip'
    }
    $qualitySuccess = Invoke-QualityPipeline -Mode $mode -Profile $LintProfile
    exit $(if ($qualitySuccess) { 0 } else { 1 })
}

# Dependency refresh-only component
if ($Component -eq 'deps') {
    $versionInfo = Initialize-BuildVersion
    Initialize-CargoToolsDefaults | Out-Null
    $depsSuccess = Invoke-RustDependencyRefresh
    exit $(if ($depsSuccess) { 0 } else { 1 })
}

# Initialize version information (sets PCAI_* environment variables)
$versionInfo = Initialize-BuildVersion

# Initialize CargoTools defaults / wrappers
Initialize-CargoToolsDefaults | Out-Null

# Initialize CUDA environment when GPU builds are requested.
# Sets NVCC_CCBIN, INCLUDE, LIB, CUDA_PATH, and PATH so nvcc can find
# the MSVC host compiler (cl.exe) and system headers.
if ($EnableCuda) {
    $cudaInitScript = Join-Path $script:ProjectRoot 'Tools\Initialize-CudaEnvironment.ps1'
    if (Test-Path $cudaInitScript) {
        $cudaResult = & $cudaInitScript
        if ($cudaResult.Found) {
            Write-BuildStep "CUDA environment initialized: $($cudaResult.SelectedVersion) (NVCC_CCBIN=$($cudaResult.NvccCcbin))" 'success'
        } else {
            Write-BuildStep 'CUDA not found - GPU build may fail' 'warning'
        }
    }
}

# Pre-build quality gate
if (-not $SkipQualityGate) {
    $qualityGatePassed = Invoke-AstGrepCheck
    if (-not $qualityGatePassed) {
        exit 1
    }
}

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
    'media' { @('media') }
    'nukenul' { @('nukenul') }
    'lint' { @() }
    'format' { @() }
    'fix' { @() }
    'deps' { @() }
    'native' { @('pcainative', 'servicehost', 'tui', 'nukenul') }
    'all' { @('llamacpp', 'mistralrs', 'functiongemma', 'media', 'pcainative', 'servicehost', 'tui', 'nukenul') }
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
    } elseif ($target -eq 'media') {
        $result = Invoke-MediaBuild -Configuration $Configuration -EnableCuda $EnableCuda
        $results['pcai-media'] = $result
        Write-BuildResult 'pcai-media' $result.Success $result.Duration $result.Artifacts
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
$failedCount = @($results.Values | Where-Object { -not $_.Success }).Count
if ($RunTests -and -not $SkipTests -and $testResult -and -not $testResult.Success) {
    $failedCount += @($testResult.Failed).Count
}
exit $failedCount

#endregion
