#Requires -Version 7.0
<#+
.SYNOPSIS
  Rust build helper that routes through CargoTools.

.DESCRIPTION
  Standardizes Rust builds with CargoTools env setup, sccache, and optional
  lld-link configuration. Intended for repeatable, LLM-friendly builds.

.PARAMETER Path
  Working directory to run cargo from.

.PARAMETER UseLld
  Enable lld-link (CARGO_USE_LLD=1).

.PARAMETER NoLld
  Force link.exe (CARGO_USE_LLD=0).

.PARAMETER DisableCache
  Disable cache helpers (sccache/ccache) for this invocation.

.PARAMETER LlmDebug
  Enable CargoTools LLM debug defaults (RUST_BACKTRACE, verbose traces).

.PARAMETER RaPreflight
  Enable rust-analyzer diagnostics during CargoTools preflight.

.PARAMETER Preflight
  Enable CargoTools preflight checks (cargo check/clippy/fmt).

.PARAMETER PreflightMode
  Preflight mode: check|clippy|fmt|all.

.PARAMETER PreflightBlocking
  Fail build if preflight fails.

.EXAMPLE
  .\Tools\Invoke-RustBuild.ps1 -Path Deploy\rust-functiongemma-train test

.EXAMPLE
  .\Tools\Invoke-RustBuild.ps1 -Path Deploy\rust-functiongemma-train --use-lld build --release
#>
[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$Path = (Get-Location).Path,
    [switch]$UseLld,
    [switch]$NoLld,
    [switch]$DisableCache,
    [switch]$LlmDebug,
    [switch]$RaPreflight,
    [switch]$Preflight,
    [ValidateSet('check','clippy','fmt','all')]
    [string]$PreflightMode = 'check',
    [switch]$PreflightBlocking,
    [string[]]$CargoArgs,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RemainingArgs
)

$ErrorActionPreference = 'Stop'

$pcaiModuleBootstrap = Join-Path $PSScriptRoot 'PcaiModuleBootstrap.ps1'
if (Test-Path -LiteralPath $pcaiModuleBootstrap) {
    . $pcaiModuleBootstrap
}

$script:UseCargoTools = $true
$cargoToolsManifest = if (Get-Command Resolve-PcaiModuleManifestPath -ErrorAction SilentlyContinue) {
    Resolve-PcaiModuleManifestPath -ModuleName 'CargoTools' -RepoRoot (Split-Path -Parent $PSScriptRoot)
} else {
    $null
}

if (-not $cargoToolsManifest) {
    Write-Warning 'CargoTools module not found. Falling back to direct cargo invocation.'
    $script:UseCargoTools = $false
} elseif (-not (Get-Module CargoTools)) {
    Import-Module -Name $cargoToolsManifest -ErrorAction Stop | Out-Null
}

$versionScript = Join-Path $PSScriptRoot 'Get-BuildVersion.ps1'
if (Test-Path -LiteralPath $versionScript) {
    try {
        & $versionScript -SetEnv -Quiet | Out-Null
    } catch {
        Write-Verbose "Build version initialization failed: $($_.Exception.Message)"
    }
}

if ($script:UseCargoTools -and -not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
    Write-Verbose 'MSVC cl.exe not detected in this shell; bypassing CargoTools route and using cargo.exe directly.'
    $script:UseCargoTools = $false
}

# Ensure LLVM lld-link path is configured for optional use
$llvmBin = 'C:\Program Files\LLVM\bin'
$lldPath = Join-Path $llvmBin 'lld-link.exe'
if (Test-Path $lldPath) {
    if (-not $env:CARGO_LLD_PATH) { $env:CARGO_LLD_PATH = $lldPath }
    if ($env:PATH -notlike "*${llvmBin}*") {
        $env:PATH = "$llvmBin;$env:PATH"
    }
}
$libclangDll = Join-Path $llvmBin 'libclang.dll'
if (Test-Path $libclangDll) {
    $env:LIBCLANG_PATH = $llvmBin
}

# Let CargoTools choose the best linker by default. Only override when the
# caller explicitly requests lld-link or disables it for a problematic project.
if ($UseLld) { $env:CARGO_USE_LLD = '1' }
if ($NoLld) { $env:CARGO_USE_LLD = '0' }
if ($env:CARGO_USE_LLD) {
    $env:PCAI_USE_LLD = $env:CARGO_USE_LLD
}

# Configure CUDA environment for candle-core/cudarc builds
$cudaHelper = Join-Path $PSScriptRoot 'Initialize-CudaEnvironment.ps1'
if (Test-Path $cudaHelper) {
    . $cudaHelper
    $cudaInfo = Initialize-CudaEnvironment -Quiet
    if ($cudaInfo.Found) {
        Write-Verbose "CUDA environment configured: $($cudaInfo.CudaPath)"
    }
} else {
    Write-Verbose "CUDA helper not found at $cudaHelper"
}

# Normalize CMake environment for crates that use cmake/cc build scripts
$cmakeHelper = Join-Path $PSScriptRoot 'Initialize-CmakeEnvironment.ps1'
if (Test-Path $cmakeHelper) {
    . $cmakeHelper
    $cmakeInfo = Initialize-CmakeEnvironment -Quiet
    if ($cmakeInfo.Found -and $cmakeInfo.CmakeRoot) {
        Write-Verbose "CMake environment configured: $($cmakeInfo.CmakeRoot)"
    }
} else {
    Write-Verbose "CMake helper not found at $cmakeHelper"
}

# Configure cache helpers (sccache/ccache) for Rust + C/C++ builds
$cacheHelper = Join-Path $PSScriptRoot 'Initialize-CacheEnvironment.ps1'
if (Test-Path $cacheHelper) {
    . $cacheHelper
    $cacheInfo = Initialize-CacheEnvironment -DisableCache:$DisableCache -Quiet
    if ($cacheInfo.SccacheEnabled) { Write-Verbose 'sccache enabled' }
    if ($cacheInfo.CcacheEnabled) { Write-Verbose 'ccache enabled' }
} else {
    Write-Verbose "Cache helper not found at $cacheHelper"
}

# Default: do not use lld unless explicitly requested
if ($Preflight) {
    $env:CARGO_PREFLIGHT = '1'
    $env:CARGO_PREFLIGHT_MODE = $PreflightMode
} else {
    $env:CARGO_PREFLIGHT = '0'
}

# rust-analyzer preflight remains caller-driven; otherwise defer to CargoTools
# or environment defaults so CI/agent policies do not get overridden here.
if ($RaPreflight) {
    $env:CARGO_RA_PREFLIGHT = '1'
}

if ($PreflightBlocking) {
    $env:CARGO_PREFLIGHT_BLOCKING = '1'
}

$wrapperArgs = @()
if ($LlmDebug) { $wrapperArgs += '--llm-debug' }
if ($UseLld) { $wrapperArgs += '--use-lld' }
if ($NoLld) { $wrapperArgs += '--no-lld' }

$finalArgs = if ($CargoArgs -and $CargoArgs.Count -gt 0) { $CargoArgs } else { $RemainingArgs }
if ($finalArgs -is [string]) { $finalArgs = @($finalArgs) }
$wrapperArgs += @($finalArgs)

Push-Location $Path
try {
    if ($script:UseCargoTools) {
        Invoke-CargoWrapper @wrapperArgs
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    } else {
        $cargoCommand = Get-Command cargo -ErrorAction SilentlyContinue
        if (-not $cargoCommand) {
            throw 'cargo not found on PATH. Install Rust and ensure cargo is available.'
        }
        if (-not $finalArgs -or $finalArgs.Count -eq 0) {
            throw 'No cargo arguments provided.'
        }

        $cargoPath = $cargoCommand.Path
        # In some environments cargo is wrapped by a PowerShell script that may
        # enforce extra preflight checks. Prefer the rustup cargo.exe directly.
        if ($cargoPath -and $cargoPath.EndsWith('.ps1', [System.StringComparison]::OrdinalIgnoreCase)) {
            $candidateCargoExe = Join-Path $HOME '.cargo\bin\cargo.exe'
            if (Test-Path -LiteralPath $candidateCargoExe) {
                $cargoPath = $candidateCargoExe
            }
        }

        & $cargoPath @finalArgs
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
} finally {
    Pop-Location
}
