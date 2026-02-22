#Requires -Version 5.1
<#
.SYNOPSIS
    Consolidated build script for pcai-inference with dual-backend support.

.DESCRIPTION
    Orchestrates building pcai-inference with llamacpp and/or mistralrs backends.
    Integrates sccache, ccache, MSVC, and CUDA environment initialization.

.PARAMETER Backend
    Which backend(s) to build: llamacpp, mistralrs, or all (default)

.PARAMETER Configuration
    Build configuration: Debug or Release (default)

.PARAMETER EnableCuda
    Enable CUDA support for GPU acceleration

.PARAMETER DisableCache
    Disable sccache/ccache configuration for this build

.PARAMETER Clean
    Clean build directories before building
#>

[CmdletBinding()]
param(
    [ValidateSet('llamacpp', 'mistralrs', 'all')]
    [string]$Backend = 'all',

    [ValidateSet('Debug', 'Release')]
    [string]$Configuration = 'Release',

    [switch]$EnableCuda,
    [switch]$DisableCache,
    [switch]$Clean
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = $PSScriptRoot
$RepoRoot = Split-Path -Parent $ProjectRoot
$ToolsDir = Join-Path $RepoRoot 'Tools'
if (-not (Test-Path $ToolsDir)) {
    $RepoRoot = Split-Path -Parent $RepoRoot
    $ToolsDir = Join-Path $RepoRoot 'Tools'
}
if (-not (Test-Path $ToolsDir)) {
    $RepoRoot = Split-Path -Parent $RepoRoot
    $ToolsDir = Join-Path $RepoRoot 'Tools'
}

# Build output directories
$script:ArtifactsRoot = if ($env:PCAI_ARTIFACTS_ROOT) {
    $env:PCAI_ARTIFACTS_ROOT
} else {
    Join-Path $RepoRoot '.pcai'
}
$script:BuildRoot = Join-Path $script:ArtifactsRoot 'build'
$script:BuildLogsDir = Join-Path $script:BuildRoot 'logs'
$script:BuildArtifactsDir = Join-Path $script:BuildRoot 'artifacts'
$script:BuildStartTime = Get-Date

#region Output Formatting

function Write-BuildBanner {
    param([string]$Title)
    $width = 70
    $line = '=' * $width
    $padding = [math]::Max(0, ($width - $Title.Length - 2) / 2)
    $paddedTitle = (' ' * [math]::Floor($padding)) + $Title + (' ' * [math]::Ceiling($padding))
    Write-Host "`n$line" -ForegroundColor Cyan
    Write-Host $paddedTitle -ForegroundColor Cyan
    Write-Host "$line`n" -ForegroundColor Cyan
}

function Write-BuildSection {
    param([string]$Name)
    $elapsed = (Get-Date) - $script:BuildStartTime
    Write-Host "`n[$($elapsed.ToString('mm\:ss'))] " -ForegroundColor DarkGray -NoNewline
    Write-Host $Name -ForegroundColor Yellow
}

function Write-BuildInfo {
    param([string]$Label, [string]$Value, [string]$Color = 'White')
    Write-Host "  $($Label): " -ForegroundColor DarkGray -NoNewline
    Write-Host $Value -ForegroundColor $Color
}

function Write-BuildStatus {
    param([string]$Message, [ValidateSet('Info', 'Success', 'Warning', 'Error')]$Level = 'Info')
    $symbol = switch ($Level) {
        'Info' { '[..]' }
        'Success' { '[OK]' }
        'Warning' { '[!!]' }
        'Error' { '[XX]' }
    }
    $color = switch ($Level) {
        'Info' { 'White' }
        'Success' { 'Green' }
        'Warning' { 'Yellow' }
        'Error' { 'Red' }
    }
    Write-Host "  $symbol " -ForegroundColor $color -NoNewline
    Write-Host $Message
}

#endregion

$script:HasMkl = $false
$script:HasCudnn = $false
$script:HasTensorRt = $false
$script:MsVcRuntime = $null

function Add-EnvPath {
    param([string]$Value)
    if (-not $Value) { return $false }
    if (-not (Test-Path $Value)) { return $false }
    if ($env:PATH -notlike "*$Value*") {
        $env:PATH = "$Value;$env:PATH"
        return $true
    }
    return $false
}

function Add-EnvList {
    param(
        [string]$Name,
        [string]$Value
    )
    if (-not $Name -or -not $Value) { return $false }
    if (-not (Test-Path $Value)) { return $false }
    $item = Get-Item -Path "Env:$Name" -ErrorAction SilentlyContinue
    $current = if ($item) { $item.Value } else { $null }
    if ($current -notlike "*$Value*") {
        if ($current) {
            Set-Item -Path "Env:$Name" -Value "$Value;$current"
        } else {
            Set-Item -Path "Env:$Name" -Value "$Value"
        }
        return $true
    }
    return $false
}

function Initialize-OneApiEnvironment {
    $script:HasMkl = $false

    $oneApiRoot = $env:ONEAPI_ROOT
    if (-not $oneApiRoot) { $oneApiRoot = 'C:\Program Files (x86)\Intel\oneAPI' }
    if (-not (Test-Path $oneApiRoot)) { return }

    if (-not $env:ONEAPI_ROOT) { $env:ONEAPI_ROOT = $oneApiRoot }

    $mklRoot = $env:MKLROOT
    if (-not $mklRoot) {
        $candidate = Join-Path $oneApiRoot 'mkl\latest'
        if (Test-Path $candidate) { $mklRoot = $candidate }
    }

    if ($mklRoot -and (Test-Path $mklRoot)) {
        $env:MKLROOT = $mklRoot
        Add-EnvPath (Join-Path $mklRoot 'bin')
        Add-EnvPath (Join-Path $mklRoot 'redist\intel64')
        Add-EnvList -Name 'LIB' -Value (Join-Path $mklRoot 'lib\intel64')
        Add-EnvList -Name 'INCLUDE' -Value (Join-Path $mklRoot 'include')
        $script:HasMkl = $true
        Write-Host "  Intel oneAPI MKL detected: $mklRoot" -ForegroundColor Green
    }
}

function Initialize-CudnnEnvironment {
    param([string]$CudaPath)

    $script:HasCudnn = $false
    $candidates = @()
    if ($env:CUDNN_PATH) { $candidates += $env:CUDNN_PATH }
    if ($env:CUDNN_HOME) { $candidates += $env:CUDNN_HOME }
    if ($CudaPath) { $candidates += $CudaPath }
    $candidates += 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA'

    $resolved = $null
    foreach ($candidate in $candidates | Select-Object -Unique) {
        if (-not $candidate) { continue }
        if (-not (Test-Path $candidate)) { continue }

        $bin = Join-Path $candidate 'bin'
        $lib = Join-Path $candidate 'lib\x64'
        $include = Join-Path $candidate 'include'
        $dll = Get-ChildItem -Path $bin -Filter 'cudnn64*.dll' -ErrorAction SilentlyContinue | Select-Object -First 1
        $libFile = Get-ChildItem -Path $lib -Filter 'cudnn*.lib' -ErrorAction SilentlyContinue | Select-Object -First 1

        if ($dll -and $libFile) {
            $resolved = $candidate
            break
        }
    }

    if ($resolved) {
        $env:CUDNN_PATH = $resolved
        $env:CUDNN_HOME = $resolved
        Add-EnvPath (Join-Path $resolved 'bin')
        Add-EnvList -Name 'LIB' -Value (Join-Path $resolved 'lib\x64')
        Add-EnvList -Name 'INCLUDE' -Value (Join-Path $resolved 'include')
        $script:HasCudnn = $true
        Write-Host "  cuDNN detected: $resolved" -ForegroundColor Green
    }
}

function Initialize-TensorRtEnvironment {
    $script:HasTensorRt = $false
    $candidates = @()
    if ($env:TENSORRT_PATH) { $candidates += $env:TENSORRT_PATH }
    if ($env:TENSORRT_HOME) { $candidates += $env:TENSORRT_HOME }
    $candidates += 'C:\Program Files\NVIDIA GPU Computing Toolkit\TensorRT'

    $resolved = $null
    foreach ($candidate in $candidates | Select-Object -Unique) {
        if (-not $candidate) { continue }
        if (-not (Test-Path $candidate)) { continue }

        $bin = Join-Path $candidate 'bin'
        $lib = Join-Path $candidate 'lib'
        $dll = Get-ChildItem -Path $bin -Filter 'nvinfer*.dll' -ErrorAction SilentlyContinue | Select-Object -First 1
        $libFile = Get-ChildItem -Path $lib -Filter 'nvinfer*.lib' -ErrorAction SilentlyContinue | Select-Object -First 1

        if ($dll -and $libFile) {
            $resolved = $candidate
            break
        }
    }

    if ($resolved) {
        $env:TENSORRT_PATH = $resolved
        $env:TENSORRT_HOME = $resolved
        Add-EnvPath (Join-Path $resolved 'bin')
        Add-EnvList -Name 'LIB' -Value (Join-Path $resolved 'lib')
        Add-EnvList -Name 'INCLUDE' -Value (Join-Path $resolved 'include')
        $script:HasTensorRt = $true
        Write-Host "  TensorRT detected: $resolved" -ForegroundColor Green
    }
}

#region Environment Initialization

function Initialize-BuildEnvironment {
    Write-BuildSection 'Environment Setup'

    # Detect real cargo (bypass wrappers like cargo.ps1)
    $script:cargoExe = Join-Path $env:USERPROFILE '.cargo\bin\cargo.exe'
    if (-not (Test-Path $script:cargoExe)) {
        $realCargo = (Get-Command cargo.exe -ErrorAction SilentlyContinue | Where-Object { $_.Source -notlike '*local\bin*' } | Select-Object -First 1)
        if ($realCargo) { $script:cargoExe = $realCargo.Source }
    }
    if (-not (Test-Path $script:cargoExe)) { $script:cargoExe = 'cargo.exe' } # Last resort fallback
    Write-Host "  Cargo identified: $script:cargoExe" -ForegroundColor Gray

    # [CUDA Logic moved to end of function to avoid being cleared by MSVC init]

    # 2. Locate stable CMake
    Write-Host '  Locating stable CMake (VS 3.x)...' -ForegroundColor Cyan
    $vsCmake = 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe'
    if (-not (Test-Path $vsCmake)) {
        $vsCmake = (Get-Command cmake -ErrorAction SilentlyContinue).Source
    }

    if ($vsCmake) {
        # CRITICAL: Clean up PATH of ANY other CMake-related entries
        $cmakeBin = Split-Path $vsCmake -Parent
        $cleanPath = ($env:PATH -split ';' | Where-Object { $_ -notlike '*CMake*' -and $_ -ne '' }) -join ';'
        $env:PATH = "$cmakeBin;$cleanPath"

        $env:CMAKE = $vsCmake
        $env:CMAKE_COMMAND = $vsCmake
        $env:CMAKE_PROGRAM = $vsCmake
        $env:CMAKE_EXECUTABLE = $vsCmake

        # CRITICAL: Do NOT set CMAKE_ROOT manually unless you are 100% sure of the versioned module path.
        # Most modern CMake installations handle this automatically.
        if (Test-Path Env:CMAKE_ROOT) { Remove-Item Env:CMAKE_ROOT }

        Write-Host "  CMake prioritized: $vsCmake" -ForegroundColor Green
        & $vsCmake --version | Select-Object -First 1
    } else {
        throw 'Stable CMake not found'
    }

    # Clear variables that might still pollute CMake
    @('CMAKE_MODULE_PATH', 'CMAKE_TOOLCHAIN_FILE', 'CMAKE_PREFIX_PATH', 'CMAKE_ROOT', 'CMAKE_ROOT_PowerToys_code', 'CMAKE_PREFIX_PATH_PowerToys_code', 'CMAKE_PROGRAM_PowerToys_code') | ForEach-Object {
        if (Get-Item "Env:$_" -ErrorAction SilentlyContinue) {
            Write-Host "    Clearing $_" -ForegroundColor Yellow
            Set-Item "Env:$_" -Value ''
        }
    }

    # 3. Initialize MSVC and SDK (Clear polluted env vars, set cl.exe and rc.exe)
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vswhere) {
        $vsPath = & $vswhere -latest -property installationPath
        $devShell = Join-Path $vsPath 'Common7\Tools\Launch-VsDevShell.ps1'
        if (Test-Path $devShell) {
            Write-Host '  Initializing MSVC via Launch-VsDevShell.ps1...' -ForegroundColor Cyan
            & $devShell -SkipAutomaticLocation -HostArch amd64 -Arch amd64

            # CRITICAL: Clear polluted environment variables (Only ones that actually cause issues)
            @('LINK', '_LINK_') | ForEach-Object {
                if (Get-Item "Env:$_" -ErrorAction SilentlyContinue) {
                    Write-Host "    Clearing polluted env var: $_" -ForegroundColor Yellow
                    Remove-Item "Env:$_" -ErrorAction SilentlyContinue
                }
            }

            # Set CC/CXX for crates using cc-rs or cmake
            $clPath = (Get-Command cl.exe -ErrorAction SilentlyContinue).Source -replace '\\', '/'
            if ($clPath) {
                $env:CC = $clPath
                $env:CXX = $clPath
                $env:CMAKE_C_COMPILER = $clPath
                $env:CMAKE_CXX_COMPILER = $clPath
                $clDir = Split-Path $clPath -Parent
                if ($env:PATH -notlike "*$clDir*") {
                    $env:PATH = "$clDir;$env:PATH"
                }
                Write-Host "    cl.exe identified: $clPath" -ForegroundColor Green
            }

            # Find Windows SDK rc.exe (required by CMake on Windows)
            $rcPath = Get-ChildItem 'C:\Program Files (x86)\Windows Kits\10\bin' -Recurse -Filter 'rc.exe' -ErrorAction SilentlyContinue |
                Where-Object { $_.FullName -like '*x64*' } |
                Sort-Object { [version]($_.FullName -replace '^.*\\(\d+\.\d+\.\d+\.\d+)\\.*$', '$1') } -Descending |
                Select-Object -First 1 -ExpandProperty FullName
            if ($rcPath) {
                $rcPath = $rcPath -replace '\\', '/'
                $env:CMAKE_RC_COMPILER = $rcPath
                $rcDir = Split-Path $rcPath -Parent
                $env:PATH = "$rcDir;$env:PATH"
                Write-Host "    rc.exe identified: $rcPath" -ForegroundColor Green
            }

            # vcpkg toolchain
            $vcpkgToolchain = Join-Path $vsPath 'VC\vcpkg\scripts\buildsystems\vcpkg.cmake'
            if (Test-Path $vcpkgToolchain) {
                $env:CMAKE_TOOLCHAIN_FILE = $vcpkgToolchain
                Write-Host '    vcpkg toolchain enabled' -ForegroundColor Green
            }
        }
    }

    # Align runtime across MSVC/CMake/CUDA (Debug -> /MDd, Release -> /MD)
    if ($Configuration -eq 'Debug') {
        $script:MsVcRuntime = 'MultiThreadedDebugDLL'
    } else {
        $script:MsVcRuntime = 'MultiThreadedDLL'
    }

    # Intel oneAPI/MKL environment (CPU acceleration)
    Initialize-OneApiEnvironment

    # 4. Configure Caching and Generator
    if (Get-Command ninja -ErrorAction SilentlyContinue) {
        $env:CMAKE_GENERATOR = 'Ninja'
        $env:CMAKE_MAKE_PROGRAM = (Get-Command ninja.exe).Source
        Write-Host "  Generator: Ninja enabled ($($env:CMAKE_MAKE_PROGRAM))" -ForegroundColor Green
    }

    if (-not $env:PCAI_USE_LLD) {
        $lldPath = $env:CARGO_LLD_PATH
        if (-not $lldPath) { $lldPath = 'C:\Program Files\LLVM\bin\lld-link.exe' }
        if (Test-Path $lldPath) { $env:PCAI_USE_LLD = '1' }
    }
    $useLld = ($env:PCAI_USE_LLD -eq '1') -or ($env:CARGO_USE_LLD -eq '1')
    if ($useLld) {
        $lldPath = $env:CARGO_LLD_PATH
        if (-not $lldPath) {
            $lldPath = 'C:\Program Files\LLVM\bin\lld-link.exe'
        }
        if (Test-Path $lldPath) {
            $env:CMAKE_LINKER = $lldPath -replace '\\', '/'
            Write-Host "  Linker: lld-link enabled ($($env:CMAKE_LINKER))" -ForegroundColor Green
        } else {
            Write-Host '  Linker: lld-link requested but not found' -ForegroundColor Yellow
        }
    }

    if ($EnableCuda) {
        if ($env:PCAI_DISABLE_CUDA_LAUNCHER -eq '1') {
            Write-Host '  CUDA cache launcher disabled via PCAI_DISABLE_CUDA_LAUNCHER=1' -ForegroundColor Yellow
        } else {
            Write-Host '  CUDA cache launcher enabled for NVCC (set PCAI_DISABLE_CUDA_LAUNCHER=1 to override)' -ForegroundColor Green
        }
    }

    $cacheHelper = Join-Path $ToolsDir 'Initialize-CacheEnvironment.ps1'
    if (Test-Path $cacheHelper) {
        . $cacheHelper
        $cacheInfo = Initialize-CacheEnvironment -DisableCache:$DisableCache -Quiet

        if ($DisableCache) {
            Write-Host '  Caching: explicitly disabled via parameter' -ForegroundColor Yellow
        } elseif ($cacheInfo.SccacheEnabled -or $cacheInfo.CcacheEnabled) {
            $cacheModes = @()
            if ($cacheInfo.SccacheEnabled) { $cacheModes += 'sccache' }
            if ($cacheInfo.CcacheEnabled) { $cacheModes += 'ccache' }
            Write-Host "  Caching: $($cacheModes -join ', ') enabled" -ForegroundColor Green
        } else {
            Write-Host '  Caching: no cache tool detected' -ForegroundColor Gray
        }

        if (-not $DisableCache -and $cacheInfo.SccacheEnabled -and (Get-Command sccache -ErrorAction SilentlyContinue)) {
            sccache --start-server
        }
    } elseif ($DisableCache) {
        Write-Host '  Caching: explicitly disabled via parameter' -ForegroundColor Yellow
        $env:RUSTC_WRAPPER = ''
        $env:CMAKE_C_COMPILER_LAUNCHER = ''
        $env:CMAKE_CXX_COMPILER_LAUNCHER = ''
        $env:CMAKE_CUDA_COMPILER_LAUNCHER = ''
    } elseif (Get-Command sccache -ErrorAction SilentlyContinue) {
        $env:RUSTC_WRAPPER = 'sccache'
        $env:CMAKE_C_COMPILER_LAUNCHER = 'sccache'
        $env:CMAKE_CXX_COMPILER_LAUNCHER = 'sccache'
        $env:CMAKE_CUDA_COMPILER_LAUNCHER = 'sccache'
        $env:SCCACHE_SERVER_PORT = '4226'
        $env:SCCACHE_CACHE_COMPRESSION = 'zstd'
        $env:SCCACHE_DIRECT = 'true'
        Write-Host '  Caching: sccache enabled' -ForegroundColor Green
        sccache --start-server
    } else {
        Write-Host '  Caching: sccache not found in PATH' -ForegroundColor Gray
    }

    # llamacpp specific optimizations
    $env:LLAMA_NO_OPENMP = '1'
    $env:LLAMA_CURL = 'OFF'
    $env:LLAMA_BUILD_TESTS = 'OFF'
    $env:LLAMA_BUILD_EXAMPLES = 'OFF'
    $env:LLAMA_BUILD_TOOLS = 'OFF'

    # 5. Initialize CUDA and CRT OVERRIDES (Must be last to survive sanitization)
    Write-Host "  DEBUG: Step 5 EnableCuda check (Value: $EnableCuda, Script: $($script:EnableCuda))" -ForegroundColor Gray
    if ($EnableCuda -or $script:EnableCuda) {
        $cudaHelper = Join-Path $ToolsDir 'Initialize-CudaEnvironment.ps1'
        if (Test-Path $cudaHelper) {
            . $cudaHelper
            $cudaInfo = Initialize-CudaEnvironment -Quiet
            if ($cudaInfo.Found) {
                Write-Host "  CUDA initialized: $($cudaInfo.CudaPath)" -ForegroundColor Green
                $env:LLAMA_CUDA = '1'
                $env:GGML_CUDA = 'ON'

                # CRT Mismatch Resolution: Force /MD (Dynamic Release) globally
                # This fixes the mismatch between llama-cpp (/MD) and mistralrs-cuda (often /MT)
                $env:RUSTFLAGS = '-Ctarget-feature=-crt-static'
                $crtFlag = if ($Configuration -eq 'Debug') { '/MDd' } else { '/MD' }
                $env:CUDA_NVCC_FLAGS = $crtFlag
                $env:NVCCFLAGS = "--compiler-options $crtFlag"
                $runtimeArg = "-DCMAKE_MSVC_RUNTIME_LIBRARY=$script:MsVcRuntime"
                if ($env:CMAKE_ARGS) {
                    if ($env:CMAKE_ARGS -notmatch 'CMAKE_MSVC_RUNTIME_LIBRARY') {
                        $env:CMAKE_ARGS = "$($env:CMAKE_ARGS) $runtimeArg"
                    }
                } else {
                    $env:CMAKE_ARGS = $runtimeArg
                }

                # FORCE MSVC to use dynamic runtime via internal compiler vars
                $env:CL = $crtFlag
                $env:_CL_ = $crtFlag

                # Help bindgen find CUDA
                $env:LIB += ";$($cudaInfo.CudaPath)\lib\x64"
                $env:INCLUDE += ";$($cudaInfo.CudaPath)\include"

                # Explicitly set CUDA compiler for CMake
                $nvccPath = (Join-Path $cudaInfo.CudaPath 'bin\nvcc.exe') -replace '\\', '/'
                if (Test-Path $nvccPath) {
                    $env:CMAKE_CUDA_COMPILER = $nvccPath
                    Write-Host "    nvcc identified (forcing /MD): $nvccPath" -ForegroundColor Green

                    # Set NVCC_CCBIN for candle/mistralrs
                    $clPath = (Get-Command cl.exe -ErrorAction SilentlyContinue).Source -replace '\\', '/'
                    if ($clPath) {
                        $env:NVCC_CCBIN = $clPath
                    }
                }

                # Optional CUDA libraries (cuDNN/TensorRT)
                Initialize-CudnnEnvironment -CudaPath $cudaInfo.CudaPath
                Initialize-TensorRtEnvironment
            } else {
                Write-Host '  CUDA not found, building CPU-only' -ForegroundColor Yellow
                $script:EnableCuda = $false
            }
        }
    }
}

function Initialize-LlamaCpp {
    param([string]$Configuration)

    Write-Host "`n=== Pre-configuring llama-cpp-sys-2 CMake ===" -ForegroundColor Cyan
    $targetDir = if ($env:CARGO_TARGET_DIR) { $env:CARGO_TARGET_DIR } else { Join-Path $ProjectRoot 'target' }
    $profile = if ($Configuration -eq 'Release') { 'release' } else { 'debug' }
    $buildBase = Join-Path $targetDir "$profile\build"

    if (-not (Test-Path $buildBase)) {
        Write-Host "  Build base not found: $buildBase" -ForegroundColor Yellow
        return
    }

    # Find the llama-cpp-sys-2 directory (hashes change)
    $llamaBuildDirs = Get-ChildItem $buildBase -Filter 'llama-cpp-sys-2-*' -Directory -ErrorAction SilentlyContinue
    if (-not $llamaBuildDirs) {
        Write-Host '  llama-cpp-sys-2 build dir not found yet.' -ForegroundColor Yellow
        return
    }

    # Find the llama-cpp-sys-2 directory reliably using cargo metadata
    Write-Host '  Locating llama-cpp-sys-2 source...' -ForegroundColor Cyan
    try {
        $meta = cargo metadata --format-version 1 | ConvertFrom-Json
        $pkg = $meta.packages | Where-Object { $_.name -eq 'llama-cpp-sys-2' }
        if (-not $pkg) {
            Write-Host '    llama-cpp-sys-2 not found in metadata.' -ForegroundColor Yellow
            return
        }

        $llamaSrcDir = (Join-Path (Split-Path $pkg.manifest_path -Parent) 'llama.cpp') -replace '\\', '/'
        Write-Host "    Source identified: $llamaSrcDir" -ForegroundColor Green
    } catch {
        Write-Host "    Failed to get metadata: $($_.Exception.Message)" -ForegroundColor Red
        return
    }

    foreach ($dir in $llamaBuildDirs) {
        $outDir = Join-Path $dir.FullName 'out'
        $cmakeBuildDir = Join-Path $outDir 'build'

        if (-not (Test-Path $llamaSrcDir)) {
            Write-Host "    Source not found at $llamaSrcDir" -ForegroundColor Red
            continue
        }

        if (-not (Test-Path $cmakeBuildDir)) { New-Item -ItemType Directory -Path $cmakeBuildDir -Force | Out-Null }

        $clPath = (Get-Command cl.exe -ErrorAction SilentlyContinue).Source
        $ninjaPath = (Get-Command ninja.exe -ErrorAction SilentlyContinue).Source
        $rcPath = $env:CMAKE_RC_COMPILER

        $cmakeArgs = @(
            '-S', $llamaSrcDir,
            '-B', $cmakeBuildDir,
            '-G', 'Ninja',
            "-DCMAKE_MAKE_PROGRAM=$ninjaPath",
            "-DCMAKE_C_COMPILER=$clPath",
            "-DCMAKE_CXX_COMPILER=$clPath",
            "-DCMAKE_RC_COMPILER=$rcPath",
            "-DCMAKE_BUILD_TYPE=$Configuration",
            "-DCMAKE_MSVC_RUNTIME_LIBRARY=$script:MsVcRuntime",
            "-DCMAKE_INSTALL_PREFIX=$outDir",
            '-DGGML_CUDA=ON',
            '-DLLAMA_CURL=OFF',
            '-DLLAMA_BUILD_TESTS=OFF',
            '-DLLAMA_BUILD_EXAMPLES=OFF',
            '-DLLAMA_BUILD_TOOLS=OFF',
            '-DBUILD_SHARED_LIBS=ON'
        )

        if ($EnableCuda) {
            $cudaPath = $env:CUDA_PATH -replace '\\', '/'
            $nvccPath = "$cudaPath/bin/nvcc.exe"
            if (Test-Path $nvccPath) {
                $cmakeArgs += "-DCMAKE_CUDA_COMPILER=$nvccPath"
                if ($env:CUDAARCHS) {
                    $cmakeArgs += "-DCMAKE_CUDA_ARCHITECTURES=$($env:CUDAARCHS)"
                }
            }
        }

        & $env:CMAKE @cmakeArgs
        if ($LASTEXITCODE -ne 0) {
            Write-Host "  CMake pre-config failed for $dir" -ForegroundColor Red
        } else {
            Write-Host "  CMake pre-config SUCCESS for $dir" -ForegroundColor Green
        }
    }
}

function Clear-IncompleteCmakeConfig {
    Write-Host 'Checking for incomplete CMake configurations (Nuclear Evasion)...' -ForegroundColor DarkCyan

    # 1. Use cargo metadata to find registry paths reliably
    Write-Host '  Locating build artifacts folder...' -ForegroundColor Gray
    try {
        $metadata = & $script:cargoExe metadata --format-version 1 --no-deps | ConvertFrom-Json
        $targetDir = $metadata.target_directory
    } catch {
        $targetDir = if ($env:CARGO_TARGET_DIR) { $env:CARGO_TARGET_DIR } else { Join-Path $ProjectRoot 'target' }
    }

    if (-not (Test-Path $targetDir)) { return }

    # Targeted cleaning for troublesome crates (ALWAYS wipe when changing CRT/CUDA flags)
    Write-Host '  Aggressive cleaning for candle and mistralrs (Due to CRT changes)...' -ForegroundColor Cyan
    $badCrates = @('mistralrs-core-*', 'candle-core-*', 'candle-kernels-*', 'llama-cpp-sys-2-*')
    foreach ($pattern in $badCrates) {
        Get-ChildItem $targetDir -Recurse -Filter $pattern -Directory -ErrorAction SilentlyContinue | ForEach-Object {
            Write-Host "    Wiping stale build artifacts: $($_.Name)" -ForegroundColor Yellow
            Remove-Item $_.FullName -Recurse -Force -ErrorAction SilentlyContinue
        }
    }

    Get-ChildItem $targetDir -Recurse -Filter 'CMakeCache.txt' -ErrorAction SilentlyContinue | ForEach-Object {
        $dir = $_.DirectoryName
        $ninjaFile = Join-Path $dir 'build.ninja'
        if (-not (Test-Path $ninjaFile)) {
            Write-Host "    Cleaning incomplete config in: $dir" -ForegroundColor Yellow
            Remove-Item (Split-Path $dir -Parent) -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
}

#endregion

#region Build Logic

function Invoke-Build {
    param([string]$BackendName)

    if ($BackendName -eq 'all') {
        Invoke-Build -BackendName 'llamacpp'
        Invoke-Build -BackendName 'mistralrs'
        return
    }

    $binName = if ($BackendName -eq 'llamacpp') { 'pcai-llamacpp' } else { 'pcai-mistralrs' }
    Write-Host "`n=== Building Binary: $binName ($Configuration) ===" -ForegroundColor Magenta

    $features = @('ffi', 'server')
    if ($BackendName -eq 'llamacpp') {
        $features += 'llamacpp'
    } else {
        $features += 'mistralrs-backend'
    }

    $useMkl = $script:HasMkl -and ($env:PCAI_DISABLE_MKL -ne '1')
    $useCudnn = $script:HasCudnn -and ($env:PCAI_DISABLE_CUDNN -ne '1')

    if ($EnableCuda) {
        if ($BackendName -eq 'llamacpp') {
            $features += 'cuda-llamacpp'
        } else {
            $features += 'cuda-mistralrs'
            if ($useCudnn) { $features += 'mistralrs/cudnn' }
            if ($env:PCAI_ENABLE_FLASH_ATTN -eq '1') { $features += 'mistralrs/flash-attn' }
            if ($env:PCAI_ENABLE_FLASH_ATTN_V3 -eq '1') { $features += 'mistralrs-core/flash-attn-v3' }
        }
    } elseif ($BackendName -eq 'mistralrs') {
        if ($useMkl) { $features += 'mistralrs/mkl' }
    }

    $featureString = ($features | Select-Object -Unique) -join ','
    Write-Host "  Features: $featureString" -ForegroundColor Cyan

    $cargoArgs = @('build', '--bin', $binName, '--features', $featureString, '--message-format=json')
    if ($Configuration -eq 'Release') { $cargoArgs += '--release' }

    # Configure log files in .pcai/build/logs directory
    $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    $logDir = if (Test-Path $script:BuildLogsDir) { $script:BuildLogsDir } else { $ProjectRoot }
    $logFile = Join-Path $logDir "cargo_build_${BackendName}_$stamp.log"
    $errorLogFile = Join-Path $logDir "cargo_errors_${BackendName}_$stamp.log"
    $jsonLogFile = Join-Path $logDir "cargo_output_${BackendName}_$stamp.json"

    # Initialize logs
    '' | Out-File $logFile -Force
    '' | Out-File $errorLogFile -Force
    '' | Out-File $jsonLogFile -Force
    Write-BuildStatus "Logs: $logDir" 'Info'

    Push-Location $ProjectRoot
    $startTime = Get-Date
    $script:BuildStartTime = $startTime
    $script:BuildLastLineTime = $startTime
    $script:BuildLastPkg = 'initializing'
    $heartbeatSeconds = 120
    if ($env:PCAI_BUILD_HEARTBEAT_SECONDS) {
        $parsed = 0
        if ([int]::TryParse($env:PCAI_BUILD_HEARTBEAT_SECONDS, [ref]$parsed)) {
            if ($parsed -gt 0) { $heartbeatSeconds = $parsed }
        }
    }
    $heartbeatTimer = $null
    $heartbeatEvent = $null
    if ($heartbeatSeconds -gt 0) {
        $heartbeatTimer = New-Object System.Timers.Timer
        $heartbeatTimer.Interval = $heartbeatSeconds * 1000
        $heartbeatTimer.AutoReset = $true
        $heartbeatEvent = Register-ObjectEvent -InputObject $heartbeatTimer -EventName Elapsed -Action {
            $now = Get-Date
            $elapsed = $now - $script:BuildStartTime
            $since = $now - $script:BuildLastLineTime
            $pkg = if ($script:BuildLastPkg) { $script:BuildLastPkg } else { 'unknown' }
            Write-Host ('  Build heartbeat: {0:mm\\:ss} elapsed, last package: {1}, last output {2:mm\\:ss} ago' -f $elapsed, $pkg, $since) -ForegroundColor DarkGray
        }
        $heartbeatTimer.Start()
    }

    if (Get-Command sccache -ErrorAction SilentlyContinue) {
        Write-Host '  Sccache status (Start):' -ForegroundColor Gray
        sccache --show-stats | Select-Object -First 3
    }

    # CRT Mismatch Resolution: Force /MD (Dynamic Release) globally
    if ($EnableCuda) {
        $env:RUSTFLAGS = '-Ctarget-feature=-crt-static'
        $crtFlag = if ($Configuration -eq 'Debug') { '/MDd' } else { '/MD' }
        $env:CUDA_NVCC_FLAGS = $crtFlag
        $env:NVCCFLAGS = "--compiler-options $crtFlag"
        $env:CL = $crtFlag
        $env:_CL_ = $crtFlag
        $runtimeArg = "-DCMAKE_MSVC_RUNTIME_LIBRARY=$script:MsVcRuntime"
        if ($env:CMAKE_ARGS) {
            if ($env:CMAKE_ARGS -notmatch 'CMAKE_MSVC_RUNTIME_LIBRARY') {
                $env:CMAKE_ARGS = "$($env:CMAKE_ARGS) $runtimeArg"
            }
        } else {
            $env:CMAKE_ARGS = $runtimeArg
        }
        Write-Host "  Forced CRT: $crtFlag enabled for CUDA build." -ForegroundColor Green
    }

    # Low-memory CUDA build mode (favor success over peak optimization)
    $lowMemCuda = ($env:PCAI_CUDA_FULL_OPT -ne '1') -and $EnableCuda -and ($BackendName -eq 'mistralrs')
    if ($lowMemCuda) {
        if (-not $env:CARGO_BUILD_JOBS) { $env:CARGO_BUILD_JOBS = '1' }
        if (-not $env:RAYON_NUM_THREADS) { $env:RAYON_NUM_THREADS = '1' }
        if ($env:RUSTFLAGS) {
            if ($env:RUSTFLAGS -notmatch 'opt-level') { $env:RUSTFLAGS = "$($env:RUSTFLAGS) -C opt-level=2" }
        } else {
            $env:RUSTFLAGS = '-C opt-level=2'
        }
        Write-Host '  CUDA low-mem mode: CARGO_BUILD_JOBS=1, RAYON_NUM_THREADS=1, opt-level=2' -ForegroundColor Yellow
    }

    $prevErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        Write-Host "  Executing: cargo $($cargoArgs -join ' ')" -ForegroundColor Gray

        $lastPkg = ''
        & $script:cargoExe @cargoArgs 2>&1 | ForEach-Object {
            $line = $_.ToString()
            $script:BuildLastLineTime = Get-Date
            if ($line -match '^{') {
                $line | Out-File $jsonLogFile -Append
                try {
                    $msg = $line | ConvertFrom-Json -ErrorAction SilentlyContinue
                    if ($msg -and $msg.reason -eq 'compiler-artifact') {
                        $pkg = $msg.package_id.Split(' ')[0]
                        if ($pkg -ne $lastPkg) {
                            Write-Progress -Activity "Building $binName" -Status "Compiling: $pkg"
                            $lastPkg = $pkg
                            $script:BuildLastPkg = $pkg
                        }
                    }
                } catch {}
            } else {
                if ($line -match 'Compiling ([^ ]+)') {
                    $script:BuildLastPkg = $Matches[1]
                }
                $line | Out-File $logFile -Append
                if ($line -match 'error(\[|:)' -or $line -match 'panic') {
                    $line | Out-File $errorLogFile -Append
                    Write-Host $line -ForegroundColor Red
                } elseif ($line -match 'warning:') {
                    # Write-Host $line -ForegroundColor Yellow
                } else {
                    Write-Host $line
                }
            }
        }

        if ($LASTEXITCODE -ne 0) {
            Write-Host "`nBuild FAILED for $binName with Exit Code: $LASTEXITCODE" -ForegroundColor Red
            throw "Build failed for $BackendName"
        }
    } finally {
        if ($heartbeatTimer) { $heartbeatTimer.Stop() }
        if ($heartbeatEvent) { Unregister-Event -SourceIdentifier $heartbeatEvent.SourceIdentifier -ErrorAction SilentlyContinue }
        if ($heartbeatTimer) { $heartbeatTimer.Dispose() }
        $ErrorActionPreference = $prevErrorActionPreference
        Pop-Location
        $duration = (Get-Date) - $startTime
        Write-Host ('  Duration: {0:n2}s' -f $duration.TotalSeconds) -ForegroundColor Gray
    }
}

#endregion

function Copy-CompiledArtifacts {
    param(
        [string]$Configuration = 'Release',
        [string]$BackendBuilt = 'all'
    )

    Write-BuildSection 'Collecting Artifacts'

    $targetRoot = if ($env:CARGO_TARGET_DIR) { $env:CARGO_TARGET_DIR } else { Join-Path $ProjectRoot 'target' }
    $configDir = if ($Configuration -eq 'Debug') { 'debug' } else { 'release' }
    $targetDir = Join-Path $targetRoot $configDir

    if (-not (Test-Path $targetDir)) {
        Write-BuildStatus "Target directory not found: $targetDir" 'Warning'
        return
    }

    # Ensure output directories exist
    $localBin = Join-Path $env:USERPROFILE '.local\bin'
    if (-not (Test-Path $localBin)) {
        New-Item -ItemType Directory -Path $localBin -Force | Out-Null
    }

    # Create Build/artifacts directories if they exist at repo level
    $llamacppArtifacts = Join-Path $script:BuildArtifactsDir 'pcai-llamacpp'
    $mistralrsArtifacts = Join-Path $script:BuildArtifactsDir 'pcai-mistralrs'

    if (Test-Path (Split-Path $script:BuildArtifactsDir -Parent)) {
        if (-not (Test-Path $llamacppArtifacts)) { New-Item -ItemType Directory -Path $llamacppArtifacts -Force | Out-Null }
        if (-not (Test-Path $mistralrsArtifacts)) { New-Item -ItemType Directory -Path $mistralrsArtifacts -Force | Out-Null }
    }

    $artifacts = @(
        @{ Name = 'pcai-llamacpp.exe'; Backend = 'llamacpp'; Dir = $llamacppArtifacts },
        @{ Name = 'pcai-mistralrs.exe'; Backend = 'mistralrs'; Dir = $mistralrsArtifacts },
        @{ Name = 'pcai_inference.dll'; Backend = 'shared'; Dir = $null },
        @{ Name = 'pcai_inference_lib.dll'; Backend = 'shared'; Dir = $null },
        @{ Name = 'pcai_core_lib.dll'; Backend = 'shared'; Dir = $null }
    )

    $copiedCount = 0
    foreach ($artifact in $artifacts) {
        $source = Join-Path $targetDir $artifact.Name
        if (Test-Path $source) {
            # Copy to ~/.local/bin
            Copy-Item -Path $source -Destination $localBin -Force
            $copiedCount++

            # Copy to build artifacts if applicable
            if ($artifact.Dir -and (Test-Path (Split-Path $artifact.Dir -Parent))) {
                Copy-Item -Path $source -Destination $artifact.Dir -Force
            }

            $size = [math]::Round((Get-Item $source).Length / 1MB, 2)
            Write-BuildStatus "$($artifact.Name) -> ~/.local/bin ($size MB)" 'Success'
        }
    }

    if ($copiedCount -eq 0) {
        Write-BuildStatus 'No artifacts found to copy' 'Warning'
    } else {
        Write-BuildStatus "Copied $copiedCount artifacts to $localBin" 'Success'
    }
}

# Main Execution
Write-BuildBanner 'PCAI-INFERENCE BUILD'

# Initialize version information
$versionScript = Join-Path $RepoRoot 'Tools\Get-BuildVersion.ps1'
if (Test-Path $versionScript) {
    $versionInfo = & $versionScript -SetEnv -Quiet
    Write-BuildInfo 'Version' $versionInfo.Version 'Cyan'
    Write-BuildInfo 'Git' "$($versionInfo.GitHashShort) ($($versionInfo.GitBranch))"
} else {
    Write-BuildInfo 'Version' '0.1.0+unknown (version script not found)' 'Yellow'
}

Write-BuildInfo 'Backend' $Backend
Write-BuildInfo 'Configuration' $Configuration
Write-BuildInfo 'CUDA' $(if ($EnableCuda) { 'Enabled' } else { 'Disabled' }) $(if ($EnableCuda) { 'Green' } else { 'DarkGray' })
Write-BuildInfo 'Cache' $(if ($DisableCache) { 'Disabled' } else { 'Enabled' })

if ($Clean) {
    Write-BuildSection 'Cleaning'
    if (Test-Path (Join-Path $ProjectRoot 'target')) {
        Remove-Item (Join-Path $ProjectRoot 'target') -Recurse -Force
        Write-BuildStatus 'Removed target directory' 'Success'
    }
}

# Ensure artifact/log directories exist
if (-not (Test-Path $script:BuildRoot)) {
    New-Item -ItemType Directory -Path $script:BuildRoot -Force | Out-Null
}
if (-not (Test-Path $script:BuildArtifactsDir)) {
    New-Item -ItemType Directory -Path $script:BuildArtifactsDir -Force | Out-Null
}
if (-not (Test-Path $script:BuildLogsDir)) {
    New-Item -ItemType Directory -Path $script:BuildLogsDir -Force | Out-Null
}

Initialize-BuildEnvironment
Clear-IncompleteCmakeConfig

Invoke-Build -BackendName $Backend
Copy-CompiledArtifacts -Configuration $Configuration -BackendBuilt $Backend

# Build Summary
$totalDuration = (Get-Date) - $script:BuildStartTime
Write-BuildBanner 'BUILD COMPLETE'
Write-BuildInfo 'Total Time' $totalDuration.ToString('hh\:mm\:ss') 'Cyan'
Write-BuildInfo 'Artifacts' "$env:USERPROFILE\.local\bin" 'Green'
if (Test-Path $script:BuildLogsDir) {
    Write-BuildInfo 'Logs' $script:BuildLogsDir 'DarkGray'
}
Write-Host ''
