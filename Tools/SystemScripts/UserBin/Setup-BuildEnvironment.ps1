#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Setup and configure comprehensive build environment for C/C++, Rust, Python, and Node.js

.DESCRIPTION
    This script configures environment variables, toolchain settings, and verifies
    all build tools are properly configured for optimal development experience.

.PARAMETER Profile
    Whether to add settings to PowerShell profile

.EXAMPLE
    .\Setup-BuildEnvironment.ps1
    .\Setup-BuildEnvironment.ps1 -Profile
#>

param(
    [Parameter(Mandatory=$false)]
    [switch]$Profile
)

function Write-ColoredOutput {
    param([string]$Message, [string]$Color = 'White')
    Write-Host $Message -ForegroundColor $Color
}

function Add-UserPathEntry([string]$entry) {
    if (-not (Test-Path $entry)) { return }
    $current = [Environment]::GetEnvironmentVariable('PATH', 'User')
    if (-not $current) { $current = "" }
    $parts = $current -split ';' | Where-Object { $_ -and $_.Trim() -ne "" }
    if ($parts -contains $entry) { return }
    $newPath = ($parts + $entry) -join ';'
    [Environment]::SetEnvironmentVariable('PATH', $newPath, 'User')
    $env:PATH = ($env:PATH.TrimEnd(';') + ';' + $entry)
    Write-ColoredOutput "  Added to PATH: $entry" "Green"
}

Write-ColoredOutput "=== Build Environment Setup ===" "Cyan"

# Essential environment variables
$envVars = @{
    # Python
    'PYTHON' = 'C:\Users\david\AppData\Local\Programs\Python\Python312\python.exe'

    # Rust/Cargo optimization
    'RUSTC_WRAPPER' = 'sccache'
    'CARGO_INCREMENTAL' = '0'  # Required for sccache compatibility
    'RUST_BACKTRACE' = '1'
    'CARGO_HOME' = 'T:\RustCache\cargo-home'
    'RUSTUP_HOME' = 'T:\RustCache\rustup'
    'CARGO_TARGET_DIR' = 'T:\RustCache\cargo-target'  # Centralized target dir
    'RUST_ANALYZER_CACHE_DIR' = 'T:\RustCache\ra-cache'
    'CARGO_WSL_CACHE' = 'native'
    'CARGO_ROUTE_WASM' = 'wsl'
    'CARGO_ROUTE_MACOS' = 'docker'
    'RA_SINGLETON' = '1'
    'CARGO_PREFLIGHT' = '1'
    'CARGO_PREFLIGHT_MODE' = 'all'
    'CARGO_PREFLIGHT_STRICT' = '0'
    'CARGO_RA_PREFLIGHT' = '1'
    'RA_DIAGNOSTICS_FLAGS' = '--disable-build-scripts --disable-proc-macros'
    'CARGO_PREFLIGHT_BLOCKING' = '0'
    'CARGO_PREFLIGHT_IDE_GUARD' = '1'
    'CARGO_PREFLIGHT_FORCE' = '0'

    # sccache configuration
    'SCCACHE_DIR' = 'T:\RustCache\sccache'
    'SCCACHE_CACHE_SIZE' = '30G'
    'SCCACHE_IDLE_TIMEOUT' = '1800'
    'SCCACHE_CACHE_COMPRESSION' = 'zstd'
    'SCCACHE_SERVER_PORT' = '4226'
    'SCCACHE_STARTUP_TIMEOUT' = '15'
    'SCCACHE_REQUEST_TIMEOUT' = '60'
    'SCCACHE_LOG' = 'warn'
    'SCCACHE_ERROR_LOG' = 'T:\RustCache\sccache\error.log'
    'SCCACHE_DIRECT' = 'true'
    'SCCACHE_NO_DAEMON' = '0'

    # LLVM/MSVC linker acceleration
    'CARGO_USE_LLD' = '1'
    'CARGO_LLD_PATH' = 'C:\Program Files\LLVM\bin\lld-link.exe'
    'CARGO_USE_FASTLINK' = '0'

    # C/C++ build optimization
    'CL' = '/MP'  # Enable parallel compilation in MSVC
    'MAKEFLAGS' = '-j8'  # Parallel make jobs

    # CMake
    'CMAKE_GENERATOR' = 'Ninja'  # Use Ninja by default
    'CMAKE_BUILD_PARALLEL_LEVEL' = '8'

    # Node.js/npm
    'NODE_OPTIONS' = '--max-old-space-size=8192'  # 8GB heap for Node

    # General
    'VCPKG_ROOT' = 'C:\vcpkg'  # If using vcpkg
}

Write-ColoredOutput "`nEnsuring shared Rust cache directories..." "Yellow"
$cacheDirs = @(
    'T:\RustCache\sccache',
    'T:\RustCache\cargo-target',
    'T:\RustCache\cargo-home',
    'T:\RustCache\rustup',
    'T:\RustCache\ra-cache'
)
foreach ($dir in $cacheDirs) {
    try {
        if (-not (Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
            Write-ColoredOutput "  Created: $dir" "Green"
        } else {
            Write-ColoredOutput "  OK: $dir" "Green"
        }
    } catch {
        Write-ColoredOutput "  Skipped: $dir ($($_.Exception.Message))" "Gray"
    }
}

Write-ColoredOutput "`nConfiguring environment variables..." "Yellow"

# Ensure vswhere.exe is on PATH (needed to locate VS installs)
$vswhereDir = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer"
if (Test-Path (Join-Path $vswhereDir "vswhere.exe")) {
    Add-UserPathEntry $vswhereDir
}

# Ensure Chocolatey bin is on PATH (safe to append)
$chocoRoot = if ($env:ChocolateyInstall) { $env:ChocolateyInstall } else { "C:\ProgramData\chocolatey" }
$chocoBin = Join-Path $chocoRoot "bin"
if (Test-Path $chocoBin) {
    Add-UserPathEntry $chocoBin
}

# Ensure user tool bins are on PATH
Add-UserPathEntry "$env:USERPROFILE\bin"
Add-UserPathEntry "$env:USERPROFILE\.local\bin"

foreach ($var in $envVars.GetEnumerator()) {
    $name = $var.Key
    $value = $var.Value

    # Check if path exists (for paths)
    if ($value -match '^[A-Z]:\\' -and $value -notmatch '\.exe$') {
        if (-not (Test-Path $value)) {
            Write-ColoredOutput "  Skipped $name (path doesn't exist): $value" "Gray"
            continue
        }
    }

    $current = [Environment]::GetEnvironmentVariable($name, 'User')
    if ($current -eq $value) {
        Write-ColoredOutput "  OK: $name already set" "Green"
    } else {
        try {
            [Environment]::SetEnvironmentVariable($name, $value, 'User')
            Set-Item -Path "env:$name" -Value $value
            Write-ColoredOutput "  Set: $name = $value" "Green"
        } catch {
            Write-ColoredOutput "  Failed: $name - $($_.Exception.Message)" "Red"
        }
    }
}

# Create PowerShell profile configuration
if ($Profile) {
    $profileContent = @"

# === Developer Tools Configuration ===
# Auto-generated by Setup-BuildEnvironment.ps1

# sccache for Rust
`$env:RUSTC_WRAPPER = 'sccache'
`$env:CARGO_INCREMENTAL = '0'
`$env:SCCACHE_DIR = 'T:\RustCache\sccache'
`$env:SCCACHE_SERVER_PORT = '4226'
`$env:SCCACHE_LOG = 'warn'
`$env:SCCACHE_ERROR_LOG = 'T:\RustCache\sccache\error.log'
`$env:SCCACHE_DIRECT = 'true'
`$env:SCCACHE_NO_DAEMON = '0'
`$env:CARGO_TARGET_DIR = 'T:\RustCache\cargo-target'
`$env:CARGO_HOME = 'T:\RustCache\cargo-home'
`$env:RUSTUP_HOME = 'T:\RustCache\rustup'
`$env:RUST_ANALYZER_CACHE_DIR = 'T:\RustCache\ra-cache'
`$env:CARGO_WSL_CACHE = 'native'
`$env:CARGO_ROUTE_WASM = 'wsl'
`$env:CARGO_ROUTE_MACOS = 'docker'
`$env:RA_SINGLETON = '1'
`$env:CARGO_PREFLIGHT = '1'
`$env:CARGO_PREFLIGHT_MODE = 'all'
`$env:CARGO_PREFLIGHT_STRICT = '0'
`$env:CARGO_RA_PREFLIGHT = '1'
`$env:RA_DIAGNOSTICS_FLAGS = '--disable-build-scripts --disable-proc-macros'
`$env:CARGO_PREFLIGHT_BLOCKING = '0'
`$env:CARGO_PREFLIGHT_IDE_GUARD = '1'
`$env:CARGO_PREFLIGHT_FORCE = '0'
`$env:CARGO_USE_LLD = '1'
`$env:CARGO_LLD_PATH = 'C:\Program Files\LLVM\bin\lld-link.exe'
`$env:CARGO_USE_FASTLINK = '0'

# Build parallelization
`$env:MAKEFLAGS = '-j8'
`$env:CMAKE_BUILD_PARALLEL_LEVEL = '8'

# Python
`$env:PYTHON = 'C:\Users\david\AppData\Local\Programs\Python\Python312\python.exe'

# Aliases for common tasks
function cargo-quick { cargo build --profile quick-dev `$args }
function cargo-release-fast { cargo build --profile release-dev `$args }
function sccache-stats { sccache --show-stats }
function sccache-reset { sccache --zero-stats }
function build-rust {
    sccache --zero-stats
    cargo build `$args
    sccache --show-stats
}

Write-Host "Developer environment loaded" -ForegroundColor Green
"@

    Write-ColoredOutput "`nPowerShell profile configuration:" "Cyan"
    Write-ColoredOutput $profileContent "White"

    $addToProfile = Read-Host "`nAdd to your PowerShell profile? (Y/N)"
    if ($addToProfile -eq 'Y' -or $addToProfile -eq 'y') {
        $profilePath = $PROFILE.CurrentUserAllHosts
        if (-not (Test-Path $profilePath)) {
            New-Item -ItemType File -Path $profilePath -Force | Out-Null
        }
        Add-Content -Path $profilePath -Value $profileContent
        Write-ColoredOutput "Added to profile: $profilePath" "Green"
        Write-ColoredOutput "Run '. `$PROFILE' to reload" "Yellow"
    }
}

# Verify toolchains
Write-ColoredOutput "`n=== Verifying Toolchains ===" "Cyan"

$tests = @(
    @{Name='MSVC'; Cmd='cl'; Args='/? 2>&1'; Pattern='Version'},
    @{Name='GCC'; Cmd='gcc'; Args='--version'; Pattern='gcc'},
    @{Name='Rust'; Cmd='rustc'; Args='--version'; Pattern='rustc'},
    @{Name='Cargo'; Cmd='cargo'; Args='--version'; Pattern='cargo'},
    @{Name='Python'; Cmd='python'; Args='--version'; Pattern='Python'},
    @{Name='Node.js'; Cmd='node'; Args='--version'; Pattern='v'},
    @{Name='CMake'; Cmd='cmake'; Args='--version'; Pattern='cmake'},
    @{Name='Make'; Cmd='make'; Args='--version'; Pattern='GNU'},
    @{Name='Ninja'; Cmd='ninja'; Args='--version'; Pattern='\d'},
    @{Name='sccache'; Cmd='sccache'; Args='--version'; Pattern='sccache'}
)

foreach ($test in $tests) {
    try {
        $output = Invoke-Expression "$($test.Cmd) $($test.Args)" 2>&1
        $match = $output | Select-String $test.Pattern | Select-Object -First 1
        if ($match) {
            Write-ColoredOutput "  ✓ $($test.Name): $match" "Green"
        } else {
            Write-ColoredOutput "  ✗ $($test.Name): Found but unexpected output" "Yellow"
        }
    } catch {
        Write-ColoredOutput "  ✗ $($test.Name): Not available" "Red"
    }
}

# Test sccache integration
Write-ColoredOutput "`n=== Testing sccache ===" "Cyan"
try {
    sccache --show-stats | Select-Object -First 5 | ForEach-Object {
        Write-ColoredOutput "  $_" "White"
    }
    Write-ColoredOutput "  sccache is working correctly" "Green"
} catch {
    Write-ColoredOutput "  sccache test failed" "Red"
}

# Print optimization tips
Write-ColoredOutput "`n=== Optimization Tips ===" "Cyan"
Write-ColoredOutput @"

1. Rust builds:
   - Use 'cargo build --profile quick-dev' for fastest iteration
   - Use 'cargo build --profile release-dev' for fast release builds
   - Monitor sccache: 'sccache --show-stats'

2. C/C++ builds:
   - Use 'cmake -G Ninja' for fastest builds
   - Enable parallel compilation: 'cmake --build . -j8'
   - MSVC automatically uses /MP (parallel compilation)
   - Prefer LLVM lld-link when available (CARGO_USE_LLD=1)
   - Use Developer PowerShell or run vcvars64.bat to init MSVC env
   - VS 2026 removes /DEBUG:FASTLINK; keep CARGO_USE_FASTLINK=0 for VS 2026

3. Clean builds:
   - Rust: 'cargo clean && cargo build'
   - CMake: 'rm -r build && cmake -B build'
   - sccache: 'sccache --stop-server' to clear memory

4. Profile-guided optimization:
   - Enable in cargo with [profile.release]
   - Use 'lto = "thin"' for faster link-time optimization

5. Common issues:
   - If sccache shows 0 requests, check RUSTC_WRAPPER is set
   - If builds are slow, check CARGO_INCREMENTAL = 0
   - Monitor cache hit rate: should be >50% on rebuilds

"@ "White"

Write-ColoredOutput "=== Setup Complete ===" "Cyan"
Write-ColoredOutput "Your build environment is configured!" "Green"
Write-ColoredOutput "`nNext steps:" "Yellow"
Write-ColoredOutput "  1. Reload your terminal or run: . `$PROFILE" "White"
Write-ColoredOutput "  2. Test with: cargo build (in a Rust project)" "White"
Write-ColoredOutput "  3. Check cache: sccache --show-stats" "White"
Write-ColoredOutput "  4. Run: .\Manage-DevTools.ps1 -Action Symlink (as admin)" "White"
