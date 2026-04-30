# Setup sccache environment variables for current PowerShell session
# This can be sourced in PowerShell profile for automatic activation

Write-Host "Setting up sccache environment..." -ForegroundColor Cyan

# Set required environment variables
$env:RUSTC_WRAPPER = "sccache"
$env:SCCACHE_DIR = "T:\RustCache\sccache"
$env:SCCACHE_CACHE_COMPRESSION = "zstd"
$env:SCCACHE_CACHE_SIZE = "30G"
$env:SCCACHE_IDLE_TIMEOUT = "1800"
$env:SCCACHE_DIRECT = "true"
$env:SCCACHE_SERVER_PORT = "4400"
$env:SCCACHE_LOG = "warn"
$env:SCCACHE_ERROR_LOG = "T:\RustCache\sccache\error.log"
$env:SCCACHE_NO_DAEMON = "0"
$env:CARGO_TARGET_DIR = "T:\RustCache\cargo-target"

# Optional MSVC linker accelerators (wrapper-controlled)
$lldPath = "C:\Program Files\LLVM\bin\lld-link.exe"
if (Test-Path $lldPath) {
    $env:CARGO_LLD_PATH = $lldPath
    if (-not $env:CARGO_USE_LLD) {
        $env:CARGO_USE_LLD = "1"
    }
} elseif (-not $env:CARGO_USE_LLD) {
    $env:CARGO_USE_LLD = "0"
}

if (-not $env:CARGO_USE_NATIVE) {
    $env:CARGO_USE_NATIVE = "0"
}

if (-not $env:CARGO_USE_FASTLINK) {
    $env:CARGO_USE_FASTLINK = "0"
}

Write-Host "✅ sccache environment configured:" -ForegroundColor Green
Write-Host "   RUSTC_WRAPPER = $env:RUSTC_WRAPPER" -ForegroundColor Gray
Write-Host "   SCCACHE_DIR = $env:SCCACHE_DIR" -ForegroundColor Gray
Write-Host "   SCCACHE_CACHE_COMPRESSION = $env:SCCACHE_CACHE_COMPRESSION" -ForegroundColor Gray
Write-Host "   SCCACHE_CACHE_SIZE = $env:SCCACHE_CACHE_SIZE" -ForegroundColor Gray
Write-Host "   SCCACHE_DIRECT = $env:SCCACHE_DIRECT" -ForegroundColor Gray
Write-Host "   SCCACHE_SERVER_PORT = $env:SCCACHE_SERVER_PORT" -ForegroundColor Gray
Write-Host "   SCCACHE_ERROR_LOG = $env:SCCACHE_ERROR_LOG" -ForegroundColor Gray
Write-Host "   CARGO_TARGET_DIR = $env:CARGO_TARGET_DIR" -ForegroundColor Gray
Write-Host "   CARGO_USE_LLD = $env:CARGO_USE_LLD" -ForegroundColor Gray
Write-Host "   CARGO_LLD_PATH = $env:CARGO_LLD_PATH" -ForegroundColor Gray
Write-Host "   CARGO_USE_NATIVE = $env:CARGO_USE_NATIVE" -ForegroundColor Gray
Write-Host "   CARGO_USE_FASTLINK = $env:CARGO_USE_FASTLINK" -ForegroundColor Gray
Write-Host ""

# Define helper functions
function cargo-stats {
    Write-Host "Cache Statistics:" -ForegroundColor Cyan
    sccache --show-stats
}

function cargo-reset {
    Write-Host "Resetting sccache statistics..." -ForegroundColor Yellow
    sccache --zero-stats
    Write-Host "✅ Statistics reset" -ForegroundColor Green
}

function cargo-clear {
    Write-Host "WARNING: This will clear all cached compilations!" -ForegroundColor Red
    $confirm = Read-Host "Are you sure? (yes/no)"
    if ($confirm -eq "yes") {
        sccache --stop-server
        Remove-Item -Recurse -Force "$env:SCCACHE_DIR\*"
        sccache --start-server
        Write-Host "✅ Cache cleared and daemon restarted" -ForegroundColor Green
    } else {
        Write-Host "❌ Cancelled" -ForegroundColor Yellow
    }
}

function cargo-verify {
    & 'C:\Users\david\.local\etc\verify-sccache.ps1' @args
}

Write-Host "Available commands:" -ForegroundColor Cyan
Write-Host "  cargo-stats      - Show cache statistics" -ForegroundColor Gray
Write-Host "  cargo-reset      - Reset statistics counters" -ForegroundColor Gray
Write-Host "  cargo-clear      - Clear all cache (WARNING: destructive)" -ForegroundColor Gray
Write-Host "  cargo-verify     - Run full verification check" -ForegroundColor Gray
Write-Host ""
Write-Host "Verify acceleration is active:" -ForegroundColor Cyan
Write-Host "  `$(cargo-stats)" -ForegroundColor Gray

try {
    sccache --start-server | Out-Null
} catch {
    Write-Host "⚠️  sccache server did not start automatically: $($_.Exception.Message)" -ForegroundColor Yellow
}
