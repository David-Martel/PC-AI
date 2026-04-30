# Simplified Rust Build Optimization Script for Windows
# Optimized for frequent recompilation

[CmdletBinding()]
param(
    [Parameter(Mandatory=$false)]
    [ValidateSet("dev", "quick-dev", "release-dev", "release")]
    [string]$Profile = "dev",

    [Parameter(Mandatory=$false)]
    [ValidateSet("check", "build", "test", "clippy")]
    [string]$Action = "build",

    [Parameter(Mandatory=$false)]
    [ValidateSet("rust-fs", "rust-commander", "both")]
    [string]$Project = "both"
)

$RustFsPath = "T:\projects\rust\rust-fs"
$RustCommanderPath = "T:\projects\rust\rust-commander"

function Setup-Environment {
    Write-Host "Setting up optimized build environment..." -ForegroundColor Green

    # sccache configuration
    $env:RUSTC_WRAPPER = "sccache"
    $env:SCCACHE_CACHE_SIZE = "15G"
    $env:SCCACHE_DIR = "C:\Users\david\AppData\Local\sccache\cache"
    $env:SCCACHE_IDLE_TIMEOUT = "1800"
    $env:SCCACHE_CACHE_COMPRESSION = "zstd"
    $env:SCCACHE_DIRECT_MODE = "true"

    # Rust optimization
    $env:RUST_BACKTRACE = "1"
    $env:CARGO_TERM_COLOR = "always"

    # Enable incremental for dev builds
    if ($Profile -eq "dev" -or $Profile -eq "quick-dev") {
        $env:CARGO_INCREMENTAL = "1"
    } else {
        $env:CARGO_INCREMENTAL = "0"
    }

    # Start sccache server if needed
    $stats = sccache --show-stats 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Starting sccache server..." -ForegroundColor Yellow
        sccache --start-server
    }
}

function Build-Project {
    param([string]$Path, [string]$Name)

    Write-Host "Building $Name..." -ForegroundColor Cyan
    Write-Host "Path: $Path" -ForegroundColor Gray

    Push-Location $Path

    $startTime = Get-Date

    try {
        switch ($Action) {
            "check" {
                cargo check --workspace --all-features --message-format=short
            }
            "build" {
                if ($Profile -eq "dev") {
                    cargo build --workspace
                } else {
                    cargo build --workspace --profile $Profile
                }
            }
            "test" {
                if ($Profile -eq "dev") {
                    cargo test --workspace
                } else {
                    cargo test --workspace --profile $Profile
                }
            }
            "clippy" {
                cargo clippy --workspace --all-features --all-targets -- -D warnings
            }
        }

        $endTime = Get-Date
        $duration = ($endTime - $startTime).TotalSeconds

        if ($LASTEXITCODE -eq 0) {
            Write-Host "SUCCESS: $Name completed in $($duration.ToString('F2'))s" -ForegroundColor Green
            return $true
        } else {
            Write-Host "FAILED: $Name failed with exit code $LASTEXITCODE" -ForegroundColor Red
            return $false
        }
    }
    finally {
        Pop-Location
    }
}

# Main execution
Write-Host "Rust Build Optimizer - Profile: $Profile, Action: $Action, Project: $Project" -ForegroundColor Yellow

Setup-Environment

$projects = @()
switch ($Project) {
    "both" { $projects = @(("rust-fs", $RustFsPath), ("rust-commander", $RustCommanderPath)) }
    "rust-fs" { $projects = @(("rust-fs", $RustFsPath)) }
    "rust-commander" { $projects = @(("rust-commander", $RustCommanderPath)) }
}

$overallStart = Get-Date
$results = @()

foreach ($project in $projects) {
    $name = $project[0]
    $path = $project[1]

    if (Test-Path $path) {
        $success = Build-Project -Path $path -Name $name
        $results += @{Name = $name; Success = $success}
    } else {
        Write-Host "WARNING: Project path not found: $path" -ForegroundColor Yellow
    }
}

$overallEnd = Get-Date
$totalTime = ($overallEnd - $overallStart).TotalSeconds

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Build Summary" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Total Time: $($totalTime.ToString('F2'))s" -ForegroundColor White

foreach ($result in $results) {
    $status = if ($result.Success) { "PASS" } else { "FAIL" }
    $color = if ($result.Success) { "Green" } else { "Red" }
    Write-Host "$status $($result.Name)" -ForegroundColor $color
}

Write-Host ""
Write-Host "sccache Statistics:" -ForegroundColor Yellow
sccache --show-stats

Write-Host "Build optimization complete!" -ForegroundColor Green