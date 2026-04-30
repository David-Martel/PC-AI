#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Comprehensive sccache diagnostic and troubleshooting tool

.DESCRIPTION
    Diagnoses common sccache issues on Windows and provides solutions

.EXAMPLE
    .\Diagnose-Sccache.ps1
#>

function Write-ColoredOutput {
    param([string]$Message, [string]$Color = 'White')
    Write-Host $Message -ForegroundColor $Color
}

Write-ColoredOutput "=== sccache Diagnostic Tool ===" "Cyan"

# Check if sccache is installed
Write-ColoredOutput "`n[1] Checking sccache installation..." "Yellow"
try {
    $sccachePath = (Get-Command sccache -ErrorAction Stop).Source
    $sccacheVersion = & sccache --version
    Write-ColoredOutput "  ✓ sccache found: $sccachePath" "Green"
    Write-ColoredOutput "  ✓ Version: $sccacheVersion" "Green"
} catch {
    Write-ColoredOutput "  ✗ sccache not found in PATH" "Red"
    Write-ColoredOutput "  Install with: cargo install sccache" "Yellow"
    exit 1
}

# Check environment variables
Write-ColoredOutput "`n[2] Checking environment variables..." "Yellow"
$requiredVars = @{
    'RUSTC_WRAPPER' = 'sccache'
    'CARGO_INCREMENTAL' = '0'
}

$optionalVars = @{
    'SCCACHE_DIR' = 'Cache directory'
    'SCCACHE_CACHE_SIZE' = 'Cache size limit'
    'SCCACHE_IDLE_TIMEOUT' = 'Server idle timeout'
}

$issues = @()

foreach ($var in $requiredVars.GetEnumerator()) {
    $value = [Environment]::GetEnvironmentVariable($var.Key, 'User')
    $processValue = Get-Item -Path "env:$($var.Key)" -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Value

    if ($value -eq $var.Value -or $processValue -eq $var.Value) {
        Write-ColoredOutput "  ✓ $($var.Key) = $($var.Value)" "Green"
    } else {
        Write-ColoredOutput "  ✗ $($var.Key) not set or incorrect" "Red"
        Write-ColoredOutput "    Expected: $($var.Value)" "Yellow"
        Write-ColoredOutput "    Current: $value" "Gray"
        $issues += "Set $($var.Key)=$($var.Value)"
    }
}

foreach ($var in $optionalVars.GetEnumerator()) {
    $value = [Environment]::GetEnvironmentVariable($var.Key, 'User')
    if ($value) {
        Write-ColoredOutput "  ✓ $($var.Key) = $value" "Green"
    } else {
        Write-ColoredOutput "  ⚠ $($var.Key) not set ($($var.Value))" "Yellow"
    }
}

# Check cargo config
Write-ColoredOutput "`n[3] Checking Cargo configuration..." "Yellow"
$cargoConfig = "$env:USERPROFILE\.cargo\config.toml"
if (Test-Path $cargoConfig) {
    Write-ColoredOutput "  ✓ Found: $cargoConfig" "Green"

    $content = Get-Content $cargoConfig -Raw

    # Check for incremental = false
    if ($content -match 'incremental\s*=\s*false') {
        Write-ColoredOutput "  ✓ incremental = false in profiles" "Green"
    } else {
        Write-ColoredOutput "  ✗ incremental should be false for sccache" "Red"
        $issues += "Set incremental = false in [profile.dev]"
    }

    # Check for RUSTC_WRAPPER in [env]
    if ($content -match 'RUSTC_WRAPPER\s*=\s*"sccache"') {
        Write-ColoredOutput "  ✓ RUSTC_WRAPPER in config [env]" "Green"
    } else {
        Write-ColoredOutput "  ⚠ RUSTC_WRAPPER not in config (using env var)" "Yellow"
    }
} else {
    Write-ColoredOutput "  ⚠ No cargo config found at $cargoConfig" "Yellow"
}

# Check cache directory
Write-ColoredOutput "`n[4] Checking cache status..." "Yellow"
try {
    $stats = & sccache --show-stats
    $cacheDir = $stats | Select-String "Cache location.*Local disk: \"(.*)\"" | ForEach-Object { $_.Matches.Groups[1].Value }

    if ($cacheDir) {
        Write-ColoredOutput "  Cache directory: $cacheDir" "White"

        if (Test-Path $cacheDir) {
            $files = Get-ChildItem $cacheDir -Recurse -File -ErrorAction SilentlyContinue
            $size = ($files | Measure-Object -Property Length -Sum).Sum / 1GB
            $count = $files.Count

            Write-ColoredOutput "  ✓ Cache exists: $count files, $([math]::Round($size, 2)) GB" "Green"
        } else {
            Write-ColoredOutput "  ⚠ Cache directory doesn't exist yet" "Yellow"
            Write-ColoredOutput "    It will be created on first compilation" "Gray"
        }
    }

    # Parse statistics
    Write-ColoredOutput "`n  Current statistics:" "Cyan"
    $compileRequests = $stats | Select-String "Compile requests\s+(\d+)" | ForEach-Object { $_.Matches.Groups[1].Value }
    $cacheHits = $stats | Select-String "Cache hits\s+(\d+)" | ForEach-Object { $_.Matches.Groups[1].Value }
    $cacheMisses = $stats | Select-String "Cache misses\s+(\d+)" | ForEach-Object { $_.Matches.Groups[1].Value }

    Write-ColoredOutput "    Compile requests: $compileRequests" "White"
    Write-ColoredOutput "    Cache hits: $cacheHits" "White"
    Write-ColoredOutput "    Cache misses: $cacheMisses" "White"

    if ([int]$compileRequests -eq 0) {
        Write-ColoredOutput "`n  ⚠ No compile requests yet - sccache hasn't been used" "Yellow"
        Write-ColoredOutput "    This is normal if you haven't built anything yet" "Gray"
    } elseif ([int]$compileRequests -gt 0 -and [int]$cacheHits -eq 0 -and [int]$cacheMisses -eq 0) {
        Write-ColoredOutput "`n  ⚠ Compile requests but no hits/misses" "Yellow"
        Write-ColoredOutput "    This may indicate non-compilation calls (like --print cfg)" "Gray"
    }

} catch {
    Write-ColoredOutput "  ✗ Failed to get sccache stats" "Red"
}

# Check server status
Write-ColoredOutput "`n[5] Checking sccache server..." "Yellow"
try {
    # Try to show stats (which requires server)
    $null = & sccache --show-stats 2>&1
    Write-ColoredOutput "  ✓ sccache server is running" "Green"
} catch {
    Write-ColoredOutput "  ✗ sccache server not responding" "Red"
    Write-ColoredOutput "    Run: sccache --start-server" "Yellow"
    $issues += "Start sccache server"
}

# Check for common issues
Write-ColoredOutput "`n[6] Checking for common issues..." "Yellow"

# Check rustup proxy issue
try {
    $rustcPath = & where.exe rustc 2>$null | Select-Object -First 1
    $rustupPath = & where.exe rustup 2>$null | Select-Object -First 1

    if ($rustcPath -and $rustupPath) {
        $rustcDir = Split-Path $rustcPath
        $rustupDir = Split-Path $rustupPath

        if ($rustcDir -ne $rustupDir) {
            Write-ColoredOutput "  ⚠ rustc and rustup in different directories" "Yellow"
            Write-ColoredOutput "    rustc: $rustcDir" "Gray"
            Write-ColoredOutput "    rustup: $rustupDir" "Gray"
            Write-ColoredOutput "    This is OK but may cause warnings in logs" "Gray"
        } else {
            Write-ColoredOutput "  ✓ rustc and rustup properly located" "Green"
        }
    }
} catch {}

# Check for incremental compilation
$incrementalUser = [Environment]::GetEnvironmentVariable('CARGO_INCREMENTAL', 'User')
$incrementalProcess = $env:CARGO_INCREMENTAL
if ($incrementalUser -eq '1' -or $incrementalProcess -eq '1') {
    Write-ColoredOutput "  ✗ CARGO_INCREMENTAL=1 (incompatible with sccache!)" "Red"
    $issues += "Set CARGO_INCREMENTAL=0"
}

# Summary and recommendations
Write-ColoredOutput "`n=== Diagnosis Summary ===" "Cyan"

if ($issues.Count -eq 0) {
    Write-ColoredOutput "✓ No critical issues found!" "Green"
    Write-ColoredOutput "`nsccache appears to be configured correctly." "Green"
    Write-ColoredOutput "If you're not seeing cache acceleration:" "Yellow"
    Write-ColoredOutput "  1. Build something: cargo build" "White"
    Write-ColoredOutput "  2. Clean and rebuild: cargo clean && cargo build" "White"
    Write-ColoredOutput "  3. Check stats: sccache --show-stats" "White"
    Write-ColoredOutput "  4. Check logs: T:\RustCache\sccache\error.log" "White"
} else {
    Write-ColoredOutput "Found $($issues.Count) issue(s) to fix:" "Red"
    foreach ($issue in $issues) {
        Write-ColoredOutput "  • $issue" "Yellow"
    }

    Write-ColoredOutput "`nQuick fix:" "Cyan"
    Write-ColoredOutput "  Run: .\Setup-BuildEnvironment.ps1" "White"
    Write-ColoredOutput "  This will configure all required settings" "Gray"
}

# Test compilation
Write-ColoredOutput "`n=== Quick Test ===" "Cyan"
$testResponse = Read-Host "Do you want to run a quick sccache test? (Y/N)"
if ($testResponse -eq 'Y' -or $testResponse -eq 'y') {
    Write-ColoredOutput "`nRunning test..." "Yellow"

    # Create temp Rust project
    $tempDir = Join-Path $env:TEMP "sccache-test-$(Get-Random)"
    & cargo new $tempDir --quiet 2>&1 | Out-Null

    if (Test-Path $tempDir) {
        Push-Location $tempDir

        Write-ColoredOutput "Building test project (first build)..." "White"
        & sccache --zero-stats 2>&1 | Out-Null
        $build1 = Measure-Command { & cargo build --quiet 2>&1 | Out-Null }

        Write-ColoredOutput "Building test project (second build)..." "White"
        & cargo clean --quiet 2>&1 | Out-Null
        $build2 = Measure-Command { & cargo build --quiet 2>&1 | Out-Null }

        $stats = & sccache --show-stats
        $hits = $stats | Select-String "Cache hits\s+(\d+)" | ForEach-Object { $_.Matches.Groups[1].Value }
        $misses = $stats | Select-String "Cache misses\s+(\d+)" | ForEach-Object { $_.Matches.Groups[1].Value }

        Write-ColoredOutput "`nResults:" "Cyan"
        Write-ColoredOutput "  First build: $([math]::Round($build1.TotalSeconds, 1))s" "White"
        Write-ColoredOutput "  Second build: $([math]::Round($build2.TotalSeconds, 1))s" "White"
        Write-ColoredOutput "  Cache hits: $hits" "White"
        Write-ColoredOutput "  Cache misses: $misses" "White"

        if ([int]$hits -gt 0) {
            $speedup = $build1.TotalSeconds / $build2.TotalSeconds
            Write-ColoredOutput "`n  ✓ sccache is working! Speedup: $([math]::Round($speedup, 2))x" "Green"
        } else {
            Write-ColoredOutput "`n  ⚠ No cache hits detected" "Yellow"
            Write-ColoredOutput "  This may be normal for very small projects" "Gray"
        }

        Pop-Location
        Remove-Item $tempDir -Recurse -Force -ErrorAction SilentlyContinue
    }
}

Write-ColoredOutput "`n=== Diagnostic Complete ===" "Cyan"
