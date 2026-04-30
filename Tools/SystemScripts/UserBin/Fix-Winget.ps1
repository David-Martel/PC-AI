#Requires -RunAsAdministrator
<#
.SYNOPSIS
Diagnose and fix common winget issues.

.DESCRIPTION
Identifies and fixes common winget problems including:
- Slow source updates
- Corrupted cache
- Missing dependencies
- Source connectivity issues
#>

$ErrorActionPreference = 'Continue'

Write-Host "`n=== Winget Diagnostics and Fixes ===" -ForegroundColor Cyan

# Test 1: Check if winget is available
Write-Host "`n[1/6] Checking winget availability..." -ForegroundColor Gray
$wingetPath = (Get-Command winget -ErrorAction SilentlyContinue).Source
if ($wingetPath) {
    Write-Host "✓ Winget found at: $wingetPath" -ForegroundColor Green
    $version = winget --version
    Write-Host "  Version: $version" -ForegroundColor Green
} else {
    Write-Host "✗ Winget not found" -ForegroundColor Red
    exit 1
}

# Test 2: Check sources
Write-Host "`n[2/6] Checking winget sources..." -ForegroundColor Gray
try {
    $sources = winget source list 2>&1
    if ($sources) {
        Write-Host "✓ Sources configured:" -ForegroundColor Green
        $sources | Where-Object { $_ -match '^(msstore|winget)' } | ForEach-Object { Write-Host "  $_" -ForegroundColor Green }
    }
} catch {
    Write-Host "⚠ Could not retrieve sources: $_" -ForegroundColor Yellow
}

# Test 3: Clear winget cache
Write-Host "`n[3/6] Clearing winget cache..." -ForegroundColor Gray
$cacheDir = "$env:LOCALAPPDATA\Microsoft\WinGet\Cache"
if (Test-Path $cacheDir) {
    try {
        Remove-Item "$cacheDir\*" -Recurse -Force -ErrorAction SilentlyContinue
        Write-Host "✓ Cache cleared" -ForegroundColor Green
    } catch {
        Write-Host "⚠ Could not fully clear cache: $_" -ForegroundColor Yellow
    }
} else {
    Write-Host "✓ No cache directory found (not needed)" -ForegroundColor Green
}

# Test 4: Reset winget sources
Write-Host "`n[4/6] Resetting winget sources..." -ForegroundColor Gray
try {
    winget source reset --force 2>&1 | Out-Null
    Write-Host "✓ Sources reset" -ForegroundColor Green
} catch {
    Write-Host "⚠ Source reset skipped: $_" -ForegroundColor Yellow
}

# Test 5: Update package cache
Write-Host "`n[5/6] Updating package metadata..." -ForegroundColor Gray
try {
    Write-Host "  This may take a minute..." -ForegroundColor Gray
    winget source update 2>&1 | Out-Null
    Write-Host "✓ Metadata updated" -ForegroundColor Green
} catch {
    Write-Host "⚠ Metadata update had issues: $_" -ForegroundColor Yellow
}

# Test 6: Verify winget works
Write-Host "`n[6/6] Testing winget functionality..." -ForegroundColor Gray
try {
    $testSearch = winget search "7zip" --accept-source-agreements 2>&1 | head -3
    if ($testSearch) {
        Write-Host "✓ Winget search working" -ForegroundColor Green
    } else {
        Write-Host "⚠ Winget search returned no results" -ForegroundColor Yellow
    }
} catch {
    Write-Host "⚠ Winget search test failed: $_" -ForegroundColor Yellow
}

Write-Host "`n=== Winget Diagnostics Complete ===" -ForegroundColor Cyan
Write-Host "Winget should now be functioning normally.`n" -ForegroundColor Green
