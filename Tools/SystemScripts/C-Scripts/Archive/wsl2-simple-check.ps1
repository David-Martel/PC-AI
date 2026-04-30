# Simple WSL2 Configuration Check
# Run this script to check basic WSL2 configuration

Write-Host "WSL2 Simple Configuration Check" -ForegroundColor Cyan
Write-Host "===============================" -ForegroundColor Cyan
Write-Host ""

# Check WSL version
Write-Host "WSL Version:" -ForegroundColor Yellow
wsl --version

Write-Host "`nWSL Distributions:" -ForegroundColor Yellow
wsl --list --verbose

Write-Host "`n.wslconfig File Analysis:" -ForegroundColor Yellow
$wslConfigPath = "$env:USERPROFILE\.wslconfig"
if (Test-Path $wslConfigPath) {
    Write-Host "✅ .wslconfig exists at: $wslConfigPath" -ForegroundColor Green

    $content = Get-Content $wslConfigPath -Raw
    Write-Host "`nKey Configuration Settings:" -ForegroundColor Cyan

    # Check critical settings
    if ($content -match "networkingMode\s*=\s*mirrored") {
        Write-Host "✅ Mirrored networking mode enabled" -ForegroundColor Green
    } else {
        Write-Host "⚠️  Mirrored networking mode not found" -ForegroundColor Yellow
    }

    if ($content -match "memory\s*=\s*(\d+)([GMgm])") {
        Write-Host "✅ Custom memory allocation: $($matches[1])$($matches[2])" -ForegroundColor Green
    } else {
        Write-Host "⚠️  No custom memory allocation found" -ForegroundColor Yellow
    }

    if ($content -match "processors\s*=\s*(\d+)") {
        Write-Host "✅ Custom processor allocation: $($matches[1]) cores" -ForegroundColor Green
    } else {
        Write-Host "⚠️  No custom processor allocation found" -ForegroundColor Yellow
    }

    if ($content -match "swap\s*=\s*(\d+)([GMgm])") {
        Write-Host "✅ Custom swap allocation: $($matches[1])$($matches[2])" -ForegroundColor Green
    } else {
        Write-Host "⚠️  No custom swap allocation found" -ForegroundColor Yellow
    }

    if ($content -match "\[experimental\]") {
        Write-Host "✅ Experimental features section present" -ForegroundColor Green
    } else {
        Write-Host "⚠️  No experimental features section" -ForegroundColor Yellow
    }

} else {
    Write-Host "❌ No .wslconfig file found" -ForegroundColor Red
    Write-Host "   Create one at: $wslConfigPath" -ForegroundColor Yellow
}

# Test WSL connectivity
Write-Host "`nTesting WSL Connectivity:" -ForegroundColor Yellow
try {
    $wslTest = wsl -- echo "WSL connectivity test successful"
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ WSL is responsive" -ForegroundColor Green
    } else {
        Write-Host "❌ WSL is not responding" -ForegroundColor Red
    }
} catch {
    Write-Host "❌ Cannot test WSL connectivity" -ForegroundColor Red
}

# Test internet connectivity from WSL
Write-Host "`nTesting WSL Internet Connectivity:" -ForegroundColor Yellow
try {
    $internetTest = wsl -- ping -c 2 8.8.8.8 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ WSL internet connectivity working" -ForegroundColor Green
    } else {
        Write-Host "❌ WSL internet connectivity issues" -ForegroundColor Red
    }
} catch {
    Write-Host "❌ Cannot test WSL internet connectivity" -ForegroundColor Red
}

# Check Docker
Write-Host "`nDocker Integration:" -ForegroundColor Yellow
try {
    $dockerVersion = docker --version 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Docker available: $dockerVersion" -ForegroundColor Green
    } else {
        Write-Host "⚠️  Docker not available" -ForegroundColor Yellow
    }
} catch {
    Write-Host "⚠️  Docker not accessible" -ForegroundColor Yellow
}

Write-Host "`n📋 To perform full optimization:" -ForegroundColor Cyan
Write-Host "   1. Open PowerShell as Administrator" -ForegroundColor Yellow
Write-Host "   2. Run: .\wsl2-hyperv-optimization.ps1 -Action Check" -ForegroundColor Yellow
Write-Host "   3. Run: .\wsl2-hyperv-optimization.ps1 -Action Optimize" -ForegroundColor Yellow
Write-Host ""
Write-Host "💡 Quick fixes you can do now:" -ForegroundColor Cyan
Write-Host "   • wsl --shutdown (if WSL needs restart)" -ForegroundColor Yellow
Write-Host "   • Restart Docker Desktop (if using Docker)" -ForegroundColor Yellow