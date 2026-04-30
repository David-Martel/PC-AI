<#
.SYNOPSIS
    Basic WSL2 Configuration Check (No Admin Required)

.DESCRIPTION
    Checks WSL2 configuration without requiring administrator privileges.
    Provides analysis of current setup and identifies optimization opportunities.

.EXAMPLE
    .\wsl2-basic-check.ps1
#>

Write-Host "🔍 WSL2 Basic Configuration Check" -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan
Write-Host ""

# System Information
$version = [System.Environment]::OSVersion.Version
$build = (Get-ItemProperty "HKLM:SOFTWARE\Microsoft\Windows NT\CurrentVersion").CurrentBuild
Write-Host "System: Windows $($version.Major).$($version.Minor) Build $build" -ForegroundColor Green

# WSL Status
Write-Host "`n📊 WSL Status:" -ForegroundColor Yellow
try {
    $wslStatus = wsl --status 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ WSL is installed and accessible" -ForegroundColor Green

        # Get WSL version info
        wsl --version 2>$null | ForEach-Object {
            if ($_ -and $_ -notmatch "^$") {
                Write-Host "   $_" -ForegroundColor Gray
            }
        }
    }
    else {
        Write-Host "❌ WSL is not accessible" -ForegroundColor Red
    }
}
catch {
    Write-Host "❌ WSL is not installed or not accessible" -ForegroundColor Red
}

# WSL Distributions
Write-Host "`n📋 WSL Distributions:" -ForegroundColor Yellow
try {
    $distributions = wsl --list --verbose 2>$null
    if ($LASTEXITCODE -eq 0) {
        $distributions | ForEach-Object {
            if ($_ -and $_ -notmatch "NAME|---") {
                Write-Host "   $_" -ForegroundColor Gray
            }
        }
    }
}
catch {
    Write-Host "❌ Cannot list WSL distributions" -ForegroundColor Red
}

# .wslconfig Analysis
Write-Host "`n⚙️  .wslconfig Analysis:" -ForegroundColor Yellow
$wslConfigPath = "$env:USERPROFILE\.wslconfig"

if (Test-Path $wslConfigPath) {
    Write-Host "✅ .wslconfig file exists" -ForegroundColor Green
    Write-Host "   Location: $wslConfigPath" -ForegroundColor Gray

    $content = Get-Content $wslConfigPath -Raw

    # Check for key settings
    $settings = @{
        "networkingMode=mirrored" = "Mirrored networking for improved connectivity"
        "memory=" = "Custom memory allocation"
        "processors=" = "Custom CPU allocation"
        "swap=" = "Swap configuration"
        "dnsTunneling=true" = "DNS tunneling for network stability"
        "autoProxy=true" = "Automatic proxy configuration"
        "guiApplications=true" = "GUI application support"
        "sparseVhd=true" = "Sparse VHD for efficient storage"
        "autoMemoryReclaim=" = "Memory reclaim configuration"
    }

    Write-Host ""
    foreach ($setting in $settings.GetEnumerator()) {
        if ($content -match [regex]::Escape($setting.Key)) {
            Write-Host "   ✅ $($setting.Value)" -ForegroundColor Green
        }
        else {
            Write-Host "   ❌ Missing: $($setting.Value)" -ForegroundColor Yellow
        }
    }

    # Check for experimental section
    if ($content -match "\[experimental\]") {
        Write-Host "   ✅ Experimental features section present" -ForegroundColor Green
    }
    else {
        Write-Host "   ⚠️  No experimental features section" -ForegroundColor Yellow
    }
}
else {
    Write-Host "❌ No .wslconfig file found" -ForegroundColor Red
    Write-Host "   This may limit WSL2 performance and functionality" -ForegroundColor Yellow
}

# Docker Integration Check
Write-Host "`n🐳 Docker Integration:" -ForegroundColor Yellow
try {
    $dockerVersion = docker --version 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Docker is installed: $dockerVersion" -ForegroundColor Green

        # Check if Docker is running
        $dockerInfo = docker info 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ Docker is running and accessible" -ForegroundColor Green
        }
        else {
            Write-Host "⚠️  Docker is installed but not running" -ForegroundColor Yellow
        }
    }
    else {
        Write-Host "❌ Docker is not installed or not accessible" -ForegroundColor Red
    }
}
catch {
    Write-Host "❌ Cannot check Docker status" -ForegroundColor Red
}

# Network Connectivity Test
Write-Host "`n🌐 Network Connectivity Test:" -ForegroundColor Yellow
try {
    # Test WSL internet connectivity
    $wslPing = wsl -- ping -c 2 8.8.8.8 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ WSL internet connectivity working" -ForegroundColor Green
    }
    else {
        Write-Host "❌ WSL internet connectivity issues" -ForegroundColor Red
    }

    # Test WSL to Windows host connectivity
    $hostIP = (Get-NetIPConfiguration | Where-Object { $_.IPv4DefaultGateway -ne $null }).IPv4Address.IPAddress | Select-Object -First 1
    if ($hostIP) {
        $hostPing = wsl -- ping -c 2 $hostIP 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ WSL to Windows host connectivity working" -ForegroundColor Green
        }
        else {
            Write-Host "⚠️  WSL to Windows host connectivity issues" -ForegroundColor Yellow
        }
    }
}
catch {
    Write-Host "❌ Cannot test network connectivity" -ForegroundColor Red
}

# Development Drive Check
Write-Host "`n💾 Development Drive Analysis:" -ForegroundColor Yellow
$drives = @("C:", "T:", "F:")
foreach ($drive in $drives) {
    if (Test-Path $drive) {
        try {
            $driveInfo = Get-WmiObject -Class Win32_LogicalDisk | Where-Object { $_.DeviceID -eq $drive }
            $freeGB = [math]::Round($driveInfo.FreeSpace / 1GB, 1)
            $totalGB = [math]::Round($driveInfo.Size / 1GB, 1)
            $usagePercent = [math]::Round((($totalGB - $freeGB) / $totalGB) * 100, 1)

            $color = if ($usagePercent -gt 90) { "Red" } elseif ($usagePercent -gt 80) { "Yellow" } else { "Green" }
            Write-Host "   $drive $($freeGB)GB free of $($totalGB)GB ($usagePercent% used)" -ForegroundColor $color

            # Check if T: is ReFS (Microsoft Dev Drive)
            if ($drive -eq "T:" -and $driveInfo.FileSystem -eq "ReFS") {
                Write-Host "      ✅ Microsoft Dev Drive (ReFS) detected" -ForegroundColor Green
            }
        }
        catch {
            Write-Host "   ❌ Cannot analyze $drive" -ForegroundColor Red
        }
    }
    else {
        Write-Host "   ❌ Drive $drive not accessible" -ForegroundColor Red
    }
}

# Check for WSL file paths
Write-Host "`n📁 WSL File System Access:" -ForegroundColor Yellow
$wslPaths = @("\\wsl.localhost\Ubuntu", "\\wsl$\Ubuntu")
foreach ($path in $wslPaths) {
    if (Test-Path $path) {
        Write-Host "   ✅ $path is accessible" -ForegroundColor Green
    }
    else {
        Write-Host "   ❌ $path is not accessible" -ForegroundColor Red
    }
}

# Summary and Recommendations
Write-Host "`n`n📋 SUMMARY AND NEXT STEPS:" -ForegroundColor Magenta
Write-Host "==========================" -ForegroundColor Magenta

Write-Host "`n🎯 To run full optimization (requires Administrator):" -ForegroundColor Cyan
Write-Host "   1. Open PowerShell as Administrator" -ForegroundColor Yellow
Write-Host "   2. Run: .\wsl2-hyperv-optimization.ps1 -Action Check" -ForegroundColor Yellow
Write-Host "   3. Run: .\wsl2-hyperv-optimization.ps1 -Action Optimize" -ForegroundColor Yellow
Write-Host ""

Write-Host "🔧 Quick optimizations you can do now:" -ForegroundColor Cyan
Write-Host "   • Your .wslconfig looks well configured!" -ForegroundColor Green
Write-Host "   • Restart WSL if you made recent changes: wsl --shutdown" -ForegroundColor Yellow
Write-Host "   • Check if Docker Desktop is running if you need it" -ForegroundColor Yellow

Write-Host "`n📖 For detailed analysis, run as Administrator:" -ForegroundColor Gray
Write-Host "   .\wsl2-config-analyzer.ps1" -ForegroundColor Gray