#Requires -RunAsAdministrator

<#
.SYNOPSIS
    WSL2 Quick Optimization Script - One-Click Solution

.DESCRIPTION
    Combines analysis and optimization in a single script for Windows 11 WSL2.
    Automatically detects issues and applies optimizations with user confirmation.

.PARAMETER AnalyzeOnly
    Only run analysis without applying optimizations

.PARAMETER AutoOptimize
    Apply optimizations automatically without prompts (use with caution)

.EXAMPLE
    .\wsl2-quick-optimize.ps1
    .\wsl2-quick-optimize.ps1 -AnalyzeOnly
    .\wsl2-quick-optimize.ps1 -AutoOptimize

.NOTES
    This is a simplified version that combines the analyzer and optimizer
    For detailed control, use wsl2-config-analyzer.ps1 and wsl2-hyperv-optimization.ps1
#>

param(
    [switch]$AnalyzeOnly,
    [switch]$AutoOptimize
)

# Quick configuration check
function Test-QuickConfiguration {
    Write-Host "🔍 Quick WSL2 Configuration Check" -ForegroundColor Cyan
    Write-Host "================================" -ForegroundColor Cyan

    $issues = @()
    $optimizations = @()

    # Check if running as admin
    if (-not ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        Write-Host "❌ Not running as Administrator" -ForegroundColor Red
        return $false
    }

    Write-Host "✅ Running as Administrator" -ForegroundColor Green

    # Check WSL installation
    try {
        $null = wsl --status 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ WSL is installed and accessible" -ForegroundColor Green
        }
        else {
            throw "WSL not accessible"
        }
    }
    catch {
        Write-Host "❌ WSL is not properly installed" -ForegroundColor Red
        $issues += "WSL_NOT_INSTALLED"
        return $false
    }

    # Check Hyper-V features
    $hypervFeatures = @("Microsoft-Hyper-V-All", "VirtualMachinePlatform", "Microsoft-Windows-Subsystem-Linux")
    foreach ($feature in $hypervFeatures) {
        try {
            $status = Get-WindowsOptionalFeature -Online -FeatureName $feature -ErrorAction SilentlyContinue
            if ($status.State -eq "Enabled") {
                Write-Host "✅ $feature is enabled" -ForegroundColor Green
            }
            else {
                Write-Host "❌ $feature is $($status.State)" -ForegroundColor Red
                $issues += "HYPERV_FEATURE_DISABLED"
            }
        }
        catch {
            Write-Host "⚠️  Cannot check $feature status" -ForegroundColor Yellow
        }
    }

    # Check VMQ on WSL adapters
    $wslAdapters = Get-NetAdapter | Where-Object { $_.Name -like "*WSL*" -or $_.InterfaceDescription -like "*WSL*" }
    foreach ($adapter in $wslAdapters) {
        try {
            $vmq = Get-NetAdapterVmq -Name $adapter.Name -ErrorAction SilentlyContinue
            if ($vmq -and $vmq.Enabled) {
                Write-Host "⚠️  VMQ is enabled on WSL adapter: $($adapter.Name)" -ForegroundColor Yellow
                $optimizations += "DISABLE_VMQ"
            }
            else {
                Write-Host "✅ VMQ is disabled on WSL adapter: $($adapter.Name)" -ForegroundColor Green
            }
        }
        catch {
            Write-Host "⚠️  Cannot check VMQ status for $($adapter.Name)" -ForegroundColor Yellow
        }
    }

    # Check Windows Defender exclusions
    try {
        $preferences = Get-MpPreference
        $wslPaths = @(
            "$env:LOCALAPPDATA\Docker",
            "$env:LOCALAPPDATA\Packages\MicrosoftCorporationII.WindowsSubsystemForLinux_8wekyb3d8bbwe",
            "\\wsl.localhost"
        )

        $missingExclusions = 0
        foreach ($path in $wslPaths) {
            if ($preferences.ExclusionPath -notcontains $path) {
                $missingExclusions++
            }
        }

        if ($missingExclusions -gt 0) {
            Write-Host "⚠️  $missingExclusions Windows Defender exclusions are missing" -ForegroundColor Yellow
            $optimizations += "ADD_DEFENDER_EXCLUSIONS"
        }
        else {
            Write-Host "✅ Windows Defender exclusions are configured" -ForegroundColor Green
        }
    }
    catch {
        Write-Host "⚠️  Cannot check Windows Defender configuration" -ForegroundColor Yellow
    }

    # Check .wslconfig
    $wslConfigPath = "$env:USERPROFILE\.wslconfig"
    if (Test-Path $wslConfigPath) {
        $content = Get-Content $wslConfigPath -Raw
        if ($content -match "networkingMode\s*=\s*mirrored") {
            Write-Host "✅ .wslconfig has mirrored networking" -ForegroundColor Green
        }
        else {
            Write-Host "⚠️  .wslconfig exists but may need optimization" -ForegroundColor Yellow
            $optimizations += "UPDATE_WSLCONFIG"
        }
    }
    else {
        Write-Host "⚠️  No .wslconfig file found" -ForegroundColor Yellow
        $optimizations += "CREATE_WSLCONFIG"
    }

    Write-Host ""
    Write-Host "📊 Analysis Summary:" -ForegroundColor Cyan
    Write-Host "Critical Issues: $($issues.Count)" -ForegroundColor $(if ($issues.Count -gt 0) { "Red" } else { "Green" })
    Write-Host "Optimizations Available: $($optimizations.Count)" -ForegroundColor $(if ($optimizations.Count -gt 0) { "Yellow" } else { "Green" })

    return @{
        Issues = $issues
        Optimizations = $optimizations
        CanOptimize = $issues.Count -eq 0 -and $optimizations.Count -gt 0
    }
}

# Apply quick optimizations
function Apply-QuickOptimizations {
    param(
        [array]$Optimizations,
        [switch]$Auto
    )

    if ($Optimizations.Count -eq 0) {
        Write-Host "🎉 No optimizations needed - your system is already configured well!" -ForegroundColor Green
        return
    }

    Write-Host "🔧 Applying WSL2 Optimizations" -ForegroundColor Cyan
    Write-Host "==============================" -ForegroundColor Cyan

    # Disable VMQ if needed
    if ($Optimizations -contains "DISABLE_VMQ") {
        if ($Auto -or (Read-Host "Disable VMQ on WSL adapters? (y/N)").StartsWith('y')) {
            Write-Host "Disabling VMQ on WSL adapters..." -ForegroundColor Yellow

            $wslAdapters = Get-NetAdapter | Where-Object { $_.Name -like "*WSL*" -or $_.InterfaceDescription -like "*WSL*" }
            foreach ($adapter in $wslAdapters) {
                try {
                    Disable-NetAdapterVmq -Name $adapter.Name
                    Write-Host "✅ Disabled VMQ on $($adapter.Name)" -ForegroundColor Green
                }
                catch {
                    Write-Host "❌ Failed to disable VMQ on $($adapter.Name): $($_.Exception.Message)" -ForegroundColor Red
                }
            }
        }
    }

    # Add Defender exclusions if needed
    if ($Optimizations -contains "ADD_DEFENDER_EXCLUSIONS") {
        if ($Auto -or (Read-Host "Add Windows Defender exclusions for WSL? (y/N)").StartsWith('y')) {
            Write-Host "Adding Windows Defender exclusions..." -ForegroundColor Yellow

            $wslPaths = @(
                "$env:LOCALAPPDATA\Docker",
                "$env:LOCALAPPDATA\Packages\MicrosoftCorporationII.WindowsSubsystemForLinux_8wekyb3d8bbwe",
                "$env:USERPROFILE\AppData\Local\Temp\.X11-unix",
                "$env:PROGRAMDATA\Docker",
                "\\wsl.localhost",
                "\\wsl$"
            )

            $wslProcesses = @(
                "wsl.exe",
                "wslhost.exe",
                "wslservice.exe",
                "vmcompute.exe",
                "vmms.exe",
                "docker.exe"
            )

            foreach ($path in $wslPaths) {
                try {
                    Add-MpPreference -ExclusionPath $path
                    Write-Host "✅ Added path exclusion: $path" -ForegroundColor Green
                }
                catch {
                    Write-Host "⚠️  Path exclusion may already exist: $path" -ForegroundColor Yellow
                }
            }

            foreach ($process in $wslProcesses) {
                try {
                    Add-MpPreference -ExclusionProcess $process
                    Write-Host "✅ Added process exclusion: $process" -ForegroundColor Green
                }
                catch {
                    Write-Host "⚠️  Process exclusion may already exist: $process" -ForegroundColor Yellow
                }
            }
        }
    }

    # Create/update .wslconfig if needed
    if ($Optimizations -contains "CREATE_WSLCONFIG" -or $Optimizations -contains "UPDATE_WSLCONFIG") {
        if ($Auto -or (Read-Host "Create/update .wslconfig file with optimal settings? (y/N)").StartsWith('y')) {
            Write-Host "Creating optimized .wslconfig..." -ForegroundColor Yellow

            $wslConfig = @"
[wsl2]
# Network configuration
networkingMode=mirrored
localhostForwarding=true

# Memory and CPU allocation
memory=8GB
processors=4
swap=2GB

# Performance optimizations
pageReporting=false
guiApplications=true

# Kernel configuration
debugConsole=false
"@

            try {
                $wslConfig | Out-File -FilePath "$env:USERPROFILE\.wslconfig" -Encoding UTF8 -Force
                Write-Host "✅ Created optimized .wslconfig file" -ForegroundColor Green
                Write-Host "   Location: $env:USERPROFILE\.wslconfig" -ForegroundColor Gray
            }
            catch {
                Write-Host "❌ Failed to create .wslconfig: $($_.Exception.Message)" -ForegroundColor Red
            }
        }
    }
}

# Main execution
function Main {
    try {
        Write-Host "🚀 WSL2 Quick Optimization Tool for Windows 11" -ForegroundColor Magenta
        Write-Host "===============================================" -ForegroundColor Magenta
        Write-Host ""

        # Run analysis
        $analysis = Test-QuickConfiguration

        if ($analysis -eq $false) {
            Write-Host ""
            Write-Host "❌ Critical issues detected. Please address the following:" -ForegroundColor Red
            Write-Host "1. Ensure you're running as Administrator" -ForegroundColor Yellow
            Write-Host "2. Install WSL2 if not present: wsl --install" -ForegroundColor Yellow
            Write-Host "3. Enable required Windows features in 'Turn Windows features on or off'" -ForegroundColor Yellow
            return
        }

        if ($AnalyzeOnly) {
            Write-Host ""
            Write-Host "📋 Analysis complete. Use the full scripts for detailed optimization:" -ForegroundColor Cyan
            Write-Host "   .\wsl2-config-analyzer.ps1 - Detailed analysis" -ForegroundColor Gray
            Write-Host "   .\wsl2-hyperv-optimization.ps1 -Action Optimize - Full optimization" -ForegroundColor Gray
            return
        }

        # Apply optimizations if any are available
        if ($analysis.CanOptimize) {
            Write-Host ""
            Apply-QuickOptimizations -Optimizations $analysis.Optimizations -Auto:$AutoOptimize

            Write-Host ""
            Write-Host "✅ Optimization complete!" -ForegroundColor Green
            Write-Host ""
            Write-Host "📋 Next Steps:" -ForegroundColor Cyan
            Write-Host "1. Restart WSL: wsl --shutdown && wsl" -ForegroundColor Yellow
            Write-Host "2. Restart Docker Desktop if you use Docker" -ForegroundColor Yellow
            Write-Host "3. Test WSL performance: wsl -- uname -a" -ForegroundColor Yellow
            Write-Host ""
            Write-Host "🔍 For detailed monitoring, run:" -ForegroundColor Gray
            Write-Host "   .\wsl2-hyperv-optimization.ps1 -Action Monitor" -ForegroundColor Gray
        }
        else {
            Write-Host ""
            if ($analysis.Issues.Count -gt 0) {
                Write-Host "⚠️  Critical issues need to be resolved before optimization" -ForegroundColor Yellow
            }
            else {
                Write-Host "🎉 Your WSL2 configuration is already optimized!" -ForegroundColor Green
            }
        }

    }
    catch {
        Write-Host "Script failed: $($_.Exception.Message)" -ForegroundColor Red
        Write-Host $_.Exception.ToString() -ForegroundColor Red
        exit 1
    }
}

# Execute main function
Main