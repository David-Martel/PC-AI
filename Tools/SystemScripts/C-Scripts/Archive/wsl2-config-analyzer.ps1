#Requires -RunAsAdministrator

<#
.SYNOPSIS
    WSL2 Configuration Analysis and Issue Detection Script

.DESCRIPTION
    Analyzes current WSL2, Hyper-V, and Docker configuration to identify
    potential performance issues and configuration problems.

.EXAMPLE
    .\wsl2-config-analyzer.ps1

.NOTES
    Companion script to wsl2-hyperv-optimization.ps1
    Provides detailed analysis and recommendations
#>

# Configuration analysis functions
function Test-WSLConfigFile {
    Write-Host "=== WSL Configuration Analysis ===" -ForegroundColor Cyan

    $wslConfigPath = "$env:USERPROFILE\.wslconfig"
    $dockerDesktopConfigPath = "$env:USERPROFILE\.docker\daemon.json"

    $analysis = @{
        WSLConfigExists = Test-Path $wslConfigPath
        WSLConfigContent = $null
        DockerConfigExists = Test-Path $dockerDesktopConfigPath
        DockerConfigContent = $null
        Issues = @()
        Recommendations = @()
    }

    # Analyze .wslconfig
    if ($analysis.WSLConfigExists) {
        Write-Host "Found .wslconfig file" -ForegroundColor Green
        $analysis.WSLConfigContent = Get-Content $wslConfigPath -Raw
        Write-Host "Current .wslconfig content:" -ForegroundColor Yellow
        Write-Host $analysis.WSLConfigContent -ForegroundColor Gray

        # Check for mirrored networking
        if ($analysis.WSLConfigContent -match "networkingMode\s*=\s*mirrored") {
            Write-Host "✓ Mirrored networking mode detected" -ForegroundColor Green
        }
        elseif ($analysis.WSLConfigContent -match "networkingMode") {
            $analysis.Issues += "Non-mirrored networking mode may cause connectivity issues"
        }
        else {
            $analysis.Recommendations += "Consider adding 'networkingMode=mirrored' for improved networking"
        }

        # Check memory allocation
        if ($analysis.WSLConfigContent -match "memory\s*=\s*(\d+)([GMgm])") {
            $memoryValue = $matches[1]
            $memoryUnit = $matches[2].ToUpper()
            Write-Host "✓ Custom memory allocation: $memoryValue$memoryUnit" -ForegroundColor Green
        }
        else {
            $analysis.Recommendations += "Consider setting explicit memory limits for better performance"
        }

        # Check processor allocation
        if ($analysis.WSLConfigContent -match "processors\s*=\s*(\d+)") {
            $processors = $matches[1]
            Write-Host "✓ Custom processor allocation: $processors cores" -ForegroundColor Green
        }
        else {
            $analysis.Recommendations += "Consider setting explicit processor limits"
        }

    }
    else {
        Write-Host "No .wslconfig file found" -ForegroundColor Yellow
        $analysis.Recommendations += "Create .wslconfig file for optimal WSL2 configuration"
    }

    # Analyze Docker Desktop config
    if ($analysis.DockerConfigExists) {
        Write-Host "Found Docker Desktop configuration" -ForegroundColor Green
        try {
            $dockerConfig = Get-Content $dockerDesktopConfigPath | ConvertFrom-Json
            $analysis.DockerConfigContent = $dockerConfig

            # Check for experimental features
            if ($dockerConfig.experimental -eq $true) {
                Write-Host "✓ Docker experimental features enabled" -ForegroundColor Green
            }

            # Check for WSL integration
            if ($dockerConfig."use-wsl2-based-engine" -eq $true) {
                Write-Host "✓ WSL2-based Docker engine enabled" -ForegroundColor Green
            }
        }
        catch {
            $analysis.Issues += "Docker configuration file exists but cannot be parsed"
        }
    }

    return $analysis
}

function Test-HyperVConfiguration {
    Write-Host "`n=== Hyper-V Configuration Analysis ===" -ForegroundColor Cyan

    $analysis = @{
        Features = @{}
        Services = @{}
        Issues = @()
        Recommendations = @()
    }

    # Check Windows features
    $features = @(
        "Microsoft-Hyper-V-All",
        "VirtualMachinePlatform",
        "Microsoft-Windows-Subsystem-Linux",
        "HypervisorPlatform"
    )

    foreach ($feature in $features) {
        try {
            $status = Get-WindowsOptionalFeature -Online -FeatureName $feature -ErrorAction SilentlyContinue
            $analysis.Features[$feature] = $status.State

            if ($status.State -eq "Enabled") {
                Write-Host "✓ $feature: Enabled" -ForegroundColor Green
            }
            else {
                Write-Host "✗ $feature: $($status.State)" -ForegroundColor Red
                $analysis.Issues += "$feature is not enabled"
            }
        }
        catch {
            Write-Host "? $feature: Unknown" -ForegroundColor Yellow
            $analysis.Issues += "Cannot determine status of $feature"
        }
    }

    # Check Hyper-V services
    $services = @("vmms", "vmcompute", "vmickvpexchange", "vmicguestinterface")

    foreach ($serviceName in $services) {
        try {
            $service = Get-Service -Name $serviceName -ErrorAction SilentlyContinue
            if ($service) {
                $analysis.Services[$serviceName] = $service.Status

                if ($service.Status -eq "Running") {
                    Write-Host "✓ Service $serviceName: Running" -ForegroundColor Green
                }
                else {
                    Write-Host "⚠ Service $serviceName: $($service.Status)" -ForegroundColor Yellow
                    if ($serviceName -eq "vmms") {
                        $analysis.Issues += "Hyper-V Virtual Machine Management service is not running"
                    }
                }
            }
            else {
                Write-Host "✗ Service $serviceName: Not found" -ForegroundColor Red
                $analysis.Issues += "Service $serviceName is not installed"
            }
        }
        catch {
            $analysis.Issues += "Cannot check status of service $serviceName"
        }
    }

    return $analysis
}

function Test-NetworkConfiguration {
    Write-Host "`n=== Network Configuration Analysis ===" -ForegroundColor Cyan

    $analysis = @{
        Adapters = @()
        VMQStatus = @{}
        FirewallRules = @()
        Issues = @()
        Recommendations = @()
    }

    # Get all network adapters related to virtualization
    $adapters = Get-NetAdapter | Where-Object {
        $_.Name -like "*WSL*" -or
        $_.Name -like "*Docker*" -or
        $_.Name -like "*vEthernet*" -or
        $_.InterfaceDescription -like "*Hyper-V*"
    }

    foreach ($adapter in $adapters) {
        $adapterAnalysis = @{
            Name = $adapter.Name
            InterfaceDescription = $adapter.InterfaceDescription
            Status = $adapter.Status
            LinkSpeed = $adapter.LinkSpeed
            VMQ = $null
            Issues = @()
        }

        # Check VMQ status
        try {
            $vmq = Get-NetAdapterVmq -Name $adapter.Name -ErrorAction SilentlyContinue
            if ($vmq) {
                $adapterAnalysis.VMQ = @{
                    Enabled = $vmq.Enabled
                    BaseProcessorNumber = $vmq.BaseProcessorNumber
                    MaxProcessorNumber = $vmq.MaxProcessorNumber
                }

                if ($vmq.Enabled -and ($adapter.Name -like "*WSL*" -or $adapter.InterfaceDescription -like "*WSL*")) {
                    Write-Host "⚠ VMQ enabled on WSL adapter $($adapter.Name)" -ForegroundColor Yellow
                    $analysis.Issues += "VMQ is enabled on WSL adapter '$($adapter.Name)' which may cause network issues"
                    $analysis.Recommendations += "Disable VMQ on WSL adapter '$($adapter.Name)' using Disable-NetAdapterVmq"
                }
                else {
                    Write-Host "✓ VMQ settings for $($adapter.Name): Enabled=$($vmq.Enabled)" -ForegroundColor Green
                }
            }
        }
        catch {
            $adapterAnalysis.Issues += "Cannot retrieve VMQ settings"
        }

        # Check adapter status
        if ($adapter.Status -ne "Up") {
            Write-Host "⚠ Adapter $($adapter.Name) is $($adapter.Status)" -ForegroundColor Yellow
            $adapterAnalysis.Issues += "Adapter is not in Up status"
        }
        else {
            Write-Host "✓ Adapter $($adapter.Name): $($adapter.Status)" -ForegroundColor Green
        }

        $analysis.Adapters += $adapterAnalysis
    }

    # Check Windows Firewall rules for WSL
    try {
        $firewallRules = Get-NetFirewallRule | Where-Object {
            $_.DisplayName -like "*WSL*" -or
            $_.DisplayName -like "*Docker*" -or
            $_.DisplayName -like "*vEthernet*"
        }

        $analysis.FirewallRules = $firewallRules | ForEach-Object {
            @{
                DisplayName = $_.DisplayName
                Enabled = $_.Enabled
                Direction = $_.Direction
                Action = $_.Action
            }
        }

        Write-Host "Found $($analysis.FirewallRules.Count) relevant firewall rules" -ForegroundColor Green
    }
    catch {
        $analysis.Issues += "Cannot retrieve firewall rules"
    }

    return $analysis
}

function Test-WindowsDefenderConfiguration {
    Write-Host "`n=== Windows Defender Configuration Analysis ===" -ForegroundColor Cyan

    $analysis = @{
        ExclusionPaths = @()
        ExclusionProcesses = @()
        MissingExclusions = @()
        Issues = @()
        Recommendations = @()
    }

    try {
        $preferences = Get-MpPreference

        # Define expected WSL exclusions
        $expectedPaths = @(
            "$env:LOCALAPPDATA\Docker",
            "$env:LOCALAPPDATA\Packages\MicrosoftCorporationII.WindowsSubsystemForLinux_8wekyb3d8bbwe",
            "$env:USERPROFILE\AppData\Local\Temp\.X11-unix",
            "$env:PROGRAMDATA\Docker",
            "\\wsl.localhost",
            "\\wsl$"
        )

        $expectedProcesses = @(
            "wsl.exe",
            "wslhost.exe",
            "wslservice.exe",
            "vmcompute.exe",
            "vmms.exe",
            "docker.exe",
            "com.docker.backend.exe"
        )

        $analysis.ExclusionPaths = $preferences.ExclusionPath
        $analysis.ExclusionProcesses = $preferences.ExclusionProcess

        # Check for missing path exclusions
        foreach ($path in $expectedPaths) {
            if ($preferences.ExclusionPath -notcontains $path) {
                $analysis.MissingExclusions += "Path: $path"
                Write-Host "✗ Missing path exclusion: $path" -ForegroundColor Red
            }
            else {
                Write-Host "✓ Path exclusion present: $path" -ForegroundColor Green
            }
        }

        # Check for missing process exclusions
        foreach ($process in $expectedProcesses) {
            if ($preferences.ExclusionProcess -notcontains $process) {
                $analysis.MissingExclusions += "Process: $process"
                Write-Host "✗ Missing process exclusion: $process" -ForegroundColor Red
            }
            else {
                Write-Host "✓ Process exclusion present: $process" -ForegroundColor Green
            }
        }

        if ($analysis.MissingExclusions.Count -gt 0) {
            $analysis.Issues += "$($analysis.MissingExclusions.Count) Windows Defender exclusions are missing"
            $analysis.Recommendations += "Add missing Windows Defender exclusions to improve WSL2 performance"
        }

        # Check real-time protection status
        if ($preferences.DisableRealtimeMonitoring -eq $false) {
            Write-Host "✓ Real-time protection is enabled" -ForegroundColor Green
        }
        else {
            Write-Host "⚠ Real-time protection is disabled" -ForegroundColor Yellow
            $analysis.Issues += "Real-time protection is disabled"
        }

    }
    catch {
        $analysis.Issues += "Cannot retrieve Windows Defender configuration: $($_.Exception.Message)"
        Write-Host "✗ Cannot retrieve Windows Defender settings" -ForegroundColor Red
    }

    return $analysis
}

function Test-SystemPerformance {
    Write-Host "`n=== System Performance Analysis ===" -ForegroundColor Cyan

    $analysis = @{
        CPU = @{}
        Memory = @{}
        Disk = @{}
        Network = @{}
        Issues = @()
        Recommendations = @()
    }

    try {
        # CPU Analysis
        $cpu = Get-WmiObject -Class Win32_Processor
        $analysis.CPU = @{
            Name = $cpu.Name
            Cores = $cpu.NumberOfCores
            LogicalProcessors = $cpu.NumberOfLogicalProcessors
            VirtualizationEnabled = $cpu.VirtualizationFirmwareEnabled
            LoadPercentage = $cpu.LoadPercentage
        }

        Write-Host "CPU: $($cpu.Name)" -ForegroundColor Green
        Write-Host "Cores: $($cpu.NumberOfCores), Logical Processors: $($cpu.NumberOfLogicalProcessors)" -ForegroundColor Green

        if ($cpu.VirtualizationFirmwareEnabled) {
            Write-Host "✓ Hardware virtualization is enabled" -ForegroundColor Green
        }
        else {
            Write-Host "✗ Hardware virtualization is disabled" -ForegroundColor Red
            $analysis.Issues += "Hardware virtualization is not enabled in BIOS/UEFI"
        }

        # Memory Analysis
        $memory = Get-WmiObject -Class Win32_ComputerSystem
        $totalMemoryGB = [math]::Round($memory.TotalPhysicalMemory / 1GB, 2)
        $availableMemory = Get-Counter '\Memory\Available MBytes'
        $availableMemoryGB = [math]::Round($availableMemory.CounterSamples.CookedValue / 1024, 2)

        $analysis.Memory = @{
            TotalGB = $totalMemoryGB
            AvailableGB = $availableMemoryGB
            UsagePercent = [math]::Round(((($totalMemoryGB * 1024) - $availableMemory.CounterSamples.CookedValue) / ($totalMemoryGB * 1024)) * 100, 1)
        }

        Write-Host "Memory: $($totalMemoryGB)GB total, $($availableMemoryGB)GB available ($($analysis.Memory.UsagePercent)% used)" -ForegroundColor Green

        if ($totalMemoryGB -lt 8) {
            $analysis.Issues += "System has less than 8GB RAM, which may limit WSL2 performance"
            $analysis.Recommendations += "Consider upgrading system memory for better WSL2 performance"
        }

        if ($analysis.Memory.UsagePercent -gt 85) {
            $analysis.Issues += "High memory usage detected ($($analysis.Memory.UsagePercent)%)"
            $analysis.Recommendations += "Close unnecessary applications to free memory for WSL2"
        }

        # Disk Analysis (focusing on system drive)
        $systemDrive = Get-WmiObject -Class Win32_LogicalDisk | Where-Object { $_.DeviceID -eq $env:SystemDrive }
        $freeSpaceGB = [math]::Round($systemDrive.FreeSpace / 1GB, 2)
        $totalSpaceGB = [math]::Round($systemDrive.Size / 1GB, 2)
        $usagePercent = [math]::Round((($totalSpaceGB - $freeSpaceGB) / $totalSpaceGB) * 100, 1)

        $analysis.Disk = @{
            Drive = $systemDrive.DeviceID
            TotalGB = $totalSpaceGB
            FreeGB = $freeSpaceGB
            UsagePercent = $usagePercent
        }

        Write-Host "System Drive ($($systemDrive.DeviceID)): $($freeSpaceGB)GB free of $($totalSpaceGB)GB ($usagePercent% used)" -ForegroundColor Green

        if ($freeSpaceGB -lt 10) {
            $analysis.Issues += "Low disk space on system drive ($($freeSpaceGB)GB free)"
            $analysis.Recommendations += "Free up disk space on system drive for better WSL2 performance"
        }

    }
    catch {
        $analysis.Issues += "Cannot retrieve system performance data: $($_.Exception.Message)"
        Write-Host "✗ Cannot retrieve system performance information" -ForegroundColor Red
    }

    return $analysis
}

function Generate-OptimizationReport {
    param(
        [hashtable]$WSLConfig,
        [hashtable]$HyperVConfig,
        [hashtable]$NetworkConfig,
        [hashtable]$DefenderConfig,
        [hashtable]$Performance
    )

    Write-Host "`n`n=== WSL2 OPTIMIZATION RECOMMENDATIONS ===" -ForegroundColor Magenta
    Write-Host "=============================================" -ForegroundColor Magenta

    $allIssues = @()
    $allRecommendations = @()

    # Collect all issues and recommendations
    $allIssues += $WSLConfig.Issues
    $allIssues += $HyperVConfig.Issues
    $allIssues += $NetworkConfig.Issues
    $allIssues += $DefenderConfig.Issues
    $allIssues += $Performance.Issues

    $allRecommendations += $WSLConfig.Recommendations
    $allRecommendations += $HyperVConfig.Recommendations
    $allRecommendations += $NetworkConfig.Recommendations
    $allRecommendations += $DefenderConfig.Recommendations
    $allRecommendations += $Performance.Recommendations

    # Priority Issues
    Write-Host "`n🚨 CRITICAL ISSUES FOUND:" -ForegroundColor Red
    $criticalIssues = $allIssues | Where-Object { $_ -match "virtualization|VMQ|not enabled|not running" }
    if ($criticalIssues.Count -eq 0) {
        Write-Host "  ✓ No critical issues detected" -ForegroundColor Green
    }
    else {
        foreach ($issue in $criticalIssues) {
            Write-Host "  ✗ $issue" -ForegroundColor Red
        }
    }

    # All Issues
    if ($allIssues.Count -gt $criticalIssues.Count) {
        Write-Host "`n⚠️  OTHER ISSUES:" -ForegroundColor Yellow
        $otherIssues = $allIssues | Where-Object { $_ -notin $criticalIssues }
        foreach ($issue in $otherIssues) {
            Write-Host "  ⚠ $issue" -ForegroundColor Yellow
        }
    }

    # Recommendations
    Write-Host "`n💡 OPTIMIZATION RECOMMENDATIONS:" -ForegroundColor Cyan
    if ($allRecommendations.Count -eq 0) {
        Write-Host "  ✓ System appears to be well configured" -ForegroundColor Green
    }
    else {
        foreach ($i, $recommendation in $allRecommendations | ForEach-Object { $i = 0 } { @{ Index = ++$i; Value = $_ } }) {
            Write-Host "  $($i.Index). $($i.Value)" -ForegroundColor Cyan
        }
    }

    # Specific commands to run
    Write-Host "`n🔧 SUGGESTED COMMANDS TO RUN:" -ForegroundColor Green
    Write-Host ""

    # Check if optimization script should be run
    if ($NetworkConfig.Issues -contains "VMQ is enabled on WSL adapter" -or
        $DefenderConfig.MissingExclusions.Count -gt 0) {
        Write-Host "# Run the optimization script:" -ForegroundColor White
        Write-Host ".\wsl2-hyperv-optimization.ps1 -Action Optimize" -ForegroundColor Yellow
        Write-Host ""
    }

    # WSL config recommendations
    if (-not $WSLConfig.WSLConfigExists -or $WSLConfig.Recommendations.Count -gt 0) {
        Write-Host "# Create or update .wslconfig file:" -ForegroundColor White
        Write-Host '@"' -ForegroundColor Yellow
        Write-Host "[wsl2]" -ForegroundColor Yellow
        Write-Host "networkingMode=mirrored" -ForegroundColor Yellow
        Write-Host "memory=8GB" -ForegroundColor Yellow
        Write-Host "processors=4" -ForegroundColor Yellow
        Write-Host "swap=2GB" -ForegroundColor Yellow
        Write-Host "localhostForwarding=true" -ForegroundColor Yellow
        Write-Host '"@ | Out-File -FilePath "$env:USERPROFILE\.wslconfig" -Encoding UTF8' -ForegroundColor Yellow
        Write-Host ""
    }

    # Restart commands
    Write-Host "# Restart WSL and Docker after optimization:" -ForegroundColor White
    Write-Host "wsl --shutdown" -ForegroundColor Yellow
    Write-Host "# Wait 10 seconds, then restart Docker Desktop" -ForegroundColor Yellow
    Write-Host ""

    # Final validation
    Write-Host "# Validate changes:" -ForegroundColor White
    Write-Host ".\wsl2-hyperv-optimization.ps1 -Action Check" -ForegroundColor Yellow
    Write-Host ""

    # Summary
    $issueCount = $allIssues.Count
    $recommendationCount = $allRecommendations.Count

    Write-Host "=== ANALYSIS SUMMARY ===" -ForegroundColor Magenta
    Write-Host "Issues found: $issueCount" -ForegroundColor $(if ($issueCount -eq 0) { "Green" } elseif ($issueCount -lt 5) { "Yellow" } else { "Red" })
    Write-Host "Recommendations: $recommendationCount" -ForegroundColor $(if ($recommendationCount -eq 0) { "Green" } else { "Cyan" })

    if ($issueCount -eq 0 -and $recommendationCount -eq 0) {
        Write-Host "`n🎉 Your WSL2 configuration appears to be optimal!" -ForegroundColor Green
    }
    elseif ($issueCount -lt 3) {
        Write-Host "`n✅ Your WSL2 configuration is mostly good with minor improvements needed" -ForegroundColor Yellow
    }
    else {
        Write-Host "`n🔧 Your WSL2 configuration needs optimization for best performance" -ForegroundColor Red
    }
}

# Main execution
function Main {
    try {
        # Administrator check
        if (-not ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
            throw "This script must be run as Administrator for complete analysis."
        }

        Write-Host "WSL2 Configuration Analyzer" -ForegroundColor Magenta
        Write-Host "===========================" -ForegroundColor Magenta
        Write-Host "Analyzing your WSL2, Hyper-V, and Docker configuration..." -ForegroundColor Gray
        Write-Host ""

        # Run all analysis functions
        $wslConfig = Test-WSLConfigFile
        $hypervConfig = Test-HyperVConfiguration
        $networkConfig = Test-NetworkConfiguration
        $defenderConfig = Test-WindowsDefenderConfiguration
        $performance = Test-SystemPerformance

        # Generate comprehensive report
        Generate-OptimizationReport -WSLConfig $wslConfig -HyperVConfig $hypervConfig -NetworkConfig $networkConfig -DefenderConfig $defenderConfig -Performance $performance

    }
    catch {
        Write-Host "Analysis failed: $($_.Exception.Message)" -ForegroundColor Red
        Write-Host $_.Exception.ToString() -ForegroundColor Red
        exit 1
    }
}

# Execute main function
Main