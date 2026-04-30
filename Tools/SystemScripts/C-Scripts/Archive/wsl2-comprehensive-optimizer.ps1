#Requires -RunAsAdministrator

<#
.SYNOPSIS
    Comprehensive WSL2 Hyper-V and Network Optimization Script

.DESCRIPTION
    Production-ready script for optimizing WSL2 performance on Windows 11 Build 26100+
    Features safe execution, rollback capabilities, and comprehensive validation.

.PARAMETER Action
    Optimize - Apply all optimizations
    Check - Analyze current configuration
    Rollback - Remove all optimizations
    Monitor - Performance monitoring
    Repair - Fix common issues

.PARAMETER Force
    Skip confirmation prompts

.PARAMETER Verbose
    Show detailed output

.EXAMPLE
    .\wsl2-comprehensive-optimizer.ps1 -Action Check -Verbose
    .\wsl2-comprehensive-optimizer.ps1 -Action Optimize -Force

.NOTES
    Designed for Windows 11 Build 26100.3323 with mirrored networking
    Compatible with Docker Desktop and Microsoft Dev Drive on T:
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("Optimize", "Check", "Rollback", "Monitor", "Repair")]
    [string]$Action,

    [switch]$Force,
    [switch]$Verbose
)

# Configuration for your specific system
$Config = @{
    SystemInfo = @{
        MinWindowsBuild = 26100
        RequiredFeatures = @(
            "Microsoft-Hyper-V-All"
            "VirtualMachinePlatform"
            "Microsoft-Windows-Subsystem-Linux"
        )
    }

    DefenderExclusions = @{
        Paths = @(
            "$env:LOCALAPPDATA\Docker"
            "$env:LOCALAPPDATA\Packages\MicrosoftCorporationII.WindowsSubsystemForLinux_8wekyb3d8bbwe"
            "$env:USERPROFILE\AppData\Local\Temp\.X11-unix"
            "$env:PROGRAMDATA\Docker"
            "T:\vm"  # Your swap file location
            "\\wsl.localhost"
            "\\wsl$"
        )
        Processes = @(
            "wsl.exe"
            "wslhost.exe"
            "wslservice.exe"
            "vmcompute.exe"
            "vmms.exe"
            "docker.exe"
            "com.docker.backend.exe"
            "com.docker.service"
            "com.docker.proxy.exe"
        )
    }

    NetworkOptimizations = @{
        DisableVMQOnWSL = $true
        EnableLargeReceiveOffload = $false
        EnableTcpChimneyOffload = $false
    }

    LogPath = "$env:TEMP\WSL2-Optimization-$(Get-Date -Format 'yyyy-MM-dd-HH-mm-ss').log"
}

# Enhanced logging
function Write-Log {
    param(
        [string]$Message,
        [ValidateSet("INFO", "SUCCESS", "WARNING", "ERROR", "DEBUG")]
        [string]$Level = "INFO",
        [ConsoleColor]$Color = "White"
    )

    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logEntry = "[$timestamp] [$Level] $Message"

    if ($Level -eq "DEBUG" -and -not $Verbose) {
        # Skip debug messages unless verbose mode
        Add-Content -Path $Config.LogPath -Value $logEntry -Encoding UTF8
        return
    }

    Write-Host $logEntry -ForegroundColor $Color
    Add-Content -Path $Config.LogPath -Value $logEntry -Encoding UTF8
}

# System validation functions
function Test-Prerequisites {
    Write-Log "Validating system prerequisites..." "INFO" "Cyan"

    # Admin check
    $currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($currentUser)
    if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        throw "This script must be run as Administrator"
    }
    Write-Log "✓ Running as Administrator" "SUCCESS" "Green"

    # Windows version check
    $build = (Get-ItemProperty "HKLM:SOFTWARE\Microsoft\Windows NT\CurrentVersion").CurrentBuild
    if ([int]$build -lt $Config.SystemInfo.MinWindowsBuild) {
        throw "This script requires Windows Build $($Config.SystemInfo.MinWindowsBuild) or later. Current: $build"
    }
    Write-Log "✓ Windows Build $build is supported" "SUCCESS" "Green"

    # WSL check
    try {
        $wslStatus = wsl --status 2>$null
        if ($LASTEXITCODE -ne 0) {
            throw "WSL is not installed or accessible"
        }
        Write-Log "✓ WSL is accessible" "SUCCESS" "Green"
    }
    catch {
        throw "WSL validation failed: $($_.Exception.Message)"
    }

    # Feature check
    foreach ($feature in $Config.SystemInfo.RequiredFeatures) {
        try {
            $featureStatus = Get-WindowsOptionalFeature -Online -FeatureName $feature -ErrorAction SilentlyContinue
            if ($featureStatus.State -ne "Enabled") {
                Write-Log "✗ $feature is $($featureStatus.State)" "WARNING" "Yellow"
            }
            else {
                Write-Log "✓ $feature is enabled" "SUCCESS" "Green"
            }
        }
        catch {
            Write-Log "? Cannot check $feature status" "WARNING" "Yellow"
        }
    }

    return $true
}

function Get-WSLNetworkAdapters {
    Write-Log "Detecting WSL network adapters..." "DEBUG"

    $adapters = @()

    try {
        # Get all network adapters and filter for WSL/Hyper-V related ones
        $allAdapters = Get-NetAdapter -ErrorAction SilentlyContinue

        foreach ($adapter in $allAdapters) {
            if ($adapter.Name -like "*WSL*" -or
                $adapter.Name -like "*vEthernet*" -or
                $adapter.InterfaceDescription -like "*WSL*" -or
                $adapter.InterfaceDescription -like "*Hyper-V Virtual Ethernet*") {

                $adapterInfo = @{
                    Name = $adapter.Name
                    InterfaceDescription = $adapter.InterfaceDescription
                    Status = $adapter.Status
                    LinkSpeed = $adapter.LinkSpeed
                    VMQ = $null
                }

                # Get VMQ status safely
                try {
                    $vmqStatus = Get-NetAdapterVmq -Name $adapter.Name -ErrorAction SilentlyContinue
                    if ($vmqStatus) {
                        $adapterInfo.VMQ = @{
                            Enabled = $vmqStatus.Enabled
                            BaseProcessorNumber = $vmqStatus.BaseProcessorNumber
                            MaxProcessorNumber = $vmqStatus.MaxProcessorNumber
                        }
                    }
                }
                catch {
                    Write-Log "Cannot get VMQ status for $($adapter.Name): $($_.Exception.Message)" "DEBUG"
                }

                $adapters += $adapterInfo
            }
        }
    }
    catch {
        Write-Log "Error detecting network adapters: $($_.Exception.Message)" "ERROR" "Red"
    }

    return $adapters
}

function Test-DefenderConfiguration {
    Write-Log "Analyzing Windows Defender configuration..." "INFO" "Cyan"

    try {
        $mpPreference = Get-MpPreference -ErrorAction Stop

        $analysis = @{
            CurrentPaths = $mpPreference.ExclusionPath
            CurrentProcesses = $mpPreference.ExclusionProcess
            MissingPaths = @()
            MissingProcesses = @()
            RealtimeProtection = -not $mpPreference.DisableRealtimeMonitoring
        }

        # Check for missing path exclusions
        foreach ($path in $Config.DefenderExclusions.Paths) {
            if ($mpPreference.ExclusionPath -notcontains $path) {
                $analysis.MissingPaths += $path
                Write-Log "Missing path exclusion: $path" "WARNING" "Yellow"
            }
            else {
                Write-Log "✓ Path exclusion present: $path" "DEBUG"
            }
        }

        # Check for missing process exclusions
        foreach ($process in $Config.DefenderExclusions.Processes) {
            if ($mpPreference.ExclusionProcess -notcontains $process) {
                $analysis.MissingProcesses += $process
                Write-Log "Missing process exclusion: $process" "WARNING" "Yellow"
            }
            else {
                Write-Log "✓ Process exclusion present: $process" "DEBUG"
            }
        }

        Write-Log "Defender analysis complete. Missing: $($analysis.MissingPaths.Count) paths, $($analysis.MissingProcesses.Count) processes" "INFO"

        return $analysis
    }
    catch {
        Write-Log "Cannot access Windows Defender configuration: $($_.Exception.Message)" "ERROR" "Red"
        return $null
    }
}

function Apply-DefenderOptimizations {
    param([hashtable]$DefenderAnalysis, [switch]$Force)

    if (-not $DefenderAnalysis) {
        Write-Log "Skipping Defender optimizations - analysis failed" "WARNING" "Yellow"
        return @{ Applied = $false; Reason = "Analysis failed" }
    }

    $totalMissing = $DefenderAnalysis.MissingPaths.Count + $DefenderAnalysis.MissingProcesses.Count

    if ($totalMissing -eq 0) {
        Write-Log "✓ All Windows Defender exclusions are already configured" "SUCCESS" "Green"
        return @{ Applied = $false; Reason = "Already optimized" }
    }

    if (-not $Force) {
        $response = Read-Host "Add $totalMissing Windows Defender exclusions for WSL? (y/N)"
        if ($response -notlike "y*") {
            Write-Log "Skipping Defender exclusions by user choice" "INFO"
            return @{ Applied = $false; Reason = "User declined" }
        }
    }

    $results = @{
        Applied = $true
        AddedPaths = @()
        AddedProcesses = @()
        Errors = @()
    }

    # Add path exclusions
    foreach ($path in $DefenderAnalysis.MissingPaths) {
        try {
            Add-MpPreference -ExclusionPath $path -ErrorAction Stop
            $results.AddedPaths += $path
            Write-Log "✓ Added path exclusion: $path" "SUCCESS" "Green"
        }
        catch {
            $error = "Failed to add path exclusion '$path': $($_.Exception.Message)"
            $results.Errors += $error
            Write-Log $error "ERROR" "Red"
        }
    }

    # Add process exclusions
    foreach ($process in $DefenderAnalysis.MissingProcesses) {
        try {
            Add-MpPreference -ExclusionProcess $process -ErrorAction Stop
            $results.AddedProcesses += $process
            Write-Log "✓ Added process exclusion: $process" "SUCCESS" "Green"
        }
        catch {
            $error = "Failed to add process exclusion '$process': $($_.Exception.Message)"
            $results.Errors += $error
            Write-Log $error "ERROR" "Red"
        }
    }

    Write-Log "Defender optimization complete. Added: $($results.AddedPaths.Count) paths, $($results.AddedProcesses.Count) processes" "INFO"

    return $results
}

function Apply-NetworkOptimizations {
    param([array]$NetworkAdapters, [switch]$Force)

    Write-Log "Applying network optimizations..." "INFO" "Cyan"

    $results = @{
        Applied = $false
        ModifiedAdapters = @()
        Errors = @()
    }

    $wslAdapters = $NetworkAdapters | Where-Object {
        $_.Name -like "*WSL*" -or $_.InterfaceDescription -like "*WSL*"
    }

    if ($wslAdapters.Count -eq 0) {
        Write-Log "No WSL network adapters found for optimization" "WARNING" "Yellow"
        return $results
    }

    foreach ($adapter in $wslAdapters) {
        if ($adapter.VMQ -and $adapter.VMQ.Enabled) {
            if (-not $Force) {
                $response = Read-Host "Disable VMQ for adapter '$($adapter.Name)'? (y/N)"
                if ($response -notlike "y*") {
                    Write-Log "Skipping VMQ optimization for $($adapter.Name) by user choice" "INFO"
                    continue
                }
            }

            try {
                Disable-NetAdapterVmq -Name $adapter.Name -ErrorAction Stop
                $results.ModifiedAdapters += $adapter.Name
                Write-Log "✓ Disabled VMQ for adapter: $($adapter.Name)" "SUCCESS" "Green"
                $results.Applied = $true
            }
            catch {
                $error = "Failed to disable VMQ for '$($adapter.Name)': $($_.Exception.Message)"
                $results.Errors += $error
                Write-Log $error "ERROR" "Red"
            }
        }
        else {
            Write-Log "✓ VMQ already optimized for adapter: $($adapter.Name)" "SUCCESS" "Green"
        }
    }

    return $results
}

function Test-SystemPerformance {
    Write-Log "Running system performance tests..." "INFO" "Cyan"

    $results = @{
        WSLConnectivity = $false
        WSLPerformance = $null
        NetworkLatency = $null
        Issues = @()
    }

    # Test WSL basic connectivity
    try {
        $wslTest = wsl -- echo "test" 2>$null
        $results.WSLConnectivity = $LASTEXITCODE -eq 0

        if ($results.WSLConnectivity) {
            Write-Log "✓ WSL connectivity test passed" "SUCCESS" "Green"
        }
        else {
            $results.Issues += "WSL connectivity test failed"
            Write-Log "✗ WSL connectivity test failed" "ERROR" "Red"
        }
    }
    catch {
        $results.Issues += "WSL connectivity test error: $($_.Exception.Message)"
        Write-Log "✗ WSL connectivity test error" "ERROR" "Red"
    }

    # Test network performance
    try {
        $pingStart = Get-Date
        $pingResult = wsl -- ping -c 3 8.8.8.8 2>$null
        $pingEnd = Get-Date

        if ($LASTEXITCODE -eq 0) {
            $results.NetworkLatency = ($pingEnd - $pingStart).TotalMilliseconds
            Write-Log "✓ Network latency test: $($results.NetworkLatency)ms" "SUCCESS" "Green"
        }
        else {
            $results.Issues += "Network latency test failed"
            Write-Log "✗ Network latency test failed" "ERROR" "Red"
        }
    }
    catch {
        $results.Issues += "Network test error: $($_.Exception.Message)"
        Write-Log "✗ Network test error" "ERROR" "Red"
    }

    return $results
}

function Show-ConfigurationReport {
    Write-Log "`n=== WSL2 CONFIGURATION REPORT ===" "INFO" "Magenta"
    Write-Log "=================================" "INFO" "Magenta"

    # System Information
    $version = [System.Environment]::OSVersion.Version
    $build = (Get-ItemProperty "HKLM:SOFTWARE\Microsoft\Windows NT\CurrentVersion").CurrentBuild
    Write-Log "System: Windows $($version.Major).$($version.Minor) Build $build" "INFO"

    # WSL Information
    Write-Log "`n--- WSL Status ---" "INFO" "Cyan"
    try {
        $wslVersion = (wsl --version 2>$null) -join " "
        Write-Log "WSL Version: $wslVersion" "INFO"

        $distributions = wsl --list --verbose 2>$null
        Write-Log "Active Distributions:" "INFO"
        $distributions | ForEach-Object {
            if ($_ -and $_ -notmatch "NAME|STATE|VERSION|----|^$") {
                Write-Log "  $_" "INFO" "Gray"
            }
        }
    }
    catch {
        Write-Log "Cannot retrieve WSL information" "ERROR" "Red"
    }

    # Network Adapter Status
    Write-Log "`n--- Network Adapters ---" "INFO" "Cyan"
    $adapters = Get-WSLNetworkAdapters
    if ($adapters.Count -gt 0) {
        foreach ($adapter in $adapters) {
            Write-Log "Adapter: $($adapter.Name)" "INFO" "Yellow"
            Write-Log "  Status: $($adapter.Status)" "INFO"
            Write-Log "  Speed: $($adapter.LinkSpeed)" "INFO"

            if ($adapter.VMQ) {
                $vmqColor = if ($adapter.VMQ.Enabled) { "Red" } else { "Green" }
                $vmqSymbol = if ($adapter.VMQ.Enabled) { "⚠" } else { "✓" }
                Write-Log "  $vmqSymbol VMQ Enabled: $($adapter.VMQ.Enabled)" "INFO" $vmqColor
            }
            else {
                Write-Log "  ? VMQ Status: Unknown" "INFO" "Yellow"
            }
        }
    }
    else {
        Write-Log "No WSL network adapters detected" "WARNING" "Yellow"
    }

    # Windows Defender Status
    Write-Log "`n--- Windows Defender Status ---" "INFO" "Cyan"
    $defenderAnalysis = Test-DefenderConfiguration
    if ($defenderAnalysis) {
        $totalPaths = $Config.DefenderExclusions.Paths.Count
        $totalProcesses = $Config.DefenderExclusions.Processes.Count
        $missingPaths = $defenderAnalysis.MissingPaths.Count
        $missingProcesses = $defenderAnalysis.MissingProcesses.Count

        Write-Log "Path Exclusions: $($totalPaths - $missingPaths)/$totalPaths configured" "INFO"
        Write-Log "Process Exclusions: $($totalProcesses - $missingProcesses)/$totalProcesses configured" "INFO"

        if ($missingPaths -gt 0 -or $missingProcesses -gt 0) {
            Write-Log "⚠ $($missingPaths + $missingProcesses) exclusions missing" "WARNING" "Yellow"
        }
        else {
            Write-Log "✓ All exclusions configured" "SUCCESS" "Green"
        }
    }

    # Performance Test
    Write-Log "`n--- Performance Test ---" "INFO" "Cyan"
    $perfTest = Test-SystemPerformance

    if ($perfTest.WSLConnectivity) {
        Write-Log "✓ WSL Connectivity: Working" "SUCCESS" "Green"
    }
    else {
        Write-Log "✗ WSL Connectivity: Failed" "ERROR" "Red"
    }

    if ($perfTest.NetworkLatency) {
        $latencyColor = if ($perfTest.NetworkLatency -lt 100) { "Green" } elseif ($perfTest.NetworkLatency -lt 500) { "Yellow" } else { "Red" }
        Write-Log "Network Latency: $($perfTest.NetworkLatency)ms" "INFO" $latencyColor
    }

    # Summary
    Write-Log "`n--- Optimization Summary ---" "INFO" "Cyan"
    $issueCount = $perfTest.Issues.Count
    $optimizationCount = 0

    if ($defenderAnalysis) {
        $optimizationCount += $defenderAnalysis.MissingPaths.Count + $defenderAnalysis.MissingProcesses.Count
    }

    $adapters | ForEach-Object {
        if ($_.VMQ -and $_.VMQ.Enabled) {
            $optimizationCount++
        }
    }

    if ($issueCount -eq 0 -and $optimizationCount -eq 0) {
        Write-Log "🎉 System is fully optimized!" "SUCCESS" "Green"
    }
    elseif ($optimizationCount -gt 0) {
        Write-Log "🔧 $optimizationCount optimizations available" "INFO" "Yellow"
    }

    if ($issueCount -gt 0) {
        Write-Log "⚠ $issueCount issues detected" "WARNING" "Red"
    }
}

function Invoke-OptimizeAction {
    Write-Log "Starting WSL2 optimization process..." "INFO" "Cyan"

    Test-Prerequisites

    $results = @{
        DefenderOptimization = $null
        NetworkOptimization = $null
        OverallSuccess = $false
    }

    # Apply Defender optimizations
    $defenderAnalysis = Test-DefenderConfiguration
    if ($defenderAnalysis) {
        $results.DefenderOptimization = Apply-DefenderOptimizations -DefenderAnalysis $defenderAnalysis -Force:$Force
    }

    # Apply network optimizations
    $networkAdapters = Get-WSLNetworkAdapters
    if ($networkAdapters.Count -gt 0) {
        $results.NetworkOptimization = Apply-NetworkOptimizations -NetworkAdapters $networkAdapters -Force:$Force
    }

    # Determine overall success
    $errors = @()
    if ($results.DefenderOptimization -and $results.DefenderOptimization.Errors) {
        $errors += $results.DefenderOptimization.Errors
    }
    if ($results.NetworkOptimization -and $results.NetworkOptimization.Errors) {
        $errors += $results.NetworkOptimization.Errors
    }

    $results.OverallSuccess = $errors.Count -eq 0

    # Show summary
    Write-Log "`n=== OPTIMIZATION COMPLETE ===" "INFO" "Magenta"

    if ($results.OverallSuccess) {
        Write-Log "✓ All optimizations applied successfully!" "SUCCESS" "Green"
        Write-Log "`nNext steps:" "INFO" "Cyan"
        Write-Log "1. Restart WSL: wsl --shutdown && wsl" "INFO" "Yellow"
        Write-Log "2. Restart Docker Desktop if using Docker" "INFO" "Yellow"
        Write-Log "3. Test performance: .\wsl2-comprehensive-optimizer.ps1 -Action Monitor" "INFO" "Yellow"
    }
    else {
        Write-Log "⚠ Optimization completed with $($errors.Count) errors" "WARNING" "Yellow"
        Write-Log "Check the log file for details: $($Config.LogPath)" "INFO"
    }

    return $results
}

function Invoke-RollbackAction {
    Write-Log "Starting WSL2 optimization rollback..." "INFO" "Cyan"

    Test-Prerequisites

    # Rollback Defender exclusions
    $defenderAnalysis = Test-DefenderConfiguration
    if ($defenderAnalysis) {
        $pathsToRemove = $Config.DefenderExclusions.Paths | Where-Object { $defenderAnalysis.CurrentPaths -contains $_ }
        $processesToRemove = $Config.DefenderExclusions.Processes | Where-Object { $defenderAnalysis.CurrentProcesses -contains $_ }

        foreach ($path in $pathsToRemove) {
            try {
                Remove-MpPreference -ExclusionPath $path
                Write-Log "✓ Removed path exclusion: $path" "SUCCESS" "Green"
            }
            catch {
                Write-Log "Failed to remove path exclusion '$path': $($_.Exception.Message)" "ERROR" "Red"
            }
        }

        foreach ($process in $processesToRemove) {
            try {
                Remove-MpPreference -ExclusionProcess $process
                Write-Log "✓ Removed process exclusion: $process" "SUCCESS" "Green"
            }
            catch {
                Write-Log "Failed to remove process exclusion '$process': $($_.Exception.Message)" "ERROR" "Red"
            }
        }
    }

    # Re-enable VMQ
    $networkAdapters = Get-WSLNetworkAdapters
    $wslAdapters = $networkAdapters | Where-Object {
        $_.Name -like "*WSL*" -or $_.InterfaceDescription -like "*WSL*"
    }

    foreach ($adapter in $wslAdapters) {
        if ($adapter.VMQ -and -not $adapter.VMQ.Enabled) {
            try {
                Enable-NetAdapterVmq -Name $adapter.Name
                Write-Log "✓ Re-enabled VMQ for adapter: $($adapter.Name)" "SUCCESS" "Green"
            }
            catch {
                Write-Log "Failed to re-enable VMQ for '$($adapter.Name)': $($_.Exception.Message)" "ERROR" "Red"
            }
        }
    }

    Write-Log "Rollback complete" "SUCCESS" "Green"
}

function Invoke-RepairAction {
    Write-Log "Starting WSL2 repair process..." "INFO" "Cyan"

    # Restart WSL
    Write-Log "Restarting WSL..." "INFO" "Yellow"
    try {
        wsl --shutdown
        Start-Sleep 3
        $wslTest = wsl -- echo "WSL restart successful"
        if ($LASTEXITCODE -eq 0) {
            Write-Log "✓ WSL restart successful" "SUCCESS" "Green"
        }
    }
    catch {
        Write-Log "WSL restart failed: $($_.Exception.Message)" "ERROR" "Red"
    }

    # Reset network adapters
    Write-Log "Resetting WSL network adapters..." "INFO" "Yellow"
    try {
        $adapters = Get-WSLNetworkAdapters
        foreach ($adapter in $adapters) {
            if ($adapter.Status -ne "Up") {
                try {
                    Restart-NetAdapter -Name $adapter.Name
                    Write-Log "✓ Reset adapter: $($adapter.Name)" "SUCCESS" "Green"
                }
                catch {
                    Write-Log "Failed to reset adapter '$($adapter.Name)': $($_.Exception.Message)" "ERROR" "Red"
                }
            }
        }
    }
    catch {
        Write-Log "Network adapter reset failed: $($_.Exception.Message)" "ERROR" "Red"
    }

    Write-Log "Repair process complete" "SUCCESS" "Green"
}

function Invoke-MonitorAction {
    Write-Log "Starting WSL2 performance monitoring..." "INFO" "Cyan"

    Write-Log "Collecting baseline performance data..." "INFO"
    $baseline = Test-SystemPerformance

    Write-Log "Monitoring for 60 seconds... Press Ctrl+C to stop early" "INFO" "Yellow"

    $samples = @()
    $monitorStart = Get-Date

    for ($i = 0; $i -lt 12; $i++) {
        Start-Sleep 5

        $sample = @{
            Timestamp = Get-Date
            Performance = Test-SystemPerformance
        }

        $samples += $sample
        Write-Host "." -NoNewline -ForegroundColor Yellow
    }

    Write-Host ""

    # Analyze results
    $successfulTests = ($samples | Where-Object { $_.Performance.WSLConnectivity }).Count
    $successRate = if ($samples.Count -gt 0) { ($successfulTests / $samples.Count) * 100 } else { 0 }

    $avgLatency = if ($samples.Count -gt 0) {
        ($samples | Where-Object { $_.Performance.NetworkLatency } | Measure-Object -Property { $_.Performance.NetworkLatency } -Average).Average
    } else { 0 }

    Write-Log "`n=== MONITORING RESULTS ===" "INFO" "Magenta"
    Write-Log "Monitoring Duration: 60 seconds" "INFO"
    Write-Log "Samples Collected: $($samples.Count)" "INFO"
    Write-Log "Success Rate: $([math]::Round($successRate, 1))%" "INFO" $(if ($successRate -gt 95) { "Green" } elseif ($successRate -gt 80) { "Yellow" } else { "Red" })

    if ($avgLatency -gt 0) {
        Write-Log "Average Latency: $([math]::Round($avgLatency, 1))ms" "INFO" $(if ($avgLatency -lt 100) { "Green" } elseif ($avgLatency -lt 500) { "Yellow" } else { "Red" })
    }
}

# Main execution
try {
    Write-Log "WSL2 Comprehensive Optimizer v2.0" "INFO" "Magenta"
    Write-Log "=================================" "INFO" "Magenta"
    Write-Log "Action: $Action" "INFO"
    Write-Log "Log file: $($Config.LogPath)" "INFO" "Gray"
    Write-Log ""

    switch ($Action) {
        "Check" {
            Show-ConfigurationReport
        }
        "Optimize" {
            Invoke-OptimizeAction
        }
        "Rollback" {
            Invoke-RollbackAction
        }
        "Monitor" {
            Invoke-MonitorAction
        }
        "Repair" {
            Invoke-RepairAction
        }
    }

    Write-Log "`nScript completed successfully" "SUCCESS" "Green"
    Write-Log "Full log available at: $($Config.LogPath)" "INFO" "Gray"
}
catch {
    Write-Log "Script failed: $($_.Exception.Message)" "ERROR" "Red"
    Write-Log "Stack trace:" "ERROR"
    Write-Log $_.Exception.ToString() "ERROR"

    Write-Log "Log file: $($Config.LogPath)" "INFO" "Gray"
    exit 1
}