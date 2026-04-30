#Requires -RunAsAdministrator

<#
.SYNOPSIS
    WSL2 Hyper-V and Network Optimization Script for Windows 11

.DESCRIPTION
    Comprehensive PowerShell script to optimize WSL2 performance through:
    - Windows Defender exclusions for WSL paths and processes
    - VMQ (Virtual Machine Queue) optimization for WSL virtual network adapter
    - Hyper-V and WSL network configuration validation
    - Performance monitoring and verification

.PARAMETER Action
    The action to perform: Optimize, Check, Rollback, or Monitor

.PARAMETER Force
    Skip confirmation prompts

.EXAMPLE
    .\wsl2-hyperv-optimization.ps1 -Action Check
    .\wsl2-hyperv-optimization.ps1 -Action Optimize -Force
    .\wsl2-hyperv-optimization.ps1 -Action Rollback

.NOTES
    Requires Administrator privileges
    Compatible with Windows 11 Build 26100+ and WSL2
    Created for mirrored networking mode with Docker Desktop integration
#>

param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("Optimize", "Check", "Rollback", "Monitor")]
    [string]$Action,

    [switch]$Force
)

# Configuration
$script:Config = @{
    WSLPaths = @(
        "$env:LOCALAPPDATA\Docker"
        "$env:LOCALAPPDATA\Packages\MicrosoftCorporationII.WindowsSubsystemForLinux_8wekyb3d8bbwe"
        "$env:USERPROFILE\AppData\Local\Temp\.X11-unix"
        "$env:PROGRAMDATA\Docker"
        "\\wsl.localhost"
        "\\wsl$"
    )

    WSLProcesses = @(
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

    HyperVFeatures = @(
        "Microsoft-Hyper-V-All"
        "VirtualMachinePlatform"
        "Microsoft-Windows-Subsystem-Linux"
    )

    LogFile = "$env:TEMP\WSL2-Optimization-$(Get-Date -Format 'yyyy-MM-dd-HH-mm-ss').log"
}

# Logging functions
function Write-Log {
    param(
        [string]$Message,
        [string]$Level = "INFO",
        [string]$Color = "White"
    )

    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logEntry = "[$timestamp] [$Level] $Message"

    Write-Host $logEntry -ForegroundColor $Color
    Add-Content -Path $script:Config.LogFile -Value $logEntry -Encoding UTF8
}

function Write-Success { param([string]$Message) Write-Log $Message "SUCCESS" "Green" }
function Write-Warning { param([string]$Message) Write-Log $Message "WARNING" "Yellow" }
function Write-Error { param([string]$Message) Write-Log $Message "ERROR" "Red" }

# Utility functions
function Test-Administrator {
    $currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($currentUser)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Test-WindowsVersion {
    $version = [System.Environment]::OSVersion.Version
    $build = (Get-ItemProperty "HKLM:SOFTWARE\Microsoft\Windows NT\CurrentVersion").CurrentBuild

    Write-Log "Windows Version: $($version.Major).$($version.Minor) Build $build"

    if ($version.Major -lt 10 -or [int]$build -lt 19041) {
        throw "This script requires Windows 10 Build 19041 (20H1) or later for WSL2 support"
    }

    return $true
}

function Test-WSL2Status {
    try {
        $wslStatus = wsl --status 2>$null
        if ($LASTEXITCODE -ne 0) {
            throw "WSL is not installed or not accessible"
        }

        Write-Log "WSL Status check passed"

        # Check for WSL2 distributions
        $distributions = wsl --list --verbose | Where-Object { $_ -match '\s+2\s+' }
        if (-not $distributions) {
            Write-Warning "No WSL2 distributions found. WSL2 may not be properly configured."
        }

        return $true
    }
    catch {
        throw "WSL2 is not properly installed or configured: $($_.Exception.Message)"
    }
}

function Get-HyperVStatus {
    Write-Log "Checking Hyper-V and virtualization features..."

    $features = @{}
    foreach ($feature in $script:Config.HyperVFeatures) {
        try {
            $status = Get-WindowsOptionalFeature -Online -FeatureName $feature -ErrorAction SilentlyContinue
            $features[$feature] = @{
                State = $status.State
                Enabled = $status.State -eq "Enabled"
            }
        }
        catch {
            $features[$feature] = @{
                State = "Unknown"
                Enabled = $false
            }
        }
    }

    # Check processor virtualization support
    $vmxSupport = (Get-WmiObject -Class Win32_Processor).VirtualizationFirmwareEnabled
    $features["ProcessorVirtualization"] = @{
        State = if ($vmxSupport) { "Enabled" } else { "Disabled" }
        Enabled = $vmxSupport
    }

    # Check Hyper-V service status
    try {
        $hvService = Get-Service -Name "vmms" -ErrorAction SilentlyContinue
        $features["HyperVService"] = @{
            State = $hvService.Status
            Enabled = $hvService.Status -eq "Running"
        }
    }
    catch {
        $features["HyperVService"] = @{
            State = "Not Found"
            Enabled = $false
        }
    }

    return $features
}

function Get-WSLNetworkConfiguration {
    Write-Log "Analyzing WSL network configuration..."

    $config = @{
        Adapters = @()
        VMQ = @{}
        NetworkMode = "Unknown"
    }

    # Get WSL network adapters
    $wslAdapters = Get-NetAdapter | Where-Object { $_.Name -like "*WSL*" -or $_.InterfaceDescription -like "*WSL*" }

    foreach ($adapter in $wslAdapters) {
        $adapterInfo = @{
            Name = $adapter.Name
            InterfaceDescription = $adapter.InterfaceDescription
            Status = $adapter.Status
            LinkSpeed = $adapter.LinkSpeed
            MacAddress = $adapter.MacAddress
        }

        # Check VMQ status
        try {
            $vmqSettings = Get-NetAdapterVmq -Name $adapter.Name -ErrorAction SilentlyContinue
            if ($vmqSettings) {
                $config.VMQ[$adapter.Name] = @{
                    Enabled = $vmqSettings.Enabled
                    BaseProcessorNumber = $vmqSettings.BaseProcessorNumber
                    MaxProcessorNumber = $vmqSettings.MaxProcessorNumber
                    NumaNode = $vmqSettings.NumaNode
                }
                $adapterInfo.VMQEnabled = $vmqSettings.Enabled
            }
        }
        catch {
            Write-Warning "Could not retrieve VMQ settings for adapter $($adapter.Name): $($_.Exception.Message)"
            $adapterInfo.VMQEnabled = "Unknown"
        }

        $config.Adapters += $adapterInfo
    }

    # Detect network mode from .wslconfig if it exists
    $wslConfigPath = "$env:USERPROFILE\.wslconfig"
    if (Test-Path $wslConfigPath) {
        $wslConfigContent = Get-Content $wslConfigPath -Raw
        if ($wslConfigContent -match "networkingMode\s*=\s*(\w+)") {
            $config.NetworkMode = $matches[1]
        }
    }

    return $config
}

function Get-DefenderExclusions {
    Write-Log "Checking current Windows Defender exclusions..."

    try {
        $preferences = Get-MpPreference
        return @{
            ExclusionPath = $preferences.ExclusionPath
            ExclusionProcess = $preferences.ExclusionProcess
            ExclusionExtension = $preferences.ExclusionExtension
        }
    }
    catch {
        Write-Error "Failed to retrieve Windows Defender preferences: $($_.Exception.Message)"
        return @{}
    }
}

function Add-DefenderExclusions {
    param([switch]$Force)

    Write-Log "Adding Windows Defender exclusions for WSL2..."

    if (-not $Force -and -not (Read-Host "Add Windows Defender exclusions? (y/N)").StartsWith('y')) {
        Write-Warning "Skipping Defender exclusions"
        return
    }

    $currentExclusions = Get-DefenderExclusions
    $addedPaths = @()
    $addedProcesses = @()

    # Add path exclusions
    foreach ($path in $script:Config.WSLPaths) {
        if ($currentExclusions.ExclusionPath -notcontains $path) {
            try {
                Add-MpPreference -ExclusionPath $path
                $addedPaths += $path
                Write-Success "Added path exclusion: $path"
            }
            catch {
                Write-Error "Failed to add path exclusion '$path': $($_.Exception.Message)"
            }
        }
        else {
            Write-Log "Path exclusion already exists: $path"
        }
    }

    # Add process exclusions
    foreach ($process in $script:Config.WSLProcesses) {
        if ($currentExclusions.ExclusionProcess -notcontains $process) {
            try {
                Add-MpPreference -ExclusionProcess $process
                $addedProcesses += $process
                Write-Success "Added process exclusion: $process"
            }
            catch {
                Write-Error "Failed to add process exclusion '$process': $($_.Exception.Message)"
            }
        }
        else {
            Write-Log "Process exclusion already exists: $process"
        }
    }

    if ($addedPaths.Count -gt 0 -or $addedProcesses.Count -gt 0) {
        Write-Success "Added $($addedPaths.Count) path exclusions and $($addedProcesses.Count) process exclusions"
    }
    else {
        Write-Log "No new exclusions were needed"
    }

    return @{
        AddedPaths = $addedPaths
        AddedProcesses = $addedProcesses
    }
}

function Disable-WSLAdapterVMQ {
    param([switch]$Force)

    Write-Log "Configuring VMQ settings for WSL network adapters..."

    $networkConfig = Get-WSLNetworkConfiguration
    $modifiedAdapters = @()

    foreach ($adapter in $networkConfig.Adapters) {
        if ($adapter.VMQEnabled -eq $true) {
            if (-not $Force -and -not (Read-Host "Disable VMQ for adapter '$($adapter.Name)'? (y/N)").StartsWith('y')) {
                Write-Warning "Skipping VMQ configuration for $($adapter.Name)"
                continue
            }

            try {
                Disable-NetAdapterVmq -Name $adapter.Name
                $modifiedAdapters += $adapter.Name
                Write-Success "Disabled VMQ for adapter: $($adapter.Name)"
            }
            catch {
                Write-Error "Failed to disable VMQ for adapter '$($adapter.Name)': $($_.Exception.Message)"
            }
        }
        elseif ($adapter.VMQEnabled -eq $false) {
            Write-Log "VMQ already disabled for adapter: $($adapter.Name)"
        }
        else {
            Write-Warning "VMQ status unknown for adapter: $($adapter.Name)"
        }
    }

    return $modifiedAdapters
}

function Enable-WSLAdapterVMQ {
    Write-Log "Re-enabling VMQ for WSL network adapters..."

    $networkConfig = Get-WSLNetworkConfiguration
    $modifiedAdapters = @()

    foreach ($adapter in $networkConfig.Adapters) {
        if ($adapter.VMQEnabled -eq $false) {
            try {
                Enable-NetAdapterVmq -Name $adapter.Name
                $modifiedAdapters += $adapter.Name
                Write-Success "Enabled VMQ for adapter: $($adapter.Name)"
            }
            catch {
                Write-Error "Failed to enable VMQ for adapter '$($adapter.Name)': $($_.Exception.Message)"
            }
        }
    }

    return $modifiedAdapters
}

function Remove-DefenderExclusions {
    Write-Log "Removing WSL-related Windows Defender exclusions..."

    $currentExclusions = Get-DefenderExclusions
    $removedPaths = @()
    $removedProcesses = @()

    # Remove path exclusions
    foreach ($path in $script:Config.WSLPaths) {
        if ($currentExclusions.ExclusionPath -contains $path) {
            try {
                Remove-MpPreference -ExclusionPath $path
                $removedPaths += $path
                Write-Success "Removed path exclusion: $path"
            }
            catch {
                Write-Error "Failed to remove path exclusion '$path': $($_.Exception.Message)"
            }
        }
    }

    # Remove process exclusions
    foreach ($process in $script:Config.WSLProcesses) {
        if ($currentExclusions.ExclusionProcess -contains $process) {
            try {
                Remove-MpPreference -ExclusionProcess $process
                $removedProcesses += $process
                Write-Success "Removed process exclusion: $process"
            }
            catch {
                Write-Error "Failed to remove process exclusion '$process': $($_.Exception.Message)"
            }
        }
    }

    return @{
        RemovedPaths = $removedPaths
        RemovedProcesses = $removedProcesses
    }
}

function Test-NetworkPerformance {
    Write-Log "Running network performance tests..."

    try {
        # Test WSL connectivity
        $wslTest = wsl -- ping -c 3 8.8.8.8 2>$null
        $wslConnectivity = $LASTEXITCODE -eq 0

        # Test Docker connectivity if available
        $dockerTest = $false
        try {
            $dockerStatus = docker version --format "{{.Server.Version}}" 2>$null
            $dockerTest = $LASTEXITCODE -eq 0
        }
        catch {
            # Docker not available
        }

        # Get network adapter statistics
        $networkStats = @{}
        $wslAdapters = Get-NetAdapter | Where-Object { $_.Name -like "*WSL*" }
        foreach ($adapter in $wslAdapters) {
            $stats = Get-NetAdapterStatistics -Name $adapter.Name
            $networkStats[$adapter.Name] = @{
                BytesReceived = $stats.ReceivedBytes
                BytesSent = $stats.SentBytes
                PacketsReceived = $stats.ReceivedUnicastPackets
                PacketsSent = $stats.SentUnicastPackets
            }
        }

        return @{
            WSLConnectivity = $wslConnectivity
            DockerConnectivity = $dockerTest
            NetworkStats = $networkStats
        }
    }
    catch {
        Write-Error "Network performance test failed: $($_.Exception.Message)"
        return $null
    }
}

function Show-SystemReport {
    Write-Log "=== WSL2 Hyper-V System Report ===" "INFO" "Cyan"
    Write-Log ""

    # System Information
    $version = [System.Environment]::OSVersion.Version
    $build = (Get-ItemProperty "HKLM:SOFTWARE\Microsoft\Windows NT\CurrentVersion").CurrentBuild
    Write-Log "Operating System: Windows $($version.Major).$($version.Minor) Build $build"

    # Hyper-V Status
    Write-Log ""
    Write-Log "=== Hyper-V and Virtualization Status ===" "INFO" "Cyan"
    $hypervStatus = Get-HyperVStatus
    foreach ($feature in $hypervStatus.GetEnumerator()) {
        $color = if ($feature.Value.Enabled) { "Green" } else { "Red" }
        Write-Log "$($feature.Key): $($feature.Value.State)" "INFO" $color
    }

    # WSL Network Configuration
    Write-Log ""
    Write-Log "=== WSL Network Configuration ===" "INFO" "Cyan"
    $networkConfig = Get-WSLNetworkConfiguration
    Write-Log "Network Mode: $($networkConfig.NetworkMode)"

    foreach ($adapter in $networkConfig.Adapters) {
        Write-Log ""
        Write-Log "Adapter: $($adapter.Name)" "INFO" "Yellow"
        Write-Log "  Description: $($adapter.InterfaceDescription)"
        Write-Log "  Status: $($adapter.Status)"
        Write-Log "  Link Speed: $($adapter.LinkSpeed)"
        Write-Log "  MAC Address: $($adapter.MacAddress)"

        $vmqColor = if ($adapter.VMQEnabled -eq $true) { "Red" } else { "Green" }
        Write-Log "  VMQ Enabled: $($adapter.VMQEnabled)" "INFO" $vmqColor
    }

    # Windows Defender Exclusions
    Write-Log ""
    Write-Log "=== Windows Defender Exclusions ===" "INFO" "Cyan"
    $exclusions = Get-DefenderExclusions

    $wslPathsCount = ($exclusions.ExclusionPath | Where-Object { $_ -in $script:Config.WSLPaths }).Count
    $wslProcessesCount = ($exclusions.ExclusionProcess | Where-Object { $_ -in $script:Config.WSLProcesses }).Count

    Write-Log "WSL Path Exclusions: $wslPathsCount / $($script:Config.WSLPaths.Count)"
    Write-Log "WSL Process Exclusions: $wslProcessesCount / $($script:Config.WSLProcesses.Count)"

    # Performance Test
    Write-Log ""
    Write-Log "=== Network Performance Test ===" "INFO" "Cyan"
    $perfTest = Test-NetworkPerformance
    if ($perfTest) {
        $wslColor = if ($perfTest.WSLConnectivity) { "Green" } else { "Red" }
        $dockerColor = if ($perfTest.DockerConnectivity) { "Green" } else { "Yellow" }

        Write-Log "WSL Connectivity: $($perfTest.WSLConnectivity)" "INFO" $wslColor
        Write-Log "Docker Connectivity: $($perfTest.DockerConnectivity)" "INFO" $dockerColor
    }

    Write-Log ""
    Write-Log "Log file: $($script:Config.LogFile)" "INFO" "Gray"
}

# Main action functions
function Invoke-OptimizeAction {
    param([switch]$Force)

    Write-Log "=== Starting WSL2 Hyper-V Optimization ===" "INFO" "Cyan"

    # Pre-optimization checks
    Test-WindowsVersion
    Test-WSL2Status

    $results = @{
        DefenderExclusions = $null
        VMQDisabled = $null
        Errors = @()
    }

    try {
        # Add Windows Defender exclusions
        $results.DefenderExclusions = Add-DefenderExclusions -Force:$Force
    }
    catch {
        $error = "Failed to configure Windows Defender exclusions: $($_.Exception.Message)"
        Write-Error $error
        $results.Errors += $error
    }

    try {
        # Disable VMQ for WSL adapters
        $results.VMQDisabled = Disable-WSLAdapterVMQ -Force:$Force
    }
    catch {
        $error = "Failed to configure VMQ settings: $($_.Exception.Message)"
        Write-Error $error
        $results.Errors += $error
    }

    # Show results
    Write-Log ""
    Write-Log "=== Optimization Results ===" "INFO" "Cyan"

    if ($results.DefenderExclusions) {
        Write-Log "Defender path exclusions added: $($results.DefenderExclusions.AddedPaths.Count)"
        Write-Log "Defender process exclusions added: $($results.DefenderExclusions.AddedProcesses.Count)"
    }

    if ($results.VMQDisabled) {
        Write-Log "VMQ disabled on adapters: $($results.VMQDisabled.Count)"
        foreach ($adapter in $results.VMQDisabled) {
            Write-Log "  - $adapter"
        }
    }

    if ($results.Errors.Count -gt 0) {
        Write-Warning "Optimization completed with $($results.Errors.Count) errors"
        Write-Log "Review the log file for details: $($script:Config.LogFile)"
    }
    else {
        Write-Success "Optimization completed successfully!"
        Write-Log ""
        Write-Log "RECOMMENDED NEXT STEPS:"
        Write-Log "1. Restart WSL2: wsl --shutdown && wsl"
        Write-Log "2. Restart Docker Desktop if using Docker"
        Write-Log "3. Run this script with -Action Monitor to verify improvements"
    }
}

function Invoke-CheckAction {
    Show-SystemReport
}

function Invoke-RollbackAction {
    Write-Log "=== Rolling Back WSL2 Hyper-V Optimizations ===" "INFO" "Cyan"

    $results = @{
        DefenderExclusionsRemoved = $null
        VMQEnabled = $null
        Errors = @()
    }

    try {
        # Remove Windows Defender exclusions
        $results.DefenderExclusionsRemoved = Remove-DefenderExclusions
    }
    catch {
        $error = "Failed to remove Windows Defender exclusions: $($_.Exception.Message)"
        Write-Error $error
        $results.Errors += $error
    }

    try {
        # Re-enable VMQ for WSL adapters
        $results.VMQEnabled = Enable-WSLAdapterVMQ
    }
    catch {
        $error = "Failed to re-enable VMQ: $($_.Exception.Message)"
        Write-Error $error
        $results.Errors += $error
    }

    # Show results
    Write-Log ""
    Write-Log "=== Rollback Results ===" "INFO" "Cyan"

    if ($results.DefenderExclusionsRemoved) {
        Write-Log "Defender path exclusions removed: $($results.DefenderExclusionsRemoved.RemovedPaths.Count)"
        Write-Log "Defender process exclusions removed: $($results.DefenderExclusionsRemoved.RemovedProcesses.Count)"
    }

    if ($results.VMQEnabled) {
        Write-Log "VMQ re-enabled on adapters: $($results.VMQEnabled.Count)"
        foreach ($adapter in $results.VMQEnabled) {
            Write-Log "  - $adapter"
        }
    }

    if ($results.Errors.Count -gt 0) {
        Write-Warning "Rollback completed with $($results.Errors.Count) errors"
    }
    else {
        Write-Success "Rollback completed successfully!"
    }
}

function Invoke-MonitorAction {
    Write-Log "=== WSL2 Performance Monitoring ===" "INFO" "Cyan"

    # Show current system status
    Show-SystemReport

    Write-Log ""
    Write-Log "=== Performance Monitoring (60 seconds) ===" "INFO" "Cyan"
    Write-Log "Monitoring network performance... Press Ctrl+C to stop early"

    $monitorStart = Get-Date
    $samples = @()

    for ($i = 0; $i -lt 12; $i++) {
        Start-Sleep 5

        $sample = @{
            Timestamp = Get-Date
            Performance = Test-NetworkPerformance
        }

        $samples += $sample

        Write-Host "." -NoNewline -ForegroundColor Yellow
    }

    Write-Host ""
    Write-Log "Monitoring completed"

    # Analyze results (basic implementation)
    if ($samples.Count -gt 0) {
        $avgTime = ($samples | Measure-Object { ($_.Timestamp - $monitorStart).TotalSeconds } -Average).Average
        Write-Log "Average sample time: $([math]::Round($avgTime, 2)) seconds"

        $successfulSamples = ($samples | Where-Object { $_.Performance.WSLConnectivity }).Count
        $successRate = ($successfulSamples / $samples.Count) * 100
        Write-Log "WSL connectivity success rate: $([math]::Round($successRate, 1))%"
    }
}

# Main execution
function Main {
    try {
        # Administrator check
        if (-not (Test-Administrator)) {
            throw "This script must be run as Administrator. Please restart PowerShell as Administrator and try again."
        }

        Write-Log "Starting WSL2 Hyper-V Optimization Script"
        Write-Log "Action: $Action"
        Write-Log "Log file: $($script:Config.LogFile)"
        Write-Log ""

        switch ($Action) {
            "Optimize" { Invoke-OptimizeAction -Force:$Force }
            "Check" { Invoke-CheckAction }
            "Rollback" { Invoke-RollbackAction }
            "Monitor" { Invoke-MonitorAction }
        }

        Write-Log ""
        Write-Log "Script completed successfully"

    }
    catch {
        Write-Error "Script failed: $($_.Exception.Message)"
        Write-Log "Full error details:" "ERROR"
        Write-Log $_.Exception.ToString() "ERROR"
        exit 1
    }
}

# Execute main function
Main