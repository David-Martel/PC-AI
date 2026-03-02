#Requires -Version 5.1
function Watch-SystemResources {
    <#
    .SYNOPSIS
        Real-time monitoring of CPU, memory, and disk I/O.

    .DESCRIPTION
        Provides continuous real-time monitoring of system resources with
        color-coded output based on utilization thresholds. Useful for
        observing system behavior during testing or troubleshooting.

    .PARAMETER RefreshInterval
        Time in seconds between updates. Default is 2 seconds.
        Minimum is 1 second.

    .PARAMETER Duration
        Total duration to monitor in seconds. If not specified,
        monitoring continues until Ctrl+C is pressed.

    .PARAMETER IncludeTopProcesses
        Include top CPU and memory consuming processes in each update.

    .PARAMETER TopProcessCount
        Number of top processes to show. Default is 3.

    .PARAMETER OutputMode
        Output mode: 'Console' for live display, 'Object' for collectible output.
        Default is 'Console'.

    .PARAMETER WarningThreshold
        Percentage threshold for warning color (yellow). Default is 75.

    .PARAMETER CriticalThreshold
        Percentage threshold for critical color (red). Default is 90.

    .EXAMPLE
        Watch-SystemResources
        Starts real-time monitoring with default settings (Ctrl+C to stop).

    .EXAMPLE
        Watch-SystemResources -Duration 60 -RefreshInterval 1
        Monitors for 60 seconds with 1-second refresh rate.

    .EXAMPLE
        Watch-SystemResources -IncludeTopProcesses -TopProcessCount 5
        Shows top 5 processes by CPU/memory with each update.

    .EXAMPLE
        Watch-SystemResources -Duration 30 -OutputMode Object | Export-Csv metrics.csv
        Collects 30 seconds of metrics and exports to CSV.

    .OUTPUTS
        When OutputMode is 'Object', returns PSCustomObject with:
        - Timestamp: When the sample was taken
        - CpuPercent: Overall CPU usage percentage
        - MemoryPercent: Memory usage percentage
        - MemoryUsedGB: Memory used in GB
        - MemoryTotalGB: Total memory in GB
        - DiskReadMBps: Disk read speed in MB/s
        - DiskWriteMBps: Disk write speed in MB/s
        - NetworkInMbps: Network receive in Mbps
        - NetworkOutMbps: Network send in Mbps

    .NOTES
        Author: PC_AI Project
        Version: 1.0.0

        Press Ctrl+C to stop monitoring at any time.
        Console mode uses color coding:
        - Green: Normal (below warning threshold)
        - Yellow: Warning (above warning threshold)
        - Red: Critical (above critical threshold)
    #>
    [CmdletBinding()]
    [OutputType([PSCustomObject[]])]
    param(
        [Parameter()]
        [ValidateRange(1, 3600)]
        [int]$RefreshInterval = 2,

        [Parameter()]
        [ValidateRange(1, 86400)]
        [int]$Duration,

        [Parameter()]
        [switch]$IncludeTopProcesses,

        [Parameter()]
        [ValidateRange(1, 20)]
        [int]$TopProcessCount = 3,

        [Parameter()]
        [ValidateSet('Console', 'Object')]
        [string]$OutputMode = 'Console',

        [Parameter()]
        [ValidateRange(1, 99)]
        [int]$WarningThreshold = 75,

        [Parameter()]
        [ValidateRange(1, 99)]
        [int]$CriticalThreshold = 90
    )

    begin {
        Write-Verbose "Starting system resource monitoring (Refresh: ${RefreshInterval}s)"

        # Get system info
        $computerInfo = Get-CimInstance -ClassName Win32_ComputerSystem
        $totalMemoryGB = [math]::Round($computerInfo.TotalPhysicalMemory / 1GB, 2)
        $processorCount = $computerInfo.NumberOfLogicalProcessors

        # Initialize metrics collection
        $metrics = [System.Collections.ArrayList]::new()

        # Calculate end time if duration specified
        $endTime = if ($Duration) {
            (Get-Date).AddSeconds($Duration)
        }
        else {
            [datetime]::MaxValue
        }

        # Helper function to get color based on value
        function Get-ThresholdColor {
            param([double]$Value)

            if ($Value -ge $CriticalThreshold) {
                return [System.ConsoleColor]::Red
            }
            elseif ($Value -ge $WarningThreshold) {
                return [System.ConsoleColor]::Yellow
            }
            else {
                return [System.ConsoleColor]::Green
            }
        }

        # Helper function to create progress bar
        function Get-ProgressBar {
            param(
                [double]$Percent,
                [int]$Width = 20
            )

            $filledWidth = [math]::Floor($Percent / 100 * $Width)
            $emptyWidth = $Width - $filledWidth

            $filled = [string]::new([char]0x2588, $filledWidth)
            $empty = [string]::new([char]0x2591, $emptyWidth)

            return "$filled$empty"
        }
    }

    process {
        $iteration = 0
        $running = $true

        try {
            while ($running -and (Get-Date) -lt $endTime) {
                $iteration++
                $timestamp = Get-Date

                # Get CPU usage
                $cpuCounter = Get-Counter -Counter '\Processor(_Total)\% Processor Time' -ErrorAction SilentlyContinue
                $cpuPercent = if ($cpuCounter) {
                    [math]::Round($cpuCounter.CounterSamples[0].CookedValue, 1)
                }
                else { 0 }

                # Get memory usage
                $os = Get-CimInstance -ClassName Win32_OperatingSystem
                $memoryUsedBytes = $os.TotalVisibleMemorySize * 1KB - $os.FreePhysicalMemory * 1KB
                $memoryTotalBytes = $os.TotalVisibleMemorySize * 1KB
                $memoryPercent = [math]::Round(($memoryUsedBytes / $memoryTotalBytes) * 100, 1)
                $memoryUsedGB = [math]::Round($memoryUsedBytes / 1GB, 2)

                # Get disk I/O
                $diskIO = Get-DiskIOCounters
                $diskReadMBps = [math]::Round($diskIO.ReadBytesPerSec / 1MB, 2)
                $diskWriteMBps = [math]::Round($diskIO.WriteBytesPerSec / 1MB, 2)

                # Get network I/O
                $networkIn = 0
                $networkOut = 0
                try {
                    $networkCounters = Get-Counter -Counter @(
                        '\Network Interface(*)\Bytes Received/sec',
                        '\Network Interface(*)\Bytes Sent/sec'
                    ) -ErrorAction SilentlyContinue

                    if ($networkCounters) {
                        $networkIn = ($networkCounters.CounterSamples |
                            Where-Object { $_.Path -like '*Received*' -and $_.InstanceName -ne '_total' } |
                            Measure-Object -Property CookedValue -Sum).Sum

                        $networkOut = ($networkCounters.CounterSamples |
                            Where-Object { $_.Path -like '*Sent*' -and $_.InstanceName -ne '_total' } |
                            Measure-Object -Property CookedValue -Sum).Sum
                    }
                }
                catch {
                    Write-Verbose "Could not get network counters"
                }

                $networkInMbps = [math]::Round(($networkIn * 8) / 1MB, 2)
                $networkOutMbps = [math]::Round(($networkOut * 8) / 1MB, 2)

                # Create metrics object
                $sample = [PSCustomObject]@{
                    Timestamp      = $timestamp
                    Iteration      = $iteration
                    CpuPercent     = $cpuPercent
                    MemoryPercent  = $memoryPercent
                    MemoryUsedGB   = $memoryUsedGB
                    MemoryTotalGB  = $totalMemoryGB
                    DiskReadMBps   = $diskReadMBps
                    DiskWriteMBps  = $diskWriteMBps
                    NetworkInMbps  = $networkInMbps
                    NetworkOutMbps = $networkOutMbps
                }

                [void]$metrics.Add($sample)

                # Output based on mode
                if ($OutputMode -eq 'Console') {
                    # Clear screen for refresh (only after first iteration)
                    if ($iteration -gt 1) {
                        Clear-Host
                    }

                    # Header
                    Write-Host "`n====== System Resource Monitor ======" -ForegroundColor Cyan
                    Write-Host "Time: $($timestamp.ToString('HH:mm:ss'))  |  Refresh: ${RefreshInterval}s  |  Press Ctrl+C to stop`n" -ForegroundColor Gray

                    # CPU
                    $cpuColor = Get-ThresholdColor -Value $cpuPercent
                    $cpuBar = Get-ProgressBar -Percent $cpuPercent
                    Write-Host "CPU:    " -NoNewline
                    Write-Host "$cpuBar " -ForegroundColor $cpuColor -NoNewline
                    Write-Host "$($cpuPercent.ToString('F1').PadLeft(5))%" -ForegroundColor $cpuColor

                    # Memory
                    $memColor = Get-ThresholdColor -Value $memoryPercent
                    $memBar = Get-ProgressBar -Percent $memoryPercent
                    Write-Host "Memory: " -NoNewline
                    Write-Host "$memBar " -ForegroundColor $memColor -NoNewline
                    Write-Host "$($memoryPercent.ToString('F1').PadLeft(5))% " -ForegroundColor $memColor -NoNewline
                    Write-Host "($memoryUsedGB / $totalMemoryGB GB)" -ForegroundColor Gray

                    # Disk I/O
                    Write-Host "`nDisk I/O:" -ForegroundColor White
                    Write-Host "  Read:  $($diskReadMBps.ToString('F2').PadLeft(8)) MB/s" -ForegroundColor $(if ($diskReadMBps -gt 100) { 'Yellow' } else { 'Green' })
                    Write-Host "  Write: $($diskWriteMBps.ToString('F2').PadLeft(8)) MB/s" -ForegroundColor $(if ($diskWriteMBps -gt 100) { 'Yellow' } else { 'Green' })

                    # Network I/O
                    Write-Host "`nNetwork:" -ForegroundColor White
                    Write-Host "  In:  $($networkInMbps.ToString('F2').PadLeft(8)) Mbps" -ForegroundColor Cyan
                    Write-Host "  Out: $($networkOutMbps.ToString('F2').PadLeft(8)) Mbps" -ForegroundColor Cyan

                    # Top processes if requested
                    if ($IncludeTopProcesses) {
                        Write-Host "`n------ Top Processes by CPU ------" -ForegroundColor Yellow

                        $topCpu = Get-Process |
                            Where-Object { $_.ProcessName -ne 'Idle' } |
                            Sort-Object CPU -Descending |
                            Select-Object -First $TopProcessCount

                        foreach ($proc in $topCpu) {
                            $procCpu = if ($proc.CPU) { $proc.CPU.ToString('F1') } else { '0.0' }
                            Write-Host "  $($proc.ProcessName.PadRight(25)) CPU: $($procCpu.PadLeft(8))s  Mem: $([math]::Round($proc.WorkingSet64 / 1MB, 0).ToString().PadLeft(6)) MB" -ForegroundColor Gray
                        }

                        Write-Host "`n------ Top Processes by Memory ------" -ForegroundColor Yellow

                        $topMem = Get-Process |
                            Sort-Object WorkingSet64 -Descending |
                            Select-Object -First $TopProcessCount

                        foreach ($proc in $topMem) {
                            $memMB = [math]::Round($proc.WorkingSet64 / 1MB, 0)
                            Write-Host "  $($proc.ProcessName.PadRight(25)) Mem: $($memMB.ToString().PadLeft(6)) MB" -ForegroundColor Gray
                        }
                    }

                    # Duration countdown
                    if ($Duration) {
                        $remaining = [math]::Max(0, ($endTime - (Get-Date)).TotalSeconds)
                        Write-Host "`nRemaining: $([math]::Round($remaining, 0)) seconds" -ForegroundColor DarkGray
                    }
                }

                # Wait for next interval
                if ((Get-Date) -lt $endTime) {
                    Start-Sleep -Seconds $RefreshInterval
                }
            }
        }
        catch [System.Management.Automation.PipelineStoppedException] {
            # User pressed Ctrl+C
            Write-Host "`n`nMonitoring stopped by user." -ForegroundColor Yellow
            $running = $false
        }
        catch {
            Write-Error "Monitoring error: $_"
            $running = $false
        }
    }

    end {
        Write-Verbose "Monitoring complete. Collected $($metrics.Count) samples."

        if ($OutputMode -eq 'Object') {
            return $metrics
        }
        elseif ($metrics.Count -gt 0) {
            # Show summary statistics in console mode
            Write-Host "`n====== Session Summary ======" -ForegroundColor Cyan
            Write-Host "Duration: $($metrics.Count * $RefreshInterval) seconds ($($metrics.Count) samples)" -ForegroundColor Gray

            $avgCpu = ($metrics | Measure-Object -Property CpuPercent -Average).Average
            $maxCpu = ($metrics | Measure-Object -Property CpuPercent -Maximum).Maximum
            $avgMem = ($metrics | Measure-Object -Property MemoryPercent -Average).Average
            $maxMem = ($metrics | Measure-Object -Property MemoryPercent -Maximum).Maximum

            Write-Host "`nCPU:    Avg: $([math]::Round($avgCpu, 1))%  Max: $([math]::Round($maxCpu, 1))%"
            Write-Host "Memory: Avg: $([math]::Round($avgMem, 1))%  Max: $([math]::Round($maxMem, 1))%"
        }
    }
}
