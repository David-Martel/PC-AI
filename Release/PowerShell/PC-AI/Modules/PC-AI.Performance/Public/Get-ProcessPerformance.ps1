#Requires -Version 5.1
function Get-ProcessPerformance {
    <#
    .SYNOPSIS
        Gets top processes sorted by CPU or memory usage.

    .DESCRIPTION
        Retrieves process information including CPU usage, memory consumption,
        process path, and owner. Useful for identifying resource-intensive
        processes and troubleshooting performance issues.

    .PARAMETER Top
        Number of top processes to return. Default is 10.

    .PARAMETER SortBy
        Sort processes by 'CPU', 'Memory', or 'Both'.
        'Both' returns two result sets: top by CPU and top by memory.
        Default is 'Both'.

    .PARAMETER IncludeSystemProcesses
        Include system processes (those without a main window and running as SYSTEM).
        By default, system processes are included.

    .PARAMETER ExcludeIdle
        Exclude the System Idle Process from results.

    .PARAMETER MinimumCpuPercent
        Only include processes using at least this percentage of CPU.
        Default is 0 (include all).

    .PARAMETER MinimumMemoryMB
        Only include processes using at least this amount of memory in MB.
        Default is 0 (include all).

    .EXAMPLE
        Get-ProcessPerformance
        Returns top 10 processes by both CPU and memory.

    .EXAMPLE
        Get-ProcessPerformance -Top 20 -SortBy Memory
        Returns top 20 processes sorted by memory usage.

    .EXAMPLE
        Get-ProcessPerformance -MinimumCpuPercent 5
        Returns processes using at least 5% CPU.

    .EXAMPLE
        Get-ProcessPerformance -SortBy CPU -Top 5 | Format-Table -AutoSize
        Shows top 5 CPU-consuming processes in a table.

    .OUTPUTS
        PSCustomObject with properties:
        - ProcessName: Name of the process
        - ProcessId: Process ID (PID)
        - CpuPercent: CPU usage percentage (approximate)
        - MemoryMB: Working set memory in MB
        - MemoryPercent: Percentage of total system memory used
        - ThreadCount: Number of threads
        - HandleCount: Number of handles
        - StartTime: When the process started
        - Path: Full path to the executable
        - Owner: Process owner (user account)
        - Description: Process description (if available)

    .NOTES
        Author: PC_AI Project
        Version: 1.0.0

        CPU percentage is calculated based on processor time since process start,
        which provides an average rather than instantaneous CPU usage.
        For real-time monitoring, use Watch-SystemResources.
    #>
    [CmdletBinding()]
    [OutputType([PSCustomObject[]])]
    param(
        [Parameter()]
        [ValidateRange(1, 1000)]
        [int]$Top = 10,

        [Parameter()]
        [ValidateSet('CPU', 'Memory', 'Both')]
        [string]$SortBy = 'Both',

        [Parameter()]
        [switch]$IncludeSystemProcesses = $true,

        [Parameter()]
        [switch]$ExcludeIdle,

        [Parameter()]
        [ValidateRange(0, 100)]
        [double]$MinimumCpuPercent = 0,

        [Parameter()]
        [ValidateRange(0, [int]::MaxValue)]
        [int]$MinimumMemoryMB = 0
    )

    begin {
        Write-Verbose "Getting process performance data (Top $Top by $SortBy)"

        # Get total physical memory for percentage calculation
        $totalMemory = (Get-CimInstance -ClassName Win32_ComputerSystem).TotalPhysicalMemory
        $processorCount = (Get-CimInstance -ClassName Win32_ComputerSystem).NumberOfLogicalProcessors

        # Get all processes with performance data
        $processData = [System.Collections.ArrayList]::new()
    }

    process {
        try {
            $processes = Get-Process -ErrorAction SilentlyContinue

            foreach ($proc in $processes) {
                try {
                    # Skip idle process if requested
                    if ($ExcludeIdle -and $proc.ProcessName -eq 'Idle') {
                        continue
                    }

                    # Calculate CPU percentage (average since process start)
                    $cpuPercent = 0
                    if ($proc.StartTime -and $proc.TotalProcessorTime) {
                        $runtime = (Get-Date) - $proc.StartTime
                        if ($runtime.TotalSeconds -gt 0) {
                            $cpuPercent = [math]::Round(
                                ($proc.TotalProcessorTime.TotalSeconds / $runtime.TotalSeconds / $processorCount) * 100,
                                2
                            )
                        }
                    }

                    # Calculate memory metrics
                    $memoryBytes = $proc.WorkingSet64
                    $memoryMB = [math]::Round($memoryBytes / 1MB, 2)
                    $memoryPercent = [math]::Round(($memoryBytes / $totalMemory) * 100, 2)

                    # Apply minimum filters
                    if ($cpuPercent -lt $MinimumCpuPercent -and $memoryMB -lt $MinimumMemoryMB) {
                        if ($MinimumCpuPercent -gt 0 -or $MinimumMemoryMB -gt 0) {
                            continue
                        }
                    }

                    # Get process path safely
                    $path = 'N/A'
                    try {
                        if ($proc.Path) {
                            $path = $proc.Path
                        }
                    }
                    catch {
                        # Access denied for some processes
                    }

                    # Get process description
                    $description = 'N/A'
                    try {
                        if ($proc.MainModule -and $proc.MainModule.FileVersionInfo) {
                            $desc = $proc.MainModule.FileVersionInfo.FileDescription
                            if ($desc) {
                                $description = $desc
                            }
                        }
                    }
                    catch {
                        # Access denied for some processes
                    }

                    # Get process owner
                    $owner = Get-ProcessOwner -ProcessId $proc.Id

                    # Get start time safely
                    $startTime = $null
                    try {
                        $startTime = $proc.StartTime
                    }
                    catch {
                        # Some processes don't expose start time
                    }

                    $procInfo = [PSCustomObject]@{
                        ProcessName   = $proc.ProcessName
                        ProcessId     = $proc.Id
                        CpuPercent    = $cpuPercent
                        MemoryMB      = $memoryMB
                        MemoryPercent = $memoryPercent
                        ThreadCount   = $proc.Threads.Count
                        HandleCount   = $proc.HandleCount
                        StartTime     = $startTime
                        Path          = $path
                        Owner         = $owner
                        Description   = $description
                        Priority      = $proc.PriorityClass
                    }

                    # Add custom type for formatting
                    $procInfo.PSObject.TypeNames.Insert(0, 'PC-AI.Performance.ProcessInfo')

                    [void]$processData.Add($procInfo)
                }
                catch {
                    Write-Verbose "Could not get info for process $($proc.ProcessName) ($($proc.Id)): $_"
                }
            }
        }
        catch {
            Write-Error "Failed to retrieve process information: $_"
        }
    }

    end {
        $results = @{}

        switch ($SortBy) {
            'CPU' {
                $results['TopByCPU'] = $processData |
                    Sort-Object CpuPercent -Descending |
                    Select-Object -First $Top
            }
            'Memory' {
                $results['TopByMemory'] = $processData |
                    Sort-Object MemoryMB -Descending |
                    Select-Object -First $Top
            }
            'Both' {
                $results['TopByCPU'] = $processData |
                    Sort-Object CpuPercent -Descending |
                    Select-Object -First $Top

                $results['TopByMemory'] = $processData |
                    Sort-Object MemoryMB -Descending |
                    Select-Object -First $Top
            }
        }

        # Calculate summary statistics
        $totalCpuUsage = ($processData | Measure-Object -Property CpuPercent -Sum).Sum
        $totalMemoryUsage = ($processData | Measure-Object -Property MemoryMB -Sum).Sum

        Write-Verbose "Total processes analyzed: $($processData.Count)"
        Write-Verbose "Total CPU usage (sum of averages): $([math]::Round($totalCpuUsage, 2))%"
        Write-Verbose "Total memory usage: $([math]::Round($totalMemoryUsage / 1024, 2)) GB"

        # Return results based on SortBy parameter
        if ($SortBy -eq 'Both') {
            # Return a combined object with both result sets
            [PSCustomObject]@{
                TopByCPU    = $results['TopByCPU']
                TopByMemory = $results['TopByMemory']
                Summary     = [PSCustomObject]@{
                    TotalProcesses = $processData.Count
                    TotalMemoryGB  = [math]::Round($totalMemoryUsage / 1024, 2)
                    Timestamp      = Get-Date
                }
            }
        }
        elseif ($SortBy -eq 'CPU') {
            return $results['TopByCPU']
        }
        else {
            return $results['TopByMemory']
        }
    }
}
