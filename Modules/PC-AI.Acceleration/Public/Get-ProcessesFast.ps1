#Requires -Version 5.1
<#
.SYNOPSIS
    Fast process listing using procs

.DESCRIPTION
    Lists processes using native PCAI performance APIs when available,
    then procs (Rust process viewer) when available, with fallback to
    Get-Process. Procs provides better formatting and tree view output.

.PARAMETER Name
    Filter by process name

.PARAMETER SortBy
    Sort by: cpu, mem, pid, name, user

.PARAMETER Top
    Show only top N processes

.PARAMETER Tree
    Show process tree view

.PARAMETER Watch
    Enable watch mode (continuous refresh)

.EXAMPLE
    Get-ProcessesFast -SortBy cpu -Top 10
    Shows top 10 CPU-consuming processes

.EXAMPLE
    Get-ProcessesFast -Name "chrome" -Tree
    Shows Chrome processes in tree view

.OUTPUTS
    PSCustomObject[] with process information
#>
function Get-ProcessesFast {
    [CmdletBinding()]
    [OutputType([PSCustomObject[]])]
    param(
        [Parameter(Position = 0)]
        [string]$Name,

        [Parameter()]
        [ValidateSet('cpu', 'mem', 'pid', 'name', 'user', 'read', 'write')]
        [string]$SortBy = 'cpu',

        [Parameter()]
        [int]$Top = 0,

        [Parameter()]
        [switch]$Tree,

        [Parameter()]
        [switch]$Watch,

        [Parameter()]
        [switch]$RawOutput
    )

    $preferRustCli = $env:PCAI_PREFER_RUST_CLI -eq '1'
    $pcaiPerfPath = Get-PcaiPerfToolPath
    $procsPath = Get-RustToolPath -ToolName 'procs'
    $useProcs = $null -ne $procsPath -and (Test-Path $procsPath)
    $nativeType = ([System.Management.Automation.PSTypeName]'PcaiNative.PerformanceModule').Type

    if ($preferRustCli -and $pcaiPerfPath -and -not $RawOutput -and -not $Tree -and -not $Watch) {
        $rustSupportedSort = $SortBy -in @('cpu', 'mem')
        if (-not $Name -and $Top -gt 0 -and $rustSupportedSort) {
            $rustResults = Get-ProcessesWithPcaiPerf -Top $Top -SortBy $SortBy -ToolPath $pcaiPerfPath
            if ($null -ne $rustResults) {
                return $rustResults
            }
        }
    }

    if ($nativeType -and [PcaiNative.PcaiCore]::IsAvailable -and -not $RawOutput -and -not $Tree -and -not $Watch) {
        $nativeSupportedSort = $SortBy -in @('cpu', 'mem')
        if (-not $Name -and $Top -gt 0 -and $nativeSupportedSort) {
            $nativeResults = Get-ProcessesWithNative -Top $Top -SortBy $SortBy
            if ($null -ne $nativeResults) {
                return $nativeResults
            }
        }
    }

    if ($useProcs -and $RawOutput) {
        return Get-ProcessesWithProcs @PSBoundParameters -ProcsPath $procsPath
    }
    else {
        # For structured output, use .NET parallel processing
        return Get-ProcessesParallel @PSBoundParameters
    }
}

function Get-ProcessesWithPcaiPerf {
    [CmdletBinding()]
    param(
        [int]$Top,
        [string]$SortBy,
        [Parameter(Mandatory)]
        [string]$ToolPath
    )

    try {
        $sortKey = if ($SortBy -eq 'cpu') { 'cpu' } else { 'memory' }
        $json = & $ToolPath 'processes' '--top' $Top '--sort-by' $sortKey 2>$null
        if (-not $json) {
            return $null
        }

        return @(ConvertFrom-Json -InputObject $json)
    } catch {
        return $null
    }
}

function Convert-PcaiNativeProcessRows {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [object[]]$Rows,
        [Parameter(Mandatory)]
        [scriptblock]$Mapper
    )

    $results = [System.Collections.Generic.List[object]]::new()
    foreach ($row in $Rows) {
        if ($null -eq $row) {
            continue
        }
        $results.Add((& $Mapper $row))
    }

    return @($results)
}

<#
.SYNOPSIS
    Retrieve top processes using the native PCAI performance module.

.PARAMETER Top
    Number of processes to return.

.PARAMETER SortBy
    Sort order (cpu or mem).
#>
function Get-ProcessesWithNative {
    [CmdletBinding()]
    param(
        [int]$Top,
        [string]$SortBy
    )

    try {
        $sortKey = if ($SortBy -eq 'cpu') { 'cpu' } else { 'memory' }
        $preferCompact = $env:PCAI_PREFER_COMPACT_TRANSPORT -eq '1'
        $performanceType = ([System.Management.Automation.PSTypeName]'PcaiNative.PerformanceModule').Type
        $typedMethod = if ($performanceType) {
            $performanceType.GetMethod('GetTopProcesses', [System.Reflection.BindingFlags]'Public,Static', $null, [Type[]]@([uint32], [string]), $null)
        } else {
            $null
        }
        $jsonMethod = if ($performanceType) {
            $performanceType.GetMethod('GetTopProcessesJson', [System.Reflection.BindingFlags]'Public,Static', $null, [Type[]]@([uint32], [string]), $null)
        } else {
            $null
        }

        $json = $null
        if ($jsonMethod -and -not $preferCompact) {
            $json = [PcaiNative.PerformanceModule]::GetTopProcessesJson([uint32]$Top, $sortKey)
        }
        if (-not $json) {
            $typedResult = $null
            if ($typedMethod) {
                $typedResult = [PcaiNative.PerformanceModule]::GetTopProcesses([uint32]$Top, $sortKey)
            }

            if ($typedResult -and $typedResult.Processes) {
                return @(Convert-PcaiNativeProcessRows -Rows @($typedResult.Processes) -Mapper {
                        param($row)
                        [PSCustomObject]@{
                            PID       = $row.Pid
                            Name      = $row.Name
                            CPU       = [Math]::Round([double]$row.CpuUsage, 2)
                            MemoryMB  = [Math]::Round([double]$row.MemoryBytes / 1MB, 2)
                            Threads   = $null
                            Handles   = $null
                            Owner     = $null
                            Path      = $row.ExecutablePath
                            StartTime = $null
                            Status    = $row.Status
                            Tool      = 'pcai_native'
                        }
                    })
            }

            if ($jsonMethod -and $preferCompact) {
                $json = [PcaiNative.PerformanceModule]::GetTopProcessesJson([uint32]$Top, $sortKey)
            }
        }

        if (-not $json) {
            return $null
        }

        $result = ConvertFrom-Json -InputObject $json
        if (-not $result.processes) {
            return @()
        }

        return @(Convert-PcaiNativeProcessRows -Rows @($result.processes) -Mapper {
                param($row)
                [PSCustomObject]@{
                    PID       = $row.pid
                    Name      = $row.name
                    CPU       = [Math]::Round([double]$row.cpu_usage, 2)
                    MemoryMB  = [Math]::Round([double]$row.memory_bytes / 1MB, 2)
                    Threads   = $null
                    Handles   = $null
                    Owner     = $null
                    Path      = $row.exe_path
                    StartTime = $null
                    Status    = $row.status
                    Tool      = 'pcai_native'
                }
            })
    }
    catch {
        return $null
    }
}

function Get-ProcessesWithProcs {
    [CmdletBinding()]
    param(
        [string]$Name,
        [string]$SortBy,
        [int]$Top,
        [switch]$Tree,
        [switch]$Watch,
        [switch]$RawOutput,
        [string]$ProcsPath
    )

    $args = @()

    # Sort
    if ($SortBy) {
        $args += '--sortd'
        $args += $SortBy
    }

    # Tree view
    if ($Tree) {
        $args += '--tree'
    }

    # Watch mode
    if ($Watch) {
        $args += '--watch'
    }

    # Name filter
    if ($Name) {
        $args += $Name
    }

    try {
        $output = & $ProcsPath @args 2>&1

        if ($Top -gt 0) {
            # Return top N lines (plus header)
            return ($output | Select-Object -First ($Top + 1)) -join "`n"
        }

        return $output -join "`n"
    }
    catch {
        Write-Warning "procs failed: $_"
        return Get-ProcessesParallel -Name $Name -SortBy $SortBy -Top $Top
    }
}

function Get-ProcessesParallel {
    [CmdletBinding()]
    param(
        [string]$Name,
        [string]$SortBy,
        [int]$Top,
        [switch]$Tree,
        [switch]$Watch,
        [switch]$RawOutput
    )

    # Get processes
    $processes = if ($Name) {
        Get-Process -Name "*$Name*" -ErrorAction SilentlyContinue
    }
    else {
        Get-Process -ErrorAction SilentlyContinue
    }

    if (-not $processes) {
        return @()
    }

    $requiresOwnerLookup = $SortBy -eq 'user'
    $results = [System.Collections.Generic.List[object]]::new()

    if (-not $requiresOwnerLookup) {
        foreach ($proc in $processes) {
            $results.Add([PSCustomObject]@{
                    PID       = $proc.Id
                    Name      = $proc.ProcessName
                    CPU       = [Math]::Round($proc.CPU, 2)
                    MemoryMB  = [Math]::Round($proc.WorkingSet64 / 1MB, 2)
                    Threads   = $proc.Threads.Count
                    Handles   = $proc.HandleCount
                    Owner     = $null
                    Path      = $proc.Path
                    StartTime = $proc.StartTime
                })
        }
    }
    else {
        $throttleLimit = [Math]::Min(8, [Environment]::ProcessorCount)
        $parallelResults = $processes | ForEach-Object -Parallel {
            $proc = $_
            $owner = $null

            try {
                $wmiProcess = Get-CimInstance Win32_Process -Filter "ProcessId = $($proc.Id)" -ErrorAction SilentlyContinue
                if ($wmiProcess) {
                    $ownerInfo = Invoke-CimMethod -InputObject $wmiProcess -MethodName GetOwner -ErrorAction SilentlyContinue
                    if ($ownerInfo -and $ownerInfo.User) {
                        $owner = "$($ownerInfo.Domain)\$($ownerInfo.User)"
                    }
                }
            }
            catch {
            }

            [PSCustomObject]@{
                PID       = $proc.Id
                Name      = $proc.ProcessName
                CPU       = [Math]::Round($proc.CPU, 2)
                MemoryMB  = [Math]::Round($proc.WorkingSet64 / 1MB, 2)
                Threads   = $proc.Threads.Count
                Handles   = $proc.HandleCount
                Owner     = $owner
                Path      = $proc.Path
                StartTime = $proc.StartTime
            }
        } -ThrottleLimit $throttleLimit

        foreach ($item in $parallelResults) {
            if ($item) {
                $results.Add($item)
            }
        }
    }

    # Sort
    $results = switch ($SortBy) {
        'cpu'   { $results | Sort-Object CPU -Descending }
        'mem'   { $results | Sort-Object MemoryMB -Descending }
        'pid'   { $results | Sort-Object PID }
        'name'  { $results | Sort-Object Name }
        'user'  { $results | Sort-Object Owner }
        default { $results | Sort-Object CPU -Descending }
    }

    # Top N
    if ($Top -gt 0) {
        $results = $results | Select-Object -First $Top
    }

    return $results
}
