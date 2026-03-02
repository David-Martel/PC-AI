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

    $procsPath = Get-RustToolPath -ToolName 'procs'
    $useProcs = $null -ne $procsPath -and (Test-Path $procsPath)
    $nativeType = ([System.Management.Automation.PSTypeName]'PcaiNative.PerformanceModule').Type

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
        $json = [PcaiNative.PerformanceModule]::GetTopProcessesJson([uint32]$Top, $sortKey)
        if (-not $json) {
            return $null
        }

        $result = $json | ConvertFrom-Json
        if (-not $result.processes) {
            return @()
        }

        return @(
            $result.processes | ForEach-Object {
                [PSCustomObject]@{
                    PID       = $_.pid
                    Name      = $_.name
                    CPU       = [Math]::Round([double]$_.cpu_usage, 2)
                    MemoryMB  = [Math]::Round([double]$_.memory_bytes / 1MB, 2)
                    Threads   = $null
                    Handles   = $null
                    Owner     = $null
                    Path      = $_.exe_path
                    StartTime = $null
                    Status    = $_.status
                    Tool      = 'pcai_native'
                }
            }
        )
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

    # Build results using PS7+ parallel processing for enhanced info
    $throttleLimit = [Math]::Min(8, [Environment]::ProcessorCount)

    $results = $processes | ForEach-Object -Parallel {
        $proc = $_
        $owner = $null

        # Try to get owner (this is the slow operation)
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
            # Skip owner lookup errors
        }

        [PSCustomObject]@{
            PID          = $proc.Id
            Name         = $proc.ProcessName
            CPU          = [Math]::Round($proc.CPU, 2)
            MemoryMB     = [Math]::Round($proc.WorkingSet64 / 1MB, 2)
            Threads      = $proc.Threads.Count
            Handles      = $proc.HandleCount
            Owner        = $owner
            Path         = $proc.Path
            StartTime    = $proc.StartTime
        }
    } -ThrottleLimit $throttleLimit

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
