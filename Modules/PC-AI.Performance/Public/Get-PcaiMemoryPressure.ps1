function Get-PcaiMemoryPressure {
    <#
    .SYNOPSIS
        Analyzes system memory pressure for LLM agent optimization.
    .DESCRIPTION
        Uses native Rust FFI to analyze memory pressure including paging rates,
        pool memory, handle leaks, and orphaned terminal processes. Falls back
        to PowerShell-based analysis when native DLL is unavailable.
    .OUTPUTS
        PSCustomObject with pressure level, metrics, and recommendations.
    .EXAMPLE
        Get-PcaiMemoryPressure
    .EXAMPLE
        Get-PcaiMemoryPressure -Detailed
    #>
    [CmdletBinding()]
    param(
        [switch]$Detailed,
        [switch]$AsJson
    )

    Import-Module PC-AI.Common -ErrorAction SilentlyContinue
    $nativeAvailable = $false
    try { $nativeAvailable = Initialize-PcaiNative } catch {}

    if ($nativeAvailable) {
        if ($Detailed -or $AsJson) {
            $json = [PcaiNative.OptimizerModule]::GetMemoryPressureJson()
            if ($AsJson) { return $json }
            if ($json) { return $json | ConvertFrom-Json }
        }

        $report = [PcaiNative.OptimizerModule]::AnalyzeMemoryPressure()
        if ($report -and $report.IsSuccess) {
            return [PSCustomObject]@{
                PressureLevel        = $report.PressureLevelName
                PressureLevelCode    = [int]$report.PressureLevel
                AvailableMB          = [uint64]$report.AvailableMB
                CommittedPct         = [float]$report.CommittedPct
                PoolNonpagedMB       = [uint64]$report.PoolNonpagedMB
                PagesPerSec          = [uint64]$report.PagesPerSec
                TopConsumerCount     = [uint32]$report.TopConsumerCount
                HandleLeakCount      = [uint32]$report.HandleLeakCount
                OrphanTerminalCount  = [uint32]$report.OrphanTerminalCount
                ElapsedMs            = [uint64]$report.ElapsedMs
                Source               = 'PcaiNative.OptimizerModule'
            }
        }
    }

    # Fallback: PowerShell-based analysis
    Write-Verbose 'Native DLL unavailable, using PowerShell fallback'
    $os = Get-CimInstance Win32_OperatingSystem
    $cs = Get-CimInstance Win32_ComputerSystem
    $availMB = [math]::Round($os.FreePhysicalMemory / 1KB, 0)
    $totalMB = [math]::Round($cs.TotalPhysicalMemory / 1MB, 0)
    $usedPct = [math]::Round((1 - $os.FreePhysicalMemory * 1KB / $cs.TotalPhysicalMemory) * 100, 1)

    # Detect handle leaks (>100K handles)
    $handleLeaks = (Get-Process | Where-Object { $_.HandleCount -gt 100000 } | Measure-Object).Count

    # Detect orphan terminals
    $allPids = (Get-Process).Id
    $orphans = Get-Process -Name 'cmd', 'conhost' -ErrorAction SilentlyContinue | Where-Object {
        try {
            $parent = (Get-CimInstance Win32_Process -Filter "ProcessId=$($_.Id)").ParentProcessId
            $parent -notin $allPids
        } catch { $false }
    }
    $orphanCount = ($orphans | Measure-Object).Count

    # Top consumers (>500MB working set)
    $topConsumers = (Get-Process | Where-Object { $_.WorkingSet64 -gt 500MB } | Measure-Object).Count

    # Pressure level
    $level = if ($availMB -lt 1024) { 3 }
             elseif ($availMB -lt 2048) { 2 }
             elseif ($availMB -lt 4096) { 1 }
             else { 0 }

    $levelName = @('Low', 'Moderate', 'High', 'Critical')[$level]

    # Pool nonpaged (requires perf counter)
    $poolNP = 0
    try {
        $counter = Get-Counter '\Memory\Pool Nonpaged Bytes' -ErrorAction SilentlyContinue
        if ($counter) {
            $poolNP = [math]::Round($counter.CounterSamples[0].CookedValue / 1MB, 0)
        }
    } catch {}

    # Pages/sec
    $pagesSec = 0
    try {
        $counter = Get-Counter '\Memory\Pages/sec' -ErrorAction SilentlyContinue
        if ($counter) {
            $pagesSec = [math]::Round($counter.CounterSamples[0].CookedValue, 0)
        }
    } catch {}

    [PSCustomObject]@{
        PressureLevel        = $levelName
        PressureLevelCode    = $level
        AvailableMB          = $availMB
        CommittedPct         = $usedPct
        PoolNonpagedMB       = $poolNP
        PagesPerSec          = $pagesSec
        TopConsumerCount     = $topConsumers
        HandleLeakCount      = $handleLeaks
        OrphanTerminalCount  = $orphanCount
        ElapsedMs            = 0
        Source               = 'PowerShell-Fallback'
    }
}
