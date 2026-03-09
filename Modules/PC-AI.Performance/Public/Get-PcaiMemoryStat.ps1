function Get-PcaiMemoryStat {
    <#
    .SYNOPSIS
        Gets system memory statistics.
    #>
    [CmdletBinding()]
    param()

    Import-Module PC-AI.Common -ErrorAction SilentlyContinue
    if (-not (Initialize-PcaiNative)) { return }

    $stats = [PcaiNative.PerformanceModule]::GetMemoryStats()
    if ($stats -and $stats.IsSuccess) {
        return [PSCustomObject]@{
            Status               = $stats.Status.ToString()
            TotalMemoryBytes     = [uint64]$stats.TotalMemoryBytes
            UsedMemoryBytes      = [uint64]$stats.UsedMemoryBytes
            AvailableMemoryBytes = [uint64]$stats.AvailableMemoryBytes
            TotalSwapBytes       = [uint64]$stats.TotalSwapBytes
            UsedSwapBytes        = [uint64]$stats.UsedSwapBytes
            ElapsedMs            = [uint64]$stats.ElapsedMs
            MemoryUsagePercent   = [double]$stats.MemoryUsagePercent
            SwapUsagePercent     = [double]$stats.SwapUsagePercent
            Source               = 'PcaiNative.PerformanceModule.GetMemoryStats'
        }
    }

    $json = [PcaiNative.PerformanceModule]::GetMemoryStatsJson()
    if ($json) {
        return $json | ConvertFrom-Json
    }
}
