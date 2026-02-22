#Requires -Version 5.1

<#
.SYNOPSIS
    PC-AI Performance Module — native-accelerated system performance diagnostics.

.DESCRIPTION
    Bridges PowerShell to the Rust/C# native layer (PcaiNative.dll,
    PerformanceModule) for high-throughput diagnostics that would be prohibitively
    slow with pure PowerShell cmdlets. Falls back to WMI/CIM implementations
    automatically when PcaiNative.dll is not present.

    Exported functions:
      Get-DiskSpace          - Report used/free space for all local fixed volumes.
      Get-ProcessPerformance - Snapshot CPU and working-set usage for all running
                               processes using low-overhead native counters.
      Watch-SystemResources  - Continuously monitor CPU, memory, and disk I/O,
                               emitting structured objects at a configurable interval.
      Optimize-Disks         - Run TRIM (SSD) or defragmentation (HDD) with
                               smart detection of drive type; supports scheduled
                               task registration.
      Get-PcaiDiskUsage      - Native-accelerated recursive directory size scan.
                               Processes millions of files per second via Rust FFI.
                               Mapped by FunctionGemma router as pcai_get_disk_usage.
      Get-PcaiTopProcess     - Return the top N processes sorted by CPU or memory
                               using native performance counters.
                               Mapped by FunctionGemma router as pcai_get_top_processes.
      Get-PcaiMemoryStat     - Return system-wide memory statistics (total, used,
                               available, commit charge) via native query.
                               Mapped by FunctionGemma router as pcai_get_memory_stats.
      Test-PcaiNative        - Probe PcaiNative.dll availability and verify that
                               PerformanceModule is loadable; returns a status object.

    Native acceleration (PcaiNative.dll — PerformanceModule):
      Get-PcaiDiskUsage, Get-PcaiTopProcess, and Get-PcaiMemoryStat require
      bin\PcaiNative.dll built from Native\PcaiNative\. Test-PcaiNative confirms
      whether the DLL is present and the [PcaiNative.PerformanceModule] type can
      be resolved. The remaining functions (Get-DiskSpace, Get-ProcessPerformance,
      Watch-SystemResources, Optimize-Disks) work without the DLL.

    Dependencies:
      - PowerShell 5.1 or later
      - Windows 10/11
      - bin\PcaiNative.dll (optional, required for Get-PcaiDiskUsage /
        Get-PcaiTopProcess / Get-PcaiMemoryStat)
      - Administrator rights recommended for Watch-SystemResources disk I/O counters
#>

# Module variables
$script:ModuleRoot = $PSScriptRoot

# Import private functions
$privatePath = Join-Path $PSScriptRoot 'Private'
if (Test-Path $privatePath) {
    Get-ChildItem -Path $privatePath -Filter '*.ps1' | ForEach-Object {
        . $_.FullName
    }
}

# Import public functions
$publicPath = Join-Path $PSScriptRoot 'Public'
if (Test-Path $publicPath) {
    Get-ChildItem -Path $publicPath -Filter '*.ps1' | ForEach-Object {
        . $_.FullName
    }
}

Export-ModuleMember -Function @(
    'Get-DiskSpace',
    'Get-ProcessPerformance',
    'Watch-SystemResources',
    'Optimize-Disks',
    'Get-PcaiDiskUsage',
    'Get-PcaiTopProcess',
    'Get-PcaiMemoryStat',
    'Test-PcaiNative'
)
