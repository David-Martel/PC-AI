#Requires -Version 7.0
<#
.SYNOPSIS
    Returns a capability registry for native and LLM components.

.DESCRIPTION
    Aggregates native DLL availability, feature flags, CPU/GPU info, and
    optional service health data for pcai-inference and router runtimes.

.PARAMETER IncludeGpu
    Include GPU inventory details.

.PARAMETER IncludeServices
    Include Get-PcaiServiceHealth output when available.

.EXAMPLE
    Get-PcaiCapabilities

.EXAMPLE
    Get-PcaiCapabilities -IncludeGpu -IncludeServices
#>
function Get-PcaiCapabilities {
    [CmdletBinding()]
    [OutputType([PSCustomObject])]
    param(
        [Parameter()]
        [switch]$IncludeGpu,

        [Parameter()]
        [switch]$IncludeServices
    )

    $native = Get-PcaiNativeStatus
    $modules = $native.Modules

    $features = [PSCustomObject]@{
        JsonExtraction = $native.CoreAvailable
        PromptAssembly = $native.CoreAvailable
        FileSearch     = $modules.Search
        ContentSearch  = $modules.Search
        DuplicateScan  = $modules.Search
        LogSearch      = $modules.System
        DiskUsage      = $modules.Performance
        MemoryStats    = $modules.Performance
        FsReplace      = $modules.Fs
    }

    $cpu = [PSCustomObject]@{
        LogicalCores = $native.CpuCount
        Architecture = if ([Environment]::Is64BitProcess) { 'x64' } else { 'x86' }
    }

    $gpu = $null
    if ($IncludeGpu) {
        try {
            $gpus = Get-CimInstance Win32_VideoController -ErrorAction Stop
            $gpu = @(
                $gpus | ForEach-Object {
                    [PSCustomObject]@{
                        Name          = $_.Name
                        DriverVersion = $_.DriverVersion
                        Status        = $_.Status
                        PnpDeviceId   = $_.PNPDeviceID
                    }
                }
            )
        } catch {
            $gpu = @()
        }
    }

    $services = $null
    if ($IncludeServices) {
        $serviceCmd = Get-Command Get-PcaiServiceHealth -ErrorAction SilentlyContinue
        if ($serviceCmd) {
            try {
                $services = Get-PcaiServiceHealth
            } catch {
                $services = $null
            }
        }
    }

    [PSCustomObject]@{
        Timestamp = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss')
        Native    = $native
        Features  = $features
        Cpu       = $cpu
        Gpu       = $gpu
        Services  = $services
    }
}
