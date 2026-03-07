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

    $backendCoverage = @(
        [PSCustomObject]@{
            Operation            = 'TokenEstimate'
            Category             = 'Core'
            RustAvailable        = [bool]$native.CoreAvailable
            CSharpBridgeAvailable = [bool]$native.CoreAvailable
            PowerShellSurface    = [bool](Get-Command Get-PcaiTokenEstimate -ErrorAction SilentlyContinue)
            ManagedBaseline      = $true
            PreferredBackend     = if ($native.CoreAvailable) { 'Rust+C#' } else { 'PowerShell' }
        }
        [PSCustomObject]@{
            Operation            = 'DirectoryManifest'
            Category             = 'Search'
            RustAvailable        = [bool]$modules.Search
            CSharpBridgeAvailable = [bool]$modules.Search
            PowerShellSurface    = [bool](Get-Command Invoke-PcaiNativeDirectoryManifest -ErrorAction SilentlyContinue)
            ManagedBaseline      = $true
            PreferredBackend     = if ($modules.Search) { 'Rust+C#' } else { 'PowerShell' }
        }
        [PSCustomObject]@{
            Operation            = 'FileSearch'
            Category             = 'Search'
            RustAvailable        = [bool]$modules.Search
            CSharpBridgeAvailable = [bool]$modules.Search
            PowerShellSurface    = [bool](Get-Command Invoke-PcaiNativeFileSearch -ErrorAction SilentlyContinue)
            ManagedBaseline      = $true
            PreferredBackend     = if ($modules.Search) { 'Rust+C#' } else { 'PowerShell/fd' }
        }
        [PSCustomObject]@{
            Operation            = 'ContentSearch'
            Category             = 'Search'
            RustAvailable        = [bool]$modules.Search
            CSharpBridgeAvailable = [bool]$modules.Search
            PowerShellSurface    = [bool](Get-Command Invoke-PcaiNativeContentSearch -ErrorAction SilentlyContinue)
            ManagedBaseline      = $true
            PreferredBackend     = if ($modules.Search) { 'Rust+C#' } else { 'PowerShell/rg' }
        }
        [PSCustomObject]@{
            Operation            = 'FullContext'
            Category             = 'Diagnostics'
            RustAvailable        = [bool]$native.CoreAvailable
            CSharpBridgeAvailable = [bool]$native.CoreAvailable
            PowerShellSurface    = [bool](Get-Command Invoke-PcaiNativeSystemInfo -ErrorAction SilentlyContinue)
            ManagedBaseline      = $true
            PreferredBackend     = if ($native.CoreAvailable) { 'Rust+C#' } else { 'PowerShell' }
        }
        [PSCustomObject]@{
            Operation            = 'DiskUsage'
            Category             = 'Performance'
            RustAvailable        = [bool]$modules.Performance
            CSharpBridgeAvailable = [bool]$modules.Performance
            PowerShellSurface    = [bool]$modules.Performance
            ManagedBaseline      = $true
            PreferredBackend     = if ($modules.Performance) { 'Rust+C#' } else { 'PowerShell' }
        }
    ) | ForEach-Object {
        $_ | Add-Member -NotePropertyName CoverageState -NotePropertyValue (
            if ($_.RustAvailable -and $_.CSharpBridgeAvailable -and $_.PowerShellSurface) { 'Rust+CSharp+PS' }
            elseif ($_.CSharpBridgeAvailable -and $_.PowerShellSurface) { 'CSharp+PS' }
            elseif ($_.PowerShellSurface) { 'PSOnly' }
            else { 'Unavailable' }
        ) -PassThru | Add-Member -NotePropertyName Gap -NotePropertyValue (
            if ($_.RustAvailable -and $_.CSharpBridgeAvailable -and $_.PowerShellSurface) { '' }
            elseif (-not $_.RustAvailable) { 'Native/Rust coverage missing' }
            elseif (-not $_.PowerShellSurface) { 'PowerShell wrapper missing' }
            else { 'Bridge availability incomplete' }
        ) -PassThru
    }

    $features = [PSCustomObject]@{
        JsonExtraction = $native.CoreAvailable
        PromptAssembly = $native.CoreAvailable
        DirectoryManifest = $modules.Search
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
        BackendCoverage = $backendCoverage
        Cpu       = $cpu
        Gpu       = $gpu
        Services  = $services
    }
}
