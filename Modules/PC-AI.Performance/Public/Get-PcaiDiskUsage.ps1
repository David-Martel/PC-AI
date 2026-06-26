function Get-PcaiDiskUsage {
    <#
    .SYNOPSIS
        Gets disk usage statistics for a directory.
    .DESCRIPTION
        Uses native Rust traversal for high-performance analysis.
    .PARAMETER Path
        Directory to analyze. Defaults to current location.
    .PARAMETER Top
        Number of top subdirectories to return.
    #>
    [CmdletBinding()]
    param(
        [string]$Path = $PWD,
        [int]$Top = 10
    )

    Import-Module PC-AI.Common -ErrorAction SilentlyContinue
    if (-not (Initialize-PcaiNative)) { return }

    $performanceType = ([System.Management.Automation.PSTypeName]'PcaiNative.PerformanceModule').Type
    if (-not $performanceType) {
        $loadedAssembly = [System.AppDomain]::CurrentDomain.GetAssemblies() |
            Where-Object { $_.GetName().Name -eq 'PcaiNative' } |
            Select-Object -First 1
        $loadedLocation = if ($loadedAssembly) { $loadedAssembly.Location } else { '<not loaded>' }
        throw "PcaiNative.PerformanceModule is unavailable after native initialization. Loaded PcaiNative assembly: $loadedLocation"
    }

    $Json = $performanceType::GetDiskUsageJson($Path, $Top)
    if ($Json) {
        return $Json | ConvertFrom-Json
    }
}
