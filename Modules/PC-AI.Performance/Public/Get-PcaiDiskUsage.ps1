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

    $Json = [PcaiNative.PerformanceModule]::GetDiskUsageJson($Path, $Top)
    if ($Json) {
        return $Json | ConvertFrom-Json
    }
}
