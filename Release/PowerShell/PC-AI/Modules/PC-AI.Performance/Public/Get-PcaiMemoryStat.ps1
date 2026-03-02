function Get-PcaiMemoryStat {
    <#
    .SYNOPSIS
        Gets system memory statistics.
    #>
    [CmdletBinding()]
    param()

    Import-Module PC-AI.Common -ErrorAction SilentlyContinue
    if (-not (Initialize-PcaiNative)) { return }

    $Json = [PcaiNative.PerformanceModule]::GetMemoryStatsJson()
    if ($Json) {
        return $Json | ConvertFrom-Json
    }
}
