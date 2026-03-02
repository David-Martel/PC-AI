function Get-PcaiTopProcess {
    <#
    .SYNOPSIS
        Gets top resource-consuming processes.
    .DESCRIPTION
        Returns a snapshot of processes sorted by CPU or Memory.
    #>
    [CmdletBinding()]
    param(
        [ValidateSet('memory', 'cpu')]
        [string]$SortBy = 'memory',
        [int]$Top = 20
    )

    Import-Module PC-AI.Common -ErrorAction SilentlyContinue
    if (-not (Initialize-PcaiNative)) { return }

    $Json = [PcaiNative.PerformanceModule]::GetTopProcessesJson($Top, $SortBy)
    if ($Json) {
        return $Json | ConvertFrom-Json
    }
}
