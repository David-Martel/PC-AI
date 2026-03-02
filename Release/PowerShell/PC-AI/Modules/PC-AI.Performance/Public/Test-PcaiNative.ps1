function Test-PcaiNative {
    <#
    .SYNOPSIS
        Verifies native DLL is loaded and working.
    #>
    [CmdletBinding()]
    param()

    Import-Module PC-AI.Common -ErrorAction SilentlyContinue
    if (-not (Initialize-PcaiNative)) { return $false }

    return [PcaiNative.PerformanceModule]::Test()
}
