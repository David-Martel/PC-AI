#Requires -Version 5.1
<#
.SYNOPSIS
    Stops WinSocat HVSOCK proxies started by Start-HVSockProxy.
#>
function Stop-HVSockProxy {
    [CmdletBinding()]
    [OutputType([PSCustomObject])]
    param(
        [Parameter()]
        [string]$StatePath = "$env:ProgramData\PC_AI\hvsock-proxy\state.json"
    )

    if (-not (Test-Path $StatePath)) {
        return [PSCustomObject]@{
            Stopped = 0
            Message = 'No state file found.'
        }
    }

    $state = Get-Content $StatePath -Raw | ConvertFrom-Json
    $stopped = 0
    foreach ($entry in $state) {
        try {
            Stop-Process -Id $entry.Pid -Force -ErrorAction SilentlyContinue
            $stopped++
        }
        catch {
        }
    }

    Remove-Item $StatePath -ErrorAction SilentlyContinue

    return [PSCustomObject]@{
        Stopped = $stopped
        StatePath = $StatePath
    }
}
