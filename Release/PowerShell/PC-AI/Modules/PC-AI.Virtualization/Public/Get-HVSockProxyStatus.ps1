#Requires -Version 5.1
<#
.SYNOPSIS
    Returns status of WinSocat HVSOCK proxies.
#>
function Get-HVSockProxyStatus {
    [CmdletBinding()]
    [OutputType([PSCustomObject])]
    param(
        [Parameter()]
        [string]$StatePath = "$env:ProgramData\PC_AI\hvsock-proxy\state.json"
    )

    if (-not (Test-Path $StatePath)) {
        return [PSCustomObject]@{
            Running = 0
            Entries = @()
            StatePath = $StatePath
        }
    }

    $state = Get-Content $StatePath -Raw | ConvertFrom-Json
    $entries = @()
    foreach ($entry in $state) {
        $running = $false
        try {
            $proc = Get-Process -Id $entry.Pid -ErrorAction SilentlyContinue
            if ($proc) { $running = $true }
        } catch {
        }

        $entries += [PSCustomObject]@{
            Name = $entry.Name
            Pid = $entry.Pid
            Running = $running
            ServiceId = $entry.ServiceId
            TcpTarget = $entry.TcpTarget
            Started = $entry.Started
        }
    }

    return [PSCustomObject]@{
        Running = ($entries | Where-Object { $_.Running }).Count
        Entries = $entries
        StatePath = $StatePath
    }
}
