#Requires -Version 5.1
<#
.SYNOPSIS
    Starts, stops, or restarts a Windows service by name.
#>
function Set-PCaiServiceState {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$Name,

        [Parameter(Mandatory)]
        [ValidateSet('start','stop','restart')]
        [string]$Action
    )

    switch ($Action) {
        'start' { Start-Service -Name $Name -ErrorAction Stop }
        'stop' { Stop-Service -Name $Name -ErrorAction Stop }
        'restart' { Restart-Service -Name $Name -ErrorAction Stop }
    }

    return Get-Service -Name $Name -ErrorAction SilentlyContinue
}
