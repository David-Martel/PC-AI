#Requires -Version 5.1

function Resolve-HardwareId {
    [CmdletBinding()]
    [OutputType([PSCustomObject])]
    param(
        [Parameter(Mandatory)]
        [string]$HardwareId
    )

    $bus = $null
    $vid = $null
    $devPid = $null
    $rev = $null
    $subSystem = $null

    if ($HardwareId -match '^([^\\]+)\\') {
        $bus = $Matches[1].ToUpper()
    }

    if ($HardwareId -match 'VID_([0-9A-Fa-f]{4})') {
        $vid = $Matches[1].ToUpper()
    }
    elseif ($HardwareId -match 'VEN_([0-9A-Fa-f]{4})') {
        $vid = $Matches[1].ToUpper()
    }

    if ($HardwareId -match 'PID_([0-9A-Fa-f]{4})') {
        $devPid = $Matches[1].ToUpper()
    }
    elseif ($HardwareId -match 'DEV_([0-9A-Fa-f]{4})') {
        $devPid = $Matches[1].ToUpper()
    }

    if ($HardwareId -match 'REV_([0-9A-Fa-f]{2,4})') {
        $rev = $Matches[1].ToUpper()
    }

    if ($HardwareId -match 'SUBSYS_([0-9A-Fa-f]{8})') {
        $subSystem = $Matches[1].ToUpper()
    }
    elseif ($HardwareId -match 'FUNC_([0-9A-Fa-f]{2})') {
        $subSystem = $Matches[1].ToUpper()
    }

    return [PSCustomObject]@{
        Bus       = $bus
        VID       = $vid
        PID       = $devPid
        REV       = $rev
        SubSystem = $subSystem
        RawId     = $HardwareId
    }
}
