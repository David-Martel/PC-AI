#Requires -Version 5.1
<#
.SYNOPSIS
    Internal helper functions for PC-AI.Hardware module

.DESCRIPTION
    Provides formatting and utility functions used by public module functions.
#>

function Format-DeviceErrorCode {
    <#
    .SYNOPSIS
        Formats ConfigManagerErrorCode into human-readable description
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [int]$ErrorCode
    )

    $errorDescriptions = @{
        0  = 'Working properly'
        1  = 'Device not configured correctly'
        2  = 'Windows cannot load the driver'
        3  = 'Driver corrupted or system low on memory'
        4  = 'Device not working properly (driver or registry issue)'
        5  = 'Driver requires resource Windows cannot manage'
        6  = 'Boot configuration conflict'
        7  = 'Cannot filter'
        8  = 'Driver loader missing'
        9  = 'Firmware reporting resources incorrectly'
        10 = 'Device cannot start'
        11 = 'Device failed'
        12 = 'Cannot find enough free resources'
        13 = 'Cannot verify resources'
        14 = 'Restart required'
        15 = 'Re-enumeration problem'
        16 = 'Cannot identify all resources'
        17 = 'Unknown resource type requested'
        18 = 'Reinstall drivers'
        19 = 'Registry failure'
        20 = 'VxD loader failure'
        21 = 'System failure'
        22 = 'Device is disabled'
        23 = 'System failure'
        24 = 'Device missing or not working'
        25 = 'Setup incomplete'
        26 = 'Setup incomplete'
        27 = 'Invalid log configuration'
        28 = 'Drivers not installed'
        29 = 'Firmware did not provide resources'
        30 = 'IRQ conflict'
        31 = 'Device not working properly'
        32 = 'Driver service disabled'
        33 = 'Cannot determine resource requirements'
        34 = 'Cannot determine device settings'
        35 = 'Cannot determine device settings (missing firmware)'
        36 = 'PCI IRQ conflict'
        37 = 'Cannot initialize'
        38 = 'Cannot load driver (already loaded by another device)'
        39 = 'Cannot load driver (driver corrupted)'
        40 = 'Service key information missing'
        41 = 'Cannot load driver'
        42 = 'Duplicate device running'
        43 = 'Device stopped responding (Code 43)'
        44 = 'Application or service shut down device'
        45 = 'Device not connected'
        46 = 'Cannot access device (Windows shutting down)'
        47 = 'Safe removal prepared'
        48 = 'Firmware has blocked device'
        49 = 'Registry size limit exceeded'
        50 = 'Cannot apply properties'
        51 = 'Device waiting on another device'
        52 = 'Cannot verify digital signature'
        53 = 'Reserved for Windows'
        54 = 'ACPI failure'
    }

    if ($errorDescriptions.ContainsKey($ErrorCode)) {
        return $errorDescriptions[$ErrorCode]
    }
    return "Unknown error code: $ErrorCode"
}

function Get-SeverityFromErrorCode {
    <#
    .SYNOPSIS
        Returns severity level based on ConfigManagerErrorCode
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [int]$ErrorCode
    )

    $critical = @(1, 10, 12, 28, 31, 43)
    $warning = @(22, 29, 32, 44)

    if ($ErrorCode -eq 0) { return 'OK' }
    if ($ErrorCode -in $critical) { return 'Critical' }
    if ($ErrorCode -in $warning) { return 'Warning' }
    return 'Error'
}

function ConvertTo-ReportSection {
    <#
    .SYNOPSIS
        Formats data into a report section
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$Title,

        [Parameter()]
        [object]$Data,

        [Parameter()]
        [string]$EmptyMessage = 'No data found.'
    )

    $output = @()
    $output += "== $Title =="
    $output += ''

    if ($null -eq $Data -or ($Data -is [array] -and $Data.Count -eq 0)) {
        $output += $EmptyMessage
    }
    else {
        if ($Data -is [string]) {
            $output += $Data
        }
        else {
            $output += ($Data | Format-Table -AutoSize | Out-String).Trim()
        }
    }

    $output += ''
    return $output -join "`r`n"
}
