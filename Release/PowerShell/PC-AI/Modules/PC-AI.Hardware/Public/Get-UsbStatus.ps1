#Requires -Version 5.1
<#
.SYNOPSIS
    Gets USB device and controller status

.DESCRIPTION
    Queries all USB devices and controllers, identifying any with errors.
    Useful for diagnosing USB connectivity and driver issues.

.PARAMETER OnlyErrors
    Only return devices with errors

.PARAMETER Filter
    Filter by device name pattern

.EXAMPLE
    Get-UsbStatus
    Returns all USB devices and their status

.EXAMPLE
    Get-UsbStatus -OnlyErrors
    Returns only USB devices with errors

.OUTPUTS
    PSCustomObject[] with properties: Name, PNPClass, Status, ErrorCode, ErrorDescription, Severity
#>
function Get-UsbStatus {
    [CmdletBinding()]
    [OutputType([PSCustomObject[]])]
    param(
        [Parameter()]
        [switch]$OnlyErrors,

        [Parameter()]
        [string]$Filter
    )

    try {
        $query = Get-CimInstance -ClassName Win32_PnPEntity -ErrorAction Stop |
            Where-Object { $_.PNPClass -eq 'USB' -or $_.Name -like '*USB*' }

        if ($Filter) {
            $query = $query | Where-Object { $_.Name -like "*$Filter*" }
        }

        if ($OnlyErrors) {
            $query = $query | Where-Object { $_.ConfigManagerErrorCode -ne 0 }
        }

        $results = $query | ForEach-Object {
            $severity = Get-SeverityFromErrorCode -ErrorCode $_.ConfigManagerErrorCode

            [PSCustomObject]@{
                Name             = $_.Name
                PNPClass         = $_.PNPClass
                Status           = $_.Status
                ErrorCode        = $_.ConfigManagerErrorCode
                ErrorDescription = Format-DeviceErrorCode -ErrorCode $_.ConfigManagerErrorCode
                Severity         = $severity
                DeviceID         = $_.DeviceID
                Manufacturer     = $_.Manufacturer
            }
        }

        return $results | Sort-Object -Property @{Expression = 'ErrorCode'; Descending = $true}, Name

    }
    catch {
        Write-Error "Failed to query USB devices: $($_.Exception.Message)"
        return @()
    }
}
