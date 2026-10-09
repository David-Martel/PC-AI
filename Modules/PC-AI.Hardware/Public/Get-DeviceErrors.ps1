#Requires -Version 7.0
<#
.SYNOPSIS
    Gets devices with errors from Device Manager

.DESCRIPTION
    Queries Win32_PnPEntity for devices with ConfigManagerErrorCode != 0,
    indicating device or driver issues.

.PARAMETER IncludeOK
    Include devices with no errors (ConfigManagerErrorCode = 0)

.PARAMETER Class
    Filter by PNP device class (e.g., 'USB', 'Net', 'DiskDrive')

.EXAMPLE
    Get-DeviceErrors
    Returns all devices with errors

.EXAMPLE
    Get-DeviceErrors -Class 'USB'
    Returns only USB devices with errors

.OUTPUTS
    PSCustomObject[] with properties: Name, PNPClass, Manufacturer, ErrorCode, ErrorDescription, Severity, Status
#>
function Get-DeviceErrors {
    [CmdletBinding()]
    [OutputType([PSCustomObject[]])]
    param(
        [Parameter()]
        [switch]$IncludeOK,

        [Parameter()]
        [string]$Class
    )

    try {
        $results = @()
        $nativeAvailable = $false

        # Attempt to use Native Core if available
        try {
          $json = Get-HardwarePnpDevicesNative -Class $Class
          if ($json) {
            # Preserve the top-level shape: null and a single object cannot
            # establish that the native backend queried an empty inventory.
            $nativeDevices = ConvertFrom-Json -InputObject $json -NoEnumerate -ErrorAction Stop
            if ($nativeDevices -isnot [array]) {
                throw 'Native PnP response must be a JSON array.'
            }
            foreach ($dev in $nativeDevices) {
                $code = [uint32]0
                if ($null -eq $dev -or $null -eq $dev.PSObject.Properties['config_error_code'] -or
                    -not [uint32]::TryParse([string]$dev.config_error_code, [ref]$code)) {
                    throw 'Native PnP response has no valid config_error_code.'
                }
                if ($Class -and $dev.pnp_class -ne $Class -and $dev.name -notlike "*$Class*") { continue }
                if ($IncludeOK -or $code -ne 0) {
                    $results += [PSCustomObject]@{
                        Name             = $dev.name
                        PNPClass         = $dev.pnp_class
                        Manufacturer     = $dev.manufacturer
                        ErrorCode        = $code
                        ErrorDescription = if ($dev.error_summary) { $dev.error_summary } else { Format-DeviceErrorCode -ErrorCode $code }
                        Severity         = Get-SeverityFromErrorCode -ErrorCode $code
                        Status           = $dev.status
                        DeviceID         = $dev.device_id
                    }
                }
            }
            $nativeAvailable = $true
          }
        } catch {
            # Reject the whole native result, including any rows already accumulated.
            # Missing fields must never become hundreds of false hardware errors.
            $results = @()
            Write-Verbose "Native PnP response unavailable or incompatible: $($_.Exception.Message)"
        }

        if (-not $nativeAvailable) {
            Write-Verbose 'Native PnP interrogation unavailable, using CIM fallback.'
            $query = Get-CimInstance -ClassName Win32_PnPEntity -ErrorAction Stop

            if (-not $IncludeOK) {
                $query = $query | Where-Object { $_.ConfigManagerErrorCode -ne 0 }
            }

            if ($Class) {
                $query = $query | Where-Object { $_.PNPClass -eq $Class -or $_.Name -like "*$Class*" }
            }

            $results = $query | ForEach-Object {
                [PSCustomObject]@{
                    Name             = $_.Name
                    PNPClass         = $_.PNPClass
                    Manufacturer     = $_.Manufacturer
                    ErrorCode        = $_.ConfigManagerErrorCode
                    ErrorDescription = Format-DeviceErrorCode -ErrorCode $_.ConfigManagerErrorCode
                    Severity         = Get-SeverityFromErrorCode -ErrorCode $_.ConfigManagerErrorCode
                    Status           = $_.Status
                    DeviceID         = $_.DeviceID
                }
            }
        }

        return $results | Sort-Object -Property @{Expression = 'Severity'; Descending = $true }, Name

    } catch {
        Write-Error "Failed to query PnP devices: $($_.Exception.Message)"
        return @()
    }
}

function Get-HardwarePnpDevicesNative {
    param($Class)
    if ($null -ne (Get-Module -Name 'PC-AI.Common' -ErrorAction SilentlyContinue) -and [PcaiNative.HardwareModule]::IsAvailable) {
        return [PcaiNative.HardwareModule]::GetPnpDevicesJson($Class)
    }
    return $null
}
