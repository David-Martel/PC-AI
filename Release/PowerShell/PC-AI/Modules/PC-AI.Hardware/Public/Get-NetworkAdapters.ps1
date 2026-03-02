#Requires -Version 5.1
<#
.SYNOPSIS
    Gets physical network adapter status

.DESCRIPTION
    Queries physical network adapters for configuration and health status.
    Filters out virtual adapters by default.

.PARAMETER IncludeVirtual
    Include virtual network adapters

.PARAMETER OnlyEnabled
    Only return enabled adapters

.EXAMPLE
    Get-NetworkAdapters
    Returns all physical network adapters

.EXAMPLE
    Get-NetworkAdapters -IncludeVirtual
    Returns all adapters including virtual ones

.OUTPUTS
    PSCustomObject[] with properties: Name, NetEnabled, Status, MACAddress, Speed, InterfaceType, Severity
#>
function Get-NetworkAdapters {
    [CmdletBinding()]
    [OutputType([PSCustomObject[]])]
    param(
        [Parameter()]
        [switch]$IncludeVirtual,

        [Parameter()]
        [switch]$OnlyEnabled
    )

    try {
        $query = Get-CimInstance -ClassName Win32_NetworkAdapter -ErrorAction Stop

        if (-not $IncludeVirtual) {
            $query = $query | Where-Object { $_.PhysicalAdapter -eq $true }
        }

        if ($OnlyEnabled) {
            $query = $query | Where-Object { $_.NetEnabled -eq $true }
        }

        $results = $query | ForEach-Object {
            # Determine severity
            $severity = 'OK'
            if ($_.NetEnabled -eq $false -and $_.PhysicalAdapter -eq $true) {
                $severity = 'Info'  # Disabled adapters are informational
            }
            if ($_.Status -and $_.Status -ne 'OK' -and $_.Status -ne 'Connected') {
                $severity = 'Warning'
            }

            # Calculate speed in Mbps
            $speedMbps = if ($_.Speed) { [math]::Round($_.Speed / 1000000, 0) } else { 0 }

            [PSCustomObject]@{
                Name          = $_.Name
                NetEnabled    = $_.NetEnabled
                Status        = $_.Status
                MACAddress    = $_.MACAddress
                SpeedMbps     = $speedMbps
                AdapterType   = $_.AdapterType
                InterfaceType = $_.NetConnectionID
                PhysicalAdapter = $_.PhysicalAdapter
                Severity      = $severity
                DeviceID      = $_.DeviceID
            }
        }

        return $results | Sort-Object -Property Name

    }
    catch {
        Write-Error "Failed to query network adapters: $($_.Exception.Message)"
        return @()
    }
}
