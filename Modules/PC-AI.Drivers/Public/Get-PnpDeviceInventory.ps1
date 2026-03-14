#Requires -Version 5.1
<#
.SYNOPSIS
    Discovers all PnP devices and extracts structured driver metadata.

.DESCRIPTION
    Enumerates PnP devices using Get-PnpDevice (with a CIM fallback for environments
    where the PnP provider is unavailable). For each device that passes the specified
    filters, driver properties are queried via Get-PnpDeviceProperty. Hardware IDs are
    parsed by Resolve-HardwareId to extract VID, PID, and bus type.

    No administrator rights are required for basic enumeration. Some device properties
    (e.g. DriverVersion, DriverDate) may be unavailable without elevation on certain
    device classes.

.PARAMETER Class
    Filter results to a specific PnP device class (e.g. 'Net', 'USB', 'Display').
    Matched against the PnpClass property returned by Get-PnpDevice.

.PARAMETER VidPid
    Filter results by a hardware ID substring pattern (e.g. 'VID_0BDA', 'PID_8156').
    Applied after all other filters, against the raw InstanceId string.

.PARAMETER ActiveOnly
    When specified, only devices with Status equal to 'OK' are returned.

.EXAMPLE
    Get-PnpDeviceInventory
    Returns all PnP devices with driver metadata.

.EXAMPLE
    Get-PnpDeviceInventory -Class 'Net' -ActiveOnly
    Returns only active network adapter devices with driver metadata.

.EXAMPLE
    Get-PnpDeviceInventory -VidPid 'VID_0BDA'
    Returns all Realtek USB devices (VID 0BDA) regardless of class or status.

.OUTPUTS
    PSCustomObject[] with properties: Name, PnpClass, VID, PID, Bus, InstanceId,
    DriverVersion, DriverDate, Manufacturer, Status
#>
function Get-PnpDeviceInventory {
    [CmdletBinding()]
    [OutputType([PSCustomObject[]])]
    param(
        [Parameter()]
        [string]$Class,

        [Parameter()]
        [string]$VidPid,

        [Parameter()]
        [switch]$ActiveOnly
    )

    try {
        # --- Phase 0: Try native Rust DLL for high-performance enumeration ---
        $nativeAvailable = $false
        if ($null -ne (Get-Module -Name 'PC-AI.Common' -ErrorAction SilentlyContinue)) {
            try {
                $nativeJson = [PcaiNative.HardwareModule]::GetPnpDevicesJson($Class)
                if ($nativeJson) {
                    Write-Verbose 'Using native PcaiNative for device enumeration.'
                    $nativeDevices = $nativeJson | ConvertFrom-Json
                    $results = @()
                    foreach ($dev in $nativeDevices) {
                        if ($ActiveOnly -and $dev.status -ne 'OK') { continue }
                        $hwParsed = Resolve-HardwareId -HardwareId $dev.device_id
                        if ($VidPid -and $dev.device_id -notlike "*$VidPid*") { continue }
                        $results += [PSCustomObject]@{
                            Name          = $dev.name
                            PnpClass      = $dev.pnp_class
                            VID           = $hwParsed.VID
                            PID           = $hwParsed.PID
                            Bus           = $hwParsed.Bus
                            InstanceId    = $dev.device_id
                            DriverVersion = $dev.driver_version
                            DriverDate    = $dev.driver_date
                            Manufacturer  = $dev.manufacturer
                            Status        = $dev.status
                        }
                    }
                    $nativeAvailable = $true
                    return $results | Sort-Object PnpClass, Name
                }
            } catch {
                Write-Verbose "Native PnP unavailable: $($_.Exception.Message)"
            }
        }

        # --- Phase 1: Enumerate devices, apply class and status filters early ---
        $devices = $null
        $usedCimFallback = $false

        if (Get-Command -Name 'Get-PnpDevice' -ErrorAction SilentlyContinue) {
            Write-Verbose 'Using Get-PnpDevice for enumeration.'

            $pnpParams = @{ ErrorAction = 'Stop' }
            if ($Class) {
                $pnpParams['Class'] = $Class
            }
            if ($ActiveOnly) {
                $pnpParams['Status'] = 'OK'
            }

            $devices = @(Get-PnpDevice @pnpParams)
        }
        else {
            Write-Verbose 'Get-PnpDevice not available — falling back to Win32_PnPEntity via CIM.'
            $usedCimFallback = $true

            $cimDevices = @(Get-CimInstance -ClassName Win32_PnPEntity -ErrorAction Stop)

            if ($Class) {
                $cimDevices = @($cimDevices | Where-Object { $_.PNPClass -eq $Class })
            }
            if ($ActiveOnly) {
                $cimDevices = @($cimDevices | Where-Object { $_.Status -eq 'OK' })
            }

            $devices = $cimDevices
        }

        if ($devices.Count -eq 0) {
            Write-Verbose 'No devices matched the specified filters.'
            return @()
        }

        Write-Verbose "Querying driver properties for $($devices.Count) device(s)."

        # --- Phase 2: For each filtered device, query driver properties ---
        $results = @()

        foreach ($device in $devices) {
            # Resolve InstanceId to extract VID/PID/Bus
            $instanceId = if ($usedCimFallback) { $device.DeviceID } else { $device.InstanceId }
            $hwParsed = Resolve-HardwareId -HardwareId $instanceId

            # Apply VidPid substring filter now (before the expensive property query)
            if ($VidPid -and $instanceId -notlike "*$VidPid*") {
                continue
            }

            # Query driver-specific device properties; SilentlyContinue because many
            # devices (root enumerators, virtual devices) do not expose these keys.
            $driverVersion = $null
            $driverDate    = $null
            $manufacturer  = $null

            $versionProp = Get-PnpDeviceProperty -InstanceId $instanceId `
                -KeyName 'DEVPKEY_Device_DriverVersion' `
                -ErrorAction SilentlyContinue
            if ($versionProp -and $versionProp.Data) {
                $driverVersion = $versionProp.Data
            }

            $dateProp = Get-PnpDeviceProperty -InstanceId $instanceId `
                -KeyName 'DEVPKEY_Device_DriverDate' `
                -ErrorAction SilentlyContinue
            if ($dateProp -and $dateProp.Data) {
                $driverDate = $dateProp.Data
            }

            $mfgProp = Get-PnpDeviceProperty -InstanceId $instanceId `
                -KeyName 'DEVPKEY_Device_Manufacturer' `
                -ErrorAction SilentlyContinue
            if ($mfgProp -and $mfgProp.Data) {
                $manufacturer = $mfgProp.Data
            }

            # Fall back to the CIM Manufacturer field when available
            if (-not $manufacturer -and $usedCimFallback -and $device.Manufacturer) {
                $manufacturer = $device.Manufacturer
            }

            $results += [PSCustomObject]@{
                Name          = $device.FriendlyName
                PnpClass      = if ($usedCimFallback) { $device.PNPClass } else { $device.Class }
                VID           = $hwParsed.VID
                PID           = $hwParsed.PID
                Bus           = $hwParsed.Bus
                InstanceId    = $instanceId
                DriverVersion = $driverVersion
                DriverDate    = $driverDate
                Manufacturer  = $manufacturer
                Status        = $device.Status
            }
        }

        Write-Verbose "Inventory complete: $($results.Count) device(s) returned."

        return @($results | Sort-Object -Property PnpClass, Name)

    }
    catch {
        Write-Error "Get-PnpDeviceInventory failed: $($_.Exception.Message)"
        return @()
    }
}
