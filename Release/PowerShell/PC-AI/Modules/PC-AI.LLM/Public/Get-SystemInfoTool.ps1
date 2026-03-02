#Requires -Version 5.1

function Get-SystemInfoTool {
    <#
    .SYNOPSIS
        Active system interrogation tool for the AI agent.

    .DESCRIPTION
        Allows the agent to query specific WMI/CIM properties for devices,
        storage, and networking to verify hypotheses.

    .PARAMETER Category
        The category of information: Storage, Network, USB, BIOS, OS, GPU, DiskDrive, Net, Display, Media, HIDClass

    .PARAMETER Detail
        Specific detail requested (e.g., "FullStatus", "DriverVersion", "Capabilities")

    .EXAMPLE
        Get-SystemInfoTool -Category Network -Detail DriverVersion
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory, Position = 0)]
        [ValidateSet('Storage', 'Network', 'USB', 'BIOS', 'OS', 'GPU', 'DiskDrive', 'Net', 'Display', 'Media', 'HIDClass')]
        [string]$Category,

        [Parameter()]
        [string]$Detail = 'Summary'
    )

    Write-Verbose "Interrogating $Category for $Detail"

    $normalizedCategory = switch ($Category) {
        'DiskDrive' { 'Storage' }
        'Net' { 'Network' }
        'Display' { 'GPU' }
        default { $Category }
    }

    # Use native acceleration if available for general OS/System summary
    if ($normalizedCategory -eq 'OS' -and $Detail -eq 'Summary' -and (Get-Command Invoke-PcaiNativeSystemInfo -ErrorAction SilentlyContinue)) {
        try {
            $nativeInfo = Invoke-PcaiNativeSystemInfo
            if ($nativeInfo) {
                return $nativeInfo | ConvertTo-Json -Depth 5
            }
        } catch {
            Write-Verbose "Native system info failed, falling back to CIM: $_"
        }
    }
    try {
        $result = switch ($normalizedCategory) {
            'Storage' {
                if ($Detail -ne 'DriverVersion') {
                    if (-not (Get-Command Get-DiskHealth -ErrorAction SilentlyContinue)) {
                        Import-Module PC-AI.Hardware -ErrorAction SilentlyContinue
                    }
                }

                if ($Detail -ne 'DriverVersion' -and (Get-Command Get-DiskHealth -ErrorAction SilentlyContinue)) {
                    Get-DiskHealth
                } else {
                    $disks = Get-CimInstance Win32_DiskDrive
                    $disks | Select-Object Model, InterfaceType, PNPDeviceID | ForEach-Object {
                        $pnp = Get-CimInstance Win32_PnPEntity -Filter "DeviceID = '$($_.PNPDeviceID.Replace('\', '\\'))'"
                        @{ Model = $_.Model; Driver = $pnp.DriverVersion }
                    }
                }
            }
            'Network' {
                if ($Detail -ne 'DriverVersion') {
                    if (-not (Get-Command Get-NetworkAdapters -ErrorAction SilentlyContinue)) {
                        Import-Module PC-AI.Hardware -ErrorAction SilentlyContinue
                    }
                }

                if ($Detail -ne 'DriverVersion' -and (Get-Command Get-NetworkAdapters -ErrorAction SilentlyContinue)) {
                    Get-NetworkAdapters
                } else {
                    $adapters = Get-CimInstance Win32_NetworkAdapter -Filter 'PhysicalAdapter = True'
                    $adapters | ForEach-Object {
                        $pnp = Get-CimInstance Win32_PnPEntity -Filter "DeviceID = '$($_.PNPDeviceID.Replace('\', '\\'))'"
                        @{ Name = $_.Name; Driver = $pnp.DriverVersion; Provider = $pnp.Manufacturer }
                    }
                }
            }
            'USB' {
                if ($Detail -ne 'DriverVersion') {
                    if (-not (Get-Command Get-UsbStatus -ErrorAction SilentlyContinue)) {
                        Import-Module PC-AI.Hardware -ErrorAction SilentlyContinue
                    }
                    if (-not (Get-Command Get-UsbDeviceList -ErrorAction SilentlyContinue)) {
                        Import-Module PC-AI.USB -ErrorAction SilentlyContinue
                    }
                }

                if ($Detail -eq 'DriverVersion') {
                    $devices = Get-CimInstance Win32_PnPEntity |
                        Where-Object { $_.PNPClass -eq 'USB' -or $_.Name -like '*USB*' }
                    $devices | ForEach-Object {
                        @{ Name = $_.Name; Driver = $_.DriverVersion; Provider = $_.Manufacturer; Status = $_.Status }
                    }
                } elseif ($Detail -eq 'FullStatus' -and (Get-Command Get-UsbStatus -ErrorAction SilentlyContinue)) {
                    Get-UsbStatus
                } elseif (Get-Command Get-UsbDeviceList -ErrorAction SilentlyContinue) {
                    Get-UsbDeviceList
                } else {
                    $hubs = Get-CimInstance Win32_USBHub
                    $hubs | Select-Object Name, DeviceID, Status
                }
            }
            'BIOS' {
                Get-CimInstance Win32_BIOS | Select-Object Manufacturer, Version, ReleaseDate
            }
            'OS' {
                Get-CimInstance Win32_OperatingSystem | Select-Object Caption, Version, BuildNumber, OSArchitecture
            }
            'GPU' {
                $cim = @()
                $nvidia = @()
                $health = $null

                try {
                    $cim = Get-CimInstance Win32_VideoController | Select-Object Name, Status, DriverVersion, AdapterRAM, VideoProcessor
                } catch { }

                $smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
                if ($smi) {
                    try {
                        $lines = & $smi.Source '--query-gpu=name,driver_version,pci.bus_id,memory.total,memory.used,utilization.gpu' '--format=csv,noheader,nounits' 2>$null
                        if ($lines) {
                            foreach ($line in @($lines)) {
                                $parts = $line -split '\s*,\s*'
                                if ($parts.Count -ge 6) {
                                    $nvidia += [PSCustomObject]@{
                                        Name = $parts[0]
                                        DriverVersion = $parts[1]
                                        PciBusId = $parts[2]
                                        MemoryTotalMB = $parts[3]
                                        MemoryUsedMB = $parts[4]
                                        UtilizationGpu = $parts[5]
                                    }
                                }
                            }
                        }
                    } catch { }
                }

                if (Get-Command Get-PcaiServiceHealth -ErrorAction SilentlyContinue) {
                    try {
                        $health = (Get-PcaiServiceHealth).Gpu
                    } catch { }
                }

                if ($Detail -eq 'DriverVersion') {
                    $cim | Select-Object Name, DriverVersion
                } else {
                    [PSCustomObject]@{
                        Cim = $cim
                        NvidiaSmi = $nvidia
                        ServiceHealth = $health
                    }
                }
            }
            'Media' {
                $audio = Get-CimInstance Win32_SoundDevice
                if ($Detail -eq 'DriverVersion') {
                    $audio | Select-Object Name, DriverVersion, Status, Manufacturer
                } else {
                    $audio | Select-Object Name, Status, Manufacturer
                }
            }
            'HIDClass' {
                $hid = Get-CimInstance Win32_PnPEntity -Filter "PNPClass = 'HIDClass'"
                if ($Detail -eq 'DriverVersion') {
                    $hid | Select-Object Name, DriverVersion, Status, Manufacturer
                } else {
                    $hid | Select-Object Name, Status, Manufacturer
                }
            }
        }

        return $result | ConvertTo-Json -Depth 5
    } catch {
        return @{ Error = $_.Exception.Message; Category = $Category } | ConvertTo-Json
    }
}
