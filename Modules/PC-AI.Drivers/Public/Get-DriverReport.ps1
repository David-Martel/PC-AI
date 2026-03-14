#Requires -Version 5.1
<#
.SYNOPSIS
    Generates a consolidated driver status report for all matched PnP devices.

.DESCRIPTION
    Orchestrates Get-PnpDeviceInventory, Get-DriverRegistry, and Compare-DriverVersion
    into a single call. Returns one row per device with its comparison status, installed
    version, target version, and registry metadata.

    By default all non-Unknown, non-Current statuses are returned. Use -OnlyActionable
    to restrict to Outdated and NoDriver items, or -IncludeUnknown to include devices
    that have no matching registry entry.

.PARAMETER RegistryPath
    Full path to an alternate driver-registry.json. When omitted the default path
    under PC_AI\Config\ is resolved automatically.

.PARAMETER Category
    Filter the registry (and therefore the results) to a single category such as
    'network', 'hub', or 'bluetooth'.

.PARAMETER OnlyActionable
    When specified, only Outdated and NoDriver items are returned. Current, NoUpdate,
    and Unknown rows are suppressed.

.PARAMETER IncludeUnknown
    When specified, devices with no matching registry entry (status Unknown) are
    included in the output.

.EXAMPLE
    Get-DriverReport
    Returns all non-current, non-unknown matched devices.

.EXAMPLE
    Get-DriverReport -OnlyActionable
    Returns only devices that need a driver download or install action.

.EXAMPLE
    Get-DriverReport -Category 'network' -IncludeUnknown
    Returns network-category devices including those not in the registry.

.OUTPUTS
    PSCustomObject[] with properties: Status, Category, DeviceName, VID, PID,
    PnpClass, InstalledVersion, TargetVersion, RegistryId, SourceId, Notes
#>
function Get-DriverReport {
    [CmdletBinding()]
    [OutputType([PSCustomObject[]])]
    param(
        [Parameter()]
        [string]$RegistryPath,

        [Parameter()]
        [string]$Category,

        [Parameter()]
        [switch]$OnlyActionable,

        [Parameter()]
        [switch]$IncludeUnknown
    )

    try {
        # Load registry first so we can pre-filter the inventory scan to only
        # devices that could match a registry entry (massive performance win).
        Write-Verbose "Get-DriverReport: loading driver registry (Category='$Category') ..."

        $regParams = @{}
        if ($RegistryPath) { $regParams['RegistryPath'] = $RegistryPath }
        if ($Category)     { $regParams['Category']     = $Category     }
        $reg = Get-DriverRegistry @regParams

        if (-not $reg) {
            Write-Error 'Get-DriverReport: failed to load driver registry.'
            return @()
        }

        # Build a lightweight PnP device list (no property queries) and pre-filter
        # to only devices that match at least one registry entry. This avoids
        # calling Get-PnpDeviceProperty on hundreds of unrelated devices.
        Write-Verbose 'Get-DriverReport: performing targeted device scan ...'

        $vidPids = @()
        $namePatterns = @()
        foreach ($entry in $reg.Devices) {
            if (-not $entry.matchRules) { continue }
            foreach ($rule in $entry.matchRules) {
                if ($rule.type -eq 'vid_pid') {
                    $vidPids += "VID_$($rule.vid)&PID_$($rule.pid)"
                }
                elseif ($rule.type -eq 'vid') {
                    $vidPids += "VID_$($rule.vid)"
                }
                elseif ($rule.type -eq 'friendly_name') {
                    $namePatterns += $rule.pattern
                }
            }
        }

        # Get raw device list (fast — no property queries)
        $allDevices = @(Get-PnpDevice -Status OK -ErrorAction SilentlyContinue)

        # Filter to devices matching any registry rule
        $matchedDeviceIds = @()
        foreach ($dev in $allDevices) {
            $matched = $false
            foreach ($vp in $vidPids) {
                if ($dev.InstanceId -like "*$vp*") { $matched = $true; break }
            }
            if (-not $matched) {
                foreach ($pat in $namePatterns) {
                    if ($dev.FriendlyName -like $pat) { $matched = $true; break }
                }
            }
            if ($matched) {
                $matchedDeviceIds += $dev.InstanceId
            }
        }

        Write-Verbose "Get-DriverReport: $($matchedDeviceIds.Count) device(s) match registry rules (from $($allDevices.Count) total)."

        # Build targeted inventory objects for matched devices only (avoids
        # full PnP property scan which takes minutes on systems with 200+ devices).
        $inv = @()
        foreach ($mid in $matchedDeviceIds) {
            $dev = $allDevices | Where-Object { $_.InstanceId -eq $mid } | Select-Object -First 1
            if (-not $dev) { continue }

            $hwParsed = Resolve-HardwareId -HardwareId $mid

            $driverVersion = $null
            $driverDate    = $null
            $manufacturer  = $null

            # Suppress WhatIf/Confirm propagation: these are read-only property queries
            $savedWhatIf = $WhatIfPreference
            $savedConfirm = $ConfirmPreference
            try {
                $WhatIfPreference = $false
                $ConfirmPreference = 'None'
                $vProp = Get-PnpDeviceProperty -InstanceId $mid -KeyName 'DEVPKEY_Device_DriverVersion' -ErrorAction SilentlyContinue
                if ($vProp -and $vProp.Data) { $driverVersion = $vProp.Data }

                $dProp = Get-PnpDeviceProperty -InstanceId $mid -KeyName 'DEVPKEY_Device_DriverDate' -ErrorAction SilentlyContinue
                if ($dProp -and $dProp.Data) { $driverDate = $dProp.Data }

                $mProp = Get-PnpDeviceProperty -InstanceId $mid -KeyName 'DEVPKEY_Device_Manufacturer' -ErrorAction SilentlyContinue
                if ($mProp -and $mProp.Data) { $manufacturer = $mProp.Data }
            } finally {
                $WhatIfPreference = $savedWhatIf
                $ConfirmPreference = $savedConfirm
            }

            $inv += [PSCustomObject]@{
                Name          = $dev.FriendlyName
                PnpClass      = $dev.Class
                VID           = $hwParsed.VID
                PID           = $hwParsed.PID
                Bus           = $hwParsed.Bus
                InstanceId    = $mid
                DriverVersion = $driverVersion
                DriverDate    = $driverDate
                Manufacturer  = $manufacturer
                Status        = $dev.Status
            }
        }

        if ($inv.Count -eq 0 -and -not $IncludeUnknown) {
            Write-Verbose 'No matching active PnP devices found in registry.'
            return @()
        }

        # IncludeUpToDate is the inverse of OnlyActionable.
        # IncludeUnknown is passed through directly.
        $compareParams = @{
            Inventory      = $inv
            Registry       = $reg
            IncludeUpToDate = (-not $OnlyActionable.IsPresent)
            IncludeUnknown  = $IncludeUnknown.IsPresent
        }
        $results = @(Compare-DriverVersion @compareParams)

        Write-Verbose "Get-DriverReport: $($results.Count) result(s) before sort."

        if ($results.Count -eq 0) {
            return @()
        }

        # Sort: Outdated first, then NoDriver, then remaining statuses; secondary by
        # Category and DeviceName so the table reads cleanly.
        $statusOrder = @{
            'Outdated'    = 0
            'ManualCheck' = 1
            'NoDriver'    = 2
            'NoUpdate'    = 3
            'Current'     = 4
            'Unknown'     = 5
        }

        $sorted = $results | Sort-Object -Property `
            @{ Expression = { $order = $statusOrder[$_.Status]; if ($null -eq $order) { 9 } else { $order } } },
            @{ Expression = 'Category' },
            @{ Expression = 'DeviceName' }

        return @($sorted)

    }
    catch {
        Write-Error "Get-DriverReport failed: $($_.Exception.Message)"
        return @()
    }
}
