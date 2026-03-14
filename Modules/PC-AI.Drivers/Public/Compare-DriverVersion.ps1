#Requires -Version 5.1
<#
.SYNOPSIS
    Compares installed driver versions against registry target versions and classifies
    each device.

.DESCRIPTION
    Iterates over each device in $Inventory and attempts to find a matching entry in
    $Registry.Devices using the registry entry's matchRules array. Three rule types are
    supported:

        vid_pid       - Matches when both $device.VID and $device.PID equal the rule's
                        "vid" and "pid" values (case-insensitive).
        friendly_name - Matches when $device.Name satisfies a -like wildcard comparison
                        against the rule's "pattern" value.
        vid           - Matches when $device.VID equals the rule's "vid" value. Used for
                        multi-device vendors (e.g. Intel Bluetooth).

    Version comparison uses [System.Version]::TryParse when both strings are parseable
    as dotted version quads; otherwise falls back to a string equality check.

    Output status values:
        Current   - Installed version matches or is newer than the registry target.
        Outdated  - Installed version is older than the registry target.
        NoDriver  - Device has no installed driver version string.
        NoUpdate  - Registry entry exists but latestVersion is null (no update tracked).
        Unknown   - No matching registry entry found for this device.

.PARAMETER Inventory
    Array of PSCustomObjects as produced by Get-PnpDeviceInventory.

.PARAMETER Registry
    Registry object as produced by Get-DriverRegistry.

.PARAMETER IncludeUpToDate
    When specified, 'Current' status devices are included in the output.
    By default they are suppressed (the common case is acting on actionable results).

.PARAMETER IncludeUnknown
    When specified, 'Unknown' status devices (no registry match) are included.
    By default they are suppressed.

.EXAMPLE
    $inv = Get-PnpDeviceInventory
    $reg = Get-DriverRegistry
    Compare-DriverVersion -Inventory $inv -Registry $reg

    Returns Outdated, NoDriver, and NoUpdate devices only.

.EXAMPLE
    Compare-DriverVersion -Inventory $inv -Registry $reg -IncludeUpToDate -IncludeUnknown
    Returns every device from the inventory with its comparison status.

.OUTPUTS
    PSCustomObject[] with properties: DeviceName, VID, PID, PnpClass,
    InstalledVersion, TargetVersion, RegistryId, Status, Category,
    SourceId, Notes
#>
function Compare-DriverVersion {
    [CmdletBinding()]
    [OutputType([PSCustomObject[]])]
    param(
        [Parameter(Mandatory)]
        [PSCustomObject[]]$Inventory,

        [Parameter(Mandatory)]
        [PSCustomObject]$Registry,

        [Parameter()]
        [switch]$IncludeUpToDate,

        [Parameter()]
        [switch]$IncludeUnknown
    )

    try {
        $results = [System.Collections.Generic.List[PSCustomObject]]::new()

        foreach ($device in $Inventory) {
            # --- Match device against registry entries ---
            $matchedEntry = $null

            foreach ($entry in $Registry.Devices) {
                if (-not $entry.matchRules) { continue }

                $isMatch = $false
                foreach ($rule in $entry.matchRules) {
                    if ($rule.type -eq 'vid_pid') {
                        $ruleVid = if ($rule.vid) { $rule.vid.ToUpper() } else { '' }
                        $rulePid = if ($rule.pid) { $rule.pid.ToUpper() } else { '' }
                        if ($device.VID -and $device.PID -and
                            $device.VID -eq $ruleVid -and
                            $device.PID -eq $rulePid) {
                            $isMatch = $true
                            break
                        }
                    }
                    elseif ($rule.type -eq 'friendly_name') {
                        if ($device.Name -and $device.Name -like $rule.pattern) {
                            $isMatch = $true
                            break
                        }
                    }
                    elseif ($rule.type -eq 'vid') {
                        $ruleVid = if ($rule.vid) { $rule.vid.ToUpper() } else { '' }
                        if ($device.VID -and $device.VID -eq $ruleVid) {
                            $isMatch = $true
                            break
                        }
                    }
                    # pci_class and other future types: not matched here (no PCI class
                    # data in the inventory object); treat as no-match and fall through.
                }

                if ($isMatch) {
                    $matchedEntry = $entry
                    break
                }
            }

            # --- Determine status ---
            $status          = 'Unknown'
            $installedVer    = $device.DriverVersion
            $targetVer       = $null
            $registryId      = $null
            $category        = $null
            $sourceId        = $null
            $notes           = $null

            if ($matchedEntry) {
                $registryId = $matchedEntry.id
                $category   = $matchedEntry.category
                $targetVer  = $matchedEntry.driver.latestVersion
                $sourceId   = $matchedEntry.driver.sourceId
                $notes      = $matchedEntry.driver.notes

                if (-not $installedVer) {
                    $status = 'NoDriver'
                }
                elseif (-not $targetVer) {
                    # Registry tracks the device but deliberately has no target (inbox/WU)
                    $status = 'NoUpdate'
                }
                else {
                    # Check if registry explicitly marks versions as non-comparable
                    # (e.g. firmware version vs Windows driver version)
                    $versionComparable = $true
                    if ($null -ne $matchedEntry.driver.versionComparable) {
                        $versionComparable = [bool]$matchedEntry.driver.versionComparable
                    }

                    if (-not $versionComparable) {
                        $status = 'ManualCheck'
                        Write-Verbose "Version not comparable for '$($device.Name)': installed='$installedVer' target='$targetVer' (versionComparable=false)."
                    }
                    else {
                        # Attempt typed version comparison; fall back to string equality
                        $parsedInstalled = $null
                        $parsedTarget    = $null
                        $installedParsed = [System.Version]::TryParse($installedVer, [ref]$parsedInstalled)
                        $targetParsed    = [System.Version]::TryParse($targetVer,    [ref]$parsedTarget)

                        if ($installedParsed -and $targetParsed) {
                            if ($parsedInstalled -ge $parsedTarget) {
                                $status = 'Current'
                            }
                            else {
                                $status = 'Outdated'
                            }
                        }
                        else {
                            # Non-parseable versions: string comparison
                            if ($installedVer -eq $targetVer) {
                                $status = 'Current'
                            }
                            else {
                                Write-Verbose "Non-comparable versions for '$($device.Name)': installed='$installedVer' target='$targetVer' - reporting ManualCheck."
                                $status = 'ManualCheck'
                            }
                        }
                    }
                }
            }

            # --- Apply output filters ---
            if ($status -eq 'Current' -and -not $IncludeUpToDate) { continue }
            if ($status -eq 'Unknown' -and -not $IncludeUnknown)  { continue }

            $results.Add([PSCustomObject]@{
                DeviceName        = $device.Name
                VID               = $device.VID
                PID               = $device.PID
                PnpClass          = $device.PnpClass
                InstalledVersion  = $installedVer
                TargetVersion     = $targetVer
                RegistryId        = $registryId
                Status            = $status
                Category          = $category
                SourceId          = $sourceId
                Notes             = $notes
            })
        }

        Write-Verbose "Compare-DriverVersion complete: $($results.Count) result(s) after filtering."

        return @($results)

    }
    catch {
        Write-Error "Compare-DriverVersion failed: $($_.Exception.Message)"
        return @()
    }
}
