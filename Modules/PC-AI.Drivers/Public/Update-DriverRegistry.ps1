#Requires -Version 5.1
<#
.SYNOPSIS
    Adds or updates a device entry in the driver registry JSON.

.DESCRIPTION
    Loads the registry JSON, locates the entry whose "id" matches $DeviceId (or
    creates a new entry if none exists), updates only the fields that were explicitly
    provided, sets lastUpdated to the current UTC timestamp, and writes the file back.

    Fields that are omitted (empty string or not supplied) are left unchanged in
    existing entries. For brand-new entries, omitted fields are written as $null.

.PARAMETER DeviceId
    Registry id for the device (e.g. 'realtek-rtl8156'). Mandatory.

.PARAMETER Name
    Human-readable device name.

.PARAMETER Category
    Category slug (e.g. 'network', 'hub', 'bluetooth').

.PARAMETER Vid
    Vendor ID (4 hex chars, e.g. '0BDA').

.PARAMETER Pid
    Product ID (4 hex chars, e.g. '8156').

.PARAMETER LatestVersion
    Version string for the latest known driver.

.PARAMETER DownloadUrl
    Direct download URL for the installer.

.PARAMETER ReleaseDate
    Release date string (ISO 8601 preferred, e.g. '2026-02-27').

.PARAMETER RegistryPath
    Full path to an alternate driver-registry.json. When omitted the default path
    under PC_AI\Config\ is resolved automatically.

.EXAMPLE
    Update-DriverRegistry -DeviceId 'realtek-rtl8156' -LatestVersion '11.23.1.0'
    Updates only the latestVersion field for the existing entry.

.EXAMPLE
    Update-DriverRegistry -DeviceId 'new-device' -Name 'My Device' -Category 'network' `
        -Vid '1234' -Pid 'ABCD'
    Creates a new minimal entry.

.OUTPUTS
    None. Writes back to the registry JSON file.
#>
function Update-DriverRegistry {
    [CmdletBinding(SupportsShouldProcess)]
    [OutputType([void])]
    param(
        [Parameter(Mandatory)]
        [string]$DeviceId,

        [Parameter()]
        [string]$Name,

        [Parameter()]
        [string]$Category,

        [Parameter()]
        [string]$Vid,

        [Parameter()]
        [string]$Pid,

        [Parameter()]
        [string]$LatestVersion,

        [Parameter()]
        [string]$DownloadUrl,

        [Parameter()]
        [string]$ReleaseDate,

        [Parameter()]
        [string]$RegistryPath
    )

    try {
        # --- Resolve registry path ---
        if (-not $RegistryPath) {
            $moduleRoot   = $script:ModuleRoot
            $modulesDir   = Split-Path $moduleRoot  -Parent
            $pcAiRoot     = Split-Path $modulesDir   -Parent
            $RegistryPath = Join-Path $pcAiRoot 'Config\driver-registry.json'
        }

        if (-not (Test-Path -LiteralPath $RegistryPath)) {
            Write-Error "Driver registry not found at: $RegistryPath"
            return
        }

        $raw      = [System.IO.File]::ReadAllText($RegistryPath)
        $registry = $raw | ConvertFrom-Json

        if (-not $registry.devices) {
            Write-Error "Registry file is missing the 'devices' array: $RegistryPath"
            return
        }

        # --- Find or create the entry ---
        $isNew = $false
        $entry = $registry.devices | Where-Object { $_.id -eq $DeviceId } | Select-Object -First 1

        if (-not $entry) {
            Write-Verbose "Update-DriverRegistry: '$DeviceId' not found - creating new entry."
            $isNew = $true

            # Build a minimal new entry matching the registry schema
            $entry = [PSCustomObject]@{
                id              = $DeviceId
                name            = $null
                category        = $null
                matchRules      = @()
                driver          = [PSCustomObject]@{
                    sourceId        = $null
                    latestVersion   = $null
                    releaseDate     = $null
                    certification   = $null
                    downloadUrl     = $null
                    manualDownloadUrl = $null
                    installerType   = 'exe'
                    sha256          = $null
                    notes           = $null
                }
                sharedDriverGroup = $null
            }

            # Seed matchRules from VID/PID if provided
            if ($Vid -and $Pid) {
                $rule = [PSCustomObject]@{ type = 'vid_pid'; vid = $Vid.ToUpper(); pid = $Pid.ToUpper() }
                $entry.matchRules = @($rule)
            }
            elseif ($Vid) {
                $rule = [PSCustomObject]@{ type = 'vid'; vid = $Vid.ToUpper() }
                $entry.matchRules = @($rule)
            }
        }

        # --- Apply provided (non-empty) fields ---
        if ($Name)          { $entry.name            = $Name }
        if ($Category)      { $entry.category        = $Category }
        if ($LatestVersion) { $entry.driver.latestVersion  = $LatestVersion }
        if ($DownloadUrl)   { $entry.driver.downloadUrl    = $DownloadUrl }
        if ($ReleaseDate)   { $entry.driver.releaseDate    = $ReleaseDate }

        # Update VID/PID match rules on existing entries only when explicitly supplied
        # (avoids destroying existing multi-rule entries on a partial update).
        if (-not $isNew) {
            if ($Vid -and $Pid) {
                # Replace any existing vid_pid rule that matches this VID to keep
                # things tidy; leave other rule types (friendly_name, etc.) intact.
                $otherRules = @($entry.matchRules | Where-Object { $_.type -ne 'vid_pid' })
                $vidPidRule = [PSCustomObject]@{ type = 'vid_pid'; vid = $Vid.ToUpper(); pid = $Pid.ToUpper() }
                $entry.matchRules = @($otherRules) + @($vidPidRule)
            }
        }

        # --- Set lastUpdated on the registry root ---
        $nowUtc = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
        $registry.lastUpdated = $nowUtc

        # --- Splice the entry back into the devices array ---
        if ($isNew) {
            # ConvertFrom-Json gives us a fixed-size PSObject array; rebuild as a list
            $list = [System.Collections.Generic.List[object]]::new()
            foreach ($d in $registry.devices) { $list.Add($d) }
            $list.Add($entry)
            $registry.devices = $list.ToArray()
        }
        # (Existing entries are modified by reference through $entry.)

        # --- Write back ---
        if ($PSCmdlet.ShouldProcess($RegistryPath, "Update registry entry '$DeviceId'")) {
            $json = $registry | ConvertTo-Json -Depth 10
            [System.IO.File]::WriteAllText($RegistryPath, $json, [System.Text.Encoding]::UTF8)
            Write-Verbose "Update-DriverRegistry: wrote '$DeviceId' to $RegistryPath (lastUpdated=$nowUtc)"
        }

    }
    catch {
        Write-Error "Update-DriverRegistry failed: $($_.Exception.Message)"
    }
}
