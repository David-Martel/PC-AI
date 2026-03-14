#Requires -Version 5.1
<#
.SYNOPSIS
    Loads the PC-AI driver registry JSON file and returns it as a structured object.

.DESCRIPTION
    Reads driver-registry.json from the default location (Config\driver-registry.json
    relative to the PC_AI root, derived from the module's $script:ModuleRoot), or from
    an explicit path supplied via -RegistryPath. The full registry object is returned
    with its .devices array optionally filtered by -DeviceId or -Category.

    The returned object preserves the original structure — trustedSources, categories,
    and the (filtered) devices array — so callers can pass it directly to
    Compare-DriverVersion or inspect registry metadata.

.PARAMETER RegistryPath
    Full path to an alternate driver-registry.json file. When omitted, the function
    resolves the path automatically from the module root.

.PARAMETER DeviceId
    Filter the .devices array to the single entry whose "id" property matches this
    value exactly.

.PARAMETER Category
    Filter the .devices array to entries whose "category" property matches this value
    exactly (e.g. 'network', 'hub', 'bluetooth').

.EXAMPLE
    Get-DriverRegistry
    Loads and returns the full registry with all devices.

.EXAMPLE
    Get-DriverRegistry -Category 'network'
    Returns the registry with only network-category device entries.

.EXAMPLE
    Get-DriverRegistry -DeviceId 'realtek-rtl8156'
    Returns the registry filtered to the Realtek RTL8156 entry.

.EXAMPLE
    Get-DriverRegistry -RegistryPath 'C:\Custom\my-registry.json'
    Loads a registry file from a custom path.

.OUTPUTS
    PSCustomObject representing the registry: .version, .lastUpdated,
    .trustedSources, .categories, .devices (filtered when -DeviceId or -Category
    is specified).
#>
function Get-DriverRegistry {
    [CmdletBinding()]
    [OutputType([PSCustomObject])]
    param(
        [Parameter()]
        [string]$RegistryPath,

        [Parameter()]
        [string]$DeviceId,

        [Parameter()]
        [string]$Category
    )

    try {
        # --- Resolve registry file path ---
        if (-not $RegistryPath) {
            # $script:ModuleRoot is set to $PSScriptRoot in PC-AI.Drivers.psm1 (the Modules\PC-AI.Drivers\ dir).
            # Navigate up two levels: PC-AI.Drivers -> Modules -> PC_AI, then into Config\.
            $moduleRoot   = $script:ModuleRoot
            $modulesDir   = Split-Path $moduleRoot  -Parent   # Modules\
            $pcAiRoot     = Split-Path $modulesDir   -Parent   # PC_AI\
            $RegistryPath = Join-Path $pcAiRoot 'Config\driver-registry.json'
        }

        Write-Verbose "Loading driver registry from: $RegistryPath"

        if (-not (Test-Path -LiteralPath $RegistryPath)) {
            Write-Error "Driver registry not found at path: $RegistryPath"
            return $null
        }

        $raw      = [System.IO.File]::ReadAllText($RegistryPath)
        $registry = $raw | ConvertFrom-Json

        if (-not $registry.devices) {
            Write-Error "Registry file is missing the 'devices' array: $RegistryPath"
            return $null
        }

        # --- Apply filters to .devices ---
        $filtered = @($registry.devices)

        if ($DeviceId) {
            Write-Verbose "Filtering registry to DeviceId: $DeviceId"
            $filtered = @($filtered | Where-Object { $_.id -eq $DeviceId })
        }

        if ($Category) {
            Write-Verbose "Filtering registry to Category: $Category"
            $filtered = @($filtered | Where-Object { $_.category -eq $Category })
        }

        Write-Verbose "Registry loaded: $($filtered.Count) device entry/entries matched."

        # Return a new object so we don't mutate the deserialized original
        return [PSCustomObject]@{
            Version        = $registry.version
            LastUpdated    = $registry.lastUpdated
            TrustedSources = $registry.trustedSources
            Categories     = $registry.categories
            Devices        = $filtered
        }

    }
    catch {
        Write-Error "Get-DriverRegistry failed: $($_.Exception.Message)"
        return $null
    }
}
