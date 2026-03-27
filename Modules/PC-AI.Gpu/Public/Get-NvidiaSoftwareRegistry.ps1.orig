#Requires -Version 5.1
<#
.SYNOPSIS
    Loads the PC-AI NVIDIA software registry JSON and returns it as a structured
    object, with optional filtering by component ID or category.

.DESCRIPTION
    Reads nvidia-software-registry.json from the default location
    (Config\nvidia-software-registry.json relative to the PC_AI root, derived
    from the module's $script:ModuleRoot), or from an explicit path supplied via
    -RegistryPath.

    The full registry object is returned with its .components array optionally
    filtered by -ComponentId or -Category. The returned object preserves the
    original structure — trustedSources, categories, and the (filtered)
    components array — so callers can pass it directly to Get-NvidiaSoftwareStatus
    or inspect registry metadata.

.PARAMETER RegistryPath
    Full path to an alternate nvidia-software-registry.json file. When omitted
    the function resolves the path automatically from the module root.

.PARAMETER ComponentId
    Filter the .components array to the single entry whose "id" property matches
    this value exactly (e.g. 'cuda', 'cudnn', 'tensorrt').

.PARAMETER Category
    Filter the .components array to entries whose "category" property matches
    this value exactly (e.g. 'toolkit', 'profiling', 'inference').

.OUTPUTS
    PSCustomObject representing the registry: .Version, .LastUpdated,
    .TrustedSources, .Categories, .Components (filtered when -ComponentId or
    -Category is specified).

.EXAMPLE
    Get-NvidiaSoftwareRegistry
    Loads and returns the full NVIDIA software registry.

.EXAMPLE
    Get-NvidiaSoftwareRegistry -Category 'toolkit'
    Returns only toolkit-category component entries.

.EXAMPLE
    Get-NvidiaSoftwareRegistry -ComponentId 'cuda'
    Returns the registry filtered to the CUDA Toolkit entry.

.EXAMPLE
    Get-NvidiaSoftwareRegistry -RegistryPath 'C:\Custom\my-registry.json'
    Loads a registry file from a custom path.
#>
function Get-NvidiaSoftwareRegistry {
    [CmdletBinding()]
    [OutputType([PSCustomObject])]
    param(
        [Parameter()]
        [string]$RegistryPath,

        [Parameter()]
        [string]$ComponentId,

        [Parameter()]
        [string]$Category
    )

    $ErrorActionPreference = 'Stop'

    try {
        # --- Resolve registry file path ---
        if (-not $RegistryPath) {
            # $script:ModuleRoot is set to $PSScriptRoot in PC-AI.Gpu.psm1 (the Modules\PC-AI.Gpu\ dir).
            # Navigate up two levels: PC-AI.Gpu -> Modules -> PC_AI, then into Config\.
            $moduleRoot   = $script:ModuleRoot
            $modulesDir   = Split-Path $moduleRoot  -Parent   # Modules\
            $pcAiRoot     = Split-Path $modulesDir   -Parent   # PC_AI\
            $RegistryPath = Join-Path $pcAiRoot 'Config\nvidia-software-registry.json'
        }

        Write-Verbose "Loading NVIDIA software registry from: $RegistryPath"

        if (-not (Test-Path -LiteralPath $RegistryPath)) {
            Write-Error "NVIDIA software registry not found at path: $RegistryPath"
            return $null
        }

        $raw      = [System.IO.File]::ReadAllText($RegistryPath)
        $registry = $raw | ConvertFrom-Json

        if (-not $registry.components) {
            Write-Error "Registry file is missing the 'components' array: $RegistryPath"
            return $null
        }

        # --- Apply filters to .components ---
        $filtered = @($registry.components)

        if ($ComponentId) {
            Write-Verbose "Filtering registry to ComponentId: $ComponentId"
            $filtered = @($filtered | Where-Object { $_.id -eq $ComponentId })
        }

        if ($Category) {
            Write-Verbose "Filtering registry to Category: $Category"
            $filtered = @($filtered | Where-Object { $_.category -eq $Category })
        }

        Write-Verbose "Registry loaded: $($filtered.Count) component entry/entries matched."

        # Return a new object so we do not mutate the deserialized original
        return [PSCustomObject]@{
            Version        = $registry.version
            LastUpdated    = $registry.lastUpdated
            TrustedSources = $registry.trustedSources
            Categories     = $registry.categories
            Components     = $filtered
        }
    }
    catch {
        Write-Error "Get-NvidiaSoftwareRegistry failed: $($_.Exception.Message)"
        return $null
    }
}
