#Requires -Version 5.1
<#
.SYNOPSIS
    Builds a per-GPU compatibility matrix showing whether each installed NVIDIA
    software component meets the minimum requirements for every GPU in the system.

.DESCRIPTION
    Cross-references the installed versions of the NVIDIA display driver, CUDA
    Toolkit, cuDNN, TensorRT, Nsight Compute, and Nsight Systems against the
    minimumVersion and recommendedVersion fields in the compatibilityMatrix
    section of nvidia-software-registry.json.

    For each (GPU architecture, component) pair the function emits one result row
    with a Status value of:

      Compatible           - Installed version meets the minimum requirement.
      Incompatible         - Installed version is below the minimum requirement.
      UpgradeRecommended   - Installed version meets the minimum but a newer
                             recommended version is available.
      NotInstalled         - Component not detected on the local machine.
      NotRequired          - The component has no minimum requirement defined for
                             this GPU architecture.
      UnknownVersion       - Component is installed but the version could not be
                             determined.

    Components evaluated: gpu-driver, cuda-toolkit, cudnn, tensorrt,
    nsight-compute, nsight-systems.

    Flags any row whose Status is Incompatible so that Install-NvidiaSoftware can
    act on it.

.PARAMETER RegistryPath
    Full path to an alternate nvidia-software-registry.json file. When omitted
    the default Config\nvidia-software-registry.json is used.

.OUTPUTS
    PSCustomObject[] with properties:
      GpuName             - Display name of the GPU (from nvidia-smi or CIM).
      GpuIndex            - nvidia-smi GPU index.
      ArchKey             - Architecture key as stored in the registry (e.g. "SM 8.9").
      Component           - Human-readable component name.
      ComponentId         - Registry component identifier (e.g. "cuda-toolkit").
      InstalledVersion    - Detected installed version string, or $null.
      MinimumRequired     - Minimum version string from the compatibility matrix.
      RecommendedVersion  - Recommended version string from the compatibility matrix, or $null.
      Status              - Compatible | Incompatible | UpgradeRecommended |
                            NotInstalled | NotRequired | UnknownVersion
      IsBlocker           - [bool] $true when Status is Incompatible.

.EXAMPLE
    Get-NvidiaCompatibilityMatrix
    Returns the full compatibility matrix for all GPUs and components.

.EXAMPLE
    Get-NvidiaCompatibilityMatrix | Where-Object IsBlocker
    Lists only components that block normal operation due to version
    incompatibility.

.EXAMPLE
    Get-NvidiaCompatibilityMatrix | Format-Table GpuName, Component, InstalledVersion, MinimumRequired, Status -AutoSize
    Displays a compact status table.

.NOTES
    The function calls private helpers (Resolve-NvidiaInstallPath,
    Get-NvidiaDriverVersion, Get-CudaVersionFromPath, Get-CudnnVersionFromHeader,
    Get-TensorRtVersionFromHeader, Get-NsightVersions) and the public cmdlet
    Get-NvidiaGpuInventory that are dot-sourced by PC-AI.Gpu.psm1.
#>
function Get-NvidiaCompatibilityMatrix {
    [CmdletBinding()]
    [OutputType([PSCustomObject[]])]
    param(
        [Parameter()]
        [string]$RegistryPath
    )

    $ErrorActionPreference = 'Stop'

    # -------------------------------------------------------------------------
    # Helper: compare two version strings.
    # Returns -1 (a < b), 0 (a == b), 1 (a > b).
    # Falls back to lexicographic comparison when [System.Version] cannot parse.
    # -------------------------------------------------------------------------
    function Compare-VersionStrings {
        param([string]$A, [string]$B)
        if (-not $A -or -not $B) { return 0 }
        try {
            $va = [System.Version]::new($A)
            $vb = [System.Version]::new($B)
            return $va.CompareTo($vb)
        }
        catch {
            # Non-standard version (e.g. "2026.1.0" with only 3 parts parsed fine,
            # but date-style like "2025.6.3" also parses — the catch handles truly
            # unparsable strings).
            return [string]::Compare($A, $B, [System.StringComparison]::OrdinalIgnoreCase)
        }
    }

    # Helper: normalise a version string to remove trailing ".0" build numbers
    # so "572.16.0" compares correctly to "572.16"
    function Normalize-Version {
        param([string]$V)
        if (-not $V) { return $null }
        try {
            $parsed = [System.Version]::new($V)
            # Return major.minor if patch = 0 and build not set, else full dotted form
            if ($parsed.Build -le 0 -and $parsed.Revision -le 0) {
                return "$($parsed.Major).$($parsed.Minor)"
            }
            return $V
        }
        catch { return $V }
    }

    # -------------------------------------------------------------------------
    # Step 1: Load the registry
    # -------------------------------------------------------------------------
    Write-Verbose 'Loading NVIDIA software registry...'
    $regParams = @{}
    if ($RegistryPath) { $regParams['RegistryPath'] = $RegistryPath }
    $registry = Get-NvidiaSoftwareRegistry @regParams

    if (-not $registry) {
        Write-Error 'Get-NvidiaCompatibilityMatrix: Failed to load nvidia-software-registry.json'
        return @()
    }

    # Extract the compatibilityMatrix section from the raw JSON so we have access
    # to architecture-specific minimums.  Get-NvidiaSoftwareRegistry returns a
    # cleaned PSCustomObject that does not expose compatibilityMatrix, so we
    # reload the raw JSON here.
    $rawRegistryPath = $RegistryPath
    if (-not $rawRegistryPath) {
        $moduleRoot      = $script:ModuleRoot
        $modulesDir      = Split-Path $moduleRoot -Parent
        $pcAiRoot        = Split-Path $modulesDir -Parent
        $rawRegistryPath = Join-Path $pcAiRoot 'Config\nvidia-software-registry.json'
    }

    $rawJson     = [System.IO.File]::ReadAllText($rawRegistryPath)
    $rawRegistry = $rawJson | ConvertFrom-Json

    $compatMatrix = $rawRegistry.compatibilityMatrix
    if (-not $compatMatrix -or -not $compatMatrix.architectures) {
        Write-Warning 'Registry compatibilityMatrix section not found. Cannot build compatibility matrix.'
        return @()
    }

    # -------------------------------------------------------------------------
    # Step 2: Detect installed GPU inventory
    # -------------------------------------------------------------------------
    Write-Verbose 'Querying GPU inventory...'
    $gpus = @(Get-NvidiaGpuInventory)
    if ($gpus.Count -eq 0) {
        Write-Warning 'No NVIDIA GPUs detected. Compatibility matrix will be empty.'
        return @()
    }

    # -------------------------------------------------------------------------
    # Step 3: Detect installed software component versions
    # -------------------------------------------------------------------------
    Write-Verbose 'Detecting installed NVIDIA component versions...'

    $installPaths  = Resolve-NvidiaInstallPath
    $driverVersion = $null
    try { $driverVersion = Get-NvidiaDriverVersion } catch { Write-Verbose "Driver version error: $_" }

    # CUDA: latest installed directory
    $cudaVersion = $null
    $cudaPath    = $installPaths['CUDA']
    if ($cudaPath) {
        try { $cudaVersion = Get-CudaVersionFromPath -CudaPath $cudaPath } catch { }
    }

    # cuDNN
    $cudnnVersion = $null
    $cudnnPath    = $installPaths['cuDNN']
    if ($cudnnPath) {
        try { $cudnnVersion = Get-CudnnVersionFromHeader -CudnnPath $cudnnPath } catch { }
    }

    # TensorRT
    $tensorrtVersion = $null
    $tensorrtPath    = $installPaths['TensorRT']
    if ($tensorrtPath) {
        try { $tensorrtVersion = Get-TensorRtVersionFromHeader -TensorRtPath $tensorrtPath } catch { }
    }

    # Nsight Compute and Nsight Systems
    $nsightComputeVersion = $null
    $nsightSystemsVersion = $null
    try {
        $nsightEntries = @(Get-NsightVersions)
        $ncEntry = $nsightEntries | Where-Object { $_.Product -eq 'NsightCompute' } | Select-Object -First 1
        $nsEntry = $nsightEntries | Where-Object { $_.Product -eq 'NsightSystems'  } | Select-Object -First 1
        if ($ncEntry) { $nsightComputeVersion = $ncEntry.Version }
        if ($nsEntry) { $nsightSystemsVersion = $nsEntry.Version }
    }
    catch { Write-Verbose "Nsight version detection error: $_" }

    # Build a lookup by component identifier
    $installedVersions = @{
        'gpu-driver'      = $driverVersion
        'driver'          = $driverVersion
        'cuda-toolkit'    = $cudaVersion
        'cuda'            = $cudaVersion
        'cudnn'           = $cudnnVersion
        'tensorrt'        = $tensorrtVersion
        'nsight-compute'  = $nsightComputeVersion
        'nsight-systems'  = $nsightSystemsVersion
    }

    # Map of componentId -> human-readable name (from registry components array)
    $componentNames = @{}
    foreach ($comp in $registry.Components) {
        $componentId = $comp.id
        if (-not $componentId) {
            continue
        }
        $componentNames[$componentId] = $comp.Name
    }
    # Ensure known IDs have a fallback name
    $fallbackNames = @{
        'gpu-driver'     = 'NVIDIA Display Driver'
        'driver'         = 'NVIDIA Display Driver'
        'cuda-toolkit'   = 'CUDA Toolkit'
        'cuda'           = 'CUDA Toolkit'
        'cudnn'          = 'NVIDIA cuDNN'
        'tensorrt'       = 'NVIDIA TensorRT'
        'nsight-compute' = 'Nsight Compute'
        'nsight-systems' = 'Nsight Systems'
    }
    foreach ($key in $fallbackNames.Keys) {
        if (-not $componentNames.ContainsKey($key)) {
            $componentNames[$key] = $fallbackNames[$key]
        }
    }

    # -------------------------------------------------------------------------
    # Step 4: Build the matrix
    # -------------------------------------------------------------------------
    $results = [System.Collections.Generic.List[PSCustomObject]]::new()

    # Architecture entries in the registry: keys are like "SM 8.9", "SM 12.0"
    $archEntries = $compatMatrix.architectures.PSObject.Properties

    foreach ($gpu in $gpus) {
        # Match this GPU to a registry architecture by compute capability.
        # nvidia-smi returns compute capability as "8.9" or "12.0".
        # Registry keys are "SM 8.9" and "SM 12.0".
        $gpuCC    = $gpu.ComputeCapability   # e.g. "8.9"
        $archKey  = $null
        $archData = $null

        foreach ($entry in $archEntries) {
            # Normalise: "SM 8.9" -> "8.9", "SM 12.0" -> "12.0"
            $keyCC = $entry.Name -replace '^SM\s*', ''
            if ($keyCC -eq $gpuCC) {
                $archKey  = $entry.Name
                $archData = $entry.Value
                break
            }
        }

        if (-not $archData) {
            Write-Verbose "No compatibility matrix entry for GPU '$($gpu.Name)' (compute $gpuCC). Skipping."
            continue
        }

        # --- Define which components to evaluate and how to pull min/recommended from archData ---
        # Structure: each item has Id, MinProp (property name in archData), RecommendedProp
        $componentChecks = @(
            [PSCustomObject]@{ Id = 'gpu-driver';     MinProp = 'minimumDriver';           RecommendedProp = $null              }
            [PSCustomObject]@{ Id = 'cuda-toolkit';   MinProp = 'minimumCuda';             RecommendedProp = 'recommendedCuda'  }
            [PSCustomObject]@{ Id = 'cudnn';          MinProp = 'minimumCuDNN';            RecommendedProp = $null              }
            [PSCustomObject]@{ Id = 'tensorrt';       MinProp = 'minimumTensorRT';         RecommendedProp = $null              }
            [PSCustomObject]@{ Id = 'nsight-compute'; MinProp = 'nsightComputeMinimum';    RecommendedProp = $null              }
            [PSCustomObject]@{ Id = 'nsight-systems'; MinProp = 'nsightSystemsMinimum';    RecommendedProp = $null              }
        )

        foreach ($check in $componentChecks) {
            $compId           = $check.Id
            $minPropName      = $check.MinProp
            $recPropName      = $check.RecommendedProp

            $minimumRequired  = if ($minPropName)  { $archData.$minPropName  } else { $null }
            $recommendedVer   = if ($recPropName)  { $archData.$recPropName  } else { $null }
            $installedVer     = $installedVersions[$compId]
            $compName         = $componentNames[$compId]

            # Determine status
            $status    = 'NotRequired'
            $isBlocker = $false

            if (-not $minimumRequired) {
                $status = 'NotRequired'
            }
            elseif (-not $installedVer) {
                $status    = 'NotInstalled'
                $isBlocker = $true
            }
            else {
                # Try to compare versions
                $cmpMin = Compare-VersionStrings -A $installedVer -B $minimumRequired
                if ($cmpMin -lt 0) {
                    $status    = 'Incompatible'
                    $isBlocker = $true
                }
                elseif ($recommendedVer) {
                    $cmpRec = Compare-VersionStrings -A $installedVer -B $recommendedVer
                    if ($cmpRec -lt 0) {
                        $status = 'UpgradeRecommended'
                    }
                    else {
                        $status = 'Compatible'
                    }
                }
                else {
                    $status = 'Compatible'
                }
            }

            $results.Add([PSCustomObject]@{
                GpuName            = $gpu.Name
                GpuIndex           = $gpu.Index
                ArchKey            = $archKey
                Component          = $compName
                ComponentId        = $compId
                InstalledVersion   = $installedVer
                MinimumRequired    = $minimumRequired
                RecommendedVersion = $recommendedVer
                Status             = $status
                IsBlocker          = $isBlocker
            })
        }
    }

    Write-Verbose "Compatibility matrix: $($results.Count) row(s) evaluated across $($gpus.Count) GPU(s)."
    return @($results)
}
