#Requires -Version 5.1
<#
.SYNOPSIS
    Compares installed NVIDIA software component versions against the curated
    software registry and returns a per-component status table.

.DESCRIPTION
    Orchestrates the private detection functions (Resolve-NvidiaInstallPath,
    Get-CudaVersionFromPath, Get-CudnnVersionFromHeader,
    Get-TensorRtVersionFromHeader, Get-NsightVersions) to build a complete
    picture of what is installed on the local machine, then compares each
    installed version against the latestVersion recorded in
    nvidia-software-registry.json.

    Status values:
      Current      - Installed version matches or is newer than registry latest.
      Outdated     - Installed version is older than registry latest.
      NotInstalled - Component not found on disk.
      Unknown      - Installed but registry has no latestVersion to compare.

.PARAMETER RegistryPath
    Full path to an alternate nvidia-software-registry.json. When omitted the
    default Config\nvidia-software-registry.json is used.

.PARAMETER ComponentId
    Limit the status check to a single component ID (e.g. 'cuda', 'cudnn').

.OUTPUTS
    PSCustomObject[] with properties:
      ComponentId       - Registry component identifier
      Name              - Human-readable component name
      InstalledVersion  - Detected installed version string, or $null
      LatestVersion     - Registry latest version string, or $null
      Status            - Current | Outdated | NotInstalled | Unknown
      Path              - Resolved install path string, or $null
      SideBySideCount   - Count of parallel installs discovered for that component

.EXAMPLE
    Get-NvidiaSoftwareStatus
    Returns status for all components in the registry.

.EXAMPLE
    Get-NvidiaSoftwareStatus -ComponentId 'cuda'
    Returns status for the CUDA Toolkit only.

.EXAMPLE
    Get-NvidiaSoftwareStatus | Where-Object Status -eq 'Outdated'
    Lists only components that have a newer version available.
#>
function Get-NvidiaSoftwareStatus {
    [CmdletBinding()]
    [OutputType([PSCustomObject[]])]
    param(
        [Parameter()]
        [string]$RegistryPath,

        [Parameter()]
        [string]$ComponentId
    )

    $ErrorActionPreference = 'Stop'

    try {
        # --- Load registry ---
        $registryParams = @{}
        if ($RegistryPath)  { $registryParams['RegistryPath'] = $RegistryPath }
        if ($ComponentId)   { $registryParams['ComponentId']  = $ComponentId }

        $registry = Get-NvidiaSoftwareRegistry @registryParams
        if (-not $registry -or -not $registry.Components) {
            Write-Warning 'NVIDIA software registry could not be loaded or contains no components.'
            return @()
        }

        # --- Resolve all install paths once (avoids repeated filesystem scans) ---
        Write-Verbose 'Resolving NVIDIA component install paths...'
        $installPaths = Resolve-NvidiaInstallPath

        # Collect Nsight tool entries indexed by Product type
        Write-Verbose 'Scanning for Nsight installations...'
        $nsightVersions = Get-NsightVersions

        # Detect driver version once
        Write-Verbose 'Querying NVIDIA driver version...'
        $driverVersion = Get-NvidiaDriverVersion

        # --- Detect installed versions per well-known component ID ---
        # Build a lookup: componentId -> (installedVersion, path)
        $detectedVersions = @{}

        # CUDA
        $cudaPath = $installPaths['CUDA']
        if ($cudaPath) {
            # Check for multiple side-by-side CUDA installs
            $cudaRoot = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA'
            $sideByCount = 0
            if ($null -ne $cudaRoot) {
                # Test-Path can return true or false but we should be robust
                try {
                    $sideByCount = @(Get-ChildItem -Path $cudaRoot -Filter 'v*' -ErrorAction SilentlyContinue).Count
                } catch { }
            }
            $cudaVer = Get-CudaVersionFromPath -CudaPath $cudaPath
            $detectedVersions['cuda'] = @{
                InstalledVersion = $cudaVer
                Path             = $cudaPath
                SideByCount      = $sideByCount
            }
            # Registry alias used in enriched registry snapshot
            $detectedVersions['cuda-toolkit'] = $detectedVersions['cuda']
        }
        else {
            $detectedVersions['cuda']         = @{ InstalledVersion = $null; Path = $null; SideByCount = 0 }
            $detectedVersions['cuda-toolkit'] = $detectedVersions['cuda']
        }

        # cuDNN
        $cudnnPath = $installPaths['cuDNN']
        if ($cudnnPath) {
            $cudnnVer = Get-CudnnVersionFromHeader -CudnnPath $cudnnPath
            $detectedVersions['cudnn'] = @{ InstalledVersion = $cudnnVer; Path = $cudnnPath; SideByCount = 0 }
        }
        else {
            $detectedVersions['cudnn'] = @{ InstalledVersion = $null; Path = $null; SideByCount = 0 }
        }

        # TensorRT
        $tensorrtPath = $installPaths['TensorRT']
        if ($tensorrtPath) {
            $tensorrtVer = Get-TensorRtVersionFromHeader -TensorRtPath $tensorrtPath
            $detectedVersions['tensorrt'] = @{ InstalledVersion = $tensorrtVer; Path = $tensorrtPath; SideByCount = 0 }
        }
        else {
            $detectedVersions['tensorrt'] = @{ InstalledVersion = $null; Path = $null; SideByCount = 0 }
        }

        # NVIDIA Display Driver
        $detectedVersions['driver']     = @{ InstalledVersion = $driverVersion; Path = $null; SideByCount = 0 }
        # Registry alias used in enriched registry snapshot
        $detectedVersions['gpu-driver'] = $detectedVersions['driver']

        # Nsight Compute
        $nsightComputeEntries = @($nsightVersions | Where-Object { $_.Product -eq 'NsightCompute' })
        if ($nsightComputeEntries.Count -gt 0) {
            $detectedVersions['nsight-compute'] = @{
                InstalledVersion = $nsightComputeEntries[0].Version
                Path             = $nsightComputeEntries[0].Path
                SideByCount      = $nsightComputeEntries.Count
            }
        }
        else {
            $detectedVersions['nsight-compute'] = @{ InstalledVersion = $null; Path = $null; SideByCount = 0 }
        }

        # Nsight Systems
        $nsightSystemsEntries = @($nsightVersions | Where-Object { $_.Product -eq 'NsightSystems' })
        if ($nsightSystemsEntries.Count -gt 0) {
            $detectedVersions['nsight-systems'] = @{
                InstalledVersion = $nsightSystemsEntries[0].Version
                Path             = $nsightSystemsEntries[0].Path
                SideByCount      = $nsightSystemsEntries.Count
            }
        }
        else {
            $detectedVersions['nsight-systems'] = @{ InstalledVersion = $null; Path = $null; SideByCount = 0 }
        }

        # NVML — ships with the display driver as nvml.dll in System32.
        # The DLL FilePrivatePart encodes the driver build (e.g. 8241 -> 582.41).
        $nvmlDll = $null
        if ($env:SystemRoot) {
            $nvmlDll = Join-Path ($env:SystemRoot -replace '^\s+|\s+$','') 'System32\nvml.dll'
        }
        if ($null -ne $nvmlDll -and (Test-Path -LiteralPath $nvmlDll)) {
            # NVML version is coextensive with the display driver; reuse $driverVersion
            # (already obtained from nvidia-smi) rather than parsing DLL metadata.
            $nvmlDriverVer = $driverVersion
            $detectedVersions['nvml'] = @{
                InstalledVersion = $nvmlDriverVer
                Path             = $nvmlDll
                SideByCount      = 0
            }
            Write-Verbose "NVML found at $nvmlDll (driver $nvmlDriverVer)"
        }
        else {
            $detectedVersions['nvml'] = @{ InstalledVersion = $null; Path = $null; SideByCount = 0 }
        }

        # Warp — Python GPU simulation framework installed via pip/uv into AI-Media venv.
        # Probe the venv python first, then fall back to any python on PATH.
        $warpVersion = $null
        $warpPath    = $null
        $warpPythons = @(
            # From Public\ -> up 3 levels reaches repo root, then AI-Media\.venv
            (Join-Path $PSScriptRoot '..\..\..\AI-Media\.venv\Scripts\python.exe'),
            'python'
        )
        foreach ($py in $warpPythons) {
            $pyResolved = if ([System.IO.Path]::IsPathRooted($py)) {
                $py
            } else {
                (Get-Command $py -ErrorAction SilentlyContinue)?.Source
            }
            if ($null -ne $pyResolved -and (Test-Path -LiteralPath $pyResolved)) {
                $ver = & $pyResolved -c "import warp; print(warp.__version__)" 2>$null
                if ($ver -and $ver -match '^\d+\.\d+') {
                    $warpVersion = $ver.Trim()
                    $warpPath    = $pyResolved
                    Write-Verbose "Warp $warpVersion detected via $pyResolved"
                    break
                }
            }
        }
        $detectedVersions['warp'] = @{
            InstalledVersion = $warpVersion
            Path             = $warpPath
            SideByCount      = 0
        }

        # Nsight Graphics — standalone installer, not part of CUDA toolkit.
        $nsightGraphicsEntries = @($nsightVersions | Where-Object { $_.Product -eq 'NsightGraphics' })
        if ($nsightGraphicsEntries.Count -gt 0) {
            $detectedVersions['nsight-graphics'] = @{
                InstalledVersion = $nsightGraphicsEntries[0].Version
                Path             = $nsightGraphicsEntries[0].Path
                SideByCount      = $nsightGraphicsEntries.Count
            }
        }
        else {
            $detectedVersions['nsight-graphics'] = @{ InstalledVersion = $null; Path = $null; SideByCount = 0 }
        }

        # --- Build status table ---
        $results = [System.Collections.Generic.List[PSCustomObject]]::new()

        foreach ($component in $registry.Components) {
            $id            = $component.id
            $latestVersion = $component.latestVersion

            $detected        = $detectedVersions[$id]
            $installedVersion = $null
            $path             = $null
            $sideByCount      = 0

            if ($detected) {
                $installedVersion = $detected['InstalledVersion']
                $path             = $detected['Path']
                $sideByCount      = $detected['SideByCount']
            }

            # --- Determine status ---
            $status = 'NotInstalled'

            if ($installedVersion) {
                if (-not $latestVersion) {
                    $status = 'Unknown'
                }
                else {
                    # Compare using System.Version for dotted version strings
                    try {
                        # Helper to normalize version strings for [System.Version]
                        # (e.g. "565.90.07" -> "565.90.0.7" or "13.1" -> "13.1")
                        # [System.Version] requires 2-4 numeric components.
                        $toVersion = {
                            param($v)
                            $v = $v -replace '[^0-9.]' # Remove non-numeric/non-dot chars
                            $parts = @($v.Split('.') | Where-Object { $_ -ne '' })
                            if ($parts.Count -lt 2) { $parts += '0' }
                            [System.Version]::new(($parts | Select-Object -First 4) -join '.')
                        }

                        $installed = &$toVersion $installedVersion
                        $latest    = &$toVersion $latestVersion

                        if ($installed -ge $latest) {
                            $status = 'Current'
                        }
                        else {
                            $status = 'Outdated'
                        }
                    }
                    catch {
                        # Non-standard version strings: fall back to string comparison
                        Write-Verbose "Version parse failed for '$id' ($installedVersion vs $latestVersion): $($_.Exception.Message)"
                        if ($installedVersion -eq $latestVersion) {
                            $status = 'Current'
                        }
                        else {
                            $status = 'Unknown'
                        }
                    }
                }
            }

            $results.Add([PSCustomObject]@{
                ComponentId      = $id
                Name             = $component.name
                InstalledVersion = $installedVersion
                LatestVersion    = $latestVersion
                Status           = $status
                Path             = $path
                SideBySideCount  = $sideByCount
            })
        }

        Write-Verbose "Status check complete: $($results.Count) component(s) evaluated."
        return @($results)
    }
    catch {
        Write-Error "Get-NvidiaSoftwareStatus failed: $($_.Exception.Message)"
        return @()
    }
}
