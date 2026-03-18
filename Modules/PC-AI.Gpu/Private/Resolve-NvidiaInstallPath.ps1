#Requires -Version 5.1

function Resolve-NvidiaInstallPath {
<#
.SYNOPSIS
    Scans known install locations for each NVIDIA software component and returns
    a hashtable mapping component identifiers to their resolved paths.

.DESCRIPTION
    Probes the standard install directories for CUDA Toolkit, cuDNN, TensorRT,
    Nsight Compute, and Nsight Systems on the local machine. For versioned
    components (CUDA, cuDNN) the newest installed version directory is selected
    when multiple side-by-side versions are present.

    If a component is not found on disk, its key maps to $null. Callers should
    check for $null before attempting to read version headers or binaries from
    the returned path.

.PARAMETER ComponentType
    When specified, return only the path for that component type instead of the
    full hashtable. Accepted values: 'CUDA', 'cuDNN', 'TensorRT',
    'NsightCompute', 'NsightSystems'.

.OUTPUTS
    Hashtable with keys: CUDA, cuDNN, TensorRT, NsightCompute, NsightSystems.
    Each value is an absolute path string, or $null if not installed.

.EXAMPLE
    Resolve-NvidiaInstallPath
    Returns the full component-to-path hashtable.

.EXAMPLE
    Resolve-NvidiaInstallPath -ComponentType 'CUDA'
    Returns only the CUDA toolkit root path string (or $null).
#>
    [CmdletBinding()]
    [OutputType([hashtable])]
    param(
        [Parameter()]
        [ValidateSet('CUDA', 'cuDNN', 'TensorRT', 'NsightCompute', 'NsightSystems')]
        [string]$ComponentType
    )

    $ErrorActionPreference = 'Stop'

    # --- CUDA Toolkit ---
    $cudaPath = $null
    $cudaRoot = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA'
    if (Test-Path $cudaRoot) {
        $cudaVersionDirs = @(Get-ChildItem -Path $cudaRoot -Directory -Filter 'v*' -ErrorAction SilentlyContinue |
            Sort-Object Name -Descending)
        if ($cudaVersionDirs.Count -gt 0) {
            $cudaPath = $cudaVersionDirs[0].FullName
            Write-Verbose "CUDA install resolved to: $cudaPath"
        }
    }

    # Also check CUDA_PATH environment variable as a fallback
    if (-not $cudaPath -and $env:CUDA_PATH -and (Test-Path $env:CUDA_PATH)) {
        $cudaPath = $env:CUDA_PATH
        Write-Verbose "CUDA install resolved from CUDA_PATH env: $cudaPath"
    }

    # --- cuDNN ---
    $cudnnPath = $null
    $cudnnRoot = 'C:\Program Files\NVIDIA\CUDNN'
    if (Test-Path $cudnnRoot) {
        $cudnnVersionDirs = @(Get-ChildItem -Path $cudnnRoot -Directory -Filter 'v*' -ErrorAction SilentlyContinue |
            Sort-Object Name -Descending)
        if ($cudnnVersionDirs.Count -gt 0) {
            $cudnnPath = $cudnnVersionDirs[0].FullName
            Write-Verbose "cuDNN install resolved to: $cudnnPath"
        }
    }

    # Fallback: cuDNN redistributed inside CUDA Toolkit directory
    if (-not $cudnnPath -and $cudaPath) {
        $cudnnHeaderCandidate = Join-Path $cudaPath 'include\cudnn_version.h'
        if (Test-Path $cudnnHeaderCandidate) {
            $cudnnPath = $cudaPath
            Write-Verbose "cuDNN header found inside CUDA Toolkit at: $cudnnPath"
        }
    }

    # Also check CUDNN_PATH environment variable
    if (-not $cudnnPath -and $env:CUDNN_PATH -and (Test-Path $env:CUDNN_PATH)) {
        $cudnnPath = $env:CUDNN_PATH
        Write-Verbose "cuDNN install resolved from CUDNN_PATH env: $cudnnPath"
    }

    # --- TensorRT ---
    $tensorrtPath = $null
    $tensorrtRoot = 'C:\Program Files\NVIDIA\TensorRT'
    if (Test-Path $tensorrtRoot) {
        $tensorrtPath = $tensorrtRoot
        Write-Verbose "TensorRT install resolved to: $tensorrtPath"
    }

    if (-not $tensorrtPath -and $env:TENSORRT_PATH -and (Test-Path $env:TENSORRT_PATH)) {
        $tensorrtPath = $env:TENSORRT_PATH
        Write-Verbose "TensorRT install resolved from TENSORRT_PATH env: $tensorrtPath"
    }

    # --- Nsight Compute ---
    $nsightComputePath = $null
    $corpRoot = 'C:\Program Files\NVIDIA Corporation'
    if (Test-Path $corpRoot) {
        $nsightComputeDirs = @(Get-ChildItem -Path $corpRoot -Directory -Filter 'Nsight Compute*' -ErrorAction SilentlyContinue |
            Sort-Object Name -Descending)
        if ($nsightComputeDirs.Count -gt 0) {
            $nsightComputePath = $nsightComputeDirs[0].FullName
            Write-Verbose "Nsight Compute resolved to: $nsightComputePath"
        }
    }

    # --- Nsight Systems ---
    $nsightSystemsPath = $null
    if (Test-Path $corpRoot) {
        $nsightSystemsDirs = @(Get-ChildItem -Path $corpRoot -Directory -Filter 'Nsight Systems*' -ErrorAction SilentlyContinue |
            Sort-Object Name -Descending)
        if ($nsightSystemsDirs.Count -gt 0) {
            $nsightSystemsPath = $nsightSystemsDirs[0].FullName
            Write-Verbose "Nsight Systems resolved to: $nsightSystemsPath"
        }
    }

    $result = @{
        CUDA          = $cudaPath
        cuDNN         = $cudnnPath
        TensorRT      = $tensorrtPath
        NsightCompute = $nsightComputePath
        NsightSystems = $nsightSystemsPath
    }

    if ($ComponentType) {
        return $result[$ComponentType]
    }

    return $result
}
