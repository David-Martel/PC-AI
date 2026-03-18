#Requires -Version 5.1

function Get-CudaVersionFromPath {
<#
.SYNOPSIS
    Reads the CUDA Toolkit version from its install directory.

.DESCRIPTION
    Given the root of a CUDA Toolkit installation (e.g.
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"), attempts to
    extract the version in this priority order:

      1. Parse version.json: reads the "cuda" -> "version" field.
      2. Parse version.txt: reads the first line as a bare version string.
      3. Infer from the directory name: matches trailing "vX.Y" pattern.

    Returns a version string like "12.8.0" or "12.8", or $null when none of the
    above methods succeed.

.PARAMETER CudaPath
    Root directory of the CUDA Toolkit installation.

.OUTPUTS
    [string] Version string, or $null if undetectable.

.EXAMPLE
    Get-CudaVersionFromPath -CudaPath 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8'
    Returns "12.8.0"
#>
    [CmdletBinding()]
    [OutputType([string])]
    param(
        [Parameter(Mandatory)]
        [ValidateNotNullOrEmpty()]
        [string]$CudaPath
    )

    $ErrorActionPreference = 'Stop'

    if (-not (Test-Path $CudaPath -PathType Container)) {
        Write-Verbose "CUDA path does not exist: $CudaPath"
        return $null
    }

    # --- Method 1: version.json ---
    $versionJsonPath = Join-Path $CudaPath 'version.json'
    if (Test-Path $versionJsonPath) {
        try {
            $raw      = [System.IO.File]::ReadAllText($versionJsonPath)
            $jsonData = $raw | ConvertFrom-Json
            $cudaVer  = $null

            # Structure varies across CUDA releases:
            #   { "cuda": { "version": "12.8.0" } }
            #   { "cuda_cudart": { "version": "12.8.0" } }
            if ($jsonData.cuda -and $jsonData.cuda.version) {
                $cudaVer = $jsonData.cuda.version
            }
            elseif ($jsonData.cuda_cudart -and $jsonData.cuda_cudart.version) {
                $cudaVer = $jsonData.cuda_cudart.version
            }
            else {
                # Iterate properties and find the first one with a "version" field
                foreach ($prop in $jsonData.PSObject.Properties) {
                    if ($prop.Value -and $prop.Value.version) {
                        $cudaVer = $prop.Value.version
                        break
                    }
                }
            }

            if ($cudaVer) {
                Write-Verbose "CUDA version from version.json: $cudaVer"
                return $cudaVer
            }
        }
        catch {
            Write-Verbose "Failed to parse version.json at '$versionJsonPath': $($_.Exception.Message)"
        }
    }

    # --- Method 2: version.txt ---
    $versionTxtPath = Join-Path $CudaPath 'version.txt'
    if (Test-Path $versionTxtPath) {
        try {
            $line = [System.IO.File]::ReadAllLines($versionTxtPath) | Select-Object -First 1
            if ($line -match '(\d+\.\d+[\.\d]*)') {
                $ver = $Matches[1]
                Write-Verbose "CUDA version from version.txt: $ver"
                return $ver
            }
        }
        catch {
            Write-Verbose "Failed to read version.txt at '$versionTxtPath': $($_.Exception.Message)"
        }
    }

    # --- Method 3: Infer from directory name ---
    $dirName = Split-Path $CudaPath -Leaf
    if ($dirName -match '^v(\d+\.\d+.*)$') {
        $ver = $Matches[1]
        Write-Verbose "CUDA version inferred from directory name '$dirName': $ver"
        return $ver
    }

    Write-Verbose "Could not determine CUDA version from path: $CudaPath"
    return $null
}
