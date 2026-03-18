#Requires -Version 5.1

function Get-TensorRtVersionFromHeader {
<#
.SYNOPSIS
    Parses the TensorRT version from the NvInferVersion.h header file.

.DESCRIPTION
    Searches for NvInferVersion.h under the include subdirectory of the given
    TensorRT install path and extracts the NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR,
    and NV_TENSORRT_PATCH preprocessor defines. Returns a dotted version string
    such as "10.9.0".

    Falls back to NvInfer.h if NvInferVersion.h is not present (used in older
    TensorRT distributions).

.PARAMETER TensorRtPath
    Root directory of the TensorRT installation. The function searches for header
    files under <TensorRtPath>\include\.

.OUTPUTS
    [string] Version string formatted as "MAJOR.MINOR.PATCH", or $null when the
    header is not found or the version defines cannot be parsed.

.EXAMPLE
    Get-TensorRtVersionFromHeader -TensorRtPath 'C:\Program Files\NVIDIA\TensorRT'
    Returns "10.9.0"
#>
    [CmdletBinding()]
    [OutputType([string])]
    param(
        [Parameter(Mandatory)]
        [ValidateNotNullOrEmpty()]
        [string]$TensorRtPath
    )

    $ErrorActionPreference = 'Stop'

    if (-not (Test-Path $TensorRtPath -PathType Container)) {
        Write-Verbose "TensorRT path does not exist: $TensorRtPath"
        return $null
    }

    $includePath = Join-Path $TensorRtPath 'include'

    # Priority: NvInferVersion.h (TensorRT 8+), then NvInfer.h (older distributions)
    $headerCandidates = @(
        (Join-Path $includePath 'NvInferVersion.h'),
        (Join-Path $includePath 'NvInfer.h')
    )

    $headerPath = $null
    foreach ($candidate in $headerCandidates) {
        if (Test-Path $candidate) {
            $headerPath = $candidate
            break
        }
    }

    if (-not $headerPath) {
        Write-Verbose "No TensorRT version header found under: $includePath"
        return $null
    }

    Write-Verbose "Parsing TensorRT version from: $headerPath"

    try {
        $lines = [System.IO.File]::ReadAllLines($headerPath)
    }
    catch {
        Write-Verbose "Failed to read TensorRT header '$headerPath': $($_.Exception.Message)"
        return $null
    }

    $major = $null
    $minor = $null
    $patch = $null

    foreach ($line in $lines) {
        if ($line -match '^\s*#define\s+NV_TENSORRT_MAJOR\s+(\d+)') {
            $major = $Matches[1]
        }
        elseif ($line -match '^\s*#define\s+NV_TENSORRT_MINOR\s+(\d+)') {
            $minor = $Matches[1]
        }
        elseif ($line -match '^\s*#define\s+NV_TENSORRT_PATCH\s+(\d+)') {
            $patch = $Matches[1]
        }

        # Early exit once all three are found
        if ($null -ne $major -and $null -ne $minor -and $null -ne $patch) {
            break
        }
    }

    if ($null -eq $major -or $null -eq $minor -or $null -eq $patch) {
        Write-Verbose "Could not extract all version components from '$headerPath' (major=$major minor=$minor patch=$patch)."
        return $null
    }

    $version = "$major.$minor.$patch"
    Write-Verbose "TensorRT version parsed: $version"
    return $version
}
