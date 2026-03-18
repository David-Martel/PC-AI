#Requires -Version 5.1

function Get-CudnnVersionFromHeader {
<#
.SYNOPSIS
    Parses the cuDNN version from the cudnn_version.h header file.

.DESCRIPTION
    Locates cudnn_version.h under the include subdirectory of the given cuDNN
    install path and extracts the CUDNN_MAJOR, CUDNN_MINOR, and CUDNN_PATCHLEVEL
    preprocessor defines. Returns a dotted version string like "9.8.0".

    Falls back to searching cudnn.h when cudnn_version.h is absent (older cuDNN
    distributions bundled the version defines in cudnn.h directly).

.PARAMETER CudnnPath
    Root directory of the cuDNN installation. The function searches for header
    files under <CudnnPath>\include\.

.OUTPUTS
    [string] Version string formatted as "MAJOR.MINOR.PATCH", or $null when the
    header is not found or parsing fails.

.EXAMPLE
    Get-CudnnVersionFromHeader -CudnnPath 'C:\Program Files\NVIDIA\CUDNN\v9.8'
    Returns "9.8.0"
#>
    [CmdletBinding()]
    [OutputType([string])]
    param(
        [Parameter(Mandatory)]
        [ValidateNotNullOrEmpty()]
        [string]$CudnnPath
    )

    $ErrorActionPreference = 'Stop'

    if (-not (Test-Path $CudnnPath -PathType Container)) {
        Write-Verbose "cuDNN path does not exist: $CudnnPath"
        return $null
    }

    $includePath = Join-Path $CudnnPath 'include'

    # cuDNN 9.x stores headers in version-specific subdirs: include/12.8/cudnn_version.h
    # cuDNN 8.x stores directly: include/cudnn_version.h
    # cuDNN 7.x: include/cudnn.h
    $headerCandidates = @(
        (Join-Path $includePath 'cudnn_version.h')
        (Join-Path $includePath 'cudnn.h')
    )
    # Also search version-specific subdirs (cuDNN 9.x pattern)
    if (Test-Path $includePath) {
        Get-ChildItem -Path $includePath -Directory -ErrorAction SilentlyContinue | ForEach-Object {
            $headerCandidates += (Join-Path $_.FullName 'cudnn_version.h')
            $headerCandidates += (Join-Path $_.FullName 'cudnn.h')
        }
    }

    $headerPath = $null
    foreach ($candidate in $headerCandidates) {
        if (Test-Path $candidate) {
            $headerPath = $candidate
            break
        }
    }

    if (-not $headerPath) {
        Write-Verbose "No cuDNN version header found under: $includePath"
        return $null
    }

    Write-Verbose "Parsing cuDNN version from: $headerPath"

    try {
        $lines = [System.IO.File]::ReadAllLines($headerPath)
    }
    catch {
        Write-Verbose "Failed to read cuDNN header '$headerPath': $($_.Exception.Message)"
        return $null
    }

    $major = $null
    $minor = $null
    $patch = $null

    foreach ($line in $lines) {
        if ($line -match '^\s*#define\s+CUDNN_MAJOR\s+(\d+)') {
            $major = $Matches[1]
        }
        elseif ($line -match '^\s*#define\s+CUDNN_MINOR\s+(\d+)') {
            $minor = $Matches[1]
        }
        elseif ($line -match '^\s*#define\s+CUDNN_PATCHLEVEL\s+(\d+)') {
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
    Write-Verbose "cuDNN version parsed: $version"
    return $version
}
