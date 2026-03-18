#Requires -Version 5.1

function Get-NsightVersions {
<#
.SYNOPSIS
    Scans for installed Nsight Compute and Nsight Systems installations and
    returns version information for each one found.

.DESCRIPTION
    Searches under "C:\Program Files\NVIDIA Corporation\" for directories
    matching the patterns "Nsight Compute*" and "Nsight Systems*". The version
    is extracted from the directory name when it matches the pattern
    "<Product> <Version>" (e.g. "Nsight Compute 2025.1.0").

    When a directory name does not embed the version, the function falls back to
    reading the application's .exe version resource if a recognizable binary is
    present.

    Returns an array of objects, one per discovered installation. An empty array
    is returned when neither tool is found.

.OUTPUTS
    PSCustomObject[] with properties: Name, Product, Version, Path.

.EXAMPLE
    Get-NsightVersions
    Returns objects for all installed Nsight tools, e.g.:
      Name    : Nsight Compute 2025.1.0
      Product : NsightCompute
      Version : 2025.1.0
      Path    : C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.1.0
#>
    [CmdletBinding()]
    [OutputType([PSCustomObject[]])]
    param(
        [Parameter()]
        [ValidateNotNullOrEmpty()]
        [string]$SearchPath = 'C:\Program Files\NVIDIA Corporation'
    )

    $ErrorActionPreference = 'Stop'

    $results = [System.Collections.Generic.List[PSCustomObject]]::new()

    $corpRoot = $SearchPath
    if (-not (Test-Path $corpRoot -PathType Container)) {
        Write-Verbose "NVIDIA Corporation directory not found at: $corpRoot"
        return @()
    }

    $nsightPatterns = @(
        @{ Pattern = 'Nsight Compute*';   Product = 'NsightCompute'   },
        @{ Pattern = 'Nsight Systems*';   Product = 'NsightSystems'   },
        @{ Pattern = 'Nsight Graphics*';  Product = 'NsightGraphics'  }
    )

    foreach ($entry in $nsightPatterns) {
        $dirs = @(Get-ChildItem -Path $corpRoot -Directory -Filter $entry.Pattern -ErrorAction SilentlyContinue |
            Sort-Object Name -Descending)

        foreach ($dir in $dirs) {
            $version = $null

            # Primary: extract version from directory name, e.g. "Nsight Compute 2025.1.0"
            if ($dir.Name -match '(\d+\.\d+[\.\d]*)$') {
                $version = $Matches[1]
            }

            # Fallback: read version resource from the main executable
            if (-not $version) {
                $exeCandidates = @(
                    (Join-Path $dir.FullName 'ncu.exe'),
                    (Join-Path $dir.FullName 'nsys.exe'),
                    (Join-Path $dir.FullName 'ncu-ui.exe'),
                    (Join-Path $dir.FullName 'NVIDIA Nsight Graphics.exe'),
                    (Join-Path $dir.FullName 'ngfx.exe')
                )
                foreach ($exe in $exeCandidates) {
                    if (Test-Path $exe) {
                        try {
                            $fileVersion = (Get-Item $exe -ErrorAction Stop).VersionInfo.FileVersion
                            if ($fileVersion -and $fileVersion -ne '') {
                                $version = $fileVersion
                            }
                        }
                        catch {
                            Write-Verbose "Could not read version from '$exe': $($_.Exception.Message)"
                        }
                        break
                    }
                }
            }

            $results.Add([PSCustomObject]@{
                Name    = $dir.Name
                Product = $entry.Product
                Version = $version
                Path    = $dir.FullName
            })

            Write-Verbose "Nsight tool found: $($dir.Name) version=$version path=$($dir.FullName)"
        }
    }

    return @($results)
}
