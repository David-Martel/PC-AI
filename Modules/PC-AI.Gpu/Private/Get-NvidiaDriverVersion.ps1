#Requires -Version 5.1

function Get-NvidiaDriverVersion {
<#
.SYNOPSIS
    Retrieves the installed NVIDIA display driver version string.

.DESCRIPTION
    Queries the driver version in priority order:

      1. nvidia-smi.exe --query-gpu=driver_version (fastest, most accurate)
      2. Win32_VideoController CIM class filtered to NVIDIA adapters (fallback
         when nvidia-smi is not on PATH or fails)

    The returned version string uses the standard NVIDIA dotted format, e.g.
    "572.83" or "565.90.07". Returns $null when no NVIDIA driver is detected.

.OUTPUTS
    [string] Driver version string, or $null if not found.

.EXAMPLE
    Get-NvidiaDriverVersion
    Returns "572.83" (or similar) when an NVIDIA driver is installed.
#>
    [CmdletBinding()]
    [OutputType([string])]
    param()

    $ErrorActionPreference = 'Stop'

    # --- Primary: nvidia-smi ---
    $nvidiaSmi = $null
    $smiCandidates = @(
        'nvidia-smi.exe',
        'C:\Windows\System32\nvidia-smi.exe',
        'C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe'
    )

    foreach ($candidate in $smiCandidates) {
        if (Get-Command -Name $candidate -ErrorAction SilentlyContinue) {
            $nvidiaSmi = $candidate
            break
        }
        if (Test-Path $candidate -ErrorAction SilentlyContinue) {
            $nvidiaSmi = $candidate
            break
        }
    }

    if ($nvidiaSmi) {
        try {
            $smiOutput = & $nvidiaSmi --query-gpu=driver_version --format=csv,noheader,nounits 2>&1
            if ($LASTEXITCODE -eq 0 -and $smiOutput) {
                # Sanitize output by removing empty/noise lines
                $sanitized = @($smiOutput | Where-Object { $_.Trim() -ne '' })
                if ($sanitized.Count -gt 0) {
                    $version = $sanitized[0].Trim()
                    if ($version -and $version -ne '') {
                        Write-Verbose "Driver version from nvidia-smi: $version"
                        return $version
                    }
                }
            }
        }
        catch {
            Write-Verbose "nvidia-smi query failed: $($_.Exception.Message)"
        }
    }

    # --- Fallback: CIM Win32_VideoController ---
    Write-Verbose 'nvidia-smi unavailable or failed — falling back to Win32_VideoController CIM.'
    try {
        $nvidiaAdapter = Get-CimInstance -ClassName Win32_VideoController -ErrorAction Stop |
            Where-Object { $_.Name -match 'NVIDIA' -or $_.AdapterCompatibility -match 'NVIDIA' } |
            Select-Object -First 1

        if ($nvidiaAdapter -and $nvidiaAdapter.DriverVersion) {
            Write-Verbose "Driver version from CIM Win32_VideoController: $($nvidiaAdapter.DriverVersion)"
            return $nvidiaAdapter.DriverVersion
        }
    }
    catch {
        Write-Verbose "CIM Win32_VideoController query failed: $($_.Exception.Message)"
    }

    Write-Verbose 'No NVIDIA driver version detected.'
    return $null
}
