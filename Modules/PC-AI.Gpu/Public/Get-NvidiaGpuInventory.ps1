#Requires -Version 5.1
<#
.SYNOPSIS
    Enumerates all installed NVIDIA GPUs and returns structured hardware metadata.

.DESCRIPTION
    Queries nvidia-smi.exe using its CSV output mode to collect per-GPU hardware
    properties including compute capability, memory totals, current utilization,
    and temperature. When nvidia-smi is not available the function falls back to
    Win32_VideoController CIM class and returns the subset of properties available
    from that source.

    Each returned object identifies the GPU by its nvidia-smi index and UUID so
    that callers can correlate results with other nvidia-smi queries or with
    CUDA_VISIBLE_DEVICES assignments.

.PARAMETER Index
    When specified, return only the GPU at this nvidia-smi index (0-based).
    When omitted all detected GPUs are returned.

.OUTPUTS
    PSCustomObject[] with properties:
      Index               - nvidia-smi GPU index (integer, or $null on CIM fallback)
      UUID                - GPU UUID string (e.g. "GPU-abc123..."), or $null
      Name                - GPU display name (e.g. "NVIDIA GeForce RTX 5060 Ti")
      DriverVersion       - Installed driver version string
      ComputeCapability   - SM compute capability string (e.g. "12.0"), or $null
      MemoryTotalMB       - Total VRAM in MiB (integer)
      MemoryUsedMB        - Currently used VRAM in MiB (integer), or $null
      Temperature         - GPU core temperature in degrees Celsius (integer), or $null
      Utilization         - GPU compute utilization percentage (integer), or $null
      Source              - "nvidia-smi" or "cim"

.EXAMPLE
    Get-NvidiaGpuInventory
    Returns metadata for all NVIDIA GPUs.

.EXAMPLE
    Get-NvidiaGpuInventory -Index 0
    Returns metadata for the GPU at nvidia-smi index 0.

.EXAMPLE
    Get-NvidiaGpuInventory | Select-Object Name, MemoryTotalMB, ComputeCapability
    Shows a concise table of GPU names, VRAM, and compute capabilities.
#>
function Get-NvidiaGpuInventory {
    [CmdletBinding()]
    [OutputType([PSCustomObject[]])]
    param(
        [Parameter()]
        [ValidateRange(0, 15)]
        [int]$Index = -1
    )

    $ErrorActionPreference = 'Stop'

    # --- Locate nvidia-smi ---
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

    # --- Primary path: nvidia-smi CSV query ---
    if ($nvidiaSmi) {
        try {
            $queryFields = 'index,uuid,name,driver_version,compute_cap,memory.total,memory.used,temperature.gpu,utilization.gpu'
            $smiArgs     = @(
                '--query-gpu={0}' -f $queryFields,
                '--format=csv,noheader,nounits'
            )

            Write-Verbose "Querying nvidia-smi: $nvidiaSmi $($smiArgs -join ' ')"
            $rawLines = & $nvidiaSmi @smiArgs 2>&1

            if ($LASTEXITCODE -ne 0) {
                Write-Verbose "nvidia-smi exited with code $LASTEXITCODE — falling back to CIM."
            }
            else {
                $results = [System.Collections.Generic.List[PSCustomObject]]::new()

                foreach ($line in $rawLines) {
                    $line = $line.Trim()
                    if ($line -eq '') { continue }

                    $fields = $line -split ',\s*'
                    if ($fields.Count -lt 9) {
                        Write-Verbose "Skipping malformed nvidia-smi output line: $line"
                        continue
                    }

                    $gpuIndex           = [int]($fields[0].Trim())
                    $gpuUuid            = $fields[1].Trim()
                    $gpuName            = $fields[2].Trim()
                    $driverVersion      = $fields[3].Trim()
                    $computeCapability  = $fields[4].Trim()
                    $memTotalRaw        = $fields[5].Trim()
                    $memUsedRaw         = $fields[6].Trim()
                    $tempRaw            = $fields[7].Trim()
                    $utilizationRaw     = $fields[8].Trim()

                    # Skip if -Index filter is active and this GPU does not match
                    if ($Index -ge 0 -and $gpuIndex -ne $Index) {
                        continue
                    }

                    # Parse numeric fields; treat "[Not Supported]" as $null
                    $memTotalMB  = $null
                    $memUsedMB   = $null
                    $temperature = $null
                    $utilization = $null

                    if ($memTotalRaw -match '^\d+$')  { $memTotalMB  = [int]$memTotalRaw }
                    if ($memUsedRaw  -match '^\d+$')  { $memUsedMB   = [int]$memUsedRaw }
                    if ($tempRaw     -match '^\d+$')  { $temperature = [int]$tempRaw }
                    if ($utilizationRaw -match '^\d+$') { $utilization = [int]$utilizationRaw }

                    # Normalize compute capability: nvidia-smi returns "8.9" or "8, 9"
                    $computeCapability = $computeCapability -replace '\s', ''

                    $results.Add([PSCustomObject]@{
                        Index             = $gpuIndex
                        UUID              = $gpuUuid
                        Name              = $gpuName
                        DriverVersion     = $driverVersion
                        ComputeCapability = $computeCapability
                        MemoryTotalMB     = $memTotalMB
                        MemoryUsedMB      = $memUsedMB
                        Temperature       = $temperature
                        Utilization       = $utilization
                        Source            = 'nvidia-smi'
                    })
                }

                Write-Verbose "nvidia-smi inventory: $($results.Count) GPU(s) found."
                return @($results)
            }
        }
        catch {
            Write-Verbose "nvidia-smi query failed: $($_.Exception.Message) — falling back to CIM."
        }
    }

    # --- Fallback: Win32_VideoController CIM ---
    Write-Verbose 'Using Win32_VideoController CIM fallback for GPU inventory.'
    try {
        $adapters = @(Get-CimInstance -ClassName Win32_VideoController -ErrorAction Stop |
            Where-Object { $_.Name -match 'NVIDIA' -or $_.AdapterCompatibility -match 'NVIDIA' })

        if ($adapters.Count -eq 0) {
            Write-Verbose 'No NVIDIA adapters found via Win32_VideoController.'
            return @()
        }

        $results = [System.Collections.Generic.List[PSCustomObject]]::new()
        $idx = 0

        foreach ($adapter in $adapters) {
            if ($Index -ge 0 -and $idx -ne $Index) {
                $idx++
                continue
            }

            # AdapterRAM is in bytes; convert to MiB
            $memTotalMB = $null
            if ($adapter.AdapterRAM -and $adapter.AdapterRAM -gt 0) {
                $memTotalMB = [int]($adapter.AdapterRAM / 1MB)
            }

            $results.Add([PSCustomObject]@{
                Index             = $idx
                UUID              = $null
                Name              = $adapter.Name
                DriverVersion     = $adapter.DriverVersion
                ComputeCapability = $null
                MemoryTotalMB     = $memTotalMB
                MemoryUsedMB      = $null
                Temperature       = $null
                Utilization       = $null
                Source            = 'cim'
            })

            $idx++
        }

        Write-Verbose "CIM inventory: $($results.Count) NVIDIA GPU(s) found."
        return @($results)
    }
    catch {
        Write-Error "Get-NvidiaGpuInventory failed: $($_.Exception.Message)"
        return @()
    }
}
