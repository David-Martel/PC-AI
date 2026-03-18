#Requires -Version 5.1
<#
.SYNOPSIS
    Enumerates all installed NVIDIA GPUs and returns structured hardware metadata.

.DESCRIPTION
    Queries GPU hardware via three paths in priority order:
      1. NVML FFI (pcai_core_lib.dll built with --features nvml) — direct kernel
         driver access, no subprocess, returns richer metrics (power, fan, PCIe).
      2. nvidia-smi.exe CSV query — subprocess fallback when DLL is unavailable.
      3. Win32_VideoController CIM — last resort when nvidia-smi is also absent.

    Each returned object identifies the GPU by its index and UUID so that callers
    can correlate results with CUDA_VISIBLE_DEVICES assignments.

.PARAMETER Index
    When specified, return only the GPU at this nvidia-smi index (0-based).
    When omitted all detected GPUs are returned.

.OUTPUTS
    PSCustomObject[] with properties:
      Index               - GPU index (integer, or $null on CIM fallback)
      UUID                - GPU UUID string (e.g. "GPU-abc123..."), or $null
      Name                - GPU display name (e.g. "NVIDIA GeForce RTX 5060 Ti")
      DriverVersion       - Installed driver version string
      ComputeCapability   - SM compute capability string (e.g. "12.0"), or $null
      MemoryTotalMB       - Total VRAM in MiB (integer)
      MemoryUsedMB        - Currently used VRAM in MiB (integer), or $null
      Temperature         - GPU core temperature in degrees Celsius (integer), or $null
      Utilization         - GPU compute utilization percentage (integer), or $null
      Source              - "nvml-ffi", "nvidia-smi", or "cim"

.EXAMPLE
    Get-NvidiaGpuInventory
    Returns metadata for all NVIDIA GPUs.

.EXAMPLE
    Get-NvidiaGpuInventory -Index 0
    Returns metadata for the GPU at index 0.

.EXAMPLE
    Get-NvidiaGpuInventory | Select-Object Name, MemoryTotalMB, ComputeCapability
    Shows a concise table of GPU names, VRAM, and compute capabilities.
#>

#region NVML FFI interop helpers (module-scoped, initialised once per session)
$script:NvmlInteropTypeName = 'PcaiGpuNvmlInterop'
$script:NvmlInteropLoaded   = $false
$script:NvmlInteropDllPath  = $null

function Resolve-PcaiCoreLibDll {
    <#
    .SYNOPSIS
        Locate pcai_core_lib.dll using the same search order as the rest of the
        PC-AI module family.
    #>
    $moduleDir  = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.ScriptName }
    # PC-AI.Gpu\Public\ -> PC-AI.Gpu\ -> Modules\ -> repo root -> bin\
    $repoRoot   = $moduleDir | Split-Path | Split-Path | Split-Path

    $candidates = @(
        (Join-Path $repoRoot 'bin\pcai_core_lib.dll'),
        (Join-Path $repoRoot '.pcai\build\artifacts\pcai-core-lib\pcai_core_lib.dll'),
        (Join-Path $repoRoot 'Native\pcai_core\target\release\pcai_core_lib.dll'),
        (Join-Path $repoRoot 'Native\pcai_core\target\release\deps\pcai_core_lib.dll')
    )

    foreach ($path in $candidates) {
        if (Test-Path $path -ErrorAction SilentlyContinue) {
            return $path
        }
    }
    return $null
}

function Initialize-NvmlInteropType {
    <#
    .SYNOPSIS
        Emit an Add-Type C# class that P/Invokes the three NVML FFI exports from
        pcai_core_lib.dll.  Idempotent — safe to call multiple times.
    #>
    param([Parameter(Mandatory)][string]$DllPath)

    # Already loaded with the same DLL path — nothing to do.
    if ($script:NvmlInteropLoaded -and $script:NvmlInteropDllPath -eq $DllPath) {
        return ($script:NvmlInteropTypeName -as [type])
    }

    # Add DLL directory to process PATH so the loader can find dependent DLLs.
    $dllDir = Split-Path -Parent $DllPath
    $currentPath = [System.Environment]::GetEnvironmentVariable('PATH', 'Process')
    if ($currentPath -notlike "*$dllDir*") {
        [System.Environment]::SetEnvironmentVariable('PATH', "$dllDir;$currentPath", 'Process')
    }

    # Only compile the type once per AppDomain — PowerShell 5.1 and 7+ both
    # keep Add-Type definitions for the lifetime of the process.
    if (-not ($script:NvmlInteropTypeName -as [type])) {
        $escapedPath = $DllPath.Replace('\', '\\')
        $typeDef = @"
using System;
using System.Runtime.InteropServices;
using System.Text;

public static class $($script:NvmlInteropTypeName) {
    // Returns the number of NVIDIA GPUs detected by NVML (0 = none, -1 = error).
    [DllImport(@"$escapedPath", CallingConvention = CallingConvention.Cdecl)]
    public static extern int pcai_gpu_count();

    // Returns a heap-allocated JSON array of GpuInfo objects. Caller must free
    // the pointer with pcai_free_string().
    [DllImport(@"$escapedPath", CallingConvention = CallingConvention.Cdecl)]
    public static extern IntPtr pcai_gpu_info_json();

    // Returns a heap-allocated driver version string. Caller must free with
    // pcai_free_string().
    [DllImport(@"$escapedPath", CallingConvention = CallingConvention.Cdecl)]
    public static extern IntPtr pcai_driver_version();

    // Frees a string allocated by any pcai_* function that returns *mut c_char.
    [DllImport(@"$escapedPath", CallingConvention = CallingConvention.Cdecl)]
    public static extern void pcai_free_string(IntPtr ptr);

    // Marshals an IntPtr returned by Rust into a managed string, then frees the
    // native buffer.  Returns null when ptr is IntPtr.Zero.
    public static string MarshalAndFree(IntPtr ptr) {
        if (ptr == IntPtr.Zero) return null;
        try {
            // Rust produces UTF-8; Marshal.PtrToStringAnsi reads until the first
            // null byte, which is correct for a CString.
            return Marshal.PtrToStringAnsi(ptr);
        } finally {
            pcai_free_string(ptr);
        }
    }
}
"@
        Add-Type -TypeDefinition $typeDef -Language CSharp -ErrorAction Stop | Out-Null
    }

    $script:NvmlInteropLoaded  = $true
    $script:NvmlInteropDllPath = $DllPath
    return ($script:NvmlInteropTypeName -as [type])
}
#endregion

function Get-NvidiaGpuInventory {
    [CmdletBinding()]
    [OutputType([PSCustomObject[]])]
    param(
        [Parameter()]
        [ValidateRange(0, 15)]
        [int]$Index = -1
    )

    $ErrorActionPreference = 'Stop'

    # ── Primary path: NVML FFI via pcai_core_lib.dll ──────────────────────────
    $coreDll = Resolve-PcaiCoreLibDll
    if ($coreDll) {
        try {
            $t = Initialize-NvmlInteropType -DllPath $coreDll

            $count = $t::pcai_gpu_count()
            Write-Verbose "NVML FFI: pcai_gpu_count() = $count"

            if ($count -gt 0) {
                $jsonPtr = $t::pcai_gpu_info_json()
                $json    = $t::MarshalAndFree($jsonPtr)

                if ($json -and $json -ne '[]') {
                    $raw     = $json | ConvertFrom-Json
                    $results = [System.Collections.Generic.List[PSCustomObject]]::new()

                    foreach ($g in @($raw)) {
                        $gpuIndex = [int]$g.index
                        if ($Index -ge 0 -and $gpuIndex -ne $Index) { continue }

                        # NVML reports memory in MiB already (converted in Rust).
                        # Utilization is not in GpuInfo (static snapshot); expose $null
                        # so callers know to call Get-NvidiaGpuUtilization for live data.
                        $results.Add([PSCustomObject]@{
                            Index             = $gpuIndex
                            UUID              = [string]$g.uuid
                            Name              = [string]$g.name
                            DriverVersion     = [string]$g.driver_version
                            ComputeCapability = [string]$g.compute_capability
                            MemoryTotalMB     = [int]$g.memory_total_mb
                            MemoryUsedMB      = [int]$g.memory_used_mb
                            Temperature       = [int]$g.temperature_c
                            Utilization       = $null  # use Get-NvidiaGpuUtilization for live util
                            Source            = 'nvml-ffi'
                        })
                    }

                    Write-Verbose "NVML FFI inventory: $($results.Count) GPU(s) found."
                    return @($results)
                }
            }
            elseif ($count -eq 0) {
                Write-Verbose 'NVML FFI: no NVIDIA GPUs detected — skipping to CIM fallback.'
                return @()
            }
            # count -eq -1 means NVML error; fall through to nvidia-smi.
            Write-Verbose 'NVML FFI: pcai_gpu_count returned -1 (NVML error) — falling back to nvidia-smi.'
        }
        catch {
            Write-Verbose "NVML FFI query failed: $($_.Exception.Message) — falling back to nvidia-smi."
        }
    }
    else {
        Write-Verbose 'pcai_core_lib.dll not found — skipping NVML FFI path.'
    }

    # ── Secondary path: nvidia-smi CSV query ──────────────────────────────────

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

    # --- Secondary path: nvidia-smi CSV query ---
    if ($nvidiaSmi) {
        try {
            $queryFields = 'index,uuid,name,driver_version,compute_cap,memory.total,memory.used,temperature.gpu,utilization.gpu'
            $smiArgs     = @(
                ('--query-gpu={0}' -f $queryFields),
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

    # --- Tertiary path: Win32_VideoController CIM ---
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
