#Requires -Version 5.1
<#
.SYNOPSIS
    Runs a GPU preflight readiness check for LLM inference workloads.

.DESCRIPTION
    Queries NVIDIA GPU VRAM state and optionally estimates model memory
    requirements from a GGUF file header.  Returns a structured verdict
    (Go / Warn / Fail) indicating whether the model will fit on the best
    available GPU.

    Two execution paths in priority order:
      1. FFI via pcai_core_lib.dll (pcai_gpu_preflight_json) - direct NVML
         access, no subprocess, fastest path.
      2. CLI via pcai-perf.exe preflight - subprocess fallback when the DLL
         is unavailable or the FFI call fails.

    When neither path is available the function writes a warning and returns
    a Fail verdict with Source = 'none'.

.PARAMETER ModelPath
    Path to a GGUF model file.  When provided the function parses the GGUF
    header to estimate memory requirements and compares against available
    VRAM.  When omitted the function returns a VRAM inventory snapshot.

.PARAMETER ContextLength
    Context length override in tokens.  When non-zero, overrides the default
    context length embedded in the GGUF metadata for memory estimation.
    Valid range: 0 to 1048576.

.PARAMETER RequiredMB
    Minimum VRAM required in mebibytes.  Used when ModelPath is not provided
    to check whether at least this much free VRAM is available.
    Valid range: 0 to 1048576.

.PARAMETER AsJson
    When specified, returns the raw JSON string from the preflight check
    instead of a parsed PSCustomObject.

.OUTPUTS
    PSCustomObject with properties:
      Verdict         - "go", "warn", or "fail"
      Reason          - Human-readable explanation of the verdict
      ModelEstimateMB - Estimated model memory in MiB (0 when no model given)
      BestGpuIndex    - Zero-based index of the best GPU, or $null
      Gpus            - Array of per-GPU VRAM snapshots
      Source          - "ffi", "cli", or "none"

    When -AsJson is specified, returns a JSON string instead.

.EXAMPLE
    Test-PcaiGpuReadiness
    Returns a VRAM inventory snapshot with verdict "go" (inventory-only mode).

.EXAMPLE
    Test-PcaiGpuReadiness -ModelPath "C:\Models\llama-7b.Q4_K_M.gguf"
    Checks whether the 7B model fits on the best available GPU.

.EXAMPLE
    Test-PcaiGpuReadiness -ModelPath "C:\Models\qwen-30b.gguf" -ContextLength 8192
    Estimates memory for 30B model at 8K context and returns verdict.

.EXAMPLE
    Test-PcaiGpuReadiness -RequiredMB 4096
    Checks whether at least 4096 MiB of free VRAM is available.

.EXAMPLE
    Test-PcaiGpuReadiness -AsJson | ConvertFrom-Json
    Returns the raw JSON string for programmatic consumption.
#>

#region Preflight FFI interop (module-scoped, initialised once per session)
$script:PreflightInteropTypeName = 'PcaiPreflightInterop'
$script:PreflightInteropLoaded   = $false
$script:PreflightInteropDllPath  = $null

function Initialize-PreflightInteropType {
    <#
    .SYNOPSIS
        Emit an Add-Type C# class that P/Invokes the preflight FFI exports
        from pcai_core_lib.dll.  Idempotent -- safe to call multiple times.
    #>
    param([Parameter(Mandatory)][string]$DllPath)

    # Already loaded with the same DLL path -- nothing to do.
    if ($script:PreflightInteropLoaded -and $script:PreflightInteropDllPath -eq $DllPath) {
        return ($script:PreflightInteropTypeName -as [type])
    }

    # Add DLL directory to process PATH so the loader can find dependent DLLs.
    $dllDir = Split-Path -Parent $DllPath
    $currentPath = [System.Environment]::GetEnvironmentVariable('PATH', 'Process')
    if ($currentPath -notlike "*$dllDir*") {
        [System.Environment]::SetEnvironmentVariable('PATH', "$dllDir;$currentPath", 'Process')
    }

    # Only compile the type once per AppDomain.
    if (-not ($script:PreflightInteropTypeName -as [type])) {
        $escapedPath = $DllPath.Replace('\', '\\')
        $typeDef = @"
using System;
using System.Runtime.InteropServices;

public static class $($script:PreflightInteropTypeName) {
    // Runs the GPU preflight check.  Returns a heap-allocated JSON string
    // containing the PreflightResult.  Caller must free with pcai_free_string().
    //
    // model_path: null for inventory/required-mb mode, or a UTF-8 GGUF path.
    // context_length: context override (0 = use GGUF default).
    // required_mb: minimum VRAM required (0 = no threshold).
    [DllImport(@"$escapedPath", CallingConvention = CallingConvention.Cdecl)]
    public static extern IntPtr pcai_gpu_preflight_json(
        [MarshalAs(UnmanagedType.LPUTF8Str)] string modelPath,
        ulong contextLength,
        ulong requiredMb);

    // Frees a string allocated by any pcai_* function that returns *mut c_char.
    [DllImport(@"$escapedPath", CallingConvention = CallingConvention.Cdecl)]
    public static extern void pcai_free_string(IntPtr ptr);

    // Marshals an IntPtr returned by Rust into a managed string, then frees
    // the native buffer.  Returns null when ptr is IntPtr.Zero.
    public static string MarshalAndFree(IntPtr ptr) {
        if (ptr == IntPtr.Zero) return null;
        try {
            return Marshal.PtrToStringAnsi(ptr);
        } finally {
            pcai_free_string(ptr);
        }
    }
}
"@
        Add-Type -TypeDefinition $typeDef -Language CSharp -ErrorAction Stop | Out-Null
    }

    $script:PreflightInteropLoaded  = $true
    $script:PreflightInteropDllPath = $DllPath
    return ($script:PreflightInteropTypeName -as [type])
}
#endregion

function Test-PcaiGpuReadiness {
    [CmdletBinding()]
    [OutputType([PSCustomObject])]
    param(
        [Parameter()]
        [string]$ModelPath,

        [Parameter()]
        [ValidateRange(0, 1048576)]
        [int]$ContextLength = 0,

        [Parameter()]
        [ValidateRange(0, 1048576)]
        [int]$RequiredMB = 0,

        [Parameter()]
        [switch]$AsJson
    )

    $ErrorActionPreference = 'Stop'

    # Determine model argument for FFI/CLI (null/empty means no model).
    $modelArg = if ($ModelPath) { $ModelPath } else { $null }

    # ── Primary path: FFI via pcai_core_lib.dll ──────────────────────────────
    $coreDll = Resolve-PcaiCoreLibDll
    if ($coreDll) {
        try {
            $t = Initialize-PreflightInteropType -DllPath $coreDll
            Write-Verbose "Preflight FFI: using $coreDll"

            $jsonPtr = $t::pcai_gpu_preflight_json(
                $modelArg,
                [uint64]$ContextLength,
                [uint64]$RequiredMB
            )
            $json = $t::MarshalAndFree($jsonPtr)

            if ($json) {
                if ($AsJson) {
                    return $json
                }

                $raw = $json | ConvertFrom-Json
                $result = [PSCustomObject]@{
                    Verdict         = [string]$raw.verdict
                    Reason          = [string]$raw.reason
                    ModelEstimateMB = [long]$raw.model_estimate_mb
                    BestGpuIndex    = if ($null -ne $raw.best_gpu_index) { [int]$raw.best_gpu_index } else { $null }
                    Gpus            = @($raw.gpus)
                    Source          = 'ffi'
                }

                Write-Verbose "Preflight FFI: verdict=$($result.Verdict)"
                return $result
            }

            Write-Verbose 'Preflight FFI: pcai_gpu_preflight_json returned null -- falling back to CLI.'
        }
        catch {
            Write-Verbose "Preflight FFI failed: $($_.Exception.Message) -- falling back to CLI."
        }
    }
    else {
        Write-Verbose 'pcai_core_lib.dll not found -- skipping FFI path.'
    }

    # ── Fallback path: pcai-perf.exe CLI ─────────────────────────────────────
    $perfExe = $null
    $moduleDir = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.ScriptName }
    $repoRoot  = $moduleDir | Split-Path | Split-Path | Split-Path

    $perfCandidates = @(
        (Join-Path $repoRoot 'Native\pcai_core\target\release\pcai-perf.exe'),
        (Join-Path $env:USERPROFILE '.local\bin\pcai-perf.exe')
    )

    foreach ($candidate in $perfCandidates) {
        if (Test-Path $candidate -ErrorAction SilentlyContinue) {
            $perfExe = $candidate
            break
        }
    }

    if ($perfExe) {
        try {
            Write-Verbose "Preflight CLI: using $perfExe"

            $cliArgs = @('preflight')
            if ($ModelPath) {
                $cliArgs += '--model'
                $cliArgs += $ModelPath
            }
            if ($ContextLength -gt 0) {
                $cliArgs += '--ctx'
                $cliArgs += $ContextLength.ToString()
            }
            if ($RequiredMB -gt 0) {
                $cliArgs += '--required-mb'
                $cliArgs += $RequiredMB.ToString()
            }

            Write-Verbose "Preflight CLI: $perfExe $($cliArgs -join ' ')"
            $rawOutput = & $perfExe @cliArgs 2>&1

            # pcai-perf exits 0=go, 1=warn, 2=fail -- all are valid.
            # Only treat truly unexpected exit codes as errors.
            $exitCode = $LASTEXITCODE
            if ($null -ne $exitCode -and $exitCode -gt 2) {
                Write-Verbose "pcai-perf.exe exited with unexpected code $exitCode."
                throw "pcai-perf.exe exited with code $exitCode"
            }

            # Extract the first line of valid JSON (skip any stderr noise).
            $jsonLine = $null
            foreach ($line in @($rawOutput)) {
                $trimmed = "$line".Trim()
                if ($trimmed.StartsWith('{')) {
                    $jsonLine = $trimmed
                    break
                }
            }

            if ($jsonLine) {
                if ($AsJson) {
                    return $jsonLine
                }

                $raw = $jsonLine | ConvertFrom-Json
                $result = [PSCustomObject]@{
                    Verdict         = [string]$raw.verdict
                    Reason          = [string]$raw.reason
                    ModelEstimateMB = [long]$raw.model_estimate_mb
                    BestGpuIndex    = if ($null -ne $raw.best_gpu_index) { [int]$raw.best_gpu_index } else { $null }
                    Gpus            = @($raw.gpus)
                    Source          = 'cli'
                }

                Write-Verbose "Preflight CLI: verdict=$($result.Verdict)"
                return $result
            }

            Write-Verbose 'Preflight CLI: no JSON output received.'
        }
        catch {
            Write-Verbose "Preflight CLI failed: $($_.Exception.Message)"
        }
    }
    else {
        Write-Verbose 'pcai-perf.exe not found -- skipping CLI path.'
    }

    # ── No backend available ─────────────────────────────────────────────────
    Write-Warning 'Test-PcaiGpuReadiness: neither pcai_core_lib.dll nor pcai-perf.exe found. Install pcai_core with NVML support or place pcai-perf.exe on PATH.'

    $failResult = [PSCustomObject]@{
        Verdict         = 'fail'
        Reason          = 'No preflight backend available (pcai_core_lib.dll and pcai-perf.exe not found)'
        ModelEstimateMB = 0
        BestGpuIndex    = $null
        Gpus            = @()
        Source          = 'none'
    }

    if ($AsJson) {
        return ($failResult | ConvertTo-Json -Depth 5 -Compress)
    }

    return $failResult
}
