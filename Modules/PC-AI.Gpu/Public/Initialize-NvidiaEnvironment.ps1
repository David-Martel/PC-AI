#Requires -Version 5.1
<#
.SYNOPSIS
    Configures NVIDIA-related environment variables for the current session or
    persistently for the current user or machine.

.DESCRIPTION
    Orchestrates the following steps:

      1. Backs up the current NVIDIA environment via Backup-NvidiaEnvironment.
      2. Resolves install paths for all NVIDIA components via
         Resolve-NvidiaInstallPath.
      3. Selects the target CUDA installation: the highest version directory
         under the standard CUDA root, or the version specified by
         -PreferredCudaVersion.
      4. Sets the following environment variables in the requested -Scope:
           CUDA_PATH          - Selected CUDA Toolkit root
           CUDA_HOME          - Same as CUDA_PATH
           CUDA_DIR           - Same as CUDA_PATH
           CUDNN_PATH         - cuDNN root (when detected)
           TENSORRT_PATH      - TensorRT root (when detected)
           CUDA_COMPUTE_CAPS  - "89,120" for the two installed GPUs
           CUDA_DEVICE_ORDER  - "PCI_BUS_ID"
      5. Prepends to PATH: CUDA bin, CUDA nvvm\bin, cuDNN bin, TensorRT lib.
      6. When Tools\Initialize-CudaEnvironment.ps1 exists in the PC_AI tree it
         is dot-sourced and Initialize-CudaEnvironment is called with matching
         parameters to apply its additional MSVC / CMake / SDK configuration.
         Env vars already set by that function are not re-applied.
      7. Returns a summary PSCustomObject with all env vars set and PATH
         entries added.

    -Scope controls persistence:
      Process  (default) - Current PowerShell session only.
      User               - Written to HKCU via [System.Environment]::SetEnvironmentVariable.
      Machine            - Written to HKLM; requires an elevated session.

    Use -WhatIf to preview all changes without applying them.

.PARAMETER Scope
    Target persistence scope. Accepted values: Process (default), User, Machine.
    Machine scope requires Administrator privileges.

.PARAMETER PreferredCudaVersion
    Selects a specific CUDA version directory name to use as CUDA_PATH (e.g.
    "v13.1"). When omitted the highest-versioned installed directory is used.

.PARAMETER SkipBackup
    Suppresses the initial Backup-NvidiaEnvironment call. Useful when the
    caller has already taken a snapshot.

.PARAMETER Quiet
    Suppresses Write-Host output from the delegated Initialize-CudaEnvironment
    helper. Verbose stream is unaffected.

.OUTPUTS
    PSCustomObject with properties:
      BackupFile         - Path of the environment backup JSON, or $null.
      CudaPath           - Selected CUDA Toolkit root.
      CudaVersion        - Detected version string of the selected CUDA install.
      CudnnPath          - Resolved cuDNN root, or $null.
      TensorRtPath       - Resolved TensorRT root, or $null.
      Scope              - The -Scope value applied.
      EnvVarsSet         - Hashtable of variable-name -> value set by this call.
      PathEntriesAdded   - String[] of PATH segments added during this call.
      DelegatedToCudaEnv - $true if Initialize-CudaEnvironment was also called.
      Notes              - String[] of informational messages.

.EXAMPLE
    Initialize-NvidiaEnvironment
    Sets CUDA_PATH, CUDNN_PATH, etc. for the current process using the latest
    installed CUDA version.

.EXAMPLE
    Initialize-NvidiaEnvironment -PreferredCudaVersion 'v12.8'
    Configures the environment for CUDA v12.8 specifically.

.EXAMPLE
    Initialize-NvidiaEnvironment -Scope User
    Sets environment variables persistently for the current user account.

.EXAMPLE
    Initialize-NvidiaEnvironment -WhatIf
    Shows what environment changes would be made without applying any.

.NOTES
    The function calls private helpers (Backup-NvidiaEnvironment,
    Resolve-NvidiaInstallPath, Get-CudaVersionFromPath,
    Get-CudnnVersionFromHeader, Get-TensorRtVersionFromHeader) that are
    dot-sourced by PC-AI.Gpu.psm1. When testing outside the module those
    helpers must be in scope.
#>
function Initialize-NvidiaEnvironment {
    [CmdletBinding(SupportsShouldProcess)]
    [OutputType([PSCustomObject])]
    param(
        [Parameter()]
        [ValidateSet('Process', 'User', 'Machine')]
        [string]$Scope = 'Process',

        [Parameter()]
        [string]$PreferredCudaVersion,

        [Parameter()]
        [switch]$SkipBackup,

        [Parameter()]
        [switch]$Quiet
    )

    $ErrorActionPreference = 'Stop'

    # Locate the PC_AI root: module root -> Modules -> PC_AI
    $moduleRoot = $script:ModuleRoot
    $modulesDir = Split-Path $moduleRoot -Parent
    $pcAiRoot   = Split-Path $modulesDir -Parent

    $notes           = [System.Collections.Generic.List[string]]::new()
    $envVarsSet      = [ordered]@{}
    $pathEntries     = [System.Collections.Generic.List[string]]::new()
    $backupFilePath  = $null
    $delegated       = $false

    # -------------------------------------------------------------------------
    # Helper: set an env var in the requested scope (no-op under -WhatIf)
    # -------------------------------------------------------------------------
    function Set-NvidiaEnvVar {
        param([string]$Name, [string]$Value)
        if (-not $Value) { return }
        if ($PSCmdlet.ShouldProcess("$Name = $Value", "Set environment variable ($Scope)")) {
            Set-Item -Path "Env:$Name" -Value $Value
            if ($Scope -in 'User', 'Machine') {
                $envScope = [System.EnvironmentVariableTarget]::$Scope
                [System.Environment]::SetEnvironmentVariable($Name, $Value, $envScope)
            }
            $envVarsSet[$Name] = $Value
            Write-Verbose "Set $Name = $Value  (scope: $Scope)"
        }
    }

    # Helper: prepend a directory to PATH (no-op if already present or missing)
    function Add-NvidiaPathEntry {
        param([string]$PathSegment)
        if (-not $PathSegment) { return }
        if (-not (Test-Path $PathSegment)) {
            Write-Verbose "PATH entry skipped (not found): $PathSegment"
            return
        }
        if ($env:PATH -like "*$PathSegment*") {
            Write-Verbose "PATH entry already present: $PathSegment"
            return
        }
        if ($PSCmdlet.ShouldProcess($PathSegment, "Prepend to PATH ($Scope)")) {
            $env:PATH = "$PathSegment;$env:PATH"
            $pathEntries.Add($PathSegment)
            if ($Scope -in 'User', 'Machine') {
                $envScope   = [System.EnvironmentVariableTarget]::$Scope
                $currentPersisted = [System.Environment]::GetEnvironmentVariable('PATH', $envScope)
                if ($currentPersisted -notlike "*$PathSegment*") {
                    [System.Environment]::SetEnvironmentVariable(
                        'PATH',
                        "$PathSegment;$currentPersisted",
                        $envScope
                    )
                }
            }
            Write-Verbose "Added to PATH: $PathSegment"
        }
    }

    # -------------------------------------------------------------------------
    # Step 1: Backup current environment
    # -------------------------------------------------------------------------
    if (-not $SkipBackup) {
        try {
            $backupFilePath = Backup-NvidiaEnvironment
            Write-Verbose "Environment backup: $backupFilePath"
        }
        catch {
            $msg = "Backup-NvidiaEnvironment failed (non-fatal): $($_.Exception.Message)"
            Write-Warning $msg
            $notes.Add($msg)
        }
    }

    # -------------------------------------------------------------------------
    # Step 2: Resolve all NVIDIA component install paths
    # -------------------------------------------------------------------------
    Write-Verbose 'Resolving NVIDIA component install paths...'
    $installPaths = Resolve-NvidiaInstallPath

    $cudaRootDir       = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA'
    $selectedCuda      = $null
    $registryCudaEntry = $null

    try {
        $gpuRegistry = Get-NvidiaSoftwareRegistry
        $registryCudaEntry = @($gpuRegistry.Components | Where-Object { $_.id -eq 'cuda-toolkit' } | Select-Object -First 1)[0]
    }
    catch {
        Write-Verbose "Unable to load NVIDIA software registry for CUDA selection: $($_.Exception.Message)"
    }

    if ($PreferredCudaVersion) {
        # Normalise: accept "12.8", "v12.8", "12.8.0" etc.
        $normalized = $PreferredCudaVersion.TrimStart('v')
        # Match major.minor only for directory name matching
        $majorMinor = ($normalized -split '\.')[0..1] -join '.'
        $candidate  = Join-Path $cudaRootDir "v$majorMinor"
        if (Test-Path $candidate) {
            $selectedCuda = $candidate
            Write-Verbose "PreferredCudaVersion: selected $selectedCuda"
        }
        else {
            $msg = "Preferred CUDA version 'v$majorMinor' not found at $candidate — falling back to latest."
            Write-Warning $msg
            $notes.Add($msg)
        }
    }

    if (-not $selectedCuda -and $registryCudaEntry) {
        $defaultInstall = @($registryCudaEntry.installedVersions | Where-Object { $_.isDefault -eq $true } | Select-Object -First 1)[0]
        if ($defaultInstall -and $defaultInstall.path -and (Test-Path $defaultInstall.path)) {
            $selectedCuda = $defaultInstall.path
            Write-Verbose "Registry default CUDA install selected: $selectedCuda"
        }
        elseif ($registryCudaEntry.envVars -and $registryCudaEntry.envVars.CUDA_PATH) {
            $registryCudaPath = $registryCudaEntry.envVars.CUDA_PATH
            if ($registryCudaPath -and $registryCudaPath -notmatch 'not set' -and (Test-Path $registryCudaPath)) {
                $selectedCuda = $registryCudaPath
                Write-Verbose "Registry CUDA_PATH selected: $selectedCuda"
            }
        }
    }

    if (-not $selectedCuda) {
        # Pick the highest installed v* directory
        if (Test-Path $cudaRootDir) {
            $latest = Get-ChildItem -Path $cudaRootDir -Directory -Filter 'v*' -ErrorAction SilentlyContinue |
                Sort-Object {
                    $n = ($_.Name -replace '^v', '') -split '\.'
                    [System.Version](($n[0..([Math]::Min(1, $n.Count - 1))] -join '.') + '.0')
                } -Descending |
                Select-Object -First 1
            if ($latest) { $selectedCuda = $latest.FullName }
        }

        if (-not $selectedCuda -and $installPaths['CUDA']) {
            $selectedCuda = $installPaths['CUDA']
        }
    }

    if (-not $selectedCuda) {
        Write-Warning 'No CUDA Toolkit installation found. CUDA environment variables will not be set.'
        $notes.Add('CUDA not found — CUDA_PATH/CUDA_HOME/CUDA_DIR not set.')
    }

    # -------------------------------------------------------------------------
    # Step 3: Detect version of the selected CUDA install
    # -------------------------------------------------------------------------
    $cudaVersion = $null
    if ($selectedCuda) {
        try { $cudaVersion = Get-CudaVersionFromPath -CudaPath $selectedCuda } catch { }
    }

    # -------------------------------------------------------------------------
    # Step 4: Set CUDA env vars
    # -------------------------------------------------------------------------
    if ($selectedCuda) {
        Set-NvidiaEnvVar -Name 'CUDA_PATH' -Value $selectedCuda
        Set-NvidiaEnvVar -Name 'CUDA_HOME' -Value $selectedCuda
        Set-NvidiaEnvVar -Name 'CUDA_DIR'  -Value $selectedCuda
    }

    # cuDNN
    $cudnnPath = $installPaths['cuDNN']
    if ($cudnnPath) {
        Set-NvidiaEnvVar -Name 'CUDNN_PATH' -Value $cudnnPath
    }
    else {
        $notes.Add('cuDNN not found — CUDNN_PATH not set.')
    }

    # TensorRT
    $tensorrtPath = $installPaths['TensorRT']
    if ($tensorrtPath) {
        Set-NvidiaEnvVar -Name 'TENSORRT_PATH' -Value $tensorrtPath
    }
    else {
        $notes.Add('TensorRT not found — TENSORRT_PATH not set.')
    }

    # GPU compute capabilities — queried dynamically so this works on any system
    try {
        $gpus = @(Get-NvidiaGpuInventory -ErrorAction Stop)
        $caps = ($gpus |
            Where-Object { $_.ComputeCapability } |
            ForEach-Object { ($_.ComputeCapability -replace '\.', '') } |
            Sort-Object -Unique) -join ','
        if ($caps) {
            Set-NvidiaEnvVar -Name 'CUDA_COMPUTE_CAPS' -Value $caps
        }
        else {
            $notes.Add('Get-NvidiaGpuInventory returned no ComputeCapability values — CUDA_COMPUTE_CAPS not set.')
            Write-Warning 'Initialize-NvidiaEnvironment: No GPU compute capabilities detected — CUDA_COMPUTE_CAPS not set.'
        }
    }
    catch {
        $msg = "Get-NvidiaGpuInventory failed (non-fatal): $($_.Exception.Message) — CUDA_COMPUTE_CAPS not set."
        Write-Warning "Initialize-NvidiaEnvironment: $msg"
        $notes.Add($msg)
    }
    Set-NvidiaEnvVar -Name 'CUDA_DEVICE_ORDER'  -Value 'PCI_BUS_ID'

    # -------------------------------------------------------------------------
    # Step 5: Update PATH
    # -------------------------------------------------------------------------
    if ($selectedCuda) {
        Add-NvidiaPathEntry (Join-Path $selectedCuda 'bin')
        Add-NvidiaPathEntry (Join-Path $selectedCuda 'nvvm\bin')
        Add-NvidiaPathEntry (Join-Path $selectedCuda 'libnvvp')
    }

    if ($cudnnPath) {
        # cuDNN bin dir contains cudnn*.dll files
        $cudnnBin = Join-Path $cudnnPath 'bin'
        if (-not (Test-Path $cudnnBin)) {
            # Some cuDNN layouts have CUDA-versioned subdirs under bin
            # e.g. C:\Program Files\NVIDIA\CUDNN\v9.8\bin (flat) — already checked
            # Try the flat bin
            $cudnnBin = $cudnnPath
        }
        Add-NvidiaPathEntry $cudnnBin
    }

    if ($tensorrtPath) {
        # TensorRT lib dir contains inference DLLs
        $trtLib = Join-Path $tensorrtPath 'lib'
        Add-NvidiaPathEntry $trtLib
        $trtBin = Join-Path $tensorrtPath 'bin'
        Add-NvidiaPathEntry $trtBin
    }

    # -------------------------------------------------------------------------
    # Step 6: Delegate to Tools\Initialize-CudaEnvironment.ps1 when present
    # -------------------------------------------------------------------------
    $cudaEnvScript = Join-Path $pcAiRoot 'Tools\Initialize-CudaEnvironment.ps1'
    if (Test-Path $cudaEnvScript) {
        Write-Verbose "Delegating to Tools\Initialize-CudaEnvironment.ps1 for MSVC/CMake/SDK configuration..."
        try {
            # Dot-source to bring Initialize-CudaEnvironment into scope
            . $cudaEnvScript

            $cudaEnvParams = @{
                Quiet = $Quiet.IsPresent
            }
            if ($selectedCuda) {
                $cudaEnvParams['CudaPath'] = $selectedCuda
            }
            if ($PreferredCudaVersion) {
                # Tools\Initialize-CudaEnvironment accepts PreferredVersions list
                $normalized  = $PreferredCudaVersion.TrimStart('v')
                $majorMinor  = ($normalized -split '\.')[0..1] -join '.'
                $cudaEnvParams['PreferredVersions'] = @("v$majorMinor")
            }

            if ($PSCmdlet.ShouldProcess('Tools\Initialize-CudaEnvironment.ps1', 'Invoke CUDA environment helper')) {
                $cudaResult = Initialize-CudaEnvironment @cudaEnvParams
                $delegated  = $true

                if ($cudaResult -and $cudaResult.PathUpdated) {
                    $notes.Add('Initialize-CudaEnvironment updated PATH with additional MSVC/SDK entries.')
                }
                if ($cudaResult -and $cudaResult.CompatibilityWarning) {
                    $notes.Add($cudaResult.CompatibilityWarning)
                }
            }
        }
        catch {
            $msg = "Delegation to Initialize-CudaEnvironment failed (non-fatal): $($_.Exception.Message)"
            Write-Warning $msg
            $notes.Add($msg)
        }
    }
    else {
        $notes.Add('Tools\Initialize-CudaEnvironment.ps1 not found — MSVC/CMake/SDK env vars not configured.')
    }

    # -------------------------------------------------------------------------
    # Step 7: Build and return the summary object
    # -------------------------------------------------------------------------
    return [PSCustomObject]@{
        BackupFile         = $backupFilePath
        CudaPath           = $selectedCuda
        CudaVersion        = $cudaVersion
        CudnnPath          = $cudnnPath
        TensorRtPath       = $tensorrtPath
        Scope              = $Scope
        EnvVarsSet         = $envVarsSet
        PathEntriesAdded   = $pathEntries.ToArray()
        DelegatedToCudaEnv = $delegated
        Notes              = $notes.ToArray()
    }
}
