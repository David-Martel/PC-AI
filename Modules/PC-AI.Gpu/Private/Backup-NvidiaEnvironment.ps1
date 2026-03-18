#Requires -Version 5.1

function Backup-NvidiaEnvironment {
<#
.SYNOPSIS
    Captures the current NVIDIA-related environment variables to a timestamped
    JSON backup file, with an optional restore mode.

.DESCRIPTION
    Snapshots all NVIDIA-relevant process-level environment state, including:

      - Core variables: CUDA_PATH, CUDA_HOME, CUDA_DIR, CUDNN_PATH,
        TENSORRT_PATH, CUDA_VISIBLE_DEVICES, CUDA_DEVICE_ORDER,
        CUDA_COMPUTE_CAPS, CUDAARCHS, NVCC_CCBIN
      - All versioned CUDA_PATH_V* variables
      - PATH segments containing "NVIDIA" or "CUDA" (case-insensitive)
      - Detected installed component versions collected via the private
        detection functions: driver, CUDA per-install, cuDNN, TensorRT

    The backup is written to .pcai\nvidia-backup\nvidia-env-{timestamp}.json
    under the PC_AI root directory (two levels above the module root), or to an
    explicit path supplied via -BackupRoot.

    When -Restore is specified the function reads a backup JSON produced by a
    prior invocation and restores the captured environment variables into the
    current PowerShell process scope. PATH nvidia/CUDA segments are prepended
    to the current PATH, with no duplicates added.

.PARAMETER BackupRoot
    Directory in which to write the backup JSON. Defaults to
    <PC_AI root>\.pcai\nvidia-backup\.

.PARAMETER Restore
    When specified, treat -BackupRoot (or the default directory) as a source.
    Requires -BackupFile to identify which snapshot to restore. If -BackupFile
    is omitted the most recent backup in the directory is used.

.PARAMETER BackupFile
    Full path to a specific backup JSON file to restore. Used only when
    -Restore is also specified.

.OUTPUTS
    [string] Full path of the written backup file (default mode), or the path
    of the restored backup file (-Restore mode). Returns $null on failure.

.EXAMPLE
    Backup-NvidiaEnvironment
    Writes a timestamped backup to .pcai\nvidia-backup\ under the PC_AI root.

.EXAMPLE
    Backup-NvidiaEnvironment -BackupRoot 'D:\Backups\nvidia'
    Writes the backup to a custom directory.

.EXAMPLE
    Backup-NvidiaEnvironment -Restore
    Restores environment variables from the most recent backup in the default
    directory.

.EXAMPLE
    Backup-NvidiaEnvironment -Restore -BackupFile 'C:\path\to\nvidia-env-20260318T120000.json'
    Restores environment variables from the specified backup file.

.NOTES
    The function calls Resolve-NvidiaInstallPath, Get-NvidiaDriverVersion,
    Get-CudaVersionFromPath, Get-CudnnVersionFromHeader, and
    Get-TensorRtVersionFromHeader, which are dot-sourced private helpers in the
    same PC-AI.Gpu module. When called outside the module context those helpers
    must be in scope.
#>
    [CmdletBinding(SupportsShouldProcess, DefaultParameterSetName = 'Backup')]
    [OutputType([string])]
    param(
        [Parameter(ParameterSetName = 'Backup')]
        [Parameter(ParameterSetName = 'Restore')]
        [string]$BackupRoot,

        [Parameter(ParameterSetName = 'Restore', Mandatory)]
        [switch]$Restore,

        [Parameter(ParameterSetName = 'Restore')]
        [string]$BackupFile
    )

    $ErrorActionPreference = 'Stop'

    # -------------------------------------------------------------------------
    # Resolve the backup root directory
    # -------------------------------------------------------------------------
    if (-not $BackupRoot) {
        # $script:ModuleRoot is set in PC-AI.Gpu.psm1 to the module directory.
        # Navigate: PC-AI.Gpu\ -> Modules\ -> PC_AI root
        $moduleRoot  = $script:ModuleRoot
        $modulesDir  = Split-Path $moduleRoot -Parent
        $pcAiRoot    = Split-Path $modulesDir -Parent
        $BackupRoot  = Join-Path $pcAiRoot '.pcai\nvidia-backup'
    }

    # -------------------------------------------------------------------------
    # RESTORE mode
    # -------------------------------------------------------------------------
    if ($Restore) {
        # Locate the backup file to restore
        if (-not $BackupFile) {
            if (-not (Test-Path $BackupRoot)) {
                Write-Error "Backup directory not found: $BackupRoot"
                return $null
            }
            $latestFile = Get-ChildItem -Path $BackupRoot -Filter 'nvidia-env-*.json' -ErrorAction Stop |
                Sort-Object Name -Descending |
                Select-Object -First 1
            if (-not $latestFile) {
                Write-Error "No backup files found in: $BackupRoot"
                return $null
            }
            $BackupFile = $latestFile.FullName
        }

        if (-not (Test-Path $BackupFile)) {
            Write-Error "Backup file not found: $BackupFile"
            return $null
        }

        Write-Verbose "Restoring NVIDIA environment from: $BackupFile"

        $raw     = [System.IO.File]::ReadAllText($BackupFile)
        $backup  = $raw | ConvertFrom-Json

        $envVars = $backup.EnvVars
        if (-not $envVars) {
            Write-Error "Backup file has no EnvVars section: $BackupFile"
            return $null
        }

        if ($PSCmdlet.ShouldProcess($BackupFile, 'Restore NVIDIA environment variables')) {
            # Restore named environment variables
            foreach ($prop in $envVars.PSObject.Properties) {
                $name  = $prop.Name
                $value = $prop.Value
                if ($null -ne $value) {
                    Set-Item -Path "Env:$name" -Value $value
                    Write-Verbose "Restored: $name = $value"
                }
            }

            # Restore nvidia/CUDA PATH segments (prepend, no duplicates)
            if ($backup.NvidiaPathSegments) {
                $currentPath = $env:PATH
                foreach ($segment in [array]($backup.NvidiaPathSegments)) {
                    if ($segment -and (Test-Path $segment) -and
                        $currentPath -notlike "*$segment*") {
                        $env:PATH  = "$segment;$env:PATH"
                        $currentPath = $env:PATH
                        Write-Verbose "Restored PATH segment: $segment"
                    }
                }
            }

            Write-Verbose "Restore complete from: $BackupFile"
        }

        return $BackupFile
    }

    # -------------------------------------------------------------------------
    # BACKUP mode
    # -------------------------------------------------------------------------

    # Create the backup directory if needed
    if (-not (Test-Path $BackupRoot)) {
        if ($PSCmdlet.ShouldProcess($BackupRoot, 'Create backup directory')) {
            [System.IO.Directory]::CreateDirectory($BackupRoot) | Out-Null
            Write-Verbose "Created backup directory: $BackupRoot"
        }
    }

    # --- Capture explicit NVIDIA environment variables -----------------------
    $namedVarNames = @(
        'CUDA_PATH',
        'CUDA_HOME',
        'CUDA_DIR',
        'CUDNN_PATH',
        'TENSORRT_PATH',
        'CUDA_VISIBLE_DEVICES',
        'CUDA_DEVICE_ORDER',
        'CUDA_COMPUTE_CAPS',
        'CUDAARCHS',
        'NVCC_CCBIN',
        'CMAKE_CUDA_COMPILER',
        'CUDAToolkit_ROOT',
        'NVCUDASAMPLES_ROOT',
        'NVTOOLSEXT_PATH'
    )

    $envVarsCapture = [ordered]@{}

    foreach ($name in $namedVarNames) {
        $item = Get-Item -Path "Env:$name" -ErrorAction SilentlyContinue
        $envVarsCapture[$name] = if ($item) { $item.Value } else { $null }
    }

    # Capture all CUDA_PATH_V* versioned variables
    $cudaPathVars = Get-ChildItem -Path 'Env:CUDA_PATH_V*' -ErrorAction SilentlyContinue
    foreach ($var in $cudaPathVars) {
        $envVarsCapture[$var.Name] = $var.Value
    }

    # Capture any remaining CUDA_* or NVIDIA_* variables not already captured
    $allEnv = Get-ChildItem -Path 'Env:' -ErrorAction SilentlyContinue
    foreach ($var in $allEnv) {
        if (($var.Name -like 'CUDA_*' -or $var.Name -like 'NVIDIA_*') -and
            -not $envVarsCapture.Contains($var.Name)) {
            $envVarsCapture[$var.Name] = $var.Value
        }
    }

    # --- Capture NVIDIA/CUDA PATH segments -----------------------------------
    $nvidiaPathSegments = @(
        $env:PATH -split ';' |
        Where-Object { $_ -and ($_ -imatch 'NVIDIA' -or $_ -imatch 'CUDA') }
    )

    # --- Collect installed component versions via detection helpers ----------
    $componentVersions = [ordered]@{}

    # Driver version
    try {
        $driverVer = Get-NvidiaDriverVersion
        $componentVersions['Driver'] = $driverVer
    }
    catch {
        Write-Verbose "Driver version detection failed: $($_.Exception.Message)"
        $componentVersions['Driver'] = $null
    }

    # All CUDA installs side-by-side
    $cudaRoot      = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA'
    $cudaInstalls  = [System.Collections.Generic.List[hashtable]]::new()

    if (Test-Path $cudaRoot) {
        $cudaDirs = Get-ChildItem -Path $cudaRoot -Directory -Filter 'v*' -ErrorAction SilentlyContinue |
            Sort-Object Name -Descending

        foreach ($dir in $cudaDirs) {
            try {
                $ver = Get-CudaVersionFromPath -CudaPath $dir.FullName
                $cudaInstalls.Add(@{ Path = $dir.FullName; Version = $ver })
            }
            catch {
                Write-Verbose "CUDA version detection failed for '$($dir.FullName)': $($_.Exception.Message)"
                $cudaInstalls.Add(@{ Path = $dir.FullName; Version = $null })
            }
        }
    }
    $componentVersions['CudaInstalls'] = $cudaInstalls

    # cuDNN
    try {
        $paths   = Resolve-NvidiaInstallPath
        $cudnnP  = $paths['cuDNN']
        $componentVersions['CuDNNPath']    = $cudnnP
        $componentVersions['CuDNNVersion'] = if ($cudnnP) {
            Get-CudnnVersionFromHeader -CudnnPath $cudnnP
        }
        else { $null }
    }
    catch {
        Write-Verbose "cuDNN version detection failed: $($_.Exception.Message)"
        $componentVersions['CuDNNVersion'] = $null
    }

    # TensorRT
    try {
        $trtP  = (Resolve-NvidiaInstallPath)['TensorRT']
        $componentVersions['TensorRTPath']    = $trtP
        $componentVersions['TensorRTVersion'] = if ($trtP) {
            Get-TensorRtVersionFromHeader -TensorRtPath $trtP
        }
        else { $null }
    }
    catch {
        Write-Verbose "TensorRT version detection failed: $($_.Exception.Message)"
        $componentVersions['TensorRTVersion'] = $null
    }

    # --- Assemble the backup object ------------------------------------------
    $timestamp  = [System.DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ')
    $backupObj  = [ordered]@{
        SchemaVersion      = '1.0'
        Timestamp          = $timestamp
        MachineName        = $env:COMPUTERNAME
        UserName           = $env:USERNAME
        EnvVars            = $envVarsCapture
        NvidiaPathSegments = $nvidiaPathSegments
        ComponentVersions  = $componentVersions
    }

    $backupJson = $backupObj | ConvertTo-Json -Depth 10

    $backupFileName = "nvidia-env-$timestamp.json"
    $backupFilePath = Join-Path $BackupRoot $backupFileName

    if ($PSCmdlet.ShouldProcess($backupFilePath, 'Write NVIDIA environment backup')) {
        [System.IO.File]::WriteAllText($backupFilePath, $backupJson, [System.Text.Encoding]::UTF8)
        Write-Verbose "NVIDIA environment backup written to: $backupFilePath"
    }

    return $backupFilePath
}
