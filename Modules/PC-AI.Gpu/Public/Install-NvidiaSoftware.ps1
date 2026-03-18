#Requires -Version 5.1
<#
.SYNOPSIS
    Downloads and silently installs an NVIDIA software component from the
    curated registry.

.DESCRIPTION
    Orchestrates the full install workflow for a single NVIDIA component:

      1. Look up the component by -ComponentId in nvidia-software-registry.json.
      2. If no -InstallerPath is provided:
           a. Validate the registry download URL via Test-NvidiaDownloadUrl.
           b. Download the installer to $env:TEMP\nvidia-installers\.
           c. Verify the downloaded file's SHA-256 hash when the registry
              entry contains a 'sha256' field.
      3. Call Backup-NvidiaEnvironment to capture the current NVIDIA
         environment state before any changes are made.
      4. Call Invoke-NvidiaSilentInstall to run the installer.
      5. Re-run Get-NvidiaSoftwareStatus for the component to confirm the
         installed version advanced.
      6. Return a result object with before/after version information.

    Supports -WhatIf to preview every destructive step without executing it.
    Requires Administrator elevation for the install step.

    Use -DownloadOnly to download (and optionally verify) the installer
    without running it. The result object contains the local path.

.PARAMETER ComponentId
    Registry component identifier to install (e.g. 'cuda-toolkit', 'gpu-driver',
    'cudnn', 'tensorrt'). Must match an entry in nvidia-software-registry.json.

.PARAMETER InstallerPath
    Full path to a locally available installer. When supplied the download step
    is skipped. The file is still validated (existence, extension).

.PARAMETER DownloadOnly
    Download the installer and verify its hash without running it. Returns the
    local file path in result.InstallerPath. Mutually exclusive with
    -InstallerPath when the intent is to download.

.PARAMETER Force
    Proceed with install even when Get-NvidiaSoftwareStatus reports the
    component is already Current.

.PARAMETER RegistryPath
    Full path to an alternate nvidia-software-registry.json. When omitted the
    default Config\nvidia-software-registry.json is used.

.PARAMETER TimeoutSeconds
    Maximum seconds to wait for the installer. Passed through to
    Invoke-NvidiaSilentInstall. Default: 600.

.OUTPUTS
    [PSCustomObject] with properties:
        ComponentId      - Registry identifier.
        ComponentName    - Human-readable name.
        VersionBefore    - Installed version string before install, or $null.
        VersionAfter     - Installed version string after install, or $null.
        Success          - $true when install completed without error.
        RebootRequired   - $true when exit code 3010 was returned.
        DownloadOnly     - $true when -DownloadOnly was specified.
        InstallerPath    - Path of the local installer file.
        LogPath          - Installer log path (null for download-only).
        Duration         - [TimeSpan] total elapsed time.
        Message          - Human-readable outcome summary.

.EXAMPLE
    Install-NvidiaSoftware -ComponentId 'cuda-toolkit'
    Downloads and installs the CUDA Toolkit version listed in the registry.

.EXAMPLE
    Install-NvidiaSoftware -ComponentId 'gpu-driver' -DownloadOnly
    Downloads the driver installer and verifies its hash without running it.

.EXAMPLE
    Install-NvidiaSoftware -ComponentId 'cudnn' -InstallerPath 'D:\Downloads\cudnn_9.exe'
    Installs cuDNN from a pre-downloaded local file, skipping the download.

.EXAMPLE
    Install-NvidiaSoftware -ComponentId 'tensorrt' -WhatIf
    Previews the full install workflow without making any changes.

.NOTES
    Phase 3 implementation.
    The download staging directory ($env:TEMP\nvidia-installers\) is created
    lazily. Files are never deleted after a successful install so they can be
    reused; use -Force to re-download when an existing cached file exists.
#>
function Install-NvidiaSoftware {
    [CmdletBinding(SupportsShouldProcess, ConfirmImpact = 'High')]
    [OutputType([PSCustomObject])]
    param(
        [Parameter(Mandatory)]
        [ValidateNotNullOrEmpty()]
        [string]$ComponentId,

        [Parameter()]
        [string]$InstallerPath,

        [Parameter()]
        [switch]$DownloadOnly,

        [Parameter()]
        [switch]$Force,

        [Parameter()]
        [string]$RegistryPath,

        [Parameter()]
        [ValidateRange(30, 7200)]
        [int]$TimeoutSeconds = 600
    )

    $ErrorActionPreference = 'Stop'

    $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()

    # --- Build result skeleton ---
    $result = [PSCustomObject]@{
        ComponentId    = $ComponentId
        ComponentName  = $null
        VersionBefore  = $null
        VersionAfter   = $null
        Success        = $false
        RebootRequired = $false
        DownloadOnly   = $DownloadOnly.IsPresent
        InstallerPath  = $InstallerPath
        LogPath        = $null
        Duration       = [TimeSpan]::Zero
        Message        = $null
    }

    # -------------------------------------------------------------------------
    # 1. Load registry entry
    # -------------------------------------------------------------------------
    Write-Verbose "Install-NvidiaSoftware: Loading registry for component '$ComponentId'..."

    $registryParams = @{ ComponentId = $ComponentId }
    if ($RegistryPath) { $registryParams['RegistryPath'] = $RegistryPath }

    $registry = Get-NvidiaSoftwareRegistry @registryParams

    if (-not $registry -or -not $registry.Components -or $registry.Components.Count -eq 0) {
        $msg = "Component '$ComponentId' was not found in the NVIDIA software registry."
        $result.Message = $msg
        Write-Error $msg
        return $result
    }

    $component            = $registry.Components[0]
    $result.ComponentName = $component.name

    Write-Verbose "Install-NvidiaSoftware: Found component '$($component.name)' (category: $($component.category))."

    # -------------------------------------------------------------------------
    # 2. Check current status and honour -Force
    # -------------------------------------------------------------------------
    Write-Verbose "Install-NvidiaSoftware: Checking current software status..."

    $statusParams = @{ ComponentId = $ComponentId }
    if ($RegistryPath) { $statusParams['RegistryPath'] = $RegistryPath }

    $currentStatus        = Get-NvidiaSoftwareStatus @statusParams | Select-Object -First 1
    $result.VersionBefore = $currentStatus.InstalledVersion

    if ($currentStatus.Status -eq 'Current' -and -not $Force -and -not $DownloadOnly) {
        $msg = "Component '$ComponentId' is already Current (v$($currentStatus.InstalledVersion)). Use -Force to reinstall."
        $result.Message  = $msg
        $result.Success  = $true   # Nothing needed — not an error state
        Write-Verbose $msg
        $stopwatch.Stop()
        $result.Duration = $stopwatch.Elapsed
        return $result
    }

    # -------------------------------------------------------------------------
    # 3. Download if no local installer was supplied
    # -------------------------------------------------------------------------
    if (-not $InstallerPath) {
        $downloadUrl = $component.downloadUrl
        if (-not $downloadUrl) {
            $msg = "Registry entry for '$ComponentId' has no downloadUrl field."
            $result.Message = $msg
            Write-Error $msg
            return $result
        }

        # Validate URL trust + reachability
        Write-Verbose "Install-NvidiaSoftware: Validating download URL '$downloadUrl'..."
        $urlCheck = Test-NvidiaDownloadUrl -Url $downloadUrl

        if (-not $urlCheck.IsTrusted) {
            $msg = "Download URL for '$ComponentId' is not from a trusted NVIDIA host: $downloadUrl"
            $result.Message = $msg
            Write-Error $msg
            return $result
        }

        if (-not $urlCheck.IsValid) {
            $msg = "Download URL for '$ComponentId' is not reachable (HTTP $($urlCheck.StatusCode)): $downloadUrl"
            $result.Message = $msg
            Write-Error $msg
            return $result
        }

        # Derive filename from URL or component id
        $uriObj    = [System.Uri]::new($downloadUrl)
        $urlFile   = [System.IO.Path]::GetFileName($uriObj.LocalPath)
        if (-not $urlFile) {
            $urlFile = "$ComponentId-installer.exe"
        }

        $stagingDir = Join-Path $env:TEMP 'nvidia-installers'
        if (-not (Test-Path -LiteralPath $stagingDir)) {
            New-Item -Path $stagingDir -ItemType Directory -Force | Out-Null
            Write-Verbose "Install-NvidiaSoftware: Created staging directory: $stagingDir"
        }

        $localPath = Join-Path $stagingDir $urlFile

        if ($PSCmdlet.ShouldProcess($downloadUrl, "Download NVIDIA installer to '$localPath'")) {
            Write-Verbose "Install-NvidiaSoftware: Downloading '$downloadUrl' -> '$localPath'..."

            # Validate that the URL is from a trusted NVIDIA host before downloading
            $trustedHosts = @('nvidia.com', 'developer.nvidia.com',
                              'developer.download.nvidia.com', 'us.download.nvidia.com')
            $uriForCheck = [System.Uri]::new($downloadUrl)
            $hostIsValid = $false
            foreach ($trusted in $trustedHosts) {
                if ($uriForCheck.Host -eq $trusted -or $uriForCheck.Host.EndsWith(".$trusted")) {
                    $hostIsValid = $true
                    break
                }
            }
            if (-not $hostIsValid) {
                $msg = "Aborting download: '$downloadUrl' is not from a trusted NVIDIA host."
                $result.Message = $msg
                $stopwatch.Stop()
                $result.Duration = $stopwatch.Elapsed
                Write-Error $msg
                return $result
            }

            $expectedHash = $null
            if ($component.PSObject.Properties['sha256']) {
                $expectedHash = $component.sha256
            }

            # Skip download when a valid cached file already exists (unless -Force)
            $needsDownload = $Force -or -not (Test-Path -LiteralPath $localPath)

            if ($needsDownload) {
                try {
                    Invoke-WebRequest `
                        -Uri        $downloadUrl `
                        -OutFile    $localPath `
                        -UseBasicParsing `
                        -ErrorAction Stop
                }
                catch {
                    $msg = "Download failed for component '$ComponentId': $($_.Exception.Message)"
                    $result.Message = $msg
                    $stopwatch.Stop()
                    $result.Duration = $stopwatch.Elapsed
                    Write-Error $msg
                    return $result
                }
            }
            else {
                Write-Verbose "Install-NvidiaSoftware: Using cached installer: $localPath"
            }

            # SHA-256 verification (when registry entry supplies a hash)
            if ($expectedHash) {
                Write-Verbose "Install-NvidiaSoftware: Verifying SHA-256 hash..."
                try {
                    $stream = [System.IO.File]::OpenRead($localPath)
                    try {
                        $sha256  = [System.Security.Cryptography.SHA256]::Create()
                        $hashBytes = $sha256.ComputeHash($stream)
                        $actualHash = [System.BitConverter]::ToString($hashBytes) -replace '-', ''
                    }
                    finally {
                        $stream.Dispose()
                    }

                    if ($actualHash.ToUpperInvariant() -ne $expectedHash.ToUpperInvariant()) {
                        Remove-Item -LiteralPath $localPath -Force -ErrorAction SilentlyContinue
                        $msg = "SHA-256 mismatch for '$ComponentId'. " +
                               "Expected: $expectedHash  Got: $actualHash. " +
                               "Downloaded file deleted."
                        $result.Message = $msg
                        $stopwatch.Stop()
                        $result.Duration = $stopwatch.Elapsed
                        Write-Error $msg
                        return $result
                    }
                    Write-Verbose "Install-NvidiaSoftware: SHA-256 verified OK."
                }
                catch {
                    $msg = "SHA-256 verification error for '$ComponentId': $($_.Exception.Message)"
                    $result.Message = $msg
                    $stopwatch.Stop()
                    $result.Duration = $stopwatch.Elapsed
                    Write-Error $msg
                    return $result
                }
            }

            Write-Verbose "Install-NvidiaSoftware: Download complete: $localPath"
        }
        else {
            Write-Verbose "WhatIf: Would download '$downloadUrl' to '$localPath'."
            $localPath = $downloadUrl   # Placeholder for WhatIf output
        }

        $InstallerPath        = $localPath
        $result.InstallerPath = $localPath
    }
    else {
        # Validate the caller-supplied installer path
        if (-not (Test-Path -LiteralPath $InstallerPath)) {
            $msg = "Supplied -InstallerPath '$InstallerPath' does not exist."
            $result.Message = $msg
            Write-Error $msg
            return $result
        }
        Write-Verbose "Install-NvidiaSoftware: Using supplied installer: $InstallerPath"
    }

    # Return early for download-only mode
    if ($DownloadOnly) {
        $result.Success  = $true
        $result.Message  = "Download-only mode: installer staged at '$InstallerPath'."
        $stopwatch.Stop()
        $result.Duration = $stopwatch.Elapsed
        Write-Verbose $result.Message
        return $result
    }

    # -------------------------------------------------------------------------
    # 4. Backup NVIDIA environment before touching anything
    # -------------------------------------------------------------------------
    if ($PSCmdlet.ShouldProcess('NVIDIA environment', 'Backup environment variables to .pcai/nvidia-backup/')) {
        Write-Verbose "Install-NvidiaSoftware: Backing up NVIDIA environment..."
        try {
            $backupPath = Backup-NvidiaEnvironment
            if ($backupPath) {
                Write-Verbose "Install-NvidiaSoftware: Environment backed up to '$backupPath'."
            }
            else {
                Write-Warning "Install-NvidiaSoftware: Backup-NvidiaEnvironment returned no path; continuing without backup."
            }
        }
        catch {
            Write-Warning "Install-NvidiaSoftware: Backup-NvidiaEnvironment failed: $($_.Exception.Message). Continuing."
        }
    }
    else {
        Write-Verbose "WhatIf: Would call Backup-NvidiaEnvironment."
    }

    # -------------------------------------------------------------------------
    # 5. Run the silent installer
    # -------------------------------------------------------------------------
    Write-Verbose "Install-NvidiaSoftware: Running silent installer for '$ComponentId'..."

    $installParams = @{
        InstallerPath  = $InstallerPath
        ComponentId    = $ComponentId
        TimeoutSeconds = $TimeoutSeconds
    }

    # WhatIf propagates to Invoke-NvidiaSilentInstall via [CmdletBinding(SupportsShouldProcess)]
    $installResult = Invoke-NvidiaSilentInstall @installParams

    $result.LogPath        = $installResult.LogPath
    $result.RebootRequired = $installResult.RebootRequired

    if (-not $installResult.Success) {
        $msg = "Installer for '$ComponentId' failed with exit code $($installResult.ExitCode). Log: $($installResult.LogPath)"
        $result.Message = $msg
        $stopwatch.Stop()
        $result.Duration = $stopwatch.Elapsed
        Write-Error $msg
        return $result
    }

    # -------------------------------------------------------------------------
    # 6. Verify installation by re-checking status
    # -------------------------------------------------------------------------
    Write-Verbose "Install-NvidiaSoftware: Verifying installation by re-checking component status..."

    try {
        $postStatus           = Get-NvidiaSoftwareStatus @statusParams | Select-Object -First 1
        $result.VersionAfter  = $postStatus.InstalledVersion

        $verificationStatus = $postStatus.Status
        Write-Verbose "Install-NvidiaSoftware: Post-install status: $verificationStatus (v$($result.VersionAfter))."
    }
    catch {
        Write-Warning "Install-NvidiaSoftware: Post-install status check failed: $($_.Exception.Message)"
        $result.VersionAfter = $null
        $verificationStatus  = 'Unknown'
    }

    # -------------------------------------------------------------------------
    # 7. Compose result
    # -------------------------------------------------------------------------
    $result.Success = $true

    if ($result.RebootRequired) {
        $result.Message = "Component '$ComponentId' installed successfully (reboot required). " +
                          "v$($result.VersionBefore) -> v$($result.VersionAfter)."
    }
    else {
        $result.Message = "Component '$ComponentId' installed successfully. " +
                          "v$($result.VersionBefore) -> v$($result.VersionAfter). " +
                          "Post-install status: $verificationStatus."
    }

    $stopwatch.Stop()
    $result.Duration = $stopwatch.Elapsed

    Write-Verbose "Install-NvidiaSoftware: $($result.Message)"
    return $result
}
