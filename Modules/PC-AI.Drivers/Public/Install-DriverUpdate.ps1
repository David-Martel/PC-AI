#Requires -Version 5.1
<#
.SYNOPSIS
    Downloads and installs a driver update for a registry-tracked device.

.DESCRIPTION
    Looks up the specified device by its registry ID, downloads the installer from
    the device's trusted source URL, and runs the appropriate installation method
    (exe, zip-with-exe, inf, or msi).

    Devices with installerType 'none' or 'windows-update' are reported but skipped;
    no download or execution is attempted.

    Actual installation requires administrator rights and a trusted registry SHA256
    digest of exactly 64 hexadecimal characters. The downloaded or cached installer
    is checked again immediately before use. An archive is extracted into a fresh
    per-run directory under $DownloadDir.

    -DownloadOnly leaves the installer file in $DownloadDir for manual use. The
    digest is optional in this mode and is checked when present. Existing bytes
    with a different digest are preserved. -WhatIf suppresses directory creation,
    download and execution in both modes.

.LINK
    docs/driver-installation.md

.PARAMETER DeviceId
    The registry entry id (e.g. 'realtek-rtl8156'). Must match exactly.

.PARAMETER RegistryPath
    Full path to an alternate driver-registry.json. When omitted the default path
    under PC_AI\Config\ is resolved automatically.

.PARAMETER DownloadDir
    Directory for downloaded installer files. Created if it does not exist.
    Defaults to $env:TEMP\DriverUpdates.

.PARAMETER Force
    Re-download the installer even if the file already exists in $DownloadDir.

.PARAMETER DownloadOnly
    Download the installer but do not execute it. Admin rights are not required.

.EXAMPLE
    Install-DriverUpdate -DeviceId 'realtek-rtl8156'
    Downloads and installs the Realtek USB ethernet driver.

.EXAMPLE
    Install-DriverUpdate -DeviceId 'caldigit-element-hub' -DownloadOnly
    Downloads the CalDigit firmware updater without launching it.

.EXAMPLE
    Install-DriverUpdate -DeviceId 'realtek-rtl8156' -WhatIf
    Shows what would be downloaded and installed without doing anything.

.OUTPUTS
    PSCustomObject with properties: DeviceId, DeviceName, Action, FilePath,
    ExitCode, Success, Message
#>
function Install-DriverUpdate {
    [CmdletBinding(SupportsShouldProcess, ConfirmImpact = 'High')]
    [OutputType([PSCustomObject])]
    param(
        [Parameter(Mandatory)]
        [string]$DeviceId,

        [Parameter()]
        [string]$RegistryPath,

        [Parameter()]
        [string]$DownloadDir = (Join-Path $env:TEMP 'DriverUpdates'),

        [Parameter()]
        [switch]$Force,

        [Parameter()]
        [switch]$DownloadOnly
    )

    # Helper: produce a consistent result object
    function New-InstallResult {
        param(
            [string]$Id,
            [string]$Name,
            [string]$Action,
            [string]$FilePath,
            [object]$ExitCode,
            [bool]$Success,
            [string]$Message
        )
        return [PSCustomObject]@{
            DeviceId   = $Id
            DeviceName = $Name
            Action     = $Action
            FilePath   = $FilePath
            ExitCode   = $ExitCode
            Success    = $Success
            Message    = $Message
        }
    }

    # Load the registry before evaluating the requirements for the selected action.
    $regParams = @{}
    if ($RegistryPath) { $regParams['RegistryPath'] = $RegistryPath }
    $reg = Get-DriverRegistry @regParams

    if (-not $reg) {
        Write-Error 'Install-DriverUpdate: could not load driver registry.'
        return $null
    }

    $entry = $reg.Devices | Where-Object { $_.id -eq $DeviceId } | Select-Object -First 1

    # --- Step 3: Validate entry ---
    if (-not $entry) {
        Write-Error "Install-DriverUpdate: device ID '$DeviceId' not found in registry." -ErrorAction $(if ($PSBoundParameters.ContainsKey('ErrorAction')) { $PSBoundParameters['ErrorAction'] } else { $ErrorActionPreference })
        return $null
    }

    $deviceName    = $entry.name
    $installerType = $entry.driver.installerType
    $downloadUrl   = $entry.driver.downloadUrl
    $manualUrl     = $entry.driver.manualDownloadUrl
    $sha256        = $entry.driver.sha256
    $sharedGroup   = $entry.sharedDriverGroup

    Write-Verbose "Install-DriverUpdate: '$DeviceId' - name='$deviceName' installerType='$installerType'"

    # --- Step 4: Skip unsupported installer types ---
    if ($installerType -eq 'none') {
        $msg = "No driver update available for '$deviceName' (uses inbox driver). No action taken."
        Write-Host "  [--] $msg" -ForegroundColor DarkGray
        return New-InstallResult -Id $DeviceId -Name $deviceName -Action 'Skipped' `
            -FilePath $null -ExitCode $null -Success $true -Message $msg
    }

    if ($installerType -eq 'windows-update') {
        $msg = "Driver for '$deviceName' is managed by Windows Update. Run Windows Update to get the latest version."
        Write-Host "  [--] $msg" -ForegroundColor DarkGray
        return New-InstallResult -Id $DeviceId -Name $deviceName -Action 'Skipped' `
            -FilePath $null -ExitCode $null -Success $true -Message $msg
    }

    # --- Step 5: Build trusted hosts list from registry ---
    $trustedHosts = @()
    if ($reg.TrustedSources) {
        foreach ($src in $reg.TrustedSources) {
            if ($src.baseUrl) {
                $uri = $null
                try { $uri = [Uri]$src.baseUrl } catch { }
                if ($uri -and $uri.Host) {
                    $trustedHosts += $uri.Host
                }
            }
        }
    }
    Write-Verbose "Trusted hosts: $($trustedHosts -join ', ')"

    # --- Step 6: Determine output filename without writing anything ---
    # Use the shared driver group name (if present) to avoid downloading the same
    # package multiple times for sibling devices (e.g. RTL8156 and RTL8157).
    $fileBaseName = if ($sharedGroup) { $sharedGroup } else { $DeviceId }

    $fileExtension = '.exe'
    if ($installerType -eq 'zip-with-exe') { $fileExtension = '.zip' }
    elseif ($installerType -eq 'inf')      { $fileExtension = '.zip' }
    elseif ($installerType -eq 'msi')      { $fileExtension = '.msi' }

    $outFile = Join-Path $DownloadDir ($fileBaseName + $fileExtension)
    if ($DownloadOnly -and -not $sha256) {
        # Unpinned bytes cannot be proven to belong to the selected package URL.
        $downloadId = [guid]::NewGuid().ToString('N')
        $outFile = Join-Path $DownloadDir "$fileBaseName-unverified-$downloadId$fileExtension"
    }

    if (-not $downloadUrl) {
        $msg = "No download URL configured for '$deviceName'."
        if ($manualUrl) { $msg += " Manual download: $manualUrl" }
        Write-Warning $msg
        return New-InstallResult -Id $DeviceId -Name $deviceName -Action 'Failed' `
            -FilePath $null -ExitCode $null -Success $false -Message $msg
    }

    # --- Step 7: Approve either mode before directory creation or download ---
    $downloadAction = if ($DownloadOnly) { "Download driver from $downloadUrl" } else { "Download and install driver from $downloadUrl" }
    if (-not $PSCmdlet.ShouldProcess($deviceName, $downloadAction)) {
        return New-InstallResult -Id $DeviceId -Name $deviceName -Action 'WhatIf' `
            -FilePath $null -ExitCode $null -Success $true -Message 'WhatIf: no changes made.'
    }

    if (-not $DownloadOnly) {
        $prerequisiteError = if ($installerType -notin @('exe', 'zip-with-exe', 'inf', 'msi')) {
            "Unsupported installerType '$installerType' for device '$DeviceId'."
        }
        elseif (-not (Test-AdminElevation)) {
            "Administrator elevation is required to install '$deviceName'. Re-run as Administrator, or use -DownloadOnly."
        }
        elseif ($sha256 -isnot [string] -or $sha256 -notmatch '\A[0-9a-fA-F]{64}\z') {
            "A 64-character hexadecimal SHA256 digest is required to install '$deviceName'. Supply a trusted digest in the device registry, or use -DownloadOnly."
        }

        if ($prerequisiteError) {
            Write-Warning $prerequisiteError
            return New-InstallResult -Id $DeviceId -Name $deviceName -Action 'Failed' `
                -FilePath $null -ExitCode $null -Success $false -Message $prerequisiteError
        }
    }

    # Create the directory only after approval and admission.
    if (-not (Test-Path -LiteralPath $DownloadDir)) {
        New-Item -ItemType Directory -Path $DownloadDir -Force | Out-Null
        Write-Verbose "Created download directory: $DownloadDir"
    }

    # --- Step 9: Download ---
    Write-Host "  [*] Downloading driver for '$deviceName' ..." -ForegroundColor Gray

    $downloadParams = @{
        Url          = $downloadUrl
        OutFile      = $outFile
        TrustedHosts = $trustedHosts
        ForceDownload = $Force.IsPresent
    }
    if ($sha256) { $downloadParams['ExpectedSha256'] = $sha256 }

    $downloaded = Invoke-TrustedDownload @downloadParams

    # --- Step 9: Handle download failure ---
    if (-not $downloaded) {
        $msg = "Download failed for '$deviceName'."
        if ($manualUrl) {
            Write-Warning "  [!!] $msg"
            Write-Warning "  [!!] Manual download: $manualUrl"
            Write-Warning "  [!!] Save as: $outFile  then re-run."
        }
        else {
            Write-Warning "  [!!] $msg"
        }
        return New-InstallResult -Id $DeviceId -Name $deviceName -Action 'Failed' `
            -FilePath $null -ExitCode $null -Success $false -Message $msg
    }

    Write-Host "  [OK] Downloaded: $(Split-Path $downloaded -Leaf)" -ForegroundColor Green

    # --- Step 10: DownloadOnly exit ---
    if ($DownloadOnly) {
        $msg = "Download complete. Installer saved to: $downloaded"
        Write-Host "  [--] $msg" -ForegroundColor DarkGray
        return New-InstallResult -Id $DeviceId -Name $deviceName -Action 'DownloadOnly' `
            -FilePath $downloaded -ExitCode $null -Success $true -Message $msg
    }

    # --- Step 11: Install ---
    $action  = 'Install'
    $exitCode = $null

    try {
        # Verify the exact package path used by the installer or archive reader.
        $actualSha256 = (Get-FileHash -LiteralPath $downloaded -Algorithm SHA256 -ErrorAction Stop).Hash
        if ($actualSha256 -ne $sha256) {
            throw "SHA256 mismatch for installer '$downloaded'."
        }

        if ($installerType -in @('zip-with-exe', 'inf')) {
            $extractDir = Join-Path $DownloadDir ("install-" + [guid]::NewGuid().ToString('N'))
            New-Item -ItemType Directory -Path $extractDir -ErrorAction Stop | Out-Null
            Expand-Archive -LiteralPath $downloaded -DestinationPath $extractDir -ErrorAction Stop
        }

        if ($installerType -eq 'exe') {
            Write-Host "  [*] Launching installer: $(Split-Path $downloaded -Leaf) ..." -ForegroundColor Gray
            $proc = Start-Process -FilePath $downloaded -Wait -PassThru -ErrorAction Stop
            $exitCode = $proc.ExitCode
        }
        elseif ($installerType -eq 'zip-with-exe') {
            $exeFile = Get-ChildItem -LiteralPath $extractDir -Recurse -File -Filter '*.exe' -ErrorAction Stop |
                Select-Object -First 1
            if (-not $exeFile) {
                $msg = "No .exe found in extracted archive at: $extractDir"
                Write-Warning "  [!!] $msg"
                return New-InstallResult -Id $DeviceId -Name $deviceName -Action 'Failed' `
                    -FilePath $downloaded -ExitCode $null -Success $false -Message $msg
            }
            Write-Host "  [*] Launching installer: $($exeFile.Name) ..." -ForegroundColor Gray
            $proc = Start-Process -FilePath $exeFile.FullName -Wait -PassThru -ErrorAction Stop
            $exitCode = $proc.ExitCode
        }
        elseif ($installerType -eq 'inf') {
            $infFile = Get-ChildItem -LiteralPath $extractDir -Recurse -File -Filter '*.inf' -ErrorAction Stop |
                Select-Object -First 1
            if (-not $infFile) {
                $msg = "No .inf found in extracted archive at: $extractDir"
                Write-Warning "  [!!] $msg"
                return New-InstallResult -Id $DeviceId -Name $deviceName -Action 'Failed' `
                    -FilePath $downloaded -ExitCode $null -Success $false -Message $msg
            }
            Write-Host "  [*] Installing via pnputil: $($infFile.Name) ..." -ForegroundColor Gray
            $proc = Start-Process -FilePath 'pnputil.exe' `
                -ArgumentList "/add-driver `"$($infFile.FullName)`" /install" `
                -Wait -PassThru -ErrorAction Stop
            $exitCode = $proc.ExitCode
        }
        elseif ($installerType -eq 'msi') {
            Write-Host "  [*] Launching MSI installer: $(Split-Path $downloaded -Leaf) ..." -ForegroundColor Gray
            $proc = Start-Process -FilePath 'msiexec.exe' `
                -ArgumentList "/i `"$downloaded`" /qb" `
                -Wait -PassThru -ErrorAction Stop
            $exitCode = $proc.ExitCode
        }

        $rebootRequired = ($exitCode -eq 3010)
        $success = ($exitCode -eq 0 -or $rebootRequired)
        if ($rebootRequired) {
            $msg = "Installation succeeded (exit code 3010). A reboot is required to complete."
        } elseif ($exitCode -eq 0) {
            $msg = "Installation completed (exit code 0)."
        } else {
            $msg = "Installer exited with code $exitCode. Check for errors."
        }

        if ($success) {
            Write-Host "  [OK] $msg" -ForegroundColor Green
        }
        else {
            Write-Warning "  [!!] $msg"
        }

        return New-InstallResult -Id $DeviceId -Name $deviceName -Action $action `
            -FilePath $downloaded -ExitCode $exitCode -Success $success -Message $msg

    }
    catch {
        $msg = "Installer preparation or launch failed: $($_.Exception.Message)"
        Write-Warning "  [!!] $msg"
        return New-InstallResult -Id $DeviceId -Name $deviceName -Action 'Failed' `
            -FilePath $downloaded -ExitCode $null -Success $false -Message $msg
    }
}
