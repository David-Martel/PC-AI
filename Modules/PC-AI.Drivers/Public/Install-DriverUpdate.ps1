#Requires -Version 5.1
<#
.SYNOPSIS
    Downloads and installs a driver update for a registry-tracked device.

.DESCRIPTION
    Looks up the specified device by its registry ID, downloads the installer from
    the device's trusted source URL, and runs the appropriate installation method
    (exe, zip-with-exe, or msi).

    Devices with installerType 'none' or 'windows-update' are reported but skipped;
    no download or execution is attempted.

    Actual installation requires elevation and a registry under the protected
    Program Files PC-AI.DriverTrust directory. The registry pins both the package
    SHA256 and the signer's certificate SHA256. Each install uses a fresh protected
    stage, verifies the actual executable or MSI, and removes the stage after use.
    INF packages remain available through -DownloadOnly.

    -DownloadOnly leaves the installer file in $DownloadDir for manual use. The
    digest is optional in this mode and is checked when present. Existing bytes
    with a different digest are preserved. -WhatIf suppresses directory creation,
    download and execution in both modes.

.LINK
    docs/driver-installation.md

.PARAMETER DeviceId
    The registry entry id (e.g. 'realtek-rtl8156'). Must match exactly.

.PARAMETER RegistryPath
    Registry path for a preview or download. Installs use driver-registry.json in
    the protected Program Files PC-AI.DriverTrust directory.

.PARAMETER DownloadDir
    Directory for -DownloadOnly output, created if needed. Defaults to
    $env:TEMP\DriverUpdates. Installation stages fresh bytes in its protected directory.

.PARAMETER Force
    Re-download the installer even if the file already exists in $DownloadDir.

.PARAMETER DownloadOnly
    Download the installer but do not execute it. Admin rights are not required.

.EXAMPLE
    Install-DriverUpdate -DeviceId $deviceId -Confirm
    Installs an entry from the configured machine trust registry.

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

    # A preview or download may use the repository registry without elevation.
    # Executing always loads the protected machine registry before selecting bytes.
    $trustRoot = $null
    if (-not $DownloadOnly -and -not $WhatIfPreference) {
        try {
            if (-not (Test-AdminElevation)) {
                throw 'Administrator elevation is required to install drivers. Use -DownloadOnly to acquire the package.'
            }
            $trustRoot = Get-DriverInstallTrustRoot
            $trustedRegistry = Join-Path $trustRoot 'driver-registry.json'
            if ($RegistryPath -and [IO.Path]::GetFullPath($RegistryPath) -ine $trustedRegistry) {
                throw "The installation registry must be '$trustedRegistry'. Use -DownloadOnly with any other registry."
            }
            Assert-DriverInstallTrustPath -Path $trustRoot -Directory
            Assert-DriverInstallTrustPath -Path $trustedRegistry
            $RegistryPath = $trustedRegistry
        }
        catch {
            return New-InstallResult -Id $DeviceId -Action 'Failed' -Success $false -Message $_.Exception.Message
        }
    }

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
    $publisherSha256 = $entry.driver.publisherCertificateSha256
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
    if ($fileBaseName -notmatch '\A[a-zA-Z0-9][a-zA-Z0-9._-]{0,126}\z' -or
        $fileBaseName -match '\A(?:AUX|CON|NUL|PRN|COM[1-9]|LPT[1-9])(?:\.|\z)') {
        return New-InstallResult -Id $DeviceId -Name $deviceName -Action 'Failed' -Success $false `
            -Message 'The registry must identify an ordinary installer filename.'
    }

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
        $prerequisiteError = if ($installerType -notin @('exe', 'zip-with-exe', 'msi')) {
            "Automatic installation does not support '$installerType'. Use -DownloadOnly or Windows Update."
        }
        elseif ($sha256 -isnot [string] -or $sha256 -notmatch '\A[0-9a-fA-F]{64}\z') {
            "A 64-character hexadecimal SHA256 digest is required to install '$deviceName'."
        }
        elseif ($publisherSha256 -isnot [string] -or $publisherSha256 -notmatch '\A[0-9a-fA-F]{64}\z') {
            "A 64-character publisher certificate SHA256 pin is required to install '$deviceName'."
        }
        elseif (-not $trustedHosts.Count -or $downloadUrl -notmatch '\Ahttps://') {
            "An HTTPS package URL and a trusted source are required to install '$deviceName'."
        }

        if ($prerequisiteError) {
            Write-Warning $prerequisiteError
            return New-InstallResult -Id $DeviceId -Name $deviceName -Action 'Failed' `
                -FilePath $null -ExitCode $null -Success $false -Message $prerequisiteError
        }
    }

    # Separate the reusable download cache from the one-time execution stage.
    $stage = $null
    $downloaded = $null
    try {
        if ($DownloadOnly) {
            if (-not (Test-Path -LiteralPath $DownloadDir)) {
                New-Item -ItemType Directory -Path $DownloadDir -Force -ErrorAction Stop | Out-Null
            }
        }
        else {
            $stage = New-DriverInstallStage -TrustRoot $trustRoot
            Assert-DriverInstallTrustPath -Path $stage -Directory
            $outFile = Join-Path $stage ($fileBaseName + $fileExtension)
        }

        Write-Host "  [*] Downloading driver for '$deviceName' ..." -ForegroundColor Gray
        $downloadParams = @{
            Url           = $downloadUrl
            OutFile       = $outFile
            TrustedHosts  = $trustedHosts
            ForceDownload = $Force.IsPresent
        }
        if ($sha256) { $downloadParams['ExpectedSha256'] = $sha256 }
        $downloaded = Invoke-TrustedDownload @downloadParams
        if (-not $downloaded) {
            throw "Download failed for '$deviceName'."
        }
        if ($DownloadOnly) {
            return New-InstallResult -Id $DeviceId -Name $deviceName -Action 'DownloadOnly' `
                -FilePath $downloaded -ExitCode $null -Success $true -Message "Download complete. Installer saved to: $downloaded"
        }
        if ($downloaded -ine $outFile) {
            throw 'The download did not return the expected protected installer path.'
        }

        Assert-DriverInstallTrustPath -Path $outFile
        $actualSha256 = (Get-FileHash -LiteralPath $outFile -Algorithm SHA256 -ErrorAction Stop).Hash
        if ($actualSha256 -ne $sha256) {
            throw "SHA256 mismatch for installer '$outFile'."
        }

        $payload = $outFile
        if ($installerType -eq 'zip-with-exe') {
            $extractDir = Join-Path $stage 'payload'
            New-Item -ItemType Directory -Path $extractDir -ErrorAction Stop | Out-Null
            Assert-DriverInstallTrustPath -Path $extractDir -Directory
            Expand-Archive -LiteralPath $outFile -DestinationPath $extractDir -ErrorAction Stop
            $payload = Assert-DriverArchivePayload -Path $extractDir
        }

        Assert-DriverInstallerSignature -Path $payload -PublisherCertificateSha256 $publisherSha256
        $processParams = @{
            FilePath = $payload
            WorkingDirectory = (Split-Path $payload -Parent)
            Wait = $true
            PassThru = $true
            ErrorAction = 'Stop'
        }
        if ($installerType -eq 'msi') {
            $processParams.FilePath = Join-Path ([Environment]::SystemDirectory) 'msiexec.exe'
            $processParams.ArgumentList = "/i `"$payload`" /qb"
        }
        $exitCode = (Start-Process @processParams).ExitCode

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

        return New-InstallResult -Id $DeviceId -Name $deviceName -Action 'Install' `
            -FilePath $null -ExitCode $exitCode -Success $success -Message $msg

    }
    catch {
        $msg = "Installer preparation or launch failed: $($_.Exception.Message)"
        Write-Warning "  [!!] $msg"
        return New-InstallResult -Id $DeviceId -Name $deviceName -Action 'Failed' `
            -FilePath $null -ExitCode $null -Success $false -Message $msg
    }
    finally {
        if ($stage) {
            try {
                $expectedParent = [IO.Path]::GetFullPath($trustRoot)
                $resolvedStage = [IO.Path]::GetFullPath($stage)
                if ([IO.Path]::GetDirectoryName($resolvedStage) -ine $expectedParent -or
                    [IO.Path]::GetFileName($resolvedStage) -notmatch '\Ainstall-[0-9a-f]{32}\z') {
                    throw 'The staging path is outside its expected install directory.'
                }
                $stageItem = Get-Item -LiteralPath $resolvedStage -Force -ErrorAction Stop
                if (-not $stageItem.PSIsContainer -or $stageItem.Attributes -band [IO.FileAttributes]::ReparsePoint) {
                    throw 'The installation stage is not an ordinary directory.'
                }
                Remove-Item -LiteralPath $stageItem.FullName -Recurse -Force -ErrorAction Stop
            }
            catch {
                Write-Warning "Driver staging cleanup failed: $($_.Exception.Message)"
            }
        }
    }
}
