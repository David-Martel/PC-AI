#Requires -RunAsAdministrator
<#
.SYNOPSIS
    Downloads and installs latest drivers for USB hubs, docks, and ethernet adapters.

.DESCRIPTION
    Idempotent, severable driver update script for USB peripherals detected on this system.
    Each section is independent — the script can be interrupted and re-run safely.

    Detected devices:
      - Realtek RTL8156 (USB 2.5GbE) — VID_0BDA&PID_8156
      - Realtek RTL8157 (USB 5GbE)   — VID_0BDA&PID_8157
      - CalDigit Element Hub (TB4/USB4) — VID_8087&PID_0B26

    The Realtek NICs share a single unified driver package.
    The CalDigit Element Hub uses a standalone firmware updater.

.PARAMETER DownloadOnly
    Download drivers without installing them.

.PARAMETER SkipRealtek
    Skip the Realtek USB ethernet driver update.

.PARAMETER SkipCalDigit
    Skip the CalDigit firmware update.

.PARAMETER Force
    Force re-download and reinstall even if versions match.

.PARAMETER DownloadDir
    Directory for downloaded driver packages. Defaults to $env:TEMP\UsbDriverUpdates.

.EXAMPLE
    .\Update-UsbDrivers.ps1
    .\Update-UsbDrivers.ps1 -DownloadOnly
    .\Update-UsbDrivers.ps1 -SkipCalDigit -Force

.NOTES
    Created: 2026-03-11
    Requires: Administrator privileges, internet access
    Re-runnable: Yes — skips already-current drivers unless -Force is specified.
#>
[CmdletBinding(SupportsShouldProcess)]
param(
    [switch]$DownloadOnly,
    [switch]$SkipRealtek,
    [switch]$SkipCalDigit,
    [switch]$Force,
    [string]$DownloadDir = (Join-Path $env:TEMP 'UsbDriverUpdates')
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# ---------------------------------------------------------------------------
# Constants — defined at script scope so Summary section can always read them
# ---------------------------------------------------------------------------
$realtekTargetVer = '11.22.20.0206.2026'
$caldigitTargetFwVer = '45.1'

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

function Write-Section {
    param([string]$Title)
    $bar = '=' * 70
    Write-Host "`n$bar" -ForegroundColor Cyan
    Write-Host "  $Title" -ForegroundColor Cyan
    Write-Host "$bar" -ForegroundColor Cyan
}

function Write-Status {
    param([string]$Message, [string]$Color = 'Gray')
    Write-Host "  [*] $Message" -ForegroundColor $Color
}

function Write-OK {
    param([string]$Message)
    Write-Host "  [OK] $Message" -ForegroundColor Green
}

function Write-Skip {
    param([string]$Message)
    Write-Host "  [--] $Message" -ForegroundColor DarkGray
}

function Write-Warn {
    param([string]$Message)
    Write-Host "  [!!] $Message" -ForegroundColor Yellow
}

function Get-InstalledDriverVersion {
    param([string]$HardwareIdPattern)

    # Query all PnP devices (not just Net class) — the device may not yet have a
    # class-driver loaded and could appear under an Unknown or error state.
    $dev = Get-PnpDevice -ErrorAction SilentlyContinue |
        Where-Object { $_.InstanceId -like $HardwareIdPattern } |
        Select-Object -First 1

    if (-not $dev) { return $null }

    $prop = Get-PnpDeviceProperty -InstanceId $dev.InstanceId `
        -KeyName 'DEVPKEY_Device_DriverVersion' -ErrorAction SilentlyContinue
    if (-not $prop) { return $null }
    return $prop.Data
}

function Invoke-Download {
    param(
        [string]$Url,
        [string]$OutFile,
        [switch]$ForceDownload
    )

    if ((Test-Path $OutFile) -and -not $ForceDownload) {
        Write-Skip "Already downloaded: $(Split-Path $OutFile -Leaf)"
        return $OutFile
    }

    Write-Status "Downloading $(Split-Path $OutFile -Leaf) ..."
    $ProgressPreference = 'SilentlyContinue'
    try {
        Invoke-WebRequest -Uri $Url -OutFile $OutFile -UseBasicParsing
        $sizeKb = [math]::Round((Get-Item $OutFile).Length / 1KB, 1)
        Write-OK "Downloaded $sizeKb KB"
        return $OutFile
    }
    catch {
        Write-Warn "Download failed: $_"
        # Remove a partial file so a re-run does not skip the download
        if (Test-Path $OutFile) { Remove-Item -Force $OutFile -ErrorAction SilentlyContinue }
        return $null
    }
}

# ---------------------------------------------------------------------------
# Ensure download directory
# ---------------------------------------------------------------------------
if (-not (Test-Path $DownloadDir)) {
    New-Item -ItemType Directory -Path $DownloadDir -Force | Out-Null
    Write-Status "Created download directory: $DownloadDir"
}

# ===========================================================================
# SECTION 1: Realtek USB Ethernet Drivers (RTL8156 / RTL8157)
# ===========================================================================
if (-not $SkipRealtek) {
    Write-Section 'Realtek USB Ethernet Drivers (RTL8156 2.5GbE / RTL8157 5GbE)'

    $rtl8156Ver = Get-InstalledDriverVersion -HardwareIdPattern '*VID_0BDA&PID_8156*'
    $rtl8157Ver = Get-InstalledDriverVersion -HardwareIdPattern '*VID_0BDA&PID_8157*'

    $realtekUrl    = 'https://www.realtek.com/Download/List?cate_id=585'
    $realtekExeUrl = 'https://www.realtek.com/Download/ToLatestDownloadPage?downloadType=6&cate_id=585'
    $realtekFile   = Join-Path $DownloadDir 'Realtek-USB-Ethernet-Win11.exe'

    Write-Status "RTL8156 (2.5GbE) installed: $(if ($rtl8156Ver) { $rtl8156Ver } else { 'NOT FOUND' })"
    Write-Status "RTL8157 (5GbE)   installed: $(if ($rtl8157Ver) { $rtl8157Ver } else { 'NOT FOUND' })"
    Write-Status "Latest available:           $realtekTargetVer (WHQL, 2026-02-27)"

    # Determine whether an update is needed.
    # A device that is absent from this machine is skipped (N/A), but if at least
    # one is detected and its version differs from target, update is required.
    $needsUpdate = $Force.IsPresent
    if (-not $needsUpdate) {
        if ($rtl8156Ver -and $rtl8156Ver -ne $realtekTargetVer) { $needsUpdate = $true }
        if ($rtl8157Ver -and $rtl8157Ver -ne $realtekTargetVer) { $needsUpdate = $true }
    }

    if (-not $needsUpdate) {
        if (-not $rtl8156Ver -and -not $rtl8157Ver) {
            Write-Skip 'No Realtek USB ethernet adapter detected on this system.'
        }
        else {
            Write-OK 'Realtek USB ethernet drivers are already up to date.'
        }
    }
    else {
        Write-Status 'Update available. Downloading unified driver package ...' -Color Yellow

        # Realtek's redirect endpoint sometimes requires browser interaction.
        # Invoke-Download returns $null on failure and cleans up partial files.
        $downloaded = Invoke-Download -Url $realtekExeUrl -OutFile $realtekFile -ForceDownload:$Force

        if (-not $downloaded) {
            Write-Warn @"
Automatic download from Realtek was not possible (site may require a browser).
Please download the latest Windows 11 USB Ethernet driver manually:

  1. Visit: $realtekUrl
  2. Download the 'Win11 Auto Installation Program' (.exe)
  3. Save it as: $realtekFile

Then re-run this script.
"@
        }
        elseif ($DownloadOnly) {
            Write-OK "Driver package saved to: $realtekFile"
            Write-Status 'Skipping installation (-DownloadOnly mode).'
        }
        else {
            Write-Status 'Launching Realtek driver installer (follow the prompts) ...' -Color Yellow
            if ($PSCmdlet.ShouldProcess('Realtek USB Ethernet driver', 'Install')) {
                try {
                    $proc = Start-Process -FilePath $realtekFile -Wait -PassThru
                    if ($proc.ExitCode -eq 0) {
                        Write-OK 'Realtek driver installation completed successfully.'
                    }
                    else {
                        Write-Warn "Installer exited with code $($proc.ExitCode). Check for errors."
                    }
                }
                catch {
                    Write-Warn "Failed to launch installer: $_"
                }
            }

            $newVer8156 = Get-InstalledDriverVersion -HardwareIdPattern '*VID_0BDA&PID_8156*'
            $newVer8157 = Get-InstalledDriverVersion -HardwareIdPattern '*VID_0BDA&PID_8157*'
            if ($newVer8156) { Write-Status "RTL8156 now at: $newVer8156" -Color Green }
            if ($newVer8157) { Write-Status "RTL8157 now at: $newVer8157" -Color Green }
        }
    }
}
else {
    Write-Section 'Realtek USB Ethernet Drivers'
    Write-Skip 'Skipped (-SkipRealtek)'
}

# ===========================================================================
# SECTION 2: CalDigit Element Hub Thunderbolt Firmware
# ===========================================================================
if (-not $SkipCalDigit) {
    Write-Section 'CalDigit Element Hub — Thunderbolt Firmware Update'

    $caldigitDevices = Get-PnpDevice -ErrorAction SilentlyContinue |
        Where-Object { $_.FriendlyName -like '*CalDigit*Element*' }

    if (-not $caldigitDevices) {
        Write-Skip 'No CalDigit Element Hub detected. Skipping.'
    }
    else {
        $activeCount = @($caldigitDevices | Where-Object { $_.Status -eq 'OK' }).Count
        $totalCount  = @($caldigitDevices).Count
        Write-Status "CalDigit Element Hub(s) found: $totalCount ($activeCount currently connected)"

        $caldigitUrl = 'https://downloads.caldigit.com/CalDigit-TBT-Firmware-Updater-v7.3.zip'
        $caldigitZip = Join-Path $DownloadDir 'CalDigit-TBT-Firmware-Updater-v7.3.zip'
        $caldigitDir = Join-Path $DownloadDir 'CalDigit-Firmware-Updater'

        Write-Status "Latest firmware: v$caldigitTargetFwVer via Updater v7.3 (2026-01-26)"

        $downloaded = Invoke-Download -Url $caldigitUrl -OutFile $caldigitZip -ForceDownload:$Force

        if (-not $downloaded) {
            Write-Warn 'Failed to download CalDigit firmware updater.'
        }
        else {
            if (-not (Test-Path $caldigitDir) -or $Force) {
                if (Test-Path $caldigitDir) {
                    Remove-Item -Recurse -Force $caldigitDir
                }
                Write-Status 'Extracting firmware updater ...'
                Expand-Archive -Path $caldigitZip -DestinationPath $caldigitDir -Force
                Write-OK 'Extracted.'
            }
            else {
                Write-Skip 'Already extracted.'
            }

            if ($DownloadOnly) {
                Write-OK "Firmware updater saved to: $caldigitDir"
                Write-Status 'Skipping launch (-DownloadOnly mode).'
            }
            else {
                $updaterExe = Get-ChildItem -Path $caldigitDir -Recurse -Filter '*.exe' |
                    Select-Object -First 1

                if (-not $updaterExe) {
                    Write-Warn 'Could not find .exe in extracted archive. Contents:'
                    Get-ChildItem -Path $caldigitDir -Recurse |
                        ForEach-Object { Write-Status "  $($_.FullName)" }
                }
                else {
                    Write-Status "Found updater: $($updaterExe.Name)"
                    Write-Warn @"
IMPORTANT: The CalDigit firmware updater requires:
  - The Element Hub must be connected via Thunderbolt/USB4
  - Do NOT disconnect during the update
  - Follow all on-screen prompts
"@
                    if ($PSCmdlet.ShouldProcess('CalDigit Element Hub firmware', 'Update')) {
                        try {
                            Write-Status 'Launching CalDigit firmware updater ...' -Color Yellow
                            $proc = Start-Process -FilePath $updaterExe.FullName -Wait -PassThru
                            if ($proc.ExitCode -eq 0) {
                                Write-OK 'CalDigit firmware update completed.'
                            }
                            else {
                                Write-Warn "Updater exited with code $($proc.ExitCode)."
                            }
                        }
                        catch {
                            Write-Warn "Failed to launch updater: $_"
                        }
                    }
                }
            }
        }
    }
}
else {
    Write-Section 'CalDigit Element Hub Firmware'
    Write-Skip 'Skipped (-SkipCalDigit)'
}

# ===========================================================================
# SECTION 3: Summary
# ===========================================================================
Write-Section 'Summary'

$summary = @()

if (-not $SkipRealtek) {
    $finalRtl8156 = Get-InstalledDriverVersion -HardwareIdPattern '*VID_0BDA&PID_8156*'
    $finalRtl8157 = Get-InstalledDriverVersion -HardwareIdPattern '*VID_0BDA&PID_8157*'

    $drv8156 = if ($finalRtl8156) { $finalRtl8156 } else { 'Not present' }
    $sts8156 = if ($finalRtl8156 -eq $realtekTargetVer) { 'Current' }
               elseif (-not $finalRtl8156) { 'N/A' }
               else { 'Needs update' }
    $summary += [PSCustomObject]@{
        Device  = 'Realtek RTL8156 (2.5GbE)'
        HwId    = 'VID_0BDA&PID_8156'
        Driver  = $drv8156
        Target  = $realtekTargetVer
        Status  = $sts8156
    }

    $drv8157 = if ($finalRtl8157) { $finalRtl8157 } else { 'Not present' }
    $sts8157 = if ($finalRtl8157 -eq $realtekTargetVer) { 'Current' }
               elseif (-not $finalRtl8157) { 'N/A' }
               else { 'Needs update' }
    $summary += [PSCustomObject]@{
        Device  = 'Realtek RTL8157 (5GbE)'
        HwId    = 'VID_0BDA&PID_8157'
        Driver  = $drv8157
        Target  = $realtekTargetVer
        Status  = $sts8157
    }
}

if (-not $SkipCalDigit) {
    $calActive = @(Get-PnpDevice -ErrorAction SilentlyContinue |
        Where-Object { $_.FriendlyName -like '*CalDigit*Element*' -and $_.Status -eq 'OK' }).Count
    $summary += [PSCustomObject]@{
        Device  = 'CalDigit Element Hub (TB4/USB4)'
        HwId    = 'VID_8087&PID_0B26'
        Driver  = 'FW updater v7.3'
        Target  = "FW v$caldigitTargetFwVer"
        Status  = if ($calActive -gt 0) { 'Connected' } else { 'Disconnected' }
    }
}

$summary += [PSCustomObject]@{
    Device = 'Cable Matters 107064 (USB4)'
    HwId   = 'VID_8087&PID_5786'
    Driver = 'USB4 class driver'
    Target = 'N/A'
    Status = 'No update available'
}
$summary += [PSCustomObject]@{
    Device = 'ACASIS TBU405Pro (TB3)'
    HwId   = 'VID_8086&PID_15EF'
    Driver = 'Plug-and-play'
    Target = 'N/A'
    Status = 'No update available'
}

$summary | Format-Table -AutoSize -Wrap

Write-Host "`nDownloads stored in: $DownloadDir" -ForegroundColor DarkGray
Write-Host 'Re-run this script at any time to check for updates.' -ForegroundColor DarkGray
Write-Host ''
