#Requires -Version 5.1
<#
.SYNOPSIS
    Scans all PnP devices and reports/updates drivers from trusted sources.

.DESCRIPTION
    Phase 1 (scan): No admin required. Discovers devices, compares against the curated
    driver registry, reports status as a formatted table with summary counts.

    Phase 2 (install): Requires admin. Downloads and installs updates with user approval.
    Devices that share a driver package (sharedDriverGroup) are processed once - only the
    first representative in each group triggers a download and install.

    Scan, preview and download modes use the bundled registry unless overridden.
    Installation scans and groups devices using the protected machine registry
    it passes to Install-DriverUpdate.

.PARAMETER ScanOnly
    Run Phase 1 (report) only. No downloads or installs are attempted.

.PARAMETER Category
    Limit scanning and installation to a single registry category (e.g. 'network', 'hub').

.PARAMETER DownloadOnly
    Download installer files without executing them. Admin rights are not required.

.PARAMETER Force
    Re-download installer packages even if the file already exists on disk.

.PARAMETER DownloadDir
    Directory for downloaded driver packages. Defaults to $env:TEMP\DriverUpdates.

.PARAMETER RegistryPath
    Optional registry for scans, previews and downloads. Installation uses the
    protected PC-AI.DriverTrust registry in Program Files.

.EXAMPLE
    .\Update-Drivers.ps1
    Full scan + install (requires admin).

.EXAMPLE
    .\Update-Drivers.ps1 -ScanOnly
    Report only - no changes made.

.EXAMPLE
    .\Update-Drivers.ps1 -Category network -DownloadOnly
    Download Realtek/network packages without installing.

.EXAMPLE
    .\Update-Drivers.ps1 -WhatIf
    Show what would be installed without making any changes.

.NOTES
    Successor to: Tools\Update-UsbDrivers.ps1
    Requires:     PC-AI.Drivers module (auto-resolved from script location)
    Downloads can reuse verified cached files. Installs use a fresh protected stage.
#>
[CmdletBinding(SupportsShouldProcess)]
param(
    [switch]$ScanOnly,
    [string]$Category,
    [switch]$DownloadOnly,
    [switch]$Force,
    [string]$DownloadDir = (Join-Path $env:TEMP 'DriverUpdates'),
    [string]$RegistryPath
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# ---------------------------------------------------------------------------
# Helper functions (same pattern as Update-UsbDrivers.ps1)
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

# ---------------------------------------------------------------------------
# Import PC-AI.Drivers module (path relative to this script's location)
# ---------------------------------------------------------------------------

$scriptDir  = $PSScriptRoot
$moduleRoot = Join-Path $scriptDir '..\Modules\PC-AI.Drivers'

if (-not (Test-Path (Join-Path $moduleRoot 'PC-AI.Drivers.psd1'))) {
    Write-Error "PC-AI.Drivers module not found at: $moduleRoot"
    exit 1
}

Import-Module $moduleRoot -Force -ErrorAction Stop

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

Write-Section 'PC-AI Driver Update Utility'
Write-Status "Date      : $(Get-Date -Format 'yyyy-MM-dd HH:mm')"
if ($Category)    { Write-Status "Category  : $Category" }
if ($ScanOnly)    { Write-Status "Mode      : Scan only (no installs)" -Color Yellow }
elseif ($DownloadOnly) { Write-Status "Mode      : Download only (no installs)" -Color Yellow }
else              { Write-Status "Mode      : Scan + install" }

# ---------------------------------------------------------------------------
# Phase 1: Scan
# ---------------------------------------------------------------------------

Write-Section 'Phase 1 - Device Scan'
Write-Status 'Enumerating PnP devices and comparing against registry ...'

$regLookupParams = @{}
if ($RegistryPath) { $regLookupParams['RegistryPath'] = $RegistryPath }

# Resolve installation trust before scanning so the offered version and package
# grouping come from the same registry passed to the installer.
$reg = $null
if (-not ($ScanOnly -or $DownloadOnly -or $WhatIfPreference)) {
    $reg = Get-DriverRegistry @regLookupParams -ForInstall
    $RegistryPath = $reg.RegistryPath
}

$reportParams = @{}
if ($RegistryPath) { $reportParams['RegistryPath'] = $RegistryPath }
if ($Category)     { $reportParams['Category']     = $Category     }

# Suppress WhatIf/Confirm during scan -- this is a read-only operation.
# CIM cmdlets like Get-PnpDeviceProperty inherit ShouldProcess from the caller.
$savedWhatIf = $WhatIfPreference
$savedConfirm = $ConfirmPreference
$WhatIfPreference = $false
$ConfirmPreference = 'None'
try {
    $fullReport = @(Get-DriverReport @reportParams)
} finally {
    $WhatIfPreference = $savedWhatIf
    $ConfirmPreference = $savedConfirm
}

$countTotal      = $fullReport.Count
$countCurrent    = @($fullReport | Where-Object { $_.Status -eq 'Current' }).Count
$countOutdated   = @($fullReport | Where-Object { $_.Status -eq 'Outdated' }).Count
$countNoDriver   = @($fullReport | Where-Object { $_.Status -eq 'NoDriver' }).Count
$countNoUpdate   = @($fullReport | Where-Object { $_.Status -eq 'NoUpdate' }).Count
$countManual     = @($fullReport | Where-Object { $_.Status -eq 'ManualCheck' }).Count

# Display the report table (suppress Current/NoUpdate in the default view to keep it readable)
$displayReport = @($fullReport | Where-Object { $_.Status -ne 'Current' -and $_.Status -ne 'NoUpdate' })

if ($displayReport.Count -gt 0) {
    Write-Host ''
    $displayReport |
        Select-Object Status, Category, DeviceName, InstalledVersion, TargetVersion |
        Format-Table -AutoSize -Wrap
}
else {
    Write-Host ''
    Write-OK 'All registered drivers are current.'
    Write-Host ''
}

# Summary counts
Write-Host ''
Write-Host ('  Devices checked : ' + $countTotal)   -ForegroundColor White
Write-Host ('  Current         : ' + $countCurrent)  -ForegroundColor Green
$outdatedColor = if ($countOutdated -gt 0) { 'Yellow' } else { 'Green' }
Write-Host ('  Outdated        : ' + $countOutdated) -ForegroundColor $outdatedColor
$noDriverColor = if ($countNoDriver -gt 0) { 'Yellow' } else { 'Green' }
Write-Host ('  No driver       : ' + $countNoDriver) -ForegroundColor $noDriverColor
Write-Host ('  No update avail : ' + $countNoUpdate) -ForegroundColor DarkGray
$manualColor = if ($countManual -gt 0) { 'Magenta' } else { 'Green' }
Write-Host ('  Manual check    : ' + $countManual) -ForegroundColor $manualColor
Write-Host ''

# ---------------------------------------------------------------------------
# ScanOnly exit
# ---------------------------------------------------------------------------

if ($ScanOnly) {
    Write-Skip 'Scan complete (-ScanOnly). No changes made.'
    Write-Host ''
    exit 0
}

# ---------------------------------------------------------------------------
# Phase 2: Install
# ---------------------------------------------------------------------------

Write-Section 'Phase 2 - Driver Installation'

# Actionable = Outdated, NoDriver, or ManualCheck (NoUpdate/Current/Unknown are not installable)
$actionable = @($fullReport | Where-Object { $_.Status -eq 'Outdated' -or $_.Status -eq 'NoDriver' -or $_.Status -eq 'ManualCheck' })

if ($actionable.Count -eq 0) {
    Write-Skip 'All registered drivers are current. Nothing to install.'
    Write-Host ''
    exit 0
}

# Process items, respecting sharedDriverGroup - download/install a shared package only once.
$processedGroups = [System.Collections.Generic.HashSet[string]]::new(
    [System.StringComparer]::OrdinalIgnoreCase
)

$installResults = @()

if (-not $reg) {
    $reg = Get-DriverRegistry @regLookupParams
}

foreach ($item in $actionable) {
    $deviceId    = $item.RegistryId
    $deviceName  = $item.DeviceName
    $groupKey    = $null

    # Look up sharedDriverGroup and installerType from the pre-loaded registry
    $regEntry = $null
    if ($reg) {
        $regEntry = $reg.Devices | Where-Object { $_.id -eq $deviceId } | Select-Object -First 1
        if ($regEntry -and $regEntry.sharedDriverGroup) {
            $groupKey = $regEntry.sharedDriverGroup
        }
    }

    # Skip if this shared group was already handled
    if ($groupKey -and $processedGroups.Contains($groupKey)) {
        Write-Skip "  $deviceName - package '$groupKey' already processed for this group."
        $installResults += [PSCustomObject]@{
            DeviceId   = $deviceId
            DeviceName = $deviceName
            Action     = 'SharedGroup-Skipped'
            Success    = $true
            Message    = "Shared package '$groupKey' already installed for this group."
        }
        continue
    }

    # Handle manual-download devices (open browser instead of auto-download)
    if ($regEntry -and $regEntry.driver.installerType -eq 'manual') {
        $manualUrl = $regEntry.driver.manualDownloadUrl
        if ($manualUrl) {
            Write-Warn "  $deviceName requires manual download."
            Write-Status "  Download address: $manualUrl" -Color Magenta
        } else {
            Write-Warn "  $deviceName requires manual update (no URL available)."
        }
        $installResults += [PSCustomObject]@{
            DeviceId   = $deviceId
            DeviceName = $deviceName
            Action     = 'ManualDownload'
            Success    = $true
            Message    = "Manual download address: $manualUrl"
        }
        if ($groupKey) { $processedGroups.Add($groupKey) | Out-Null }
        continue
    }

    # Display what we are about to do
    $targetVer = if ($item.TargetVersion) { $item.TargetVersion } else { 'unknown' }
    Write-Status "Updating '$deviceName'  $($item.InstalledVersion) --> $targetVer" -Color Yellow

    $installParams = @{
        DeviceId    = $deviceId
        DownloadDir = $DownloadDir
        Force       = $Force.IsPresent
        DownloadOnly = $DownloadOnly.IsPresent
    }
    if ($RegistryPath) { $installParams['RegistryPath'] = $RegistryPath }

    $result = Install-DriverUpdate @installParams

    if ($result) {
        $installResults += $result

        # Mark group as processed after a successful (or download-only) run
        if ($groupKey) {
            if ($result.Success) {
                $processedGroups.Add($groupKey) | Out-Null
            }
        }
    }
}

# ---------------------------------------------------------------------------
# Final summary table
# ---------------------------------------------------------------------------

Write-Section 'Summary'

if ($installResults.Count -gt 0) {
    $installResults |
        Select-Object DeviceName, Action, Success, Message |
        Format-Table -AutoSize -Wrap
}

$successCount = @($installResults | Where-Object { $_.Success -eq $true  }).Count
$failCount    = @($installResults | Where-Object { $_.Success -eq $false }).Count

Write-Host ''
if ($failCount -eq 0) {
    Write-OK "All operations completed successfully ($successCount item(s))."
}
else {
    Write-Warn "$failCount operation(s) failed, $successCount succeeded."
    Write-Warn 'Review the messages above and re-run for failed items.'
}

Write-Host ''
if ($DownloadOnly) {
    Write-Host "  Downloads stored in: $DownloadDir" -ForegroundColor DarkGray
}
Write-Host '  Re-run this script at any time to check for updates.' -ForegroundColor DarkGray
Write-Host ''
