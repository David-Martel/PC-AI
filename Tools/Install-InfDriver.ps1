#Requires -RunAsAdministrator
<#
.SYNOPSIS
    Installs Windows drivers from INF files using pnputil, bypassing broken setup.exe wrappers.

.DESCRIPTION
    Many driver packages (Realtek, etc.) ship as WinZip SFX / 7z self-extracting archives
    containing INF+SYS+CAT files. Their setup.exe wrappers often hang or fail.

    This utility:
      1. Extracts the archive using 7-Zip (if needed)
      2. Locates INF files matching a filter pattern
      3. Installs them via pnputil /add-driver /install
      4. Verifies the installed driver versions

    It is idempotent - re-running skips already-installed drivers unless -Force is specified.

.PARAMETER ArchivePath
    Path to the self-extracting .exe or .7z archive containing driver files.
    If an 'extracted' subdirectory already exists (from a prior run), extraction is skipped.

.PARAMETER ExtractPath
    Override the extraction directory. Defaults to <ArchivePath-dir>\extracted.

.PARAMETER InfFilter
    Wildcard filter for INF files to install. Examples:
      'rtu56*'  - only RTL8156 driver
      'rtu5*'   - all Realtek USB ethernet INFs
      '*'       - all INFs in the package (default)

.PARAMETER Architecture
    Target architecture subfolder. Default: '64' (amd64).
    Use 'arm64' for ARM devices.

.PARAMETER VerifyHardwareId
    Optional hardware ID pattern (e.g., '*VID_0BDA&PID_8156*') to query
    post-install and display the updated driver version.

.PARAMETER SevenZipPath
    Path to 7z.exe. Auto-detected from Program Files if not specified.

.PARAMETER Force
    Re-extract and reinstall even if already present.

.EXAMPLE
    # Install all Realtek USB ethernet drivers from extracted package
    .\Install-InfDriver.ps1 `
        -ArchivePath 'C:\Users\david\Downloads\Install_USB_Win11_11021_20_11102025_01302026\Install_USB_Win11_11021_20_11102025_01302026.exe' `
        -InfFilter 'rtu5*' `
        -VerifyHardwareId '*VID_0BDA&PID_8156*','*VID_0BDA&PID_8157*'

.EXAMPLE
    # Install a specific INF from an already-extracted directory
    .\Install-InfDriver.ps1 `
        -ExtractPath 'C:\Drivers\Realtek\WIN11\cx\64' `
        -InfFilter 'rtu56cx22x64sta'

.NOTES
    Created: 2026-03-11
    Origin:  Extracted from pnputil install approach that bypasses Realtek's broken setup.exe
    Requires: Administrator privileges, 7-Zip (for archive extraction)
#>
[CmdletBinding(SupportsShouldProcess)]
param(
    [Parameter(Mandatory = $false)]
    [string]$ArchivePath,

    [Parameter(Mandatory = $false)]
    [string]$ExtractPath,

    [string]$InfFilter = '*',

    [string]$Architecture = '64',

    [string[]]$VerifyHardwareId,

    [string]$SevenZipPath,

    [switch]$Force
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
function Write-Status {
    param([string]$Message, [string]$Color = 'Gray')
    Write-Host "  [*] $Message" -ForegroundColor $Color
}
function Write-OK {
    param([string]$Message)
    Write-Host "  [OK] $Message" -ForegroundColor Green
}
function Write-Warn {
    param([string]$Message)
    Write-Host "  [!!] $Message" -ForegroundColor Yellow
}

# ---------------------------------------------------------------------------
# Resolve 7-Zip
# ---------------------------------------------------------------------------
function Find-SevenZip {
    param([string]$Hint)
    if ($Hint -and (Test-Path $Hint)) { return $Hint }

    $candidates = @(
        "$env:ProgramFiles\7-Zip\7z.exe",
        "${env:ProgramFiles(x86)}\7-Zip\7z.exe"
    )
    foreach ($c in $candidates) {
        if (Test-Path $c) { return $c }
    }
    return $null
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
Write-Host ''
Write-Host '======================================================================' -ForegroundColor Cyan
Write-Host '  Install-InfDriver - pnputil-based INF driver installer'             -ForegroundColor Cyan
Write-Host '======================================================================'  -ForegroundColor Cyan
Write-Host ''

# Determine extraction path
if (-not $ExtractPath -and -not $ArchivePath) {
    Write-Error 'You must specify either -ArchivePath or -ExtractPath.'
}

if (-not $ExtractPath) {
    $ExtractPath = Join-Path (Split-Path $ArchivePath -Parent) 'extracted'
}

# Step 1: Extract if needed
if ($ArchivePath -and (-not (Test-Path $ExtractPath) -or $Force)) {
    if (-not (Test-Path $ArchivePath)) {
        Write-Error "Archive not found: $ArchivePath"
    }

    $szPath = Find-SevenZip -Hint $SevenZipPath
    if (-not $szPath) {
        Write-Error '7-Zip not found. Install it or specify -SevenZipPath.'
    }

    Write-Status 'Extracting archive with 7-Zip ...'
    if ($Force -and (Test-Path $ExtractPath)) {
        Remove-Item -Recurse -Force $ExtractPath
    }

    $szArgs = @('x', "`"$ArchivePath`"", "-o`"$ExtractPath`"", '-y')
    $proc = Start-Process -FilePath $szPath -ArgumentList $szArgs -Wait -PassThru -NoNewWindow
    if ($proc.ExitCode -ne 0 -and $proc.ExitCode -ne 1) {
        Write-Error "7-Zip extraction failed with exit code $($proc.ExitCode)"
    }
    Write-OK 'Extraction complete.'
}
elseif (Test-Path $ExtractPath) {
    Write-Status "Using existing extraction: $ExtractPath"
}
else {
    Write-Error "Extraction path does not exist: $ExtractPath"
}

# Step 2: Find INF files
# Search recursively - driver packages nest INFs under arch-specific subdirs
$infSearch = Get-ChildItem -Path $ExtractPath -Recurse -Filter "$InfFilter.INF" -ErrorAction SilentlyContinue
if (-not $infSearch) {
    # Try lowercase extension
    $infSearch = Get-ChildItem -Path $ExtractPath -Recurse -Filter "$InfFilter.inf" -ErrorAction SilentlyContinue
}

if (-not $infSearch) {
    Write-Warn "No INF files matching filter found in $ExtractPath"
    Write-Status 'Listing available INF files:'
    Get-ChildItem -Path $ExtractPath -Recurse -Include '*.inf','*.INF' | ForEach-Object {
        Write-Status "  $($_.FullName)"
    }
    return
}

# Filter to target architecture if there are arch subdirs
$archInfs = @($infSearch | Where-Object {
    $_.DirectoryName -like "*\$Architecture*" -or $_.DirectoryName -like "*\$Architecture"
})
if ($archInfs.Count -gt 0) {
    $infFiles = $archInfs
    Write-Status "Found $($infFiles.Count) INF file(s) for architecture '$Architecture'"
}
else {
    $infFiles = @($infSearch)
    Write-Status "Found $($infFiles.Count) INF file(s) (no architecture filter applied)"
}

foreach ($inf in $infFiles) {
    Write-Status "  $($inf.Name) in $($inf.DirectoryName)"
}

# Step 3: Install via pnputil
Write-Host ''
$installed = 0
$failed = 0
foreach ($inf in $infFiles) {
    Write-Status "Installing: $($inf.Name) ..." -Color Yellow
    if ($PSCmdlet.ShouldProcess($inf.FullName, 'pnputil /add-driver /install')) {
        $output = & pnputil /add-driver "$($inf.FullName)" /install 2>&1
        $exitCode = $LASTEXITCODE
        foreach ($line in $output) {
            Write-Host "    $line"
        }
        if ($exitCode -eq 0 -or $exitCode -eq 3010) {
            Write-OK "$($inf.Name) installed successfully."
            $installed++
            if ($exitCode -eq 3010) {
                Write-Warn 'Reboot required to complete driver installation.'
            }
        }
        else {
            Write-Warn "$($inf.Name) failed with exit code $exitCode"
            $failed++
        }
    }
}

Write-Host ''
$resultColor = 'Green'
if ($failed -gt 0) { $resultColor = 'Yellow' }
Write-Status "Results: $installed installed, $failed failed" -Color $resultColor

# Step 4: Verify
if ($VerifyHardwareId -and $VerifyHardwareId.Count -gt 0) {
    Write-Host ''
    Write-Status 'Verifying installed driver versions ...'
    foreach ($pattern in $VerifyHardwareId) {
        $dev = Get-PnpDevice -ErrorAction SilentlyContinue |
            Where-Object { $_.InstanceId -like $pattern -and $_.Status -eq 'OK' } |
            Select-Object -First 1
        if ($dev) {
            $ver = (Get-PnpDeviceProperty -InstanceId $dev.InstanceId `
                -KeyName 'DEVPKEY_Device_DriverVersion' -ErrorAction SilentlyContinue).Data
            Write-OK "$($dev.FriendlyName): $ver"
        }
        else {
            Write-Status "No active device matching pattern" -Color DarkGray
        }
    }
}

Write-Host ''
Write-Host 'Done.' -ForegroundColor Green
Write-Host ''
