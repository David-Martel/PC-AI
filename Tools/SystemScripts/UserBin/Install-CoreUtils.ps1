# Install-CoreUtils.ps1
# Installs uutils coreutils with uu- prefix to avoid conflicts with system utilities

param(
    [string]$CoreutilsPath = "T:\projects\coreutils\target\release\coreutils.exe",
    [string]$InstallPath = "$env:USERPROFILE\.local\bin",
    [switch]$Force
)

if (-not (Test-Path $CoreutilsPath)) {
    Write-Error "Coreutils executable not found at: $CoreutilsPath"
    Write-Host "Please build coreutils first using: cd T:\projects\coreutils && make release" -ForegroundColor Yellow
    exit 1
}

# Ensure install directory exists
if (-not (Test-Path $InstallPath)) {
    New-Item -ItemType Directory -Path $InstallPath -Force | Out-Null
    Write-Host "Created install directory: $InstallPath" -ForegroundColor Green
}

# Get list of all utilities
$utilities = & $CoreutilsPath --list 2>$null | Where-Object { $_ -and $_ -ne '[' -and $_ -ne ']' }

if (-not $utilities) {
    Write-Error "Failed to get utilities list from coreutils"
    exit 1
}

Write-Host "`nInstalling $($utilities.Count) utilities to $InstallPath with 'uu-' prefix..." -ForegroundColor Cyan

$installed = @()
$failed = @()

foreach ($utility in $utilities) {
    $linkPath = Join-Path $InstallPath "uu-$utility.exe"

    try {
        # Remove existing item if Force is specified
        if ($Force -and (Test-Path $linkPath)) {
            Remove-Item $linkPath -Force
        }

        # Create symlink
        New-Item -ItemType SymbolicLink -Path $linkPath -Target $CoreutilsPath -Force -ErrorAction Stop | Out-Null
        $installed += "uu-$utility"
        Write-Host "  ✓ uu-$utility" -ForegroundColor Green
    }
    catch {
        $failed += "uu-$utility"
        Write-Host "  ✗ uu-$utility - $($_.Exception.Message)" -ForegroundColor Red
    }
}

# Create wrapper script for multicall dispatching
$wrapperContent = @'
@echo off
setlocal
set "UTILITY_NAME=%~n0"
set "UTILITY_NAME=%UTILITY_NAME:uu-=%"
"T:\projects\coreutils\target\release\coreutils.exe" "%UTILITY_NAME%" %*
'@

$wrapperBatPath = Join-Path $InstallPath "uu-dispatch.bat"
Set-Content -Path $wrapperBatPath -Value $wrapperContent -Force

Write-Host "`n=== Installation Summary ===" -ForegroundColor Cyan
Write-Host "Successfully installed: $($installed.Count) utilities" -ForegroundColor Green
if ($failed.Count -gt 0) {
    Write-Host "Failed to install: $($failed.Count) utilities" -ForegroundColor Red
    Write-Host "Failed utilities: $($failed -join ', ')" -ForegroundColor Red
}

Write-Host "`n=== Usage ===" -ForegroundColor Cyan
Write-Host "Utilities are available with 'uu-' prefix. Examples:" -ForegroundColor White
Write-Host "  uu-ls       # uutils version of ls"
Write-Host "  uu-cat      # uutils version of cat"
Write-Host "  uu-echo     # uutils version of echo"

Write-Host "`n=== PATH Configuration ===" -ForegroundColor Cyan
if ($env:PATH -notmatch [regex]::Escape($InstallPath)) {
    Write-Host "WARNING: $InstallPath is not in your PATH" -ForegroundColor Yellow
    Write-Host "Add it to PATH to use utilities from any location" -ForegroundColor Yellow
} else {
    Write-Host "$InstallPath is in PATH ✓" -ForegroundColor Green
}

# Save installation manifest
$manifest = @{
    InstallDate = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    CoreutilsPath = $CoreutilsPath
    InstallPath = $InstallPath
    InstalledUtilities = $installed
    FailedUtilities = $failed
}

$manifestPath = Join-Path $InstallPath "uu-coreutils-manifest.json"
$manifest | ConvertTo-Json -Depth 3 | Set-Content -Path $manifestPath -Force
Write-Host "`nManifest saved to: $manifestPath" -ForegroundColor Gray