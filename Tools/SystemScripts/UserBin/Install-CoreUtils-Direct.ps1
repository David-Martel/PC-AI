# Install-CoreUtils-Direct.ps1
# Installs uutils coreutils as direct replacements for system utilities

param(
    [string]$CoreutilsPath = "T:\projects\coreutils\target\release\coreutils.exe",
    [switch]$Force,
    [switch]$BackupExisting
)

if (-not (Test-Path $CoreutilsPath)) {
    Write-Error "Coreutils executable not found at: $CoreutilsPath"
    Write-Host "Please build coreutils first using: cd T:\projects\coreutils && make release" -ForegroundColor Yellow
    exit 1
}

# Install to both directories
$installPaths = @(
    "$env:USERPROFILE\bin",
    "$env:USERPROFILE\.local\bin"
)

# Get list of all utilities
$utilities = & $CoreutilsPath --list 2>$null | Where-Object { $_ -and $_ -ne '[' -and $_ -ne ']' }

if (-not $utilities) {
    Write-Error "Failed to get utilities list from coreutils"
    exit 1
}

Write-Host "`nInstalling $($utilities.Count) coreutils utilities as direct replacements..." -ForegroundColor Cyan

$installedCount = 0
$backupCount = 0

foreach ($installPath in $installPaths) {
    # Ensure install directory exists
    if (-not (Test-Path $installPath)) {
        New-Item -ItemType Directory -Path $installPath -Force | Out-Null
        Write-Host "Created directory: $installPath" -ForegroundColor Green
    }

    Write-Host "`nInstalling to: $installPath" -ForegroundColor Yellow

    # First, clean up old uu-* prefixed versions
    Write-Host "  Removing old uu-* prefixed versions..." -ForegroundColor DarkGray
    Get-ChildItem "$installPath\uu-*.exe" -ErrorAction SilentlyContinue | Remove-Item -Force
    Get-ChildItem "$installPath\wu-*.exe" -ErrorAction SilentlyContinue | Remove-Item -Force

    foreach ($utility in $utilities) {
        $targetPath = Join-Path $installPath "$utility.exe"

        try {
            # Backup existing if requested
            if ($BackupExisting -and (Test-Path $targetPath) -and (-not (Get-Item $targetPath).LinkType)) {
                $backupPath = "$targetPath.backup"
                Move-Item $targetPath $backupPath -Force
                $backupCount++
                Write-Host "    Backed up: $utility.exe -> $utility.exe.backup" -ForegroundColor DarkGray
            }

            # Remove existing item if Force is specified
            if ($Force -and (Test-Path $targetPath)) {
                Remove-Item $targetPath -Force
            }

            # Create symlink to coreutils multicall binary
            New-Item -ItemType SymbolicLink -Path $targetPath -Target $CoreutilsPath -Force -ErrorAction Stop | Out-Null
            Write-Host "    ✓ $utility" -ForegroundColor Green
            $installedCount++
        }
        catch {
            Write-Host "    ✗ $utility - $($_.Exception.Message)" -ForegroundColor Red
        }
    }
}

# Create batch wrapper for CMD compatibility
$wrapperContent = @'
@echo off
setlocal
set "UTILITY_NAME=%~n0"
"T:\projects\coreutils\target\release\coreutils.exe" "%UTILITY_NAME%" %*
'@

foreach ($installPath in $installPaths) {
    $wrapperPath = Join-Path $installPath "coreutils-dispatch.bat"
    Set-Content -Path $wrapperPath -Value $wrapperContent -Force
}

Write-Host "`n=== Installation Summary ===" -ForegroundColor Cyan
Write-Host "Successfully installed: $installedCount utilities across both directories" -ForegroundColor Green
if ($backupCount -gt 0) {
    Write-Host "Backed up: $backupCount existing utilities" -ForegroundColor Yellow
}

Write-Host "`n=== Testing Installation ===" -ForegroundColor Cyan
# Test a few utilities
$testUtils = @('ls', 'cat', 'echo', 'pwd')
foreach ($test in $testUtils) {
    $result = & $test --version 2>&1 | Select-Object -First 1
    if ($result -match 'uutils') {
        Write-Host "  ✓ $test is using coreutils version" -ForegroundColor Green
    } else {
        Write-Host "  ⚠ $test might not be using coreutils version" -ForegroundColor Yellow
    }
}

Write-Host "`n=== PATH Priority ===" -ForegroundColor Cyan
$pathDirs = $env:PATH -split ';'
$binIndex = -1
$localBinIndex = -1
$systemIndex = -1

for ($i = 0; $i -lt $pathDirs.Count; $i++) {
    if ($pathDirs[$i] -match [regex]::Escape("$env:USERPROFILE\bin")) {
        $binIndex = $i
    }
    if ($pathDirs[$i] -match [regex]::Escape("$env:USERPROFILE\.local\bin")) {
        $localBinIndex = $i
    }
    if ($pathDirs[$i] -match 'System32') {
        $systemIndex = $i
        break  # Stop at first System32
    }
}

Write-Host "PATH order:" -ForegroundColor White
if ($binIndex -ge 0) { Write-Host "  Position $($binIndex): ~/bin" -ForegroundColor Green }
if ($localBinIndex -ge 0) { Write-Host "  Position $($localBinIndex): ~/.local/bin" -ForegroundColor Green }
if ($systemIndex -ge 0) { Write-Host "  Position $($systemIndex): System32" -ForegroundColor Gray }

if ($binIndex -ge 0 -and $binIndex -lt $systemIndex) {
    Write-Host "`n✓ ~/bin comes before System32 - coreutils will be used by default" -ForegroundColor Green
} elseif ($localBinIndex -ge 0 -and $localBinIndex -lt $systemIndex) {
    Write-Host "`n✓ ~/.local/bin comes before System32 - coreutils will be used by default" -ForegroundColor Green
} else {
    Write-Host "`n⚠ User bin directories come after System32 - system utilities will be used by default" -ForegroundColor Yellow
    Write-Host "  Consider reordering PATH to prioritize user directories" -ForegroundColor Yellow
}

# Save installation manifest
$manifest = @{
    InstallDate = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    CoreutilsPath = $CoreutilsPath
    InstallPaths = $installPaths
    InstalledUtilities = $utilities
    BackupCreated = $BackupExisting
    DirectReplacement = $true
}

foreach ($installPath in $installPaths) {
    $manifestPath = Join-Path $installPath "coreutils-manifest.json"
    $manifest | ConvertTo-Json -Depth 3 | Set-Content -Path $manifestPath -Force
}

Write-Host "`nManifests saved to both directories" -ForegroundColor Gray
Write-Host "`nInstallation complete! Coreutils are now your default utilities." -ForegroundColor Green
Write-Host "Note: You may need to restart your terminal for PATH changes to take effect." -ForegroundColor Yellow