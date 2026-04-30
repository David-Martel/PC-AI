# Fix GCP Profile System Issues
# This script addresses all the identified problems with the GCP profile system

Write-Host "Fixing GCP Profile System Issues..." -ForegroundColor Yellow

# 1. Remove the corrupted GcpProfileManager.ps1 from OneDrive location
$corruptedFile = "C:\Users\david\OneDrive\Documents\PowerShell\Modules\GcpUtils\GcpProfileManager.ps1"
if (Test-Path $corruptedFile) {
    try {
        Remove-Item $corruptedFile -Force
        Write-Host "[SUCCESS] Removed corrupted GcpProfileManager.ps1 from OneDrive location" -ForegroundColor Green
    }
    catch {
        Write-Host "[ERROR] Failed to remove corrupted file: $_" -ForegroundColor Red
    }
}

# 2. Ensure correct PowerShell module path priority
$psModulePaths = $env:PSModulePath -split ';'
$localPath = "C:\Users\david\Documents\PowerShell\Modules"
$oneDrivePath = "C:\Users\david\OneDrive\Documents\PowerShell\Modules"

# Make sure local path comes before OneDrive path
if ($localPath -notin $psModulePaths) {
    $env:PSModulePath = "$localPath;$env:PSModulePath"
    Write-Host "[SUCCESS] Added local PowerShell modules path" -ForegroundColor Green
}

# 3. Test paths that should exist
$requiredPaths = @(
    "$env:USERPROFILE\.auth",
    "$env:USERPROFILE\.auth\business",
    "$env:USERPROFILE\.auth\personal",
    "$env:USERPROFILE\.gcp",
    "$env:USERPROFILE\.gcp\business.env",
    "$env:USERPROFILE\.gcp\personal.env",
    "$env:USERPROFILE\.gcp\current-profile.txt"
)

Write-Host "`nChecking required paths..." -ForegroundColor Cyan
foreach ($path in $requiredPaths) {
    if (Test-Path $path) {
        Write-Host "[OK] $path" -ForegroundColor Green
    } else {
        Write-Host "[MISSING] $path" -ForegroundColor Red
    }
}

# 4. Try to reload the GcpUtils module cleanly
Write-Host "`nAttempting to reload GcpUtils module..." -ForegroundColor Cyan
try {
    # Remove any loaded module instances
    Get-Module GcpUtils | Remove-Module -Force -ErrorAction SilentlyContinue

    # Import from the correct location
    Import-Module "C:\Users\david\Documents\PowerShell\Modules\GcpUtils\GcpUtils.psm1" -Force
    Write-Host "[SUCCESS] GcpUtils module reloaded successfully" -ForegroundColor Green

    # Test basic functionality
    $currentProfile = Get-GcpProfile
    if ($currentProfile) {
        Write-Host "[SUCCESS] Profile system working - Current: $($currentProfile.Name)" -ForegroundColor Green
    } else {
        Write-Host "[WARNING] Profile system loaded but no active profile" -ForegroundColor Yellow
    }
}
catch {
    Write-Host "[ERROR] Failed to reload GcpUtils: $_" -ForegroundColor Red
}

Write-Host "`nGCP Profile system fix complete!" -ForegroundColor Green
Write-Host "You may need to restart PowerShell for all changes to take effect." -ForegroundColor Cyan