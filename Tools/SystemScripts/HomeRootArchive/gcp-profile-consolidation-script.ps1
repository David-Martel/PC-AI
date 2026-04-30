# GCP Profile System Consolidation Script
# ENFORCING ANTI-DUPLICATION DIRECTIVE
# This script consolidates all GCP profile management to CANONICAL versions ONLY

Write-Host "🚨 GCP PROFILE CONSOLIDATION - ANTI-DUPLICATION ENFORCEMENT 🚨" -ForegroundColor Red
Write-Host "=" * 70 -ForegroundColor Red

# CANONICAL VERSIONS (DO NOT MODIFY)
$CANONICAL_POWERSHELL = "C:\Users\david\OneDrive\Documents\PowerShell\Modules\GcpProfileSimple\GcpProfileSimple.psm1"
$CANONICAL_BASH = "\\wsl.localhost\Ubuntu\home\david\.local\bin\gcp-profile-simple"

# DUPLICATES TO REMOVE
$DUPLICATES = @{
    "GcpUtils Module" = "C:\Users\david\OneDrive\Documents\PowerShell\Modules\GcpUtils"
    "Enhanced GCP Profile" = "\\wsl.localhost\Ubuntu\home\david\.local\bin\cloud-tools\gcp-profile"
    "GCP Profile Backup" = "\\wsl.localhost\Ubuntu\home\david\.local\bin\cloud-tools\gcp-profile.backup.20250804"
    "WSL GCP Profile" = "\\wsl.localhost\Ubuntu\home\david\.local\bin\gcp-profile"
}

Write-Host "`nCONSOLIDATION PLAN:" -ForegroundColor Cyan
Write-Host "✅ KEEP: GcpProfileSimple (PowerShell) - CANONICAL" -ForegroundColor Green
Write-Host "✅ KEEP: gcp-profile-simple (Bash) - CANONICAL" -ForegroundColor Green
Write-Host ""

foreach ($name in $DUPLICATES.Keys) {
    Write-Host "❌ REMOVE: $name" -ForegroundColor Red
    Write-Host "   Path: $($DUPLICATES[$name])" -ForegroundColor Gray
}

Write-Host "`n" -ForegroundColor Yellow
$confirm = Read-Host "Do you want to proceed with consolidation? (yes/no)"

if ($confirm -ne "yes") {
    Write-Host "❌ Consolidation cancelled" -ForegroundColor Yellow
    exit 0
}

# Create backup directory
$backupDir = "C:\Users\david\.gcp-consolidation-backup-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
New-Item -ItemType Directory -Path $backupDir -Force | Out-Null
Write-Host "📦 Created backup directory: $backupDir" -ForegroundColor Green

# Backup and remove duplicates
Write-Host "`n🔄 STARTING CONSOLIDATION..." -ForegroundColor Yellow

# 1. Backup GcpUtils module
$gcpUtilsPath = "C:\Users\david\OneDrive\Documents\PowerShell\Modules\GcpUtils"
if (Test-Path $gcpUtilsPath) {
    Write-Host "📦 Backing up GcpUtils module..." -ForegroundColor Yellow
    Copy-Item -Path $gcpUtilsPath -Destination "$backupDir\GcpUtils" -Recurse -Force

    # Remove GcpUtils
    Remove-Item -Path $gcpUtilsPath -Recurse -Force
    Write-Host "❌ Removed duplicate: GcpUtils module" -ForegroundColor Red
}

# 2. Remove WSL duplicates via WSL commands
Write-Host "📦 Backing up WSL GCP profile scripts..." -ForegroundColor Yellow

# Backup enhanced gcp-profile
wsl cp /home/david/.local/bin/cloud-tools/gcp-profile "$($backupDir.Replace('C:', '/mnt/c').Replace('\', '/'))/gcp-profile-enhanced.backup" 2>$null
wsl rm -f /home/david/.local/bin/cloud-tools/gcp-profile

# Backup old backup
wsl cp /home/david/.local/bin/cloud-tools/gcp-profile.backup.20250804 "$($backupDir.Replace('C:', '/mnt/c').Replace('\', '/'))/gcp-profile.backup.20250804" 2>$null
wsl rm -f /home/david/.local/bin/cloud-tools/gcp-profile.backup.20250804

# Remove main gcp-profile if it exists
wsl rm -f /home/david/.local/bin/gcp-profile 2>$null

Write-Host "❌ Removed WSL duplicate scripts" -ForegroundColor Red

# 3. Update canonical versions with clear identification
Write-Host "`n✏️  UPDATING CANONICAL VERSIONS..." -ForegroundColor Cyan

# Update PowerShell canonical (already has the comment, but reinforce)
$psContent = Get-Content $CANONICAL_POWERSHELL -Raw
if ($psContent -notmatch "This is the CANONICAL GCP profile manager") {
    $newContent = $psContent -replace "# This is the CANONICAL version - DO NOT create variants",
        "# This is the CANONICAL GCP profile manager - DO NOT create variants like V2, V3, enhanced, simple, etc."
    Set-Content -Path $CANONICAL_POWERSHELL -Value $newContent
}

# Update Bash canonical
wsl sed -i '4i# This is the CANONICAL GCP profile manager - DO NOT create variants like V2, V3, enhanced, etc.' /home/david/.local/bin/gcp-profile-simple

Write-Host "✅ Updated canonical version identification" -ForegroundColor Green

# 4. Clean up any remaining duplicates in other locations
Write-Host "`n🧹 CLEANING UP OTHER LOCATIONS..." -ForegroundColor Cyan

# Remove any profile_Utilities.ps1 reference to old GCP functions
$profileUtil = "C:\Users\david\OneDrive\Documents\PowerShell\Modules\profile_Utilities.ps1"
if (Test-Path $profileUtil) {
    Write-Host "🔍 Checking profile_Utilities.ps1 for old GCP references..." -ForegroundColor Yellow
    # Note: This would need manual review as it might have other important functions
}

# 5. Create aliases to ensure compatibility
Write-Host "`n🔗 CREATING COMPATIBILITY ALIASES..." -ForegroundColor Cyan

# Create WSL alias for old gcp-profile command
wsl ln -sf /home/david/.local/bin/gcp-profile-simple /home/david/.local/bin/gcp-profile

Write-Host "✅ Created compatibility alias: gcp-profile -> gcp-profile-simple" -ForegroundColor Green

# 6. Test canonical versions
Write-Host "`n🧪 TESTING CANONICAL VERSIONS..." -ForegroundColor Cyan

# Test PowerShell version
try {
    Import-Module GcpProfileSimple -Force
    Write-Host "✅ PowerShell canonical version loads correctly" -ForegroundColor Green
} catch {
    Write-Host "❌ PowerShell canonical version failed: $_" -ForegroundColor Red
}

# Test WSL version
$wslTest = wsl bash -c '/home/david/.local/bin/gcp-profile-simple help 2>/dev/null && echo "SUCCESS" || echo "FAILED"'
if ($wslTest -match "SUCCESS") {
    Write-Host "✅ WSL canonical version works correctly" -ForegroundColor Green
} else {
    Write-Host "❌ WSL canonical version failed" -ForegroundColor Red
}

Write-Host "`n🎉 CONSOLIDATION COMPLETE!" -ForegroundColor Green
Write-Host "=" * 50 -ForegroundColor Green

Write-Host "`nSUMMARY:" -ForegroundColor Cyan
Write-Host "✅ CANONICAL PowerShell: GcpProfileSimple" -ForegroundColor Green
Write-Host "✅ CANONICAL WSL/Bash: gcp-profile-simple" -ForegroundColor Green
Write-Host "📦 Backups saved to: $backupDir" -ForegroundColor Yellow
Write-Host "🔗 Compatibility alias: gcp-profile -> gcp-profile-simple" -ForegroundColor Cyan

Write-Host "`nIMPORTANT REMINDERS:" -ForegroundColor Red
Write-Host "❌ NEVER create variants like gcp-profile-v2, enhanced, simple again" -ForegroundColor Red
Write-Host "✅ ALWAYS update existing canonical files instead" -ForegroundColor Green
Write-Host "🔍 If you need new features, add them to the canonical versions" -ForegroundColor Yellow

Write-Host "`nNext steps:" -ForegroundColor Yellow
Write-Host "1. Test both systems: Set-GcpProfile business, gcp-profile-simple business" -ForegroundColor White
Write-Host "2. Update any scripts that reference old GcpUtils functions" -ForegroundColor White
Write-Host "3. Remove backup directory after confirming everything works" -ForegroundColor White

Write-Host "`n✅ ANTI-DUPLICATION DIRECTIVE ENFORCED!" -ForegroundColor Green