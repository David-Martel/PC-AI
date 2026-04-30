# Initialize-GcpProfile.ps1
# Comprehensive GCP Profile System Initialization and Fix Script
# This addresses all identified issues with the GCP profile system

[CmdletBinding()]
param(
    [Parameter(Mandatory = $false)]
    [ValidateSet('business', 'personal')]
    [string]$DefaultProfile = 'business',

    [switch]$CleanStart,
    [switch]$ShowDetails
)

Write-Host "=== GCP Profile System Initialization ===" -ForegroundColor Green
Write-Host "Fixing all identified issues and initializing clean system..." -ForegroundColor Cyan

# Step 1: Fix PowerShell Module Path Priority
Write-Host "`n[1/7] Fixing PowerShell module path priority..." -ForegroundColor Yellow

$localModulePath = "C:\Users\david\Documents\PowerShell\Modules"
$oneDriveModulePath = "C:\Users\david\OneDrive\Documents\PowerShell\Modules"

# Get current PSModulePath
$currentPaths = $env:PSModulePath -split ';'

# Remove OneDrive path if present and add local path first
$cleanedPaths = $currentPaths | Where-Object { $_ -ne $oneDriveModulePath }
if ($localModulePath -notin $cleanedPaths) {
    $env:PSModulePath = "$localModulePath;" + ($cleanedPaths -join ';')
} else {
    # Move local path to front
    $cleanedPaths = $cleanedPaths | Where-Object { $_ -ne $localModulePath }
    $env:PSModulePath = "$localModulePath;" + ($cleanedPaths -join ';')
}

Write-Host "[SUCCESS] Local module path prioritized" -ForegroundColor Green

# Step 2: Ensure Required Directories Exist
Write-Host "`n[2/7] Ensuring required directories exist..." -ForegroundColor Yellow

$requiredDirs = @(
    "$env:USERPROFILE\.auth",
    "$env:USERPROFILE\.auth\business",
    "$env:USERPROFILE\.auth\personal",
    "$env:USERPROFILE\.auth\shared",
    "$env:USERPROFILE\.gcp"
)

foreach ($dir in $requiredDirs) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "[CREATED] $dir" -ForegroundColor Green
    } else {
        Write-Host "[EXISTS] $dir" -ForegroundColor Green
    }
}

# Step 3: Synchronize Profile Indicators
Write-Host "`n[3/7] Synchronizing profile indicators..." -ForegroundColor Yellow

$gcpProfilePath = "$env:USERPROFILE\.gcp\current-profile.txt"
$authProfilePath = "$env:USERPROFILE\.auth\shared\current-profile.txt"

# Ensure both profile indicators exist and match
if (Test-Path $gcpProfilePath) {
    $currentProfile = Get-Content $gcpProfilePath -Raw | ForEach-Object { $_.Trim() }
    if (-not (Test-Path $authProfilePath) -or (Get-Content $authProfilePath -Raw).Trim() -ne $currentProfile) {
        $currentProfile | Out-File $authProfilePath -Encoding ASCII -NoNewline
        Write-Host "[SYNCED] Profile indicators synchronized to: $currentProfile" -ForegroundColor Green
    }
} else {
    # Set default profile
    $DefaultProfile | Out-File $gcpProfilePath -Encoding ASCII -NoNewline
    $DefaultProfile | Out-File $authProfilePath -Encoding ASCII -NoNewline
    Write-Host "[CREATED] Default profile set to: $DefaultProfile" -ForegroundColor Green
}

# Step 4: Validate Profile Environment Files
Write-Host "`n[4/7] Validating profile environment files..." -ForegroundColor Yellow

$businessEnv = "$env:USERPROFILE\.gcp\business.env"
$personalEnv = "$env:USERPROFILE\.gcp\personal.env"

if (-not (Test-Path $businessEnv)) {
    $businessContent = @"
# GCP Profile: business
# Generated: $(Get-Date)
GOOGLE_CLOUD_PROJECT=auricleinc-gemini
GOOGLE_CLOUD_LOCATION=us-central1
GOOGLE_APPLICATION_CREDENTIALS=$env:USERPROFILE\.auth\business\service-account-key.json
GCP_PROFILE_NAME=business
GCP_PROFILE_EMAIL=david.martel@auricleinc.com
"@
    $businessContent | Out-File $businessEnv -Encoding UTF8
    Write-Host "[CREATED] Business profile environment" -ForegroundColor Green
}

if (-not (Test-Path $personalEnv)) {
    $personalContent = @"
# GCP Profile: personal
# Generated: $(Get-Date)
GOOGLE_CLOUD_PROJECT=dtm-gemini-ai
GOOGLE_CLOUD_LOCATION=us-central1
GOOGLE_APPLICATION_CREDENTIALS=$env:USERPROFILE\.auth\personal\service-account-key.json
GCP_PROFILE_NAME=personal
GCP_PROFILE_EMAIL=dtmartel1@gmail.com
"@
    $personalContent | Out-File $personalEnv -Encoding UTF8
    Write-Host "[CREATED] Personal profile environment" -ForegroundColor Green
}

# Step 5: Create Current Profile Symlink
Write-Host "`n[5/7] Creating current profile symlink..." -ForegroundColor Yellow

$currentProfile = Get-Content $gcpProfilePath -Raw | ForEach-Object { $_.Trim() }
$currentEnvPath = "$env:USERPROFILE\.gcp\current-profile.env"
$targetEnvPath = "$env:USERPROFILE\.gcp\$currentProfile.env"

if (Test-Path $currentEnvPath) {
    Remove-Item $currentEnvPath -Force
}

if (Test-Path $targetEnvPath) {
    Copy-Item $targetEnvPath $currentEnvPath -Force
    Write-Host "[CREATED] Current profile environment link" -ForegroundColor Green
}

# Step 6: Clean Import GcpUtils Module
Write-Host "`n[6/7] Loading GcpUtils module cleanly..." -ForegroundColor Yellow

try {
    # Remove any existing module
    Get-Module GcpUtils -ErrorAction SilentlyContinue | Remove-Module -Force

    # Import from correct location with error action
    $moduleManifest = "$localModulePath\GcpUtils\GcpUtils.psd1"
    if (Test-Path $moduleManifest) {
        Import-Module $moduleManifest -Force -ErrorAction SilentlyContinue
        Write-Host "[SUCCESS] GcpUtils module loaded from local path" -ForegroundColor Green
    } else {
        # Fallback to .psm1
        $moduleScript = "$localModulePath\GcpUtils\GcpUtils.psm1"
        if (Test-Path $moduleScript) {
            Import-Module $moduleScript -Force -ErrorAction SilentlyContinue
            Write-Host "[SUCCESS] GcpUtils module loaded from .psm1" -ForegroundColor Green
        } else {
            Write-Host "[WARNING] GcpUtils module files not found" -ForegroundColor Yellow
        }
    }

    # Test basic functionality
    if (Get-Command Get-GcpProfile -ErrorAction SilentlyContinue) {
        Write-Host "[VERIFIED] GcpUtils functions available" -ForegroundColor Green
    } else {
        Write-Host "[WARNING] GcpUtils functions not loaded" -ForegroundColor Yellow
    }
}
catch {
    Write-Host "[ERROR] Failed to load GcpUtils: $($_.Exception.Message)" -ForegroundColor Red
}

# Step 7: Final Validation and Environment Setup
Write-Host "`n[7/7] Final validation and environment setup..." -ForegroundColor Yellow

# Load current profile environment variables
$currentEnvFile = "$env:USERPROFILE\.gcp\current-profile.env"
if (Test-Path $currentEnvFile) {
    Get-Content $currentEnvFile | ForEach-Object {
        if ($_ -match '^([^#][^=]+)=(.+)$') {
            $varName = $matches[1].Trim()
            $varValue = $matches[2].Trim()

            # Only set non-sensitive variables
            if ($varName -in @('GOOGLE_CLOUD_PROJECT', 'GOOGLE_CLOUD_LOCATION', 'GCP_PROFILE_NAME', 'GCP_PROFILE_EMAIL')) {
                Set-Item -Path "Env:\$varName" -Value $varValue
                Write-Host "[SET] $varName = $varValue" -ForegroundColor Cyan
            }
        }
    }
}

Write-Host "`n=== Initialization Complete ===" -ForegroundColor Green
Write-Host "Current Profile: $(Get-Content $gcpProfilePath -Raw)" -ForegroundColor Cyan
Write-Host "Available Commands:" -ForegroundColor Yellow
Write-Host "  Get-GcpProfile      - Check current profile" -ForegroundColor White
Write-Host "  Set-GcpProfile      - Switch profiles" -ForegroundColor White
Write-Host "  Test-GcpProfile     - Validate profile" -ForegroundColor White

# Return status information
return @{
    Success = $true
    CurrentProfile = Get-Content $gcpProfilePath -Raw -ErrorAction SilentlyContinue
    ModuleLoaded = (Get-Module GcpUtils) -ne $null
    RequiredPathsExist = ($requiredDirs | ForEach-Object { Test-Path $_ }) -notcontains $false
}