# Test script for GCP Profile Integration
# This script tests the new GCP profile integration without running the full profile

Write-Host "🧪 Testing GCP Profile Integration..." -ForegroundColor Cyan
Write-Host ""

# Test 1: Check if GcpUtils module is available
Write-Host "Test 1: Checking GcpUtils module availability" -ForegroundColor Yellow
$gcpUtilsAvailable = Get-Module -ListAvailable -Name GcpUtils
if ($gcpUtilsAvailable) {
    Write-Host "  ✅ GcpUtils module found (Version: $($gcpUtilsAvailable.Version))" -ForegroundColor Green
} else {
    Write-Host "  ❌ GcpUtils module not found" -ForegroundColor Red
    Write-Host "     Please ensure GcpUtils v3.0.0 is installed" -ForegroundColor Gray
    return
}

# Test 2: Import GcpUtils module
Write-Host "`nTest 2: Importing GcpUtils module" -ForegroundColor Yellow
try {
    Import-Module GcpUtils -Force -ErrorAction Stop
    Write-Host "  ✅ GcpUtils module imported successfully" -ForegroundColor Green

    # List available functions
    $gcpFunctions = Get-Command -Module GcpUtils | Select-Object -ExpandProperty Name
    Write-Host "     Available functions: $($gcpFunctions.Count)" -ForegroundColor Gray
    Write-Host "     Key functions: Get-GcpProfile, Set-GcpProfile, Test-GcpProfile" -ForegroundColor Gray
} catch {
    Write-Host "  ❌ Failed to import GcpUtils: $_" -ForegroundColor Red
    return
}

# Test 3: Check current profile
Write-Host "`nTest 3: Checking current GCP profile" -ForegroundColor Yellow
try {
    $currentProfile = Get-GcpProfile
    if ($currentProfile) {
        Write-Host "  ✅ Current profile loaded: $($currentProfile.name)" -ForegroundColor Green
        Write-Host "     Project ID: $($currentProfile.projectId)" -ForegroundColor Gray
        Write-Host "     Service Account: $(if ($currentProfile.serviceAccountKey) { 'Configured' } else { 'Not set' })" -ForegroundColor Gray
    } else {
        Write-Host "  ⚠️  No active profile found" -ForegroundColor Yellow
        Write-Host "     This is normal if no profile is currently set" -ForegroundColor Gray
    }
} catch {
    Write-Host "  ❌ Error checking current profile: $_" -ForegroundColor Red
}

# Test 4: Check authentication directory structure
Write-Host "`nTest 4: Checking authentication directory structure" -ForegroundColor Yellow
$authDir = "$env:USERPROFILE\.auth"
if (Test-Path $authDir) {
    Write-Host "  ✅ Auth directory exists: $authDir" -ForegroundColor Green

    $currentProfileFile = Join-Path $authDir "shared\current-profile.txt"
    if (Test-Path $currentProfileFile) {
        $profileName = Get-Content $currentProfileFile -ErrorAction SilentlyContinue | Select-Object -First 1
        Write-Host "     Current profile file: $profileName" -ForegroundColor Gray
    } else {
        Write-Host "     ⚠️  Current profile file not found" -ForegroundColor Yellow
    }

    # Check profile directories
    $businessDir = Join-Path $authDir "business"
    $personalDir = Join-Path $authDir "personal"
    Write-Host "     Business profile: $(if (Test-Path $businessDir) { '✅' } else { '❌' })" -ForegroundColor $(if (Test-Path $businessDir) { 'Green' } else { 'Red' })
    Write-Host "     Personal profile: $(if (Test-Path $personalDir) { '✅' } else { '❌' })" -ForegroundColor $(if (Test-Path $personalDir) { 'Green' } else { 'Red' })
} else {
    Write-Host "  ❌ Auth directory not found: $authDir" -ForegroundColor Red
    Write-Host "     Please run the Google Cloud authentication setup" -ForegroundColor Gray
}

# Test 5: Test profile functionality
Write-Host "`nTest 5: Testing profile functionality" -ForegroundColor Yellow
try {
    if (Get-Command Test-GcpProfile -ErrorAction SilentlyContinue) {
        $profileTest = Test-GcpProfile
        Write-Host "  Profile test result: $(if ($profileTest) { '✅ Working' } else { '⚠️  Issues detected' })" -ForegroundColor $(if ($profileTest) { 'Green' } else { 'Yellow' })
    } else {
        Write-Host "  ⚠️  Test-GcpProfile function not available" -ForegroundColor Yellow
    }
} catch {
    Write-Host "  ❌ Error testing profile: $_" -ForegroundColor Red
}

# Test 6: Environment variables
Write-Host "`nTest 6: Checking environment variables" -ForegroundColor Yellow
$envVars = @(
    @{ Name = "GOOGLE_CLOUD_PROJECT"; Value = $env:GOOGLE_CLOUD_PROJECT },
    @{ Name = "GOOGLE_APPLICATION_CREDENTIALS"; Value = $env:GOOGLE_APPLICATION_CREDENTIALS }
)

foreach ($var in $envVars) {
    if ($var.Value) {
        Write-Host "  ✅ $($var.Name): $($var.Value)" -ForegroundColor Green
    } else {
        Write-Host "  ⚠️  $($var.Name): Not set" -ForegroundColor Yellow
    }
}

# Summary
Write-Host "`n📊 Integration Test Summary:" -ForegroundColor Cyan
Write-Host "   - GcpUtils module: $(if ($gcpUtilsAvailable) { '✅ Available' } else { '❌ Missing' })" -ForegroundColor $(if ($gcpUtilsAvailable) { 'Green' } else { 'Red' })
Write-Host "   - Auth directory: $(if (Test-Path $authDir) { '✅ Present' } else { '❌ Missing' })" -ForegroundColor $(if (Test-Path $authDir) { 'Green' } else { 'Red' })
Write-Host "   - Active profile: $(if ($currentProfile) { "✅ $($currentProfile.name)" } else { '⚠️  None' })" -ForegroundColor $(if ($currentProfile) { 'Green' } else { 'Yellow' })

Write-Host ""
Write-Host "🎯 Integration Status: $(if ($gcpUtilsAvailable -and (Test-Path $authDir)) { 'READY' } else { 'NEEDS SETUP' })" -ForegroundColor $(if ($gcpUtilsAvailable -and (Test-Path $authDir)) { 'Green' } else { 'Red' })
Write-Host ""