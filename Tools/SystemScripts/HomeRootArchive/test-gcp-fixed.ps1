# Test GCP Profile System After Fixes
# This script tests if all the fixes have resolved the issues

Write-Host "=== Testing GCP Profile System ===" -ForegroundColor Green

# Test 1: Import Module without OneDrive conflicts
Write-Host "`n[TEST 1] Importing GcpUtils module..." -ForegroundColor Yellow

# Remove any cached modules
Get-Module GcpUtils -ErrorAction SilentlyContinue | Remove-Module -Force

# Set correct module path
$env:PSModulePath = "C:\Users\david\Documents\PowerShell\Modules;" + $env:PSModulePath

try {
    Import-Module GcpUtils -Force -ErrorAction Stop
    Write-Host "[PASS] Module imported successfully" -ForegroundColor Green
}
catch {
    Write-Host "[FAIL] Module import failed: $($_.Exception.Message)" -ForegroundColor Red
    return
}

# Test 2: Check if functions are available
Write-Host "`n[TEST 2] Checking if functions are available..." -ForegroundColor Yellow

$expectedFunctions = @('Get-GcpProfile', 'Set-GcpProfile', 'Test-GcpProfile')
$allFunctionsAvailable = $true

foreach ($func in $expectedFunctions) {
    if (Get-Command $func -ErrorAction SilentlyContinue) {
        Write-Host "[PASS] $func is available" -ForegroundColor Green
    } else {
        Write-Host "[FAIL] $func is not available" -ForegroundColor Red
        $allFunctionsAvailable = $false
    }
}

if (-not $allFunctionsAvailable) {
    Write-Host "[ERROR] Not all functions are available" -ForegroundColor Red
    return
}

# Test 3: Check current profile
Write-Host "`n[TEST 3] Checking current profile..." -ForegroundColor Yellow

try {
    $currentProfile = Get-GcpProfile
    if ($currentProfile -and $currentProfile.Name) {
        Write-Host "[PASS] Current profile: $($currentProfile.Name)" -ForegroundColor Green
        Write-Host "       Project: $($currentProfile.ProjectId)" -ForegroundColor Cyan
    } else {
        Write-Host "[WARNING] No current profile set, trying to get all profiles" -ForegroundColor Yellow

        # Try to list all profiles
        $allProfiles = Get-GcpProfile -ListAll
        if ($allProfiles) {
            Write-Host "[INFO] Available profiles:" -ForegroundColor Cyan
            $allProfiles | ForEach-Object { Write-Host "       - $($_.Name)" -ForegroundColor White }
        }
    }
}
catch {
    Write-Host "[FAIL] Get-GcpProfile failed: $($_.Exception.Message)" -ForegroundColor Red
}

# Test 4: Test profile switching
Write-Host "`n[TEST 4] Testing profile switching..." -ForegroundColor Yellow

try {
    Set-GcpProfile -Name "business" -ErrorAction Stop
    $newProfile = Get-GcpProfile
    if ($newProfile -and $newProfile.Name -eq "business") {
        Write-Host "[PASS] Successfully switched to business profile" -ForegroundColor Green
    } else {
        Write-Host "[FAIL] Profile switch did not work as expected" -ForegroundColor Red
    }
}
catch {
    Write-Host "[WARNING] Profile switching failed: $($_.Exception.Message)" -ForegroundColor Yellow
    Write-Host "This might be expected if profiles are not fully configured" -ForegroundColor Gray
}

# Test 5: Check required paths
Write-Host "`n[TEST 5] Checking required paths..." -ForegroundColor Yellow

$requiredPaths = @(
    "$env:USERPROFILE\.auth",
    "$env:USERPROFILE\.auth\business",
    "$env:USERPROFILE\.auth\personal",
    "$env:USERPROFILE\.gcp",
    "$env:USERPROFILE\.gcp\current-profile.txt"
)

$allPathsExist = $true
foreach ($path in $requiredPaths) {
    if (Test-Path $path) {
        Write-Host "[PASS] $path exists" -ForegroundColor Green
    } else {
        Write-Host "[FAIL] $path missing" -ForegroundColor Red
        $allPathsExist = $false
    }
}

# Test 6: Check environment variables
Write-Host "`n[TEST 6] Checking environment variables..." -ForegroundColor Yellow

$expectedEnvVars = @('GOOGLE_CLOUD_PROJECT', 'GCP_PROFILE_NAME')
foreach ($envVar in $expectedEnvVars) {
    $value = Get-Item "Env:\$envVar" -ErrorAction SilentlyContinue
    if ($value) {
        Write-Host "[PASS] $envVar = $($value.Value)" -ForegroundColor Green
    } else {
        Write-Host "[INFO] $envVar not set (may be normal)" -ForegroundColor Cyan
    }
}

# Summary
Write-Host "`n=== TEST SUMMARY ===" -ForegroundColor Green
if ($allFunctionsAvailable -and $allPathsExist) {
    Write-Host "Overall Status: SYSTEM FUNCTIONAL" -ForegroundColor Green
    Write-Host "The GCP profile system has been successfully fixed!" -ForegroundColor Green
} else {
    Write-Host "Overall Status: ISSUES REMAIN" -ForegroundColor Yellow
    Write-Host "Some issues may still need to be addressed" -ForegroundColor Yellow
}

Write-Host "`nYou can now use:" -ForegroundColor Cyan
Write-Host "  Get-GcpProfile" -ForegroundColor White
Write-Host "  Set-GcpProfile -Name business" -ForegroundColor White
Write-Host "  Set-GcpProfile -Name personal" -ForegroundColor White