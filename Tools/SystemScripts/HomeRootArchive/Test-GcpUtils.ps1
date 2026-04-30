# Test script for GcpUtils module
Write-Host "=== Testing GcpUtils Module ===" -ForegroundColor Green

# Import the specific file directly to bypass module issues
try {
    . "C:\Users\david\OneDrive\Documents\PowerShell\Modules\GcpUtils\GcpProfileManagerV2.ps1"
    Write-Host "✓ GcpProfileManagerV2.ps1 loaded" -ForegroundColor Green
} catch {
    Write-Error "Failed to load GcpProfileManagerV2.ps1: $_"
    exit 1
}

# Test Get-GcpProfile
Write-Host "`n--- Testing Get-GcpProfile ---" -ForegroundColor Yellow
try {
    $profile = Get-GcpProfile
    if ($profile) {
        Write-Host "Active Profile: $($profile.ActiveProfile)" -ForegroundColor Cyan
        Write-Host "Project ID: $($profile.ProjectId)" -ForegroundColor Cyan
        Write-Host "Is Valid: $($profile.IsValid)" -ForegroundColor Cyan
        if ($profile.ErrorMessage) {
            Write-Host "Error: $($profile.ErrorMessage)" -ForegroundColor Red
        }
    } else {
        Write-Host "No profile returned" -ForegroundColor Red
    }
} catch {
    Write-Error "Failed to get profile: $_"
}

# Test Remove-ApiKeysFromWslEnv
Write-Host "`n--- Testing API Key Cleanup ---" -ForegroundColor Yellow
try {
    Remove-ApiKeysFromWslEnv
    Write-Host "✓ API keys removed from WSLENV" -ForegroundColor Green
} catch {
    Write-Error "Failed to clean API keys: $_"
}

# Check environment variables
Write-Host "`n--- Checking Environment Variables ---" -ForegroundColor Yellow
$apiKeys = @('GEMINI_API_KEY', 'VERTEX_AI_API_KEY', 'GOOGLE_API_KEY', 'GENERATIVE_AI_API_KEY')
foreach ($key in $apiKeys) {
    $value = [Environment]::GetEnvironmentVariable($key)
    if ($value) {
        Write-Host "WARNING: $key is still set!" -ForegroundColor Red
    } else {
        Write-Host "✓ $key is not exposed" -ForegroundColor Green
    }
}

# Check WSLENV
Write-Host "`n--- Checking WSLENV ---" -ForegroundColor Yellow
$wslenv = [Environment]::GetEnvironmentVariable('WSLENV', 'User')
if ($wslenv) {
    Write-Host "WSLENV: $wslenv" -ForegroundColor Cyan
    foreach ($key in $apiKeys) {
        if ($wslenv -match $key) {
            Write-Host "WARNING: $key found in WSLENV!" -ForegroundColor Red
        }
    }
} else {
    Write-Host "WSLENV not set" -ForegroundColor Yellow
}

Write-Host "`n=== Test Complete ===" -ForegroundColor Green