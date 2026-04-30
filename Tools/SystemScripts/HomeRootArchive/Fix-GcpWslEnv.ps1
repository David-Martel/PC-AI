# Script to remove API keys from WSLENV and clean up environment

Write-Host "=== Fixing GCP Environment Configuration ===" -ForegroundColor Green

# Remove API keys from WSLENV
Write-Host "`n1. Cleaning WSLENV..." -ForegroundColor Yellow
$currentWslEnv = [Environment]::GetEnvironmentVariable('WSLENV', 'User')
if ($currentWslEnv) {
    Write-Host "   Current WSLENV entries:" -ForegroundColor Cyan
    $currentWslEnv -split ':' | ForEach-Object { Write-Host "   - $_" }

    $keysToRemove = @('GEMINI_API_KEY', 'VERTEX_AI_API_KEY', 'GOOGLE_API_KEY', 'GENERATIVE_AI_API_KEY')
    $wslEnvVars = $currentWslEnv -split ':' | Where-Object {
        $varName = $_ -replace '/.*$', ''
        $varName -notin $keysToRemove
    }

    $newWslEnv = $wslEnvVars -join ':'
    [Environment]::SetEnvironmentVariable('WSLENV', $newWslEnv, 'User')
    $env:WSLENV = $newWslEnv

    Write-Host "`n   Removed from WSLENV:" -ForegroundColor Red
    $keysToRemove | ForEach-Object { Write-Host "   - $_" }

    Write-Host "`n   New WSLENV entries:" -ForegroundColor Green
    $wslEnvVars | ForEach-Object { Write-Host "   - $_" }
}

# Remove API keys from current session
Write-Host "`n2. Removing API keys from current session..." -ForegroundColor Yellow
$apiKeys = @('GEMINI_API_KEY', 'VERTEX_AI_API_KEY', 'GOOGLE_API_KEY', 'GENERATIVE_AI_API_KEY')
foreach ($key in $apiKeys) {
    if (Test-Path "Env:\$key") {
        Remove-Item "Env:\$key" -Force
        Write-Host "   ✓ Removed $key" -ForegroundColor Green
    }
}

# Remove API keys from user environment
Write-Host "`n3. Removing API keys from user environment..." -ForegroundColor Yellow
foreach ($key in $apiKeys) {
    $value = [Environment]::GetEnvironmentVariable($key, 'User')
    if ($value) {
        [Environment]::SetEnvironmentVariable($key, $null, 'User')
        Write-Host "   ✓ Removed $key from user environment" -ForegroundColor Green
    }
}

# Check current GCP profile from WSL
Write-Host "`n4. Checking GCP profile in WSL..." -ForegroundColor Yellow
try {
    $currentProfile = wsl cat /home/david/.gcp/shared/current-profile.txt 2`>`/dev/null
    if ($currentProfile) {
        Write-Host "   Current profile: $currentProfile" -ForegroundColor Cyan

        # Set non-sensitive environment variables
        switch ($currentProfile.Trim()) {
            'business' {
                $env:GOOGLE_CLOUD_PROJECT = 'auricleinc-gemini'
                $env:GCP_PROJECT = 'auricleinc-gemini'
                Write-Host "   Project: auricleinc-gemini" -ForegroundColor Cyan
            }
            'personal' {
                $env:GOOGLE_CLOUD_PROJECT = 'dtm-gemini-ai'
                $env:GCP_PROJECT = 'dtm-gemini-ai'
                Write-Host "   Project: dtm-gemini-ai" -ForegroundColor Cyan
            }
        }

        $env:GOOGLE_CLOUD_LOCATION = 'us-central1'
        $env:GOOGLE_GENAI_USE_VERTEXAI = 'true'
        $env:GCP_CURRENT_PROFILE = $currentProfile.Trim()
    } else {
        Write-Host "   No active profile found" -ForegroundColor Yellow
    }
} catch {
    Write-Host "   Could not access WSL profile: $_" -ForegroundColor Red
}

Write-Host "`n=== Configuration Fixed ===" -ForegroundColor Green
Write-Host "API keys have been removed from environment variables and WSLENV." -ForegroundColor Cyan
Write-Host "Keys remain securely stored in WSL at ~/.gcp/" -ForegroundColor Cyan
Write-Host "`nTo switch profiles, run from WSL:" -ForegroundColor Yellow
Write-Host "  gcp-profile business" -ForegroundColor White
Write-Host "  gcp-profile personal" -ForegroundColor White