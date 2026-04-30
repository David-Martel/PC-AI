# Simple test script
Write-Host "Testing PowerShell syntax..." -ForegroundColor Green

# Test prerequisites function
function Test-Prerequisites {
    Write-Host "Checking prerequisites..." -ForegroundColor Cyan

    try {
        $uvVersion = uv --version 2>$null
        Write-Host "✓ uv found: $uvVersion" -ForegroundColor Green
    }
    catch {
        Write-Host "✗ uv not found" -ForegroundColor Red
    }
}

Test-Prerequisites
Write-Host "Simple test completed!" -ForegroundColor Green