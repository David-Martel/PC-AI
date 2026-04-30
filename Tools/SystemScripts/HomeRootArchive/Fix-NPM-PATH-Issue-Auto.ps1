# Fix NPM PATH Issue - Add npm global directory to PATH and remove duplicates
# AUTO-APPLY VERSION - No confirmation prompt

Write-Host "=== NPM PATH Fix Script (Auto-Apply) ===" -ForegroundColor Cyan
Write-Host "Root Cause: C:\Users\david\AppData\Roaming\npm is not in PATH" -ForegroundColor Yellow
Write-Host ""

# Get current User PATH
$userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
Write-Host "Current User PATH has $($userPath -split ';' | Where-Object { $_ -ne '' } | Measure-Object | Select-Object -ExpandProperty Count) entries" -ForegroundColor Green

# Define the npm directory that needs to be added
$npmGlobalDir = "C:\Users\david\AppData\Roaming\npm"

# Split PATH into array and remove duplicates while preserving order
$pathArray = $userPath -split ';' | Where-Object { $_ -ne '' }
$uniquePath = [System.Collections.Generic.List[string]]::new()
$seen = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
$duplicatesRemoved = 0

foreach ($entry in $pathArray) {
    # Normalize path (remove trailing slashes)
    $normalized = $entry.TrimEnd('\', '/')

    if (-not $seen.Contains($normalized)) {
        $uniquePath.Add($entry)
        $seen.Add($normalized) | Out-Null
    } else {
        Write-Host "  Removing duplicate: $entry" -ForegroundColor Red
        $duplicatesRemoved++
    }
}

Write-Host "Removed $duplicatesRemoved duplicate PATH entries" -ForegroundColor Yellow

# Check if npm global directory is already in PATH
$npmAdded = $false
if (-not $seen.Contains($npmGlobalDir)) {
    Write-Host "Adding npm global directory to PATH: $npmGlobalDir" -ForegroundColor Cyan
    $uniquePath.Insert(0, $npmGlobalDir)  # Add at beginning for priority
    $npmAdded = $true
} else {
    Write-Host "npm global directory already in PATH" -ForegroundColor Green
}

# Reconstruct PATH
$newPath = $uniquePath -join ';'

try {
    # Set the new PATH
    [Environment]::SetEnvironmentVariable('Path', $newPath, 'User')

    # Update current session PATH
    $env:Path = $newPath + ';' + [Environment]::GetEnvironmentVariable('Path', 'Machine')

    Write-Host ""
    Write-Host "=== SUCCESS ===" -ForegroundColor Green
    Write-Host "User PATH updated:" -ForegroundColor Green
    if ($npmAdded) {
        Write-Host "  [+] Added: $npmGlobalDir" -ForegroundColor Green
    }
    if ($duplicatesRemoved -gt 0) {
        Write-Host "  [-] Removed $duplicatesRemoved duplicate entries" -ForegroundColor Green
    }
    Write-Host ""
    Write-Host "New PATH has $($uniquePath.Count) unique entries" -ForegroundColor Cyan
    Write-Host ""

    Write-Host "Testing npm global packages..." -ForegroundColor Cyan
    Write-Host ""

    # Test npm packages
    Write-Host "1. Testing tsc (TypeScript):" -ForegroundColor White
    try {
        $tscVersion = & tsc --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "   ✓ SUCCESS: $tscVersion" -ForegroundColor Green
        } else {
            Write-Host "   ✗ FAILED: $tscVersion" -ForegroundColor Red
        }
    } catch {
        Write-Host "   ✗ FAILED: $_" -ForegroundColor Red
    }

    Write-Host ""
    Write-Host "2. Testing eslint:" -ForegroundColor White
    try {
        $eslintVersion = & eslint --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "   ✓ SUCCESS: $eslintVersion" -ForegroundColor Green
        } else {
            Write-Host "   ✗ FAILED: $eslintVersion" -ForegroundColor Red
        }
    } catch {
        Write-Host "   ✗ FAILED: $_" -ForegroundColor Red
    }

    Write-Host ""
    Write-Host "3. Testing prettier:" -ForegroundColor White
    try {
        $prettierVersion = & prettier --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "   ✓ SUCCESS: $prettierVersion" -ForegroundColor Green
        } else {
            Write-Host "   ✗ FAILED: $prettierVersion" -ForegroundColor Red
        }
    } catch {
        Write-Host "   ✗ FAILED: $_" -ForegroundColor Red
    }

    Write-Host ""
    Write-Host "4. Verifying npm is still accessible:" -ForegroundColor White
    try {
        $npmVersion = & npm --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "   ✓ SUCCESS: npm $npmVersion" -ForegroundColor Green
        } else {
            Write-Host "   ✗ FAILED: $npmVersion" -ForegroundColor Red
        }
    } catch {
        Write-Host "   ✗ FAILED: $_" -ForegroundColor Red
    }

    Write-Host ""
    Write-Host "=== FIX COMPLETE ===" -ForegroundColor Green
    Write-Host "All npm-installed global packages should now be accessible!" -ForegroundColor Green
    Write-Host ""
    Write-Host "NOTE: New terminal windows will automatically have the updated PATH." -ForegroundColor Yellow
    Write-Host "      This current session has been updated and tests above verify functionality." -ForegroundColor Yellow

} catch {
    Write-Host ""
    Write-Host "ERROR: Failed to update PATH: $_" -ForegroundColor Red
    exit 1
}
