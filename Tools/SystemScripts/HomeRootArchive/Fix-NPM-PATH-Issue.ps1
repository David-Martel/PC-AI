# Fix NPM PATH Issue - Add npm global directory to PATH and remove duplicates
# This script fixes the issue where npm-installed global packages are inaccessible

Write-Host "=== NPM PATH Fix Script ===" -ForegroundColor Cyan
Write-Host "Root Cause: C:\Users\david\AppData\Roaming\npm is not in PATH" -ForegroundColor Yellow
Write-Host ""

# Get current User PATH
$userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
Write-Host "Current User PATH entries:" -ForegroundColor Green
$userPath -split ';' | ForEach-Object { Write-Host "  $_" }
Write-Host ""

# Define the npm directory that needs to be added
$npmGlobalDir = "C:\Users\david\AppData\Roaming\npm"

# Split PATH into array and remove duplicates while preserving order
$pathArray = $userPath -split ';' | Where-Object { $_ -ne '' }
$uniquePath = [System.Collections.Generic.List[string]]::new()
$seen = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)

foreach ($entry in $pathArray) {
    # Normalize path (remove trailing slashes)
    $normalized = $entry.TrimEnd('\', '/')

    if (-not $seen.Contains($normalized)) {
        $uniquePath.Add($entry)
        $seen.Add($normalized) | Out-Null
        Write-Host "  Keeping: $entry" -ForegroundColor Green
    } else {
        Write-Host "  Removing duplicate: $entry" -ForegroundColor Red
    }
}

# Check if npm global directory is already in PATH
if (-not $seen.Contains($npmGlobalDir)) {
    Write-Host ""
    Write-Host "Adding npm global directory to PATH: $npmGlobalDir" -ForegroundColor Cyan
    $uniquePath.Insert(0, $npmGlobalDir)  # Add at beginning for priority
} else {
    Write-Host ""
    Write-Host "npm global directory already in PATH" -ForegroundColor Green
}

# Reconstruct PATH
$newPath = $uniquePath -join ';'

Write-Host ""
Write-Host "=== New User PATH ===" -ForegroundColor Cyan
$newPath -split ';' | ForEach-Object { Write-Host "  $_" -ForegroundColor Green }

# Prompt for confirmation
Write-Host ""
$response = Read-Host "Apply these changes? (y/n)"

if ($response -eq 'y' -or $response -eq 'Y') {
    try {
        # Set the new PATH
        [Environment]::SetEnvironmentVariable('Path', $newPath, 'User')

        # Update current session PATH
        $env:Path = $newPath + ';' + [Environment]::GetEnvironmentVariable('Path', 'Machine')

        Write-Host ""
        Write-Host "=== SUCCESS ===" -ForegroundColor Green
        Write-Host "PATH has been updated successfully!" -ForegroundColor Green
        Write-Host ""
        Write-Host "IMPORTANT: Close and reopen any terminal windows for changes to take effect" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "Testing npm global packages in current session..." -ForegroundColor Cyan

        # Test a few common npm packages
        Write-Host ""
        Write-Host "Testing tsc (TypeScript):" -ForegroundColor Cyan
        & tsc --version 2>&1

        Write-Host ""
        Write-Host "Testing eslint:" -ForegroundColor Cyan
        & eslint --version 2>&1

        Write-Host ""
        Write-Host "Testing prettier:" -ForegroundColor Cyan
        & prettier --version 2>&1

        Write-Host ""
        Write-Host "If commands above show version numbers, the fix is working!" -ForegroundColor Green

    } catch {
        Write-Host "ERROR: Failed to update PATH: $_" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "Changes cancelled." -ForegroundColor Yellow
}
