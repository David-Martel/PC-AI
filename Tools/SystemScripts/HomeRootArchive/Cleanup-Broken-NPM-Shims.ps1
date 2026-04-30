# Cleanup Broken NPM Shims from .local\bin
# These broken shims shadow the real npm global executables

Write-Host "=== Cleanup Broken NPM Shims ===" -ForegroundColor Cyan
Write-Host ""

$localBin = "C:\users\david\.local\bin"
$backupDir = "$localBin\backup"
$brokenShims = @("tsc.cmd", "npm.cmd", "npx.cmd")

# Create backup directory
if (-not (Test-Path $backupDir)) {
    New-Item -ItemType Directory -Path $backupDir | Out-Null
    Write-Host "Created backup directory: $backupDir" -ForegroundColor Green
}

Write-Host "Broken shims identified in $localBin`:" -ForegroundColor Yellow
Write-Host "  - tsc.cmd (points to non-existent .claude workspace TypeScript)" -ForegroundColor Yellow
Write-Host "  - npm.cmd (points to non-existent C:\Program Files\nodejs)" -ForegroundColor Yellow
Write-Host "  - npx.cmd (points to non-existent C:\Program Files\nodejs)" -ForegroundColor Yellow
Write-Host ""

# Backup and remove each broken shim
$movedCount = 0
foreach ($shim in $brokenShims) {
    $sourcePath = Join-Path $localBin $shim
    $backupPath = Join-Path $backupDir $shim

    if (Test-Path $sourcePath) {
        try {
            # Backup the file
            Copy-Item -Path $sourcePath -Destination $backupPath -Force
            Write-Host "[BACKUP] $shim -> backup\$shim" -ForegroundColor Cyan

            # Remove the broken shim
            Remove-Item -Path $sourcePath -Force
            Write-Host "[REMOVE] $shim removed from .local\bin" -ForegroundColor Green
            $movedCount++
        } catch {
            Write-Host "[ERROR] Failed to process $shim`: $_" -ForegroundColor Red
        }
    } else {
        Write-Host "[SKIP] $shim not found" -ForegroundColor Gray
    }
}

Write-Host ""
Write-Host "=== Summary ===" -ForegroundColor Cyan
Write-Host "Backed up and removed $movedCount broken shims" -ForegroundColor Green
Write-Host "Backup location: $backupDir" -ForegroundColor Cyan
Write-Host ""

# Verify the real npm commands are now accessible
Write-Host "=== Verification Tests ===" -ForegroundColor Cyan
Write-Host ""

# Test 1: Check where commands are found
Write-Host "1. Locating npm executable:" -ForegroundColor White
$npmLocation = (Get-Command npm -ErrorAction SilentlyContinue).Source
if ($npmLocation) {
    Write-Host "   Found: $npmLocation" -ForegroundColor Green
    if ($npmLocation -like "*AppData\Roaming\npm*") {
        Write-Host "   ✓ Correct location (AppData\Roaming\npm)" -ForegroundColor Green
    } else {
        Write-Host "   ⚠ Unexpected location" -ForegroundColor Yellow
    }
} else {
    Write-Host "   ✗ npm not found in PATH" -ForegroundColor Red
}

Write-Host ""
Write-Host "2. Locating tsc executable:" -ForegroundColor White
$tscLocation = (Get-Command tsc -ErrorAction SilentlyContinue).Source
if ($tscLocation) {
    Write-Host "   Found: $tscLocation" -ForegroundColor Green
    if ($tscLocation -like "*AppData\Roaming\npm*") {
        Write-Host "   ✓ Correct location (AppData\Roaming\npm)" -ForegroundColor Green
    } else {
        Write-Host "   ⚠ Unexpected location" -ForegroundColor Yellow
    }
} else {
    Write-Host "   ✗ tsc not found in PATH" -ForegroundColor Red
}

Write-Host ""
Write-Host "3. Testing npm functionality:" -ForegroundColor White
try {
    $npmVersion = & npm --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "   ✓ npm $npmVersion" -ForegroundColor Green
    } else {
        Write-Host "   ✗ npm failed: $npmVersion" -ForegroundColor Red
    }
} catch {
    Write-Host "   ✗ npm error: $_" -ForegroundColor Red
}

Write-Host ""
Write-Host "4. Testing TypeScript (tsc):" -ForegroundColor White
try {
    $tscVersion = & tsc --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "   ✓ $tscVersion" -ForegroundColor Green
    } else {
        Write-Host "   ✗ tsc failed: $tscVersion" -ForegroundColor Red
    }
} catch {
    Write-Host "   ✗ tsc error: $_" -ForegroundColor Red
}

Write-Host ""
Write-Host "5. Testing eslint:" -ForegroundColor White
try {
    $eslintVersion = & eslint --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "   ✓ $eslintVersion" -ForegroundColor Green
    } else {
        Write-Host "   ✗ eslint failed: $eslintVersion" -ForegroundColor Red
    }
} catch {
    Write-Host "   ✗ eslint error: $_" -ForegroundColor Red
}

Write-Host ""
Write-Host "6. Testing prettier:" -ForegroundColor White
try {
    $prettierVersion = & prettier --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "   ✓ prettier $prettierVersion" -ForegroundColor Green
    } else {
        Write-Host "   ✗ prettier failed: $prettierVersion" -ForegroundColor Red
    }
} catch {
    Write-Host "   ✗ prettier error: $_" -ForegroundColor Red
}

Write-Host ""
Write-Host "=== CLEANUP COMPLETE ===" -ForegroundColor Green
Write-Host "All npm-installed global packages should now work correctly!" -ForegroundColor Green
Write-Host ""
Write-Host "To restore backed up shims (if needed): Copy from $backupDir back to $localBin" -ForegroundColor Cyan
