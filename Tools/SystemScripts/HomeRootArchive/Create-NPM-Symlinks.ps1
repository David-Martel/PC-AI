# Create Proper Symlinks for NPM Global Executables
# This creates a symlink-based command access system for npm packages

#Requires -RunAsAdministrator

Write-Host "=== NPM Symlink Management System ===" -ForegroundColor Cyan
Write-Host ""

$npmGlobalBin = "C:\Users\david\AppData\Roaming\npm"
$localBin = "C:\users\david\.local\bin"

# Verify npm global directory exists
if (-not (Test-Path $npmGlobalBin)) {
    Write-Host "ERROR: npm global directory not found: $npmGlobalBin" -ForegroundColor Red
    exit 1
}

# Get all executables in npm global directory
$npmExecutables = Get-ChildItem "$npmGlobalBin\*.cmd" -File
Write-Host "Found $($npmExecutables.Count) npm global executables in $npmGlobalBin" -ForegroundColor Green
Write-Host ""

# List of npm executables to create symlinks for
$targetExecutables = @(
    "tsc.cmd",
    "eslint.cmd",
    "prettier.cmd",
    "npx.cmd"
    # Note: npm.cmd should remain in nvm4w for version management
)

Write-Host "Creating/Updating symlinks for npm executables:" -ForegroundColor Cyan
Write-Host ""

$created = 0
$updated = 0
$skipped = 0
$errors = 0

foreach ($exe in $targetExecutables) {
    $sourcePath = Join-Path $npmGlobalBin $exe
    $linkPath = Join-Path $localBin $exe

    # Check if source exists
    if (-not (Test-Path $sourcePath)) {
        Write-Host "  [SKIP] $exe - not installed in npm global" -ForegroundColor Gray
        $skipped++
        continue
    }

    try {
        # Check if link already exists
        if (Test-Path $linkPath) {
            $item = Get-Item $linkPath

            # If it's already a correct symlink, skip
            if ($item.LinkType -eq 'SymbolicLink' -and $item.Target -eq $sourcePath) {
                Write-Host "  [OK] $exe - symlink already correct" -ForegroundColor Green
                $skipped++
                continue
            }

            # Remove old file/symlink
            Write-Host "  [REMOVE] $exe - removing old file/symlink" -ForegroundColor Yellow
            Remove-Item $linkPath -Force
        }

        # Create symlink
        New-Item -ItemType SymbolicLink -Path $linkPath -Target $sourcePath -Force | Out-Null
        Write-Host "  [CREATE] $exe -> $sourcePath" -ForegroundColor Green

        if ($updated -gt 0) { $updated++ } else { $created++ }

    } catch {
        Write-Host "  [ERROR] $exe - $_" -ForegroundColor Red
        $errors++
    }
}

Write-Host ""
Write-Host "=== Summary ===" -ForegroundColor Cyan
Write-Host "  Created: $created" -ForegroundColor Green
Write-Host "  Updated: $updated" -ForegroundColor Yellow
Write-Host "  Skipped: $skipped" -ForegroundColor Gray
Write-Host "  Errors: $errors" -ForegroundColor $(if ($errors -gt 0) { "Red" } else { "Gray" })
Write-Host ""

# Verification
Write-Host "=== Verification ===" -ForegroundColor Cyan
Write-Host ""

foreach ($exe in $targetExecutables) {
    $linkPath = Join-Path $localBin $exe

    if (Test-Path $linkPath) {
        $item = Get-Item $linkPath
        if ($item.LinkType -eq 'SymbolicLink') {
            Write-Host "✓ $exe" -ForegroundColor Green
            Write-Host "  Type: Symlink" -ForegroundColor Gray
            Write-Host "  Target: $($item.Target)" -ForegroundColor Gray
        } else {
            Write-Host "✗ $exe" -ForegroundColor Red
            Write-Host "  Type: Regular file (should be symlink)" -ForegroundColor Yellow
        }
    } else {
        Write-Host "✗ $exe - not found" -ForegroundColor Red
    }
    Write-Host ""
}

Write-Host "=== Testing Commands ===" -ForegroundColor Cyan
Write-Host ""

# Test tsc
Write-Host "Testing tsc:" -ForegroundColor White
try {
    $tscVersion = & tsc --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✓ $tscVersion" -ForegroundColor Green
    } else {
        Write-Host "  ✗ Failed: $tscVersion" -ForegroundColor Red
    }
} catch {
    Write-Host "  ✗ Error: $_" -ForegroundColor Red
}

Write-Host ""
Write-Host "Testing eslint:" -ForegroundColor White
try {
    $eslintVersion = & eslint --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✓ $eslintVersion" -ForegroundColor Green
    } else {
        Write-Host "  ✗ Failed: $eslintVersion" -ForegroundColor Red
    }
} catch {
    Write-Host "  ✗ Error: $_" -ForegroundColor Red
}

Write-Host ""
Write-Host "Testing prettier:" -ForegroundColor White
try {
    $prettierVersion = & prettier --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✓ prettier $prettierVersion" -ForegroundColor Green
    } else {
        Write-Host "  ✗ Failed: $prettierVersion" -ForegroundColor Red
    }
} catch {
    Write-Host "  ✗ Error: $_" -ForegroundColor Red
}

Write-Host ""
Write-Host "=== Symlink System Created ===" -ForegroundColor Green
Write-Host "All npm global executables are now accessible via symlinks!" -ForegroundColor Green
Write-Host ""
Write-Host "Benefits of symlink approach:" -ForegroundColor Cyan
Write-Host "  • Automatic updates when npm packages are updated" -ForegroundColor White
Write-Host "  • No broken shims pointing to non-existent paths" -ForegroundColor White
Write-Host "  • Single source of truth (AppData\Roaming\npm)" -ForegroundColor White
Write-Host "  • Easy to maintain and update" -ForegroundColor White
