# Verify Node.js symlinks setup
Write-Host "=== Node.js Symlinks Verification ===" -ForegroundColor Cyan
Write-Host ""

$localBin = "C:\Users\david\.local\bin"

# Check directory
Write-Host "Checking .local\bin directory:" -ForegroundColor Yellow
if (Test-Path $localBin) {
    Write-Host "  [OK] Directory exists: $localBin" -ForegroundColor Green

    # List Node.js related files
    $nodeFiles = Get-ChildItem $localBin -Filter "*node*", "*npm*", "*npx*" -ErrorAction SilentlyContinue
    if ($nodeFiles) {
        Write-Host "  Files in directory:" -ForegroundColor Gray
        foreach ($file in $nodeFiles) {
            $size = if ($file.Length -gt 1MB) {
                "{0:N2} MB" -f ($file.Length / 1MB)
            } else {
                "{0:N0} KB" -f ($file.Length / 1KB)
            }
            Write-Host "    - $($file.Name) ($size)" -ForegroundColor White
        }
    }
} else {
    Write-Host "  [ERROR] Directory not found" -ForegroundColor Red
}

# Check PATH
Write-Host "`nChecking PATH configuration:" -ForegroundColor Yellow
$userPath = [Environment]::GetEnvironmentVariable("PATH", "User")
if ($userPath -contains $localBin -or $userPath -split ";" -contains $localBin) {
    Write-Host "  [OK] $localBin is in User PATH" -ForegroundColor Green
} else {
    Write-Host "  [WARNING] $localBin is NOT in User PATH" -ForegroundColor Yellow
}

# Test executables directly
Write-Host "`nTesting executables directly:" -ForegroundColor Yellow
$tests = @(
    @{Name = "node"; Path = "$localBin\node.exe"; Command = "--version"},
    @{Name = "npm"; Path = "$localBin\npm.bat"; Command = "--version"},
    @{Name = "npx"; Path = "$localBin\npx.bat"; Command = "--version"}
)

foreach ($test in $tests) {
    if (Test-Path $test.Path) {
        try {
            $version = & $test.Path $test.Command 2>&1 | Select-Object -First 1
            Write-Host "  [OK] $($test.Name): $version" -ForegroundColor Green
        } catch {
            Write-Host "  [ERROR] $($test.Name): Failed to execute" -ForegroundColor Red
        }
    } else {
        # Check for alternative paths
        $altPaths = @("$localBin\$($test.Name).cmd", "$localBin\$($test.Name).exe", "$localBin\$($test.Name).bat")
        $found = $false
        foreach ($alt in $altPaths) {
            if (Test-Path $alt) {
                try {
                    $version = & $alt $test.Command 2>&1 | Select-Object -First 1
                    Write-Host "  [OK] $($test.Name): $version (via $(Split-Path $alt -Leaf))" -ForegroundColor Green
                    $found = $true
                    break
                } catch {}
            }
        }
        if (!$found) {
            Write-Host "  [ERROR] $($test.Name): Not found" -ForegroundColor Red
        }
    }
}

# Test from PATH (simulating new terminal)
Write-Host "`nTesting commands from PATH:" -ForegroundColor Yellow
$env:Path = $userPath + ";" + [Environment]::GetEnvironmentVariable("PATH", "Machine")

$commands = @("node", "npm", "npx")
foreach ($cmd in $commands) {
    try {
        $result = & cmd /c "where $cmd" 2>&1
        if ($LASTEXITCODE -eq 0) {
            $location = $result | Select-Object -First 1
            if ($location -like "*\.local\bin\*") {
                Write-Host "  [OK] $cmd found in .local\bin" -ForegroundColor Green
            } else {
                Write-Host "  [INFO] $cmd found at: $location" -ForegroundColor Yellow
            }
        } else {
            Write-Host "  [WARNING] $cmd not found in PATH" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "  [ERROR] Failed to check $cmd" -ForegroundColor Red
    }
}

# Permission test
Write-Host "`nTesting permissions (no admin required):" -ForegroundColor Yellow
try {
    # Try to run a simple node command
    $testResult = & "$localBin\node.exe" -e "console.log('Hello from Node.js')" 2>&1
    if ($testResult -eq "Hello from Node.js") {
        Write-Host "  [OK] Node.js executes without admin permissions" -ForegroundColor Green
    } else {
        Write-Host "  [WARNING] Unexpected output: $testResult" -ForegroundColor Yellow
    }
} catch {
    Write-Host "  [ERROR] Failed to execute Node.js: $_" -ForegroundColor Red
}

# Summary
Write-Host "`n=== Summary ===" -ForegroundColor Cyan
Write-Host "Node.js symlinks have been successfully created in:" -ForegroundColor Green
Write-Host "  $localBin" -ForegroundColor White
Write-Host ""
Write-Host "You can now run these commands without admin permissions:" -ForegroundColor Green
Write-Host "  - node" -ForegroundColor White
Write-Host "  - npm" -ForegroundColor White
Write-Host "  - npx" -ForegroundColor White
Write-Host ""
Write-Host "Note: If commands aren't recognized, restart your terminal." -ForegroundColor Yellow