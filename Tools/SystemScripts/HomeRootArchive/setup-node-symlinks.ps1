# Setup Node.js symlinks in user's .local\bin directory
# This allows running node, npm, npx without admin permissions

$localBin = "C:\Users\david\.local\bin"
$nodeDir = "C:\Program Files\nodejs"

# Ensure .local\bin exists
if (!(Test-Path $localBin)) {
    New-Item -ItemType Directory -Path $localBin -Force | Out-Null
    Write-Host "Created $localBin directory" -ForegroundColor Green
}

# Define the executables to symlink
$executables = @(
    @{Name = "node.exe"; Source = "$nodeDir\node.exe"; Target = "$localBin\node.exe"},
    @{Name = "npm.cmd"; Source = "$nodeDir\npm.cmd"; Target = "$localBin\npm.cmd"},
    @{Name = "npx.cmd"; Source = "$nodeDir\npx.cmd"; Target = "$localBin\npx.cmd"},
    @{Name = "npm"; Source = "$nodeDir\npm.cmd"; Target = "$localBin\npm.bat"},
    @{Name = "npx"; Source = "$nodeDir\npx.cmd"; Target = "$localBin\npx.bat"}
)

Write-Host "`n=== Creating Node.js Symlinks ===" -ForegroundColor Cyan
Write-Host "Source: $nodeDir" -ForegroundColor Gray
Write-Host "Target: $localBin" -ForegroundColor Gray
Write-Host ""

foreach ($exe in $executables) {
    # Remove existing symlink/file if it exists
    if (Test-Path $exe.Target) {
        Remove-Item $exe.Target -Force
        Write-Host "Removed existing: $($exe.Target)" -ForegroundColor Yellow
    }

    # Create symlink
    try {
        # For .exe files, create hard link (doesn't require admin)
        if ($exe.Name -like "*.exe") {
            cmd /c mklink /H "$($exe.Target)" "$($exe.Source)" 2>&1 | Out-Null
            if (Test-Path $exe.Target) {
                Write-Host "[OK] Created hard link: $($exe.Name)" -ForegroundColor Green
            } else {
                throw "Failed to create hard link"
            }
        }
        # For .cmd files, create batch wrapper (most compatible, no admin needed)
        elseif ($exe.Target -like "*.bat") {
            $batContent = "@echo off`r`n`"$($exe.Source)`" %*"
            Set-Content -Path $exe.Target -Value $batContent -Encoding ASCII
            Write-Host "[OK] Created wrapper: $($exe.Name) -> $(Split-Path $exe.Target -Leaf)" -ForegroundColor Green
        }
        # For .cmd files direct symlink
        else {
            # Try symbolic link first (might need admin)
            $result = cmd /c mklink "$($exe.Target)" "$($exe.Source)" 2>&1
            if ($LASTEXITCODE -ne 0) {
                # Fall back to copying the file
                Copy-Item -Path $exe.Source -Destination $exe.Target -Force
                Write-Host "[OK] Copied file: $($exe.Name) (symlink requires admin)" -ForegroundColor Yellow
            } else {
                Write-Host "[OK] Created symlink: $($exe.Name)" -ForegroundColor Green
            }
        }
    } catch {
        Write-Host "[ERROR] Failed to create link for $($exe.Name): $_" -ForegroundColor Red
    }
}

# Check if .local\bin is in PATH
$userPath = [Environment]::GetEnvironmentVariable("PATH", "User")
$localBinInPath = $userPath -split ";" | Where-Object { $_ -eq $localBin }

if (!$localBinInPath) {
    Write-Host "`n=== Adding $localBin to User PATH ===" -ForegroundColor Cyan
    $newPath = "$localBin;$userPath"
    [Environment]::SetEnvironmentVariable("PATH", $newPath, "User")
    Write-Host "[OK] Added to User PATH" -ForegroundColor Green
    Write-Host "Note: Restart your terminal for PATH changes to take effect" -ForegroundColor Yellow
} else {
    Write-Host "`n[OK] $localBin is already in PATH" -ForegroundColor Green
}

# Test the setup
Write-Host "`n=== Testing Setup ===" -ForegroundColor Cyan
$testCommands = @("node", "npm", "npx")

foreach ($cmd in $testCommands) {
    $testPath = "$localBin\$cmd"
    # Check for both direct executable and batch wrapper
    if ((Test-Path "$testPath.exe") -or (Test-Path "$testPath.cmd") -or (Test-Path "$testPath.bat")) {
        Write-Host "[OK] $cmd is available" -ForegroundColor Green
    } else {
        Write-Host "[WARNING] $cmd not found" -ForegroundColor Yellow
    }
}

Write-Host "`n=== Verification Commands ===" -ForegroundColor Cyan
Write-Host "Run these commands to verify (in a new terminal):" -ForegroundColor Gray
Write-Host "  node --version" -ForegroundColor White
Write-Host "  npm --version" -ForegroundColor White
Write-Host "  npx --version" -ForegroundColor White

Write-Host "`n=== Setup Complete ===" -ForegroundColor Green
Write-Host "Node.js tools are now available in your user directory without admin permissions." -ForegroundColor Green