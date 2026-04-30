# Fix Docker Desktop WSL Distro Registration
# Workaround for Dev Drive VHDX mount limitation

$ErrorActionPreference = "Stop"

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Docker Desktop Dev Drive Fix" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

# Paths
$devDriveVhdx = "T:\vm\docker\DockerDesktopWSL\main\ext4.vhdx"
$tempLocation = "C:\DockerTemp"
$tempVhdx = "$tempLocation\ext4.vhdx"

# Stop Docker Desktop first
Write-Host ""
Write-Host "Step 1: Stopping Docker Desktop..." -ForegroundColor Yellow
Get-Process -Name "Docker Desktop", "Docker*" -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 3

# Check if distro exists
Write-Host ""
Write-Host "Step 2: Checking current WSL distros..." -ForegroundColor Yellow
$wslList = wsl -l -q 2>&1
if ($wslList -match "docker-desktop") {
    Write-Host "  docker-desktop distro already exists, unregistering..." -ForegroundColor Yellow
    wsl --unregister docker-desktop 2>&1
}

# Create temp directory
Write-Host ""
Write-Host "Step 3: Creating temp directory on C: drive..." -ForegroundColor Yellow
if (-not (Test-Path $tempLocation)) {
    New-Item -ItemType Directory -Path $tempLocation -Force | Out-Null
}

# Copy VHDX to C: drive (non-Dev-Drive)
Write-Host ""
Write-Host "Step 4: Copying VHDX to C: drive (this may take a moment)..." -ForegroundColor Yellow
Write-Host "  From: $devDriveVhdx"
Write-Host "  To: $tempVhdx"

if (Test-Path $tempVhdx) {
    Remove-Item $tempVhdx -Force
}
Copy-Item $devDriveVhdx $tempVhdx -Force
Write-Host "  Copy complete!" -ForegroundColor Green

# Register the distro from C: drive
Write-Host ""
Write-Host "Step 5: Registering docker-desktop distro from C: drive..." -ForegroundColor Yellow
$result = wsl --import-in-place docker-desktop $tempVhdx 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "  Registration successful!" -ForegroundColor Green
} else {
    Write-Host "  Registration failed: $result" -ForegroundColor Red
    exit 1
}

# Verify registration
Write-Host ""
Write-Host "Step 6: Verifying registration..." -ForegroundColor Yellow
wsl -l -v

Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host "Fix Complete!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""
Write-Host "The docker-desktop distro is now registered from C:\DockerTemp"
Write-Host "You can now start Docker Desktop."
Write-Host ""
Write-Host "NOTE: The VHDX is now on C: drive, not T: drive."
Write-Host "Docker Desktop should reconfigure itself on next start."
Write-Host ""
Write-Host "Starting Docker Desktop..." -ForegroundColor Yellow
Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"
