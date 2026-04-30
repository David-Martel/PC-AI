# Check VHDX Access and Permissions
$vhdxPath = 'T:\vm\docker\DockerDesktopWSL\main\ext4.vhdx'
$dataVhdxPath = 'T:\vm\docker\DockerDesktopWSL\disk\docker_data.vhdx'

Write-Host "=== VHDX File Permissions ===" -ForegroundColor Cyan
if (Test-Path $vhdxPath) {
    $acl = Get-Acl $vhdxPath
    Write-Host "Path: $vhdxPath"
    Write-Host "Owner: $($acl.Owner)"
    Write-Host "Access Rules:"
    $acl.Access | ForEach-Object {
        Write-Host "  $($_.IdentityReference): $($_.FileSystemRights) ($($_.AccessControlType))"
    }
}

Write-Host ""
Write-Host "=== Checking if VHDX files are locked ===" -ForegroundColor Cyan

foreach ($path in @($vhdxPath, $dataVhdxPath)) {
    if (Test-Path $path) {
        Write-Host "Checking: $path"
        try {
            $fileStream = [System.IO.File]::Open($path, 'Open', 'Read', 'ReadWrite')
            $fileStream.Close()
            Write-Host "  Status: NOT LOCKED (accessible)" -ForegroundColor Green
        } catch {
            Write-Host "  Status: LOCKED or ACCESS DENIED" -ForegroundColor Red
            Write-Host "  Error: $_" -ForegroundColor Red
        }
    }
}

Write-Host ""
Write-Host "=== T: Drive Info ===" -ForegroundColor Cyan
Get-Volume -DriveLetter T -ErrorAction SilentlyContinue |
    Select-Object DriveLetter, FileSystemLabel, FileSystem, DriveType, HealthStatus, SizeRemaining, Size |
    Format-List

Write-Host ""
Write-Host "=== Attempting Manual WSL Import Test ===" -ForegroundColor Cyan
Write-Host "Testing if we can register the docker-desktop distro..."

# First check current WSL distros
Write-Host ""
Write-Host "Current WSL distros:"
wsl -l -v

# Try to import with --import-in-place (this is what Docker Desktop does)
Write-Host ""
Write-Host "Attempting import-in-place (dry run analysis)..."
Write-Host "Command that would run: wsl --import-in-place docker-desktop $vhdxPath"

# Check if Hyper-V is enabled
Write-Host ""
Write-Host "=== Hyper-V Status ===" -ForegroundColor Cyan
$hyperv = Get-WindowsOptionalFeature -Online -FeatureName Microsoft-Hyper-V -ErrorAction SilentlyContinue
if ($hyperv) {
    Write-Host "Hyper-V State: $($hyperv.State)"
} else {
    Write-Host "Could not check Hyper-V status"
}

# Check WSL version
Write-Host ""
Write-Host "=== WSL Version ===" -ForegroundColor Cyan
wsl --version 2>&1

Write-Host ""
Write-Host "=== Diagnosis Complete ===" -ForegroundColor Cyan
