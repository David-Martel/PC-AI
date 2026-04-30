# Migrate-UbuntuWSLtoVHDX.ps1
param (
    [switch]$UnregisterOriginal = $false
)

$BackupDir = "T:\wsl-backups"
$VHDXRoot  = "T:\vm\wsl-vhdx"

# Ensure folders exist
New-Item -ItemType Directory -Path $BackupDir -Force | Out-Null
New-Item -ItemType Directory -Path $VHDXRoot -Force | Out-Null

# Get list of WSL distros
$wslList = wsl --list --quiet
$ubuntuDistros = $wslList | Where-Object {
    ($_ -ieq "Ubuntu") -and ($_ -notmatch "(?i)docker")
}

if ($ubuntuDistros.Count -eq 0) {
    Write-Host "No eligible Ubuntu WSL distros found to migrate." -ForegroundColor Yellow
    exit
}

foreach ($distro in $ubuntuDistros) {
    Write-Host "`nProcessing: $distro" -ForegroundColor Cyan

    $safeName = $distro -replace '[^a-zA-Z0-9_\-]', '_'
    $tarPath = Join-Path $BackupDir "$safeName.tar"
    $targetDir = Join-Path $VHDXRoot $safeName

    # Step 1: Export
    Write-Host " → Exporting to $tarPath..."
    wsl --export $distro $tarPath

    # Step 2: Create target directory
    Write-Host " → Creating target directory $targetDir..."
    New-Item -ItemType Directory -Path $targetDir -Force | Out-Null

    # Step 3: Import
    Write-Host " → Importing into $targetDir..."
    wsl --import $safeName $targetDir $tarPath --version 2

    # Step 4: Optionally unregister
    if ($UnregisterOriginal) {
        Write-Host " → Unregistering original distro $distro..."
        wsl --unregister $distro
    }

    Write-Host " ✅ Migration complete for: $distro" -ForegroundColor Green
}

Write-Host "`nAll eligible distros have been processed." -ForegroundColor Magenta
