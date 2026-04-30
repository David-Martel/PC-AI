# Create backup directory with timestamp
$timestamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$backupDir = "C:\Backup\tinntester-pre-lfs-$timestamp"

# Create C:\Backup if it doesn't exist
if (!(Test-Path "C:\Backup")) {
    New-Item -ItemType Directory -Path "C:\Backup" -Force | Out-Null
}

# Create timestamped backup directory
New-Item -ItemType Directory -Path $backupDir -Force | Out-Null

# Output the backup directory path
Write-Output $backupDir

# Start robocopy backup
Write-Output "Starting backup from C:\codedev\tinntester to $backupDir"
robocopy "C:\codedev\tinntester" "$backupDir" /E /COPYALL /R:3 /W:5 /NP /NFL /NDL

# Get backup size
$totalSize = (Get-ChildItem -Path $backupDir -Recurse -File | Measure-Object -Property Length -Sum).Sum
$sizeMB = [math]::Round($totalSize / 1MB, 2)
$sizeGB = [math]::Round($totalSize / 1GB, 2)

Write-Output ""
Write-Output "Backup completed!"
Write-Output "Location: $backupDir"
Write-Output "Size: $sizeMB MB ($sizeGB GB)"
Write-Output ""

# Return the backup directory for verification
return $backupDir
