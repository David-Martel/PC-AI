#Requires -Version 5.1
<#
.SYNOPSIS
    Creates a backup of .wslconfig

.DESCRIPTION
    Backs up the current .wslconfig file with a timestamp.

.PARAMETER Path
    Custom backup path. Default: same directory with timestamp suffix.

.EXAMPLE
    Backup-WSLConfig
    Creates timestamped backup

.OUTPUTS
    String (path to backup file) or $null if no config exists
#>
function Backup-WSLConfig {
    [CmdletBinding()]
    [OutputType([string])]
    param(
        [Parameter()]
        [string]$Path
    )

    $wslConfigPath = "$env:USERPROFILE\.wslconfig"

    if (-not (Test-Path $wslConfigPath)) {
        Write-Warning "No .wslconfig file found at: $wslConfigPath"
        return $null
    }

    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $backupPath = if ($Path) { $Path } else { "$wslConfigPath.backup_$timestamp" }

    try {
        Copy-Item -Path $wslConfigPath -Destination $backupPath -Force
        Write-Host "[+] Backup created: $backupPath" -ForegroundColor Green
        return $backupPath
    }
    catch {
        Write-Error "Failed to create backup: $($_.Exception.Message)"
        return $null
    }
}
