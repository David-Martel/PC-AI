# Step 1 - System Restore to RP 65 (4/21 Scheduled Checkpoint).
# This is the heaviest step. ONLY run if Steps 0+2 do not resolve the touchpad
# glitches AND user has confirmed they understand the rollback consequences:
#   - Reverts all driver/registry changes since 4/21
#   - Apps installed since 4/21 may need reinstall
#   - WILL revert 4/30 PL UI/sync tuning - re-apply afterwards
#   - WILL revert any registry tweaks done since 4/21
# Pause Windows Update first to prevent immediate reapplication of the rollup.
[CmdletBinding(SupportsShouldProcess=$true, ConfirmImpact='High')]
param(
    [int]$RestorePointSequence = 65,
    [switch]$Confirm
)
$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot

# Confirm RP exists
$rp = Get-ComputerRestorePoint | Where-Object SequenceNumber -eq $RestorePointSequence
if (-not $rp) {
    Write-Error "Restore Point #$RestorePointSequence not found. Available:"
    Get-ComputerRestorePoint | Format-Table SequenceNumber, CreationTime, Description -AutoSize
    return
}

Write-Host "About to restore to:" -ForegroundColor Yellow
$rp | Format-Table SequenceNumber, CreationTime, Description, RestorePointType -AutoSize

# Document apps installed since RP 65 timestamp
Write-Host "`nApps installed since RP target timestamp ($($rp.CreationTime)):" -ForegroundColor Yellow
$rpDate = $rp.CreationTime
try {
    $sinceApps = Get-WmiObject -Class Win32_Product -ErrorAction SilentlyContinue |
        Where-Object { $_.InstallDate -and ([datetime]::ParseExact($_.InstallDate, 'yyyyMMdd', $null) -ge $rpDate) } |
        Select-Object Name, Version, Vendor, InstallDate
    $appList = Join-Path $root "apps-installed-since-RP$RestorePointSequence.txt"
    $sinceApps | Out-String | Set-Content $appList
    Write-Host "  Documented to: $appList ($($sinceApps.Count) apps)"
} catch {
    Write-Warning "Could not enumerate Win32_Product: $($_.Exception.Message)"
}

# Pause Windows Update for 7 days (prevents immediate rollup reapply)
Write-Host "`nPausing Windows Update for 7 days..." -ForegroundColor Yellow
$pauseUntil = (Get-Date).AddDays(7).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
$wuKey = 'HKLM:\SOFTWARE\Microsoft\WindowsUpdate\UX\Settings'
if (Test-Path $wuKey) {
    Set-ItemProperty -Path $wuKey -Name 'PauseUpdatesExpiryTime' -Value $pauseUntil -Type String -ErrorAction SilentlyContinue
    Set-ItemProperty -Path $wuKey -Name 'PauseFeatureUpdatesEndTime' -Value $pauseUntil -Type String -ErrorAction SilentlyContinue
    Set-ItemProperty -Path $wuKey -Name 'PauseQualityUpdatesEndTime' -Value $pauseUntil -Type String -ErrorAction SilentlyContinue
    Write-Host "  Paused until $pauseUntil" -ForegroundColor Green
} else {
    Write-Warning "  WindowsUpdate UX Settings key not found - pause may not stick"
}

if (-not $Confirm) {
    Write-Host "`nDry run complete. To execute restore:" -ForegroundColor Cyan
    Write-Host "  & '$PSCommandPath' -RestorePointSequence $RestorePointSequence -Confirm" -ForegroundColor Cyan
    Write-Host "Restore-Computer will reboot the machine. Save work first." -ForegroundColor Yellow
    return
}

if ($PSCmdlet.ShouldProcess("Restore Point #$RestorePointSequence ($($rp.CreationTime))", "System Restore")) {
    Write-Host "`nInitiating Restore-Computer..." -ForegroundColor Red
    Restore-Computer -RestorePoint $RestorePointSequence -Confirm:$false
}
