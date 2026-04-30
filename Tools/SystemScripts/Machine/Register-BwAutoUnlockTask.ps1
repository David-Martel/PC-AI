<#
.SYNOPSIS
    Registers a scheduled task that auto-unlocks Bitwarden at user login.
.DESCRIPTION
    Creates a scheduled task 'BW-Auto-Unlock' that runs Unlock-BwVault.ps1
    at logon for the current user. The task:
    - Reads master password from Windows Credential Manager
    - Handles login if vault is unauthenticated
    - Unlocks the vault and persists BW_SESSION as a User env var
    - Runs hidden (no console window)
.PARAMETER Unregister
    Removes the scheduled task instead of creating it.
#>
[CmdletBinding()]
param(
    [switch]$Unregister
)

$TaskName = 'BW-Auto-Unlock'
$ScriptRoot = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
$ScriptPath = Join-Path $ScriptRoot 'Unlock-BwVault.ps1'

if ($Unregister) {
    $existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($existing) {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        Write-Host "Scheduled task '$TaskName' removed." -ForegroundColor Green
    } else {
        Write-Host "Scheduled task '$TaskName' not found." -ForegroundColor Yellow
    }
    return
}

if (-not (Test-Path $ScriptPath)) {
    Write-Error "Unlock script not found: $ScriptPath"
    return
}

# Remove existing task if present
$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existing) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "Replacing existing '$TaskName' task..." -ForegroundColor Yellow
}

# Action: run pwsh with the unlock script
$pwshPath = (Get-Command pwsh -ErrorAction SilentlyContinue).Source
if (-not $pwshPath) { $pwshPath = 'pwsh.exe' }

$action = New-ScheduledTaskAction `
    -Execute $pwshPath `
    -Argument "-NoLogo -NoProfile -NonInteractive -WindowStyle Hidden -File `"$ScriptPath`" -Quiet -PersistSession" `
    -WorkingDirectory $ScriptRoot

# Trigger: at user logon
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME

# Settings: allow running on battery, don't stop on idle, retry on failure
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 5) `
    -Hidden

# Principal: run as current user, highest available
$principal = New-ScheduledTaskPrincipal `
    -UserId $env:USERNAME `
    -LogonType Interactive `
    -RunLevel Limited

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Principal $principal `
    -Description "Auto-unlocks Bitwarden CLI vault at login using Windows Credential Manager. Sets BW_SESSION as User environment variable." `
    -Force | Out-Null

Write-Host "Scheduled task '$TaskName' registered successfully." -ForegroundColor Green
Write-Host "  Trigger: At logon for $env:USERNAME" -ForegroundColor Gray
Write-Host "  Action: $pwshPath -File $ScriptPath -Quiet -PersistSession" -ForegroundColor Gray
Write-Host ""
Write-Host "To test now: Start-ScheduledTask -TaskName '$TaskName'" -ForegroundColor Cyan
