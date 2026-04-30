#Requires -Version 5.1
#Requires -RunAsAdministrator
<#
.SYNOPSIS
    Creates a scheduled task to sync profile logs every 15 minutes.

.DESCRIPTION
    Sets up a Windows Scheduled Task that runs Sync-ProfileLogs.ps1 every 15 minutes.
    The task runs whether the user is logged on or not, and starts up if the
    scheduled time was missed (e.g., computer was sleeping).

.PARAMETER Uninstall
    Removes the scheduled task instead of creating it.

.EXAMPLE
    .\Install-ProfileLogSyncTask.ps1
    Creates the scheduled task.

.EXAMPLE
    .\Install-ProfileLogSyncTask.ps1 -Uninstall
    Removes the scheduled task.

.NOTES
    Version: 1.0.0
    Created: 2026-01-05
    Requires: Administrator privileges
#>
[CmdletBinding()]
param(
    [switch]$Uninstall
)

$TaskName = 'ProfileLogSync'
$TaskPath = '\PowerShell\'
$ScriptRoot = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
$SyncScript = Join-Path $ScriptRoot 'Sync-ProfileLogs.ps1'
$LogPath = Join-Path $env:USERPROFILE '.machine\logs\task-scheduler.log'

function Remove-ProfileLogSyncTask {
    try {
        $existingTask = Get-ScheduledTask -TaskName $TaskName -TaskPath $TaskPath -ErrorAction SilentlyContinue
        if ($existingTask) {
            Unregister-ScheduledTask -TaskName $TaskName -TaskPath $TaskPath -Confirm:$false
            Write-Host "Removed scheduled task: $TaskPath$TaskName" -ForegroundColor Green
        } else {
            Write-Host "Task not found: $TaskPath$TaskName" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "Failed to remove task: $_" -ForegroundColor Red
        exit 1
    }
}

function Install-ProfileLogSyncTask {
    # Verify sync script exists
    if (-not (Test-Path $SyncScript)) {
        Write-Host "Sync script not found: $SyncScript" -ForegroundColor Red
        exit 1
    }

    # Remove existing task if present
    $existingTask = Get-ScheduledTask -TaskName $TaskName -TaskPath $TaskPath -ErrorAction SilentlyContinue
    if ($existingTask) {
        Write-Host "Removing existing task..." -ForegroundColor Yellow
        Unregister-ScheduledTask -TaskName $TaskName -TaskPath $TaskPath -Confirm:$false
    }

    # Create the task folder if it doesn't exist
    $scheduler = New-Object -ComObject Schedule.Service
    $scheduler.Connect()
    $rootFolder = $scheduler.GetFolder('\')
    try {
        $null = $rootFolder.GetFolder('PowerShell')
    } catch {
        $rootFolder.CreateFolder('PowerShell') | Out-Null
        Write-Host "Created task folder: \PowerShell" -ForegroundColor Gray
    }

    # Define the action - run PowerShell with the sync script
    $action = New-ScheduledTaskAction `
        -Execute 'pwsh.exe' `
        -Argument "-NoProfile -NonInteractive -WindowStyle Hidden -File `"$SyncScript`"" `
        -WorkingDirectory (Split-Path $SyncScript -Parent)

    # Define the trigger - every 15 minutes
    $trigger = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Minutes 15)

    # Define settings
    $settings = New-ScheduledTaskSettingsSet `
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries `
        -StartWhenAvailable `
        -RunOnlyIfNetworkAvailable `
        -MultipleInstances IgnoreNew `
        -ExecutionTimeLimit (New-TimeSpan -Minutes 5)

    # Define principal (run as current user)
    $principal = New-ScheduledTaskPrincipal `
        -UserId $env:USERNAME `
        -LogonType S4U `
        -RunLevel Limited

    # Create the task
    try {
        $task = Register-ScheduledTask `
            -TaskName $TaskName `
            -TaskPath $TaskPath `
            -Action $action `
            -Trigger $trigger `
            -Settings $settings `
            -Principal $principal `
            -Description 'Syncs PowerShell profile logs to OneDrive every 15 minutes.'

        Write-Host "Created scheduled task: $TaskPath$TaskName" -ForegroundColor Green
        Write-Host ""
        Write-Host "Task Details:" -ForegroundColor Cyan
        Write-Host "  Name: $TaskName"
        Write-Host "  Path: $TaskPath"
        Write-Host "  Interval: Every 15 minutes"
        Write-Host "  Script: $SyncScript"
        Write-Host "  Run as: $env:USERNAME"
        Write-Host ""
        Write-Host "To verify: Get-ScheduledTask -TaskName '$TaskName' -TaskPath '$TaskPath'"
        Write-Host "To run now: Start-ScheduledTask -TaskName '$TaskName' -TaskPath '$TaskPath'"
        Write-Host "To uninstall: .\Install-ProfileLogSyncTask.ps1 -Uninstall"
    } catch {
        Write-Host "Failed to create task: $_" -ForegroundColor Red
        exit 1
    }
}

# Main
if ($Uninstall) {
    Remove-ProfileLogSyncTask
} else {
    Install-ProfileLogSyncTask
}
