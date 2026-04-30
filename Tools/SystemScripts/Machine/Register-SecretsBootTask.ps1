<#
.SYNOPSIS
    Registers Windows Task Scheduler task for secrets initialization at boot
.DESCRIPTION
    Creates a scheduled task that runs Initialize-SecretsAtBoot.ps1 at user logon
.NOTES
    Run this script once with administrator privileges to register the task
#>

$taskName = "Initialize-MachineSecrets"
$taskPath = "\Bitwarden\"
$scriptRoot = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
$scriptPath = Join-Path $scriptRoot "Initialize-SecretsAtBoot.ps1"

# Check if running as admin (recommended but not required)
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

Write-Host "=== Registering Secrets Boot Task ===" -ForegroundColor Cyan
Write-Host ""

# Check if task already exists
$existingTask = Get-ScheduledTask -TaskName $taskName -TaskPath $taskPath -ErrorAction SilentlyContinue

if ($existingTask) {
    Write-Host "Task already exists. Updating..." -ForegroundColor Yellow
    Unregister-ScheduledTask -TaskName $taskName -TaskPath $taskPath -Confirm:$false
}

# Create the action
$action = New-ScheduledTaskAction -Execute "pwsh.exe" -Argument "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$scriptPath`" -Silent" -WorkingDirectory $scriptRoot

# Create the trigger - at user logon
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME

# Create task settings
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 5) `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1)

# Create the principal (run as current user)
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited

# Register the task
try {
    Register-ScheduledTask `
        -TaskName $taskName `
        -TaskPath $taskPath `
        -Action $action `
        -Trigger $trigger `
        -Settings $settings `
        -Principal $principal `
        -Description "Initializes Bitwarden Secrets Manager secrets at Windows logon. Sets persistent environment variables for CLI applications." `
        | Out-Null

    Write-Host "Task registered successfully!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Task Details:" -ForegroundColor Yellow
    Write-Host "  Name: $taskPath$taskName"
    Write-Host "  Trigger: At logon for user $env:USERNAME"
    Write-Host "  Action: Initialize secrets from BWS Cloud or cache"
    Write-Host "  Script: $scriptPath"
    Write-Host ""

    # Also create a manual run option
    Write-Host "To run manually: " -NoNewline
    Write-Host "schtasks /run /tn `"$taskPath$taskName`"" -ForegroundColor White

    # Test the task
    Write-Host ""
    Write-Host "Testing task execution..." -ForegroundColor Yellow
    Start-ScheduledTask -TaskName $taskName -TaskPath $taskPath

    Start-Sleep -Seconds 3

    $taskInfo = Get-ScheduledTaskInfo -TaskName $taskName -TaskPath $taskPath
    if ($taskInfo.LastTaskResult -eq 0) {
        Write-Host "Test execution: SUCCESS" -ForegroundColor Green
    } else {
        Write-Host "Test execution: Result code $($taskInfo.LastTaskResult)" -ForegroundColor Yellow
        Write-Host "Check logs at: $env:USERPROFILE\.machine\logs\boot-init.log" -ForegroundColor Gray
    }

} catch {
    Write-Error "Failed to register task: $_"
    exit 1
}

Write-Host ""
Write-Host "=== Setup Complete ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "The task will run automatically at each Windows logon." -ForegroundColor White
Write-Host "Secrets will be available as User environment variables." -ForegroundColor White
