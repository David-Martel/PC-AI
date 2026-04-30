# Install-DevTasks.ps1
# Registers the Scheduled Task for Dev Environment Startup

$TaskName = "DevEnvironmentStartup"
$ScriptRoot = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
$ScriptPath = Join-Path $ScriptRoot "Start-DevEnvironment.ps1"

# Unregister if exists
Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue

# Create Trigger (At LogOn)
$Trigger = New-ScheduledTaskTrigger -AtLogOn

# Create Action (Run pwsh hidden)
$Action = New-ScheduledTaskAction -Execute "pwsh.exe" -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$ScriptPath`" -WindowStyle Hidden" -WorkingDirectory $ScriptRoot

# Create Settings (Allow running on battery, etc.)
$Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Hours 0)

# Register Task
Register-ScheduledTask -TaskName $TaskName -Trigger $Trigger -Action $Action -Settings $Settings -Description "Starts Docker, WSL Bridge, and warms up WSL environment."

Write-Host "Task '$TaskName' registered successfully." -ForegroundColor Green
Write-Host "It will run automatically at next login."
Write-Host "To test now, run: Start-ScheduledTask -TaskName '$TaskName'"
