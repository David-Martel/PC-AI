# PowerShell script to create WSL/Docker startup task
# This creates a scheduled task that runs at system startup (before user logon)

$TaskName = "WSL-Docker-Startup"
$TaskDescription = "Starts WSL and Docker Desktop services at system startup (before user logon)"
$ScriptPath = "C:\Scripts\Startup\docker-wsl-startup.bat"

# Check if task already exists and remove it
$ExistingTask = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($ExistingTask) {
    Write-Host "Removing existing task: $TaskName"
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

# Create the scheduled task action
$Action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c `"$ScriptPath`""

# Create the trigger (at startup)
$Trigger = New-ScheduledTaskTrigger -AtStartup

# Create the principal (run as SYSTEM with highest privileges)
$Principal = New-ScheduledTaskPrincipal -UserId "SYSTEM" -LogonType ServiceAccount -RunLevel Highest

# Create task settings
$Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)

# Register the scheduled task
Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Principal $Principal -Settings $Settings -Description $TaskDescription

Write-Host "Task '$TaskName' created successfully!"
Write-Host "The task will run at system startup with SYSTEM privileges."
Write-Host "Log file location: C:\Scripts\Startup\docker-wsl-startup.log"