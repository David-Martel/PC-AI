# WSL and Docker Early Startup Configuration

## Summary of Changes

This document describes the migration of WSL and Docker startup scripts from user-level startup to system-level startup via Task Scheduler.

### What Was Done

1. **Found Original Script**: Located `docker-wsl-startup.bat` in the Windows user startup folder
2. **Created System Location**: Created `C:\Scripts\Startup\` folder for system-level scripts
3. **Enhanced Script**: Improved the original script with better logging and error handling
4. **Created Scheduled Task**: Set up "WSL-Docker-Startup" task to run at system boot (before user logon)
5. **Removed from User Startup**: Moved original script out of user startup folder

### Current Configuration

**Script Location**: `C:\Scripts\Startup\docker-wsl-startup.bat`
**Backup Location**: `C:\Scripts\Startup\docker-wsl-startup-original.bat`
**Log File**: `C:\Scripts\Startup\docker-wsl-startup.log`
**Task Name**: `WSL-Docker-Startup`

### Task Details
- **Trigger**: System startup (before user logon)
- **Security**: Runs as SYSTEM account with highest privileges
- **Restart Policy**: Will retry 3 times with 1-minute intervals if it fails

## Management Commands

### View Task Status
```powershell
Get-ScheduledTask -TaskName "WSL-Docker-Startup"
```

### View Task History
```powershell
Get-WinEvent -FilterHashtable @{LogName='Microsoft-Windows-TaskScheduler/Operational'} | Where-Object {$_.Message -like "*WSL-Docker-Startup*"}
```

### Check Log File
```batch
type "C:\Scripts\Startup\docker-wsl-startup.log"
```

### Disable Task (if needed)
```powershell
Disable-ScheduledTask -TaskName "WSL-Docker-Startup"
```

### Enable Task
```powershell
Enable-ScheduledTask -TaskName "WSL-Docker-Startup"
```

### Remove Task (if needed)
```powershell
Unregister-ScheduledTask -TaskName "WSL-Docker-Startup" -Confirm:$false
```

### Test Script Manually
```batch
C:\Scripts\Startup\docker-wsl-startup.bat
```

## Benefits of This Approach

1. **Earlier Startup**: Services start during system boot, before user logon
2. **More Reliable**: Runs with SYSTEM privileges, avoiding permission issues
3. **Better Logging**: Comprehensive logging for troubleshooting
4. **Centralized Management**: All startup scripts in one system location
5. **Retry Logic**: Automatic retry if startup fails

## Troubleshooting

If you experience issues:

1. Check the log file at `C:\Scripts\Startup\docker-wsl-startup.log`
2. Run the script manually to test: `C:\Scripts\Startup\docker-wsl-startup.bat`
3. Verify task status: `Get-ScheduledTask -TaskName "WSL-Docker-Startup"`
4. Check Windows Event Logs for Task Scheduler events

## File Locations

```
C:\Scripts\Startup\
├── docker-wsl-startup.bat          # Enhanced startup script
├── docker-wsl-startup-original.bat # Backup of original script
├── create-startup-task.ps1          # Script to recreate the task
├── docker-wsl-startup.log          # Runtime log file
└── README.md                       # This documentation
```

---
Created: %DATE% %TIME%
Last Updated: %DATE% %TIME%