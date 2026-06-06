# Touchpad glitch evidence collector - read-only
[CmdletBinding()]
param(
    [string]$OutputDir = "C:\codedev\PC_AI\Reports\touchpad-glitch-investigation-20260502"
)

$ErrorActionPreference = 'Continue'
if (-not (Test-Path $OutputDir)) { New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null }
$ts = Get-Date -Format "yyyy-MM-ddTHH:mm:ssK"

#region 1. HID Touchpad devices
"# Touchpad investigation - generated $ts" | Out-File "$OutputDir\01-devices.txt" -Encoding utf8
"`r`n## 1. HID Touchpad / I2C / TrackPoint devices" | Out-File "$OutputDir\01-devices.txt" -Append -Encoding utf8
$tpAll = Get-PnpDevice -PresentOnly | Where-Object {
    $_.InstanceId -match 'SNSL|ELAN|HID\\|I2C|VID_06CB|VID_04F3|VID_044E' -and
    ($_.FriendlyName -match 'touch|track|HID|I2C|Serial IO|mouse|pad' -or $_.Class -eq 'HIDClass' -or $_.Class -eq 'Mouse')
}
$tpAll | Sort-Object FriendlyName | Select-Object Status, Class, FriendlyName, InstanceId | Format-Table -AutoSize | Out-File "$OutputDir\01-devices.txt" -Append -Encoding utf8
#endregion

#region 2. Driver versions for the touchpad/HID/I2C stack
"# Driver versions for touchpad/HID/I2C stack" | Out-File "$OutputDir\02-driver-versions.txt" -Encoding utf8
$keys = @(
    'DEVPKEY_Device_DriverVersion',
    'DEVPKEY_Device_DriverDate',
    'DEVPKEY_Device_DriverProvider',
    'DEVPKEY_Device_DriverInfPath',
    'DEVPKEY_Device_DriverDesc'
)
$tpStack = Get-PnpDevice -PresentOnly | Where-Object {
    $_.InstanceId -match 'SNSL|ELAN|HID|I2C|VID_06CB|VID_04F3|VID_044E' -or
    $_.Service -match 'iaLPSS2i|hidi2c|HIDClass|mouclass|mouhid|hidusb|SynRpcServer'
}
foreach ($d in $tpStack) {
    "" | Out-File "$OutputDir\02-driver-versions.txt" -Append -Encoding utf8
    "--- $($d.FriendlyName) ---" | Out-File "$OutputDir\02-driver-versions.txt" -Append -Encoding utf8
    "InstanceId: $($d.InstanceId)" | Out-File "$OutputDir\02-driver-versions.txt" -Append -Encoding utf8
    "Service: $($d.Service)  Status: $($d.Status)" | Out-File "$OutputDir\02-driver-versions.txt" -Append -Encoding utf8
    foreach ($k in $keys) {
        $p = Get-PnpDeviceProperty -InstanceId $d.InstanceId -KeyName $k -ErrorAction SilentlyContinue
        if ($p -and $p.Data) {
            "  $k = $($p.Data)" | Out-File "$OutputDir\02-driver-versions.txt" -Append -Encoding utf8
        }
    }
}
#endregion

#region 3. pnputil enumerated drivers (full + filtered)
"# pnputil /enum-drivers full output" | Out-File "$OutputDir\03-pnputil-full.txt" -Encoding utf8
$pnp = pnputil /enum-drivers
$pnp | Out-File "$OutputDir\03-pnputil-full.txt" -Append -Encoding utf8

$relevant = $pnp | Out-String
$blocks = $relevant -split "(?ms)^Published Name\s*:" | Where-Object { $_ -match '\S' }
"# Touchpad/HID/I2C driver blocks" | Out-File "$OutputDir\03-pnputil-touchpad.txt" -Encoding utf8
foreach ($b in $blocks) {
    if ($b -match 'sensel|synaptic|elan|hidi2c|iaLPSS2|mouclass|mouhid|trackpoint|touchpad|i2chid|Lenovo') {
        ("Published Name :" + $b) | Out-File "$OutputDir\03-pnputil-touchpad.txt" -Append -Encoding utf8
    }
}
#endregion

#region 4. setupapi.dev.log recent device install events
"# setupapi.dev.log device install events (recent)" | Out-File "$OutputDir\04-setupapi-recent.txt" -Encoding utf8
$setupapi = "$env:WINDIR\INF\setupapi.dev.log"
if (Test-Path $setupapi) {
    "Path: $setupapi" | Out-File "$OutputDir\04-setupapi-recent.txt" -Append -Encoding utf8
    $matches = Select-String -Path $setupapi -Pattern '^>>>\s+\[Device Install|^>>>\s+Section start' -ErrorAction SilentlyContinue
    if ($matches) {
        $matches | Select-Object -Last 80 | ForEach-Object { "{0}:{1}: {2}" -f $_.Filename, $_.LineNumber, $_.Line } | Out-File "$OutputDir\04-setupapi-recent.txt" -Append -Encoding utf8
    } else {
        "No device install markers found" | Out-File "$OutputDir\04-setupapi-recent.txt" -Append -Encoding utf8
    }
}
#endregion

#region 5. Recent system events
"# Recent system events for HID/I2C/PnP/WUDFHost/Power (last 7 days)" | Out-File "$OutputDir\05-events.txt" -Encoding utf8
$providers = @(
    'Microsoft-Windows-Kernel-PnP',
    'Microsoft-Windows-Kernel-Power',
    'Microsoft-Windows-WUDFHost',
    'Microsoft-Windows-UserModeDriverFramework',
    'Microsoft-Windows-DeviceSetupManager',
    'Microsoft-Windows-DriverFrameworks-UserMode',
    'Microsoft-Windows-Power-Troubleshooter'
)
foreach ($p in $providers) {
    try {
        $evs = Get-WinEvent -FilterHashtable @{
            LogName = 'System'
            Level = 1,2,3
            StartTime = (Get-Date).AddDays(-7)
            ProviderName = $p
        } -ErrorAction Stop -MaxEvents 30
        "" | Out-File "$OutputDir\05-events.txt" -Append -Encoding utf8
        "=== Provider: $p (count=$($evs.Count)) ===" | Out-File "$OutputDir\05-events.txt" -Append -Encoding utf8
        foreach ($e in $evs) {
            $msg = ($e.Message -replace "`r?`n", ' | ' -replace '\s+', ' ')
            if ($msg.Length -gt 220) { $msg = $msg.Substring(0,220) }
            "{0}  Id={1}  Lvl={2}  {3}" -f $e.TimeCreated, $e.Id, $e.LevelDisplayName, $msg | Out-File "$OutputDir\05-events.txt" -Append -Encoding utf8
        }
    } catch {
        "" | Out-File "$OutputDir\05-events.txt" -Append -Encoding utf8
        "=== Provider: $p (no events or query failed) ===" | Out-File "$OutputDir\05-events.txt" -Append -Encoding utf8
    }
}
#endregion

#region 6. Recent hotfixes + Windows Update events
"# Recent Windows hotfixes (last 30 days)" | Out-File "$OutputDir\06-hotfixes.txt" -Encoding utf8
Get-HotFix | Where-Object { $_.InstalledOn -gt (Get-Date).AddDays(-30) } | Sort-Object InstalledOn -Descending | Select-Object HotFixID, Description, InstalledOn, InstalledBy | Format-Table -AutoSize | Out-File "$OutputDir\06-hotfixes.txt" -Append -Encoding utf8

"`r`n# WindowsUpdateClient events (last 14 days)" | Out-File "$OutputDir\06-hotfixes.txt" -Append -Encoding utf8
try {
    Get-WinEvent -FilterHashtable @{LogName='System'; ProviderName='Microsoft-Windows-WindowsUpdateClient'; StartTime=(Get-Date).AddDays(-14)} -MaxEvents 25 -ErrorAction Stop |
        Select-Object TimeCreated, Id, LevelDisplayName, @{N='Message';E={ $m = $_.Message; if ($m.Length -gt 180) { $m.Substring(0,180) } else { $m } }} |
        Format-Table -AutoSize | Out-File "$OutputDir\06-hotfixes.txt" -Append -Encoding utf8
} catch {
    "WindowsUpdateClient query failed: $_" | Out-File "$OutputDir\06-hotfixes.txt" -Append -Encoding utf8
}
#endregion

#region 7. Process Lasso state
"# Process Lasso state right now" | Out-File "$OutputDir\08-processlasso-current.txt" -Encoding utf8
Get-Process -Name 'ProcessGovernor','ProcessLasso','bitsumsessionagent' -ErrorAction SilentlyContinue |
    Select-Object Name, Id, StartTime, Responding, BasePriority,
        @{N='CPU_seconds';E={[math]::Round($_.CPU,2)}},
        @{N='WorkingSetMB';E={[math]::Round($_.WorkingSet64/1MB,1)}}, Path |
    Format-List | Out-File "$OutputDir\08-processlasso-current.txt" -Append -Encoding utf8
#endregion

#region 8. Top processes
"# Top processes right now" | Out-File "$OutputDir\09-top-processes.txt" -Encoding utf8
"`r`n## Top 25 by CPU time" | Out-File "$OutputDir\09-top-processes.txt" -Append -Encoding utf8
Get-Process | Sort-Object CPU -Descending | Select-Object -First 25 Name, Id,
    @{N='CPU';E={[math]::Round($_.CPU,1)}},
    @{N='RAM_MB';E={[math]::Round($_.WorkingSet64/1MB,1)}},
    BasePriority, Responding, StartTime |
    Format-Table -AutoSize | Out-File "$OutputDir\09-top-processes.txt" -Append -Encoding utf8

"`r`n## Watched processes - input/shell/sync paths" | Out-File "$OutputDir\09-top-processes.txt" -Append -Encoding utf8
$watched = 'dwm','explorer','sihost','StartMenuExperienceHost','ShellExperienceHost','TextInputHost','ctfmon','SynRpcServer','Lenovo.Modern.ImController','LenovoVantageService','OneDrive','OneDrive.Sync.Service','FileSyncHelper','GoogleDriveFS','rclone','Docker Desktop','com.docker.backend','redis-server','node','sccache','cargo','rustc','NVIDIA Broadcast','PresentMonService','NVDisplay.Container','WUDFHost'
$watched | ForEach-Object {
    Get-Process -Name $_ -ErrorAction SilentlyContinue |
        Select-Object Name, Id, BasePriority, Responding,
            @{N='CPU';E={[math]::Round($_.CPU,1)}},
            @{N='RAM_MB';E={[math]::Round($_.WorkingSet64/1MB,1)}}
} | Format-Table -AutoSize | Out-File "$OutputDir\09-top-processes.txt" -Append -Encoding utf8
#endregion

#region 9. OneDrive IO right now
"# OneDrive / sync process IO right now" | Out-File "$OutputDir\10-onedrive-io-now.txt" -Encoding utf8
$syncProcs = @('OneDrive.exe','FileSyncHelper.exe','OneDrive.Sync.Service.exe','FileCoAuth.exe','GoogleDriveFS.exe','Dropbox.exe')
$filter = ($syncProcs | ForEach-Object { "Name='$_'" }) -join ' OR '
Get-CimInstance Win32_Process -Filter $filter -ErrorAction SilentlyContinue |
    Select-Object Name, ProcessId, WorkingSetSize, ReadOperationCount, WriteOperationCount, ReadTransferCount, WriteTransferCount |
    Format-List | Out-File "$OutputDir\10-onedrive-io-now.txt" -Append -Encoding utf8
#endregion

#region 10. Power scheme + power events
"# Power scheme + recent power events" | Out-File "$OutputDir\11-power-mgmt.txt" -Encoding utf8
powercfg /getactivescheme | Out-File "$OutputDir\11-power-mgmt.txt" -Append -Encoding utf8

"`r`n## Recent Kernel-Power events (last 3 days)" | Out-File "$OutputDir\11-power-mgmt.txt" -Append -Encoding utf8
try {
    Get-WinEvent -FilterHashtable @{LogName='System'; ProviderName='Microsoft-Windows-Kernel-Power'; StartTime=(Get-Date).AddDays(-3)} -MaxEvents 15 -ErrorAction Stop |
        Select-Object TimeCreated, Id, LevelDisplayName, @{N='Msg';E={ $m=$_.Message; if($m.Length -gt 150){$m.Substring(0,150)}else{$m}}} |
        Format-Table -AutoSize | Out-File "$OutputDir\11-power-mgmt.txt" -Append -Encoding utf8
} catch {}

# HID device power capabilities
"`r`n## HID device power capabilities" | Out-File "$OutputDir\11-power-mgmt.txt" -Append -Encoding utf8
foreach ($d in $tpStack) {
    "" | Out-File "$OutputDir\11-power-mgmt.txt" -Append -Encoding utf8
    "--- $($d.FriendlyName) ---" | Out-File "$OutputDir\11-power-mgmt.txt" -Append -Encoding utf8
    "InstanceId: $($d.InstanceId)" | Out-File "$OutputDir\11-power-mgmt.txt" -Append -Encoding utf8
    $caps = Get-PnpDeviceProperty -InstanceId $d.InstanceId -KeyName 'DEVPKEY_Device_PowerData' -ErrorAction SilentlyContinue
    if ($caps) {
        "PowerData length: $($caps.Data.Length) bytes (raw - decoded via CM_GetDevNodeStatus)" | Out-File "$OutputDir\11-power-mgmt.txt" -Append -Encoding utf8
    }
}
#endregion

#region 11. Lenovo Vantage
"# Lenovo Vantage" | Out-File "$OutputDir\12-lenovo-vantage.txt" -Encoding utf8
Get-AppxPackage -Name '*Vantage*' -ErrorAction SilentlyContinue |
    Select-Object Name, Version, InstallLocation, PackageFullName |
    Format-List | Out-File "$OutputDir\12-lenovo-vantage.txt" -Append -Encoding utf8
Get-Service -Name 'LenovoVantageService','ImControllerService' -ErrorAction SilentlyContinue |
    Select-Object Name, Status, StartType, DisplayName |
    Format-Table -AutoSize | Out-File "$OutputDir\12-lenovo-vantage.txt" -Append -Encoding utf8

$lenovoLogs = "$env:WINDIR\Lenovo\ImController\PluginHost\Logs"
if (Test-Path $lenovoLogs) {
    "`r`n## Lenovo IM Controller log files (recent)" | Out-File "$OutputDir\12-lenovo-vantage.txt" -Append -Encoding utf8
    Get-ChildItem $lenovoLogs -Recurse -File -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 20 FullName, Length, LastWriteTime |
        Format-Table -AutoSize | Out-File "$OutputDir\12-lenovo-vantage.txt" -Append -Encoding utf8
}
#endregion

#region 12. Process Lasso log tail
"# Process Lasso log tail" | Out-File "$OutputDir\13-pl-log-tail.txt" -Encoding utf8
$plPaths = @(
    "$env:LOCALAPPDATA\ProcessLasso\logs",
    "C:\ProgramData\ProcessLasso\logs",
    "C:\ProgramData\ProcessLasso"
)
foreach ($p in $plPaths) {
    if (Test-Path $p) {
        "Found: $p" | Out-File "$OutputDir\13-pl-log-tail.txt" -Append -Encoding utf8
        $latest = Get-ChildItem $p -File -Recurse -Filter '*.log' -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTime -Descending | Select-Object -First 3
        foreach ($f in $latest) {
            "" | Out-File "$OutputDir\13-pl-log-tail.txt" -Append -Encoding utf8
            "=== $($f.FullName) (LastWrite: $($f.LastWriteTime)) ===" | Out-File "$OutputDir\13-pl-log-tail.txt" -Append -Encoding utf8
            Get-Content $f.FullName -Tail 60 -ErrorAction SilentlyContinue | Out-File "$OutputDir\13-pl-log-tail.txt" -Append -Encoding utf8
        }
    }
}
#endregion

Write-Host "Evidence collection complete -> $OutputDir"
Get-ChildItem $OutputDir -File | Sort-Object Name | Select-Object Name, Length, LastWriteTime | Format-Table -AutoSize
