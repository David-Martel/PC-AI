$ErrorActionPreference = 'Continue'
$out = 'C:\codedev\PC_AI\Reports\onedrive-triage-20260509\25-wpn-restart.txt'
"=== WpnUserService restart + OneDrive re-launch at $(Get-Date -Format o) ===" | Out-File $out

"`n--- PRE: WpnUserService state ---" | Out-File $out -Append
Get-Service -Name 'WpnUserService*' -ErrorAction SilentlyContinue |
    Select-Object Name, Status, StartType |
    Format-Table -AutoSize | Out-String | Out-File $out -Append

"`n--- PRE: OneDrive WNS channel state ---" | Out-File $out -Append
$wpnDb = "$env:LOCALAPPDATA\Microsoft\Windows\Notifications\wpndatabase.db"
$wpnCopy = 'C:\codedev\PC_AI\Reports\onedrive-triage-20260509\wpndatabase.copy.db'
Copy-Item $wpnDb $wpnCopy -Force -ErrorAction SilentlyContinue
$preRow = sqlite3.exe $wpnCopy "SELECT COUNT(*) FROM WNSPushChannel WHERE HandlerId IN (2951,2952,3161);"
"OneDrive (2951/2952/3161) channel count PRE: $preRow" | Out-File $out -Append

"`n=== STOPPING OneDrive processes ===" | Out-File $out -Append
Get-Process OneDrive,FileSyncHelper,FileCoAuth -ErrorAction SilentlyContinue |
    ForEach-Object {
        "  Stopping $($_.Name) PID $($_.Id)" | Out-File $out -Append
        try { Stop-Process -Id $_.Id -Force -ErrorAction Stop } catch { "    err: $($_.Exception.Message)" | Out-File $out -Append }
    }
Start-Sleep -Seconds 5

"`n=== RESTARTING WpnUserService ===" | Out-File $out -Append
$wpnSvcs = Get-Service -Name 'WpnUserService*' -ErrorAction SilentlyContinue
foreach ($s in $wpnSvcs) {
    "Restart $($s.Name)..." | Out-File $out -Append
    try {
        Restart-Service -Name $s.Name -Force -ErrorAction Stop
        "  OK" | Out-File $out -Append
    } catch {
        "  ERR: $($_.Exception.Message)" | Out-File $out -Append
    }
}

"`n--- POST: WpnUserService state ---" | Out-File $out -Append
Get-Service -Name 'WpnUserService*' -ErrorAction SilentlyContinue |
    Select-Object Name, Status, StartType |
    Format-Table -AutoSize | Out-String | Out-File $out -Append

Start-Sleep -Seconds 5

"`n=== STARTING OneDrive via Startup Task ===" | Out-File $out -Append
$mySid = ([System.Security.Principal.WindowsIdentity]::GetCurrent()).User.Value
$taskName = "OneDrive Startup Task-$mySid"
"Task: $taskName" | Out-File $out -Append
try {
    Start-ScheduledTask -TaskName $taskName -ErrorAction Stop
    "  Triggered" | Out-File $out -Append
} catch {
    "  ERR: $($_.Exception.Message)" | Out-File $out -Append
}

# Wait for OneDrive to come up
Start-Sleep -Seconds 15

"`n--- 15s post-launch: OneDrive procs ---" | Out-File $out -Append
Get-Process OneDrive,FileSyncHelper,FileCoAuth,OneDrive.Sync.Service -ErrorAction SilentlyContinue |
    Select-Object Name, Id, StartTime, SessionId |
    Format-Table -AutoSize | Out-String | Out-File $out -Append

"`n--- 15s post-launch: TCP connections ---" | Out-File $out -Append
$pids = (Get-Process OneDrive,FileSyncHelper -ErrorAction SilentlyContinue).Id
foreach ($p in $pids) {
    "PID ${p}:" | Out-File $out -Append
    Get-NetTCPConnection -OwningProcess $p -State Established -ErrorAction SilentlyContinue |
        Select-Object LocalPort, RemoteAddress, RemotePort |
        Format-Table -AutoSize | Out-String | Out-File $out -Append
}

# Wait longer to allow channel registration
Start-Sleep -Seconds 60

"`n--- 75s post-launch: Channel registration check ---" | Out-File $out -Append
Copy-Item $wpnDb $wpnCopy -Force -ErrorAction SilentlyContinue
$postRow = sqlite3.exe $wpnCopy "SELECT COUNT(*) FROM WNSPushChannel WHERE HandlerId IN (2951,2952,3161);"
"OneDrive (2951/2952/3161) channel count POST: $postRow" | Out-File $out -Append

"`n--- All NEW channels registered (since 1 hour ago) ---" | Out-File $out -Append
sqlite3.exe $wpnCopy ".mode column" "SELECT c.HandlerId, h.PrimaryId, datetime(c.CreatedTime/10000-11644473600,'unixepoch') AS CreatedAt FROM WNSPushChannel c LEFT JOIN NotificationHandler h ON c.HandlerId=h.RecordId WHERE c.CreatedTime > $((Get-Date).AddHours(-1).ToFileTimeUtc()) ORDER BY c.CreatedTime DESC;" 2>&1 | Out-File $out -Append

"`n--- All OneDrive-related notification handlers + channel state ---" | Out-File $out -Append
sqlite3.exe $wpnCopy ".mode line" "SELECT h.RecordId, h.PrimaryId, c.ChannelId IS NOT NULL AS HasChannel, datetime(c.CreatedTime/10000-11644473600,'unixepoch') AS ChannelCreatedAt FROM NotificationHandler h LEFT JOIN WNSPushChannel c ON h.RecordId=c.HandlerId WHERE h.PrimaryId LIKE '%OneDrive%' OR h.PrimaryId LIKE '%FileSync%' OR h.PrimaryId LIKE '%SkyDrive%';" 2>&1 | Out-File $out -Append

"`n--- Live SyncEngineDatabase op count ---" | Out-File $out -Append
$db = "$env:LOCALAPPDATA\Microsoft\OneDrive\settings\Personal\SyncEngineDatabase.db"
Copy-Item "$db*" 'C:\codedev\PC_AI\Reports\onedrive-triage-20260509\' -Force -ErrorAction SilentlyContinue
$count = sqlite3.exe 'C:\codedev\PC_AI\Reports\onedrive-triage-20260509\SyncEngineDatabase.db' 'SELECT COUNT(*) FROM od_ServiceOperationHistory;'
"od_ServiceOperationHistory rows: $count" | Out-File $out -Append
"Last 5 ops:" | Out-File $out -Append
sqlite3.exe 'C:\codedev\PC_AI\Reports\onedrive-triage-20260509\SyncEngineDatabase.db' "SELECT id, datetime(timestamp,'unixepoch'), operationName, resultCode, scenarioName FROM od_ServiceOperationHistory ORDER BY id DESC LIMIT 5;" | Out-File $out -Append

"`n--- Test file shell sync status ---" | Out-File $out -Append
$tf = "$env:USERPROFILE\OneDrive\_synctest_20260509\synctest-20260509-151440.txt"
if (Test-Path $tf) {
    try {
        $shell = New-Object -ComObject Shell.Application
        $folder = $shell.Namespace((Split-Path $tf))
        $item = $folder.ParseName((Split-Path $tf -Leaf))
        $availStatus = $folder.GetDetailsOf($item, 306)
        "Availability status: $availStatus" | Out-File $out -Append
    } catch {}
}

Write-Host "Wrote $out"
Get-Content $out
