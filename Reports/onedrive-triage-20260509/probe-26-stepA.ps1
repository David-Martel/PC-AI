$ErrorActionPreference = 'Continue'
$out = 'C:\codedev\PC_AI\Reports\onedrive-triage-20260509\26-stepA.txt'
"=== Step A: Clear OneAuth WAM cache + restart at $(Get-Date -Format o) ===" | Out-File $out

# Capture pre-state
$wamRoot = "$env:LOCALAPPDATA\Microsoft\OneAuth"
"`n--- PRE: OneAuth files for cid 8d77a48e98fdb6dc ---" | Out-File $out -Append
$preFiles = Get-ChildItem $wamRoot -Recurse -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -match '8d77a48e98fdb6dc' }
$preFiles | Select-Object FullName, Length, LastWriteTime |
    Format-Table -AutoSize | Out-String -Width 200 | Out-File $out -Append
"PRE total: $($preFiles.Count) files" | Out-File $out -Append

# Backup the files we are about to delete
$backupDir = 'C:\codedev\PC_AI\Reports\onedrive-triage-20260509\oneauth-backup-stepA'
$null = New-Item -ItemType Directory -Path $backupDir -Force
foreach ($f in $preFiles) {
    Copy-Item $f.FullName $backupDir -Force -ErrorAction SilentlyContinue
}
"Backed up to: $backupDir" | Out-File $out -Append

# Stop OneDrive
"`n=== STOPPING OneDrive ===" | Out-File $out -Append
Get-Process OneDrive,FileSyncHelper,FileCoAuth -ErrorAction SilentlyContinue |
    ForEach-Object {
        "  Stopping $($_.Name) PID $($_.Id)" | Out-File $out -Append
        Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
    }
Start-Sleep -Seconds 5

# Delete the WAM cache files for this cid
"`n=== Deleting WAM cache files ===" | Out-File $out -Append
foreach ($f in $preFiles) {
    try {
        Remove-Item $f.FullName -Force -ErrorAction Stop
        "  DELETED: $($f.FullName)" | Out-File $out -Append
    } catch {
        "  FAILED: $($f.FullName) - $($_.Exception.Message)" | Out-File $out -Append
    }
}

"`n--- POST-DELETE: OneAuth files for cid 8d77a48e98fdb6dc ---" | Out-File $out -Append
$postFiles = Get-ChildItem $wamRoot -Recurse -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -match '8d77a48e98fdb6dc' }
"POST-DELETE count: $($postFiles.Count) files" | Out-File $out -Append

# Restart via Startup Task (proper session)
"`n=== STARTING OneDrive via Startup Task ===" | Out-File $out -Append
$mySid = ([System.Security.Principal.WindowsIdentity]::GetCurrent()).User.Value
$taskName = "OneDrive Startup Task-$mySid"
try {
    Start-ScheduledTask -TaskName $taskName -ErrorAction Stop
    "Triggered: $taskName" | Out-File $out -Append
} catch {
    "ERR: $($_.Exception.Message)" | Out-File $out -Append
}

# Wait 90s for sign-in + channel registration
Start-Sleep -Seconds 90

"`n=== POST: 90s after launch ===" | Out-File $out -Append
"OneDrive procs:" | Out-File $out -Append
Get-Process OneDrive,FileSyncHelper,FileCoAuth -ErrorAction SilentlyContinue |
    Select-Object Name, Id, StartTime, SessionId, Responding, @{n='Mem(MB)';e={[math]::Round($_.WorkingSet64/1MB,1)}} |
    Format-Table -AutoSize | Out-String | Out-File $out -Append

"`n--- POST: New OneAuth files for cid 8d77a48e98fdb6dc (should be regenerated) ---" | Out-File $out -Append
Get-ChildItem $wamRoot -Recurse -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -match '8d77a48e98fdb6dc' } |
    Select-Object FullName, Length, LastWriteTime |
    Format-Table -AutoSize | Out-String -Width 200 | Out-File $out -Append

"`n--- POST: WNS channel registration check ---" | Out-File $out -Append
$wpnDb = "$env:LOCALAPPDATA\Microsoft\Windows\Notifications\wpndatabase.db"
$wpnCopy = 'C:\codedev\PC_AI\Reports\onedrive-triage-20260509\wpndatabase.copy.db'
Copy-Item $wpnDb $wpnCopy -Force -ErrorAction SilentlyContinue
sqlite3.exe $wpnCopy ".mode line" "SELECT h.RecordId, h.PrimaryId, c.ChannelId IS NOT NULL AS HasChannel, datetime(c.CreatedTime/10000-11644473600,'unixepoch') AS ChannelCreatedAt, substr(c.Uri,1,80) AS UriPreview FROM NotificationHandler h LEFT JOIN WNSPushChannel c ON h.RecordId=c.HandlerId WHERE h.PrimaryId LIKE '%OneDrive%' OR h.PrimaryId LIKE '%FileSync%' OR h.PrimaryId LIKE '%SkyDrive%';" 2>&1 | Out-File $out -Append

"`n--- POST: SyncEngineDatabase op count + last 10 ops ---" | Out-File $out -Append
$db = "$env:LOCALAPPDATA\Microsoft\OneDrive\settings\Personal\SyncEngineDatabase.db"
Copy-Item "${db}*" 'C:\codedev\PC_AI\Reports\onedrive-triage-20260509\' -Force -ErrorAction SilentlyContinue
$count = sqlite3.exe 'C:\codedev\PC_AI\Reports\onedrive-triage-20260509\SyncEngineDatabase.db' 'SELECT COUNT(*) FROM od_ServiceOperationHistory;'
"od_ServiceOperationHistory rows: $count" | Out-File $out -Append
sqlite3.exe 'C:\codedev\PC_AI\Reports\onedrive-triage-20260509\SyncEngineDatabase.db' "SELECT id, datetime(timestamp,'unixepoch'), operationName, resultCode, scenarioName FROM od_ServiceOperationHistory ORDER BY id DESC LIMIT 10;" 2>&1 | Out-File $out -Append

"`n--- POST: Test file shell sync status ---" | Out-File $out -Append
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
