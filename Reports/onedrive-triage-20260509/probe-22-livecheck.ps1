$ErrorActionPreference = 'Continue'
$out = 'C:\codedev\PC_AI\Reports\onedrive-triage-20260509\22-livecheck.txt'
"=== Live sync check at $(Get-Date -Format o) ===" | Out-File $out

$db = "$env:LOCALAPPDATA\Microsoft\OneDrive\settings\Personal\SyncEngineDatabase.db"
$dbCopy = 'C:\codedev\PC_AI\Reports\onedrive-triage-20260509\SyncEngineDatabase.live3.db'
Copy-Item $db $dbCopy -Force -ErrorAction SilentlyContinue

"`n--- Op count ---" | Out-File $out -Append
sqlite3.exe $dbCopy 'SELECT COUNT(*) FROM od_ServiceOperationHistory;' | Out-File $out -Append

"`n--- Last 20 ops ---" | Out-File $out -Append
sqlite3.exe $dbCopy "SELECT id, datetime(timestamp,'unixepoch') AS t, operationName, resultCode, scenarioName FROM od_ServiceOperationHistory ORDER BY id DESC LIMIT 20;" | Out-File $out -Append

"`n--- Op count by name (entire history) ---" | Out-File $out -Append
sqlite3.exe $dbCopy "SELECT operationName, COUNT(*) FROM od_ServiceOperationHistory GROUP BY operationName ORDER BY 2 DESC;" | Out-File $out -Append

"`n--- Any new UPLOAD operations today? ---" | Out-File $out -Append
sqlite3.exe $dbCopy "SELECT id, datetime(timestamp,'unixepoch'), operationName, resultCode, scenarioName FROM od_ServiceOperationHistory WHERE datetime(timestamp,'unixepoch') >= '2026-05-09 19:35:00' OR scenarioName LIKE '%Upload%' ORDER BY id DESC LIMIT 25;" | Out-File $out -Append

"`n--- Last NotificationReceived ---" | Out-File $out -Append
sqlite3.exe $dbCopy "SELECT id, datetime(timestamp,'unixepoch'), operationName, resultCode FROM od_ServiceOperationHistory WHERE operationName='NotificationReceived' ORDER BY id DESC LIMIT 5;" | Out-File $out -Append

"`n--- DB mtimes ---" | Out-File $out -Append
Get-ChildItem "$env:LOCALAPPDATA\Microsoft\OneDrive\settings\Personal" -Filter '*.db*' |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 10 Name, LastWriteTime |
    Format-Table -AutoSize | Out-String | Out-File $out -Append

"`n--- TCP from new OneDrive ---" | Out-File $out -Append
$pid4496 = (Get-Process -Id 4496 -ErrorAction SilentlyContinue)
if ($pid4496) {
    Get-NetTCPConnection -OwningProcess 4496 -ErrorAction SilentlyContinue |
        Where-Object { $_.RemoteAddress -ne '::' -and $_.RemoteAddress -ne '0.0.0.0' } |
        Select-Object LocalPort, RemoteAddress, RemotePort, State, CreationTime |
        Format-Table -AutoSize | Out-String -Width 200 | Out-File $out -Append
} else {
    "PID 4496 no longer running" | Out-File $out -Append
}

"`n--- Test file shell sync status ---" | Out-File $out -Append
$tf = "$env:USERPROFILE\OneDrive\_synctest_20260509\synctest-20260509-151440.txt"
if (Test-Path $tf) {
    try {
        $shell = New-Object -ComObject Shell.Application
        $folder = $shell.Namespace((Split-Path $tf))
        $item = $folder.ParseName((Split-Path $tf -Leaf))
        $availStatus = $folder.GetDetailsOf($item, 306)
        $status = $folder.GetDetailsOf($item, 307)
        "Availability status (col 306): $availStatus" | Out-File $out -Append
        "Status (col 307): $($status.Substring(0,[Math]::Min(50,$status.Length)))..." | Out-File $out -Append
        "File size: $((Get-Item $tf).Length) bytes" | Out-File $out -Append
    } catch {
        "shell err: $($_.Exception.Message)" | Out-File $out -Append
    }
}

"`n--- All 4 OneDrive procs current state ---" | Out-File $out -Append
Get-Process OneDrive,FileSyncHelper,FileCoAuth,OneDrive.Sync.Service -ErrorAction SilentlyContinue |
    Select-Object Name, Id, StartTime, @{n='UpMin';e={[math]::Round((New-TimeSpan -Start $_.StartTime -End (Get-Date)).TotalMinutes,1)}}, @{n='Mem(MB)';e={[math]::Round($_.WorkingSet64/1MB,1)}}, Responding, @{n='IO_R(KB)';e={[math]::Round((Get-CimInstance Win32_Process -Filter "ProcessId=$($_.Id)").ReadTransferCount/1KB,0)}}, @{n='IO_W(KB)';e={[math]::Round((Get-CimInstance Win32_Process -Filter "ProcessId=$($_.Id)").WriteTransferCount/1KB,0)}} |
    Format-Table -AutoSize | Out-String -Width 200 | Out-File $out -Append

Write-Host "Wrote $out"
Get-Content $out
