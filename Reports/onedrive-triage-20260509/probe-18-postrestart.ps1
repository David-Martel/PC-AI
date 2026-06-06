$ErrorActionPreference = 'Continue'
$out = 'C:\codedev\PC_AI\Reports\onedrive-triage-20260509\18-postrestart.txt'
"=== Post-restart inspection at $(Get-Date -Format o) ===" | Out-File $out

"`n--- DB file mtimes ---" | Out-File $out -Append
$set = "$env:LOCALAPPDATA\Microsoft\OneDrive\settings\Personal"
Get-ChildItem $set -Filter '*.db*' -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object Name, LastWriteTime, @{n='AgeMin';e={[math]::Round((New-TimeSpan -Start $_.LastWriteTime -End (Get-Date)).TotalMinutes,1)}} |
    Format-Table -AutoSize | Out-String | Out-File $out -Append

"`n--- OneDrive.exe + parents ---" | Out-File $out -Append
Get-CimInstance Win32_Process -Filter "Name LIKE 'OneDrive%' OR Name LIKE 'FileSync%' OR Name LIKE 'FileCoAuth%'" -ErrorAction SilentlyContinue |
    Select-Object Name, ProcessId, ParentProcessId, ExecutablePath, CreationDate, CommandLine |
    Format-List | Out-String | Out-File $out -Append

"`n--- HKCU OneDrive root keys (live state) ---" | Out-File $out -Append
Get-ItemProperty 'HKCU:\Software\Microsoft\OneDrive' -ErrorAction SilentlyContinue |
    Select-Object Version, UserCount, IsLoggedIn, ProcessRunningCount, BootTimeOfLastSync, OneAuthAccountId |
    Format-List | Out-String | Out-File $out -Append

"`n--- HKCU Personal keys (live state) ---" | Out-File $out -Append
Get-ItemProperty 'HKCU:\Software\Microsoft\OneDrive\Accounts\Personal' -ErrorAction SilentlyContinue |
    Select-Object UserEmail, IsLoggedIn, LastSignInTime, LastSignInResult, LastAttemptedSignInTime, LastSyncTimeStamp, GetOnlineStatus |
    Format-List | Out-String | Out-File $out -Append

"`n--- Newest 8 log files ---" | Out-File $out -Append
$logRoot = "$env:LOCALAPPDATA\Microsoft\OneDrive\logs\Personal"
Get-ChildItem $logRoot -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 8 Name, LastWriteTime, @{n='SizeKB';e={[math]::Round($_.Length/1KB,1)}} |
    Format-Table -AutoSize | Out-String | Out-File $out -Append

"`n--- TCP from OneDrive PIDs (Established) ---" | Out-File $out -Append
$pids = (Get-Process -Name OneDrive,FileSyncHelper -ErrorAction SilentlyContinue).Id
"OneDrive PIDs: $($pids -join ',')" | Out-File $out -Append
foreach ($p in $pids) {
    "PID ${p}:" | Out-File $out -Append
    Get-NetTCPConnection -OwningProcess $p -State Established -ErrorAction SilentlyContinue |
        Select-Object LocalPort, RemoteAddress, RemotePort, CreationTime |
        Format-Table -AutoSize | Out-String | Out-File $out -Append
}

"`n--- Test file shell sync status ---" | Out-File $out -Append
$testFile = Get-ChildItem "$env:USERPROFILE\OneDrive\_synctest_20260509\synctest-*.txt" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($testFile) {
    try {
        $shell = New-Object -ComObject Shell.Application
        $folder = $shell.Namespace($testFile.DirectoryName)
        $item = $folder.ParseName($testFile.Name)
        for ($i = 280; $i -lt 320; $i++) {
            $hdr = $folder.GetDetailsOf($null, $i)
            if ($hdr -in 'Status','Sync status','Availability status','State','Sharing status') {
                $v = $folder.GetDetailsOf($item, $i)
                if ($v) { "  $hdr (col $i) = $v" | Out-File $out -Append }
            }
        }
        "  File present: $($testFile.FullName) (size=$($testFile.Length))" | Out-File $out -Append
    } catch {
        "  shell err: $($_.Exception.Message)" | Out-File $out -Append
    }
}

"`n--- Live SyncEngine DB tail ---" | Out-File $out -Append
$dbCopy = 'C:\codedev\PC_AI\Reports\onedrive-triage-20260509\SyncEngineDatabase.live2.db'
try {
    Copy-Item "$set\SyncEngineDatabase.db" $dbCopy -Force -ErrorAction Stop
    $maxRow = sqlite3.exe $dbCopy "SELECT MAX(id), datetime(MAX(timestamp),'unixepoch') FROM od_ServiceOperationHistory;"
    "Max op id+ts: $maxRow" | Out-File $out -Append
    sqlite3.exe $dbCopy "SELECT id, datetime(timestamp,'unixepoch'), operationName, resultCode, scenarioName FROM od_ServiceOperationHistory ORDER BY id DESC LIMIT 5;" | Out-File $out -Append
    "Total rows: $(sqlite3.exe $dbCopy 'SELECT COUNT(*) FROM od_ServiceOperationHistory;')" | Out-File $out -Append
} catch {
    "Copy failed: $($_.Exception.Message)" | Out-File $out -Append
}

Write-Host "Wrote $out"
Get-Content $out
