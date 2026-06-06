$ErrorActionPreference = 'Continue'
$out = 'C:\codedev\PC_AI\Reports\onedrive-triage-20260509\17-restart-and-monitor.txt'
"=== OneDrive restart + monitor at $(Get-Date -Format o) ===" | Out-File $out

$db = "$env:LOCALAPPDATA\Microsoft\OneDrive\settings\Personal\SyncEngineDatabase.db"
$dbCopy = 'C:\codedev\PC_AI\Reports\onedrive-triage-20260509\SyncEngineDatabase.live.db'

function Snapshot-OpsHistory ([string]$label) {
    "`n--- ops snapshot @ $label ($(Get-Date -Format HH:mm:ss)) ---" | Out-File $out -Append
    try {
        Copy-Item $db $dbCopy -Force -ErrorAction Stop
        $cnts = sqlite3.exe $dbCopy "SELECT operationName, COUNT(*) FROM od_ServiceOperationHistory GROUP BY operationName ORDER BY 2 DESC;"
        $cnts | Out-File $out -Append
        $maxId = sqlite3.exe $dbCopy "SELECT MAX(id), datetime(MAX(timestamp),'unixepoch') FROM od_ServiceOperationHistory;"
        "max id+ts: $maxId" | Out-File $out -Append
        "last 5:" | Out-File $out -Append
        sqlite3.exe $dbCopy "SELECT id, datetime(timestamp,'unixepoch'), operationName, resultCode, scenarioName FROM od_ServiceOperationHistory ORDER BY id DESC LIMIT 5;" | Out-File $out -Append
    } catch { "  Copy failed: $($_.Exception.Message)" | Out-File $out -Append }
}

# PRE
"`n=== PRE-RESTART ===" | Out-File $out -Append
Snapshot-OpsHistory 'pre'
"`nProcesses pre-stop:" | Out-File $out -Append
Get-Process OneDrive,FileSyncHelper,FileCoAuth -ErrorAction SilentlyContinue |
    Select-Object Name, Id, StartTime |
    Format-Table -AutoSize | Out-String | Out-File $out -Append

# STOP
"`n=== STOPPING OneDrive ecosystem at $(Get-Date -Format o) ===" | Out-File $out -Append
Get-Process OneDrive,FileSyncHelper,FileCoAuth -ErrorAction SilentlyContinue |
    ForEach-Object {
        "  Stopping $($_.Name) PID $($_.Id)..." | Out-File $out -Append
        try { Stop-Process -Id $_.Id -Force -ErrorAction Stop } catch { "    err: $($_.Exception.Message)" | Out-File $out -Append }
    }
Start-Sleep -Seconds 5
"`nPost-stop process check:" | Out-File $out -Append
$still = Get-Process OneDrive,FileSyncHelper,FileCoAuth -ErrorAction SilentlyContinue
if ($still) {
    $still | Format-Table Name, Id | Out-String | Out-File $out -Append
    "  WARN: some still running, waiting another 5s" | Out-File $out -Append
    Start-Sleep -Seconds 5
} else {
    "  All OneDrive processes exited cleanly" | Out-File $out -Append
}

# START
"`n=== STARTING OneDrive at $(Get-Date -Format o) ===" | Out-File $out -Append
$exe = 'C:\Program Files\Microsoft OneDrive\OneDrive.exe'
"  Launching $exe /background" | Out-File $out -Append
Start-Process $exe -ArgumentList '/background'
Start-Sleep -Seconds 10

"`nProcess check (10s post-start):" | Out-File $out -Append
Get-Process OneDrive,FileSyncHelper,FileCoAuth -ErrorAction SilentlyContinue |
    Select-Object Name, Id, StartTime, Responding |
    Format-Table -AutoSize | Out-String | Out-File $out -Append

# MONITOR @ 30s, 60s, 2min, 5min
$startTime = Get-Date
foreach ($wait in 30, 30, 60, 180) {
    Start-Sleep -Seconds $wait
    $elapsed = [math]::Round((Get-Date - $startTime).TotalSeconds)
    Snapshot-OpsHistory "T+${elapsed}s post-start"
}

"`n=== Post-restart TCP from OneDrive (5 min after start) ===" | Out-File $out -Append
$pids = (Get-Process OneDrive,FileSyncHelper -ErrorAction SilentlyContinue).Id
foreach ($p in $pids) {
    "PID ${p}:" | Out-File $out -Append
    Get-NetTCPConnection -OwningProcess $p -State Established -ErrorAction SilentlyContinue |
        Select-Object LocalPort, RemoteAddress, RemotePort, CreationTime |
        Format-Table -AutoSize | Out-String | Out-File $out -Append
}

"`n=== Post-restart synctest file status ===" | Out-File $out -Append
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
    } catch { "shell err: $($_.Exception.Message)" | Out-File $out -Append }
}

Write-Host "Wrote $out"
Get-Content $out
