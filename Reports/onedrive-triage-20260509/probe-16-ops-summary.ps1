$ErrorActionPreference = 'Continue'
$out = 'C:\codedev\PC_AI\Reports\onedrive-triage-20260509\16-ops-summary.txt'
"=== Operation summary at $(Get-Date -Format o) ===" | Out-File $out

$db = 'C:\codedev\PC_AI\Reports\onedrive-triage-20260509\SyncEngineDatabase.copy.db'

"`n--- Operation counts (entire 1320-row history) ---" | Out-File $out -Append
sqlite3.exe $db "SELECT operationName, COUNT(*) AS cnt, MIN(datetime(timestamp,'unixepoch')) AS first_ts, MAX(datetime(timestamp,'unixepoch')) AS last_ts FROM od_ServiceOperationHistory GROUP BY operationName ORDER BY cnt DESC;" 2>&1 | Out-File $out -Append

"`n--- Result code distribution ---" | Out-File $out -Append
sqlite3.exe $db "SELECT resultCode, COUNT(*) AS cnt FROM od_ServiceOperationHistory GROUP BY resultCode ORDER BY cnt DESC;" 2>&1 | Out-File $out -Append

"`n--- Any upload/put/save/commit operations EVER? ---" | Out-File $out -Append
sqlite3.exe $db "SELECT id, datetime(timestamp,'unixepoch') AS t, operationName, resultCode, scenarioName FROM od_ServiceOperationHistory WHERE operationName LIKE '%Upload%' OR operationName LIKE '%Put%' OR operationName LIKE '%Commit%' OR operationName LIKE '%Save%' OR operationName LIKE '%Create%' OR scenarioName LIKE '%Upload%' OR scenarioName LIKE '%Put%' OR scenarioName LIKE '%Commit%' ORDER BY id DESC LIMIT 30;" 2>&1 | Out-File $out -Append

"`n--- Distinct scenarios containing 'WNS' or 'Notification' ---" | Out-File $out -Append
sqlite3.exe $db "SELECT scenarioName, COUNT(*) FROM od_ServiceOperationHistory WHERE scenarioName LIKE '%WNS%' OR scenarioName LIKE '%Notification%' OR scenarioName LIKE '%Polling%' GROUP BY scenarioName ORDER BY 2 DESC;" 2>&1 | Out-File $out -Append

"`n--- WNS-related Application events (last 24h) ---" | Out-File $out -Append
$apps = Get-WinEvent -FilterHashtable @{LogName='Application'; Level=2,3; StartTime=(Get-Date).AddHours(-24)} -ErrorAction SilentlyContinue
$wnsEvents = $apps | Where-Object { ($_.Message) -and ($_.Message -match 'WNS|WpnUser|push notification|notification channel|WPN') }
"Found $($wnsEvents.Count) WNS-related events" | Out-File $out -Append
$wnsEvents | Select-Object -First 15 TimeCreated, Id, ProviderName, @{n='Snippet';e={($_.Message -split [Environment]::NewLine)[0]}} |
    Format-Table -AutoSize -Wrap | Out-String -Width 200 | Out-File $out -Append

"`n--- WpnService and WpnUserService_* details ---" | Out-File $out -Append
Get-CimInstance Win32_Service -Filter "Name LIKE 'Wpn%'" -ErrorAction SilentlyContinue |
    Select-Object Name, State, StartMode, Status, ProcessId |
    Format-Table -AutoSize | Out-String | Out-File $out -Append

"`n--- WNS HKCU registration entries ---" | Out-File $out -Append
$wnsKeys = @(
    'HKCU:\Software\Microsoft\Windows NT\CurrentVersion\PushNotifications',
    'HKCU:\Software\Microsoft\Windows\CurrentVersion\PushNotifications',
    'HKCU:\Software\Microsoft\Windows\CurrentVersion\Notifications\Settings'
)
foreach ($k in $wnsKeys) {
    "--- $k ---" | Out-File $out -Append
    Get-ItemProperty $k -ErrorAction SilentlyContinue | Select-Object * -ExcludeProperty PS* | Format-List | Out-String | Out-File $out -Append
}

"`n--- WNS network endpoints ---" | Out-File $out -Append
foreach ($h in 'client.wns.windows.com','sso.wns.windows.com','login.wns.windows.com') {
    try {
        $r = Resolve-DnsName $h -Type A -ErrorAction Stop | Select-Object -First 3
        $ips = ($r | Where-Object IPAddress).IPAddress -join ', '
        $cnames = ($r | Where-Object NameHost).NameHost -join ', '
        "${h}: ips=$ips ; cnames=$cnames" | Out-File $out -Append
    } catch {
        "${h}: FAIL: $($_.Exception.Message)" | Out-File $out -Append
    }
    $tnc = Test-NetConnection $h -Port 443 -WarningAction SilentlyContinue -InformationLevel Quiet
    "${h}:443 reachable=$tnc" | Out-File $out -Append
}

"`n--- Port 443 connections from OneDrive process to wns.* ---" | Out-File $out -Append
$pids = (Get-Process OneDrive,FileSyncHelper -ErrorAction SilentlyContinue).Id
foreach ($p in $pids) {
    "--- PID $p ---" | Out-File $out -Append
    Get-NetTCPConnection -OwningProcess $p -ErrorAction SilentlyContinue |
        Select-Object LocalAddress, LocalPort, RemoteAddress, RemotePort, State, CreationTime |
        Format-Table -AutoSize | Out-String -Width 200 | Out-File $out -Append
}

Write-Host "Wrote $out"
Get-Content $out
