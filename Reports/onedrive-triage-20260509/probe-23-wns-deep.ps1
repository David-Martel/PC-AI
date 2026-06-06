$ErrorActionPreference = 'Continue'
$out = 'C:\codedev\PC_AI\Reports\onedrive-triage-20260509\23-wns-deep.txt'
"=== WNS / WAM / WebView2 deep probe at $(Get-Date -Format o) ===" | Out-File $out

"`n--- WebView2 Runtime ---" | Out-File $out -Append
$wv = Get-ItemProperty 'HKLM:\SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}' -ErrorAction SilentlyContinue
if ($wv) { "WebView2 ver: $($wv.pv) name: $($wv.name)" | Out-File $out -Append } else { "WebView2 NOT INSTALLED at expected key!" | Out-File $out -Append }
$evrt = Get-AppxPackage -Name 'Microsoft.WebView2*' -ErrorAction SilentlyContinue | Select-Object -First 1
if ($evrt) { "Appx WebView2: $($evrt.Name) $($evrt.Version)" | Out-File $out -Append }
$evdir = 'C:\Program Files (x86)\Microsoft\EdgeWebView\Application'
if (Test-Path $evdir) {
    $vers = Get-ChildItem $evdir -Directory | Sort-Object Name -Descending | Select-Object -First 3 Name, LastWriteTime
    "EdgeWebView versions present:" | Out-File $out -Append
    $vers | Format-Table -AutoSize | Out-String | Out-File $out -Append
}

"`n--- OneDrive Personal AAD/MSA WAM account state ---" | Out-File $out -Append
$wamRoot = "$env:LOCALAPPDATA\Microsoft\OneAuth"
if (Test-Path $wamRoot) {
    "OneAuth dir present at $wamRoot" | Out-File $out -Append
    Get-ChildItem $wamRoot -Recurse -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending |
        Select-Object -First 15 FullName, LastWriteTime, @{n='SizeKB';e={[math]::Round($_.Length/1KB,1)}} |
        Format-Table -AutoSize | Out-String -Width 200 | Out-File $out -Append
}

"`n--- Win32WebViewHost / OneDrive AppLifetime ---" | Out-File $out -Append
Get-AppxPackage -Name '*OneDrive*' -ErrorAction SilentlyContinue | Format-List Name, Version, InstallLocation | Out-String | Out-File $out -Append

"`n--- Windows Push Notification (Wpn*) state ---" | Out-File $out -Append
$wpnKeys = @(
    'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\WpnService',
    'HKCU:\Software\Microsoft\Windows\CurrentVersion\PushNotifications',
    'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\PushNotifications'
)
foreach ($k in $wpnKeys) {
    "--- $k ---" | Out-File $out -Append
    Get-ItemProperty $k -ErrorAction SilentlyContinue | Select-Object * -ExcludeProperty PS* | Format-List | Out-String | Out-File $out -Append
}

"`n--- WnsId / channels ---" | Out-File $out -Append
$wpnDb = "$env:LOCALAPPDATA\Microsoft\Windows\Notifications\wpndatabase.db"
if (Test-Path $wpnDb) {
    "WPN DB: $wpnDb size=$([math]::Round((Get-Item $wpnDb).Length/1KB,1)) KB" | Out-File $out -Append
    $wpnCopy = 'C:\codedev\PC_AI\Reports\onedrive-triage-20260509\wpndatabase.copy.db'
    try {
        Copy-Item $wpnDb $wpnCopy -Force -ErrorAction Stop
        "Tables:" | Out-File $out -Append
        sqlite3.exe $wpnCopy '.tables' 2>&1 | Out-File $out -Append
        "Channel-like records:" | Out-File $out -Append
        sqlite3.exe $wpnCopy "SELECT * FROM sqlite_master WHERE name LIKE '%notif%' OR name LIKE '%chan%';" 2>&1 | Out-File $out -Append
    } catch {
        "WPN DB copy: $($_.Exception.Message)" | Out-File $out -Append
    }
}

"`n--- WNS Pending tracking via PLM (Process Lifetime Manager) ---" | Out-File $out -Append
"OneDrive UWP app users:" | Out-File $out -Append
$plm = Get-ItemProperty 'HKLM:\SOFTWARE\Microsoft\PolicyManager\current\device\Privacy' -ErrorAction SilentlyContinue
"Privacy policy: $(if ($plm) { 'present' } else { 'absent' })" | Out-File $out -Append

"`n--- Per-user WNS connection state via Get-AppxNotification ---" | Out-File $out -Append
try {
    $notifSvc = New-Object -ComObject Windows.UI.Notifications.NotificationData -ErrorAction SilentlyContinue
    "NotificationData COM available: $($notifSvc -ne $null)" | Out-File $out -Append
} catch {}

"`n--- Test WNS reachability over time ---" | Out-File $out -Append
$testHosts = 'client.wns.windows.com','wns.notify.windows.com','client.wns.windows.com','db5p.wns.windows.com'
foreach ($h in $testHosts) {
    $ok = Test-NetConnection -ComputerName $h -Port 443 -WarningAction SilentlyContinue -InformationLevel Quiet
    "$h:443 = $ok" | Out-File $out -Append
}

"`n--- TLS 1.2/1.3 reachability test ---" | Out-File $out -Append
foreach ($h in 'login.live.com','client.wns.windows.com') {
    try {
        $req = [System.Net.HttpWebRequest]::Create("https://$h/")
        $req.Method = 'HEAD'
        $req.Timeout = 5000
        $resp = $req.GetResponse()
        "$h responded: $($resp.StatusCode)" | Out-File $out -Append
        $resp.Close()
    } catch {
        "$h failed: $($_.Exception.Message)" | Out-File $out -Append
    }
}

"`n--- Defender real-time scan stats (large counts can starve OneDrive IO) ---" | Out-File $out -Append
$mp = Get-MpComputerStatus -ErrorAction SilentlyContinue
if ($mp) {
    $mp | Select-Object RealTimeProtectionEnabled, AntivirusSignatureLastUpdated, NISSignatureLastUpdated, OnAccessProtectionEnabled, AntivirusEnabled, IoavProtectionEnabled |
        Format-List | Out-String | Out-File $out -Append
}

Write-Host "Wrote $out"
