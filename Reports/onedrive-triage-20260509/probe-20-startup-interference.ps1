$ErrorActionPreference = 'Continue'
$out = 'C:\codedev\PC_AI\Reports\onedrive-triage-20260509\20-startup-interference.txt'
"=== Startup interference audit at $(Get-Date -Format o) ===" | Out-File $out

"`n--- Active scheduled tasks that touch network/files at startup ---" | Out-File $out -Append
$tasks = Get-ScheduledTask -ErrorAction SilentlyContinue | Where-Object {
    $_.State -eq 'Ready' -and (
        $_.Triggers | Where-Object { $_.CimClass.CimClassName -in 'MSFT_TaskBootTrigger','MSFT_TaskLogonTrigger' }
    )
}
"Boot/logon-triggered tasks: $($tasks.Count)" | Out-File $out -Append

$risky = $tasks | Where-Object {
    $exec = ($_.Actions.Execute -join ';')
    $exec -match '(?i)(rclone|gdrive|google.?drive|dropbox|proton|icloud|dnsproxy|acrylic|RAGRedis|wsl|docker|sync|drive|kbysitime|udm|unifi)'
}
"Risky boot/logon tasks (sync/network/cloud-related): $($risky.Count)" | Out-File $out -Append
$risky | Select-Object TaskName, State, @{n='Trigger';e={($_.Triggers[0].CimClass.CimClassName) -replace 'MSFT_Task',''}}, @{n='Exec';e={($_.Actions.Execute -join ';')}} |
    Format-Table -AutoSize -Wrap | Out-String -Width 250 | Out-File $out -Append

"`n--- Currently running Cloud-sync / DNS / network services ---" | Out-File $out -Append
$svcNames = @('CloudflareWARP','AcrylicDNSProxySvc','OpenVPN*','WireGuard*','NordVPN*','ExpressVPN*','tailscale*','OneDrive*','Dropbox*','GoogleDrive*','GoogleDriveFS','iCloud*','ProtonDrive*','wsl*','vmms','Hyper-V*','Mullvad*','Twingate*','Zerotier*')
foreach ($n in $svcNames) {
    $s = Get-Service -Name $n -ErrorAction SilentlyContinue
    if ($s) { $s | Select-Object Name, Status, StartType | Format-Table -AutoSize | Out-String | Out-File $out -Append }
}

"`n--- Currently running cloud sync processes ---" | Out-File $out -Append
$pp = @('OneDrive','FileSyncHelper','FileCoAuth','GoogleDriveFS','dropbox','rclone','iCloudDrive','iCloudServices','ProtonDrive','tailscaled','ts-postgresql','warp-svc','msedge','chrome','RagRedis','redis-server','docker')
$running = Get-Process -Name $pp -ErrorAction SilentlyContinue
$running | Select-Object Name, Id, @{n='Mem(MB)';e={[math]::Round($_.WorkingSet64/1MB,1)}}, @{n='Threads';e={$_.Threads.Count}}, StartTime, Responding |
    Sort-Object Mem(MB) -Descending | Format-Table -AutoSize | Out-String | Out-File $out -Append

"`n--- HKCU/HKLM Run / RunOnce keys ---" | Out-File $out -Append
$runKeys = @(
    'HKCU:\Software\Microsoft\Windows\CurrentVersion\Run',
    'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Run',
    'HKLM:\SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Run'
)
foreach ($k in $runKeys) {
    "--- $k ---" | Out-File $out -Append
    $items = Get-ItemProperty $k -ErrorAction SilentlyContinue
    if ($items) {
        $items.PSObject.Properties | Where-Object { $_.Name -notmatch '^PS' } |
            ForEach-Object { "  $($_.Name) = $($_.Value)" } | Out-File $out -Append
    }
}

"`n--- Startup folder shortcuts ---" | Out-File $out -Append
$startups = @(
    "$env:APPDATA\Microsoft\Windows\Start Menu\Programs\Startup",
    "$env:ALLUSERSPROFILE\Microsoft\Windows\Start Menu\Programs\Startup"
)
foreach ($s in $startups) {
    "--- $s ---" | Out-File $out -Append
    Get-ChildItem $s -ErrorAction SilentlyContinue | Select-Object Name, Length, LastWriteTime |
        Format-Table -AutoSize | Out-String | Out-File $out -Append
}

"`n--- Active TCP listeners that may interfere with OneDrive (DNS hijacking, proxies) ---" | Out-File $out -Append
Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue |
    Where-Object { $_.LocalPort -in 53,80,443,8080,3128,3142,8888,9999,10800 } |
    ForEach-Object {
        $p = Get-Process -Id $_.OwningProcess -ErrorAction SilentlyContinue
        [PSCustomObject]@{
            LocalAddr=$_.LocalAddress
            Port=$_.LocalPort
            PID=$_.OwningProcess
            Process=$p.Name
            Path=$p.Path
        }
    } | Format-Table -AutoSize -Wrap | Out-String -Width 200 | Out-File $out -Append

"`n--- OneDrive boot.TODO indicators (boot.TODO references) ---" | Out-File $out -Append
"Items from boot.TODO.md still flagged as risk to OneDrive:" | Out-File $out -Append
'- WARP exclude mode could route OneDrive through Cloudflare gateway (org=auricleinc Zero Trust)' | Out-File $out -Append
'- iCloud sync root present without provider running (per 4/30 sync-provider-health)' | Out-File $out -Append
'- UnifiUdmDriveStackStartup task disabled (intentional per ledger, OK)' | Out-File $out -Append
'- VHDs cloud-cache-disk on F: hosts Dropbox + Proton Drive — racing OneDrive at boot' | Out-File $out -Append

Write-Host "Wrote $out"
