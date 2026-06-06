$ErrorActionPreference = 'Continue'
$out = 'C:\codedev\PC_AI\Reports\onedrive-triage-20260509\07-dns-deep.txt'
"=== DNS deep probe at $(Get-Date -Format o) ===" | Out-File $out

$hosts = 'oneclient.sfx.ms','onedrive.live.com','login.live.com','d.docs.live.net','my.microsoftpersonalcontent.com','graph.microsoft.com','officeclient.microsoft.com','login.microsoftonline.com','sb.cdn.skype.com'

"`n--- Resolve via system resolver (with cache) ---" | Out-File $out -Append
foreach ($h in $hosts) {
    try {
        $r = Resolve-DnsName -Name $h -Type A -ErrorAction Stop | Select-Object -First 5
        $ips = ($r | Where-Object IPAddress).IPAddress -join ', '
        $cnames = ($r | Where-Object NameHost).NameHost -join ', '
        "${h}: ips=$ips ; cnames=$cnames" | Out-File $out -Append
    } catch {
        "${h}: FAIL: $($_.Exception.Message)" | Out-File $out -Append
    }
}

"`n--- Resolve via 1.1.1.1 (Cloudflare) ---" | Out-File $out -Append
foreach ($h in $hosts) {
    try {
        $r = Resolve-DnsName -Name $h -Type A -Server 1.1.1.1 -ErrorAction Stop | Select-Object -First 5
        $ips = ($r | Where-Object IPAddress).IPAddress -join ', '
        "${h}: $ips" | Out-File $out -Append
    } catch {
        "${h}: FAIL: $($_.Exception.Message)" | Out-File $out -Append
    }
}

"`n--- Resolve via 8.8.8.8 (Google) ---" | Out-File $out -Append
foreach ($h in $hosts) {
    try {
        $r = Resolve-DnsName -Name $h -Type A -Server 8.8.8.8 -ErrorAction Stop | Select-Object -First 5
        $ips = ($r | Where-Object IPAddress).IPAddress -join ', '
        "${h}: $ips" | Out-File $out -Append
    } catch {
        "${h}: FAIL: $($_.Exception.Message)" | Out-File $out -Append
    }
}

"`n--- nslookup (uses default DNS) ---" | Out-File $out -Append
foreach ($h in 'login.live.com','d.docs.live.net','my.microsoftpersonalcontent.com') {
    "--- nslookup $h ---" | Out-File $out -Append
    nslookup $h 2>&1 | Out-File $out -Append
}

"`n--- WARP runtime status ---" | Out-File $out -Append
$warpExe = 'C:\Program Files\Cloudflare\Cloudflare WARP\warp-cli.exe'
if (Test-Path $warpExe) {
    & $warpExe --accept-tos status 2>&1 | Out-File $out -Append
    "`n--- WARP settings ---" | Out-File $out -Append
    & $warpExe --accept-tos settings 2>&1 | Out-File $out -Append
    "`n--- WARP virtual networks (split tunnel) ---" | Out-File $out -Append
    & $warpExe --accept-tos virtual-networks 2>&1 | Out-File $out -Append
    "`n--- WARP DNS stats ---" | Out-File $out -Append
    & $warpExe --accept-tos dns stats 2>&1 | Out-File $out -Append
} else {
    "warp-cli.exe NOT FOUND at expected path" | Out-File $out -Append
}

"`n--- Active TCP connections from OneDrive (re-check with broader filter) ---" | Out-File $out -Append
$pids = (Get-Process OneDrive,FileSyncHelper,FileCoAuth -ErrorAction SilentlyContinue).Id
"OneDrive PIDs: $($pids -join ',')" | Out-File $out -Append
foreach ($p in $pids) {
    "--- PID $p (all states) ---" | Out-File $out -Append
    Get-NetTCPConnection -OwningProcess $p -ErrorAction SilentlyContinue |
        Select-Object LocalAddress, LocalPort, RemoteAddress, RemotePort, State, CreationTime |
        Format-Table -AutoSize | Out-String -Width 200 | Out-File $out -Append
    "--- PID $p UDP ---" | Out-File $out -Append
    Get-NetUDPEndpoint -OwningProcess $p -ErrorAction SilentlyContinue |
        Select-Object LocalAddress, LocalPort |
        Format-Table -AutoSize | Out-String | Out-File $out -Append
}

"`n--- All OneDrive-process-related TCP from netstat ---" | Out-File $out -Append
$pidStrs = $pids -join '|'
netstat -ano 2>&1 | Select-String -Pattern "($pidStrs)" -SimpleMatch:$false | Out-File $out -Append

Write-Host "Wrote $out"
Get-Content $out
