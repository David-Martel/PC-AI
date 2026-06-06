$ErrorActionPreference = 'Continue'
$out = 'C:\codedev\PC_AI\Reports\onedrive-triage-20260509\03-network-probes.txt'
"=== Network probes at $(Get-Date -Format o) ===" | Out-File $out

"`n--- DNS resolution ---" | Out-File $out -Append
$hosts = 'oneclient.sfx.ms','onedrive.live.com','login.live.com','d.docs.live.net','my.microsoftpersonalcontent.com','graph.microsoft.com','officeclient.microsoft.com'
foreach ($h in $hosts) {
    try {
        $r = Resolve-DnsName -Name $h -Type A -ErrorAction Stop -DnsOnly | Select-Object -First 3
        "${h}: $($r.IPAddress -join ', ')" | Out-File $out -Append
    } catch {
        "${h}: RESOLVE_FAIL: $($_.Exception.Message)" | Out-File $out -Append
    }
}

"`n--- TCP 443 reachability ---" | Out-File $out -Append
foreach ($h in 'oneclient.sfx.ms','onedrive.live.com','login.live.com','graph.microsoft.com') {
    $tnc = Test-NetConnection -ComputerName $h -Port 443 -WarningAction SilentlyContinue -InformationLevel Quiet
    "${h}:443 -> Reachable=$tnc" | Out-File $out -Append
}

"`n--- DNS client config ---" | Out-File $out -Append
Get-DnsClientServerAddress -AddressFamily IPv4 |
    Where-Object ServerAddresses |
    Select-Object InterfaceAlias, ServerAddresses |
    Format-Table -AutoSize | Out-String | Out-File $out -Append

"`n--- Proxy / WARP / Acrylic presence ---" | Out-File $out -Append
$warp = Get-Service -Name 'CloudflareWARP' -ErrorAction SilentlyContinue
$acrylic = Get-Service -Name 'AcrylicDNSProxySvc' -ErrorAction SilentlyContinue
"WARP service: $(if ($warp) { $warp.Status } else { 'NOT_INSTALLED' })" | Out-File $out -Append
"Acrylic service: $(if ($acrylic) { $acrylic.Status } else { 'NOT_INSTALLED' })" | Out-File $out -Append
"`nnetsh winhttp show proxy:" | Out-File $out -Append
netsh winhttp show proxy 2>&1 | Out-File $out -Append
"`nIE/WinINET proxy:" | Out-File $out -Append
$wi = Get-ItemProperty 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Internet Settings' -ErrorAction SilentlyContinue
"ProxyEnable=$($wi.ProxyEnable)  ProxyServer=$($wi.ProxyServer)  AutoConfigURL=$($wi.AutoConfigURL)" | Out-File $out -Append

"`n--- Active TCP from OneDrive PIDs ---" | Out-File $out -Append
$pids = (Get-Process OneDrive,FileSyncHelper -ErrorAction SilentlyContinue).Id
"OneDrive PIDs: $($pids -join ',')" | Out-File $out -Append
foreach ($p in $pids) {
    "--- PID $p ---" | Out-File $out -Append
    Get-NetTCPConnection -OwningProcess $p -ErrorAction SilentlyContinue |
        Where-Object { $_.RemoteAddress -ne '::' -and $_.RemoteAddress -ne '0.0.0.0' } |
        Select-Object -First 20 LocalAddress, LocalPort, RemoteAddress, RemotePort, State |
        Format-Table -AutoSize | Out-String | Out-File $out -Append
}

Write-Host "Wrote $out"
Get-Content $out
