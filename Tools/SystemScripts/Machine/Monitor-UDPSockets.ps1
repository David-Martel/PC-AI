# Monitor-UDPSockets.ps1 — Detect UDP socket leaks before they cause DNS failure
# Runs as scheduled task every 5 minutes
# Logs to ~/.machine/logs/udp-monitor.log
# Threshold: 5000 UDP endpoints (normal is <200)

param(
    [int]$Threshold = 5000,
    [string]$LogDir = "$env:USERPROFILE\.machine\logs"
)

$ErrorActionPreference = 'SilentlyContinue'

# Ensure log dir exists
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir -Force | Out-Null }

$udpCount = (Get-NetUDPEndpoint | Measure-Object).Count
$timestamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'

if ($udpCount -gt $Threshold) {
    $top = Get-NetUDPEndpoint | Group-Object OwningProcess | Sort-Object Count -Descending | Select-Object -First 5
    $msg = "[$timestamp] ALERT: UDP socket count $udpCount exceeds threshold $Threshold`n"
    foreach ($t in $top) {
        $proc = Get-Process -Id $t.Name -ErrorAction SilentlyContinue
        $name = if ($proc) { $proc.ProcessName } else { "unknown" }
        $msg += "  PID $($t.Name) ($name): $($t.Count) sockets`n"
    }
    $msg | Out-File -Append "$LogDir\udp-monitor.log" -Encoding utf8

    # Kill the worst offender if it has >10000 sockets and is dllhost
    $worst = $top | Select-Object -First 1
    $worstProc = Get-Process -Id $worst.Name -ErrorAction SilentlyContinue
    if ($worstProc -and $worstProc.ProcessName -eq 'dllhost' -and $worst.Count -gt 10000) {
        "[$timestamp] AUTO-KILL: dllhost PID $($worst.Name) with $($worst.Count) sockets" |
            Out-File -Append "$LogDir\udp-monitor.log" -Encoding utf8
        Stop-Process -Id $worst.Name -Force
        ipconfig /flushdns | Out-Null
    }
} elseif ((Get-Date).Minute -eq 0) {
    # Hourly health log (on the hour)
    "[$timestamp] OK: $udpCount UDP sockets" |
        Out-File -Append "$LogDir\udp-monitor.log" -Encoding utf8
}
