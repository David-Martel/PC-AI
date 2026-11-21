<#
.SYNOPSIS
  Detects UDP socket leaks before they cause DNS failure. Runs as a scheduled task every 5 minutes.

.DESCRIPTION
  Counts UDP endpoints. Above -Threshold (normal is under 200) it logs the five worst owners, and
  force-stops a dllhost process holding more than 10000 sockets, then flushes the DNS cache.
  Otherwise it writes one "OK" heartbeat line per -HeartbeatMinutes so a reader of the log can tell
  the monitor is alive.

  Log: <LogDir>\udp-monitor.log (default: the real user profile's .machine\logs).

  Why the heartbeat uses a stamp file: the old check was "the minute is 0". A task that repeats every
  5 minutes from a start time such as 06:14:34 only ever runs at minutes 4, 9, 14 and so on, so it
  never hit minute 0 and the OK line was never written (dtm-work, 2026-11-21: 0 OK lines in the log).
  The stamp file makes the heartbeat independent of the trigger's phase.

  Exit codes: 0 normal (including an alert), 1 when the UDP endpoint list could not be read. The
  previous script ran with ErrorActionPreference SilentlyContinue and counted 0 endpoints on failure,
  which looks exactly like a healthy host.

  Test seams: dot-sourcing the file only defines the functions; the monitor runs when it is executed.
  Tests: Tests/Unit/Monitor-UDPSockets.Tests.ps1. Task registry entry: docs/scheduled-tasks/TASK_REGISTRY.md.

.PARAMETER Threshold
  Endpoint count above which an ALERT is logged. The "-On4266" variant task passes 0 to record every run.

.PARAMETER LogDir
  Directory for udp-monitor.log and udp-monitor.heartbeat.

.PARAMETER HeartbeatMinutes
  Minimum minutes between OK lines. Default 60.

.EXAMPLE
  pwsh -NoProfile -File Monitor-UDPSockets.ps1 -WhatIf   # log, but never stop a process or flush DNS
#>
[CmdletBinding(SupportsShouldProcess)]
param(
    [int]$Threshold = 5000,
    # GetFolderPath, not $env:USERPROFILE: agent homes redirect USERPROFILE and the log would land there.
    [string]$LogDir = (Join-Path ([Environment]::GetFolderPath('UserProfile')) '.machine\logs'),
    [int]$HeartbeatMinutes = 60
)

function Write-UdpLog {
    param([string]$LogDir, [string]$Message)
    # -WhatIf:$false: a dry run must still write its own log; only the kill and the DNS flush are suppressed.
    $Message | Out-File -Append -LiteralPath (Join-Path $LogDir 'udp-monitor.log') -Encoding utf8 -WhatIf:$false
}

function Test-HeartbeatDue {
    <# True when no heartbeat was written in the last $Minutes minutes, judged by the stamp file's mtime. #>
    param([string]$LogDir, [int]$Minutes, [datetime]$Now = (Get-Date))
    $stamp = Join-Path $LogDir 'udp-monitor.heartbeat'
    if (-not (Test-Path -LiteralPath $stamp)) { return $true }
    return (($Now - (Get-Item -LiteralPath $stamp).LastWriteTime).TotalMinutes -ge $Minutes)
}

function Invoke-UdpMonitor {
    [CmdletBinding(SupportsShouldProcess)]
    param([int]$Threshold, [string]$LogDir, [int]$HeartbeatMinutes)

    if (-not (Test-Path -LiteralPath $LogDir)) { New-Item -ItemType Directory -Path $LogDir -Force -WhatIf:$false | Out-Null }
    $timestamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'

    try { $endpoints = @(Get-NetUDPEndpoint -ErrorAction Stop) }
    catch {
        Write-UdpLog $LogDir "[$timestamp] ERROR: cannot read UDP endpoints: $($_.Exception.Message)"
        return 1
    }
    $udpCount = $endpoints.Count

    if ($udpCount -gt $Threshold) {
        $top = $endpoints | Group-Object OwningProcess | Sort-Object Count -Descending | Select-Object -First 5
        $msg = "[$timestamp] ALERT: UDP socket count $udpCount exceeds threshold $Threshold`n"
        foreach ($t in $top) {
            $proc = Get-Process -Id $t.Name -ErrorAction SilentlyContinue
            $name = if ($proc) { $proc.ProcessName } else { 'unknown' }
            $msg += "  PID $($t.Name) ($name): $($t.Count) sockets`n"
        }
        Write-UdpLog $LogDir $msg

        # Stop the worst offender only when it is dllhost holding more than 10000 sockets.
        $worst = $top | Select-Object -First 1
        $worstProc = Get-Process -Id $worst.Name -ErrorAction SilentlyContinue
        if ($worstProc -and $worstProc.ProcessName -eq 'dllhost' -and $worst.Count -gt 10000) {
            if ($PSCmdlet.ShouldProcess("dllhost PID $($worst.Name) ($($worst.Count) sockets)", 'Stop-Process and flush DNS')) {
                Write-UdpLog $LogDir "[$timestamp] AUTO-KILL: dllhost PID $($worst.Name) with $($worst.Count) sockets"
                Stop-Process -Id $worst.Name -Force
                ipconfig /flushdns | Out-Null
            }
        }
    }
    elseif (Test-HeartbeatDue -LogDir $LogDir -Minutes $HeartbeatMinutes) {
        Write-UdpLog $LogDir "[$timestamp] OK: $udpCount UDP sockets"
        Set-Content -LiteralPath (Join-Path $LogDir 'udp-monitor.heartbeat') -Value $timestamp -WhatIf:$false
    }
    return 0
}

# Run only when executed, not when dot-sourced by the tests.
if ($MyInvocation.InvocationName -ne '.') {
    $code = Invoke-UdpMonitor -Threshold $Threshold -LogDir $LogDir -HeartbeatMinutes $HeartbeatMinutes -WhatIf:$WhatIfPreference
    exit $code
}
