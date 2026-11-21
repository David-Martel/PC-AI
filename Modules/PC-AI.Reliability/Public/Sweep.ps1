#Requires -Version 5.1
<#
Entry points: the task wrapper (Invoke-FleetTask), the health sweep (Invoke-FleetHealthSweep) and
the daily digest (Send-FleetDigest). Tools/SystemScripts/Reliability/*.ps1 are thin CLIs over these.
#>

function Invoke-FleetTask {
    <#
    Runs a scheduled task's real command and records what actually happened. A task that exits 0
    while its log says it failed ("integrity check did not pass", "STALE", "[ERROR]") is recorded
    as failed and the wrapper exits non-zero, so Task Scheduler stops showing 0x0 for it.
    Output goes to <home>\logs\<task>.log; -LogPath names the tool's own log, whose bytes written
    during this run are scanned too. Returns the exit code the task should report.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$Name,
        [Parameter(Mandatory)][string]$FilePath,
        [string[]]$ArgumentList = @(),
        [string]$LogPath,
        [string[]]$FailurePattern = @('\[ERROR\]', '\bERROR:', '\bSTALE\b', 'integrity check did not pass', 'ALERT not sent'),
        [int]$TimeoutMinutes = 120,
        [string]$Severity = 'medium',
        [string]$WorkingDirectory
    )
    $start = Get-FleetNow
    $logOffset = [int64]0
    if ($LogPath -and (Test-Path -LiteralPath $LogPath)) { $logOffset = (Get-Item -LiteralPath $LogPath).Length }
    $r = Invoke-FleetProcess -FilePath $FilePath -ArgumentList $ArgumentList -TimeoutSeconds ($TimeoutMinutes * 60) -WorkingDirectory $WorkingDirectory
    $end = Get-FleetNow

    $safe = ($Name -replace '[\\/:*?"<>|]+', '_').Trim('_')
    $ownLog = Join-Path (Get-FleetPath -Kind logs) ("{0}.log" -f $safe)
    $header = "=== {0} {1} exit={2} timedout={3}`n" -f (ConvertTo-FleetIso $start), $Name, $r.ExitCode, $r.TimedOut
    [IO.File]::AppendAllText($ownLog, $header + $r.StdOut + $(if ($r.StdErr) { "`n--- stderr ---`n" + $r.StdErr } else { '' }) + "`n", (New-Object System.Text.UTF8Encoding($false)))

    $scan = $r.StdOut + "`n" + $r.StdErr
    if ($LogPath -and (Test-Path -LiteralPath $LogPath)) {
        $fs = [IO.File]::Open($LogPath, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::ReadWrite)
        try {
            if ($fs.Length -lt $logOffset) { $logOffset = 0 }   # rotated during the run
            [void]$fs.Seek($logOffset, [IO.SeekOrigin]::Begin)
            $sr = New-Object IO.StreamReader($fs)
            $scan += "`n" + $sr.ReadToEnd()
        }
        finally { $fs.Dispose() }
    }
    $hit = $null
    foreach ($line in ($scan -split "`r?`n")) {
        foreach ($p in $FailurePattern) { if ($line -match $p) { $hit = $line.Trim(); break } }
        if ($hit) { break }
    }

    $status = 'ok'; $code = 0; $msg = 'exit 0'
    if ($r.TimedOut) { $status = 'failed'; $code = 124; $msg = "timed out after $TimeoutMinutes min" }
    elseif ($r.ExitCode -ne 0) { $status = 'failed'; $code = $r.ExitCode; $msg = "exit $($r.ExitCode)" + $(if ($hit) { ": $hit" } else { '' }) }
    elseif ($hit) { $status = 'failed'; $code = 1; $msg = "exit 0 but the output says it failed: $hit" }
    if ($r.ExitCode -eq -1 -and -not $r.TimedOut -and $r.StdErr -like 'failed to start*') { $msg = $r.StdErr }

    $logForEvent = $ownLog; if ($LogPath) { $logForEvent = $LogPath }
    $exitForEvent = $null; if (-not $r.TimedOut) { $exitForEvent = $r.ExitCode }
    $run = New-FleetLedgerEvent -Source task -Name $Name -Check run -Kind run -Status $status -Severity $(if ($status -eq 'ok') { 'info' } else { $Severity }) `
        -Message $msg -Start $start -End $end -ExitCode $exitForEvent -LogPath $logForEvent -IncidentKey ('{0}|task|{1}|run' -f (Get-FleetHostName), $Name)
    $null = Write-FleetLedgerEvent -LedgerEvent $run
    $obs = New-FleetObservation -Source task -Name $Name -Check run -Status $status -Severity $Severity -Message $msg -LogPath $logForEvent `
        -ExitCode $exitForEvent -Start $start -End $end -OpenAfter 1 -ResolveAfter 1
    $null = Submit-FleetObservation -Observation $obs
    return $code
}

function Invoke-FleetHealthSweep {
    <#
    One sweep: replay the bus spool, run every detector (a detector that throws becomes its own
    'detector-error' observation instead of silently passing), apply observations to the incident
    store, run bounded playbooks for open incidents, resolve incidents whose subject vanished, and
    write a summary. -NoPlaybooks observes only.
    #>
    [CmdletBinding()]
    param(
        [string]$PolicyPath,
        [string[]]$Detector,
        [switch]$NoPlaybooks
    )
    $policy = Get-FleetPolicy -Path $PolicyPath
    $started = Get-FleetNow
    $spool = Invoke-FleetSpoolReplay

    $detectors = [ordered]@{
        task     = { param($p) Get-FleetTaskObservations -Policy $p }
        disk     = { param($p) Get-FleetDiskObservations -Policy $p }
        postgres = { param($p) Get-FleetPostgresObservations -Policy $p }
        process  = { param($p) Get-FleetOrphanProcessObservations -Policy $p }
        cert     = { param($p) Get-FleetCertObservations -Policy $p }
    }
    if ($policy.fleet_checks) {
        $detectors['runner'] = { param($p) Get-FleetRunnerObservations -Policy $p }
        $detectors['workflow'] = { param($p) Get-FleetWorkflowObservations -Policy $p }
    }
    $all = @(); $completed = @(); $errors = @()
    foreach ($name in @($detectors.Keys)) {
        if ($Detector -and $Detector -notcontains $name) { continue }
        try {
            $all += @(& $detectors[$name] $policy)
            $completed += $name
            $all += New-FleetObservation -Source sweeper -Name $name -Check detector-error -Status ok -ResolveAfter 1
        }
        catch {
            $errors += "${name}: $($_.Exception.Message)"
            $all += New-FleetObservation -Source sweeper -Name $name -Check detector-error -Status failed -Severity low -ResolveAfter 1 -Message $_.Exception.Message
        }
    }

    $keys = @(); $opened = 0; $resolved = 0; $playbooks = @()
    foreach ($o in $all) {
        $t = Submit-FleetObservation -Observation $o
        $keys += $t.Key
        if ($t.Transition -eq 'opened') { $opened++ }
        if ($t.Transition -eq 'resolved') { $resolved++ }
        if (-not $NoPlaybooks -and $o.Playbook -and $t.Bad) {
            $state = (Get-FleetIncidentState)[$t.Key]
            if ($state -and $state.state -eq 'open') {
                $outcome = Invoke-FleetPlaybookForObservation -Observation $o -IncidentKey $t.Key -Policy $policy
                $playbooks += ('{0}:{1}={2}' -f $o.Playbook, $o.Name, $outcome)
            }
        }
    }
    $late = @(Resolve-FleetUnobservedIncidents -ObservedKeys $keys -CompletedSources @($completed + 'sweeper'))
    $resolved += @($late | Where-Object { $_.Transition -eq 'resolved' }).Count

    $openNow = @((Get-FleetIncidentState).GetEnumerator() | Where-Object { $_.Value.state -eq 'open' -and $_.Key.StartsWith((Get-FleetHostName) + '|') })
    $summary = [ordered]@{
        started = ConvertTo-FleetIso $started; ended = ConvertTo-FleetIso (Get-FleetNow)
        observations = $all.Count; bad = @($all | Where-Object { @('ok', 'skipped') -notcontains $_.Status }).Count
        opened = $opened; resolved = $resolved; open_incidents = $openNow.Count
        playbooks = $playbooks; detector_errors = $errors; spool = $spool
        disks = @($all | Where-Object { $_.Source -eq 'disk' } | ForEach-Object { $_.Message })
    }
    Write-FleetJsonFile -Path (Join-Path (Get-FleetPath -Kind state) 'last-sweep.json') -Value $summary
    $sweepStatus = 'ok'; if ($errors.Count -gt 0) { $sweepStatus = 'degraded' }
    $ev = New-FleetLedgerEvent -Source sweeper -Name 'health-sweep' -Check run -Kind run -Status $sweepStatus -Severity info `
        -Message ("{0} observations, {1} bad, {2} opened, {3} resolved, {4} open, playbooks [{5}]" -f $summary.observations, $summary.bad, $opened, $resolved, $openNow.Count, ($playbooks -join ', ')) `
        -Start $started -End (Get-FleetNow) -Details @{ detector_errors = $errors }
    $null = Write-FleetLedgerEvent -LedgerEvent $ev
    return [pscustomobject]$summary
}

function Read-FleetLedger {
    <# Ledger events since a time, newest files included across one rotation. #>
    param([datetime]$Since)
    $dir = Get-FleetPath -Kind ledger
    $events = @()
    foreach ($f in @('ledger.1.jsonl', 'ledger.jsonl')) {
        $p = Join-Path $dir $f
        if (-not (Test-Path -LiteralPath $p)) { continue }
        foreach ($line in [IO.File]::ReadAllLines($p)) {
            if ([string]::IsNullOrWhiteSpace($line)) { continue }
            try { $e = $line | ConvertFrom-Json } catch { continue }
            if ((ConvertFrom-FleetIso $e.ts) -ge $Since) { $events += $e }
        }
    }
    return $events
}

function Get-FleetBusLedgerEvents {
    <# Fleet-wide: every host's forwarded events from agent-bus (the event rides in metadata). #>
    param([int]$SinceMinutes = 1440)
    $r = Invoke-FleetProcess -FilePath (Get-FleetAgentBusPath) -ArgumentList @('read', '--topic', 'fleet-failures', '--since-minutes', [string]$SinceMinutes, '--limit', '500', '--encoding', 'json') -TimeoutSeconds 30
    if ($r.TimedOut -or $r.ExitCode -ne 0) { throw "agent-bus read failed (exit $($r.ExitCode))" }
    $msgs = @($r.StdOut | ConvertFrom-Json)
    if ($msgs.Count -eq 1 -and $msgs[0].PSObject.Properties.Name -contains 'messages') { $msgs = @($msgs[0].messages) }
    $events = @()
    foreach ($m in $msgs) {
        $md = $m.metadata
        if ($md -and $md.schema -eq 'fleet-ledger/1') { $events += $md }
    }
    return $events
}

function Send-FleetDigest {
    <#
    Short health summary for the last SinceHours: failures, recoveries, escalations, incidents still
    open, disk lines. With -Fleet (or policy digest.fleet) it summarises every host from agent-bus;
    otherwise this host from its own ledger. Posted as a 'digest' event (agent-bus status + Discord).
    #>
    [CmdletBinding()]
    param([string]$PolicyPath, [int]$SinceHours, [switch]$Fleet, [switch]$NoSend)
    $policy = Get-FleetPolicy -Path $PolicyPath
    if (-not $SinceHours) { $SinceHours = 24; if ($policy.digest.since_hours) { $SinceHours = [int]$policy.digest.since_hours } }
    if (-not $Fleet -and $policy.digest.fleet) { $Fleet = [switch]$true }
    $now = Get-FleetNow
    $since = $now.AddHours(-$SinceHours)
    if ($Fleet) { $events = @(Get-FleetBusLedgerEvents -SinceMinutes ($SinceHours * 60)) }
    else { $events = @(Read-FleetLedger -Since $since) }

    $byKind = @{}
    foreach ($e in $events) { $byKind[[string]$e.kind] = 1 + [int]$byKind[[string]$e.kind] }
    $latest = @{}
    # Sort by time, then by ledger order: Windows PowerShell 5.1's Sort-Object is not stable, and an
    # open and its resolution can share a timestamp, which would otherwise reorder them.
    $i = 0
    $ordered = foreach ($e in $events) { [pscustomobject]@{ E = $e; T = (ConvertFrom-FleetIso $e.ts); I = $i++ } }
    foreach ($e in (@($ordered | Where-Object { $_.E.incident_key } | Sort-Object T, I) | ForEach-Object { $_.E })) {
        if (@('incident-open', 'incident-reminder', 'incident-flapping', 'escalation', 'incident-resolved') -contains $e.kind) { $latest[[string]$e.incident_key] = $e }
    }
    $stillOpen = @($latest.Values | Where-Object { $_.kind -ne 'incident-resolved' } | Sort-Object host, name)
    $hosts = @($events | ForEach-Object { $_.host } | Sort-Object -Unique)
    $lines = @()
    $lines += ("{0} {1} h to {2:yyyy-MM-dd HH:mm}Z, hosts [{3}]: opened {4}, resolved {5}, escalations {6}, remediations {7}, still open {8}" -f `
            $(if ($Fleet) { 'fleet' } else { Get-FleetHostName }), $SinceHours, $now, ($hosts -join ' '),
        [int]$byKind['incident-open'], [int]$byKind['incident-resolved'], [int]$byKind['escalation'], [int]$byKind['remediation'], $stillOpen.Count)
    foreach ($o in ($stillOpen | Select-Object -First 12)) {
        $lines += ("- OPEN [{0}] {1} {2}/{3} ({4}) since {5}: {6}" -f $o.severity, $o.host, $o.source, $o.name, $o.check, (ConvertTo-FleetIso (ConvertFrom-FleetIso $o.ts)).Substring(0, 16), ([string]$o.message).Substring(0, [Math]::Min(120, ([string]$o.message).Length)))
    }
    if ($stillOpen.Count -gt 12) { $lines += ("- ... and {0} more" -f ($stillOpen.Count - 12)) }
    $last = Read-FleetJsonFile -Path (Join-Path (Get-FleetPath -Kind state) 'last-sweep.json') -Default @{}
    foreach ($d in @($last.disks)) { if ($d) { $lines += ("- disk {0} {1}" -f (Get-FleetHostName), $d) } }
    $msg = ($lines -join "`n")
    $ev = New-FleetLedgerEvent -Source digest -Name $(if ($Fleet) { 'fleet' } else { Get-FleetHostName }) -Check daily -Kind digest -Status $(if ($stillOpen.Count) { 'degraded' } else { 'ok' }) -Severity info -Message $msg `
        -Details @{ opened = [int]$byKind['incident-open']; resolved = [int]$byKind['incident-resolved']; escalations = [int]$byKind['escalation']; still_open = $stillOpen.Count; hosts = $hosts }
    $null = Write-FleetLedgerEvent -LedgerEvent $ev -NoForward:$NoSend
    return [pscustomobject]@{ Message = $msg; StillOpen = $stillOpen.Count; Event = $ev }
}
