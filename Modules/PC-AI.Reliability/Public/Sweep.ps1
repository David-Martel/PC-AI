#Requires -Version 5.1
<#
Entry points: the task wrapper (Invoke-FleetTask), the health sweep (Invoke-FleetHealthSweep) and
the daily digest (Send-FleetDigest). Tools/SystemScripts/Reliability/*.ps1 are thin CLIs over these.
#>

function Get-FleetFilePrefixHash {
    <# SHA-256 of the first Bytes bytes of a file (shared read, so a writer may keep it open). #>
    param([Parameter(Mandatory)][string]$Path, [int64]$Bytes)
    $fs = [IO.File]::Open($Path, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::ReadWrite)
    try {
        $buf = New-Object byte[] ([int]$Bytes)
        $n = $fs.Read($buf, 0, $buf.Length)
        $sha = [Security.Cryptography.SHA256]::Create()
        try { return [BitConverter]::ToString($sha.ComputeHash($buf, 0, $n)) } finally { $sha.Dispose() }
    }
    finally { $fs.Dispose() }
}

function Get-FleetStreamEncoding {
    <# Encoding named by the byte-order mark at the start of a stream (UTF-8 without one). Leaves the position at 0. #>
    param([Parameter(Mandatory)][IO.Stream]$Stream)
    $bom = New-Object byte[] 4
    [void]$Stream.Seek(0, [IO.SeekOrigin]::Begin)
    $n = $Stream.Read($bom, 0, 4)
    [void]$Stream.Seek(0, [IO.SeekOrigin]::Begin)
    if ($n -ge 4 -and $bom[0] -eq 0xFF -and $bom[1] -eq 0xFE -and $bom[2] -eq 0 -and $bom[3] -eq 0) { return (New-Object Text.UTF32Encoding($false, $true)) }
    if ($n -ge 2 -and $bom[0] -eq 0xFF -and $bom[1] -eq 0xFE) { return (New-Object Text.UnicodeEncoding($false, $true)) }
    if ($n -ge 2 -and $bom[0] -eq 0xFE -and $bom[1] -eq 0xFF) { return (New-Object Text.UnicodeEncoding($true, $true)) }
    if ($n -ge 3 -and $bom[0] -eq 0xEF -and $bom[1] -eq 0xBB -and $bom[2] -eq 0xBF) { return (New-Object Text.UTF8Encoding($true)) }
    return (New-Object Text.UTF8Encoding($false))
}

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
        [string]$WorkingDirectory,
        [string]$PolicyPath,
        [switch]$DryRun
    )
    if ($DryRun) {
        # Host stream, not output: the caller captures the return value as the exit code.
        Write-Host ("DRY RUN: would run {0} {1} as task '{2}' (timeout {3} min), scan its output{4} for /{5}/, record a run event and update the incident" -f `
                $FilePath, ($ArgumentList -join ' '), $Name, $TimeoutMinutes, $(if ($LogPath) { " and $LogPath" } else { '' }), ($FailurePattern -join '|'))
        return 0
    }
    # The wrapped job always runs. A policy that cannot be loaded is recorded as its own incident after the run;
    # throwing here stopped the job itself over a typo in -PolicyPath (sweep row 12).
    $policyError = $null
    if ($PolicyPath) {
        try { $script:FleetEffectivePolicy = Get-FleetPolicy -Path $PolicyPath }
        catch { $policyError = $_.Exception.Message; $script:FleetEffectivePolicy = $null }
    }
    try {
        return (Invoke-FleetTaskCore -Name $Name -FilePath $FilePath -ArgumentList $ArgumentList -LogPath $LogPath -FailurePattern $FailurePattern `
                -TimeoutMinutes $TimeoutMinutes -Severity $Severity -WorkingDirectory $WorkingDirectory -PolicyPath $PolicyPath -PolicyError $policyError)
    }
    finally { $script:FleetEffectivePolicy = $null }
}

function Invoke-FleetTaskCore {
    <# The body of Invoke-FleetTask, run with the effective policy set; its caller always clears it. #>
    param([string]$Name, [string]$FilePath, [string[]]$ArgumentList, [string]$LogPath, [string[]]$FailurePattern, [int]$TimeoutMinutes,
        [string]$Severity, [string]$WorkingDirectory, [string]$PolicyPath, [string]$PolicyError)
    $start = Get-FleetNow
    $logOffset = [int64]0
    $logPrefix = $null
    $logError = $null
    if ($LogPath -and (Test-Path -LiteralPath $LogPath)) {
        # A log that cannot be read before the run (locked, denied) must not stop the job: scan it whole afterwards.
        try {
            $logOffset = (Get-Item -LiteralPath $LogPath).Length
            $logPrefix = Get-FleetFilePrefixHash -Path $LogPath -Bytes ([Math]::Min(4096, $logOffset))
        }
        catch { $logOffset = [int64]0; $logPrefix = $null }
    }
    $r = Invoke-FleetProcess -FilePath $FilePath -ArgumentList $ArgumentList -TimeoutSeconds ($TimeoutMinutes * 60) -WorkingDirectory $WorkingDirectory
    $end = Get-FleetNow

    $safe = ($Name -replace '[\\/:*?"<>|]+', '_').Trim('_')
    $ownLog = Join-Path (Get-FleetPath -Kind logs) ("{0}.log" -f $safe)
    $header = "=== {0} {1} exit={2} timedout={3}`n" -f (ConvertTo-FleetIso $start), $Name, $r.ExitCode, $r.TimedOut
    [IO.File]::AppendAllText($ownLog, $header + $r.StdOut + $(if ($r.StdErr) { "`n--- stderr ---`n" + $r.StdErr } else { '' }) + "`n", (New-Object System.Text.UTF8Encoding($false)))

    $scan = $r.StdOut + "`n" + $r.StdErr
    if ($LogPath -and -not (Test-Path -LiteralPath $LogPath)) {
        # The job was told to write this log and did not: exit 0 is not backed by its own record (sweep row 19).
        $logError = "configured log is missing after the run: $LogPath"
    }
    elseif ($LogPath) {
        # A log that cannot be opened (locked, denied, removed mid-check) is reported; the run is still recorded
        # with its exit code. The open used to throw and lose the run record (sweep row 20).
        $fs = $null
        try { $fs = [IO.File]::Open($LogPath, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::ReadWrite) }
        catch { $logError = "cannot read ${LogPath}: $($_.Exception.Message)" }
    }
    if ($LogPath -and $fs) {
        try {
            if ($fs.Length -lt $logOffset) { $logOffset = 0 }   # rotated during the run
            # Replaced by an equal or larger file: the bytes before the old offset are no longer the ones seen
            # before the run, so scan from the start (ninth review of #346; size alone missed this).
            elseif ($logOffset -gt 0 -and (Get-FleetFilePrefixHash -Path $LogPath -Bytes ([Math]::Min(4096, $logOffset))) -ne $logPrefix) { $logOffset = 0 }
            # The encoding comes from the byte-order mark at the START of the file; seeking first skipped it, and
            # appended UTF-16 text was decoded as UTF-8 so no failure pattern matched (sixth review of #346).
            $enc = Get-FleetStreamEncoding -Stream $fs
            $bomLength = $enc.GetPreamble().Length
            if ($logOffset -lt $bomLength) { $logOffset = $bomLength }
            $unit = 1; if ($enc -is [Text.UnicodeEncoding]) { $unit = 2 } elseif ($enc -is [Text.UTF32Encoding]) { $unit = 4 }
            $logOffset -= (($logOffset - $bomLength) % $unit)   # stay on a character boundary
            [void]$fs.Seek($logOffset, [IO.SeekOrigin]::Begin)
            $sr = New-Object IO.StreamReader($fs, $enc, $false)
            $scan += "`n" + $sr.ReadToEnd()
        }
        catch { $logError = "cannot read ${LogPath}: $($_.Exception.Message)" }
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
    # A descendant held the output pipes past the drain deadline, so a failure line may be missing. Exit 0 is then
    # no verified recovery: the run is neutral and the incomplete capture is its own incident (review of b5f31a5c).
    $partial = ($r.PSObject.Properties.Name -contains 'OutputComplete') -and ($r.OutputComplete -eq $false)
    $obsStatus = $status
    if ($partial -and $status -eq 'ok') { $obsStatus = 'skipped'; $msg = 'exit 0, but the output was not fully captured (a descendant kept the pipes open)' }

    $logForEvent = $ownLog; if ($LogPath) { $logForEvent = $LogPath }
    $exitForEvent = $null; if (-not $r.TimedOut) { $exitForEvent = $r.ExitCode }
    $run = New-FleetLedgerEvent -Source task -Name $Name -Check run -Kind run -Status $status -Severity $(if ($status -eq 'ok') { 'info' } else { $Severity }) `
        -Message $msg -Start $start -End $end -ExitCode $exitForEvent -LogPath $logForEvent -IncidentKey ('{0}|task|{1}|run' -f (Get-FleetHostName), $Name)
    $null = Write-FleetLedgerEvent -LedgerEvent $run
    $obs = New-FleetObservation -Source task -Name $Name -Check run -Status $obsStatus -Severity $Severity -Message $msg -LogPath $logForEvent `
        -ExitCode $exitForEvent -Start $start -End $end -OpenAfter 1 -ResolveAfter 1 -Origin wrapper
    $null = Submit-FleetObservation -Observation $obs
    $cst = 'ok'; $cmsg = 'output fully captured'
    if ($partial) { $cst = 'degraded'; $cmsg = 'output not fully captured: a descendant process kept the pipes open past the drain deadline' }
    $null = Submit-FleetObservation -Observation (New-FleetObservation -Source task -Name $Name -Check output-capture -Status $cst -Severity low -Message $cmsg -OpenAfter 1 -ResolveAfter 1 -Origin wrapper)
    if ($LogPath) {
        $lst = 'ok'; $lmsg = "read $LogPath"
        if ($logError) { $lst = 'failed'; $lmsg = $logError }
        $null = Submit-FleetObservation -Observation (New-FleetObservation -Source task -Name $Name -Check log-read -Status $lst -Severity low -Message $lmsg -LogPath $LogPath -OpenAfter 1 -ResolveAfter 1 -Origin wrapper)
    }
    if ($PolicyPath) {
        $pst = 'ok'; $pmsg = "loaded $PolicyPath"
        if ($PolicyError) { $pst = 'failed'; $pmsg = $PolicyError }
        $null = Submit-FleetObservation -Observation (New-FleetObservation -Source sweeper -Name 'policy' -Check policy-load -Status $pst -Severity medium -Message $pmsg -OpenAfter 1 -ResolveAfter 1 -Origin wrapper)
    }
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
        [switch]$NoPlaybooks,
        [switch]$DryRun
    )
    if ($DryRun) {
        $script:FleetReadOnly = $true
        try { return (Invoke-FleetDryRunSweep -Policy (Get-FleetPolicy -Path $PolicyPath) -Detector $Detector) }
        finally { $script:FleetReadOnly = $false }
    }
    $policy = Get-FleetPolicy -Path $PolicyPath
    $script:FleetEffectivePolicy = $policy
    try { return (Invoke-FleetHealthSweepCore -Policy $policy -Detector $Detector -NoPlaybooks:$NoPlaybooks) }
    finally { $script:FleetEffectivePolicy = $null }
}

function Invoke-FleetHealthSweepCore {
    <# The body of a sweep, run with the effective policy set; its caller always clears it (even on a throw). #>
    param([hashtable]$Policy, [string[]]$Detector, [switch]$NoPlaybooks)
    $policy = $Policy
    $started = Get-FleetNow
    $spool = Invoke-FleetSpoolReplay

    $detectors = Get-FleetDetectorTable -Policy $policy
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
    # 'sweeper' is NOT a completed source: a detector-error clears only when that detector runs and
    # succeeds, never because a sweep was limited to other detectors (codex review, #346).
    $late = @(Resolve-FleetUnobservedIncidents -ObservedKeys $keys -CompletedSources @($completed) -Policy $policy)
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

function Get-FleetDetectorTable {
    param([hashtable]$Policy)
    $detectors = [ordered]@{
        task     = { param($p) Get-FleetTaskObservations -Policy $p }
        disk     = { param($p) Get-FleetDiskObservations -Policy $p }
        postgres = { param($p) Get-FleetPostgresObservations -Policy $p }
        process  = { param($p) Get-FleetOrphanProcessObservations -Policy $p }
        cert     = { param($p) Get-FleetCertObservations -Policy $p }
    }
    if ($Policy.fleet_checks) {
        $detectors['runner'] = { param($p) Get-FleetRunnerObservations -Policy $p }
        $detectors['workflow'] = { param($p) Get-FleetWorkflowObservations -Policy $p }
    }
    return $detectors
}

function Invoke-FleetDryRunSweep {
    <#
    Runs the detectors and returns their observations without any side effect on this host. The caller sets
    read-only mode, which every detector honours: no directory is created, the disk samples and DRAIN state
    are read but not written, and the task-health child reports to a temp file removed at once. The old
    approach ran the detectors against a temporary copy of the home, which still wrote files (sweep row 23).
    Nothing is submitted, forwarded, replayed or remediated.
    #>
    param([hashtable]$Policy, [string[]]$Detector)
    if (-not $script:FleetReadOnly) { throw 'Invoke-FleetDryRunSweep must run in read-only mode' }
    $results = @()
    $table = Get-FleetDetectorTable -Policy $Policy
    foreach ($name in @($table.Keys)) {
        if ($Detector -and $Detector -notcontains $name) { continue }
        try { $results += @(& $table[$name] $Policy) }
        catch { $results += New-FleetObservation -Source sweeper -Name $name -Check detector-error -Status failed -Severity low -Message $_.Exception.Message }
    }
    return $results
}

function Read-FleetLedger {
    <# Ledger events since a time, newest files included across one rotation. #>
    param([datetime]$Since)
    $dir = Get-FleetPath -Kind ledger -NoCreate
    $events = @()
    # Unparseable lines are counted, not silently dropped: the digest says how much history it could not read
    # (sweep row 22). $script:FleetLedgerBadLines is read by Send-FleetDigest right after this call.
    $script:FleetLedgerBadLines = 0
    foreach ($f in @('ledger.1.jsonl', 'ledger.jsonl')) {
        $p = Join-Path $dir $f
        if (-not (Test-Path -LiteralPath $p)) { continue }
        foreach ($line in [IO.File]::ReadAllLines($p)) {
            if ([string]::IsNullOrWhiteSpace($line)) { continue }
            try { $e = $line | ConvertFrom-Json -ErrorAction Stop; $ts = ConvertFrom-FleetIso $e.ts }
            catch { $script:FleetLedgerBadLines++; continue }
            if ($null -eq $ts) { $script:FleetLedgerBadLines++; continue }
            if ($ts -ge $Since) { $events += $e }
        }
    }
    return $events
}

function Get-FleetBusLedgerEvents {
    <# Fleet-wide: every host's forwarded events from agent-bus (the event rides in metadata). #>
    param([int]$SinceMinutes = 1440)
    $r = Invoke-FleetProcess -FilePath (Get-FleetAgentBusPath) -ArgumentList @('read', '--topic', 'fleet-failures', '--since-minutes', [string]$SinceMinutes, '--limit', '500', '--encoding', 'json') -TimeoutSeconds 30
    if ($r.TimedOut -or $r.ExitCode -ne 0) { throw "agent-bus read failed (exit $($r.ExitCode))" }
    $msgs = @($r.StdOut | ConvertFrom-Json | ForEach-Object { $_ })   # 5.1 emits a JSON array as one object
    if ($msgs.Count -eq 1 -and $msgs[0].PSObject.Properties.Name -contains 'messages') { $msgs = @($msgs[0].messages) }
    $events = @()
    foreach ($m in $msgs) {
        $md = $m.metadata
        if ($md -and $md.schema -eq 'fleet-ledger/1') { $events += $md }
    }
    return $events
}

function Get-FleetOpenIncidentRows {
    <# This host's open incidents from the durable store, shaped like ledger events for the digest. #>
    $rows = @()
    $all = Get-FleetIncidentState
    foreach ($key in @($all.Keys)) {
        $s = $all[$key]
        if ($s.state -ne 'open') { continue }
        $parts = ([string]$key).Split('|')
        if ($parts.Count -lt 4) { continue }
        $rows += [pscustomobject]@{
            host = $parts[0]; source = $parts[1]; name = ($parts[2..($parts.Count - 2)] -join '|'); check = $parts[-1]
            severity = $(if ($s.last_severity) { $s.last_severity } else { 'medium' }); ts = $s.opened_at
            message = [string]$s.message; incident_key = $key; kind = 'incident-open'
        }
    }
    return $rows
}

function Send-FleetDigest {
    <#
    Short health summary for the last SinceHours: failures, recoveries, escalations, incidents still
    open, disk lines. With -Fleet (or policy digest.fleet) it summarises every host from agent-bus;
    otherwise this host from its own ledger. Posted as a 'digest' event (agent-bus status + Discord).
    #>
    [CmdletBinding()]
    param([string]$PolicyPath, [int]$SinceHours, [switch]$Fleet, [switch]$NoSend, [switch]$DryRun)
    # A dry run never moves or rewrites a live file (second review of #346).
    $savedReadOnly = $script:FleetReadOnly
    if ($DryRun) { $script:FleetReadOnly = $true }
    try {
        $policy = Get-FleetPolicy -Path $PolicyPath
        if (-not $SinceHours) { $SinceHours = 24; if ($policy.digest.since_hours) { $SinceHours = [int]$policy.digest.since_hours } }
        if (-not $Fleet -and $policy.digest.fleet) { $Fleet = [switch]$true }
        $now = Get-FleetNow
        $since = $now.AddHours(-$SinceHours)
        $problems = @()
        if ($Fleet) { $events = @(Get-FleetBusLedgerEvents -SinceMinutes ($SinceHours * 60)) }
        else {
            $events = @(Read-FleetLedger -Since $since)
            if ($script:FleetLedgerBadLines -gt 0) { $problems += ('- {0} unreadable ledger line(s) on {1}: counts may be low' -f $script:FleetLedgerBadLines, (Get-FleetHostName)) }
        }

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
        # "Still open" is current state, not window activity: this host's open incidents come from the durable
        # incident store, so one opened before the window with no event since still counts (seventh review of
        # #346). In fleet mode, other hosts are known only from bus events, so their open incidents show while
        # inside the window or refreshed by a reminder.
        $self = Get-FleetHostName
        # An unreadable incident store is reported, never shown as "still open 0" (sweep row 14).
        try { $durable = @(Get-FleetOpenIncidentRows) }
        catch { $durable = @(); $problems += ('- incident store unreadable on {0}: open incidents unknown ({1})' -f $self, $_.Exception.Message) }
        $others = @()
        if ($Fleet) { $others = @($latest.Values | Where-Object { $_.kind -ne 'incident-resolved' -and $_.host -ne $self }) }
        $stillOpen = @(@($durable) + @($others) | Sort-Object host, name)
        $hosts = @($events | ForEach-Object { $_.host } | Sort-Object -Unique)
        $lines = @()
        $lines += ("{0} {1} h to {2:yyyy-MM-dd HH:mm}Z, hosts [{3}]: opened {4}, resolved {5}, escalations {6}, remediations {7}, still open {8}" -f `
                $(if ($Fleet) { 'fleet' } else { Get-FleetHostName }), $SinceHours, $now, ($hosts -join ' '),
            [int]$byKind['incident-open'], [int]$byKind['incident-resolved'], [int]$byKind['escalation'], [int]$byKind['remediation'], $stillOpen.Count)
        foreach ($o in ($stillOpen | Select-Object -First 12)) {
            $lines += ("- OPEN [{0}] {1} {2}/{3} ({4}) since {5}: {6}" -f $o.severity, $o.host, $o.source, $o.name, $o.check, (ConvertTo-FleetIso (ConvertFrom-FleetIso $o.ts)).Substring(0, 16), ([string]$o.message).Substring(0, [Math]::Min(120, ([string]$o.message).Length)))
        }
        if ($stillOpen.Count -gt 12) { $lines += ("- ... and {0} more" -f ($stillOpen.Count - 12)) }
        # The last sweep's age is stated: a dead sweeper left a "healthy" digest with old disk lines (sweep row 21).
        $maxAge = 2.0; if ($policy.digest.max_sweep_age_hours) { $maxAge = [double]$policy.digest.max_sweep_age_hours }
        $last = $null
        try { $last = Read-FleetJsonFile -Path (Join-Path (Get-FleetPath -Kind state -NoCreate) 'last-sweep.json') -Default $null -ThrowOnError }
        catch { $problems += ('- last sweep record unreadable on {0}: {1}' -f $self, $_.Exception.Message) }
        $lastEnd = $null; if ($last) { try { $lastEnd = ConvertFrom-FleetIso $last.ended } catch { $lastEnd = $null } }
        if ($null -eq $last) { if (-not ($problems -match 'last sweep record')) { $problems += ('- no sweep recorded on {0}' -f $self) } }
        elseif ($null -eq $lastEnd) { $problems += ('- last sweep on {0} has no end time' -f $self) }
        else {
            $ageH = [math]::Round(($now - $lastEnd).TotalHours, 1)
            if ($ageH -gt $maxAge) { $problems += ('- last sweep {0} h ago (stale) on {1}; disk lines below are from then' -f $ageH, $self) }
        }
        if ($last) { foreach ($d in @($last.disks)) { if ($d) { $lines += ("- disk {0} {1}" -f (Get-FleetHostName), $d) } } }
        $lines += $problems
        $msg = ($lines -join "`n")
        $digestStatus = 'ok'; if ($stillOpen.Count -or $problems.Count) { $digestStatus = 'degraded' }
        $ev = New-FleetLedgerEvent -Source digest -Name $(if ($Fleet) { 'fleet' } else { Get-FleetHostName }) -Check daily -Kind digest -Status $digestStatus -Severity info -Message $msg `
            -Details @{ opened = [int]$byKind['incident-open']; resolved = [int]$byKind['incident-resolved']; escalations = [int]$byKind['escalation']; still_open = $stillOpen.Count; hosts = $hosts }
        if (-not $DryRun) {
            $script:FleetEffectivePolicy = $policy
            try { $null = Write-FleetLedgerEvent -LedgerEvent $ev -NoForward:$NoSend }
            finally { $script:FleetEffectivePolicy = $null }
        }
        return [pscustomobject]@{ Message = $msg; StillOpen = $stillOpen.Count; Event = $ev }
    }
    finally { $script:FleetReadOnly = $savedReadOnly }
}
