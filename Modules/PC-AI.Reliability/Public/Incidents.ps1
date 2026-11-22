#Requires -Version 5.1
<#
Incident state: one alert per incident, a reminder every 24 h while it stays open, one resolution
event when it recovers, and flap suppression. The wrapper (Invoke-FleetTask) and the sweeper feed
the same store, so a failure seen by both opens one incident, not two.

Why this exists: on 2026-11-21 every alert path either never fired (exit 0 while failing) or fired
into nothing ("ALERT not sent: no bot token"); monitors that do post repeat the same line every
30 s (the gitops monitor). Both are fixed by deciding, per incident key, when a message is news.
#>

function New-FleetObservation {
    <# A detector result. Status 'ok' and 'skipped' are good/neutral; anything else is bad. #>
    param(
        [Parameter(Mandatory)][string]$Source,
        [Parameter(Mandatory)][string]$Name,
        [Parameter(Mandatory)][string]$Check,
        [Parameter(Mandatory)][ValidateSet('ok', 'failed', 'stale', 'skipped', 'never-ran', 'degraded', 'unknown')][string]$Status,
        [ValidateSet('info', 'low', 'medium', 'high', 'critical')][string]$Severity = 'medium',
        [string]$Message = '',
        [string]$LogPath,
        [Nullable[int]]$ExitCode,
        [Nullable[datetime]]$Start,
        [Nullable[datetime]]$End,
        [hashtable]$Details = @{},
        [int]$OpenAfter = 1,
        [int]$ResolveAfter = 2,
        [string]$Playbook,
        [hashtable]$PlaybookArgs = @{},
        [ValidateSet('sweeper', 'wrapper')][string]$Origin = 'sweeper'
    )
    # Redacted here, at the source: the message reaches incident state, the ledger, agent-bus and Discord.
    $Message = ConvertTo-FleetSafeMessage $Message
    return [pscustomobject]@{
        Origin = $Origin
        Source = $Source; Name = $Name; Check = $Check; Status = $Status; Severity = $Severity
        Message = $Message; LogPath = $LogPath; ExitCode = $ExitCode; Start = $Start; End = $End
        Details = $Details; OpenAfter = $OpenAfter; ResolveAfter = $ResolveAfter
        Playbook = $Playbook; PlaybookArgs = $PlaybookArgs
    }
}

function Get-FleetIncidentKey {
    param([Parameter(Mandatory)]$Observation)
    return '{0}|{1}|{2}|{3}' -f (Get-FleetHostName), $Observation.Source, $Observation.Name, $Observation.Check
}

function Get-FleetIncidentState {
    <# The incident store. Throws when it exists but cannot be read: an unreadable store is not "no incidents". #>
    return (Read-FleetIncidentStore)
}

function Read-FleetIncidentStore {
    <#
    Reads state\incidents.json. A corrupt or empty store used to be moved aside and silently replaced by an
    empty one: every open incident, including keys only the wrapper reports, was forgotten without a word,
    and the digest said "still open 0" (sweep row 14). Now a plain read throws; with -Reset (only under the
    incidents lock, never in a dry run) the store is preserved under a unique name, a forwarded high-severity
    'incident-store' event says what was lost, and an empty store is returned.
    #>
    param([switch]$Reset)
    $path = Join-Path (Get-FleetPath -Kind state -NoCreate) 'incidents.json'
    try { return (Read-FleetJsonFile -Path $path -Default @{} -ThrowOnError) }
    catch {
        $err = $_.Exception.Message
        if (-not $Reset -or $script:FleetReadOnly) { throw "incident store unreadable: $err" }
        $bad = '{0}.corrupt-{1}-{2}' -f $path, (Get-FleetNow).ToString("yyyyMMdd'T'HHmmss'Z'", [Globalization.CultureInfo]::InvariantCulture), [guid]::NewGuid().ToString('N').Substring(0, 6)
        Move-Item -LiteralPath $path -Destination $bad -Force -ErrorAction Stop
        $ev = New-FleetLedgerEvent -Source sweeper -Name 'incident-store' -Check store-reset -Kind incident-open -Status failed -Severity high `
            -Message ("the incident store was unreadable and has been reset: incidents open before now are forgotten and re-open only if still failing; wrapper-only keys re-open on their next failing run. Preserved at {0}. {1}" -f $bad, $err) `
            -IncidentKey ('{0}|sweeper|incident-store|store-reset' -f (Get-FleetHostName))
        $null = Add-FleetIncidentOutbox -LedgerEvent $ev -Forward $true
        return @{}
    }
}

function Submit-FleetObservation {
    <#
    Applies one observation to the incident store and writes the resulting ledger events.
    Returns the transition: 'opened', 'resolved', 'reminder', 'flapping', 'still-open', 'quiet',
    'neutral' or 'suppressed' (a transition recorded but not forwarded because the key is flapping).

    Rules (docs/reliability/fleet-ledger.md, section Incident state):
      bad  -> consecutive_bad++; open when consecutive_bad >= OpenAfter.
      good -> consecutive_good++; resolve when open and consecutive_good >= ResolveAfter.
      open -> reminder when 24 h passed since the last alert, or at once when severity rises.
      flap -> 3 opens inside 6 h set flapping_until = now + 6 h and send ONE flapping event;
              while flapping, transitions are written to the ledger but not forwarded.
    #>
    [CmdletBinding()]
    param([Parameter(Mandatory, ValueFromPipeline)]$Observation)
    process {
        $key = Get-FleetIncidentKey -Observation $Observation
        $now = Get-FleetNow
        $bad = @('ok', 'skipped') -notcontains $Observation.Status
        $events = New-Object System.Collections.ArrayList
        # known: what the readable store said about this key before the update ('open', 'muted' or ''), for
        # the fallback alert when the update cannot be saved.
        # committed: the store write succeeded, so a later failure (promoting the staged events) must not send the
        # fallback; outbox recovery sends the committed events (review of d5d030fd).
        $ctx = @{ transition = 'quiet'; known = ''; committed = $false }

        # Any failure below (state path, lock, read, staging, save) still throws, but a bad observation first
        # sends one unsaved alert: a full disk otherwise silenced the alert that reports it (sentinel-utils #1600).
        # The try body is not re-indented, so the change reads as the two lines that wrap it.
        try {
        $statePath = Join-Path (Get-FleetPath -Kind state) 'incidents.json'
        $acquired = $false
        $null = Invoke-FleetWithLock -LockName 'incidents' -TimeoutSeconds 30 -Acquired ([ref]$acquired) -ScriptBlock {
            $all = Read-FleetIncidentStore -Reset
            $s = $all[$key]
            if ($null -ne $s) {
                $knownFlap = ConvertFrom-FleetIso $s.flapping_until
                if ($s.state -eq 'open') { $ctx.known = 'open' }
                elseif ($null -ne $knownFlap -and $now -lt $knownFlap -and ($knownFlap - $now).TotalHours -le 6) { $ctx.known = 'muted' }
            }
            if ($Observation.Status -eq 'skipped') { $ctx.transition = 'neutral'; return }
            if ($null -eq $s) {
                if (-not $bad) { $ctx.transition = 'quiet'; return }
                $s = @{ state = 'resolved'; consecutive_bad = 0; consecutive_good = 0; opened_at = $null; resolved_at = $null
                    last_seen = $null; last_status = $null; last_severity = $null; last_alert_at = $null; open_times = @(); flapping_until = $null; message = '' }
            }
            $flapUntil = ConvertFrom-FleetIso $s.flapping_until
            $flapping = ($null -ne $flapUntil -and $now -lt $flapUntil)
            $s.last_seen = ConvertTo-FleetIso $now
            # Who last saw this key: the sweeper may auto-resolve only keys it observes itself.
            $s.last_origin = [string]$Observation.Origin
            if (-not $s.last_origin) { $s.last_origin = 'sweeper' }

            if ($bad) {
                $s.consecutive_bad = [int]$s.consecutive_bad + 1
                $s.consecutive_good = 0
                $prevSev = $s.last_severity
                $prevStatus = $s.last_status
                $s.last_status = $Observation.Status
                $s.message = $Observation.Message
                if ($s.state -ne 'open' -and $s.consecutive_bad -ge $Observation.OpenAfter) {
                    $s.state = 'open'
                    $s.opened_at = ConvertTo-FleetIso $now
                    $s.last_severity = $Observation.Severity
                    $recent = @(@($s.open_times) | Where-Object { $_ -and (ConvertFrom-FleetIso $_) -gt $now.AddHours(-6) })
                    $s.open_times = @($recent + (ConvertTo-FleetIso $now))
                    if ($flapping) {
                        $ctx.transition = 'suppressed'
                        [void]$events.Add(@{ kind = 'incident-open'; forward = $false })
                    }
                    elseif (@($s.open_times).Count -ge 3) {
                        $s.flapping_until = ConvertTo-FleetIso $now.AddHours(6)
                        $s.last_alert_at = ConvertTo-FleetIso $now
                        $ctx.transition = 'flapping'
                        [void]$events.Add(@{ kind = 'incident-flapping'; forward = $true })
                    }
                    else {
                        $s.last_alert_at = ConvertTo-FleetIso $now
                        $ctx.transition = 'opened'
                        [void]$events.Add(@{ kind = 'incident-open'; forward = $true })
                    }
                }
                elseif ($s.state -eq 'open') {
                    $ctx.transition = 'still-open'
                    $lastAlert = ConvertFrom-FleetIso $s.last_alert_at
                    $rose = ($null -ne $prevSev) -and ($script:FleetSeverityRank[[string]$Observation.Severity] -gt $script:FleetSeverityRank[[string]$prevSev])
                    if ($rose) { $s.last_severity = $Observation.Severity }
                    if (-not $flapping -and ($rose -or $null -eq $lastAlert -or ($now - $lastAlert).TotalHours -ge 24)) {
                        $s.last_alert_at = ConvertTo-FleetIso $now
                        $ctx.transition = 'reminder'
                        [void]$events.Add(@{ kind = 'incident-reminder'; forward = $true })
                    }
                    elseif ($prevStatus -ne $Observation.Status) {
                        # Status changed while open (e.g. stale -> failed): record it locally.
                        [void]$events.Add(@{ kind = 'observation'; forward = $false })
                    }
                }
                else {
                    # Bad but below OpenAfter: record the first sighting only.
                    if ($s.consecutive_bad -eq 1) { [void]$events.Add(@{ kind = 'observation'; forward = $false }) }
                }
            }
            else {
                $s.consecutive_good = [int]$s.consecutive_good + 1
                $s.consecutive_bad = 0
                $s.last_status = 'ok'
                if ($s.state -eq 'open' -and $s.consecutive_good -ge $Observation.ResolveAfter) {
                    $s.state = 'resolved'
                    $s.resolved_at = ConvertTo-FleetIso $now
                    if ($flapping) {
                        $ctx.transition = 'suppressed'
                        [void]$events.Add(@{ kind = 'incident-resolved'; forward = $false })
                    }
                    else {
                        $ctx.transition = 'resolved'
                        [void]$events.Add(@{ kind = 'incident-resolved'; forward = $true })
                    }
                }
                elseif ($s.state -eq 'open') { $ctx.transition = 'still-open' }
                else { $ctx.transition = 'quiet' }
            }
            # The events go into the durable outbox BEFORE the state that implies them is committed. Written
            # after the commit, a failed ledger append lost the opening alert until the 24 h reminder, and a
            # resolution for good (third review of #346). They are staged as .pending, the state records their
            # ids, and only after the state write succeeds do they become sendable: a transition whose state
            # write failed was otherwise sent and then queued again (sweep row 15).
            $pending = @(); $ids = @()
            foreach ($e in $events) {
                $status = $Observation.Status
                $severity = $Observation.Severity
                if ($e.kind -eq 'incident-resolved') { $status = 'ok'; $severity = 'info' }
                $msg = $Observation.Message
                if ($e.kind -eq 'incident-resolved' -and [string]::IsNullOrWhiteSpace($msg)) { $msg = 'recovered' }
                $ev = New-FleetLedgerEvent -Source $Observation.Source -Name $Observation.Name -Check $Observation.Check `
                    -Kind $e.kind -Status $status -Severity $severity -Message $msg -LogPath $Observation.LogPath `
                    -ExitCode $Observation.ExitCode -Start $Observation.Start -End $Observation.End -IncidentKey $key -Details $Observation.Details
                $pending += Add-FleetIncidentOutbox -LedgerEvent $ev -Forward ([bool]$e.forward) -Pending
                $ids += [string]$ev.event_id
            }
            # Events this key committed earlier but that an interrupted run left staged are promoted HERE, under the
            # lock, before their marker is replaced: overwriting committed_events first made recovery treat them as
            # uncommitted and set them aside unsent (review of b5f31a5c).
            $outboxDir = Join-Path (Get-FleetPath -Kind state) 'outbox'
            foreach ($prior in @($s.committed_events)) {
                if ([string]::IsNullOrWhiteSpace([string]$prior)) { continue }
                foreach ($pf in @(Get-ChildItem -LiteralPath $outboxDir -Filter ('*-{0}.pending' -f $prior) -File -ErrorAction SilentlyContinue)) {
                    Move-Item -LiteralPath $pf.FullName -Destination ($pf.FullName -replace '\.pending$', '.json') -Force -ErrorAction Stop
                }
            }
            $s.committed_events = @($ids)
            $all[$key] = $s
            try { Write-FleetJsonFile -Path $statePath -Value $all }
            catch {
                # Not committed, so these transitions never happened; the next observation recomputes them.
                foreach ($p in $pending) { Remove-Item -LiteralPath $p -Force -ErrorAction SilentlyContinue }
                throw
            }
            $ctx.committed = $true
            foreach ($p in $pending) { Move-Item -LiteralPath $p -Destination ($p -replace '\.pending$', '.json') -Force -ErrorAction Stop }
        }
        if (-not $acquired) { throw "fleet incidents: could not lock the incident store; observation for $key not applied" }
        }
        catch {
            if ($bad -and -not $ctx.committed -and @('open', 'muted') -notcontains $ctx.known) { Send-FleetUnsavedAlert -Observation $Observation -Key $key }
            throw
        }
        $transition = $ctx.transition
        Invoke-FleetIncidentOutboxFlush
        return [pscustomobject]@{ Key = $key; Transition = $transition; Bad = $bad }
    }
}

function Send-FleetUnsavedAlert {
    <#
    The alert for a bad observation whose incident store could not be read or saved (a full disk, a broken store):
    one incident-open with details.incident_state = 'unsaved', without the OpenAfter threshold (the bad count cannot
    advance unsaved). It repeats on every sweep while the store stays unsaved: repetition over silence. It never
    depends on local storage: the append and the forward are separate steps, so a failed append still forwards, and
    the forward runs exactly once, since it sends before it records the delivery and a failed record must not send
    again (review of d5d030fd). Never throws. Mirrors sentinel-utils #1600 (_fl_observe_unsaved).
    #>
    param([Parameter(Mandatory)]$Observation, [Parameter(Mandatory)][string]$Key)
    try {
        $details = @{}
        if ($Observation.Details) { foreach ($k in @($Observation.Details.Keys)) { $details[$k] = $Observation.Details[$k] } }
        $details['incident_state'] = 'unsaved'
        $ev = New-FleetLedgerEvent -Source $Observation.Source -Name $Observation.Name -Check $Observation.Check `
            -Kind incident-open -Status $Observation.Status -Severity $Observation.Severity `
            -Message ('incident state could not be saved, so this alert repeats each sweep: ' + $Observation.Message) `
            -LogPath $Observation.LogPath -IncidentKey $Key -Details $details
        try { $null = Write-FleetLedgerEvent -LedgerEvent $ev -NoForward }
        catch { Write-Warning "fleet incidents: unsaved alert for $Key not appended ($($_.Exception.Message)); forwarding it anyway" }
        try { $null = Invoke-FleetEventForward -LedgerEvent $ev }
        catch { Write-Warning "fleet incidents: unsaved alert for $Key forwarded, but its delivery was not recorded: $($_.Exception.Message)" }
    }
    catch { Write-Warning "fleet incidents: could not send the unsaved alert for ${Key}: $($_.Exception.Message)" }
}

function Add-FleetIncidentOutbox {
    <#
    One file per event, named so that a sorted listing is the order the events happened in. -Pending stages it
    as <name>.pending, which the flush does not send until the incident state that records it is committed.
    Returns the file path.
    #>
    param([Parameter(Mandatory)][System.Collections.IDictionary]$LedgerEvent, [bool]$Forward, [switch]$Pending)
    $dir = Join-Path (Get-FleetPath -Kind state) 'outbox'
    if (-not (Test-Path -LiteralPath $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
    $name = '{0:D19}-{1}.{2}' -f [datetime]::UtcNow.Ticks, $LedgerEvent.event_id, $(if ($Pending) { 'pending' } else { 'json' })
    # The alert settings in effect NOW travel with the event: a retry under another call's -PolicyPath (or
    # the default) sent it to another channel or skipped it (fourteenth review of #346).
    $policy = $script:FleetEffectivePolicy
    if (-not $policy) { try { $policy = Get-FleetPolicy } catch { $policy = $null } }
    $entry = @{ forward = $Forward; event = $LedgerEvent; alerts_captured = ($null -ne $policy); alerts = $null }
    if ($policy) { $entry.alerts = $policy.alerts }
    $path = Join-Path $dir $name
    Write-FleetJsonFile -Path $path -Value $entry
    return $path
}

function Invoke-FleetIncidentOutboxFlush {
    <#
    Writes pending incident events to the ledger (and forwards them) in order, deleting each one only
    after it was written. Stops at the first failure so later events wait behind it; never throws.
    One flusher at a time: a concurrent caller skips instead of waiting.
    #>
    $dir = Join-Path (Get-FleetPath -Kind state -NoCreate) 'outbox'
    if (-not (Test-Path -LiteralPath $dir)) { return }
    $order = @((New-FleetLedgerEvent -Source sweeper -Name order -Check order -Kind observation -Status ok -Severity info -Message order).Keys)
    $null = Invoke-FleetWithLock -LockName 'incident-outbox' -NoWait -ScriptBlock {
        # Staged entries left by a process that died between its state commit and making them sendable: promote
        # those the state records as committed; an uncommitted one older than ten minutes is kept aside, unsent.
        foreach ($p in @(Get-ChildItem -LiteralPath $dir -Filter '*.pending' -File | Sort-Object Name)) {
            if ($script:FleetReadOnly) { break }
            try {
                $staged = Read-FleetJsonFile -Path $p.FullName -ThrowOnError
                $store = Read-FleetIncidentStore
                $st = $store[[string]$staged.event.incident_key]
                if ($st -and (@($st.committed_events) -contains [string]$staged.event.event_id)) {
                    Move-Item -LiteralPath $p.FullName -Destination ($p.FullName -replace '\.pending$', '.json') -Force -ErrorAction Stop
                }
                elseif ($p.LastWriteTimeUtc -lt [datetime]::UtcNow.AddMinutes(-10)) {
                    $u = Join-Path $dir 'uncommitted'
                    if (-not (Test-Path -LiteralPath $u)) { New-Item -ItemType Directory -Path $u -Force | Out-Null }
                    Move-Item -LiteralPath $p.FullName -Destination (Join-Path $u $p.Name) -Force -ErrorAction Stop
                }
            }
            catch { Write-Warning ("fleet incidents: staged outbox entry {0} left in place: {1}" -f $p.Name, $_.Exception.Message) }
        }
        foreach ($f in @(Get-ChildItem -LiteralPath $dir -Filter '*.json' -File | Sort-Object Name)) {
            # An entry that cannot be read is never deleted: it is quarantined for inspection (left in place in
            # read-only mode), and the entries behind it still flush.
            try { $item = Read-FleetJsonFile -Path $f.FullName -ThrowOnError } catch { $item = $null }
            if ($null -eq $item -or $null -eq $item.event) {
                if ($script:FleetReadOnly) { continue }
                $q = Join-Path $dir 'corrupt'
                try {
                    if (-not (Test-Path -LiteralPath $q)) { New-Item -ItemType Directory -Path $q -Force | Out-Null }
                    Move-Item -LiteralPath $f.FullName -Destination (Join-Path $q $f.Name) -Force -ErrorAction Stop
                    Write-Warning ("fleet incidents: unreadable outbox entry quarantined: {0}" -f (Join-Path $q $f.Name))
                }
                catch { Write-Warning ("fleet incidents: unreadable outbox entry left in place: {0}: {1}" -f $f.FullName, $_.Exception.Message) }
                continue
            }
            $ev = [ordered]@{}
            foreach ($k in $order) { $ev[$k] = $item.event[$k] }
            $alertPolicy = $null
            if ($item.alerts_captured) { $alertPolicy = @{ alerts = $item.alerts } }
            $delivery = 'not-forwarded'
            try {
                if ($item.ledger_written) {
                    # Already in the ledger; only the forward failed last time. Never append it twice.
                    $delivery = Invoke-FleetEventForward -LedgerEvent $ev -AlertPolicy $alertPolicy
                }
                else {
                    $null = Write-FleetLedgerEvent -LedgerEvent $ev -NoForward:(-not $item.forward) -AlertPolicy $alertPolicy
                    $delivery = $script:FleetLastDelivery
                }
            }
            catch { Write-Warning ("fleet incidents: event {0} kept for retry: {1}" -f $ev.event_id, $_.Exception.Message); break }
            if ($item.forward -and $delivery -eq 'failed') {
                # Written, but nobody was told: the bus send AND its spool failed and Discord did not send. Keep it
                # for a forward-only retry instead of retiring an alert that never left the host (data-loss item).
                $item.ledger_written = $true
                $item.forward_attempts = 1 + [int]$item.forward_attempts
                try { Write-FleetJsonFile -Path $f.FullName -Value $item } catch { Write-Warning ("fleet incidents: could not mark {0} for retry: {1}" -f $f.Name, $_.Exception.Message) }
                Write-Warning ("fleet incidents: event {0} written but not delivered; kept for a forward-only retry (attempt {1})" -f $ev.event_id, $item.forward_attempts)
                break
            }
            Remove-Item -LiteralPath $f.FullName -Force -ErrorAction SilentlyContinue
        }
    }
}

function Resolve-FleetUnobservedIncidents {
    <#
    An open incident whose subject vanished (task deleted, orphan reaped, runner removed) is never
    observed again and would stay open forever. After a sweep, every open incident from a detector
    that completed this sweep, but was not observed in it, receives one good observation.
    #>
    param([string[]]$ObservedKeys, [string[]]$CompletedSources, [hashtable]$Policy)
    $hostName = Get-FleetHostName
    $all = Get-FleetIncidentState
    $results = @()
    foreach ($key in @($all.Keys)) {
        $s = $all[$key]
        if ($s.state -ne 'open') { continue }
        if ($ObservedKeys -contains $key) { continue }
        # Only the wrapper reports this key (task outside task_scope, or a -Name that is not the task path):
        # the sweeper's silence about it proves nothing, so leave it to the wrapper's next run.
        if ($s.last_origin -eq 'wrapper') { continue }
        $parts = $key.Split('|')
        if ($parts.Count -lt 4 -or $parts[0] -ne $hostName) { continue }
        if ($CompletedSources -notcontains $parts[1]) { continue }
        $name = ($parts[2..($parts.Count - 2)] -join '|')
        # A task outside this sweep's task_scope (or excluded) was not looked at, so its absence proves nothing.
        if ($parts[1] -eq 'task' -and $Policy -and -not (Test-FleetTaskInScope -Policy $Policy -FullName $name)) { continue }
        $obs = New-FleetObservation -Source $parts[1] -Name $name -Check $parts[-1] -Status ok -Message 'no longer observed by its detector' -ResolveAfter 2
        $results += Submit-FleetObservation -Observation $obs
    }
    return $results
}
