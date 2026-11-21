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
    $path = Join-Path (Get-FleetPath -Kind state -NoCreate) 'incidents.json'
    return (Read-FleetJsonFile -Path $path -Default @{})
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
        $statePath = Join-Path (Get-FleetPath -Kind state) 'incidents.json'
        $bad = @('ok', 'skipped') -notcontains $Observation.Status
        $events = New-Object System.Collections.ArrayList
        $ctx = @{ transition = 'quiet' }

        $acquired = $false
        $null = Invoke-FleetWithLock -LockName 'incidents' -TimeoutSeconds 30 -Acquired ([ref]$acquired) -ScriptBlock {
            $all = Read-FleetJsonFile -Path $statePath -Default @{}
            $s = $all[$key]
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
            # resolution for good (third review of #346). The outbox is flushed below and on every later call.
            foreach ($e in $events) {
                $status = $Observation.Status
                $severity = $Observation.Severity
                if ($e.kind -eq 'incident-resolved') { $status = 'ok'; $severity = 'info' }
                $msg = $Observation.Message
                if ($e.kind -eq 'incident-resolved' -and [string]::IsNullOrWhiteSpace($msg)) { $msg = 'recovered' }
                $ev = New-FleetLedgerEvent -Source $Observation.Source -Name $Observation.Name -Check $Observation.Check `
                    -Kind $e.kind -Status $status -Severity $severity -Message $msg -LogPath $Observation.LogPath `
                    -ExitCode $Observation.ExitCode -Start $Observation.Start -End $Observation.End -IncidentKey $key -Details $Observation.Details
                Add-FleetIncidentOutbox -LedgerEvent $ev -Forward ([bool]$e.forward)
            }
            $all[$key] = $s
            Write-FleetJsonFile -Path $statePath -Value $all
        }
        if (-not $acquired) { throw "fleet incidents: could not lock the incident store; observation for $key not applied" }
        $transition = $ctx.transition
        Invoke-FleetIncidentOutboxFlush
        return [pscustomobject]@{ Key = $key; Transition = $transition; Bad = $bad }
    }
}

function Add-FleetIncidentOutbox {
    <# One file per event, named so that a sorted listing is the order the events happened in. #>
    param([Parameter(Mandatory)][System.Collections.IDictionary]$LedgerEvent, [bool]$Forward)
    $dir = Join-Path (Get-FleetPath -Kind state) 'outbox'
    if (-not (Test-Path -LiteralPath $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
    $name = '{0:D19}-{1}.json' -f [datetime]::UtcNow.Ticks, $LedgerEvent.event_id
    # The alert settings in effect NOW travel with the event: a retry under another call's -PolicyPath (or
    # the default) sent it to another channel or skipped it (fourteenth review of #346).
    $policy = $script:FleetEffectivePolicy
    if (-not $policy) { try { $policy = Get-FleetPolicy } catch { $policy = $null } }
    $entry = @{ forward = $Forward; event = $LedgerEvent; alerts_captured = ($null -ne $policy); alerts = $null }
    if ($policy) { $entry.alerts = $policy.alerts }
    Write-FleetJsonFile -Path (Join-Path $dir $name) -Value $entry
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
            try { $null = Write-FleetLedgerEvent -LedgerEvent $ev -NoForward:(-not $item.forward) -AlertPolicy $alertPolicy }
            catch { Write-Warning ("fleet incidents: event {0} kept for retry: {1}" -f $ev.event_id, $_.Exception.Message); break }
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
