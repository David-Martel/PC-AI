#Requires -Version 5.1
<#
Semantic health detectors. Each returns observations (New-FleetObservation) for healthy AND
unhealthy subjects, so the incident store can resolve as well as open. Every external read goes
through one Get-Fleet*Data function so tests can mock it and prove each detector fires.

The scheduled-task axis reuses Tools/Test-ScheduledTaskHealth.ps1 (result, staleness, dtm-task-run
launcher status) instead of re-deriving it, and adds only what it does not cover: Interactive tasks
with nobody logged on, exit 0 with ERROR/STALE in the log, tasks that must be enabled, and
known-benign result codes.
#>

# The staleness reasons Test-ScheduledTaskHealth.ps1 adds (its Axis 2 and launcher-status checks). A failed
# task's overall status hides them, so the staleness axis reads them directly (sweep row 4).
$script:FleetStaleReasonPattern = '^(Last ran .* but its cadence|Has a .* cadence, has never run|Triggers on boot or logon but|Launcher status is .* old|Launcher status predates|Task Scheduler last ran it)'

# ---------------------------------------------------------------- data sources (mocked in tests)

function Get-FleetTaskHealthData {
    <# Runs Test-ScheduledTaskHealth.ps1 in a child pwsh and returns its parsed report. #>
    param([hashtable]$Policy)
    $script = [IO.Path]::GetFullPath([string]$Policy.task_health_script)
    if (-not (Test-Path -LiteralPath $script)) { throw "task health script not found: $script" }
    # A dry run writes nothing under the home: the child's report goes to a uniquely named temp file that is
    # removed afterwards (sweep row 23; the child can only report to a file).
    if ($script:FleetReadOnly) { $out = Join-Path ([IO.Path]::GetTempPath()) ('fleet-tsh-{0}.json' -f [guid]::NewGuid().ToString('N')) }
    else { $out = Join-Path (Get-FleetPath -Kind state) 'scheduled-task-health.json' }
    # Delete the previous report first: a child that fails after a good sweep must not leave the old
    # report looking current (codex review, #346).
    if (Test-Path -LiteralPath $out) { Remove-Item -LiteralPath $out -Force }
    $pwsh = (Get-Process -Id $PID).Path
    if ($pwsh -notmatch '(?i)pwsh(\.exe)?$') {
        $cmd = Get-Command pwsh -ErrorAction SilentlyContinue
        if ($cmd) { $pwsh = $cmd.Source } else { throw 'Test-ScheduledTaskHealth.ps1 needs PowerShell 7 (pwsh) and none was found' }
    }
    try {
        $r = Invoke-FleetProcess -FilePath $pwsh -ArgumentList @('-NoProfile', '-NonInteractive', '-File', $script, '-OutputJson', $out, '-StaleFactor', [string]$Policy.task_stale_factor) -TimeoutSeconds 180
        # Exit 0 = report written (no -FailOnIssue is passed); anything else, or no fresh report, is a failure.
        if ($r.TimedOut -or $r.ExitCode -ne 0 -or -not (Test-Path -LiteralPath $out)) {
            throw "task health script failed (exit $($r.ExitCode), timed out=$($r.TimedOut), report written=$(Test-Path -LiteralPath $out))"
        }
        return (ConvertTo-FleetHashtable ([IO.File]::ReadAllText($out) | ConvertFrom-Json))
    }
    finally { if ($script:FleetReadOnly) { Remove-Item -LiteralPath $out -Force -ErrorAction SilentlyContinue } }
}

function Get-FleetScheduledTaskData {
    <# Principal logon type per task, for the Interactive-without-session check. #>
    $rows = @()
    foreach ($t in @(Get-ScheduledTask -ErrorAction Stop)) {
        $rows += [pscustomobject]@{
            FullName  = '{0}{1}' -f $t.TaskPath, $t.TaskName
            TaskPath  = $t.TaskPath
            TaskName  = $t.TaskName
            State     = [string]$t.State
            LogonType = [string]$t.Principal.LogonType
            UserId    = [string]$t.Principal.UserId
        }
    }
    return $rows
}

function Get-FleetInteractiveUserData {
    <# Users with an interactive desktop right now (owners of explorer.exe), lowercase 'domain\user'. #>
    $users = @()
    # A failed query is not "nobody is logged on": raise it, so the sweep records a detector error.
    $cimErrors = @()
    $shells = @(Get-CimInstance Win32_Process -Filter "Name='explorer.exe'" -ErrorAction SilentlyContinue -ErrorVariable cimErrors)
    if (@($cimErrors).Count -gt 0) { throw "interactive-session query failed: $($cimErrors[0].Exception.Message)" }
    foreach ($p in $shells) {
        $o = Invoke-CimMethod -InputObject $p -MethodName GetOwner -ErrorAction SilentlyContinue
        if ($o -and $o.User) { $users += ('{0}\{1}' -f $o.Domain, $o.User).ToLowerInvariant() }
    }
    return @($users | Sort-Object -Unique)
}

function Get-FleetDriveData {
    $rows = @()
    foreach ($d in [IO.DriveInfo]::GetDrives()) {
        if ($d.DriveType -ne 'Fixed' -or -not $d.IsReady) { continue }
        $rows += [pscustomobject]@{ Name = $d.Name.Substring(0, 1).ToUpperInvariant(); FreeBytes = [int64]$d.AvailableFreeSpace; TotalBytes = [int64]$d.TotalSize }
    }
    return $rows
}

function Invoke-FleetGhApi {
    <#
    gh api GET; returns parsed JSON (array pages flattened). -NoPaginate for "newest N" queries:
    --paginate with per_page=1 walks a repository's whole run history page by page and timed out.
    #>
    param([Parameter(Mandatory)][string]$Path, [switch]$NoPaginate)
    $gh = (Get-Command gh -ErrorAction SilentlyContinue)
    if (-not $gh) { throw 'gh CLI not found' }
    $ghArgs = @('api')
    if (-not $NoPaginate) { $ghArgs += '--paginate' }
    $r = Invoke-FleetProcess -FilePath $gh.Source -ArgumentList ($ghArgs + $Path) -TimeoutSeconds 60
    if ($r.TimedOut) { throw "gh api $Path timed out after 60 s" }
    if ($r.ExitCode -ne 0) { throw "gh api $Path failed (exit $($r.ExitCode)): $($r.StdErr.Trim())" }
    # --paginate concatenates JSON documents; split them by re-reading each top-level value.
    $text = $r.StdOut.Trim()
    $docs = @()
    if ($text -match '^\{') { $text = '[' + ($text -replace '\}\s*\{', '},{') + ']' }
    elseif ($text -match '^\[') { $text = '[' + ($text -replace '\]\s*\[', '],[') + ']' }
    foreach ($d in @($text | ConvertFrom-Json | ForEach-Object { $_ })) { $docs += $d }
    return (ConvertTo-FleetHashtable $docs)
}

function Get-FleetDrainState {
    <#
    Whether a planned DRAIN (vct-nightly compaction) has the runners down on purpose, from the drain
    agent's newest lifecycle message on agent-bus: '<agent> START ...' (active), '... ended ...' or
    '... restored cleanly ...' (over), or 'ALERT ...' (state unknown: e.g. a run died before its restore).
    Matching is case-SENSITIVE: case-insensitive 'START' matched "restart" inside an ALERT and read a
    dead run as an active DRAIN. Returns @{ Active; StartedAt; Minutes; Alert; Unknown; Error }.
    A failed, timed-out or unparseable read is Unknown, never "no DRAIN": reading it as inactive let an
    ongoing overrun resolve during a bus outage (thirteenth review of #346).
    #>
    param([hashtable]$Policy)
    $agent = [string]$Policy.drain.bus_agent
    $none = @{ Active = $false; StartedAt = $null; Minutes = 0; Alert = $null; Unknown = $false; Error = $null }
    if ([string]::IsNullOrWhiteSpace($agent)) { return $none }
    $unknown = { param($why) @{ Active = $false; StartedAt = $null; Minutes = 0; Alert = $null; Unknown = $true; Error = $why } }
    $r = Invoke-FleetProcess -FilePath (Get-FleetAgentBusPath) -ArgumentList @('read', '--from-agent', $agent, '--since-minutes', '600', '--limit', '20', '--encoding', 'json') -TimeoutSeconds 20
    if ($r.TimedOut) { return (& $unknown 'agent-bus read timed out after 20 s') }
    if ($r.ExitCode -ne 0) { return (& $unknown ("agent-bus read failed (exit {0}): {1}" -f $r.ExitCode, ([string]$r.StdErr).Trim())) }
    try { $msgs = @($r.StdOut | ConvertFrom-Json -ErrorAction Stop | ForEach-Object { $_ }) }   # 5.1 emits a JSON array as one object
    catch { return (& $unknown ("agent-bus read returned unparseable output: {0}" -f $_.Exception.Message)) }
    if ($msgs.Count -eq 1 -and $msgs[0].PSObject.Properties.Name -contains 'messages') { $msgs = @($msgs[0].messages) }
    $life = @($msgs | Where-Object { $_.body -cmatch '^ALERT\b' -or $_.body -cmatch '\bSTART\b' -or $_.body -cmatch '\bended\b|restored cleanly' } |
            Sort-Object { try { ConvertFrom-FleetIso $_.timestamp_utc } catch { [datetime]::MinValue } })
    # The read window is bounded (600 min), so the newest lifecycle state is remembered per agent: an empty
    # window after a START is "still running", not "no DRAIN". Forgetting it resolved an overrun once its
    # START aged past ten hours, with no end message (fourteenth review of #346).
    $statePath = Join-Path (Get-FleetPath -Kind state -NoCreate) ('drain-{0}.json' -f ($agent -replace '[^A-Za-z0-9_.-]', '_'))
    if ($life.Count -eq 0) {
        try { $known = Read-FleetJsonFile -Path $statePath -Default $null -ThrowOnError }
        catch { return (& $unknown ("remembered DRAIN state unreadable: {0}" -f $_.Exception.Message)) }
        if ($null -eq $known) { return $none }
        if ($known.active) {
            $started = ConvertFrom-FleetIso ([string]$known.started_at)
            if ($null -eq $started) { return (& $unknown 'remembered DRAIN state has no start time') }
            return @{ Active = $true; StartedAt = $started; Minutes = [int]((Get-FleetNow) - $started).TotalMinutes; Alert = $null; Unknown = $false; Error = $null }
        }
        if ($known.alert) { $none.Alert = [string]$known.alert }
        return $none
    }
    $last = $life[-1]
    $body = [string]$last.body
    $result = $none
    if ($body -cmatch '^ALERT\b') { $none.Alert = $body.Substring(0, [Math]::Min(240, $body.Length)) }
    elseif ($body -cmatch '\bended\b|restored cleanly') { }
    else {
        $started = ConvertFrom-FleetIso $last.timestamp_utc
        $result = @{ Active = $true; StartedAt = $started; Minutes = [int]((Get-FleetNow) - $started).TotalMinutes; Alert = $null; Unknown = $false; Error = $null }
    }
    if (-not $script:FleetReadOnly) {
        $startedIso = $null; if ($result.Active) { $startedIso = ConvertTo-FleetIso $result.StartedAt }
        Write-FleetJsonFile -Path $statePath -Value @{ agent = $agent; active = [bool]$result.Active; started_at = $startedIso; alert = $result.Alert; seen_at = [string]$last.timestamp_utc }
    }
    return $result
}

function Get-FleetTcpListenerConnectionCount {
    <# Established server-side connections on a local port (each client connection counted once). #>
    param([int]$Port)
    # "No matching connections" is a real zero; any other failure must surface as a detector error,
    # not as an idle server (fourth review of #346).
    # Errors are collected rather than raised so both kinds are seen however they arrive.
    $queryErrors = @()
    $rows = @(Get-NetTCPConnection -LocalPort $Port -State Established -ErrorAction SilentlyContinue -ErrorVariable queryErrors)
    $real = @($queryErrors | Where-Object { $_.CategoryInfo.Category -ne 'ObjectNotFound' -and $_.FullyQualifiedErrorId -notlike 'CmdletizationQuery_NotFound*' })
    if ($real.Count -gt 0) { throw "connection query for port $Port failed: $($real[0].Exception.Message)" }
    return $rows.Count
}

function Test-FleetTcpListening {
    <# Whether anything listens on a local TCP port. "No listener" is a real answer; any other query failure throws. #>
    param([int]$Port)
    $queryErrors = @()
    $rows = @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue -ErrorVariable queryErrors)
    $real = @($queryErrors | Where-Object { $_.CategoryInfo.Category -ne 'ObjectNotFound' -and $_.FullyQualifiedErrorId -notlike 'CmdletizationQuery_NotFound*' })
    if ($real.Count -gt 0) { throw "listener query for port $Port failed: $($real[0].Exception.Message)" }
    return ($rows.Count -gt 0)
}

function Get-FleetPostgresLiveMaxConnections {
    <#
    The server's live max_connections via psql (SHOW max_connections), when it can be read without a password
    prompt (-w) within 10 s; $null when it cannot. Only used to report drift from the policy value.
    #>
    param([Parameter(Mandatory)][hashtable]$Pg)
    $psql = [string]$Pg.psql
    if (-not $psql) { $cmd = Get-Command psql -ErrorAction SilentlyContinue; if ($cmd) { $psql = $cmd.Source } }
    if (-not $psql) { return $null }
    $user = 'postgres'; if ($Pg.user) { $user = [string]$Pg.user }
    $r = Invoke-FleetProcess -FilePath $psql -ArgumentList @('-w', '-X', '-A', '-t', '-h', '127.0.0.1', '-p', [string]$Pg.port, '-U', $user, '-d', 'postgres', '-c', 'SHOW max_connections') -TimeoutSeconds 10
    if ($r.TimedOut -or $r.ExitCode -ne 0) { return $null }
    $v = ([string]$r.StdOut).Trim()
    if ($v -notmatch '^\d+$') { return $null }
    return [int]$v
}

function Get-FleetProcessData {
    $rows = @()
    # A failed enumeration must not read as "no processes, no orphans" (fourth review of #346).
    $cimErrors = @()
    $all = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue -ErrorVariable cimErrors)
    if (@($cimErrors).Count -gt 0) { throw "process enumeration failed: $($cimErrors[0].Exception.Message)" }
    if ($all.Count -eq 0) { throw 'process enumeration returned nothing' }
    foreach ($p in $all) {
        $rows += [pscustomobject]@{ ProcessId = [int]$p.ProcessId; ParentProcessId = [int]$p.ParentProcessId; Name = [string]$p.Name; CreationDate = $p.CreationDate }
    }
    return $rows
}

function Get-FleetRemoteCertificate {
    <# Leaf certificate of host:port and whether its chain validated. Never sends application data. #>
    param([string]$HostName, [int]$Port = 443)
    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $iar = $client.BeginConnect($HostName, $Port, $null, $null)
        if (-not $iar.AsyncWaitHandle.WaitOne(10000)) { throw "connect timeout to ${HostName}:$Port" }
        $client.EndConnect($iar)
        $script:FleetLastCertErrors = 'None'
        $cb = [System.Net.Security.RemoteCertificateValidationCallback] { param($s, $c, $ch, $e) $script:FleetLastCertErrors = [string]$e; return $true }
        $ssl = New-Object System.Net.Security.SslStream($client.GetStream(), $false, $cb)
        $ssl.ReadTimeout = 10000
        $ssl.AuthenticateAsClient($HostName)
        $cert = New-Object System.Security.Cryptography.X509Certificates.X509Certificate2($ssl.RemoteCertificate)
        $ssl.Dispose()
        return [pscustomobject]@{ NotAfter = $cert.NotAfter.ToUniversalTime(); Subject = $cert.Subject; PolicyErrors = $script:FleetLastCertErrors }
    }
    finally { $client.Dispose() }
}

# ---------------------------------------------------------------- detectors

function Test-FleetWildcard {
    param([string]$Value, [object[]]$Patterns)
    foreach ($p in @($Patterns)) { if ($p -and $Value -like [string]$p) { return $true } }
    return $false
}

function Test-FleetTaskInScope {
    <#
    Whether the policy's task_scope / task_exclude cover a task. The resolver uses it too: a sweep can
    vouch only for tasks it was allowed to look at (fourth review of #346).
    #>
    param([Parameter(Mandatory)][hashtable]$Policy, [Parameter(Mandatory)][string]$FullName)
    if (@($Policy.task_scope | Where-Object { $_ }).Count -gt 0 -and -not (Test-FleetWildcard $FullName $Policy.task_scope)) { return $false }
    return -not (Test-FleetWildcard $FullName $Policy.task_exclude)
}

function Get-FleetTaskObservations {
    [CmdletBinding()]
    param([Parameter(Mandatory)][hashtable]$Policy)
    $obs = @()
    $report = Get-FleetTaskHealthData -Policy $Policy
    $okCodes = @($Policy.task_ok_codes | ForEach-Object { ([string]$_).ToUpperInvariant() })
    $sevMap = $Policy.task_severity
    $principals = @(Get-FleetScheduledTaskData)
    # Read once: an unreadable store throws (detector error) rather than reading as "no incidents".
    $incidentState = Get-FleetIncidentState
    $hostName = Get-FleetHostName
    # A report without tasks while in-scope tasks are registered is a broken report, not "every task gone":
    # producing no task observations let the stale-key resolver resolve every open task incident (sweep row 1).
    $hasTasks = ($report -is [System.Collections.IDictionary] -and $report.Contains('Tasks')) -or ($null -ne $report -and $report.PSObject.Properties.Name -contains 'Tasks')
    if (-not $hasTasks) { throw 'task health report has no Tasks list' }
    $tasks = @($report.Tasks | Where-Object { $null -ne $_ })
    if ($tasks.Count -eq 0) {
        $registered = @($principals | Where-Object { Test-FleetTaskInScope -Policy $Policy -FullName ([string]$_.FullName) })
        if ($registered.Count -gt 0) { throw ("task health report lists no tasks, but {0} in-scope task(s) are registered (e.g. {1})" -f $registered.Count, $registered[0].FullName) }
    }
    $byName = @{}
    foreach ($t in $tasks) {
        $full = '{0}{1}' -f $t.TaskPath, $t.TaskName
        $byName[$full] = $t
        $included = ($t.Owner -eq 'Local') -or (Test-FleetWildcard $full $Policy.task_include)
        # task_scope narrows a staged rollout to the named tasks only (empty = no narrowing).
        if (@($Policy.task_scope | Where-Object { $_ }).Count -gt 0) { $included = Test-FleetWildcard $full $Policy.task_scope }
        if (-not $included -or (Test-FleetWildcard $full $Policy.task_exclude)) { continue }
        $sev = 'medium'
        foreach ($k in @($sevMap.Keys)) { if ($full -like $k) { $sev = [string]$sevMap[$k] } }
        $reasons = (@($t.Reasons) -join '; ')
        # TSH reports LastResult as an int64. A benign code (IgnoreNew keepalives 0x800710E0, policy
        # task_ok_codes) downgrades Failed only when that code is the failure; with 0x0 the Failed
        # came from the launcher axis and must stand.
        $resultHex = $null
        if ($null -ne $t.LastResult -and "$($t.LastResult)" -ne '') { $resultHex = ('0x{0:X}' -f [int64]$t.LastResult).ToUpperInvariant() }
        # Per-task benign codes, e.g. long-running forwarders that end with 0xC000013A at logoff.
        $taskOk = @($okCodes)
        $byTask = $Policy.task_ok_codes_by_task
        if ($byTask) { foreach ($k in @($byTask.Keys)) { if ($full -like $k) { $taskOk += @($byTask[$k] | ForEach-Object { ([string]$_).ToUpperInvariant() }) } } }
        # Benign only when the result code is the ONLY reason for Failed. Test-ScheduledTaskHealth words its
        # result-axis reason "Last run <code> ..."; any other reason (a launcher timeout or exit, an unreadable
        # status file, an unavailable task) is an independent failure and must stand (second review of #346).
        $otherReasons = @(@($t.Reasons) | Where-Object { $_ -and ([string]$_) -notlike 'Last run *' })
        $benign = ($null -ne $resultHex) -and ($resultHex -ne '0X0') -and ($taskOk -contains $resultHex) -and ($otherReasons.Count -eq 0)
        $rerun = $null
        if (Test-FleetWildcard $full $Policy.rerun_tasks) { $rerun = 'rerun-task' }

        $status = [string]$t.Status
        if (@('Disabled', 'Ignored') -contains $status) { continue }
        # Test-ScheduledTaskHealth.ps1 reports exactly Healthy / Failed / Stalled / Disabled / Ignored. Anything
        # else (a newer or broken report) used to fall through to "ok" on both axes and resolve open incidents:
        # it is neutral now, and the unreadable row is reported (sweep row 2).
        $known = @('Healthy', 'Failed', 'Stalled') -contains $status
        $obs += New-FleetObservation -Source task -Name $full -Check task-report -Severity low -ResolveAfter 1 `
            -Status $(if ($known) { 'ok' } else { 'failed' }) -Message $(if ($known) { "status $status" } else { "unrecognised task status '$status' in the task health report" })

        # Result axis -> check 'run' (the same key Invoke-FleetTask writes, so one incident per task).
        $runStatus = 'ok'
        if (-not $known) { $runStatus = 'skipped' }
        elseif ($status -eq 'Failed' -and -not $benign) { $runStatus = 'failed' }
        # Running (0x41301) or queued (0x41325): no completed result yet, so neither failure nor recovery.
        # As 'ok' it resolved the previous failure while its retry was still running (fourth review of #346).
        elseif ($resultHex -in @('0X41301', '0X41325')) { $runStatus = 'skipped' }
        # The wrapper's last capture of this task was incomplete (its output-capture incident is open): a clean Task
        # Scheduler result is not evidence of recovery until a run is captured in full (review of bcae775f).
        # Likewise while its configured log cannot be read (log-read open): the failure patterns were never checked
        # (review of a8570e8c).
        $captureNote = ''
        if ($runStatus -eq 'ok') {
            $cap = $incidentState[('{0}|task|{1}|output-capture' -f $hostName, $full)]
            $lr = $incidentState[('{0}|task|{1}|log-read' -f $hostName, $full)]
            if ($cap -and $cap.state -eq 'open') { $runStatus = 'skipped'; $captureNote = ' (not verified: the last wrapped run''s output was not fully captured)' }
            elseif ($lr -and $lr.state -eq 'open') { $runStatus = 'skipped'; $captureNote = ' (not verified: its configured log cannot be read)' }
        }
        $obs += New-FleetObservation -Source task -Name $full -Check run -Status $runStatus -Severity $sev `
            -Message (("last result {0} {1}; {2}" -f $t.LastResult, $t.ResultText, $reasons).Trim() + $captureNote) -ResolveAfter 1 `
            -Details @{ last_run = $t.LastRunTime; next_run = $t.NextRunTime; missed_runs = $t.MissedRuns } `
            -Playbook $rerun -PlaybookArgs @{ TaskPath = $t.TaskPath; TaskName = $t.TaskName }

        # Staleness axis. 'Failed' wins over 'Stalled' in the report's overall status, so a failed task's
        # staleness is read from its own reasons; without one, there is no evidence either way. It used to read
        # "ok" whenever the status was not Stalled, resolving a stale incident on a failed run (sweep row 4).
        $staleStatus = 'skipped'
        if ($status -eq 'Healthy') { $staleStatus = 'ok' }
        elseif ($status -eq 'Stalled') { $staleStatus = 'stale'; if ([string]$t.Staleness -like 'never run*') { $staleStatus = 'never-ran' } }
        elseif ($status -eq 'Failed' -and @(@($t.Reasons) | Where-Object { [string]$_ -match $script:FleetStaleReasonPattern }).Count -gt 0) {
            $staleStatus = 'stale'; if ([string]$t.Staleness -like 'never run*') { $staleStatus = 'never-ran' }
        }
        $obs += New-FleetObservation -Source task -Name $full -Check task-stale -Status $staleStatus -Severity $sev `
            -Message ("{0}; {1}" -f $t.Staleness, $reasons).Trim('; ') -ResolveAfter 1
    }

    # Tasks that must be enabled (an offsite backup task sat Disabled and never ran).
    foreach ($pattern in @($Policy.expect_enabled)) {
        # Staged rollouts: task_scope / task_exclude govern every task check (sixth review of #346), applied to
        # the CONCRETE tasks a wildcard matches, not to the wildcard text: '\backups\*' is out of a
        # '\backups\offsite-copy' scope as a string, but offsite-copy itself is in it (ninth review).
        $hits = @($tasks | Where-Object { ('{0}{1}' -f $_.TaskPath, $_.TaskName) -like $pattern } |
                Where-Object { Test-FleetTaskInScope -Policy $Policy -FullName ('{0}{1}' -f $_.TaskPath, $_.TaskName) })
        $allHits = @($tasks | Where-Object { ('{0}{1}' -f $_.TaskPath, $_.TaskName) -like $pattern })
        if ($hits.Count -eq 0 -and $allHits.Count -gt 0) { continue }   # matches exist, none in scope
        if ($hits.Count -eq 0 -and -not (Test-FleetTaskInScope -Policy $Policy -FullName ([string]$pattern))) { continue }
        if ($hits.Count -eq 0) {
            $obs += New-FleetObservation -Source task -Name ([string]$pattern) -Check task-expected -Status failed -Severity medium -Message 'expected task is not registered' -ResolveAfter 1
            continue
        }
        foreach ($t in $hits) {
            $full = '{0}{1}' -f $t.TaskPath, $t.TaskName
            $neverRan = [string]::IsNullOrWhiteSpace([string]$t.LastRunTime)
            $st = 'ok'; $msg = 'enabled'
            if ($t.State -eq 'Disabled') { $st = 'failed'; $msg = 'expected enabled but Disabled' }
            if ($neverRan) { $st = 'never-ran'; $msg = ('{0}; has never run' -f $msg) }
            $obs += New-FleetObservation -Source task -Name $full -Check task-expected -Status $st -Severity medium -Message $msg -ResolveAfter 1
        }
    }

    # Interactive tasks with no interactive session never fire (about 25 tasks on one host stopped for 11 days).
    $users = @(Get-FleetInteractiveUserData)
    foreach ($p in $principals) {
        if ($p.State -eq 'Disabled' -or $p.LogonType -notmatch '^Interactive') { continue }
        $t = $byName[$p.FullName]
        $included = ($t -and $t.Owner -eq 'Local') -or (Test-FleetWildcard $p.FullName $Policy.task_include)
        if (@($Policy.task_scope | Where-Object { $_ }).Count -gt 0) { $included = Test-FleetWildcard $p.FullName $Policy.task_scope }
        if (-not $included -or (Test-FleetWildcard $p.FullName $Policy.task_exclude)) { continue }
        $uid = ([string]$p.UserId).ToLowerInvariant()
        # Full identities when the task names a domain: 'WORKSTATION\build' at the desktop is not 'CONTOSO\build',
        # and matching on the user part alone cleared the no-session alert for a task that cannot run (sweep row 9).
        $present = $false
        if ($uid.Contains('\')) { $present = $users -contains $uid }
        else { foreach ($u in $users) { if ($u -eq $uid -or $u.EndsWith('\' + $uid)) { $present = $true } } }
        $st = 'ok'; $msg = 'interactive session present'
        if (-not $present) { $st = 'degraded'; $msg = ('LogonType Interactive and {0} has no interactive session: the task will not run (convert to S4U)' -f $p.UserId) }
        $obs += New-FleetObservation -Source task -Name $p.FullName -Check task-interactive-no-session -Status $st -Severity medium -Message $msg -ResolveAfter 1
    }

    # Exit 0 while the log says it failed (a database pull, a history export printing STALE, a logger).
    foreach ($rule in @($Policy.log_rules)) {
        $ruleName = [string]$rule.task; if (-not $ruleName) { $ruleName = [string]$rule.log }
        if (-not (Test-FleetTaskInScope -Policy $Policy -FullName $ruleName)) { continue }
        $obs += Get-FleetLogRuleObservation -Rule $rule -Tasks $byName
    }
    return $obs
}

function Read-FleetLogTail {
    <# The last Lines lines of a log. Throws on ANY read error (locked, denied, vanished): a failed read must not look like a clean log. #>
    param([Parameter(Mandatory)][string]$Path, [int]$Lines = 400)
    return @(Get-Content -LiteralPath $Path -Tail $Lines -ErrorAction Stop)
}

function Get-FleetLogRuleObservation {
    <#
    One log rule: { task, log (wildcard path), patterns[], tail_lines, severity }. Reads the newest
    matching log that was written at or after the task's last run, and reports 'failed' when a
    pattern matches in its tail. A log older than the last run proves nothing and is 'skipped'.
    #>
    param([hashtable]$Rule, [hashtable]$Tasks)
    $taskName = [string]$Rule.task
    $name = if ($taskName) { $taskName } else { [string]$Rule.log }
    $sev = 'medium'; if ($Rule.severity) { $sev = [string]$Rule.severity }
    $logGlob = Resolve-FleetPolicyPath ([string]$Rule.log)
    $files = @(Get-ChildItem -Path $logGlob -File -ErrorAction SilentlyContinue | Sort-Object LastWriteTimeUtc -Descending)
    if ($files.Count -eq 0) {
        # No log is no evidence, so the rule stays neutral. But a configured log that is missing is itself a
        # failure to report (the job writes nowhere, or the log moved), unless the task has never run yet.
        $skip = New-FleetObservation -Source task -Name $name -Check log-rule -Status skipped -Message "no log matches $($Rule.log)"
        $task = $null; if ($taskName) { $task = $Tasks[$taskName] }
        if ($task -and [string]::IsNullOrWhiteSpace([string]$task.LastRunTime)) { return $skip }
        return @($skip, (New-FleetObservation -Source task -Name $name -Check log-read -Status failed -Severity low -Message "configured log is missing: no file matches $($Rule.log)"))
    }
    $log = $files[0]
    $t = $null; if ($taskName) { $t = $Tasks[$taskName] }
    if ($t -and $t.LastRunTime) {
        $lastRun = ConvertFrom-FleetIso $t.LastRunTime
        if ($log.LastWriteTimeUtc -lt $lastRun.AddMinutes(-1)) {
            return (New-FleetObservation -Source task -Name $name -Check log-rule -Status skipped -Message "log $($log.Name) predates the last run")
        }
    }
    $tail = 400; if ($Rule.tail_lines) { $tail = [int]$Rule.tail_lines }
    try { $lines = @(Read-FleetLogTail -Path $log.FullName -Lines $tail) }
    catch {
        # No evidence is not good evidence: the rule stays neutral (an open incident neither resolves nor
        # reopens), and the read failure itself is reported (fifth review of #346).
        return @(
            (New-FleetObservation -Source task -Name $name -Check log-rule -Status skipped -LogPath $log.FullName -Message ("log unreadable: {0}" -f $_.Exception.Message)),
            (New-FleetObservation -Source task -Name $name -Check log-read -Status failed -Severity low -LogPath $log.FullName -Message ("cannot read {0}: {1}" -f $log.FullName, $_.Exception.Message)))
    }
    $patterns = @($Rule.patterns | Where-Object { $_ }); if ($patterns.Count -eq 0) { $patterns = @('\bERROR\b', '\bSTALE\b') }
    foreach ($line in $lines) {
        foreach ($pat in $patterns) {
            if ($line -match $pat) {
                return (New-FleetObservation -Source task -Name $name -Check log-rule -Status failed -Severity $sev -LogPath $log.FullName -ResolveAfter 1 `
                        -Message ("log matches /{0}/: {1}" -f $pat, $line.Trim()))
            }
        }
    }
    return (New-FleetObservation -Source task -Name $name -Check log-rule -Status ok -LogPath $log.FullName -ResolveAfter 1 -Message 'no failure pattern in the latest log')
}

function Get-FleetDiskObservations {
    <#
    Free space per drive with thresholds, fail-closed on a missing required drive (the T: watchdog
    read a missing T: as 0 GB and stopped every runner), plus a least-squares trend over the last
    window: "T: hits 15 GB in 7.5 h". Samples persist in state\disk-samples.json for 48 h.
    #>
    [CmdletBinding()]
    param([Parameter(Mandatory)][hashtable]$Policy)
    $obs = @()
    $now = Get-FleetNow
    $drives = @{}
    foreach ($d in @(Get-FleetDriveData)) { $drives[[string]$d.Name] = $d }
    $rules = @{}
    foreach ($r in @($Policy.disks)) { $rules[([string]$r.name).Substring(0, 1).ToUpperInvariant()] = $r }
    foreach ($n in @($drives.Keys)) { if (-not $rules.ContainsKey($n)) { $rules[$n] = @{ name = $n } } }

    $samplesPath = Join-Path (Get-FleetPath -Kind state -NoCreate) 'disk-samples.json'
    $samples = Read-FleetJsonFile -Path $samplesPath -Default @{}
    foreach ($n in @($rules.Keys | Sort-Object)) {
        $r = $rules[$n]
        $minGb = [double]$Policy.disk_default_min_free_gb; if ($null -ne $r.min_free_gb) { $minGb = [double]$r.min_free_gb }
        $minPct = [double]$Policy.disk_default_min_free_pct; if ($null -ne $r.min_free_pct) { $minPct = [double]$r.min_free_pct }
        $d = $drives[$n]
        if ($null -eq $d) {
            if ($r.required) {
                $obs += New-FleetObservation -Source disk -Name "${n}:" -Check disk-free -Status failed -Severity high -Message "${n}: is required but missing (treated as a failure, never as 0 GB free)"
            }
            continue
        }
        $freeGb = [math]::Round($d.FreeBytes / 1GB, 1)
        $pct = 100.0; if ($d.TotalBytes -gt 0) { $pct = [math]::Round(100.0 * $d.FreeBytes / $d.TotalBytes, 1) }
        $st = 'ok'; $sev = 'info'
        if ($freeGb -lt $minGb -or $pct -lt $minPct) { $st = 'degraded'; $sev = 'high' }
        $obs += New-FleetObservation -Source disk -Name "${n}:" -Check disk-free -Status $st -Severity $sev -ResolveAfter 2 `
            -Message ("{0}: {1} GB free ({2}%), floor {3} GB / {4}%" -f $n, $freeGb, $pct, $minGb, $minPct) `
            -Details @{ free_gb = $freeGb; free_pct = $pct } -Playbook $(if ($st -ne 'ok') { 'disk-low' } else { $null }) -PlaybookArgs @{ Drive = $n; FreeGb = $freeGb; MinGb = $minGb }

        $list = @(@($samples[$n]) | Where-Object { $_ -and (ConvertFrom-FleetIso $_.ts) -gt $now.AddHours(-48) })
        $list += @{ ts = (ConvertTo-FleetIso $now); free = [int64]$d.FreeBytes }
        $samples[$n] = $list
        $trend = Get-FleetDiskTrend -Samples $list -Now $now -WindowHours ([double]$Policy.disk_trend_window_hours) -FloorBytes ([int64]($minGb * 1GB))
        if ($null -ne $trend) {
            $tst = 'ok'
            $msg = "${n}: trend {0} GB/h over {1} h" -f ([math]::Round($trend.SlopeBytesPerHour / 1GB, 2)), $trend.SpanHours
            if ($null -ne $trend.EtaHours -and $trend.EtaHours -lt [double]$Policy.disk_trend_alert_hours) {
                $tst = 'degraded'
                $msg = "${n}: hits $minGb GB in {0} h (free {1} GB, {2} GB/h over {3} h)" -f ([math]::Round($trend.EtaHours, 1)), $freeGb, ([math]::Round($trend.SlopeBytesPerHour / 1GB, 2)), $trend.SpanHours
            }
            $obs += New-FleetObservation -Source disk -Name "${n}:" -Check disk-trend -Status $tst -Severity medium -Message $msg -ResolveAfter 2 `
                -Details @{ eta_hours = $trend.EtaHours; slope_gb_per_hour = [math]::Round($trend.SlopeBytesPerHour / 1GB, 3) } `
                -Playbook $(if ($tst -ne 'ok') { 'disk-low' } else { $null }) -PlaybookArgs @{ Drive = $n; FreeGb = $freeGb; MinGb = $minGb; EtaHours = $trend.EtaHours }
        }
        else {
            # Too few samples (after a monitoring gap or a lost samples file) is no evidence: neutral, and the key
            # still counts as observed. Emitting nothing let the stale-key resolver "recover" an open trend
            # incident after two sweeps (sweep row 3; fifteenth review of #346).
            $obs += New-FleetObservation -Source disk -Name "${n}:" -Check disk-trend -Status skipped -Message ("{0}: not enough samples yet for a trend" -f $n)
        }
    }
    if (-not $script:FleetReadOnly) { Write-FleetJsonFile -Path $samplesPath -Value $samples }
    return $obs
}

function Get-FleetDiskTrend {
    <#
    Least-squares slope of free bytes over the samples inside the window. Needs >= 3 samples spanning
    >= 30 min, else $null. EtaHours is set only when free space is falling and above the floor.
    #>
    param([object[]]$Samples, [datetime]$Now, [double]$WindowHours = 6, [int64]$FloorBytes)
    $pts = @()
    foreach ($s in @($Samples)) {
        $t = ConvertFrom-FleetIso $s.ts
        if ($t -ge $Now.AddHours(-$WindowHours)) { $pts += , @(($t - $Now).TotalHours, [double]$s.free) }
    }
    if ($pts.Count -lt 3) { return $null }
    $xs = $pts | ForEach-Object { $_[0] }; $ys = $pts | ForEach-Object { $_[1] }
    $span = ([double]($xs | Measure-Object -Maximum).Maximum) - ([double]($xs | Measure-Object -Minimum).Minimum)
    if ($span -lt 0.5) { return $null }
    $n = $pts.Count
    $mx = ($xs | Measure-Object -Average).Average; $my = ($ys | Measure-Object -Average).Average
    $num = 0.0; $den = 0.0
    for ($i = 0; $i -lt $n; $i++) { $num += ($xs[$i] - $mx) * ($ys[$i] - $my); $den += ($xs[$i] - $mx) * ($xs[$i] - $mx) }
    if ($den -eq 0) { return $null }
    $slope = $num / $den
    $latest = [double]$ys[-1]
    $eta = $null
    if ($slope -lt 0 -and $latest -gt $FloorBytes) { $eta = ($latest - $FloorBytes) / (-$slope) }
    return [pscustomobject]@{ SlopeBytesPerHour = $slope; EtaHours = $eta; SpanHours = [math]::Round($span, 1) }
}

function Get-FleetRunnerObservations {
    <#
    Self-hosted runners per scope ('repos/<owner>/<repo>' or 'orgs/<org>'). Offline is an incident,
    except during a planned DRAIN within its budget (vct-nightly takes runners down 10-30 min by
    design); a DRAIN that outlives drain.max_minutes is its own incident. Run on ONE host
    (policy fleet_checks) so the fleet does not report the same runner N times.
    #>
    [CmdletBinding()]
    param([Parameter(Mandatory)][hashtable]$Policy)
    $obs = @()
    $drain = Get-FleetDrainState -Policy $Policy
    $max = 90; if ($Policy.drain.max_minutes) { $max = [int]$Policy.drain.max_minutes }
    $drainAgent = [string]$Policy.drain.bus_agent
    if ($drainAgent) {
        # The read itself is reported, so a bus outage is visible and clears once a read succeeds.
        $rs = 'ok'; $rm = 'DRAIN state read'
        if ($drain.Unknown) { $rs = 'failed'; $rm = "DRAIN state unknown: $($drain.Error)" }
        $obs += New-FleetObservation -Source runner -Name $drainAgent -Check drain-state-read -Status $rs -Severity low -Message $rm -ResolveAfter 1
    }
    if (-not $drainAgent) {
        # No drain agent configured: the DRAIN check is off, but the runners are still polled. The empty name
        # used to fail observation validation and abort the whole detector before GitHub was asked (sweep row 8).
    }
    elseif ($drain.Unknown) {
        # No evidence is not good evidence: 'skipped' is neutral, so an open overrun neither resolves nor
        # reopens, and the stale-key resolver sees the key as observed. Offline runners are still reported
        # below, because an unknown DRAIN cannot excuse them.
        $obs += New-FleetObservation -Source runner -Name $drainAgent -Check drain-overrun -Status skipped -Message "DRAIN state unknown: $($drain.Error)"
    }
    elseif ($drain.Active) {
        $st = 'ok'; $msg = "DRAIN active for $($drain.Minutes) min (budget $max)"
        if ($drain.Minutes -gt $max) { $st = 'failed'; $msg = "DRAIN has lasted $($drain.Minutes) min, over its $max min budget" }
        $obs += New-FleetObservation -Source runner -Name ([string]$Policy.drain.bus_agent) -Check drain-overrun -Status $st -Severity high -Message $msg -ResolveAfter 1
    }
    elseif ($drain.Alert) {
        # The drain agent itself raised an ALERT (e.g. a run died before its restore): STOP/DRAIN flags may
        # be left behind, so the runners' state cannot be assumed. Surface it; do not act on it.
        $obs += New-FleetObservation -Source runner -Name ([string]$Policy.drain.bus_agent) -Check drain-overrun -Status degraded -Severity high -ResolveAfter 1 `
            -Message ("drain agent ALERT, state unknown: {0}" -f $drain.Alert)
    }
    else {
        $obs += New-FleetObservation -Source runner -Name ([string]$Policy.drain.bus_agent) -Check drain-overrun -Status ok -Message 'no DRAIN active' -ResolveAfter 1
    }
    foreach ($scope in @($Policy.runner_scopes)) {
        $pages = @(Invoke-FleetGhApi -Path ("{0}/actions/runners?per_page=100" -f $scope))
        $runners = @()
        foreach ($p in $pages) { if ($p.runners) { $runners += @($p.runners) } }
        # A configured scope that lists no runners is a broken query or a scope to remove from policy, not
        # "every runner is gone": resolving all of the scope's runner incidents by absence was the old outcome
        # (sweep row 10). The detector error stays open until the scope lists runners or leaves the policy.
        if ($runners.Count -eq 0) { throw ("runner scope {0} lists no runners (remove it from runner_scopes if that is intended)" -f $scope) }
        foreach ($r in $runners) {
            $name = '{0}/{1}' -f ($scope -replace '^(repos|orgs)/', ''), $r.name
            $st = 'ok'; $msg = "online (busy=$($r.busy))"
            if ($r.status -ne 'online') {
                $expected = $drain.Active -and $drain.Minutes -le $max -and (Test-FleetWildcard ([string]$r.name) $Policy.drain.runner_patterns)
                if ($expected) { $st = 'skipped'; $msg = 'offline during a planned DRAIN' }
                else { $st = 'failed'; $msg = "runner $($r.name) is $($r.status)" }
            }
            $obs += New-FleetObservation -Source runner -Name $name -Check runner-offline -Status $st -Severity high -Message $msg -ResolveAfter 2 -Details @{ labels = @($r.labels | ForEach-Object { $_.name }) }
        }
    }
    return $obs
}

function Get-FleetWorkflowObservations {
    <# Expected workflows that produced no run for max_age_days (one repository's QA was silent for over three weeks). #>
    [CmdletBinding()]
    param([Parameter(Mandatory)][hashtable]$Policy)
    $obs = @()
    $now = Get-FleetNow
    foreach ($w in @($Policy.workflow_expectations)) {
        $name = '{0}:{1}' -f $w.repo, $w.workflow
        $days = 3; if ($w.max_age_days) { $days = [double]$w.max_age_days }
        $page = Invoke-FleetGhApi -NoPaginate -Path ("repos/{0}/actions/workflows/{1}/runs?per_page=1" -f $w.repo, $w.workflow)
        $runs = @(); foreach ($p in @($page)) { if ($p.workflow_runs) { $runs += @($p.workflow_runs) } }
        if ($runs.Count -eq 0) {
            $obs += New-FleetObservation -Source workflow -Name $name -Check workflow-no-runs -Status never-ran -Severity medium -Message 'no runs at all' -ResolveAfter 1
            continue
        }
        $created = ConvertFrom-FleetIso $runs[0].created_at
        $age = ($now - $created).TotalDays
        $st = 'ok'; if ($age -gt $days) { $st = 'stale' }
        $obs += New-FleetObservation -Source workflow -Name $name -Check workflow-no-runs -Status $st -Severity medium -ResolveAfter 1 `
            -Message ("newest run {0:N1} days old (limit {1}), {2}" -f $age, $days, $runs[0].conclusion)
    }
    return $obs
}

function Get-FleetPostgresObservations {
    <#
    Connections near max_connections, counted from the server's established TCP sockets so the check
    needs no database credentials (the agent-bus Postgres hit its 40-connection limit through leaked qmd
    MCP processes). max_connections and port come from validated policy ONLY: inferring the limit from
    conf files and a pending-restart timestamp drew three review rounds and is gone (coordinator-approved
    scope cut). When the live value is readable without a password it is compared with the policy, and a
    disagreement is reported as drift rather than trusting either.
    #>
    [CmdletBinding()]
    param([Parameter(Mandatory)][hashtable]$Policy)
    $obs = @()
    foreach ($pg in @($Policy.postgres)) {
        $name = [string]$pg.name
        if (-not (Test-FleetPolicyNumber -Value $pg.port -Min 1 -Max 65535 -Integer) -or -not (Test-FleetPolicyNumber -Value $pg.max_connections -Min 1 -Integer)) {
            throw ("postgres entry '{0}' needs an integer port (1-65535) and max_connections (>= 1) in policy" -f $name)
        }
        $port = [int]$pg.port
        $max = [int]$pg.max_connections
        # Nothing listening is a stopped server (or a wrong port), never an idle one: 0 connections read as
        # "ok" and resolved a saturation incident (sweep row 7).
        if (-not (Test-FleetTcpListening -Port $port)) {
            $obs += New-FleetObservation -Source postgres -Name $name -Check pg-server -Status failed -Severity high -ResolveAfter 1 -Message ("nothing is listening on port {0}" -f $port)
            $obs += New-FleetObservation -Source postgres -Name $name -Check pg-connections -Status skipped -Message ("server not listening on port {0}" -f $port)
            continue
        }
        $obs += New-FleetObservation -Source postgres -Name $name -Check pg-server -Status ok -ResolveAfter 1 -Message ("listening on port {0}" -f $port)
        $used = Get-FleetTcpListenerConnectionCount -Port $port
        $live = Get-FleetPostgresLiveMaxConnections -Pg $pg
        if ($null -eq $live) {
            $obs += New-FleetObservation -Source postgres -Name $name -Check postgres-policy-drift -Status skipped -Message 'live max_connections not readable without a password; using the policy value'
        }
        elseif ([int]$live -ne $max) {
            $obs += New-FleetObservation -Source postgres -Name $name -Check postgres-policy-drift -Status degraded -Severity low -ResolveAfter 1 `
                -Message ("live max_connections is {0} but policy says {1}: update the policy (or restart the server if a change is pending)" -f $live, $max)
            $obs += New-FleetObservation -Source postgres -Name $name -Check pg-connections -Status skipped `
                -Message ("{0} connections on port {1}; limit uncertain (live {2}, policy {3})" -f $used, $port, $live, $max)
            continue
        }
        else {
            $obs += New-FleetObservation -Source postgres -Name $name -Check postgres-policy-drift -Status ok -ResolveAfter 1 -Message ("live max_connections matches policy ({0})" -f $max)
        }
        $pct = [math]::Round(100.0 * $used / $max, 0)
        $warn = 80; if ($pg.warn_pct) { $warn = [int]$pg.warn_pct }
        $st = 'ok'; $sev = 'info'
        if ($pct -ge $warn) { $st = 'degraded'; $sev = 'medium' }
        if ($pct -ge 95) { $sev = 'high' }
        $obs += New-FleetObservation -Source postgres -Name $name -Check pg-connections -Status $st -Severity $sev -ResolveAfter 2 `
            -Message ("{0}/{1} connections ({2}%) on port {3}" -f $used, $max, $pct, $port) `
            -Playbook $(if ($st -ne 'ok' -and $pg.reaper) { 'reap-orphans' } else { $null }) -PlaybookArgs @{ Reaper = [string]$pg.reaper }
    }
    return $obs
}

function Get-FleetOrphanProcessObservations {
    <#
    Processes matching a policy pattern whose parent is gone or whose parent PID was reused by a
    younger process. Detection only; the reaper playbook re-checks identity before it kills.
    #>
    [CmdletBinding()]
    param([Parameter(Mandatory)][hashtable]$Policy)
    $obs = @()
    $procs = @(Get-FleetProcessData)
    $byId = @{}; foreach ($p in $procs) { $byId[[int]$p.ProcessId] = $p }
    foreach ($rule in @($Policy.orphan_processes)) {
        $pattern = [string]$rule.pattern
        $orphans = @()
        foreach ($p in @($procs | Where-Object { $_.Name -like $pattern })) {
            $parent = $byId[[int]$p.ParentProcessId]
            if ($null -eq $parent -or ($parent.CreationDate -and $p.CreationDate -and $parent.CreationDate -gt $p.CreationDate)) { $orphans += $p }
        }
        $st = 'ok'; $msg = 'no orphans'
        if ($orphans.Count -gt 0) { $st = 'degraded'; $msg = ('{0} orphan(s): PIDs {1}' -f $orphans.Count, (($orphans | ForEach-Object { $_.ProcessId }) -join ',')) }
        $pb = $null; if ($st -ne 'ok' -and $rule.reaper) { $pb = 'reap-orphans' }
        $obs += New-FleetObservation -Source process -Name $pattern -Check orphan-process -Status $st -Severity low -Message $msg -ResolveAfter 1 `
            -Playbook $pb -PlaybookArgs @{ Reaper = [string]$rule.reaper; Pattern = $pattern }
    }
    return $obs
}

function Get-FleetCertObservations {
    [CmdletBinding()]
    param([Parameter(Mandatory)][hashtable]$Policy)
    $obs = @()
    $now = Get-FleetNow
    foreach ($ep in @($Policy.cert_endpoints)) {
        $hostName = [string]$ep; $port = 443
        if ($hostName -match '^(.+):(\d+)$') { $hostName = $Matches[1]; $port = [int]$Matches[2] }
        try { $c = Get-FleetRemoteCertificate -HostName $hostName -Port $port }
        catch {
            $obs += New-FleetObservation -Source cert -Name "${hostName}:$port" -Check cert-expiry -Status unknown -Severity low -Message "TLS probe failed: $($_.Exception.Message)" -ResolveAfter 1
            continue
        }
        $days = [math]::Floor(($c.NotAfter - $now).TotalDays)
        $st = 'ok'; $sev = 'info'
        if ($c.PolicyErrors -ne 'None') { $st = 'failed'; $sev = 'critical' }
        elseif ($days -lt [int]$Policy.cert_high_days) { $st = 'degraded'; $sev = 'high' }
        elseif ($days -lt [int]$Policy.cert_warn_days) { $st = 'degraded'; $sev = 'medium' }
        $obs += New-FleetObservation -Source cert -Name "${hostName}:$port" -Check cert-expiry -Status $st -Severity $sev -ResolveAfter 1 `
            -Message ("expires {0:yyyy-MM-dd} ({1} days); chain {2}" -f $c.NotAfter, $days, $c.PolicyErrors)
    }
    return $obs
}
