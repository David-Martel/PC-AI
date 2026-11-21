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

# ---------------------------------------------------------------- data sources (mocked in tests)

function Get-FleetTaskHealthData {
    <# Runs Test-ScheduledTaskHealth.ps1 in a child pwsh and returns its parsed report. #>
    param([hashtable]$Policy)
    $script = [IO.Path]::GetFullPath([string]$Policy.task_health_script)
    if (-not (Test-Path -LiteralPath $script)) { throw "task health script not found: $script" }
    $out = Join-Path (Get-FleetPath -Kind state) 'scheduled-task-health.json'
    $pwsh = (Get-Process -Id $PID).Path
    if ($pwsh -notmatch '(?i)pwsh(\.exe)?$') {
        $cmd = Get-Command pwsh -ErrorAction SilentlyContinue
        if ($cmd) { $pwsh = $cmd.Source } else { throw 'Test-ScheduledTaskHealth.ps1 needs PowerShell 7 (pwsh) and none was found' }
    }
    $r = Invoke-FleetProcess -FilePath $pwsh -ArgumentList @('-NoProfile', '-NonInteractive', '-File', $script, '-OutputJson', $out, '-StaleFactor', [string]$Policy.task_stale_factor) -TimeoutSeconds 180
    if ($r.TimedOut -or -not (Test-Path -LiteralPath $out)) { throw "task health script failed (exit $($r.ExitCode), timed out=$($r.TimedOut))" }
    return (ConvertTo-FleetHashtable ([IO.File]::ReadAllText($out) | ConvertFrom-Json))
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
    foreach ($p in @(Get-CimInstance Win32_Process -Filter "Name='explorer.exe'" -ErrorAction SilentlyContinue)) {
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
    <# gh api GET with pagination; returns parsed JSON (array pages flattened). #>
    param([Parameter(Mandatory)][string]$Path)
    $gh = (Get-Command gh -ErrorAction SilentlyContinue)
    if (-not $gh) { throw 'gh CLI not found' }
    $r = Invoke-FleetProcess -FilePath $gh.Source -ArgumentList @('api', '--paginate', $Path) -TimeoutSeconds 60
    if ($r.TimedOut -or $r.ExitCode -ne 0) { throw "gh api $Path failed: $($r.StdErr.Trim())" }
    # --paginate concatenates JSON documents; split them by re-reading each top-level value.
    $text = $r.StdOut.Trim()
    $docs = @()
    if ($text -match '^\{') { $text = '[' + ($text -replace '\}\s*\{', '},{') + ']' }
    elseif ($text -match '^\[') { $text = '[' + ($text -replace '\]\s*\[', '],[') + ']' }
    foreach ($d in @($text | ConvertFrom-Json)) { $docs += $d }
    return (ConvertTo-FleetHashtable $docs)
}

function Get-FleetDrainState {
    <#
    Whether a planned DRAIN (vct-nightly compaction) has the runners down on purpose: the newest
    START/ended status from the drain agent on agent-bus. Returns @{ Active; StartedAt; Minutes }.
    #>
    param([hashtable]$Policy)
    $agent = [string]$Policy.drain.bus_agent
    $none = @{ Active = $false; StartedAt = $null; Minutes = 0 }
    if ([string]::IsNullOrWhiteSpace($agent)) { return $none }
    $r = Invoke-FleetProcess -FilePath (Get-FleetAgentBusPath) -ArgumentList @('read', '--from-agent', $agent, '--since-minutes', '600', '--limit', '20', '--encoding', 'json') -TimeoutSeconds 20
    if ($r.TimedOut -or $r.ExitCode -ne 0) { return $none }
    $msgs = @($r.StdOut | ConvertFrom-Json)
    if ($msgs.Count -eq 1 -and $msgs[0].PSObject.Properties.Name -contains 'messages') { $msgs = @($msgs[0].messages) }
    $last = $msgs | Where-Object { $_.body -match 'START|ended|restored' } | Sort-Object timestamp_utc | Select-Object -Last 1
    if ($null -eq $last -or $last.body -notmatch 'START') { return $none }
    $started = ConvertFrom-FleetIso $last.timestamp_utc
    return @{ Active = $true; StartedAt = $started; Minutes = [int]((Get-FleetNow) - $started).TotalMinutes }
}

function Get-FleetTcpListenerConnectionCount {
    <# Established server-side connections on a local port (each client connection counted once). #>
    param([int]$Port)
    return @(Get-NetTCPConnection -LocalPort $Port -State Established -ErrorAction SilentlyContinue).Count
}

function Get-FleetProcessData {
    $rows = @()
    foreach ($p in @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue)) {
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

function Get-FleetTaskObservations {
    [CmdletBinding()]
    param([Parameter(Mandatory)][hashtable]$Policy)
    $obs = @()
    $report = Get-FleetTaskHealthData -Policy $Policy
    $okCodes = @($Policy.task_ok_codes | ForEach-Object { ([string]$_).ToUpperInvariant() })
    $sevMap = $Policy.task_severity
    $tasks = @($report.Tasks)
    $byName = @{}
    foreach ($t in $tasks) {
        $full = '{0}{1}' -f $t.TaskPath, $t.TaskName
        $byName[$full] = $t
        $included = ($t.Owner -eq 'Local') -or (Test-FleetWildcard $full $Policy.task_include)
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
        $benign = ($null -ne $resultHex) -and ($resultHex -ne '0X0') -and ($taskOk -contains $resultHex)
        $rerun = $null
        if (Test-FleetWildcard $full $Policy.rerun_tasks) { $rerun = 'rerun-task' }

        # Result axis -> check 'run' (the same key Invoke-FleetTask writes, so one incident per task).
        $runStatus = 'ok'
        if ($t.Status -eq 'Failed' -and -not $benign) { $runStatus = 'failed' }
        if (@('Disabled', 'Ignored') -notcontains $t.Status) {
            $obs += New-FleetObservation -Source task -Name $full -Check run -Status $runStatus -Severity $sev `
                -Message ("last result {0} {1}; {2}" -f $t.LastResult, $t.ResultText, $reasons).Trim() -ResolveAfter 1 `
                -Details @{ last_run = $t.LastRunTime; next_run = $t.NextRunTime; missed_runs = $t.MissedRuns } `
                -Playbook $rerun -PlaybookArgs @{ TaskPath = $t.TaskPath; TaskName = $t.TaskName }
        }
        # Staleness axis.
        if (@('Disabled', 'Ignored') -notcontains $t.Status) {
            $staleStatus = 'ok'
            if ($t.Status -eq 'Stalled') { $staleStatus = 'stale'; if ([string]$t.Staleness -like 'never run*') { $staleStatus = 'never-ran' } }
            $obs += New-FleetObservation -Source task -Name $full -Check task-stale -Status $staleStatus -Severity $sev `
                -Message ("{0}; {1}" -f $t.Staleness, $reasons).Trim('; ') -ResolveAfter 1
        }
    }

    # Tasks that must be enabled (an offsite backup task sat Disabled and never ran).
    foreach ($pattern in @($Policy.expect_enabled)) {
        $hits = @($tasks | Where-Object { ('{0}{1}' -f $_.TaskPath, $_.TaskName) -like $pattern })
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
    $principals = @(Get-FleetScheduledTaskData)
    $users = @(Get-FleetInteractiveUserData)
    foreach ($p in $principals) {
        if ($p.State -eq 'Disabled' -or $p.LogonType -notmatch '^Interactive') { continue }
        $t = $byName[$p.FullName]
        $included = ($t -and $t.Owner -eq 'Local') -or (Test-FleetWildcard $p.FullName $Policy.task_include)
        if (-not $included -or (Test-FleetWildcard $p.FullName $Policy.task_exclude)) { continue }
        $uid = ([string]$p.UserId).ToLowerInvariant()
        $present = $false
        foreach ($u in $users) { if ($u -eq $uid -or $u.EndsWith('\' + $uid) -or $uid.EndsWith('\' + $u.Split('\')[-1])) { $present = $true } }
        $st = 'ok'; $msg = 'interactive session present'
        if (-not $present) { $st = 'degraded'; $msg = ('LogonType Interactive and {0} has no interactive session: the task will not run (convert to S4U)' -f $p.UserId) }
        $obs += New-FleetObservation -Source task -Name $p.FullName -Check task-interactive-no-session -Status $st -Severity medium -Message $msg -ResolveAfter 1
    }

    # Exit 0 while the log says it failed (a database pull, a history export printing STALE, a logger).
    foreach ($rule in @($Policy.log_rules)) {
        $obs += Get-FleetLogRuleObservation -Rule $rule -Tasks $byName
    }
    return $obs
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
        return (New-FleetObservation -Source task -Name $name -Check log-rule -Status skipped -Message "no log matches $($Rule.log)")
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
    $lines = @(Get-Content -LiteralPath $log.FullName -Tail $tail -ErrorAction SilentlyContinue)
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

    $samplesPath = Join-Path (Get-FleetPath -Kind state) 'disk-samples.json'
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
    }
    Write-FleetJsonFile -Path $samplesPath -Value $samples
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
    if ($drain.Active) {
        $st = 'ok'; $msg = "DRAIN active for $($drain.Minutes) min (budget $max)"
        if ($drain.Minutes -gt $max) { $st = 'failed'; $msg = "DRAIN has lasted $($drain.Minutes) min, over its $max min budget" }
        $obs += New-FleetObservation -Source runner -Name ([string]$Policy.drain.bus_agent) -Check drain-overrun -Status $st -Severity high -Message $msg -ResolveAfter 1
    }
    else {
        $obs += New-FleetObservation -Source runner -Name ([string]$Policy.drain.bus_agent) -Check drain-overrun -Status ok -Message 'no DRAIN active' -ResolveAfter 1
    }
    foreach ($scope in @($Policy.runner_scopes)) {
        $pages = @(Invoke-FleetGhApi -Path ("{0}/actions/runners?per_page=100" -f $scope))
        $runners = @()
        foreach ($p in $pages) { if ($p.runners) { $runners += @($p.runners) } }
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
        $page = Invoke-FleetGhApi -Path ("repos/{0}/actions/workflows/{1}/runs?per_page=1" -f $w.repo, $w.workflow)
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
    Connections near max_connections, counted from the server's established TCP sockets so the
    check needs no database credentials (the agent-bus Postgres hit its 40-connection limit through
    leaked qmd MCP processes). max_connections comes from policy or the conf files.
    #>
    [CmdletBinding()]
    param([Parameter(Mandatory)][hashtable]$Policy)
    $obs = @()
    foreach ($pg in @($Policy.postgres)) {
        $name = [string]$pg.name
        $max = $null
        if ($pg.max_connections) { $max = [int]$pg.max_connections }
        $dataDir = Resolve-FleetPolicyPath ([string]$pg.data_dir)
        $confList = @($pg.conf_files | Where-Object { $_ })   # @($null) has Count 1
        if ($dataDir -and $confList.Count -eq 0) { $confList = @((Join-Path $dataDir 'postgresql.conf'), (Join-Path $dataDir 'postgresql.auto.conf')) }
        # A conf edited after the server started is not live yet: the configured max_connections may
        # be higher than the running one (a 40 -> 100 raise sat pending a restart), so say so.
        if ($dataDir -and (Test-Path -LiteralPath (Join-Path $dataDir 'postmaster.pid'))) {
            $started = (Get-Item -LiteralPath (Join-Path $dataDir 'postmaster.pid')).LastWriteTimeUtc
            $changed = @($confList | Where-Object { $_ -and (Test-Path -LiteralPath $_) -and (Get-Item -LiteralPath $_).LastWriteTimeUtc -gt $started })
            $pst = 'ok'; $pmsg = 'configuration matches the running server'
            if ($changed.Count -gt 0) { $pst = 'degraded'; $pmsg = ('{0} changed after the server started; a restart is pending and live settings may differ' -f (($changed | ForEach-Object { Split-Path -Leaf $_ }) -join ', ')) }
            $obs += New-FleetObservation -Source postgres -Name $name -Check pg-config-pending -Status $pst -Severity low -Message $pmsg -ResolveAfter 1
        }
        foreach ($confRaw in $confList) {
            $conf = Resolve-FleetPolicyPath ([string]$confRaw)
            if ($conf -and (Test-Path -LiteralPath $conf)) {
                foreach ($line in @(Get-Content -LiteralPath $conf)) { if ($line -match '^\s*max_connections\s*=\s*(\d+)') { $max = [int]$Matches[1] } }
            }
        }
        if (-not $max) {
            $obs += New-FleetObservation -Source postgres -Name $name -Check pg-connections -Status unknown -Severity low -Message 'max_connections unknown (set max_connections or conf_files in policy)'
            continue
        }
        $used = Get-FleetTcpListenerConnectionCount -Port ([int]$pg.port)
        $pct = [math]::Round(100.0 * $used / $max, 0)
        $warn = 80; if ($pg.warn_pct) { $warn = [int]$pg.warn_pct }
        $st = 'ok'; $sev = 'info'
        if ($pct -ge $warn) { $st = 'degraded'; $sev = 'medium' }
        if ($pct -ge 95) { $sev = 'high' }
        $obs += New-FleetObservation -Source postgres -Name $name -Check pg-connections -Status $st -Severity $sev -ResolveAfter 2 `
            -Message ("{0}/{1} connections ({2}%) on port {3}" -f $used, $max, $pct, $pg.port) `
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
