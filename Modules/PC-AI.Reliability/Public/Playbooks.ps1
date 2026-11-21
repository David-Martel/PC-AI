#Requires -Version 5.1
<#
Bounded auto-reconciliation. Every playbook runs under a non-blocking lock, inside a retry budget
per incident and window (default 1 per 24 h), with exponential backoff, and emits a remediation
event; when the budget is spent it emits ONE escalation and stops acting.

Built-in playbooks are deliberately narrow:
  rerun-task    Start-ScheduledTask once for a task on the policy's rerun_tasks allowlist
                (idempotent tasks only), never when it is running or disabled.
  reap-orphans  run the configured reaper (Stop-OrphanQmdMcp.ps1 -Apply kills only processes
                whose parent is dead, re-checking identity before each kill).
  disk-low      prune ONLY policy-listed cache directories (files older than N days) and the
                uv cache, then REQUEST compaction from the storage lane over agent-bus.
Never: reboot, DRAIN, stop runners, delete user data or change network configuration.
#>

function Get-FleetPlaybookSettings {
    param([hashtable]$Policy, [string]$Name)
    $pb = $Policy.playbooks
    $s = @{ budget = 1; window_hours = 24; backoff_minutes = 30; enabled = $true }
    foreach ($k in @('budget', 'window_hours', 'backoff_minutes')) { if ($pb -and $null -ne $pb[$k]) { $s[$k] = $pb[$k] } }
    if ($pb -and $pb[$Name] -is [hashtable]) { foreach ($k in @($pb[$Name].Keys)) { $s[$k] = $pb[$Name][$k] } }
    return $s
}

function Invoke-FleetPlaybook {
    <#
    Runs $Action for one incident inside the budget. $Action returns @{ outcome; detail } with
    outcome in succeeded|failed|triggered|requested|skipped. Returns the outcome string, or
    'locked', 'backoff', 'disabled', 'budget-exhausted'.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$Playbook,
        [Parameter(Mandatory)][string]$IncidentKey,
        [Parameter(Mandatory)][hashtable]$Policy,
        [Parameter(Mandatory)][scriptblock]$Action,
        [hashtable]$Arguments = @{},
        [string]$Source = 'playbook',
        [string]$SubjectName
    )
    $settings = Get-FleetPlaybookSettings -Policy $Policy -Name $Playbook
    if (-not $settings.enabled) { return 'disabled' }
    if (-not $SubjectName) { $SubjectName = $IncidentKey }
    $stateKey = '{0}|{1}' -f $Playbook, $IncidentKey
    $statePath = Join-Path (Get-FleetPath -Kind state) 'playbooks.json'
    $now = Get-FleetNow
    $ctx = @{ outcome = 'locked' }
    $acquired = $false
    $null = Invoke-FleetWithLock -LockName ('playbook-{0}' -f (Get-FleetShortHash $stateKey)) -NoWait -Acquired ([ref]$acquired) -ScriptBlock {
        $all = Read-FleetJsonFile -Path $statePath -Default @{}
        $st = $all[$stateKey]
        if ($null -eq $st) { $st = @{ attempts = @(); escalated_at = $null; last_outcome = $null } }
        $window = $now.AddHours(-[double]$settings.window_hours)
        $attempts = @(@($st.attempts) | Where-Object { $_ -and (ConvertFrom-FleetIso $_) -gt $window })
        if ($attempts.Count -ge [int]$settings.budget) {
            $esc = ConvertFrom-FleetIso $st.escalated_at
            if ($null -eq $esc -or $esc -lt $window) {
                $st.escalated_at = ConvertTo-FleetIso $now
                $ev = New-FleetLedgerEvent -Source playbook -Name $SubjectName -Check $Playbook -Kind escalation -Status failed -Severity high -IncidentKey $IncidentKey `
                    -Message ("playbook {0} spent its budget ({1} attempt(s) in {2} h) and the incident is still open; needs a human" -f $Playbook, $attempts.Count, $settings.window_hours) `
                    -Remediation @{ playbook = $Playbook; attempt = $attempts.Count; budget = [int]$settings.budget; outcome = 'budget-exhausted'; detail = 'escalated' }
                $null = Write-FleetLedgerEvent -LedgerEvent $ev
            }
            $ctx.outcome = 'budget-exhausted'
        }
        else {
            $last = $null; if ($attempts.Count -gt 0) { $last = ConvertFrom-FleetIso $attempts[-1] }
            $wait = [double]$settings.backoff_minutes * [math]::Pow(2, [math]::Max(0, $attempts.Count - 1))
            if ($null -ne $last -and ($now - $last).TotalMinutes -lt $wait) { $ctx.outcome = 'backoff' }
            else {
                $attempt = $attempts.Count + 1
                $attempts = @($attempts + (ConvertTo-FleetIso $now))
                $st.attempts = $attempts
                $res = $null
                try { $res = & $Action $Arguments }
                catch { $res = @{ outcome = 'failed'; detail = $_.Exception.Message } }
                if ($null -eq $res) { $res = @{ outcome = 'failed'; detail = 'playbook returned nothing' } }
                $st.last_outcome = $res.outcome
                $status = 'ok'; if (@('failed') -contains $res.outcome) { $status = 'failed' }
                $ev = New-FleetLedgerEvent -Source playbook -Name $SubjectName -Check $Playbook -Kind remediation -Status $status -Severity info -IncidentKey $IncidentKey `
                    -Message ("{0} attempt {1}/{2}: {3} ({4})" -f $Playbook, $attempt, $settings.budget, $res.outcome, $res.detail) `
                    -Remediation @{ playbook = $Playbook; attempt = $attempt; budget = [int]$settings.budget; outcome = [string]$res.outcome; detail = [string]$res.detail }
                $null = Write-FleetLedgerEvent -LedgerEvent $ev
                $ctx.outcome = [string]$res.outcome
            }
        }
        $all[$stateKey] = $st
        Write-FleetJsonFile -Path $statePath -Value $all
    }
    if (-not $acquired) { return 'locked' }
    return $ctx.outcome
}

# ---------------------------------------------------------------- built-in actions (each mockable)

function Get-FleetTaskStateData {
    param([string]$TaskPath, [string]$TaskName)
    $t = Get-ScheduledTask -TaskPath $TaskPath -TaskName $TaskName -ErrorAction Stop
    return [string]$t.State
}

function Start-FleetScheduledTask {
    param([string]$TaskPath, [string]$TaskName)
    Start-ScheduledTask -TaskPath $TaskPath -TaskName $TaskName -ErrorAction Stop
}

function Invoke-FleetRerunTaskAction {
    param([hashtable]$Arguments)
    $state = Get-FleetTaskStateData -TaskPath $Arguments.TaskPath -TaskName $Arguments.TaskName
    if ($state -eq 'Running') { return @{ outcome = 'skipped'; detail = 'task is running' } }
    if ($state -eq 'Disabled') { return @{ outcome = 'skipped'; detail = 'task is disabled (a human disabled it; not re-enabling)' } }
    Start-FleetScheduledTask -TaskPath $Arguments.TaskPath -TaskName $Arguments.TaskName
    return @{ outcome = 'triggered'; detail = ('Start-ScheduledTask {0}{1}' -f $Arguments.TaskPath, $Arguments.TaskName) }
}

function Resolve-FleetPolicyPath {
    <# Expands {profile} (real profile) and {home} (fleet-reliability home) in a policy path. #>
    param([string]$Path)
    if (-not $Path) { return $null }
    return ($Path.Replace('{profile}', (Get-FleetRealProfile)).Replace('{home}', (Get-FleetReliabilityHome)))
}

function Invoke-FleetReaperAction {
    param([hashtable]$Arguments)
    $reaper = Resolve-FleetPolicyPath $Arguments.Reaper
    if (-not $reaper -or -not (Test-Path -LiteralPath $reaper)) { return @{ outcome = 'skipped'; detail = "reaper not found: $reaper" } }
    $pwsh = (Get-Process -Id $PID).Path
    $r = Invoke-FleetProcess -FilePath $pwsh -ArgumentList @('-NoProfile', '-NonInteractive', '-File', $reaper, '-Apply') -TimeoutSeconds 120
    if (-not $r.TimedOut -and $r.ExitCode -eq 0) { return @{ outcome = 'succeeded'; detail = ("{0} -Apply exit 0" -f (Split-Path -Leaf $reaper)) } }
    return @{ outcome = 'failed'; detail = ("{0} -Apply exit {1} timedout={2}" -f (Split-Path -Leaf $reaper), $r.ExitCode, $r.TimedOut) }
}

function Test-FleetCacheDirAllowed {
    <#
    The only directories disk-low may prune: absolute, at least three levels deep, with a path
    segment that names a cache, and not inside user documents, source trees or a git directory.
    A policy typo must not turn this into "delete user data".
    #>
    param([string]$Path)
    if ([string]::IsNullOrWhiteSpace($Path) -or -not [IO.Path]::IsPathRooted($Path)) { return $false }
    $full = [IO.Path]::GetFullPath($Path).TrimEnd('\')
    $segments = @($full.Split('\') | Where-Object { $_ })
    if ($segments.Count -lt 3) { return $false }
    if (-not (@($segments | Where-Object { $_ -match '(?i)cache' }).Count)) { return $false }
    foreach ($bad in @('Documents', 'Desktop', 'Pictures', 'Videos', 'Music', 'OneDrive', 'Dropbox', 'codedev', 'projects', '.git', '.ssh', '.gnupg', 'secrets')) {
        if ($segments -contains $bad) { return $false }
    }
    return $true
}

function Invoke-FleetCachePrune {
    <# Deletes files older than MinAgeDays under an allowed cache dir. Returns bytes freed. #>
    param([string]$Path, [double]$MinAgeDays = 7, [switch]$WhatIfOnly)
    if (-not (Test-FleetCacheDirAllowed $Path)) { throw "refusing to prune a path that is not an allowed cache directory: $Path" }
    if (-not (Test-Path -LiteralPath $Path)) { return 0 }
    $cut = (Get-FleetNow).AddDays(-$MinAgeDays)
    $freed = [int64]0
    foreach ($f in @(Get-ChildItem -LiteralPath $Path -File -Recurse -Force -ErrorAction SilentlyContinue | Where-Object { $_.LastWriteTimeUtc -lt $cut -and -not ($_.Attributes -band [IO.FileAttributes]::ReparsePoint) })) {
        if (-not $WhatIfOnly) {
            try { Remove-Item -LiteralPath $f.FullName -Force -ErrorAction Stop; $freed += $f.Length } catch { Write-Verbose "skip $($f.FullName): $($_.Exception.Message)" }
        }
        else { $freed += $f.Length }
    }
    return $freed
}

function Send-FleetCompactionRequest {
    param([hashtable]$Policy, [hashtable]$Arguments)
    $agent = [string]$Policy.compaction.request_agent
    $body = ("REQUEST from fleet-ledger-{0}: drive {1}: {2} GB free (floor {3} GB){4}. Please run your compaction/cleanup when safe. This sender never stops runners or sets DRAIN itself." -f `
            (Get-FleetHostName), $Arguments.Drive, $Arguments.FreeGb, $Arguments.MinGb, $(if ($Arguments.EtaHours) { ', projected to reach the floor in {0:N1} h' -f $Arguments.EtaHours } else { '' }))
    $r = Invoke-FleetProcess -FilePath (Get-FleetAgentBusPath) -ArgumentList @('send', '--from-agent', ('fleet-ledger-{0}' -f (Get-FleetHostName)), '--to-agent', $agent,
        '--topic', 'coordination', '--schema', 'status', '--thread-id', 'machine-hygiene', '--tag', ('host:{0}' -f (Get-FleetHostName)), '--body', $body, '--encoding', 'compact') -TimeoutSeconds 20
    return (-not $r.TimedOut -and $r.ExitCode -eq 0)
}

function Invoke-FleetDiskLowAction {
    param([hashtable]$Arguments, [hashtable]$Policy)
    $freed = [int64]0
    $notes = @()
    foreach ($c in @($Policy.cache_prune)) {
        if (([string]$c.drive).Substring(0, 1).ToUpperInvariant() -ne [string]$Arguments.Drive) { continue }
        foreach ($d in @($c.dirs)) {
            $p = Resolve-FleetPolicyPath ([string]$d.path)
            $age = 7; if ($d.min_age_days) { $age = [double]$d.min_age_days }
            try { $freed += Invoke-FleetCachePrune -Path $p -MinAgeDays $age } catch { $notes += $_.Exception.Message }
        }
        if (@($c.commands) -contains 'uv-cache-prune') {
            $uv = Get-Command uv -ErrorAction SilentlyContinue
            if ($uv) { $r = Invoke-FleetProcess -FilePath $uv.Source -ArgumentList @('cache', 'prune') -TimeoutSeconds 300; $notes += "uv cache prune exit $($r.ExitCode)" }
        }
    }
    $requested = $false
    if ($Policy.compaction.request_agent -and @($Policy.compaction.drives) -contains [string]$Arguments.Drive) {
        $requested = Send-FleetCompactionRequest -Policy $Policy -Arguments $Arguments
        $notes += "compaction requested from $($Policy.compaction.request_agent): $requested"
    }
    $detail = ("freed {0:N2} GB; {1}" -f ($freed / 1GB), ($notes -join '; '))
    if ($freed -gt 0) { return @{ outcome = 'succeeded'; detail = $detail } }
    if ($requested) { return @{ outcome = 'requested'; detail = $detail } }
    return @{ outcome = 'skipped'; detail = $detail }
}

function Invoke-FleetPlaybookForObservation {
    <# Dispatches an observation's named playbook through the bounded runner. #>
    param([Parameter(Mandatory)]$Observation, [Parameter(Mandatory)][string]$IncidentKey, [Parameter(Mandatory)][hashtable]$Policy)
    $a = $Observation.PlaybookArgs
    switch ($Observation.Playbook) {
        'rerun-task' { $action = { param($x) Invoke-FleetRerunTaskAction -Arguments $x } }
        'reap-orphans' { $action = { param($x) Invoke-FleetReaperAction -Arguments $x } }
        'disk-low' { $action = { param($x) Invoke-FleetDiskLowAction -Arguments $x -Policy $Policy } }
        default { return 'unknown-playbook' }
    }
    return (Invoke-FleetPlaybook -Playbook $Observation.Playbook -IncidentKey $IncidentKey -Policy $Policy -Action $action -Arguments $a -SubjectName $Observation.Name)
}
