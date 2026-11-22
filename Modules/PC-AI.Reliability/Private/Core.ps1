#Requires -Version 5.1
<#
Core helpers for PC-AI.Reliability: paths, the injectable clock, a cross-session file lock,
a bounded child-process runner and JSON helpers that behave the same on Windows PowerShell 5.1
and PowerShell 7.

Every call that leaves the process (agent-bus, gh, Task Scheduler, the network) goes through one
small function here or in the detector file, so the Pester suite can mock it with -ModuleName.
Contract: docs/reliability/fleet-ledger.md.
#>

$script:FleetLedgerSchema = 'fleet-ledger/1'
$script:FleetForwardKinds = @('incident-open', 'incident-resolved', 'incident-flapping', 'incident-reminder', 'escalation', 'remediation', 'digest')
$script:FleetFindingKinds = @('incident-open', 'incident-flapping', 'incident-reminder', 'escalation')
$script:FleetSeverityRank = @{ info = 0; low = 1; medium = 2; high = 3; critical = 4 }

function Assert-FleetNotDeferred {
    <# Throws when a deferred feature is called (see $script:FleetDeferred in the .psm1). #>
    param([Parameter(Mandatory)][ValidateSet('remediation', 'privileged_setup')][string]$Feature, [Parameter(Mandatory)][string]$What)
    if ($script:FleetDeferred -and $script:FleetDeferred[$Feature]) {
        throw "$What is deferred to a follow-up PR and refuses to run (docs/reliability/fleet-ledger.md, Deferred)."
    }
}

function Get-FleetNow {
    <# UTC now. Tests set $env:FLEET_NOW (ISO-8601) so time-based rules are deterministic. #>
    if (-not [string]::IsNullOrWhiteSpace($env:FLEET_NOW)) {
        return ([datetime]::Parse($env:FLEET_NOW, [Globalization.CultureInfo]::InvariantCulture,
                [Globalization.DateTimeStyles]::AdjustToUniversal -bor [Globalization.DateTimeStyles]::AssumeUniversal))
    }
    return [datetime]::UtcNow
}

function ConvertTo-FleetIso {
    param([Nullable[datetime]]$Value)
    if ($null -eq $Value) { return $null }
    $v = [datetime]$Value
    if ($v.Kind -eq [DateTimeKind]::Local) { $v = $v.ToUniversalTime() }
    return $v.ToString("yyyy-MM-dd'T'HH:mm:ss.fff'Z'", [Globalization.CultureInfo]::InvariantCulture)
}

function ConvertFrom-FleetIso {
    <#
    ISO-8601 string (or a DateTime: PowerShell 7's ConvertFrom-Json turns ISO strings into
    DateTime, 5.1 does not) to a UTC DateTime; $null for empty input.
    #>
    param([object]$Value)
    if ($null -eq $Value) { return $null }
    if ($Value -is [psobject]) { $Value = $Value.psobject.BaseObject }
    if ($Value -is [datetime]) {
        if ($Value.Kind -eq [DateTimeKind]::Local) { return $Value.ToUniversalTime() }
        return [datetime]::SpecifyKind($Value, [DateTimeKind]::Utc)
    }
    $text = [string]$Value
    if ([string]::IsNullOrWhiteSpace($text)) { return $null }
    return [datetime]::Parse($text, [Globalization.CultureInfo]::InvariantCulture,
        [Globalization.DateTimeStyles]::AdjustToUniversal -bor [Globalization.DateTimeStyles]::AssumeUniversal)
}

function Get-FleetHostName {
    if (-not [string]::IsNullOrWhiteSpace($env:FLEET_HOST)) { return $env:FLEET_HOST.ToLowerInvariant() }
    return ([Environment]::MachineName).ToLowerInvariant()
}

function Get-FleetRealProfile {
    <#
    The real user profile. USERPROFILE/HOME are redirected for agent sessions (audit finding #15),
    which silently pointed scripts at an empty tree. The known-folder lookup reads the process token,
    so it is right for interactive, S4U and agent-redirected processes alike.
    #>
    $p = [Environment]::GetFolderPath('UserProfile')
    if ([string]::IsNullOrWhiteSpace($p)) { $p = $env:USERPROFILE }
    return $p
}

function Get-FleetReliabilityHome {
    if (-not [string]::IsNullOrWhiteSpace($env:FLEET_RELIABILITY_HOME)) { return $env:FLEET_RELIABILITY_HOME }
    return (Join-Path ([Environment]::GetFolderPath('CommonApplicationData')) 'fleet-reliability')
}

function Get-FleetPath {
    # -NoCreate for read paths: a reader (digest, dry-run) must not create directories as a side effect.
    param([Parameter(Mandatory)][ValidateSet('ledger', 'state', 'spool', 'quarantine', 'locks', 'secrets', 'logs')][string]$Kind, [switch]$NoCreate)
    $homeDir = Get-FleetReliabilityHome
    $rel = switch ($Kind) {
        'quarantine' { Join-Path 'spool' 'quarantine' }
        default { $Kind }
    }
    $dir = Join-Path $homeDir $rel
    # A dry run (read-only) never creates a directory, whichever caller asks (sweep row 23).
    if (-not $NoCreate -and -not $script:FleetReadOnly -and -not (Test-Path -LiteralPath $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
    return $dir
}

function Invoke-FleetWithLock {
    <#
    Runs $ScriptBlock while holding an exclusive lock file. A file lock (FileShare.None) works across
    sessions and accounts, which a named mutex does not without an explicit ACL: SYSTEM watchdogs,
    S4U tasks and interactive shells all append to the same ledger.
    Returns $null without running when the lock is not obtained within TimeoutSeconds and -NoWait
    is not set; with -NoWait a busy lock returns immediately (used by playbooks).
    #>
    param(
        [Parameter(Mandatory)][string]$LockName,
        [Parameter(Mandatory)][scriptblock]$ScriptBlock,
        [int]$TimeoutSeconds = 30,
        [switch]$NoWait,
        [ref]$Acquired
    )
    $lockPath = Join-Path (Get-FleetPath -Kind locks) ("{0}.lock" -f $LockName)
    $deadline = [datetime]::UtcNow.AddSeconds($TimeoutSeconds)
    $stream = $null
    while ($null -eq $stream) {
        try {
            $stream = [IO.File]::Open($lockPath, [IO.FileMode]::OpenOrCreate, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
        }
        catch [System.IO.IOException] {
            if ($NoWait -or [datetime]::UtcNow -ge $deadline) {
                if ($Acquired) { $Acquired.Value = $false }
                return $null
            }
            Start-Sleep -Milliseconds 100
        }
    }
    if ($Acquired) { $Acquired.Value = $true }
    try { return (& $ScriptBlock) }
    finally { $stream.Dispose() }
}

function ConvertTo-FleetCommandLine {
    <#
    Quotes an argv list for CreateProcess the way the MSVC runtime (and Rust's std) splits it back:
    backslashes are literal unless they precede a quote. Windows PowerShell 5.1 has no
    ProcessStartInfo.ArgumentList, and agent-bus --metadata carries JSON full of quotes.
    #>
    param([string[]]$Arguments)
    $parts = foreach ($a in $Arguments) {
        if ($null -eq $a) { $a = '' }
        if ($a.Length -gt 0 -and $a -notmatch '[\s"]') { $a; continue }
        $sb = New-Object System.Text.StringBuilder
        [void]$sb.Append('"')
        $bs = 0
        foreach ($ch in $a.ToCharArray()) {
            if ($ch -eq '\') { $bs++; continue }
            if ($ch -eq '"') {
                [void]$sb.Append([char]'\', [int](2 * $bs + 1)); [void]$sb.Append('"'); $bs = 0; continue
            }
            if ($bs -gt 0) { [void]$sb.Append([char]'\', [int]$bs); $bs = 0 }
            [void]$sb.Append($ch)
        }
        if ($bs -gt 0) { [void]$sb.Append([char]'\', [int](2 * $bs)) }
        [void]$sb.Append('"')
        $sb.ToString()
    }
    return ($parts -join ' ')
}

function Stop-FleetProcessTree {
    <# Kills a process and all its descendants: Kill(entireProcessTree) on .NET Core, taskkill /T on 5.1. #>
    param([Parameter(Mandatory)][System.Diagnostics.Process]$Process)
    try {
        $killTree = $Process.GetType().GetMethod('Kill', [type[]]@([bool]))
        if ($null -ne $killTree) { $Process.Kill($true); return }
        & taskkill.exe /T /F /PID $Process.Id 2>&1 | Out-Null
    }
    catch { Write-Verbose "process tree kill failed: $($_.Exception.Message)" }
}

function Invoke-FleetProcess {
    <#
    Runs an executable with a hard timeout and captured output. Never hangs a scheduled task: on
    timeout the process tree is killed and TimedOut is set. Never throws for a missing binary; the
    caller decides (ExitCode -1, StdErr explains).
    #>
    param(
        [Parameter(Mandatory)][string]$FilePath,
        [string[]]$ArgumentList = @(),
        [int]$TimeoutSeconds = 20,
        [string]$WorkingDirectory
    )
    $result = [pscustomobject]@{ ExitCode = -1; StdOut = ''; StdErr = ''; TimedOut = $false; OutputComplete = $true }
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $FilePath
    $psi.Arguments = ConvertTo-FleetCommandLine -Arguments $ArgumentList
    $psi.UseShellExecute = $false
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.CreateNoWindow = $true
    if ($WorkingDirectory) { $psi.WorkingDirectory = $WorkingDirectory }
    $proc = New-Object System.Diagnostics.Process
    $proc.StartInfo = $psi
    try { [void]$proc.Start() }
    catch {
        $result.StdErr = "failed to start ${FilePath}: $($_.Exception.Message)"
        return $result
    }
    # Both pipes are read in chunks while the job runs, so a full pipe never blocks the child and every
    # chunk that arrived is kept. A descendant that outlives the child (a daemon, `start /b`, `cmd &`)
    # keeps the pipes open; reading to the end would hang the caller as long as it lives (measured: 26 s
    # for a 2 s timeout), and a whole-stream read abandoned at a deadline lost what the job HAD printed,
    # including its [ERROR] line (second review of #346). After exit, drain for a bounded time.
    $readers = @(
        @{ Name = 'out'; Reader = $proc.StandardOutput; Text = New-Object System.Text.StringBuilder; Task = $null; Eof = $false; Buf = New-Object char[] 8192 },
        @{ Name = 'err'; Reader = $proc.StandardError; Text = New-Object System.Text.StringBuilder; Task = $null; Eof = $false; Buf = New-Object char[] 8192 })
    foreach ($rd in $readers) { $rd.Task = $rd.Reader.ReadAsync($rd.Buf, 0, $rd.Buf.Length) }
    $drainMs = 10000
    if ($env:FLEET_PIPE_DRAIN_MS) { $drainMs = [int]$env:FLEET_PIPE_DRAIN_MS }
    $clock = [Diagnostics.Stopwatch]::StartNew()
    $limitMs = [Math]::Max(1, $TimeoutSeconds) * 1000
    $exitedAtMs = $null
    while ($true) {
        if ($null -eq $exitedAtMs) {
            if ($proc.HasExited) { $result.ExitCode = $proc.ExitCode; $exitedAtMs = $clock.ElapsedMilliseconds }
            elseif ($clock.ElapsedMilliseconds -ge $limitMs) {
                $result.TimedOut = $true
                Stop-FleetProcessTree -Process $proc
                [void]$proc.WaitForExit(5000)
                $exitedAtMs = $clock.ElapsedMilliseconds
            }
        }
        $open = @($readers | Where-Object { -not $_.Eof })
        if ($open.Count -eq 0 -and $null -ne $exitedAtMs) { break }
        if ($null -ne $exitedAtMs -and ($clock.ElapsedMilliseconds - $exitedAtMs) -ge $drainMs) { $result.OutputComplete = ($open.Count -eq 0); break }
        if ($open.Count -gt 0) { [void][System.Threading.Tasks.Task]::WaitAny([System.Threading.Tasks.Task[]]@($open | ForEach-Object { $_.Task }), 100) }
        else { Start-Sleep -Milliseconds 50 }
        foreach ($rd in $open) {
            if (-not $rd.Task.IsCompleted) { continue }
            if ($rd.Task.IsFaulted -or $rd.Task.IsCanceled -or $rd.Task.Result -le 0) { $rd.Eof = $true; continue }
            [void]$rd.Text.Append($rd.Buf, 0, $rd.Task.Result)
            $rd.Task = $rd.Reader.ReadAsync($rd.Buf, 0, $rd.Buf.Length)
        }
    }
    $result.StdOut = $readers[0].Text.ToString()
    $result.StdErr = $readers[1].Text.ToString()
    if (-not $result.OutputComplete) { $result.StdErr += "`n[fleet] output not fully captured: a descendant process still holds the output pipe (what arrived before the deadline is kept)" }
    $proc.Dispose()
    return $result
}

function ConvertTo-FleetHashtable {
    <# PowerShell 5.1 has no ConvertFrom-Json -AsHashtable; state files are read through this. #>
    # `-is [pscustomobject]` is true for ANY PSObject-wrapped value (a DateTime from ConvertFrom-Json
    # included), so test the unwrapped base object. Dates go back to ISO strings so state files
    # read the same on PowerShell 5.1 (strings) and 7 (DateTime).
    param($InputObject)
    if ($null -eq $InputObject) { return $null }
    $base = $InputObject
    if ($InputObject -is [psobject]) { $base = $InputObject.psobject.BaseObject }
    if ($base -is [datetime]) { return (ConvertTo-FleetIso $base) }
    if ($base -is [string] -or $base -is [ValueType]) { return $base }
    if ($base -is [System.Collections.IDictionary]) {
        $h = @{}
        foreach ($k in $base.Keys) { $h[[string]$k] = ConvertTo-FleetHashtable $base[$k] }
        return $h
    }
    if ($base -is [System.Management.Automation.PSCustomObject]) {
        $h = @{}
        foreach ($p in $InputObject.PSObject.Properties) { $h[$p.Name] = ConvertTo-FleetHashtable $p.Value }
        return $h
    }
    if ($base -is [System.Collections.IEnumerable]) {
        return , @($base | ForEach-Object { ConvertTo-FleetHashtable $_ })
    }
    return $InputObject
}

function Read-FleetJsonFile {
    # -ThrowOnError: an existing file that cannot be read or parsed throws and is left in place, for
    # callers where "unreadable" must not mean "empty" (playbook budgets).
    param([Parameter(Mandatory)][string]$Path, $Default = @{}, [switch]$ThrowOnError)
    if (-not (Test-Path -LiteralPath $Path)) { return $Default }
    try {
        $raw = [IO.File]::ReadAllText($Path)
        if ([string]::IsNullOrWhiteSpace($raw)) {
            if ($ThrowOnError) { throw "$Path is empty" }
            return $Default
        }
        return (ConvertTo-FleetHashtable ($raw | ConvertFrom-Json))
    }
    catch {
        if ($ThrowOnError) { throw "cannot read ${Path}: $($_.Exception.Message)" }
        # A dry run never moves a live file, even a corrupt one (second review of #346).
        if ($script:FleetReadOnly) { Write-Verbose "unreadable JSON left in place (read-only): $Path"; return $Default }
        # A corrupt state file must not stop the sweeper; keep it for inspection and start fresh.
        $bad = '{0}.corrupt-{1}' -f $Path, ([datetime]::UtcNow.ToString('yyyyMMddTHHmmss'))
        try { Move-Item -LiteralPath $Path -Destination $bad -Force } catch { Write-Verbose "could not move corrupt state: $($_.Exception.Message)" }
        return $Default
    }
}

function Write-FleetJsonFile {
    <# Atomic replace: write a sibling temp file, then move it over the target. #>
    param([Parameter(Mandatory)][string]$Path, [Parameter(Mandatory)]$Value)
    $dir = Split-Path -Parent $Path
    if (-not (Test-Path -LiteralPath $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
    $tmp = '{0}.tmp-{1}' -f $Path, ([guid]::NewGuid().ToString('N').Substring(0, 8))
    $json = $Value | ConvertTo-Json -Depth 12
    [IO.File]::WriteAllText($tmp, $json, (New-Object System.Text.UTF8Encoding($false)))
    # Terminating: a state file that could not be replaced must stop the caller. A playbook reserves its
    # attempt with this write before acting, and a silent failure here let it act without spending budget
    # (sixth review of #346).
    try { Move-Item -LiteralPath $tmp -Destination $Path -Force -ErrorAction Stop }
    catch {
        Remove-Item -LiteralPath $tmp -Force -ErrorAction SilentlyContinue
        throw "could not write ${Path}: $($_.Exception.Message)"
    }
}

function Get-FleetShortHash {
    param([string]$Text)
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try {
        $bytes = $sha.ComputeHash([Text.Encoding]::UTF8.GetBytes([string]$Text))
        return (-join ($bytes[0..5] | ForEach-Object { $_.ToString('x2') }))
    }
    finally { $sha.Dispose() }
}

function Protect-FleetText {
    <#
    Redacts credentials from a text that may leave the host (ledger, incident state, agent-bus, Discord):
    key=value / key: value pairs with secret-like keys (quoted values too), Authorization schemes,
    user:password@ in URLs, and well-known token shapes. Throws on any internal error; callers use
    ConvertTo-FleetSafeMessage, which then withholds the text (codex P1 on #346).
    #>
    param([string]$Text)
    if ([string]::IsNullOrEmpty($Text)) { return $Text }
    $t = $Text
    $t = [regex]::Replace($t, '(?i)\b([a-z][a-z0-9+.\-]*://)[^\s/:@]+:[^\s/@]+@', '$1***@')
    $t = [regex]::Replace($t, '(?i)\b(bearer|basic|token|digest)\s+[A-Za-z0-9._~+/=\-]{6,}', '$1 ***')
    $keys = 'password|passwd|pwd|passphrase|secret|token|api[_\-]?key|apikey|access[_\-]?key|private[_\-]?key|client[_\-]?secret|authorization|auth|credential|cookie|session[_\-]?id|sig|signature'
    # The key may be quoted ("password": / 'secret':), and a quoted value may contain escapes ("ab\"cd"): a
    # JSON key's closing quote between key and ':' used to stop the match and forward the value (review of 45461ab9).
    $t = [regex]::Replace($t, ('(?i)\b([A-Za-z0-9_.\-]*(?:{0})[A-Za-z0-9_.\-]*)(["'']?\s*[:=]\s*)("(?:[^"\\]|\\.)*"?|''(?:[^''\\]|\\.)*''?|[^\s;,&}}]+)' -f $keys), '$1$2***')
    $t = [regex]::Replace($t, '\bsk-[A-Za-z0-9_\-]{12,}', 'sk-***')
    $t = [regex]::Replace($t, '\b(?:AKIA|ASIA|AGPA|AIDA|AROA)[A-Z0-9]{16}\b', 'AWS-KEY-***')
    $t = [regex]::Replace($t, '\bgh[pousr]_[A-Za-z0-9]{20,}', 'gh_***')
    $t = [regex]::Replace($t, '\bxox[abprs]-[A-Za-z0-9\-]{10,}', 'xox-***')
    $t = [regex]::Replace($t, '\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{4,}', 'jwt-***')
    $t = [regex]::Replace($t, '\b[MN][A-Za-z0-9]{23,27}\.[A-Za-z0-9_\-]{6}\.[A-Za-z0-9_\-]{27,}', 'discord-token-***')
    return $t
}

function ConvertTo-FleetSafeMessage {
    <#
    The redacted text, or, when redaction fails, a placeholder carrying only a hash of the withheld excerpt:
    redaction fails CLOSED. Raw text stays only in the tool's own log (-LogPath), which this module never copies.
    #>
    param([string]$Text)
    if ([string]::IsNullOrEmpty($Text)) { return $Text }
    try { return [string](Protect-FleetText -Text $Text) }
    catch { return ('[excerpt withheld: redaction failed; sha256 {0}]' -f (Get-FleetShortHash $Text)) }
}

# Keys whose values are credentials whatever they look like. Shared with the Linux ledger (sentinel-utils #1600).
$script:FleetSensitiveKeyPattern = '(?i)password|passwd|passphrase|pwd|secret|token|api[_-]?key|auth|credential|cookie|private[_-]?key|access[_-]?key|session[_-]?id'

function Protect-FleetStructure {
    <#
    A redacted COPY of structured event data (Details, Remediation): any non-null value under a sensitive key is
    replaced whole, at any depth and of any type, since a plain value under such a key ({ password = 'v' }, a
    number under pin_secret) matches no text rule; every other string goes through Protect-FleetText. The
    caller's object is not changed. Throws on any error; callers use ConvertTo-FleetSafeStructure.
    #>
    param($Value)
    if ($null -eq $Value) { return $null }
    if ($Value -is [string]) { return [string](Protect-FleetText -Text $Value) }
    if ($Value -is [System.Collections.IDictionary]) {
        $out = @{}
        foreach ($k in @($Value.Keys)) {
            $v = $Value[$k]
            if ([string]$k -match $script:FleetSensitiveKeyPattern -and $null -ne $v) { $out[[string]$k] = '[REDACTED]' }
            else { $out[[string]$k] = Protect-FleetStructure -Value $v }
        }
        return $out
    }
    if ($Value -is [System.Management.Automation.PSCustomObject]) {
        $out = @{}
        foreach ($p in $Value.PSObject.Properties) {
            if ($p.Name -match $script:FleetSensitiveKeyPattern -and $null -ne $p.Value) { $out[$p.Name] = '[REDACTED]' }
            else { $out[$p.Name] = Protect-FleetStructure -Value $p.Value }
        }
        return $out
    }
    if ($Value -is [System.Collections.IEnumerable]) {
        $items = @(foreach ($i in $Value) { , (Protect-FleetStructure -Value $i) })
        return , $items
    }
    # Only plain values pass unchanged. Any other object is withheld by type name: returned as-is, a
    # NetworkCredential under details.account was serialized with its plaintext Password (review of e62fcadc).
    $t = $Value.GetType()
    if ($t.IsPrimitive -or $t.IsEnum -or $Value -is [decimal] -or $Value -is [datetime] -or $Value -is [datetimeoffset] -or
        $Value -is [timespan] -or $Value -is [guid] -or $Value -is [System.Numerics.BigInteger]) { return $Value }
    return ('[withheld object: {0}]' -f $t.FullName)
}

function ConvertTo-FleetSafeStructure {
    <# The redacted copy, or, when redaction fails, @{ redaction = 'failed' }: structured data fails CLOSED too. #>
    param($Value)
    if ($null -eq $Value) { return $null }
    try { return (Protect-FleetStructure -Value $Value) }
    catch { return @{ redaction = 'failed' } }
}

function Set-FleetNormalPriority {
    <#
    Raises this process from BelowNormal/Idle to Normal and returns the previous class. Task Scheduler
    starts tasks at priority 7 (BelowNormal) unless told otherwise, and a starved monitor goes quiet
    precisely when its host is overloaded. Never lowers a priority and never goes above Normal.
    #>
    $p = Get-Process -Id $PID
    $before = [string]$p.PriorityClass
    if (@('BelowNormal', 'Idle') -contains $before) {
        try { $p.PriorityClass = 'Normal' } catch { Write-Verbose "could not raise priority: $($_.Exception.Message)" }
    }
    return $before
}

function Get-FleetAgentBusPath {
    if (-not [string]::IsNullOrWhiteSpace($env:AGENT_BUS_EXE)) { return $env:AGENT_BUS_EXE }
    return (Join-Path (Get-FleetRealProfile) 'bin\agent-bus.exe')
}

function Get-FleetPolicy {
    <#
    Host policy: <home>\policy.json (not in the repository: it names hosts, tasks, endpoints and
    alert recipients). Missing keys fall back to the defaults below, so an empty policy still runs
    the generic detectors. An example lives next to the module manifest.
    #>
    param([string]$Path)
    $named = -not [string]::IsNullOrWhiteSpace($Path)
    if (-not $named) { $Path = Join-Path (Get-FleetReliabilityHome) 'policy.json' }
    $defaults = @{
        task_health_script    = (Join-Path $PSScriptRoot '..\..\..\Tools\Test-ScheduledTaskHealth.ps1')
        task_stale_factor     = 2
        task_include          = @()
        task_scope            = @()
        task_exclude          = @('\Microsoft\*')
        task_ok_codes         = @('0x0', '0x41301', '0x41325', '0x800710E0')
        task_severity         = @{}
        task_ok_codes_by_task = @{}
        expect_enabled        = @()
        log_rules             = @()
        rerun_tasks           = @()
        disks                 = @()
        disk_default_min_free_gb  = 10
        disk_default_min_free_pct = 5
        disk_trend_window_hours   = 6
        disk_trend_alert_hours    = 24
        runner_scopes         = @()
        workflow_expectations = @()
        drain                 = @{ bus_agent = 'vct-nightly'; max_minutes = 90; runner_patterns = @() }
        postgres              = @()
        orphan_processes      = @()
        cert_endpoints        = @()
        cert_warn_days        = 21
        cert_high_days        = 7
        cache_prune           = @()
        compaction            = @{ request_agent = $null; drives = @(); task = $null }
        playbooks             = @{ budget = 1; window_hours = 24; backoff_minutes = 30 }
        alerts                = @{ discord_min_severity = 'medium'; discord_dm_user_id = $null; discord_channel_id = $null }
        fleet_checks          = $false
        digest                = @{ fleet = $false; since_hours = 24 }
    }
    # A policy that exists but cannot be read must fail the run: the defaults have no required drives,
    # fleet checks or recipients, so falling back silently stopped monitoring and let two sweeps resolve
    # a still-missing required drive (eighth review of #346). The file is left in place. No file = defaults.
    # A missing file is "defaults" only on a fresh host. Named by -PolicyPath, or missing where incident state
    # already exists, it means the policy was lost or mistyped, and defaults would drop required drives and
    # recipients and let absence resolve their incidents (sweep row 12).
    if (-not (Test-Path -LiteralPath $Path)) {
        if ($named) { throw "fleet policy not found: $Path (named by -PolicyPath); refusing to run on defaults" }
        $store = Join-Path (Join-Path (Get-FleetReliabilityHome) 'state') 'incidents.json'
        if (Test-Path -LiteralPath $store) { throw "fleet policy not found: $Path, but this host already has incident state; refusing to run on defaults" }
        return $defaults
    }
    try { $loaded = Read-FleetJsonFile -Path $Path -Default @{} -ThrowOnError }
    catch { throw "fleet policy is unreadable, refusing to run on defaults: $($_.Exception.Message)" }
    # Sub-objects merge key by key: '"drain": {"max_minutes": 120}' replaced the whole object and silently
    # dropped bus_agent, turning the DRAIN check off (sweep row 13).
    foreach ($k in @($loaded.Keys)) {
        if (@('drain', 'playbooks', 'alerts', 'digest', 'compaction') -contains $k -and $loaded[$k] -is [System.Collections.IDictionary] -and $defaults[$k] -is [System.Collections.IDictionary]) {
            foreach ($sub in @($loaded[$k].Keys)) { $defaults[$k][$sub] = $loaded[$k][$sub] }
        }
        else { $defaults[$k] = $loaded[$k] }
    }
    Assert-FleetPolicy -Policy $defaults -Path $Path
    return $defaults
}

function Test-FleetPolicyNumber {
    <# A JSON number or a plain numeric string ('15', '1.5'), finite and in range. Null, '', '10%', '-5' and booleans are not. #>
    param($Value, [double]$Min, [double]$Max = [double]::MaxValue, [switch]$Integer, [switch]$ExclusiveMin)
    if ($null -eq $Value -or $Value -is [bool]) { return $false }
    if ($Value -is [int] -or $Value -is [long] -or $Value -is [double] -or $Value -is [decimal] -or $Value -is [single]) { $num = [double]$Value }
    elseif ($Value -is [string] -and $Value -match '^\s*\d+(\.\d+)?\s*$') { $num = [double]::Parse($Value.Trim(), [Globalization.CultureInfo]::InvariantCulture) }
    else { return $false }
    if ([double]::IsNaN($num) -or [double]::IsInfinity($num)) { return $false }
    if ($Integer -and $num -ne [math]::Floor($num)) { return $false }
    if ($ExclusiveMin) { if ($num -le $Min) { return $false } } elseif ($num -lt $Min) { return $false }
    return ($num -le $Max)
}

function Assert-FleetPolicy {
    <#
    Every number the detectors and playbooks compute with is checked once, at load. An explicit null used to
    replace the default and cast to 0 (a free-space floor of 0 GB is always "ok"), and a negative or '10%'
    value either passed or failed deep inside one detector (sweep row 11). Any invalid value refuses the run.
    #>
    param([Parameter(Mandatory)][hashtable]$Policy, [string]$Path)
    $errs = New-Object System.Collections.ArrayList
    $chk = {
        param([string]$Label, $Value, [double]$Min, [double]$Max, [bool]$Int, [bool]$Excl)
        if (-not (Test-FleetPolicyNumber -Value $Value -Min $Min -Max $Max -Integer:$Int -ExclusiveMin:$Excl)) {
            [void]$errs.Add(('{0} = {1}' -f $Label, $(if ($null -eq $Value) { 'null' } else { "'$Value'" })))
        }
    }
    $big = [double]::MaxValue
    & $chk 'task_stale_factor' $Policy.task_stale_factor 0 $big $false $true
    & $chk 'disk_default_min_free_gb' $Policy.disk_default_min_free_gb 0 $big $false $false
    & $chk 'disk_default_min_free_pct' $Policy.disk_default_min_free_pct 0 100 $false $false
    & $chk 'disk_trend_window_hours' $Policy.disk_trend_window_hours 0 $big $false $true
    & $chk 'disk_trend_alert_hours' $Policy.disk_trend_alert_hours 0 $big $false $true
    & $chk 'cert_warn_days' $Policy.cert_warn_days 0 $big $false $false
    & $chk 'cert_high_days' $Policy.cert_high_days 0 $big $false $false
    if ($Policy.drain -is [System.Collections.IDictionary]) { & $chk 'drain.max_minutes' $Policy.drain.max_minutes 0 $big $false $true }
    if ($Policy.playbooks -is [System.Collections.IDictionary]) {
        & $chk 'playbooks.budget' $Policy.playbooks.budget 0 1000 $true $false
        & $chk 'playbooks.window_hours' $Policy.playbooks.window_hours 0 $big $false $true
        & $chk 'playbooks.backoff_minutes' $Policy.playbooks.backoff_minutes 0 $big $false $false
    }
    if ($Policy.digest -is [System.Collections.IDictionary]) {
        & $chk 'digest.since_hours' $Policy.digest.since_hours 0 8760 $true $true
        if ($Policy.digest.Contains('max_sweep_age_hours')) { & $chk 'digest.max_sweep_age_hours' $Policy.digest.max_sweep_age_hours 0 $big $false $true }
    }
    $i = 0
    foreach ($d in @($Policy.disks)) {
        if ($d -is [System.Collections.IDictionary]) {
            if ($d.Contains('min_free_gb')) { & $chk "disks[$i].min_free_gb" $d.min_free_gb 0 $big $false $false }
            if ($d.Contains('min_free_pct')) { & $chk "disks[$i].min_free_pct" $d.min_free_pct 0 100 $false $false }
        }
        $i++
    }
    $i = 0
    foreach ($pg in @($Policy.postgres)) {
        if ($pg -isnot [System.Collections.IDictionary] -or [string]::IsNullOrWhiteSpace([string]$pg.name)) { [void]$errs.Add("postgres[$i].name is missing") }
        else {
            # max_connections and port come from policy only: inferring them from conf files drew three review rounds.
            & $chk "postgres[$i].port" $pg.port 1 65535 $true $false
            & $chk "postgres[$i].max_connections" $pg.max_connections 1 $big $true $false
            if ($pg.Contains('warn_pct')) { & $chk "postgres[$i].warn_pct" $pg.warn_pct 1 100 $false $false }
        }
        $i++
    }
    $i = 0
    foreach ($w in @($Policy.workflow_expectations)) {
        if ($w -is [System.Collections.IDictionary] -and $w.Contains('max_age_days')) { & $chk "workflow_expectations[$i].max_age_days" $w.max_age_days 0 $big $false $true }
        $i++
    }
    $i = 0
    foreach ($r in @($Policy.log_rules)) {
        if ($r -is [System.Collections.IDictionary] -and $r.Contains('tail_lines')) { & $chk "log_rules[$i].tail_lines" $r.tail_lines 1 1000000 $true $false }
        $i++
    }
    if ($errs.Count -gt 0) { throw ("fleet policy {0} is invalid, refusing to run: {1}" -f $Path, ($errs -join '; ')) }
}
