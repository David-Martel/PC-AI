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
    param([Parameter(Mandatory)][ValidateSet('ledger', 'state', 'spool', 'quarantine', 'locks', 'secrets', 'logs')][string]$Kind)
    $homeDir = Get-FleetReliabilityHome
    $rel = switch ($Kind) {
        'quarantine' { 'spool\quarantine' }
        default { $Kind }
    }
    $dir = Join-Path $homeDir $rel
    if (-not (Test-Path -LiteralPath $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
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
    $result = [pscustomobject]@{ ExitCode = -1; StdOut = ''; StdErr = ''; TimedOut = $false }
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
    $outTask = $proc.StandardOutput.ReadToEndAsync()
    $errTask = $proc.StandardError.ReadToEndAsync()
    if (-not $proc.WaitForExit([Math]::Max(1, $TimeoutSeconds) * 1000)) {
        $result.TimedOut = $true
        try { $proc.Kill() } catch { Write-Verbose "kill after timeout failed: $($_.Exception.Message)" }
        [void]$proc.WaitForExit(5000)
    }
    else {
        $proc.WaitForExit()
        $result.ExitCode = $proc.ExitCode
    }
    try { $result.StdOut = $outTask.Result } catch { $result.StdOut = '' }
    try { $result.StdErr = $errTask.Result } catch { $result.StdErr = '' }
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
    param([Parameter(Mandatory)][string]$Path, $Default = @{})
    if (-not (Test-Path -LiteralPath $Path)) { return $Default }
    try {
        $raw = [IO.File]::ReadAllText($Path)
        if ([string]::IsNullOrWhiteSpace($raw)) { return $Default }
        return (ConvertTo-FleetHashtable ($raw | ConvertFrom-Json))
    }
    catch {
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
    Move-Item -LiteralPath $tmp -Destination $Path -Force
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
    if (-not $Path) { $Path = Join-Path (Get-FleetReliabilityHome) 'policy.json' }
    $defaults = @{
        task_health_script    = (Join-Path $PSScriptRoot '..\..\..\Tools\Test-ScheduledTaskHealth.ps1')
        task_stale_factor     = 2
        task_include          = @()
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
    $loaded = Read-FleetJsonFile -Path $Path -Default @{}
    foreach ($k in @($loaded.Keys)) { $defaults[$k] = $loaded[$k] }
    return $defaults
}
