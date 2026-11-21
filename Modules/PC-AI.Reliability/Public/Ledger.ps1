#Requires -Version 5.1
<#
The failure ledger (fleet-ledger/1): one JSONL line per run, anomaly, incident transition,
remediation or digest, kept per host with rotation and forwarded to agent-bus when it is an
alertable kind. The local line is always written first, so a dead bus never loses the record.
Contract: docs/reliability/fleet-ledger.md.
#>

function New-FleetLedgerEvent {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][ValidateSet('task', 'timer', 'service', 'runner', 'job', 'workflow', 'disk', 'postgres', 'process', 'cert', 'sweeper', 'playbook', 'digest', 'alert')][string]$Source,
        [Parameter(Mandatory)][string]$Name,
        [string]$Check = 'run',
        [ValidateSet('run', 'observation', 'incident-open', 'incident-resolved', 'incident-flapping', 'incident-reminder', 'remediation', 'escalation', 'digest')][string]$Kind = 'run',
        [ValidateSet('ok', 'failed', 'stale', 'skipped', 'never-ran', 'degraded', 'unknown')][string]$Status = 'ok',
        [ValidateSet('info', 'low', 'medium', 'high', 'critical')][string]$Severity = 'info',
        [string]$Message = '',
        [Nullable[datetime]]$Start,
        [Nullable[datetime]]$End,
        [Nullable[int]]$ExitCode,
        [string]$LogPath,
        [string]$IncidentKey,
        [hashtable]$Remediation,
        [hashtable]$Details = @{}
    )
    $now = Get-FleetNow
    $hostName = Get-FleetHostName
    if ($Message.Length -gt 2000) { $Message = $Message.Substring(0, 1997) + '...' }
    $rand = -join ((1..8) | ForEach-Object { '{0:x}' -f (Get-Random -Maximum 16) })
    return [ordered]@{
        schema       = $script:FleetLedgerSchema
        event_id     = '{0}-{1}-{2}' -f $hostName, $now.ToString("yyyyMMdd'T'HHmmssfff'Z'", [Globalization.CultureInfo]::InvariantCulture), $rand
        ts           = ConvertTo-FleetIso $now
        host         = $hostName
        source       = $Source
        name         = $Name
        check        = $Check
        kind         = $Kind
        status       = $Status
        severity     = $Severity
        start        = ConvertTo-FleetIso $Start
        end          = ConvertTo-FleetIso $End
        exit_code    = $ExitCode
        message      = $Message
        log_path     = if ([string]::IsNullOrWhiteSpace($LogPath)) { $null } else { $LogPath }
        incident_key = if ([string]::IsNullOrWhiteSpace($IncidentKey)) { $null } else { $IncidentKey }
        remediation  = $Remediation
        details      = $Details
    }
}

function Write-FleetLedgerEvent {
    <#
    Appends one event to <home>\ledger\ledger.jsonl (rotating at FLEET_LEDGER_MAX_BYTES, default
    5 MiB, keeping 5 files) and forwards alertable kinds to agent-bus and Discord. Returns the event.
    Never throws because a transport failed; it throws only when the local append itself fails,
    because then the failure would otherwise go unrecorded.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][System.Collections.IDictionary]$LedgerEvent,
        [switch]$NoForward
    )
    $ledgerDir = Get-FleetPath -Kind ledger
    $ledger = Join-Path $ledgerDir 'ledger.jsonl'
    $maxBytes = 5MB
    if ($env:FLEET_LEDGER_MAX_BYTES) { $maxBytes = [int64]$env:FLEET_LEDGER_MAX_BYTES }
    $line = ($LedgerEvent | ConvertTo-Json -Depth 10 -Compress) + "`n"
    $acquired = $false
    $null = Invoke-FleetWithLock -LockName 'ledger' -TimeoutSeconds 30 -Acquired ([ref]$acquired) -ScriptBlock {
        if ((Test-Path -LiteralPath $ledger) -and ((Get-Item -LiteralPath $ledger).Length -ge $maxBytes)) {
            for ($i = 4; $i -ge 1; $i--) {
                $from = Join-Path $ledgerDir ("ledger.{0}.jsonl" -f $i)
                if (Test-Path -LiteralPath $from) { Move-Item -LiteralPath $from -Destination (Join-Path $ledgerDir ("ledger.{0}.jsonl" -f ($i + 1))) -Force }
            }
            Move-Item -LiteralPath $ledger -Destination (Join-Path $ledgerDir 'ledger.1.jsonl') -Force
        }
        [IO.File]::AppendAllText($ledger, $line, (New-Object System.Text.UTF8Encoding($false)))
    }
    if (-not $acquired) { throw "fleet ledger: could not lock $ledger within 30 s; event not recorded: $($LedgerEvent.event_id)" }

    if (-not $NoForward -and ($script:FleetForwardKinds -contains $LedgerEvent.kind)) {
        $bus = Send-FleetBusEvent -LedgerEvent $LedgerEvent
        $discord = Send-FleetDiscordEvent -LedgerEvent $LedgerEvent
        # Record what each transport did. "Alerts that never arrive" was invisible until someone read a
        # text log; this line makes a lost or skipped delivery queryable from the ledger and the digest.
        $deliveryStatus = 'ok'
        if ($bus -eq 'lost' -and $discord -notlike 'sent*') { $deliveryStatus = 'failed' }
        elseif ($bus -ne 'sent' -or $discord -like 'failed*') { $deliveryStatus = 'degraded' }
        $delivery = New-FleetLedgerEvent -Source alert -Name ([string]$LedgerEvent.event_id) -Check delivery -Kind observation `
            -Status $deliveryStatus -Severity info -Message ("bus={0} discord={1}" -f $bus, $discord) -IncidentKey $LedgerEvent.incident_key `
            -Details @{ for_kind = [string]$LedgerEvent.kind; bus = [string]$bus; discord = [string]$discord }
        $null = Write-FleetLedgerEvent -LedgerEvent $delivery -NoForward
    }
    return $LedgerEvent
}

function Format-FleetBusBody {
    param([Parameter(Mandatory)][System.Collections.IDictionary]$LedgerEvent)
    $subject = '{0} {1}/{2} ({3})' -f $LedgerEvent.host, $LedgerEvent.source, $LedgerEvent.name, $LedgerEvent.check
    $tail = ''
    if ($LedgerEvent.log_path) { $tail += "`nLOG: $($LedgerEvent.log_path)" }
    $tail += "`nEVENT: $($LedgerEvent.event_id)"
    if ($script:FleetFindingKinds -contains $LedgerEvent.kind) {
        $label = if ($LedgerEvent.kind -eq 'incident-open') { $LedgerEvent.status } else { "$($LedgerEvent.kind) $($LedgerEvent.status)" }
        $head = 'FINDING: [{0}] {1}: ' -f $label, $subject
        $sev = "`nSEVERITY: $(([string]$LedgerEvent.severity).ToUpperInvariant())"
    }
    else {
        $prefix = switch ($LedgerEvent.kind) { 'incident-resolved' { 'RESOLVED' } 'remediation' { 'REMEDIATION' } 'digest' { 'DIGEST' } default { 'STATUS' } }
        $head = '{0}: {1}: ' -f $prefix, $subject
        if ($LedgerEvent.kind -eq 'digest') { $head = 'DIGEST: ' }
        $sev = ''
    }
    $room = 1900 - $head.Length - $sev.Length - $tail.Length
    $msg = [string]$LedgerEvent.message
    if ($msg.Length -gt $room) { $msg = $msg.Substring(0, [Math]::Max(0, $room - 3)) + '...' }
    return $head + $msg + $sev + $tail
}

function Get-FleetBusArguments {
    param([Parameter(Mandatory)][System.Collections.IDictionary]$LedgerEvent)
    $schema = if ($script:FleetFindingKinds -contains $LedgerEvent.kind) { 'finding' } else { 'status' }
    $busArgList = @(
        '--from-agent', ('fleet-ledger-{0}' -f $LedgerEvent.host),
        '--to-agent', 'all',
        '--topic', 'fleet-failures',
        '--thread-id', 'fleet-failures',
        '--schema', $schema,
        '--tag', ('host:{0}' -f $LedgerEvent.host),
        '--tag', ('source:{0}' -f $LedgerEvent.source)
    )
    if ($LedgerEvent.incident_key) { $busArgList += @('--tag', ('incident:{0}' -f (Get-FleetShortHash $LedgerEvent.incident_key))) }
    if (@('high', 'critical') -contains $LedgerEvent.severity) { $busArgList += @('--priority', 'high') }
    $busArgList += @('--metadata', ($LedgerEvent | ConvertTo-Json -Depth 10 -Compress), '--body', (Format-FleetBusBody -LedgerEvent $LedgerEvent))
    return , $busArgList
}

function Send-FleetBusEvent {
    <#
    Posts the event to agent-bus (topic and thread fleet-failures). On any failure the same request
    goes to the agent-bus offline spool (<home>\spool\bus-spool.ndjson), which the next sweep replays.
    Returns 'sent', 'spooled' or 'lost' (lost = the local ledger line is the only record).
    #>
    param([Parameter(Mandatory)][System.Collections.IDictionary]$LedgerEvent)
    $exe = Get-FleetAgentBusPath
    $busArgs = Get-FleetBusArguments -LedgerEvent $LedgerEvent
    $timeout = 20
    if ($env:FLEET_BUS_TIMEOUT_SEC) { $timeout = [int]$env:FLEET_BUS_TIMEOUT_SEC }
    $r = Invoke-FleetProcess -FilePath $exe -ArgumentList (@('send') + $busArgs + @('--encoding', 'compact')) -TimeoutSeconds $timeout
    if (-not $r.TimedOut -and $r.ExitCode -eq 0) { return 'sent' }
    $spool = Join-Path (Get-FleetPath -Kind spool) 'bus-spool.ndjson'
    $s = Invoke-FleetWithLock -LockName 'spool' -TimeoutSeconds 30 -ScriptBlock {
        Invoke-FleetProcess -FilePath $exe -ArgumentList (@('spool-send') + $busArgs + @('--spool', $spool, '--encoding', 'compact')) -TimeoutSeconds $timeout
    }
    if ($null -ne $s -and -not $s.TimedOut -and $s.ExitCode -eq 0) { return 'spooled' }
    Write-Warning "fleet ledger: agent-bus send and spool both failed for $($LedgerEvent.event_id); the ledger line is the record."
    return 'lost'
}

function Invoke-FleetSpoolReplay {
    <#
    Replays spooled bus messages. The live spool is renamed first, under the spool lock, so an
    emitter appending during the replay writes a new file instead of being deleted with the old one
    (spool-replay --delete-on-success removes the whole file). A spool that does not parse would fail
    every replay forever, so it is quarantined. Returns a summary object.
    #>
    [CmdletBinding()]
    param()
    $dir = Get-FleetPath -Kind spool
    $live = Join-Path $dir 'bus-spool.ndjson'
    $exe = Get-FleetAgentBusPath
    $summary = [pscustomobject]@{ Replayed = 0; Pending = 0; Quarantined = 0 }
    $null = Invoke-FleetWithLock -LockName 'spool' -TimeoutSeconds 30 -ScriptBlock {
        if ((Test-Path -LiteralPath $live) -and (Get-Item -LiteralPath $live).Length -gt 0) {
            $stamp = (Get-FleetNow).ToString("yyyyMMdd'T'HHmmssfff'Z'", [Globalization.CultureInfo]::InvariantCulture)
            Move-Item -LiteralPath $live -Destination (Join-Path $dir ("bus-spool.{0}.replaying.ndjson" -f $stamp)) -Force
        }
    }
    foreach ($f in @(Get-ChildItem -LiteralPath $dir -Filter 'bus-spool.*.replaying.ndjson' -File | Sort-Object Name)) {
        $r = Invoke-FleetProcess -FilePath $exe -ArgumentList @('spool-replay', '--spool', $f.FullName, '--delete-on-success', '--encoding', 'compact') -TimeoutSeconds 60
        if (-not $r.TimedOut -and $r.ExitCode -eq 0) {
            if (Test-Path -LiteralPath $f.FullName) { Remove-Item -LiteralPath $f.FullName -Force }
            $summary.Replayed++
        }
        elseif (($r.StdErr + $r.StdOut) -match 'JSON parse error') {
            Move-Item -LiteralPath $f.FullName -Destination (Join-Path (Get-FleetPath -Kind quarantine) $f.Name) -Force
            $summary.Quarantined++
        }
        else { $summary.Pending++ }
    }
    return $summary
}

function Get-FleetDiscordToken {
    <#
    Bot token for the Discord transport, readable by S4U and boot tasks without an unlocked vault:
    $env:DISCORD_BOT_TOKEN, else <home>\secrets\discord-bot-token.dpapi (LocalMachine DPAPI).
    LocalMachine scope because CurrentUser DPAPI Protect/Unprotect fails with "Access is denied" in a
    passwordless logon (measured over an SSH key logon with no interactive session, 2026-11-21); the
    file ACL is the boundary.
    Returns $null when no token is available. Never logs the value.
    #>
    if (-not [string]::IsNullOrWhiteSpace($env:DISCORD_BOT_TOKEN)) { return $env:DISCORD_BOT_TOKEN }
    $file = Join-Path (Join-Path (Get-FleetReliabilityHome) 'secrets') 'discord-bot-token.dpapi'
    if (-not (Test-Path -LiteralPath $file)) { return $null }
    try {
        Add-Type -AssemblyName System.Security -ErrorAction SilentlyContinue
        $entropy = [Text.Encoding]::UTF8.GetBytes('fleet-reliability/discord-bot-token/v1')
        $plain = [Security.Cryptography.ProtectedData]::Unprotect([IO.File]::ReadAllBytes($file), $entropy, 'LocalMachine')
        return [Text.Encoding]::UTF8.GetString($plain)
    }
    catch {
        Write-Warning "fleet alerts: the Discord token file exists but could not be decrypted: $($_.Exception.GetType().Name)"
        return $null
    }
}

function Set-FleetDiscordToken {
    <#
    One-time provisioning, run interactively by the owner on each host. Reads the token from a
    SecureString (or from an unlocked Bitwarden item with -FromBitwardenItem) and writes the
    LocalMachine-DPAPI file with an ACL limited to SYSTEM, Administrators and the current user.
    The token is never written to the console or a log.
    #>
    [CmdletBinding(SupportsShouldProcess)]
    param(
        [Parameter(ParameterSetName = 'Secure', Mandatory)][securestring]$Token,
        [Parameter(ParameterSetName = 'Bw', Mandatory)][string]$FromBitwardenItem,
        [Parameter(ParameterSetName = 'Bw', Mandatory)][string]$Field
    )
    if ($PSCmdlet.ParameterSetName -eq 'Bw') {
        $item = (& bw get item $FromBitwardenItem 2>$null) | ConvertFrom-Json
        if ($null -eq $item) { throw 'Bitwarden item not readable (is the vault unlocked?)' }
        $plain = @($item.fields | Where-Object { $_.name -eq $Field } | Select-Object -ExpandProperty value -First 1)[0]
        if ([string]::IsNullOrWhiteSpace($plain)) { throw "Bitwarden item has no field '$Field'" }
    }
    else {
        $ptr = [Runtime.InteropServices.Marshal]::SecureStringToGlobalAllocUnicode($Token)
        try { $plain = [Runtime.InteropServices.Marshal]::PtrToStringUni($ptr) }
        finally { [Runtime.InteropServices.Marshal]::ZeroFreeGlobalAllocUnicode($ptr) }
    }
    $dir = Join-Path (Get-FleetReliabilityHome) 'secrets'
    $file = Join-Path $dir 'discord-bot-token.dpapi'
    if (-not $PSCmdlet.ShouldProcess($file, 'write LocalMachine-DPAPI Discord token')) { return }
    if (-not (Test-Path -LiteralPath $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
    # The ACL is the only boundary for a LocalMachine-DPAPI blob, so it must be in place and verified
    # BEFORE the secret is written; a failed icacls aborts (codex review, #346).
    $aclExit = Set-FleetSecretDirAcl -Path $dir
    if ($aclExit -ne 0) { throw "could not restrict $dir (icacls exit $aclExit); token NOT written" }
    $extra = @(Get-FleetSecretDirAclViolation -Path $dir)
    if ($extra.Count -gt 0) { throw "$dir still grants access to: $($extra -join ', '); token NOT written" }
    Add-Type -AssemblyName System.Security -ErrorAction SilentlyContinue
    $entropy = [Text.Encoding]::UTF8.GetBytes('fleet-reliability/discord-bot-token/v1')
    $blob = [Security.Cryptography.ProtectedData]::Protect([Text.Encoding]::UTF8.GetBytes($plain), $entropy, 'LocalMachine')
    [IO.File]::WriteAllBytes($file, $blob)
    $plain = $null
    return [pscustomobject]@{ Path = $file; Bytes = $blob.Length }
}

function Set-FleetSecretDirAcl {
    <# Restricts a directory to SYSTEM, Administrators and the current user (inheritance removed). Returns the icacls exit code. #>
    param([Parameter(Mandatory)][string]$Path)
    $user = [Security.Principal.WindowsIdentity]::GetCurrent().Name
    & icacls.exe $Path /inheritance:r /grant:r '*S-1-5-18:(OI)(CI)F' '*S-1-5-32-544:(OI)(CI)F' "${user}:(OI)(CI)F" | Out-Null
    return $LASTEXITCODE
}

function Get-FleetSecretDirAclViolation {
    <# Identities other than SYSTEM, Administrators and the current user that have any access to Path. #>
    param([Parameter(Mandatory)][string]$Path)
    $allowed = @('S-1-5-18', 'S-1-5-32-544', [Security.Principal.WindowsIdentity]::GetCurrent().User.Value)
    $bad = @()
    foreach ($rule in (Get-Acl -LiteralPath $Path).Access) {
        if ($rule.AccessControlType -ne 'Allow') { continue }
        try { $sid = $rule.IdentityReference.Translate([Security.Principal.SecurityIdentifier]).Value } catch { $sid = [string]$rule.IdentityReference }
        if ($allowed -notcontains $sid) { $bad += [string]$rule.IdentityReference }
    }
    return $bad
}

function Invoke-FleetDiscordPost {
    <# The one network call of the Discord transport; mocked in tests. #>
    param([string]$Uri, [hashtable]$Headers, [string]$Body)
    return Invoke-RestMethod -Method Post -Uri $Uri -Headers $Headers -Body $Body -ContentType 'application/json' -TimeoutSec 15
}

function Send-FleetDiscordEvent {
    <#
    Second transport. Sends alertable kinds at or above alerts.discord_min_severity (resolutions and
    digests always) to a channel (alerts.discord_channel_id) or as a DM (alerts.discord_dm_user_id).
    Returns 'sent', 'skipped(<why>)' or 'failed(<type>)'; never throws.
    #>
    param([Parameter(Mandatory)][System.Collections.IDictionary]$LedgerEvent, [hashtable]$Policy)
    # The entry points (sweep, digest, wrapper) set the policy they loaded from -PolicyPath, so forwarding
    # deep inside the incident code uses the same recipients and floor (codex review, #346).
    if (-not $Policy) { $Policy = $script:FleetEffectivePolicy }
    if (-not $Policy) { $Policy = Get-FleetPolicy }
    $alerts = $Policy.alerts
    $min = 'medium'
    if ($alerts -and $alerts.discord_min_severity) { $min = $alerts.discord_min_severity }
    $always = @('incident-resolved', 'digest')
    if (($always -notcontains $LedgerEvent.kind) -and ($script:FleetSeverityRank[[string]$LedgerEvent.severity] -lt $script:FleetSeverityRank[$min])) { return 'skipped(severity)' }
    if ($LedgerEvent.kind -eq 'remediation') { return 'skipped(kind)' }
    $channel = $null; $dm = $null
    if ($alerts) { $channel = $alerts.discord_channel_id; $dm = $alerts.discord_dm_user_id }
    if (-not $channel -and -not $dm) { return 'skipped(no recipient)' }
    $token = Get-FleetDiscordToken
    if (-not $token) { return 'skipped(no token)' }
    $headers = @{ Authorization = "Bot $token"; 'User-Agent' = 'fleet-reliability (PC-AI, 1.0)' }
    try {
        if (-not $channel) {
            $dmChannel = Invoke-FleetDiscordPost -Uri 'https://discord.com/api/v10/users/@me/channels' -Headers $headers -Body (@{ recipient_id = [string]$dm } | ConvertTo-Json -Compress)
            $channel = $dmChannel.id
        }
        $content = Format-FleetBusBody -LedgerEvent $LedgerEvent
        if ($content.Length -gt 1900) { $content = $content.Substring(0, 1897) + '...' }
        $null = Invoke-FleetDiscordPost -Uri ("https://discord.com/api/v10/channels/{0}/messages" -f $channel) -Headers $headers -Body (@{ content = $content } | ConvertTo-Json -Compress)
        return 'sent'
    }
    catch { return "failed($($_.Exception.GetType().Name))" }
    finally { $token = $null; $headers = $null }
}
