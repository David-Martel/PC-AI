#Requires -Version 7.2
<#
.SYNOPSIS
Re-validates this host's USB4/Thunderbolt links after boot, logon, replug or a network change.
.DESCRIPTION
Rediscovers the USB4 network adapters by interface GUID and MAC, compares them with the recorded
`pcai-tb route-plan` output, checks link state, planned addresses, host routes, forwarding and
peer reachability, reports PC-AI's own USB4 tasks or units, and writes a status file. When the
state fingerprint changes it posts one compact status to agent-bus.

Observe mode (the default) changes no address, route, forwarding or adapter setting. When it
finds drift it writes this host's guarded apply block to a review file and prints the command
that runs it elevated.

Apply mode is the owner opt-in. It runs only when the process is elevated, an admin-owned
auto-apply policy says enabled, the plan file is admin-owned, and every guard (ifIndex, MAC,
interface GUID) still matches. It then runs this host's guard and apply lines from the plan.
A replaced adapter instance (new GUID) is never applied: the plan must be regenerated from a
fresh observation of every host, which `route-plan` itself requires.

See docs/usb4/link-health.md.
.PARAMETER Mode
Observe (default) or Apply. Apply still refuses unless the admin-owned policy enables it.
.PARAMETER Trigger
What started this run (Manual, Logon, Boot, Event, Timer, Systemd). Event, Logon, Boot and
Timer runs are debounced by MinIntervalSeconds.
.PARAMETER StatusRoot
Directory for status.json, history.jsonl and review files. Defaults per user and platform.
.PARAMETER ConfigPath
Optional JSON configuration; see docs/usb4/link-health.md.
.PARAMETER PlanPath
Route plan to compare against. Overrides the configuration.
.PARAMETER NoBus
Do not post to agent-bus.
.PARAMETER ForceNotify
Post to agent-bus even when the fingerprint did not change.
.PARAMETER DryRun
Collect and print the status only. Writes no file, posts nothing, restarts nothing.
.EXAMPLE
pwsh -NoProfile -File Tools/SystemScripts/Networking/Invoke-PcaiLinkHealth.ps1 -DryRun
.EXAMPLE
pwsh -NoProfile -File Tools/SystemScripts/Networking/Invoke-PcaiLinkHealth.ps1 -Trigger Event
#>
[CmdletBinding(PositionalBinding = $false)]
param(
    [ValidateSet('Observe', 'Apply')][string]$Mode = 'Observe',
    [ValidateSet('Manual', 'Logon', 'Boot', 'Event', 'Timer', 'Systemd')][string]$Trigger = 'Manual',
    [string]$StatusRoot,
    [string]$ConfigPath,
    [string]$PlanPath,
    [ValidateRange(0, 3600)][int]$MinIntervalSeconds = 20,
    [switch]$NoBus,
    [switch]$ForceNotify,
    [switch]$DryRun,
    [switch]$PassThru,
    [Alias('h')][switch]$Help
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$script:PcaiLinkHealthSchema = 'pcai.link-health-status/v1'
$script:PcaiAutoApplySchema = 'pcai.link-health-auto-apply/v1'
$script:PcaiAdminSids = @('S-1-5-32-544', 'S-1-5-18')
# Rights that let an account change a file or directory: write/append data, write attributes and
# extended attributes, delete, delete child, change permissions, take ownership, generic write/all.
$script:PcaiWriteRightsMask = [int64]0x500D0156

function Get-PcaiHostName {
    [CmdletBinding()]
    param()
    $name = if ($env:COMPUTERNAME) { $env:COMPUTERNAME } else { [System.Net.Dns]::GetHostName() }
    return ($name -split '\.')[0].ToLowerInvariant()
}

function Test-PcaiElevated {
    [CmdletBinding()]
    param()
    if ($IsWindows) {
        $id = [Security.Principal.WindowsIdentity]::GetCurrent()
        return ([Security.Principal.WindowsPrincipal]$id).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
    }
    return ((& id -u) -eq '0')
}

function Get-PcaiLinkHealthDefaultRoot {
    [CmdletBinding()]
    param([ValidateSet('Observe', 'Apply')][string]$RunMode = 'Observe')
    if ($IsWindows) {
        if ($RunMode -eq 'Apply' -or [Security.Principal.WindowsIdentity]::GetCurrent().User.Value -eq 'S-1-5-18') {
            return (Join-Path $env:ProgramData 'pcai-usb4\link-health')
        }
        return (Join-Path $env:LOCALAPPDATA 'PC-AI\link-health')
    }
    $state = if ($env:XDG_STATE_HOME) { $env:XDG_STATE_HOME } else { Join-Path $HOME '.local/state' }
    return (Join-Path $state 'pcai/link-health')
}

function Read-PcaiLinkHealthConfig {
    [CmdletBinding()]
    param([string]$Path, [Parameter(Mandatory)][string]$Root)
    $config = [ordered]@{
        planPath             = $null
        busEnabled           = $true
        busThread            = 'machine-hygiene'
        restartAllowList     = @('\PC-AI\Configure-ThunderboltSwitchRoutes', 'pcai-usb4-observer.service')
        tcpProbePort         = 22
        regenerateHint       = 'Recollect every host and rerun pcai-tb route-plan (docs/usb4/multi-hop-routing.md), then replace the recorded plan.'
    }
    if ($Path -and (Test-Path -LiteralPath $Path)) {
        $raw = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json -AsHashtable
        foreach ($key in @($raw.Keys)) {
            if (-not $config.Contains($key)) { throw "Unknown link-health configuration key '$key' in $Path" }
            $config[$key] = $raw[$key]
        }
    }
    if (-not $config.planPath) { $config.planPath = Join-Path $Root 'route-plan.json' }
    return [pscustomobject]$config
}

function Invoke-PcaiBoundedCommand {
    <# Runs a native command with a deadline; never throws for a non-zero exit. #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$FilePath,
        [string[]]$ArgumentList = @(),
        [ValidateRange(1, 600)][int]$TimeoutSeconds = 20
    )
    $psi = [Diagnostics.ProcessStartInfo]::new($FilePath)
    foreach ($a in $ArgumentList) { $psi.ArgumentList.Add($a) }
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true
    try { $p = [Diagnostics.Process]::Start($psi) } catch {
        return [pscustomobject]@{ ExitCode = -1; Stdout = ''; Stderr = $_.Exception.Message; TimedOut = $false }
    }
    $out = $p.StandardOutput.ReadToEndAsync()
    $err = $p.StandardError.ReadToEndAsync()
    $timedOut = -not $p.WaitForExit($TimeoutSeconds * 1000)
    if ($timedOut) { try { $p.Kill($true) } catch { Write-Verbose "kill failed: $_" } ; $null = $p.WaitForExit(5000) }
    $code = if ($timedOut) { -2 } else { $p.ExitCode }
    return [pscustomobject]@{ ExitCode = $code; Stdout = $out.GetAwaiter().GetResult(); Stderr = $err.GetAwaiter().GetResult(); TimedOut = $timedOut }
}

function Get-PcaiUsb4Adapter {
    <#
    .SYNOPSIS
    Lists present USB4/Thunderbolt network adapters with identity, link state, IPv4 and forwarding.
    #>
    [CmdletBinding()]
    param()
    $result = [Collections.Generic.List[object]]::new()
    if ($IsWindows) {
        $adapters = @(Get-NetAdapter -ErrorAction SilentlyContinue | Where-Object { $_.InterfaceDescription -match 'USB4|Thunderbolt' })
        foreach ($a in $adapters) {
            $ipv4 = @(Get-NetIPAddress -InterfaceIndex $a.ifIndex -AddressFamily IPv4 -ErrorAction SilentlyContinue |
                    ForEach-Object { '{0}/{1}' -f $_.IPAddress, $_.PrefixLength })
            $ipif = Get-NetIPInterface -InterfaceIndex $a.ifIndex -AddressFamily IPv4 -ErrorAction SilentlyContinue
            $result.Add([pscustomobject]@{
                    InterfaceIndex = [int]$a.ifIndex
                    Name           = [string]$a.Name
                    Description    = [string]$a.InterfaceDescription
                    Guid           = ([string]$a.InterfaceGuid).Trim('{}').ToLowerInvariant()
                    Mac            = ([string]$a.MacAddress).ToUpperInvariant()
                    Status         = [string]$a.Status
                    LinkSpeed      = [string]$a.LinkSpeed
                    Ipv4           = @($ipv4 | Sort-Object)
                    Forwarding     = if ($ipif) { [string]$ipif.Forwarding } else { 'Unknown' }
                })
        }
    } elseif ($IsLinux) {
        foreach ($dir in @(Get-ChildItem -LiteralPath '/sys/class/net' -ErrorAction SilentlyContinue)) {
            $driverLink = Join-Path $dir.FullName 'device/driver'
            if (-not (Test-Path -LiteralPath $driverLink)) { continue }
            $driver = Split-Path -Leaf ((Get-Item -LiteralPath $driverLink).ResolvedTarget)
            if ($driver -notin @('thunderbolt-net', 'thunderbolt_net')) { continue }
            $ifname = $dir.Name
            $read = { param($f) $p = Join-Path $dir.FullName $f; if (Test-Path -LiteralPath $p) { (Get-Content -LiteralPath $p -Raw).Trim() } else { '' } }
            $ipv4 = @()
            $ipJson = Invoke-PcaiBoundedCommand -FilePath 'ip' -ArgumentList @('-j', '-4', 'addr', 'show', 'dev', $ifname) -TimeoutSeconds 5
            if ($ipJson.ExitCode -eq 0 -and $ipJson.Stdout.Trim()) {
                $ipv4 = @(($ipJson.Stdout | ConvertFrom-Json) | ForEach-Object { $_.addr_info } | ForEach-Object { '{0}/{1}' -f $_.local, $_.prefixlen })
            }
            $fwdPath = "/proc/sys/net/ipv4/conf/$ifname/forwarding"
            $fwd = if (Test-Path -LiteralPath $fwdPath) { if ((Get-Content -LiteralPath $fwdPath -Raw).Trim() -eq '1') { 'Enabled' } else { 'Disabled' } } else { 'Unknown' }
            $result.Add([pscustomobject]@{
                    InterfaceIndex = [int](& $read 'ifindex')
                    Name           = $ifname
                    Description    = $driver
                    Guid           = ''
                    Mac            = ((& $read 'address') -replace ':', '-').ToUpperInvariant()
                    Status         = if ((& $read 'operstate') -eq 'up') { 'Up' } else { 'Down' }
                    LinkSpeed      = ''
                    Ipv4           = @($ipv4 | Sort-Object)
                    Forwarding     = $fwd
                })
        }
    }
    return , $result.ToArray()
}

function Get-PcaiIpv4HostRoute {
    [CmdletBinding()]
    param([int[]]$InterfaceIndex)
    if (-not $IsWindows -or -not $InterfaceIndex) { return , @() }
    $routes = foreach ($i in $InterfaceIndex) {
        Get-NetRoute -InterfaceIndex $i -AddressFamily IPv4 -ErrorAction SilentlyContinue |
            ForEach-Object { [pscustomobject]@{ InterfaceIndex = [int]$_.ifIndex; DestinationPrefix = [string]$_.DestinationPrefix; NextHop = [string]$_.NextHop } }
    }
    return , @($routes)
}

function Read-PcaiRoutePlan {
    [CmdletBinding()]
    param([string]$Path)
    if (-not $Path -or -not (Test-Path -LiteralPath $Path)) { return $null }
    $plan = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
    if ($plan.schema -ne 'pcai.usb4-route-plan/v1') { throw "Route plan $Path has schema '$($plan.schema)', expected pcai.usb4-route-plan/v1" }
    return $plan
}

function Get-PcaiRoutePlanId {
    <# The planner embeds its plan ID in every journal path; there is no separate field. #>
    [CmdletBinding()]
    param([Parameter(Mandatory)]$Plan)
    foreach ($h in @($Plan.hosts)) {
        foreach ($a in @($h.actions)) {
            if ($a.apply -and ([string]$a.apply -match "'([0-9a-f]{16})\.journal'")) { return $Matches[1] }
        }
    }
    return $null
}

function Compare-PcaiLinkPlan {
    <#
    .SYNOPSIS
    Classifies this host's live USB4 state against its block of a route plan. Pure function.
    .OUTPUTS
    State: NoPlan, NotInPlan, Absent, Healthy, InSync, Drift, LinkDown or InstanceChanged.
    #>
    [CmdletBinding()]
    param(
        $Plan,
        [Parameter(Mandatory)][string]$HostName,
        [AllowEmptyCollection()][object[]]$Adapter = @(),
        [AllowEmptyCollection()][object[]]$Route = @()
    )
    $findings = [Collections.Generic.List[object]]::new()
    $up = @($Adapter | Where-Object Status -eq 'Up')
    if (-not $Plan) {
        $state = if ($Adapter.Count -eq 0) { 'Absent' } else { 'NoPlan' }
        return [pscustomobject]@{ State = $state; GuardsMatch = $false; Findings = $findings.ToArray(); PlanHost = $null }
    }
    $planHost = @($Plan.hosts | Where-Object host -eq $HostName) | Select-Object -First 1
    $endpoints = @(foreach ($e in @($Plan.edges)) { foreach ($ep in @($e.endpoints)) { if ($ep.host -eq $HostName) { $ep } } })
    if (-not $planHost -and $endpoints.Count -eq 0) {
        $state = if ($Adapter.Count -eq 0) { 'Absent' } else { 'NotInPlan' }
        return [pscustomobject]@{ State = $state; GuardsMatch = $false; Findings = $findings.ToArray(); PlanHost = $null }
    }

    $guardsMatch = $true
    $linkDown = $false
    foreach ($ep in $endpoints) {
        $guid = ([string]$ep.interface_guid).Trim('{}').ToLowerInvariant()
        $mac = ([string]$ep.mac).ToUpperInvariant()
        $atIndex = @($Adapter | Where-Object InterfaceIndex -eq ([int]$ep.interface_index)) | Select-Object -First 1
        $byGuid = @($Adapter | Where-Object Guid -eq $guid) | Select-Object -First 1
        if ($atIndex -and $atIndex.Guid -eq $guid -and $atIndex.Mac -eq $mac) {
            $status = if ($atIndex.Status -eq 'Up') { 'Match' } else { $linkDown = $true; 'LinkDown' }
        } elseif ($byGuid) {
            $guardsMatch = $false; $status = 'IndexChanged'
        } elseif ($atIndex) {
            $guardsMatch = $false; $status = 'InstanceReplaced'
        } else {
            $guardsMatch = $false; $status = 'AdapterAbsent'
        }
        $findings.Add([pscustomobject]@{ Kind = 'Guard'; InterfaceIndex = [int]$ep.interface_index; Target = "$mac $guid"; Status = $status })
    }
    $plannedGuids = @($endpoints | ForEach-Object { ([string]$_.interface_guid).Trim('{}').ToLowerInvariant() })
    foreach ($a in $up) {
        if ($a.Guid -and $a.Guid -notin $plannedGuids) {
            $findings.Add([pscustomobject]@{ Kind = 'UnplannedAdapter'; InterfaceIndex = $a.InterfaceIndex; Target = "$($a.Mac) $($a.Guid)"; Status = 'Unplanned' })
        }
    }

    $missing = 0
    foreach ($action in @(if ($planHost) { $planHost.actions })) {
        $idx = [int]$action.interface_index
        $live = @($Adapter | Where-Object InterfaceIndex -eq $idx) | Select-Object -First 1
        $present = $false
        switch ([string]$action.kind) {
            'EdgeAddress' { $present = [bool]($live -and ([string]$action.target -in @($live.Ipv4))) }
            'HostRoute' {
                if ([string]$action.target -match '^(?<dst>\S+)\s+via\s+(?<hop>\S+)$') {
                    $dst = $Matches.dst; $hop = $Matches.hop
                    $present = [bool](@($Route | Where-Object { $_.InterfaceIndex -eq $idx -and $_.DestinationPrefix -eq $dst -and $_.NextHop -eq $hop }).Count)
                }
            }
            'Forwarding' { $present = [bool]($live -and $live.Forwarding -eq 'Enabled') }
            default { $present = $false }
        }
        if (-not $present) { $missing++ }
        $findings.Add([pscustomobject]@{ Kind = [string]$action.kind; InterfaceIndex = $idx; Target = [string]$action.target; Status = $(if ($present) { 'Present' } else { 'Missing' }) })
    }

    $state = if (-not $guardsMatch) { 'InstanceChanged' } elseif ($linkDown) { 'LinkDown' } elseif ($missing -gt 0) { 'Drift' } else { 'InSync' }
    return [pscustomobject]@{ State = $state; GuardsMatch = $guardsMatch; Findings = $findings.ToArray(); PlanHost = $planHost }
}

function Get-PcaiReachabilityTarget {
    [CmdletBinding()]
    param($Plan, [Parameter(Mandatory)][string]$HostName)
    if (-not $Plan) { return , @() }
    return , @(@($Plan.paths) | Where-Object { $_.from -eq $HostName -and $_.status -in @('Direct', 'Routed') -and $_.destination_ipv4 } |
            ForEach-Object { [pscustomobject]@{ Peer = [string]$_.to; Address = [string]$_.destination_ipv4; Status = [string]$_.status } })
}

function Test-PcaiTcpPort {
    [CmdletBinding()]
    param([Parameter(Mandatory)][string]$Address, [int]$Port = 22, [int]$TimeoutMilliseconds = 1500)
    $client = [Net.Sockets.TcpClient]::new()
    try {
        $task = $client.ConnectAsync($Address, $Port)
        return ($task.Wait($TimeoutMilliseconds) -and $client.Connected)
    } catch { return $false } finally { $client.Dispose() }
}

function Test-PcaiPeerReachability {
    [CmdletBinding()]
    param([AllowEmptyCollection()][object[]]$Target = @(), [int]$TcpPort = 22)
    return , @(foreach ($t in $Target) {
            $ping = $false
            try { $ping = [bool](Test-Connection -TargetName $t.Address -Count 1 -TimeoutSeconds 1 -Quiet -ErrorAction SilentlyContinue) } catch { $ping = $false }
            $tcp = if ($TcpPort -gt 0) { Test-PcaiTcpPort -Address $t.Address -Port $TcpPort } else { $null }
            [pscustomobject]@{ Peer = $t.Peer; Address = $t.Address; Path = $t.Status; Ping = $ping; Tcp = $tcp; TcpPort = $TcpPort }
        })
}

function Get-PcaiUsb4Service {
    <# Reports PC-AI's own USB4 scheduled tasks (Windows) or systemd units (Linux). Read-only. #>
    [CmdletBinding()]
    param()
    $items = [Collections.Generic.List[object]]::new()
    if ($IsWindows) {
        $tasks = @(Get-ScheduledTask -ErrorAction SilentlyContinue | Where-Object {
                ($_.TaskPath -eq '\PC-AI\' -and $_.TaskName -match 'Thunderbolt|Usb4') -or $_.TaskName -match '^Rclone-.*Thunderbolt' })
        foreach ($t in $tasks) {
            $info = $t | Get-ScheduledTaskInfo -ErrorAction SilentlyContinue
            $items.Add([pscustomobject]@{
                    Name       = "$($t.TaskPath)$($t.TaskName)"
                    Kind       = 'ScheduledTask'
                    State      = [string]$t.State
                    Principal  = [string]$t.Principal.UserId
                    LastResult = if ($info) { [int64]$info.LastTaskResult } else { $null }
                    LastRun    = if ($info -and $info.LastRunTime) { $info.LastRunTime.ToUniversalTime().ToString('o') } else { $null }
                })
        }
    } elseif ($IsLinux -and (Get-Command systemctl -ErrorAction SilentlyContinue)) {
        foreach ($scope in @('system', 'user')) {
            $scopeArgs = if ($scope -eq 'user') { @('--user') } else { @() }
            $r = Invoke-PcaiBoundedCommand -FilePath 'systemctl' -ArgumentList ($scopeArgs + @('list-unit-files', 'pcai-usb4*', '--no-legend', '--no-pager')) -TimeoutSeconds 10
            foreach ($line in @($r.Stdout -split "`n" | Where-Object { $_.Trim() })) {
                $parts = $line.Trim() -split '\s+'
                $active = Invoke-PcaiBoundedCommand -FilePath 'systemctl' -ArgumentList ($scopeArgs + @('is-active', $parts[0])) -TimeoutSeconds 5
                $items.Add([pscustomobject]@{ Name = $parts[0]; Kind = "systemd-$scope"; State = "$($parts[1])/$($active.Stdout.Trim())"; Principal = $scope; LastResult = $active.ExitCode; LastRun = $null })
            }
        }
    }
    return , $items.ToArray()
}

function Get-PcaiServiceRecoveryPlan {
    <#
    .SYNOPSIS
    Chooses which PC-AI USB4 tasks or units to restart. Pure function.
    .DESCRIPTION
    Only allow-listed, enabled, read-only observers that last failed are restarted. A disabled
    task stays disabled: enabling one is an owner decision (\PC-AI\Usb4PeerNetwork runs -Apply).
    #>
    [CmdletBinding()]
    param([AllowEmptyCollection()][object[]]$Service = @(), [string[]]$AllowList = @())
    return , @(foreach ($s in $Service) {
            $allowed = $s.Name -in $AllowList -or (Split-Path -Leaf $s.Name) -in $AllowList
            $decision = if (-not $allowed) { 'ReportOnly' }
            elseif ($s.State -match '^Disabled' -or $s.State -match '^disabled') { 'LeaveDisabled' }
            elseif ($s.State -eq 'Running' -or $s.State -match '/active$') { 'Running' }
            elseif ($null -ne $s.LastResult -and $s.LastResult -ne 0) { 'Restart' }
            else { 'Healthy' }
            [pscustomobject]@{ Name = $s.Name; Kind = $s.Kind; State = $s.State; Decision = $decision }
        })
}

function Invoke-PcaiServiceRecovery {
    [CmdletBinding()]
    param([AllowEmptyCollection()][object[]]$Decision = @())
    return , @(foreach ($d in @($Decision | Where-Object Decision -eq 'Restart')) {
            $outcome = 'Started'
            try {
                if ($d.Kind -eq 'ScheduledTask') {
                    $path = $d.Name.Substring(0, $d.Name.LastIndexOf('\') + 1)
                    Start-ScheduledTask -TaskPath $path -TaskName (Split-Path -Leaf $d.Name) -ErrorAction Stop
                } elseif ($d.Kind -eq 'systemd-user') {
                    $r = Invoke-PcaiBoundedCommand -FilePath 'systemctl' -ArgumentList @('--user', 'restart', $d.Name) -TimeoutSeconds 30
                    if ($r.ExitCode -ne 0) { $outcome = "Failed: $($r.Stderr.Trim())" }
                } else {
                    $outcome = 'NeedsRoot'
                }
            } catch { $outcome = "Failed: $($_.Exception.Message)" }
            [pscustomobject]@{ Name = $d.Name; Outcome = $outcome }
        })
}

function Get-PcaiObserverReceipt {
    <# The existing read-only startup observer's receipt (docs/usb4/windows-boot.md), if any. #>
    [CmdletBinding()]
    param([string]$Path = (Join-Path $HOME '.machine/logs/usb4-boot/current.json'))
    if (-not (Test-Path -LiteralPath $Path)) { return $null }
    $item = Get-Item -LiteralPath $Path
    $state = $null
    # pcai.usb4-boot-observation/v1 keeps the verdict in result.State.
    try {
        $doc = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json -AsHashtable
        $state = if ($doc.ContainsKey('result') -and $doc.result -is [Collections.IDictionary]) { [string]$doc.result['State'] } else { 'Unknown' }
    } catch { $state = 'Unreadable' }
    return [pscustomobject]@{ Path = $Path; State = $state; AgeMinutes = [math]::Round(([DateTime]::UtcNow - $item.LastWriteTimeUtc).TotalMinutes, 1) }
}

function Test-PcaiAdminOwnedPath {
    <#
    .SYNOPSIS
    True when a file or directory is safe for a SYSTEM task to trust.
    .DESCRIPTION
    Auto-apply executes command text from the plan, so a user-writable policy, plan or status
    root would be a privilege escalation. The path and its parent must be owned by
    Administrators or SYSTEM and grant no other account a right to change them. Every further
    ancestor up to the volume root must not be a reparse point and must be owned by
    Administrators, SYSTEM or TrustedInstaller; an ancestor may still let Users create entries
    (as %ProgramData% does), because that does not let them replace an existing child.
    #>
    [CmdletBinding()]
    param([Parameter(Mandatory)][string]$Path)
    if (-not $IsWindows) { return $false }
    if (-not (Test-Path -LiteralPath $Path)) { return $false }
    $trustedOwners = $script:PcaiAdminSids + @('S-1-5-80-956008885-3418522649-1831038044-1853292631-2271478464')
    $chain = [Collections.Generic.List[string]]::new()
    $cursor = (Get-Item -LiteralPath $Path -Force).FullName
    while ($cursor) { $chain.Add($cursor); $cursor = Split-Path -Parent $cursor }
    for ($i = 0; $i -lt $chain.Count; $i++) {
        $p = $chain[$i]
        $item = Get-Item -LiteralPath $p -Force
        if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) { return $false }
        if ($i -eq $chain.Count - 1) { break }   # the volume root itself
        $acl = Get-Acl -LiteralPath $p
        $owner = $acl.GetOwner([Security.Principal.SecurityIdentifier]).Value
        if ($i -le 1 -and $owner -notin $script:PcaiAdminSids) { return $false }
        if ($owner -notin $trustedOwners) { return $false }
        # The path itself: no other account may write it at all. Every ancestor: no other account
        # may delete, rename, re-permission or take it over (Delete, DeleteChild, WriteDAC,
        # WriteOwner, generic write/all); creating new entries, as %ProgramData% allows, is fine.
        $mask = if ($i -eq 0) { $script:PcaiWriteRightsMask } else { [int64]0x500D0040 }
        foreach ($rule in $acl.GetAccessRules($true, $true, [Security.Principal.SecurityIdentifier])) {
            if ($rule.AccessControlType -ne 'Allow' -or $rule.IdentityReference.Value -in $script:PcaiAdminSids) { continue }
            if ($rule.PropagationFlags -band [Security.AccessControl.PropagationFlags]::InheritOnly) { continue }
            if ([int64]$rule.FileSystemRights -band $mask) { return $false }
        }
    }
    return $true
}

function New-PcaiProtectedStatusRoot {
    <#
    .SYNOPSIS
    Creates a directory owned by Administrators with full control for Administrators and SYSTEM
    and read access for Users, so a standard user cannot plant files the SYSTEM task trusts.
    #>
    [CmdletBinding()]
    param([Parameter(Mandatory)][string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) { $null = New-Item -ItemType Directory -Path $Path -Force }
    $acl = [Security.AccessControl.DirectorySecurity]::new()
    $acl.SetAccessRuleProtection($true, $false)
    $acl.SetOwner([Security.Principal.SecurityIdentifier]'S-1-5-32-544')
    foreach ($sid in $script:PcaiAdminSids) {
        $acl.AddAccessRule([Security.AccessControl.FileSystemAccessRule]::new([Security.Principal.SecurityIdentifier]$sid, 'FullControl', 'ContainerInherit, ObjectInherit', 'None', 'Allow'))
    }
    $acl.AddAccessRule([Security.AccessControl.FileSystemAccessRule]::new([Security.Principal.SecurityIdentifier]'S-1-5-32-545', 'ReadAndExecute', 'ContainerInherit, ObjectInherit', 'None', 'Allow'))
    Set-Acl -LiteralPath $Path -AclObject $acl
}

function Get-PcaiAutoApplyPolicy {
    [CmdletBinding()]
    param([string]$Path = $(if ($IsWindows) { Join-Path $env:ProgramData 'pcai-usb4\link-health\auto-apply.json' } else { '/etc/pcai-usb4/link-health-auto-apply.json' }))
    $policy = [pscustomobject]@{ Path = $Path; Enabled = $false; PlanPath = $null; Reason = 'No auto-apply policy (default off)' }
    if (-not (Test-Path -LiteralPath $Path)) { return $policy }
    if (-not (Test-PcaiAdminOwnedPath -Path $Path)) { $policy.Reason = 'Policy is not admin-owned or is writable by another account; ignored'; return $policy }
    $raw = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
    if ($raw.schema -ne $script:PcaiAutoApplySchema) { $policy.Reason = "Policy schema '$($raw.schema)' is not $script:PcaiAutoApplySchema"; return $policy }
    $policy.PlanPath = [string]$raw.planPath
    if ($raw.enabled -ne $true) { $policy.Reason = 'Policy present, enabled=false'; return $policy }
    if (-not $policy.PlanPath -or -not (Test-PcaiAdminOwnedPath -Path $policy.PlanPath)) { $policy.Reason = 'Policy planPath is missing or not admin-owned'; return $policy }
    $policy.Enabled = $true
    $policy.Reason = 'Enabled by admin-owned policy'
    return $policy
}

function Get-PcaiAutoApplyDecision {
    <# Decides whether Apply mode may run this host's plan block. Pure function. #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$RunMode,
        [Parameter(Mandatory)][bool]$Elevated,
        [Parameter(Mandatory)]$Policy,
        [Parameter(Mandatory)]$Comparison
    )
    $refuse = {
        param($r) [pscustomobject]@{ Apply = $false; Reason = $r }
    }
    if ($RunMode -ne 'Apply') { return (& $refuse 'Observe mode') }
    if (-not $IsWindows) { return (& $refuse 'route-plan apply lines are Windows-only') }
    if (-not $Policy.Enabled) { return (& $refuse $Policy.Reason) }
    if (-not $Elevated) { return (& $refuse 'Apply mode needs an elevated process') }
    switch ($Comparison.State) {
        'Drift' { return [pscustomobject]@{ Apply = $true; Reason = 'Guards match and planned state is missing' } }
        'InstanceChanged' { return (& $refuse 'Adapter instance changed; the plan must be regenerated from a fresh observation of every host') }
        default { return (& $refuse "Nothing to apply in state $($Comparison.State)") }
    }
}

function New-PcaiHostApplyScript {
    <# Renders this host's guard, preflight and apply lines. The planner's lines are idempotent and journaled. #>
    [CmdletBinding()]
    param([Parameter(Mandatory)]$PlanHost, [Parameter(Mandatory)][string]$PlanId, [string]$Source)
    $lines = [Collections.Generic.List[string]]::new()
    $lines.Add('#Requires -Version 7.0')
    $lines.Add('#Requires -RunAsAdministrator')
    $lines.Add("# Host $($PlanHost.host), plan $PlanId, from $Source. Generated by Invoke-PcaiLinkHealth; review before running.")
    $lines.Add('$ErrorActionPreference = ''Stop''')
    foreach ($g in @($PlanHost.guard)) { $lines.Add([string]$g) }
    foreach ($q in @($PlanHost.preflight)) { $lines.Add("$q | Format-Table -AutoSize | Out-String -Width 200") }
    foreach ($a in @($PlanHost.actions)) {
        if ($a.apply) { $lines.Add("# $($a.kind) ifIndex $($a.interface_index) $($a.target)"); $lines.Add([string]$a.apply) }
    }
    return ($lines -join [Environment]::NewLine) + [Environment]::NewLine
}

function Get-PcaiLinkHealthSeverity {
    [CmdletBinding()]
    param([Parameter(Mandatory)][string]$State, [AllowEmptyCollection()][object[]]$Reachability = @())
    $unreachable = @($Reachability | Where-Object { -not $_.Ping }).Count
    switch ($State) {
        { $_ -in @('InSync', 'Absent', 'NoPlan', 'NotInPlan', 'Healthy') } { if ($unreachable) { return 'warn' } return 'ok' }
        default { return 'warn' }
    }
}

function Get-PcaiLinkHealthFingerprint {
    <# Hash of the facts that matter for a notification; timestamps and ages are excluded. #>
    [CmdletBinding()]
    param([Parameter(Mandatory)]$Status)
    $facts = [ordered]@{
        state    = $Status.state
        severity = $Status.severity
        adapters = @($Status.adapters | Sort-Object Guid, Mac | ForEach-Object { "$($_.Guid)|$($_.Mac)|$($_.Status)|$($_.Ipv4 -join ',')|$($_.Forwarding)" })
        findings = @($Status.comparison.Findings | ForEach-Object { "$($_.Kind)|$($_.InterfaceIndex)|$($_.Target)|$($_.Status)" })
        reach    = @($Status.reachability | ForEach-Object { "$($_.Address)|$($_.Ping)" })
        services = @($Status.services | ForEach-Object { "$($_.Name)|$($_.State)" })
        planId   = $Status.plan.planId
    }
    $bytes = [Text.Encoding]::UTF8.GetBytes(($facts | ConvertTo-Json -Depth 5 -Compress))
    return [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($bytes)).ToLowerInvariant()
}

function Format-PcaiLinkHealthSummary {
    [CmdletBinding()]
    param([Parameter(Mandatory)]$Status)
    $up = @($Status.adapters | Where-Object Status -eq 'Up').Count
    $reach = @($Status.reachability)
    $ok = @($reach | Where-Object Ping).Count
    $missing = @($Status.comparison.Findings | Where-Object { $_.Status -in @('Missing', 'InstanceReplaced', 'AdapterAbsent', 'IndexChanged', 'LinkDown') } |
            ForEach-Object { "$($_.Kind) if$($_.InterfaceIndex) $($_.Target) $($_.Status)" })
    $text = "LINK-HEALTH $($Status.host) [$($Status.trigger)] state=$($Status.state) severity=$($Status.severity) usb4 up=$up/$(@($Status.adapters).Count) peers=$ok/$($reach.Count)"
    if ($Status.plan.planId) { $text += " plan=$($Status.plan.planId)" }
    if ($missing) { $text += '; ' + ($missing -join '; ') }
    if ($Status.autoApply.Applied) { $text += "; auto-apply: $($Status.autoApply.Outcome)" }
    foreach ($g in @($Status.guidance)) { $text += "; $g" }
    return $text
}

function Write-PcaiAtomicFile {
    [CmdletBinding()]
    param([Parameter(Mandatory)][string]$Path, [Parameter(Mandatory)][string]$Content)
    $tmp = "$Path.tmp-$PID"
    [IO.File]::WriteAllText($tmp, $Content, [Text.UTF8Encoding]::new($false))
    Move-Item -LiteralPath $tmp -Destination $Path -Force
}

function Send-PcaiBusStatus {
    [CmdletBinding()]
    param([Parameter(Mandatory)][string]$Body, [Parameter(Mandatory)][string]$HostName, [string]$Thread = 'machine-hygiene')
    # Scheduled tasks and systemd user units can start with a shorter PATH than a login shell.
    $exe = if ($IsWindows) { 'agent-bus.exe' } else { 'agent-bus' }
    $candidates = @(
        (Get-Command agent-bus -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1 | ForEach-Object Source)
        (Join-Path $HOME "bin/$exe")
        (Join-Path $HOME ".local/bin/$exe")
    ) | Where-Object { $_ -and (Test-Path -LiteralPath $_ -PathType Leaf) }
    $bus = @($candidates) | Select-Object -First 1
    if (-not $bus) { return 'agent-bus not found' }
    $r = Invoke-PcaiBoundedCommand -FilePath $bus -TimeoutSeconds 30 -ArgumentList @(
        'send', '--from-agent', "pcai-link-health-$HostName", '--to-agent', 'all', '--topic', 'status',
        '--thread-id', $Thread, '--tag', 'repo:PC-AI', '--tag', "host:$HostName", '--body', $Body, '--encoding', 'compact')
    if ($r.ExitCode -eq 0) { return 'posted' }
    return "agent-bus exit $($r.ExitCode): $(($r.Stderr + $r.Stdout).Trim() -replace '\s+', ' ')"
}

function Enter-PcaiRunLock {
    [CmdletBinding()]
    param([Parameter(Mandatory)][string]$Root)
    try {
        return [IO.File]::Open((Join-Path $Root 'run.lock'), 'OpenOrCreate', 'ReadWrite', 'None')
    } catch [IO.IOException] { return $null }
}

function Invoke-PcaiLinkHealth {
    [CmdletBinding()]
    param(
        [string]$RunMode = 'Observe', [string]$RunTrigger = 'Manual', [string]$Root, [string]$Config,
        [string]$Plan, [int]$MinInterval = 20, [switch]$SkipBus, [switch]$Notify, [switch]$IsDryRun
    )
    if (-not $Root) { $Root = Get-PcaiLinkHealthDefaultRoot -RunMode $RunMode }
    if (-not $Config) { $Config = Join-Path $Root 'config.json' }
    $rootTrusted = $false
    if ($RunMode -eq 'Apply' -and $IsWindows -and -not $IsDryRun) {
        # The SYSTEM task writes and runs apply scripts under this root, so it must not inherit
        # %ProgramData%'s "Users may create" right. Create it protected; never adopt a root that
        # another account created or can change, and only read a config file it cannot change.
        if (-not (Test-Path -LiteralPath $Root) -and (Test-PcaiElevated)) { New-PcaiProtectedStatusRoot -Path $Root }
        $rootTrusted = Test-PcaiAdminOwnedPath -Path $Root
        if ((Test-Path -LiteralPath $Config) -and -not ($rootTrusted -and (Test-PcaiAdminOwnedPath -Path $Config))) { $Config = $null }
    }
    $cfg = Read-PcaiLinkHealthConfig -Path $Config -Root $Root
    if ($Plan) { $cfg.planPath = $Plan }
    $hostName = Get-PcaiHostName
    $lock = $null
    if (-not $IsDryRun) {
        $null = New-Item -ItemType Directory -Force -Path $Root
        $lock = Enter-PcaiRunLock -Root $Root
        if (-not $lock) { Write-Output "LINK-HEALTH $hostName skipped: another run holds $Root/run.lock"; return $null }
        $lastRunPath = Join-Path $Root 'last-run.txt'
        if ($RunTrigger -ne 'Manual' -and $MinInterval -gt 0 -and (Test-Path -LiteralPath $lastRunPath)) {
            $last = [DateTime]::Parse((Get-Content -LiteralPath $lastRunPath -Raw).Trim(), [Globalization.CultureInfo]::InvariantCulture, 'RoundtripKind')
            if (([DateTime]::UtcNow - $last).TotalSeconds -lt $MinInterval) {
                $lock.Dispose(); Write-Output "LINK-HEALTH $hostName debounced ($RunTrigger within $MinInterval s)"; return $null
            }
        }
        Set-Content -LiteralPath $lastRunPath -Value ([DateTime]::UtcNow.ToString('o')) -NoNewline
    }
    try {
        $policy = Get-PcaiAutoApplyPolicy
        if ($RunMode -eq 'Apply' -and $policy.Enabled -and -not $Plan) { $cfg.planPath = $policy.PlanPath }
        $planObj = $null; $planError = $null
        try { $planObj = Read-PcaiRoutePlan -Path $cfg.planPath } catch { $planError = $_.Exception.Message }
        $adapters = Get-PcaiUsb4Adapter
        $routes = Get-PcaiIpv4HostRoute -InterfaceIndex @($adapters | ForEach-Object InterfaceIndex)
        $comparison = Compare-PcaiLinkPlan -Plan $planObj -HostName $hostName -Adapter $adapters -Route $routes
        if ($comparison.State -eq 'NoPlan' -and @($adapters | Where-Object Status -eq 'Up').Count) { $comparison.State = 'Healthy' }
        $reach = Test-PcaiPeerReachability -Target (Get-PcaiReachabilityTarget -Plan $planObj -HostName $hostName) -TcpPort ([int]$cfg.tcpProbePort)
        $services = Get-PcaiUsb4Service
        $decisions = Get-PcaiServiceRecoveryPlan -Service $services -AllowList @($cfg.restartAllowList)
        $recovery = if ($IsDryRun) { @() } else { Invoke-PcaiServiceRecovery -Decision $decisions }
        $planId = if ($planObj) { Get-PcaiRoutePlanId -Plan $planObj } else { $null }
        $guidance = [Collections.Generic.List[string]]::new()
        if ($planError) { $guidance.Add("Plan unreadable: $planError") }

        $auto = [ordered]@{ Enabled = $policy.Enabled; Reason = $policy.Reason; Applied = $false; Outcome = $null }
        $decision = Get-PcaiAutoApplyDecision -RunMode $RunMode -Elevated (Test-PcaiElevated) -Policy $policy -Comparison $comparison
        if ($decision.Apply -and -not $rootTrusted) { $decision = [pscustomobject]@{ Apply = $false; Reason = "Status root $Root is not admin-owned and protected; not applying" } }
        $auto.Decision = $decision.Reason
        if ($comparison.State -eq 'Drift' -and $comparison.PlanHost -and $planId) {
            $body = New-PcaiHostApplyScript -PlanHost $comparison.PlanHost -PlanId $planId -Source $cfg.planPath
            if ($decision.Apply -and -not $IsDryRun) {
                $applyDir = Join-Path $Root 'apply'
                if (-not (Test-Path -LiteralPath $applyDir)) { $null = New-Item -ItemType Directory -Path $applyDir }
                if (-not (Test-PcaiAdminOwnedPath -Path $applyDir)) { throw "Apply directory $applyDir is not admin-owned and protected; review and remove it" }
                $applyFile = Join-Path $applyDir ("{0}-{1}.ps1" -f $planId, [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ'))
                Write-PcaiAtomicFile -Path $applyFile -Content $body
                $pwsh = (Get-Process -Id $PID).Path
                $r = Invoke-PcaiBoundedCommand -FilePath $pwsh -ArgumentList @('-NoLogo', '-NoProfile', '-NonInteractive', '-File', $applyFile) -TimeoutSeconds 120
                Set-Content -LiteralPath "$applyFile.log" -Value ("exit=$($r.ExitCode)`n$($r.Stdout)`n$($r.Stderr)")
                $auto.Applied = $true
                $auto.Outcome = if ($r.ExitCode -eq 0) { 'applied' } else { "failed exit $($r.ExitCode); see $applyFile.log" }
                $adapters = Get-PcaiUsb4Adapter
                $routes = Get-PcaiIpv4HostRoute -InterfaceIndex @($adapters | ForEach-Object InterfaceIndex)
                $comparison = Compare-PcaiLinkPlan -Plan $planObj -HostName $hostName -Adapter $adapters -Route $routes
                $reach = Test-PcaiPeerReachability -Target (Get-PcaiReachabilityTarget -Plan $planObj -HostName $hostName) -TcpPort ([int]$cfg.tcpProbePort)
            } elseif (-not $IsDryRun) {
                $review = Join-Path $Root "pending-apply-$planId.ps1"
                Write-PcaiAtomicFile -Path $review -Content $body
                $guidance.Add("Guards match; review then run elevated: pwsh -NoProfile -File `"$review`"")
            } else {
                $guidance.Add('Guards match; a run without -DryRun writes the elevated apply script for review')
            }
        } elseif ($comparison.State -eq 'InstanceChanged') {
            $guidance.Add("Adapter replaced or moved; not applying. $($cfg.regenerateHint)")
        } elseif ($comparison.State -eq 'LinkDown') {
            $guidance.Add('A planned USB4 link is down: check the cable and the peer; nothing to apply')
        }

        $status = [ordered]@{
            schema       = $script:PcaiLinkHealthSchema
            host         = $hostName
            evaluatedAtUtc = [DateTime]::UtcNow.ToString('o')
            trigger      = $RunTrigger
            mode         = $RunMode
            dryRun       = [bool]$IsDryRun
            state        = $comparison.State
            severity     = Get-PcaiLinkHealthSeverity -State $comparison.State -Reachability $reach
            adapters     = $adapters
            plan         = [ordered]@{ path = $cfg.planPath; planId = $planId; expiresAtUtc = if ($planObj) { $planObj.expires_at_utc } else { $null } }
            comparison   = [ordered]@{ GuardsMatch = $comparison.GuardsMatch; Findings = $comparison.Findings }
            reachability = $reach
            services     = $decisions
            recovery     = $recovery
            observer     = Get-PcaiObserverReceipt
            autoApply    = $auto
            guidance     = $guidance.ToArray()
        }
        $status.fingerprint = Get-PcaiLinkHealthFingerprint -Status ([pscustomobject]$status)
        $summary = Format-PcaiLinkHealthSummary -Status ([pscustomobject]$status)
        $status.summary = $summary
        $status.bus = 'not posted'
        if (-not $IsDryRun) {
            $fpPath = Join-Path $Root 'last-fingerprint.txt'
            $previous = if (Test-Path -LiteralPath $fpPath) { (Get-Content -LiteralPath $fpPath -Raw).Trim() } else { '' }
            $changed = $previous -ne $status.fingerprint
            if (($changed -or $Notify) -and -not $SkipBus -and $cfg.busEnabled) {
                $status.bus = Send-PcaiBusStatus -Body $summary -HostName $hostName -Thread $cfg.busThread
            } elseif (-not $changed) { $status.bus = 'unchanged' }
            $json = [pscustomobject]$status | ConvertTo-Json -Depth 8
            Write-PcaiAtomicFile -Path (Join-Path $Root 'status.json') -Content $json
            if ($status.bus -eq 'posted' -or $status.bus -eq 'unchanged' -or $SkipBus -or -not $cfg.busEnabled) {
                Set-Content -LiteralPath $fpPath -Value $status.fingerprint -NoNewline
            }
            $history = Join-Path $Root 'history.jsonl'
            Add-Content -LiteralPath $history -Value ([pscustomobject]@{ t = $status.evaluatedAtUtc; trigger = $RunTrigger; state = $status.state; severity = $status.severity; fp = $status.fingerprint; bus = $status.bus } | ConvertTo-Json -Compress)
            $lines = @(Get-Content -LiteralPath $history)
            if ($lines.Count -gt 1000) { Set-Content -LiteralPath $history -Value $lines[-500..-1] }
        }
        Write-Output $summary
        return [pscustomobject]$status
    } finally {
        if ($lock) { $lock.Dispose() }
    }
}

if ($MyInvocation.InvocationName -ne '.') {
    if ($Help) { Get-Help $PSCommandPath -Detailed; return }
    $result = Invoke-PcaiLinkHealth -RunMode $Mode -RunTrigger $Trigger -Root $StatusRoot -Config $ConfigPath -Plan $PlanPath `
        -MinInterval $MinIntervalSeconds -SkipBus:$NoBus -Notify:$ForceNotify -IsDryRun:$DryRun
    $statusObject = @($result | Where-Object { $_ -isnot [string] }) | Select-Object -Last 1
    @($result | Where-Object { $_ -is [string] }) | ForEach-Object { Write-Host $_ }
    if ($PassThru -and $statusObject) { $statusObject }
    if (-not $statusObject) { exit 0 }
    exit $(if ($statusObject.severity -eq 'ok') { 0 } else { 2 })
}
