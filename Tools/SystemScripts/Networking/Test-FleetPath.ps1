#Requires -Version 7.0
<#
.SYNOPSIS
Read-only fleet path diagnostic: which address reaches each host and service from here.

.DESCRIPTION
Answers one question per host and service: which candidate address should I use
right now? For every target and service it tries the candidate addresses in order
(LAN addresses, the tailnet address, then the MagicDNS name) with a hard
per-probe deadline, and reports for each candidate:

  * whether a TCP connect succeeded, and how long it took;
  * the egress interface Windows would use (Find-NetRoute);
  * the Tailscale path to that peer (direct, relay "<region>", peer-relay,
    offline), read from `tailscale status --json` after the probes have run so a
    sleeping path has been woken first.

It also decides whether this machine is on the home LAN. A hotel or cafe network
can use 192.168.1.0/24 too, so the subnet alone proves nothing. Home requires an
inventory gateway that answers a TCP connect AND whose on-link MAC address matches
the value recorded in the inventory. A missing, incomplete or all-zero MAC fails
closed to off-LAN. While off the home LAN, LAN candidates are neither probed nor
recommended: a connect to a LAN address on a hotel's 192.168.1.0/24 would reach a stranger's machine.
Home is verified per interface: each LAN candidate's egress interface is resolved with
Find-NetRoute before it is probed, and it is probed only if that interface is one that
verified a gateway MAC. On a host with Ethernet on the home LAN and Wi-Fi on a network
that reuses the subnet, a LAN address routed out of Wi-Fi is skipped with the reason
egress-not-verified, as is one whose route cannot be resolved.

The tool changes nothing. It calls Get-NetNeighbor, Find-NetRoute and
`tailscale status --json`, and it opens TCP connections that it closes at once. It
never edits routes, DNS, Tailscale, SMB mappings or credentials. Targets see a
connection that sends no data and closes.

Recommendation rule: LAN candidates (only when on the home LAN) first, in listed
order; then tailnet candidates ranked direct, peer-relay, relay; the MagicDNS name
ranks just behind the tailnet address of the same path state. The first reachable
candidate in that ranking is recommended.

Probes run in parallel (ForEach-Object -Parallel) in two waves. Wave A: the gateway
probes that decide the location, plus every tailnet and MagicDNS candidate. Wave B:
the LAN candidates, only if wave A proved home. The first Get-NetNeighbor and
Find-NetRoute calls cost about 1.2 s of CIM start-up, so they run while wave A is in
flight. With the default 1500 ms budget the whole run takes 3 to 4 s on the home LAN
(one blocked LAN candidate costs one budget) and about the same when away. If no
gateway answers, or the MAC cannot be verified, the gateway probes are retried once
before the verdict is off-LAN; a wrong MAC is a final answer and is not retried.

The inventory is private and lives outside the repository: $env:PCAI_FLEET_PATHS if set,
else ~/.machine/fleet-paths.json (override with
-InventoryPath). Tests: Tests/Unit/FleetPath.Tests.ps1.

.PARAMETER Target
Inventory host names to test. Default: every host in the inventory.

.PARAMETER Service
Limit the test to these services (ssh, smb, rdp, https, agentbus). Default: every
service each target defines.

.PARAMETER BudgetMs
Hard deadline per probe in milliseconds, 100 to 30000. Default 1500.

.PARAMETER InventoryPath
Path of the inventory JSON. Default: $env:PCAI_FLEET_PATHS if set, else
~/.machine/fleet-paths.json. The file holds real addresses and MACs and must not be
committed. Copy Tools/SystemScripts/Networking/fleet-paths.example.json (reserved
documentation values only) and replace them with your own.

.PARAMETER AsJson
Emit one JSON document (schema pcai.fleet-path/1) instead of an object, suitable for
~/.machine/logs. The tailnet path may include a peer's public endpoint address.

.EXAMPLE
# Run in a PowerShell 7 session so the result is objects, not text:
& .\Tools\SystemScripts\Networking\Test-FleetPath.ps1 | Select-Object -ExpandProperty Summary | Format-Table

.EXAMPLE
# -AsJson emits text, so a child process is fine here:
pwsh -NoProfile -File .\Tools\SystemScripts\Networking\Test-FleetPath.ps1 -Target nas -Service smb -AsJson

.EXAMPLE
pwsh -NoProfile -File .\Tools\SystemScripts\Networking\Test-FleetPath.ps1 -AsJson |
    Set-Content -Encoding utf8NoBOM (Join-Path $HOME ('.machine/logs/fleet-path-{0:yyyyMMdd-HHmmss}.json' -f (Get-Date)))
#>
[CmdletBinding(PositionalBinding = $false)]
param(
    [string[]]$Target,
    [ValidateSet('ssh', 'smb', 'rdp', 'https', 'agentbus')][string[]]$Service,
    [ValidateRange(100, 30000)][int]$BudgetMs = 1500,
    [string]$InventoryPath = $(if ($env:PCAI_FLEET_PATHS) { $env:PCAI_FLEET_PATHS } else { Join-Path $HOME '.machine\fleet-paths.json' }),
    [switch]$AsJson
)

Set-StrictMode -Version Latest

$script:FleetSchema = 'pcai.fleet-path/1'
$script:FleetInventorySchema = 'pcai.fleet-paths/1'
$script:FleetServices = @('ssh', 'smb', 'rdp', 'https', 'agentbus')
$script:FleetKinds = @('lan', 'tailnet', 'magicdns')

function Get-FleetProperty {
    # Optional JSON members and optional tailscale fields are absent, not null, in
    # some captures; StrictMode turns a bare member access into an error.
    [CmdletBinding()]
    param($InputObject, [Parameter(Mandatory)][string]$Name, $Default = $null)
    if ($null -eq $InputObject) { return $Default }
    $property = $InputObject.PSObject.Properties[$Name]
    if ($property) { return $property.Value }
    return $Default
}

function ConvertTo-FleetMac {
    # Returns AA-BB-CC-DD-EE-FF, or $null for anything that is not a usable MAC. An
    # all-zero MAC is what an unresolved neighbour entry reports.
    [CmdletBinding()]
    [OutputType([string])]
    param([AllowNull()][AllowEmptyString()][string]$Value)
    if ([string]::IsNullOrWhiteSpace($Value)) { return $null }
    $clean = $Value.Trim() -replace '[-:.]', ''
    if ($clean -notmatch '^[0-9A-Fa-f]{12}$') { return $null }
    $clean = $clean.ToUpperInvariant()
    if ($clean -eq '000000000000') { return $null }
    return ([regex]::Matches($clean, '..').Value -join '-')
}

function Import-FleetInventory {
    [CmdletBinding()]
    param([Parameter(Mandatory)][string]$Path)
    # .NET file calls, not Test-Path/Get-Content: the first provider call in a fresh
    # process costs about 250 ms, and this tool has a runtime budget.
    $full = [IO.Path]::GetFullPath($Path, $PWD.ProviderPath)
    if (-not [IO.File]::Exists($full)) {
        throw "Fleet inventory not found: $Path. The real inventory is private and kept outside the repository: copy Tools/SystemScripts/Networking/fleet-paths.example.json to ~/.machine/fleet-paths.json (or set PCAI_FLEET_PATHS, or pass -InventoryPath) and replace the example values with your own."
    }
    try { $raw = [IO.File]::ReadAllText($full) | ConvertFrom-Json -ErrorAction Stop }
    catch { throw "Fleet inventory is not valid JSON ($Path): $($_.Exception.Message)" }
    if ((Get-FleetProperty $raw 'schema') -ne $script:FleetInventorySchema) {
        throw "Fleet inventory schema must be '$script:FleetInventorySchema' ($Path)."
    }
    $testIpv4 = { param($Text) $parsed = $null; [ipaddress]::TryParse([string]$Text, [ref]$parsed) -and $parsed.AddressFamily -eq 'InterNetwork' }
    $testPort = { param($Value) ($Value -is [int] -or $Value -is [long]) -and $Value -ge 1 -and $Value -le 65535 }

    $gateways = foreach ($g in @(Get-FleetProperty $raw 'gateways' @())) {
        $name = Get-FleetProperty $g 'name'
        $ip = Get-FleetProperty $g 'ip'
        if (-not $name -or -not (& $testIpv4 $ip)) { throw "Gateway entries need a name and an IPv4 ip ($Path)." }
        $macText = Get-FleetProperty $g 'mac'
        $mac = ConvertTo-FleetMac $macText
        if ($macText -and -not $mac) { throw "Gateway '$name' has an unusable MAC '$macText' ($Path)." }
        $ports = @(Get-FleetProperty $g 'probePorts' @(443))
        foreach ($p in $ports) { if (-not (& $testPort $p)) { throw "Gateway '$name' has an invalid probe port '$p' ($Path)." } }
        [pscustomobject]@{ Name = [string]$name; Ip = [string]$ip; Mac = $mac; ProbePorts = [int[]]$ports }
    }

    $hostsNode = Get-FleetProperty $raw 'hosts'
    if ($null -eq $hostsNode) { throw "Fleet inventory has no hosts ($Path)." }
    $hostList = foreach ($entry in $hostsNode.PSObject.Properties) {
        $name = $entry.Name
        $h = $entry.Value
        $tailnet = Get-FleetProperty $h 'tailnet'
        if ($tailnet -and -not (& $testIpv4 $tailnet)) { throw "Host '$name' has an invalid tailnet address '$tailnet' ($Path)." }
        $lan = @(Get-FleetProperty $h 'lan' @())
        foreach ($a in $lan) { if (-not (& $testIpv4 $a)) { throw "Host '$name' has an invalid LAN address '$a' ($Path)." } }
        $services = foreach ($svc in (Get-FleetProperty $h 'services').PSObject.Properties) {
            if ($svc.Name -notin $script:FleetServices) { throw "Host '$name' has unknown service '$($svc.Name)' ($Path)." }
            $ports = @(Get-FleetProperty $svc.Value 'ports' @())
            if ($ports.Count -eq 0) { throw "Host '$name' service '$($svc.Name)' lists no ports ($Path)." }
            foreach ($p in $ports) { if (-not (& $testPort $p)) { throw "Host '$name' service '$($svc.Name)' has an invalid port '$p' ($Path)." } }
            $via = @(Get-FleetProperty $svc.Value 'via' $script:FleetKinds)
            foreach ($v in $via) { if ($v -notin $script:FleetKinds) { throw "Host '$name' service '$($svc.Name)' has unknown via '$v' ($Path)." } }
            [pscustomobject]@{ Service = $svc.Name; Ports = [int[]]$ports; Via = [string[]]($via | ForEach-Object { $_.ToLowerInvariant() }) }
        }
        [pscustomobject]@{
            Name = $name; Tailnet = $tailnet; MagicDns = Get-FleetProperty $h 'magicDns'
            Lan = [string[]]$lan; Services = @($services)
        }
    }
    [pscustomobject]@{
        Schema = $script:FleetInventorySchema; Path = $Path
        Measured = Get-FleetProperty $raw 'measured'; MeasuredFrom = Get-FleetProperty $raw 'measuredFrom'
        Gateways = @($gateways); Hosts = @($hostList)
    }
}

function New-FleetCandidatePlan {
    # Candidate order is the contract: LAN addresses as listed, then the tailnet
    # address, then the MagicDNS name. Order is recorded so ranking can break ties.
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]$FleetHost,
        [Parameter(Mandatory)][string[]]$Via,
        [Parameter(Mandatory)][int]$Port
    )
    $plan = [Collections.Generic.List[object]]::new()
    $add = {
        param($Kind, $Address)
        $plan.Add([pscustomobject]@{ Kind = $Kind; Address = [string]$Address; Port = $Port; Order = $plan.Count })
    }
    if ('lan' -in $Via) { foreach ($a in $FleetHost.Lan) { & $add 'lan' $a } }
    if ('tailnet' -in $Via -and $FleetHost.Tailnet) { & $add 'tailnet' $FleetHost.Tailnet }
    if ('magicdns' -in $Via -and $FleetHost.MagicDns) { & $add 'magicdns' $FleetHost.MagicDns }
    return $plan.ToArray()
}

# The TCP probe is a script block, not a function, so the parallel runspaces can
# rebuild it from text. ForEach-Object -Parallel cannot see script functions or
# Pester mocks, and refuses a script block passed through $using:. The connector
# contract: a block taking (Address, Port) that returns an object with a Task (the
# connect, which the caller waits on with the budget) and a Client (disposed
# afterwards, or $null). The deadline lives here, outside the connector, so a
# connector that never completes cannot defeat it.
$script:FleetTcpProbe = {
    param([string]$Address, [int]$Port, [int]$BudgetMs, [string]$ConnectorText)
    $clock = [Diagnostics.Stopwatch]::StartNew()
    $connection = $null
    $result = [ordered]@{
        Address = $Address; Port = $Port; Reachable = $false
        LatencyMs = $null; Error = $null; RemoteAddress = $null
    }
    try {
        $connector = if ($ConnectorText) { [scriptblock]::Create($ConnectorText) }
        else {
            {
                param($a, $p)
                $client = [Net.Sockets.TcpClient]::new()
                [pscustomobject]@{ Task = $client.ConnectAsync($a, $p); Client = $client }
            }
        }
        $connection = & $connector $Address $Port
        if (-not $connection.Task.Wait($BudgetMs)) {
            $result.Error = "timeout after ${BudgetMs}ms"
        }
        elseif ($connection.Client -and -not $connection.Client.Connected) {
            $result.Error = 'not connected'
        }
        else {
            $result.Reachable = $true
            $result.LatencyMs = [math]::Round($clock.Elapsed.TotalMilliseconds, 1)
            if ($connection.Client) {
                try {
                    # A name resolved to IPv4 over a dual-mode socket reads ::ffff:a.b.c.d.
                    $remote = $connection.Client.Client.RemoteEndPoint.Address
                    if ($remote.IsIPv4MappedToIPv6) { $remote = $remote.MapToIPv4() }
                    $result.RemoteAddress = $remote.ToString()
                }
                catch { }
            }
        }
    }
    catch {
        # A refused or unreachable connect throws from Task.Wait as an
        # AggregateException wrapped in a MethodInvocationException.
        $root = $_.Exception.GetBaseException()
        $result.Error = if ($root -is [Net.Sockets.SocketException]) { [string]$root.SocketErrorCode }
        else { "$($root.GetType().Name): $($root.Message)" }
    }
    finally {
        if ($connection -and $connection.Client) { try { $connection.Client.Dispose() } catch { } }
    }
    [pscustomobject]$result
}

function Test-FleetTcpPort {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$Address,
        [Parameter(Mandatory)][int]$Port,
        [int]$BudgetMs = 1500,
        [scriptblock]$Connector
    )
    $connectorText = if ($Connector) { $Connector.ToString() } else { '' }
    & $script:FleetTcpProbe $Address $Port $BudgetMs $connectorText
}

function Invoke-FleetProbeBatch {
    # Every probe gets its own runspace slot: with the throttle at the probe count,
    # a wave of hung probes costs one budget, not one budget per wave.
    [CmdletBinding()]
    param(
        [AllowEmptyCollection()][object[]]$Probes = @(),
        [int]$BudgetMs = 1500,
        [scriptblock]$Connector,
        # Runs in this runspace while the probes are in flight. The first
        # Get-NetNeighbor / Find-NetRoute call costs up to 1.7 s of CIM start-up, so
        # callers use the wait to pay it instead of adding it to the total.
        [scriptblock]$WhileWaiting
    )
    $items = @($Probes)
    if ($items.Count -eq 0) {
        if ($WhileWaiting) { $null = & $WhileWaiting }
        return @()
    }
    $probeText = $script:FleetTcpProbe.ToString()
    $connectorText = if ($Connector) { $Connector.ToString() } else { '' }
    $throttle = [Math]::Min($items.Count, 64)
    $job = $items | ForEach-Object -ThrottleLimit $throttle -AsJob -Parallel {
        $probe = [scriptblock]::Create($using:probeText)
        & $probe $_.Address $_.Port $using:BudgetMs $using:connectorText
    }
    try {
        if ($WhileWaiting) { $null = & $WhileWaiting }
    }
    finally {
        # Receive-Job -Wait -AutoRemoveJob raced on a -Parallel job and dropped its
        # output once in about ten runs, which read as "no gateway answered". Wait,
        # receive and remove as separate steps.
        $null = $job | Wait-Job
        $results = @($job | Receive-Job)
        $job | Remove-Job -Force
    }
    if ($results.Count -ne $items.Count) {
        throw "Probe batch returned $($results.Count) results for $($items.Count) probes; refusing to report from partial data."
    }
    $results
}

function Get-FleetNeighbor {
    # Wrapper so tests need no live ARP table. Failure reads as "no rows", which the
    # caller treats as unverifiable, never as a match.
    [CmdletBinding()]
    param([Parameter(Mandatory)][string[]]$IPAddress)
    try {
        # One table read for all addresses: each Get-NetNeighbor call costs about 200 ms.
        Get-NetNeighbor -AddressFamily IPv4 -ErrorAction Stop | Where-Object { $_.IPAddress -in $IPAddress } | ForEach-Object {
            [pscustomobject]@{
                IPAddress = [string]$_.IPAddress; LinkLayerAddress = [string]$_.LinkLayerAddress
                State = [string]$_.State; InterfaceAlias = [string]$_.InterfaceAlias
                InterfaceIndex = $_.InterfaceIndex
            }
        }
    }
    catch { @() }
}

function Find-FleetRoute {
    [CmdletBinding()]
    param([Parameter(Mandatory)][string]$Address)
    try {
        $found = @(Find-NetRoute -RemoteIPAddress $Address -ErrorAction Stop)
        $source = $found | Where-Object { $_.PSObject.Properties['IPAddress'] } | Select-Object -First 1
        $route = $found | Where-Object { $_.PSObject.Properties['NextHop'] } | Select-Object -First 1
        $alias = if ($route) { $route.InterfaceAlias } elseif ($source) { $source.InterfaceAlias }
        $index = if ($route) { $route.InterfaceIndex } elseif ($source) { $source.InterfaceIndex }
        if (-not $alias) { return $null }
        $nextHop = if ($route) { [string]$route.NextHop } else { '' }
        [pscustomobject]@{
            Interface = [string]$alias
            InterfaceIndex = if ($null -ne $index) { [int]$index } else { $null }
            SourceAddress = if ($source) { [string]$source.IPAddress } else { $null }
            NextHop = if ($nextHop -in '', '0.0.0.0', '::') { 'on-link' } else { $nextHop }
        }
    }
    catch { $null }
}

function Resolve-FleetLocation {
    # Home needs two independent facts per gateway: a TCP connect succeeded (so the
    # ARP row is fresh, not left over from another network) AND the on-link MAC
    # equals the inventory MAC. Every other outcome is off-LAN.
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][AllowEmptyCollection()][object[]]$Gateways,
        [Parameter(Mandatory)][AllowEmptyCollection()][object[]]$ProbeResults
    )
    $byKey = @{}
    foreach ($r in $ProbeResults) { $byKey["$($r.Address):$($r.Port)"] = $r }

    # Read the neighbour table after the probes, so a reached gateway has a fresh row.
    $neighbors = @(if (@($Gateways).Count -gt 0) { Get-FleetNeighbor -IPAddress @($Gateways | ForEach-Object { $_.Ip }) })

    $entries = foreach ($g in $Gateways) {
        $probes = @($g.ProbePorts | ForEach-Object { $byKey["$($g.Ip):$_"] } | Where-Object { $_ })
        $reached = @($probes | Where-Object { $_.Reachable })
        $reachable = $reached.Count -gt 0
        $rows = @($neighbors | Where-Object {
                $_.IPAddress -eq $g.Ip -and $_.State -notin 'Unreachable', 'Incomplete' -and (ConvertTo-FleetMac $_.LinkLayerAddress)
            })
        $observed = @($rows | ForEach-Object { ConvertTo-FleetMac $_.LinkLayerAddress } | Sort-Object -Unique)
        # Verification is per interface: a row proves "home" only for the interface it was
        # read on. A row without an interface index cannot be matched to an egress, so it
        # verifies nothing.
        $verified = @($rows | Where-Object { $g.Mac -and (ConvertTo-FleetMac $_.LinkLayerAddress) -eq $g.Mac -and $null -ne $_.InterfaceIndex } |
                ForEach-Object { [int]$_.InterfaceIndex } | Sort-Object -Unique)
        $colliding = @($rows | Where-Object { $g.Mac -and (ConvertTo-FleetMac $_.LinkLayerAddress) -ne $g.Mac -and $null -ne $_.InterfaceIndex } |
                ForEach-Object { [int]$_.InterfaceIndex } | Sort-Object -Unique)
        $mismatching = @($observed | Where-Object { $_ -ne $g.Mac })
        $match = $reachable -and $verified.Count -gt 0
        $collision = $reachable -and [bool]$g.Mac -and $mismatching.Count -gt 0
        [pscustomobject]@{
            Name = $g.Name; Ip = $g.Ip; ExpectedMac = $g.Mac
            ObservedMac = ($observed -join ',')
            Reachable = $reachable
            LatencyMs = if ($reachable) { ($reached | Measure-Object -Property LatencyMs -Minimum).Minimum } else { $null }
            MacMatch = $match; Collision = $collision
            VerifiedInterfaces = $(if ($reachable) { $verified } else { @() })
            CollidingInterfaces = $(if ($collision) { $colliding } else { @() })
            Interface = (@($rows | ForEach-Object { $_.InterfaceAlias } | Sort-Object -Unique) -join ',')
        }
    }
    $entries = @($entries)
    # An interface that disagrees with the inventory for any gateway is not trusted, even
    # if another gateway row matched on it.
    $collidingInterfaces = @($entries | ForEach-Object { $_.CollidingInterfaces } | Where-Object { $null -ne $_ } | Sort-Object -Unique)
    $verifiedInterfaces = @($entries | ForEach-Object { $_.VerifiedInterfaces } | Where-Object { $null -ne $_ -and $_ -notin $collidingInterfaces } | Sort-Object -Unique)
    $homeGateways = @($entries | Where-Object { $_.MacMatch })
    $collided = @($entries | Where-Object { $_.Collision })
    $unverified = @($entries | Where-Object { $_.Reachable -and -not $_.MacMatch -and -not $_.Collision })

    if ($verifiedInterfaces.Count -gt 0 -and $homeGateways.Count -gt 0) {
        $state = 'home-lan'
        $reason = "Gateway $($homeGateways[0].Ip) answered and its MAC $($homeGateways[0].ObservedMac) matches the inventory on interface $($verifiedInterfaces -join ', ')."
        if ($collidingInterfaces.Count -gt 0) {
            $reason += " Interface $($collidingInterfaces -join ', ') holds a different MAC for a gateway address and is not trusted; LAN addresses routed through it are skipped."
        }
    }
    elseif ($collided.Count -gt 0) {
        $state = 'off-lan'
        $reason = "Gateway $($collided[0].Ip) answered but its MAC $($collided[0].ObservedMac) is not the inventory MAC $($collided[0].ExpectedMac): another network is using the home subnet."
    }
    elseif ($unverified.Count -gt 0) {
        $state = 'off-lan'
        $reason = "Gateway $($unverified[0].Ip) answered but no usable MAC could be read or none is recorded, so home cannot be verified."
    }
    else {
        $state = 'off-lan'
        $reason = 'No inventory gateway answered.'
    }
    [pscustomobject]@{
        State = $state; Reason = $reason; Collision = ($collided.Count -gt 0)
        VerifiedInterfaces = $(if ($state -eq 'home-lan') { $verifiedInterfaces } else { @() })
        CollidingInterfaces = $collidingInterfaces
        Gateways = $entries
    }
}

function Test-FleetLanEgress {
    # Pure. A LAN candidate is eligible only when the interface Windows would send it out
    # of is one that verified a home gateway. Unknown route or ifIndex: not eligible.
    [CmdletBinding()]
    [OutputType([bool])]
    param($Route, [AllowEmptyCollection()][object[]]$VerifiedInterfaces = @())
    if ($null -eq $Route) { return $false }
    $index = Get-FleetProperty $Route 'InterfaceIndex'
    if ($null -eq $index) { return $false }
    return (@($VerifiedInterfaces) -contains [int]$index)
}
function Invoke-FleetNative {
    # Runs one native process with a hard deadline and closed stdin, so a wedged
    # CLI cannot hang the diagnostic. Same approach as ~/.machine/Test-FleetAccess.ps1.
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$FilePath,
        [string[]]$Arguments = @(),
        [int]$TimeoutMs = 5000
    )
    $info = [Diagnostics.ProcessStartInfo]::new($FilePath)
    foreach ($a in $Arguments) { $info.ArgumentList.Add($a) }
    $info.RedirectStandardOutput = $true
    $info.RedirectStandardError = $true
    $info.RedirectStandardInput = $true
    $info.UseShellExecute = $false
    $info.CreateNoWindow = $true
    $process = $null
    try {
        $process = [Diagnostics.Process]::Start($info)
        $process.StandardInput.Close()
        $out = $process.StandardOutput.ReadToEndAsync()
        $err = $process.StandardError.ReadToEndAsync()
        if (-not $process.WaitForExit($TimeoutMs)) {
            try { $process.Kill($true) } catch { }
            return [pscustomobject]@{ ExitCode = -1; Stdout = ''; Stderr = ''; TimedOut = $true }
        }
        $process.WaitForExit()
        [pscustomobject]@{ ExitCode = $process.ExitCode; Stdout = $out.Result; Stderr = $err.Result; TimedOut = $false }
    }
    catch { [pscustomobject]@{ ExitCode = -1; Stdout = ''; Stderr = $_.Exception.Message; TimedOut = $false } }
    finally { if ($process) { $process.Dispose() } }
}

function Get-FleetTailscaleStatus {
    # Never throws: a missing, wedged or confused tailscale CLI is data, because the
    # probes still tell the truth about reachability without it.
    [CmdletBinding()]
    param([int]$TimeoutMs = 5000)
    $fail = { param($Message) [pscustomobject]@{ Available = $false; Error = $Message; Status = $null } }
    $command = Get-Command -Name 'tailscale' -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $command) { return (& $fail 'tailscale binary not found') }
    $run = Invoke-FleetNative -FilePath $command.Source -Arguments @('status', '--json') -TimeoutMs $TimeoutMs
    if ($run.TimedOut) { return (& $fail "tailscale status timed out after ${TimeoutMs}ms") }
    if ($run.ExitCode -ne 0) { return (& $fail "tailscale status exited $($run.ExitCode)") }
    if ([string]::IsNullOrWhiteSpace($run.Stdout)) { return (& $fail 'tailscale status returned empty output') }
    try { $parsed = $run.Stdout | ConvertFrom-Json -ErrorAction Stop }
    catch { return (& $fail 'tailscale status returned invalid JSON') }
    if ($null -eq $parsed -or $parsed -isnot [System.Management.Automation.PSCustomObject]) { return (& $fail 'tailscale status returned an unexpected document') }
    [pscustomobject]@{ Available = $true; Error = $null; Status = $parsed }
}

function Resolve-FleetTailscalePath {
    # Peers are matched by tailnet IP, never by name (a NAS appliance's peer name
    # differs from its inventory key). `Relay` is the home DERP region and is set even on direct paths,
    # so it only means "relayed" when no direct or peer-relay endpoint exists.
    [CmdletBinding()]
    param($Status, [Parameter(Mandatory)][string]$TailnetAddress)
    $path = {
        param($State, $Display, $Region = $null, $Endpoint = $null, $Active = $null)
        [pscustomobject]@{ State = $State; Display = $Display; Region = $Region; Endpoint = $Endpoint; Active = $Active }
    }
    if ($null -eq $Status) { return (& $path 'unavailable' 'unavailable') }
    $backend = Get-FleetProperty $Status 'BackendState' 'Unknown'
    if ($backend -ne 'Running') { return (& $path 'unavailable' "unavailable (backend $backend)") }
    $self = Get-FleetProperty $Status 'Self'
    if ($self -and $TailnetAddress -in @(Get-FleetProperty $self 'TailscaleIPs' @())) { return (& $path 'self' 'self') }

    $peer = $null
    $peers = Get-FleetProperty $Status 'Peer'
    if ($peers) {
        foreach ($entry in $peers.PSObject.Properties) {
            if ($TailnetAddress -in @(Get-FleetProperty $entry.Value 'TailscaleIPs' @())) { $peer = $entry.Value; break }
        }
    }
    if (-not $peer) { return (& $path 'unknown' 'unknown (not in netmap)') }

    $active = [bool](Get-FleetProperty $peer 'Active' $false)
    $peerRelay = [string](Get-FleetProperty $peer 'PeerRelay' '')
    $current = [string](Get-FleetProperty $peer 'CurAddr' '')
    $relay = [string](Get-FleetProperty $peer 'Relay' '')
    if (-not (Get-FleetProperty $peer 'Online' $true)) { return (& $path 'offline' 'offline' -Active $active) }
    if ($peerRelay) { return (& $path 'peer-relay' 'peer-relay' -Endpoint $peerRelay -Active $active) }
    if ($current) { return (& $path 'direct' 'direct' -Endpoint $current -Active $active) }
    if ($relay) { return (& $path 'relay' "relay `"$relay`"" -Region $relay -Active $active) }
    return (& $path 'unknown' 'unknown (no path fields)' -Active $active)
}

function Get-FleetCandidateRank {
    # Lower is better. LAN beats every tailnet path; among tailnet paths direct beats
    # peer-relay beats DERP relay; the MagicDNS name trails its tailnet address.
    [CmdletBinding()]
    [OutputType([double])]
    param([Parameter(Mandatory)]$Candidate)
    if ($Candidate.Kind -eq 'lan') { return 0.0 }
    $state = if ($Candidate.TailscalePath) { $Candidate.TailscalePath.State } else { 'unknown' }
    $rank = switch ($state) { 'self' { 1.0 } 'direct' { 1.0 } 'peer-relay' { 2.0 } 'relay' { 3.0 } default { 4.0 } }
    if ($Candidate.Kind -eq 'magicdns') { $rank += 0.5 }
    return $rank
}

function Select-FleetRecommendation {
    # Pure. LAN candidates are ineligible off the home LAN even if they connected: the
    # host that answered is probably someone else's.
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][AllowEmptyCollection()][object[]]$Candidates,
        [Parameter(Mandatory)][bool]$OnHomeLan
    )
    $eligible = @($Candidates | Where-Object {
            $_.Reachable -and -not $_.Skipped -and ($OnHomeLan -or $_.Kind -ne 'lan')
        })
    if ($eligible.Count -eq 0) { return $null }
    $eligible | Sort-Object -Property @{ Expression = { Get-FleetCandidateRank $_ } }, @{ Expression = { $_.Order } } | Select-Object -First 1
}

function Invoke-FleetPathTest {
    [CmdletBinding()]
    param(
        [string[]]$Target,
        [string[]]$Service,
        [ValidateRange(100, 30000)][int]$BudgetMs = 1500,
        [Parameter(Mandatory)][string]$InventoryPath,
        [switch]$AsJson,
        # Test seam: replaces the TCP connect. See the connector contract above.
        [scriptblock]$Connector
    )
    $clock = [Diagnostics.Stopwatch]::StartNew()
    $stage = { param($Name) Write-Verbose ('{0,6} ms  {1}' -f [math]::Round($clock.Elapsed.TotalMilliseconds), $Name) }
    $inventory = Import-FleetInventory -Path $InventoryPath
    & $stage 'inventory loaded'

    $selected = if ($Target) {
        foreach ($name in $Target) {
            $match = @($inventory.Hosts | Where-Object { $_.Name -ieq $name })
            if ($match.Count -eq 0) { throw "Unknown target '$name'. Known: $(($inventory.Hosts.Name) -join ', ')." }
            $match[0]
        }
    }
    else { $inventory.Hosts }
    $selected = @($selected | Sort-Object -Property Name -Unique | Sort-Object { [array]::IndexOf(@($inventory.Hosts.Name), $_.Name) })

    # Plan every endpoint and candidate. Candidates start unskipped; the location
    # decision below skips the LAN ones when home is not proven.
    $endpoints = [Collections.Generic.List[object]]::new()
    foreach ($h in $selected) {
        foreach ($svc in $h.Services) {
            if ($Service -and $svc.Service -notin $Service) { continue }
            foreach ($port in $svc.Ports) {
                $candidates = foreach ($c in (New-FleetCandidatePlan -FleetHost $h -Via $svc.Via -Port $port)) {
                    [pscustomobject]([ordered]@{
                            Kind = $c.Kind; Address = $c.Address; Port = $c.Port; Order = $c.Order
                            Reachable = $false; LatencyMs = $null; Skipped = $null
                            Error = $null; RemoteAddress = $null
                            Interface = $null; SourceAddress = $null; NextHop = $null; TailscalePath = $null
                        })
                }
                $endpoints.Add([pscustomobject]@{ FleetHost = $h; Service = $svc.Service; Port = $port; Candidates = @($candidates); Recommended = $null })
            }
        }
    }

    # Wave A: the gateway probes that decide the location, plus every candidate that
    # does not depend on it (tailnet and MagicDNS). Each distinct address:port is
    # probed once. While they run, pay the CIM start-up cost of the neighbour and
    # route cmdlets in this runspace.
    $results = @{}
    $waveA = [ordered]@{}
    $gatewayProbes = @(foreach ($g in $inventory.Gateways) { foreach ($p in $g.ProbePorts) { [pscustomobject]@{ Address = $g.Ip; Port = $p } } })
    foreach ($probe in $gatewayProbes) { $waveA["$($probe.Address):$($probe.Port)"] = $probe }
    foreach ($e in $endpoints) {
        foreach ($c in $e.Candidates) {
            if ($c.Kind -ne 'lan') { $waveA["$($c.Address):$($c.Port)"] = [pscustomobject]@{ Address = $c.Address; Port = $c.Port } }
        }
    }
    $warmAddress = if (@($inventory.Gateways).Count -gt 0) { @($inventory.Gateways)[0].Ip } else { $null }
    $warmUp = { if ($warmAddress) { $null = Get-FleetNeighbor -IPAddress $warmAddress; $null = Find-FleetRoute -Address $warmAddress } }
    foreach ($r in (Invoke-FleetProbeBatch -Probes @($waveA.Values) -BudgetMs $BudgetMs -Connector $Connector -WhileWaiting $warmUp)) {
        $results["$($r.Address):$($r.Port)"] = $r
    }
    & $stage 'wave A done (gateways and tailnet candidates)'
    $location = Resolve-FleetLocation -Gateways $inventory.Gateways -ProbeResults @($results.Values)
    # One transient stall (seen about once in 40 runs: both ports on the gateway timing
    # out together) would otherwise read as "away from home". A colliding MAC is a
    # definite answer and is not retried; anything else gets one more look.
    if ($location.State -ne 'home-lan' -and -not $location.Collision -and $gatewayProbes.Count -gt 0) {
        & $stage 'no home verdict; retrying the gateway probes once'
        foreach ($r in (Invoke-FleetProbeBatch -Probes $gatewayProbes -BudgetMs $BudgetMs -Connector $Connector)) {
            $results["$($r.Address):$($r.Port)"] = $r
        }
        $location = Resolve-FleetLocation -Gateways $inventory.Gateways -ProbeResults @($results.Values)
    }
    $onHomeLan = $location.State -eq 'home-lan'
    & $stage "location: $($location.State)"

    # Wave B: LAN candidates, only once home is proven. Off the home LAN a connect to
    # a LAN address would reach whatever network this machine is on. Route lookups for
    # every address run while these probes are in flight.
    $routes = @{}
    $routeFor = {
        param($Address)
        if ($Address -and -not $routes.ContainsKey($Address)) { $routes[$Address] = Find-FleetRoute -Address $Address }
    }
    $waveB = [ordered]@{}
    foreach ($e in $endpoints) {
        foreach ($c in $e.Candidates) {
            if ($c.Kind -ne 'lan') { continue }
            if (-not $onHomeLan) { $c.Skipped = 'not on the home LAN'; continue }
            # Home is verified per interface. Resolve this candidate's egress BEFORE probing
            # it: on a multihomed host Ethernet can verify the home gateway while Wi-Fi sits
            # on an unrelated network that reuses the subnet, and a connect routed out of
            # Wi-Fi would reach a stranger's machine.
            & $routeFor $c.Address
            $route = $routes[$c.Address]
            if ($route) { $c.Interface = $route.Interface; $c.SourceAddress = $route.SourceAddress; $c.NextHop = $route.NextHop }
            if (Test-FleetLanEgress -Route $route -VerifiedInterfaces $location.VerifiedInterfaces) {
                $waveB["$($c.Address):$($c.Port)"] = [pscustomobject]@{ Address = $c.Address; Port = $c.Port }
            }
            else { $c.Skipped = 'egress-not-verified' }
        }
    }
    $routeLookups = {
        foreach ($e in $endpoints) {
            foreach ($c in $e.Candidates) { if (-not $c.Skipped -and $c.Kind -ne 'magicdns') { & $routeFor $c.Address } }
        }
    }
    foreach ($r in (Invoke-FleetProbeBatch -Probes @($waveB.Values) -BudgetMs $BudgetMs -Connector $Connector -WhileWaiting $routeLookups)) {
        $results["$($r.Address):$($r.Port)"] = $r
    }

    & $stage 'wave B done (LAN candidates and route lookups)'

    # Egress interface: one Find-NetRoute per distinct IP, in this runspace (it is
    # slower in parallel runspaces). A MagicDNS name is routed by the address it
    # actually connected to, so it is looked up after the probes.
    foreach ($e in $endpoints) {
        foreach ($c in $e.Candidates) {
            if ($c.Skipped) { continue }
            $r = $results["$($c.Address):$($c.Port)"]
            $c.Reachable = [bool]$r.Reachable; $c.LatencyMs = $r.LatencyMs; $c.Error = $r.Error; $c.RemoteAddress = $r.RemoteAddress
            $ip = if ($c.Kind -eq 'magicdns') { $c.RemoteAddress } else { $c.Address }
            if (-not $ip) { continue }
            & $routeFor $ip
            $route = $routes[$ip]
            if ($route) { $c.Interface = $route.Interface; $c.SourceAddress = $route.SourceAddress; $c.NextHop = $route.NextHop }
        }
    }
    # Tailscale is read after the probes so a sleeping peer path has been exercised.
    & $stage 'candidate results and routes applied'
    $tailscale = Get-FleetTailscaleStatus
    & $stage 'tailscale status read'
    $hostPaths = @{}
    foreach ($h in $selected) {
        $hostPaths[$h.Name] = if ($h.Tailnet) { Resolve-FleetTailscalePath -Status $tailscale.Status -TailnetAddress $h.Tailnet } else { $null }
    }
    foreach ($e in $endpoints) {
        foreach ($c in $e.Candidates) {
            if ($c.Kind -in 'tailnet', 'magicdns') { $c.TailscalePath = $hostPaths[$e.FleetHost.Name] }
        }
        $e.Recommended = Select-FleetRecommendation -Candidates $e.Candidates -OnHomeLan $onHomeLan
    }

    $toRecommendation = {
        param($Candidate)
        if (-not $Candidate) { return $null }
        [pscustomobject]@{
            Kind = $Candidate.Kind; Address = $Candidate.Address; Port = $Candidate.Port; LatencyMs = $Candidate.LatencyMs
            Interface = $Candidate.Interface; Path = if ($Candidate.TailscalePath) { $Candidate.TailscalePath.Display } else { $null }
        }
    }
    $targets = foreach ($h in $selected) {
        $mine = @($endpoints | Where-Object { $_.FleetHost.Name -eq $h.Name })
        $best = @($mine | Where-Object { $_.Recommended } | ForEach-Object { $_.Recommended } |
                Sort-Object -Property @{ Expression = { Get-FleetCandidateRank $_ } }, @{ Expression = { $_.LatencyMs } }) | Select-Object -First 1
        [pscustomobject]@{
            Name = $h.Name
            Tailscale = if ($hostPaths[$h.Name]) { $hostPaths[$h.Name] } else { [pscustomobject]@{ State = 'none'; Display = 'no tailnet address'; Region = $null; Endpoint = $null; Active = $null } }
            Recommended = & $toRecommendation $best
            Endpoints = @($mine | ForEach-Object {
                    [pscustomobject]@{ Service = $_.Service; Port = $_.Port; Recommended = (& $toRecommendation $_.Recommended); Candidates = $_.Candidates }
                })
        }
    }
    $targets = @($targets)

    $summary = foreach ($t in $targets) {
        foreach ($e in $t.Endpoints) {
            [pscustomobject]@{
                Target = $t.Name; Service = $e.Service; Port = $e.Port
                Recommended = if ($e.Recommended) { "$($e.Recommended.Kind) $($e.Recommended.Address)" } else { 'none' }
                LatencyMs = if ($e.Recommended) { $e.Recommended.LatencyMs } else { $null }
                Path = if ($e.Recommended) { $e.Recommended.Path } else { $null }
                Interface = if ($e.Recommended) { $e.Recommended.Interface } else { $null }
            }
        }
    }

    $report = [pscustomobject]([ordered]@{
            Schema = $script:FleetSchema
            GeneratedAt = (Get-Date).ToUniversalTime().ToString('o')
            ComputerName = [Environment]::MachineName
            BudgetMs = $BudgetMs
            ElapsedMs = [math]::Round($clock.Elapsed.TotalMilliseconds)
            InventoryPath = $inventory.Path
            InventoryMeasured = $inventory.Measured
            Location = $location
            Tailscale = [pscustomobject]@{
                Available = $tailscale.Available; Error = $tailscale.Error
                BackendState = if ($tailscale.Status) { Get-FleetProperty $tailscale.Status 'BackendState' } else { $null }
            }
            Targets = $targets
            Summary = @($summary)
        })
    if ($AsJson) { return ($report | ConvertTo-Json -Depth 10) }
    $report
}

if ($MyInvocation.InvocationName -ne '.') {
    $forward = @{ BudgetMs = $BudgetMs; InventoryPath = $InventoryPath }
    if ($PSBoundParameters.ContainsKey('Target')) { $forward.Target = $Target }
    if ($PSBoundParameters.ContainsKey('Service')) { $forward.Service = $Service }
    if ($AsJson) { $forward.AsJson = $true }
    Invoke-FleetPathTest @forward
}
