#Requires -Version 7.0
<#
.SYNOPSIS
Reconcile an explicitly enrolled USB4 peer and optional Windows internet sharing.
.DESCRIPTION
This standalone controller has no application, agent-bus, or user-profile dependency.
Enrollment pins the local adapter and directly observed peer MAC. This is physical
link enrollment, not cryptographic authentication. Unknown peers are never adopted.
Plan/DryRun performs observations but writes no addresses, routes, DNS, ICS, or files.
Apply preserves foreign routes, DNS, sharing pairs, and every WinNAT instance.
Existing MTU, SMB configuration, and firewall rules are never changed.
.PARAMETER PolicyPath
Administrator-controlled JSON policy for exactly one local host and peer link.
.PARAMETER StatePath
Controller-owned state; preexisting addresses and routes are never silently adopted.
.PARAMETER ReportPath
Atomic generic transport report, independent of application or laboratory readiness.
.PARAMETER Apply
Apply the validated plan. Without this switch the controller only observes/plans.
.PARAMETER DryRun
Explicitly prohibit all mutations, including state/report publication.
.EXAMPLE
pwsh -NoProfile -File Invoke-Usb4PeerNetwork.ps1 -PolicyPath policy.json -DryRun
#>
[CmdletBinding(SupportsShouldProcess)]
param(
    [string]$PolicyPath,
    [string]$StatePath,
    [string]$ReportPath,
    [switch]$Apply,
    [switch]$DryRun,
    [Alias('h', 'help')][switch]$ShowHelp
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Copy-Usb4NetworkValue {
    <# .SYNOPSIS
    Copy plain policy/state data without retaining mutable references.
    #>
    param([object]$Value)
    , (ConvertFrom-Json -InputObject (ConvertTo-Json -InputObject $Value -Depth 40 -Compress) -AsHashtable)
}

function ConvertTo-Usb4Mac {
    <# .SYNOPSIS
    Validate and normalize a unicast, nonempty hardware address.
    #>
    param([string]$Address)
    $value = $Address.Replace(':', '').Replace('-', '').ToUpperInvariant()
    if ($value -notmatch '^[0-9A-F]{12}$' -or $value -eq '000000000000' -or
        ([Convert]::ToByte($value.Substring(0, 2), 16) -band 1)) { throw 'A nonempty unicast MAC address is required.' }
    $value
}

function ConvertTo-Usb4IPv4Number {
    <# .SYNOPSIS
    Convert a strict dotted IPv4 address to an unsigned network-order number.
    #>
    param([string]$Address)
    if ($Address -notmatch '^(0|[1-9]\d{0,2})(\.(0|[1-9]\d{0,2})){3}$') { throw "Invalid IPv4 address: $Address" }
    $bytes = [Net.IPAddress]::Parse($Address).GetAddressBytes()
    if ($bytes.Count -ne 4) { throw 'An IPv4 address is required.' }
    ([uint64]$bytes[0] * 16777216) + ([uint64]$bytes[1] * 65536) + ([uint64]$bytes[2] * 256) + [uint64]$bytes[3]
}

function Get-Usb4IPv4Network {
    <# .SYNOPSIS
    Return the numeric network and broadcast boundaries for a prefix.
    #>
    param([string]$Address, [int]$PrefixLength)
    if ($PrefixLength -lt 0 -or $PrefixLength -gt 32) { throw 'Invalid IPv4 prefix length.' }
    $number = ConvertTo-Usb4IPv4Number $Address
    $size = [uint64][Math]::Pow(2, 32 - $PrefixLength)
    $network = [uint64]([Math]::Floor($number / $size) * $size)
    @{ first = $network; last = $network + $size - 1; number = $number }
}

function Test-Usb4PrivateIPv4 {
    <# .SYNOPSIS
    Accept only RFC1918 addresses for routed peer and sharing subnets.
    #>
    param([string]$Address)
    $n = ConvertTo-Usb4IPv4Number $Address
    ($n -ge 167772160 -and $n -le 184549375) -or
    ($n -ge 2886729728 -and $n -le 2887778303) -or
    ($n -ge 3232235520 -and $n -le 3232301055)
}

function Assert-Usb4NetworkPolicy {
    <# .SYNOPSIS
    Validate one explicit host, physical peer, and optional sharing role.
    #>
    param([Parameter(Mandatory)][hashtable]$Policy, [string]$CurrentHost)
    $required = @('schemaVersion', 'hostName', 'mode', 'adapterMac', 'peerMac', 'peerLinkLocal', 'localIPv4', 'peerIPv4', 'prefixLength')
    foreach ($key in $required) { if (-not $Policy.ContainsKey($key)) { throw "Policy lacks $key." } }
    foreach ($key in $Policy.Keys) {
        if ($key -notin ($required + @('localUla', 'peerUla', 'sharing'))) { throw "Unknown policy key: $key" }
    }
    if ($Policy.schemaVersion -isnot [long] -and $Policy.schemaVersion -isnot [int]) { throw 'schemaVersion must be an integer.' }
    if ($Policy.schemaVersion -ne 1 -or $Policy.mode -cnotin @('Peer', 'Provider', 'Consumer') -or
        $Policy.hostName -isnot [string] -or $Policy.hostName -notmatch '^[a-zA-Z0-9][a-zA-Z0-9-]{0,62}$') { throw 'Unsupported policy schema, mode, or host.' }
    if ($CurrentHost -and $Policy.hostName -ine $CurrentHost) { throw 'Policy does not enroll this host.' }
    $p = Copy-Usb4NetworkValue $Policy
    $p.adapterMac = ConvertTo-Usb4Mac $p.adapterMac; $p.peerMac = ConvertTo-Usb4Mac $p.peerMac
    if ($p.adapterMac -eq $p.peerMac) { throw 'Local and peer MACs must differ.' }
    if ($p.peerLinkLocal -isnot [string] -or $p.peerLinkLocal.Contains('%')) { throw 'Peer link-local policy address must be unscoped.' }
    $ll = [Net.IPAddress]::Parse($p.peerLinkLocal)
    if (-not $ll.IsIPv6LinkLocal) { throw 'Peer bootstrap must use scoped IPv6 link-local.' }
    $p.peerLinkLocal = $ll.ToString()
    if (($p.prefixLength -isnot [int] -and $p.prefixLength -isnot [long]) -or $p.prefixLength -ne 30) { throw 'Transport policy requires an integer /30.' }
    $local = Get-Usb4IPv4Network $p.localIPv4 $p.prefixLength
    $peer = Get-Usb4IPv4Network $p.peerIPv4 $p.prefixLength
    if (-not (Test-Usb4PrivateIPv4 $p.localIPv4) -or -not (Test-Usb4PrivateIPv4 $p.peerIPv4) -or
        $local.first -ne $peer.first -or $local.number -eq $peer.number -or
        $local.number -in @($local.first, $local.last) -or $peer.number -in @($peer.first, $peer.last)) { throw 'Invalid isolated private transport allocation.' }
    if ($p.ContainsKey('localUla') -ne $p.ContainsKey('peerUla')) { throw 'Both ULA endpoints must be specified together.' }
    if ($p.ContainsKey('localUla')) {
        foreach ($name in @('localUla', 'peerUla')) {
            $ip = [Net.IPAddress]::Parse($p[$name])
            if ($ip.AddressFamily -ne [Net.Sockets.AddressFamily]::InterNetworkV6 -or
                ($ip.GetAddressBytes()[0] -band 254) -ne 252 -or $p[$name].Contains('%')) { throw 'Only unscoped ULA addresses are allowed.' }
            $p[$name] = $ip.ToString()
        }
        if ($p.localUla -eq $p.peerUla) { throw 'ULA endpoints must differ.' }
    }
    if ($p.mode -eq 'Peer') {
        if ($p.ContainsKey('sharing')) { throw 'Peer-only policy cannot request sharing.' }
    }
    else {
        if (-not $p.ContainsKey('sharing') -or $p.sharing -isnot [hashtable]) { throw 'An explicit sharing policy is required.' }
        $keys = @('providerAddress', 'consumerAddress', 'prefixLength', 'preferSharedInternet', 'routeMetric', 'probeAddress', 'probePort', 'allowWirelessFallback')
        foreach ($key in $keys) { if (-not $p.sharing.ContainsKey($key)) { throw "Sharing policy lacks $key." } }
        $s = $p.sharing
        if (-not $s.ContainsKey('backend')) { $s.backend = 'Ics' }
        if ($s.backend -cnotin @('Ics', 'ExistingWinNat')) { throw 'Unsupported explicit sharing backend.' }
        $extra = $(if ($s.backend -eq 'ExistingWinNat') { @('natName', 'natPrefix', 'dnsServers') } else { @() })
        foreach ($key in $s.Keys) { if ($key -notin ($keys + @('backend') + $extra)) { throw "Unknown sharing policy key: $key" } }
        foreach ($key in $extra) { if (-not $s.ContainsKey($key)) { throw "Existing WinNAT policy lacks $key." } }
        $expectedPrefix = $(if ($s.backend -eq 'ExistingWinNat') { 30 } else { 24 })
        if (($s.prefixLength -isnot [int] -and $s.prefixLength -isnot [long]) -or $s.prefixLength -ne $expectedPrefix -or
            ($s.routeMetric -isnot [int] -and $s.routeMetric -isnot [long]) -or ($s.probePort -isnot [int] -and $s.probePort -isnot [long]) -or
            $s.preferSharedInternet -isnot [bool] -or $s.allowWirelessFallback -isnot [bool] -or
            $s.routeMetric -lt 1 -or $s.routeMetric -gt 9999 -or $s.probePort -ne 443) { throw 'Unsupported sharing prefix, preference, metric, or HTTPS probe port.' }
        $a = Get-Usb4IPv4Network $s.providerAddress $s.prefixLength; $b = Get-Usb4IPv4Network $s.consumerAddress $s.prefixLength
        if (-not (Test-Usb4PrivateIPv4 $s.providerAddress) -or -not (Test-Usb4PrivateIPv4 $s.consumerAddress) -or
            $a.first -ne $b.first -or $a.number -eq $b.number -or
            $a.number -in @($a.first, $a.last) -or $b.number -in @($b.first, $b.last) -or
            ($a.first -le $local.last -and $local.first -le $a.last)) { throw 'Invalid or overlapping private sharing subnet.' }
        $probe = ConvertTo-Usb4IPv4Number $s.probeAddress
        if (Test-Usb4PrivateIPv4 $s.probeAddress) { throw 'Internet probe must not be a private endpoint.' }
        if ($probe -lt 16777216 -or $probe -ge 3758096384 -or ($probe -ge 2130706432 -and $probe -le 2147483647) -or
            ($probe -ge 2851995648 -and $probe -le 2852061183)) { throw 'Invalid internet probe address.' }
        if ($s.backend -eq 'ExistingWinNat') {
            if ($s.natName -isnot [string] -or $s.natName -notmatch '^[A-Za-z0-9][A-Za-z0-9 _.-]{0,63}$' -or
                $s.natPrefix -isnot [string] -or $s.natPrefix -notmatch '^([^/]+)/([0-9]{1,2})$') { throw 'An explicit existing NAT name and IPv4 prefix are required.' }
            $parent = Get-Usb4IPv4Network $Matches[1] ([int]$Matches[2])
            if (-not (Test-Usb4PrivateIPv4 (($s.natPrefix -split '/')[0])) -or $parent.first -gt $a.first -or $parent.last -lt $a.last) { throw 'Sharing child must be inside the approved existing private NAT prefix.' }
            if ($s.dnsServers -isnot [array] -or $s.dnsServers.Count -eq 0 -or $s.dnsServers.Count -gt 3 -or @($s.dnsServers | Select-Object -Unique).Count -ne $s.dnsServers.Count) { throw 'Existing WinNAT requires an explicit nonempty unique DNS server list.' }
            foreach ($server in $s.dnsServers) {
                $number = ConvertTo-Usb4IPv4Number $server
                if ($number -lt 16777216 -or $number -ge 3758096384 -or ($number -ge 2130706432 -and $number -le 2147483647) -or
                    ($number -ge 2851995648 -and $number -le 2852061183) -or
                    ($number -ge $a.first -and $number -le $a.last)) { throw 'DNS servers cannot be loopback, multicast, or a private sharing endpoint.' }
            }
        }
    }
    $p
}

function Get-Usb4Ipv6DnsConfiguration {
    <# .SYNOPSIS
    Observe configured IPv6 DNS separately from Windows' legacy resolver fallback.
    #>
    param([string]$InterfaceGuid)
    $path = "HKLM:\SYSTEM\CurrentControlSet\Services\Tcpip6\Parameters\Interfaces\{$InterfaceGuid}"
    if (-not (Test-Path -LiteralPath $path -ErrorAction Stop)) { return @{ observed = $false } }
    $registry = Get-ItemProperty -LiteralPath $path -ErrorAction Stop
    $static = $registry.PSObject.Properties['NameServer']
    $dhcp = $registry.PSObject.Properties['Dhcpv6DNSServers']
    @{ observed = $true; nameServer = $(if ($static) { [string]$static.Value } else { '' });
        dhcpServers = $(if ($dhcp -and $null -ne $dhcp.Value) { [BitConverter]::ToString([byte[]]$dhcp.Value).Replace('-', '') } else { '' })
    }
}

function Get-Usb4NetworkSnapshot {
    <# .SYNOPSIS
    Read current adapters, addresses, routes, sharing ownership, DNS, and WinNAT.
    #>
    param([switch]$SkipNat)
    $addresses = @(Get-NetIPAddress -PolicyStore ActiveStore -ErrorAction Stop | ForEach-Object {
            @{ address = ($_.IPAddress -split '%', 2)[0]; prefix = [int]$_.PrefixLength; index = [int]$_.InterfaceIndex; family = [string]$_.AddressFamily; state = [string]$_.AddressState;
                prefixOrigin = [string]$_.PrefixOrigin; suffixOrigin = [string]$_.SuffixOrigin; skipAsSource = [bool]$_.SkipAsSource
            }
        })
    $interfaces = @(Get-NetIPInterface -ErrorAction Stop)
    $dns = @(Get-DnsClientServerAddress -ErrorAction Stop)
    $adapters = @(Get-NetAdapter -IncludeHidden -ErrorAction Stop | ForEach-Object {
            $adapter = $_; $ipif = @($interfaces | Where-Object { $_.InterfaceIndex -eq $adapter.ifIndex -and $_.AddressFamily -eq 'IPv4' })
            $guid = ([string]$adapter.InterfaceGuid).Trim('{}').ToLowerInvariant()
            $registryPath = "HKLM:\SYSTEM\CurrentControlSet\Services\Tcpip\Parameters\Interfaces\{$guid}"
            $dnsRegistry = $(if (Test-Path -LiteralPath $registryPath -ErrorAction Stop) { Get-ItemProperty -LiteralPath $registryPath -ErrorAction Stop } else { $null })
            $explicitDns = $(if ($dnsRegistry) { $dnsRegistry.PSObject.Properties['NameServer'] } else { $null })
            @{ mac                = $(if ($adapter.MacAddress) { $adapter.MacAddress.Replace('-', '').Replace(':', '').ToUpperInvariant() } else { '' });
                guid = ([string]$adapter.InterfaceGuid).Trim('{}').ToLowerInvariant(); index = [int]$adapter.ifIndex; name = [string]$adapter.Name;
                description = [string]$adapter.InterfaceDescription; up = ([string]$adapter.Status -eq 'Up'); hardware = [bool]$adapter.HardwareInterface;
                wired             = ([int]$adapter.InterfaceType -eq 6 -and [bool]$adapter.HardwareInterface);
                wireless          = ([int]$adapter.InterfaceType -eq 71 -and [bool]$adapter.HardwareInterface);
                metric            = $(if ($ipif.Count -eq 1) { [int]$ipif[0].InterfaceMetric } else { 9999 });
                dhcp              = $(if ($ipif.Count -eq 1) { [string]$ipif[0].Dhcp } else { 'Unknown' });
                mtu               = $(if ($ipif.Count -eq 1) { [int]$ipif[0].NlMtu } else { 0 });
                weakHostSend      = $(if ($ipif.Count -eq 1) { [string]$ipif[0].WeakHostSend } else { 'Unknown' });
                weakHostReceive   = $(if ($ipif.Count -eq 1) { [string]$ipif[0].WeakHostReceive } else { 'Unknown' });
                forwarding        = $(if ($ipif.Count -eq 1) { [string]$ipif[0].Forwarding } else { 'Unknown' });
                addresses         = @($addresses | Where-Object index -EQ $adapter.ifIndex);
                dns               = @($dns | Where-Object { $_.InterfaceIndex -eq $adapter.ifIndex -and $_.AddressFamily -eq 2 } | ForEach-Object { $_.ServerAddresses });
                dns6              = @($dns | Where-Object { $_.InterfaceIndex -eq $adapter.ifIndex -and $_.AddressFamily -eq 23 } | ForEach-Object { $_.ServerAddresses });
                dns6Configuration = $(if ($adapter.InterfaceDescription -match 'USB4|Thunderbolt.*(Network|Ethernet)') { Get-Usb4Ipv6DnsConfiguration $guid } else { @{ observed = $false } });
                dnsMode           = $(if (-not $dnsRegistry) { 'Unknown' } elseif ($explicitDns -and [string]$explicitDns.Value) { 'Static' } else { 'Dhcp' })
            }
        })
    $routes = @(Get-NetRoute -PolicyStore ActiveStore -ErrorAction Stop | ForEach-Object {
            @{ destination = [string]$_.DestinationPrefix; nextHop = [string]$_.NextHop; index = [int]$_.InterfaceIndex; metric = [int]$_.RouteMetric }
        })
    $manager = New-Object -ComObject HNetCfg.HNetShare
    $sharing = @(foreach ($connection in $manager.EnumEveryConnection()) {
            $properties = $manager.NetConnectionProps($connection); $configuration = $manager.INetSharingConfigurationForINetConnection($connection)
            if ($configuration.SharingEnabled) { @{ guid = ([string]$properties.Guid).Trim('{}').ToLowerInvariant(); type = [int]$configuration.SharingConnectionType } }
        })
    $nat = $(if ($SkipNat) { @() } else { @(Get-NetNat -ErrorAction Stop | ForEach-Object { @{ name = [string]$_.Name; prefix = [string]$_.InternalIPInterfaceAddressPrefix } }) })
    @{ hostName = [Environment]::MachineName; adapters = $adapters; addresses = $addresses; routes = $routes; sharing = $sharing; nat = @($nat); natObserved = -not $SkipNat }
}

function Invoke-Usb4BoundTcpProbe {
    <# .SYNOPSIS
    Require source-bound TLS and a bounded authenticated HTTPS response.
    #>
    param([string]$SourceAddress, [string]$RemoteAddress, [int]$Port = 443, [int]$TimeoutMilliseconds = 3500)
    $client = $null
    $tls = $null
    try {
        $client = [Net.Sockets.TcpClient]::new([Net.IPEndPoint]::new([Net.IPAddress]::Parse($SourceAddress), 0))
        if (-not $client.ConnectAsync($RemoteAddress, $Port).Wait($TimeoutMilliseconds)) { throw 'Source-bound TCP probe timed out.' }
        $tls = [Net.Security.SslStream]::new($client.GetStream(), $false)
        if (-not $tls.AuthenticateAsClientAsync($RemoteAddress).Wait($TimeoutMilliseconds)) { throw 'Authenticated TLS probe timed out.' }
        $request = [Text.Encoding]::ASCII.GetBytes("HEAD / HTTP/1.1`r`nHost: $RemoteAddress`r`nConnection: close`r`n`r`n")
        $tls.WriteTimeout = $TimeoutMilliseconds; $tls.ReadTimeout = $TimeoutMilliseconds
        $tls.Write($request, 0, $request.Length)
        $buffer = [byte[]]::new(2048); $count = $tls.Read($buffer, 0, $buffer.Length)
        $status = [Text.Encoding]::ASCII.GetString($buffer, 0, $count)
        if ($status -notmatch '^HTTP/1\.[01] ([23]\d\d)\b') { throw "HTTPS probe returned no successful response: $($status.Split("`r")[0])" }
        @{ success = $true; source = $client.Client.LocalEndPoint.Address.ToString(); remote = $RemoteAddress; tlsAuthenticated = $true; httpStatus = [int]$Matches[1] }
    }
    catch { @{ success = $false; source = $SourceAddress; remote = $RemoteAddress; error = $_.Exception.Message } }
    finally { if ($tls) { $tls.Dispose() }; if ($client) { $client.Dispose() } }
}

function Invoke-Usb4BoundDnsProbe {
    <# .SYNOPSIS
    Verify DNS through the sharing gateway using the USB4 source address.
    #>
    param([string]$SourceAddress, [string]$ServerAddress)
    $client = $null
    try {
        $client = [Net.Sockets.UdpClient]::new([Net.IPEndPoint]::new([Net.IPAddress]::Parse($SourceAddress), 0))
        $client.Client.ReceiveTimeout = 2000
        $id = [Security.Cryptography.RandomNumberGenerator]::GetInt32(1, 65536)
        $bytes = [Collections.Generic.List[byte]]::new()
        $bytes.AddRange([byte[]]@(($id -shr 8), ($id -band 255), 1, 0, 0, 1, 0, 0, 0, 0, 0, 0))
        foreach ($label in @('www', 'microsoft', 'com')) { $bytes.Add([byte]$label.Length); $bytes.AddRange([Text.Encoding]::ASCII.GetBytes($label)) }
        $bytes.AddRange([byte[]]@(0, 0, 1, 0, 1))
        $client.Connect($ServerAddress, 53)
        if ($client.Send($bytes.ToArray(), $bytes.Count) -ne $bytes.Count) { throw 'DNS request was not completely sent.' }
        $endpoint = [Net.IPEndPoint]::new([Net.IPAddress]::Any, 0); $reply = $client.Receive([ref]$endpoint)
        if ($reply.Length -lt 12 -or $reply[0] -ne ($id -shr 8) -or $reply[1] -ne ($id -band 255) -or
            ($reply[2] -band 128) -eq 0 -or ($reply[3] -band 15) -ne 0 -or ($reply[6] * 256 + $reply[7]) -lt 1) { throw 'DNS gateway returned no matching successful answer.' }
        @{ success = $true; source = $client.Client.LocalEndPoint.Address.ToString(); server = $endpoint.Address.ToString(); query = 'www.microsoft.com' }
    }
    catch { @{ success = $false; source = $SourceAddress; server = $ServerAddress; error = $_.Exception.Message } }
    finally { if ($client) { $client.Dispose() } }
}

function Set-Usb4IcsPair {
    <# .SYNOPSIS
    Change only the explicitly checked sharing pair, retaining its exact preimage.
    #>
    [CmdletBinding(SupportsShouldProcess)]
    param([string]$PublicGuid, [string]$PrivateGuid, [object[]]$Expected)
    $manager = New-Object -ComObject HNetCfg.HNetShare
    $connections = @{}; $current = @()
    foreach ($connection in $manager.EnumEveryConnection()) {
        $properties = $manager.NetConnectionProps($connection); $configuration = $manager.INetSharingConfigurationForINetConnection($connection)
        $guid = ([string]$properties.Guid).Trim('{}').ToLowerInvariant(); $connections[$guid] = $configuration
        if ($configuration.SharingEnabled) { $current += @{ guid = $guid; type = [int]$configuration.SharingConnectionType } }
    }
    $fingerprint = { param($Value) (@($Value | Sort-Object guid, type | ForEach-Object { "$($_.guid):$($_.type)" }) -join '|') }
    if ((& $fingerprint $current) -cne (& $fingerprint $Expected)) { throw 'ICS ownership changed after observation.' }
    if ($PublicGuid -and (-not $connections.ContainsKey($PublicGuid) -or -not $connections.ContainsKey($PrivateGuid))) { throw 'Selected sharing adapter disappeared.' }
    if (-not $PSCmdlet.ShouldProcess("$PublicGuid / $PrivateGuid", 'Change the explicitly observed ICS pair')) { return }
    try {
        foreach ($entry in $current) { $connections[$entry.guid].DisableSharing() }
        if ($PublicGuid) { $connections[$PublicGuid].EnableSharing(0); $connections[$PrivateGuid].EnableSharing(1) }
    }
    catch {
        $failure = $_.Exception.Message; $restoreErrors = [Collections.Generic.List[string]]::new()
        foreach ($guid in @($PublicGuid, $PrivateGuid) | Where-Object { $_ -and $connections.ContainsKey($_) }) {
            try { if ($connections[$guid].SharingEnabled) { $connections[$guid].DisableSharing() } } catch { $restoreErrors.Add($_.Exception.Message) }
        }
        foreach ($entry in @($current | Sort-Object type)) {
            try { $connections[$entry.guid].EnableSharing($entry.type) } catch { $restoreErrors.Add($_.Exception.Message) }
        }
        throw "ICS change failed: $failure. Restoration errors: $($restoreErrors -join '; ')"
    }
}

function Get-Usb4SelectedIPv4Route {
    <# .SYNOPSIS
    Require a unique current longest-prefix and effective-metric egress route.
    #>
    param([string]$RemoteAddress)
    $target = ConvertTo-Usb4IPv4Number $RemoteAddress
    $metrics = @(Get-NetIPInterface -AddressFamily IPv4 -ErrorAction Stop)
    $choices = @(foreach ($route in @(Get-NetRoute -AddressFamily IPv4 -PolicyStore ActiveStore -ErrorAction Stop)) {
            $parts = $route.DestinationPrefix.Split('/'); $range = Get-Usb4IPv4Network $parts[0] ([int]$parts[1])
            if ($range.first -le $target -and $target -le $range.last) {
                $metric = @($metrics | Where-Object InterfaceIndex -EQ $route.InterfaceIndex)
                if ($metric.Count -eq 1) { @{ index = [int]$route.InterfaceIndex; prefix = [int]$parts[1]; score = [int]$route.RouteMetric + [int]$metric[0].InterfaceMetric; destination = $route.DestinationPrefix; nextHop = $route.NextHop } }
            }
        })
    $best = @($choices | Sort-Object @{ Expression = 'prefix'; Descending = $true }, score | Select-Object -First 1)
    if ($best.Count -eq 1 -and @($choices | Where-Object { $_.prefix -eq $best[0].prefix -and $_.score -eq $best[0].score -and $_.index -ne $best[0].index }).Count -eq 0) { $best[0] }
}

function Get-Usb4NetworkReadback {
    <# .SYNOPSIS
    Read current mutable custody without repeating global NAT/physical inventory.
    #>
    param([ValidateSet('Address', 'Route', 'Adapter', 'Sharing')][string]$Kind, [hashtable]$Adapter)
    $result = @{ addresses = @(); routes = @(); adapters = @(); sharing = @() }
    if ($Kind -in @('Address', 'Sharing')) {
        $result.addresses = @(Get-NetIPAddress -PolicyStore ActiveStore -ErrorAction Stop | ForEach-Object {
                @{ address = ($_.IPAddress -split '%', 2)[0]; prefix = [int]$_.PrefixLength; index = [int]$_.InterfaceIndex;
                    family = [string]$_.AddressFamily; state = [string]$_.AddressState; prefixOrigin = [string]$_.PrefixOrigin;
                    suffixOrigin = [string]$_.SuffixOrigin; skipAsSource = [bool]$_.SkipAsSource
                }
            })
    }
    if ($Kind -in @('Route', 'Adapter')) {
        $result.routes = @(Get-NetRoute -PolicyStore ActiveStore -ErrorAction Stop | ForEach-Object {
                @{ destination = [string]$_.DestinationPrefix; nextHop = [string]$_.NextHop; index = [int]$_.InterfaceIndex; metric = [int]$_.RouteMetric }
            })
    }
    if ($Kind -in @('Adapter', 'Sharing')) {
        $current = @(Get-NetAdapter -IncludeHidden -ErrorAction Stop | Where-Object ifIndex -EQ $Adapter.index)
        if ($current.Count -ne 1 -or ([string]$current[0].InterfaceGuid).Trim('{}').ToLowerInvariant() -cne $Adapter.guid -or
            (ConvertTo-Usb4Mac $current[0].MacAddress) -cne $Adapter.mac) { throw 'Managed adapter identity changed during readback.' }
        $ipif = @(Get-NetIPInterface -InterfaceIndex $Adapter.index -AddressFamily IPv4 -ErrorAction Stop)
        if ($ipif.Count -ne 1) { throw 'Managed adapter lacks a unique IPv4 interface.' }
        $dns = @(Get-DnsClientServerAddress -InterfaceIndex $Adapter.index -ErrorAction Stop)
        $path = "HKLM:\SYSTEM\CurrentControlSet\Services\Tcpip\Parameters\Interfaces\{$($Adapter.guid)}"
        $registry = Get-ItemProperty -LiteralPath $path -ErrorAction Stop; $explicit = $registry.PSObject.Properties['NameServer']
        $entry = Copy-Usb4NetworkValue $Adapter
        $entry.name = [string]$current[0].Name; $entry.up = [string]$current[0].Status -eq 'Up'
        $entry.metric = [int]$ipif[0].InterfaceMetric; $entry.mtu = [int]$ipif[0].NlMtu; $entry.dhcp = [string]$ipif[0].Dhcp
        $entry.weakHostSend = [string]$ipif[0].WeakHostSend; $entry.weakHostReceive = [string]$ipif[0].WeakHostReceive
        $entry.forwarding = [string]$ipif[0].Forwarding
        $entry.dns = @($dns | Where-Object AddressFamily -EQ 2 | ForEach-Object { $_.ServerAddresses })
        $entry.dns6 = @($dns | Where-Object AddressFamily -EQ 23 | ForEach-Object { $_.ServerAddresses })
        $entry.dns6Configuration = Get-Usb4Ipv6DnsConfiguration $Adapter.guid
        $entry.dnsMode = $(if ($explicit -and [string]$explicit.Value) { 'Static' } else { 'Dhcp' })
        $entry.addresses = $(if ($Kind -eq 'Sharing') { @($result.addresses | Where-Object index -EQ $Adapter.index) } else { @() })
        $entry.addressesObserved = $Kind -eq 'Sharing'
        $result.adapters = @($entry)
    }
    if ($Kind -eq 'Sharing') {
        $manager = New-Object -ComObject HNetCfg.HNetShare
        $result.sharing = @(foreach ($connection in $manager.EnumEveryConnection()) {
                $properties = $manager.NetConnectionProps($connection); $configuration = $manager.INetSharingConfigurationForINetConnection($connection)
                if ($configuration.SharingEnabled) { @{ guid = ([string]$properties.Guid).Trim('{}').ToLowerInvariant(); type = [int]$configuration.SharingConnectionType } }
            })
    }
    $result
}

function Assert-Usb4CurrentAdapter {
    <# .SYNOPSIS
    Revalidate the exact adapter incarnation immediately before index-based writes.
    #>
    param([Parameter(Mandatory)][hashtable]$Expected)
    $current = @(Get-NetAdapter -IncludeHidden -ErrorAction Stop | Where-Object ifIndex -EQ $Expected.index)
    if ($current.Count -ne 1 -or ([string]$current[0].InterfaceGuid).Trim('{}').ToLowerInvariant() -cne $Expected.guid -or
        (ConvertTo-Usb4Mac $current[0].MacAddress) -cne $Expected.mac) { throw 'Managed adapter incarnation changed before mutation.' }
}

function Get-Usb4ObjectProperty {
    <# .SYNOPSIS
    Read known native API fields without treating missing data as success.
    #>
    param([object]$Object, [string[]]$Names)
    foreach ($name in $Names) {
        if ($Object -is [Collections.IDictionary]) { if ($Object.Contains($name)) { return $Object[$name] } }
        elseif ($null -ne $Object -and $Object.PSObject.Properties[$name]) { return $Object.PSObject.Properties[$name].Value }
    }
    $null
}

function Get-Usb4ExistingNatInventory {
    <# .SYNOPSIS
    Read existing NAT configuration and all HNS/allocation reservations without mutation.
    #>
    $nats = @(Get-NetNat -ErrorAction Stop | ForEach-Object {
            $values = @{}
            foreach ($field in @('Name', 'InternalIPInterfaceAddressPrefix', 'ExternalIPInterfaceAddressPrefix', 'Active',
                    'IcmpQueryTimeout', 'TcpEstablishedConnectionTimeout', 'TcpTransientConnectionTimeout', 'TcpFilteringBehavior',
                    'UdpFilteringBehavior', 'UdpIdleSessionTimeout', 'UdpInboundRefresh', 'Store')) {
                $property = $_.PSObject.Properties[$field]
                if (-not $property) { throw "NAT configuration lacks $field." }
                $values[$field] = $property.Value
            }
            $values
        })
    if (-not (Get-Command Get-HnsNetwork -ErrorAction SilentlyContinue) -or -not (Get-Command Get-HnsEndpoint -ErrorAction SilentlyContinue)) {
        $module = Join-Path $env:SystemRoot 'System32/WindowsPowerShell/v1.0/Modules/HostNetworkingService/HostNetworkingService.psd1'
        if (-not (Test-Path -LiteralPath $module -ErrorAction Stop)) { throw 'HNS allocation inventory is unavailable; existing NAT sharing is not admitted.' }
        Import-Module $module -ErrorAction Stop
    }
    $networks = @(foreach ($network in @(Get-HnsNetwork -ErrorAction Stop)) {
            $id = Get-Usb4ObjectProperty $network @('Id', 'ID'); $type = Get-Usb4ObjectProperty $network @('Type')
            if (-not $id -or -not $type) { throw 'HNS network identity/type is unknown.' }
            $prefixes = [Collections.Generic.List[string]]::new()
            foreach ($subnet in @(Get-Usb4ObjectProperty $network @('Subnets'))) {
                if ($null -eq $subnet) { continue }
                $prefix = Get-Usb4ObjectProperty $subnet @('AddressPrefix', 'IpAddressPrefix')
                if ($prefix) { $prefixes.Add([string]$prefix) }
                foreach ($ipSubnet in @(Get-Usb4ObjectProperty $subnet @('IpSubnets'))) {
                    if ($null -eq $ipSubnet) { continue }
                    $prefix = Get-Usb4ObjectProperty $ipSubnet @('IpAddressPrefix', 'AddressPrefix')
                    if (-not $prefix) { throw 'Unknown HNS IP subnet allocation.' }; $prefixes.Add([string]$prefix)
                }
            }
            foreach ($ipam in @(Get-Usb4ObjectProperty $network @('Ipams'))) {
                if ($null -eq $ipam) { continue }
                foreach ($subnet in @(Get-Usb4ObjectProperty $ipam @('Subnets'))) {
                    if ($null -eq $subnet) { continue }
                    $prefix = Get-Usb4ObjectProperty $subnet @('IpAddressPrefix', 'AddressPrefix')
                    if (-not $prefix) { throw 'Unknown HNS IPAM allocation.' }; $prefixes.Add([string]$prefix)
                }
            }
            if ($prefixes.Count -eq 0 -and $type -notin @('Transparent', 'L2Bridge', 'L2Tunnel')) { throw 'HNS network has unknown address allocations.' }
            @{ id = [string]$id; type = [string]$type; prefixes = @($prefixes | Sort-Object -Unique) }
        })
    $endpoints = @(foreach ($endpoint in @(Get-HnsEndpoint -ErrorAction Stop)) {
            $id = Get-Usb4ObjectProperty $endpoint @('Id', 'ID'); $addresses = [Collections.Generic.List[string]]::new()
            $ip = Get-Usb4ObjectProperty $endpoint @('IPAddress')
            if ($ip) { $addresses.Add([string]$ip) }
            foreach ($configuration in @(Get-Usb4ObjectProperty $endpoint @('IpConfigurations'))) {
                if ($null -eq $configuration) { continue }
                $ip = Get-Usb4ObjectProperty $configuration @('IpAddress', 'IPAddress')
                if (-not $ip) { throw 'Unknown HNS endpoint IP configuration.' }; $addresses.Add([string]$ip)
            }
            if (-not $id -or $addresses.Count -eq 0) { throw 'HNS endpoint reservation is unknown.' }
            @{ id = [string]$id; addresses = @($addresses | Sort-Object -Unique) }
        })
    $mappings = @(Get-NetNatStaticMapping -ErrorAction Stop | ForEach-Object {
            $address = Get-Usb4ObjectProperty $_ @('InternalIPAddress')
            if (-not $address) { throw 'Static NAT mapping reservation is unknown.' }
            @{ internalAddress = [string]$address; natName = [string](Get-Usb4ObjectProperty $_ @('NatName')); configuration = [string]($_ | Select-Object NatName, StaticMappingID, Protocol, ExternalIPAddress, ExternalPort, InternalIPAddress, InternalPort | ConvertTo-Json -Compress) }
        })
    # Get-NetNatSession has no NatName parameter on supported Windows builds.
    # Inspect every returned reservation; never infer absence from a failed query.
    $sessions = @(Get-NetNatSession -ErrorAction Stop | ForEach-Object {
            $source = Get-Usb4ObjectProperty $_ @('InternalSourceAddress'); $destination = Get-Usb4ObjectProperty $_ @('InternalDestinationAddress')
            if (-not $source -or -not $destination) { throw 'NAT session reservation is unknown.' }
            @{ source = [string]$source; destination = [string]$destination }
        })
    @{ complete = $true; nat = $nats; networks = $networks; endpoints = $endpoints; mappings = $mappings; sessions = $sessions }
}

function Get-Usb4NatConfigurationFingerprint {
    <# .SYNOPSIS
    Bind immutable NAT/HNS configuration while excluding live session counters.
    #>
    param([hashtable]$Inventory)
    $entries = [Collections.Generic.List[string]]::new()
    foreach ($nat in @($Inventory.nat)) {
        $fields = @($nat.Keys | Sort-Object | ForEach-Object { "$($_)=$($nat[$_])" }) -join ';'
        $entries.Add("nat:$fields")
    }
    foreach ($network in @($Inventory.networks)) { $entries.Add("network:$($network.id):$($network.type):$(@($network.prefixes | Sort-Object) -join ',')") }
    foreach ($endpoint in @($Inventory.endpoints)) { $entries.Add("endpoint:$($endpoint.id):$(@($endpoint.addresses | Sort-Object) -join ',')") }
    foreach ($mapping in @($Inventory.mappings)) { $entries.Add("mapping:$($mapping.natName):$($mapping.configuration)") }
    @($entries | Sort-Object) -join '|'
}

function Assert-Usb4ExistingNatReservation {
    <# .SYNOPSIS
    Refuse unknown or occupied child allocations inside one explicitly approved NAT.
    #>
    param([hashtable]$Policy, [hashtable]$Inventory, [hashtable]$Snapshot, [hashtable]$State, [hashtable]$Adapter)
    if (-not $Inventory.complete -or @($Inventory.nat).Count -ne 1) { throw 'A complete single existing NAT inventory is required.' }
    $sharing = $Policy.sharing; $nat = $Inventory.nat[0]
    $approved = $sharing.natPrefix.Split('/'); $actual = ([string]$nat.InternalIPInterfaceAddressPrefix).Split('/')
    if ($approved.Count -ne 2 -or $actual.Count -ne 2) { throw 'Unknown NAT prefix shape.' }
    $parent = Get-Usb4IPv4Network $approved[0] ([int]$approved[1]); $observed = Get-Usb4IPv4Network $actual[0] ([int]$actual[1])
    if ($nat.Name -ine $sharing.natName -or -not [bool]$nat.Active -or $parent.first -ne $observed.first -or $parent.last -ne $observed.last) { throw 'The existing active NAT does not match the approved identity/prefix.' }
    $child = Get-Usb4IPv4Network $sharing.providerAddress $sharing.prefixLength
    $owned = $State.ContainsKey('natReservation') -and $State.natReservation -and $State.natReservation.committed -and
    $State.natReservation.natName -ieq $sharing.natName -and $State.natReservation.parentFirst -eq $parent.first -and
    $State.natReservation.childFirst -eq $child.first -and $State.natReservation.childLast -eq $child.last -and
    $State.natReservation.adapterGuid -ceq $Adapter.guid -and $State.natReservation.peerMac -ceq $Policy.peerMac -and
    $State.natReservation.providerAddress -eq $sharing.providerAddress -and $State.natReservation.consumerAddress -eq $sharing.consumerAddress
    if ($State.natReservation -and $State.natReservation.committed -and -not $owned) { throw 'Changing a committed NAT child reservation requires explicit ownership recovery.' }
    foreach ($network in @($Inventory.networks)) {
        foreach ($prefix in @($network.prefixes)) {
            $parts = $prefix.Split('/'); if ($parts.Count -ne 2) { throw 'Unknown HNS subnet prefix.' }
            $ip = [Net.IPAddress]::Parse($parts[0]); if ($ip.AddressFamily -ne [Net.Sockets.AddressFamily]::InterNetwork) { continue }
            $range = Get-Usb4IPv4Network $parts[0] ([int]$parts[1])
            if ($range.first -le $child.last -and $child.first -le $range.last) { throw 'The sharing child overlaps an HNS subnet/IPAM allocation.' }
        }
    }
    foreach ($endpoint in @($Inventory.endpoints)) {
        foreach ($address in @($endpoint.addresses)) {
            $ip = [Net.IPAddress]::Parse($address); if ($ip.AddressFamily -ne [Net.Sockets.AddressFamily]::InterNetwork) { continue }
            $number = ConvertTo-Usb4IPv4Number $address
            if ($number -ge $child.first -and $number -le $child.last) { throw 'The sharing child contains an HNS endpoint.' }
        }
    }
    foreach ($address in @($Snapshot.addresses | Where-Object { $_.address.Contains('.') })) {
        $number = ConvertTo-Usb4IPv4Number $address.address
        $range = Get-Usb4IPv4Network $address.address $address.prefix
        if ($address.index -ne $Adapter.index -and $range.first -le $child.last -and $child.first -le $range.last) { throw 'The sharing child overlaps another host interface subnet.' }
        if ($number -ge $child.first -and $number -le $child.last -and
            (-not $owned -or $address.index -ne $Adapter.index -or $address.address -ne $sharing.providerAddress)) { throw 'The sharing child contains an unowned host address.' }
    }
    foreach ($mapping in @($Inventory.mappings)) {
        $number = ConvertTo-Usb4IPv4Number $mapping.internalAddress
        if ($number -ge $child.first -and $number -le $child.last) { throw 'The sharing child conflicts with a static NAT mapping.' }
    }
    foreach ($session in @($Inventory.sessions)) {
        $source = ConvertTo-Usb4IPv4Number $session.source; $destination = ConvertTo-Usb4IPv4Number $session.destination
        if (($source -ge $child.first -and $source -le $child.last -and (-not $owned -or $session.source -ne $sharing.consumerAddress)) -or
            ($destination -ge $child.first -and $destination -le $child.last)) { throw 'The sharing child conflicts with an unowned NAT session.' }
    }
    $octets = [Net.IPAddress]::Parse($sharing.providerAddress).GetAddressBytes()
    $childCidr = '{0}.{1}.{2}.{3}/30' -f $octets[0], $octets[1], $octets[2], ($octets[3] -band 252)
    @{ natName = [string]$nat.Name; parentFirst = $parent.first; parentLast = $parent.last; childFirst = $child.first; childLast = $child.last; childCidr = $childCidr;
        adapterGuid = $Adapter.guid; peerMac = $Policy.peerMac; providerAddress = $sharing.providerAddress; consumerAddress = $sharing.consumerAddress;
        configurationFingerprint = Get-Usb4NatConfigurationFingerprint $Inventory; committed = [bool]$owned; externalPrefix = $nat.ExternalIPInterfaceAddressPrefix
    }
}

function Invoke-Usb4NetworkNativeOperation {
    <# .SYNOPSIS
    Execute explicit external I/O while leaving reconciliation logic independently testable.
    #>
    param([string]$Name, [hashtable]$Arguments)
    if ($Name -in @('AddAddress', 'RemoveAddress', 'AddRoute', 'RemoveRoute', 'SetDns', 'SetForwarding')) {
        if (-not $Arguments.ContainsKey('expectedAdapter')) { throw 'Index-based mutation lacks an enrolled adapter precondition.' }
        Assert-Usb4CurrentAdapter $Arguments.expectedAdapter
    }
    switch ($Name) {
        'Snapshot' { return Get-Usb4NetworkSnapshot -SkipNat:($Arguments.ContainsKey('skipNat') -and $Arguments.skipNat) }
        'ReadExistingNat' { return Get-Usb4ExistingNatInventory }
        'ReadAddressState' { return Get-Usb4NetworkReadback -Kind Address -Adapter $Arguments.adapter }
        'ReadRouteState' { return Get-Usb4NetworkReadback -Kind Route -Adapter $Arguments.adapter }
        'ReadAdapterState' { return Get-Usb4NetworkReadback -Kind Adapter -Adapter $Arguments.adapter }
        'ReadSharingState' { return Get-Usb4NetworkReadback -Kind Sharing -Adapter $Arguments.adapter }
        'ProbePeer' {
            $adapter = $Arguments.adapter; $address = "$($Arguments.address)%$($adapter.index)"
            $ping = [Net.NetworkInformation.Ping]::new()
            try {
                $reply = $ping.Send([Net.IPAddress]::Parse($address), 1500)
                if ($reply.Status -ne [Net.NetworkInformation.IPStatus]::Success) { return @{ success = $false; error = "Scoped peer ICMP: $($reply.Status)" } }
                $neighbors = @(Get-NetNeighbor -InterfaceIndex $adapter.index -AddressFamily IPv6 -ErrorAction Stop | Where-Object { ($_.IPAddress -split '%', 2)[0] -eq $Arguments.address })
                if ($neighbors.Count -ne 1) { return @{ success = $false; error = 'Scoped peer has no unique fresh neighbor record.' } }
                return @{ success = $true; interfaceIndex = $adapter.index; peerMac = ConvertTo-Usb4Mac $neighbors[0].LinkLayerAddress;
                    peerAddress = $address; roundTripMilliseconds = $reply.RoundtripTime; identity = 'PinnedPhysicalLink'; authentication = 'not-cryptographic'
                }
            }
            catch { return @{ success = $false; error = $_.Exception.Message } } finally { $ping.Dispose() }
        }
        'ProbeTransport' {
            $adapter = $Arguments.adapter; $best = @(Get-Usb4SelectedIPv4Route $Arguments.peerIPv4)
            if ($best.Count -ne 1 -or $best[0].index -ne $adapter.index -or $best[0].nextHop -ne '0.0.0.0') {
                return @{ success = $false; error = 'IPv4 peer route does not uniquely select the enrolled USB4 link.' }
            }
            $ping = [Net.NetworkInformation.Ping]::new()
            try {
                $reply = $ping.Send($Arguments.peerIPv4, 1500)
                $neighbors = @(Get-NetNeighbor -InterfaceIndex $adapter.index -AddressFamily IPv4 -ErrorAction Stop | Where-Object IPAddress -EQ $Arguments.peerIPv4)
                if ($reply.Status -ne [Net.NetworkInformation.IPStatus]::Success -or $neighbors.Count -ne 1 -or (ConvertTo-Usb4Mac $neighbors[0].LinkLayerAddress) -cne $Arguments.peerMac) { return @{ success = $false; error = 'IPv4 peer echo/neighbor did not match the enrolled physical peer.' } }
                return @{ success = $true; requestedSource = $Arguments.localIPv4; peerAddress = $Arguments.peerIPv4; interfaceIndex = $adapter.index; route = $best[0]; peerMac = $Arguments.peerMac; roundTripMilliseconds = $reply.RoundtripTime }
            }
            catch { return @{ success = $false; error = $_.Exception.Message } } finally { $ping.Dispose() }
        }
        'AddAddress' {
            $skip = $Arguments.ContainsKey('skipAsSource') -and [bool]$Arguments.skipAsSource
            New-NetIPAddress -InterfaceIndex $Arguments.index -IPAddress $Arguments.address -PrefixLength $Arguments.prefix -SkipAsSource:$skip -PolicyStore ActiveStore -ErrorAction Stop | Out-Null
            return
        }
        'RemoveAddress' {
            $found = @(Get-NetIPAddress -PolicyStore ActiveStore -ErrorAction Stop | Where-Object { $_.InterfaceIndex -eq $Arguments.index -and $_.IPAddress -eq $Arguments.address })
            if (@($found | Where-Object PrefixLength -NE $Arguments.prefix).Count -gt 0 -or $found.Count -gt 1) { throw 'Address changed before rollback.' }
            if ($found.Count -eq 1) { Remove-NetIPAddress -InterfaceIndex $Arguments.index -IPAddress $Arguments.address -PrefixLength $Arguments.prefix -PolicyStore ActiveStore -Confirm:$false -ErrorAction Stop }
            return
        }
        'AddRoute' { New-NetRoute -InterfaceIndex $Arguments.index -DestinationPrefix $Arguments.destination -NextHop $Arguments.nextHop -RouteMetric $Arguments.metric -PolicyStore ActiveStore -ErrorAction Stop | Out-Null; return }
        'RemoveRoute' {
            $found = @(Get-NetRoute -PolicyStore ActiveStore -ErrorAction Stop | Where-Object { $_.InterfaceIndex -eq $Arguments.index -and $_.DestinationPrefix -eq $Arguments.destination -and $_.NextHop -eq $Arguments.nextHop })
            if (@($found | Where-Object RouteMetric -NE $Arguments.metric).Count -gt 0 -or $found.Count -gt 1) { throw 'Route changed before rollback.' }
            if ($found.Count -eq 1) { Remove-NetRoute -InterfaceIndex $Arguments.index -DestinationPrefix $Arguments.destination -NextHop $Arguments.nextHop -RouteMetric $Arguments.metric -PolicyStore ActiveStore -Confirm:$false -ErrorAction Stop }
            return
        }
        'SetDns' {
            $inputObjects = @(Get-DnsClientServerAddress -InterfaceIndex $Arguments.index -AddressFamily IPv4 -ErrorAction Stop)
            if ($inputObjects.Count -ne 1) { throw 'No unique IPv4 DNS input object exists.' }
            if ($Arguments.ContainsKey('mode') -and $Arguments.mode -eq 'Dhcp') { $inputObjects[0] | Set-DnsClientServerAddress -ResetServerAddresses -ErrorAction Stop }
            elseif (@($Arguments.servers).Count -gt 0) { $inputObjects[0] | Set-DnsClientServerAddress -ServerAddresses $Arguments.servers -ErrorAction Stop }
            else { throw 'Static IPv4 DNS requires explicit servers.' }
            return
        }
        'SetForwarding' {
            $current = @(Get-NetIPInterface -InterfaceIndex $Arguments.index -AddressFamily IPv4 -PolicyStore ActiveStore -ErrorAction Stop)
            if ($current.Count -eq 1 -and $Arguments.ContainsKey('allowAlreadyDesired') -and $Arguments.allowAlreadyDesired -and [string]$current[0].Forwarding -ceq $Arguments.mode) { return }
            if ($current.Count -ne 1 -or [string]$current[0].Forwarding -cne $Arguments.expectedCurrent -or $Arguments.mode -notin @('Enabled', 'Disabled')) { throw 'IPv4 forwarding ownership changed before mutation.' }
            Set-NetIPInterface -InterfaceIndex $Arguments.index -AddressFamily IPv4 -Forwarding $Arguments.mode -PolicyStore ActiveStore -ErrorAction Stop
            $after = @(Get-NetIPInterface -InterfaceIndex $Arguments.index -AddressFamily IPv4 -PolicyStore ActiveStore -ErrorAction Stop)
            if ($after.Count -ne 1 -or [string]$after[0].Forwarding -cne $Arguments.mode) { throw 'IPv4 forwarding did not match its exact readback.' }
            return
        }
        'SetSharing' { Set-Usb4IcsPair -PublicGuid $Arguments.publicGuid -PrivateGuid $Arguments.privateGuid -Expected $Arguments.expected -Confirm:$false; return }
        'RestoreDhcp' {
            $matching = @(Get-NetAdapter -IncludeHidden -ErrorAction Stop | Where-Object { ([string]$_.InterfaceGuid).Trim('{}').ToLowerInvariant() -eq $Arguments.guid -and $_.ifIndex -eq $Arguments.index })
            if ($matching.Count -ne 1 -or $Arguments.mode -notin @('Enabled', 'Disabled')) { throw 'DHCP restoration requires the exact original adapter and known mode.' }
            Set-NetIPInterface -InterfaceIndex $Arguments.index -AddressFamily IPv4 -Dhcp $Arguments.mode -ErrorAction Stop
            return
        }
        'ProbeInternet' {
            $best = @(Get-Usb4SelectedIPv4Route $Arguments.remote)
            if (-not $Arguments.ContainsKey('index') -or $best.Count -ne 1 -or $best[0].index -ne $Arguments.index) { return @{ success = $false; source = $Arguments.source; error = 'Internet probe egress does not uniquely select the approved interface.' } }
            $result = Invoke-Usb4BoundTcpProbe -SourceAddress $Arguments.source -RemoteAddress $Arguments.remote -Port $Arguments.port
            $result.route = $best[0]; return $result
        }
        'ProbeDns' { return Invoke-Usb4BoundDnsProbe -SourceAddress $Arguments.source -ServerAddress $Arguments.server }
        default { throw "Unsupported USB4 external operation: $Name" }
    }
}

function Remove-Usb4OwnedConsumerInternet {
    <# .SYNOPSIS
    Withdraw only a previously owned default and unchanged owned DNS settings.
    #>
    [CmdletBinding(SupportsShouldProcess)]
    param([hashtable]$State, [hashtable]$Snapshot, [hashtable]$Adapter, [scriptblock]$Operation, [switch]$Apply)
    if ($State.interfaceGuid -cne $Adapter.guid) { throw 'Consumer cleanup cannot cross an adapter incarnation.' }
    $routes = @($State.ownedRoutes | Where-Object { $_.destination -eq '0.0.0.0/0' -or ($_.destination -match '^\d+\.\d+\.\d+\.\d+/32$' -and $_.nextHop -ne '0.0.0.0') })
    foreach ($owned in $routes) {
        $now = @($Snapshot.routes | Where-Object { $_.index -eq $Adapter.index -and $_.destination -eq $owned.destination })
        if (@($now | Where-Object { $_.nextHop -ne $owned.nextHop -or $_.metric -ne $owned.metric }).Count -or $now.Count -gt 1) { throw 'Consumer default changed; refusing to remove foreign state.' }
    }
    $currentDns = @($Adapter.dns)
    $dnsChanged = $State.dns -and ($currentDns -join '|') -cne (@($State.dns.applied) -join '|')
    if ($Apply -and $PSCmdlet.ShouldProcess($Adapter.guid, 'Withdraw owned consumer routes and restore owned IPv4 DNS')) {
        foreach ($owned in $routes) { & $Operation -Name 'RemoveRoute' -Arguments @{ index = $Adapter.index; destination = $owned.destination; nextHop = $owned.nextHop; metric = $owned.metric } | Out-Null }
        $State.ownedRoutes = @($State.ownedRoutes | Where-Object { $_ -notin $routes })
        if ($dnsChanged) { throw 'Owned routes were withdrawn, but foreign DNS was preserved unchanged.' }
        if ($State.dns) { & $Operation -Name 'SetDns' -Arguments @{ index = $Adapter.index; servers = @($State.dns.before); mode = $State.dns.beforeMode } | Out-Null; $State.dns = $null }
    }
}

function Invoke-Usb4ForwardingRestore {
    <# .SYNOPSIS
    Restore only exact forwarding transitions owned by this link policy.
    #>
    param([hashtable]$State, [hashtable]$Snapshot, [scriptblock]$Operation, [switch]$Apply)
    foreach ($owned in @($State.ownedForwarding)) {
        $adapters = @($Snapshot.adapters | Where-Object { $_.guid -ceq $owned.guid -and $_.mac -ceq $owned.mac })
        if ($adapters.Count -ne 1) { throw 'An owned forwarding adapter is absent or changed; custody retained.' }
        $readback = & $Operation -Name 'ReadAdapterState' -Arguments @{ adapter = $adapters[0] }
        $adapter = @($readback.adapters | Where-Object guid -EQ $owned.guid)[0]
        if ($adapter.forwarding -notin @($owned.before, $owned.applied)) { throw 'IPv4 forwarding has foreign ownership changes.' }
        if ($Apply) {
            if ($adapter.forwarding -ne $owned.before) {
                & $Operation -Name 'SetForwarding' -Arguments @{ index = $adapter.index; expectedAdapter = @{ index = $adapter.index; guid = $adapter.guid; mac = $adapter.mac };
                    expectedCurrent = $owned.applied; mode = $owned.before; allowAlreadyDesired = $true
                } | Out-Null
            }
            $State.ownedForwarding = @($State.ownedForwarding | Where-Object guid -NE $owned.guid)
        }
    }
}

function Get-Usb4UnrelatedNetworkFingerprint {
    <# .SYNOPSIS
    Bind all other interfaces' addressing, routes, and DNS to the preimage.
    #>
    param([hashtable]$Snapshot, [int]$ManagedIndex, [string[]]$AllowedForwardingGuids = @(), [switch]$AllowManagedAutomaticDnsFallback)
    $items = [Collections.Generic.List[string]]::new()
    foreach ($address in @($Snapshot.addresses | Where-Object index -NE $ManagedIndex)) { $items.Add("ip:$($address.index):$($address.address)/$($address.prefix):$($address.prefixOrigin):$($address.suffixOrigin):$($address.skipAsSource)") }
    foreach ($route in @($Snapshot.routes | Where-Object index -NE $ManagedIndex)) { $items.Add("route:$($route.index):$($route.destination):$($route.nextHop):$($route.metric)") }
    foreach ($adapter in @($Snapshot.adapters)) {
        $dns6 = @($adapter.dns6) -join ','
        if ($adapter.ContainsKey('dns6Configuration') -and $adapter.dns6Configuration.observed) {
            $configuration = $adapter.dns6Configuration
            $items.Add("dns6Configuration:$($adapter.guid):$($configuration.nameServer):$($configuration.dhcpServers)")
            # Windows suppresses this exact automatic fallback when IPv4 DNS is set.
            # Retain raw observations and exact rollback comparisons; only the managed
            # interface's positively unconfigured fallback is semantically equivalent.
            $fallback = @($adapter.dns6 | Sort-Object) -join ','
            if ($AllowManagedAutomaticDnsFallback -and $adapter.index -eq $ManagedIndex -and -not $configuration.nameServer -and -not $configuration.dhcpServers -and
                ($fallback -eq '' -or $fallback -ceq 'fec0:0:0:ffff::1,fec0:0:0:ffff::2,fec0:0:0:ffff::3')) { $dns6 = 'AutomaticUnconfigured' }
        }
        $items.Add("dns6:$($adapter.guid):$dns6")
        $items.Add("readOnly:$($adapter.guid):$($adapter.metric):$($adapter.mtu):$($adapter.weakHostSend):$($adapter.weakHostReceive)")
        if ($adapter.index -ne $ManagedIndex -and $adapter.guid -notin $AllowedForwardingGuids) { $items.Add("forwarding:$($adapter.guid):$(if ($adapter.ContainsKey('forwarding')) { $adapter.forwarding } else { 'Unknown' })") }
        if ($adapter.index -ne $ManagedIndex) { $items.Add("dns:$($adapter.guid):$($adapter.dnsMode):$(@($adapter.dns) -join ',')"); $items.Add("dhcp:$($adapter.guid):$($adapter.dhcp)") }
    }
    foreach ($nat in @($Snapshot.nat)) { $items.Add("nat:$($nat.name):$($nat.prefix)") }
    @($items | Sort-Object) -join '|'
}

function Get-Usb4NetworkFingerprint {
    <# .SYNOPSIS
    Bind the complete managed and unrelated preimage for rollback admission.
    #>
    param([hashtable]$Snapshot)
    (Get-Usb4UnrelatedNetworkFingerprint $Snapshot -1) + '|ics:' + (@($Snapshot.sharing | Sort-Object guid, type | ForEach-Object { "$($_.guid):$($_.type)" }) -join '|')
}

function Invoke-Usb4NetworkReconcile {
    <# .SYNOPSIS
    Apply a source-bound peer/sharing plan with exact ownership and rollback.
    #>
    param([Parameter(Mandatory)][hashtable]$Policy, [Parameter(Mandatory)][hashtable]$Snapshot,
        [hashtable]$PreviousState = @{}, [scriptblock]$Operation = ${function:Invoke-Usb4NetworkNativeOperation},
        [scriptblock]$StateCheckpoint, [scriptblock]$ReportCheckpoint, [switch]$Apply)
    $attemptClock = [Diagnostics.Stopwatch]::StartNew()
    $p = Assert-Usb4NetworkPolicy -Policy $Policy -CurrentHost $Snapshot.hostName
    $backend = $(if ($p.mode -eq 'Peer') { 'PeerOnly' } else { $p.sharing.backend })
    $operationInvoker = $Operation
    $reportPublisher = $ReportCheckpoint
    $state = $(if ($PreviousState.Count) { Copy-Usb4NetworkValue $PreviousState } else {
            @{ schemaVersion = 1; hostName = $p.hostName; mode = $p.mode; backend = $backend; adapterMac = $p.adapterMac; interfaceGuid = ''; ownedAddresses = @(); ownedRoutes = @(); ownedForwarding = @(); natReservation = $null; dns = $null; ics = $null }
        })
    if ($state.schemaVersion -ne 1 -or $state.hostName -ine $p.hostName -or $state.adapterMac -cne $p.adapterMac) { throw 'State belongs to another host or adapter policy.' }
    if ($state.mode -cne $p.mode) { throw 'Changing sharing roles requires an explicit owned-state transition.' }
    if (-not $state.ContainsKey('ownedForwarding')) { $state.ownedForwarding = @() }
    if (-not $state.ContainsKey('natReservation')) { $state.natReservation = $null }
    if (-not $state.ContainsKey('backend')) { $state.backend = $(if ($p.mode -eq 'Peer') { 'PeerOnly' } else { 'Ics' }) }
    if ($state.backend -cne $backend) {
        if ($state.ics -or $state.ownedForwarding.Count -gt 0 -or $state.natReservation -or $state.dns -or @($state.ownedRoutes | Where-Object destination -EQ '0.0.0.0/0').Count) { throw 'Changing active sharing backend requires explicit ownership recovery.' }
        $state.backend = $backend
    }
    $record = @{ schema = 'pcai.usb4-peer-network/v1'; observedAt = [DateTimeOffset]::UtcNow.ToString('o'); hostName = $p.hostName;
        role = $p.mode; backend = $backend; scope = 'generic-usb4-transport'; linkLocalReady = $false; ipv4Ready = $false; peerReady = $false; internetReady = $false; networkReady = $false;
        sharingConfigured = $false; upstreamInternetReady = $false; peerInternetVerified = $null; qualification = 'Unqualified'; allowedForwardingGuids = @($state.ownedForwarding | ForEach-Object guid);
        applyRequested = [bool]$Apply; exitCode = 2; errors = @(); changes = @(); planned = @(); rollbackErrors = @(); state = $state;
        preservedNat = $(if ($Snapshot.ContainsKey('natObserved') -and -not $Snapshot.natObserved) { $null } else { Copy-Usb4NetworkValue @($Snapshot.nat) }); authentication = 'PinnedPhysicalLink; not cryptographic authentication';
        attemptId = [guid]::NewGuid().ToString('D'); phase = 'Preflight'; reconcileElapsedMilliseconds = 0;
        operationTimings = [Collections.Generic.List[hashtable]]::new(); firstFailure = $null; diagnosticPublicationErrors = @(); lastOperation = $null
    }
    if ($state.ContainsKey('pendingTransaction')) {
        $record.exitCode = 1; $record.phase = 'PendingOwnershipReview'; $record.preservedNat = $null
        $record.pendingTransaction = Copy-Usb4NetworkValue $state.pendingTransaction
        $record.originalAttemptId = $(if ($state.pendingTransaction.ContainsKey('attemptId')) { $state.pendingTransaction.attemptId } else { $null })
        $record.firstFailure = $(if ($state.pendingTransaction.ContainsKey('firstFailure')) { Copy-Usb4NetworkValue $state.pendingTransaction.firstFailure } else { $null })
        $record.originalFailureKnown = $null -ne $record.firstFailure
        $record.errors += 'An interrupted or incompletely rolled-back transaction requires exact preimage review before another apply.'
        if ($record.firstFailure) { $record.errors += "Original failure: $($record.firstFailure.message)" }
        else { $record.errors += 'The interrupted journal does not contain the original failure cause.' }
        $record.reconcileElapsedMilliseconds = $attemptClock.ElapsedMilliseconds
        return $record
    }
    $undo = [Collections.Generic.List[hashtable]]::new()
    $publishFailure = {
        param([Management.Automation.ErrorRecord]$FailureRecord)
        $record.exitCode = 1; $record.networkReady = $false; $record.internetReady = $false; $record.peerReady = $false; $record.ipv4Ready = $false
        $record.sharingConfigured = $false; $record.upstreamInternetReady = $false; $record.peerInternetVerified = $null
        if (-not $record.firstFailure) {
            $record.firstFailure = @{ attemptId = $record.attemptId; observedAt = [DateTimeOffset]::UtcNow.ToString('o');
                phase = $record.phase; operation = $(if ($record.lastOperation) { $record.lastOperation.name } else { $null });
                message = $FailureRecord.Exception.Message; errorId = $FailureRecord.FullyQualifiedErrorId;
                scriptStackTrace = $FailureRecord.ScriptStackTrace; elapsedMilliseconds = $attemptClock.ElapsedMilliseconds
            }
        }
        if ($Apply -and $state.ContainsKey('pendingTransaction')) {
            if (-not $state.pendingTransaction.ContainsKey('firstFailure')) { $state.pendingTransaction.firstFailure = Copy-Usb4NetworkValue $record.firstFailure }
            $state.pendingTransaction.phase = 'FailureRecordedBeforeRollback'
            if ($StateCheckpoint) {
                try { & $StateCheckpoint $state | Out-Null } catch { $record.diagnosticPublicationErrors += $_.Exception.Message }
            }
        }
        if ($Apply -and $reportPublisher) {
            $record.reconcileElapsedMilliseconds = $attemptClock.ElapsedMilliseconds
            try { & $reportPublisher $record | Out-Null } catch { $record.diagnosticPublicationErrors += $_.Exception.Message }
        }
    }
    $call = {
        param([string]$Name, [hashtable]$Arguments)
        if ($Name -eq 'Snapshot' -and $p.mode -eq 'Consumer' -and $backend -eq 'ExistingWinNat') { $Arguments.skipNat = $true }
        if ($Name -in @('AddAddress', 'RemoveAddress', 'AddRoute', 'RemoveRoute', 'SetDns') -or ($Name -eq 'SetForwarding' -and -not $Arguments.ContainsKey('expectedAdapter'))) {
            $Arguments.expectedAdapter = @{ index = $record.adapter.index; guid = $record.adapter.guid; mac = $record.adapter.mac }
        }
        $operationClock = [Diagnostics.Stopwatch]::StartNew()
        $entry = @{ name = $Name; phase = $record.phase; startedAt = [DateTimeOffset]::UtcNow.ToString('o');
            startedAfterMilliseconds = $attemptClock.ElapsedMilliseconds; completed = $false; elapsedMilliseconds = $null;
            checkpointMilliseconds = 0; ioMilliseconds = $null; error = $null; arguments = @{}
        }
        foreach ($key in @('index', 'address', 'prefix', 'destination', 'nextHop', 'metric', 'source', 'remote', 'port', 'publicGuid', 'privateGuid', 'mode', 'expectedCurrent')) {
            if ($Arguments.ContainsKey($key)) { $entry.arguments[$key] = $Arguments[$key] }
        }
        $record.lastOperation = $entry
        if ($Name -eq 'SetForwarding' -and $record.phase -ne 'Rollback' -and $Arguments.mode -ne $Arguments.expectedCurrent) {
            $inverse = Copy-Usb4NetworkValue $Arguments
            $inverse.mode = $Arguments.expectedCurrent; $inverse.expectedCurrent = $Arguments.mode; $inverse.allowAlreadyDesired = $true
            $undo.Add(@{ name = 'SetForwarding'; arguments = $inverse })
        }
        $ioClock = $null
        try {
            if ($Apply -and $state.ContainsKey('pendingTransaction')) {
                $state.pendingTransaction.lastOperation = Copy-Usb4NetworkValue $entry
                if ($StateCheckpoint) { & $StateCheckpoint $state | Out-Null }
            }
            $entry.checkpointMilliseconds = $operationClock.ElapsedMilliseconds
            $ioClock = [Diagnostics.Stopwatch]::StartNew()
            $value = & $operationInvoker -Name $Name -Arguments $Arguments
            $entry.ioMilliseconds = $ioClock.ElapsedMilliseconds
            $entry.completed = $true
            return $value
        }
        catch {
            $entry.error = $_.Exception.Message
            if ($ioClock) { $entry.ioMilliseconds = $ioClock.ElapsedMilliseconds }
            $entry.elapsedMilliseconds = $operationClock.ElapsedMilliseconds
            & $publishFailure $_
            $_.Exception.Data['Usb4Attempt'] = Copy-Usb4NetworkValue $record
            throw
        }
        finally {
            $entry.elapsedMilliseconds = $operationClock.ElapsedMilliseconds
            $record.operationTimings.Add($entry)
        }
    }
    $ensureTransaction = {
        if ($Apply -and -not $state.ContainsKey('pendingTransaction')) {
            $state.pendingTransaction = @{ observedAt = $record.observedAt; attemptId = $record.attemptId; phase = $record.phase;
                policy = Copy-Usb4NetworkValue $p; preimage = Copy-Usb4NetworkValue $Snapshot
            }
            if ($StateCheckpoint) { & $StateCheckpoint $state | Out-Null }
        }
    }
    $commitTransaction = {
        if ($state.ContainsKey('pendingTransaction')) {
            $committed = Copy-Usb4NetworkValue $state
            $committed.Remove('pendingTransaction')
            $record.phase = 'CommitOwnedState'
            $commitClock = [Diagnostics.Stopwatch]::StartNew()
            $record.lastOperation = @{ name = 'CommitState'; phase = $record.phase; startedAt = [DateTimeOffset]::UtcNow.ToString('o'); completed = $false; elapsedMilliseconds = $null }
            try {
                if ($StateCheckpoint) { & $StateCheckpoint $committed | Out-Null }
                # Retain the in-memory journal until durable commit succeeded.
                # A commit exception must still be able to retain its first cause.
                $state.Remove('pendingTransaction')
                $record.lastOperation.completed = $true
            }
            finally { $record.lastOperation.elapsedMilliseconds = $commitClock.ElapsedMilliseconds }
        }
    }
    $peerLossFallback = {
        param([hashtable]$SelectedAdapter)
        try {
            $record.phase = $(if ($p.mode -eq 'Provider') { 'WithdrawProviderSharing' } else { 'WithdrawConsumerInternet' })
            if ($p.mode -eq 'Provider') {
                if ($backend -eq 'ExistingWinNat') {
                    if ($Apply -and $state.ownedForwarding.Count -gt 0) {
                        & $ensureTransaction
                        Invoke-Usb4ForwardingRestore -State $state -Snapshot $Snapshot -Operation $call -Apply
                        $after = & $call 'Snapshot' @{}
                        if ((Get-Usb4UnrelatedNetworkFingerprint $Snapshot $SelectedAdapter.index $record.allowedForwardingGuids) -cne (Get-Usb4UnrelatedNetworkFingerprint $after $SelectedAdapter.index $record.allowedForwardingGuids)) { throw 'Forwarding withdrawal changed unrelated networking.' }
                        & $commitTransaction; $record.changes += 'RestoreOwnedForwarding'
                    }
                    return
                }
                $pair = @($Snapshot.sharing)
                if (@($pair | Where-Object { -not (($_.guid -eq $state.ics.publicGuid -and $_.type -eq 0) -or ($_.guid -eq $state.ics.privateGuid -and $_.type -eq 1)) }).Count -gt 0 -or $pair.Count -gt 2) { throw 'Provider peer loss found foreign sharing; it was preserved.' }
                if ($Apply) {
                    & $ensureTransaction
                    if ($pair.Count -gt 0) { & $call 'SetSharing' @{ publicGuid = ''; privateGuid = ''; expected = $pair } | Out-Null }
                    $after = & $call 'Snapshot' @{}
                    if ($after.sharing.Count -gt 0 -or (Get-Usb4UnrelatedNetworkFingerprint $Snapshot $SelectedAdapter.index) -cne (Get-Usb4UnrelatedNetworkFingerprint $after $SelectedAdapter.index)) { throw 'Provider withdrawal could not verify sharing removal and unrelated-network preservation.' }
                    $state.ics = $null; & $commitTransaction
                    $record.changes += 'WithdrawOwnedProviderSharing'
                }
                return
            }
            if ($Apply) {
                $state.pendingTransaction = @{ observedAt = $record.observedAt; attemptId = $record.attemptId; phase = 'WithdrawConsumerInternet'; preimage = Copy-Usb4NetworkValue $Snapshot }
                if ($StateCheckpoint) { & $StateCheckpoint $state | Out-Null }
            }
            Remove-Usb4OwnedConsumerInternet -State $state -Snapshot $Snapshot -Adapter $SelectedAdapter -Operation $call -Apply:$Apply -Confirm:$false
            if ($Apply) {
                & $commitTransaction
                $record.changes += 'WithdrawOwnedConsumerInternet'
            }
        }
        catch { & $publishFailure $_; $record.errors += $_.Exception.Message; $record.exitCode = 1 }
    }
    $adapters = @($Snapshot.adapters | Where-Object { $_.mac -eq $p.adapterMac -and $_.description -match 'USB4|Thunderbolt.*(Network|Ethernet)' })
    if ($adapters.Count -eq 1) { $record.adapter = Copy-Usb4NetworkValue $adapters[0] }
    if ($adapters.Count -ne 1 -or -not $adapters[0].up) {
        if ($p.mode -eq 'Consumer' -and $adapters.Count -eq 1 -and $state.interfaceGuid -eq $adapters[0].guid) {
            & $peerLossFallback $adapters[0]
        }
        if ($p.mode -eq 'Provider' -and ($state.ics -or $state.ownedForwarding.Count -gt 0)) { & $peerLossFallback $(if ($adapters.Count -eq 1) { $adapters[0] } else { @{ guid = $state.interfaceGuid; index = -1 } }) }
        $record.errors += 'Enrolled USB4 adapter is absent, disconnected, or ambiguous.'; $record.reconcileElapsedMilliseconds = $attemptClock.ElapsedMilliseconds; return $record
    }
    $adapter = $adapters[0]; $record.adapter = Copy-Usb4NetworkValue $adapter
    if ($state.interfaceGuid -and $state.interfaceGuid -cne $adapter.guid) { throw 'Adapter incarnation changed; existing ownership requires explicit review.' }
    $state.interfaceGuid = $adapter.guid
    $proof = & $call 'ProbePeer' @{ adapter = $adapter; address = $p.peerLinkLocal }
    if (-not $proof.success -or -not $proof.ContainsKey('peerMac') -or $proof.peerMac -cne $p.peerMac -or $proof.interfaceIndex -ne $adapter.index) {
        if ($p.mode -eq 'Consumer' -or ($p.mode -eq 'Provider' -and ($state.ics -or $state.ownedForwarding.Count -gt 0))) { & $peerLossFallback $adapter }
        $record.errors += 'Direct peer probe did not match the enrolled physical peer.'; $record.peerProof = $proof; $record.reconcileElapsedMilliseconds = $attemptClock.ElapsedMilliseconds; return $record
    }
    $record.peerProof = $proof
    $record.linkLocalReady = $true
    # Conflicting sharing ownership is rejected before the first network write.
    $sharing = @($Snapshot.sharing)
    if ($p.mode -eq 'Consumer' -and (@($sharing | Where-Object guid -EQ $adapter.guid).Count -gt 0 -or
            @($Snapshot.addresses | Where-Object { $_.index -eq $adapter.index -and $_.address -eq $p.sharing.providerAddress }).Count -gt 0)) { throw 'Consumer adapter already participates in sharing or owns its proposed gateway address.' }
    if ($p.mode -eq 'Provider' -and $sharing.Count -gt 0) {
        if ($backend -eq 'ExistingWinNat') { throw 'Existing WinNAT policy cannot change or coexist with an active ICS pair.' }
        if (-not $state.ics -or $sharing.Count -ne 2 -or
            @($sharing | Where-Object { $_.type -eq 0 -and $_.guid -eq $state.ics.publicGuid }).Count -ne 1 -or
            @($sharing | Where-Object { $_.type -eq 1 -and $_.guid -eq $adapter.guid }).Count -ne 1) { throw 'Existing internet sharing is not owned by this controller.' }
    }
    $natInventory = $null; $natReservation = $null
    if ($p.mode -eq 'Provider' -and $backend -eq 'ExistingWinNat') {
        $natInventory = & $call 'ReadExistingNat' @{}
        $natReservation = Assert-Usb4ExistingNatReservation -Policy $p -Inventory $natInventory -Snapshot $Snapshot -State $state -Adapter $adapter
        if ($adapter.forwarding -notin @('Enabled', 'Disabled') -or ($adapter.forwarding -eq 'Enabled' -and @($state.ownedForwarding | Where-Object guid -EQ $adapter.guid).Count -ne 1)) { throw 'Peer forwarding is unknown or already enabled without this controller custody.' }
        $record.natReservation = Copy-Usb4NetworkValue $natReservation
    }
    $desired = @(@{ address = $p.localIPv4; prefix = $p.prefixLength })
    if ($p.ContainsKey('localUla')) { $desired += @{ address = $p.localUla; prefix = 128; allowedPrefixes = @(64, 128) } }
    if ($p.mode -eq 'Consumer') { $desired += @{ address = $p.sharing.consumerAddress; prefix = $p.sharing.prefixLength } }
    foreach ($item in $desired) {
        $existing = @($Snapshot.addresses | Where-Object address -EQ $item.address)
        $allowed = $(if ($item.ContainsKey('allowedPrefixes')) { @($item.allowedPrefixes) } else { @($item.prefix) })
        if ($existing.Count -gt 1 -or @($existing | Where-Object { $_.index -ne $adapter.index -or $_.prefix -notin $allowed }).Count) { throw 'Requested address has foreign interface or prefix custody.' }
        if ($item.address.Contains('.')) {
            $wanted = Get-Usb4IPv4Network $item.address $item.prefix
            foreach ($other in @($Snapshot.addresses | Where-Object { $_.index -ne $adapter.index -and $_.address.Contains('.') })) {
                $range = Get-Usb4IPv4Network $other.address $other.prefix
                if ($wanted.first -le $range.last -and $range.first -le $wanted.last) { throw 'Requested subnet overlaps another interface.' }
            }
        }
    }
    $ensureAddress = {
        param([hashtable]$Item)
        $current = & $call 'ReadAddressState' @{ adapter = $adapter }
        $exists = @($current.addresses | Where-Object { $_.address -eq $Item.address })
        $allowed = $(if ($Item.ContainsKey('allowedPrefixes')) { @($Item.allowedPrefixes) } else { @($Item.prefix) })
        if (@($exists | Where-Object { $_.index -ne $adapter.index -or $_.prefix -notin $allowed }).Count -or $exists.Count -gt 1) { throw 'Address custody changed during apply.' }
        if ($exists.Count -eq 0) {
            $operationArgs = @{ index = $adapter.index; address = $Item.address; prefix = $Item.prefix }
            $record.planned += @{ operation = 'AddAddress'; arguments = $operationArgs }
            if ($Apply) {
                & $ensureTransaction
                $undo.Add(@{ name = 'RemoveAddress'; arguments = $operationArgs })
                & $call 'AddAddress' $operationArgs | Out-Null
                $state.ownedAddresses = @($state.ownedAddresses | Where-Object { $_.address -ne $Item.address }) + @(@{ address = $Item.address; prefix = $Item.prefix })
                $record.changes += "AddAddress:$($Item.address)"
            }
        }
    }
    $ensureRoute = {
        param([string]$Destination, [string]$NextHop, [int]$Metric, [bool]$AllowExistingMetric = $false)
        $current = & $call 'ReadRouteState' @{ adapter = $adapter }
        $routes = @($current.routes | Where-Object { $_.destination -eq $Destination -and ($Destination -ne '0.0.0.0/0' -or $_.index -eq $adapter.index) })
        if (@($routes | Where-Object { $_.index -ne $adapter.index -or $_.nextHop -ne $NextHop -or (-not $AllowExistingMetric -and $_.metric -ne $Metric) }).Count) { throw "Foreign route conflicts with $Destination." }
        if ($routes.Count -gt 1) { throw 'Ambiguous existing route.' }
        if ($routes.Count -eq 0) {
            $operationArgs = @{ index = $adapter.index; destination = $Destination; nextHop = $NextHop; metric = $Metric }
            $record.planned += @{ operation = 'AddRoute'; arguments = $operationArgs }
            if ($Apply) {
                & $ensureTransaction
                $undo.Add(@{ name = 'RemoveRoute'; arguments = $operationArgs }); & $call 'AddRoute' $operationArgs | Out-Null
                $state.ownedRoutes = @($state.ownedRoutes | Where-Object destination -NE $Destination) + @(@{ destination = $Destination; nextHop = $NextHop; metric = $Metric })
                $record.changes += "AddRoute:$Destination"
            }
        }
    }
    $withdrawConsumer = {
        $current = & $call 'ReadAdapterState' @{ adapter = $adapter }; $nowAdapter = @($current.adapters | Where-Object guid -EQ $adapter.guid)[0]
        if ($state.dns -or @($state.ownedRoutes | Where-Object { $_.destination -eq '0.0.0.0/0' -or $_.destination -match '^\d+\.\d+\.\d+\.\d+/32$' }).Count) { & $ensureTransaction }
        Remove-Usb4OwnedConsumerInternet -State $state -Snapshot $current -Adapter $nowAdapter -Operation $call -Apply:$Apply -Confirm:$false
        if ($Apply) { $record.changes += 'WithdrawOwnedConsumerInternet' }
    }
    $ensureForwarding = {
        param([hashtable]$Target)
        $current = & $call 'ReadAdapterState' @{ adapter = $Target }
        $matching = @($current.adapters | Where-Object guid -EQ $Target.guid)
        if ($matching.Count -ne 1) { throw 'Forwarding readback did not select the exact approved adapter.' }
        $fresh = $matching[0]
        if ($fresh.forwarding -notin @('Enabled', 'Disabled')) { throw 'IPv4 forwarding state is unknown.' }
        $owned = @($state.ownedForwarding | Where-Object guid -EQ $fresh.guid)
        if ($owned.Count -gt 1 -or ($owned.Count -eq 1 -and ($owned[0].mac -cne $fresh.mac -or $fresh.forwarding -ne $owned[0].applied))) { throw 'Owned forwarding has foreign drift.' }
        if ($fresh.guid -eq $adapter.guid -and $fresh.forwarding -eq 'Enabled' -and $owned.Count -eq 0) { throw 'Peer forwarding was enabled without controller custody after preflight.' }
        if ($fresh.forwarding -eq 'Disabled') {
            $record.planned += @{ operation = 'SetForwarding'; guid = $fresh.guid; mode = 'Enabled' }
            if ($Apply) {
                & $ensureTransaction
                & $call 'SetForwarding' @{ index = $fresh.index; expectedAdapter = @{ index = $fresh.index; guid = $fresh.guid; mac = $fresh.mac }; expectedCurrent = 'Disabled'; mode = 'Enabled' } | Out-Null
                $state.ownedForwarding = @($state.ownedForwarding | Where-Object guid -NE $fresh.guid) + @(@{ guid = $fresh.guid; mac = $fresh.mac; before = 'Disabled'; applied = 'Enabled' })
                $record.allowedForwardingGuids = @($record.allowedForwardingGuids + @($fresh.guid) | Select-Object -Unique)
                $record.changes += "EnableOwnedForwarding:$($fresh.guid)"
            }
        }
    }
    try {
        $record.phase = 'ReconcileTransport'
        foreach ($item in $desired) { & $ensureAddress $item }
        if ($p.ContainsKey('peerUla')) { & $ensureRoute "$($p.peerUla)/128" '::' 10 $true }
        $transport = & $call 'ProbeTransport' @{ adapter = $adapter; localIPv4 = $p.localIPv4; peerIPv4 = $p.peerIPv4; peerMac = $p.peerMac }
        $record.transportProof = $transport
        $record.ipv4Ready = $transport.success -and $transport.interfaceIndex -eq $adapter.index -and $transport.peerAddress -eq $p.peerIPv4 -and $transport.peerMac -eq $p.peerMac
        $record.peerReady = $record.ipv4Ready
        if (-not $record.peerReady) {
            if ($p.mode -eq 'Consumer') { & $withdrawConsumer }
            $record.errors += 'Configured IPv4 transport has not qualified on the enrolled USB4 edge.'
        }
        elseif ($p.mode -eq 'Peer') { $record.networkReady = $true; $record.exitCode = 0 }
        elseif ($p.mode -eq 'Provider') {
            $candidates = @(foreach ($candidate in $Snapshot.adapters) {
                    if (-not $candidate.up -or $candidate.guid -eq $adapter.guid -or -not $candidate.hardware -or
                        $candidate.description -match 'USB4|Thunderbolt|Virtual|VPN|Tunnel|Wintun|Tailscale' -or
                        (-not $candidate.wired -and -not ($p.sharing.allowWirelessFallback -and $candidate.wireless))) { continue }
                    foreach ($route in @($Snapshot.routes | Where-Object { $_.destination -eq '0.0.0.0/0' -and $_.index -eq $candidate.index -and $_.nextHop -ne '0.0.0.0' })) {
                        foreach ($address in @($candidate.addresses | Where-Object { $_.address.Contains('.') -and $_.address -notmatch '^(169\.254|127)\.' })) {
                            @{ adapter = $candidate; address = $address.address; route = $route; score = $route.metric + $candidate.metric + $(if ($candidate.wireless) { 10000 } else { 0 }) }
                        }
                    }
                })
            $upstream = $null
            foreach ($candidate in @($candidates | Sort-Object score)) {
                $probe = & $call 'ProbeInternet' @{ source = $candidate.address; remote = $p.sharing.probeAddress; port = $p.sharing.probePort; index = $candidate.adapter.index }
                if ($probe.success -and $probe.source -eq $candidate.address) { $upstream = $candidate; $record.upstreamProof = $probe; break }
            }
            if (-not $upstream) {
                if ($state.ics -or $state.ownedForwarding.Count -gt 0) { & $peerLossFallback $adapter }
                $record.errors += 'No approved healthy wired upstream is available.'
            }
            else {
                $record.upstream = @{ guid = $upstream.adapter.guid; name = $upstream.adapter.name; source = $upstream.address; gateway = $upstream.route.nextHop }
                if ($backend -eq 'ExistingWinNat') {
                    if ($natReservation.externalPrefix) {
                        $parts = ([string]$natReservation.externalPrefix).Split('/'); if ($parts.Count -ne 2) { throw 'Unknown NAT external prefix constraint.' }
                        $external = Get-Usb4IPv4Network $parts[0] ([int]$parts[1]); $source = ConvertTo-Usb4IPv4Number $upstream.address
                        if ($source -lt $external.first -or $source -gt $external.last) { throw 'Approved wired source is outside the existing NAT external constraint.' }
                    }
                    $record.phase = 'ConfigureExistingNatChild'
                    if ($Apply -and -not $natReservation.committed) {
                        & $ensureTransaction; $state.natReservation = Copy-Usb4NetworkValue $natReservation
                        if ($StateCheckpoint) { & $StateCheckpoint $state | Out-Null }
                    }
                    $keep = @($adapter.guid, $upstream.adapter.guid)
                    if (@($state.ownedForwarding | Where-Object { $_.guid -notin $keep }).Count -gt 0) {
                        if ($Apply) { & $ensureTransaction; Invoke-Usb4ForwardingRestore -State $state -Snapshot $Snapshot -Operation $call -Apply }
                    }
                    $private = @{ address = $p.sharing.providerAddress; prefix = $p.sharing.prefixLength }
                    $desired += $private; & $ensureAddress $private
                    & $ensureForwarding $adapter; & $ensureForwarding $upstream.adapter
                    $record.sharingConfigured = [bool]$Apply -or ($adapter.forwarding -eq 'Enabled' -and $upstream.adapter.forwarding -eq 'Enabled')
                    $record.upstreamInternetReady = $true; $record.internetReady = $false; $record.peerInternetVerified = $null
                    $record.qualification = 'ProviderConfigurationOnly; consumer packet proof required'
                    $record.networkReady = $record.sharingConfigured; $record.exitCode = $(if ($record.sharingConfigured) { 0 } else { 2 })
                }
                else {
                    $target = @(@{ guid = $upstream.adapter.guid; type = 0 }, @{ guid = $adapter.guid; type = 1 })
                    $exact = $sharing.Count -eq 2 -and @($sharing | Where-Object { $_.guid -eq $upstream.adapter.guid -and $_.type -eq 0 }).Count -eq 1 -and @($sharing | Where-Object { $_.guid -eq $adapter.guid -and $_.type -eq 1 }).Count -eq 1
                    if (-not $exact) {
                        $record.planned += @{ operation = 'SetSharing'; publicGuid = $upstream.adapter.guid; privateGuid = $adapter.guid }
                        if ($Apply) {
                            & $ensureTransaction
                            if (@($Snapshot.addresses | Where-Object { $_.index -eq $adapter.index -and $_.address -eq $p.sharing.providerAddress }).Count -eq 0) {
                                $undo.Add(@{ name = 'RemoveAddress'; arguments = @{ index = $adapter.index; address = $p.sharing.providerAddress; prefix = $p.sharing.prefixLength } })
                            }
                            $undo.Add(@{ name = 'SetSharing'; arguments = @{ publicGuid = $(if ($state.ics) { $state.ics.publicGuid } else { '' }); privateGuid = $(if ($state.ics) { $state.ics.privateGuid } else { '' }); expected = $target } })
                            & $call 'SetSharing' @{ publicGuid = $upstream.adapter.guid; privateGuid = $adapter.guid; expected = $sharing } | Out-Null
                            $state.ics = @{ publicGuid = $upstream.adapter.guid; privateGuid = $adapter.guid;
                                privatePreimage = $(if ($state.ics) { $state.ics.privatePreimage } else { Copy-Usb4NetworkValue $adapter })
                            }
                            # ICS can reset private IPv4. Restore preexisting private
                            # addresses without converting them into owned resources.
                            $postIcs = & $call 'ReadSharingState' @{ adapter = $adapter }
                            foreach ($original in @($Snapshot.addresses | Where-Object { $_.index -eq $adapter.index -and $_.address.Contains('.') })) {
                                if (@($postIcs.addresses | Where-Object { $_.index -eq $adapter.index -and $_.address -eq $original.address -and $_.prefix -eq $original.prefix }).Count -eq 0) {
                                    & $call 'AddAddress' @{ index = $adapter.index; address = $original.address; prefix = $original.prefix; skipAsSource = $original.skipAsSource } | Out-Null
                                }
                            }
                            $privateAfter = @($postIcs.adapters | Where-Object guid -EQ $adapter.guid)[0]
                            if ($privateAfter.dnsMode -ne $adapter.dnsMode -or (@($privateAfter.dns) -join '|') -cne (@($adapter.dns) -join '|')) {
                                & $call 'SetDns' @{ index = $adapter.index; servers = @($adapter.dns); mode = $adapter.dnsMode } | Out-Null
                            }
                            foreach ($item in $desired) { & $ensureAddress $item }
                            $after = & $call 'ReadSharingState' @{ adapter = $adapter }
                            if (@($after.sharing | Where-Object { $_.guid -eq $upstream.adapter.guid -and $_.type -eq 0 }).Count -ne 1 -or
                                @($after.sharing | Where-Object { $_.guid -eq $adapter.guid -and $_.type -eq 1 }).Count -ne 1 -or
                                @($after.addresses | Where-Object { $_.index -eq $adapter.index -and $_.address -eq $p.sharing.providerAddress -and $_.prefix -eq $p.sharing.prefixLength }).Count -ne 1) { throw 'ICS pair/private scope did not match the explicit policy after apply.' }
                            $record.changes += 'SetSharing'
                        }
                    }
                    $record.internetReady = $exact -or [bool]$Apply; $record.networkReady = $record.internetReady
                    $record.exitCode = $(if ($record.networkReady) { 0 } else { 2 })
                }
            }
        }
        else {
            $desiredDns = @(if ($backend -eq 'ExistingWinNat') { $p.sharing.dnsServers } else { $p.sharing.providerAddress })
            foreach ($probeTarget in @(@($p.sharing.probeAddress) + $(if ($backend -eq 'ExistingWinNat') { $desiredDns } else { @() }) | Select-Object -Unique)) { & $ensureRoute "$probeTarget/32" $p.sharing.providerAddress 5 }
            $probe = & $call 'ProbeInternet' @{ source = $p.sharing.consumerAddress; remote = $p.sharing.probeAddress; port = $p.sharing.probePort; index = $adapter.index }
            $dnsProbe = & $call 'ProbeDns' @{ source = $p.sharing.consumerAddress; server = $desiredDns[0] }
            $record.internetProof = $probe; $record.dnsProof = $dnsProbe
            if (-not $probe.success -or $probe.source -ne $p.sharing.consumerAddress -or -not $dnsProbe.success) {
                & $withdrawConsumer; $record.errors += 'USB4-source HTTPS or sharing-gateway DNS qualification failed.'
            }
            else {
                $record.internetReady = $true
                $record.peerInternetVerified = $true; $record.sharingConfigured = $true; $record.qualification = 'ConsumerSourceBoundHttpsAndDns'
                if ($p.sharing.preferSharedInternet) {
                    & $ensureRoute '0.0.0.0/0' $p.sharing.providerAddress $p.sharing.routeMetric
                    $current = & $call 'ReadAdapterState' @{ adapter = $adapter }; $nowAdapter = @($current.adapters | Where-Object guid -EQ $adapter.guid)[0]
                    if ($state.dns -and (@($nowAdapter.dns) -join '|') -cne (@($state.dns.applied) -join '|')) { throw 'DNS ownership changed; refusing to overwrite it.' }
                    if ((@($nowAdapter.dns) -join '|') -cne ($desiredDns -join '|')) {
                        $record.planned += @{ operation = 'SetDns'; servers = $desiredDns }
                        if ($Apply) {
                            & $ensureTransaction
                            $before = @($nowAdapter.dns)
                            $undo.Add(@{ name = 'SetDns'; arguments = @{ index = $adapter.index; servers = $before; mode = $nowAdapter.dnsMode } })
                            & $call 'SetDns' @{ index = $adapter.index; servers = $desiredDns; mode = 'Static' } | Out-Null
                            $state.dns = @{ before = $before; beforeMode = $nowAdapter.dnsMode; applied = $desiredDns }; $record.changes += 'SetDns'
                        }
                    }
                }
                elseif ($state.dns -or @($state.ownedRoutes | Where-Object destination -EQ '0.0.0.0/0').Count) { & $withdrawConsumer }
                $record.networkReady = $true; $record.exitCode = 0
            }
        }
        if ($Apply) {
            $record.phase = 'FinalInvariantReadback'
            $after = & $call 'Snapshot' @{}
            $ownedDnsTransition = $p.mode -eq 'Consumer' -and ($state.dns -or ($PreviousState.ContainsKey('dns') -and $PreviousState.dns))
            if ((Get-Usb4UnrelatedNetworkFingerprint $Snapshot $adapter.index $record.allowedForwardingGuids -AllowManagedAutomaticDnsFallback:$ownedDnsTransition) -cne
                (Get-Usb4UnrelatedNetworkFingerprint $after $adapter.index $record.allowedForwardingGuids -AllowManagedAutomaticDnsFallback:$ownedDnsTransition)) { throw 'Unrelated addressing, routes, DNS, or WinNAT changed during reconciliation; refusing readiness.' }
            foreach ($item in $desired) {
                $allowed = $(if ($item.ContainsKey('allowedPrefixes')) { @($item.allowedPrefixes) } else { @($item.prefix) })
                if (@($after.addresses | Where-Object { $_.index -eq $adapter.index -and $_.address -eq $item.address -and $_.prefix -in $allowed }).Count -ne 1) { throw 'Applied address was not present in final readback.' }
            }
            foreach ($ownedRoute in @($state.ownedRoutes)) {
                if (@($after.routes | Where-Object { $_.index -eq $adapter.index -and $_.destination -eq $ownedRoute.destination -and $_.nextHop -eq $ownedRoute.nextHop -and $_.metric -eq $ownedRoute.metric }).Count -ne 1) { throw 'Owned route did not survive final readback.' }
            }
            if ($p.mode -eq 'Provider' -and $backend -eq 'ExistingWinNat') {
                $natAfter = & $call 'ReadExistingNat' @{}
                if ((Get-Usb4NatConfigurationFingerprint $natInventory) -cne (Get-Usb4NatConfigurationFingerprint $natAfter)) { throw 'Existing NAT configuration or HNS allocations changed; no NAT/HNS restoration is authorized.' }
                if ($record.sharingConfigured) {
                    foreach ($guid in @($adapter.guid, $record.upstream.guid)) {
                        if (@($after.adapters | Where-Object { $_.guid -eq $guid -and $_.forwarding -eq 'Enabled' }).Count -ne 1) { throw 'Approved forwarding did not survive exact readback.' }
                    }
                    if (-not $state.natReservation) { throw 'Existing NAT child has no durable reservation identity.' }
                    $state.natReservation.committed = $true
                    $record.natReservation = Copy-Usb4NetworkValue $state.natReservation
                }
            }
            if ($record.internetReady -and $p.mode -eq 'Provider') {
                if ($after.sharing.Count -ne 2 -or @($after.sharing | Where-Object { $_.guid -eq $adapter.guid -and $_.type -eq 1 }).Count -ne 1 -or
                    @($after.sharing | Where-Object { $_.guid -eq $record.upstream.guid -and $_.type -eq 0 }).Count -ne 1 -or
                    @($after.addresses | Where-Object { $_.index -eq $adapter.index -and $_.address -eq $p.sharing.providerAddress -and $_.prefix -eq $p.sharing.prefixLength }).Count -ne 1) { throw 'Final ICS readback no longer matches the qualified provider pair/scope.' }
            }
            if ($record.internetReady -and $p.mode -eq 'Consumer' -and $p.sharing.preferSharedInternet) {
                $finalAdapter = @($after.adapters | Where-Object guid -EQ $adapter.guid)[0]
                if (@($after.routes | Where-Object { $_.index -eq $adapter.index -and $_.destination -eq '0.0.0.0/0' -and $_.nextHop -eq $p.sharing.providerAddress -and $_.metric -eq $p.sharing.routeMetric }).Count -ne 1 -or
                    (@($finalAdapter.dns) -join '|') -cne ($desiredDns -join '|')) { throw 'Consumer default/DNS promotion did not survive final readback.' }
            }
            if ($record.exitCode -ne 1 -and $state.ContainsKey('pendingTransaction')) { & $commitTransaction }
        }
        elseif ($record.planned.Count -gt 0) {
            $record.networkReady = $false; $record.exitCode = 2
        }
    }
    catch {
        & $publishFailure $_
        $record.errors += $_.Exception.Message
        $record.phase = 'Rollback'
        if ($state.ContainsKey('pendingTransaction')) {
            $state.pendingTransaction.phase = 'Rollback'
            if ($StateCheckpoint) {
                try { & $StateCheckpoint $state | Out-Null } catch { $record.diagnosticPublicationErrors += $_.Exception.Message }
            }
        }
        for ($i = $undo.Count - 1; $i -ge 0; $i--) {
            try {
                if ($undo[$i].name -eq 'SetSharing') {
                    $current = & $call 'ReadSharingState' @{ adapter = $adapter }
                    $currentPair = @($current.sharing | Sort-Object guid, type | ForEach-Object { "$($_.guid):$($_.type)" }) -join '|'
                    $beforePair = @($Snapshot.sharing | Sort-Object guid, type | ForEach-Object { "$($_.guid):$($_.type)" }) -join '|'
                    if ($currentPair -ceq $beforePair) { continue }
                }
                & $call $undo[$i].name $undo[$i].arguments | Out-Null
            }
            catch { $record.rollbackErrors += $_.Exception.Message }
        }
        if ($Apply) {
            try {
                $rollbackSnapshot = & $call 'Snapshot' @{}
                $rollbackAdapter = @($rollbackSnapshot.adapters | Where-Object guid -EQ $adapter.guid)
                if ($rollbackAdapter.Count -eq 1 -and $rollbackAdapter[0].dhcp -ne $adapter.dhcp -and $adapter.dhcp -in @('Enabled', 'Disabled')) {
                    & $call 'RestoreDhcp' @{ index = $adapter.index; guid = $adapter.guid; mode = $adapter.dhcp } | Out-Null
                    $rollbackSnapshot = & $call 'Snapshot' @{}
                }
                if ((Get-Usb4NetworkFingerprint $Snapshot) -cne (Get-Usb4NetworkFingerprint $rollbackSnapshot)) { $record.rollbackErrors += 'Complete network preimage was not restored; pending custody retained.' }
                if ($p.mode -eq 'Provider' -and $backend -eq 'ExistingWinNat' -and $natInventory) {
                    $natRollback = & $call 'ReadExistingNat' @{}
                    if ((Get-Usb4NatConfigurationFingerprint $natInventory) -cne (Get-Usb4NatConfigurationFingerprint $natRollback)) { $record.rollbackErrors += 'Existing NAT/HNS configuration changed; foreign configuration was not restored and pending custody is retained.' }
                }
            }
            catch { $record.rollbackErrors += "Rollback readback failed: $($_.Exception.Message)" }
        }
        if ($record.rollbackErrors.Count -eq 0) { $record.state = Copy-Usb4NetworkValue $PreviousState }
        else { $record.state = $state }
        if ($Apply -and $StateCheckpoint) {
            try { & $StateCheckpoint $record.state | Out-Null } catch { $record.rollbackErrors += "Rollback state publication failed: $($_.Exception.Message)" }
        }
        $record.networkReady = $false; $record.internetReady = $false; $record.peerReady = $false; $record.ipv4Ready = $false; $record.exitCode = 1
    }
    $record.phase = $(if ($record.exitCode -eq 0) { 'Complete' } elseif ($record.exitCode -eq 2) { 'PendingReadiness' } else { 'Failed' })
    $record.reconcileElapsedMilliseconds = $attemptClock.ElapsedMilliseconds
    $record
}

function Write-Usb4NetworkJson {
    <# .SYNOPSIS
    Publish JSON atomically without accepting reserved Windows output paths.
    #>
    param([Parameter(Mandatory)][string]$Path, [Parameter(Mandatory)][object]$Value)
    $full = [IO.Path]::GetFullPath($Path); $leaf = [IO.Path]::GetFileName($full).TrimEnd(' ', '.')
    if ($leaf -match '^(\$null|CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(\.|$)' -or -not $leaf) { throw 'Unsafe Windows output leaf.' }
    [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($full)) | Out-Null
    $temporary = $full + '.pending'
    [IO.File]::WriteAllText($temporary, (ConvertTo-Json -InputObject $Value -Depth 40), [Text.UTF8Encoding]::new($false))
    if ($IsWindows) {
        $acl = [Security.AccessControl.FileSecurity]::new()
        $administrators = [Security.Principal.SecurityIdentifier]::new('S-1-5-32-544')
        $system = [Security.Principal.SecurityIdentifier]::new('S-1-5-18')
        $acl.SetOwner($administrators); $acl.SetAccessRuleProtection($true, $false)
        foreach ($identity in @($administrators, $system)) { $acl.AddAccessRule([Security.AccessControl.FileSystemAccessRule]::new($identity, 'FullControl', 'Allow')) }
        Set-Acl -LiteralPath $temporary -AclObject $acl -ErrorAction Stop
    }
    [IO.File]::Move($temporary, $full, $true)
}

function Publish-Usb4NetworkResult {
    <# .SYNOPSIS
    Keep publication failure nonzero rather than declaring an unrecorded success.
    #>
    param([hashtable]$Result, [scriptblock]$StateWriter, [scriptblock]$ReportWriter, [switch]$Apply)
    if ($Apply) {
        try { & $StateWriter $Result.state | Out-Null; & $ReportWriter $Result | Out-Null }
        catch { $Result.errors += "Publication failed: $($_.Exception.Message)"; $Result.networkReady = $false; $Result.exitCode = 1 }
    }
    $Result
}

if ($MyInvocation.InvocationName -eq '.') { return }
if ($ShowHelp) { Get-Help $PSCommandPath -Detailed; return }
if (-not $PolicyPath -or -not $StatePath -or -not $ReportPath) { throw 'Explicit PolicyPath, StatePath, and ReportPath are required.' }
if (@(@($PolicyPath, $StatePath, $ReportPath) | ForEach-Object { [IO.Path]::GetFullPath($_).ToLowerInvariant() } | Select-Object -Unique).Count -ne 3) { throw 'Policy, state, and report paths must be distinct.' }
if ($Apply -and $DryRun) { throw 'Apply and DryRun are mutually exclusive.' }
if (-not $IsWindows) { throw 'Live USB4 network reconciliation requires Windows.' }
if ($Apply -and -not ([Security.Principal.WindowsPrincipal]::new([Security.Principal.WindowsIdentity]::GetCurrent())).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) { throw 'Apply requires an administrator token.' }
$execute = $Apply -and -not $DryRun -and $PSCmdlet.ShouldProcess([Environment]::MachineName, 'Reconcile explicitly enrolled USB4 peer networking')
$mutex = $null; $locked = $false; $previous = @{}; $scriptClock = [Diagnostics.Stopwatch]::StartNew()
$invocationId = [guid]::NewGuid().ToString('D'); $mainPhase = 'AcquireMutex'
try {
    if ($execute) {
        $mutex = [Threading.Mutex]::new($false, 'Global\PcaiUsb4PeerNetwork')
        try { $locked = $mutex.WaitOne(0) } catch [Threading.AbandonedMutexException] { $locked = $true }
        if (-not $locked) { throw 'Another USB4 peer-network reconciliation is active.' }
    }
    $mainPhase = 'ReadPolicyAndState'
    $policy = Get-Content -LiteralPath $PolicyPath -Raw -ErrorAction Stop | ConvertFrom-Json -AsHashtable
    $previous = $(if (Test-Path -LiteralPath $StatePath) { Get-Content -LiteralPath $StatePath -Raw -ErrorAction Stop | ConvertFrom-Json -AsHashtable } else { @{} })
    $mainPhase = 'InitialObservation'
    if ($execute) {
        Write-Usb4NetworkJson -Path $ReportPath -Value @{ schema = 'pcai.usb4-peer-network/v1'; observedAt = [DateTimeOffset]::UtcNow.ToString('o');
            scope = 'generic-usb4-transport'; invocationId = $invocationId; phase = $mainPhase; inProgress = $true;
            networkReady = $false; peerReady = $false; ipv4Ready = $false; internetReady = $false; exitCode = 2
        }
    }
    $snapshotClock = [Diagnostics.Stopwatch]::StartNew()
    if ($previous.ContainsKey('pendingTransaction')) {
        # A blocked journal is not authority to rescan or change networking. Its
        # original attempt/failure is returned without replacing that evidence.
        $snapshot = @{ hostName = [Environment]::MachineName; adapters = @(); addresses = @(); routes = @(); nat = @(); sharing = @() }
    }
    else {
        $skipNat = $policy.ContainsKey('mode') -and $policy.mode -eq 'Consumer' -and $policy.ContainsKey('sharing') -and
        $policy.sharing.ContainsKey('backend') -and $policy.sharing.backend -eq 'ExistingWinNat'
        $snapshot = Get-Usb4NetworkSnapshot -SkipNat:$skipNat
    }
    $snapshotClock.Stop()
    $mainPhase = 'Reconcile'
    $result = Invoke-Usb4NetworkReconcile -Policy $policy -Snapshot $snapshot -PreviousState $previous -Apply:$execute -StateCheckpoint { param($Value); Write-Usb4NetworkJson -Path $StatePath -Value $Value } -ReportCheckpoint { param($Value); Write-Usb4NetworkJson -Path $ReportPath -Value $Value }
    $result.initialSnapshotMilliseconds = $snapshotClock.ElapsedMilliseconds
    $result.invocationId = $invocationId; $result.inProgress = $false
    $result.initialSnapshotSkippedForPending = $previous.ContainsKey('pendingTransaction')
    $result.totalElapsedMilliseconds = $scriptClock.ElapsedMilliseconds
    $result.timingBoundary = 'before_final_publication'
    $result.controllerSha256 = (Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256 -ErrorAction Stop).Hash
    $mainPhase = 'FinalPublication'
    $result = Publish-Usb4NetworkResult -Result $result -Apply:$execute -StateWriter { param($Value); Write-Usb4NetworkJson -Path $StatePath -Value $Value } -ReportWriter { param($Value); Write-Usb4NetworkJson -Path $ReportPath -Value $Value }
    $result | ConvertTo-Json -Depth 40
    exit $result.exitCode
}
catch {
    $failure = $(if ($_.Exception.Data.Contains('Usb4Attempt')) { Copy-Usb4NetworkValue $_.Exception.Data['Usb4Attempt'] } else {
            @{ schema = 'pcai.usb4-peer-network/v1'; observedAt = [DateTimeOffset]::UtcNow.ToString('o');
                scope = 'generic-usb4-transport'; invocationId = $invocationId; phase = $mainPhase; inProgress = $false;
                peerReady = $false; internetReady = $false; networkReady = $false; exitCode = 1; errors = @($_.Exception.Message)
            }
        })
    $failure.networkReady = $false; $failure.exitCode = 1; $failure.totalElapsedMilliseconds = $scriptClock.ElapsedMilliseconds
    if ($execute -and $locked) {
        try { Write-Usb4NetworkJson -Path $ReportPath -Value $failure } catch { $failure.errors += "Failure report publication failed: $($_.Exception.Message)" }
    }
    $failure | ConvertTo-Json -Depth 20
    exit 1
}
finally { if ($locked) { $mutex.ReleaseMutex() }; if ($mutex) { $mutex.Dispose() } }
