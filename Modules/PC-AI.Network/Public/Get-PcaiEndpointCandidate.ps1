#Requires -Version 5.1
function Get-PcaiEndpointCandidate {
    <#
    .SYNOPSIS
        Lists the addresses a peer may currently be reached at, IPv6 link-local included, scoped to live interfaces.

    .DESCRIPTION
        Builds dual-stack endpoint candidates for a peer from independent providers:

          KnownPeer  Config/thunderbolt-peers.json: the peer's short and hash-derived link-local and ULA
                     addresses and its registry IPv4 addresses.
          Eui64      the modified EUI-64 link-local address derived from a MAC (supplied, or the registry's
                     expectedPeerMacs).
          Neighbor   link-local entries in the IPv6 neighbour cache that match a known MAC or a known peer address;
                     -Wake first sends an all-nodes multicast ping so a sleeping cache is filled.
          Dns        A and AAAA answers for -HostName.

        A link-local address means nothing without its interface, so every link-local candidate carries the zone
        (the Windows ifIndex): fe80::...%25. With no interface pinned, one candidate is emitted per LIVE USB4 /
        Thunderbolt / P2P adapter that is Up. The ifIndex is resolved from the adapter table on EVERY call and never
        cached: it changes on replug, another port, a hub change or a driver reinstall, and a stale zone would
        point at the wrong link.

        Identifiers LOCATE a peer; they never authenticate it. MAC spoofing is not a concern of this command: SSH
        host keys and stronger checks authenticate the peer.

        Output is deduplicated by (address, zone) and returned best first (the order Connect-PcaiEndpoint uses).

        Egress proof: a registry ULA or IPv4 address is not tied to an interface by a zone, and a roaming network can
        route the same prefix to an unrelated host. Such a candidate is emitted only when the route Windows would use
        for it leaves over a live USB4/P2P adapter and the route's source address is that adapter's own. The candidate
        then carries SourceAddress, and Connect-PcaiEndpoint binds its socket to it. With no live USB4 adapter, or a
        route through any other interface, the candidate is omitted (see -IncludeRejected). Scoped link-local
        candidates need no such proof: the zone is the interface. DNS answers are returned as given; the caller named
        that host.

    .PARAMETER HostName
        A name or literal address. Also matched against registry peer names, aliases and tbHostAlias.

    .PARAMETER Port
        Recorded on each candidate; Connect-PcaiEndpoint uses it when -Port is not given.

    .PARAMETER PeerId
        Registry peer name, alias, tbHostAlias or node id (nodeIdHex).

    .PARAMETER MacAddress
        The peer's MAC (any of AA-BB-.., aa:bb:.., aabb.., aabb.ccdd.eeff). Used by Eui64 and Neighbor.

    .PARAMETER InterfaceAlias
        Pin to one adapter by alias. Resolved to the live ifIndex on each call.

    .PARAMETER InterfaceIndex
        Pin to one interface index.

    .PARAMETER Provider
        KnownPeer, Eui64, Neighbor, Dns. Default: all four.

    .PARAMETER IncludeIPv4
        Emit IPv4 candidates. Neither -IncludeIPv4 nor -IncludeIPv6 means both; one of them means only that family.

    .PARAMETER IncludeIPv6
        Emit IPv6 candidates.

    .PARAMETER Wake
        Before reading the neighbour cache, send one 'ping -6 -n 1 -w 500 ff02::1%<ifIndex>' per interface.

    .PARAMETER RegistryPath
        Peer registry file. Default: Config/thunderbolt-peers.json, then ~/.machine/thunderbolt-peers.json.

    .PARAMETER InterfaceSource
        Test seam. Script block returning adapters (Name/InterfaceAlias, InterfaceDescription, ifIndex or
        InterfaceIndex, Status, MacAddress). Default: the CIM class MSFT_NetAdapter.

    .PARAMETER RegistrySource
        Test seam. Script block taking the registry path and returning the parsed registry.

    .PARAMETER NeighborSource
        Test seam. Script block taking an interface index and returning neighbour rows
        (IPAddress, LinkLayerAddress, State, InterfaceIndex). Default: the CIM class MSFT_NetNeighbor, IPv6.

    .PARAMETER WakeAction
        Test seam. Script block taking an interface index. Default: the multicast ping above.

    .PARAMETER DnsSource
        Test seam. Script block taking a name and returning [ipaddress[]]. Default: Dns.GetHostAddressesAsync.

    .PARAMETER RouteSource
        Test seam. Script block taking a destination address and returning the route Windows would use (InterfaceIndex,
        SourceAddress). Default: Find-NetRoute.

    .PARAMETER LocalAddressSource
        Test seam. Script block taking an interface index and returning the addresses configured on it.
        Default: the CIM class MSFT_NetIPAddress.

    .PARAMETER IncludeRejected
        Also return registry candidates that failed the egress proof, flagged Admitted = $false with the evidence
        reason 'unverified-egress'. Connect-PcaiEndpoint never dials them.

    .EXAMPLE
        Get-PcaiEndpointCandidate -PeerId dtm-super-nuc -Provider KnownPeer, Eui64, Neighbor -Wake |
            Format-Table Address, Class, Source, NeighborState

    .EXAMPLE
        Get-PcaiEndpointCandidate -MacAddress 00-11-22-33-44-55 -Provider Eui64 -InterfaceAlias 'Ethernet 4'
        # Address : fe80::211:22ff:fe33:4455%25 (zone is whatever ifIndex 'Ethernet 4' has right now)
    #>
    [CmdletBinding()]
    [OutputType([pscustomobject])]
    param(
        [string]$HostName,
        [ValidateRange(1, 65535)][int]$Port,
        [string]$PeerId,
        [string]$MacAddress,
        [string]$InterfaceAlias,
        [int]$InterfaceIndex,
        [ValidateSet('KnownPeer', 'Eui64', 'Neighbor', 'Dns')][string[]]$Provider = @('KnownPeer', 'Eui64', 'Neighbor', 'Dns'),
        [switch]$IncludeIPv4,
        [switch]$IncludeIPv6,
        [switch]$Wake,
        [string]$RegistryPath,
        [scriptblock]$InterfaceSource,
        [scriptblock]$RegistrySource,
        [scriptblock]$NeighborSource,
        [scriptblock]$WakeAction,
        [scriptblock]$DnsSource,
        [scriptblock]$RouteSource,
        [scriptblock]$LocalAddressSource,
        [switch]$IncludeRejected
    )

    if (-not $HostName -and -not $PeerId -and -not $MacAddress) {
        throw [System.ArgumentException]::new('Give at least one of -HostName, -PeerId or -MacAddress so the peer can be located.')
    }
    $wantV6 = ($IncludeIPv6 -or -not $IncludeIPv4)
    $wantV4 = ($IncludeIPv4 -or -not $IncludeIPv6)
    $candidatePort = $null
    if ($PSBoundParameters.ContainsKey('Port')) { $candidatePort = $Port }

    if (-not $InterfaceSource) { $InterfaceSource = { Get-PcaiLiveInterface } }
    if (-not $RegistrySource) { $RegistrySource = { param($Path) Read-PcaiPeerRegistry -Path $Path } }
    if (-not $NeighborSource) { $NeighborSource = { param($InterfaceIndex) Get-PcaiNeighborEntry -InterfaceIndex $InterfaceIndex } }
    if (-not $WakeAction) { $WakeAction = { param($InterfaceIndex) Invoke-PcaiNeighborWake -InterfaceIndex $InterfaceIndex } }
    if (-not $DnsSource) { $DnsSource = { param($Name) Resolve-PcaiDnsAddress -Name $Name } }
    if (-not $RouteSource) { $RouteSource = { param($Address) Find-PcaiRoute -Address $Address } }
    if (-not $LocalAddressSource) { $LocalAddressSource = { param($InterfaceIndex) Get-PcaiLocalAddress -InterfaceIndex $InterfaceIndex } }

    # Identity: the registry peer (if any) and every MAC we may match against.
    $peer = $null
    if ($PeerId -or $HostName) {
        $registry = $null
        try { $registry = & $RegistrySource $RegistryPath } catch { Write-Verbose "Peer registry unavailable: $($_.Exception.Message)" }
        $peer = Resolve-PcaiPeer -Registry $registry -PeerId $PeerId -HostName $HostName
    }
    $peerName = $PeerId
    if ($peer) { $peerName = $peer.Name }
    $knownMacs = New-Object 'System.Collections.Generic.List[string]'
    $suppliedMac = ConvertTo-PcaiMacAddress -Value $MacAddress
    if ($MacAddress -and -not $suppliedMac) { Write-Warning "MacAddress '$MacAddress' is not a usable MAC and is ignored." }
    if ($suppliedMac) { $knownMacs.Add($suppliedMac) }
    if ($peer) { foreach ($m in $peer.MacAddresses) { if (-not $knownMacs.Contains($m)) { $knownMacs.Add($m) } } }

    # Live scope interfaces, read now (never cached across calls). Needed for IPv6 link-local and for the egress proof.
    $scopeInterfaces = @()
    if ($wantV6 -or 'KnownPeer' -in $Provider) {
        $scopeInterfaces = @(Select-PcaiScopeInterface -Source $InterfaceSource -Alias $InterfaceAlias -Index $InterfaceIndex)
    }

    $found = New-Object 'System.Collections.Generic.List[object]'
    $rejected = New-Object 'System.Collections.Generic.List[object]'
    # Admits a routed (not zone-scoped) registry address only with USB4 egress proof; otherwise records the reason.
    $addRouted = {
        param($Ip, $Evidence)
        $proof = Test-PcaiUsb4Egress -IPAddress $Ip -Interface $scopeInterfaces -RouteSource $RouteSource -LocalAddressSource $LocalAddressSource
        if ($proof.Admitted) {
            $iface = @($scopeInterfaces | Where-Object { [int]$_.InterfaceIndex -eq [int]$proof.InterfaceIndex }) | Select-Object -First 1
            $found.Add((New-PcaiEndpointCandidate -IPAddress $Ip -Source 'KnownPeer' -PeerId $peerName -Interface $iface -Port $candidatePort -Evidence "$Evidence; egress verified via ifIndex $($proof.InterfaceIndex)" -SourceAddress $proof.SourceAddress -EgressInterfaceIndex $proof.InterfaceIndex))
        }
        else {
            $rejected.Add((New-PcaiEndpointCandidate -IPAddress $Ip -Source 'KnownPeer' -PeerId $peerName -Port $candidatePort -Evidence "$Evidence; $($proof.Reason)" -Admitted $false))
        }
    }
    $peerLinkLocal = New-Object 'System.Collections.Generic.List[string]'
    if ($peer) { foreach ($a in $peer.LinkLocalAddresses) { $peerLinkLocal.Add(([System.Net.IPAddress]::Parse($a)).ToString()) } }

    # Neighbour rows per interface, read at most once per call.
    $neighborCache = @{}
    $readNeighbors = {
        param($Interface)
        $idx = [int]$Interface.InterfaceIndex
        if (-not $neighborCache.ContainsKey($idx)) {
            $rows = @()
            try { $rows = @(& $NeighborSource $idx) } catch { Write-Verbose "Neighbour read failed on ifIndex ${idx}: $($_.Exception.Message)" }
            $neighborCache[$idx] = $rows
        }
        $neighborCache[$idx]
    }
    $goodStates = @('Reachable', 'Stale', 'Delay', 'Probe', 'Permanent')

    if ($wantV6 -and 'KnownPeer' -in $Provider -and $peer) {
        foreach ($text in $peer.LinkLocalAddresses) {
            $base = [System.Net.IPAddress]::Parse($text)
            foreach ($iface in $scopeInterfaces) {
                $scoped = New-PcaiScopedIPAddress -IPAddress $base -ScopeId $iface.InterfaceIndex
                $found.Add((New-PcaiEndpointCandidate -IPAddress $scoped -Source 'KnownPeer' -PeerId $peerName -Interface $iface -Port $candidatePort -Evidence 'registry link-local'))
            }
        }
        foreach ($text in $peer.UlaAddresses) {
            $ip = [System.Net.IPAddress]::Parse($text)
            & $addRouted $ip 'registry ULA'
        }
    }
    if ($wantV4 -and 'KnownPeer' -in $Provider -and $peer) {
        foreach ($text in $peer.LanAddresses) {
            $ip = $null
            if ([System.Net.IPAddress]::TryParse($text, [ref]$ip) -and $ip.AddressFamily -eq [System.Net.Sockets.AddressFamily]::InterNetwork) {
                & $addRouted $ip 'registry IPv4'
            }
        }
    }

    if ($wantV6 -and 'Eui64' -in $Provider) {
        foreach ($mac in $knownMacs) {
            foreach ($iface in $scopeInterfaces) {
                $derived = ConvertTo-PcaiEui64LinkLocal -MacAddress $mac -ScopeId $iface.InterfaceIndex
                $found.Add((New-PcaiEndpointCandidate -IPAddress $derived -Source 'Eui64' -PeerId $peerName -MacAddress $mac -Interface $iface -Port $candidatePort -Eui64Derived $true -Evidence 'EUI-64 from MAC'))
            }
        }
    }

    if ($wantV6 -and 'Neighbor' -in $Provider) {
        # Without a known MAC or a registry peer there is nothing to match, so nothing is offered: every
        # neighbour on a link is not "the peer".
        if ($knownMacs.Count -gt 0 -or $peerLinkLocal.Count -gt 0) {
            foreach ($iface in $scopeInterfaces) {
                if ($Wake) { try { & $WakeAction ([int]$iface.InterfaceIndex) } catch { Write-Verbose "Wake failed: $($_.Exception.Message)" } }
                foreach ($row in @(& $readNeighbors $iface)) {
                    $rawAddress = (([string](Get-PcaiProperty $row 'IPAddress' '')) -split '%')[0]
                    $parsed = $null
                    if (-not [System.Net.IPAddress]::TryParse($rawAddress, [ref]$parsed)) { continue }
                    if ((Get-PcaiAddressClass -IPAddress $parsed) -ne 'LinkLocal') { continue }
                    $state = [string](Get-PcaiProperty $row 'State' '')
                    if ($state -notin $goodStates) { continue }
                    $rowMac = ConvertTo-PcaiMacAddress -Value ([string](Get-PcaiProperty $row 'LinkLayerAddress' ''))
                    if (-not $rowMac) { continue }
                    $macMatch = $knownMacs.Contains($rowMac)
                    $addressMatch = $peerLinkLocal.Contains($parsed.ToString())
                    if (-not $macMatch -and -not $addressMatch) { continue }
                    $scoped = New-PcaiScopedIPAddress -IPAddress $parsed -ScopeId $iface.InterfaceIndex
                    $derived = ConvertTo-PcaiEui64LinkLocal -MacAddress $rowMac
                    $isEui = ($null -ne $derived -and $derived.ToString() -eq $parsed.ToString())
                    $evidence = "neighbor $state on ifIndex $($iface.InterfaceIndex)"
                    if ($macMatch) { $evidence += '; MAC match' }
                    if ($isEui) { $evidence += '; EUI-64 of that MAC' }
                    $found.Add((New-PcaiEndpointCandidate -IPAddress $scoped -Source 'Neighbor' -PeerId $peerName -MacAddress $rowMac -Interface $iface -Port $candidatePort -MacMatched $macMatch -Eui64Derived $isEui -NeighborState $state -Evidence $evidence))
                }
            }
        }
    }

    if ('Dns' -in $Provider -and $HostName) {
        $literal = $null
        $answers = @()
        if ([System.Net.IPAddress]::TryParse($HostName.Trim().TrimStart('[').TrimEnd(']'), [ref]$literal)) { $answers = @($literal) }
        else { try { $answers = @(& $DnsSource $HostName) } catch { Write-Verbose "DNS lookup failed: $($_.Exception.Message)" } }
        foreach ($answer in $answers) {
            if ($answer -isnot [System.Net.IPAddress]) { continue }
            $isV6 = ($answer.AddressFamily -eq [System.Net.Sockets.AddressFamily]::InterNetworkV6)
            if ($isV6 -and -not $wantV6) { continue }
            if (-not $isV6 -and -not $wantV4) { continue }
            $evidence = 'DNS answer'
            if ($literal) { $evidence = 'literal address' }
            if ($isV6 -and (Get-PcaiAddressClass -IPAddress $answer) -eq 'LinkLocal' -and $answer.ScopeId -le 0) {
                # getaddrinfo drops a zone-less fe80; a literal one gets the live USB4 zones instead.
                if ($literal) {
                    foreach ($iface in $scopeInterfaces) {
                        $scoped = New-PcaiScopedIPAddress -IPAddress $answer -ScopeId $iface.InterfaceIndex
                        $found.Add((New-PcaiEndpointCandidate -IPAddress $scoped -Source 'Dns' -PeerId $peerName -Interface $iface -Port $candidatePort -Evidence 'literal link-local, live zone'))
                    }
                }
                continue
            }
            $found.Add((New-PcaiEndpointCandidate -IPAddress $answer -Source 'Dns' -PeerId $peerName -Port $candidatePort -Evidence $evidence))
        }
    }

    # Annotate scoped link-local candidates with the neighbour state of THEIR interface, so the connector can
    # prefer the interface whose entry is Reachable when one address is valid on several.
    $merged = @(Merge-PcaiEndpointCandidate -Candidate $found.ToArray())
    foreach ($c in $merged) {
        if ($c.Class -ne 'LinkLocal' -or $null -eq $c.InterfaceIndex -or $c.Family -ne 'InterNetworkV6') { continue }
        if ($c.NeighborState) { continue }
        $iface = [pscustomobject]@{ InterfaceIndex = [int]$c.InterfaceIndex }
        foreach ($row in @(& $readNeighbors $iface)) {
            $rawAddress = (([string](Get-PcaiProperty $row 'IPAddress' '')) -split '%')[0]
            $parsed = $null
            if (-not [System.Net.IPAddress]::TryParse($rawAddress, [ref]$parsed)) { continue }
            if ($parsed.ToString() -ne ($c.Address -split '%')[0]) { continue }
            $state = [string](Get-PcaiProperty $row 'State' '')
            if ($state -notin $goodStates) { continue }
            $c.NeighborState = $state
            $rowMac = ConvertTo-PcaiMacAddress -Value ([string](Get-PcaiProperty $row 'LinkLayerAddress' ''))
            if ($rowMac) {
                if (-not $c.MacAddress) { $c.MacAddress = $rowMac }
                if ($knownMacs.Contains($rowMac)) { $c.MacMatched = $true }
            }
            break
        }
    }
    $ordered = @(Get-PcaiConnectionOrder -Candidate $merged)
    if ($IncludeRejected) { $ordered = @($ordered) + @($rejected.ToArray()) }
    return $ordered
}
