#Requires -Version 5.1
<#
.SYNOPSIS
    Private helpers for the dual-stack endpoint commands (Get-PcaiEndpointCandidate, Connect-PcaiEndpoint).

.DESCRIPTION
    Pure address logic (EUI-64, classification, scoping, ordering, merging) plus thin OS
    wrappers that the public commands call through injectable script blocks, so tests never
    touch the adapter table, the neighbour cache, DNS or the network.

    Identifiers LOCATE a peer; they never authenticate it. SSH host keys and stronger checks
    authenticate. Nothing here treats a MAC match as proof of identity.

    Compatibility: Windows PowerShell 5.1 and PowerShell 7. No ternary, no null-coalescing,
    no ForEach-Object -Parallel, and only .NET Framework 4.7.2 APIs. ASCII only: Windows
    PowerShell reads a BOM-less file as ANSI.

    See docs/networking/dual-stack.md.
#>

# Source rank when the same (address, zone) is reported by several providers: lower wins.
$script:PcaiSourceRank = @{ Neighbor = 0; KnownPeer = 1; Eui64 = 2; Dns = 3 }

function Get-PcaiProperty {
    # Optional members are absent, not null, on some inputs; strict mode turns a bare access into an error.
    [CmdletBinding()]
    param($InputObject, [Parameter(Mandatory)][string]$Name, $Default = $null)
    if ($null -eq $InputObject) { return $Default }
    $property = $InputObject.PSObject.Properties[$Name]
    if ($property -and $null -ne $property.Value) { return $property.Value }
    return $Default
}

function ConvertTo-PcaiMacAddress {
    # Returns AA-BB-CC-DD-EE-FF, or $null for anything that is not a usable MAC.
    # Mirrors ConvertTo-FleetMac (Tools/SystemScripts/Networking/Test-FleetPath.ps1:131); also accepts the
    # dotted Cisco form. An all-zero MAC is what an unresolved neighbour entry reports.
    [CmdletBinding()]
    [OutputType([string])]
    param([AllowNull()][AllowEmptyString()][string]$Value)
    if ([string]::IsNullOrWhiteSpace($Value)) { return $null }
    $clean = $Value.Trim() -replace '[-:.]', ''
    if ($clean -notmatch '^[0-9A-Fa-f]{12}$') { return $null }
    $clean = $clean.ToUpperInvariant()
    if ($clean -eq '000000000000') { return $null }
    return ([regex]::Matches($clean, '..') | ForEach-Object { $_.Value }) -join '-'
}

function Get-PcaiAddressClass {
    # LinkLocal / ULA / GUA for IPv6; IPv4Private / IPv4LinkLocal / IPv4Public for IPv4; Loopback for both.
    # IPv4 shared address space (100.64.0.0/10, the tailnet range) counts as IPv4Private: it is not
    # globally routable and is only reachable over a private path.
    [CmdletBinding()]
    [OutputType([string])]
    param([Parameter(Mandatory)][System.Net.IPAddress]$IPAddress)
    if ([System.Net.IPAddress]::IsLoopback($IPAddress)) { return 'Loopback' }
    $b = $IPAddress.GetAddressBytes()
    if ($IPAddress.AddressFamily -eq [System.Net.Sockets.AddressFamily]::InterNetworkV6) {
        if ($b[0] -eq 0xFE -and (($b[1] -band 0xC0) -eq 0x80)) { return 'LinkLocal' }
        # IsIPv6UniqueLocal is not in .NET Framework 4.7.2; fc00::/7 by hand.
        if (($b[0] -band 0xFE) -eq 0xFC) { return 'ULA' }
        return 'GUA'
    }
    if ($b[0] -eq 169 -and $b[1] -eq 254) { return 'IPv4LinkLocal' }
    if ($b[0] -eq 10) { return 'IPv4Private' }
    if ($b[0] -eq 172 -and $b[1] -ge 16 -and $b[1] -le 31) { return 'IPv4Private' }
    if ($b[0] -eq 192 -and $b[1] -eq 168) { return 'IPv4Private' }
    if ($b[0] -eq 100 -and (($b[1] -band 0xC0) -eq 0x40)) { return 'IPv4Private' }
    return 'IPv4Public'
}

function New-PcaiScopedIPAddress {
    # A link-local address is meaningless without its zone, so the ScopeId (the Windows ifIndex) travels
    # with the address object and prints as fe80::...%<ifIndex>.
    [CmdletBinding()]
    [OutputType([System.Net.IPAddress])]
    param([Parameter(Mandatory)][System.Net.IPAddress]$IPAddress, [Parameter(Mandatory)][long]$ScopeId)
    $bytes = $IPAddress.GetAddressBytes()
    return [System.Net.IPAddress]::new($bytes, $ScopeId)
}

function ConvertTo-PcaiEui64LinkLocal {
    # Modified EUI-64 (RFC 4291 appendix A): flip the universal/local bit of the first octet and insert
    # ff:fe in the middle. 00-11-22-33-44-55 becomes fe80::211:22ff:fe33:4455.
    [CmdletBinding()]
    [OutputType([System.Net.IPAddress])]
    param([Parameter(Mandatory)][string]$MacAddress, [long]$ScopeId = 0)
    $mac = ConvertTo-PcaiMacAddress -Value $MacAddress
    if (-not $mac) { return $null }
    $octets = @($mac -split '-' | ForEach-Object { [Convert]::ToByte($_, 16) })
    $bytes = New-Object byte[] 16
    $bytes[0] = 0xFE
    $bytes[1] = 0x80
    $bytes[8] = [byte]($octets[0] -bxor 0x02)
    $bytes[9] = $octets[1]
    $bytes[10] = $octets[2]
    $bytes[11] = 0xFF
    $bytes[12] = 0xFE
    $bytes[13] = $octets[3]
    $bytes[14] = $octets[4]
    $bytes[15] = $octets[5]
    if ($ScopeId -gt 0) { return [System.Net.IPAddress]::new($bytes, $ScopeId) }
    return [System.Net.IPAddress]::new($bytes)
}

function Get-PcaiPeerNodeHextet {
    # Mirrors Get-DeterministicNodeHextets (Modules/PC-AI.Drivers/Public/Get-ThunderboltPeerIdentity.ps1:81-92).
    # Same hash input and formatting so both modules derive the same fe80::d700:<h0>:<h1>:<h2> and ULA.
    # SHA256.Create() because SHA256.HashData does not exist in .NET Framework (Windows PowerShell 5.1).
    # A unit test cross-checks this against the Drivers function (pwsh 7) so the mirror cannot drift silently.
    # Candidate for later consolidation into one shared helper (owner: Drivers).
    [CmdletBinding()]
    [OutputType([string])]
    param([Parameter(Mandatory)][string]$CanonicalName)
    $bytes = [System.Text.Encoding]::UTF8.GetBytes("pc-ai:usb4-peer:v1:$CanonicalName")
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try { $hash = $sha.ComputeHash($bytes) } finally { $sha.Dispose() }
    $h0 = ([int]$hash[0] -shl 8) -bor [int]$hash[1]
    $h1 = ([int]$hash[2] -shl 8) -bor [int]$hash[3]
    $h2 = ([int]$hash[4] -shl 8) -bor [int]$hash[5]
    if ($h0 -eq 0) { $h0 = 1 }
    if ($h1 -eq 0) { $h1 = 1 }
    if ($h2 -eq 0) { $h2 = 1 }
    return ('{0:x4}:{1:x4}:{2:x4}' -f $h0, $h1, $h2)
}

function ConvertTo-PcaiCanonicalPeerName {
    # Mirrors ConvertTo-CanonicalPeerName (Get-ThunderboltPeerIdentity.ps1:50-77): lower case, no domain suffix,
    # no -tb alias suffix.
    [CmdletBinding()]
    [OutputType([string])]
    param([AllowNull()][AllowEmptyString()][string]$Name)
    if ([string]::IsNullOrWhiteSpace($Name)) { return '' }
    $cleaned = $Name.Trim().ToLowerInvariant()
    $cleaned = $cleaned -replace '\.(local|lan|radius\.dtmventures\.com|dtmventures\.com)$', ''
    $cleaned = $cleaned -replace '-tb$', ''
    return $cleaned
}

function Read-PcaiPeerRegistry {
    # Default RegistrySource. Search order mirrors Get-ThunderboltPeerIdentity.ps1:145-148
    # (repo Config, then ~/.machine). First parseable file wins; none is not an error.
    [CmdletBinding()]
    param([string]$Path)
    $candidates = New-Object 'System.Collections.Generic.List[string]'
    if (-not [string]::IsNullOrWhiteSpace($Path)) {
        $candidates.Add($Path)
    }
    else {
        $candidates.Add((Join-Path $script:ConfigPath 'thunderbolt-peers.json'))
        $candidates.Add((Join-Path (Join-Path $HOME '.machine') 'thunderbolt-peers.json'))
    }
    foreach ($candidate in $candidates) {
        if ($candidate -and [System.IO.File]::Exists($candidate)) {
            try { return ([System.IO.File]::ReadAllText($candidate) | ConvertFrom-Json) }
            catch { Write-Verbose "Peer registry '$candidate' is not valid JSON: $($_.Exception.Message)" }
        }
    }
    return $null
}

function Resolve-PcaiPeer {
    # Finds a registry peer by PeerId (name, alias, tbHostAlias or node id) or by HostName (name, alias, tbHostAlias)
    # and returns its known addressing, or $null. Both registry shapes the Drivers module reads are accepted:
    # fleetPeers[] or peers{} keyed by name. The registry's expectedPeerMacs (topologyLinks[]) supply known MACs.
    [CmdletBinding()]
    param($Registry, [string]$PeerId, [string]$HostName)
    if ($null -eq $Registry) { return $null }
    $peers = @()
    $fleet = Get-PcaiProperty $Registry 'fleetPeers'
    $keyed = Get-PcaiProperty $Registry 'peers'
    if ($fleet) { $peers = @($fleet) }
    elseif ($keyed) {
        $peers = @(foreach ($prop in $keyed.PSObject.Properties) {
                $value = $prop.Value
                if ($null -ne $value -and -not $value.PSObject.Properties['name']) { $value | Add-Member -NotePropertyName 'name' -NotePropertyValue $prop.Name -Force }
                $value
            })
    }
    $wanted = New-Object 'System.Collections.Generic.List[string]'
    foreach ($text in @($PeerId, $HostName)) {
        $canonical = ConvertTo-PcaiCanonicalPeerName -Name $text
        if ($canonical) { $wanted.Add($canonical) }
    }
    if ($wanted.Count -eq 0) { return $null }
    $match = $null
    foreach ($peer in $peers) {
        if ($null -eq $peer -or -not $peer.PSObject.Properties['name']) { continue }
        $names = New-Object 'System.Collections.Generic.List[string]'
        $names.Add((ConvertTo-PcaiCanonicalPeerName -Name ([string]$peer.name)))
        foreach ($alias in @(Get-PcaiProperty $peer 'aliases' @())) { $names.Add((ConvertTo-PcaiCanonicalPeerName -Name ([string]$alias))) }
        $tbAlias = Get-PcaiProperty $peer 'tbHostAlias'
        if ($tbAlias) { $names.Add((ConvertTo-PcaiCanonicalPeerName -Name ([string]$tbAlias))) }
        # A node id is only meaningful as an explicit PeerId, never as a host name.
        $nodeId = Get-PcaiProperty $peer 'nodeIdHex'
        $idMatch = ($PeerId -and $nodeId -and ([string]$PeerId).Trim().ToLowerInvariant() -eq ([string]$nodeId).ToLowerInvariant())
        foreach ($w in $wanted) { if ($names.Contains($w)) { $match = $peer; break } }
        if (-not $match -and $idMatch) { $match = $peer }
        if ($match) { break }
    }
    if (-not $match) { return $null }

    $name = ConvertTo-PcaiCanonicalPeerName -Name ([string]$match.name)
    $ulaPrefix = [string](Get-PcaiProperty $Registry 'ulaPrefix' 'fd00:7462:7032')
    $hextets = Get-PcaiPeerNodeHextet -CanonicalName $name
    $shortLinkLocal = Get-PcaiProperty $match 'shortLinkLocal'
    $shortUla = Get-PcaiProperty $match 'shortUla'
    $linkLocal = New-Object 'System.Collections.Generic.List[string]'
    $ula = New-Object 'System.Collections.Generic.List[string]'
    # Short (human-readable) alias first, then the hash-derived one, as Get-ThunderboltPeerIdentity orders them.
    if ($shortLinkLocal) { $linkLocal.Add([string]$shortLinkLocal) }
    $linkLocal.Add("fe80::d700:$hextets")
    if ($shortUla) { $ula.Add([string]$shortUla) }
    $ula.Add("${ulaPrefix}:d700::$hextets")

    $macs = New-Object 'System.Collections.Generic.List[string]'
    foreach ($link in @(Get-PcaiProperty $Registry 'topologyLinks' @())) {
        $expected = Get-PcaiProperty $link 'expectedPeerMacs'
        if ($null -eq $expected) { continue }
        foreach ($entry in $expected.PSObject.Properties) {
            if ((ConvertTo-PcaiCanonicalPeerName -Name $entry.Name) -eq $name) {
                $mac = ConvertTo-PcaiMacAddress -Value ([string]$entry.Value)
                if ($mac -and -not $macs.Contains($mac)) { $macs.Add($mac) }
            }
        }
    }
    [pscustomobject]@{
        Name = $name
        LinkLocalAddresses = $linkLocal.ToArray()
        UlaAddresses = $ula.ToArray()
        LanAddresses = @(Get-PcaiProperty $match 'lanAddresses' @() | ForEach-Object { [string]$_ })
        MacAddresses = $macs.ToArray()
    }
}

function Get-PcaiLiveInterface {
    # Default InterfaceSource: the live adapter table, read on EVERY call. An ifIndex is never remembered: it
    # changes on replug, another port, a hub change or a driver reinstall.
    # Read through CIM (root/StandardCimv2 MSFT_NetAdapter), as Find-ThunderboltPeer does. In a cold process
    # Get-NetAdapter costs about 2 s of CIM start-up and this about 0.4 s, which matters for a reconnect.
    # InterfaceOperationalStatus 1 is Up.
    [CmdletBinding()]
    param()
    try {
        Get-CimInstance -Namespace root/StandardCimv2 -ClassName MSFT_NetAdapter -ErrorAction Stop | ForEach-Object {
            $status = 'Disconnected'
            if ([int]$_.InterfaceOperationalStatus -eq 1) { $status = 'Up' }
            [pscustomobject]@{
                Name = [string]$_.Name
                InterfaceAlias = [string]$_.Name
                InterfaceDescription = [string]$_.InterfaceDescription
                ifIndex = [int]$_.InterfaceIndex
                Status = $status
                MacAddress = [string]$_.PermanentAddress
            }
        }
    }
    catch {
        Write-Verbose "Adapter table read failed: $($_.Exception.Message)"
        @()
    }
}

function Select-PcaiScopeInterface {
    # Returns the interfaces a link-local candidate is scoped to. A link-local address is ambiguous across
    # interfaces, so with nothing pinned this is every LIVE USB4/P2P adapter. The matcher mirrors
    # Find-ThunderboltPeer.ps1:40-48 (description or name contains USB4, Thunderbolt or P2P); Status must be Up.
    # A pinned alias or index overrides the matcher and is looked up in the live table each time.
    [CmdletBinding()]
    param([Parameter(Mandatory)][scriptblock]$Source, [string]$Alias, [int]$Index)
    $all = @(& $Source | ForEach-Object {
            $idx = Get-PcaiProperty $_ 'ifIndex' (Get-PcaiProperty $_ 'InterfaceIndex' 0)
            $name = [string](Get-PcaiProperty $_ 'InterfaceAlias' (Get-PcaiProperty $_ 'Name' ''))
            [pscustomobject]@{
                InterfaceIndex = [int]$idx
                InterfaceAlias = $name
                Description = [string](Get-PcaiProperty $_ 'InterfaceDescription' '')
                Name = [string](Get-PcaiProperty $_ 'Name' $name)
                Status = [string](Get-PcaiProperty $_ 'Status' '')
                MacAddress = [string](Get-PcaiProperty $_ 'MacAddress' '')
            }
        })
    if ($Alias -or $Index -gt 0) {
        $pinned = @($all | Where-Object {
                ((-not $Alias) -or $_.InterfaceAlias -ieq $Alias -or $_.Name -ieq $Alias) -and (($Index -le 0) -or $_.InterfaceIndex -eq $Index)
            })
        if ($pinned.Count -eq 0 -and $Index -gt 0 -and -not $Alias) {
            # Index pinned but absent from the table: trust the caller; the connect will say if it is wrong.
            return @([pscustomobject]@{ InterfaceIndex = $Index; InterfaceAlias = $null; Description = ''; Name = ''; Status = 'Unknown'; MacAddress = '' })
        }
        return @($pinned | Where-Object { $_.Status -eq 'Up' -or $_.Status -eq 'Unknown' })
    }
    return @($all | Where-Object {
            $_.InterfaceIndex -gt 0 -and $_.Status -eq 'Up' -and
            ($_.Description -match 'USB4|Thunderbolt|P2P' -or $_.Name -match 'USB4|Thunderbolt|P2P')
        })
}

function Get-PcaiNeighborEntry {
    # Default NeighborSource: the IPv6 neighbour cache for one interface, read through CIM
    # (root/StandardCimv2 MSFT_NetNeighbor, AddressFamily 23) as Find-ThunderboltPeer.ps1:100-122 does; in a cold
    # process that is about 0.2 s against about 2 s for Get-NetNeighbor. State is the byte the provider reports:
    # 0 Unreachable, 1 Incomplete, 2 Probe, 3 Delay, 4 Stale, 5 Reachable, 6 Permanent. Failure reads as "no rows".
    [CmdletBinding()]
    param([Parameter(Mandatory)][int]$InterfaceIndex)
    $names = @('Unreachable', 'Incomplete', 'Probe', 'Delay', 'Stale', 'Reachable', 'Permanent')
    try {
        Get-CimInstance -Namespace root/StandardCimv2 -ClassName MSFT_NetNeighbor -Filter "InterfaceIndex=$InterfaceIndex AND AddressFamily=23" -ErrorAction Stop | ForEach-Object {
            $code = [int]$_.State
            $state = 'Unknown'
            if ($code -ge 0 -and $code -lt $names.Count) { $state = $names[$code] }
            [pscustomobject]@{
                IPAddress = [string]$_.IPAddress
                LinkLayerAddress = [string]$_.LinkLayerAddress
                State = $state
                InterfaceIndex = [int]$_.InterfaceIndex
            }
        }
    }
    catch { @() }
}

function Invoke-PcaiNeighborWake {
    # Default WakeAction: an all-nodes link-local multicast ping fills the neighbour cache for that interface
    # (Modules/PC-AI.Drivers/Public/Register-ThunderboltPeer.ps1:358-362 does the same with a 200 ms wait).
    # Best effort; the reply content is not used, only the cache it fills.
    [CmdletBinding()]
    param([Parameter(Mandatory)][int]$InterfaceIndex)
    try {
        & ping.exe -6 -n 1 -w 500 "ff02::1%$InterfaceIndex" 2>$null | Out-Null
        if ($LASTEXITCODE -ne 0) { Write-Verbose "Neighbour wake on ifIndex ${InterfaceIndex}: ping exited $LASTEXITCODE (no reply is normal on an idle link)." }
    }
    catch { Write-Verbose "Neighbour wake failed: $($_.Exception.Message)" }
}

function Resolve-PcaiDnsAddress {
    # Default DnsSource: A and AAAA through getaddrinfo, with a deadline (GetHostAddresses itself has none).
    # getaddrinfo drops a zone-less fe80 answer, so link-local never comes from DNS.
    [CmdletBinding()]
    param([Parameter(Mandatory)][string]$Name, [int]$TimeoutMs = 2000)
    try {
        $task = [System.Net.Dns]::GetHostAddressesAsync($Name)
        if ($task.Wait($TimeoutMs)) { return @($task.Result) }
    }
    catch { Write-Verbose "DNS lookup of '$Name' failed: $($_.Exception.GetBaseException().Message)" }
    return @()
}

function New-PcaiEndpointCandidate {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][System.Net.IPAddress]$IPAddress,
        [Parameter(Mandatory)][string]$Source,
        [string]$PeerId,
        [string]$MacAddress,
        $Interface,
        [string]$Evidence,
        $Port,
        [bool]$MacMatched = $false,
        [bool]$Eui64Derived = $false,
        [string]$NeighborState,
        # The local address the socket must bind to (set when egress was proven), and the interface it leaves by.
        [string]$SourceAddress,
        $EgressInterfaceIndex,
        # $false for a candidate omitted by the egress proof; the connector never dials it.
        [bool]$Admitted = $true
    )
    $class = Get-PcaiAddressClass -IPAddress $IPAddress
    $zone = $null
    if ($IPAddress.AddressFamily -eq [System.Net.Sockets.AddressFamily]::InterNetworkV6 -and $IPAddress.ScopeId -gt 0) { $zone = [int]$IPAddress.ScopeId }
    elseif ($null -ne $EgressInterfaceIndex) { $zone = [int]$EgressInterfaceIndex }
    $alias = $null
    if ($Interface) { $alias = [string](Get-PcaiProperty $Interface 'InterfaceAlias') }
    [pscustomobject]@{
        PSTypeName = 'PC-AI.Network.EndpointCandidate'
        Address = $IPAddress.ToString()
        IPAddress = $IPAddress
        Family = $IPAddress.AddressFamily.ToString()
        Class = $class
        InterfaceIndex = $zone
        InterfaceAlias = $alias
        Source = $Source
        PeerId = $PeerId
        MacAddress = $MacAddress
        Evidence = $Evidence
        Port = $Port
        MacMatched = $MacMatched
        Eui64Derived = $Eui64Derived
        NeighborState = $NeighborState
        SourceAddress = $SourceAddress
        Admitted = $Admitted
    }
}

function Merge-PcaiEndpointCandidate {
    # Dedupe by (address, zone): the address string already carries the zone (fe80::1%25). The surviving
    # candidate keeps the strongest source and the evidence, MAC, neighbour state and flags of every duplicate,
    # because the connector's tie-breaks (MAC matched, neighbour Reachable) read them.
    [CmdletBinding()]
    param([AllowEmptyCollection()][object[]]$Candidate = @())
    $map = @{}
    $order = New-Object 'System.Collections.Generic.List[string]'
    foreach ($c in $Candidate) {
        if ($null -eq $c) { continue }
        $key = ([string]$c.Address).ToLowerInvariant()
        if (-not $map.ContainsKey($key)) { $map[$key] = $c; $order.Add($key); continue }
        $kept = $map[$key]
        $keptRank = 9; $newRank = 9
        if ($script:PcaiSourceRank.ContainsKey([string]$kept.Source)) { $keptRank = $script:PcaiSourceRank[[string]$kept.Source] }
        if ($script:PcaiSourceRank.ContainsKey([string]$c.Source)) { $newRank = $script:PcaiSourceRank[[string]$c.Source] }
        if ($newRank -lt $keptRank) { $kept.Source = $c.Source }
        if ($c.Evidence -and ([string]$kept.Evidence) -notlike "*$($c.Evidence)*") {
            if ($kept.Evidence) { $kept.Evidence = "$($kept.Evidence); $($c.Evidence)" } else { $kept.Evidence = $c.Evidence }
        }
        if (-not $kept.MacAddress -and $c.MacAddress) { $kept.MacAddress = $c.MacAddress }
        if (-not $kept.PeerId -and $c.PeerId) { $kept.PeerId = $c.PeerId }
        if (-not $kept.NeighborState -and $c.NeighborState) { $kept.NeighborState = $c.NeighborState }
        if (-not $kept.InterfaceAlias -and $c.InterfaceAlias) { $kept.InterfaceAlias = $c.InterfaceAlias }
        if ($null -eq $kept.Port -and $null -ne $c.Port) { $kept.Port = $c.Port }
        $kept.MacMatched = [bool]($kept.MacMatched -or $c.MacMatched)
        $kept.Eui64Derived = [bool]($kept.Eui64Derived -or $c.Eui64Derived)
    }
    foreach ($key in $order) { $map[$key] }
}

function ConvertTo-PcaiConnectCandidate {
    # Normalises whatever a caller supplies (a Get-PcaiEndpointCandidate object, or any object with an Address or
    # IPAddress) into the full candidate shape. Returns $null for an address that cannot be parsed.
    [CmdletBinding()]
    param([Parameter(Mandatory)]$InputObject)
    $ip = $null
    $given = Get-PcaiProperty $InputObject 'IPAddress'
    if ($given -is [System.Net.IPAddress]) { $ip = $given }
    else {
        $text = [string](Get-PcaiProperty $InputObject 'Address' '')
        if (-not $text -and $InputObject -is [string]) { $text = $InputObject }
        $text = $text.Trim().TrimStart('[').TrimEnd(']')
        $parsed = $null
        if ($text -and [System.Net.IPAddress]::TryParse($text, [ref]$parsed)) { $ip = $parsed }
    }
    if ($null -eq $ip) { return $null }
    if ($ip.AddressFamily -eq [System.Net.Sockets.AddressFamily]::InterNetworkV6 -and $ip.IsIPv4MappedToIPv6) {
        $tail = $ip.GetAddressBytes()
        $ip = [System.Net.IPAddress]::new([byte[]]@($tail[12], $tail[13], $tail[14], $tail[15]))
    }
    $source = [string](Get-PcaiProperty $InputObject 'Source' 'Caller')
    $candidate = New-PcaiEndpointCandidate -IPAddress $ip -Source $source `
        -PeerId ([string](Get-PcaiProperty $InputObject 'PeerId' '')) `
        -MacAddress ([string](Get-PcaiProperty $InputObject 'MacAddress' '')) `
        -Evidence ([string](Get-PcaiProperty $InputObject 'Evidence' '')) `
        -Port (Get-PcaiProperty $InputObject 'Port') `
        -MacMatched ([bool](Get-PcaiProperty $InputObject 'MacMatched' $false)) `
        -Eui64Derived ([bool](Get-PcaiProperty $InputObject 'Eui64Derived' $false)) `
        -NeighborState ([string](Get-PcaiProperty $InputObject 'NeighborState' ''))
    $alias = Get-PcaiProperty $InputObject 'InterfaceAlias'
    if ($alias) { $candidate.InterfaceAlias = [string]$alias }
    $bind = Get-PcaiProperty $InputObject 'SourceAddress'
    if ($bind) { $candidate.SourceAddress = [string]$bind }
    $zoneIndex = Get-PcaiProperty $InputObject 'InterfaceIndex'
    if ($null -eq $candidate.InterfaceIndex -and $null -ne $zoneIndex) { $candidate.InterfaceIndex = [int]$zoneIndex }
    $candidate.Admitted = [bool](Get-PcaiProperty $InputObject 'Admitted' $true)
    return $candidate
}

function Get-PcaiConnectionOrder {
    # RFC 8305 section 4 ordering, specialised for scoped link-local peers.
    #   IPv6 list: scoped link-local first (observed-neighbour and registry candidates ahead of EUI-64 guesses;
    #   MAC-matched ahead of unmatched; an interface whose neighbour entry is Reachable ahead of
    #   Stale/Delay/Probe/Permanent ahead of unknown), then loopback, ULA, GUA.
    #   IPv4 list: loopback, private, link-local, public.
    #   Merge: the first PreferredFamilyCount IPv6 candidates, then IPv4, IPv6, IPv4, ... (First Address Family Count,
    #   RFC 8305 section 4; default 1). Link-local stays first WITHIN IPv6, but it can no longer starve IPv4: with six
    #   link-local candidates pending, the first IPv4 attempt is the second attempt, not the seventh (PR 402 finding).
    # The original position is the last sort key, so the order never depends on Sort-Object being stable
    # (it is not guaranteed in Windows PowerShell 5.1).
    [CmdletBinding()]
    param([AllowEmptyCollection()][object[]]$Candidate = @(), [ValidateRange(1, 64)][int]$PreferredFamilyCount = 1)
    $scoped = New-Object 'System.Collections.Generic.List[object]'
    $v6 = New-Object 'System.Collections.Generic.List[object]'
    $v4 = New-Object 'System.Collections.Generic.List[object]'
    $index = 0
    foreach ($c in $Candidate) {
        if ($null -eq $c) { continue }
        $index++
        $isV6 = ([string]$c.Family -eq 'InterNetworkV6')
        if ($isV6 -and $c.Class -eq 'LinkLocal' -and $null -ne $c.InterfaceIndex -and [int]$c.InterfaceIndex -gt 0) {
            $tier = 0
            if ($c.Source -eq 'Eui64') { $tier = 1 }
            $mac = 1
            if ($c.MacMatched) { $mac = 0 }
            $neighbor = 2
            if ($c.NeighborState -eq 'Reachable') { $neighbor = 0 }
            elseif ($c.NeighborState) { $neighbor = 1 }
            $scoped.Add([pscustomobject]@{ Tier = $tier; Mac = $mac; Neighbor = $neighbor; Index = $index; Candidate = $c })
        }
        elseif ($isV6) {
            $rank = 3
            if ($c.Class -eq 'Loopback') { $rank = 0 } elseif ($c.Class -eq 'ULA') { $rank = 1 } elseif ($c.Class -eq 'GUA') { $rank = 2 }
            $v6.Add([pscustomobject]@{ Rank = $rank; Index = $index; Candidate = $c })
        }
        else {
            $rank = 3
            if ($c.Class -eq 'Loopback') { $rank = 0 } elseif ($c.Class -eq 'IPv4Private') { $rank = 1 } elseif ($c.Class -eq 'IPv4LinkLocal') { $rank = 2 }
            $v4.Add([pscustomobject]@{ Rank = $rank; Index = $index; Candidate = $c })
        }
    }
    $first = New-Object 'System.Collections.Generic.List[object]'
    foreach ($item in @($scoped | Sort-Object -Property Tier, Mac, Neighbor, Index)) { $first.Add($item.Candidate) }
    foreach ($item in @($v6 | Sort-Object -Property Rank, Index)) { $first.Add($item.Candidate) }
    $second = @($v4 | Sort-Object -Property Rank, Index | ForEach-Object { $_.Candidate })
    $result = New-Object 'System.Collections.Generic.List[object]'
    $i = 0; $j = 0
    while ($i -lt $first.Count -and $i -lt $PreferredFamilyCount) { $result.Add($first[$i]); $i++ }
    while ($i -lt $first.Count -or $j -lt $second.Count) {
        if ($j -lt $second.Count) { $result.Add($second[$j]); $j++ }
        if ($i -lt $first.Count) { $result.Add($first[$i]); $i++ }
    }
    return $result.ToArray()
}

function Get-PcaiAttemptOutcome {
    # Maps a failed connect to Refused, Timeout, Unreachable or Cancelled, and a readable message.
    [CmdletBinding()]
    param([AllowNull()]$Exception)
    $root = $Exception
    while ($root -and $root.InnerException) { $root = $root.InnerException }
    if ($null -eq $root) { return [pscustomobject]@{ Outcome = 'Unreachable'; Message = 'connect failed' } }
    $outcome = 'Unreachable'
    if ($root -is [System.Net.Sockets.SocketException]) {
        $code = $root.SocketErrorCode.ToString()
        if ($code -eq 'ConnectionRefused') { $outcome = 'Refused' }
        elseif ($code -eq 'TimedOut') { $outcome = 'Timeout' }
    }
    elseif ($root -is [System.OperationCanceledException] -or $root -is [System.ObjectDisposedException]) { $outcome = 'Cancelled' }
    [pscustomobject]@{ Outcome = $outcome; Message = [string]$root.Message }
}

function New-PcaiTcpConnection {
    # Default Connector: a TcpClient of the candidate's own address family and its ConnectAsync task. The scope
    # (zone) rides inside the IPAddress. A candidate that carries a SourceAddress (egress proven by
    # Test-PcaiUsb4Egress) is dialled FROM that address: the client is bound to it first, so the OS cannot pick another
    # egress for a prefix that a roaming network also routes. The contract is an object with Task (a
    # System.Threading.Tasks.Task) and Client (disposed by the caller, or $null). Same shape as the Test-FleetPath
    # connector seam (which also binds a source: Test-FleetPath.ps1 $script:FleetTcpProbe).
    [CmdletBinding()]
    param([Parameter(Mandatory)][System.Net.IPAddress]$IPAddress, [Parameter(Mandatory)][int]$Port, $Candidate)
    $bindTo = $null
    $text = [string](Get-PcaiProperty $Candidate 'SourceAddress' '')
    if ($text) {
        $parsed = $null
        if (-not [System.Net.IPAddress]::TryParse($text, [ref]$parsed)) { throw [System.ArgumentException]::new("SourceAddress '$text' is not an IP address") }
        if ($parsed.AddressFamily -ne $IPAddress.AddressFamily) { throw [System.ArgumentException]::new('SourceAddress and destination are different address families') }
        $bindTo = $parsed
    }
    if ($bindTo) { $client = [System.Net.Sockets.TcpClient]::new([System.Net.IPEndPoint]::new($bindTo, 0)) }
    else { $client = [System.Net.Sockets.TcpClient]::new($IPAddress.AddressFamily) }
    try { $task = $client.ConnectAsync($IPAddress, $Port) }
    catch { $client.Dispose(); throw }
    [pscustomobject]@{ Task = $task; Client = $client }
}

function Find-PcaiRoute {
    # Default RouteSource: which interface and source address Windows would use for a destination.
    # Mirrors Find-FleetRoute (Tools/SystemScripts/Networking/Test-FleetPath.ps1:362). $null when there is no route.
    [CmdletBinding()]
    param([Parameter(Mandatory)][string]$Address)
    try {
        $found = @(Find-NetRoute -RemoteIPAddress $Address -ErrorAction Stop)
        $source = $found | Where-Object { $_.PSObject.Properties['IPAddress'] } | Select-Object -First 1
        $route = $found | Where-Object { $_.PSObject.Properties['NextHop'] } | Select-Object -First 1
        $index = $null
        if ($route) { $index = $route.InterfaceIndex } elseif ($source) { $index = $source.InterfaceIndex }
        if ($null -eq $index) { return $null }
        $sourceAddress = $null
        if ($source) { $sourceAddress = ([string]$source.IPAddress -split '%')[0] }
        [pscustomobject]@{ InterfaceIndex = [int]$index; SourceAddress = $sourceAddress; NextHop = $(if ($route) { [string]$route.NextHop } else { '' }) }
    }
    catch { $null }
}

function Get-PcaiLocalAddress {
    # Default LocalAddressSource: the addresses configured on one interface (CIM MSFT_NetIPAddress).
    [CmdletBinding()]
    param([Parameter(Mandatory)][int]$InterfaceIndex)
    try {
        Get-CimInstance -Namespace root/StandardCimv2 -ClassName MSFT_NetIPAddress -Filter "InterfaceIndex=$InterfaceIndex" -ErrorAction Stop |
            ForEach-Object { ([string]$_.IPAddress -split '%')[0] }
    }
    catch { @() }
}

function Test-PcaiUsb4Egress {
    # Egress proof for a candidate that is not scoped to an interface by its own zone (a registry ULA or IPv4
    # address). It is admitted only when the route Windows would use for it leaves over a LIVE USB4/P2P adapter and the
    # route's source address is one of that adapter's own addresses. On a roaming network that routes the same prefix
    # the route selects another interface, and the candidate is omitted ("unverified-egress"). The caller then binds
    # the socket to the proven source address. Anything missing or failing is "not proven".
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][System.Net.IPAddress]$IPAddress,
        [AllowEmptyCollection()][object[]]$Interface = @(),
        [Parameter(Mandatory)][scriptblock]$RouteSource,
        [Parameter(Mandatory)][scriptblock]$LocalAddressSource
    )
    $fail = {
        param($Why)
        [pscustomobject]@{ Admitted = $false; SourceAddress = $null; InterfaceIndex = $null; Reason = "unverified-egress: $Why" }
    }
    if (@($Interface).Count -eq 0) { return (& $fail 'no live USB4/P2P adapter') }
    $route = $null
    try { $route = & $RouteSource $IPAddress.ToString() } catch { Write-Verbose "Route lookup failed: $($_.Exception.Message)" }
    if ($null -eq $route) { return (& $fail 'no route') }
    $routeIndex = Get-PcaiProperty $route 'InterfaceIndex'
    if ($null -eq $routeIndex) { return (& $fail 'route has no interface') }
    $adapter = @($Interface | Where-Object { [int]$_.InterfaceIndex -eq [int]$routeIndex }) | Select-Object -First 1
    if (-not $adapter) { return (& $fail "route leaves via ifIndex $routeIndex, which is not a live USB4/P2P adapter") }
    $sourceText = [string](Get-PcaiProperty $route 'SourceAddress' '')
    $sourceIp = $null
    if (-not [System.Net.IPAddress]::TryParse(($sourceText -split '%')[0], [ref]$sourceIp)) { return (& $fail 'route has no source address') }
    if ($sourceIp.AddressFamily -ne $IPAddress.AddressFamily) { return (& $fail 'route source address is of another family') }
    $owned = @()
    try { $owned = @(& $LocalAddressSource ([int]$routeIndex)) } catch { Write-Verbose "Local address lookup failed: $($_.Exception.Message)" }
    $match = $false
    foreach ($text in $owned) {
        $ip = $null
        if ([System.Net.IPAddress]::TryParse(([string]$text -split '%')[0], [ref]$ip) -and $ip.Equals($sourceIp)) { $match = $true; break }
    }
    if (-not $match) { return (& $fail "route source address is not an address of ifIndex $routeIndex") }
    [pscustomobject]@{ Admitted = $true; SourceAddress = $sourceIp.ToString(); InterfaceIndex = [int]$routeIndex; Reason = 'egress verified' }
}
