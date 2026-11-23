#Requires -Version 5.1
function Resolve-PcaiPeerEndpoint {
    <#
    .SYNOPSIS
        Finds the address of a peer that answers on a port right now, for ssh and probe consumers.

    .DESCRIPTION
        Runs Get-PcaiEndpointCandidate, then Connect-PcaiEndpoint (RFC 8305 race), and returns the winning candidate,
        or nothing when no address answered. The connect is a reachability test; the socket is closed again.

        The winner is where the peer is, not who it is. Authenticate with the SSH host key. Note that ssh itself does
        not race: pass the winning Address (zone included, for example fe80::1%25) to it.

    .PARAMETER HostName
        Name or address of the peer.

    .PARAMETER PeerId
        Registry peer name, alias or node id.

    .PARAMETER MacAddress
        Peer MAC, used by the Eui64 and Neighbor providers.

    .PARAMETER Port
        TCP port to test. Mandatory.

    .PARAMETER InterfaceAlias
        Pin to one adapter by alias (resolved live).

    .PARAMETER InterfaceIndex
        Pin to one interface index.

    .PARAMETER Provider
        KnownPeer, Eui64, Neighbor, Dns. Default: all.

    .PARAMETER Wake
        Send the all-nodes multicast ping per interface before reading the neighbour cache.

    .PARAMETER AttemptDelayMs
        Connection Attempt Delay (clamped 10..2000).

    .PARAMETER TimeoutMs
        Overall deadline.

    .PARAMETER Detailed
        Return the full Connect-PcaiEndpoint result (attempts and timing) instead of only the winner.

    .PARAMETER Connector
        Test seam, see Connect-PcaiEndpoint.

    .PARAMETER ResolverParameter
        Test seam, see Connect-PcaiEndpoint.

    .EXAMPLE
        $peer = Resolve-PcaiPeerEndpoint -PeerId dtm-super-nuc -Port 22 -Wake
        if ($peer) { ssh "david@$($peer.Address)" }
    #>
    [CmdletBinding()]
    [OutputType([pscustomobject])]
    param(
        [string]$HostName,
        [string]$PeerId,
        [string]$MacAddress,
        [Parameter(Mandatory)][ValidateRange(1, 65535)][int]$Port,
        [string]$InterfaceAlias,
        [int]$InterfaceIndex,
        [ValidateSet('KnownPeer', 'Eui64', 'Neighbor', 'Dns')][string[]]$Provider,
        [switch]$Wake,
        [int]$AttemptDelayMs = 250,
        [int]$TimeoutMs = 5000,
        [switch]$Detailed,
        [scriptblock]$Connector,
        [hashtable]$ResolverParameter
    )
    $connect = @{ Port = $Port; AttemptDelayMs = $AttemptDelayMs; TimeoutMs = $TimeoutMs }
    foreach ($name in 'HostName', 'PeerId', 'MacAddress', 'InterfaceAlias', 'InterfaceIndex', 'Provider', 'Connector', 'ResolverParameter') {
        if ($PSBoundParameters.ContainsKey($name)) { $connect[$name] = $PSBoundParameters[$name] }
    }
    if ($Wake) { $connect['Wake'] = $true }
    $result = Connect-PcaiEndpoint @connect
    if ($Detailed) { return $result }
    if ($result.Connected) {
        Write-Verbose ('Winner {0} ({1}, {2}) in {3} ms after {4} attempt(s).' -f $result.Winner.Class, $result.Winner.Source, $result.Winner.Family, $result.ElapsedMs, @($result.Attempts).Count)
        return $result.Winner
    }
    Write-Verbose "No endpoint answered on port $Port after $(@($result.Attempts).Count) attempt(s)."
}
