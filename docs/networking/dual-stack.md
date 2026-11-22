# Dual-stack endpoints and the RFC 8305 connector

`Get-PcaiEndpointCandidate`, `Connect-PcaiEndpoint` and `Resolve-PcaiPeerEndpoint` in
[PC-AI.Network](../../Modules/PC-AI.Network/README.md) find a peer again, quickly, after it was unplugged and plugged
back in somewhere else, and connect to it without stalling on a dead address family. IPv6 link-local is a first-class
address next to IPv4 and ULA, and Happy Eyeballs (RFC 8305) decides which one wins.

## Install, test, run

```powershell
Import-Module .\Modules\PC-AI.Network\PC-AI.Network.psd1 -Force      # no build step; works in 5.1 and 7
Import-Module Pester -MinimumVersion 5.0
Invoke-Pester -Path .\Tests\Unit\PC-AI.Network.Tests.ps1             # injected providers and connector, no network
pwsh -NoProfile -File .\Tests\Portable\Test-DualStackConnector.ps1   # real loopback sockets; also runs in powershell.exe
Resolve-PcaiPeerEndpoint -PeerId dtm-super-nuc -Port 22 -Wake -Verbose
```

## Locate, never authenticate

Every identifier here **locates** a peer. None **authenticates** it. The identifiers are:

| Identifier | Where it comes from | What it gives |
| --- | --- | --- |
| Registry peer id | `Config/thunderbolt-peers.json` (name, alias, `tbHostAlias`, `nodeIdHex`) | Short and hash-derived link-local and ULA addresses, registry IPv4 addresses |
| MAC | `-MacAddress`, or `expectedPeerMacs` in the registry's `topologyLinks` | The modified EUI-64 link-local address; a filter for neighbour-cache rows |
| Neighbour entry | IPv6 neighbour cache (CIM `MSFT_NetNeighbor`, the same source as `Find-ThunderboltPeer`) on the live USB4 interfaces | An address that was actually seen on a link, with its MAC and state |
| Deterministic address | SHA-256 of the canonical peer name, the same derivation as Drivers | `fe80::d700:<h0>:<h1>:<h2>` and `fd00:7462:7032:d700::<h0>:<h1>:<h2>` |

MAC spoofing is not a concern of this code. A connect that succeeds means something listens at that address. SSH host
keys (pinned in `known_hosts`) or another authenticated protocol decide whether it is the right machine, as in
[peer networking](../usb4/peer-networking.md): MAC enrollment limits a link, it is not authentication.

## Providers

`Get-PcaiEndpointCandidate -Provider KnownPeer, Eui64, Neighbor, Dns` (default all four). Each reads one source:

* **KnownPeer** resolves the registry peer by `-PeerId` or `-HostName` and emits its link-local addresses (short alias
  first, then the derived one), its ULA addresses and its registry IPv4 addresses.
* **Eui64** derives `fe80::<iid>` from the MAC: flip the universal/local bit of the first octet and insert `ff:fe`. The
  vector `00-11-22-33-44-55` gives `fe80::211:22ff:fe33:4455`.
* **Neighbor** keeps link-local rows in state Reachable, Stale, Delay, Probe or Permanent whose MAC matches (separators
  and case do not matter) or whose address is one the registry peer is known to use. With neither a MAC nor a registry
  peer it offers nothing, because "every neighbour on the link" is not "the peer". `-Wake` first runs
  `ping -6 -n 1 -w 500 ff02::1%<ifIndex>` once per interface to fill a cold cache.
* **Dns** uses `GetHostAddressesAsync` (A and AAAA, 2 s deadline). `getaddrinfo` drops a zone-less `fe80`, so link-local
  never comes from DNS.

Candidates carry `Address` (with the zone, `fe80::...%25`), `IPAddress` (an `[ipaddress]` with `ScopeId` set),
`Family`, `Class` (`LinkLocal`, `ULA`, `GUA`, `IPv4Private`, `IPv4LinkLocal`, `IPv4Public`, `Loopback`),
`InterfaceIndex`, `InterfaceAlias`, `Source`, `PeerId`, `MacAddress`, `Evidence`, plus `Port`, `MacMatched`,
`Eui64Derived` and `NeighborState`. Duplicates are merged by (address, zone): the strongest source is kept
(Neighbor, then KnownPeer, Eui64, Dns) with the evidence, MAC and neighbour state of all of them.

## Reconnecting across topologies

A link-local address is ambiguous: `fe80::d700:50` can exist on every interface. The zone is the Windows `ifIndex`, and
the `ifIndex` changes with a replug, another port, a different hub or a driver reinstall. So:

* With no interface pinned, one candidate per live USB4, Thunderbolt or P2P adapter that is Up is emitted. The matcher
  mirrors `Find-ThunderboltPeer` (description or name).
* The adapter table is read on every call. Nothing remembers an `ifIndex`. A pinned `-InterfaceAlias` is looked up again
  each time.
* Each scoped candidate is annotated with the neighbour state of its own interface, so when one address exists on two
  links the connector tries the link whose entry is Reachable first.

After a replug from ifIndex 19 to 25 the next call returns `%25` candidates and no `%19` ones. Unit tests drive exactly
that.

## Ordering and timing (RFC 8305)

1. Scoped IPv6 link-local candidates first. Registry and neighbour-observed candidates come before EUI-64 guesses;
   MAC-matched before unmatched; Reachable before Stale/Delay/Probe/Permanent before unknown.
2. Then IPv6 and IPv4 interleaved, starting with IPv6 (ULA before GUA; IPv4 private before link-local before public).

`Connect-PcaiEndpoint` starts attempt 1 immediately and each next attempt after `-AttemptDelayMs` (default 250, clamped
to 10..2000), **or immediately when the attempt in flight fails**. The first success wins; every other client is
disposed and recorded as `Cancelled`; whatever is still pending at `-TimeoutMs` (default 5000) is disposed and recorded
as `Timeout`. The result lists every attempt that was started, with class, zone, `StartMs`, `EndMs`, outcome
(`Connected`, `Refused`, `Timeout`, `Unreachable`, `Cancelled`) and the error text. A failed connect is data
(`Connected = $false`); only an explicit `-ErrorAction Stop` throws.

## Usage

```powershell
# Which addresses could be the peer right now?
Get-PcaiEndpointCandidate -PeerId dtm-super-nuc -Provider KnownPeer, Eui64, Neighbor -Wake |
    Format-Table Address, Class, Source, NeighborState

# Does anything answer SSH, and over which family and link?
$r = Connect-PcaiEndpoint -PeerId dtm-super-nuc -Port 22 -Wake
$r.Connected; $r.Winner.Class; $r.ElapsedMs; $r.Attempts | Format-Table Class, Zone, StartMs, EndMs, Outcome

# Take the open socket
$r = Get-PcaiEndpointCandidate -PeerId dtm-super-nuc | Connect-PcaiEndpoint -Port 22 -PassThru
if ($r.Connected) { $r.Client.Dispose() }

# Just the winner, for ssh
$peer = Resolve-PcaiPeerEndpoint -PeerId dtm-super-nuc -Port 22
if ($peer) { ssh "david@$($peer.Address)" }
```

`Tools/SystemScripts/Networking/Test-FleetPath.ps1 -DualStack` adds the same race to the fleet path diagnostic. It
feeds the connector only candidates that already passed that tool's home-LAN and tailnet-egress gates plus the scoped
USB4 link-local and ULA addresses, and it adds `DualStack` members to the report only when the switch is given.

## Known limits

* `ssh` does not race. `Resolve-PcaiPeerEndpoint` tells you the winning address; you pass it to `ssh`. A native
  ProxyCommand relay that races inside the SSH connection is a later change (PR-C).
* The Eui64 provider only matches Windows peers whose IPv6 interface identifiers are not randomized. Windows defaults to
  `RandomizeIdentifiers = Enabled`, and `Register-ThunderboltPeer` sets it to `Disabled`
  ([source](../../Modules/PC-AI.Drivers/Public/Register-ThunderboltPeer.ps1)). On a peer that still randomizes, the
  EUI-64 candidate is a guess that does not exist on the wire. The Neighbor provider works either way, because it reads
  what was actually seen.
* The default adapter and neighbour reads use CIM, not `Get-NetAdapter` or `Get-NetNeighbor`: in a cold process that is about 0.6 s instead of about 4 s, which matters for a reconnect.
* A neighbour entry is only as fresh as the cache. `-Wake` helps on a sleeping link; a peer in modern standby may still
  not answer.
* The deterministic hextets mirror `Get-ThunderboltPeerIdentity` (Drivers). A unit test compares both on PowerShell 7 so
  they cannot drift apart; consolidating them into one helper is a follow-up for the Drivers owner.
* `Test-PortConnectivity` in `Network-Helpers.ps1` still makes a single `BeginConnect` attempt. Its contract tests pin
  that exact call, so switching it to this connector is a separate change.
