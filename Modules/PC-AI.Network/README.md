# PC-AI.Network

Network diagnostics, VSock tuning and WSL connectivity checks for PC-AI, plus the dual-stack endpoint commands that
find a USB4 peer and connect to it with RFC 8305 (Happy Eyeballs v2).

## Install, build, test, run

```powershell
# Install: nothing to build. The module is a plain script module; PowerShell 5.1 and 7 both load it.
Import-Module .\Modules\PC-AI.Network\PC-AI.Network.psd1 -Force

# Test (no network, no hardware; every provider and connector is injected)
Import-Module Pester -MinimumVersion 5.0
Invoke-Pester -Path .\Tests\Unit\PC-AI.Network.Tests.ps1
pwsh -NoProfile -File .\Tests\Portable\Test-DualStackConnector.ps1     # real loopback sockets

# Run
Get-NetworkDiagnostics
Resolve-PcaiPeerEndpoint -PeerId dtm-super-nuc -Port 22 -Wake
```

## Commands

| Command | Purpose |
| --- | --- |
| `Get-NetworkDiagnostics`, `Test-WSLConnectivity`, `Optimize-VSock`, `Watch-VSockPerformance` | Network stack, WSL and VSock diagnostics |
| `Get-StableNetAdapter`, `Repair-UsbNetAdapter`, `Test-NetPathHealth` | Adapter identity, repair and link measurement |
| `Get-PcaiEndpointCandidate` | Addresses a peer may be reached at now, IPv6 link-local with its live zone included |
| `Connect-PcaiEndpoint` | Races those addresses per RFC 8305 and reports the winner and every attempt |
| `Resolve-PcaiPeerEndpoint` | The winning address for ssh and probe consumers |

The design, the identity model (identifiers locate a peer, SSH host keys authenticate it), the ordering and timing rules
and the known limits are in [docs/networking/dual-stack.md](../../docs/networking/dual-stack.md). The USB4 peer
framework these commands sit beside is described in [peer networking](../../docs/usb4/peer-networking.md).

## Layout

* `Public/` one file per exported command; the export list is in `PC-AI.Network.psd1` and `PC-AI.Network.psm1` and the
  two must agree (a unit test checks it).
* `Private/Endpoint-Helpers.ps1` address logic for the dual-stack commands: EUI-64, classification, ordering, merging.
* Compatibility: the manifest declares Windows PowerShell 5.1, so the dual-stack code uses no ternary, null-coalescing
  or `ForEach-Object -Parallel`, and only .NET Framework 4.7.2 APIs. Keep new sources ASCII.
