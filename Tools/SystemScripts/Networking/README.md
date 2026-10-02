# Thunderbolt Linux peers

`Invoke-ThunderboltLinuxPeer.ps1` uses an existing LAN SSH alias and verified SSH
host key to prepare an Ubuntu peer, discover a real Thunderbolt peer interface,
configure an isolated /30 and measure the actual link. The maintained wrapper is
`Tools/Invoke-ThunderboltNetworking.ps1 -Mode LinuxPeer`.

```powershell
./Tools/Invoke-ThunderboltNetworking.ps1 -Mode LinuxPeer -Peer millylaptop1
./Tools/Invoke-ThunderboltNetworking.ps1 -Mode LinuxPeer -Peer millylaptop1 -Action Prepare -Apply
./Tools/Invoke-ThunderboltNetworking.ps1 -Mode LinuxPeer -Peer millylaptop1 -Action Configure -DryRun
./Tools/Invoke-ThunderboltNetworking.ps1 -Mode LinuxPeer -Peer millylaptop1 -Action Configure -Apply
./Tools/Invoke-ThunderboltNetworking.ps1 -Mode LinuxPeer -Peer millylaptop1 -Action Benchmark -Apply -IperfPath C:/Tools/iperf3.exe
```

Status and default action planning are read-only. `-DryRun` and `-WhatIf` suppress
all writes and benchmark traffic even when `-Apply` is present. `-h` / `-help`
display help. No report files are created automatically; output is a structured
object that an operator may explicitly export.

Profiles live in `Config/thunderbolt-peers.json`. Milly reserves
172.31.240.1/172.31.240.2; the future ASUS profile reserves
172.31.240.5/172.31.240.6. Both use /30 and MTU 1500. The actual hostname, SSH host
key, adapter driver, carrier and nonconflicting routes must match before applying.
ASUS's physical link and profile identity must be checked when it is connected.

Prepare loads `thunderbolt_net`, persists it in
`/etc/modules-load.d/pcai-thunderbolt-net.conf`, and installs iperf3 with its Debian
daemon option disabled. It does not upgrade packages or enable a service. Linux
requires Python 3, `ip`, passwordless `sudo` and NetworkManager for Configure;
Windows Configure requires an elevated PowerShell session. Missing peer hardware
is a Blocked status; Prepare may still run, but Configure and Benchmark refuse.

Selection accepts only an Up Windows adapter whose description identifies
Thunderbolt networking or USB4 peer networking, plus a Linux interface backed by
the `thunderbolt-net` driver with carrier. Exact `-InterfaceAlias` and
`-LinuxInterface` overrides disambiguate multiple peer links; ordinary USB
Ethernet adapters cannot satisfy these checks. The SSH control path stays on LAN.

Configuration adds the Windows address without deleting unrelated addresses.
Linux creates the persistent NetworkManager profile `pcai-tb-<peer>` and rejects
ambiguous names or an existing profile with unexpected interface, address,
IP method, gateway, DNS, routes, IPv6 settings, or autoconnect policy. Existing
profiles are selected by UUID. It sets no
gateway or DNS, disables default routing, and uses MTU 1500 at both ends. If a
later Windows step fails, the Linux isolated profile can remain configured; the
error is surfaced and LAN SSH remains available for recovery. Re-running is
idempotent. Rollback removes only the added Windows /30 address and owned Linux
profile; undo module persistence only if it is no longer needed.

After applying Configure, the tool verifies SSH over the Thunderbolt address with
the local source bound and checks Linux's return route. Its `ThunderboltSsh`
result includes a usable command that reuses the existing alias's user/key and
LAN host-key identity with strict checking. This leaves the LAN alias unchanged.

Benchmark requires configured addresses and a source-bound Windows route through
the selected adapter. It runs forward and reverse iperf3 against peer-address-bound
one-shot servers. Server lifetime and local processes are bounded; raw iperf3 JSON
is returned. Any native exit failure, route mismatch or missing hardware prevents
a throughput result. Rates are measurements, not claims about cable or dock speed.

The explicit Linux-runtime regression suite executes the generated Bash profile
logic with hardware and NetworkManager I/O fixtures. Run it only against an
authorized Linux SSH alias; it creates and removes temporary fixture files and
never invokes real sudo or NetworkManager:

```powershell
$env:PCAI_TB_TEST_SSH_ALIAS = 'millylaptop1'
Invoke-Pester ./Tests/Integration/ThunderboltLinuxProfile.Tests.ps1 -Output Detailed
```
