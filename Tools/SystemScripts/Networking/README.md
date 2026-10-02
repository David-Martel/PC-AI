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

## Controller discovery traces

When no peer interface exists, packet capture cannot diagnose the missing
interdomain tunnel. `Invoke-Usb4DiscoveryTrace.ps1` captures the two Microsoft
USB4 router discovery providers instead. It plans by default; `-Apply` requires
elevation and an explicit report directory. Duration is bounded to 1–60 seconds
and the circular ETL to 32–128 MiB. Each run owns a unique collector, removes it
before decoding, and retains native failure output. DryRun, WhatIf and help have
no side effects. This captures router metadata rather than network payloads.

```powershell
./Tools/SystemScripts/Networking/Invoke-Usb4DiscoveryTrace.ps1 -DryRun
./Tools/SystemScripts/Networking/Invoke-Usb4DiscoveryTrace.ps1 -ReportDirectory C:/Reports/usb4 -DurationSeconds 20 -Apply
```

On Linux, deploy `capture_thunderbolt_peer_trace.py` to the authorized peer and
run its default plan before opting into a root capture:

```bash
python3 capture_thunderbolt_peer_trace.py --dry-run
sudo python3 capture_thunderbolt_peer_trace.py --apply --expected-hostname millylaptop1 --duration 20 --tools-dir /home/cog/.local/bin --output-dir /home/cog/.cache/pcai/thunderbolt-trace-new
```

The output directory must be new. The helper uses its own tracefs instance with
only supported UCSI and Thunderbolt networking events. It temporarily enables
previously disabled Thunderbolt dynamic-debug printing, restores those exact
callsites, removes its instance and verifies global tracing was unchanged.
Cleanup and evidence failures return failure; signal handlers restore even if
final evidence writes fail. Optional Intel `tbtools` binaries are used only for
read-only inventories. Neither capture helper rebinds a driver, flashes firmware
or configures network addresses. Empty trace data is an observation, not proof
of a cable defect or a working connection.
