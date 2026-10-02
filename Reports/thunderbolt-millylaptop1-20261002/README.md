# millylaptop1 Thunderbolt networking — 2026-10-02

Software preparation is complete. The physical Thunderbolt network is not
established, and no Thunderbolt throughput has been measured. The final live
status is `Blocked`: Windows has zero Thunderbolt/USB4 peer network adapters;
Linux has zero `thunderbolt-net` interfaces with carrier and exposes only its
local Thunderbolt domain and host router (`domain0`, `0-0`).

## Connection and observed hardware

The user confirms an Apple Thunderbolt 4 cable connects a rear downstream
Thunderbolt/lightning port on the CalDigit Element 5 hub attached to dtm-p1gen7
to millylaptop1. A direct computer-to-computer test is currently unavailable.
Windows enumerates the Element 5 hub and its other peripherals. The ThinkPad's
Maple Ridge controller is present; Thunderbolt BIOS access is enabled, XDomain
is allowed, and the domain security level is `none`. Windows has Microsoft's
inbox USB4 peer networking driver. Device rescanning, Linux driver loading and a
temporary controller runtime-power test did not reveal a peer. Runtime-power
policy was restored to `auto`.

The ThinkPad logged repeated USB port enable/configuration errors until
14:48:33 EDT; these stopped after the user's reseat. Stopping those errors did
not establish a Thunderbolt peer. No specific cable, hub or firmware defect has
been proved. The Linux USB device running at 10000M with `cdc_ncm` is the Realtek
0bda:8157 USB Ethernet adapter, not a Thunderbolt peer. Its 5 GbE product label
and USB bus rate are not measured network throughput.

## Installed and reusable tooling

On millylaptop1, `thunderbolt_net` is loaded and persisted in
`/etc/modules-load.d/pcai-thunderbolt-net.conf`. Ubuntu's iperf3 3.9 is installed;
its service is inactive and port 5201 has no listener. Python, NetworkManager
and bolt were already available. Windows iperf3 3.21 is available. Existing LAN
SSH access remains working. No owned Thunderbolt NetworkManager profile or
172.31.240.* address has been activated while peer hardware is missing.

Maintained entrypoint: `Tools/Invoke-ThunderboltNetworking.ps1 -Mode LinuxPeer`.
The helper lives in `Tools/SystemScripts/Networking/Invoke-ThunderboltLinuxPeer.ps1`.
`Config/thunderbolt-peers.json` contains separate isolated /30 profiles:

| Peer | Windows endpoint | Linux endpoint | MTU | Current status |
| --- | --- | --- | --- | --- |
| millylaptop1 | 172.31.240.1 | 172.31.240.2 | 1500 | Prepared; no peer link |
| asuspro13 | 172.31.240.5 | 172.31.240.6 | 1500 | Future profile; not applied |

Each profile requires the expected SSH hostname/key, a genuine Windows peer
adapter, a Linux `thunderbolt-net` driver with carrier, and nonconflicting
addresses/routes. Configuration preserves other interfaces, adds no gateway or
DNS, disables Linux default routing and verifies source-bound direct SSH and
the Linux return route. It rejects existing profiles with unexpected routes,
IPv6 or other settings before activation. Benchmarking binds both endpoints to
the peer addresses, runs forward/reverse one-shot iperf3 servers and rejects a
route through a different interface. LAN SSH remains the recovery channel.

From an elevated PowerShell session, after peer discovery succeeds:

```powershell
$tool = 'C:/codedev/PC_AI/Tools/Invoke-ThunderboltNetworking.ps1'
& $tool -Mode LinuxPeer -Peer millylaptop1 -Action Status
& $tool -Mode LinuxPeer -Peer millylaptop1 -Action Configure -DryRun
& $tool -Mode LinuxPeer -Peer millylaptop1 -Action Configure -Apply
& $tool -Mode LinuxPeer -Peer millylaptop1 -Action Benchmark -Apply
```

Use the returned `ThunderboltSsh.Command` for verified direct control. SSH/SFTP
can carry commands, telemetry and files over this ordinary IP transport. Future
ASUS links use the same tooling with `-Peer asuspro13`, after physical connection
and identity validation. These profiles do not assume a hub supports three
simultaneous computer hosts or acts as an Ethernet switch.

## Intel software and alternatives

Intel Thunderbolt Share is a Windows application requiring installation on
both PCs and a manufacturer-licensed PC/accessory. Its application licence is
not required for standard Thunderbolt IP networking. Microsoft's native USB4NET
and Linux's `thunderbolt-net` provide the transport without that application.
Linux's kernel documentation explicitly supports Windows/Linux networking.
Installing Share on Ubuntu is therefore neither supported nor necessary.

SSH supplies remote command/control and SFTP supplies file transfer over the
native link. Folder synchronization or a graphical remote desktop can be added
as separate applications if desired; this work has not installed them or
validated graphical control. These are alternatives to Share's application
features and cannot create a host-to-host tunnel when the controllers have not
enumerated each other.

Official sources checked:

- [Microsoft USB4 interdomain connections](https://learn.microsoft.com/en-us/windows-hardware/design/component-guidelines/usb4-interdomain-connections)
- [Linux 6.8 Thunderbolt networking](https://www.kernel.org/doc/html/v6.8/admin-guide/thunderbolt.html#networking-over-thunderbolt-cable)
- [Intel Thunderbolt Share setup and requirements](https://www.intel.com/content/www/us/en/support/articles/000098894/software.html)
- [CalDigit Element 5 support](https://www.caldigit.com/element-5-hub-support/)

## Verification and remaining work

The final scoped run passed 34 new tests (29 unit tests plus five Linux Bash
runtime fixtures) and all 20 selected existing Thunderbolt tests, with no
failures or skips. The five fixtures execute the actual generated configuration
logic, replacing only hardware/NM I/O, and verify valid UUID selection, new
profile creation and rejection of static routes, automatic IPv6 and duplicate
names. PSScriptAnalyzer reported zero diagnostics for the changed PowerShell
files. Direct SSH proxy/source-binding options, dry-run/WhatIf, overlapping
routes, native exit failures and timeout termination are covered.

Live Status confirms the absent peer. Live Configure correctly refuses before
any network mutation. Actual Configure activation and the iperf3 benchmark
process lifecycle/throughput remain unverified on Thunderbolt hardware. Passing
fixtures do not substitute for that physical validation.

The next diagnostic step is link enumeration: isolate alternate downstream hub
and ThinkPad Thunderbolt ports, then a direct cable path when practical, and
check OEM hub/controller/BIOS firmware if the peer still does not appear. No
firmware flashing, controller reset, reboot or global security change was done.
A missing peer is not evidence that the Apple cable is faulty or that Intel
Share would fix it.

For rollback, no network profile currently needs removal. If later configured,
remove only `pcai-tb-millylaptop1` and the added Windows 172.31.240.1/30 address.
Remove the owned module-load file only if persistent Thunderbolt networking is
no longer desired; iperf3 is a utility and has no active server service here.

Evidence: `status-final.json`, `linux-prepared-final.txt`, `verification.json`,
`windows-topology.json`, `linux-topology.txt` and `prepare-result.json`.
`status-before-prepare.json` is historical; use final status for current state.
