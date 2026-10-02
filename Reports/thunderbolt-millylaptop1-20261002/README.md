# millylaptop1 Thunderbolt networking — 2026-10-02

**BIOS 2.00 is installed and LAN SSH has recovered.** Lenovo system firmware was
staged with `fwupdmgr` local installation, followed by a reboot at approximately
19:56 UTC. The ThinkPad returned after approximately 11 minutes of firmware
progress. Live postboot DMI identifies `N3TET64W (2.00)` with recorded BIOS date
`02/12/2026`; fwupd history reports successful update state 2 and current system
firmware 0.2.0 without an update error. VirtualizationTechnology, VTdFeature and
ThunderboltAccess are all verified `Enable`. SecureBoot is verified `Disable`;
it was not changed as a prerequisite for this work. The first postboot snapshot
confirms LAN access, the persistent Thunderbolt module and NVIDIA GPU discovery,
but Linux lane adapters remain in `CLd` with no peer. DMI still reports EC
firmware revision 1.1; the separate EC update remains unapplied. Evidence:
[postboot verification](thinkpad-after-bios-reboot.txt),
[staging result](thinkpad-firmware-stage-result.txt) and
[pre-reboot network state](thinkpad-before-reboot.txt).
[Final recovery check](recovery-final.txt) confirms SSH user `cog`, passwordless
sudo after ticket invalidation, the staged ISO's remote SHA-256, restored `auto`
runtime policy and zero owned Thunderbolt network profiles.

The physical Thunderbolt network remains unresolved, and no Thunderbolt
throughput has been measured. The final guarded status after BIOS update,
Thunderbolt driver rebind and the user's alternate ThinkPad port test is
`Blocked`: Windows has zero Thunderbolt/USB4 peer network adapters; Linux has
zero `thunderbolt-net` interfaces with carrier and exposes only its local domain
and host router (`domain0`, `0-0`). The historical software-preparation snapshot
is [status-final.json](status-final.json). The initial postboot register check
still shows no Linux peer. Use [final alternate-port status](status-alternate-port.json)
for the latest host state rather than the historical preparation snapshot.

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

Low-level diagnostics reached the controller/PHY layer. Intel's decoded Linux
adapter registers reported all four lane adapters in `CLd`, no router detected
and no tunnels. Windows router rundown showed CalDigit lane 3 in `CL0` with
Thunderbolt 3 compatibility, but no interdomain event or peer network adapter.
That Windows lane observation does not by itself identify a working link to
the ThinkPad. These results leave physical negotiation unresolved and do not
prove the Apple cable, hub or ThinkPad port is faulty. See
[Linux register decoding](intel-tbtools-validation-20261002.txt),
[Windows router rundown](windows-router-rundown.json) and
[Windows rundown during the reboot observation](windows-reboot-router-rundown.json).

The CalDigit tree also contains an ACASIS Thunderbolt storage device hosting
Windows drive `D:`. Preserve its active storage when considering controller or
hub resets; no CalDigit firmware flash was performed.

## Installed and reusable tooling

On millylaptop1, `thunderbolt_net` is loaded and persisted in
`/etc/modules-load.d/pcai-thunderbolt-net.conf`. Ubuntu's iperf3 3.9 is installed;
its service is inactive and port 5201 has no listener. Python, NetworkManager
and bolt were already available. Windows iperf3 3.21 is available. LAN
SSH worked before the firmware reboot and has recovered afterward.
No owned Thunderbolt NetworkManager profile or
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
route through a different interface. LAN SSH is the intended recovery channel.

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

## Discovery capture and performance baseline

Two maintained helpers add bounded discovery telemetry:

- `Tools/SystemScripts/Networking/Invoke-Usb4DiscoveryTrace.ps1` collects Windows
  USB4 HostRouter and DeviceRouter ETW metadata into a unique bounded collector.
  Planning, `-DryRun` and `-WhatIf` avoid writes and native processes.
- `Tools/SystemScripts/Networking/capture_thunderbolt_peer_trace.py` creates a
  separate Linux tracefs instance for supported UCSI and Thunderbolt-network
  events, temporarily enables selected Thunderbolt dynamic-debug printing, then
  restores the settings and removes its trace instance. It plans by default;
  actual capture requires root, an exact hostname and a new output directory.

The Windows helper passed 21 tests and a real 60-second capture, with its native
commands succeeding. The final Linux helper passed seven tests, Ruff and
independent review, then a fresh live capture around the NHI driver rebind.
Its real 20-second pre-reboot capture contained zero trace
data events. All 264 changed dynamic-debug selectors were restored, the saved
debug-control snapshots were byte-identical, its trace instance was removed,
and the global tracing switch was unchanged. Zero events in this interval do
not prove that future reconnects would produce none. Evidence:
[Windows tests](capture-pester-verification.json),
[Windows live capture result](windows-reboot-discovery-result.json), and
[Linux live capture summary](thunderbolt-diag-20261002-trace1/summary.json).
The Windows decoder returned exit 0 while warning that some system events lacked
a matching schema; the USB4 router metadata was decoded, including 62 USB4 events
in the reboot capture. No capture completion is being used as evidence of a peer.

Intel's read-only diagnostic executables `tblist`, `tbadapters`, `tbdump`,
`tbget`, `tbmonitor`, `tbtrace` and `tbtunnels` were built on Ubuntu from upstream
commit `aa0b1be590443d7074e799ebd6c72e308c5bdd02` and installed in
`/home/cog/.local/bin`. A minimal Rust 1.99 toolchain was installed for that
build; existing native build dependencies were sufficient. Recorded provenance
includes the upstream source, installer and binary hashes, and successful help
checks for all seven tools. See [build provenance](intel-tbtools-provenance-20261002.txt)
and [validation](intel-tbtools-validation-20261002.txt).

Ubuntu `ethtool` 1:5.16-1ubuntu0.2 was installed; `tcpdump` 4.99.1 was already
available. Windows tshark can enumerate the existing Npcap and USBPcap capture
interfaces. Driver presence is documented in
[capture drivers](windows-capture-drivers.json) and
[tshark interfaces](windows-tshark-interfaces.txt); it does not demonstrate a
Thunderbolt peer or a successful USBPcap packet capture. Router ETW and Linux
controller traces address discovery below the absent IP interface.

The working **wired LAN**, before reboot, measured the following iperf3 receiver
throughput. These measurements are a recovery-path baseline, not Thunderbolt
performance:

| Direction | Receiver throughput | Evidence |
| --- | --- | --- |
| Windows to Linux | 933.4 Mbit/s | [iperf3 JSON](lan-baseline-windows-to-linux.json) |
| Linux to Windows | 744.59 Mbit/s | [iperf3 JSON](lan-baseline-linux-to-windows.json) |

## Firmware custody and boot verification

The user explicitly authorized relevant firmware, BIOS, settings and software
updates on the ThinkPad and attached dtm-p1gen7 devices. That authorization
supersedes earlier plans to request firmware approval again. Updates still
require matching supported hardware and verifiable package custody.

Lenovo/LVFS packages were downloaded, their SHA-256 values checked and local
`fwupdmgr` details inspected against the P16 Gen 2 device identifiers:

| Component | Recorded installed version | Package target | Verified action |
| --- | --- | --- | --- |
| P16 system firmware | 0.1.3 before; 0.2.0 after | 0.2.00 (BIOS 2.00) | Installed; live DMI and successful fwupd history verified |
| P16 embedded controller | DMI firmware revision 1.1 after | Current combined package EC 1.58 | Bootable updater staged; no EC update applied |
| CalDigit Element 5 | 61.1 | 71.71 | Vendor updater inspected; no flash performed |

The complete package sources, hashes and release notes are in
[firmware plan](thinkpad-firmware-plan.json), with Linux file hashes and device
matching in [CAB inspection](thinkpad-firmware-cab-details.txt). Official package
sources are the [Lenovo P16 system firmware 2.00 CAB](https://fwupd.org/downloads/1e5bab9a315b9e57a779dc4f2c7faa62251a5b51e0a3cc89a265fee072384993-Lenovo-ThinkPad-P16Gen2-SystemFirmware-2.00.cab)
and [Lenovo P16 EC 1.50 CAB](https://fwupd.org/downloads/814be1e5d51644e5d72e6576f7ebd1f80ef9207a27d4170da51e475dcd1f2919-Lenovo-ThinkPad-P16Gen2-EmbeddedControllerFirmware-1.50.cab).
The captured BIOS 2.00 release notes do not establish a Thunderbolt fix.

The CalDigit v7.5 firmware package documents Element 5 target 71.71 and fixes
including intermittent monitor flicker; its earlier 64.1 change log mentions
Thunderbolt 5 host link stability. The signed vendor updater GUI enumerated
only ACASIS, not the CalDigit hub, so no firmware was sent to either device.
Do not select the ACASIS device for CalDigit firmware. See the
[vendor updater README](firmware-staging/unpacked/CalDigit-TBT-Firmware-Updater-v7.5/ReadMe.txt)
and [CalDigit support page](https://www.caldigit.com/element-5-hub-support/).
[Final GUI evidence](caldigit-updater-discovery.json) verifies the valid vendor
signature, absent Element5, disabled Next button and graceful updater cleanup.

Before reboot the ThinkPad had AC connected, a full battery and a working wired
connection at 192.168.50.43. Firmware processing temporarily interrupted LAN
SSH; recovery took approximately 11 minutes. Live postboot DMI confirms BIOS
`N3TET64W (2.00)` and fwupd confirms successful update history and system firmware
0.2.0 without an update error. The written virtualization and VT-d settings are
now verified enabled, as is Thunderbolt access. Secure Boot is disabled, as
observed after boot. The named embedded-controller device is absent from the new
fwupd inventory and its GUID is absent from the actual ESRT table. DMI reports
firmware revision 1.1. The older trusted EC 1.50 CAB was not forced against a
missing or different device. See [post-BIOS EC inventory](thinkpad-ec-after-bios.txt)
and [ESRT/preflight inventory](linux-rebind-preflight.txt).

Lenovo's current combined package N3TUR13W pairs BIOS 2.00/N3TET64W with EC
1.58/N3THT58W. Its supported bootable updater updates both components and runs
independently of the installed OS. The official README documents a USB optical
drive and F12 USB CD boot; arbitrary ISO-to-USB writing or remote chainloading
was not validated. The official `n3tur13w.iso` was downloaded and SHA-256 checked
against Lenovo's published checksum, then copied to
`/home/cog/.cache/pcai/firmware/n3tur13w.iso` with the same hash. It has not been
booted. This remaining update requires a supported boot path and local recovery.
Sources: [Lenovo package](https://support.lenovo.com/vc/en/downloads/ds563392),
[bootable updater instructions](https://download.lenovo.com/pccbbs/mobiles/n3tur13w.html).
[Firmware custody](firmware-custody.json) records exact sources, hashes and actions.
Lenovo documents that BIOS 1.99 and later cannot roll back below 1.99; do not
attempt to restore the original 1.03 capsule as a rollback procedure.

The [initial postboot snapshot](thinkpad-after-bios-reboot.txt) confirms wired
LAN 192.168.50.43, `thunderbolt_net` loaded, controller NVM 41.82 and working
read-only Intel diagnostic tools. NVIDIA reports the RTX 5000 Ada Laptop GPU,
16,376 MiB and driver 580.178.04; GPU discovery is not a compute benchmark.
All four Thunderbolt lane adapters still report `CLd`. VT-d initialization and
IOMMU groups are present; UCSI_GET_PDOS -95 messages persist, along with DMAR
feature-consistency messages. Their presence does not establish the cause of
the missing peer.
[Postboot capability inventory](capabilities-post-bios.txt) confirms the
i9-13950HX's 24 cores/32 threads, approximately 64 GB installed memory, 1 TB
KIOXIA NVMe, VT-x on all 32 logical CPUs, `/dev/kvm` present and 23 IOMMU groups.
These are discovered capabilities, not VM, passthrough or compute benchmarks.

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

The earlier configuration checkpoint passed 34 new tests (29 unit tests plus five Linux Bash
runtime fixtures) and all 20 selected existing Thunderbolt tests, with no
failures or skips. The five fixtures execute the actual generated configuration
logic, replacing only hardware/NM I/O, and verify valid UUID selection, new
profile creation and rejection of static routes, automatic IPv6 and duplicate
names. PSScriptAnalyzer reported zero diagnostics for the changed PowerShell
files. These same 54 tests passed again after the BIOS reboot. Direct SSH
proxy/source-binding options, dry-run/WhatIf, overlapping
routes, native exit failures and timeout termination are covered.

The two new capture helpers pass 21 PowerShell tests and seven Python tests;
the combined scoped total is 82 passing tests, with no selected test failures or
skips. Independent review found and reproduced a late-evidence I/O signal-handler
restoration defect in the Python helper; it was fixed and re-reviewed. Its added
test exercises actual cleanup with four late I/O failures and SIGTERM interruption.
Ruff passes and PSScriptAnalyzer reports zero diagnostics for the Windows helper.
Evidence: [capture Pester results](capture-pester-verification.json),
[postboot profile tests](profile-post-bios-verification.json),
[selected existing tests](existing-thunderbolt-post-bios-verification.json).

Postboot live Status confirmed the absent peer. Live Configure correctly refused before
any network mutation. Actual Configure activation and the iperf3 benchmark
process lifecycle/throughput remain unverified on Thunderbolt hardware. Passing
fixtures do not substitute for that physical validation.

The ThinkPad completed firmware processing and regained LAN SSH; BIOS update
history and effective virtualization settings have been verified. Initial GPU
discovery and Thunderbolt tooling/module persistence also passed. Rebinding only
the NHI driver at 0000:22:00.0 preserved LAN on the separate 0000:48:00.0 USB
controller. The final fixed Python helper captured this operation, restored all
264 changed printing selectors, removed its instance and preserved global
tracing, with no capture or evidence errors. No peer appeared. Evidence:
[rebind result](linux-rebind-result.txt),
[cleanup summary](thunderbolt-diag-20261002-rebind/summary.json).

The user then moved the Apple cable to the other rear ThinkPad Thunderbolt port.
The fresh Linux inventory and five-second Windows router capture still show no
peer, and configuration remains unapplied. Evidence:
[Linux alternate port](linux-alternate-port.txt),
[Windows capture](windows-alternate-port-result.json),
[decoded router state](windows-alternate-port-router-rundown.json).
Remaining physical isolation is an alternate downstream CalDigit port and a
direct computer-to-computer cable path when practical. CalDigit firmware
requires correct hub enumeration in its supported updater and storage-aware
maintenance. A missing peer is not evidence that the Apple cable is faulty or
that Intel Share would fix it. Actual network configuration and performance
validation remain dependent on genuine peer enumeration and carrier.

For rollback, no network profile currently needs removal. If later configured,
remove only `pcai-tb-millylaptop1` and the added Windows 172.31.240.1/30 address.
Remove the owned module-load file only if persistent Thunderbolt networking is
no longer desired; iperf3 is a utility and has no active server service here.

Earlier preparation evidence: [status](status-final.json),
[Linux preparation](linux-prepared-final.txt), [test results](verification.json),
[Windows topology](windows-topology.json), [Linux topology](linux-topology.txt)
and [preparation result](prepare-result.json). These and
`status-before-prepare.json` are historical pre-reboot observations. Use a new
recovered-host capture to establish postboot peer and performance state. The
latest such capture is [status-alternate-port.json](status-alternate-port.json).
Raw ETL/XML, debug snapshots and vendor downloads are retained locally and
ignored by Git; [raw capture custody](raw-capture-custody.json) preserves hashes.
[Normalization custody](text-evidence-normalization.json) records original and
normalized hashes for the listed text artifacts, with their original bytes
retained under the ignored `raw-originals` directory.
