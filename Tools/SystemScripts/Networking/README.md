# Thunderbolt Linux peers

For Windows USB4NET endpoints, use the [private-access installer](../../../docs/usb4/private-access.md)
to set the enrolled adapter's network profile and scoped IPv6 echo policy.

`Invoke-ThunderboltLinuxPeer.ps1` uses an existing LAN SSH alias and verified SSH
host key to prepare an Ubuntu peer, discover a real Thunderbolt peer interface,
configure an isolated /30 and measure the actual link. The maintained wrapper is
`Tools/Invoke-ThunderboltNetworking.ps1 -Mode LinuxPeer`.

```powershell
$peer = 'your-reviewed-profile'
$peerConfig = 'C:/pcai-config/lan-peers.json'
./Tools/Invoke-ThunderboltNetworking.ps1 -Mode LinuxPeer -Peer $peer -ConfigPath $peerConfig
./Tools/Invoke-ThunderboltNetworking.ps1 -Mode LinuxPeer -Peer $peer -ConfigPath $peerConfig -Action Prepare -Apply
./Tools/Invoke-ThunderboltNetworking.ps1 -Mode LinuxPeer -Peer $peer -ConfigPath $peerConfig -Action Configure -DryRun
./Tools/Invoke-ThunderboltNetworking.ps1 -Mode LinuxPeer -Peer $peer -ConfigPath $peerConfig -Action Configure -Apply
./Tools/Invoke-ThunderboltNetworking.ps1 -Mode LinuxPeer -Peer $peer -ConfigPath $peerConfig -Action Benchmark -Apply -IperfPath C:/Tools/iperf3.exe
./Tools/Invoke-ThunderboltNetworking.ps1 -Mode LinuxPeer -Peer $peer -ConfigPath $peerConfig -Action Benchmark -FullDuplex -Apply -IperfPath C:/Tools/iperf3.exe
```

Status and default action planning are read-only. `-DryRun` and `-WhatIf` suppress
all writes and benchmark traffic even when `-Apply` is present. `-h` / `-help`
display help. No report files are created automatically; output is a structured
object that an operator may explicitly export.

Supply a reviewed `-ConfigPath` whose `peers` map includes the explicit `-Peer`.
The bundled [USB4 peer registry](../../../Config/thunderbolt-peers.json) leaves
the `peers` map empty. Its `fleetPeers` entries describe native CLI identities.
For a current LAN deployment, verify the hostname, SSH host key, adapter driver,
carrier and nonconflicting routes before applying. See the
[fleet deployment scope](../../../docs/usb4/fabric-deployment.md#fleet-deployment-scope)
and the [explicit peer test review](../../../Tests/Unit/thunderbolt-linux-peer-entrypoint-review.json).

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
IP method, gateway, DNS, routes, or autoconnect policy. Existing profiles are
selected by UUID. Configure sets `ipv6.method link-local`, `ipv6.never-default yes`
and `ipv6.addr-gen-mode eui64` for the isolated USB4 profile. A stored IPv6 token
blocks this change. The hardware-derived address helps locate the peer. The
live neighbor binding and SSH host key continue to govern identity checks.
An explicit Apply can migrate a matching `disabled` profile after it verifies
that the profile has no stored IPv6 addresses, DNS, gateway, or routes. The
IPv4 address pair still provides the source-bound SSH and benchmark paths.
See the [Linux profile tests](../../../Tests/Integration/ThunderboltLinuxProfile.Tests.ps1)
and the [profile migration](../../../Tests/Integration/thunderbolt-linux-profile-review.json)
and [repeat Configure](../../../Tests/Integration/thunderbolt-linux-idempotency-review.json) review records.
The [EUI-64 review record](../../../Tests/Integration/thunderbolt-linux-eui64-review.json)
covers creation, migration and activation guards.
Run `Invoke-Pester Tests/Integration/ThunderboltLinuxProfile.Tests.ps1` on Linux
to execute the generated Bash against fixture-only NetworkManager commands.
On Windows, set `PCAI_TB_TEST_SSH_ALIAS` to a key-authenticated Linux host.
CI runs this fixture on a disposable Linux runner.
Configure sets no gateway or DNS and uses the configured MTU for Windows IPv4
and the Linux link. A repeat Apply leaves Linux traffic undisturbed when the
stored profile, active connection UUID, live MTU and addresses all match and
the device has no gateway or DNS entries. Activating a profile can interrupt the data link,
so use the LAN management alias for changes. If a later Windows step fails,
the Linux profile can remain configured and the error is surfaced. Rollback
removes only the added Windows /30 address and owned Linux profile. Undo
module persistence only if it is no longer needed.

After applying Configure, the tool verifies SSH over the Thunderbolt address with
the local source bound and checks Linux's return route. Its `ThunderboltSsh`
result includes a usable command that reuses the existing alias's user/key and
LAN host-key identity with strict checking. This leaves the LAN alias unchanged.

Benchmark requires configured addresses and a source-bound Windows route through
the selected adapter. The default uses separate forward and reverse iperf3 trials
against peer-address-bound one-shot servers. `-FullDuplex` runs both directions
simultaneously in one TCP session and checks the direct source-bound Linux return
route before opening the listener. Both endpoints need iperf3 3.11 or newer for
separately named receiver summaries. The duplex result includes the raw JSON,
both tool versions and `ReceiverBitsPerSecond.WindowsToLinux` and `.LinuxToWindows`.
A missing or invalid receiver summary, native failure, route mismatch or missing
hardware prevents a throughput result. The server and local process waits have
deadlines. These are synthetic TCP measurements of the selected link under load.
[Duplex test review](../../../Tests/Unit/thunderbolt-linux-peer-duplex-review.json).
[iperf3 duplex option](https://software.es.net/iperf/invoking.html) ·
[iperf3 3.11 receiver summaries](https://github.com/esnet/iperf/blob/3.11/src/iperf_api.c#L3587-L3605)

The explicit Linux-runtime regression suite executes the generated Bash profile
logic with hardware and NetworkManager I/O fixtures. Run it only against an
authorized Linux SSH alias; it creates and removes temporary fixture files and
never invokes real sudo or NetworkManager:

```powershell
$env:PCAI_TB_TEST_SSH_ALIAS = 'your-authorized-lan-ssh-alias'
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
sudo python3 capture_thunderbolt_peer_trace.py --apply --expected-hostname your-verified-hostname --duration 20 --output-dir "$HOME/.cache/pcai/thunderbolt-trace-new"
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

## Retired ASUS NFS share on Windows

The former fleet NFS share is retired. The [compatibility command](Mount-VigilFleetShare.ps1)
reports `Retired` for status, dry runs and WhatIf. Apply requests to mount or
unmount and requests for a task template fail before contacting a host. Existing
drive mappings and scheduled-task definitions remain under their owners'
custody. The [October Gen7 task inventory](../../../docs/scheduled-tasks/TASK_REGISTRY.md)
preserves the historical task registration.

```powershell
./Tools/SystemScripts/Networking/Mount-VigilFleetShare.ps1 -Action Mount -DryRun
Invoke-Pester ./Tests/Unit/VigilFleetShare.Tests.ps1 -Output Detailed
```

The [retirement test review](../../../Tests/Unit/vigil-fleet-share-retirement-review.json)
binds these checks to the reviewed helper and fixture.

For current QNAP storage, select an explicit signed SMB route, share and drive
letter. The QNAP Windows mapper supports a preview when qualifying a route.
Gen9 verified signed SMB 3.1.1 for the QNAP `net_drive` and DTM-WORK `net-share`
shares. Use the maintained QNAP and Jones runbooks for their respective mappings.
[QNAP Windows mapping](https://github.com/David-Martel/qnap/blob/main/docs/RESILIENCE-RUNBOOK.md) ·
[Jones on-demand mappings](https://github.com/David-Martel/jones_analysis/pull/31)

For PC-AI deployment, use the [LinkHealth installation guide](../../../docs/usb4/link-health.md).
It stages an immutable payload with file hashes on each target before changing
the task or service definition. Shared storage can transport the release
payload, with its accepted revision and hashes verified on the destination.

## Cloud mounts after the cache image

`Start-PcaiCloudMount.ps1` admits an existing read-only rclone provider task in
the same Limited interactive owner session. Configure U or V explicitly; R is
reserved for the dtm-work share. Planning, DryRun, WhatIf and help perform no
Storage, Scheduler, mapping or process probes and write no files.

```powershell
./Tools/SystemScripts/Networking/Start-PcaiCloudMount.ps1 -TaskName Rclone-Dropbox-Gmail-U -Drive U -DryRun
```

Runtime admission requires the expected owner SID, the SHA256 of the exact
exported provider-task XML, provider UNC and cache directory. Inspect the task's
read-only action and principal before deriving its pin. XML hashing uses UTF8
bytes of the `Export-ScheduledTask` string, rather than a BOM-bearing file.
Permanent cache/backing volume GUIDs, the actual image-to-partition association,
backing-file ancestry/identity and free-space reserves must also match. Machine
defaults describe this workstation; override them only with reviewed identities
for another host. A nullable image Number is resolved from its canonical live
PhysicalDrive path, with malformed/conflicting identities rejected.

```powershell
$xml = Export-ScheduledTask -TaskName Rclone-Dropbox-Gmail-U
$pin = [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData([Text.Encoding]::UTF8.GetBytes($xml)))
$sid = [Security.Principal.WindowsIdentity]::GetCurrent().User.Value
./Tools/SystemScripts/Networking/Start-PcaiCloudMount.ps1 -TaskName Rclone-Dropbox-Gmail-U -Drive U -ProviderUnc '\\dropbox-cloud\gmail-u' -CacheDirectory 'F:\RcloneCache\gmail-u' -ExpectedUserSid $sid -ExpectedTaskXmlSha256 $pin -Apply -ValidateOnly
```

ValidateOnly performs real admission but prohibits starting an absent provider.
Ordinary Apply mode requests the existing task at most once, then qualifies its
mapping and owner/process creation identity. Missing or delayed volumes retry
within one monotonic deadline; wrong GUIDs, reparse backing paths, changed task
definitions and mapping collisions are refused. Scheduler, Storage, native
compilation and owner probes run in owned children with capped output and bounded
termination. An uncertain Scheduler start outcome is reported as Hold/Unknown
and is never hidden by restarting or stopping a provider.

The wrapper does not register tasks, change cloud files, move Dropbox, or stop
mounts. Deployment must preserve old source/task XML, bind the reviewed helper
bytes, and validate the actual Limited task namespace. Keep source tests, an
AlreadyRunning probe, first provider start, and subsequent logon/reboot/resume
acceptance as distinct evidence. Changing a provider action requires fresh task
review and a new XML pin.

Portable controls run in `Tests/Boot/CloudMountGate.Tests.ps1`; attached-machine
controls are explicitly in `Tests/LocalMachine/CloudMountGate.Local.ps1`. The
existing repair/startup suites cover the shared nullable-number fix. Neither
suite qualifies an offline dtm-work share or fleet deployment.

`Invoke-PcaiCloudMountGate.ps1` is the maintained scheduled-task receipt runner.
Apply additionally requires the reviewed wrapper/helper SHA256 values and an
existing private local `ReceiptDirectory`. It holds read-only source leases,
checks the Limited owner, and delegates the admission above. Per-drive
`deployment-gate-U-latest.json` / `deployment-gate-U-previous.json` (and V)
retain bounded success, Hold/Unknown and failure results through flushed atomic
rotation. Foreign receipts, reparse paths and competing writers are refused;
failed publication preserves existing receipts. Default, DryRun, WhatIf and
help perform no runtime probes or writes. The runner does not register tasks or
restart providers. Its controls are in `Tests/Boot/CloudMountGateRunner.Tests.ps1`.
