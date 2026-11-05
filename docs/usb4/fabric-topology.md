# USB4 topology without a sharing-product dependency

Scope, actual observation results and deployment limits are recorded in the
[maintenance review](../../Reports/usb4-topology-maintenance.md); remaining hardware
qualification is tracked in [boot.TODO.md](../../boot.TODO.md#usb4-topology-observation-and-hot-plug-planning).

Use Windows' USB4NET transport and existing authenticated sharing services. The
working Thunderbolt Share session is a useful connection reference; the planner
does not require that application, its licensing metadata or its private APIs.
Microsoft documents [native USB4 interdomain networking](https://learn.microsoft.com/windows-hardware/design/component-guidelines/usb4-interdomain-connections).
Intel describes the separate [Share product requirements](https://www.intel.com/content/www/us/en/support/articles/000099003/software.html).

Build and test the existing native CLI from the repository root:

```powershell
cargo build --manifest-path Native/pcai_core/Cargo.toml -p pcai-tb --bin pcai-tb
cargo test --manifest-path Native/pcai_core/Cargo.toml -p pcai-tb fabric_plan
uv run --no-project python Tests/Portable/test_usb4_fabric_cli.py --exe PATH_TO_BUILT_PCAI_TB
./Tools/SystemScripts/Networking/Get-Usb4FabricObservation.ps1 -DryRun
```

The [Windows collector](../../Tools/SystemScripts/Networking/Get-Usb4FabricObservation.ps1)
queries current adapters and present PnP ancestry. It retains interface GUID, MAC,
USB4NET instance, root router, local index, addresses and neighbor-cache records.
Phantom registry entries and display names do not select devices. A missing root
stays unknown. Provider errors fail collection rather than reporting an empty
successful inventory. Help and DryRun invoke no hardware provider or child.
Normal collection uses the existing [hidden bounded runner](../../Tools/SystemScripts/Networking/Invoke-VigilBoundedProcess.ps1).
Normal output is bounded by interface/address/neighbor cardinality and checked
against 256 KiB; the shared runner's text readers do not impose a streaming memory
cap on exceptional provider output. No report is written automatically.

Collect a second host through an existing independently trusted management
channel. Export each returned object as JSON only to an explicitly chosen private
file. Combine the two `hosts` arrays into one observation envelope with schema
`pcai.usb4-fabric-observation/v1`, one `ttl_seconds` value in 30–300, and the
original per-host observation times. Do not refresh timestamps on cached data.
The `management-ssh` source label records collection provenance; the JSON field
does not authenticate a peer. Do not expose SSH keys, credentials or share content
in these records.

```text
pcai-tb topology-plan --input current-observation.json --compact
pcai-tb topology-plan --input next-observation.json --previous previous-plan.json
```

The [planner](../../Native/pcai_core/pcai_tb_cli/src/fabric_plan.rs) accepts explicit
bounded input, never discovers hosts or changes networking. Its
[CLI entrypoint](../../Native/pcai_core/pcai_tb_cli/src/main.rs) refuses Apply,
configuration overrides and output-file writes for this command. A segment needs
fresh, up USB4 endpoints and unique reciprocal neighbor MACs whose addresses
belong to the observed remote interfaces. This is `ObservedUnverified`, not
cryptographic authentication or application readiness. IP address, alias, service
banner or a running sharing application alone cannot establish the segment.

Each segment retains both endpoint identities. Link-local connection targets use
the outgoing endpoint's interface index, not the peer's zone number. Edge IDs
remain stable when aliases or indices change; a new GUID, MAC or root creates a
different edge. Expired, future, down or missing observations cannot carry the old
edge forward. Differences from `--previous` are withdrawn **observation IDs**;
they never remove OS routes, shares or vendor state. Recollect after hot-plug and
evaluate TTL on every consumption. This command is a projection, not a deployed
event service or an unattended provisioning controller.

APIPA prefixes on parallel point-to-point interfaces can overlap. The planner
warns rather than treating those links as one LAN. A shared USB4 router tree also
does not prove a shared Ethernet segment. Preserve independent management access;
never bridge a USB4 path to Work, QNAP or UniFi solely from physical adjacency.

## Promotion to a configured connection

1. Preserve the working Share session, existing addresses, routes and management
   path. Capture fresh endpoint identity before proposing changes.
2. Qualify the chosen service through the exact source/interface: a TCP 445
   handshake proves transport availability, not SMB identity, authentication,
   share permissions or successful file transfer. An SSH timeout remains unknown
   until listener, firewall and source-policy evidence resolves it.
3. Use the existing [enrolled policy](enrolled-observer.md) and
   [Windows ownership/recovery contract](windows-boot.md) for reviewed addresses,
   freshness, custody and revocation. The current Windows reconciliation helper
   consumes a Linux enrolled observer; Windows-to-Windows enrollment parity is
   still required before automatic address adoption. Do not label a topology plan
   as that implementation.
4. Admit a disjoint per-edge IPAM allocation, authenticated bilateral identity,
   direct/return route, service permissions and explicit owner before applying
   configuration. Keep default routes, DNS and foreign interface/share state.
   A bridge or transit role needs its own loop/domain and return-path review.

The [CLI controls](../../Tests/Portable/test_usb4_fabric_cli.py) exercise the actual
binary's bounded input, no-write planning, Apply refusal and disconnected-edge
differences. Native planner fixtures cover alias changes, replacement devices,
freshness, local link-local scope and parallel APIPA segments. The
[collector controls](../../Tests/Portable/Test-Usb4FabricObservation.ps1) separate
provider failures, absent devices and unknown roots. These are software controls;
physical unplug, sleep, reboot and newly started application sessions require
their own machine evidence. Nominal Windows link speed is not measured throughput.
