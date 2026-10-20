# `pcai_tb_cli` (`pcai-tb`)

This crate imports Windows and Linux Thunderbolt/USB4 and SFP+ switch control-plane observation work for review. Its source custody, exact revisions, observed limitations and isolated Linux/Windows QA are recorded in `source-custody.json`. It is not an upstream release or a qualified automatic network provisioning service.

## Install, build, test and run

From the `Native/pcai_core` workspace:

```bash
cargo check -p pcai-tb --all-targets
cargo clippy -p pcai-tb --all-targets -- -D warnings
cargo test -p pcai-tb
cargo install --path pcai_tb_cli --locked --bin pcai-tb
pcai-tb status --compact
```

## Subcommands

```bash
pcai-tb status --compact
pcai-tb identity --computer minisforum --peer millylaptop1 --registry ../../../Config/thunderbolt-peers.json --compact
pcai-tb discover --timeout-ms 220 --registry ../../../Config/thunderbolt-peers.json --compact
pcai-tb register --peer millylaptop1 --registry ../../../Config/thunderbolt-peers.json --compact
pcai-tb switch --registry ../../../Config/thunderbolt-peers.json --compact
pcai-tb verify-custody --registry ../../../Config/thunderbolt-peers.json --compact
pcai-tb observe --registry ../../../Config/thunderbolt-peers.json --compact
pcai-tb vnv --registry ../../../Config/thunderbolt-peers.json --output vnv-evidence.json --compact
```

`status`, `switch`, `verify-custody`, `observe`, and `vnv` are strictly read-only by default. Route mutation (`--apply-routes`) additionally requires `PC_AI_ALLOW_ROUTE_MUTATION=1` and fails closed on pre-existing conflicting routes.

## Failure contracts and tests

`tests/registry_cli.rs` invokes the actual CLI executable. It protects the entrypoint against malformed JSON, unknown schemas, duplicate fleet addresses and missing explicit configuration; these must fail before producing success JSON or installing files. A valid configuration is a positive control. `src/configure.rs` tests failed file copies and failed OS action reports. `src/platform/linux.rs` tests missing, negative, overflowing and malformed speed measurements. The existing integer JSON schema uses zero for unknown speed; zero is not observed throughput.

Unknown link speed must not become an invented 20Gb/s measurement. Failed installation copies propagate an error. Applied registration emits its action report then exits nonzero if an action failed.

Headless Windows identity falls back to `GetComputerNameW` when process hostname
variables are missing or blank. Registration with no connected USB4 adapters
reports `OverallSuccess=false` and an unapplied `RegistrationPresenceGate` error.
`register --apply` additionally requires the resolved local fleet identity to
match the OS machine identity, independently of environment overrides. It rejects
`--computer`; explicit host selection remains available for `identity` inspection.
These gates run before peer discovery and adapter mutation. Ordinary dry-run
inspection remains available for unknown machines.

All Windows child commands in configuration, lifecycle, switch observation and
the legacy benchmark use [the background command helper](src/platform/mod.rs),
which sets `CREATE_NO_WINDOW`; Linux commands retain their existing behavior.
[CLI regressions](tests/registry_cli.rs) test a real sanitized child process and
a real child with no console window; [configuration regressions](src/configure.rs)
cover absence, identity mismatch, unknown identity and admitted matching identity.
The [runtime test review register](tests/runtime-review.json) records the exact
source reviewed. These controls do not qualify physical unplug, reboot or sleep.

`src/switch.rs` tests fail-closed evaluation for:
- Exact Linux custody `ifalias` equality (`ms01-switch:<owner>` and `ms01-usb4:<owner>`), rejecting prefix collisions (`ms01-switch:<owner>-extra`), wrong `phase`, missing displaced interfaces, or mismatched veth peer `ifindex`/`link_index`.
- Shared `flock(LOCK_SH | LOCK_NB)` acquisition and root-owned non-symlink directory/file mode verification before reading `/var/lib/ms01-network-switch/current.json` and `/var/lib/ms01-network-switch/usb4-milly/current.json`.
- Read-only invocation and validation of `/usr/local/sbin/ms01-usb4-extension check` (`schema: "vigil.ms01-usb4-check/v1"`), distinct from PC-AI's internal `PcaiLinuxUsb4Observation` (`schema: "pcai.linux-usb4-observation/v1"`).
- Exact IPv4 route destination + netmask + gateway verification (`Active Routes:` on Windows, `ip -4 -j route show` on Linux) and exact IPv6 address token matching on Windows (`::9` does not match `::99`).
- Fail-closed Windows interface metric hierarchy checks when `entries`/`connected` is empty or any connected `usb4-p2p`/`wired-lan`/`wifi` interface has `metric: None`.
- Pending TPM intake gates (`PENDING-FABRIC-F09..F13`, `tpm_admission_status: "PENDING_TPM_INTAKE"`, `acceptance_credit: false`) using the closed verdict enum (`PASS`, `FAIL`, `SKIP_NOT_APPLICABLE`, `BLOCKED_MISSING_PROVIDER`, `HOLD_PENDING_TPM_INTAKE`).

## Deployment limits

Linux currently reports hard-coded interface metrics; MTU configuration does not set the requested metric. Registration is temporary `ip` configuration, without NetworkManager ownership or reboot persistence. Windows metric changes likewise require review. Do not deploy automatic apply until those contracts, strict CLI argument validation, cancellation/timeouts, identity authentication, route rollback and management-path preservation are implemented and tested.

No throughput, reconnect, DHCP forwarding, bridge-loop protection, or hardware acceptance is established by the unit tests. Discovery hints and SSH/NTLM banners are not authenticated host identity. Benchmark collection is not a link throughput benchmark.

## Source custody and review follow-through

[source-custody.json](source-custody.json) preserves baseline hashes from both Windows development variants alongside the current reconciled candidate file hashes. Files in this candidate include subsequent Linux/Windows reliability fixes and review blocker resolutions, so baseline hashes are historical provenance while `reconciled_candidate` records the current file hashes.

Future addressing work should prioritize explicit IPv6 link-local interface scopes and an allocated fleet ULA prefix, with IPv4 fallback where needed. Do not infer a trusted peer from an unauthenticated hostname, hard-code new addresses, or bridge independent upstream networks without loop/domain admission checks.

## Shared Windows/Linux topology contract

The [fleet control-plane proposal](https://github.com/David-Martel/vigil-spark/pull/2804)
and [namespace observation integration](https://github.com/David-Martel/vigil-spark/pull/2805)
define the current integration direction. These are reviewed candidates, not a
claim that this CLI already implements their complete provisioning workflow.

Windows/Linux providers must distinguish approved host identity, replaceable
adapter instance, current interface/namespace/zone, L2 segment, L3 route and
service/transit authorization. Link-local discovery is per-segment; it cannot
resolve an intermediate routed hop by itself. A hub/router tree is not proof of
an Ethernet path. Parallel SFP and USB4 paths require explicit cycle/return-route
checks before creating any bridge or forwarding grant.

Extend the same observation/plan/verify contract to available Wi-Fi AP/P2P radios,
Bluetooth/BLE service hints and platform diagnostics. Read actual driver binding,
firmware, interface modes, concurrency/regulatory limits and effective permissions.
Do not claim AP readiness from USB enumeration or station bridging without
compatible four-address support. Windows and Linux need separate native providers
for Wi-Fi, firewall, capture drivers and privilege grants; portable contracts do
not imply identical implementation or automatic NAT/bridge equivalence.

Approved provisioning should generate minimal owned group/device/helper/firewall
changes, retain source epoch and accounting, preserve independent management,
verify loaded state and support explicit revocation. Capture plans bind exact
asset/bus/namespace, lease, filter and duration/size limits; usbmon, USBPcap, Npcap,
ETW and packet capture have distinct capability and permission states. Discovery
never starts unrestricted payload collection or retunes an active sensor radio.

Preserve the read-only MS01 schema's structural scope (`network_ready: null`),
fail-closed registry behavior and isolated per-variant build targets. New
traceability IDs (`PENDING-FABRIC-F09..F13`) remain pending TPM admission
(`acceptance_credit: false`); USB4 evidence cannot credit RDMA-specific gates.
