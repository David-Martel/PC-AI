# USB4 fleet source, releases and deployment

PC-AI is the maintained executor code home. A fleet atlas, agent-bus, VIGIL
application or an AI session is not required by the standalone Windows peer
controller or the native Linux peer agent. Optional fleet observers may publish
their receipts; they must not become a prerequisite for basic peer networking.

## Source of truth

| Item | Maintained location |
| --- | --- |
| Fleet catalog and support boundaries | `Config/usb4-fleet/fleet.json`, `fleet.schema.json` |
| Approved Windows host policies | `Config/usb4-fleet/hosts/dtm-p1gen7.json`, `dtm-p1gen9.json` |
| Existing Linux enrolled policy snapshot | `Config/usb4-fleet/hosts/millylaptop1.json` (epoch2, unchanged; existing protection/unknown-controller holds remain) |
| Windows controller and installer | `Tools/SystemScripts/Networking/Invoke-Usb4PeerNetwork.ps1`, `Install-Usb4PeerNetwork.ps1` |
| Read-only Windows deployment/status command | `Tools/SystemScripts/Networking/Get-Usb4PeerDeployment.ps1` |
| Source-bound release packaging | `Tools/SystemScripts/Networking/usb4_fleet_release.py` |
| Native Linux agent, identity, lifecycle and NM/networkd backends | `Native/pcai_core/pcai_tb_cli/` |
| Native Linux deployment templates | `Deploy/usb4/` |
| Linux enrollment/adoption and status contracts | `docs/usb4/enrolled-observer.md`, `docs/usb4-multiedge-reconciliation.md` |
| Detecting regression controls | `Tests/Portable/Test-Usb4PeerNetwork*.ps1`, `test_usb4_fleet_release.py` |

The repository is `David-Martel/PC-AI`. Runtime commits and CI follow-ups must
remain distinguishable from a merged default branch. Use the release manifest's
full source commit and SHA-256 inventory, not a remembered branch name. Actual
observations in the fleet catalog are dated snapshots, not declarations that a
host is currently reachable or ready. Runtime state and credentials stay out of
the desired policy files.

The two host policies preserve the approved IPv4 sharing and IPv6 peer intent.
Their source formatting is normalized to LF; a policy hash may therefore differ
from an older locally prepared JSON with equivalent data. The installer binds
the exact new policy bytes and preserves the previous revision. Do not insert
catalog metadata into the schema1 controller policy: unknown keys are refused.

## Reproducible packages

The release publisher uses Python's standard library. It does not SSH to hosts,
change routes or install services. Run from a committed checkout with Python 3:

```powershell
uv run --no-project python Tools/SystemScripts/Networking/usb4_fleet_release.py validate
uv run --no-project python Tools/SystemScripts/Networking/usb4_fleet_release.py build --host dtm-p1gen9 --output C:/approved/pcai-usb4-dtm-p1gen9.zip
```

Read `--help` for the exact supported arguments. The output parent must already
exist and the artifact must not exist. A build refuses untracked or modified
selected payloads/policy/catalog, unsafe paths, symlinks and source mismatch.
Only pure CRLF checkout differences for positively identified Git text inputs are accepted; attribute evidence is Git's effective checkout policy, including local/global/system overrides. Artifacts always contain the original committed Git blobs. Substantive edits, binary/unknown attribute mismatches and general clean/smudge transformations are refused. Unrelated work is preserved. The archive records the exact Git commit, host,
platform, architecture, payload kind, policy and all payload hashes. Retain that
manifest beside each installed deployment and in the controlled release store.

`windows-peer` is a deployable script payload. Its supported source architectures
do not prove that Windows ARM64 USB4 drivers or PowerShell on a particular ARM64
host have been qualified. `linux-native` packages are explicitly templates-only:
they do not contain an unverified or wrong-architecture executable. Build the
native agent from the locked workspace for its actual target, bind the binary
hash to source/toolchain/build evidence, and then follow the existing native
adoption procedure. A macOS package is inventory/design material, not an
implemented macOS executor. No 32-bit USB4 target is qualified.

Private keys, tokens, credential files and raw application/health data do not
belong in a release. Use existing host secret storage and pinned SSH identities.

## Windows deployment and verification

Verify archive hashes before extraction into a new controlled release directory.
From that extracted tree, use elevated PowerShell 7:

```powershell
$installer = './Tools/SystemScripts/Networking/Install-Usb4PeerNetwork.ps1'
$policy = './Config/usb4-fleet/hosts/dtm-p1gen9.json'
& $installer -PolicyPath ([IO.Path]::GetFullPath($policy)) -DryRun
# Bind ExpectedControllerSha256 to the reviewed archive manifest value.
& $installer -PolicyPath ([IO.Path]::GetFullPath($policy)) -ExpectedControllerSha256 '<reviewed hash>' -ReleaseManifestPath ([IO.Path]::GetFullPath('./manifest.json')) -Apply -ReplaceLegacyTasks
& $installer -Verify
./Tools/SystemScripts/Networking/Get-Usb4PeerDeployment.ps1
./Tools/SystemScripts/Networking/Get-Usb4PeerDeployment.ps1 -ProbeInternet
```

The installer stages immutable `rN` revisions under
`C:\ProgramData\pcai-usb4\peer-network` and registers
`\PC-AI\Usb4PeerNetwork`. `installed.json` points to the active controller,
policy, state and report. `owned-state.json` is the mutation/rollback journal;
`current.json` is the latest attempt, which may be in progress. Previous
revisions and exact task XML preimages remain available. Registration verification
is distinct from an actual completed SYSTEM run and end-to-end traffic proof.
The optional release manifest is validated against this host, OS architecture and
the controller/installer/policy bytes before installation, then retained in the
immutable revision. Its location/hash and source declaration are recorded in
`installed.json` and rechecked by `-Verify`. The status command distinguishes
byte-matched provenance declarations from cryptographic source authenticity;
verify trusted source and artifact signatures/hashes separately.

The status command reads real routing, installed hashes, task state and policy
identity. `-ProbeInternet` additionally opens a fresh unbound TLS/HTTPS connection
and reports the OS-selected source; the default command sends no probe traffic.
It makes no configuration changes or report files. Keep its JSON as a deployment
receipt by explicitly redirecting stdout to controlled storage when needed.

Windows UI/NCSI connectivity labels are not a routing-table selection proof.
Compare effective default metrics (route plus interface), the selected source
for an ordinary destination, and fresh traffic. Existing source-bound sockets,
corporate/VPN destination routes or application policy can still use another
interface. Preserve those paths and distinguish them from the general default.
An owned host route to a health endpoint proves that shared path only; it cannot
prove that the ordinary default also prefers USB4.

## Linux deployment and host acceptance

Reuse the existing native agent and its policy/node identity; do not install a
second coordinator. The usual installed paths are:

- `/usr/local/libexec/pcai-usb4-agent`: target-native executable.
- `/etc/pcai-usb4/policy.json`: root-owned enrolled policy with monotonic epoch.
- `/etc/pcai-usb4/node-id`: stable node identity, when provisioned there.
- `pcai-usb4-observer`, `pcai-usb4-nm` or `pcai-usb4-networkd`, and
  `pcai-usb4-qualify` systemd units/timers from `Deploy/usb4/`.
- `/run/pcai-usb4-status/`: private atomic runtime status records.

Follow the native documentation for the exact node-ID search locations and unit
arguments. Policy backend selects NetworkManager or networkd with explicit
profile custody. Interface names/indices are observations; controller allowlists
and authenticated node identity decide enrollment. A kernel carrier or a
NetworkManager label alone is not an authenticated usable peer. Source-bound
SSH qualification, route choice, freshness and owned-profile readback matter.

Linux peer-only profiles are not an internet-sharing implementation. In
particular the maintained networkd renderer is IPv6-only and disables RA/DHCP;
it must not be presented as IPv4+IPv6 internet sharing. A future Linux sharing
backend needs its own explicit provider/consumer roles, upstream selection,
firewall/NAT or routed-prefix ownership, DNS behavior and detecting rollback
tests. Optional atlas publication may observe it but cannot grant trust or
override the executor's refusals.

## Hardware changes, IPv6 and platform boundaries

Do not key desired configuration to a transient port name or interface index.
The Windows controller rediscovers the enrolled MAC and scopes link-local IPv6
to the current index. It still refuses a replacement adapter GUID/MAC, changed
role or foreign ownership. The native Linux policy supports stable node IDs,
controller UUID allowlists, epochs and multiple enrolled peers. An unrecognized
controller is a candidate for reviewed enrollment, not automatic trust.

For replacement hardware: capture old ownership/state and current identity,
prove the new node through an authenticated management path, allocate/check
addresses, review the desired policy diff and epoch, and perform an explicit
custody migration. Preserve old configuration and rollback evidence. Never
delete the journal or widen an allowlist to force readiness. A general automated
Windows multi-peer/adapter-replacement migration is not yet implemented.

IPv4 and IPv6 have separate readiness. Link-local discovery and ULA peer traffic
do not establish IPv6 internet. A genuine IPv6 internet provider needs a usable
upstream route, valid downstream prefix/delegation/routing, forwarding/firewall
ownership and end-to-end proof. Do not invent a global prefix, relay an arbitrary
LAN RA, disable IPv6 globally, or advertise `::/0` merely to change a tray icon.
On the audited Gen7/Gen9 topology the verified internet-sharing backend is IPv4;
there is no routed IPv6 internet default. Local IPv6 is retained.

Windows x86_64 currently has the two named live deployments. Linux x86_64 has an
existing native deployment on Milly, whose current carrier/enrollment and binary
provenance gaps are recorded in the catalog. The ARM64 Sparks have active USB
Ethernet fabric links, not qualified native Thunderbolt XDomain deployment.
Windows ARM64, Linux ARM64 agent artifacts and macOS need independent runtime
acceptance. Do not copy an x86_64 native binary onto ARM64 and call it deployed.

## Operational gates and recovery

Before changing an active fleet path, refresh exact source/policy/artifact hashes,
ownership, physical carrier/controller identity, routes, namespaces, NFS and
workloads. Use an independent pinned management path. Preserve HIL/runner/NFS
holds. Failed SSH is not an inactive-owner release. Stage and inspect a concrete
revision before applying; verify hash readback, actual SYSTEM/systemd execution,
peer traffic, ordinary default selection and failure withdrawal separately.

Reboot, resume, cable removal, physical upstream loss and soak need separate
operator-timed qualification when workloads can tolerate them. Do not reset
services, USB controllers or networking automatically after an unknown failure.
Retain `HOLD`, `NOT_RUN`, `UNVERIFIED` and pending custody rather than fabricate a
passing fleet state. Existing last-attempt records can be in progress; stale
success must not override current carrier/route evidence.

ASUS currently has a VIGIL-fabric internet default ahead of its direct LAN;
that observation is not a generic independent deployment. Milly has a protected
interface, changed controller identity and active hard-NFS recovery. Source and
templates can be staged independently; routing or enrollment changes on those
machines require fresh custody and workload-preservation checks.
