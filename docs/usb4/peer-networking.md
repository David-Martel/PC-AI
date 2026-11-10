# Standalone USB4 peer networking

The generic peer controller is independent of VIGIL, agent-bus, LLMs, native
PC-AI binaries and PowerShell profiles. A working peer link does not require the
separate lab observer or its clinical/hardware acceptance records to pass.
Keep those records separate; this controller does not qualify clinical systems.

## Enrollment and modes

Start with [the example policy](../../Config/usb4-peer-network.example.json).
Replace the example host, adapter MAC, peer MAC and addresses using observed
hardware and an independently verified management path. MAC enrollment limits
the selected physical link; it is not cryptographic authentication. Use pinned
SSH host keys or another authenticated application protocol for data access.
Unknown peers are not automatically enrolled from banners, NTLM names or DNS.

The adapter is rediscovered from its enrolled MAC on each run, so aliases and
interface indices may change. Link-local IPv6 addresses acquire the current
interface scope when used. The existing machine-level `thunderbolt-peers.json`
is neither replaced nor downgraded: this component uses its own explicit policy.

| Mode | Behavior |
| --- | --- |
| `Peer` | Enrolled direct connectivity and private transport addresses; no default gateway or DNS takeover. |
| `Provider` | Opt-in internet sharing from a healthy eligible wired uplink, using ICS or an explicitly reserved existing WinNAT child. |
| `Consumer` | Opt-in shared internet, with source-bound proof before an owned default route/DNS promotion. |

Provider/Consumer policies also supply `sharing`:

```json
{
  "providerAddress": "192.168.137.1",
  "consumerAddress": "192.168.137.2",
  "prefixLength": 24,
  "preferSharedInternet": true,
  "routeMetric": 10,
  "probeAddress": "1.1.1.1",
  "probePort": 443,
  "allowWirelessFallback": false
}
```

Explicitly configure complementary roles on the two endpoints. A provider must
not select its downstream peer or a VPN as an upstream. Consumer Wi-Fi and
management routes are preserved as recovery paths. Readiness is tied to the
observed interface and source address; an unbound hostname request through Wi-Fi
is not proof of USB4 sharing. Inspect `peerReady` and `internetReady` separately.

An uplink qualifies only when the current route to the probe destination selects
that interface. The controller does not rewrite other uplinks' defaults to bypass
a still-selected but unhealthy route; automatic reselection depends on OS route
changes. Losing the enrolled peer or a healthy approved upstream withdraws only
the controller-owned sharing pair.

Policy roles are fixed for an existing ownership record. Changing a deployed
`Peer` policy into `Provider` or `Consumer` requires explicit custody review;
deleting state to bypass that check is unsafe. An interrupted transaction retains
its pending journal and refuses further mutation until the preimage and current
state have been reconciled. Adapter incarnation changes likewise require review.

The [controller](../../Tools/SystemScripts/Networking/Invoke-Usb4PeerNetwork.ps1)
admits persisted state before collecting an initial OS snapshot. A missing file
permits bootstrap. An existing blank, zero-filled, null, non-object or malformed
JSON record is refused in `ReadPolicyAndState` with an explicit state-admission
error. A present `pendingTransaction` must remain an object. Refusal preserves
the journal bytes; it never treats unreadable custody as a new enrollment.
An explicit JSON `{}` remains readable because successful rollback to a missing
state preimage publishes that legacy empty-object record.
Preserve the damaged file and any independently verified backup. Restoring a
backup that contains an unfinished rollback does not resolve that rollback or
authorize another Apply. These guards do not establish how corruption occurred
or qualify durable state publication under power loss.

Windows ICS supports one public/private sharing pair. A foreign existing pair
must be refused rather than stolen. WinNAT permits one internal prefix; this
controller must not create a competing NAT or remove WSL/Hyper-V/container NAT
state. ICS may assign its private interface address, so inspect readback rather
than assuming the preexisting USB4 IPv4 address survived. APIPA is bootstrap
only and must not be forwarded as an internet-sharing subnet.

### Sharing backends

`Ics` is the default sharing backend. An ICS event-subscription error is retained
as a real failure; it does not authorize resetting services, deleting existing
NAT objects, or silently choosing another backend.

An explicitly approved `ExistingWinNat` policy can reserve a disjoint child
subnet inside an existing NAT prefix. For example, a provider/consumer pair can
use the following sharing configuration after its own allocation audit:

```json
{
  "backend": "ExistingWinNat",
  "natName": "ExistingApprovedNat",
  "natPrefix": "172.16.0.0/16",
  "providerAddress": "172.16.252.1",
  "consumerAddress": "172.16.252.2",
  "prefixLength": 30,
  "dnsServers": ["1.1.1.1"],
  "preferSharedInternet": true,
  "routeMetric": 10,
  "probeAddress": "1.1.1.1",
  "probePort": 443,
  "allowWirelessFallback": false
}
```

The named NAT and prefix must already exist. HNS/IPAM allocations, endpoints,
static mappings, sessions, host addresses and existing reservations must leave
the child range available. Missing or ambiguous inventory prevents allocation.
This backend never creates, deletes, renames or reconfigures the NAT object,
changes WSL/HNS pools, supplies DHCP, or runs a DNS proxy. The provider owns only
its secondary peer address and specifically approved IPv4 forwarding changes.
The consumer uses explicit DNS and proves authenticated HTTPS and DNS through the peer before
promoting its owned route and DNS settings.

For `ExistingWinNat`, provider configuration success is not peer internet proof:
`sharingConfigured` and `upstreamInternetReady` describe the provider, while
`internetReady` remains false until evaluated on the consumer. Qualify the
consumer's bound DNS/HTTPS and the matching kernel NAT session together with
unchanged original NAT/WSL configuration. VM/container NAT documentation alone
does not qualify a physical USB4 path on a particular Windows build.

## Planning and installation

Use PowerShell 7 in an elevated session for Apply. Paths must be explicit and
absolute. Planning/DryRun creates no files, receipts, tasks or networking state.

```powershell
$installer = 'C:\source\PC_AI\Tools\SystemScripts\Networking\Install-Usb4PeerNetwork.ps1'
& $installer -PolicyPath 'C:\approved\peer-policy.json' -DryRun
& $installer -PolicyPath 'C:\approved\peer-policy.json' -Apply -ReplaceLegacyTasks
& $installer -Verify
```

The installer copies the standalone controller and exact policy bytes into
`C:\ProgramData\pcai-usb4\peer-network\rN`, verifies SHA-256 readback, restricts
the runtime to SYSTEM/Administrators, and records the installed revision in
`installed.json`. `-ExpectedControllerSha256` binds an approved controller hash.
Revision directories use stable sequence names; observation times are metadata.
Previous revisions and task XML preimages remain available for rollback.

### Optional release binding

Supply `-ReleaseManifestPath` only for an independently approved deployable manifest. Planning checks its schema, declared Git repository/full commit, current host/platform/OS architecture, and the hashes/sizes of the supplied controller, current installer source and exact policy bytes **before dot-sourcing the controller**. `-ExpectedControllerSha256` remains an independent early approved-hash check. Input sources must stay trusted and stable; path-based loading does not close every mutable-file race or authenticate an arbitrary manifest.

The installer shape-checks all `files[]` records, but matches bytes only for those three inputs. It does not install or qualify arbitrary additional payloads. Source repository/commit fields are declarations, not proof of repository custody or a signature. Bound plans, installed metadata and Verify report `releaseAuthenticity` as NotVerified. Installation and verification still leave RuntimeQualified false.

Planning/DryRun creates no release copy, revision or task. Apply revalidates the binding, preserves an exact immutable `release.json`, checks staging readback, and records source/hash metadata alongside existing rollback preimages. Verify checks the retained manifest and installed controller/policy binding. Changed source/manifest, malformed duplicate policy scalars or inconsistent readback refuse the operation; failed publication retains existing recovery behavior. Existing unbound use and previously bound manifests remain compatible.

The [release-binding controls](../../Tests/Portable/usb4-release-binding-review.json) cover malformed/foreign declarations, controller execution ordering, post-plan drift, immutable readback and failed-install recovery. They use owned temporary files and injected task/ACL operations. They do not qualify a scheduled runtime or physical link.

This restores only the installer prerequisite from preserved source. It does not deploy an older controller over an installed newer release, clear a pending journal, isolate another link, or resolve draft233's separate coverage/integration hold. Reconcile installed provenance, recovery ownership and per-edge resource coordination before promoting an additional Peer enrollment.

The new `\PC-AI\Usb4PeerNetwork` task runs SYSTEM/Highest, with boot, network
connect/disconnect, resume and one-minute fallback triggers, IgnoreNew overlap control and
a two-minute runtime limit. It uses absolute PowerShell/controller/policy/state
paths and no profile. The installer initially registers it disabled, preserves
recognized old-task XML/state, optionally disables exactly the old
`PC-AI-ThunderboltPeerAgent` and `\PC-AI\Configure-ThunderboltSwitchRoutes`
tasks, then enables the new task. No other task is retired. Do not disable an
old enrolled-peer service if it still owns a different required physical link.
An active or queued old/new controller causes a retryable refusal; installation
does not stop it. Fresh task snapshots are checked before replacement/disable,
and a legacy controller must be disabled and quiescent before the new task is
enabled. Unknown execution states are treated as active.

Installation and `-Verify` prove installed bytes and task configuration, not
networking. Qualify an actual scheduled SYSTEM run and its receipt on both
hosts. A failed installation attempts task/configuration rollback; retained
revision or staging artifacts and any rollback errors remain explicit. Inspect
the retained preimages before restoring anything after an ownership change.

## File-only deployment inspection

Run the read-only inspector against the intended local installation root before
planning an upgrade or investigating a failed controller run:

```powershell
pwsh -NoLogo -NoProfile -NonInteractive -File Tools/SystemScripts/Networking/Get-Usb4PeerDeployment.ps1 -InstallRoot C:/ProgramData/pcai-usb4/peer-network
pwsh -NoLogo -NoProfile -NonInteractive -File Tests/Portable/Test-Usb4PeerDeployment.ps1
```

The inspector needs PowerShell 7 or newer and reads installed metadata, payload
hashes, the recorded release manifest, pending recovery state and the historical
controller report. It neither executes the controller nor queries live adapters,
routes, tasks or services. Those fields remain `NotCollected`; use the maintained
[observation collector](../../Tools/SystemScripts/Networking/Get-Usb4FabricObservation.ps1)
separately for fresh OS observations. It sends no active peer or Internet probe.

Malformed or remote pointers, unsafe path traversal, reparse points, invalid
manifest shapes and changed declarations produce explicit unknown/refusal
results. The inspector checks path components from the filesystem root toward
the leaf, refusing unsafe ancestors before querying descendant metadata. A future or invalid timestamp has no freshness verdict. A pending
rollback remains pending. Manifest byte agreement leaves authenticity
`NotVerified`; historical readiness is not current link qualification.

The [doctor controls](../../Tests/Portable/usb4-deployment-doctor-review.json)
bind the source and detecting fixtures. Windows and Linux CI exercise temporary
files and boundary doubles without changing live task or network configuration.
Linux inspection does not establish a Windows deployment or implement a Linux
installer. Path checks and later reads are not an atomic no-follow transaction;
use a trusted installation root, and retain the source/recovery evidence before
any deployment decision.

## Qualification and performance

Run the existing fixture contracts from the repository root:

```powershell
pwsh -NoLogo -NoProfile -File Tests/Portable/Test-Usb4PeerNetwork.ps1
pwsh -NoLogo -NoProfile -File Tests/Portable/Test-Usb4StateAdmission.ps1
pwsh -NoLogo -NoProfile -File Tests/Portable/Test-Usb4StatePublication.ps1
pwsh -NoLogo -NoProfile -File Tests/Portable/Test-Usb4JournalRetirement.ps1
pwsh -NoLogo -NoProfile -File Tests/Portable/Test-Usb4PeerNetworkInstall.ps1
```

The [USB4 workflow](../../.github/workflows/usb4-lifecycle.yml) runs these suites
on Windows and Linux. Network and scheduler mutations use fixtures; installer
controls create and clean their own temporary files. Windows-only task-schema
and file-ownership controls explicitly report unsupported platforms. These
contracts do not install a runtime or qualify a physical connection.
The [state-admission controls](../../Tests/Portable/Test-Usb4StateAdmission.ps1)
exercise the actual entrypoint read against real files and retain unknown fields
and pending custody. A bounded Windows child also verifies failure before the
observation phase and no DryRun writes; Linux reports that child as `NOT_RUN`.
The [independent review](../../Tests/Portable/usb4-state-admission-review.md)
and [selector register](../../Tests/Portable/usb4-state-admission-review.json)
record exact source bindings, the resolved rollback compatibility finding and
the limits of the fixture evidence.

State publication validates the exact captured predecessor bytes, preserves each
different valid generation under `owned-state.json.history/<sha256>.json`, and
uses unique create-only staging, write-through handles, `Flush(true)` and hash
readback before and after replacement. A damaged predecessor is never silently
converted into a new enrollment. Identical states do not create new generations.
History retains at most128 generations and32MiB, including the incoming exact
predecessor. Before staging, the controller checks bounded filenames, attributes
and sizes. When pruning is needed it first validates every retained content hash,
object custody and private Windows ACL, then removes oldest validated entries
to half the hard budget, reserving that predecessor. Batching avoids rehashing
the whole history on every checkpoint. It never prunes
the immediate predecessor. Unrecognized names, sizes or reparse entries refuse
publication. Substituted content or foreign ACLs refuse pruning without deleting
any history. Existing stores already exceeding either budget require
explicit recovery review instead of automatic cleanup. Recovery archives outside
this directory remain untouched. The [retention controls](../../Tests/Portable/Test-Usb4StateHistoryRetention.ps1)
repeat actual failed reconciliation/rollback attempts and exercise byte bounds,
predecessor preservation and foreign-entry/no-Apply refusal.
This reduces exposure to interrupted writes; it does not establish physical
power-loss durability or atomic comparison against arbitrary administrator writes.
Use a trusted protected runtime and the controller/installer mutexes.
Installer upgrades preserve this history only after checking administrator-only
ACLs, exact content-addressed filenames, bounded bytes and object custody.
Unknown entries, substituted generations and reparse points refuse an upgrade.
The controller protects new history directories before publication. It can seal
legacy inherited ACLs only when their owner and full-control grants already name
SYSTEM and Administrators exclusively; foreign ACLs remain unchanged. The
[actual Windows ACL controls](../../Tests/Portable/usb4-history-acl-review.json)
exercise controller-created history against real installer admission.
Keep journal-recovery archives outside the installation root, for example under
its sibling `journal-recovery` directory; the installer does not adopt them.

For the narrow case of a disabled ExistingWinNat consumer interrupted while
observing its rollback, [Resolve-Usb4PeerJournal.ps1](../../Tools/SystemScripts/Networking/Resolve-Usb4PeerJournal.ps1)
provides an explicit file-only retirement. Declare the exact host, adapter,
interrupted attempt, installed controller/policy hashes and damaged/trusted state
hashes. Run with `-DryRun` first. An incomplete identity, unknown custody field,
in-flight mutation, non-address ownership, or any remaining possibly owned
address refuses retirement. The tool archives the exact damaged bytes, pending
backup and disabled task, writes a prepared receipt, and reobserves absence under
both mutexes before publishing `{}`. It leaves the task disabled and preserves
foreign network settings. An existing archive requires explicit review of the
interrupted recovery; it is never overwritten automatically.

The [publication and retirement tests](../../Tests/Portable/Test-Usb4StatePublication.ps1)
exercise actual temporary-file publication, locks, byte preservation and refusal;
[retirement controls](../../Tests/Portable/Test-Usb4JournalRetirement.ps1) inject
only the operating-system boundaries. Neither suite qualifies a live deployment.
The [review register](../../Tests/Portable/usb4-journal-publication-review.json)
records the source snapshot and independent findings.

Run the portable controller and installer controls first, then qualify:

1. Direct IPv4/IPv6 peer connections with the actual source interface recorded.
2. Authenticated SSH or file-transfer traffic bound to the USB4 source.
3. Consumer DNS/HTTPS through the intended provider and wired uplink.
4. Unchanged management, Wi-Fi, WSL NAT and unrelated routes/firewall/SMB state.
5. A second reconcile with no duplicate addresses/routes or repeated changes.
6. Recovery after task rerun, then separately after physical reconnect/resume.

Measure bidirectional payload throughput and CPU before tuning MTU, parallelism
or SMB. Nominal 20 Gbps link reporting is not payload throughput. Preserve the
current MTU until the actual peer and upstream path are qualified. USB4 Ethernet
and SMB multichannel do not establish RDMA or SMB Direct capability.

Linux contributors should use maintained `thunderbolt_net` and native networking
such as NetworkManager's `shared` downstream mode, with explicit uplink policy.
This Windows installer does not deploy Linux configuration. Use the same
identity, route, source-bound proof and rollback requirements on either OS.

Platform references: [Microsoft USB4NET](https://learn.microsoft.com/en-us/windows-hardware/design/component-guidelines/usb4-interdomain-connections),
[ICS behavior](https://learn.microsoft.com/en-us/windows/win32/api/netcon/nf-netcon-inetsharingconfiguration-enablesharing),
[WinNAT limitations](https://learn.microsoft.com/en-us/windows-server/virtualization/hyper-v/setup-nat-network),
[shared-prefix allocation](https://learn.microsoft.com/en-us/virtualization/windowscontainers/container-networking/advanced),
[IPv4 link-local scope](https://www.rfc-editor.org/rfc/rfc3927),
[Linux Thunderbolt](https://docs.kernel.org/admin-guide/thunderbolt.html), and
[NetworkManager settings](https://networkmanager.pages.freedesktop.org/NetworkManager/NetworkManager/nm-settings-nmcli.html).

For a Public-profile peer whose existing SSH rule covers only Private networks,
use the separately reviewed [scoped peer SSH access](peer-ssh-access.md) helper.
It preserves network categories and constrains access to the declared peer,
interface, addresses, application and service.
