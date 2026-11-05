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

## Qualification and performance

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
