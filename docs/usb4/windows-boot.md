# Windows USB4 startup observation

The [startup entrypoint](../../Tools/SystemScripts/Networking/Configure-ThunderboltSwitchRoutes.ps1)
runs native status and V&V with a 20-second deadline per command. It publishes
`~/.machine/logs/usb4-boot/current.json` atomically, preserving both native output
streams and exit codes. `-DryRun` performs observation without writing files.

| Exit | State | Meaning |
|---|---|---|
| 0 | Configured | Native technical checks pass; this does not qualify a data edge. |
| 1 | Failed | Collection, receipt binding or technical checks fail. |
| 2 | Absent / Degraded | No present adapter, or no operational USB4 link. |
| 3 | PendingAcceptance | Technical checks pass while native acceptance remains pending. |
| 4 | Unsupported | The collector is running outside Windows. |

`NetworkReady` remains false in every state. Consumers must independently prove
enrolled identity, intended egress and traffic before using a data edge. Receipt
expiry is 90 seconds and is an observation freshness boundary, not a route lease.
The original native verdict remains in `vnv`; an absent adapter never rewrites a
native FAIL into PASS.

Boot no longer invokes discovery-driven registration, address installation,
metrics or route mutation. The previous machine-local wrapper treated an NTLM
hostname or banner as a verified peer and ignored the V&V exit code. Those hints
cannot authorize a route on a roaming, multihomed machine. The
[enrolled observer](enrolled-observer.md) describes the separate Linux authority
and qualification boundaries. Static adapter aliases and historical topology
records are insufficient authorization.

Run the [portable controls](../../Tests/Portable/Test-ThunderboltBootReadiness.ps1)
with `pwsh -NoProfile -File Tests/Portable/Test-ThunderboltBootReadiness.ps1`.
They exercise failure, absence, down links, intake holds, malformed/wrong-host
receipts, unsupported platforms and no-write dry-run, without changing the host.
The positive Configured control still refuses data-edge readiness.

The installed `~/.machine/Configure-ThunderboltSwitchRoutes.ps1` must forward to
this reviewed repository source and propagate its exit code. Keep a hashed copy
of the prior launcher and a source/build deployment manifest outside the
repository. Windows native inventory requires the merged present-only inventory
fix; a stale PnP record is not a present operational adapter.

## SFTP mount recovery

The [USB4 mount entrypoint](../../Tools/SystemScripts/Networking/Invoke-ThunderboltRcloneMount.ps1)
refuses to mount when the enrolled source address is absent, belongs to another
adapter or resolves through a default/gateway route. Rclone binds its connection
to that source, overrides the endpoint address explicitly and validates every
connection using the supplied pinned `known_hosts` file. A trusted key can be
transferred from the already verified management identity; a new `ssh-keyscan`
response alone is not enrollment.

The launch arguments explicitly clear external SSH and disable first-use trust,
so backend or environment changes cannot bypass the native source/key policy.

Connect timeout is five seconds, idle timeout 30 seconds, and both retry counts
are one. These bound an unavailable endpoint without imposing a runtime deadline
on a healthy foreground mount. The existing scheduled task can reattempt after
hardware returns while IgnoreNew prevents duplicate mounts. An unavailable path
returns exit 2 without launching rclone, reading remote data or touching cached
writes. The [mount gate controls](../../Tests/Portable/Test-ThunderboltMountGate.ps1)
cover a working path, default-LAN fallback, absent/foreign addresses, down hardware
and an unrelated Ethernet adapter. They do not certify a successful live mount.

The options follow the official [SFTP host-key contract](https://rclone.org/sftp/)
and [outgoing source-bind semantics](https://rclone.org/docs/#bind-string), checked
against installed rclone 1.75.1. Preserve the existing mount cache and unrelated
drive tasks during deployment.

## Enrolled Windows address leases

The [partner reconciler](../../Tools/SystemScripts/Networking/Invoke-ThunderboltEnrolledPeer.ps1)
consumes a fresh Linux `observe-current` record through source-bound, pinned SSH
on a present Windows USB4 interface. The actual SSH socket, administrative
node identity, enrolled controller pair, policy epoch and interface inventory
must agree. Its second authenticated connection uses the derived global address
pair. Management/LAN access never substitutes for either connection.

`-DryRun` observes and plans without writing. `-Apply` requires an administrative
`C:/ProgramData/pcai-usb4/windows-partners.json` allowlist and writes its private
`windows-owned-edges.json` journal. The policy pins peer node/controller identities,
minimum epoch, existing SSH identity/known-hosts paths, allowed local PnP root
identities and a ULA /48. Linux remains the address derivation authority.
No Windows IPAM table, default route, DNS, MTU or firewall change is introduced.

Present disconnected adapters remain in the identity inventory with no neighbors.
Windows reports an empty per-interface neighbor table as `ObjectNotFound` with
the exact `CmdletizationQuery_NotFound,Get-NetNeighbor` error ID; the reconciler
treats only that outcome as an empty table. Access failures, missing providers
and other query failures still refuse reconciliation. An unused adapter cannot
prevent another USB4 edge from being observed, and retaining its GUID, MAC and
PnP root keeps expired-address custody checks intact.

Only new ActiveStore addresses and explicitly transferred exact allocations
receive deletion custody. A persistent derived address is refused until its
Active/Persistent preimages have been preserved and its exact custody migrated
to ActiveStore. A pending intent left by a crash never authorizes deletion.
An unowned existing address may be tested but remains unowned. Every receipt
qualifies only its current root-namespace edge, never the entire topology.

Committed leases expire after the policy TTL (30–300 seconds). Refresh identifies
the local adapter by GUID, MAC and PnP root, allowing aliases and ifIndex values
to change. A new GUID with an address stranded on an old interface returns an
actionable custody refusal; an absent old identity never permits blind deletion.
An expired exact owned allocation is withdrawn only after fresh identity and
positive address readback. Failed new-address canaries require positive rollback
absence before custody can be forgotten. Repeated tasks must run with IgnoreNew;
the reconciler also holds a node-specific mutation mutex.

The [portable partner controls](../../Tests/Portable/Test-ThunderboltEnrolledPeer.ps1)
exercise proof freshness, socket identity, policy replay, foreign custody,
same-GUID reindex, different-GUID conflict, persistent-store refusal, no-write
planning and actual controller creation/rollback using host API fixtures. They
also exercise the real host-inventory function with a usable adapter alongside
a present disconnected adapter, and check that access failures and unrelated
`ObjectNotFound` errors remain fatal. They
do not certify physical unplug, resume, reboot or a different hardware link.

The tracked [startup coordinator](../../Tools/SystemScripts/Networking/Invoke-ThunderboltStartup.ps1)
is the deployment recipe for the existing scheduled task: a thin machine launcher
supplies explicit policy, journal, executable, registry and evidence paths. It
runs the hidden controller and observer with separate 45-second deadlines,
records both outcomes in `enrolled-last-attempt.json`, and attempts observation
even when controller launch fails. A publication failure also returns nonzero.
The [orchestration controls](../../Tests/Portable/Test-ThunderboltStartup.ps1)
cover component, launch and publication exceptions. Use IgnoreNew, a two-minute
task execution limit and a one-minute fallback with a 180-second local lease;
retain the boot/network/resume triggers and original task settings/preimage.

A SYSTEM task needs its own SSH identity. Windows OpenSSH may reject a personal
key containing that user's ACL when invoked as SYSTEM. Keep personal keys and
the existing task principal unchanged; generate a separate SYSTEM/Administrators
key under the protected ProgramData service directory and transfer the existing
verified public host pin. On the enrolled Linux partner, preserve `authorized_keys`
before adding a `restrict` entry forced to the root-owned
[fixed observer export](../../Tools/SystemScripts/Networking/Export-ThunderboltEnrolledObserver.sh).
That credential grants the fixed read-only frame, without shell selection,
forwarding or PTY. Qualify the actual scheduled SYSTEM run and repeated renewal;
a successful interactive run alone does not qualify deployment.
