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
