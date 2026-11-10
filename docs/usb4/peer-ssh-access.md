# Exact peer SSH access over USB4

Use PowerShell 7 or newer and a PC-AI repository checkout. Keep the
[helper](../../Tools/SystemScripts/Networking/Set-Usb4PeerSshAccess.ps1) in that
checkout; no compilation or build step is required. Windows Apply needs an
already running OpenSSH `sshd` service. The helper installs or starts no service.

From the repository root, test and run help:

```powershell
pwsh -NoLogo -NoProfile -NonInteractive -File Tests/Portable/Test-Usb4PeerSshAccess.ps1
pwsh -NoLogo -NoProfile -NonInteractive -File Tools/SystemScripts/Networking/Set-Usb4PeerSshAccess.ps1 --help
```

The [portable fixtures](../../Tests/Portable/Test-Usb4PeerSshAccess.ps1) match the
[SSH access workflow](../../.github/workflows/usb4-peer-ssh-access.yml).
Nine groups use inert operation providers on Windows and Linux. On Windows, a
tenth group constructs unregistered `HNetCfg.FWRule` objects for both binding
modes, using a cached Up interface alias, and checks full scope readback. It needs Windows Firewall COM
but no elevation, service, peer traffic or firewall policy API; no rule is
published. POSIX reports that group explicitly as `NOT_RUN`. Linux fixture
success does not implement or qualify a Linux firewall backend. The
[review record](../../Tests/Portable/usb4-peer-ssh-access-review.json) binds the
source, selectors and observed Work execution.

## Plan with fresh declarations

A USB4 interface can be Public while existing SSH allow rules cover only Private
networks. Recorded transport checks observed TCP445 connectivity and TCP22
timeouts in both address families with that configuration. This is an access
policy finding; a port handshake does not establish authentication or file
transfer.

Collect current local host, interface GUID, alias, local MAC and assigned addresses.
Obtain the peer addresses through an existing authenticated management path.
Keep them separate from historical adjacency records. The
[networking qualification guide](peer-networking.md) and
[fabric deployment roadmap](fabric-deployment.md) describe those boundaries.

Replace these example declarations before any Apply:

```powershell
$peerSsh = @{
    ExpectedHostName = 'YOUR-PC'
    InterfaceGuid = '11111111-2222-3333-4444-555555555555'
    ExpectedInterfaceAlias = 'USB4 peer'
    ExpectedMac = '02-00-00-00-00-01'
    LocalAddress = @('169.254.12.34', 'fe80::1234')
    RemoteAddress = @('169.254.56.78', 'fe80::5678')
    ProgramPath = 'C:\Windows\System32\OpenSSH\sshd.exe'
    ServiceBinding = 'Service'
}
& ./Tools/SystemScripts/Networking/Set-Usb4PeerSshAccess.ps1 @peerSsh -DryRun
& ./Tools/SystemScripts/Networking/Set-Usb4PeerSshAccess.ps1 @peerSsh -Apply -WhatIf
```

Default invocation, `-DryRun` and `-WhatIf` produce offline plans with no OS or
firewall provider calls. `-DryRun` suppresses providers even with `-Apply`.
The declarations themselves remain unverified in an offline plan.

Each address must be an exact assignable IPv4 link-local or IPv6 link-local host
address, without a prefix, wildcard, range or zone suffix. Supply at most one
address per family and the same families on both sides. `ExpectedMac` identifies
the local interface. It is not a peer-authentication claim.

Windows may read an exact host back with a full host mask or an equal-endpoint
range. The observed-rule matcher normalizes only those single-host equivalents
before comparing the entire rule. Wider masks, unequal endpoints and mixed
families remain refused. Declared input still rejects all masks and ranges.

## Select the binding explicitly

`ServiceBinding` defaults to `Service`. Before Add, this mode requires the
configured `sshd` service SID to be a registry DWORD of 1 (UNRESTRICTED) or 3
(RESTRICTED). SID NONE (0), missing, unknown or incorrectly typed data is refused
before mutation. The helper never falls back to another mode or changes SID
configuration. Microsoft documents the
[service SID requirement for service-specific firewall rules](https://learn.microsoft.com/en-us/windows/security/operating-system-security/network-security/windows-firewall/configure).

For a reviewed decision to bind only the exact executable, select `Executable`
explicitly before the initial Apply:

```powershell
$peerSsh.ServiceBinding = 'Executable'
& ./Tools/SystemScripts/Networking/Set-Usb4PeerSshAccess.ps1 @peerSsh -DryRun
```

Executable mode leaves the COM `ServiceName` setter unused on a fresh rule.
Observed null or empty service names match only this explicit mode; `*` is
refused. Microsoft describes
[`ServiceName` as optional and gives `*` a distinct service-only meaning](https://learn.microsoft.com/en-us/windows/win32/api/netfw/nf-netfw-inetfwrule-get_servicename).
Both modes retain the logical `sshd` service, running PID/generation and image
checks, plus the exact application, interface, addresses and TCP22 scope.

A configured SID is metadata, not proof of the SID in the running process token;
`runningTokenQualified` remains false. Microsoft describes
[when service SID types affect process tokens](https://learn.microsoft.com/en-us/windows/win32/api/winsvc/ns-winsvc-service_sid_info).
The helper does not change or restart the service. A recorded endpoint reported
SID NONE, which makes a named-service firewall filter unsupported; this finding
does not establish that it is the only cause of SSH timeouts.

## Apply and retain the original scope

Run Apply inside an endpoint-owned process with a finite deadline and captured
stdout, stderr and exit status. The existing
[bounded Windows process runner](../../Tools/SystemScripts/Networking/Invoke-VigilBoundedProcess.ps1)
provides process-tree custody for callers. Its lazy native-helper initialization
precedes its runtime deadline, so a caller still needs a bound around the whole
endpoint invocation.

Within that bounded endpoint process:

```powershell
& ./Tools/SystemScripts/Networking/Set-Usb4PeerSshAccess.ps1 @peerSsh -Apply
```

Apply requires an elevated Windows token on the declared host. Before adding a
rule, it checks the exact current GUID, alias, MAC, USB4 identity, assigned local
addresses and Up state. It also checks the running `sshd` service executable,
process ID and generation, then repeats preflight before mutation. Service mode
also requires the configured SID type to stay unchanged across that preflight.

The rule permits inbound TCP22 for one interface, exact local and remote
addresses and the running SSH application. Service mode adds the exact `sshd`
firewall service condition; Executable mode omits that condition. Its profile mask is
Domain, Private and Public; it does not change network classification. It does
not open all ports or an entire link-local subnet. Rule publication does not
prove that SSH authentication or payload transfer works. Peer identity remains
`DeclaredByCaller`.

The broad link-local allow in the older
[registration helper](../../Modules/PC-AI.Drivers/Public/Register-ThunderboltPeer.ps1)
is not this operation. Do not invoke that registration path as an SSH repair.

Keep the original declarations and receipt for cleanup:

```powershell
# Within a bounded endpoint process, using the unchanged original declarations:
& ./Tools/SystemScripts/Networking/Set-Usb4PeerSshAccess.ps1 @peerSsh -Remove -Apply
```

Remove does not require connected hardware or a running SSH service. It still
requires the correct elevated host and the identical owned rule. A missing rule
is a NoOp. A foreign, duplicated or drifted same-name rule is refused; the helper
does not adopt or overwrite it. Changed addresses or aliases produce a different
rule identity, so retain the old scope rather than reconstructing it from a new
observation. The binding mode is also part of the rule identity: retain the
original mode for cleanup, including when a later plan selects a different mode.

## Read outcomes before qualification

The helper changes only its exact firewall rule. It writes no files, events,
tasks, services, routes, DNS, NAT, MTU or network-category settings. Its named
Global mutex coordinates cooperating invocations across Windows sessions.

COM rule operations are not atomic compare/exchange against unrelated
administrator edits. OS/native queries can stall despite the service query's
three-second operation timeout. A timeout or mutation exception leaves the
outcome unknown until fresh readback; do not blindly delete a rule or infer
rollback from a failed command.

The final recorded Work run on PowerShell 7.6.9 passed strict parsing and 239
assertions across ten groups in 2.6273804 seconds. The original eight selector
blocks remain unchanged; their shared fixture now declares the supported SID
prerequisite. The new inert SID group first failed against the previous source
because SID NONE was accepted. It now proves strict SID refusal, explicit mode
selection, unchanged remaining scope and exact owned executable-rule cleanup.

The Windows COM group checks construction and full scope readback in both modes
without opening firewall policy. Earlier failures caught singleton-array
unwrapping, COM rejection of a string array for `Interfaces`, and exact IPv6 host
readback as an equal-endpoint range. The
[review record](../../Tests/Portable/usb4-peer-ssh-access-review.json) retains those
failures and earlier passing records with their original source bindings.

The fixture result covers inert controls and unregistered COM construction and
readback. A separate temporary trial on dtm-carbon-two, using this source and
explicit Executable mode, completed pinned IPv4 SSH with the expected server
nonce, host and exact connection tuple. Paced SFTP then transferred 1 MiB in
each direction in 3.7108957 seconds with equal SHA-256 hashes, under an 8192 kbit/s
rate cap. This is functional payload-integrity evidence, not peak throughput.

The earlier named-service trial timed out while the service SID was NONE;
explicit executable binding worked without changing SID configuration or
restarting the service. This supports the SID explanation within the observed
cohort; it does not establish it as the sole possible SSH blocker elsewhere.
Owned cleanup returned Removed, and fresh COM readback found zero matching
rules. The trial left no ongoing firewall rule or installed operational deployment.

These observations do not qualify reverse-initiated sessions, hot-plug/reconnect,
IPv6 SSH, other source versions, Linux or other fleet hosts, or Thunderbolt Share
features. Keep those checks in the separate
[qualification sequence](peer-networking.md#qualification-and-performance).

For native Windows SSH capture handles, memory-only stdin and reproducible qualification, see the [process capture contract](ssh-process-capture.md).

Explicit controller-enrolled RFC1918 /30 pairs use the [private peer SSH contract](private-peer-ssh-access.md), while default admission remains link-local only.
