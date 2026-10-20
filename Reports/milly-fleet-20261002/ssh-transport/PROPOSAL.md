Hanging-handshake control added — 2026-10-03 00:09 UTC:

An exclusively owned Python server bound an ephemeral port on 127.0.0.1, accepted TCP and deliberately withheld the SSH banner. A fresh SSH client with -F /dev/null, -S none, BatchMode=yes, ConnectionAttempts=1 and ConnectTimeout=1 exited 255 after 1005.981 ms with “Connection timed out during banner exchange”. The separate four-second process deadline did not fire. The owned listener and server thread both closed; no production master or network configuration was touched. This complements the earlier refused-port test and verifies a genuinely stalled protocol handshake. See milly-banner-timeout-validation.json and its recorded source. The release upgrade remains in progress, so repeat relevant SSH policy/reuse checks after final package installation and reboot.

---
Validated deployment update — Milly Cog and Windows, 2026-10-03 UTC:

Milly Cog now has an early Include for /home/cog/.ssh/vigil-transport.conf, scoped only to asuspro13, spark-0060 and spark-3066. Its original fleet include bytes, keys, strict pins and ProxyJump are preserved; GitHub's complete effective configuration is unchanged. Control sockets are owned by Cog in a mode-0700 directory. Windows' reviewed four-alias settings also passed a recovered-Milly hostname check, supplementing its earlier three-host validation. These checks do not establish completion of the ongoing OS upgrade.

Three fresh Milly→ASUS `true` commands took 218.362/212.913/193.780 ms; three reused commands took 92.437/16.847/16.720 ms. Medians were 212.913 versus 16.847 ms. Each reused command emitted mux_client_request_session evidence; an exclusive disposable master retained the same PID. This small-sample result concerns command latency on this route, not bulk throughput or fleet-wide speed.

Missing/stale disposable socket fallback, short disposable idle expiry, alias/requested-account path separation, fresh-handshake wrong-pin refusal, and a bounded unreachable loopback endpoint all passed. Both Spark read-only key-authentication checks passed. All three remaining exclusively owned test masters exited successfully; no foreign/default master was terminated. Linux verification used OpenSSH 8.9p1 before the in-progress release upgrade; repeat effective-policy and reuse checks after the new client packages and reboot.

The first diagnostic harness failed because a verbose ControlPersist master retained an inherited stderr PIPE, delaying subprocess capture completion. That failure is retained in milly-mux-harness-failure.json. The corrected harness writes stderr to private files and validates command-process exit independently of master lifetime; it then passed all controls. For production verbose mux diagnostics, use an owned private `ssh -E` log or file redirection with an explicit retention policy. A pool must distinguish command completion, transport lifetime, timeouts and unknown outcomes; never use a client deadline to blindly replay a remote mutation.

Keep compression disabled on connections mixing trusted commands and untrusted forwarded traffic; the current [OpenSSH manual](https://man.openbsd.org/ssh_config) documents that shared compression can leak session information. Use `-S none` for bulk/credential checks that should not reuse the control master. Existing pinned credentials and host-key policy remain mandatory for fresh transports. No live packet loss, wireless roaming, bulk acceleration, desktop stream, or clinical acceptance was tested here.

The files milly-implementation-receipt.json, milly-mux-validation.json, milly-mux-validation-source.sh and windows-milly-postmaintenance-check.json contain the current public evidence. Original before-config bytes remain privately held on each machine and are excluded from public manifests. Earlier no-change/deferred statements below belong to the original research snapshots.

---
Implementation update 2026-10-02T23:53:14.4712808Z: the four exact Windows aliases now have the reviewed bounded connection settings. See windows-implementation-receipt.json and windows-transport.applied.conf. The research-stage no-change statements below describe the earlier snapshot; Linux session sharing and other alternatives remain proposals. Three fleet SSH hostname checks passed; Milly's live check is deferred through primary-agent firmware maintenance.

# VIGIL fleet SSH and transport proposal — 2026-10-02

Status: researched and parser checked; no SSH config, service, route, firewall, keys, or remote control sockets changed. Measured only short read-only commands. Existing maintenance/HIL remains owned by the primary agents. This report is input to owner consensus, not a claim that session reuse has been deployed or benchmarked.

## Observed client state and timings

Windows executable C:/WINDOWS/System32/OpenSSH/ssh.exe reports OpenSSH_for_Windows_9.5p2, LibreSSL 3.8.2, file version 9.5.6.3. All four fleet aliases have pinned StrictHostKeyChecking=true, IdentitiesOnly=yes, ForwardAgent=no, Compression=no; ControlMaster=false and ControlPersist=no. ASUS/Sparks resolve to the verified direct 10.60.4.1/.2/.3 fabric; Milly resolves to 192.168.50.43. These Windows aliases currently need no ProxyJump. Preserve distinct host-scoped Linux routing; Milly's route to 3066 may require its existing ASUS jump.

Twelve Windows process-inclusive cold `ssh -n -o BatchMode=yes -o ConnectionAttempts=1 -o ConnectTimeout=5 HOST true` samples passed:

| Destination | Median milliseconds | Range | Count |
|---|---:|---:|---:|
| Milly | 357.54 | 337.32–383.44 | 3 |
| ASUS | 307.29 | 279.65–311.10 | 3 |
| Spark 0060 | 402.77 | 395.53–498.85 | 3 |
| Spark 3066 | 432.10 | 399.13–440.34 | 3 |

This is shell/process + TCP/KEX/auth + remote-command time, not RTT, bulk throughput, or reused-session latency. Three samples establish a baseline only. At these times repeated one-command processes add noticeable cost; one structured multi-command script per host avoids repeated connection setup. No multiplex speedup percentage is claimed.

ASUS currently reports OpenSSH_10.2p1 and rsync 3.4.3. ASUS→Milly is pinned and agent forwarding disabled. ASUS→both Sparks has ForwardAgent=yes; Spark3066 additionally has IdentitiesOnly=no; both have ConnectTimeout=none. Review owner dependencies before tightening these settings. Do not disable active forwarded signing/auth use silently.

## Decisions proposed to the owners

| Change | Priority / scope | Decision and limit |
|---|---|---|
| Batched remote scripts and SFTP batches | Immediate, every client | Existing Windows-native tooling can do this without installing another SSH stack. Capture stdout, stderr, exit status and per-operation timeout distinctly. |
| OpenSSH multiplexing | Linux only, after maintenance | Enable per owner and per pinned alias with a short idle lifetime. Separate bulk/interactive/control profiles. |
| Native Windows ControlMaster | Reject | Parser accepts directives, implementation is unsupported. A local `-O check` with a nonexistent private control path failed with `getsockname failed: Not a socket`, exit -1. Do not add ControlPath and accidentally break all commands. |
| Windows WSL OpenSSH | Optional future lane | Existing WSL distributions were inventoried but not started or enrolled. Provision an independent per-distribution key and pinned host-key config; do not copy workstation private keys into WSL. Linux mux works in that client environment, not in native ssh.exe. |
| Windows PuTTY/Plink sharing | Optional separate workflow | Installed Plink 0.84 supports upstream/downstream sharing. Its own saved sessions, key integration and pins must be reviewed; it does not automatically share native OpenSSH sessions or configs. Prefer batching until this extra configuration is justified. |
| Global compression or cipher override | Reject | Keep current security defaults and Compression=no. Benchmark payload-specific changes later. |
| New SSH pool daemon/library | Defer | First measure batch scripts and Linux mux. Persistent protocol tooling needs lifecycle, credential, timeout and replay design rather than an unreviewed dependency installation. |

The Windows implementation limitation is documented by the [Win32 project scope](https://github.com/PowerShell/Win32-OpenSSH/wiki/Project-Scope) and [Windows socket design](https://github.com/PowerShell/Win32-OpenSSH/wiki/About-Win32-OpenSSH-and-Design-Details); the runtime failure independently corroborates it. Those wiki pages also contain historical feature lists: no claim about compression support was derived solely from that old list.

The Linux proposal uses ControlMaster=auto, ControlPersist=120s and private ControlPath `~/.ssh/cm/vigil-%n-%C`. Create the directory mode 0700 as the invoking owner, never a shared root/user socket. `%n` separates alias policies; `%C` shortens the endpoint identity. No token includes a command-line key override, so use `-S none` for key/host-policy revalidation and credential rotation. Place scalar options before an earlier matching value; appending after Host * may have no effect. This follows [OpenSSH client configuration](https://man.openbsd.org/ssh_config). Linux parser validation on ASUS succeeded without creating any socket or config file.

For automation pass BatchMode=yes and enforce a total process deadline independently. A 30-second server-alive interval and three unanswered probes give a roughly 90-second failure detector, not automatic reconnection or failover. Reuse the existing pinned HostKeyAlias, identity, and known-hosts custody. Agent key caching avoids repeated unlocks but is distinct from connection reuse. Prefer existing per-host keys and ProxyJump over agent forwarding. [SSH agent documentation](https://man.openbsd.org/ssh-agent) describes signing without exposing private keys to the destination.

## Throughput and alternative transports

Keep SSH as the administrative/bootstrap path. A master removes setup latency; it does not increase physical link bandwidth. Avoid putting interactive control and a long bulk transfer into the same TCP connection, where packet loss/congestion couples them. Do not add unrestricted parallel transfers on the currently occupied fleet network. Current Milly bulk-copy and iperf results belong to the primary run receipts, not this benchmark.

Use rsync for repeated tree synchronization and preserved Linux metadata; retain the existing -aHAX/verification policy. For compressible text/artifact groups on slower routes, compare negotiated zstd/lz4 with no compression, one compression layer only. Already encoded video, JPEG, archives and typical model weights need their own data measurements. On fast SSD-to-SSD links, compare whole-file transfer with delta only when changed files and storage bandwidth justify it. Neither --checksum-choice=none nor in-place overwrite is an acceleration default. The [rsync manual](https://download.samba.org/pub/rsync/rsync.1) documents compression negotiation, partial-directory resumption, whole-file behavior and checksum guarantees.

For Windows file batches use one sftp process or one scp invocation containing multiple paths. Installed clients expose -b/-B/-R/-X; tune outstanding requests/buffer only after measuring bandwidth-delay and server limits. Preserve safe SFTP default; do not switch to legacy scp to obtain speculative speed. [OpenSSH release notes](https://www.openssh.org/releasenotes.html) confirm scp's SFTP default from 9.0, and the [SFTP manual](https://man.openbsd.org/sftp) documents batching and request buffers.

Continue using the public read-only fleet NFS share for appropriate shared assets; keep private backups outside it. Streaming/rendering should use the selected Sunshine/Moonlight or dedicated media transport rather than a bulk/control SSH master. For application telemetry/control, consider reuse of the existing authenticated HTTP/gRPC connection infrastructure with connection pooling, request IDs, deadlines and explicitly idempotent retries. This is an architectural proposal; no new listener or TLS configuration was installed.

For mobile interactive terminals, evaluate Mosh plus tmux after owner agreement. [Mosh](https://mosh.org/) bootstraps through SSH and uses UDP for roaming; it is not a file-transfer tunnel. Narrow any UDP rule to the actual approved peers/port instead of opening its whole default range. For remote fallback, inspect whether Tailscale is direct or relayed before blaming SSH encryption; [Tailscale connection types](https://tailscale.com/docs/reference/connection-types) explain direct versus relay performance. Never replace strict OpenSSH host-key trust merely because the overlay authenticates devices.

## Validation before rollout

1. Root completes Milly backup/maintenance; ASUS owner releases affected configuration claims. Capture each client's sanitized `ssh -G` and pinned-key custody before and after any edit. Compare exact intended fields, including include order.
2. On one Linux owner, use a unique mode-0700 runtime directory and disposable known-master socket for three cold and three reused `true` commands. Confirm one transport with debug/channel receipts and `ssh -O check`; then drain with `ssh -O stop`. Do not exit another agent's master. [ssh(1)](https://man.openbsd.org/ssh) documents control operations.
3. Validate idle expiry, missing/stale socket fallback, timeout on an unreachable test endpoint, alias/user separation and absence of agent forwarding. Test a deliberately wrong pinned key in an isolated temporary config with `-S none`; verify refusal without mutating production known_hosts. A still-open authenticated master will not rerun the host-key handshake for each new channel.
4. Disconnect/reboot recovery must preserve a remote tmux/systemd-owned operation and report an unknown outcome rather than blindly repeat a mutating command. Read-only retry may use bounded backoff. Reconcile request IDs and state before retrying installation, deployment, firmware or control operations.
5. After HIL release, compare isolated bulk and control latency under load with packet-loss/CPU observations. Example diagnostic only, not executed: `sudo tcpdump -ni <verified-interface> -s 96 -c 200 'host <peer-address> and tcp port 22'`. Restrict retention/permissions of captures; do not record credentials or unrelated application traffic. `tshark -r <capture> -Y 'tcp.analysis.retransmission || tcp.analysis.lost_segment'` can identify candidate transport loss; capture vantage points must be recorded.

Proposal files beside this report are intentionally inert: linux-transport.proposed.conf, windows-transport.proposed.conf and asus-spark-hardening.proposed.conf. Windows and ASUS parsers accepted their applicable transport template. The applied Windows/Linux policies and measured Linux reuse are recorded above and in implementation receipts; the adjacent templates remain inert. Roaming, bulk tuning, new transports, firewall changes and fleet acceptance are not claimed. Post-OS OpenSSH10.2 reuse is additionally verified in ../post26-final/functional-corrections.json.