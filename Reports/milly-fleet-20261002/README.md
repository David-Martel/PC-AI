# Milly fleet endpoint maintenance

Checkpoint: 2026-10-03 00:16 UTC. This report records work in progress. The
qualified backup is verified; Intel ME firmware applied and reboot recovered.
Operating-system migration, graphical login, paired stream and final system
data validation are not complete.

## Current machine and access

`millylaptop1` is a Lenovo P16 Gen 2 with an i9-13950HX, 64 GB RAM, RTX 5000 Ada
Laptop GPU with 16 GB VRAM, and a 3840x2400 panel. It currently runs Ubuntu
22.04.5 with kernel 6.8.0-138 and NVIDIA 580.178.04. `cog` has working SSH keys,
passwordless sudo, its own fleet/GitHub authentication key, and independently
verified SSH Git signing. Private keys were not copied from another host.

The endpoint is an external development and operator surface. It is not added
to the production fleet membership, compute placement, release quorum, or
static DDS peer set. Native Humble was removed before the OS migration. Current VIGIL
contracts use Jazzy/Python 3.12; separately validated, pinned Jazzy and Lyrical
containers preserve their respective ROS/Python ABI. Managed Python 3.14 is
available alongside 3.12 and 3.13. The system Python was not replaced.

## Backup and maintenance boundary

The private recovery copy is at
`asuspro13:/srv/vigil-backups/millylaptop1/20261002`, outside the NFS export,
with root-owned private parent directories. The source baseline is
775,325,002,378 bytes (775.325 GB or 722.078 GiB), including all three home
directories and a separate EFI copy. The original copy completed at 22:47:10 UTC;
the final incremental copy completed at 22:51:35 UTC with 789,737,621,693 total
file bytes and no destination deletions. Nine restore samples passed, including
byte/metadata/symlink checks, a capability attribute and a hardlink pair.
Full checksums match all three homes and EFI with zero differences; all nonvolatile
system data and metadata match. Seven classified live logging/NetworkManager/timer
exceptions remain, so this is not an exact frozen whole-system snapshot.
The hash-bound backup-validation.json records the qualification and controls.

The user approved logout and maintenance; the prior desktop user was logged
out and quiescence checked. Previous-user autologin is disabled during maintenance,
with its original configuration preserved. An independently tested key-only
recovery SSH listener and sleep inhibitor support maintenance.
The recovery listener was recreated and tested after the ME reboot. Ubuntu
24.04 maintenance is underway. The first upgrader attempt stopped on incompatible
ROS Humble; a reviewed removal affected only its 282 packages, then the upgrade
was restarted. No automatic obsolete-package removal was requested.
The approved upgrade path is
22.04 to 24.04 to 26.04, validating SSH, packages, NVIDIA, and desktop behavior
at each step. Backup cleanup waits for successful post-upgrade data validation.

## Installed and exercised tooling

Managed tooling includes uv, Python 3.12/3.13/3.14, Rust with clippy/rustfmt,
Clang 23 alongside existing compilers, GCC/G++ 12, CMake/CTest/CPack, Ninja,
sccache/ccache, nextest, just, Ruff, ty, basedpyright, pytest, Git LFS, protobuf,
grpcurl, lychee, rootless Podman, and hardware/network debugging tools.
Provenance and exact versions are in the adjacent JSON receipts.

Functional checks include 20 tool commands, six pytest cases, a CTest case,
a nextest case, invalid-input checks for linters/type checkers, two build-cache
hits, and 14,840 managed-Python RECORD file hashes. Agent-bus was built locally
for this host's glibc, rather than copying an incompatible fleet binary.

Qt painting and NVIDIA EGL rendering produced asserted pixels. A reproducible
synthetic NVENC/NVDEC check has three exact commands, zero exit statuses, empty
stderr, and 30 decoded frame records matching the CPU reference. Its bitstream,
input description, timestamps, host/kernel/driver and hashes are retained in
`validation/receipt.json`. The earlier default-thread NVDEC failure is retained:
explicitly limiting decoder threads to two avoids requesting 33 decode surfaces.
The earlier broad EGL enumeration's DRI2 warnings are retained.

Moonlight is installed. A real VIGIL CaptureProgressBar component passed ten
pixel assertions against current source without ROS imports. Full native ROS
operator nodes have not been started in the Python 3.13 UI environment; they
require matching rclpy and generated messages. Component probes do not prove a
headed fleet application or a paired remote stream.

## Wireless and shared storage

Three Windows COGROB profiles were imported into root-owned mode-0600
NetworkManager keyfiles. `COGROB_ASUS_5G-2` authenticated and obtained
192.168.50.192. The backup route remained wired, sourced from 192.168.50.43.
The original MWireless profile was preserved and had authenticated before
switching to COGROB. New enterprise templates require PEAP/MSCHAPv2, the
USERTrust CA, and the expected RADIUS hostname. Bounded eduroam attempts with
both `damartel@umich.edu` and `damartel` timed out during association; this does
not establish that the password was wrong. The attempted secret was cleared
and automatic retries disabled while the current credential provider is
identified. Windows authenticated to MWireless using its cached enterprise
profile. Its Credential Manager entry uses `damartel`, but the supported read
API does not expose that protected domain password.

Temporary cleartext Windows WLAN exports remain in private directories on
Windows and Milly. Automatic approval review rejected their removal with
"blocked by policy" and no further reason. No alternate removal path was used.
These exports and NetworkManager secrets are excluded from public reports.

Windows `N:` mounts only the read-only public `/srv/vigil-share` export through
the approved direct LAN. `Mount-VigilFleetShare.ps1` detects approved local
addresses and bounded TCP reachability, preserves foreign/offline mappings,
verifies native mount results, and skips unavailable networks. All 26 tests
pass, including real process/descendant deadlines and failed native results.
The registered per-user hidden logon/network task completed with result zero.
Milly's NFSv4.2 automount was also opened successfully over its wired LAN:
read-only, an eight-second mount deadline, and five-minute idle unmount. The
primary ASUS LAN still needs its Windows NFS firewall/RPC policy resolved before
claiming that Windows route works; the direct-LAN route was observed working.

## Firmware and Thunderbolt

The earlier BIOS update to 2.00 recovered successfully. Intel ME
16.1.42.2872 was installed from the verified payload in a real interactive terminal.
After reboot, fwupd reports current version 1.42.2872 and success state 2 without
an update error. SSH, wired/Wi-Fi addressing, NVIDIA and the package audit recovered.
AMT is enabled but unprovisioned. The current USB Ethernet adapter cannot
provide AMT out-of-band networking.

Neither host has detected a Thunderbolt peer through the present CalDigit
connection. Prepared addressing and reusable capture/configuration tools are
not activated as a working link. The Realtek RTL8157 uses CDC-NCM; current
r8152 lacks its device ID. The official newer driver download requires CAPTCHA
and was not obtained. No Ethernet driver change is attempted during backup.

## Coordination and remaining validation

ASUS Codex was asked for an approved optional video/input surface, a firewall
rule scoped to Milly, and the existing Sunshine authentication provider.
Sunshine is active but currently allowed only over Tailscale; Milly's LAN
port probes timed out. Active Clarius work and another Codex agent's ASUS seat
ownership require display coordination before any stream or input. No Sunshine
credentials, firewall policy, or service state were reset. The signed
optional-endpoint source exists on a remote
branch; its PR publication was rejected by automatic policy, and no alternate
publication path was used.

The TPM owner was contacted in the existing Milly display thread. The
hash-bound `tpm-endpoint-handoff.json` and its Markdown companion identify
existing requirement anchors and the canonical fleet evidence intake. They
preserve pending latency thresholds and exclusions. No clinical, HIL,
requirement, release or fleet acceptance is inferred from component checks.

After backup and OS validation: log `cog` into the actual desktop, establish
the authenticated stream, verify hardware decoding and video/input behavior,
measure latency/load and reconnect/stale-session behavior, and verify existing
ASUS display routing remains intact. Recheck Thunderbolt discovery under the
updated OS. Keep the private backup until the installed system and user data
have been validated.

## Tooling and documentation proposal review

The exact Claude F2 research snapshot and independent review are in
`tooling-modernization/owner-consensus/REVIEW.md`. The original proposal is preserved as a research
artifact; its unverified benchmark, safety, version and configuration statements
are not adopted as facts.

TPM and ASUS owners agree on an additive Sphinx-Needs publishing pilot over the
existing CSV registries and graph, with ID/text/status/link parity, strict warning
and duplicate/dangling-ID rejection, stable anchors and source/tool hashes. Existing
Sphinx/MyST/rosdoc2 and Vale remain the documentation foundation. StrictDoc migration,
replacement tool managers, shared Redis cache and live labgrid deployment are deferred.

The Windows documentation gates now reject requested native build failures and
stale/wrong-target artifacts. Rust defaults stay within the repository; external
roots require explicit opt-in. Cargo JSON selects actual enabled feature-gated
targets and configured target-triple paths. All 25 tests pass, both analyzers report
zero findings, and independent native controls pass 2/2. New-function coverage is
94.24%; full legacy-generator coverage remains limited at 37.20%. Existing Build.ps1,
CargoTools, platyPS, C# XML and rustdoc remain in use.

Tool acceleration needs pinned, architecture-specific source/toolchain receipts,
cold/warm/no-op/one-file-change comparisons and cache-disabled correctness parity.
No speedup is promised. Keep independent test parallelism separate from physical
hardware custody. F1 host-probe, empty-selector and colcon-interpreter findings must
be corrected and reviewed before enforcing tooling floors. Claude's corrected
adoption matrix, source-owner acknowledgement and first-change ownership are pending.
No modernization rollout was performed by this review.
## SSH transport optimization

The applied settings and research are in `ssh-transport/PROPOSAL.md`, with
sanitized before/after policies, implementation receipts and file hashes.
Windows native OpenSSH uses four exact fleet aliases with a ten-second connection
timeout, one attempt and 30-second keepalives. All four pinned hostname checks
passed. Native Windows multiplexing remains disabled; batch remote scripts or
SFTP operations in one process to amortize connection setup.

Cog on Milly uses host-scoped Linux ControlMaster auto, ControlPersist 120 seconds
and owner-only control sockets for ASUS and both Sparks. Existing identities,
host-key pins, routes and the Spark3066 jump are preserved. Three fresh ASUS
commands had median 212.913 ms; three verified reused commands had median
16.847 ms. First reuse was 92.437 ms. These small command samples measure setup
latency, not bulk bandwidth. Idle expiry, missing/stale socket fallback,
alias/account separation and incorrect-pin refusal all passed. A refused local port verifies bounded immediate failure; a separate owned
server accepted TCP and withheld its SSH banner. SSH itself timed out after
1005.981 ms, before the four-second outer deadline. Network reboot/reconnect
behavior under active bulk load still needs a separate test. Brief policy/reuse checks must be repeated after the OS
and OpenSSH upgrade.

Compression stays off on the LAN. Compare payload-specific rsync compression,
whole-file versus delta, SFTP batching and request buffers only with actual data.
Use separate bulk/control profiles, including separate jump-host transports when
needed. Keep streaming on the media transport. Existing authenticated HTTP/gRPC
pooling, Mosh with tmux, WSL OpenSSH and Plink sharing are future options with
owner-specific identity, lifecycle and firewall review. No new pool daemon,
global cipher pinning or ASUS forwarding-policy change was applied.

Upgrade status: package installation continues. During SSH/library replacement,
port22 refused connections and recovery1022 reset; they share system libraries.
The user opened Cog desktop and attached the persistent maintenance tmux, reporting
setting-up progress. This is a user-observed login, not final deployment validation.
Keep power connected and do not reboot until the upgrade completes. For the next
maintenance stage, hold an authenticated administrative session and use a recovery
transport independent of the libraries being replaced.

Normal SSH returned at00:17 UTC. The authenticated administrative session is now
held; recovery1022 was restarted and key-tested at00:19 UTC. Installation continues.

Captured diagnostic source/stdin is stored with `.txt` suffixes. Its original
bytes are unchanged; source-capture-custody.json maps producer names to local
captures. Upstream manifests remain unchanged; local manifests bind stored paths.
