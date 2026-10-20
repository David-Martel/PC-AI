# Milly fleet endpoint maintenance

Updated 2026-10-03 UTC. Milly has recovered from the final OS, graphics-driver and
firmware reboot. Cog is logged into its local Wayland desktop; temporary GDM
autologin was restored to disabled afterward. Local component validation is
complete with the qualifications below. Thunderbolt peer networking, authenticated
Sunshine streaming and full VIGIL operator-UI acceptance remain pending.

## Installed system and capability

Milly is a Lenovo ThinkPad P16 Gen 2: Intel i9-13950HX, 64 GB RAM, NVIDIA RTX 5000
Ada Laptop GPU with 16 GB VRAM, and a 3840x2400 panel. It now runs Ubuntu 26.04.1
LTS, kernel 7.0.0-38, system Python 3.14.4, and Ubuntu's recommended NVIDIA
595.91.07 open driver. Cog has its own SSH/fleet/GitHub key, passwordless sudo and
verified SSH Git signing. Private keys were not copied between hosts.

The endpoint remains optional: no production compute placement, release quorum,
required-host membership or static DDS peer was added. VIGIL currently requires
Python >=3.12,<3.14 and Jazzy; native Lyrical/Python 3.14 is a separate development
installation. Pinned Jazzy/Python 3.12 and Lyrical/Python 3.14 containers both pass
network-isolated initialization checks. No production ROS domain was used.

Managed Python 3.12.15, 3.13.16 and3.14.8 remain available. An isolated3.15.0rc2
preview now passes SSL/SQLite checks; it does not replace system Python or VIGIL's
ABI. The installed uv catalog offers rc2; upstream released rc3 on Oct2 and plans
final3.15 for Oct9, so this preview is explicitly not the latest upstream candidate.
See [Python's release notice](https://www.python.org/downloads/release/python-3150rc3/).

## Firmware, storage and maintenance

BIOS 2.00 N3TET64W and Intel ME 1.42.2872 were applied and reboot-verified earlier.
Kioxia SSD 5108APLA is now active: LVFS's release fixes a specific host-write-pattern
hang. NVMe SMART reports zero media/errors, zero critical warnings and100% spare.
The TRIM timer is active; its dry-run succeeds. The existing balanced power profile
is preserved; performance is available without degraded status. No overclock,
undervolt or speculative global networking/storage tuning was applied.

UEFI dbx 20260707 is present. Native fwupd 2.1.1 initially reported a postboot
expected-version/null failure because its recognition data lacks this release.
Independent exact signature-type/owner/data comparison finds291/291 signed
update entries and371/371 Ubuntu boot-update entries in the live database.
Four parser controls pass, including missing-entry, changed-owner and truncated
input rejection. A two-line exact checksum/version mapping from pinned upstream
fwupd 2.1.8 was installed through the documented local quirks directory. Fresh
fwupd now reports20260707; EFI bytes stayed unchanged. Original failed history
is retained. Secure Boot remains disabled, so entry presence is not enforcement.

The migration followed22.04→24.04→26.04. Native Humble's282 incompatible packages
were removed after reviewing their scope. The26 upgrader returned1 on unused
Postfix's missing-configuration migration. A reviewed removal of only Postfix
resolved it; dpkg audit, apt dependency checks and subsequent reboot passed.
No obsolete-package autoremove or backup deletion was performed. Captured native
failure and repair receipts are preserved beside the final evidence.

## vPro and remote recovery

Milly's AMTControl BIOS setting remains enabled. The earlier capability check
found AMT unprovisioned; no working out-of-band power/KVM path is claimed. Its
ordinary USB Ethernet adapter cannot substitute for an AMT-supported Intel path.
Use SSH for OS telemetry and the selected media client for normal UI. A future
MEBx/TLS provisioning step needs distinct vault-managed AMT credentials, an approved
Intel wired/dock or managed Wi-Fi profile, and explicit AC/sleep/off recovery tests;
see vpro-findings-and-sources.json for primary vendor guidance.

Fresh ASUS discovery identifies NUC13ANKH7 and an i7-13620H CPU. Do not assume
AMT from its hostname or a general vPro label; no ASUS AMT recovery path was
verified or provisioned. The CPU specification's Thunderbolt4 support alone does
not establish an AMT capability or a negotiated Thunderbolt peer link.
## Installed tools and live checks

New native tools include Lyrical desktop/dev-tools/CycloneDDS, clangd/lldb 21,
fio 3.41, OBS 32.2 from its Resolute PPA, Nsight Systems 2026.3.2 and NVIDIA Container
Toolkit 1.20.1 from official signed repositories. NVIDIA's profiling repository is
pinned to permit Nsight packages while excluding its driver/CUDA packages.
Podman's CDI spec regenerated successfully at boot and a network-isolated pinned
container sees the595 GPU. Missing IMEX utilities are retained diagnostic warnings
for a facility this laptop does not use. No Docker runtime or fleet compute role
was enabled. [NVIDIA documents CDI for Podman](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

Existing uv/Rust/Clang 23/CMake/Ninja/sccache/nextest/just, Python analysis/testing,
BLE/network/storage tools, Moonlight 6.1, Qt 6.11.2 and BitwardenCLI remain installed.
Post-upgrade native C++ build/CTest, six Python tests and Rust nextest pass. EGL
compiles with -Wall/-Wextra/-Werror and verifies an actual GPU-rendered red pixel;
its two DRI2 diagnostics remain disclosed. Synthetic NVENC H.264 encode and CPU/
NVDEC decode pass with all30 decoded frames identical. Qt's two pixel assertions
and the actual VIGIL progress widget's ten assertions pass.

Native Lyrical/CycloneDDS publishes/subscribes three exact messages inside a
private network namespace/domain231. Pinned Jazzy and Lyrical containers initialize
with network=none. These are ABI/component checks, not cross-distro fleet DDS
interop. Podman retains its legacy BoltDB warning; no destructive reset was used.
A synthetic Nsight capture generated a125KB report with no collection errors.

OBS reaches startup-complete and detects NVIDIA H.264/HEVC/AV1 encoders. VLC's
library/base plugins were added for its media-source support. The first temporary
startup opened default desktop/microphone audio inputs for 8 seconds; no streaming
or file-recording command was used. The owned process stopped. Those audio sources
were removed from the private test scene, and the second startup asserts no audio
input started. Optional AJA/DeckLink-without-hardware and Xvfb EGL/EWMH/VAAPI
warnings remain; this is startup validation, not a physical recording session.

Cog's current GNOME shell and audio services run normally, with zero audio-service
restarts and four audio control probes passing. Historical WirePlumber, greeter
and Orca crash evidence remains. The automatic crash uploader timed out; it was
not reset to manufacture a passing service state. Physical audio, accessibility,
camera DMA-buffer support, input and full operator UI remain unvalidated.

Initial final-test commands contained a module-path error, container shell-quoting
error and duplicate SSH-M/ControlMaster option. Their failure receipt is retained;
corrected targeted checks pass. The system discovery receipt's firmware-pending
expectedExit2 is a harness annotation error: JSON mode returned0 with an empty
Devices list. It is not a firmware failure or a blanket all-checks acceptance.

## Data custody

Keep the private ASUS backup at
`/srv/vigil-backups/millylaptop1/20261002`, outside the public NFS export. Its final
pre-upgrade copy contains789,737,621,693 file bytes; all three homes and EFI had
zero checksum differences, nine restore samples passed, and seven bounded live
system-log/timer/NetworkManager exceptions were qualified. It is not an exact
frozen whole-system snapshot.

Post-OS checksum dry-runs describe57 yayuanli and59 millyptg differences, all
confined to Firefox7355→8995 revision migration and parent-directory time. No
unexpected original-user path was found. Cog's863 paths are narrowly accounted
for; five SSH and two GPG files match the backup, six repositories remain clean
at their baseline heads, and prior shell history is retained as an exact prefix.
One desktop database's schema/metadata migration remains semantically unqualified;
Millyptg retains inherited unresolved Firefox theme links. These checks combine
frozen post-OS logs and later narrow live probes, not a fresh frozen whole-home
snapshot after final GUI login. User/browser/data validation still precedes cleanup.

## Network and rendering boundary

Wired192.168.50.43 and COGROB Wi-Fi192.168.50.192 are connected. Three protected
COG profiles were installed; existing MWireless was preserved. Eduroam/MWireless
have not successfully associated in the bounded tests. University profiles retain
CA/server validation and do not autoconnect with unverified credentials. No secret
was placed in the reports or agent bus.

Windows N: mounting probes LAN availability with bounded failure, preserves foreign
mappings and offers the read-only fabric fallback. Its26 tests and analyzer pass.
Milly's read-only NFSv4.2 mount works. Private backups remain outside that share.

Both final machines still expose no Thunderbolt peer/network adapter through the
CalDigit/Apple cable. Linux thunderbolt and thunderbolt_net are loaded. Windows
native USB4 inventory and reusable peer-profile tooling are prepared; no unsupported
adapter binding, guessed vendor driver or claimed40Gb/s throughput was introduced.
The observed iperf results are Ethernet, not Thunderbolt. A direct-cable isolation
or verified peer-capable topology is still needed. CalDigit's vendor updater did
not recognize Element5; no hub flash/reset was performed. Active Windows storage
and other owners' hardware sessions are preserved.

Moonlight is installed and tested. ASUS Sunshine's existing LAN/firewall, seat and
credential-provider decisions remain pending; no authentication reset, stream,
input injection or firewall change was performed. Full native VIGIL UI also needs
matching Jazzy rclpy/vigil_msgs/vigil_c2 build artifacts; generic PySide installation
does not satisfy that contract. The architecture review names existing entrypoints
and the streaming route; the browser-control gateway is not deployed.

## SSH and project coordination

[The SSH proposal](ssh-transport/PROPOSAL.md) records applied host-scoped policies.
Windows retains native OpenSSH with bounded connect/keepalive behavior on four
fleet aliases; its multiplexing probe is unsupported. Cog's Linux profiles reuse
private control sockets for ASUS and both Sparks. Actual reuse passed again after
OpenSSH 10.2. Initial three-command ASUS samples measured median212.913 ms fresh
versus16.847 ms reused, not bulk bandwidth. Pins, keys and the3066 jump are retained.
Missing/stale sockets, idle expiry, wrong-pin refusal and a1.006 s stalled-banner
timeout passed earlier. Compression stays off for LAN/media. Separate bulk and
control transports and measure payload-specific compression/buffering before tuning.

ASUS→Milly, Windows→Milly/Sparks and Milly→ASUS/bothSparks key access pass. ASUS
multiplexing/agent-forwarding policy remains with its owner; no owner's live socket
was terminated. The [TPM handoff](tpm-endpoint-handoff.md) binds supporting evidence
to existing requirements/HOLD/Pending states without new acceptance or thresholds.

Claude's exact tooling proposal was reviewed with TPM/ASUS: adopt an additive
Sphinx-Needs view of canonical CSV/schema/DAG with parity, stable anchors, negative
controls and source/tool hashes. Keep Sphinx/MyST/rosdoc2/Vale and existing Windows
Build.ps1/CargoTools/platyPS/C#XML/rustdoc. Defer global tool managers/shared Redis
cache/StrictDoc migration/labgrid/GPU floors. Invalid Cargo/ccache/colcon and
cargo-llvm-cov recipes plus unreceipted speedups need correction; a named first
publishing implementer remains pending. Windows fail-closed docs fixes are signed
and pass25 tests, both analyzers and2 native failure controls; new-function coverage
is94.24%, full legacy coverage37.20%. Do not infer rollout/performance from those checks.

Evidence is in `post26-final/`, with original failures and SHA256 custody bindings.
Local support is validated; paired streaming, Thunderbolt, university Wi-Fi,
physical rendering/control/HIL and user-data cleanup remain explicitly pending.
Private Windows WLAN-export cleanup was rejected by automatic approval policy
("blocked by policy", no more specific reason supplied); the exports remain under
private ACLs and no alternate deletion path was used.