# Milly fleet endpoint maintenance

Checkpoint: 2026-10-02 21:56 UTC. This report records work in progress. The full
backup, operating-system migration, graphical login, paired stream, and final
data validation are not complete.

## Current machine and access

`millylaptop1` is a Lenovo P16 Gen 2 with an i9-13950HX, 64 GB RAM, RTX 5000 Ada
Laptop GPU with 16 GB VRAM, and a 3840x2400 panel. It currently runs Ubuntu
22.04.5 with kernel 6.8.0-138 and NVIDIA 580.178.04. `cog` has working SSH keys,
passwordless sudo, its own fleet/GitHub authentication key, and independently
verified SSH Git signing. Private keys were not copied from another host.

The endpoint is an external development and operator surface. It is not added
to the production fleet membership, compute placement, release quorum, or
static DDS peer set. Native Humble is present on the old OS. Current VIGIL
contracts use Jazzy/Python 3.12; separately validated, pinned Jazzy and Lyrical
containers preserve their respective ROS/Python ABI. Managed Python 3.14 is
available alongside 3.12 and 3.13. The system Python was not replaced.

## Backup and maintenance boundary

The private recovery copy is at
`asuspro13:/srv/vigil-backups/millylaptop1/20261002`, outside the NFS export,
with root-owned private parent directories. The source baseline is
775,325,002,378 bytes (775.325 GB or 722.078 GiB), including all three home
directories and a separate EFI copy. At 21:54 UTC, 402.53 GB had transferred;
the original copy remained active at approximately 116 MB/s. That rate is
consistent with the ASUS destination's verified 1 Gb/s Ethernet link.

The user approved logout and maintenance; the prior desktop user was logged
out and quiescence checked. Release upgrades and the staged ME firmware flash
wait for the initial copy, a final consistent delta, complete included-data
checksum comparison, and restore checks. The approved upgrade path is
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
USERTrust CA, and the expected RADIUS hostname. One saved Bitwarden MWireless
credential failed eduroam; that rejected credential was cleared and automatic
retries disabled while the current credential provider is identified.

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
The primary ASUS LAN still needs its NFS firewall/RPC policy resolved before
claiming that Windows route works; the direct-LAN route was observed working.

## Firmware and Thunderbolt

The earlier BIOS update to 2.00 recovered successfully. Intel ME
16.1.42.2872 is staged with a verified SHA-256 and exact live device/GUID
match; fwupd reports trusted payload and metadata. It has not been flashed.
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
port probes timed out. No Sunshine credentials, firewall policy, or service
state were reset. The signed optional-endpoint source exists on a remote
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
