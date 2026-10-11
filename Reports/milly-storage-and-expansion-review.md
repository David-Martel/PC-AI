# Milly storage and expansion review

Host: `millylaptop1`. Review date: 2026-10-09.
The initial bounded assessment preserved existing data, active builds and the
Clarius operator. After reviewing the SSD contents, the user explicitly requested
wiping it and formatting it for Windows/Linux sharing. That external drive alone
was reformatted. Internal data, active builds and the operator remain preserved.

## Active migration successor

Completed at 2026-10-10 02:03:54 UTC (October 9 local): **MIGRATION COMPLETE;
LINUX CONSUMER CHECKS PASS**. The user
approved migration and keeping existing Linux paths usable. The layout and
migration results here supersede the historical exFAT configuration below.
Active Clarius builds, operator, radios, worktrees and toolchains remain protected.

The authorized WDC WDS100T2B0B SSD, serial `1830D0800483`, now has an NTFS outer
volume mounted at `/mnt/dtm-shared` (UUID `1CEDDDB5190C5CEB`). Its sparse 832 GiB
Btrfs image is mounted at `/mnt/pcai-milly-data` (UUID
`09a77a5f-17c9-4495-a88c-049f0b80af73`). The image preserves Linux ownership,
permissions, ACLs, extended attributes, links and executable environments while
shared outer-volume files remain suitable for Windows/Linux transport. Native
Btrfs contents require the Linux mount; Windows access to the raw image and
Windows DACL protection are **NOT TESTED**. The second SKHynix SSD was not formatted.

A pilot demonstrated sparse allocation, growth and remount integrity before the
actual filesystem was grown. Root and both original users were denied writes to
the underlying immutable mountpoint with the image absent; restoration and
owning-user reads passed. This prevents missing removable storage from silently
refilling the internal disk. UUID, serial, backing image, actual loop mount and
outer reserve are checked before mutation. Normal guest-before-outer unmount is
required; no force or lazy removal was used.

The WDC had fallen back to USB 2 at 480 Mb/s. After the safely paused migration
and physical reconnection, its verified route negotiated **10 Gb/s**, with matching
filesystem UUIDs, fixture hash and zero Btrfs device error counters. This is link
speed, not a raw-storage benchmark. The resumed 16 GB archive completed its copy,
verification and source retirement in approximately two minutes.

| Item | Current disposition |
| --- | --- |
| Primary model blobs and complete backup directory | Verified and relocated; seven equal backups and the distinct eighth checkpoint are all preserved. |
| CUDA 12.2 installer | Verified and relocated; installed CUDA was not removed. |
| Complete `milly-y2.tgz.1` download | Verified byte-preserving relocation; archive validity and extracted-tree equivalence remain separate untested claims. |
| `__milly_logs` | Verified and relocated at 00:07:15 UTC. |
| Three pip HTTP cache directories | Verified and relocated; locally built wheel caches retained. |
| `user_study/av_recording` | All 371 large recordings and the remaining tree verified; whole-tree cutover and original retirement complete. Existing Linux path remains usable. |

All eight smaller source objects are verified and retired, accounting for
approximately **65.653 GiB** of their measured allocated bytes. Ordinary internal
availability was **72,584,224,768 bytes (67.60 GiB)** at the study launch after the
last smaller-source retirement at 00:11:19 UTC. Active builds can change that
snapshot; do not equate a free-space difference with exact migration benefit.
Each retired original passed complete source/destination/retained SHA-256,
metadata parity, stable-source checks, fresh process-reference admission and an
actual original-user read through its retained old path. Temporary originals
were held under private root-owned custody before guarded deletion.

All 371 incremental recordings completed at 01:24:23 UTC, preserving
570,880,193,988 logical bytes. The pinned final helper completed whole-tree
cutover and original retirement at 02:03:54 UTC. Its final readback reports
inactive/dead, PID 0, Result success and exit status 0, with all 371 second full
recording hashes present. The existing study root is a symlink to the native
tree; actual owning-user access, `.venv` execution with Python 3.11.13, original
Git HEAD and status parity all passed. Both filesystem UUIDs and the exact helper
binding passed; all five Btrfs device error counters were zero.

The redundant 2,189,554,353-byte partial download was subsequently retired only
after full prefix/source/retained verification and fresh path, process-reference
and filesystem admission. The complete archive remains readable through its
original owning-user path; archive validity is still a separate untested claim.
The original ext4 root reserve of **12,495,987 blocks** was restored after exact
root UUID, NVMe serial, block count/size and current reserve checks. The emergency
30,710,140,928 bytes of ordinary-user headroom are no longer counted available.
The subsequent internal snapshot reports **634,950,225,920 bytes available**
(about 591 GiB); the outer SSD has 374,527,430,656 bytes available and the native
image has 267,087,466,496 bytes available. Concurrent work changes these figures;
they are not an exact migration-only free-space delta or a runtime speedup.

The study helpers preserve the original Git HEAD/status and untracked/ignored
work. Large untracked/ignored regular recordings are copied and independently
verified one at a time, then the original leaf becomes a link to its canonical
native file. The final tree copy explicitly excludes those leaf links so it
cannot overwrite native recordings with links back to themselves. Full recorded
file hashes/portable metadata/ACLs/xattrs are checked again after the remaining
copy. Original directory mtimes, Git status, owning-user access and existing
virtual-environment execution passed before final original-tree retirement.
Incomplete receipts or live source/native consumers were fail-closed holds.

### Lossless compression evidence

Twenty-four native zstd compression/decompression sample trials passed SHA-256
roundtrip verification with unchanged source metadata. Level 3 saved **37.16%**
on the sampled WAV bytes and **1.18%** on sampled MP4 bytes; level 1 slightly
expanded MP4. These samples do not predict Btrfs's smaller extent decisions.
The native filesystem uses automatic `zstd:3`; original file formats and bytes
are retained. No transcoding or format substitution is part of the migration.

Actual extent accounting on the relocated model trees found 9,292,673,024 disk
bytes versus 9,293,807,616 uncompressed bytes: effectively no useful compression
for those models. The read-only native `compsize` utility was extracted from a
hash-recorded Ubuntu package into private operation custody, without changing
system packages. Actual relocated logs used 23,165,292,739 disk bytes for
27,105,029,351 uncompressed bytes (about 14.5% saved). One completed WAV used
1,038,712,832 disk bytes for 1,542,963,200 uncompressed bytes (about 32.7%
saved); an inspected MP4 showed no compression savings. Final whole-study extent
accounting covered 3,763 files: **557,246,863,357 disk bytes versus
572,110,004,663 uncompressed bytes**, saving about **13.84 GiB**. The complete WAV
group (56 files) used **30,022,832,128 versus 44,744,134,656 bytes**, about **32.9%**
or **13.71 GiB** saved. The MP4 group (351 files) used **526,779,543,552 versus
526,780,878,848 bytes**: negligible savings. These are actual native extent
observations, not sample extrapolation or a storage-speed measurement. NTFS image
logical length or potentially stale `stat` block counts are not actual compressed
allocation evidence. [Btrfs compression](https://btrfs.readthedocs.io/en/latest/Compression.html).

### Deficiencies and remaining gates

- Ubuntu's installed uutils `test -r` denied an ACL-authorized user while Bash's
  builtin and that user's actual hash read succeeded. Admission now uses real
  reads. System coreutils was not replaced during active work.
- The first normal native unmount exceeded its 15-second command timeout on the
  slow connection. It subsequently completed normally before the physical move.
  The reviewed native mount-command timeout is now 300 seconds, verified on the
  active mount without remounting; boot guard
  and removable-device deadlines remain bounded separately.
- The dataset migration, original-path consumer checks, final hashes and native
  extent accounting are complete. Preserve independent backup/restore custody;
  a removable relocated copy alone is not an independent backup.
- The prefix-only partial download is verified and retired; complete archive
  validity and extracted-tree equivalence remain **NOT TESTED**. All differing
  backup checkpoints remain preserved.
- Physical Windows attachment, physical M.2 vacancy, cold-boot/removal behavior
  and application/HIL acceptance remain separate **NOT TESTED** gates.
- The failing NFS share remains **OPEN** with its existing server/mount owner;
  active Clarius build growth and release remain that lane's responsibility.

Current completion evidence:
`.pcai/integration/milly-final-progress-r9.txt`,
`.pcai/integration/milly-final-acceptance-r1.txt` and
`.pcai/integration/milly-final-partial-and-reserve-r1.txt`.
Each retained transport receipt reports SSH exit 0. Completion of these storage
checks does not establish Windows runtime, application/HIL or cold-boot acceptance.

Private hash-bound receipts and reviewed scripts are under
`.pcai/integration/milly-migration-*`; manifests and job receipts are in Milly's
root-only `/var/lib/pcai-storage/migration-r1`. Retired source bytes now reside on
the verified SSD. Temporary original copies were deleted after verification;
the private state directory is not a second retained copy.

## Historical assessment and candidate inventory

The following sections retain their original observations and qualification
scope. Their exFAT instructions and unstarted-migration state are superseded by
the active successor above.

## Capacity and active work

SSH verified the machine identity, Lenovo `21FA002UUS` / ThinkPad P16 Gen 2,
Ubuntu 26.04.1 LTS and kernel 7.0.0-38-generic. This supersedes the older Ubuntu
22.04 inventory. One KIOXIA XG8 1 TB NVMe is detected, with an ext4 root filesystem
of approximately 938 GiB. Inodes are only 4% used; this is a byte-capacity issue.

Ordinary-user available space changed during the audit: approximately 3 GiB,
then 1.6 GiB, then 897,024 bytes, then 9.4 GB, and finally 54,807,216,128 bytes
(about 51 GiB) at 20:17 UTC, then 44,008,439,808 bytes (about 41 GiB) at 20:37 UTC.
These are different live snapshots, not a stable
baseline or a cleanup result from this lane. Build-cache paths disappeared or
changed while being measured. The Clarius build owner subsequently confirmed
removing build outputs for nine merged lanes at approximately 20:16 UTC, while
preserving the bench GUI and eight active build lanes. No new CPU offload batch
will be scheduled on Milly during that work. Root reserve initially remained unchanged: 12,495,987 of
249,919,744 blocks, approximately 5%. That reserve explains why available space
can reach zero before every block is occupied; it was not reduced as a substitute
for controlling disk growth during that initial assessment. The reversible
headroom successor below supersedes that setting.

Bounded, low-priority directory scans found these major consumers. Values are
allocated-byte observations; errors and concurrent changes prevent treating them
as an exact, simultaneous accounting of the entire disk.

| Area | Observation | Custody and next action |
| --- | --- | --- |
| First other-user home | Approximately 580 GiB, including a roughly 533 GiB directory | Preserve; identify dataset owner and retention/backup requirements before relocating anything. |
| Second other-user home | Approximately 103 GiB, including about 29 GiB downloads and 25 GiB logs | Preserve; owner must classify useful recordings, logs and unique work. |
| Maintenance user's home | Approximately 155 GiB in the initial scan | Includes active development and caches; not disposable as a group. |
| Maintenance-user cache | Initially about 111 GiB, largely numerous separate Clarius Rust build targets; later scan was lower | Coordinate exact build owners. Offload inactive, preserved targets or use a shared, bounded cache only after consumer checks. |
| `/var` | Approximately 14 GiB; journal approximately 2.7 GiB | No journal evidence or package cache removed. |

A two-second process-I/O sample found Cargo PIDs 2399996 and 2353689 writing
approximately 19.7 and 4.3 MiB/s respectively. This identifies writers in that
interval, not all historical growth. Both were absent at the later 20:17 check.
Clarius operator PID 2300004 had zero `write_bytes` in two observations and was
not blamed or stopped. Active owners received capacity warnings through agent-bus.

The NFS mount `/home/cog/mnt/vigil-share` reports I/O errors. Initial `df` therefore
exited nonzero after producing useful local-device output. Subsequent scans used
explicit local filesystems and excluded the mount's parent. No forced unmount,
server restart or network change was performed. Root/home scans that reached
their timeout remain partial; permission errors and disappearing files are retained.
An initial PowerShell-to-SSH trailing carriage-return error is also preserved;
the corrected transport strips carriage returns before Bash parsing.

### Reversible root reserve headroom

Continued compilation reduced ordinary-user available space to 24.66 GB at
21:25 UTC, 12.17 GB at 21:34 UTC and 6.16 GB immediately before the adjustment.
The former build owner had no current bus presence or acknowledgment; live Cargo
and Rust compiler processes still established active work. They were preserved.

At 21:38 UTC, exact root UUID `adcf1ee1-79b8-4f3d-b170-e843d105d2bd`, KIOXIA
serial `73CFC01EF6HU`, filesystem type, block count, block size and original
reserved count were checked. The reserve was reduced from 12,495,987 to
4,998,394 blocks, approximately **5% to 2%**. With 4096-byte blocks, this exposes
**30,710,140,928 bytes** to ordinary writers while retaining
**20,473,421,824 bytes** for root. Actual available space rose from
**6,158,422,016 to 36,868,550,656 bytes**, reporting 97% usage; concurrent writes
make this a live snapshot rather than an exact measured gain. A later readback
still had 36.57 GB available.

Original metadata and exact-count rollback instructions are preserved in
`/var/backups/pcai-storage/ext4-root-reserve.before-r1.txt` and
`ext4-root-reserve.rollback-r1.txt`; directory mode 700 and file modes 600 were
independently read back. The reserved UID/GID remain root, and root UUID, fstab
and the external SSD's validation hash are unchanged. No file was deleted,
active target moved or foreign process stopped. This is temporary user-space
headroom, not new physical capacity or a completed storage migration. Root
retains about 20 GB of privileged reserve; build growth and offload still need
coordination. [e2fsprogs reserved-block controls](https://manpages.debian.org/trixie/e2fsprogs/tune2fs.8.en.html).

## External SSD

### Initial inspection

After the user reconnected it, Linux detected a **1 TB WDC WDS100T2B0B SATA SSD**
through an ASMedia USB bridge using UAS at a negotiated 10 Gb/s. This is transport
enumeration, not measured storage throughput. SMART overall status passed at 40 C,
with zero reallocated, grown-bad, reported-uncorrectable and interface CRC counts.
Historical timeouts and unexpected power losses remain recorded; no self-test or
write benchmark was performed.

The disk contains NTFS labelled `Wddrive`, starting at byte offset 2048. GPT
inspection reports a valid table with a corrupt protective MBR and an unusual
single-entry layout; ordinary Linux partition enumeration did not expose it.
Top-level files include seven `.xct`, seven `.xvi`, nine `.UWA` files and
`LastConsole`. Xbox-named application packages suggest existing console storage.
That is a content-based inference, not confirmed ownership or proof that the
console's layout should be repaired. Existing payloads were preserved.

An identity-checked, read-only loop mapping and `ntfs-3g` mount with `ro,norecover,
nodev,nosuid,noexec` exposed **1,000,204,877,824 bytes total**, **269,827,952,640
used**, and **730,376,925,184 available** (approximately **680 GiB free**).
The loop's read-only flag was independently checked. Normal unmount and exact
loop detachment succeeded at 20:37 UTC, with both absences verified.

The free capacity makes cold archives or dataset copies plausible offload
candidates. A persistent mount and safe write access remain unqualified, and
NTFS permissions, symlinks, case handling and allocation semantics have not been
validated for Linux Rust targets. Do not direct active builds to this disk yet.
Preserve its existing game/app files, verify copied data before retiring sources,
and establish the console's continuing use before changing its partition layout.
No partition repair, formatting, data removal or offload copy was performed during
that inspection. The following user-authorized successor supersedes its contents
preservation and mount restrictions for this external drive.

### Authorized portable-storage format

The user requested: "Wipe the drive and format it for dual windows and linux use
between dtm-* and milly." Before writes, the host, external drive's exact model,
serial, capacity, internal root identity and absence of mounts, loop mappings and
open users were checked. The old GPT was removed. A transient open-device check
then stopped the first sequence; fresh readback and udev settlement showed no
remaining user, and the successor completed. No unknown process was killed.

The disk now has standard GPT with one Microsoft basic-data partition, both
boundaries aligned to 1 MiB. exfatprogs 1.3.2 formatted it as **exFAT**, label
**DTM_SHARED**, UUID **FBCF-5608**, using 128 KiB clusters and formatter metadata
readback. GPT verification reports no problems and the filesystem checker reports
clean. This is repartitioning and filesystem formatting, not a forensic secure
erasure claim. Former console files are no longer present in the filesystem.

At 21:05 UTC the mounted volume reported **1,000,170,586,112 bytes total**, with
**1,000,152,236,032 available** after the small validation payloads. A 16 MiB file
produced on P1 was copied to the SSD, SHA-256 checked, synced, normally unmounted,
filesystem-checked, remounted and checked again. Its return copy to P1 matched
the original SHA-256. A filename containing spaces and Unicode also survived
remount. This qualifies Linux exFAT writes and file transport to/from P1;
**a physical Windows attachment is NOT TESTED**.

On Milly it is available at **`/mnt/dtm-shared`**. Its optional UUID-based systemd
automount uses the live maintenance account's UID/GID 1002, `umask=0022`, noatime,
nodev, nosuid and noexec. `nofail`, a five-second device timeout and 120-second
idle timeout keep removable storage out of the required boot path. Original
`/etc/fstab` bytes are preserved under root-only `/var/backups/pcai-storage`.
Automount activation and payload hash readback passed at 21:09 UTC.

Full `findmnt --verify` still reports the preexisting required NFS target's I/O
error and a swapfile advisory. The original and staged configurations had zero
parse errors and the same environmental failures; original-byte prefix and the
single appended SSD entry were checked separately before atomic installation.
The NFS entry, swap configuration and internal mounts were not changed.

Use this volume for shared files, datasets and archives. exFAT lacks NTFS's
permissions, hardlinks and journaling, so it is not an equivalent native build
filesystem. Keep permission-sensitive trees in suitable archives or qualify a
native filesystem/image workflow before relocating build targets. Always use
normal OS unmount/eject before physically moving the drive between machines.
[Microsoft filesystem comparison](https://learn.microsoft.com/en-us/windows/win32/fileio/filesystem-functionality-comparison).

## Internal expansion

Lenovo specifies **two M.2 2280 PCIe 4.0 x4 storage slots**, with tested offerings
up to 4 TB per drive / 8 TB total. One active internal drive was detected.
Therefore the second storage slot is a plausible expansion path, but software
enumeration does not prove physical vacancy: a disabled or faulty device would
also be absent. SMBIOS exposed only a SIM-slot record and did not independently
identify free M.2 positions. Confirm the vacant slot and required mounting/thermal
parts physically before purchase or installation. The SIM/WWAN connector is not
the storage expansion evidence. [Lenovo specifications](https://psref.lenovo.com/syspool/Sys/PDF/ThinkPad/ThinkPad_P16_Gen_2/ThinkPad_P16_Gen_2_Spec.PDF).

## Remaining actions

### Offload and deletion identification refresh

The user's follow-up requested identification of safe storage candidates. The
22:47–22:55 UTC read-only audit did not move or delete internal files, remove
packages, stop builds or retire worktrees. All sizes below are **allocated GiB**,
not a promise of simultaneous or immediately reclaimable space. At 22:55 UTC,
root was 96% used with 43,900,284,928 bytes available (40.89 GiB); the external
SSD had 1,000,152,236,032 bytes available (931.46 GiB). A new build target appeared
during the audit, so this headroom remains subject to active growth.

#### Prioritized offload candidates

| Source | Allocated GiB | Classification and required preservation |
| --- | ---: | --- |
| `/home/yayuanli/fun/ICON/user_study/av_recording` | 532.83 | Largest potential relief. Study recordings, embedded Git repository and environment files; preserve together until the data owner releases an exact source snapshot. Never classify the tree as junk. |
| `/home/millyptg/__milly_logs` | 25.44 | Research/session recordings and logs. Archive closed sessions with their metadata and provenance; do not apply system-log retention to this directory. |
| `/home/millyptg/__milly_blobs` | 8.02 | Model checkpoints. Preserve unique models and training provenance before moving or consolidating them. |
| `/home/millyptg/Downloads/milly-y2.tgz.1` | 14.92 | Retain the complete-sized training download; archive integrity and equivalence to the extracted tree are **NOT TESTED**. |
| `/home/millyptg/Downloads/cuda_12.2.2_535.104.05_linux.run` | 4.05 | Prefer SSD retention over deletion: CUDA 12.2 is still installed, and offline reconstruction/provenance may need this installer. Do not remove the installed toolkit. |

These non-overlapping sources occupy approximately **585.26 GiB**, comfortably
within the attached SSD's observed capacity. This is a conditional migration
budget, not a completed copy or guaranteed reclaim. The extracted
`/home/millyptg/Downloads/milly-y2` is another 8.02 GiB; keep it until its source,
local changes and consumers are reconciled. Do not count the whole 29.21 GiB
Downloads directory again on top of its listed children. The study's sibling
`realsense` directory occupies only 4096 bytes in this snapshot.

Use an owner-separated destination such as `millylaptop1/<source-user>/` on
`/mnt/dtm-shared`, with a new artifact identity for conflicting versions. Merge
only SHA-256-identical files; keep different contents separately without
overwriting either. For Linux trees, preserve ownership, permissions, timestamps,
links, Git history, ignored/untracked files and extended metadata in a suitable
archive rather than assuming exFAT preserves their original semantics. The
recording tree includes a `.venv/lib64` symlink and its own Git repository; a flat
media copy does not preserve that entire tree. Avoid redundant compression of
already compressed recordings merely to claim a smaller copy.

Before retiring any source, capture a released snapshot and manifest, copy
directly to the SSD without staging hundreds of GiB on root, sync and read back
hashes, verify archive/restoration behavior and update actual path consumers.
Ensure removable-drive absence has a clear failure path. The current exFAT mount
uses a common maintenance UID and `umask=0022`; it does not preserve per-user
confidentiality. Owner-approved protection is required for private study data.
A removable copy alone is not an independent backup. Full dataset copies,
restore tests and consumer migration are **NOT TESTED**.

#### Deletion candidates and explicit holds

| Item | Allocated size | Evidence and safe disposition |
| --- | ---: | --- |
| Seven files in `/home/millyptg/__milly_blobs_bk` | 655,335,424 bytes / 0.610 GiB | Each SHA-256 matches its same-named primary checkpoint, with distinct inodes and unchanged size/mtime/ctime across hashing. Byte-redundant copies are eligible for consolidation after preserving required backup retention and checking live consumers. Exact names/hashes remain in the private receipt. |
| `Downloads/milly-y2.QjEOhh8x.tgz.part` under `/home/millyptg` | 2.04 GiB | Every byte matches the prefix of retained `milly-y2.tgz.1`; both files stayed unchanged across comparison. Eligible redundant-download cleanup while the larger file and reconstruction length are retained. This does not prove the larger archive is valid. |
| `/home/millyptg/.cache/pip/http` and `http-v2`; `/home/yayuanli/.cache/pip/http-v2` | 13,512,777,728 bytes / 12.58 GiB combined | Conditional regenerable download-cache cleanup after checking live installers and required offline sources. Keep the separate `wheels` directories (107,380,736 bytes combined) until locally built artifacts are reconciled. Do not remove installed environments or purge both user cache roots indiscriminately. |
| Fourteen disabled Snap revisions | 3.12 GiB | Conditional manager-controlled retirement after rollback/dependency and active-operation checks. All remained mounted; never unlink their `.snap` files directly. |
| Old VS Code `.deb` in `/home/millyptg/Downloads` | 95,653,888 bytes / 0.089 GiB | Downloaded 1.83.1 installer; installed Code reports 1.140.0. Eligible replaceable-installer cleanup after preserving offline needs. No current file user was reported by the exact-path check. |
| APT download cache | 708,608 bytes | Small regenerable-cache candidate after package-manager lock/operation checks; negligible relief. |
| System journal | 2.78 GiB total | Preserve the diagnostic window before any manager-controlled archival/retention change. Archived-looking filenames are not proof of inactivity: journald has one such file open. Total journal size is not safely deletable size. |

The eighth backup checkpoint is **different** from its same-named primary:
24,084,480 bytes versus 93,617,003 bytes and different SHA-256 values. Preserve it;
the `_bk` suffix does not establish duplication. The empty training-download
placeholder would recover zero bytes and is not a meaningful pressure fix.

The HTTP/wheel distinction agrees with [pip's cache documentation](https://pip.pypa.io/en/stable/topics/caching/).
Cache removal trades space for later download/build cost; it is not a measured
runtime acceleration. Keep useful caching bounded rather than disabling it
globally. Snap revision retirement must follow [Snap's documented removal path](https://snapcraft.io/docs/tutorials/get-started/).
Its snapshot/rollback behavior can affect net space reclaimed, so the disabled
revision total is a candidate size rather than an assured immediate gain.

Twenty Clarius Cargo target roots occupied **90.06 GiB** at 22:55 UTC; the compiler
cache occupied another **7.65 GiB** in its earlier snapshot. These are **HOLD**,
pending release by the active build owner. The operator maps its executable from
`/home/cog/.cache/claude-clarius-target-overlay`, while `onebin` and the newly
created `train` target have live work. Even regenerable outputs can contain the
exact deployed executable or qualification build. Preserve those artifacts and
their hashes before selective cleanup. No cross-root shared hardlink allocation
was detected, but within-root hardlinks mean child-size sums are not independent.

Fresh Clarius custody found 29 linked worktrees plus primary, merge operations
in two bench lanes, dirty work and a held pytest-gate lock. The prior nine-lane
cleanup is historical; it does not release these current trees. The owner confirms
an active merge train and protected recipe work. All three installed Rust
toolchains have current pinned/default consumers and remain **KEEP**. Preserve
repositories, Git bundles, certificates, package environments, model caches,
`/var/cache/vigil-downloads` and the installed CUDA/NVIDIA stack until their exact
consumers and provenance are verified. `/tmp` is tmpfs, so clearing it would not
recover internal NVMe capacity.

Exact-path handle probes found no matching recording/backup users, but `lsof`
warned that existing NFS and FUSE mounts could not be inspected. This is limited
negative evidence, not proof of owner release or lack of future consumers.
Private evidence is under `.pcai/integration/milly-offload-*`,
`milly-recording-custody-r1.txt`, `milly-build-cache-audit-r1/` and
`milly-system-cache-audit-r1/`. Root owns classification and migration admission;
data owners own retention and path changes; the active Clarius lane owns build
release; system owners own package/log retirement. No cleanup benefit or
performance improvement is claimed from this identification-only refresh.

- Storage: user-authorized format, Linux write/remount checks, P1 file roundtrip
  and optional UUID automount are complete. Physically attach to a Windows host
  to qualify that host's native mount and safe-eject path.
- Build owners: bound concurrent target growth and identify inactive preserved
  Clarius outputs; root: qualify an offload destination and migration controls.
- Data owners: classify the two other-user homes before archive/offload decisions.
- Operator: confirm physical M.2 vacancy; root: document installation configuration.
- Mount owner: restore the failing NFS share through its existing server lane.

Private raw receipts are under `.pcai/integration/milly-storage-*`,
`.pcai/integration/milly-ssd-*` and `.pcai/integration/milly-writer-*`.
The scans and warnings establish current risks
and candidates; the format and validation do not claim a migration of internal
user data or a workstation speedup.

### Current storage readback and historical-summary clarification

Readback at 2026-10-10 08:43 UTC confirms the completed migration layout remains mounted. Ordinary internal root availability is **623,514,374,144 bytes (580.69 GiB)**; the NTFS outer volume has **374,527,430,656 bytes** available and the native Btrfs image has **267,087,466,496 bytes** available. These are fresh snapshots during concurrent work, not an exact migration-only space delta or a speedup measurement. Do not add outer-volume and image availability as independent physical capacity.

Both external SSD links are currently **10 Gb/s using UAS**. The WDC WDS100T2B0B, serial `1830D0800483`, maps from `/dev/sda` to USB node `4-2.3`; the SKHynix HFS001TEJ9X162N, serial `AYCCN03781CB9190E`, maps from `/dev/sdb` to USB node `2-4`. Native sysfs speed reads are 10000 for both device nodes and agree with `lsusb -t`. The second drive remains VFAT mounted at `/run/media/cog/FATDRIVE`, with **1,023,781,535,744 bytes** available; this audit did not change its contents, format or intended use.

As the original owner `yayuanli`, the existing `/home/yayuanli/fun/ICON/user_study/av_recording` path resolves to the mounted native tree, and actual directory traversal/stat succeeds. Its Btrfs UUID remains `09a77a5f-17c9-4495-a88c-049f0b80af73`, backed by `/mnt/dtm-shared/pcai-migration/milly-native-data-r1.img` on NTFS UUID `1CEDDDB5190C5CEB`. All five Btrfs device counters remain zero. This refresh checks the current directory/mount/device chain; it does not repeat full recording hashes or virtual-environment/application tests.

The report's final identification-only wording belongs to its **historical assessment**, superseded by the completed migration section and checked-off migration items in `boot.TODO.md`. Current remaining work is independent backup/restore, physical Windows mount/runtime/DACL, cold boot/removal, physical M.2 vacancy and application/HIL qualification, together with the existing NFS/build-growth owners. This readback establishes neither measured storage throughput nor workstation acceleration, and does not infer an unused M.2 slot.

Fresh native SSH exit/stdout/stderr receipts: `D:/pcai-relocation/milly-fastlink-current-readback-r4/`; their pins, scope and proposed addendum are recorded in `.pcai/integration/milly-fastlink-current-readback-r4/readback.json`. All eleven checks exited0 with empty stderr. No remote writes, remounts, deletion, data-heavy scan or data-file rehash occurred.
