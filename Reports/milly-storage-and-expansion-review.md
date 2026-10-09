# Milly storage and expansion review

Host: `millylaptop1`. Review date: 2026-10-09.
The initial bounded assessment preserved existing data, active builds and the
Clarius operator. After reviewing the SSD contents, the user explicitly requested
wiping it and formatting it for Windows/Linux sharing. That external drive alone
was reformatted. Internal data, active builds and the operator remain preserved.

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
