# Milly storage and expansion review

Host: `millylaptop1`. Review date: 2026-10-09.
This is a live, bounded, read-only assessment. Existing user data, active builds
and the Clarius operator were preserved. The external SSD was temporarily mounted
read-only and released normally; no disk was formatted, repaired, repartitioned,
cleaned or moved.

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
will be scheduled on Milly during that work. Root reserve remains unchanged: 12,495,987 of
249,919,744 blocks, approximately 5%. That reserve explains why available space
can reach zero before every block is occupied; it was not reduced as a substitute
for controlling disk growth.

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

## External SSD

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
No partition repair, formatting, data removal or offload copy was performed.

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

- Storage owner: confirm continuing console use and intended offload data; then
  qualify a persistent mount and write/copy/hash verification without altering
  existing payloads. Read-only inspection and temporary-mount cleanup are complete.
- Build owners: bound concurrent target growth and identify inactive preserved
  Clarius outputs; root: qualify an offload destination and migration controls.
- Data owners: classify the two other-user homes before archive/offload decisions.
- Operator: confirm physical M.2 vacancy; root: document installation configuration.
- Mount owner: restore the failing NFS share through its existing server lane.

Private raw receipts are under `.pcai/integration/milly-storage-*`,
`.pcai/integration/milly-ssd-*` and `.pcai/integration/milly-writer-*`.
The scans and warnings establish current risks
and candidates; they do not claim a storage migration or workstation speedup.
