# Boot storage repair — September 12, 2026

W: was missing because its startup task pointed to `T:\vm\shared-dev.vhdx`,
which does not exist. The historical disk is present at `D:\vm\shared-dev.vhdx`.
The maintained mount task restored W: successfully at 18:02:41 Eastern, with
`LastTaskResult=0`. No data was copied, formatted, resized, or replaced.

## Identity and recovery evidence

- The September 9 successful receipt
  `Logs/VHDMount/AutoMount_VHDX_shared-dev/20260909-080111-9106800a.result.json`
  identifies disk `6E7D4FB3-3CE3-41EF-B157-FB977E128EE4`, a 512 GiB dynamic
  VHDX with a 328,401,420,288-byte backing file on D:.
- Fresh metadata matched that identifier and size. W: was free, the disk was
  detached, no VM configuration referenced it, and Handle found no open handles.
- A read-only attachment with no drive letter verified healthy NTFS label
  `WSL-Shared-Dev` and volume GUID `15fbecd4-e393-4c4f-90c0-4154acfee482`.
  Only that inspection attachment was detached afterward.
- The live task's sole argument change was the backing path from T: to D:.
  Exported XML and the task security descriptor were otherwise identical.
  The normal maintained task then mounted the verified disk as disk 5 / W:;
  partition-to-disk and volume-GUID checks passed.
- Final receipt:
  `Logs/VHDMount/AutoMount_VHDX_shared-dev/20260912-180241-27fa46d6.result.json`.
  D: currently identifies as healthy 8 TB NVMe / NTFS `nvme-scratch`. Older
  descriptions of an absent removable enclosure are historical, not current
  disk identity checks. The unrelated 6 MiB `T:\vm\wsl-shared.vhdx` was untouched.

## Maintained source corrections

1. Mount dry runs now suppress directory, transcript, JSON, and Windows event
   writes across successful and failing paths, while retaining meaningful exit
   codes. They do not mount detached disks.
2. Filter Manager events during the inspection naming a different physical disk
   are retained under `UnrelatedEventId3`. For a newly attached VHD, events
   predating the actual Mount-VHD call are retained under
   `BeforeAttachmentEventId3`: Windows can reuse an earlier disk number.
   Already-attached disks retain older events conservatively because their
   earlier disk number is not proven. Matching and unclassified events,
   including unknown timestamps and `HarddiskVolume` names, remain failures.
3. Expected-drive fallback resolves the partition and verifies its disk number
   before accepting the volume. A different disk with the same letter and label
   cannot validate the VHD under inspection.
4. Registration and both maintained diagnostic inventories now use the verified
   D: backing path for shared-dev. Other VHD paths remain unchanged.

The retired `Gemini-CLI-Update-stable` task was disabled after XML backup. Its
repo-owned wrapper exists, but deliberately throws because the original
`C:\Users\david\gemini-cli\update-scripts\check-releases.ps1` is missing.
The only task XML change was explicit `Enabled=false`; the original omitted
that element and therefore defaulted to enabled. No updater was executed.

## Validation and remaining limits

- Persistent VHD/planner suite: 34 passed, zero failed or skipped, including
  disk attribution, wrong-disk rejection, dry-run failure paths, and actual
  diagnostic preview output. Healthy partition fixtures use real client-only
  CIM instances so they exercise the cmdlet's binding contract. Temporal
  regressions cover pre-attachment, at-attachment, and post-attachment events,
  and conservative handling of earlier events on already-attached disks.
- Boot validation tools: 14 passed, zero failed, one existing skip.
- PowerShell parsing and `git diff --check` passed. Unconfigured analyzer output
  retains the same three preexisting mount-script advisories; none were added.
- Actual dry runs against attached F: and ext4 wrote no logs and attempted no
  mounts. The first implementation separated their disk numbers; further event
  correlation demonstrated number reuse. The final implementation preserves
  older events conservatively for already-attached disks and separates events
  predating a new attachment, as covered by the temporal regressions.

The initial boot-wide health receipt retained the historical F:/ext4 task
results and the 10:51 Filter Manager event. Further VHDMP correlation identified
a different Windows Containers disk using number 3 before F: was attached.
After confirming the current VHD identity and filter instances, the coordinating
agent ran the existing cloud-cache validation task at 18:20:19 Eastern. It
returned 0, with `AlreadyAttached=true` and `MountAttempted=false`; receipt
`Logs/VHDMount/AutoMount_VHDX_cloud-cache-disk/20260912-182020-30383ee3.result.json`.
This validated current state without remounting or changing the gate.

The coordinating agent then ran CloudClients with `-DryRun -RequireMountLog
-AllowElevated -TimeoutSeconds 10`: exit 0, current-boot mount receipt accepted,
Google Drive already running, Dropbox only proposed, and counts started 0 /
skipped 1 / failed 0. No client was launched. Correlation and current-state
receipts are in `~/.codex/backups/machine-boot-rdp-20260912/`:
`cloud-cache-event-correlation.json`, `cloud-cache-validation-result.json`, and
`cloud-startup-dryrun.txt`.

Current filter instances on F: include FsDepends, UCPD, WdFilter, bfs, Wof, and
FileInfo. The historical event does not establish that F: lacks filters now.
Historical task results and the original event remain preserved. No reboot,
unrelated dismount, adapter removal, service reset, or active-build interruption
was performed.

## Rollback and handoff

Source preimages, task XML/security, read-only inspection, final mount result,
and health receipts are preserved under the machine-local audit directory
`C:\Users\david\audits\2026-09-12\boot-storage-c8a39153c191479397ec6534f74ffb0a`.
Restore task configuration only from the exact XML preimages after reviewing
current ownership. Restoring the old T: argument would reintroduce the missing
file failure. Re-enabling the retired updater restores its known failure.
Do not dismount W: after applications begin using it without a fresh handle and
ownership check. The unrelated watchdog JSON and keyboard investigation remain
owned by their existing workstreams.
