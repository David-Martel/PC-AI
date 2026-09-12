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
2. Filter Manager events naming a different physical disk are retained under
   `UnrelatedEventId3` instead of degrading this disk. Matching and unclassified
   events remain conservative failures. This distinguishes disk 3 from disk 4
   without discarding unknown `HarddiskVolume` evidence.
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

- Persistent VHD/planner suite: 28 passed, zero failed or skipped, including
  disk attribution, wrong-disk rejection, dry-run failure paths, and actual
  diagnostic preview output. Healthy partition fixtures use real client-only
  CIM instances so they exercise the cmdlet's binding contract.
- Boot validation tools: 14 passed, zero failed, one existing skip.
- PowerShell parsing and `git diff --check` passed. Unconfigured analyzer output
  retains the same three preexisting mount-script advisories; none were added.
- Actual dry runs against attached F: and ext4, using a lookback covering this
  boot, wrote no logs and attempted no mounts. F: remained degraded (40); ext4
  passed (0), retaining the disk-3 event separately.

The boot-wide health report still reports failures for the historical F:/ext4
task results and the 10:51 Filter Manager event. Neither task was rerun merely
to overwrite those results. CloudClients' result 3 gate was preserved and no
cloud clients were launched. Current filter instances on F: include FsDepends,
UCPD, WdFilter, bfs, Wof, and FileInfo; the historical event does not establish
that all filters are absent now. Its underlying invalid-VHD-state cause remains
unresolved. No reboot, unrelated dismount, adapter removal, service reset, or
active-build interruption was performed.

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
