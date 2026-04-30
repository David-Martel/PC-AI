# OneDrive Repair Report - 2026-04-30

## Summary

OneDrive was repaired with the Microsoft standalone installer and then reset
with `onedrive.exe /reset`. The reset/start operations were run through a
temporary limited interactive scheduled task because direct elevated launch
returned `0xC025001C`.

Current state after repair:

- `OneDrive.exe`, `FileSyncHelper.exe`, and `FileCoAuth.exe` are running.
- Short post-reset health checks pass:
  - `Reports\sync-provider-health-post-onedrive-reset-5min.json`
  - `Reports\boot-mount-health-post-onedrive-reset-5min.json`
- Long 240 minute health checks intentionally still fail because they include
  earlier same-session OneDrive WER and FilterManager evidence.
- `Tools\Repair-OneDriveSync.ps1` now records non-zero scheduled-task results
  loudly with decimal and hex result codes plus live process state.

## Repair Actions

1. Downloaded official Microsoft OneDrive setup binary from:
   `https://go.microsoft.com/fwlink/p/?LinkID=2182910`.
2. Ran installer repair:
   `Reports\onedrive-repair\20260430-install-repair-rerun\summary.json`.
   - Installer exit code: `0`.
3. Attempted elevated direct start.
   - Result: process exit `0xC025001C`.
   - Interpretation: elevated launch is not a valid health signal for this
     user-context sync client.
4. Ran reset and start through limited interactive scheduled tasks:
   `Reports\onedrive-repair\20260430-reset-repair\summary.json`.
   - Reset task result: `3011`.
   - Start task result: `0xC025001C`.
   - OneDrive nevertheless remained running after reset/start, so the code is
     recorded as warning evidence rather than treated as proof that repair
     failed.

## Key Evidence

- Pre-repair personal OneDrive diagnostics showed a large local backlog:
  `driveChangesToSend = 13348` and `driveSentChanges = 0`.
- Microsoft's current OneDrive limits guidance recommends avoiding sync sets
  over 300,000 items for performance. Captured diagnostics showed roughly
  148,425 files and 16,131 folders in the personal sync diagnostics, below that
  threshold but large enough for reset processing to be expensive.
- Registry review did not find `NoRemoteChangeNotify`,
  `NoRemoteRecursiveEvents`, or `NoRemoteRecursiveEventsEx` under the checked
  Explorer policy hives.
- The current evidence supports avoiding generic registry notification tweaks.
  Microsoft documents those remote notification keys in a DFS/network-share
  context, not as OneDrive local sync-root tuning.
- The warning that one or more cloud-sync roots are on `F:` remains important
  because `F:` is the mounted cloud-cache VHD. OneDrive's default personal root
  is still `C:\Users\david\OneDrive`, but other sync roots on VHD-backed
  storage can add boot/logon contention.

## Preserved Artifacts

- Repair dry-run:
  `Reports\onedrive-repair\20260430-pre-repair\summary.json`.
- Installer repair:
  `Reports\onedrive-repair\20260430-install-repair-rerun\summary.json`.
- Reset repair:
  `Reports\onedrive-repair\20260430-reset-repair\summary.json`.
- Post-patch dry-run:
  `Reports\onedrive-repair\20260430-postpatch-dryrun\summary.json`.
- Drive risk capture:
  `Reports\drive-performance-sync-risk\20260430-153629\summary.md`.
- Microsoft reference snapshots:
  `docs\references\onedrive\2026-04-30\README.md`.

## Remaining Follow-Up

- Let OneDrive finish rebuilding sync state, then rerun:
  `Tools\Test-SyncProviderHealth.ps1 -SinceMinutes 60 -PassThru`.
- If WER events recur after the reset window, inspect the newest
  `C:\ProgramData\Microsoft\Windows\WER\ReportQueue\*OneDrive*\Report.wer`
  and OneDrive `SyncEngine-*.aodl` references.
- Keep `UnifiUdmDriveStackStartup` disabled until OneDrive has at least one
  clean 60 minute sync-provider health window.
- Avoid reintroducing `LargeSystemCache = 1`, `NtfsMemoryUsage = 2`, or remote
  Explorer notification suppression as workstation defaults unless a new
  metric capture proves a net benefit.
