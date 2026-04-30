# Drive Performance And Cloud Sync Risk Review - 2026-04-30

## Scope

This review covers risky local performance-tweak scripts, current Windows drive
and sync-provider state, and Microsoft-supported drive optimization guidance.
The active symptom is OneDrive sync instability and UI/touchpad glitching that
improves when OneDrive real-time sync is paused.

## Current Evidence

- OneDrive is still the active failure path.
  - Running processes include `OneDrive.exe`, `FileSyncHelper.exe`, and
    `OneDrive.Sync.Service.exe`.
  - `SyncDiagnostics.log` reports `driveChangesToSend = 5203` and
    `driveSentChanges = 0`.
  - Recent WER entries show `OneDrive.exe` crashes in `ucrtbase.dll` with
    `BEX64` / `c0000409`.
  - `OneDrive Per-Machine Standalone Update Task` reports `0x8004EE04`.
- Current cloud-sync footprint is broad.
  - Sync roots are registered for OneDrive, Google Drive, Dropbox, iCloud, and
    Proton Drive.
  - OneDrive and Google Drive are running.
  - Dropbox and Proton roots are on `F:`, which is the mounted
    `cloud-cache-disk` VHD.
- Current filter stack includes cloud and virtualization filters:
  - `CldFlt`, `WdFilter`, `googledrivefs31931`, `bindflt`, `FsDepends`,
    `wcifs`, `Wof`, and `FileInfo`.
- Current global filesystem/cache values:
  - `NtfsMemoryUsage = 2`.
  - `LargeSystemCache = 1`.
  - `DisablePagingExecutive = 1`.
  - `fsutil behavior query disablelastaccess` reports system-managed last
    access behavior with updates enabled.
  - `fsutil behavior query disable8dot3` reports state `3`, disabled on all
    non-system volumes.
- Windows Search is stopped and disabled, so indexing is not the current I/O
  pressure source.
- Defender real-time/on-access protection is enabled and already has broad dev
  exclusions. Further Defender exclusions should be measurement-backed.
- `UnifiUdmDriveStackStartup` is disabled, reducing logon-time SMB/rclone load
  while OneDrive is unstable.
- Current VHD mounts pass in a post-reboot-only window. The duplicate disk ID
  event observed after boot maps to a no-media USB mass-storage slot (`Disk 3`)
  in the current state, not to the mounted VHDs.

## Risky Scripts

### `C:\Users\david\bin\scripts\home-root-archive\optimize-registry-derp.ps1`

This is a broad archived registry tweaker. It is not safe as a general
workstation optimizer.

Risky settings:

- `NtfsMemoryUsage = 2`
  - Microsoft documents this as increasing NTFS paged/nonpaged pool cache
    limits. It may help workloads opening and closing many files in the same
    set, but it can reduce memory available to other processes when the machine
    is already using memory for applications or cache.
- `LargeSystemCache = 1`
  - Server-style/application-cache bias is not current OneDrive guidance and
    the related WMI property is documented as deprecated/obsolete.
- `NtfsDisableLastAccessUpdate = 1`
  - Can reduce metadata churn, but Microsoft notes last-access behavior can
    affect software that relies on the feature. The current machine is already
    using system-managed behavior; do not force this for OneDrive.
- `NtfsDisable8dot3NameCreation = 1`
  - Reasonable only as a compatibility/per-volume decision. The current system
    state `3` is preferable: disabled on non-system volumes.
- `Disk\EnableWriteCache`, `FltMgr\CacheFlushInterval`, `ReFSCacheSize`
  - These are not supported general Windows 11/OneDrive tuning knobs in the
    researched Microsoft guidance. They should be removed from any default
    path unless a vendor-specific reason and rollback test exist.

Implementation flaws:

- If `C:\Users\david\RegistrySettings.json` does not exist, the script creates
  it but does not assign `$registrySettings`, so the first run backs up hives
  and applies nothing.
- It exports full HKLM/HKCU hives into the user profile, which can produce large
  files and possible sync churn if a profile path is cloud-backed.
- It lacks `-DryRun`, `-Apply`, `-Restore`, explicit profiles, and cloud-sync
  preflight checks.

Recommended update:

- Keep this script archived or convert it to report-only by default.
- Require explicit `-Apply -Profile ExperimentalFilesystemCache` for any
  registry writes.
- Remove undocumented `Disk`, `FltMgr`, and `ReFS` tweaks from defaults.
- Add snapshot/restore support for touched keys only, not full hive exports.
- Add a cloud-sync preflight that blocks writes when OneDrive/Google
  Drive/Dropbox/iCloud/Proton are active unless `-ForceCloudSyncRisk` is set.

### `C:\Users\david\unifi_api\submodules\qnap\scripts\QNAP_Performance_Quick_Setup.ps1`

This script is a NAS recovery/large-enumeration profile, not a general Windows
workstation optimizer.

Risky settings:

- SMB client cache expansion:
  - `MaxCmds = 2048`
  - `DirectoryCacheEntriesMax = 4096`
  - `FileInfoCacheEntriesMax = 65536`
  - `FileNotFoundCacheEntriesMax = 8192`
  - `CacheFileTimeout = 5`
  - `CacheDirTimeout = 10`
- `LargeSystemCache = 1`
- `NtfsDisableLastAccessUpdate = 1`
- MTU 9000 on `Ethernet 6`
- SMB server settings, including `AsynchronousCredits`

Current state:

- The extreme SMB client cache values are not currently active.
- `AsynchronousCredits = 0x200` is present under LanmanServer.
- MTU 9000 is active on the QNAP path and is acceptable only if that NIC is a
  dedicated jumbo-frame network path.
- The global `LargeSystemCache = 1` setting is active and should be treated as
  a rollback candidate.

Recommended update:

- Rename/document it as a temporary `QnapRecoveryEnumeration` profile.
- Default to `-DryRun`; require `-Apply`.
- Add `-RestoreFromSnapshot`.
- Add `-TargetPath` and block target paths inside cloud sync roots unless
  `-ForceCloudSyncRisk` is supplied.
- Avoid global filesystem/cache settings by default. Put `LargeSystemCache`,
  last-access changes, and MTU 9000 behind explicit switches.
- Prefer `Set-SmbClientConfiguration` and `Get-SmbClientConfiguration` where a
  supported cmdlet exists, with registry writes only for documented values not
  exposed by cmdlets.

## Microsoft-Supported Guidance

- `fsutil behavior` is supported but advanced. Microsoft specifically notes
  that `memoryusage = 2` can reduce available memory for other processes and
  may reduce overall system performance when the machine is already using large
  amounts of memory for applications or cache.
- Resetting OneDrive is a supported repair action. Microsoft documents that it
  rebuilds OneDrive sync state and does not delete files, but it triggers a
  full sync and may require reselecting synced folders.
- Defender real-time protection should stay on. Microsoft recommends using
  Defender performance tooling for evidence-based exclusions; avoid blanket
  OneDrive/cloud-root exclusions.
- Storage Sense is the supported way to manage local/cloud-backed content
  dehydration. It should not be replaced with unsupported cache or notification
  registry tweaks.
- Windows Search is not required for sync. If re-enabled, use Classic indexing
  and exclude huge cloud/dev roots.
- Do not disable Cloud Files Filter (`CldFlt`), USN/change journals, or cloud
  sync shell handlers as a performance tweak. These are core sync-provider
  infrastructure.
- Supported drive optimization is `Optimize-Volume` / Optimize Drives and TRIM
  retrim. Avoid third-party or ad hoc "SSD registry optimization" packs.

## Proposed Remediation Order

1. Preserve OneDrive evidence before repair:
   - Copy current `SyncDiagnostics.log`, WER report paths, updater task state,
     and current OneDrive task inventory into a dated report directory.
2. Apply a conservative registry rollback plan after exporting touched keys:
   - `NtfsMemoryUsage`: `2 -> 1`.
   - `LargeSystemCache`: `1 -> 0`.
   - `DisablePagingExecutive`: `1 -> 0`.
   - Leave `NoRemoteChangeNotify` and `NoRemoteRecursiveEvents` absent.
   - Leave `NtfsDisable8dot3NameCreation = 3`.
   - Leave last-access behavior system-managed.
3. Reboot and validate:
   - `Tools\Test-BootMountHealth.ps1 -SinceMinutes 60 -PassThru`.
   - `Tools\Test-SyncProviderHealth.ps1 -SinceMinutes 60 -PassThru`.
   - `Tools\Test-ProcessLassoBootSafety.ps1`.
   - Compare `driveChangesToSend`, `driveSentChanges`, WER count, OneDrive CPU,
     and touchpad responsiveness.
4. Repair OneDrive only after the registry rollback reboot sample:
   - Run supported OneDrive reset/rebuild if WER and zero-send backlog persist.
   - Clean stale OneDrive tasks for non-primary SIDs only after confirming they
     are not tied to active local profiles.
5. Reduce cloud-provider overlap:
   - Keep only required cloud providers enabled at logon.
   - Avoid placing cloud roots on mounted VHDs unless there is a specific need
     and the VHD mount order is proven before provider startup.
6. Harden scripts:
   - Convert both risky scripts to report-only/dry-run defaults with explicit
     apply/restore paths and cloud-sync preflight tests.

## Implemented Hardening - 2026-04-30

### `optimize-registry-derp.ps1`

Implemented updates:

- Replaced the broad auto-apply behavior with a report-first contract.
- Default invocation is now dry-run unless `-Apply` is supplied.
- Added `-Profile` with:
  - `ReportOnly`
  - `GeneralWorkstationSafe`
  - `ExperimentalFilesystemCache`
- Added `-DryRun`, `--DryRun`, `-Apply`, `--Apply`, `-RestoreFromSnapshot`,
  `-SnapshotPath`, `-ForceCloudSyncRisk`, `--ForceCloudSyncRisk`,
  `-OutputJson`, `-h`, and `--help`.
- Removed unsupported broad defaults for:
  - `Disk\EnableWriteCache`
  - `FltMgr\CacheFlushInterval`
  - `ReFSCacheSize`
- Replaced full HKLM/HKCU hive exports with touched-key JSON snapshots.
- Added cloud-sync preflight detection for OneDrive, Google Drive, Dropbox,
  iCloud, and Proton Drive.
- Blocks apply when cloud-sync risk exists unless `-ForceCloudSyncRisk` is
  supplied.
- Restore path can reapply prior values or remove keys that did not exist when
  captured.

Current dry-run result:

- No writes performed.
- `GeneralWorkstationSafe` proposes three changes:
  - `NtfsMemoryUsage: 2 -> 1`
  - `LargeSystemCache: 1 -> 0`
  - `DisablePagingExecutive: 1 -> 0`
- Cloud-sync risk is detected, so an apply run would require explicit
  `-ForceCloudSyncRisk`.

### `QNAP_Performance_Quick_Setup.ps1`

Implemented updates:

- Replaced immediate mutation behavior with a dry-run-first audit.
- Default invocation no longer writes registry values, changes MTU, enables RSS,
  prompts for restart, or restarts the machine.
- Added `-DryRun`, `--DryRun`, `-Apply`, `--Apply`, `-RestoreFromSnapshot`,
  `-SnapshotPath`, `-ForceCloudSyncRisk`, `--ForceCloudSyncRisk`,
  `-TargetPath`, `-OutputJson`, `-h`, and `--help`.
- Split high-risk behavior into explicit opt-in switches:
  - `-IncludeGlobalFilesystemTuning`
  - `-IncludeJumboMtu`
  - `-IncludeSmbServerTuning`
- Added touched-setting snapshot and restore support for registry and network
  settings.
- Added cloud-sync root detection and target-path checks. Applying settings is
  blocked when cloud-sync risk exists unless `-ForceCloudSyncRisk` is supplied.
- Removed the interactive restart prompt. Reboot decisions now happen outside
  the script after metrics are captured.

Current dry-run result:

- No writes performed.
- Default QNAP profile proposes SMB client settings only.
- Global filesystem tuning and jumbo MTU are not included unless explicitly
  requested.
- The QNAP host was not reachable during the dry-run, which is recorded as a
  metric rather than treated as a reason to mutate fallback settings.

### `Tools\Collect-DrivePerformanceSyncRisk.ps1`

Added a read-only collector for before/after validation.

Captured evidence:

- Risky registry values.
- Explorer remote notification policy keys.
- Running sync/provider and Process Lasso processes.
- Registered sync roots.
- Relevant scheduled tasks, including OneDrive, VHD, Process Lasso, and UDM.
- Defender settings and Windows Search state.
- Filter drivers from `fltmc filters`.
- Disk and volume inventory.
- OneDrive `SyncDiagnostics.log` metrics.
- Recent OneDrive, WER, Application Error, FilterManager, disk, NTFS, and Task
  Scheduler events.

Current dry-run result:

- No report directory was created.
- Captured warning categories:
  - recent OneDrive/FileSyncHelper event evidence
  - recent FilterManager evidence
  - cloud-sync roots on `F:`, the mounted `cloud-cache-disk` VHD

Baseline report written:

- `Reports\drive-performance-sync-risk\20260430-141404`
- Primary JSON: `drive-performance-sync-risk.json`
- Summary: `summary.md`

Validation:

- `Invoke-Pester -Path .\Tests\Boot\BootValidationTools.Tests.ps1`
- Result: 13 passed, 0 failed.

## Deployed Changes - 2026-04-30 14:32-14:34

### Conservative Registry Rollback

Applied `GeneralWorkstationSafe` using:

```powershell
pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -File `
  C:\Users\david\bin\scripts\home-root-archive\optimize-registry-derp.ps1 `
  -Profile GeneralWorkstationSafe `
  -Apply `
  -ForceCloudSyncRisk `
  -SnapshotPath C:\codedev\PC_AI\Reports\registry-tuning\general-workstation-safe-preapply-20260430-1432.snapshot.json `
  -OutputJson C:\codedev\PC_AI\Reports\registry-tuning\general-workstation-safe-apply-20260430-1432.json
```

Backup and recovery artifacts:

- Registry exports:
  - `Reports\registry-tuning\filesystem-pre-20260430-1432.reg`
  - `Reports\registry-tuning\memory-management-pre-20260430-1432.reg`
- Touched-key snapshot:
  - `Reports\registry-tuning\general-workstation-safe-preapply-20260430-1432.snapshot.json`
- Apply report:
  - `Reports\registry-tuning\general-workstation-safe-apply-20260430-1432.json`

Restore point note:

- `Checkpoint-Computer` did not create a new restore point because Windows
  reported one had already been created within the last 1440 minutes.

Verified live values after apply:

- `NtfsMemoryUsage = 1`
- `LargeSystemCache = 0`
- `DisablePagingExecutive = 0`
- `fsutil behavior query memoryusage` reports `MemoryUsage = 1`

Reboot is still required for a clean post-boot sample and for full confidence
that memory-management settings took effect across the boot lifecycle.

### Process Lasso Watchdog

Added and deployed:

- `Tools\Ensure-ProcessLassoGovernor.ps1`
- `Tools\Register-ProcessLassoGovernorWatchdog.ps1`
- Scheduled task: `PC-AI Process Lasso Governor Watchdog`

Validation:

- Task registered and started successfully.
- Last run result: `0`.
- Report:
  - `Reports\processlasso-governor-watchdog.json`
- Manual verification:
  - `Reports\processlasso-governor-watchdog-manual.json`
- `Tools\Test-ProcessLassoBootSafety.ps1` passed after deployment.

### Post-Deploy Metrics

Metric reports:

- Pre-registry apply:
  - `Reports\drive-performance-sync-risk\20260430-143204`
- Post-registry apply and watchdog deployment:
  - `Reports\drive-performance-sync-risk\20260430-143315`

Health checks:

- `Tools\Test-BootMountHealth.ps1 -SinceMinutes 60 -PassThru`: passed.
- `Tools\Test-ProcessLassoBootSafety.ps1`: passed.
- `Tools\Test-SyncProviderHealth.ps1 -SinceMinutes 60 -PassThru`: still
  failed because OneDrive/FileSyncHelper WER events remain in the current boot
  window.
- `QNAP_Performance_Quick_Setup.ps1 -DryRun`: passed; no QNAP/SMB registry or
  network settings were applied because the target was not reachable and those
  settings remain opt-in.
- `Invoke-Pester -Path .\Tests\Boot\BootValidationTools.Tests.ps1`: 14 passed,
  0 failed.

## Utility Interactions

- Task Scheduler:
  - The collector inventories relevant tasks and their last result codes.
  - `UnifiUdmDriveStackStartup` remains disabled while OneDrive is unstable.
  - Future apply/reboot validation should confirm VHD mount tasks finish before
    sync providers with roots on mounted VHDs are allowed to start.
- Process Lasso:
  - The collector captures `ProcessGovernor` / `ProcessLasso` process state.
  - `Tools\Test-ProcessLassoBootSafety.ps1` remains the stronger validator for
    exclusion/logging policy.
  - Future watchdog work should emit an event-log warning before restarting a
    missing governor.
- Defender:
  - The collector records Defender state and exclusions.
  - No new Defender exclusions were added. Any future cloud/dev exclusions
    should be based on Defender performance evidence, not blanket drive
    exclusions.
- Windows Search:
  - Current state is captured. It remains disabled and is not the current sync
    pressure source.

## Sources

- Microsoft `fsutil behavior`:
  <https://learn.microsoft.com/en-us/windows-server/administration/windows-commands/fsutil-behavior>
- Microsoft OneDrive reset:
  <https://support.microsoft.com/en-gb/office/reset-onedrive-34701e00-bf7b-42db-b960-84905399050c>
- Microsoft Defender performance troubleshooting:
  <https://learn.microsoft.com/en-us/defender-endpoint/troubleshoot-performance-issues>
- Microsoft Win32 `LargeSystemCache` property:
  <https://learn.microsoft.com/en-us/windows/win32/cimwin32prov/win32-operatingsystem>
- Microsoft Storage Sense:
  <https://learn.microsoft.com/en-us/windows/configuration/storage/storage-sense>
- Microsoft Explorer SMB change notification guidance:
  <https://learn.microsoft.com/en-us/troubleshoot/windows-server/shell-experience/increased-cpu-usage>
