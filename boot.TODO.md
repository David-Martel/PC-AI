# Boot, Mount, Sync, and UI Responsiveness TODO

Purpose: harden boot/logon automation that mounts virtual disks, starts sync
providers, and initializes workstation services. The current implementation
focuses on making filesystem/filter failures visible, reducing startup
contention, and preserving UI responsiveness while keeping VHD, WSL, OneDrive,
Google Drive, UDM, and developer workflows functional.

Last updated: 2026-04-30 after post-reboot validation on boot time
2026-04-30 12:50:02 America/New_York, UDM startup disablement, OneDrive
registry/file-notification review, conservative registry rollback, Process
Lasso watchdog deployment, `~\bin` script risk review, and OneDrive
install/reset repair. Reconciled into high-level project docs on 2026-04-30.

## Completed In This Pass

- [x] Apply Process Lasso safety changes before scheduler mutation.
  - Script: `Tools\Apply-ProcessLassoUiSyncTuning.ps1`.
  - Live backup: `C:\ProgramData\ProcessLasso\config\prolasso.ini.bak-20260428-183044-boot-safety`.
  - Validation: `Reports\processlasso-boot-safety-post-register.json`.
  - Result: governor process is running/responding; ProBalance and SmartTrim
    exclusions cover PowerShell, rclone, WSL/Docker, Google Drive, OneDrive,
    shell/input, Lenovo services, and Process Lasso helper processes.

- [x] Add Process Lasso validation tooling.
  - Script: `Tools\Test-ProcessLassoBootSafety.ps1`.
  - Captures governor state, expected exclusions, logging flags, and recent
    Process Lasso log lines.
  - Fails loudly when expected exclusions or logging flags are missing.

- [x] Replace VHD startup one-liners with a maintained wrapper.
  - Wrapper: `Tools\Mount-PersistentVHDX.ps1`.
  - Registrar: `Tools\Register-PersistentVHDXTasks.ps1`.
  - Tests: `Tests\Boot\PersistentVHDX.Tests.ps1`.
  - Live tasks now call `C:\Program Files\PowerShell\7\pwsh.exe` with the
    wrapper script instead of hidden inline `Mount-VHD` one-liners.

- [x] Make VHD mount results loud and structured.
  - Event source: `PC-AI-VHDMount`.
  - Outputs: transcript, structured JSON result, Application event-log entries,
    and nonzero exit codes for degraded or failed states.
  - Detects recent `Microsoft-Windows-FilterManager` Event ID 3 and attaches
    matching evidence to the run result.

- [x] Add VHD post-mount verification.
  - Verifies `Get-VHD`, disk identity, partition/volume state, expected drive
    letter/filesystem/label, Filter Manager visibility, and recent filter
    failures.
  - `share-ext4.vhdx` is intentionally configured as `AttachedDiskOnly`.

- [x] Stagger and re-register VHD startup tasks.
  - `AutoMount_VHDX_cloud-cache-disk`: startup delay `PT30S`.
  - `AutoMount_VHDX_shared-dev`: startup delay `PT60S`.
  - `AutoMount_VHDX_share-ext4`: startup delay `PT90S`.
  - Settings include `MultipleInstances IgnoreNew`, 10 minute execution limit,
    3 retries, and 1 minute retry interval.

- [x] Harden the UDM/rclone startup script.
  - Script: `C:\Users\david\unifi_api\scripts\windows\Start-UDMDriveStack.ps1`.
  - Adds immediate transcript logging, `events.ndjson`, `result.json`,
    Application event-log entries, dependency checks, per-run rclone logging,
    legacy rclone log rotation, and explicit exit codes.

- [x] Re-register the UDM logon task with robust settings.
  - Registrar:
    `C:\Users\david\unifi_api\scripts\windows\Register-UDMDriveStartupTask.ps1`.
  - Task: `UnifiUdmDriveStackStartup`.
  - Settings: delayed logon trigger `PT2M`, working directory, 15 minute
    execution limit, 3 retries, 1 minute retry interval, and
    `MultipleInstances IgnoreNew`.
  - Event source: `UDMDriveStack`.

- [x] Temporarily disable UDM drive-stack auto-launch while OneDrive is
  unstable.
  - Live task: `UnifiUdmDriveStackStartup`.
  - Current state: `Disabled`.
  - Reason: the latest UDM run failed loudly as designed, but its SMB leg is
    not currently healthy and adds avoidable logon-time filesystem/network
    activity while OneDrive is still crash-looping and not sending queued
    changes.
  - Follow-up before re-enabling: either repair SMB credential/network access
    or add an explicit rclone-only startup mode so SMB failure does not produce
    degraded startup noise.

- [x] Add boot/mount/sync validation tooling.
  - Collector: `Tools\Collect-BootDiagnostics.ps1`.
  - Mount health: `Tools\Test-BootMountHealth.ps1`.
  - Sync health: `Tools\Test-SyncProviderHealth.ps1`.
  - Tests: `Tests\Boot\BootValidationTools.Tests.ps1`.
  - Latest report:
    `Reports\boot-diagnostics\20260428-183808\startup-inventory.md`.

- [x] Add CLI help and dry-run contracts for session tooling.
  - All session scripts now accept `-h` and `--help`.
  - Write-capable scripts expose `-DryRun` behavior and tests that verify
    no-write/no-mutation paths.
  - Tests: `Tests\Boot\BootValidationTools.Tests.ps1` and
    `Tests\Boot\PersistentVHDX.Tests.ps1`.

- [x] Add machine-readable evidence artifacts.
  - Startup inventory JSON and markdown are emitted under
    `Reports\boot-diagnostics\<stamp>\`.
  - Raw captures include `Get-VHD`, `Get-Disk`, `Get-Partition`,
    `Get-Volume`, `fltmc filters`, and `fltmc volumes`.
  - Event profile covers FilterManager, Kernel-PnP, disk, volmgr, Ntfs,
    TaskScheduler, WER, Application Error, and optional ReFS/VHDMP/Process Lasso
    providers when present.

- [x] Validate current-window health after changes.
  - `Tools\Test-BootMountHealth.ps1 -SinceMinutes 60` passed.
  - `Tools\Test-SyncProviderHealth.ps1 -SinceMinutes 60` passed.
  - No FilterManager Event ID 3 or OneDrive/FileSyncHelper WER events were
    found in that post-change 60 minute window.

## Remaining Post-Reboot Validation

- [x] Reboot once and run the post-reboot verifier.
  - Command:
    `pwsh -NoLogo -NoProfile -File .\Tools\Collect-BootDiagnostics.ps1 -SinceMinutes 180 -PostRebootVerify -FailOnIssue`
  - Actual command run used `-SinceMinutes 360` to include the full reboot
    window.
  - Report:
    `Reports\boot-diagnostics\20260430-132534\startup-inventory.md`.
  - Result: post-reboot verifier ran and found VHD/mount health passing, but
    sync-provider health failing because OneDrive/FileSyncHelper WER events
    occurred after the 2026-04-30 12:50:02 boot.

- [x] Confirm VHD wrapper logs were produced by startup tasks after reboot.
  - Expected root: `Logs\VHDMount\AutoMount_VHDX_*`.
  - Latest post-reboot results:
    - `AutoMount_VHDX_cloud-cache-disk`:
      `Logs\VHDMount\AutoMount_VHDX_cloud-cache-disk\20260430-125139-a83d01ea.result.json`,
      `Status = Success`, `ExitCode = 0`, `EventId3Count = 0`.
    - `AutoMount_VHDX_shared-dev`:
      `Logs\VHDMount\AutoMount_VHDX_shared-dev\20260430-125140-b935fcd7.result.json`,
      `Status = Success`, `ExitCode = 0`, `EventId3Count = 0`.
    - `AutoMount_VHDX_share-ext4`:
      `Logs\VHDMount\AutoMount_VHDX_share-ext4\20260430-125204-ed0e3fed.result.json`,
      `Status = Success`, `ExitCode = 0`, `EventId3Count = 0`.
  - The earlier 2026-04-30 12:43-12:44 VHD wrapper runs correctly recorded
    degraded state from FilterManager Event ID 3 before the later 12:50 reboot.
    The current boot's VHD runs are clean.

- [x] Confirm UDM startup either succeeds or fails loudly after next logon.
  - Expected root:
    `C:\Users\david\unifi_api\logs\udm-drive-stack\<timestamp>\`.
  - Latest run:
    `C:\Users\david\unifi_api\logs\udm-drive-stack\20260430-125254\result.json`.
  - Result: the process failed loudly as designed with `status = degraded`,
    `exit_code = 50`, `exit_code_name = PartialHealth`.
  - Evidence: `rclone_health = ok`; `SMB_health = failed` with
    "The network path was not found"; SMB credential was also degraded because
    no SMB credential was supplied or found in Windows Credential Manager.
  - Follow-up action taken 2026-04-30: the scheduled task was disabled until
    OneDrive is fixed.

- [x] Re-check OneDrive task inventory after reboot/logon.
  - The collector still flags several OneDrive tasks with `267011`, meaning
    "not yet run" style scheduler state, and older OneDrive/FileSyncHelper WER
    evidence from the current boot.
  - Do not reset OneDrive by default. Reset only after explicit approval because
    it forces broad resync.
  - Post-reboot evidence is worse than stale task noise alone: the report found
    25 OneDrive/FileSyncHelper WER records after boot, including crashes for
    both OneDrive `26.062.0402.0002` and older `26.55.323.4` signatures in
    `ucrtbase.dll` with `BEX64` / `c0000409`.
  - The current OneDrive process set is running and responsive, but crash
    evidence remains the main failed post-reboot check.

- [x] Check Explorer remote file-notification policy keys.
  - Checked:
    `HKCU\Software\Microsoft\Windows\CurrentVersion\Policies\Explorer` and
    `HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\Policies\Explorer`.
  - Result: `NoRemoteChangeNotify`, `NoRemoteRecursiveEvents`, and
    `NoRemoteRecursiveEventsEx` are not present.
  - Interpretation: there is no evidence that Explorer remote change-notify
    suppression is currently breaking OneDrive. Microsoft documents these
    values as Explorer/SMB mapped-share change notification controls, not as
    local OneDrive sync-root tuning knobs.

- [x] Identify local registry-performance scripts that could reapply risky
  filesystem/cache settings.
  - `C:\Users\david\bin\scripts\home-root-archive\optimize-registry-derp.ps1`
    sets `NtfsDisableLastAccessUpdate`, `NtfsMemoryUsage = 2`,
    `NtfsDisable8dot3NameCreation`, and `LargeSystemCache = 1`.
  - `C:\Users\david\unifi_api\submodules\qnap\scripts\QNAP_Performance_Quick_Setup.ps1`
    sets `NtfsDisableLastAccessUpdate` and `LargeSystemCache = 1`.
  - These scripts should not be rerun as generic workstation optimizers until
    their scope is narrowed and their help text warns about OneDrive/sync
    provider risk.

- [x] Harden risky performance-tuning scripts so they default to report-only
  behavior.
  - `C:\Users\david\bin\scripts\home-root-archive\optimize-registry-derp.ps1`
    now requires `-Apply` before it writes registry values.
  - `C:\Users\david\unifi_api\submodules\qnap\scripts\QNAP_Performance_Quick_Setup.ps1`
    now requires `-Apply` before it writes registry/network settings.
  - Both scripts expose `-DryRun`, `--DryRun`, `-h`, `--help`, snapshot/restore
    support, output JSON support, and cloud-sync risk checks.
  - High-risk QNAP behavior is opt-in through `-IncludeGlobalFilesystemTuning`,
    `-IncludeJumboMtu`, and `-IncludeSmbServerTuning`.
  - Existing undocumented defaults for `Disk\EnableWriteCache`,
    `FltMgr\CacheFlushInterval`, and `ReFSCacheSize` were removed from the
    active default path.

- [x] Add metric capture for drive-performance/cloud-sync validation.
  - Collector: `Tools\Collect-DrivePerformanceSyncRisk.ps1`.
  - Captures registry state, sync roots/processes, relevant scheduled tasks,
    Process Lasso process state, Defender, Windows Search, filter drivers,
    disks, volumes, OneDrive diagnostics, and recent events.
  - Baseline report:
    `Reports\drive-performance-sync-risk\20260430-141404`.

- [x] Apply conservative workstation registry rollback with recovery artifacts.
  - Script:
    `C:\Users\david\bin\scripts\home-root-archive\optimize-registry-derp.ps1`.
  - Profile: `GeneralWorkstationSafe`.
  - Applied values:
    - `NtfsMemoryUsage = 1`.
    - `LargeSystemCache = 0`.
    - `DisablePagingExecutive = 0`.
  - Registry exports:
    - `Reports\registry-tuning\filesystem-pre-20260430-1432.reg`.
    - `Reports\registry-tuning\memory-management-pre-20260430-1432.reg`.
  - Restore snapshot:
    `Reports\registry-tuning\general-workstation-safe-preapply-20260430-1432.snapshot.json`.
  - Apply report:
    `Reports\registry-tuning\general-workstation-safe-apply-20260430-1432.json`.
  - Restore point note: Windows skipped a new restore point because one already
    existed within the last 1440 minutes.

- [x] Deploy Process Lasso governor watchdog.
  - Watchdog:
    `Tools\Ensure-ProcessLassoGovernor.ps1`.
  - Registrar:
    `Tools\Register-ProcessLassoGovernorWatchdog.ps1`.
  - Scheduled task: `PC-AI Process Lasso Governor Watchdog`.
  - Task last run result: `0`.
  - Reports:
    - `Reports\processlasso-governor-watchdog.json`.
    - `Reports\processlasso-governor-watchdog-manual.json`.

- [x] Apply Process Lasso touchpad/UI responsiveness tuning.
  - Script updated and applied:
    `Tools\Apply-ProcessLassoUiSyncTuning.ps1`.
  - Review:
    `Reports\touchpad-process-lasso-optimization-20260430.md`.
  - Backup:
    `C:\ProgramData\ProcessLasso\config\prolasso.ini.bak-20260430-145824-boot-safety`.
  - Reports:
    - `Reports\processlasso-touchpad-responsiveness-dryrun.json`.
    - `Reports\processlasso-touchpad-responsiveness-apply.json`.
    - `Reports\processlasso-touchpad-responsiveness-postapply-dryrun.json`.
    - `Reports\processlasso-touchpad-boot-safety-validation.json`.
  - Result: input/shell/vendor processes are protected from ProBalance and
    SmartTrim and default to Above Normal CPU / High I/O priority hints;
    OneDrive/FileSync/cloud sync, Docker/WSL/Redis, build/archive, and heavy
    dev-helper processes default to Below Normal CPU / Low I/O priority.
  - Live verification: `ProcessGovernor.exe` running/responding, Process Lasso
    log lines present, and WMIC base-priority snapshot showed input/shell
    processes at base priority `10` while sync/background competitors were at
    base priority `6`.

- [x] Apply Process Lasso dynamic hardware/display utility tuning.
  - Script updated and applied:
    `Tools\Apply-ProcessLassoUiSyncTuning.ps1`.
  - Review:
    `Reports\system-performance-optimization-20260430.md`.
  - Backup:
    `C:\ProgramData\ProcessLasso\config\prolasso.ini.bak-20260430-151250-boot-safety`.
  - Reports:
    - `Reports\processlasso-dynamic-hardware-tuning-dryrun.json`.
    - `Reports\processlasso-dynamic-hardware-tuning-apply.json`.
    - `Reports\processlasso-dynamic-hardware-tuning-postapply-dryrun.json`.
    - `Reports\processlasso-system-optimization-validation.json`.
  - Result: NVIDIA Broadcast/Overlay/App/Share, PresentMon/FrameView,
    short-lived NVIDIA `disp.exe`, Dell TechHub/DPM helpers, and HP Print Scan
    Doctor helpers now default to Below Normal CPU / Low I/O priority. Existing
    browser/Zoom GPU priority boosts were normalized from Above Normal to
    Normal so they do not compete with display composition during dock/eGPU
    churn.
  - Live verification: `ProcessGovernor.exe` running/responding, post-apply
    dry-run found no pending changes, Process Lasso log showed NVIDIA Overlay
    receiving priority/I/O adjustments, and
    `Tests\Boot\BootValidationTools.Tests.ps1` passed with 14 passed, 0 failed.

- [x] Add OneDrive repair and evidence tooling.
  - Script: `Tools\Repair-OneDriveSync.ps1`.
  - Supports `-h`, `--help`, `-DryRun`, `--DryRun`, installer download/install,
    opt-in reset, post-action evidence capture, WER queue capture, and warning
    surfacing for non-zero scheduled-task results.
  - Microsoft reference snapshots:
    `docs\references\onedrive\2026-04-30\README.md`.
  - Repair report:
    `Reports\onedrive-repair-20260430.md`.

- [x] Reconcile boot/sync/session tooling into high-level project docs.
  - Updated: `README.md`, `TODO.md`, `AGENTS.md`, `CLAUDE.md`, `GEMINI.md`,
    `Config\PROJECT_CONTEXT.md`, `llm.TODO.md`, and `CLAUDE.TODO.md`.
  - Result: high-level docs now point to this file as the live operational
    ledger and no longer leave completed boot/session contract work as active
    top-level TODO noise.

- [x] Centralize Task Scheduler and workstation system-modification scripts.
  - Migration tool: `Tools\Migrate-SystemScriptsIntoRepo.ps1`.
  - New script home: `Tools\SystemScripts`.
  - Repointed Task Scheduler actions:
    `\BW-Auto-Unlock`, `\Bitwarden\Initialize-MachineSecrets`,
    `\DevEnvironmentStartup`, `\Gemini-CLI-Update-stable`, `\LspmuxServer`,
    `\PowerShell\ProfileLogSync`, `\UDP Socket Monitor`, and
    `\UnifiUdmDriveStackStartup`.
  - Moved relevant scripts from `C:\Scripts`, `~\.machine`, `~\.local\bin`,
    `~\bin`, OneDrive PowerShell script folders, and UDM startup helper
    folders into repo-owned locations.
  - Cleaned up empty source folders after migration. The UDM Windows script
    source folder was held open by an old `rclone.exe` process using that path
    as its working directory; that stale UDM rclone mount process was stopped
    after confirming `UnifiUdmDriveStackStartup` was disabled, and the empty
    source folder was removed.
  - Validation:
    `Reports\task-scheduler-after-systemscript-migration.xml`,
    `Reports\system-script-migration-20260430-164416.json`, and
    `Reports\system-script-migration-20260430-165552.json`.
  - Notes: log files, cache files, secret material, and ignored backup files
    were not force-added to git. Secret modules and caches under `~\.machine`
    intentionally remain outside this repo.

## Remaining Active Issues After 2026-04-30 Reboot

- [ ] Monitor OneDrive after install/reset repair.
  - Evidence before repair: `Reports\boot-diagnostics\20260430-132534\boot-events.json`
    contains 25 OneDrive/FileSyncHelper WER records after boot.
  - Repair actions completed:
    - Installer repair:
      `Reports\onedrive-repair\20260430-install-repair-rerun\summary.json`.
    - Reset/start repair:
      `Reports\onedrive-repair\20260430-reset-repair\summary.json`.
    - Post-patch dry-run validation:
      `Reports\onedrive-repair\20260430-postpatch-dryrun\summary.json`.
  - Current post-reset process state: `OneDrive.exe`, `FileSyncHelper.exe`,
    and `FileCoAuth.exe` are running/responding.
  - Short post-reset health window passed:
    `Reports\sync-provider-health-post-onedrive-reset-5min.json`.
  - The 240 minute health window still fails because it includes reset-time WER
    events and earlier boot-window failures:
    `Reports\sync-provider-health-post-onedrive-reset.json`.
  - Remaining success criteria: at least one clean 60 minute
    `Tools\Test-SyncProviderHealth.ps1 -SinceMinutes 60 -PassThru` run,
    decreasing `driveChangesToSend`, increasing `driveSentChanges`, and no new
    OneDrive/FileSyncHelper WER after the reset window.
  - If WER recurs, inspect the newest
    `C:\ProgramData\Microsoft\Windows\WER\ReportQueue\*OneDrive*\Report.wer`
    and current OneDrive `SyncEngine-*.aodl` references before changing
    registry or startup policy again.

- [x] Decide and apply a OneDrive-safe registry rollback.
  - Current values:
    - `HKLM\SYSTEM\CurrentControlSet\Control\FileSystem\NtfsMemoryUsage = 2`.
    - `HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Memory Management\LargeSystemCache = 1`.
    - `HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Memory Management\DisablePagingExecutive = 1`.
  - Proposed first rollback candidate: set `NtfsMemoryUsage` to the default
    workstation behavior (`1`) because Microsoft documents that raising NTFS
    memory usage can reduce memory available to other processes and may reduce
    overall performance.
  - Proposed second rollback candidate: set `LargeSystemCache` to `0`
    application/workstation bias. Microsoft WMI documentation now marks the
    related `Win32_OperatingSystem.LargeSystemCache` property as deprecated,
    but leaving a server-cache-era tweak enabled on a sync-heavy workstation is
    not useful evidence-based tuning.
  - Proposed third rollback candidate: set `DisablePagingExecutive` to `0`
    unless a current benchmark proves it helps this workstation. It is a broad
    kernel memory residency tweak and should not be bundled with OneDrive or
    touchpad debugging.
  - Before applying: export affected registry keys, create a restore point if
    available, record current `fsutil behavior query memoryusage`, capture
    OneDrive sync diagnostics, then reboot and compare OneDrive WER count,
    `driveChangesToSend/driveSentChanges`, touchpad responsiveness, and boot
    mount health.
  - Consolidated review:
    `Reports\drive-performance-sync-risk-20260430.md`.
  - Agent/script audit conclusion: the Explorer remote notification keys are
    not the problem. The stronger rollback candidates are global
    workstation/server-cache settings that are active now:
    `NtfsMemoryUsage = 2`, `LargeSystemCache = 1`, and
    `DisablePagingExecutive = 1`.
  - Applied 2026-04-30:
    - `NtfsMemoryUsage: 2 -> 1`.
    - `LargeSystemCache: 1 -> 0`.
    - `DisablePagingExecutive: 1 -> 0`.
  - Verification before reboot:
    `fsutil behavior query memoryusage` reports `MemoryUsage = 1`.

- [ ] Reboot and validate registry rollback effect.
  - Reboot is required for a clean post-boot sample and for full confidence
    that memory-management settings took effect across the boot lifecycle.
  - Partial no-reboot validation on 2026-04-30 15:07 confirms the persisted
    registry values are now `NtfsMemoryUsage = 1`, `LargeSystemCache = 0`, and
    `DisablePagingExecutive = 0`; Explorer remote notification suppression
    keys remain absent.
  - Run after reboot:
    - `Tools\Collect-DrivePerformanceSyncRisk.ps1 -SinceMinutes 120`.
    - `Tools\Test-BootMountHealth.ps1 -SinceMinutes 120 -PassThru`.
    - `Tools\Test-SyncProviderHealth.ps1 -SinceMinutes 120 -PassThru`.
    - `Tools\Test-ProcessLassoBootSafety.ps1`.
  - Compare against:
    - `Reports\drive-performance-sync-risk\20260430-143204` pre-apply.
    - `Reports\drive-performance-sync-risk\20260430-143315` post-apply.
  - Success criteria: no new FilterManager Event ID 3, Process Governor
    present, VHD tasks return `0`, OneDrive WER rate decreases, and
    `driveChangesToSend` starts decreasing or `driveSentChanges` advances.

- [x] Add guardrails to risky local registry scripts.
  - Update `optimize-registry-derp.ps1` so it is archived or dry-run only by
    default, documents the OneDrive/sync-provider risk, and does not apply
    global filesystem/cache tweaks without explicit named switches.
  - Update the QNAP performance script so server/NAS-facing settings remain
    scoped to that use case instead of becoming generic Windows workstation
    startup advice.
  - Implemented changes:
    - Add `-DryRun`, `-Apply`, and `-RestoreFromSnapshot`.
    - Add cloud-sync preflight checks for OneDrive, Google Drive, Dropbox,
      iCloud, and Proton Drive.
    - Block target paths inside sync roots unless `-ForceCloudSyncRisk` is
      supplied.
    - Remove undocumented `Disk\EnableWriteCache`, `FltMgr\CacheFlushInterval`,
      and `ReFSCacheSize` from default paths.
    - Replace full HKLM/HKCU exports with touched-key snapshots.
    - Add Pester tests proving dry-run/no-write behavior and CLI contract
      coverage.
  - Validation:
    `Invoke-Pester -Path .\Tests\Boot\BootValidationTools.Tests.ps1` passed
    with 13 passed, 0 failed.

- [ ] Review cloud roots located on mounted VHDs.
  - Current evidence:
    - OneDrive root: `C:\Users\david\OneDrive`.
    - Dropbox root: `F:\Auricle Dropbox`.
    - Proton Drive root: `F:\Proton-Drive\My files`.
    - `F:` is the mounted `cloud-cache-disk` VHD.
  - Risk: sync providers starting before or during VHD mount/filter readiness
    can amplify FilterManager, shell overlay, and cloud file placeholder churn.
  - Proposed fix: keep VHD mount tasks early and verified; delay Dropbox/Proton
    startup until after `AutoMount_VHDX_cloud-cache-disk` passes, or move those
    sync roots off the VHD if reliability is more important than cache
    isolation.

- [ ] Explain and monitor duplicate disk identifier startup events.
  - Current read-only mapping after reboot shows `Disk 3` is a no-media
    `Generic MassStorageClass` USB slot, while VHDs are disks 4-6 and healthy.
  - System log still contains disk Event ID 158 at 2026-04-30 12:50:24:
    "Disk 3 has the same disk identifiers as one or more disks connected to the
    system."
  - Proposed fix: do not change VHD signatures based on this event alone.
    Continue collecting it as startup noise and only remediate if the event
    maps to an online mounted volume in a future capture.

- [ ] Decide the future UDM startup mode after OneDrive is stable.
  - Evidence:
    `C:\Users\david\unifi_api\logs\udm-drive-stack\20260430-125254\result.json`.
  - Current live mitigation: `UnifiUdmDriveStackStartup` is disabled.
  - If SMB is still required, validate UDM SMB service state, Windows SMB
    client access, credential storage, and `F:\udm_smb` link/target behavior.
  - If SMB is not required, update the registrar/start script with an explicit
    rclone-only mode before re-enabling the task.

- [ ] Correlate any remaining touchpad glitching against OneDrive crash and
  sync activity.
  - Fresh UI diagnostics were collected at
    `Reports\ui-glitch-diagnostics\20260430-132759`.
  - The latest dynamic-hardware review found repeated `NVIDIA Broadcast.exe`,
    `PresentMonService.exe`, Dell TechHub, camera-startup, WUDFHost, and
    Audiosrv WER entries in the same broad performance window. These are now
    Process Lasso-demoted where safe, but they should still be correlated with
    any future touchpad glitch capture.
  - Next capture should be taken immediately after a touchpad glitch and should
    include OneDrive process I/O, Process Lasso log lines, HID/I2C/Kernel-PnP
    events, top disk I/O, and current sync-provider state.

- [ ] Finish hardening newly migrated `Tools\SystemScripts` utilities before
  enabling any additional startup/logon usage.
  - Preserve current Task Scheduler repoints, but do not enable new migrated
    scripts until each write-capable script has help text, `-DryRun`,
    idempotent behavior, structured logs or clear console output, and loud
    nonzero failures.
  - Prioritize `UserBin` DNS, RAG Redis, sccache, Docker/MCP, and developer
    environment scripts because those can change services, PATH, network
    behavior, caches, or startup state.

- [ ] Decide whether NVIDIA Broadcast and FrameView/PresentMon should run at
  logon.
  - Evidence:
    `Reports\boot-diagnostics\20260430-150703\boot-events.json` includes
    repeated `NVIDIA Broadcast.exe` `BEX64` failures, a
    `PresentMonService.exe` crash in NVIDIA `nvml.dll`, repeated
    `CameraStartupFailureEvent` records, and `NVDisplay.Container.exe` crash
    evidence.
  - Current mitigation: Process Lasso demotes user-mode NVIDIA overlay,
    Broadcast, FrameView/PresentMon, and short-lived NVIDIA display helper
    processes.
  - Proposed next step: if NVIDIA Broadcast camera effects or NVIDIA
    FrameView/metrics are not actively needed, disable their startup/service
    surfaces and validate that camera/WUDFHost/Audiosrv event noise decreases.
    If they are needed, update NVIDIA App/Broadcast/display driver as a set and
    retest.

- [ ] Review optional USB filter drivers only if OneDrive-paused touchpad
  glitches continue.
  - Current running USB-related filters/services include USB4 host/device
    routers, USBPcap, VirtualBox USB monitor, Parsec virtual USB, Remote
    Desktop USB hub filters, USB mass storage, USB Ethernet, and multiple
    audio/HID roots.
  - Do not disable these blindly because several support development or remote
    workflows.
  - Proposed A/B validation: with OneDrive paused, capture a glitch window with
    the current configuration, then run one controlled boot with USBPcap
    disabled/stopped if not needed and compare WUDFHost, Kernel-PnP, USB4,
    HID/I2C, and touchpad behavior.

- [x] Add a Process Lasso governor watchdog or startup verification task.
  - Fresh validation on 2026-04-30 13:41 found `ProcessGovernor.exe` not
    running even though the Process Lasso startup tasks ran at logon.
  - Manual remediation:
    `Start-Process 'C:\Program Files\Process Lasso\ProcessGovernor.exe'`.
  - Follow-up validation at 2026-04-30 13:44 passed
    `Tools\Test-ProcessLassoBootSafety.ps1`.
  - Next implementation should either add a delayed logon/startup verification
    task that restarts `ProcessGovernor.exe` when absent or update the existing
    Process Lasso task configuration if Bitsum exposes a supported repair
    path. The failure should write a Windows event-log warning before restart
    so governor disappearances remain visible.
  - Implemented:
    `PC-AI Process Lasso Governor Watchdog` delayed logon task.
  - Validation: task last result `0`; `Tools\Test-ProcessLassoBootSafety.ps1`
    passed after deployment.

- [ ] Decode and triage OneDrive task results.
  - `OneDrive Per-Machine Standalone Update Task` now reports `-2147160572`
    / `0x8004EE04` from 2026-04-29 14:04:49.
  - Several OneDrive Reporting/Startup tasks for older SIDs still report
    `267011`.
  - Current-user tasks for SID ending `1001` are valid and have recent run
    evidence; stale/non-primary principals include raw SID `1002`,
    `WsiAccount`, `DevToolsUser`, and `CodexSandboxOffline`.
  - Determine which tasks are valid for the current user, which belong to stale
    SIDs, and whether any stale tasks are invoking obsolete OneDrive builds.

- [ ] Harden `~\bin` scripts that can amplify boot, sync, or UI latency.
  - Review:
    `Reports\bin-script-risk-review-20260430.md`.
  - Current evidence: no obvious active scheduled task was found that directly
    invokes `C:\Users\david\bin`, `Start-RAGRedis`, `RAG-Redis`, `dnsproxy`,
    Acrylic DNS, or sccache. Startup folder inventory did not show these
    scripts directly either.
  - Highest-risk script groups:
    - RAG Redis startup/health:
      `C:\Users\david\bin\Setup-RAGRedisAutoStart.ps1`,
      `C:\Users\david\bin\Start-RAGRedisNative.ps1`, and
      `C:\Users\david\bin\Test-RAGRedisHealth.ps1`.
    - DNS proxy/Acrylic DNS:
      `C:\Users\david\bin\LocalDNSProxy.ps1`,
      `C:\Users\david\bin\Install-AcrylicDNS.ps1`, and
      `C:\Users\david\bin\dns-proxy.bat`.
    - OneDrive/GCP profile scripts under
      `C:\Users\david\bin\scripts\home-root-archive`.
    - Mapped-drive/WinRM/Google Drive copy scripts under
      `C:\Users\david\bin\scripts\home-root-archive`.
    - Heavy archive/build/toolchain scripts such as `universal-archiver.ps1`,
      `create-backup.ps1`, `sccache-manager.ps1`,
      `Setup-BuildEnvironment.ps1`, `Manage-DevTools.ps1`,
      `Update-DevUtilities.ps1`, `Fix-Winget.ps1`,
      `Fix-NPM-Issues.ps1`, and `Install-CoreUtils-Direct.ps1`.
  - Required contract before any future boot/logon use:
    `-h`, `--help`, `-DryRun`, `--DryRun`, explicit `-Apply` for mutation,
    touched-key/path snapshots, structured JSON output, transcript/event-log
    evidence, nonzero failure/degraded exit codes, and sync-root preflight
    warnings.
  - Process Lasso policy should keep heavy Docker/Redis/RAG, 7-Zip, robocopy,
    cargo/rustc/linker, sccache, winget, and npm work background-friendly while
    preserving OneDrive, Explorer, shell/input, Lenovo input services, and
    Process Governor safety exclusions.
  - Do not enable RAG Redis, DNS proxy, archive, installer, or build-cache
    automation at logon until OneDrive WER and zero-send backlog are resolved.

- [ ] Convert RAG Redis startup tooling to loud, delayed, recoverable startup
  automation before considering re-enablement.
  - `Setup-RAGRedisAutoStart.ps1` should become report-only by default and
    require `-Apply` to create/remove tasks or write `Start-RAGRedis.ps1`.
  - `Start-RAGRedisNative.ps1` should add `-DryRun`, `--DryRun`, `-h`, `--help`,
    `-OutputJson`, `-NoDocker`, `-NoWSL`, `-NoProcessKill`, and event-log
    integration.
  - `Test-RAGRedisHealth.ps1` should keep ordinary health checks read-only;
    `-Fix` should write a structured action log and clearly state every service,
    process, copy, or Docker action it will take.
  - Any eventual scheduled task should be delayed until VHD mount health and
    Process Lasso governor health pass.

- [ ] Convert DNS proxy/Acrylic DNS utilities to snapshot-and-restore workflows.
  - Add adapter DNS snapshots, named-adapter targeting, explicit `-Apply`, and
    `-RestoreFromSnapshot`.
  - Avoid all-active-adapter DNS rewrites as a default.
  - Prefer local development resolver scope that does not affect global
    OneDrive/cloud-provider name resolution.

- [ ] Quarantine or modernize archived OneDrive/GCP/network-drive scripts before
  future use.
  - The archived scripts can remove OneDrive-backed PowerShell modules, copy
    files into Google Drive, mutate WSL files, and set mapped-drive
    `ProviderFlags` without a dry-run or restore plan.
  - Leave them archived unless they are converted to the same report-first,
    recoverable contract used for the registry tuning scripts.
  - If any of them are needed, run only after a fresh OneDrive evidence capture
    and outside the boot/logon contention window.

## Optional Follow-Up Optimizations

- [ ] Decide whether to delay nonessential sync providers.
  - Current tooling inventories OneDrive, Google Drive, Dropbox, iCloud, Proton,
    sync roots, shell overlays, provider processes, and related scheduled tasks.
  - Prefer app-supported startup settings over registry deletion.
  - Keep OneDrive early unless evidence shows the updated VHD/UDM sequencing
    still causes touchpad/UI stalls.

- [ ] Revisit workstation registry tuning only after OneDrive is no longer
  crash-looping.
  - Candidates promoted to active review:
    `NtfsMemoryUsage`, `LargeSystemCache`, and `DisablePagingExecutive`.
  - Do not add `NoRemoteChangeNotify` or `NoRemoteRecursiveEvents`; they are
    absent now, and adding them would reduce Explorer refresh behavior for
    mapped network shares rather than fix the local OneDrive sync root.
  - Capture before/after boot mount health, OneDrive responsiveness, touchpad
    responsiveness, and dev workload smoke tests.

- [ ] Add a reusable `~\bin` script risk collector.
  - Suggested script: `Tools\Collect-BinScriptRisk.ps1`.
  - Capture mutating command patterns, startup/profile/task references, help
    and dry-run support, Process Lasso-relevant process names, sync-root target
    paths, and likely boot/logon impact.
  - Emit JSON plus markdown so future script changes can be diffed before they
    are allowed into startup or profile code.

- [ ] Tune diagnostic severity rules after one clean reboot sample.
  - Keep true mount/filter/sync failures loud.
  - Downgrade expected long-running Process Lasso tasks and expected "not yet
    run" OneDrive updater/reporting tasks if they continue to create noise
    without indicating a real boot risk.

## Validation Commands Run

- [x] `Invoke-Pester -Path .\Tests\Boot\PersistentVHDX.Tests.ps1`
  - Result: 12 passed, 0 failed.
- [x] `Invoke-Pester -Path .\Tests\Boot\BootValidationTools.Tests.ps1`
  - Result: 5 passed, 0 failed.
- [x] `Tools\Test-ProcessLassoBootSafety.ps1`
  - Result: passed after live Process Lasso update and task registration.
- [x] `Tools\Test-BootMountHealth.ps1 -SinceMinutes 60`
  - Result: passed after VHD task registration.
- [x] `Tools\Test-SyncProviderHealth.ps1 -SinceMinutes 60`
  - Result: passed after UDM and VHD task registration.
- [x] `Tools\Collect-BootDiagnostics.ps1 -SinceMinutes 60 -PostRebootVerify`
  - Latest result: `Reports\boot-diagnostics\20260428-183808`.
  - Current boot still contains old FilterManager and OneDrive/FileSync WER
    evidence, so only a reboot can prove the startup hardening fully cleared
    those boot-time failures.
- [x] `Disable-ScheduledTask -TaskName UnifiUdmDriveStackStartup -TaskPath '\'`
  - Result: live task state is `Disabled`.
- [x] `Tools\Test-BootMountHealth.ps1 -SinceMinutes 240 -PassThru`
  - Result: `Fail` only because the 240 minute window still includes the
    pre-reboot FilterManager Event ID 3 at 2026-04-30 12:42:53. Current VHD
    tasks returned `0` after the 12:50 reboot and all expected VHDs are
    attached/mounted.
- [x] `Tools\Test-SyncProviderHealth.ps1 -SinceMinutes 240 -PassThru`
  - Result: `Fail` because OneDrive/FileSyncHelper WER events occurred after
    boot; also warns that the iCloud sync root exists while its provider
    process is not running.
- [x] `Tools\Test-BootMountHealth.ps1 -SinceMinutes 50 -PassThru`
  - Result: `Pass`. This post-reboot-only window excludes the 12:42
    FilterManager event and confirms the VHD startup path is currently clean.
- [x] `Tools\Test-SyncProviderHealth.ps1 -SinceMinutes 50 -PassThru`
  - Result: `Fail`. The post-reboot-only window still contains OneDrive WER
    events after 12:54, so OneDrive remains the active boot/sync failure.
- [x] `Tools\Test-ProcessLassoBootSafety.ps1`
  - First run at 2026-04-30 13:41 failed because `ProcessGovernor.exe` was not
    running.
  - After manually starting `C:\Program Files\Process Lasso\ProcessGovernor.exe`,
    rerun at 2026-04-30 13:44 passed with governor PID `58624`.
- [x] Registry notification/performance review commands.
  - `reg.exe query HKCU\Software\Microsoft\Windows\CurrentVersion\Policies\Explorer /s`
  - `reg.exe query HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\Policies\Explorer /s`
  - `reg.exe query HKLM\SYSTEM\CurrentControlSet\Control\FileSystem /v NtfsMemoryUsage`
  - `reg.exe query "HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Memory Management" /v LargeSystemCache`
  - `rg -n "NoRemoteChangeNotify|NoRemoteRecursiveEvents|NtfsMemoryUsage|LargeSystemCache|DisablePagingExecutive|NtfsDisableLastAccessUpdate|NtfsDisable8dot3" C:\Users\david --glob "*.ps1" --hidden`
- [x] `Tools\Collect-DrivePerformanceSyncRisk.ps1 -SinceMinutes 240`
  - Result:
    `Reports\drive-performance-sync-risk\20260430-141404`.
- [x] `Tools\Collect-DrivePerformanceSyncRisk.ps1 -SinceMinutes 240`
  - Pre-registry apply:
    `Reports\drive-performance-sync-risk\20260430-143204`.
  - Post-registry apply and watchdog deployment:
    `Reports\drive-performance-sync-risk\20260430-143315`.
- [x] `Invoke-Pester -Path .\Tests\Boot\BootValidationTools.Tests.ps1`
  - Result after risky-script hardening: 13 passed, 0 failed.
  - Result after Process Lasso watchdog deployment: 14 passed, 0 failed.
- [x] `Tools\Test-BootMountHealth.ps1 -SinceMinutes 60 -PassThru`
  - Result after registry rollback and watchdog deployment: passed.
- [x] `Tools\Test-ProcessLassoBootSafety.ps1`
  - Result after watchdog deployment: passed.
- [x] `Tools\Test-SyncProviderHealth.ps1 -SinceMinutes 60 -PassThru`
  - Result after registry rollback and watchdog deployment: still failed due
    to OneDrive/FileSyncHelper WER events in the current boot window.
- [x] `rg` review of `C:\Users\david\bin` and
  `C:\Users\david\bin\scripts\home-root-archive`
  - Result: identified risky RAG Redis, DNS, OneDrive/GCP profile,
    mapped-drive, Google Drive copy, archive/build, toolchain, and package
    repair scripts.
  - Report:
    `Reports\bin-script-risk-review-20260430.md`.
- [x] Startup-folder inventory for current user and all-users Startup folders.
  - Current-user Startup contains Google Cloud, Ollama, Proton Mail Bridge, and
    SSH-Agent links.
  - All-users Startup contains Cloudflare WARP, DeviceCenter, EPOS, miniDSP,
    SOLIDWORKS, Tailscale, and Topping USB audio links.
  - No `~\bin` RAG Redis/DNS/archive/build scripts were visible directly in
    the Startup folders.
- [x] `Tools\Repair-OneDriveSync.ps1 -OutputDirectory Reports\onedrive-repair\20260430-pre-repair -DownloadInstaller -InstallLatest -StopBeforeInstall -StartAfterRepair -DryRun`
  - Result: dry-run completed without download/install/start mutations.
- [x] `Tools\Repair-OneDriveSync.ps1 -OutputDirectory Reports\onedrive-repair\20260430-install-repair-rerun -InstallerPath Reports\onedrive-repair\20260430-install-repair\OneDriveSetup.exe -InstallLatest -StartAfterRepair -SinceMinutes 240`
  - Result: installer repair completed with installer exit code `0`.
- [x] `Tools\Repair-OneDriveSync.ps1 -OutputDirectory Reports\onedrive-repair\20260430-reset-repair -ResetOneDrive -StartAfterRepair -SinceMinutes 240`
  - Result: reset/start completed; non-zero task results were captured for
    follow-up, while OneDrive processes remained running afterward.
- [x] `Tools\Test-SyncProviderHealth.ps1 -SinceMinutes 5 -PassThru -OutputJson Reports\sync-provider-health-post-onedrive-reset-5min.json`
  - Result: passed for the short post-reset window.
- [x] `Tools\Test-BootMountHealth.ps1 -SinceMinutes 5 -PassThru -OutputJson Reports\boot-mount-health-post-onedrive-reset-5min.json`
  - Result: passed for the short post-reset window.
- [x] `Tools\Collect-DrivePerformanceSyncRisk.ps1 -SinceMinutes 240`
  - Result:
    `Reports\drive-performance-sync-risk\20260430-153629`.

## Evidence To Preserve

- Previous `AutoMount_VHDX_*` tasks were hidden startup one-liners generated
  from `C:\Users\david\Documents\PowerShell\New-PersistentVHDX.ps1`.
- `UnifiUdmDriveStackStartup` previously invoked
  `C:\Users\david\unifi_api\scripts\windows\Start-UDMDriveStack.ps1` and last
  returned `1` before the registration update.
- FilterManager Event ID 3 recurred for `\Device\Harddisk4\DR4` and
  `\Device\HarddiskVolume9`, with status `0xC03A001C`.
- Current VHDs:
  - `T:\vm\cloud-cache-disk.vhdx` -> attached, dynamic, Disk 5, drive `F:`.
  - `T:\vm\share-ext4.vhdx` -> attached, dynamic, Disk 6, no mounted data
    partition visible to Windows.
  - `T:\vm\shared-dev.vhdx` -> attached, fixed, Disk 4, drive `W:`.
- `C:\Users\david\unifi_api\docs\commands\rclone_udm_mount.log` was stale
  during investigation, last written before the current task failure.
