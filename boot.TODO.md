# Boot, Mount, Sync, and UI Responsiveness TODO

Purpose: harden boot/logon automation that mounts virtual disks, starts sync
providers, and initializes workstation services. The current implementation
focuses on making filesystem/filter failures visible, reducing startup
contention, and preserving UI responsiveness while keeping VHD, WSL, OneDrive,
Google Drive, UDM, and developer workflows functional.

Last updated: 2026-05-30 after input-stack (Shift + trackpad + fingerprint)
re-investigation — recurrence of the 2026-05-02 touchpad glitch. New reusable
toolkit at `Tools\InputDiagnostics\`; see the "## 2026-05-30 Input-Stack
Re-Investigation" section below. Key cross-finding: the 2026-05-02
Balanced→High Performance power-plan switch is being reverted to Balanced on
every boot by Process Lasso `StartWithPowerPlan=Balanced` in `prolasso.ini`.

2026-06-06 workstation remediation update:
- Evidence root: `Reports\workstation-audit-20260606-124859\`.
- Current boot diagnostics refreshed:
  `Reports\boot-diagnostics\20260606-133230`; post-reboot verifier found
  `PostRebootFailureCount = 0`.
- `Tools\Test-BootMountHealth.ps1 -SinceMinutes 240 -PassThru` now passes
  with no warnings after the fresh boot-diagnostics report.
- `Tools\Test-SyncProviderHealth.ps1 -SinceMinutes 60 -PassThru` passes for
  OneDrive/GoogleDrive; remaining warning is stale/optional iCloud sync root
  with data present and no running provider process.
- Stopped `WebManagement` and changed it from auto-start to manual after
  HTTP.sys `50443` SSL certificate churn was traced to the Web Management
  service and the `Windows Web Management` cert store. Rollback:
  `sc config WebManagement start= auto` then `sc start WebManagement`.
- Removed interactive-service flag from `MediatekSwitchUSB`; service is now
  normal own-process, auto-start, and running. Rollback:
  `sc config MediatekSwitchUSB type= interact type= own`.
- Changed `PC_AI-ToolRouter` from auto-start to manual because it depended on
  disabled `PC_AI-VLLM` and generated guaranteed boot errors. Rollback:
  `sc config PC_AI-ToolRouter start= auto` after enabling/repairing VLLM.
- Restarted Bonjour; no Bonjour errors appeared in the immediate one-hour
  post-restart sample. Service remains automatic/running for Apple/iCloud
  compatibility.
- Removed stopped Docker container `vigil-runner-docker-1` after saving
  `docker inspect`; container reclaim went from ~1.7 GB to 0 B.
- PnP rescan and targeted restart cleared no NVIDIA issue: internal
  `NVIDIA RTX 2000 Ada Generation Laptop GPU` remains Code 31
  (`CM_PROB_FAILED_ADD`, status `0xC0000182`) on `oem236.inf`. Do not
  uninstall/reinstall display drivers without a driver rollback package.

2026-06-06 direct-pass continuation:
- Docker cleanup completed without stopping running containers. Removed unused
  images and build cache: images `28 -> 8`, image reclaim `25.32 GB -> 0 B`,
  build cache `2.298 GB -> 0 B`; running containers remained up. Volumes were
  not pruned because inactive Docker volumes can contain project state.
- Disabled non-primary OneDrive startup/reporting tasks for inactive/offline
  accounts (`WsiAccount`, `DevToolsUser`, `CodexSandboxOffline`) after finding
  no repo/profile automation dependency. Current user `david` OneDrive tasks
  remain enabled and healthy. Rollback: `Enable-ScheduledTask` for the six
  disabled OneDrive task names.
- Added read-only NVIDIA split-driver checker:
  `Tools\InputDiagnostics\Test-NvidiaDualGpuDriverHealth.ps1`; Pester
  `Tests\InputDiagnostics\InputDiagnostics.Tests.ps1` now covers it and passes
  `46/46`.
- NVIDIA remains the active device issue: internal RTX 2000 Ada is on
  `32.0.15.9659` / `596.59` and Code 31; eGPU RTX 5060 Ti is on
  `32.0.15.9186` / `591.86` and OK. Both driver packages were exported under
  `Reports\workstation-audit-20260606-124859\direct-pass-20260606\driver-export\`.
- Fixed PC_AI servicehost build debt by aligning `System.Text.Json` in
  `Native\PcaiServiceHost\PcaiServiceHost.csproj` with `PcaiNative`
  (`10.0.7`) and rebuilding `Build.ps1 -Component servicehost`.
- Repointed disabled `PC_AI-HVSockProxy` and disabled `PC_AI-VLLM` NSSM entries
  from dead `C:\Users\david\PC_AI` paths to live `C:\codedev\PC_AI` servicehost
  artifacts. Services remain disabled/stopped.
- Fixed `PcaiServiceHost` HVSock status parsing so old PowerShell-written
  `state.json` files deserialize correctly and PID `0` is not reported as
  running. Cleared stale HVSock state; final status is "No HVSOCK proxies
  running."
- `PC_AI-ToolRouter` remains stale: it points to removed
  `Deploy\functiongemma-finetune\tool_router.py` and should be migrated to the
  current `Deploy\rust-functiongemma-runtime` path in a separate work item.
- Final validation: no new 30-minute critical/error/warning events,
  `Tools\Test-SyncProviderHealth.ps1 -SinceMinutes 60 -PassThru` passes, and
  `pnputil /enum-devices /problem` shows only NVIDIA Code 31.

2026-06-06 repo-fix pass:
- Fixed FunctionGemma runtime config drift:
  `Config\pcai-functiongemma.json` now points to the existing repo-local
  `Models/functiongemma-270m-it` model directory; new Pester coverage prevents
  stale user-profile model paths from returning.
- Fixed FunctionGemma runtime build drift:
  the default heuristic runtime no longer pulls the full CUDA/model core crate
  just to parse config, so `Tools\Invoke-RustBuild.ps1 -Path
  Deploy\rust-functiongemma-runtime -LlmOutput -CargoArgs @('check',
  '--no-default-features')` now passes.
- Kept the heavy `model` feature as a separate GPU build work item. Current
  blocker evidence is NVCC/MSVC setup and Windows command-line length failures
  in `candle-kernels` / `candle-flash-attn`, not the default router path.
- Fixed Rust/CUDA tooling defaults:
  `Tools\Initialize-CudaEnvironment.ps1` now prefers CUDA `v13.1` over `v13.2`
  because the current `cudarc`/Candle line rejects CUDA 13.2; static Pester
  coverage locks that preference.
- Hardened `Tools\Invoke-RustBuild.ps1` for agent usage: `-LlmOutput`,
  missing-argument failure, path resolution, and CargoTools return-code
  handling.
- Added fail-loud NVIDIA diagnostics:
  `Tools\InputDiagnostics\Test-NvidiaDualGpuDriverHealth.ps1 -FailOnIssue`
  now reports local NVIDIA App/update artifacts and exits nonzero when active
  issues are present. Do not use it as an installer; use it as pre/post driver
  validation.
- Input diagnostics now cover `Test-KeyInput.ps1` and `Watch-InputGlitch.ps1`.
  The ELAN/ELAS issue is currently a WUDFRd warning path
  (`ACPI\VEN_ELAS&DEV_B41A`); the actual touchpad device remains Sensel
  `SNSL002D`, so live symptom validation should use the key monitor and
  glitch ledger before applying driver/firmware changes.
- Validation:
  `Reports\workstation-audit-20260606-124859\Run-RepoFixValidation.ps1`
  passes all focused gates; `git diff --check` passes.

2026-06-06 touchpad power-management pass:
- Added `Tools\InputDiagnostics\Repair-TouchpadPowerManagement.ps1`, a
  targeted reversible repair for the Sensel `SNSL002D` touchpad path and Intel
  `7E78` I2C controller when `Watch-InputGlitch.ps1` shows
  `PowerDownEnabled.CanPowerDown=true`.
- Applied the fix without restarting live devices. Rollback backup:
  `Tools\InputDiagnostics\backups\touchpad-power-20260606-152612.json`.
- Post-fix validation snapshot:
  `Reports\input-glitch-watch\snapshots\snap-20260606-152639.json`.
  Result: touchpad `Status=OK`, keyboard `Status=OK`, no recent input errors,
  and both `SNSL002D` / `7E78` power-down permissions now report
  `CanPowerDown=false`.
- Remaining non-OK devices in the post-fix input collector are not touchpad
  devices: Cisco AnyConnect virtual miniport and internal
  `NVIDIA RTX 2000 Ada Generation Laptop GPU` Code 31. NVIDIA can still be an
  indirect UI/display instability contributor, but the direct touchpad evidence
  now points to the Sensel/I2C power-management path being remediated.
- Validation: `Reports\workstation-audit-20260606-124859\Run-InputDiagnosticsValidation.ps1`
  passes `65/65`.

Last updated: 2026-05-02 after touchpad glitch investigation: evidence
captured under `Reports\touchpad-glitch-investigation-20260502\`, restore
point #68 created (PreTouchpadFix), power plan switched Balanced→High
Performance, I2C HID device (ACPI\SNSL002D) disable/enable cycled. Step 1
(System Restore to RP 65) staged but not pulled pending touchpad re-test.
Previous update 2026-04-30 after post-reboot validation on boot time
2026-04-30 12:50:02 America/New_York, UDM startup disablement, OneDrive
registry/file-notification review, conservative registry rollback, Process
Lasso watchdog deployment, `~\bin` script risk review, and OneDrive
install/reset repair. Reconciled into high-level project docs on 2026-04-30.

## 2026-05-30 Input-Stack Re-Investigation (Shift / Trackpad / Fingerprint)

Recurrence of the 2026-05-02 touchpad glitch, now reported as Shift-key +
trackpad + fingerprint freezing. Investigated with systematic-debugging +
advisor, grounded in Microsoft Learn docs. Two distinct tiers identified.

New reusable toolkit (complements `Collect-BootDiagnostics.ps1` /
`Apply-ProcessLassoUiSyncTuning.ps1`):
- `Tools\InputDiagnostics\Invoke-InputStackDiagnostics.ps1` (read-only collector)
- `Tools\InputDiagnostics\Repair-InputStackQuickWins.ps1` (reversible, backup + `-Revert`)
- `Tools\InputDiagnostics\Start-LoadCapture.ps1` (native powercfg thermal/DPC capture)
- `Tools\InputDiagnostics\README.md` (findings + MS doc references)

Findings:
- Keyboard `ACPI\LEN0071` (EC) and touchpad Synaptics `SNSL002D` (I2C) are on
  different buses → co-freeze is system-wide (DPC/contention), not one device.
- Fingerprint = Synaptics USB `VID_06CB`; **USB selective suspend ON (AC+DC)**
  → resume-latency glitch on the reader.
- **Login storm: 57 autostart entries** (Docker, Ollama, LM Studio, 4×
  GoogleDriveFS, 6× USB-audio panels, Razer, MATLAB, SOLIDWORKS…).
- **Pro-audio ASIO drivers** (RME Fireface/MADIface, Focusrite, Topping,
  miniDSP) are the leading suspect for the kbd+trackpad co-freeze (DPC latency).
- **Accessibility hotkeys** were armed (FilterKeys `DelayBeforeAcceptance=1000ms`).
- Acute 5/29 cascade: 6× Kernel-Power 41 in 32 min + 7× nvlddmkm + 1 WHEA
  corrected error, **no crash dump** (CrashDumpEnabled=0x3). All 7 KP41 in 90d
  were on 5/29 → acute outlier, not the chronic complaint.
- Process Lasso EXONERATED for input freezes (live `prolasso.ini` shields
  `dwm.exe`/`explorer.exe`; no throttle/affinity on input/UI).
- `PC_AI-HVSockProxy` (this repo's PC-AI.Virtualization service) + `vtss` (Intel
  VTune sampler) fail at **every** boot ("system cannot find the path specified").
- Windows Hello *face* separately broken: IR camera in `Error`, Camera Frame
  Server crashed ×3.

Done:
- [x] Disable accidental accessibility activation hotkeys (HKCU, no admin).
  - StickyKeys 510→506, FilterKeys 126→122, ToggleKeys 58 (cleared 0x04).
  - Verified via collector baseline:
    `Logs\input-diagnostics\input-diagnostics-20260530-125420.txt`.

2026-05-30 (continuation) — Shift-specific root-cause pivot:
- Symptom refined by user: **bare Shift (L or R) does nothing; Ctrl+Shift works
  intermittently**; laptop built-in keyboard. This is NOT a DPC/freeze pattern
  (a stall drops all keys equally) — it is the signature of a **system-wide
  keyboard hook intercepting bare Shift**.
- Ruled out (live `SystemParametersInfo` GET): FilterKeys ON=False, StickyKeys
  ON=False, no Shift latched, no Scancode Map, PowerToys Keyboard Manager config
  EMPTY, single US layout. Accessibility is NOT the cause.
- Prime suspect: **Logitech Options+** (`logioptionsplus_agent` running; installs
  a system-wide kbd hook; known to intermittently break bare Shift). Razer not
  running. A/B test started: killed `logioptionsplus_agent`/`appbroker` →
  awaiting user confirm that bare Shift returns.
- New scripts: `Tools\InputDiagnostics\Reset-AccessibilityKeysLive.ps1` (live
  SPI fix, applied) and `Tools\InputDiagnostics\Repair-WorkstationInputReliability.ps1`
  (elevated; USB suspend + crash dump + per-device power + disable HVSockProxy/vtss;
  backup + `-Revert` + `-WhatIf`; elevation guard validated).
- Toolkit COMPLETE + VALIDATED 2026-05-30 (built via parallel powershell-pro agents):
  6 scripts in `Tools\InputDiagnostics\` + `Tools\InputDiagnostics\Optimize-StartupLoad.ps1`
  (login-storm trim; report-only validated: 63 startup entries → 19 essential /
  15 deferrable / 29 review; HKCU-only `-Apply`/`-Revert`, no admin) +
  `Tests\InputDiagnostics\InputDiagnostics.Tests.ps1` (Pester v5, **41/41 passing**,
  independently re-run). All scripts parse-clean, help-documented, read-only/mutating
  contracts enforced.
- [~] CORRECTION (2026-05-30 later) — earlier "Logitech Options+ = Shift culprit"
  was WRONG (coincidental timing). Shift RECURRED while Razer is UNINSTALLED and
  `logioptionsplus_agent` is NOT running. Re-verified live with a clean stack:
  FilterKeys/StickyKeys OFF, no Shift latched, no Scancode Map, keyboard class
  UpperFilters = {kbdclass} only (no orphaned rzudd), no rz*.sys, PowerToys KBM
  engine not running. => Software input hooks EXONERATED (Razer + Logi + PowerToys).
  Bare Shift fails / Ctrl+Shift works / all other keys type / TrackPoint always
  works / touchpad (I2C-HID) glitches independently.
  LEADING CAUSE: **ThinkPad keyboard EC / firmware** (intermittent), shared with the
  Synaptics I2C touchpad via the EC/Intel-SerialIO-I2C path; TrackPoint (separate
  EC/PS2 path) immune. Possible triggers: sleep/resume, eGPU Thunderbolt power events
  (WHEA PCIe corrected errors recur), EC glitch.
  - [ ] USER decisive tests: (a) Caps Lock → capitals? (b) `Test-KeyInput.ps1` → do
    Shift events reach OS? (c) external USB keyboard Shift; (d) osk.exe Shift.
  - [ ] FIX path if hardware/EC: EC power-drain reset (AC off, hold power 30s / P1
    Gen 7 emergency reset pinhole), then **Lenovo BIOS/EC update** (BIOS 1.20 → check
    Vantage) + Synaptics touchpad driver update. New tool: `Tools\InputDiagnostics\Test-KeyInput.ps1`.

2026-05-30 — eGPU + Razer resolution (user-confirmed Razer interferes with touchpad):
- **eGPU identified**: the "RTX 5060 Ti" is in a **Razer Core X V2** over USB4/TB,
  daisy-chained with Cable Matters + CalDigit Element Hub + Focusrite TB Audio.
- **Razer Synapse NOT required for Core X V2** (plug-and-play TB/USB4 PCIe enclosure;
  Synapse only drives Razer peripherals/Chroma). Safe to uninstall — eGPU keeps working.
- [x] Removed `RazerAppEngine` from HKCU\Run (stops auto-launch / touchpad+Shift
  interference). Backup: `Tools\InputDiagnostics\backups\` (reg export). Restore value:
  RazerAppEngine = '"C:\Program Files\Razer\RazerAppEngine\RazerAppEngine.exe" --url-params=apps=synapse,chroma-app --launch-force-hidden=synapse,chroma-app --autoStart=1'.
- [ ] USER: full uninstall (elevated, interactive):
  `& "C:\Windows\Installer\Razer\installer2\App\RazerInstaller.exe" /uninstall true`
  (removes Razer Synapse 4.0.662 + Razer Chroma 4.0.662 + 6 Razer services).
- **eGPU hardware link**: recurring `WHEA-Logger 17` corrected PCIe errors (5/29 21:17
  during the hard-freeze cascade, and 5/30 13:19). Likely the Thunderbolt/USB4 eGPU
  link. [ ] USER: connect Core X V2 to a DEDICATED TB4/USB4 port (NOT daisy-chained
  through the Cable Matters/CalDigit hub — chaining an eGPU is a known instability
  source), use a certified TB4 cable, update the NVIDIA driver. Crash dump (0x7 via
  Repair-WorkstationInputReliability.ps1) will capture the next hard freeze.

2026-05-30 — APPLIED (elevated, user-approved UAC; backup
`Tools\InputDiagnostics\backups\backup_20260530T135052.json`; log
`Logs\elevated\repair-Apply-20260530-135052.log`):
- [x] USB selective suspend AC/DC 0x1→0x0; CrashDumpEnabled 3→7 (+AutoReboot);
  EnhancedPowerManagementEnabled=0 on Synaptics fingerprint + 4 USB Root Hubs;
  PC_AI-HVSockProxy + vtss → Disabled. (Sign out/in for HID/BT power re-init.)

2026-05-30 — PROCESS LASSO ↔ TERMINAL ↔ eGPU review (user hypothesis CONFIRMED):
Two prolasso.ini settings create the competitive interaction:
- **EfficiencyMode (line 142)** pins `pwsh.exe` + `windowsterminal.exe` to EcoQoS
  (E-cores, capped freq) — throttles the interactive terminal driving eGPU work.
- **DefaultGPUAdapterPreferences (line 135)** sets Windows Terminal GPU pref = `2`
  (High-Perf = the **eGPU**) → terminal RENDER competes with eGPU COMPUTE over the
  same Thunderbolt link.
Touchpad/input is otherwise well-protected (syntp*/snsl*/synaptics*/elan*/etd* at
Above-Normal, IO 3, ProBalance-excluded; dwm/ctfmon/textinputhost boosted+excluded)
— so PL is NOT throttling the touchpad; that glitch is the I2C/EC-under-load issue.
- [x] FIXED & VALIDATED (2026-05-30, `Repair-ProcessLassoTerminalGpu.ps1`, elevated;
  backup `prolasso.ini.bak-20260530-140925`; log `Logs\elevated\plgpu-Apply-20260530-140925.log`):
  removed `windowsterminal.exe`+`pwsh.exe` from EfficiencyMode; set ALL
  `DefaultGPUAdapterPreferences` → auto(0) (eGPU forcing on Terminal removed). On-disk
  validated. Governor (task "Process Lasso Core Engine Only" + watchdog) reloads at sign-in.
  - [ ] Still open: reconcile `StartWithPowerPlan=Balanced` vs 2026-05-02 High-Perf intent (user decision).
  - Consolidated machine issues: see [machine-reliability.TODO.md](machine-reliability.TODO.md).

Open (need elevation / user action — preserve evidence per repo convention):
- [ ] Run elevated `Tools\InputDiagnostics\Repair-InputStackQuickWins.ps1`
  → disable USB selective suspend (AC+DC) + set CrashDumpEnabled=0x7 (MS rec for
  >32GB RAM) so the next hard freeze is captured. Backup auto-written to
  `Tools\InputDiagnostics\backups\`.
- [ ] Resolve power-plan revert: Process Lasso `StartWithPowerPlan=Balanced`
  overrides the 2026-05-02 High Performance switch on every boot. Decide intended
  plan and reconcile `prolasso.ini` via `Apply-ProcessLassoUiSyncTuning.ps1`.
- [ ] Confirm co-freeze driver with LatencyMon (~10 min under load); update/roll
  back the offending pro-audio / `nvlddmkm` / `Wdf01000` driver.
- [ ] Trim login storm (Task Manager Startup) and gate heavy AI/dev tools.
- [ ] Fix `PC_AI-HVSockProxy` (missing path) + `vtss` auto-start failures.
- [ ] Repair Windows Hello face camera stack (IR camera Error + RealSense).
- [ ] Install sensor/latency tools when online+elevated:
  `winget install REALiX.HWiNFO -e` ; `winget install Resplendence.LatencyMon -e`.

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

## 2026-05-02 Touchpad Glitch Investigation

Operational ledger for the touchpad regression discovered 2026-05-02.

### Evidence captured (read-only)

- `Reports\touchpad-glitch-investigation-20260502\01-touchpad-devices.txt` — Sensel
  HID collection inventory.
- `02-driver-versions.txt` — split-version state (COL01/COL03 at 26100.1150;
  COL02/COL04 + ACPI\SNSL002D at 26100.8328).
- `03-pnputil-touchpad.txt` — DriverStore enumeration (older drivers retained
  for iaLPSS2_I2C_MTL, etdhsa, n48et firmware).
- `05-events.txt` — Kernel-PnP / WUDFRd events; 41 critical at 5/2 13:25:42.
- `06-hotfixes.txt` — KB5083631 + 5/1 5:56-5:57 OS rollup batch.
- `09-top-processes.txt` / `10-onedrive-io-now.txt` / `11-power-mgmt.txt` /
  `12-lenovo-vantage.txt` — runtime baseline.
- `14-preplan-checks.txt`, `14a-kb-removability.txt`,
  `15-hypothesis-verification.txt` — verification of hypotheses (V1: I2C
  controller DriverDate is 2025-07-02 NOT 4/30; V2: WUDFRd events are
  chronic boot-time noise from unrelated devices, not touchpad signal).

### Hypothesis (after verification)

1. Primary: 5/1 cumulative servicing rollup `26100.8328.1.31` regressed inbox
   HID class drivers (input.inf, hidi2c.inf). Rollback path = System Restore
   only (inbox drivers; no vendor INF to swap).
2. Secondary: storage IO contention from OneDrive + GoogleDriveFS sync
   (priority demotion on 4/30 did not cap IOPS).
3. Demoted: iaLPSS2_I2C_MTL rollback (verified not changed in window).
4. Demoted: ETDHSA rollback (no evidence).
5. Demoted: WUDFRd 0xC0000365 as touchpad signal (chronic noise).

### Applied 2026-05-02 (gates + low-risk steps)

- [x] Gate A: System Restore Point #68 "PreTouchpadFix-2026-05-02" created
  2026-05-02 19:02:43 UTC (`gateA-restorepoints.txt`).
- [x] Gate B: VSS feasibility OK; C: free 588 GB
  (`gateB-vss-feasibility.txt`).
- [x] Power plan rollback artifact: GUID `381b4222-f694-41f0-9685-ff5bb260df2e`
  (Balanced) exported to `powercfg-current-backup.pow`.
- [x] Process Lasso config snapshot (20 files) → `pl-config-backup\`.
- [x] Pre-restore package list snapshot → `packages-pre-step1.txt`.
- [x] Step 2.3 — power plan switched to High Performance
  (`8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c`).
- [x] Step 0 — `pnputil /disable-device` then `/enable-device` on
  `ACPI\SNSL002D\4&39979B3E&0`. Post-cycle status: OK.

### Applied 2026-05-02 (deeper I2C / power fix after first round insufficient)

User reported continuing glitches after Step 0 + 2.3. Deep-dive into I2C /
HID filter chain / power management identified:

- Kernel-PnP Id 906 "long running thread for device add routine, 3000ms"
  events at boot (13:26:13, 14:12:41) — device enumeration exceeding SLA.
- `EnhancedPowerManagementEnabled = 1` on `ACPI\SNSL002D\4&39979B3E&0` —
  classic ThinkPad selective-suspend / D-state cycling that delays first
  touch report after idle.
- `DeviceResetNotificationEnabled = 1` — driver reset notifications enabled
  (firmware reset path active).
- Disk queue depth confirmed 0.00 across all volumes — IO contention
  eliminated as contributor.

External research corroborated this is a documented pattern:
- Lenovo forum thread "Haptic Touchpad in Thinkpad P1 Gen 7 randomly freezes"
  (post 5351108) — same model, same symptom.
- Notebookcheck review documents Sensel firmware freezing on palm + TrackPoint.
- KB5083631 introduced new haptic-signal subsystem racing HID input stream.
- n48et.inf is BIOS firmware (System Firmware 1.19 / 1.20), NOT Sensel
  touchpad firmware — earlier annex was misaimed.
- Sensel Curie firmware not visible in UEFI capsule list; delivered via
  Lenovo Vantage. No firmware push events in Setup log last 30d.

Applied:
- [x] `Apply-PowerFix.ps1` — set `EnhancedPowerManagementEnabled = 0` on
  `ACPI\SNSL002D\4&39979B3E&0` and restarted device. Pre-fix state captured
  in `powerfix-rollback.json` (was 1). Rollback: `Apply-PowerFix.ps1 -Restore`.

### Open items (next escalation steps)

- [x] Superseded 2026-06-06: touchpad re-test after ACPI EPM=0 led to a
  stronger targeted repair. `Repair-TouchpadPowerManagement.ps1` now disables
  `MSPower_DeviceEnable` for both Sensel `SNSL002D` and Intel `7E78`, and
  writes `EnhancedPowerManagementEnabled=0` on the related Sensel HID
  collections. Continue measuring recurrence with `Watch-InputGlitch.ps1
  -Mode Report -SinceFix 2026-06-06`.
- [ ] If touchpad pointer/finger-press stickiness persists after a reboot or
  sign-out/in: rerun `Watch-InputGlitch.ps1 -Mode Snapshot -Symptom touchpad`
  immediately during the symptom, then decide between `-RestartDevices`,
  Lenovo Vantage Sensel/BIOS firmware review, Dell DDPM cursor-hook testing,
  or NVIDIA driver remediation based on the captured state.
- [ ] If glitches persist: Step 2.1 — pause OneDrive + Google Drive sync
  (graceful: tray pause for 2h, OR programmatic via `OneDrive.exe /shutdown`).
  If glitches stop while paused → hypothesis #2 confirmed; investigate IO
  isolation tactics rather than System Restore.
- [ ] If glitches persist with sync paused: Step 1 — System Restore to RP 65
  (4/21 Scheduled Checkpoint, predates rollup). Pre-step actions: pause
  Windows Update 7d, document apps installed since 4/21 for reinstall list.
- [ ] After any restore: re-apply 4/30 PL UI/sync tuning via
  `Tools\Apply-ProcessLassoUiSyncTuning.ps1`.

### Rollback artifacts

| Step | Artifact | Restore command |
|---|---|---|
| Power plan (2.3) | `powercfg-current-backup.pow` + GUID in `powercfg-rollback.txt` | `powercfg /setactive 381b4222-f694-41f0-9685-ff5bb260df2e` |
| All steps (umbrella) | RP #68 PreTouchpadFix-2026-05-02 | `Restore-Computer -RestorePoint 68` |
| PL config (if 2.2 applied) | `pl-config-backup\` | Stop ProLasso, copy back, restart |

### Off-limits / annex

- n48et.inf firmware downgrade — NOT in plan. Firmware flash is irreversible
  even if older INF binds. Requires separate written consent + Lenovo Support
  consultation.
- iaLPSS2 / ETDHSA pnputil rollback — explicitly removed per V1 verification.
