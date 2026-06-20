# Workstation Audit Summary - 2026-06-06

Machine: `dtm-p1gen7`
Repo: `C:\codedev\PC_AI`

## First Fix: PowerShell Profile, Backup Files, and Codex CLI

The immediate blocker was command processing around PowerShell startup and Codex CLI resolution. The active symptoms were repeated `OpenWith.exe` "Pick an app" windows and profiled `pwsh.exe` instances waiting until those windows were closed.

Changes applied:

- Hardened `C:\Users\david\Documents\PowerShell\Microsoft.PowerShell_profile.ps1` command dispatch so `.bak` / `.backup` names and resolved paths are rejected before command lookup.
- Hardened `C:\Users\david\Documents\PowerShell\Modules\ProfileUtilities\MachineConfiguration\MachineConfiguration.psm1` so machine tool resolution does not return `.bak` / `.backup` files.
- Classified Codex-managed shells as fast/minimal at the start of the profile.
- Skipped `ProfileAccelerator` import for fast/Codex-managed profile startup.
- Moved PATH-visible backup command files out of `C:\Users\david\bin` into:
  `C:\Users\david\bin\Backups\profile-command-shims-20260606`
- Added `C:\Users\david\bin\codex.cmd` to force Windows/Codex command resolution through npm's `.cmd` shim before npm's PowerShell shim.
- Moved npm's non-CMD Codex shims into:
  `C:\Users\david\AppData\Roaming\npm\Backups\codex-command-shims-20260606`
- Restored `.ps1` association using `restore-ps1-association.reg`:
  `.ps1 -> Microsoft.PowerShellScript.1`
  open command -> PowerShell 7 `pwsh.exe -NoProfile -File "%1" %*`

Validation:

- `where codex` now returns only:
  `C:\Users\david\bin\codex.cmd`
  `C:\Users\david\AppData\Roaming\npm\codex.cmd`
- `Get-Command codex -All` now returns only `.cmd` applications.
- `codex --version` returns `codex-cli 0.137.0`.
- No `OpenWith.exe` process remained after cleanup.
- `Resolve-MachineToolPath npm.cmd.bak -BypassCache` and `npx.cmd.bak -BypassCache` returned no path after resolver hardening.

Rollback:

- Move files back from the two backup folders above if a shell-specific non-CMD shim is needed.
- Re-import or edit `restore-ps1-association.reg` if `.ps1` association should be changed again.

## Additional System Fixes: File Type and Command Search

Applied after the initial profile/Codex fix:

- Added a user-level `PATHEXT` override:
  `.COM;.EXE;.BAT;.CMD;.MSC;.PY;.PYW;.CPL`
- This removes Windows Script Host file types from command search for new user shells:
  `.VBS`, `.VBE`, `.JS`, `.JSE`, `.WSF`, `.WSH`.
- Backed up environment registry state before the change:
  `backup-hklm-environment-before-pathext-fix.reg`
  `backup-hkcu-environment-before-pathext-fix.reg`
- Broadcast `WM_SETTINGCHANGE` using:
  `Send-EnvironmentChange.ps1`
- Added a local validator:
  `Validate-PathextFix.cmd`

Validation:

- `reg.exe query HKCU\Environment /v PATHEXT` shows the sanitized user-level value.
- `Validate-PathextFix.cmd` passes:
  - bare command discovery finds `probe.cmd`
  - bare command discovery does not find a JS-only probe
- `codex --version` returns `codex-cli 0.137.0`.
- `npm --version` and `npx --version` return `11.15.0`.
- `Get-Command codex,npm,npx -All` completes without blocking.
- `taskkill /IM OpenWith.exe /F` reports no active `OpenWith.exe`.

Rollback:

- Delete the user-level `PATHEXT` value:
  `reg.exe delete HKCU\Environment /v PATHEXT /f`
- Or restore from `backup-hkcu-environment-before-pathext-fix.reg`.
- Then run `Send-EnvironmentChange.ps1` or log off/on.

## Evidence Collected

This folder contains the collected read-only evidence:

- `hid-io-regression-triage-20260606.md` - current non-GPU HID/IO,
  haptic touchpad, Shift-key, Terminal/Claude, WDF/IPF, USB/network, and power
  maintenance triage with validated next actions.
- `boot-mount-health.json`
- `sync-provider-health.json`
- `process-lasso-boot-safety.txt`
- `pnp-present-not-ok.csv`
- `events-7d-summary.txt`
- `logical-disks.txt`
- `docker-system-df.txt`
- `docker-buildx-du.txt`
- `docker-ps.txt`
- `wsl-status.txt`
- `wsl-list-verbose.txt`
- `restore-ps1-association.reg`

## System Remediation Pass

After the command-processing fixes, the machine was rechecked using repo-local
diagnostic tools plus Windows logging/service/device utilities. The following
safe remediations were applied and validated.

### Boot, VHD, and Process Lasso

- Refreshed boot diagnostics with
  `Tools\Collect-BootDiagnostics.ps1 -SinceMinutes 240 -PostRebootVerify`.
- New boot report:
  `Reports\boot-diagnostics\20260606-133230`.
- `PostRebootFailureCount` is `0`.
- `Tools\Test-BootMountHealth.ps1 -SinceMinutes 240 -PassThru` now reports:
  `Status: Pass`, no failures, no warnings.
- `Tools\Test-ProcessLassoBootSafety.ps1` still passes; Governor is running and
  exclusions/logging remain in place.

Validation artifacts:

- `boot-mount-health-after-bootdiag.txt`
- `process-lasso-refresh.txt`
- `Reports\boot-diagnostics\20260606-133230\`

### Sync Providers

- `Tools\Test-SyncProviderHealth.ps1 -SinceMinutes 60 -PassThru` passes.
- OneDrive process and SyncDiagnostics evidence are fresh since boot.
- The remaining sync warning is `C:\Users\david\iCloudDrive`: the root exists,
  contains user data, and no iCloud provider process is currently running.
- No iCloud executable/service/startup entry was found during this pass, so the
  directory was not renamed or removed.

Validation artifacts:

- `sync-provider-health-refresh.txt`
- `sync-provider-processes.txt`

Next gated action:

- Decide whether iCloud should be reinstalled/repaired or whether
  `C:\Users\david\iCloudDrive` should be archived outside active sync-provider
  monitoring.

### HTTP.sys Port 50443 Churn

Windows `Microsoft-Windows-HttpEvent` warnings were dominated by repeated SSL
certificate binding create/delete events for port `50443`. The binding traced to:

- Service: `WebManagement`
- Display name: `Web Management`
- Binary: `C:\WINDOWS\system32\WebManagement.exe`
- Startup before fix: `AUTO_START`
- Binding store: `LocalMachine\Windows Web Management`
- App ID: `{7125f84e-a584-49bf-b90f-ecaa96fe8b63}`

Applied fix:

- Backed up service configuration.
- Stopped `WebManagement`.
- Changed startup type to demand/manual.

Validation:

- `sc qc WebManagement` reports `DEMAND_START`.
- `sc queryex WebManagement` reports `STOPPED`, PID `0`.
- `netstat` shows no active listener on `:50443`.
- `Check-HttpEventRecent.ps1 -Minutes 2` produced no new HTTP Event records
  after the service was stopped.

Rollback:

```powershell
sc.exe config WebManagement start= auto
sc.exe start WebManagement
```

Validation artifacts:

- `webmanagement-sc-qc-before-fix.txt`
- `webmanagement-sc-query-before-fix.txt`
- `netsh-http-sslcert-127.0.0.1_50443.txt`
- `cert-Cert__LocalMachine_Windows_Web_Management.json`
- `Check-HttpEventRecent.ps1`

The certificate store was left intact. The service stop/manual-start change
removed the active churn without deleting local certificates.

### Service Control Manager Noise

Two SCM-backed issues were remediated.

`MediatekSwitchUSB`:

- Problem: service was registered as an interactive service, which Windows 11
  does not allow in normal desktop sessions.
- Fix: changed service type from interactive own-process to normal own-process.
- Validation: service restarted and now reports `TYPE: 10 WIN32_OWN_PROCESS`.

Rollback:

```powershell
sc.exe config MediatekSwitchUSB type= interact type= own
sc.exe stop MediatekSwitchUSB
sc.exe start MediatekSwitchUSB
```

`PC_AI-ToolRouter`:

- Problem: auto-start service depended on disabled service `PC_AI-VLLM`, causing
  repeat startup failure noise.
- Fix: changed `PC_AI-ToolRouter` startup type to demand/manual.
- Validation: `sc qc PC_AI-ToolRouter` reports `DEMAND_START`.

Rollback after `PC_AI-VLLM` is repaired/enabled:

```powershell
sc.exe config PC_AI-ToolRouter start= auto
```

Validation artifacts:

- `mediatekswitchusb-sc-qc-before-fix.txt`
- `pcai-toolrouter-sc-qc-before-fix.txt`

### Bonjour

Bonjour had recent hostname-conflict events for `dtm-p1gen7.local` falling back
to `dtm-p1gen7-2.local`.

Applied fix:

- Restarted `Bonjour Service`.

Validation:

- Service is `Running` and `Automatic`.
- The focused post-restart provider query did not find new Bonjour events in the
  current window.

Validation artifacts:

- `bonjour-before-restart.txt`
- `bonjour-after-restart.txt`
- `events-provider-Bonjour_Service.json`

### Docker

Docker showed reclaimable storage, including one stopped container consuming
about 1.7 GB.

Applied fix:

- Exported inspect data for stopped container `f2083b14a723`
  (`vigil-runner-docker-1`).
- Removed only that stopped container.

Validation:

- `docker ps -a` now shows all remaining containers running.
- `docker system df` now reports container reclaimable space as `0B`.

Validation artifacts:

- `docker-inspect-vigil-runner-before-rm.json`
- `docker-ps-after-container-rm.txt`
- `docker-system-df-after-container-rm.txt`

No image, volume, or WSL/Docker VHD prune/compaction was performed. Remaining
Docker cleanup is intentionally inventory-gated because volumes and image layers
may contain active project state.

### Device and Driver State

Safe device repair was attempted.

- Ran `pnputil /scan-devices`.
- Cisco AnyConnect virtual miniport no longer appears in `pnputil /enum-devices
  /problem`, but PowerShell PnP still reports it as disabled/error. Treat this
  as intentionally disabled unless VPN behavior is broken.
- Restarted the internal NVIDIA RTX 2000 Ada laptop GPU device using
  `pnputil /restart-device`.
- NVIDIA remains at Code 31:
  `CM_PROB_FAILED_ADD`, problem status `0xC0000182`.

Relevant NVIDIA driver packages:

- Internal RTX 2000 Ada: `oem236.inf`, `nvltwi.inf`, NVIDIA version
  `32.0.15.9659`, dated `2026-05-21`.
- eGPU RTX 5060 Ti: `oem395.inf`, `nv_dispig.inf`, NVIDIA version
  `32.0.15.9186`, dated `2026-01-20`.

Validation artifacts:

- `pnputil-scan-devices.txt`
- `nvidia-internal-before-restart.txt`
- `nvidia-internal-restart-result.txt`
- `nvidia-internal-after-restart.txt`
- `pnputil-problems-after-nvidia-restart.txt`
- `signed-drivers-display-net.json`
- `pnputil-drivers-display.txt`

Next gated action:

- Do not uninstall/reinstall display drivers blindly. Use Lenovo Vantage,
  Windows Update Optional Updates, or a known Lenovo/NVIDIA rollback package,
  preferably after a restore point and driver export.

### Post-Fix Event Check

- `Check-PostFixEvents.ps1 -Minutes 5` produced no critical/error/warning events
  in the immediate post-fix window.
- The one-hour event summary may still show pre-fix HTTP Event counts until that
  window ages out.

## Direct Pass - 2026-06-06

Evidence root for this pass:

- `direct-pass-20260606\`
- `direct-pass-20260606-after-actions\`

### Docker Cleanup

Docker image/container cleanup was completed without stopping or removing any
running containers.

Actions:

- Confirmed all running containers and their images.
- Searched repo, Codex, agent, and PowerShell automation paths for references
  to unused tagged images.
- Ran `docker image prune -f`; no dangling reclaim was available.
- Ran `docker image prune -a -f` to remove images unused by any container.
- Ran `docker builder prune -a -f` to remove build cache.

Result:

- Images: `28` -> `8`.
- Images reclaimable: `25.32 GB` -> `0 B`.
- Build cache: `2.298 GB` -> `0 B`.
- Containers: still `8` active, `0 B` reclaimable.
- Remaining images are exactly the images backing running containers.

Validation artifacts:

- `direct-pass-20260606\docker-image-prune-a.txt`
- `direct-pass-20260606\docker-builder-prune-a.txt`
- `direct-pass-20260606\docker-system-df-final.txt`
- `direct-pass-20260606\docker-ps-final.txt`

Remaining gated item:

- Local volumes still report `4.108 GB` reclaimable across inactive volumes, but
  volumes were not pruned because they can contain project state.

### OneDrive Task Triage

OneDrive itself remains healthy. The task noise was from non-primary account
tasks that never successfully ran.

Actions:

- Disabled OneDrive reporting/startup tasks for inactive `WsiAccount`.
- Searched repo/profile/machine automation for `DevToolsUser`,
  `CodexSandboxOffline`, and `WsiAccount`; no automation dependency was found.
- Disabled OneDrive reporting/startup tasks for `DevToolsUser` and
  `CodexSandboxOffline`.
- Kept the current user `david` OneDrive tasks enabled.
- Kept the per-machine updater task enabled.

Rationale:

- `WsiAccount` is inactive.
- `CodexSandboxOffline` is explicitly an offline sandbox account.
- `DevToolsUser` appears to be a development isolation account with no evidence
  of OneDrive dependency.

Rollback:

```powershell
Enable-ScheduledTask -TaskPath '\' -TaskName 'OneDrive Reporting Task-S-1-5-21-357761096-1057752491-2136275046-1003'
Enable-ScheduledTask -TaskPath '\' -TaskName 'OneDrive Startup Task-S-1-5-21-357761096-1057752491-2136275046-1003'
Enable-ScheduledTask -TaskPath '\' -TaskName 'OneDrive Reporting Task-S-1-5-21-357761096-1057752491-2136275046-1007'
Enable-ScheduledTask -TaskPath '\' -TaskName 'OneDrive Startup Task-S-1-5-21-357761096-1057752491-2136275046-1007'
Enable-ScheduledTask -TaskPath '\' -TaskName 'OneDrive Reporting Task-S-1-5-21-357761096-1057752491-2136275046-1009'
Enable-ScheduledTask -TaskPath '\' -TaskName 'OneDrive Startup Task-S-1-5-21-357761096-1057752491-2136275046-1009'
```

Validation:

- `Tools\Test-SyncProviderHealth.ps1 -SinceMinutes 60 -PassThru` reports
  `Status: Pass`.
- Current user OneDrive tasks still report `LastTaskResult = 0`.
- Remaining warning is unchanged and intentional for this pass:
  `iCloud` sync root exists while no provider process is running.

Validation artifacts:

- `direct-pass-20260606\onedrive-wsiaccount-tasks-before-disable.json`
- `direct-pass-20260606\onedrive-wsiaccount-tasks-after-disable.json`
- `direct-pass-20260606\onedrive-nonprimary-tasks-before-disable.json`
- `direct-pass-20260606\onedrive-tasks-after-nonprimary-disable.json`
- `direct-pass-20260606\sync-provider-health-post-direct-pass.txt`

### NVIDIA Dual-GPU State

The internal RTX 2000 Ada and eGPU RTX 5060 Ti remain split across NVIDIA driver
versions:

- Internal RTX 2000 Ada: `32.0.15.9659` / NVIDIA `596.59`, `Status=Error`,
  Code 31.
- eGPU RTX 5060 Ti: `32.0.15.9186` / NVIDIA `591.86`, `Status=OK`.

Actions:

- Exported both current NVIDIA display driver packages:
  - `oem236.inf` / `nvltwi.inf`
  - `oem395.inf` / `nv_dispig.inf`
- Added `Tools\InputDiagnostics\Test-NvidiaDualGpuDriverHealth.ps1`, a
  read-only checker for driver-version split, Code 31, and `nvidia-smi` state.
- Updated `Tools\InputDiagnostics\README.md`.
- Updated and ran `Tests\InputDiagnostics\InputDiagnostics.Tests.ps1`.

Validation:

- `Test-NvidiaDualGpuDriverHealth.ps1` reports
  `HasNvidiaDriverVersionSplit = true` and
  `HasNvidiaDeviceProblem = true`.
- Pester: `46 passed, 0 failed`.

Research-grounded conclusion:

- NVIDIA documents that two NVIDIA cards in one system should use the same
  driver to avoid conflicts, while GeForce plus Quadro/RTX professional mixed
  setups are not a fully supported configuration.
- NVIDIA's notebook/eGPU driver guidance also warns that notebook OEM support
  must be preserved; installing a driver that supports only the external GPU can
  break notebook-specific support.

Recommended gated fix path:

1. Create a restore point.
2. Keep the exported driver packages as rollback.
3. Prefer Lenovo Vantage / Lenovo-qualified display package first for the P1 Gen
   7 internal GPU.
4. If Lenovo does not provide a package that also supports the RTX 5060 Ti eGPU,
   test a single NVIDIA Studio/RTX package only after confirming it supports both
   device IDs.
5. After driver change, run:
   `Tools\InputDiagnostics\Test-NvidiaDualGpuDriverHealth.ps1 -AsJson`,
   `pnputil /enum-devices /problem`, and `nvidia-smi`.

Validation artifacts:

- `direct-pass-20260606\driver-export\`
- `direct-pass-20260606\nvidia-dual-gpu-health-final.json`
- `direct-pass-20260606\nvidia-dual-gpu-health-final\`

### PC_AI Service Staleness

The disabled PC_AI services were stale because NSSM still pointed to the removed
`C:\Users\david\PC_AI` tree.

Actions:

- Fixed `Native\PcaiServiceHost\PcaiServiceHost.csproj` package downgrade:
  `System.Text.Json` `8.0.5` -> `10.0.7`, matching `PcaiNative`.
- Rebuilt `Build.ps1 -Component servicehost -Configuration Release
  -SkipQualityGate`.
- Updated disabled NSSM service paths for:
  - `PC_AI-HVSockProxy`
  - `PC_AI-VLLM`
- Kept both services disabled.
- Fixed `PcaiServiceHost` HVSock status handling so stale PowerShell-created
  state files deserialize correctly and PID `0` is not treated as alive.
- Cleared stale HVSock state through `PcaiServiceHost.exe hvsock stop`.

Validation:

- `PC_AI-HVSockProxy`: disabled, stopped, NSSM points to current repo.
- `PC_AI-VLLM`: disabled, stopped, NSSM points to current repo.
- `PcaiServiceHost.exe --help` works.
- `PcaiServiceHost.exe hvsock status` now reports no proxies running after stale
  state cleanup.
- `PcaiServiceHost.exe vllm status` reports `vLLM container not running`.

Remaining service debt:

- `PC_AI-ToolRouter` still points to removed
  `Deploy\functiongemma-finetune\tool_router.py` under the old tree. The
  current repo appears to have moved FunctionGemma router work into
  `Deploy\rust-functiongemma-runtime`; service migration should be a separate
  work item.

Validation artifacts:

- `direct-pass-20260606\hvsock-nssm-before.txt`
- `direct-pass-20260606\vllm-nssm-before.txt`
- `direct-pass-20260606\pcai-servicehost-help.txt`
- `direct-pass-20260606\pcai-servicehost-hvsock-status-after-clear.txt`
- `.pcai\build\logs\build_pcai-servicehost_20260606_*.log`

### Final Direct-Pass Validation

- `Check-PostFixEvents.ps1 -Minutes 30`: no output, meaning no critical/error
  /warning records in that window.
- `Tools\Test-SyncProviderHealth.ps1 -SinceMinutes 60 -PassThru`: pass.
- `docker system df`: no image/container/build-cache reclaim remains.
- `pnputil /enum-devices /problem`: only internal NVIDIA RTX 2000 Ada Code 31
  remains.

## Current Findings

1. PowerShell/Codex command processing was unhealthy and has been fixed.
   - Rooted in active `.bak` files in PATH, broken `.ps1` association, and Codex resolving through non-CMD npm shims.
   - Follow-on validation shows `codex --version` works and no `OpenWith.exe`
     process remains.

2. The active device issue is the internal NVIDIA adapter.
   - NVIDIA RTX 2000 Ada Generation Laptop GPU still reports
     `CM_PROB_FAILED_ADD` after device rescan and restart.
   - Cisco AnyConnect no longer appears in `pnputil /enum-devices /problem`,
     though PowerShell PnP still reports it disabled/error. Treat Cisco as
     intentionally disabled unless VPN behavior is broken.

3. Event logs showed recurring platform/device issues over the last 7 days.
   - High-volume HTTP Event warnings were traced to `WebManagement` port
     `50443` SSL binding churn and remediated by stopping the service and
     changing it to demand/manual start.
   - Service Control Manager noise from `MediatekSwitchUSB` and
     `PC_AI-ToolRouter` was remediated.
   - Bonjour hostname-conflict noise was addressed with a service restart.
   - Kernel-PnP 219 warnings and DriverFrameworks-UserMode critical events
     still require correlation with the remaining NVIDIA Code 31 driver issue.
   - DriverFrameworks-UserMode critical events are present.
   - Kernel-Power 41 and EventLog 6008 indicate an unexpected shutdown/restart in the window.
   - FilterManager Event 3 and disk 158 warnings remain worth monitoring, but
     the latest boot/VHD validation passes.

4. VHD boot mount health passes with fresh post-boot evidence.
   - `cloud-cache-disk`, `share-ext4`, and `shared-dev` are attached/online.
   - AutoMount scheduled tasks show last result `0`.
   - The fresh boot diagnostics report has `PostRebootFailureCount = 0`.

5. Sync provider health currently passes, with one warning.
   - OneDrive diagnostics are fresh since boot.
   - iCloud sync root exists while the iCloud provider process is not running.

6. Process Lasso boot-safety check passes.
   - Governor is running/responding.
   - ProBalance and SmartTrim exclusions are present.
   - Logging flags and recent logs are present.

7. Docker container reclaimable space was reduced safely.
   - Removed one stopped `vigil-runner-docker-1` container after exporting
     inspect evidence.
   - Remaining containers are running and container reclaimable space is `0B`.
   - Images and volumes still have reclaimable space but were not pruned.
   - Build cache: 2.298 GB total.

8. Disk free space is acceptable but C: is approaching the range where cleanup is useful.
   - C: 416.5 GB free of 3717.4 GB, 11.2%.
   - T: 634.7 GB free of 3726 GB, 17.0%.

9. WSL inventory is stable but Docker Desktop is the only running distro.
   - Default distro: Ubuntu-22.04, stopped.
   - Running distro: docker-desktop.

## Repo Fix Pass - 2026-06-06

Read-only evidence was collected under
`repo-fix-pass-20260606`; focused validation logs are under
`repo-fix-validation-20260606`.

Completed fixes:
- Corrected `Config\pcai-functiongemma.json` to use the existing repo-local
  FunctionGemma model directory: `Models/functiongemma-270m-it`.
- Split the default FunctionGemma runtime build away from the heavy model/CUDA
  core dependency. The heuristic runtime now validates with
  `check --no-default-features`; the full `model` feature remains the gated
  CUDA/model path.
- Aligned `rust-functiongemma-core`'s direct `cudarc` dependency with
  `candle-core 0.9.2`'s `cudarc 0.19.0` resolver line.
- Changed `Tools\Initialize-CudaEnvironment.ps1` to prefer CUDA `v13.1`
  before `v13.2`, matching the repo's Rust/CUDA compatibility notes and
  avoiding the current `cudarc` CUDA 13.2 build-script panic.
- Hardened `Tools\Invoke-RustBuild.ps1` with `-LlmOutput`, path validation,
  missing-argument failure, and CargoTools returned-exit-code handling.
- Updated CargoTools tests to current exported command names and added config
  drift tests for FunctionGemma and CUDA initializer defaults.
- Brought `Test-KeyInput.ps1` and `Watch-InputGlitch.ps1` under the input
  diagnostics structural test contract.
- Extended `Test-NvidiaDualGpuDriverHealth.ps1` with local NVIDIA App/update
  artifact detection, explicit issue reporting, recommended actions, and
  `-FailOnIssue` for automation gates.
- Added and applied `Tools\InputDiagnostics\Repair-TouchpadPowerManagement.ps1`
  for the live Sensel touchpad stickiness path. It disables the
  `MSPower_DeviceEnable` power-down permission for `SNSL002D` and Intel `7E78`,
  writes reversible `EnhancedPowerManagementEnabled=0` values, and records
  rollback state in
  `Tools\InputDiagnostics\backups\touchpad-power-20260606-152612.json`.
  Evidence report:
  `Reports\workstation-audit-20260606-124859\touchpad-power-fix-20260606.md`.

Validation:
- `Reports\workstation-audit-20260606-124859\Run-RepoFixValidation.ps1` passed.
- Passing gates: `pester-invoke-rustbuild`, `pester-cargotools`,
  `pester-functiongemma-config`, `pester-cuda-initializer`,
  `pester-inputdiagnostics`, `nvidia-health-readonly`, and
  `rust-functiongemma-runtime-check`.
- `git diff --check` passed.

Remaining gated items:
- Full FunctionGemma `model` feature still needs a separate GPU build window.
  Previous validation showed `candle-kernels` / `candle-flash-attn` failures
  in NVCC/MSVC environment setup and Windows command-line length handling.
- NVIDIA driver remediation remains intentionally uninstalled/unmodified. The
  checker is now the validation gate; any install should first verify a
  candidate INF covers both `VEN_10DE&DEV_28B8` and `VEN_10DE&DEV_2D04`.
- Input symptoms still require live reproduction evidence from
  `Test-KeyInput.ps1` and/or `Watch-InputGlitch.ps1`; current logs show
  repeated WUDFRd warnings for `ACPI\VEN_ELAS&DEV_B41A` while the actual
  touchpad path remains Sensel `SNSL002D`. As of the touchpad power pass, the
  direct Sensel/I2C power-down risk has been remediated and future recurrence
  should be measured with `Watch-InputGlitch.ps1 -Mode Report -SinceFix
  2026-06-06`.

## Proposed Todo List

1. Complete NVIDIA driver remediation.
   - Create a restore point and export current display drivers.
   - Prefer Lenovo Vantage or Windows Update Optional Updates first.
   - If still broken, test a known Lenovo/NVIDIA package or rollback package.
   - Validate with `pnputil /enum-devices /problem`, Device Manager state, and a
     focused Kernel-PnP/DriverFrameworks event query.

2. Decide iCloud state.
   - If iCloud should be active, reinstall or repair iCloud and validate provider
     process plus sync diagnostics.
   - If iCloud should not be active, archive or rename
     `C:\Users\david\iCloudDrive` only after confirming the data is preserved.

3. Keep the post-fix event window under observation.
   - Re-run `Check-PostFixEvents.ps1 -Minutes 60` after the old HTTP Event
     records age out.
   - If `WebManagement` is needed later, re-enable it deliberately and monitor
     port `50443` binding churn.

4. Unexpected shutdown and storage/VHD correlation.
   - Review Kernel-Power 41/EventLog 6008 details.
   - Correlate FilterManager Event 3 and disk 158 with attached VHD disk numbers.
   - Keep `Tools\Collect-BootDiagnostics.ps1 -PostRebootVerify` as the
     validation gate after the next reboot.

5. Docker/WSL cleanup plan.
   - Do not prune blindly.
   - Inventory active containers and volumes.
   - Review unused images before pruning the reclaimable image set.
   - Consider WSL/Docker VHD compaction only after Docker is stopped and a backup/rollback path is known.

6. Profile regression tests.
   - Add a small script that verifies:
     `codex --version`, no PATH-visible `*.bak*`, `.ps1` association, and no `OpenWith.exe` spawned by a profiled `pwsh` smoke.
   - Keep this as a preflight for future workstation diagnostics.
