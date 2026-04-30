# System Performance Optimization - 2026-04-30

## Scope

This pass continued post-reboot analysis of Windows event logs, Process Lasso
logs/configuration, sync-provider state, storage/VHD topology, and dynamic
hardware surfaces on the ThinkPad P1 Gen 7.

Primary evidence:

- `Reports\boot-diagnostics\20260430-150703\startup-inventory.md`
- `Reports\boot-diagnostics\20260430-150703\boot-events.json`
- `Reports\drive-performance-sync-risk\20260430-150703\summary.md`
- `Reports\drive-performance-sync-risk\20260430-150703\drive-performance-sync-risk.json`
- `C:\ProgramData\ProcessLasso\logs\processlasso.log`
- `C:\ProgramData\ProcessLasso\config\prolasso.ini`

## Current Findings

### OneDrive remains the highest-impact unresolved issue

OneDrive is still running and responsive, but the latest 240 minute capture
shows continuing WER evidence for `OneDrive.exe` `BEX64` failures in
`ucrtbase.dll` with `c0000409`. The current version path is
`C:\Program Files\Microsoft OneDrive\26.062.0402.0002\`, while WER entries also
reference `26.55.323.4`, suggesting either queued crash reports from an older
build or a stale component path still being exercised.

The latest `SyncDiagnostics.log` values show:

- `files = 148425`
- `folders = 16131`
- `driveChangesToSend = 13348`
- `driveSentChanges = 0`
- `scanState = 3`
- `scanStateStallDetected = 0`
- `syncStallDetected = 0`
- `timeUtc = 2026-04-30T18:52:52Z`

Interpretation: OneDrive has a large local backlog and is not advancing sent
changes. This is consistent with the user's observation that pausing OneDrive
improves touchpad responsiveness: the sync engine, shell overlays, filter
driver, and antivirus/storage stack are doing high-volume file work while input
and composition are latency-sensitive.

Microsoft's current OneDrive guidance emphasizes invalid/path/file-count
restrictions, selective sync, Files On-Demand, installing the latest OneDrive,
and resetting OneDrive when needed. It also states OneDrive does not support
using network or mapped drives as the sync location and does not support
syncing through symbolic links or junction points. The live OneDrive root is
local (`C:\Users\david\OneDrive`), so the better next step is not a registry
notification tweak. It is OneDrive install/update repair plus backlog/file
restriction analysis.

### FilterManager evidence is now mostly historical to the pre-reboot window

The latest 240 minute capture contains a `Microsoft-Windows-FilterManager`
Event ID 3 at `2026-04-30T12:42:53-04:00`:

`Filter Manager failed to attach to volume '\\Device\\Harddisk4\\DR4'. This
volume will be unavailable for filtering until a reboot. The final status was
0xC03A001C.`

The later boot at `2026-04-30 12:50:02 America/New_York` shows normal filter
registration, including `CldFlt`, `PrjFlt`, `bindflt`, `storqosflt`, and
`FsDepends`. This means the prior VHD/filter attach issue was remediated by the
reboot and current VHD startup sequencing, but it remains a useful warning
signal for future VHD mount order and sync-provider startup order.

### Dynamic GPU/display utilities are noisy

Recent event evidence shows:

- `PresentMonService.exe` from Intel Graphics Software crashing in NVIDIA
  `nvml.dll` with `0xc0000005`.
- Repeated `NVIDIA Broadcast.exe` `BEX64` crashes in `ntdll.dll`.
- Repeated `CameraStartupFailureEvent` entries with
  `InitializeMediaCaptureFailed` and `0xc00d3e85`.
- `NVDisplay.Container.exe` crash evidence in the same collection window.
- Process Lasso logs showing frequent short-lived
  `C:\ProgramData\NVIDIA Corporation\nvtopps\rise\disp.exe` launches by
  `nvdisplay.container.exe`.

The system has an internal Intel Arc Pro display path, an NVIDIA RTX 2000 Ada
laptop GPU, and an external NVIDIA GeForce RTX 5060 Ti. The P1 Gen 7 platform
also supports multiple high-bandwidth external displays through Thunderbolt 4,
USB-C, and HDMI. That makes display/USB4/eGPU transitions a plausible
amplifier for user-mode display utilities, especially telemetry, overlays, and
camera effects. It does not justify de-elevating the core display driver or DWM.

### USB/Thunderbolt topology is complex enough to keep observing

Connected and running drivers include USB4 host/device router drivers,
Thunderbolt/USB audio roots, USB Ethernet, USB mass storage, USBPcap,
VirtualBox USB monitor, Parsec virtual USB, Remote Desktop USB hub filters, and
multiple HID devices. The internal touchpad remains a Sensel HID/I2C device on
the Intel Serial IO I2C path:

- `HID\SNSL002D&Col01` HID-compliant mouse
- `HID\SNSL002D&Col02` HID-compliant touch pad
- `HID\SNSL002D&Col04` vendor-defined HID
- `iaLPSS2_I2C_MTL.sys`, `hidi2c.sys`, `mshidkmdf.sys`, `mshidumdf.sys`

The touchpad controller path itself is not showing a clear driver crash in the
current evidence. The risk is contention and stalls around the broader UMDF,
display, sync, storage, and shell surfaces.

### Cloud sync roots on a mounted VHD remain a reliability risk

Current sync roots:

- OneDrive: `C:\Users\david\OneDrive`
- Dropbox: `F:\Auricle Dropbox`
- Proton Drive: `F:\Proton-Drive\My files`
- iCloud Drive: `C:\Users\david\iCloudDrive`

`F:` is `cloud-cache-disk`, a mounted VHD. That layout can be useful for
isolation, but cloud providers starting during VHD/filter readiness can amplify
startup shell-overlay, placeholder, and filter-driver churn. The current VHD
mount sequencing is improved; the next optimization is delaying Dropbox/Proton
until the VHD task has completed cleanly, or moving those sync roots off the VHD
if reliability outweighs isolation.

## Applied Process Lasso Refinements

Updated `Tools\Apply-ProcessLassoUiSyncTuning.ps1` and applied the policy.

Backup:

- `C:\ProgramData\ProcessLasso\config\prolasso.ini.bak-20260430-151250-boot-safety`

Reports:

- `Reports\processlasso-dynamic-hardware-tuning-dryrun.json`
- `Reports\processlasso-dynamic-hardware-tuning-apply.json`
- `Reports\processlasso-dynamic-hardware-tuning-postapply-dryrun.json`
- `Reports\processlasso-system-optimization-validation.json`

Changes:

- Kept input/shell/touchpad path protected and elevated:
  `dwm.exe`, `explorer.exe`, `ctfmon.exe`, `TextInputHost.exe`,
  `SynRpcServer.exe`, Sensel/Synaptics/ELAN patterns, and Lenovo input support.
- Added below-normal CPU and low-I/O defaults for noisy user-mode GPU/peripheral
  helpers:
  `NVIDIA Broadcast.exe`, `NVIDIA Overlay.exe`, `NVIDIA Share.exe`,
  `NVIDIA App.exe`, `PresentMon_x64.exe`, `PresentMonService.exe`,
  `nvfvsdksvc_x64.exe`, `disp.exe`, Dell TechHub/DPM subagents, and HP Print
  Scan Doctor helpers.
- Normalized `chrome.exe`, `brave.exe`, and `zoom.exe` GPU priority defaults
  from `3` to `2`, removing a standing Above Normal GPU boost that could
  compete with composition during dock/eGPU/display churn.

Validation:

- `Tools\Test-ProcessLassoBootSafety.ps1` passed.
- Post-apply dry-run reported no pending changes.
- `Invoke-Pester -Path .\Tests\Boot\BootValidationTools.Tests.ps1` passed:
  14 passed, 0 failed.
- Live WMIC snapshot still shows `dwm.exe` and `SynRpcServer.exe` at base
  priority `10`, while OneDrive/FileSyncHelper and NVIDIA/Dell/HP background
  helpers sit at base priority `6`.
- Process Lasso log shows NVIDIA Overlay received below-normal CPU and low-I/O
  adjustments immediately after the policy update.

## Next Optimization Candidates

1. Repair OneDrive install/update state before more registry tuning.
   - Verify the per-machine updater failure `0x8004EE04`.
   - Inspect queued WER report directories for the current and old OneDrive
     versions.
   - Install the latest OneDrive sync client over the existing install.
   - If crashes continue, plan a controlled OneDrive reset with pre-capture of
     sync diagnostics, file count, invalid names, and WER state.

2. Add a OneDrive backlog/file-restriction scanner.
   - Check invalid names, blocked/reserved names, path length, junctions,
     symlinks, PST/large archive hotspots, and directory fanout.
   - Report candidates before changing anything.

3. Delay VHD-hosted sync providers.
   - Keep OneDrive local and early.
   - Delay Dropbox and Proton Drive until `cloud-cache-disk` VHD health passes.
   - Prefer app-supported startup controls where available; otherwise use a
     scheduled task wrapper that logs loudly and exits nonzero when VHD health
     is not ready.

4. Decide whether NVIDIA Broadcast and FrameView/PresentMon are needed at
   logon.
   - If not needed, disable their startup/service surfaces rather than allowing
     repeated crash loops.
   - If needed, update NVIDIA App/Broadcast/driver as a set and retest camera
     startup events.

5. Treat USBPcap/VirtualBox USB/Parsec/RDP USB filters as observation targets.
   - Do not disable blindly; they may be required for development workflows.
   - If touchpad glitches continue with OneDrive paused, run an A/B capture
     with USBPcap stopped/disabled for one boot and compare WUDFHost,
     Kernel-PnP, USB4, HID, and touchpad behavior.

6. Use Microsoft's USB4 debugging path if dock/eGPU transitions correlate with
   glitches.
   - Capture system event log events for display tunnels and USB4 traces around
     attach/detach/resume.
   - Compare with Process Lasso log lines for display utility process churn.

## Sources

- Microsoft Support: OneDrive restrictions and limitations, including invalid
  names, file-count considerations, selective sync, and unsupported
  network/mapped-drive/symlink locations:
  https://support.microsoft.com/en-us/office/restrictions-and-limitations-in-onedrive-and-sharepoint-64883a5d-228e-48f5-b3d2-eb39e07630fa
- Microsoft Support: OneDrive error code guidance, including installing the
  latest OneDrive and reset guidance:
  https://support.microsoft.com/en-us/office/what-do-the-onedrive-error-codes-mean-f7a68338-e540-4ebf-ad5d-56c5633acded
- Microsoft Learn: USB4 debugging and troubleshooting entry points:
  https://learn.microsoft.com/en-us/windows-hardware/design/component-guidelines/usb4-debugging-and-troubleshooting
- Lenovo PSREF: ThinkPad P1 Gen 7 storage, display, Thunderbolt/USB4, camera,
  and haptic touchpad platform capabilities:
  https://psref.lenovo.com/syspool/Sys/PDF/ThinkPad/ThinkPad_P1_Gen_7/ThinkPad_P1_Gen_7_Spec.pdf
