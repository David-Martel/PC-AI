# Input-Stack Diagnostics & Remediation (DTM-P1GEN7)

Reusable toolkit for the **Shift-key / trackpad / fingerprint freezing & glitching**
investigation on this machine (Lenovo **ThinkPad P1 Gen 7**, Win 11 26200, 64 GB RAM).
Created 2026-05-30; diagnostic interpretation corrected 2026-09-12.

## Current keyboard investigation: unresolved

The user confirms failures of other keys or the whole internal keyboard while an
external USB keyboard worked normally during a recent failure. Root cause remains
unestablished. Prepare fixed-input trials across the keyboard; modifiers-only captures
cannot characterize this broader symptom. Prior declarations
of software exoneration, hardware failure and a proven typing-timing explanation were
not justified. Short successful captures cannot exclude intermittent faults; a healthy
PnP status or empty error log does not validate key delivery. See the
[September investigation](../../Reports/keyboard-investigation-20260912/README.md).

Prior workload, power, display and startup observations are useful hypotheses. They
require symptom-correlated measurements and controlled interventions before attribution.
The working USB control prioritizes the internal device/path, while device-specific
software effects remain possible; instrument this user-reported contrast next.

### Device topology
Keyboard = `ACPI\LEN0071` (PS/2), TrackPoint = `ACPI\LEN032A`, touchpad = Sensel
I2C-HID `SNSL002D`, fingerprint = Synaptics USB `VID_06CB`. Different buses help
localize tests; they do not prove independent causes or exclude shared system effects.

## Scripts

| Script | Purpose | Elevation |
|--------|---------|-----------|
| `Invoke-InputStackDiagnostics.ps1` | **Read-only** collector. Reproduces the whole investigation → timestamped `.txt`/`.json` in `PC_AI\Logs\input-diagnostics`. Run whenever freezes recur. | No (more complete elevated) |
| `Repair-InputStackQuickWins.ps1` | Applies the safe, reversible fixes (backup + `-Revert`). | Yes (HKCU part works without) |
| `Start-LoadCapture.ps1` | Native `powercfg` thermal/power/latency capture during heavy ML load; drives HWiNFO/LatencyMon if installed. | Yes |
| `Test-KeyInput.ps1` | Passive WH_KEYBOARD_LL monitor for Shift-key press/release events. Device-AGNOSTIC (sees the merged stream). Use interactively while reproducing the Shift symptom. | No |
| `Trace-ShiftKeySource.ps1` | Device-attributed Raw Input observations from an interactive desktop. Runs unfocused through INPUTSINK; verifies neither physical intent nor application output. Default captures modifiers; `-AllKeys` can reveal typed content, so use controlled test input. | No |
| `Measure-ShiftOrdering.ps1` | Compatibility entrypoint to the conservative Shift-state analyzer; the former timing/failure heuristic is retired. | No |
| `Analyze-ShiftTrace.ps1` | Per-device/side Shift state and repeat/boundary aggregates. Does not reconstruct typed content or infer hardware loss from make/break counts. | No |
| `Get-KeyboardDiagnosticSnapshot.ps1` | Repeatable sanitized snapshot with explicit probe failures, current accessibility/software/device/driver evidence; no key capture or device changes. | No |
| `Watch-InputGlitch.ps1` | Read-only symptom ledger and snapshot tool for Shift/touchpad co-freeze events. Use before/after driver or firmware changes. | No |
| `Test-NvidiaDualGpuDriverHealth.ps1` | Read-only NVIDIA internal/eGPU health check; reports driver-version split, Code 31, `nvidia-smi`, local NVIDIA App/update artifacts, and can fail automation with `-FailOnIssue`. | No |
| `Repair-TouchpadPowerManagement.ps1` | Applies/reverts the targeted Sensel `SNSL002D` + Intel `7E78` I2C power-down fix when `Watch-InputGlitch.ps1` shows `CanPowerDown=true`; opt-in `-IncludeHumanPresenceSensor` also hardens the nearby Elliptic `VEN_ELAS&DEV_B41A` sensor when WUDF timeouts implicate it. | Yes |
| `Get-SenselFirmwareState.ps1` | Read-only reconciliation of Lenovo Vantage `n48gb01w`, Sensel firmware INF/CAP metadata, Windows firmware PnP state, and `hidcfu` logs. | No |
| `Start-HapticTouchpadTrace.ps1` | Bounded ETW/logman capture for HIDI2C, HIDCLASS, Intel I2C, UMDF/WDF, and optional HIDI2C WPP during a repro. | Yes |
| `Watch-HapticTouchpadInput.ps1` | Passive pointer/button monitor for stuck press, missing button-up, and movement-stall correlation during haptic touchpad repros. | No |
| `Export-HapticTouchpadReproBundle.ps1` | Read-only evidence bundle: firmware, PnP, drivers, Precision Touchpad settings, services/processes, events, NVIDIA state, symptom ledger. | No |

### Usage
```powershell
# 1. Baseline (anytime):
pwsh -File .\Invoke-InputStackDiagnostics.ps1 -AsJson

# 2. Apply quick-wins (elevated for full effect):
Start-Process pwsh -Verb RunAs -ArgumentList '-File','C:\codedev\PC_AI\Tools\InputDiagnostics\Repair-InputStackQuickWins.ps1'
#   ...or no-admin accessibility fix only:
pwsh -File .\Repair-InputStackQuickWins.ps1 -SkipElevatedChanges
#   ...undo:
pwsh -File .\Repair-InputStackQuickWins.ps1 -Revert

# 3. Capture evidence during a heavy CUDA/ML session (elevated):
pwsh -File .\Start-LoadCapture.ps1 -EnergyDurationSec 90 -LaunchHwinfo

# 4. Check fragile internal NVIDIA + eGPU driver state:
pwsh -File .\Test-NvidiaDualGpuDriverHealth.ps1 -AsJson
pwsh -File .\Test-NvidiaDualGpuDriverHealth.ps1 -FailOnIssue

# 5. Prove whether Shift reaches Windows when the symptom is active:
pwsh -File .\Test-KeyInput.ps1 -Seconds 20
#   ...device-aware (which keyboard?) — run in background, reproduce in your real app:
pwsh -File .\Trace-ShiftKeySource.ps1 -Seconds 45        # -AllKeys to log every key

# 6. Keep a before/after symptom ledger for touchpad + Shift fixes:
pwsh -File .\Watch-InputGlitch.ps1 -Mode Snapshot -Symptom none
pwsh -File .\Watch-InputGlitch.ps1 -Mode Report -SinceFix 2026-06-06

# 7. Apply/revert the targeted touchpad/I2C power-management fix:
pwsh -File .\Repair-TouchpadPowerManagement.ps1 -WhatIf
pwsh -File .\Repair-TouchpadPowerManagement.ps1
pwsh -File .\Repair-TouchpadPowerManagement.ps1 -IncludeHumanPresenceSensor
pwsh -File .\Repair-TouchpadPowerManagement.ps1 -Revert

# 8. Haptic/Sensel diagnostic workflow:
pwsh -File .\Get-SenselFirmwareState.ps1 -AsJson
pwsh -File .\Export-HapticTouchpadReproBundle.ps1 -SinceHours 72

# 9. Live repro capture; run both at the same time in two elevated/non-elevated terminals:
pwsh -File .\Start-HapticTouchpadTrace.ps1 -DurationSeconds 90 -IncludeHidi2cWpp -Note "TrackPoint palm haptic repro"
pwsh -File .\Watch-HapticTouchpadInput.ps1 -Seconds 90 -Note "same repro window"
```

## What `Repair-InputStackQuickWins.ps1` changes (all reversible)
1. **Accessibility hotkeys** (HKCU, no admin) — clears `HOTKEYACTIVE 0x04`: StickyKeys `510→506`, FilterKeys `126→122`, ToggleKeys `62→58`. **Already applied in the 2026-05-30 session.**
2. **USB selective suspend OFF** (AC+DC) — `powercfg ... 2a737441-... 48e6b7a6-... 0`. Fixes fingerprint-wake glitches.
3. **Crash dump → Automatic (0x7)** — so the *next* hard freeze is captured (current was `0x3`).

## What `Repair-TouchpadPowerManagement.ps1` changes (all reversible)
1. **MSPower_DeviceEnable OFF** for `ACPI\SNSL002D\...\_0` and the Intel
   Serial IO I2C controller `PCI\VEN_8086&DEV_7E78...\_0`, so Windows is no
   longer allowed to power down the touchpad path.
2. **Enhanced Power Management OFF** on the ACPI Sensel device, Sensel HID
   collections, and the Intel `7E78` I2C controller.
3. **Conditional suspend values OFF** only when already present:
   `SelectiveSuspendEnabled` and `AllowIdleIrpInD3`.
4. **Optional Elliptic human-presence sensor hardening** with
   `-IncludeHumanPresenceSensor`, targeting `ACPI\VEN_ELAS&DEV_B41A`. Use this
   only when logs show repeated `WUDFHostProblem2` / `WUDFRd` warnings for that
   device. It may affect presence-detection features such as walk-away lock or
   wake-on-approach.

The 2026-06-06 apply wrote rollback state to
`Tools\InputDiagnostics\backups\touchpad-power-20260606-152612.json` and the
post-fix validation snapshot is
`Reports\input-glitch-watch\snapshots\snap-20260606-152639.json`.

## Haptic/Sensel debugging strategy

The current remaining symptom should be treated as a haptic force/firmware
interaction until disproven. The P1 Gen 7 Sensel pad exposes separate Windows
paths for:

- Sensel HID-over-I2C transport: `ACPI\SNSL002D` through `hidi2c` and
  `mshidkmdf`.
- Windows Precision Touchpad collection: `HID\SNSL002D&COL02`.
- Vendor-defined Sensel collection: `HID\SNSL002D&COL04`.
- TrackPoint / integrated button path: ELAN `EPD` driver and service.
- Intel I2C controller: `PCI\VEN_8086&DEV_7E78`.
- Nearby Elliptic human-presence sensor: `ACPI\VEN_ELAS&DEV_B41A`; not the
  touchpad itself, but current logs show repeated UMDF timeout/load warnings on
  this path while its power-down permission remains enabled.

Do not apply more settings changes until one baseline repro has:

1. `Export-HapticTouchpadReproBundle.ps1` output.
2. `Start-HapticTouchpadTrace.ps1` ETL from the repro window.
3. `Watch-HapticTouchpadInput.ps1` pointer/button output from the same window.
4. A `Watch-InputGlitch.ps1 -Mode Snapshot -Symptom touchpad` marker.

Only after that baseline should haptic settings be A/B tested by temporarily
changing Windows Settings > Touchpad feedback, because `FeedbackEnabled`,
`FeedbackIntensity`, and `ClickForceSensitivity` are part of the feature-report
path and can mask or expose firmware/force-threshold issues.

## Remaining **manual** steps (not auto-applied — device-specific or need your judgement)
- **Cut the login storm:** Task Manager → Startup apps → disable Docker, Ollama, LM Studio, duplicate GoogleDriveFS, Razer, GoPro, MATLAB, SOLIDWORKS Fast Start, Adobe. Launch heavy tools on demand.
- **Per-device power:** Device Manager → Synaptics fingerprint + Bluetooth radio + USB Root Hubs → Power Management tab → uncheck "Allow the computer to turn off this device to save power."
- **Confirm the co-freeze cause:** install + run **LatencyMon** ~10 min under load. Audio-driver / `nvlddmkm` / `Wdf01000` at top of DPC = that's it → update/rollback that driver.
- **Drivers from Lenovo (not Windows Update):** Lenovo Vantage → latest Synaptics touchpad + fingerprint + BIOS; NVIDIA Studio driver clean-install (DDU).
- **Broken services churning at login:** fix/remove `PC_AI-HVSockProxy` (points to missing path) and `vtss` (Intel VTune sampler) auto-start.
- **Windows Hello *face*** is separately broken: IR camera in `Error`, `Windows Camera Frame Server` crashed ×3 — reinstall/repair camera + RealSense drivers.

### Install the optional tools (when online + elevated)
```powershell
winget install REALiX.HWiNFO -e            # sensor logging (thermals) -> CSV into PC_AI\Logs\hwinfo
winget install Resplendence.LatencyMon -e  # DPC/ISR latency (co-freeze confirmation)
```
*(Authoring session had no outbound internet — `winget` returned `0x80072efd`, so these were not installed automatically.)*

## Microsoft documentation references
- USB selective suspend (powercfg GUIDs + rationale): https://learn.microsoft.com/troubleshoot/microsoftteams/teams-rooms-and-devices/usb-selective-suspend-status-unhealthy
- Selective suspend for HID-over-USB (resume latency): https://learn.microsoft.com/windows-hardware/drivers/hid/selective-suspend-for-hid-over-usb-devices
- FILTERKEYS / STICKYKEYS flag bits: https://learn.microsoft.com/windows/win32/api/winuser/ns-winuser-filterkeys
- Accessibility shortcut keys (accidental Shift trigger): https://learn.microsoft.com/windows/win32/dxtecharts/disabling-shortcut-keys-in-games
- Crash dump config / Automatic dump (>32 GB → 0x7): https://learn.microsoft.com/troubleshoot/windows-server/performance/troubleshoot-stop-errors-best-practices-dump-configuration-recommendations
- Memory dump registry values: https://learn.microsoft.com/troubleshoot/windows-server/performance/memory-dump-file-options
- powercfg command-line (`/energy`, `/sleepstudy`): https://learn.microsoft.com/windows-hardware/design/device-experiences/powercfg-command-line-options
