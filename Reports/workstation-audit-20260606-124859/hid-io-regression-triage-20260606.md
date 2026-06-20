# HID / IO Regression Triage - 2026-06-06

## Scope

Investigate non-GPU contributors to haptic touchpad stickiness, Shift-key
processing, Claude / Windows Terminal input processing, power maintenance, and
Windows event/logging churn.

## Immediate rollback

- Reverted the optional ELAS human-presence power-management change from
  `Tools\InputDiagnostics\backups\touchpad-power-20260606-163129.json`.
- Post-revert state:
  - `ACPI\SNSL002D...` power-down disabled.
  - `PCI\VEN_8086&DEV_7E78...` power-down disabled.
  - `ACPI\VEN_ELAS&DEV_B41A...` restored to power-down enabled.
  - ELAS registry values created by the optional pass were removed.
- A sign out/in or reboot can still be required if a driver cached the old
  power state.

## Current evidence

- `Test-KeyInput.ps1` captured left Shift down/up events. Shift reaches
  Windows, so the current failure mode is above keyboard hardware/EC in the
  focused app, hook, shell, ConPTY, or latency path.
- `Watch-HapticTouchpadInput.ps1` live capture:
  `Reports\haptic-touchpad\pointer-20260606-165443\pointer-input.json`
  - 1226 pointer events.
  - 9 left-button down events and 9 matching left-button up events.
  - 0 stuck-button warnings.
  - Max event gap: 8886 ms.
- ETW capture:
  `Reports\haptic-touchpad\trace-20260606-165443`
  - `haptic-touchpad_000001.etl`: 21 events.
  - `hidi2c-wpp_000001.etl`: 2590 HIDI2C WPP events.
  - Device snapshot showed touchpad, I2C controller, TrackPoint, ELAS, and
    ETD HSA devices all present and `OK`.
- No active internal HID/touchpad/keyboard PnP problem was found after
  filtering stale/disconnected Problem 45 devices.

## Strong non-GPU findings

1. Intel IPF / Dynamic Tuning UMDF timeouts
   - Windows Error Reporting recorded paired `WUDFHostProblem2` `HostTimeout`
     events for `ipfumdf.dll` and `ipf_umdf2.dll` at 2026-06-06 16:03.
   - WDF dumps exist at `C:\ProgramData\Microsoft\WDF\DriverManager_1984.mdmp`
     and `DriverManager_1984Heap.hdmp`.
   - This is a real driver-framework failure near the Lenovo power/thermal
     stack, not a GPU finding.

2. Driver-install and audio hangs in the same burst
   - WER recorded `drvinst.exe` `AppHangB1`.
   - WER recorded `svchost.exe_Audiosrv` / `audiodg.exe` `AppHangXProcB1`.
   - This points to broader device-framework or system latency trouble around
     the same period.

3. NETGEAR / MediaTek USB Wi-Fi driver event spam
   - System log provider `mtkwecxu` recorded 233 `Scan Abort` events in the
     last 6 hours, roughly once per minute.
   - Device: `NETGEAR A9000 Wi-Fi 7 Wireless LAN Card`.
   - Driver: MediaTek `5.3.0.3230`, date 2025-08-09.
   - Adapter is disconnected but present, while built-in Intel BE200 and wired
     Realtek USB Ethernet also exist.

4. Power maintenance issues
   - `powercfg /energy` generated 30 errors and 10 warnings.
   - USB selective suspend is globally disabled.
   - PCIe ASPM is disabled on battery and plugged in.
   - NETGEAR A9000, Intel Bluetooth, and Realtek USB 2.5GbE did not enter USB
     selective suspend.
   - `codex.exe` held a system availability request during the active turn.
   - HP Print Scan Doctor has a wake timer for `Printer Health Monitor`.

5. Terminal / Claude shell pressure
   - Process census showed high fan-out: many `conhost.exe`, `cmd.exe`,
     `node.exe`, and `pwsh.exe` processes plus active Claude and Windows
     Terminal sessions.
   - PowerShellCore produced many events in a short capture window, including
     large script-block payloads from profile / command-dispatch activity.
   - ScriptBlockLogging policy keys were not present, so this appears to be
     runtime/tooling-driven logging rather than explicit local policy.

6. Lenovo haptic/TrackPoint control layer remains relevant
   - Local topology includes Sensel `SNSL002D`, ELAN TrackPoint, ETD HSA, and
     Lenovo power/thermal services.
   - Lenovo documentation describes haptic touchpad button-area behavior,
     TrackPoint Quick Menu gestures, Quick Clean, and Intelligent Cooling as
     active platform features.

## Fixes already applied in this pass

- Reverted the optional ELAS power-management hardening.
- Patched `Tools\InputDiagnostics\Start-HapticTouchpadTrace.ps1` so manifests
  include actual emitted ETL files (`EtlFiles`, `WppEtlFiles`) when `logman`
  creates suffixed files such as `_000001.etl`.
- Added a static Pester assertion for that manifest behavior.

Validation:

```powershell
pwsh -NoLogo -NoProfile -Command "Invoke-Pester -Path .\Tests\InputDiagnostics\InputDiagnostics.Tests.ps1 -Output Detailed"
git diff --check -- Tools/InputDiagnostics/Start-HapticTouchpadTrace.ps1 Tests/InputDiagnostics/InputDiagnostics.Tests.ps1
```

Result: 87/87 Pester tests passed; `git diff --check` passed.

## Recommended next direct actions

1. Inspect WDF/IPF failure artifacts.
   - Parse or open WDF and WER dumps with WinDbg/WPA where available.
   - Cross-check Lenovo Vantage / Lenovo Commercial Vantage update state for
     Intel IPF, Dynamic Tuning, BIOS/UEFI, Lenovo Intelligent Thermal Solution,
     Sensel firmware, and ELAN TrackPoint.
   - Validate by monitoring that no new `WUDFHostProblem2` events appear after
     driver/firmware repair.

2. Clean up NETGEAR A9000 if it is not intentionally in use.
   - Candidate action: disable or remove stale NETGEAR adapter instances, or
     unplug/disable the A9000 while Ethernet is active.
   - Validation: System log should stop receiving `mtkwecxu` event 1003
     `Scan Abort` once per minute.
   - Requires user approval because it changes network-device availability.

3. Reduce power-maintenance noise.
   - Candidate action: disable HP Print Scan Doctor wake timer if not needed.
   - Candidate action: review whether global USB selective suspend and PCIe
     ASPM were intentionally disabled for eGPU / dock stability before changing
     them.
   - Validation: rerun `powercfg /energy /duration 30` and compare error count
     and specific USB/network offenders.

4. Isolate Terminal / Claude command-processing overhead.
   - Capture a no-profile baseline:
     `pwsh -NoLogo -NoProfile -Command "$PSVersionTable.PSVersion"`
   - Compare against profile-loaded shell startup and command dispatch.
   - Review active Claude / Codex / MCP process trees and collapse stale
     duplicate sessions only after confirming ownership.
   - Validation: lower `conhost`/`cmd`/`node`/`pwsh` fan-out and reduced
     PowerShellCore event volume during a 5-minute capture.

5. Reproduce the haptic failure with synchronized evidence.
   - Run:
     `Start-HapticTouchpadTrace.ps1 -DurationSeconds 90 -IncludeHidi2cWpp`
     and `Watch-HapticTouchpadInput.ps1 -Seconds 90` during an actual symptom.
   - Validation target: determine whether the live repro loses button-up,
     stalls HIDI2C, or continues OS input events while the focused app lags.
