# Haptic Touchpad Debugging Strategy - 2026-06-06

## Goal

Identify why the ThinkPad P1 Gen 7 Sensel haptic touchpad still shows pointer
or finger-press stickiness after the generic input and power-management fixes.
The next phase should gather discriminating evidence rather than applying more
settings changes.

## Current Working Hypotheses

1. **Sensel firmware / force calibration issue**
   - The symptom is haptic-specific: pointer/press stickiness rather than a
     keyboard scancode loss.
   - Lenovo Vantage has staged `n48gb01w`, the Sensel Forcepad firmware package.
   - Package metadata says version `1.4.2.0`; `SenselTrackpad.inf` describes
     `Sensel Curie Trackpad Firmware 1.4.2` and firmware DWORD `0x01040002`.
   - Vantage history says `AlreadyInstalled`, but Windows firmware state needs
     reconciliation before trusting that status.

2. **HID-over-I2C transport stall**
   - The touchpad rides through `ACPI\SNSL002D` -> `hidi2c` -> `mshidkmdf`.
   - The parent controller is Intel Serial IO I2C `7E78`, driver
     `30.100.2527.40`.
   - Microsoft's HID-over-I2C troubleshooting path is ETW/WPP tracing, looking
     for interrupt/read/complete sequences around the repro.

3. **HIDCLASS / Precision Touchpad feature-report issue**
   - The touchpad collection is `HID\SNSL002D&COL02`.
   - Haptic settings are enabled: `FeedbackEnabled`, `FeedbackIntensity=50`,
     `ClickForceSensitivity=50`.
   - Haptic intensity and click-force sensitivity are host-visible feature-report
     settings, so they should be tested only after a baseline trace exists.

4. **TrackPoint + integrated haptic button interaction**
   - The TrackPoint path is separate: `ACPI\LEN032A`, ELAN `EPD`, driver
     `31.21.51.2`, service `EPDService`.
   - The failure pattern reported in external reviews is TrackPoint movement
     plus palm/hand contact on the haptic pad.

5. **External/system contributors**
   - The internal NVIDIA RTX 2000 Ada remains Code 31 and can still affect DWM
     or display responsiveness indirectly.
   - WUDFHost, Lenovo Vantage, OneDrive, and sync/provider processes should be
     sampled during repro, but not treated as the direct cause without ETW
     correlation.

6. **Nearby UMDF sensor timeout**
   - The 2026-06-06 bundle shows repeated `WUDFHostProblem2` / `HostTimeout`
     reports and Kernel-PnP `WUDFRd failed to load` warnings.
   - One repeated warning target is the nearby Elliptic human-presence device
     `ACPI\VEN_ELAS&DEV_B41A&SUBSYS_17AA2234&REV_0003`.
   - This device is not the Sensel touchpad, but it shares the input/sensor
     reliability surface and still had `MSPower_DeviceEnable.Enable=true` while
     the Sensel touchpad and Intel `7E78` controller had already been hardened.

## Evidence Sources On This Machine

- Lenovo Vantage package cache:
  `C:\ProgramData\Lenovo\Vantage\AddinData\LenovoSystemUpdateAddin\session\Repository\n48gb01w\`
- Vantage package files:
  `SenselTrackpad.inf`, `SenselTrackpad.Cap`, `n48gb01w_2_.xml`
- Vantage state:
  `available_updates.json`, `aggregated_device_updates.json`,
  `update_history.txt`, `ProblematicUpdates.xml`
- Windows firmware resource:
  `UEFI\RES_{e3074a9c-a8f2-4ec6-8b7a-4124b1b3c134}`
- Touchpad device path:
  `ACPI\SNSL002D\4&39979B3E&0`
- Sensel HID collections:
  `HID\SNSL002D&COL01`, `COL02`, `COL03`, `COL04`
- TrackPoint path:
  `ACPI\LEN032A\4&76D3D92&0`, ELAN `EPD`
- Intel I2C controller:
  `PCI\VEN_8086&DEV_7E78&SUBSYS_223417AA&REV_20\3&11583659&1&A8`
- Elliptic human-presence sensor:
  `ACPI\VEN_ELAS&DEV_B41A&SUBSYS_17AA2234&REV_0003\2&DABA3FF&1`
- Event/trace providers:
  `Microsoft-Windows-SPB-HIDI2C`,
  `Microsoft-Windows-Input-HIDCLASS`,
  `Intel-iaLPSS2-I2C`,
  `Intel-iaLPSS-I2C`,
  `UMDF - WDF Core`,
  HIDI2C WPP `{E742C27D-29B1-4E4B-94EE-074D3AD72836}`

## New Tools

- `Tools\InputDiagnostics\Get-SenselFirmwareState.ps1`
  - Read-only firmware/Vantage/PnP reconciliation.
- `Tools\InputDiagnostics\Start-HapticTouchpadTrace.ps1`
  - Bounded ETW/logman capture for HIDI2C, HIDCLASS, Intel I2C, WDF, and
    optional HIDI2C WPP.
- `Tools\InputDiagnostics\Watch-HapticTouchpadInput.ps1`
  - Passive low-level pointer/button monitor for stuck press or missing button-up
    evidence.
- `Tools\InputDiagnostics\Export-HapticTouchpadReproBundle.ps1`
  - Read-only bundle of firmware, PnP, driver, settings, services, processes,
    events, NVIDIA state, and symptom ledger.
- `Tools\InputDiagnostics\Repair-TouchpadPowerManagement.ps1`
  - Reversible Sensel/I2C power-down hardening. New opt-in
    `-IncludeHumanPresenceSensor` also targets `VEN_ELAS&DEV_B41A` when WUDF
    timeout evidence implicates that nearby sensor path.

## Live Pass Results - 2026-06-06

- Repro bundle:
  `Reports\haptic-touchpad\bundle-20260606-161750`
- ETW trace:
  `Reports\haptic-touchpad\trace-20260606-161752`
- Converted ETW CSV:
  `haptic-touchpad.csv` and `hidi2c-wpp.csv` in the trace directory.
- Touchpad, Sensel HID collections, TrackPoint, and Intel `7E78`/`7E50` I2C
  controllers all reported `Status=OK`, `Problem=0` in the trace snapshot.
- Precision Touchpad haptics were enabled:
  `FeedbackEnabled=0xFFFFFFFF`, `FeedbackIntensity=50`,
  `ClickForceSensitivity=50`.
- `MSPower_DeviceEnable` state after the earlier fix:
  Sensel `SNSL002D=false`, Intel `7E78=false`, Elliptic
  `VEN_ELAS&DEV_B41A=true`.
- Application/System event snapshots contain repeated `WUDFHostProblem2`
  `HostTimeout` reports and repeated Kernel-PnP `WUDFRd failed to load` warnings
  for several UMDF devices, including `ACPI\VEN_ELAS&DEV_B41A`.
- NVIDIA is still unhealthy but is a secondary/system contributor for this
  symptom: internal RTX 2000 Ada reports Code 31 and a driver-version split,
  while the eGPU reports OK.

## Immediate Fix Candidate

Apply the reversible optional ELAS power-down hardening, then validate with the
same bundle/trace/snapshot path:

```powershell
pwsh -NoLogo -NoProfile -File .\Tools\InputDiagnostics\Repair-TouchpadPowerManagement.ps1 `
  -IncludeHumanPresenceSensor
```

Validation target: `device-powerdown-state.json` should show
`VEN_ELAS&DEV_B41A` with `Enable=false`; event snapshots after the change should
stop accumulating new `WUDFHostProblem2` / `WUDFRd` warnings for that device. If
presence-detection behavior regresses, revert using the backup JSON written by
the script.

## Execution Plan

1. **Static bundle before repro**

   ```powershell
   pwsh -NoLogo -NoProfile -File .\Tools\InputDiagnostics\Export-HapticTouchpadReproBundle.ps1 -SinceHours 72
   pwsh -NoLogo -NoProfile -File .\Tools\InputDiagnostics\Get-SenselFirmwareState.ps1 -AsJson
   ```

2. **Baseline live repro**

   Run these at the same time. Keep external mice idle or disconnected if
   practical so pointer events mostly represent the internal touchpad.

   ```powershell
   # Elevated terminal
   pwsh -NoLogo -NoProfile -File .\Tools\InputDiagnostics\Start-HapticTouchpadTrace.ps1 `
     -DurationSeconds 90 `
     -IncludeHidi2cWpp `
     -Note "baseline TrackPoint palm haptic repro"

   # Interactive user terminal
   pwsh -NoLogo -NoProfile -File .\Tools\InputDiagnostics\Watch-HapticTouchpadInput.ps1 `
     -Seconds 90 `
     -Note "baseline TrackPoint palm haptic repro"
   ```

3. **Immediate symptom marker**

   ```powershell
   pwsh -NoLogo -NoProfile -File .\Tools\InputDiagnostics\Watch-InputGlitch.ps1 `
     -Mode Snapshot `
     -Symptom touchpad `
     -Note "baseline haptic repro completed"
   ```

4. **Interpretation**

   - If HIDI2C interrupts arrive but read completions are missing, investigate
     Intel I2C controller / firmware / driver timing.
   - If HIDI2C completes normally but HIDCLASS/Precision Touchpad output stalls,
     investigate HID report parsing, feature reports, haptic settings, and
     Sensel firmware.
   - If pointer button-down occurs without a matching button-up in
     `Watch-HapticTouchpadInput.ps1`, prioritize force/click threshold and
     haptic firmware.
   - If pointer events continue but DWM/explorer lags, correlate with NVIDIA,
     WUDFHost, Lenovo/Vantage, and system load instead.

5. **Only after baseline evidence: haptic A/B**

   Change one Windows Touchpad feedback setting at a time through the UI:

   - Feedback off versus on.
   - Feedback intensity lower than 50 versus current 50.
   - Click force sensitivity lower/higher than current 50.

   For each setting, repeat steps 2 and 3. Do not mix multiple settings in a
   single run.

## Definition Of Done

- At least one repro bundle and one ETW trace exist for a window where the
  symptom was observed.
- The firmware state tool has reconciled Vantage and Windows firmware PnP state.
- The raw pointer monitor shows whether the issue is missing button-up,
  long-held press, movement stall, or neither.
- The next action is selected from evidence:
  firmware reinstall/escalation, Intel I2C/driver path, haptic setting A/B,
  TrackPoint/ELAN path, or external UI/DWM/system contributor.
