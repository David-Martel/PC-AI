# Haptic Touchpad Repro Bundle

- Generated: 2026-06-06T16:32:24.2956694-04:00
- Lookback hours: 1
- Bundle path: C:\codedev\PC_AI\Reports\haptic-touchpad\bundle-20260606-163147

Primary files:

- firmware\sensel-firmware-state.json
- input-devices.json
- signed-drivers.json
- precision-touchpad-settings.json
- device-powerdown-state.json
- events-System.json
- events-Microsoft-Windows-DriverFrameworks-UserMode_Operational.json
- nvidia-dual-gpu.json when available

Next capture:

```powershell
pwsh -File .\Tools\InputDiagnostics\Start-HapticTouchpadTrace.ps1 -DurationSeconds 90 -IncludeHidi2cWpp -Note "TrackPoint palm haptic repro"
pwsh -File .\Tools\InputDiagnostics\Watch-HapticTouchpadInput.ps1 -Seconds 90 -Note "same repro window"
```
