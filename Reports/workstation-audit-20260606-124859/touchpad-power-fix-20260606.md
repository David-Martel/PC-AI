# Touchpad Power-Management Fix - 2026-06-06

## Change Applied

Added and ran:

```powershell
pwsh -NoLogo -NoProfile -File .\Tools\InputDiagnostics\Repair-TouchpadPowerManagement.ps1
```

The script targets only the Sensel touchpad path (`SNSL002D`) and the Intel
Serial IO I2C controller (`VEN_8086&DEV_7E78`). It backs up WMI/registry state,
sets `MSPower_DeviceEnable.Enable=false`, writes
`EnhancedPowerManagementEnabled=0`, and leaves
`SelectiveSuspendEnabled` / `AllowIdleIrpInD3` absent unless those values
already exist.

No device restart was requested during the apply.

## Rollback

Rollback backup:

```text
Tools\InputDiagnostics\backups\touchpad-power-20260606-152612.json
```

Rollback command:

```powershell
pwsh -NoLogo -NoProfile -File .\Tools\InputDiagnostics\Repair-TouchpadPowerManagement.ps1 -Revert -BackupFile .\Tools\InputDiagnostics\backups\touchpad-power-20260606-152612.json
```

The backup captured both WMI entries as `PriorEnable=true`:

- `ACPI\SNSL002D\4&39979b3e&0_0`
- `PCI\VEN_8086&DEV_7E78&SUBSYS_223417AA&REV_20\3&11583659&1&A8_0`

## Validation

Post-fix WMI state:

```json
[
  {
    "InstanceName": "PCI\\VEN_8086&DEV_7E78&SUBSYS_223417AA&REV_20\\3&11583659&1&A8_0",
    "Enable": false
  },
  {
    "InstanceName": "ACPI\\SNSL002D\\4&39979b3e&0_0",
    "Enable": false
  }
]
```

Post-fix snapshot:

```text
Reports\input-glitch-watch\snapshots\snap-20260606-152639.json
```

Snapshot result:

- Touchpad `HID\SNSL002D&COL01\5&14B88203&0&0000`: `Status=OK`, `Problem=0`
- Keyboard `ACPI\LEN0071\4&76D3D92&0`: `Status=OK`, `Problem=0`
- `PowerDownEnabled`: both `SNSL002D` and `7E78` now `CanPowerDown=false`
- Recent Modern Standby events: none
- Recent input errors: none

Focused structural validation:

```powershell
pwsh -NoLogo -NoProfile -File .\Reports\workstation-audit-20260606-124859\Run-InputDiagnosticsValidation.ps1
```

Result: `65` passed, `0` failed.

## A2000 / NVIDIA Relationship

The internal NVIDIA RTX 2000 Ada GPU remains a real active machine issue, but
it is not the direct touchpad device path.

Current NVIDIA checker result:

- Internal `NVIDIA RTX 2000 Ada Generation Laptop GPU`: `Status=Error`,
  `Problem=31`, `CM_PROB_FAILED_ADD`
- eGPU `NVIDIA GeForce RTX 5060 Ti`: `Status=OK`
- Driver split remains: `32.0.15.9659` versus `32.0.15.9186`
- `nvidia-smi` reports `591.86`, which does not match all display adapters

Interpretation:

- Direct touchpad evidence points to Sensel/I2C power management, now fixed.
- The A2000 failure can still indirectly affect UI smoothness through display
  driver, DWM, Thunderbolt/eGPU, and power-transition churn.
- NVIDIA remediation should stay separate: verify a candidate INF supports both
  `VEN_10DE&DEV_28B8` and `VEN_10DE&DEV_2D04`, then rerun
  `Tools\InputDiagnostics\Test-NvidiaDualGpuDriverHealth.ps1 -FailOnIssue`.

## Next Measurement

After a reboot or sign-out/in, keep the same symptom ledger:

```powershell
pwsh -NoLogo -NoProfile -File .\Tools\InputDiagnostics\Watch-InputGlitch.ps1 -Mode Snapshot -Symptom touchpad -Note "PostTouchpadPowerFix recurrence"
pwsh -NoLogo -NoProfile -File .\Tools\InputDiagnostics\Watch-InputGlitch.ps1 -Mode Report -SinceFix 2026-06-06
```
