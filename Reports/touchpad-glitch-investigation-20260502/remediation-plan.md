# Touchpad glitch remediation plan — 2026-05-02 (REVISED v2)

**Host:** dtm-p1gen7 (Lenovo ThinkPad P1 Gen 7, Sensel haptic touchpad on Intel Meteor Lake I2C HID stack)
**Investigator:** Claude (under Administrator PowerShell)
**Authorization scope:** *plan only*. **No mitigation step below has been executed.** Each step requires explicit user approval before running.

> **Revision note (v2):** v1 of this plan named Intel iaLPSS2_I2C_MTL rollback as the primary fix based on a coincidence-of-timing argument (4/30 17:16 "Intel® Software Installer" restore point + WUDFRd events starting 4/30 21:36). That hypothesis was **falsified by direct verification**:
> - The I2C Host Controller's actual `DEVPKEY_Device_DriverDate` is **2025-07-02**, NOT 4/30. The 4/30 Intel installer was not the I2C driver.
> - All 96 WUDFRd 219 events resolve to chronic user-mode driver retry-on-boot patterns for benign devices (INTELAUDIO, LCI/IDDCX, ROOT/WPD, ROOT/SYSTEM, ROOT/WindowsHelloFace, ELAN human-presence sensor, Intel PCI 7D03). WUDFRd is user-mode; a kernel-mode I2C regression wouldn't surface there. The ELAN match is the **Elliptic HPD sensor**, not the touchpad.
>
> Verification artifact: `15-hypothesis-verification.txt` (V1, V2, V5).

## Summary of evidence (Phase 1–2 final)

| Source | Finding |
|---|---|
| `15...txt` V1 | I2C controller bound to oem107.inf 30.100.2527.40 dated **2025-07-02** — not changed in 5/1 window. |
| `15...txt` V5 | What actually changed within last ~7 days: USB 3.20 / USB4 Host Router (4/26 → 26100.8328), input.inf / hidi2c.inf (HID class drivers, → 26100.8328). msmouse.inf / mtconfig.inf still at 26100.1150 (split version state). |
| `02-driver-versions.txt` | Sensel HID collections: COL01 (mouse, msmouse.inf 26100.1150), **COL02 (touchpad, input.inf 26100.8328)**, COL03 (mtconfig.inf 26100.1150), COL04 (vendor, input.inf 26100.8328); ACPI\SNSL002D = hidi2c.inf 26100.8328. The touchpad's own collection got bumped; mouse class did not. |
| `14a-kb-removability.txt` | 5/1 5:56-5:57 AM: ~80 FOD/OnDemand packs all bumped to 26100.8328 + `Package_for_RollupFix~26100.8328.1.31`. This is the cumulative servicing rollup. |
| `14b` (restore points) | RP 64 (4/14), RP 65 (4/21 Scheduled), RP 66 (4/28 Windows Update — likely captures rollup start), RP 67 (4/30 17:16 Intel Software Installer). |
| `09-top-processes.txt` | Process Lasso input-priority enforcement working. Non-PL-managed top consumers: WebManagement, RzDLLService, MsMpEng, Dell.TechHub.Instrumentation, DDPM.Subagent.User. |
| `10-onedrive-io-now.txt` | OneDrive: 8.5 GB read / 459 MB write. GoogleDriveFS: 10.4 GB read / 35 MB write. PL priority demotion does NOT cap IOPS — sync IO can still saturate the storage queue. |
| `11-power-mgmt.txt` | Active scheme = **Balanced** (not High Performance). |
| `15...txt` V4 | UMDF reflector retries are boot-only events (4/30 21:36 = a reboot, 5/1 01:55-01:59 = subsequent reboots). Not glitch-time signal. |

## Hypothesis (re-ranked after verification)

1. **PRIMARY — Cumulative servicing rollup `26100.8328.1.31` regressed the inbox HID class driver chain** (input.inf, hidi2c.inf).
   The rollup landed in pieces 4/26 (USB host) through 5/1 (FOD packs). The Sensel touchpad's own collection (COL02) got bumped to 26100.8328 while sibling collections stayed at 26100.1150. The split-version state did not exist before the rollup. **Probability: medium-high.** **Rollback path: System Restore only** — these are inbox drivers, no vendor INF to swap.

2. **SECONDARY — Storage IO contention from OneDrive + GoogleDriveFS sync** is amplifying perceived glitches.
   ~19 GB cumulative read traffic. PL CPU-priority demotion does not cap IOPS. If the I2C HID poll path is queued behind storage operations under load, glitches will correlate with sync activity. **Independently testable** by capturing during a glitch with sync paused.

3. **TERTIARY — Hardware fatigue / Sensel firmware issue.**
   Sensel community has reported P1 Gen 7 touchpad degradation in some units. If Steps in #1 and #2 do not resolve, this becomes a hardware/firmware question requiring Lenovo Support engagement. **Not actionable from software.**

4. **DEMOTED out of plan — iaLPSS2_I2C_MTL rollback** (verified not changed in regression window).
5. **DEMOTED out of plan — ETDHSA rollback** (no evidence of fault in TrackPoint subsystem).
6. **DEMOTED out of plan — WUDFRd 0xC0000365 as touchpad signal** (verified to be chronic boot-time noise from unrelated devices).

## Phase 3 plan (rollback-first, evidence-driven)

### Pre-flight gates (all read-only/additive — no risk)

- [ ] **Gate A — Manual restore point.** ≥45h since last RP, default 1440-min frequency permits new RP creation.
  ```powershell
  Checkpoint-Computer -Description "PreTouchpadFix-2026-05-02" -RestorePointType MODIFY_SETTINGS
  Get-ComputerRestorePoint | Sort-Object SequenceNumber -Descending | Select-Object -First 3 |
    Format-Table SequenceNumber, CreationTime, Description, RestorePointType -AutoSize
  ```
  Verify a new RP appears with current timestamp.

- [ ] **Gate B — System Restore feasibility check.** Before relying on RP 65 as the rollback target, confirm:
  ```powershell
  vssadmin list shadowstorage
  Get-Service vss, swprv | Format-Table Name, Status, StartType
  Get-WmiObject -Class Win32_Volume | Where-Object { $_.DriveLetter -eq 'C:' } |
    Select-Object Capacity, FreeSpace, BlockSize
  ```
  Verify VSS is running, sufficient shadow storage on C:, and free space ≥10 GB.

- [ ] **Gate C — Glitch-time evidence capture (MANDATORY, not skippable).**
  Run `Tools\Collect-Evidence.ps1` (the same evidence script used today) IMMEDIATELY after the next observed glitch — within 60 seconds. This produces a glitch-state snapshot to compare against the baseline-state snapshot already captured. Specifically check whether the glitch correlates with:
  - OneDrive / GoogleDriveFS active IO bursts
  - Recent Kernel-PnP / Storage / Disk events
  - Process Lasso `actions.log` lines
  - Top disk-IO process at glitch instant

  If glitches strongly correlate with sync IO bursts, hypothesis #2 dominates and Step 1 (System Restore) is unnecessary. If glitches occur with no IO/process correlation, hypothesis #1 dominates and Step 1 is the right fix.

  This gate exists because applying System Restore without it commits a meaningful blast radius (revert ~5 days of OS state) on a coincidence-of-timing argument. **The advisor was firm: do not skip.**

### Step 0 — Layer bisect (discriminator with proper observation window)

Goal: determine whether glitches stop when the I2C HID device is reset.

```powershell
# Disable then re-enable the SNSL002D I2C HID device
pnputil /disable-device "ACPI\SNSL002D\4&39979B3E&0"
Start-Sleep -Seconds 8
pnputil /enable-device  "ACPI\SNSL002D\4&39979B3E&0"
```

Then exercise the touchpad **under normal workload for 15-30 minutes** (not 60 seconds — the user reports intermittent skipping/pausing, which a 60s sample will miss). Run a passive event logger in parallel:

```powershell
# In a separate PowerShell window — runs until Ctrl+C
Get-WinEvent -FilterHashtable @{
    LogName='System';
    ProviderName=@('Microsoft-Windows-Kernel-PnP','Microsoft-Windows-WUDFHost','Microsoft-Windows-Kernel-IO','disk','partmgr');
    StartTime=(Get-Date).AddMinutes(-1)
} -MaxEvents 200 | Format-Table TimeCreated, Id, ProviderName, LevelDisplayName, Message -AutoSize -Wrap
```

Result interpretation:
- **Glitches stop, return after several minutes** → driver-state corruption pattern → Step 1 is a good fit.
- **No improvement** → fault is elsewhere (firmware, hardware, IO contention) → skip Step 1; jump to Step 2 + 3.
- **Glitches stop and stay stopped** → transient state corruption resolved by reset → no further action; revisit only if symptom returns.

**Rollback for Step 0:** none required — disable/enable is reset by reboot.

### Step 1 — System Restore to RP 65 (4/21 Scheduled Checkpoint)

**Apply only if Steps 0 and Gate C results indicate driver-layer regression.**

RP 65 (4/21 18:something Scheduled Checkpoint) is the freshest restore point that predates BOTH the 4/26 first-wave rollup landings AND the 4/28 Windows Update RP (which likely captures the rollup itself). RP 66 is too late.

```powershell
Get-ComputerRestorePoint | Where-Object SequenceNumber -eq 65
# Confirm description / timestamp before triggering restore
Restore-Computer -RestorePoint 65
# (will reboot; UAC prompt expected)
```

**Pre-Step-1 backup actions (do these first):**
1. **Pause Windows Update for 7 days** so the rollup doesn't immediately reapply on next sync:
   ```powershell
   $until = (Get-Date).AddDays(7).ToString('yyyy-MM-ddTHH:mm:ssZ')
   Set-ItemProperty -Path 'HKLM:\SOFTWARE\Microsoft\WindowsUpdate\UX\Settings' -Name 'PauseUpdatesExpiryTime' -Value $until -Type String
   ```
   (Alternatively use `UsoClient PauseUpdates` / Settings → Windows Update → Pause for 1 week.)
2. **Capture which packages will be reverted** (for rollback documentation):
   ```powershell
   dism /online /get-packages /format:list > "C:\codedev\PC_AI\Reports\touchpad-glitch-investigation-20260502\packages-pre-restore.txt"
   ```

**Validation post-Step-1:**
- Reboot completes, login works, all critical apps launch.
- `dism /online /get-packages` shows package versions back at 26100.<earlier> instead of 26100.8328 — confirms restore landed.
- Touchpad exercised for 30 minutes — glitch frequency reduced to zero.

**Rollback for Step 1:**
- Use Gate A's manual RP to roll forward to current state:
  ```powershell
  Restore-Computer -RestorePoint <Gate-A-RP-sequence-number>
  ```
- OR allow Windows Update to reapply the rollup (will happen automatically when pause expires or is removed).

**Risks:**
- System Restore reverts driver state, registry, and some system files — does NOT revert user files, but DOES revert installed program state (newly installed apps post-RP-65 will need reinstall).
- Apps installed between 4/21 and 5/2 may need reinstall: check `Get-WmiObject -Class Win32_Product | Where-Object InstallDate -ge '20260421'` before restore to enumerate.
- **Anything Process-Lasso-policy-tuned on 4/30 (per boot.TODO.md) WILL be reverted** — re-apply via `Tools\Apply-ProcessLassoUiSyncTuning.ps1` after restore.

### Step 2 — IO contention mitigation (independent of Step 1)

Apply regardless of Step 1 outcome. Lower-risk than Step 1; can be applied first as a discriminator for hypothesis #2.

**2.1 — Pause sync clients during touchpad-active workloads.**
- Right-click OneDrive system tray → Pause syncing → 2 hours. Repeat for Google Drive.
- Verify no glitches during paused window. If glitches stop, hypothesis #2 confirmed.

**2.2 — Extend Process Lasso rules** to include non-managed top consumers (RzDLLService, Dell.TechHub.Instrumentation.SubAgent, DDPM.Subagent.User, MsMpEng).

```powershell
# Backup current PL config first
$plBackup = "C:\codedev\PC_AI\Reports\touchpad-glitch-investigation-20260502\processlasso-config-backup-$(Get-Date -Format yyyyMMdd-HHmmss).pld"
Copy-Item "C:\Program Files\Process Lasso\config\*.pld" $plBackup
# Apply via existing tool (always with -WhatIf first)
.\Tools\Apply-ProcessLassoUiSyncTuning.ps1 -WhatIf
# If output looks correct:
.\Tools\Apply-ProcessLassoUiSyncTuning.ps1
```

**Rollback Step 2.2:** restore from `$plBackup` and reload PL.

**2.3 — Switch power plan to High Performance.**
```powershell
$current = (powercfg /getactivescheme) -split 'GUID:\s*' | Select-Object -Last 1
$current = $current -replace '\s*\(.*$',''
"Previous active scheme GUID: $current" | Set-Content "C:\codedev\PC_AI\Reports\touchpad-glitch-investigation-20260502\powercfg-rollback.txt"
powercfg /export "C:\codedev\PC_AI\Reports\touchpad-glitch-investigation-20260502\powercfg-balanced-backup.pow" $current
powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c   # High Performance
```

**Rollback Step 2.3:** `powercfg /setactive <GUID-from-rollback.txt>`.

### Step 3 — Update boot.TODO.md ledger

After every applied step, append to `boot.TODO.md` (the workstation reliability ledger):
- What changed (PL rule, power plan, restore-point sequence #)
- Evidence captured (file paths under `Reports\touchpad-glitch-investigation-20260502\`)
- Validation result (touchpad behavior in next 24h)
- Rollback artifact location
- New OPEN action items (e.g., "Reapply 4/30 PL tuning after Step 1 if applied")

### Step 4 (optional) — Lenovo Vantage / Lenovo Support engagement (if hardware suspect)

If Steps 0-2 do not resolve and Step 1 either rolled back or made no difference:
- Check Lenovo Vantage for any pending firmware updates (do NOT auto-apply — annex below).
- File a Lenovo support ticket referencing this investigation directory.
- Sensel community reports of P1 Gen 7 touchpad degradation may apply.

## Annex A (separate consent gate) — Firmware

`n48et.inf` (Lenovo, Class=Firmware) versions 1.18.0.0 (oem112.inf, 10/29/2025) and 1.19.0.0 (oem183.inf, 12/30/2025) are both in the DriverStore. **Firmware downgrade is NOT reversible** — flashing an older capsule is one-way.

Before any attempt:
1. **Stop. Re-confirm consent in writing.**
2. Verify which device n48et.inf services (could be ME firmware, PCH, or peripheral — NOT necessarily touchpad). Inspect with `pnputil /enum-drivers /class Firmware /v` and cross-reference DEVPKEY_Device_HardwareIds.
3. Consult Lenovo Vantage / Lenovo Support before any pnputil-based attempt.
4. **DO NOT proceed if it cannot be confirmed which device n48et.inf services.**

This annex is intentionally outside the main rollback chain.

## Annex B — What we are NOT doing (and why)

- **No iaLPSS2_I2C_MTL rollback.** Verified DriverDate 2025-07-02; not changed in regression window.
- **No ETDHSA rollback.** No evidence of TrackPoint fault.
- **No reliance on WUDFRd 219 as touchpad signal.** Verified chronic noise from unrelated devices.
- **No KB removal.** The "KB" identifier corresponds to a cumulative rollup whose components are merged into the servicing manifest; removal is not cleanly granular. System Restore is the supported rollback.

## Decision matrix for execution

| Outcome of Step 0 + Gate C | Apply | Skip |
|---|---|---|
| Glitches stop after disable/enable AND no IO correlation | Step 1 (System Restore), then Step 2.2 + 2.3 | Step 4, Annex A |
| Glitches correlate with sync IO bursts | Step 2 (all of it), monitor 24h, escalate to Step 1 only if Step 2 inadequate | Annex A |
| Glitches persist with no driver-reset improvement and no IO correlation | Steps 2.2 + 2.3 only; escalate to Step 4 (hardware) | Step 1, Annex A |
| Glitches resolve permanently after Step 0 | Step 2 as preventative | Step 1, Annex A |

## What I am asking the user to approve

- [ ] Run **Gate A** (manual restore point) — read-only, no risk.
- [ ] Run **Gate B** (VSS/shadow storage feasibility check) — read-only, no risk.
- [ ] **Do not skip Gate C.** Wait for next glitch and run `Collect-Evidence.ps1` within 60 seconds. Without this evidence, Step 1's blast radius is not justified.
- [ ] Run **Step 0** (I2C HID disable/enable + 15-30 min observation). Reversible by reboot.
- [ ] Approval for **Step 1** (System Restore to RP 65) **gated** on Step 0 + Gate C indicating driver-layer regression and not IO contention.
- [ ] Approval for **Step 2** (sync pause + PL rule extension + power plan) — applied first if hypothesis #2 likely, or alongside Step 1 either way.
- [ ] **Separate, written consent** required for Annex A (firmware) — not implicitly bundled.

No step will run without prior approval.
