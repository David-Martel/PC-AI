# Machine Reliability TODO — DTM-P1GEN7 (ThinkPad P1 Gen 7)

Consolidated issues + resolutions from the 2026-05-30 input-stack investigation
(Shift / trackpad / fingerprint / eGPU / Terminal / Process Lasso). Companion to
[boot.TODO.md](boot.TODO.md). Toolkit: `Tools/InputDiagnostics/`.

Machine: Win 11 Pro 26200, 64 GB RAM, BIOS N48ET33W 1.20. GPUs: Intel Arc Pro (iGPU),
NVIDIA RTX 2000 Ada (internal), **NVIDIA RTX 5060 Ti = eGPU in Razer Core X V2 (USB4, direct)**.

## RESOLVED / APPLIED (2026-05-30)
- [x] **USB selective suspend OFF** (AC+DC) — validated `0x0`. Fixes fingerprint wake glitch.
- [x] **Crash dump → Automatic (0x7)** + AutoReboot — next hard freeze is now captured.
- [x] **Per-device power** `EnhancedPowerManagementEnabled=0` on Synaptics fingerprint + 4 USB Root Hubs.
- [x] **Boot-churn services disabled**: `PC_AI-HVSockProxy` (NSSM, missing path), `vtss` (VTune).
- [x] **Razer software** uninstalled by user + `RazerAppEngine` autostart removed. NOT needed for Core X V2 eGPU.
- [x] **Process Lasso optimization**: removed `windowsterminal.exe`+`pwsh.exe` from EcoQoS;
      set ALL `DefaultGPUAdapterPreferences` → auto(0) (eGPU forcing on Terminal removed). Validated on disk.
- [x] Accessibility activation hotkeys disabled (FilterKeys/StickyKeys) — was never the cause but cleaned.
  - Backups: `Tools/InputDiagnostics/backups/`. Logs: `Logs/elevated/`. Revert via each script's `-Revert`.

## ROOT CAUSES (validated)
- **Shift "doesn't register" (bare Shift fails, Ctrl+Shift works, both keys, intermittent):**
  software stack PROVEN CLEAN (no hooks/remaps/filters/accessibility). TrackPoint always works
  (separate EC/PS2 path) while touchpad (Synaptics I2C-HID) glitches → **ThinkPad EC / keyboard
  firmware** is the leading cause, triggered under the high-load eGPU+Terminal+contention state.
- **Touchpad glitch (TrackPoint immune):** **Sensel `SNSL002D`** HID-over-I2C specific (NOT Synaptics —
  Synaptics is the fingerprint `VID_06CB`; ELAN is the TrackPoint). PL is NOT throttling it
  (touchpad services are Above-Normal/IO-3/ProBalance-excluded). I2C/EC under load.
- **eGPU+Terminal+Process Lasso competition (user-confirmed pattern):** PL pinned Terminal/pwsh to
  EcoQoS AND forced Terminal to render on the eGPU → render vs compute contention over Thunderbolt.
  FIXED in PL config.
- **Acute 5/29 hard-freeze cascade:** eGPU Thunderbolt/USB4 link — recurring `WHEA-Logger 17`
  corrected PCIe errors (5/29 + 5/30) + nvlddmkm + 6× Kernel-Power 41 (one evening only).

## PENDING — USER ACTIONS (hands-on / hardware / can't be scripted here)
- [ ] **Sign out / sign back in (or reboot)** — applies HID/USB power changes AND reloads Process
      Lasso with the new config. (Single most important next step.)
- [ ] **Shift hardware confirmation**: when Shift next fails, run EC power-drain reset (AC off +
      hold power 30s / P1 Gen 7 emergency-reset pinhole), then `Test-KeyInput.ps1` (do Shift events
      reach the OS?), and test an external USB keyboard.
- [ ] **Lenovo Vantage**: update **BIOS/EC firmware** (1.20 → newer) + **Synaptics touchpad driver**
      + **NVIDIA driver**. The durable Shift+touchpad fix.
- [ ] **eGPU link**: keep Core X V2 on a dedicated TB4/USB4 port with a certified cable; monitor WHEA.
- [ ] **Behavioral validation**: after sign-in, run a heavy eGPU+Terminal workload and confirm the
      Shift/touchpad/Terminal-lag issues no longer occur (the PL fix should remove the contention trigger).
- [ ] **Decision**: `StartWithPowerPlan=Balanced` in prolasso.ini reverts the 2026-05-02 High-Performance
      intent every boot. Keep Balanced (battery) or switch to High Performance for eGPU sessions?
- [ ] Windows Hello **face** separately broken: Integrated IR camera in `Error` + Camera Frame Server
      crashed ×3 → reinstall/repair camera + RealSense drivers.
- [ ] Optional: `Optimize-StartupLoad.ps1 -Apply` to trim the 63-entry login storm (HKCU, reversible).
- [ ] Leftover stopped Razer *service* shells (Chroma SDK/Elevation/Game Manager) — harmless; remove
      with the Razer uninstaller cleanup if desired.

## 2026-06-06 RE-INVESTIGATION (read-only; evidence in `Reports/input-stack-investigation-20260606/`)
Re-ran the input-stack diagnosis from scratch (systematic-debugging). New/updated findings:
- **Device topology corrected:** Shift = `ACPI\LEN0071` PS/2 (i8042, parent PCI `7E02`); Touchpad =
  Sensel `SNSL002D` HID-over-I2C (`hidi2c`, parent Intel I2C `7E78`). **Different buses → two independent
  root causes.** Full analysis + ranked fixes + eval protocol in `…/FINDINGS.md`.
- **NEW actionable touchpad fix (T1):** "Allow the computer to turn off this device" (`MSPower_DeviceEnable`)
  is **still ON** for BOTH the Sensel touchpad AND its parent I2C controller `7E78` (confirmed live today).
  With Modern Standby (Kernel-Power 506/507 observed), this is the classic I2C-HID resume-lockup mechanism.
  The 2026-05-30 per-device-power fix covered the fingerprint + USB hubs but **not** the touchpad/I2C stack.
  → Disable power-down on `SNSL002D` + `7E78`, then measure (reversible, Device Manager Power Mgmt tab).
- **H1 Filter/Sticky Keys re-confirmed REFUTED** (currently OFF + hotkey disarmed, Flags=2).
- **Shift still has NO OS-layer evidence** (i8042 logs clean 14d) → run `Test-KeyInput.ps1` during a failure
  to decide software (app focus / DWM instability) vs EC/firmware. dwm.exe crashed 62× on 05-26 (dwmcore.dll)
  — cross-links to the open **NVIDIA RTX 2000 Ada Code 31** failure (`Reports/workstation-audit-20260606-124859`).
- **NEW eval harness:** `Watch-InputGlitch.ps1` — Gate-C glitch-time capture + glitches/day before/after a fix
  (the measurement prior fixes lacked). Run a baseline week, apply T1, compare.

### Cross-report open items (from Reports triage, regardless of session)
- [ ] **NVIDIA RTX 2000 Ada — Code 31** (CM_PROB_FAILED_ADD), persists after rescan — restore point + Vantage rollback.
- [ ] **OneDrive WNS push channel** never registers (290+ pending changes) — needs GUI unlink/relink (`onedrive-triage-20260509`).
- [ ] **OneDrive ~15 h cumulative CPU** (pathological background churn) — investigate/limit.
- [ ] **iCloud** sync root orphaned (`C:\Users\david\iCloudDrive`) — reinstall or archive.
- [ ] Gate C touchpad glitch-time capture before any System Restore (touchpad-glitch Step 1 stays unapproved).

## TOOLKIT (`Tools/InputDiagnostics/`, Pester 41/41)
| Script | Purpose | Elevation |
|--------|---------|-----------|
| `Invoke-InputStackDiagnostics.ps1` | read-only full diagnostic snapshot | no |
| `Reset-AccessibilityKeysLive.ps1` | live FilterKeys/StickyKeys off + remap check | no |
| `Test-KeyInput.ps1` | raw WH_KEYBOARD_LL monitor (does Shift reach OS?) | no |
| `Watch-InputGlitch.ps1` | Gate-C glitch capture + glitches/day before/after a fix (read-only) | no |
| `Optimize-StartupLoad.ps1` | login-storm report/trim (HKCU) | no |
| `Repair-InputStackQuickWins.ps1` | accessibility + USB suspend + crash dump | partial |
| `Repair-WorkstationInputReliability.ps1` | USB suspend + crash dump + per-device power + svc disable | yes |
| `Repair-ProcessLassoTerminalGpu.ps1` | PL EcoQoS + GPU-pref optimization | yes |
| `Launch-Elevated.ps1` / `Invoke-RepairElevatedLogged.ps1` / `Launch-RepairAndRead.ps1` | UAC launchers w/ logging | n/a |
