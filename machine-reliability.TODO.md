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
- **Touchpad glitch (TrackPoint immune):** Synaptics I2C-HID specific; PL is NOT throttling it
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

## TOOLKIT (`Tools/InputDiagnostics/`, Pester 41/41)
| Script | Purpose | Elevation |
|--------|---------|-----------|
| `Invoke-InputStackDiagnostics.ps1` | read-only full diagnostic snapshot | no |
| `Reset-AccessibilityKeysLive.ps1` | live FilterKeys/StickyKeys off + remap check | no |
| `Test-KeyInput.ps1` | raw WH_KEYBOARD_LL monitor (does Shift reach OS?) | no |
| `Optimize-StartupLoad.ps1` | login-storm report/trim (HKCU) | no |
| `Repair-InputStackQuickWins.ps1` | accessibility + USB suspend + crash dump | partial |
| `Repair-WorkstationInputReliability.ps1` | USB suspend + crash dump + per-device power + svc disable | yes |
| `Repair-ProcessLassoTerminalGpu.ps1` | PL EcoQoS + GPU-pref optimization | yes |
| `Launch-Elevated.ps1` / `Invoke-RepairElevatedLogged.ps1` / `Launch-RepairAndRead.ps1` | UAC launchers w/ logging | n/a |
