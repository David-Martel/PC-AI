# LATEST CONTEXT

**Pointer →** [`pcai-context-20260606-input-stack-reinvestigation.md`](pcai-context-20260606-input-stack-reinvestigation.md)
**ID:** `ctx-pcai-inputstack-20260606` · **Date:** 2026-06-06

## One-line state
Re-investigated **shift-key + touchpad lockup** with fresh live evidence (systematic-debugging). **Two
independent root causes** — keyboard (PS/2 `LEN0071`/i8042) and touchpad (Sensel `SNSL002D` I2C) are on
**different buses** (parents `7E02` vs `7E78`), so simultaneous failure = system-wide contention, not a shared fault.

- **Touchpad — best supported (H6):** "allow computer to turn off this device" (`MSPower_DeviceEnable=True`,
  confirmed live) on BOTH the Sensel touchpad AND its parent I2C controller `7E78`, × Modern Standby (506/507)
  = classic I2C-HID resume-lockup. Fix **T1** = disable power-down on both (reversible).
- **Shift — undecided, no OS-layer evidence** (i8042 logs clean 14 d; H1 FilterKeys REFUTED = off+disarmed now).
  Run `Test-KeyInput.ps1` during a failure to decide software (app focus / DWM instability) vs EC/firmware.
- Note: prior contexts variously blamed "Logitech Options+" then "EC firmware" then "Synaptics" — those are
  **superseded**; the touchpad is **Sensel** (Synaptics = fingerprint, ELAN = TrackPoint). H4 Process Lasso REFUTED.
- dwm.exe crashed 62× on 05-26 (dwmcore.dll) → cross-links the open **NVIDIA RTX 2000 Ada Code 31**.

## New this session
`Tools/InputDiagnostics/Watch-InputGlitch.ps1` — Gate-C glitch capture + glitches/day before/after a fix (the
measurement prior fixes lacked). CLAUDE.md accuracy reconciled; `Reports/doc-tooling-evaluation-20260606.md`.

## Resume here (USER-gated, hands-on)
1. **`Test-KeyInput.ps1`** during a Shift failure — decides the keyboard branch.
2. Apply touchpad **T1** (power-down off on `SNSL002D` + `7E78`); measure with `Watch-InputGlitch.ps1`.
3. **NVIDIA Code 31** remediation (restore point + Vantage); OneDrive WNS GUI relink; re-run stale doc pipeline.
4. Decide: delete ~934 MB OneDrive `.db` evidence copies? Consolidated: `machine-reliability.TODO.md` + `boot.TODO.md`.
