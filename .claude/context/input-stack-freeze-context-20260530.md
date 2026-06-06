# Context — Input-Stack Freeze Investigation (DTM-P1GEN7)

- **context_id:** ctx-pcai-inputfreeze-20260530
- **created_at:** 2026-05-30
- **created_by:** Claude Code (systematic-debugging skill)
- **machine:** Lenovo ThinkPad P1 Gen 7 (21KV0014US), Win 11 Pro 26200, 64 GB RAM, BIOS N48ET33W 1.20
- **project root:** C:\codedev\PC_AI (CUDA/Rust ML workspace)

## State summary
Investigated recurring **Shift-key + trackpad + fingerprint freezing/glitching**. Used systematic-debugging + advisor + Microsoft Learn docs. Built a reusable diagnostics/remediation toolkit at `C:\codedev\PC_AI\Tools\InputDiagnostics\`. Applied the one no-admin safe fix live; the elevation-gated fixes are scripted for the user to run elevated.

## Diagnosis (two unrelated tiers)
- **Tier 2 (chronic — the actual complaint):** login storm (**57** autostart entries incl. Docker/Ollama/LM Studio/4× GoogleDriveFS/6× USB-audio panels) + **USB selective suspend ON (AC+DC)** (fingerprint wake) + **pro-audio ASIO DPC-latency** (RME Fireface/MADIface, Focusrite, Topping, miniDSP → freezes kbd+trackpad together) + **accidental FilterKeys/StickyKeys hotkeys** (`DelayBeforeAcceptance=1000ms`). HIGH confidence.
- **Tier 1 (acute — one evening, 5/29):** 6× `Kernel-Power 41` in 32 min + 7× `nvlddmkm` + 1 WHEA corrected-HW-error, **no crash dump** despite dumps enabled. All 7 KP41 in 90d were on 5/29 → acute outlier, not chronic. Hardware-vs-software undecidable without a dump. MEDIUM confidence.

## Key evidence (grounded)
- Keyboard `ACPI\LEN0071` (EC) and touchpad Synaptics `SNSL002D` (**I2C**) are on **different buses** → co-freeze must be system-wide (DPC/contention), not one device.
- Fingerprint = Synaptics **USB** `VID_06CB` → affected by USB selective suspend.
- **No `Display` 4101/4109** in 30d → `nvlddmkm 14` is NOT classic GPU TDR.
- `CrashDumpEnabled=3` + empty Minidump → 5/29 hangs were not bugchecks (hard hang / power-off).

## Decisions
- **dec-001 Process Lasso exonerated** — live `prolasso.ini` shields `dwm.exe`/`explorer.exe`, no throttle/affinity/CPUSets rules on input/UI. Contradicts user's initial hunch; advisor concurred.
- **dec-002 Lead with chronic tier, not hardware** — advisor corrected an over-weighted thermal narrative; the "did not shut down properly after preshutdown" SCM messages are benign shutdown-timing, not thermal trips.
- **dec-003 Toolkit in PC_AI\Tools\InputDiagnostics** — read-only collector + reversible remediation (backup/`-Revert`) + native load-capture; all values grounded in MS Learn docs.

## Changes made this session
- **APPLIED (HKCU, reversible):** accessibility activation hotkeys disabled — StickyKeys `510→506`, FilterKeys `126→122`, ToggleKeys `62→58` (cleared `0x04 HOTKEYACTIVE`). Verified via collector (reads back 506/122/58). Fully effective after next sign-in.
- **CREATED:** `Tools\InputDiagnostics\{Invoke-InputStackDiagnostics.ps1, Repair-InputStackQuickWins.ps1, Start-LoadCapture.ps1, README.md}`. Collector run-verified; wrote baseline to `PC_AI\Logs\input-diagnostics\`.

## Pending / handoff (user actions)
1. Run elevated: `Repair-InputStackQuickWins.ps1` → disables USB selective suspend (AC+DC) + sets CrashDumpEnabled=0x7.
2. Cut login storm (Task Manager Startup): Docker, Ollama, LM Studio, dup GoogleDriveFS, Razer, GoPro, MATLAB, SOLIDWORKS, Adobe.
3. Install + run LatencyMon ~10 min under load (`winget install Resplendence.LatencyMon -e`) → confirm DPC co-freeze driver. Install HWiNFO (`winget install REALiX.HWiNFO -e`) for thermals (no internet in authoring session).
4. Lenovo Vantage: latest Synaptics touchpad+fingerprint+BIOS; NVIDIA Studio driver clean-install.
5. Fix broken auto-start services: `PC_AI-HVSockProxy` (missing path), `vtss`. Repair Windows Hello *face* (IR camera in Error, Camera Frame Server crashed ×3).

## Environment notes
- Session was NON-elevated; no outbound internet (winget `0x80072efd`). powershell-pro subagent hit an infra API error (0 tool uses) → collector authored directly instead.

## Recommended next agents
- `devops-troubleshooter` — analyze next crash dump / LatencyMon output once captured.
- `powershell-pro` — extend toolkit (e.g., scheduled-task wrapper for periodic capture).

## CHECKPOINT 2026-05-30 (continuation)
- **Shift STILL broken** after the HKCU hotkey registry change. Root insight: registry Flags are read at sign-in; they do NOT update the LIVE session. Pivoting to a live-session fix via `SystemParametersInfo` (SPI_SETFILTERKEYS/STICKYKEYS, no admin/reboot) + checking for a keyboard remap (Scancode Map / PowerToys Keyboard Manager / AutoHotkey) that the accessibility flags wouldn't touch. New script: `Tools\InputDiagnostics\Reset-AccessibilityKeysLive.ps1`.
- **Directive:** implement remediation, validate it works, then work boot.TODO.md to completion using parallel specialist agents.
- **Hard blockers this shell:** NON-elevated (USB-suspend, crash-dump, per-device power, service fixes, HKLM need admin) + NO internet (winget `0x80072efd` → can't install HWiNFO/LatencyMon). Elevated/online items must be handed to user or run via elevated script.
