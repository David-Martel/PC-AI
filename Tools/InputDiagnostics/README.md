# Input-Stack Diagnostics & Remediation (DTM-P1GEN7)

Reusable toolkit for the **Shift-key / trackpad / fingerprint freezing & glitching**
investigation on this machine (Lenovo **ThinkPad P1 Gen 7**, Win 11 26200, 64 GB RAM).
Created 2026-05-30. All findings grounded in the Microsoft docs linked below.

## TL;DR diagnosis (two unrelated tiers)

| Tier | Symptom | Root cause (evidence) | Confidence |
|------|---------|------------------------|------------|
| **2 — chronic** (your real complaint) | Everyday Shift + trackpad + fingerprint glitches | **Login storm** (40+ autostarts incl. Docker/Ollama/LM Studio/4× cloud-sync/6× USB-audio panels) + **USB selective suspend ON (AC+DC)** + **pro-audio ASIO DPC-latency** + **accidental FilterKeys/StickyKeys hotkeys** (`DelayBeforeAcceptance=1000ms`) | High |
| **1 — acute** (one bad evening) | Total hard freezes | **5/29 only**: 6× `Kernel-Power 41` in 32 min + 7× `nvlddmkm` + 1 WHEA corrected HW error, **no crash dump** despite dumps enabled. *Not* chronic (0 other days in 90d). Hardware-vs-software undecidable without a dump. | Medium |

### What was **ruled out**
- **Process Lasso** — *exonerated*. Live `prolasso.ini` shields `dwm.exe`/`explorer.exe` from ProBalance, no throttle/affinity rules on input/UI processes. It protects interactivity.
- **OneDrive** — minor I/O-filter overhead only.
- **GPU TDR** — *no* `Display` 4101/4109 events in 30d, so the `nvlddmkm 14` entries are **not** classic TDR.

### Device topology (why Shift + trackpad freeze *together*)
Keyboard = `ACPI\LEN0071` (EC), TrackPoint = `ACPI\LEN032A`, Touchpad = Synaptics **I2C-HID** `SNSL002D`, Fingerprint = Synaptics **USB** `VID_06CB`. Keyboard and touchpad are on **different buses**, so a single hardware fault can't freeze both — the co-freeze must be **system-wide** (DPC latency / contention).

## Scripts

| Script | Purpose | Elevation |
|--------|---------|-----------|
| `Invoke-InputStackDiagnostics.ps1` | **Read-only** collector. Reproduces the whole investigation → timestamped `.txt`/`.json` in `PC_AI\Logs\input-diagnostics`. Run whenever freezes recur. | No (more complete elevated) |
| `Repair-InputStackQuickWins.ps1` | Applies the safe, reversible fixes (backup + `-Revert`). | Yes (HKCU part works without) |
| `Start-LoadCapture.ps1` | Native `powercfg` thermal/power/latency capture during heavy ML load; drives HWiNFO/LatencyMon if installed. | Yes |

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
```

## What `Repair-InputStackQuickWins.ps1` changes (all reversible)
1. **Accessibility hotkeys** (HKCU, no admin) — clears `HOTKEYACTIVE 0x04`: StickyKeys `510→506`, FilterKeys `126→122`, ToggleKeys `62→58`. **Already applied in the 2026-05-30 session.**
2. **USB selective suspend OFF** (AC+DC) — `powercfg ... 2a737441-... 48e6b7a6-... 0`. Fixes fingerprint-wake glitches.
3. **Crash dump → Automatic (0x7)** — so the *next* hard freeze is captured (current was `0x3`).

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
