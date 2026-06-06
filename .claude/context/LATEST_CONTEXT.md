# LATEST CONTEXT

**Pointer →** [`input-stack-freeze-context-20260530.md`](input-stack-freeze-context-20260530.md)
**ID:** `ctx-pcai-inputfreeze-20260530` · **Date:** 2026-05-30

## One-line state (RESOLVED root causes)
- **Shift broken = Logitech Options+ keyboard hook** (v2.3.879545). Killing `logioptionsplus_agent` fixed it (user-confirmed). Durable fix: update Options+ / remove Shift Smart Action / disable Logi service.
- **Touchpad interference = Razer software** (user-confirmed). `RazerAppEngine` autostart REMOVED. Synapse NOT needed for the **Razer Core X V2 eGPU** (the "RTX 5060 Ti" is the eGPU over USB4). Full uninstall pending (elevated).
- **Acute 5/29 hard-freeze cascade = eGPU Thunderbolt/USB4 link** (recurring WHEA 17 corrected PCIe errors + nvlddmkm + KP41). eGPU is daisy-chained through hubs — move to a dedicated TB4 port.
- 6-script toolkit in `Tools/InputDiagnostics/` + Pester (41/41). Process Lasso & accessibility exonerated.

## APPLIED & VALIDATED 2026-05-30 (elevated, UAC-approved)
- USB selective suspend OFF (AC/DC=0x0 verified); CrashDumpEnabled=7; per-device power off (fingerprint+4 USB hubs); PC_AI-HVSockProxy+vtss Disabled.
- Process Lasso optimized: windowsterminal+pwsh removed from EcoQoS; ALL DefaultGPUAdapterPreferences→auto(0) (eGPU forcing on Terminal removed). On-disk validated; reloads at sign-in.
- Razer uninstalled (user) + autostart removed; was NOT needed for Core X V2 eGPU.
- Shift root cause: NOT software (stack proven clean; Razer/Logi/PowerToys all exonerated) → ThinkPad EC/keyboard firmware, triggered under eGPU+Terminal+contention load. Touchpad = Synaptics I2C-HID (TrackPoint immune). 5/29 cascade = eGPU Thunderbolt link (WHEA PCIe errors).

## Resume here (USER-gated, hands-on)
1. **Sign out/in (or reboot)** — applies HID/USB power changes + reloads Process Lasso new config. TOP priority.
2. When Shift fails: EC power-drain reset + `Tools\InputDiagnostics\Test-KeyInput.ps1` + external-keyboard test.
3. Lenovo Vantage: **BIOS/EC firmware** + Synaptics touchpad + NVIDIA driver updates (durable Shift/touchpad fix).
4. Behavioral validation: run heavy eGPU+Terminal workload post-sign-in; confirm input issues gone (PL fix removes the contention trigger).
5. Decide `StartWithPowerPlan=Balanced` vs High-Perf. Consolidated: `machine-reliability.TODO.md` + `boot.TODO.md`.
