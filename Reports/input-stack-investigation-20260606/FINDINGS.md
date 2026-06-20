> **UPDATE 2026-06-19 (rev 2) — Shift branch tested with hardware-in-the-loop; hardware-defect
> hypothesis REFUTED.** A device-aware Raw Input capture during live reproduction showed: (1) a
> 20.9 s continuous internal-Shift hold with **zero contact dropouts** (switch is mechanically sound);
> (2) the wrong-case events are Shift/letter **timing races** during fast typing, occurring at **equal
> rates on the internal Lenovo (12%) and an external USB keyboard (13%)** — i.e. NOT internal-specific
> and NOT a worn switch / EC fault. The earlier "EC firmware or physical contact" reading (rev 1) is
> superseded. Full data + revised conclusion in
> [`SHIFT-RESOLUTION-ADDENDUM-20260619.md`](SHIFT-RESOLUTION-ADDENDUM-20260619.md); tools:
> `Trace-ShiftKeySource.ps1`, `Measure-ShiftOrdering.ps1`, `Analyze-ShiftTrace.ps1`.

# Input-Stack Investigation — Shift Key Not Recognized + Touchpad Lockup

**Machine:** ThinkPad P1 Gen 7 (LENOVO 21KV0014US), BIOS N48ET33W 1.20
**Date:** 2026-06-06 · **Method:** systematic-debugging (root cause before fixes) · **Mode:** read-only diagnosis
**Evidence:** raw probes in this folder (`H1_*`–`H7_*`, `events_*`, `_SUMMARY.txt`); prior art in
`Reports/touchpad-glitch-investigation-20260502/` and `.claude/context/input-stack-freeze-context-20260530.md`.

---

## 1. The load-bearing fact: keyboard and touchpad are on different buses

| Symptom device | Device ID | Driver/bus | Parent | Status |
|---|---|---|---|---|
| **Shift key** | `ACPI\LEN0071` (Standard PS/2 Keyboard) | `i8042prt` (PS/2) | PCI `VEN_8086&DEV_7E02` (ESPI/LPC) | OK |
| **Touchpad** | `HID\SNSL002D` (Sensel) | `hidi2c` (HID-over-I2C) | PCI `VEN_8086&DEV_7E78` (Intel Serial-IO I2C, drv 30.100.2527.40) | OK |

They share **no controller**. A single hardware/bus fault therefore **cannot** explain both symptoms. This forces the
conclusion: these are **two independent root causes** (plus a shared aggravator — system-wide contention — that makes
them *appear* to co-occur). This is the central result; every fix below is scoped to one branch.

---

## 2. Hypothesis verdicts (evidence-based)

| # | Hypothesis | Verdict | Decisive evidence |
|---|---|---|---|
| H1 | Filter/Sticky/Bounce Keys swallow Shift | **REFUTED (currently)** | `Flags=2` on FilterKeys/StickyKeys/ToggleKeys = AVAILABLE, **OFF**, hotkey **disarmed** (0x04 clear); `DelayBeforeAcceptance=0`. Decoded vs `winuser.h` (0x01=ON, 0x02=AVAILABLE, 0x04=HOTKEYACTIVE). The prior registry hardening *is* in effect — accidental hold-Shift-8s toggle is disabled. |
| H2 | I2C-HID touchpad driver fault | **Inconclusive (no fault)** | Touchpad Status OK, no problem code, MS inbox `hidi2c`, no I2C error events in 14d. |
| H3 | Shared Intel Serial-IO/I2C/GPIO controller | **REFUTED** | Keyboard parent `7E02` ≠ touchpad parent `7E78`; all controllers Status OK, current drivers. |
| H4 | Process Lasso starving the input thread | **REFUTED** | `prolasso.ini` gives `syntp*/snsl*/sensel*/elan*/dwm/ctfmon/textinputhost` **above-normal** priority + IO prio 3 and excludes them from SmartTrim; `RestrainByAffinity=false`. PL **protects** input, not starves it. |
| H5 | Heavy I/O stalls the UI pipeline | **Weak / aggravator** | Instantaneous load light (CPU 15.9%, disk queue 0). But OneDrive has **~15 h cumulative CPU** (55,167 s, 10× next proc) and Defender RT+behavior+on-access all on — a credible source of *transient* UI stalls, not a steady-state cause. |
| H6 | Power management powers down the touchpad | **SUPPORTS (touchpad only)** | USB selective suspend and PCIe ASPM are **already 0** (not a factor). **But** `MSPower_DeviceEnable=True` on **both** the Sensel touchpad (`ACPI\SNSL002D`) **and** its parent I2C controller (`7E78`) = "Allow the computer to turn off this device to save power" is **ON**. With Modern Standby in use (Kernel-Power 506/507 ×3 in 14d), this is the textbook mechanism for **I2C-HID touchpad resume lockups**. The i8042 keyboard is **not** in the power-managed set → does not explain Shift. |
| H7 | Vendor driver/service interference | **Inconclusive** | ELAN PST (TrackPoint 31.21.51.2, 2025-08, current), LenovoVantageService, TPHKLOAD, ImController running; no stale touchpad driver. |

**Event-log corroboration (14 d):** zero `i8042prt/kbdhid/kbdclass/hidi2c/mouhid/WHEA` errors (input stack clean at OS
layer); Kernel-Power 41 ×1 (the 06-05 reboot); Modern Standby 506/507 ×3; **dwm.exe APPCRASH ×62 all on 05-26**
(`dwmcore.dll`) — a one-day compositor instability burst that would freeze the whole UI (incl. touchpad).

---

## 3. Root-cause reading

**Touchpad lockup — best supported (H6 × Modern Standby), plus DWM-crash aggravator.**
The I2C touchpad stack has device power-down enabled; on Modern Standby resume the controller/endpoint can fail to
re-arm its interrupt → touchpad dead until a re-enumeration. Episodic `dwmcore.dll` crashes (05-26) are a *second,
independent* way the touchpad (and everything) freezes — and those crashes plausibly tie to the **NVIDIA RTX 2000 Ada
Code 31** failure open in `workstation-audit-20260606-124859`.

**Shift key — undecided, no OS-layer evidence yet.** The i8042 path is clean in 14 d of logs and H1/H3/H6 are refuted
for the keyboard. Two live branches remain, distinguishable only by an **input-capture trace**:
- *Software/app branch:* Shift scancodes reach the OS but are dropped by a focus loss (during dwm/explorer instability)
  or a global hook (PowerToys/AutoHotkey/Scancode Map).
- *Hardware/EC branch:* Shift scancodes never reach the OS → EC/keyboard-matrix firmware or a physical contact issue.

---

## 4. Proposed fixes — ranked by confidence × reversibility, each with an evaluation method

> Apply **one at a time** and measure with the Gate-C harness (§5) so causation is attributable. Nothing here is
> auto-applied; all are reversible and have an existing scripted tool where noted.

### Touchpad branch

| ID | Fix | Confidence | Risk | Tool | How to evaluate it worked |
|---|---|---|---|---|---|
| **T1** | Disable device power-down on the Sensel touchpad **and** its parent I2C controller `7E78` (`MSPower_DeviceEnable=False`) | **High** | Low, reversible | new `Set-InputDevicePowerPolicy.ps1` (proposed) or Device Manager → Power Management tab | Run `Watch-InputGlitch.ps1` across ≥3 Modern-Standby cycles; touchpad survives resume; glitch/day count drops vs baseline |
| T2 | Capture a **glitch-time snapshot** the instant the touchpad next locks (Gate C — never completed) | n/a (diagnostic) | None | new `Watch-InputGlitch.ps1` (§5) | Confirms/refutes T1 causation with in-the-moment device + power + Modern-Standby state |
| T3 | If T1 insufficient: investigate `hidi2c` selective-suspend / Sensel firmware (one-way; needs consent) | Medium | Med | Lenovo Vantage / Sensel | Symptom recurs after T1 over ≥1 week |

### Shift / keyboard branch

| ID | Fix / step | Confidence | Risk | Tool | How to evaluate |
|---|---|---|---|---|---|
| **K1** | **Decide the branch first**: run the LL-hook trace while reproducing Shift loss | n/a (discriminator) | None (read-only) | existing `Test-KeyInput.ps1` (must be interactive) | Shift events appear → software branch; absent while other keys appear → EC/hardware branch |
| K2a | *Software branch:* address dwm/explorer instability → fix **NVIDIA Code 31** (restore point first) | Med | Med | `Reports/workstation-audit-…` remediation | DWM crash count → 0; Shift loss stops co-occurring with UI hitches |
| K2b | *Software branch:* audit global hooks (PowerToys/AHK/Scancode Map) | Med | Low | `Reset-AccessibilityKeysLive.ps1 -DiagnoseOnly` | Trace shows no remap layer intercepting Shift |
| K2c | *Hardware branch:* BIOS/EC update + external-keyboard A/B | Med | Med | Lenovo Vantage | External keyboard Shift OK while internal fails → EC/matrix confirmed |

### System-wide aggravator (reduces co-freeze probability; helps both)

| ID | Fix | Confidence | Risk | Tool |
|---|---|---|---|---|
| S1 | Reduce the 57-entry login storm | Med | Low, reversible | existing `Optimize-StartupLoad.ps1 -Apply` |
| S2 | Investigate OneDrive ~15 h CPU churn (pathological) | Med | Low | manual / `onedrive-triage` follow-up |
| S3 | Confirm/refute pro-audio ASIO DPC latency | Med | None | LatencyMon (when online), `Start-LoadCapture.ps1` |

**Explicitly NOT recommended:** System Restore to RP 65 (touchpad-glitch Step 1) — high blast radius, and its premise
(HID rollup `26100.8328` split-version regression) is **unconfirmed** (Gate C never ran). Do T2 first.

---

## 5. The missing piece — an evaluation harness for an intermittent bug

The reason prior fixes were "scripted but unverified" is there was **no way to measure an intermittent symptom's
frequency** or capture its state at the moment it fires (Gate C). `Tools/InputDiagnostics/Watch-InputGlitch.ps1`
(added this session) fills that gap:

- **Hotkey/triggered snapshot:** the instant a glitch happens, the user records it — capturing accessibility flags,
  touchpad + I2C device status & power state, recent Modern-Standby (506/507) transitions, dwm/explorer health, and
  top CPU/IO — into a timestamped JSON. *This is the Gate-C capture that was always missing.*
- **Symptom log:** appends `{timestamp, symptom, captured-state-ref}` to a running ledger so frequency is measurable.
- **Before/after report:** glitches-per-day, so each fix in §4 can be judged "worked / didn't."

Workflow: run a **baseline** week → apply **T1** → run another week → compare glitch/day. Repeat per fix.

---

## 6. Open items carried forward (see also `machine-reliability.TODO.md`)
- [ ] Run `Test-KeyInput.ps1` interactively during a Shift-loss episode (K1 — decides the keyboard branch).
- [ ] Apply **T1** (touchpad + I2C power-down off) and measure with `Watch-InputGlitch.ps1`.
- [ ] Complete **Gate C** glitch-time capture (T2) before any System Restore.
- [ ] NVIDIA RTX 2000 Ada **Code 31** remediation (cross-links to the Shift software branch via DWM stability).
- [ ] Investigate OneDrive ~15 h cumulative CPU.
