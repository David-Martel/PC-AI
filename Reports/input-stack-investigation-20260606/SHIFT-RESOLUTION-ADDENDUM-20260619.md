# Shift-Key Resolution Addendum — 2026-06-19

Follow-up to [`FINDINGS.md`](FINDINGS.md). The original investigation left the **Shift** branch
*undecided* ("software vs hardware/EC") pending an input-capture trace (open item **K1**). The user
supplied the discriminating observation that the prior analysis was missing, and it collapses the
decision tree.

## The decisive new fact

> **USB keyboards always work for every key (Shift included); the built-in keyboard's Shift is the
> only thing that intermittently fails.**

The built-in keyboard and every USB keyboard travel **completely separate paths** into Windows:

```
INTERNAL: Shift switch → keyboard matrix → ThinkPad Embedded Controller (EC) → i8042 PS/2 → i8042prt → kbdclass → Win32 input
USB:      Shift switch → USB HID keyboard MCU → USB stack → kbdhid/HID → kbdclass → Win32 input
                                                                          └── shared from here up ──┘
```

Everything **above `kbdclass`** is shared by both keyboards: accessibility filters
(Sticky/Filter/Toggle/Slow Keys), global low-level hooks (PowerToys, LogiOptions+, AHK), the
focused-app/IME layer, DWM/compositor state, the Scancode Map. **If any shared-layer cause were
swallowing Shift, the USB keyboard's Shift would fail too.** It never does. Therefore every
shared-layer hypothesis is **positively excluded by observation** — not just "refuted in steady
state" as before:

| Previously "open" software cause | Now | Why |
|---|---|---|
| Accessibility Filter/Sticky/Slow Keys | **Excluded** | Operates post-merge; would hit USB Shift too. (Also `Flags=2` = OFF, hotkeys disarmed — re-verified 2026-06-19.) |
| Global LL keyboard hook (PowerToys/Logi/AHK) | **Excluded** | Device-agnostic; would hit USB Shift too. (PowerToys KBM remap list empty.) |
| Focus loss / DWM-`dwmcore` instability | **Excluded** | Affects all keyboards equally. |
| Scancode Map / kbdclass remap | **Excluded** | Not present, and shared by both keyboards. |

**The fault is confined to the internal keyboard's private path: physical key/matrix → EC firmware →
i8042/PS-2.** The two live sub-causes:

1. **EC / keyboard firmware bug** — the EC intermittently fails to scan/report the Shift make/break.
2. **Physical contact fault** — Shift is a large, stabilizer-backed key; intermittent dome/membrane
   contact is a classic mechanical failure on this key specifically.

This product line has **documented precedent** for (1): Lenovo shipped an EC/BIOS fix for an
intermittent keystroke-drop bug on the ThinkPad X1 Extreme / P1 family
([Notebookcheck](https://www.notebookcheck.net/ThinkPad-X1-Extreme-Gen-2-Lenovo-fixes-the-keystrokes-bug.464923.0.html)).

## Live state re-verified (2026-06-19, DTM-P1GEN7)

- **BIOS** `N48ET33W (1.20)`, released **2026-03-01**; **EC `1.15`**. (Confirm current vs Lenovo
  catalog via Commercial Vantage — authoritative, already installed.)
- **Accessibility** Sticky/Filter/Toggle all `Flags=2` (OFF, available, hotkeys disarmed) — unchanged.
- **i8042prt** Running; `PollingIterations=12000`/`ResendIterations=3` (defaults); **no** Scancode Map;
  **no** upper/lower filters on `LEN0071` beyond `kbdclass`.
- Only one PS/2 keyboard present (`ACPI\LEN0071`); USB keyboards (Razer `VID_1532`, etc.) enumerate fine.

## What was built this session

`Tools/InputDiagnostics/Trace-ShiftKeySource.ps1` — a **device-aware** Raw Input trace
(`WM_INPUT` + `RIDEV_INPUTSINK`). Unlike `Test-KeyInput.ps1` (a `WH_KEYBOARD_LL` hook that sees only
the *merged* stream and cannot say *which* keyboard sent a key), this resolves
`RAWINPUTHEADER.hDevice` to a device name and classifies **INTERNAL vs USB/HID**. Because it uses an
input *sink*, it captures while unfocused — run it in the background and reproduce the failure in your
real editor. It records every Shift event per device and emits a verdict + JSON. Compiled and ran
clean (registration + pump + output verified).

## Action plan (decisive → cheap → definitive)

1. **Confirm the branch (the K1 capture, now device-resolved).** Run during a failure:
   ```powershell
   pwsh -File C:\codedev\PC_AI\Tools\InputDiagnostics\Trace-ShiftKeySource.ps1 -Seconds 60 -AllKeys
   ```
   Press internal Shift (both sides), Shift+A, and a USB Shift as control; reproduce the drop.
   - **Internal Shift events present** when it "failed" → loss is above the driver (revisit software).
     Expected to be *rare* given the USB-works fact, but this nails it.
   - **Internal Shift absent while internal letters still log, USB Shift fine** → scancode never
     reaches Windows → **EC firmware or physical fault** (the expected outcome).

2. **Cheap, reversible firmware fixes (do both before any hardware action):**
   - **Update BIOS/EC** via **Commercial Vantage** (or Lenovo support page
     [ds569195](https://support.lenovo.com/us/en/downloads/ds569195-bios-update-utility-bootable-cd-for-windows-11-64-bit-10-64-bit-thinkpad-p1-gen-7-type-21kv-21kw)).
     EC firmware ships *inside* the BIOS package; this is the documented fix path for this line.
   - **EC reset / power-drain:** shut down → unplug AC → hold power 15–30 s (or BIOS *Load Setup
     Defaults* / the emergency-reset pinhole). Clears a wedged EC state.

3. **Discriminate physical vs firmware while you wait for a fix:** when Shift next fails, note whether
   the **other** Shift works (per-key contact → physical) vs **both** fail together (EC/matrix →
   firmware); and whether it correlates with **resume-from-sleep** (EC/power) or **multi-key combos**
   like Ctrl+Shift (matrix rollover → firmware).

4. **Definitive:** if the trace shows the scancode never arrives **and** firmware is current + EC reset
   done, it's a hardware keyboard fault → **warranty / keyboard (FRU) replacement**. Lenovo built-in
   diagnostics (F10 at boot / Vantage hardware scan) can corroborate.

## HIL TEST RESULTS (2026-06-19 21:23–21:37) — supersedes the inference above

The K1 capture was run live with the user reproducing the symptom. Five device-tagged Raw Input
captures (`Logs/input-diagnostics/shift-source-*.jsonl`). **The hardware-defect hypothesis is
refuted by the data:**

| Evidence | Result |
|---|---|
| **Cadence-free hold test** (hold Shift, slowly tap `asdfghjkl`) | Internal: **20.9 s continuous hold, 0 mid-hold dropouts, 18/18 uppercase.** USB: 10.1 s, 0 dropouts, 13/13. The internal Shift contact is mechanically/electrically sound. |
| **Fast-typing ordering** (`Measure-ShiftOrdering.ps1`) | Wrong-case = letter registered before Shift. Internal **2/17 (12%)**, USB **3/23 (13%)** — statistically indistinguishable. Not internal-specific. |
| Shift scancodes reaching Windows | Always present, make + break, device = `\\?\ACPI#LEN0071`. Never "missing scancode". |
| Shift DOWN/UP "imbalance" (24/19 internal, 52/19 USB) | **Typematic auto-repeat** of held Shift, not dropped key-ups. Not a defect. |

**Revised root cause:** the intermittent wrong-case is a **Shift/letter timing race during fast
rolling** (the letter key's make registers a few–few-hundred ms before/after the Shift transition),
present on **both** the internal and external keyboards at the same rate. It is **not** a worn Shift
switch, EC firmware bug, dropped scancode, accessibility filter, or software hook. The decisive proof
is the 20.9 s zero-dropout hold: when Shift is unambiguously held, the internal keyboard is flawless.

**Caveats (honest):** (a) The tool timestamps events at user-mode processing time, not hardware time;
FIFO queue order preserves true ordering but absolute ms deltas can be jittered by system load.
(b) ~5 minutes were sampled — a rarer fault below this window isn't excluded, but the contact itself
is proven good. (c) The ~12–13% race rate is high and may be amplified by the user deliberately
rolling fast to reproduce, and/or by the input-stack latency/contention aggravators documented in
`FINDINGS.md` §2 H5 (OneDrive CPU churn, Defender, 57-entry login storm, pro-audio ASIO DPC).

**Revised recommendations (changed from rev 1):**
1. **Do NOT replace the keyboard or flash EC firmware on this evidence** — the switch is proven sound.
   (A BIOS/EC update remains harmless general hygiene but is not indicated by the data.)
2. The remaining lever is **system input-stack latency**: if jitter is reordering near-simultaneous
   Shift+letter events, lowering DPC latency / background contention (S1/S2/S3 in `FINDINGS.md`)
   should reduce the race rate. Testable: rerun `Trace-ShiftKeySource.ps1` while typing the same fast
   pattern under **low system load** (OneDrive/Defender scan/heavy procs quiesced) vs now; compare
   the late-shift % from `Measure-ShiftOrdering.ps1`.
3. To settle the daily-use "USB never fails" perception with a large natural sample, leave
   `Trace-ShiftKeySource.ps1` running in a visible window during normal work and compare per-device
   race rates over hours rather than a forced-fast minute.

## Carried-forward items updated

- [x] **K1 discriminator built** as a device-aware tool (`Trace-ShiftKeySource.ps1`).
- [x] **K1 run live (HIL)** — 5 captures; hardware-defect hypothesis **refuted** (see table above).
- [x] Internal Shift switch verified sound (20.9 s zero-dropout hold).
- [~] BIOS/EC update — *not indicated by data*; optional hygiene only.
- [~] Keyboard replacement — *not indicated by data*; switch proven good. Do not pursue on this evidence.
- [ ] Optional: low-load vs high-load fast-typing retest to test the input-latency/jitter angle.
- [ ] Optional: long natural-sample per-device race-rate comparison to settle daily-use perception.

*(Touchpad branch unchanged — see `FINDINGS.md` §4 T1/T2. This addendum is Shift-only.)*
