# Intermittent Lenovo keyboard investigation — September 12, 2026

**Status: unresolved.** The user reports continuing interference with the laptop's
internal keyboard. Historical reports describe intermittent bare Shift failures,
sometimes with Ctrl+Shift still working, plus partly separate touchpad/UI incidents.
The current exact symptom and a matched external-keyboard comparison remain to be
confirmed. No fault was reproduced during this research pass.

## What the existing evidence actually supports

The old tools could manufacture misleading conclusions. The trace analyzer counted
every repeated Shift DOWN as another held key, combined devices, called all totals
"Internal," reconstructed typed text, and treated unequal make/break counts as
hardware loss. The ordering tool inferred intended capitalization from a nearby
Shift event. Its historical 12%/13% figures are not observed application failure
rates. A clean 20.9-second hold does not rule out an intermittent keyboard fault.
Successful USB typing at another time does not categorically rule software out.

The repaired analyzers track each device and Shift side independently, retain
repeat and ambiguous-boundary counts, emit no typed content, and leave diagnosis
undetermined. The former ordering entrypoint now delegates to that shared analyzer;
its old timing heuristic is retired. Collector output and current TODOs have also
lost the unsupported hardware/software verdicts. Historical reports retain their
observations with prominent correction notices.

[History and trace-method review](history-review.md) preserves the chronology,
aggregate replay evidence and remaining diagnostic limitations.

## Current machine and software evidence

- ThinkPad P1 Gen 7, model 21KV0014US, Windows 11 Pro 25H2, build 26200.9445.
- BIOS N48ET35W 1.22, EC 1.15. These match Lenovo's N48UJ21W package for 21KV/21KW.
  No newer applicable BIOS or keyboard-specific fix was verified. The older 1.20/1.21
  update guidance is stale. [Lenovo package readme](https://download.lenovo.com/pccbbs/mobiles/n48uj21w.html)
- The started native keyboard candidate uses `kbdclass → i8042prt → ACPI`.
  Three USB HID keyboard interfaces use `kbdclass → kbdhid → HidUsb`. No additional
  third-party kernel filter appeared in those enumerated stacks. PnP health does
  not prove correct event delivery during an intermittent episode.
- Windows Update's configured service returned zero uninstalled, nonhidden driver
  offers in a successful bounded query. Vendor-only, hidden or policy-excluded
  packages are outside that result. [Driver-review evidence](driver-research.md)
- PowerToys Keyboard Manager is enabled and running, with all six observed remap
  arrays empty. Installed version 0.101.2362.0 matches GitHub's latest release result
  at this check. Logitech Options+ and Lenovo input/accessory utilities also run.
  This establishes software to test, not culpability. Microsoft documents Keyboard
  Manager's hook and suppression/injection behavior and the Settings toggle for
  disabling it. [Keyboard Manager documentation](https://learn.microsoft.com/en-us/windows/powertoys/keyboard-manager),
  [implementation](https://github.com/microsoft/PowerToys/blob/main/doc/devdocs/modules/keyboardmanager/keyboardmanager.md),
  [release](https://github.com/microsoft/PowerToys/releases/tag/v0.101.2362.0)
- Accessibility registry values currently show the three features off, with zero
  FilterKeys wait/bounce. The new snapshot also queries live SPI GET state so a
  registry-only observation is not mistaken for current session state.

## Candidate causes and the next discriminating test

These are priorities for investigation, not probability estimates.

| Candidate | Why it remains plausible | Evidence that would help separate it |
| --- | --- | --- |
| Internal key matrix/contact, connection, EC or i8042 path | Longstanding device-specific report; brief successful samples cannot exclude it | Same physical key failure in Lenovo UEFI, or a labeled internal failure with a simultaneous working USB control; account for capture health |
| User-mode hooks/input utilities | Keyboard Manager, Options+ and Lenovo utilities are active; empty mappings lower the likelihood of intentional remapping | Repeat identical trials with one utility disabled, restored, then disabled again; compare failure counts and observed exposure |
| App focus, shortcut handling, IME/text-input path | Raw receipt and rendered text are different observations | Compare a plain local app with the affected app; record exact expected outcome and received/rendered result for each labeled trial |
| Sleep, docking, power or firmware interaction | Historical resume/co-glitch reports | Matched trials before/after sleep, dock/AC changes; record precise onset/recovery and concurrent device state |
| Scheduling, ISR/DPC, GPU or UI stalls | Historical workload/display incidents justify correlation | Capture WPR scheduling/DPC/ISR around the failure and inspect the responsible stacks; distinguish a whole-UI stall from one missing modifier |
| Accessibility, scancode map or kernel filters | Current negative snapshots reduce likelihood but are not continuous observations | Compare live state and effective stacks immediately before/during failure; treat failed probes as unknown |

Serial IO/I2C and USB power tuning are secondary for native PS/2 typing; elevate
them when a matching touchpad/USB symptom or stack-specific evidence appears.
Different buses do not prove either independent faults or one shared cause.

## Better automated debugging and device testing

1. **Repeatable snapshots.** Run `Tools/InputDiagnostics/Get-KeyboardDiagnosticSnapshot.ps1`
   in good and failing intervals. It collects sanitized machine, software, driver,
   filter, live-accessibility, remap-count and recent-event metadata. Each probe
   reports empty, unavailable, error or timeout separately. Optional `-OutputPath`
   creates a new JSON file in an existing directory; `-DryRun` and `-h` do no probing
   or writing. It captures no keyboard input and changes no device settings.
2. **Labeled physical trials.** Predeclare the test: left Shift held before A,
   right Shift held before A, ordinary A, and the specific Ctrl+Shift action in a
   neutral test context. Record intended action, device, trial start/end, observed
   application outcome, symptom marker and modifier state. Test internal and USB
   under the same app/load conditions. Use fixed test input, not private text.
3. **Correlate observer layers.** Use the existing device-aware Raw Input capture
   alongside a focused application observer. Raw Input supplies device-attributed
   events; it does not supply physical intent or rendered text. A future focused
   harness should record trial/result aggregates and a shared monotonic clock,
   resolve Raw Input handles to verified PnP identities, detect device changes and
   capture gaps, and report actual failures only against
   labeled expectations. An LL hook can itself disappear after timeout; an empty
   hook trace is not hardware proof. [Raw Input](https://learn.microsoft.com/en-us/windows/win32/inputdev/about-raw-input),
   [hook timeout behavior](https://learn.microsoft.com/en-us/previous-versions/windows/desktop/legacy/ms644985(v=vs.85))
4. **Controlled A/B/A.** Start with Keyboard Manager's Settings toggle, then test
   other utilities separately. Record observed session duration/trial counts and
   whether the original failure recurs after restoration. Zero failures in a short
   run does not establish a cure. Do not divide events by first-to-last incident
   timestamps and call that a failure rate.
5. **ETW/WPR escalation.** The installed GeneralProfile includes DPC, interrupts,
   context switches, ready-thread and sampled CPU data. Inspect `wpr -status` before
   owning a trace; record a brief failing interval and analyze it in WPA. Provider
   presence alone does not prove useful per-key coverage. HID/I2C tracing does not
   automatically cover the PS/2 keyboard path. [WPR options](https://learn.microsoft.com/en-us/windows-hardware/test/wpt/wpr-command-line-options)
6. **Out-of-Windows validation.** Lenovo's interactive UEFI keyboard test exercises
   the physical native keys with Windows software absent. A reproduced failure there
   is stronger hardware/firmware evidence; a pass between episodes is inconclusive.
   Software `SendInput` or USB HID emulation exercises a different path and cannot
   validate the laptop's physical matrix/EC. Fully automated physical coverage would
   require an actuator/fixture and synchronized observation. [Lenovo keyboard diagnostics](https://support.lenovo.com/ee/en/solutions/YTV104507),
   [SendInput semantics](https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-sendinput)

HLK Device.Fundamentals testing belongs to a dedicated test environment after a
driver-specific reproduction. Driver Verifier intentionally stress-tests drivers
and can bugcheck; it is not the first collection step on this active laptop.
[Microsoft Driver Verifier guidance](https://learn.microsoft.com/en-us/windows-hardware/drivers/devtest/driver-verifier)

## Validation and remaining work

- Repaired offline analyzer: nine behavioral tests passed, including repeats,
  device/side separation, capture boundaries, empty input, malformed records and
  privacy. Historical replay produces aggregates without revealing typed contents.
- A three-second modifiers-only Raw Input smoke compiled, registered, pumped and
  exited successfully. It captured zero events; it proves startup only, not delivery
  or a successful physical-key test.
- Snapshot validation and current probe results are recorded in `validation.md`.
- No driver installation, registry/power adjustment, service stop, device restart,
  firmware update or reboot was performed. The controlled symptom experiment and
  UEFI test still require physical participation/a recurrence.

Next implementation: a labeled focused-input harness, monotonic capture metadata,
explicit session exposure, capture-health signals and an incident bundle that joins
the trial, snapshot and bounded performance trace. Preserve the present failure
before attempting resets. Closure requires reproduced failure → controlled change →
matched validation, with enough observation to address this intermittent symptom.
