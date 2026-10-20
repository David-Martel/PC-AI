# Keyboard history review — 2026-09-12

This is a history/source review, not a current hardware diagnosis. No device/service/registry changes or new input capture occurred. Historical key logs were read only to produce counts; typed contents and device identifiers are not reproduced here.

Line references below describe the pre-repair source at `4ac8d559`; subsequent
correction notices and tooling changes shift those line numbers. The root later ran
a three-second modifiers-only startup smoke with zero events, documented separately.

## Symptoms and chronology

- May 30: recurring Shift, touchpad and fingerprint glitches; Shift reportedly fails alone, Ctrl+Shift works, both Shift keys affected intermittently. TrackPoint reportedly remains usable. `machine-reliability.TODO.md:21–32` mixes these observations with an unproven EC diagnosis.
- May 2 touchpad work: boot ledger records restore point creation, Balanced-to-High-Performance change and I2C disable/enable; System Restore was deferred. `boot.TODO.md:165–169`. Earlier Intel-driver-date/boot WUDF correlations were explicitly falsified in `Reports/touchpad-glitch-investigation-20260502/remediation-plan.md:7–11`.
- May 30: accessibility hotkey registry change did not resolve Shift; live accessibility reset was then attempted. `.claude/context/input-stack-freeze-context-20260530.md:45–48`. USB suspend, fingerprint/hub power, crash-dump settings and Process Lasso/GPU preferences are recorded applied in `machine-reliability.TODO.md:10–19`; durable keyboard symptom improvement is not demonstrated there.
- June 6: internal keyboard identified as PS/2 LEN0071; touchpad Sensel SNSL002D on I2C. No recorded input-driver errors does not exclude intermittent input loss. `Reports/input-stack-investigation-20260606/FINDINGS.md:20–46`.
- June 6: targeted touchpad/I2C power-down changes applied with rollback and post-change PnP/power checks. `Reports/workstation-audit-20260606-124859/touchpad-power-fix-20260606.md:3–19,64–75`. This demonstrates applied settings, not durable keyboard/touchpad recovery.
- June 19: user reported USB keyboards always working while internal Shift intermittently failed; later brief controlled traces contained internal and USB Shift events. The addendum first declared shared software excluded, then declared hardware refuted by a short hold and 2/17 versus 3/23 proximity counts. Those conclusions exceed the evidence. `Reports/input-stack-investigation-20260606/SHIFT-RESOLUTION-ADDENDUM-20260619.md:8–34,106–156`.
- September 12: root reports the internal-keyboard complaint remains unresolved. That supersedes historical “resolved/proven” labels.

## Aggregate trace recheck and methodology gaps

Six historical live JSONL files exist for June 19; two are empty, four contain 351,17,670,147 events. In the 670-event capture, internal Shift has 24 DOWN/19 UP with five repeated DOWNs while already held; USB has 52/19 with 33 repeated DOWNs. A per-device/per-side state model closes every Shift hold in that capture. In the 147-event capture, internal has 68 DOWN/1 UP with 67 repeats; USB has 16 DOWN/no UP with 15 repeats and remains held at capture end. This is compatible with typematic and a truncated end; neither demonstrates missing hardware events.

Pre-repair analyzer source incorrectly incremented a merged scalar for every Shift DOWN, labeled all-device totals “Internal,” reconstructed private text, and asserted imbalance meant dropped make/break. The ordering script chose the nearest Shift DOWN within ±300 ms per broad class, including repeats/future transitions; it inferred intended capitals and emitted “FAIL.” Thus the 12%/13% figures are not measured user-visible failure rates, and a small similar pair of proportions does not establish equivalence.

The collector also emits a whole-window verdict if any internal Shift arrived (`Trace-ShiftKeySource.ps1:239–256`), which cannot decide a specific intermittent episode. Its timestamps are user-mode time-of-day (`:162`), not hardware timestamps; it has no trial intent, observed application result or synchronized failure marker. Root owns collector corrections; this lane is authorized to repair the two offline analyzers and behavior tests separately.

`Watch-InputGlitch.ps1:179–198` divides tagged-event counts by first-to-last tagged timestamps rather than observed session exposure. Existing ledger contains only four records, all June 6: two untagged, one touchpad and one Shift. It is not a baseline/post-fix longitudinal trial. CPU snapshots sort cumulative CPU (`:138–139`), so they do not prove load at failure time. Toolkit Pester checks largely prove contracts/structure, not clinical diagnosis of input faults.

## Prioritized hypotheses and discriminators

1. **Application/session input handling or near-simultaneous modifier ordering.** Plausible when internal Shift events arrive during the exact failure. Need labeled short trials and actual application outcome; existing proximity metric is insufficient.
2. **Intermittent internal keyboard/EC/i8042-path failure.** Still open for the user-reported device-specific symptom; a 20.9-second clean hold cannot rule it out. Compare internal/USB during the same labeled episode, separately testing left/right Shift and nonmodifier controls.
3. **System scheduling, display or driver latency during load/resume.** Historical compositor/GPU issues and workload reports justify correlation, not causation. Use aligned load/resume/event evidence and identical labeled trials; changing process priorities does not directly reprioritize kernel HID/I2C interrupts (`AGENTS.md:141–142`).
4. **Device-specific filtering/remapping or accessibility state.** Historical steady-state checks lower likelihood but do not eliminate intermittent/app/device-selective software behavior. Capture current state read-only and tie it to the exact episode; do not infer causation from installed tooling alone.
5. **Separate touchpad I2C/power/firmware issue co-occurring with keyboard symptoms.** Topology warrants separate observation, but different buses do not prove independent causes. Record whether pointer, TrackPoint and external keyboard fail simultaneously before grouping incidents.

## Unresolved questions

What exactly fails now: wrong capitalization, ignored Shift, missing all keys, stuck modifiers or focus changes? Which Shift side, applications, sleep/resume or load pattern? Do internal and USB failures reproduce in the same session under matched trials? Which BIOS/EC/driver versions and policies are current, and which previous changes were actually followed by adequate behavioral observation? No recommendation for firmware flashing, hardware replacement, broad rollback or priority tuning follows from this history alone.

## Authorized offline repair completed in this lane

Replaced `Tools/InputDiagnostics/Analyze-ShiftTrace.ps1` with device-and-side state aggregation and privacy-preserving capture-local labels. Repeated DOWNs no longer add holds; unmatched UP/end-held observations remain ambiguous. No raw typed text/device paths/timestamps or inferred intent is emitted. `Measure-ShiftOrdering.ps1` is now a compatibility entrypoint to the same conservative analysis instead of emitting the invalid nearest-transition failure metric. Both retain optional Path autodiscovery and accept PassThru. The analyzer closes its reader on malformed input and rejects non-object JSONL records without echoing payloads.

`Tests/InputDiagnostics/ShiftTraceAnalysis.Tests.ps1`: 9/9 behavioral tests passed (2026-09-12), covering repeats, two devices/sides, truncated boundaries, empty/modifier-only traces, privacy, malformed input and wrapper parity. Aggregate-only replay of the 670-event historical capture returned two devices, 33 USB/HID and 5 internal repeated DOWNs, no unmatched UPs and no end-held Shift sides. Diagnosis remains undetermined and actual failure rate unavailable. Diff whitespace check passed. No live input capture, system mutation, commit or push.

Historical source-line references above describe the pre-repair review snapshot; the owning root is separately reconciling collector/docs/TODO conclusions.
