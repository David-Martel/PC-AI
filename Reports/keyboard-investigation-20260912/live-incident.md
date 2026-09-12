# Live keyboard incident — September 12, 2026

## Reported symptom and observation

During the live session the user reported both Lenovo Shift keys failing while USB
Shift worked, then confirmed both Lenovo Shift keys still failed. Earlier in the
session the user also reported failures beyond Shift. Do not generalize this
particular capture to every historical episode.

Read-only collection preserved the episode without resetting accessibility state,
stopping applications, restarting devices or installing drivers. The initial
120-second modifiers-only capture contained zero events, including zero USB events,
and is inconclusive.

A subsequent 300-second modifiers-only capture started at 17:50:37 local. Its
17:51:57 analysis observed 24 Shift events: three left-Shift press/release pairs and
nine right-Shift pairs, all from one HID device. Exact raw-interface-to-PnP
correlation matched that device to `kbdhid`. No internal modifier events were
present in that observation. The completed JSON at 17:55:37 confirms the same 24
USB Shift events, zero internal Shift events and zero other modifier events.

An overlapping 90-second all-key capture, 17:53:07–17:54:37 local, recorded **199
non-Shift events from the internal keyboard and zero left/right Shift events**.
Exact interface correlation matched it to `i8042prt`. The completed JSON confirms
the count, with no OTHER/UNKNOWN devices; it contains no USB events in that narrower
interval. The user was instructed to use fixed test input; raw key data remains
local and is not reproduced here.

This establishes that the observer received internal-keyboard events during the
reported failure window, alongside a separate working USB Shift observation in the
overlapping collection session. It strengthens selective missing-Shift localization
at Raw Input. The individual attempts were not time-marked with expected counts,
and no synchronized focused-application observer ran. It does not establish a
hardware-versus-driver cause or a measured failure rate.

## Failure-window state

- Keyboard PnP devices reported OK with problem code zero; class filter was
  `kbdclass`, with no additional instance filters in the snapshot.
- Live FilterKeys, StickyKeys and ToggleKeys were disabled. No Scancode Map was
  present in the inspected keyboard-layout registry key.
- PowerToys process count was zero at 17:49:26, unlike the earlier baseline where
  Keyboard Manager was running. Saved settings still enabled Keyboard Manager with
  empty mappings. This weakens a currently running PowerToys-hook explanation if
  the same failure persisted; no application was stopped by this agent.
- No matching events appeared in the snapshot's bounded System-event query. That
  query is selective, not exhaustive evidence of healthy input delivery.
- Collector and input desktops were both Default in WinSta0, session 1. An
  independent Raw Input enumeration exposed one internal and three HID keyboard
  interfaces; no generic PS/2 exclusion was found in the collector.

## Next discrimination and capture corrections

Prioritize the native keyboard/connection/EC and native driver path while retaining
device-specific software effects. A reproduced UEFI failure would strengthen
hardware/firmware localization; a short UEFI pass would not exclude intermittency.
Use paired, labeled internal A/Ctrl/left-Shift/right-Shift and USB trials during a
future window, with actual app results and recovery markers.

Before interpreting missing events more strongly, add raw-read error counters,
device-name lookup health/retry, monotonic timing and explicit trial markers.
The collector silently skips raw-read errors and can cache unknown device names.
Its final pump may add events absent from live JSONL; compare completed JSON.
No unknown devices or final-count discrepancy appeared in the completed all-key
capture above, but the missing health counters remain a limitation.

Local evidence directory: `Logs/input-diagnostics/live-incident-20260912-174741/`.
Only this reviewed summary belongs in the PR; input captures remain ignored.
