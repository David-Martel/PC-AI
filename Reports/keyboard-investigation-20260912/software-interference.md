# Software interference follow-up — September 12, 2026

**Unresolved:** Shift recovered locally and in RDP, then Lenovo Right Arrow failed
in both while USB Right Arrow worked. The latest user report says Lenovo Left,
Down and Right all fail. No new matched USB result was supplied for Left/Down.
Repeated Windows reboots over months have not resolved the issue.
Down subsequently recovered after a user-reported hard press, increasing the
priority of [physical inspection](hardware-review.md) without proving causation.

## Software findings

The OneDrive and local Documents PowerShell loaders reach the canonical
`~/.config/powershell/Microsoft.PowerShell_profile.ps1`. A bounded reachable-script
audit found no global keyboard hook, input injection, accessibility setter or
scancode-map write. This does not exonerate compiled modules or helpers already
running. A new NoProfile shell cannot remove another process's hooks.

Confirmed profile defects include the ignored accelerator opt-out, contradictory
interactive command classification, a guard that blocks reload and survives partial
initialization, and a bootstrap marked loaded before its file is found. These
undermine startup reliability and profile experiments; none is an established cause
of missing native-keyboard input. See the validation report for repair status.

The installed Lenovo accessory/display HookDll has a global low-level keyboard
hook. It synchronously messages a helper window before forwarding. Start overwrites
a hook handle and target in shared writable PE data without lifecycle synchronization;
Stop ignores the unhook result and clears neither field. These are implementation
risks, not proof of this fault. The reviewed callback has no explicit key/device
suppression and forwards every normal return. Shared values in two processes do
not establish two registrations. No vendor binary was modified or code injected.

Logitech Options+, UltraslimOSD, Splashtop and WSL's remote-client binaries expose
input-related APIs; imports do not prove active interception during a failed press.
Logitech Flow was enabled with keyboard linking disabled. Stored action encodings
and runtime activation remain unverified. PowerToys was absent during later failure
checks despite the earlier running snapshot.

## Restored isolation trials

All times are local. Startup modes were preserved and every tested utility restored.

| Component | Confirmed stopped interval | Outcome and limit |
| --- | --- | --- |
| Lenovo accessory/display service and hook helpers | 18:22:53–18:25:53 | Neither Shift reportedly worked; individual press times unmarked. Trace had zero modifiers including USB controls, so is inconclusive. |
| UltraslimOSD | 18:32:21–18:34:10 | User reported no improvement. Earlier immediate-stop/restore race excluded as invalid trial. |
| Logitech updater service, agent and app broker | 18:35:28–18:38:59 | User reported continuing failure; interactions and persistent state remain possible. |

A temporary collector observed six internal Up DOWN/UP pairs and one Alt pair at
18:36:21–18:39:21, with zero Shift. Another capture at approximately
18:50:28–18:54:28 contained 24 internal left-Shift events (22 DOWN, two UP), with no
arrows or USB control. Repeated DOWN events are not verified separate physical
presses. The later report that Left/Down/Right fail arrived after this trace ended;
it cannot establish a captured failed arrow trial.

The observed mstsc process started at17:57:35, after the initial17:47–17:54 failure
captures. Its launcher/settings had no discovered keyboard override or injection.
The remote backend was not verified because read-only SSH probes timed out. The
explicit local failure means RDP cannot be the sole explanation.

## Next discrimination

Use short labeled local native/USB trials with the navigation collector, recording
intended action and actual app outcome alongside device, foreground PID, monotonic
time and capture health. Other navigation keys expose possible translation. Missing
events without working controls remain inconclusive. A focused application harness
and prospective hook tracing remain open. Investigate the physical assembly and
connectors against the exact-model service documentation; never infer a shared
electrical matrix row from physical key adjacency alone.

Raw captures, identifiers, profile backups and disassembly remain local under
`.codex/research/keyboard-interference-20260912/` and
`.codex/research/keyboard-profile-fixes-20260912/`. Only reviewed findings and
aggregates are published.

References: [callback timeout](https://learn.microsoft.com/en-us/previous-versions/windows/desktop/legacy/ms644985(v=vs.85)),
[SendMessageW](https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-sendmessagew),
[Unhook result](https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-unhookwindowshookex),
[CallNext handle contract](https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-callnexthookex).
