# Keyboard diagnostics validation — September 12, 2026

Later live-session evidence is recorded in [live-incident.md](live-incident.md).
The initial validation boundaries below describe the earlier tooling/research pass;
the follow-up captured device-correlated input during a user-reported failure.

## Completed

- `Tests/InputDiagnostics/InputDiagnostics.Tests.ps1` plus
  `Tests/InputDiagnostics/ShiftTraceAnalysis.Tests.ps1`: **96 passed, zero failed,
  skipped or inconclusive**. Nine are behavioral analyzer regressions; the existing
  suite primarily checks script contracts and syntax.
- `Tests/InputDiagnostics/KeyboardDiagnosticSnapshot.Tests.ps1`: **26 passed**,
  covering probe outcomes, dry-run/help, bounded event query behavior, output safety,
  privacy, device correlation and the observed input-software candidates. Seven
  regression cases first reproduced unsupported PowerToys schemas being treated
  as known-empty; those now return unavailable rather than zero mappings.
- Repository-configured PSScriptAnalyzer: zero findings for the new helper and
  changed analyzer/capture/reset scripts. No device-mutating script was invoked.
- Live helper: seven probes succeeded, recent Events was explicitly empty over the
  requested 120 minutes. It returned 23 relevant processes, four keyboard interfaces,
  four driver records and five class/instance-filter records. Total process runtime
  was approximately 5.45 seconds in the root invocation; this is one observation,
  not a performance guarantee. Local evidence is `current-snapshot.json`.
- Live accessibility API state: FilterKeys, StickyKeys and ToggleKeys all disabled;
  FilterKeys wait, delay, repeat and bounce were zero. PowerToys active profile was
  configured, manager enabled, key/shortcut/text remap counts all zero.
- A three-second default modifiers-only Raw Input smoke compiled, registered and
  exited; zero events were captured. It validates startup, not keyboard delivery.
- Windows Update driver search succeeded with no visible pending driver offers;
  model-specific Lenovo BIOS/EC versions match installed firmware. See
  [driver research](driver-research.md) for applicability and limitations.

## Boundaries

No physical failing trial, matched software A/B/A, out-of-Windows key test, WPR
failure trace, driver replacement, firmware flash, service change or reboot occurred.
Diagnostic tests validate the tools; they do not establish a root cause or cure.

The snapshot excludes key contents, event messages, remap values, command lines,
raw device identifiers and hardware serials. Tokens correlate devices locally.
It reads installed driver registry metadata; driver signature verification is
explicitly `not-queried`. The separate research lane checked signed-driver inventory.

Machine-specific JSON/capture artifacts remain local evidence rather than PR content.
The unrelated pre-existing process-lasso watchdog JSON change remains preserved.
