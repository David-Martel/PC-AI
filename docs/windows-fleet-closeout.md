# Windows fleet source closeout — 2026-10-11

This ledger distinguishes reviewed source, installed tooling, CI acceptance and
physical qualification. Generated machine evidence remains local under Reports;
it is not included in publication. Refresh GitHub state before acting on any
dated PR disposition here.

## Completed and validated source

The Windows fleet branch integrates upstream main at `b92e5f6` through merge
`04b5b8d`, retains the earlier setup-analyzer fix, and includes fleet tooling
commit `3369b72`. It adds machine profiles, read-only hardware/media discovery,
nullable native measurements and SMART capability honesty, routing contracts,
safe installer/release staging and corresponding tests. Previous qualification
receipts record 62 Rust core tests and 12 managed ABI/measurement tests; the
PowerShell suite receipts overlap and must not be summed as unique coverage.

The public profile and crash collector are a separate reviewed source group.
The collector writes explicitly requested bounded local telemetry; it does not
register tasks or mutate drivers/power policy. The profile workaround handles
PSReadLine console-input loss and does not explain whole-PC freezes. Unit probes
must not invoke the operator's private canonical profile.

Existing source-staging and installed-component checks do not mean the complete
source release is installed. The old umbrella/bridge remains a separate runtime
provenance concern. Compiled managed clients must rebuild for nullable metric
getter signatures. Missing input-power/SMART/temperature readings remain unknown.

## Git and pull request disposition

| Work | Current acceptance boundary |
|---|---|
| Windows fleet/profile branch | Publish reviewed source with focused tests; merge only after required final-head checks. Main then fast-forwards and owned merged branches can be removed. |
| PR182 runtime integration | Initial audited head `71ed3e9` had 21 PowerShell failures and 64.07% coverage versus the unchanged 85% target. Its owner concurrently published the 24-name admission repair in `72ee00d` and advanced to `24ae406`; refreshed exact-head CI remains required. Our tested alternate proposal was archived locally rather than overwriting that work. |
| Fleet plus PR182 | Shared Common configuration, workspace metadata and installer changes require actual conflict resolution and mixed-suite validation. A textual merge preview is not a qualified final merge. |
| Candle PR146/148/162 and coordinated draft PR156 | Choose one compatible integration path and test media/CUDA consumers; individually green default-feature CI is insufficient. PR182 already carries a coordinated update, so these must not be blindly stacked or closed before accepted integration. |
| mistral.rs PR145 | Compile/test the enabled backend and relevant runtime/FFI paths; default CPU CI omits this optional backend. |
| PC_AI worktrees | The primary checkout is the fleet branch. The temporary owned PR182 test worktree was retired after preserving its tested proposal and verifying the owner's equivalent source repair upstream. No foreign worktree was removed. |
| VIGIL-Eko worktree | Claude's locked worktree belongs to a live process; ownership acknowledgment and merge are required before the owner removes it. |

Only two machine-evidence case directories are excluded locally through
`.git/info/exclude`; source/tests remain visible to Git. Exclusion preserves
evidence in place and is not deletion. Preservation stashes may be retired only
after their source changes are reviewed, committed and backed up.

## Remaining development and qualification

PR183's first hosted run identified an installed-FFmpeg assumption in a smoke
test and a Rust core test-module ordering lint. The fixes make only actual
FFmpeg absence an explicit prerequisite skip, retain deterministic absence
coverage, and align native Clippy with that workspace's Rust 1.95 MSRV. Local
core format/lint and 62 tests pass. The PowerShell report shows 51.83% against
the unchanged 85% target on both successful main and PR183, with the same
coverage denominator. Coverage selection and enforcement need a separate audit;
the target is not claimed as met or weakened to accept this branch.

The current backlog still includes native ABI/schema versioning, full-stack
cancellation/structured errors, streaming/progress, cold-import cost and batched
search benchmarks, media fixture/performance expansion and the Windows media
server's CRT configuration. Historical TODOs can include stale states: the live
PR/CI and source audit takes precedence over an old checkbox or report.

For VIGIL-Eko, the existing decoder/replay evidence is separate from Windows
bond/notify/reconnect reliability, Linux live BLE, durable multi-site capture and
Dell/pose placement acceptance. See [integration plan](vigil-eko-integration.md).
Agent-bus services are unavailable on this host; a local coordination request has
been left for Claude, with acknowledgment pending. No physical HIL acceptance,
patient recording or medical-model validation follows from this source review.

The NUC's original-power-adapter comparison and hang/offline memory evidence are
still required to discriminate physical power, firmware/driver and hardware
causes. Maintain bounded monitoring and avoid treating Kernel-Power 41 aftermath
as proof of a particular cause.
