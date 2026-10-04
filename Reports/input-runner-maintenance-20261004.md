# DTM-P1GEN7 input and runner maintenance — 2026-10-04

The native keyboard remains unresolved. The user reports both Shift keys and
Down/Left/Right failing while USB input and the touchpad work. This maintenance
fixed task-health inspection and recovered a hung service; it did not prove a
keyboard cure or a general performance gain.

## Task inspection and oversight

`Tools/Test-ScheduledTaskHealth.ps1` previously swallowed an inspection exception
and could report an enabled task Healthy. It now records additive
`InspectionStatus: Unavailable`, a reason and Failed status for enabled,
nonignored tasks. Action results remain unknown rather than fabricated; unavailable
inspection cannot establish staleness. Disabled/Expected precedence, unsigned
result handling and the read-only/DryRun contract are preserved.

Actual-script child fixtures produced a valid baseline of 12 failed / 5 passed,
then 17 regression passes. A separate existing-test run passed 14 cases with zero
skips. Both files have zero parser errors and zero repository analyzer findings.
An independent review found no blocker and checked hashes and XML results; it
did not replay tests. Invalid harness/zero-test runs were excluded. New fixture
children have deadlines; inherited live-test child deadlines remain future work.

The installed `PC-AI Scheduled Task Health` task runs read-only as SYSTEM every
30 minutes, with IgnoreNew, a three-minute limit, no retry and no wake request.
Output is `C:/ProgramData/PC_AI/TaskHealth/scheduled-task-health.json`.
It neither starts nor repairs monitored tasks. Live execution returns gate 1
when real alerts exist. AgentHubRunner and Sam3TestbedRunner retain prior
termination alerts; Bitwarden-Archive-Refresh retains its script-failure alert.
These were not hidden with Expected overrides. A temporary trial restore task
also appeared in one intermediate report and was subsequently removed.
The 19:06 UTC snapshot had three alerts. After normalizing only this new monitor's
script argument to Windows backslashes for LocalOnly ownership recognition, its
19:17 UTC report includes the monitor itself as Healthy and adds genuine exit-1
alerts for Codedev-Worktree-Inventory and VIGIL-DashboardRefresh: 36 tasks,
24 Healthy / 5 Failed / 0 Stalled / 4 Disabled / 3 Ignored. Its completed result
remains gate 1. No existing task action or policy was changed by that normalization.
This monitor covers scheduled-task metadata, not full GitHub/Docker heartbeat
supervision. Rollback removes only this newly installed monitor.

## Input evidence and serial recovery

Three bounded captures recorded navigation/modifier events only. The first
observed USB Shift; the second observed 11 internal Up down/up pairs alongside
USB Shift/Left. Raw-read, malformed-packet, registration and device-name error
counters were zero in these two captures. Missing native events cannot establish
whether each key was physically attempted or localize an electrical fault.
The third capture overlapped Keyboard Manager isolation, observed USB navigation
and Shift with zero collector error counters, but no internal retained events.
The user subsequently replied that the same native keys still failed during the
pause. Exact physical attempt times/counts remain unrecorded; utility exclusion
is therefore not conclusive. Raw device IDs and events are retained privately.

Interhaptics HapticService is distinct from the Sensel touchpad. Its pretrial
sample used about 0.80 logical cores. A graceful stop hung in STOP_PENDING;
the A/B/A trial therefore aborted. After verifying executable, process start,
PID and binary hash, only that hung process was terminated and the automatic
service restarted. Two short subsequent samples had zero CPU delta. The user
still reported native-key failure. Temporary safety tasks were removed.

Keyboard Manager's planned off window was 18:58:13–18:59:44 UTC; the latter is
an end-of-observation marker, not successful restoration. Initial restoration
falsely rejected the unchanged PowerToys parent because JSON parsed its timestamp
as a DateTime while the guard compared it as a string. Recovery compared normalized
UTC ticks and the exact executable, then restored the engine at 19:05:29 UTC.
PowerToys stayed running, its settings hash remained unchanged, and the temporary
rollback task was removed. There is no controlled input-cure claim.

Selected PnP devices have no current problem codes. BIOS 1.22 is current on this
machine; cached offers are not proof an update applies. Serial IO/TrackPoint
versions match their cached offers, and both NVIDIA GPUs currently have matching
healthy drivers. Sensel firmware uses differing version representations that
require vendor interpretation; no firmware reinstall, controller restart or reboot
was performed here. Existing Sensel/parent power-management repairs are preserved.

## Resource and owner boundaries

Selected host samples at 18:32 UTC were about 91% CPU busy, 93.2 GB committed of
106.8 GB, 6.5–6.7 GB available and 7,839–9,269 pages/sec. These demonstrate host
pressure, not keyboard causation. A Docker stats probe timed out; its absent
result is not a usage measurement.

All four sampled runners were healthy, had zero container restarts and zero OOM
kills. Cgroup OOM counters and sampled pressure averages were zero. Historical
memory-cap hits were 96,597 / 114,797 / 219,714 for the secondary/secondary/usability
containers; these are historical reclaim/cap evidence, not current OOM proof.
Their aggregate CPU ceilings total 19 against an 18-vCPU guest. The primary TPM
job remains in owner custody; no quota, container, WSL or restart policy changes
were made. Secondary listener generation changes need owner investigation.
Process Lasso safety validation passes and existing UI/Docker/WSL priorities remain.

Coordination used the authoritative ASUS hub over verified SSH. The TPM owner
reported cleanup of its own preview/probe processes. The earlier run 37224919165
failed a PDF-test path assertion; its replacement run 37226398936, job
111506913451, was confirmed active in the owner's 19:02 UTC reply and remains
held. Other agents' worktrees, active clients, data/volumes and
deployment holds remain preserved. Local hub reachability does not establish
fleet authority. This report does not claim remote deployment/HIL closure.

## Next actions and evidence custody

1. Run Lenovo UEFI keyboard tests near the failure and inspect key travel and
   keyboard cable/latch/assembly condition as indicated. Keep hardware/EC and
   Windows paths open until matched evidence separates them.
2. Resolve the five current task alerts with their owners; retain the actual gate.
3. After job release, review runner concurrency and collect comparable pressure
   samples before making tuning/performance claims.
4. Preserve outstanding fleet/TB/HIL and native git-guard qualification work;
   the broader git-guard suite remains 266 pass / 20 fail / 2 skip.

Private evidence root:
`C:/Users/david/.cache/pcai/input-runner-maintenance-20261004`.
Key receipts are `task-health-fix/receipt.json`,
`input-task-health-independent-review.json`, `input-phase-receipt.json`,
`P2-resource-receipt.json`, `P3-coordination-receipt.json`,
`P4-haptic-service-recovery.json`, `P4-keyboard-manager-trial.json`,
`P4-input-capture-projections.json`, `P4-task-health-monitor-install.json` and
`P4-runtime-latest-monitor.json`.
Runtime follow-up and publication receipts are kept there; secrets, raw events
and private device identifiers are excluded from this public report.
