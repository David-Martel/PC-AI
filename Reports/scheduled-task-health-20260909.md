# Scheduled Task health — dtm-p1gen7, 2026-09-09

Baseline captured with `Tools\Test-ScheduledTaskHealth.ps1`. Machine-readable
companion: [`scheduled-task-health-20260909.json`](scheduled-task-health-20260909.json).

```
pwsh -File Tools\Test-ScheduledTaskHealth.ps1 -OutputJson Reports\scheduled-task-health-20260909.json
```

| Scope | Total | Healthy | Failed | Stalled | Disabled | Ignored |
|---|---|---|---|---|---|---|
| All non-Microsoft | 115 | 85 | 18 | 4 | 5 | 3 |
| `-LocalOnly` | 35 | 24 | 5 | 0 | 3 | 3 |

`-LocalOnly` is the actionable view and the form a watchdog should gate on: it
keeps tasks whose action runs something we own and drops vendor churn (Lenovo,
NI, OneDrive, npcap), which fails constantly and drowns the signal.

## Why this baseline exists

Task Scheduler has no health surface. Every failure below was invisible until
exit codes were enumerated by hand — `AgentHubRunner` had been dead for five
weeks and `ProfileLogSync` had failed hourly since February, both silently.

## Local failures at capture time

| Task | Status | Last result | Last run |
|---|---|---|---|
| `AgentHubRunner` | Failed | `0x41306` terminated | 2026-08-04 12:02 |
| `Sam3TestbedRunner` | Failed | `0x41306` terminated | 2026-08-04 10:04 |
| `RmeRunner` | Failed | `0xC000013A` killed | 2026-09-09 09:48 |
| `AutoMount_VHDX_shared-dev` | Failed | `0x33` resource not available | 2026-09-09 09:49 |
| `Gemini-CLI-Update-stable` | Failed | `0x1` generic script failure | 2026-09-09 08:05 |

**`AgentHubRunner` carries no trigger at all** — it cannot self-start, which is
why five weeks passed unnoticed. Note that `agent-bus health` passes: the
AgentHub *service* is healthy and is a different thing from this GitHub Actions
*runner*. Any workflow targeting the runner queues indefinitely. Deciding how it
should start (service vs. boot trigger) is a CI-architecture call, so it is
recorded here rather than changed.

`AutoMount_VHDX_shared-dev` is expected until the 8 TB enclosure returns — see
[`F-DRIVE-BOOT-DIAGNOSIS-20260909.md`](../docs/F-DRIVE-BOOT-DIAGNOSIS-20260909.md).

## Fixed in this workstream

| Task | Was | Now |
|---|---|---|
| `ProfileLogSync` | `0x1` on every run since **2026-02-21** — an abandoned zero-byte `.sync.lock` that `CreateNew` could never get past | `0x0` through Task Scheduler; stale locks are now broken on age |
| `CloudCache-MountWatchdog` | registered with boot+logon triggers *after* boot, so its repetition never armed — `NextRunTime` empty, no repeat until reboot | armed at install; observed firing 12:43:01 `0x0`, next 12:57:59 |

## Reading the classification

Two independent axes, because a task can be green on one and dead on the other:

- **Result** — the action's last exit code. Success, the documented `SCHED_S_*`
  informational codes, and everything else. A nonzero action exit is a failure
  deliberately: a script's `exit 51` is the signal that was being lost.
- **Staleness** — whether it ran when its own triggers say it should have. A
  boot/logon task whose last run predates `LastBootUpTime` is stalled *even if
  its last result was 0*. This axis is what catches a task that quietly stopped
  firing.

The tool fails **closed**: if Task Scheduler cannot be enumerated it exits 2 and
refuses to report health, rather than returning an empty set that would read as
"all healthy".
