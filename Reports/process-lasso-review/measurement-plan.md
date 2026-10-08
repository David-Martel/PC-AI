# Representative capture and benchmark plan

Start with the authorized 60-minute unchanged-policy capture. Then measure three
working days, extending the calendar only when necessary to fill coverage gaps.
Elapsed time alone cannot prove sufficiency; a non-meeting hour is a development
baseline, not a Zoom/Teams baseline. No calls or workload changes are manufactured
automatically. No persistent scheduled task is installed by this review.

## Three-day coverage

| Day | Work and observations |
|---|---|
| 1: unchanged baseline | Capture ordinary agents/Git/build/evaluation and sync overlap; obtain a 15–30-minute quiet window and actual meetings. Keep all policies unchanged. |
| 2: fill baseline gaps | Capture a long call, camera/effects, screen sharing and audio-only periods for each app actually used. Include a useful build/inference/evaluation load during a representative call if normal for the user. |
| 3: matched comparisons | After baseline review, test one selected policy class at a time, using at least three matched A/B repetitions for reproducible tasks. Reverse order across pairs to reduce warm-cache/time drift. Keep untested classes unchanged. |

Target at least three actual Zoom and three actual Teams call sessions if both
are used, totaling at least 90 minutes per application, plus 30-minute sustained
call+throughput overlap, screen-share and audio-only observations, a 30-minute
useful development window and a quiet window. These are planning minima, not a
statistical guarantee; extend when relevant scenarios or variance remain poorly
covered. Do not treat adjacent samples as independent experimental repetitions.
Separate cold/warm builds, different models, and AC/battery modes.

Use separate 60–90-minute captures per scenario/block, up to a full workday's
coverage. New directories prevent evidence overwrite. Default 30-second samples
yield about 120 scheduled observations per hour; an optional 10-second cadence
yields about 360 but needs a fresh overhead check. Actual sample count, overruns,
query failures and early output-budget termination must be checked. Use a
2-second 5–10-minute focused window only when needed and after checking overhead.
The 512-MiB sample budget rotates files at approximately 16 MiB and stops early
if exceeded; it is not rolling evidence deletion. Multiple blocks are needed
for a full day. Retain seven days initially and review space before extension.

## Running captures

From the repository, preview without collecting or writing:

```powershell
pwsh -NoProfile -File .\Tools\Collect-WorkloadResourceProfile.ps1 -DryRun `
  -DurationMinutes 60 -IntervalSeconds 10 -Label baseline
```

Foreground example for a later labeled block:

```powershell
pwsh -NoProfile -File .\Tools\Collect-WorkloadResourceProfile.ps1 `
  -DurationMinutes 60 -IntervalSeconds 30 `
  -Label teams-live-share -OutputPath .\Reports\workload-profiles\teams-live-share-01
```

This conservative example uses 30-second intervals without GPU counters. The
launch receipt is authoritative for the initial baseline's selected interval.
Pilot GPU enumeration materially increased overhead, so enable `-IncludeGpu`
only in a focused capture after a fresh overhead check. No-GPU data is a recorded
coverage gap, not an idle-GPU conclusion.

Use a fresh output directory. `metadata.json` records source/privacy/schema
context; `samples-*.jsonl` streams throughout capture; `summary.json` is written
when the collector finishes. A launch receipt records the actual process identity,
start/expected end and paths. A running capture with growing samples is not a
completed summary. The declared duration excludes initialization and provider
timeouts/cleanup may extend wall time; inspect actual timing, not nominal duration.

The baseline is launched hidden, without loading the interactive PowerShell
profile. It does not restart apps, stop jobs, or change Process Lasso. To end it
early, verify PID and creation time against the receipt, request Ctrl+C through
an attached console when possible, or stop only that owned collector. Forced
termination may leave flushed samples without a summary; preserve them.

Initial baseline: launched 2026-10-05 18:17:27 UTC, PID 154852, 60 minutes,
30-second cadence, GPU counters omitted. The local, untracked `capture-launch.json`
records identity, checked status, output location and exact source/policy hashes.
The no-GPU pilot at 20-second cadence lasted 66.19 seconds with four samples:
own mean CPU 1.126% of host, interval median 0.765%, p95 1.873% (only three known
CPU intervals), p95 collection wall time 4.236 seconds, no gaps above 150%.
This is a feasibility pilot, not proof of sub-1% end-to-end overhead; provider
cost remains separate. The 30-second cadence trades temporal resolution for
less observer work. See [pilot summary](collector-pilot-20s-no-gpu/summary.json).

## Scenario markers and metrics

Record UTC start/end markers for actual calls, camera on/off, effects, sharing,
app/layout/resolution, network path, AC/battery, model/job identity, policy hash
and user-perceived glitches. Record useful outputs: tests/minute, build wall
time, agent task completion, Git operation latency, inference tokens/sec and
time to first token. Process uptime or CPU usage cannot measure useful throughput.
Do not put meeting IDs, participant names, prompt content or secrets in markers.

Collector coverage:

- Process identity (PID + creation time), parent generation, interval CPU,
  private bytes, working set, handle count, and generic process I/O rates.
  Thread enumeration is deliberately omitted from continuous capture; use ETW
  or targeted snapshots when thread-level analysis is needed. The guarded parent
  inventory refreshes once a minute; newer processes can have unknown parents.
- Host busy CPU, DPC/interrupt time, available memory, commit/limit,
  page reads, disk latency/queue and adapter throughput. Optional GPU counters
  at 60-second cadence retain instance/adapter identifiers for later mapping.
- Sample timestamps/intervals, gaps, missing counters, output bounds, own CPU
  and collection wall time; host p50/p95/p99 and process-name rankings.

Limits: provider and process intervals differ; the accounting gap is diagnostic,
not culpability. Short-lived work between ticks is missed. Generic process I/O
includes multiple I/O types and is not a per-process disk/network measurement.
Pages Input/sec includes mapped-file reads, not only pagefile reads. Shared
working sets overlap. Per-name cohorts need process-tree/app context. GPU engine
sums do not mean total GPU percentage. Available-memory lower tails matter more
than upper tails. Percentiles without sample counts and scenario boundaries can
hide call-time failures.

Additional evidence required before promotion:

1. Export timestamped call quality: Zoom Autosave QoS Stats or Zoom Statistics;
   Teams Call health and available tenant analytics. Zoom supports bounded local
   JSON QoS logs; archive sanitized measurements after calls before rotation.
   Enabling this feature remains a proposed application setting change.
   [Zoom QoS logging](https://support.zoom.com/hc/en/article?id=zm_kb&sysparm_article=KB0077003),
   [Teams Call health](https://support.microsoft.com/en-us/teams/meetings/monitor-call-and-meeting-quality-in-microsoft-teams).
2. Add per-disk latency/throughput, TCP retransmits and NIC errors/drops in focused
   troubleshooting; aggregate host counters cannot identify one congested disk,
   WAN path or application. Distinguish physical links, virtual adapters and
   actual call routing. Do not sum virtual+physical interface rates as unique traffic.
3. Obtain GPU LUID-to-adapter mapping, encoder/decoder/3D/copy engine load, VRAM,
   CPU frequency/temperature/power/throttling and display/effects context. The
   baseline's optional GPU snapshots do not supply all of these.
4. Preserve sanitized Process Lasso policy/trim events promptly: current raw logs
   rotate in about ten minutes. Their launch command lines can contain secrets;
   do not upload or copy them into Git. Correlate individual trim events with
   process generation, faults and useful-work latency.
5. Short ETW traces for causes, rather than constant full-detail tracing.

## Focused WPR/WPA investigations

WPR is installed and reported not recording during this review. Its installed
profile list includes CPU, DiskIO, Audio, Video, GPU and Power. Recheck recording
ownership immediately before any trace; do not cancel another owner's session.
WPA is not on PATH in this snapshot; installation elsewhere was not searched.

For a 2–5-minute CPU diagnostic, after verifying no competing recording:

```powershell
wpr -status
wpr -profiles
wpr -start CPU.light -filemode
# Reproduce a bounded interval; mark the glitch timestamp.
wpr -stop C:\Users\david\AppData\Local\Temp\pcai-cpu-trial.etl
```

Commands come from the [WPR reference](https://learn.microsoft.com/en-us/windows-hardware/test/wpt/wpr-command-line-options)
and the local profile list. Start may require elevation; it was not executed.
Use a unique trace filename and private storage with a duration/size limit.
Choose Audio/Video/GPU/DiskIO or custom precise-CPU/DPC/ISR profiles for a specific
question after verifying installed provider coverage. CPU.light alone cannot
prove audio glitches, allocation lifetimes or driver causation.

Inspect ETW event/buffer loss before quantitative analysis. Previous repository
evidence contains a verbose trace with millions of lost events: reject such a
trace for performance claims. In WPA compare CPU stacks and ready/wait times,
DPC/ISR driver attribution, hard-fault backing files, GPU engines and glitch
timestamps. Download matching symbols when appropriate. Traces can contain paths
and sensitive process context; keep raw traces private and publish aggregates.

## Comparison and promotion gates

Preserve the exact before policy, running binary version, active power plan,
workload parameters, warm/cold state and rollback. Change one class only. Compare
three matched A/B pairs initially; expand if results vary or workload matching
is weak. Use at least 30-minute sustained runs for thermal/power claims.

Promote only with no additional audible glitches/freeze bursts, improved or
unchanged call loss/jitter/FPS and UI response, useful-work throughput within
an agreed tolerance (proposed 5%), and no worse commit/refault/disk behavior.
Target own collector CPU below 1% host as an initial engineering goal, and inspect
WMI/provider overhead separately. Own CPU excludes provider-side cost, so compare
a short matched collector-off/on pair before attributing changes to workload.

Report actual duration, sample coverage, missing scenarios, lost events,
uncertainty and differences between app versions. A quiet trace, a reduced
working set, or completion of three calendar days does not establish success.
