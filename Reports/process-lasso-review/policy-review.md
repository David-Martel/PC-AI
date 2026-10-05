# Process Lasso live policy review — 2026-10-05

Read-only review. No live setting, service, task, process priority, affinity, or existing repository file was changed. The adjacent `policy-snapshot.json` retains configuration keys and sanitized event aggregates; credential fields and raw command lines are excluded.

## Evidence boundary

- Configuration: `C:\ProgramData\ProcessLasso\config\prolasso.ini`, 22,676 bytes, modified `2026-10-03T07:22:13.9475193Z`; SHA256 `DCA0CEE62474BCCF7D4C1D8F55F62A609BBD3E09A77DF0B591DDC97C02075FA0`.
- Governor: PID 68312, `C:\Program Files\Process Lasso\processgovernor.exe`, file version 18.4.1.3, running/responding, launched October 2 at 13:53:36 EDT. Governor owner is david. GUI PID 66440 uses the same version. No explicit configuration argument appeared in the governor launch arguments; the reviewed common configuration is the repository default path, and live rule actions are consistent with it. An open-file handle check was not performed, so the process's exact loaded configuration path/revision is not conclusively proven.
- No matching Windows service was returned. Scheduled tasks `Process Lasso Core Engine Only` and `Process Lasso Management Console (GUI)` were Running; `PC-AI Process Lasso Governor Watchdog` was Ready. Do not assume a service restart applies to this launch arrangement.
- The persisted log snapshot contains 5,058 events between 13:51:28 and 14:01:01 EDT, October 5. Eleven rotating files retained only about 9.6 minutes. Files can rotate during reading; this is an observed coverage window rather than a transactional archive.
- Prior reports and `boot.TODO.md` explain historical changes; their old measurements do not prove present performance or causality.

## Current rules with practical implications

| Observed evidence | Implication and proposal |
|---|---|
| `SmartTrimIsEnabled=true`, `SmartTrimWorkingSetTrims=true`, `SmartTrimAutoMinimumRAMLoad=80`, `MinimumProcessWSSInMb=256`, `SmartTrimIntervalMins=3` | Frequent working-set trimming is active under memory pressure. Record page reads, process faults, commit, disk latency and useful work completion around every trim. Compare a matched SmartTrim-off window before assuming trimming helps. |
| `SmartTrimClearStandbyList=false`, `SmartTrimClearFileCache=false`, `DefaultMemoryPriorities=` | No blanket standby/file-cache purge or explicit memory-priority rules were observed. Keep that distinction: resident working-set reduction is not evidence of reduced private allocation or reduced commit. |
| `node.exe`, `cargo.exe`, `rustc.exe`, `cl.exe`, `link.exe`, `sccache.exe`, Docker/WSL processes and Redis have `below normal` CPU defaults; their I/O rules are `1` | These are blanket executable-name policies inherited from boot/UI contention work. They combine useful agents/MCP servers, builds, inference guests, caches, and background jobs. Define workload roles first; measure throughput and call quality before retaining or widening these demotions. Preserve trusted agent/build work as productive use. |
| `nvidia broadcast.exe,below normal` and I/O `1`; same treatment for `nvfvsdksvc_x64.exe` | Broadcast can be on the live microphone/camera path. During calls, verify the active process tree and restore Normal scheduling for confirmed media processing in a trial profile. Avoid treating it identically to overlays/update helpers. |
| `audiodg.exe`, shell/input processes, Lenovo services receive `above normal` CPU and I/O `3`; many are excluded from ProBalance and SmartTrim | Audio/input protection is intentional, but broad permanent OEM/shell elevation needs evidence. Keep audio-critical protection; measure vendor utilities individually before assuming every helper merits Above Normal. This does not reprioritize kernel interrupt/DPC work. |
| Zoom: SmartTrim excluded; I/O `3`; GPU priority `2`; `EfficiencyMode=...zoom.exe,0`. No Zoom CPU default, ProBalance exclusion or explicit affinity was found | Zoom has partial protection. A call can continue while its window is background; foreground-only ProBalance protection does not fully express a live-call role. Add call-aware trial protection only after the process tree and call symptoms are measured. |
| No `ms-teams.exe`, `teams.exe`, OBS or generic encoder rule found in priorities, trim protection or EfficiencyMode | Teams and other real-time media workloads are not represented. Build an explicit verified allowlist for live media. Do not globally elevate all `msedgewebview2.exe` processes: WebView2 is shared by many unrelated applications. |
| `DefaultAffinitiesEx=rust-analyzer.exe,0,2-9;20-21,...` and cloud/sync processes constrained to `12-21`; `CPUSets=` | Hard affinity exists for rust-analyzer and sync, while no CPU Sets rules exist. Capture actual core topology before labeling logical IDs P/E/LP-E. Compare default scheduler placement or soft CPU Sets in controlled trials rather than adding more hard masks. |
| `OocOn=true`, system threshold `65`, process threshold `10`, quota duration `2000`, `TameOnlyNormal=true`, foreground/foreground-child/service exclusions enabled, EfficiencyMode restraint off | Above/Below Normal defaults already bypass normal-only ProBalance targeting. Large exclusions also cover all Python, PowerShell, uv and virtualized workloads. Productive roles should be explicit; blanket exclusions are not a memory budget or concurrency control. |
| `StartWithPowerPlan=Balanced`, `GamingModeEnabled=false`, `EnergySaverEnabled=true`, `LoadBasedPowerSwitcherEnabled=false` | The configured gaming target `Bitsum Highest Performance` is inactive. Power-plan recommendations require active-plan, AC/battery, thermal and throughput data. Do not infer that a configured target is currently applied. |
| `SamplingEnabled=false`, `SamplingIntervalSeconds=900`, `IncludeCommandLines=true`, process-launch/termination logging on | Existing retention is too short for representative call/build/sync episodes; 15-minute sampling would also miss transients if merely enabled. Use the bounded sanitized collector and retain policy events separately. Avoid copying raw launch logs because their command-line column can contain sensitive arguments. |

## Live trimming and scheduling evidence

The sanitized logs show actual trim events, not just configured intentions:

| Local time | Process | Reported resident working-set change |
|---|---|---|
| 13:52:56 | git.exe | 273 → 1 MB; 272 MB described as restored |
| 13:52:57 | node.exe | 309 → 0 MB; 309 MB described as restored |
| 13:52:58 | rustc.exe | 419 → 95 MB; 324 MB described as restored |
| 13:56:00 | Dropbox.exe | 327 → 4 MB; 322 MB described as restored |
| 13:56:01 | node.exe | 2,158 → 250 MB; 1,907 MB described as restored |
| 13:59:03 | node.exe | 268 → 64 MB; 204 MB described as restored |

Docker backend was also trimmed from 262 to 99 MB. MsMpEng trim records restored 0, 7, and 4 MB. These records justify investigating refault/latency/throughput costs. They do not identify the node workload or prove an adverse effect, a leak, or waste. Associate each event with a PID/start-time identity and workload role during longer collection.

The persisted snapshot contains 2,531 launch and 2,464 termination events, versus 23 Efficiency Mode events, 10 trims and 3 SmartTrim completion events. Launch churn dominates retention. Many events concern conhost, git, pwsh and bash: the user explicitly considers agent/git/evaluation work valuable, so process counts alone must not be classified as waste. Measure parent families, repeated startup overhead and opportunities for worker/module/session reuse while preserving useful work.

## Tooling inconsistencies to address before applying new policy

1. `Tools/InputDiagnostics/Repair-ProcessLassoTerminalGpu.ps1` describes entries `windowsterminal.exe,0` and `pwsh.exe,0` as enabled EcoQoS/E-core pinning and removes them. Current Zoom and Chrome events explicitly say `Efficiency Mode OFF` when their rule value is 0, and the overlay/snapshot code names this `disableEfficiencyMode`/`efficiency_mode_off`. The repair's interpretation and the corresponding May 30 history in `boot.TODO.md` are reversed. Removing an OFF override gives Windows/application policy control; it does not prove EcoQoS was disabled. Correct the comments/history with provenance before reusing the repair.
2. That repair also claims auto GPU preference guarantees iGPU UI rendering, and high-performance preference names the eGPU. Actual adapter choice must be measured per GPU engine/LUID; a preference value alone does not prove either physical adapter selection.
3. The repair creates directories/transcripts and writes `last-run.txt` even during `-WhatIf`; it lacks the repository-required `-DryRun`, `-h` and `--help` contract. `Apply-ProcessLassoUiSyncTuning.ps1 -DryRun -ReportPath ...` intentionally writes the optional report, also contrary to the repository's strict no-file-write dry-run requirement. Neither script was run in this review.
4. `Apply-ProcessLassoUiSyncTuning.ps1` and `Test-ProcessLassoBootSafety.ps1` require command-line logging and many event categories on. A privacy/retention improvement must update those expectations together or the old script will re-enable logging.
5. `Tools/Invoke-ProcessLassoAnalysis.ps1` takes one counter sample and the top 15 private-memory processes. Its paging recommendation uses broad `Pages/sec` thresholds; that cannot identify pagefile I/O or process culpability by itself. It compares raw event counts over a configurable lookback with a threshold named per hour without normalizing elapsed/retained time. `Get-ProcessLassoSnapshot` fallback reads only the current log, so its nominal 120-minute lookback can contain less than one minute under current rotation. The existing analysis is a point-in-time aid, not representative profiling.
6. `Config/process-lasso.ai-dev-workstation.json` already says builds/AI should allow throughput and exclusions should remain minimal. Its role intentions conflict with the much broader live executable-name exclusions and demotions. Treat its 64-GB/22-logical-processor machine hints as a configuration hint until refreshed by inventory.

## Proposed sequence

1. Preserve this policy hash and collect unchanged policy over multiple labeled live Zoom/Teams episodes, useful agent/build/evaluation runs, sync bursts and quiet baselines. Separate call-on/call-off periods and AC/battery modes. Record missing coverage; the rolling Lasso logs alone cannot supply it.
2. Use per-workload measurements to separate justified throughput from avoidable overhead. Candidates are duplicate telemetry/updaters, redundant process startup, unnecessary background browser activity and sync contention; process names and RAM size alone are insufficient proof.
3. Prepare reversible trial profiles: (a) SmartTrim disabled during active calls/useful agent-build work; (b) live-media protection with Normal CPU as baseline, explicit trim exclusion and Efficiency Mode OFF where verified useful; (c) role-specific background sync/updater deprioritization while restoring useful shared services to Normal. Keep Real Time and blanket High priorities out of the proposal.
4. Evaluate each change independently against a matched baseline: audio glitches/dropped frames and call send/receive quality, UI latency, build/test/agent completion time, process page-read pressure, disk queue/latency, memory headroom and collector overhead. Compare before/after with useful throughput intact; do not declare success from a reduced working-set display.
5. After selecting validated trials, reconcile scripts, tests, role profile and historical documentation so routine boot safety tooling does not reintroduce conflicting defaults. Keep live policy changes a separate explicit deployment from this review.

No performance improvement is claimed by this read-only review.
