# Process Lasso, live media and useful throughput proposal

Reviewed 2026-10-05 on DTM-P1GEN7. This is a configuration review and experiment
proposal, with initial measured evidence. No live priority, affinity, memory,
power, network, service, or task policy was changed. The user selected a bounded
60-minute baseline now and an initial three-workday measurement plan.

## Findings that matter

The first priority is to protect useful resident memory and time-sensitive
media while reducing avoidable overhead. Blanket demotion of runtimes cannot
distinguish useful work from waste. More aggressive priorities cannot create
memory capacity or fix a saturated network or misbehaving driver.

Initial evidence is in [architecture-snapshot.json](architecture-snapshot.json),
[initial-interval-profile.json](initial-interval-profile.json),
[fractional-interval-profile.json](fractional-interval-profile.json), and
[policy-snapshot.json](policy-snapshot.json). These are different observation
windows and must not be treated as one atomic measurement.

| Observation | Interpretation |
|---|---|
| Core Ultra 7 155H, 16 cores / 22 logical processors; approximately 63.5 GiB installed RAM | A hybrid mobile CPU: throughput, latency and sustained thermal behavior all matter. Logical CPUs are not equal-performance cores. |
| Thirteen host samples over about 71 seconds: CPU mean 81.85%, range 55–91%; available RAM 5.00–5.63 GiB; commit 86.15–87.15% | Real concurrent load and limited headroom, but not a representative call baseline. |
| Earlier instantaneous commit approximately 95 GiB / 110 GiB limit; pagefile approximately 27.3 GiB used, 39.8 GiB peak | Pagefile occupancy is not current paging rate. Page reads and backing-file attribution are needed. |
| SmartTrim enabled: 80% RAM-load trigger, three-minute interval, 256 MB process threshold | Live logs show Git 273→1 MB, rustc 419→95 MB and Node 2,158→250 MB resident working-set reductions. Useful work may refault; throughput benefit is unproven. |
| Node, build tools, Redis, WSL/Docker and NVIDIA Broadcast have Below Normal CPU / Low I/O rules | Rules mix active agents, shared services and media processing with background work. Broadcast's live media path deserves specific treatment. |
| Zoom has trim protection and I/O/GPU/Efficiency rules; comparable Teams rules are absent | Protect verified call process trees, including relevant Teams WebView children. A running app is not evidence of an active call. |
| Logs retain only about 9.6 minutes; launch/exit events dominate; built-in sampling disabled | Historical lookback arguments cannot recover overwritten evidence. |

The CPU specification has six P-cores, eight E-cores and two LP E-cores according
to [Intel's specifications](https://www.intel.com/content/www/us/en/products/sku/236847/intel-core-ultra-7-processor-155h-24m-cache-up-to-4-80-ghz/specifications.html).
Local logical-processor numbering was not mapped to these core types. Existing
hard affinity masks therefore need topology verification before reuse.

The machine has Intel graphics, an 8-GB-class RTX 2000 Ada laptop GPU, a
16-GB-class RTX 5060 Ti, and a virtual display adapter. Their memory is separate;
do not plan around a combined 24-GB VRAM pool. NVIDIA's inventory observed both
devices, with little laptop-GPU allocation and about 686 MiB on the 5060 Ti at
one instant. Its laptop power reading was implausible and is excluded. GPU
placement, per-engine load, display routing and sustained temperatures remain
measurement gaps. Multiple physical SSDs and VHD-backed volumes are present;
physical and virtual-disk metrics must be distinguished.

An additional read-only inventory found no IFEO `PerfOptions` priority entries
in the native or 32-bit registry views. Battery status reported AC power and
95% charge. The IPv4 default route selected Ethernet 8 (1-Gbps link), while
Ethernet 15 (5-Gbps link) was also up. The latter's presence does not establish
that Internet calls use it. See
[additional-policy-snapshot.json](additional-policy-snapshot.json).

WSL is configured for 32 GB, 18 processors and 8 GB swap, with `dropCache`
reclamation; the snapshot shows about 15.9 GiB private / 13.7 GiB resident.
These are configured limits and observed allocations, not 18 reserved physical
cores or proof of waste. Do not restart WSL during active jobs. A future measured
comparison of workload budgets and `gradual` reclamation can balance Windows
headroom with repeated-build cache reuse; Microsoft documents both reclamation
modes in [WSL configuration](https://learn.microsoft.com/en-us/windows/wsl/wsl-config).

## Concrete policy proposal

First collect the current policy unchanged. Then compare one change at a time.
These are proposed trial settings, not deployed settings or promised gains.

| Workload | Proposed baseline and trial |
|---|---|
| Live Zoom/Teams, active OBS/encoders, audio processing, NVIDIA Broadcast | Start with application/Windows CPU defaults. Trial removal of Broadcast's blanket demotion, protection from SmartTrim, and narrowly scoped ProBalance/Efficiency protection during confirmed calls. Test Above Normal only if measured scheduling delay correlates with glitches. No blanket High/Realtime boosts. |
| Teams WebView2 / browsers used for live media | Identify descendants by PID plus creation time and preserve unrelated applications' defaults. Do not globally elevate or disable efficiency for every WebView/browser process. |
| LLM agents, inference, Git/GH, evaluation, builds and MCP services | Keep useful throughput. Trial Normal CPU/I/O for confirmed interactive/shared service roles; budget concurrent expensive jobs by workload, not all `node.exe`/`python.exe` instances. Protect hot model/server memory from forced trimming. |
| Sync/indexing, updaters, telemetry and optional utilities | Below Normal CPU and low memory/I/O hints are candidates for verified deferrable work. Shift bursts out of calls where feasible; protect boot/mount/provider dependencies and actual completion requirements. |
| Audio, shell, compositor and input | Keep Windows thread boosts and multimedia scheduling. Review broad OEM/shell elevation individually; process priorities do not reprioritize kernel DPC/ISR execution. |

**Memory trial comes first.** Compare SmartTrim disabled against the unchanged
policy during a matched useful workload/call. If disabling globally worsens
headroom, compare targeted hot-workload exclusions. Keep memory priority at
Normal for live media/useful hot data; experiment with lower hints only for
verified cold background work. Retain a suitable pagefile and record commit
headroom. Avoid working-set caps, standby-cache purges or pagefile removal as
substitutes for reducing allocations.

Bitsum says SmartTrim cannot reduce total virtual memory usage and active data
can be paged back in. A smaller working-set number is therefore not the success
criterion. [Bitsum FAQ](https://bitsum.com/process-lasso-faq/).

**CPU/power trial follows evidence.** Keep normal scheduling flexibility first;
compare soft CPU Sets only after mapping local topology. Above/Below Normal
classes and broad exclusions already change ProBalance eligibility. The live
power plan was Bitsum Highest Performance despite the INI's Balanced startup
setting. Compare sustained AC-powered Balanced versus the current plan while
holding workloads constant and observing useful throughput and thermals.
The GUI's configured target alone does not establish the active power plan.

**GPU and network require separate measurements.** Keep hardware acceleration
as the initial baseline; measure engine load and which GPU performs capture,
effects, decode and display. Trial effects/resolution/layout reductions where
they deliver acceptable call quality. GPU priority is not a VRAM reservation,
and child processes do not inherit it. I/O priority provides an OS hint, not
end-to-end network QoS. [Bitsum priority documentation](https://bitsum.com/apps/process-lasso/docs/rules/priorities/).

For Teams, evaluate app-specific DSCP classification only with tenant settings
and actual NIC/switch/router support. Do not mark every Teams packet as audio,
or apply Teams ports to Zoom. Ethernet link rate does not establish WAN capacity
or the call's route; measure latency/jitter/loss under competing uploads.
[Microsoft Teams QoS](https://learn.microsoft.com/en-us/microsoftteams/qos-in-teams).

## Resource use and reuse candidates

The architecture snapshot observed 80 PowerShell processes (4.14 GiB private),
54 Node processes (2.78 GiB), 58 Chrome processes (6.46 GiB), and 41 WebView
processes (3.47 GiB). Counts are not tabs, duplicate tasks, or orphan proof.
Working sets share pages and must not be added as physical consumption.
Agents' own private memory is intentional use unless a measured lifecycle leak
or unneeded residency is established.

The later 31-second fractional CPU window attributed about 17.2% host CPU to
Python, 12.4% to WSL, 9.4% to compilers, 3.1% to the Process Lasso GUI, 3.0% to
DWM, 2.8% to Defender and 2.1% to WMI. Rates cover accessible persistent process
generations and undercount exited/protected processes; collector/review and
concurrent jobs affect them. No conclusion about a call or sustained waste
follows from this short window. The earlier exploratory process rates were
integer-quantized and are explicitly qualified; use the fractional window.

| Candidate | Test before action |
|---|---|
| Process Lasso GUI drawing/log display | Compare GUI minimized/closed versus visible, keeping the governor running and all rules identical. The UI is a meaningful overhead candidate; stop neither governor nor other owners' sessions. |
| Repeated PowerShell/Node/MCP startup and scans | Measure startup/idle costs per live parent family. Pool owned PowerShell runspaces; reuse authenticated connections, repo indexes and long-lived workers where protocol/lifecycle permit. Stdio clients cannot simply share a process; a supported multi-client service must preserve isolation, concurrency, cancellation and crash recovery. |
| Browser automation | Seven Chrome processes matched Playwright-related arguments in one role snapshot. Reuse owned browser contexts where state isolation permits and close only owned completed sessions; verify other Chrome processes' purposes separately. |
| WebView applications | Teams descendants: about 1.54 GiB; Widgets: 0.49 GiB; WhatsApp: 0.64 GiB; FloatingMenu: 0.18 GiB. If optional applications are unused, compare disabling their autostart after owner confirmation. Do not fuse independent WebView trees. |
| Defender/WMI/sync bursts | Identify triggering callers, scans and sync overlap with short ETW traces. Reduce duplicate polling/scans in owned tools; retain security coverage and cloud correctness. No broad AV exclusions. |
| Compositor/display/effects | DWM's approximately 1.77 GiB private footprint and multi-adapter/display topology warrant per-engine profiling. This is not evidence of a leak or driver fault. |

Six selected processes had an absent parent in a non-atomic census. That is a
lifecycle lead, not a termination list: detached workers can legitimately outlive
launchers. The historical busy HapticService was near idle in a current separate
provider sample, so yesterday's finding is not reused as a current culprit.

## Measurement and acceptance

See [measurement-plan.md](measurement-plan.md) for the three-workday schedule,
coverage requirements, capture commands, telemetry gaps and A/B gates;
[research.md](research.md) for primary-source detail; and
[policy-review.md](policy-review.md) for exact rules and helper inconsistencies.
The new collector is [Collect-WorkloadResourceProfile.ps1](../../Tools/Collect-WorkloadResourceProfile.ps1).
Run status and validated source hashes are recorded separately in
`capture-launch.json` and `validation.json` when available.

The hour capture completed with 120 samples over 3,600.301 seconds, from
18:17:57 to 19:17:58 UTC (14:17 to 15:17 Eastern). See the
[completed baseline review](baseline-review.md) for verified integrity, separate
development/diagnostic windows, media capability checks and remaining gaps.
The operator confirmed no live calls. The archived launch source is distinct
from later maintained-source fixes; the collector was not restarted or
retroactively assigned a newer hash. Detailed telemetry remains local.

The priority order is: unchanged baseline; SmartTrim comparison; live-media
policy; reuse/deferrable-work comparisons; then CPU-placement/power/GPU/network
experiments when telemetry identifies the bottleneck. Promote a change only
when call/UI quality improves or holds steady, useful throughput remains within
an agreed tolerance (proposed 5%), and memory/refault/collector overhead does
not regress. No speedup, call improvement or multi-day sufficiency is claimed yet.
