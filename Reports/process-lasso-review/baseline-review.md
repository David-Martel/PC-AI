# Completed workload baseline and media diagnostics

Observed 2026-10-05 on DTM-P1GEN7. The authorized hour completed under the
unchanged Process Lasso policy. The operator confirmed **no live calls**.
These measurements qualify contention candidates and the next experiments;
they do not establish Zoom/Teams quality, waste, or an optimization gain.

## Completed capture and integrity

Collection ran from 18:17:57.711 to 19:17:58.353 UTC: 3,600.301 seconds,
120 samples at a requested 30-second cadence, five stream parts and 73,846,278
bytes. There were no gaps above 45 seconds; the maximum interval was 35.892
seconds. Offline verification found no malformed lines, incomplete tails,
sample-index discontinuities or summary discrepancies. Stderr was empty.

The source at launch is preserved in
[collector-source-at-launch.ps1](collector-source-at-launch.ps1), SHA256
`E5B4D816E53081D413505380A3A9AA37006866FB49DAE322C6D87FD3A271EBB9`.
It differs from the maintained collector, which received reviewed fixes after
launch. Policy SHA256 remained
`DCA0CEE62474BCCF7D4C1D8F55F62A609BBD3E09A77DF0B591DDC97C02075FA0`.
All nine closed capture/log files moved with matching hashes; the empty old
owned directory was removed only after the collector generation exited.

Detailed evidence remains local under `baseline60m/`; its summary SHA256 is
`E1A8A89ED0BEF3612F2B63C7E0A7127D8FE6FE8C3C8FD40D7EBB673DDCBA32F2`.
Local `analysis/analysis.json` and `analysis/initial-ordinary-window.json`
contain counts, quantiles, input hashes and provenance. The PNG/SVG overview
and exact offline analyzer source are alongside them. Raw streams and traces
are excluded from Git.

## Separate development from diagnostic windows

The initial window ends conservatively before the timed media diagnostics;
it includes ordinary development and this review's validation/Git work.
Later synthetic media, ETW recording/finalization, NVENC capability probes
and offline trace analysis have explicit local session markers. Overlapping
markers must not be summed as independent durations.

| Window | Samples | Mean host CPU | Available RAM minimum / p5 | Commit p95 |
|---|---:|---:|---:|---:|
| Initial development, 18:17:59–19:00:04 UTC | 85 | 89.72% | 1.120 / 1.239 GiB | 92.46% |
| Full hour, including later diagnostics | 120 | 89.02% | 0.465 / 1.120 GiB | 92.66% |

These are equal-weight sample means and nearest-rank percentiles, not
independent experimental repetitions. The full hour's CPU median was 90.78%.
Pages Input/sec p95 was about 13,941, and aggregate disk-transfer latency p95
was 2.238 ms; page reads include mapped files and do not prove pagefile swapping.
Per-disk attribution and actual call metrics remain missing.

The collector consumed 0.327% mean host CPU; interval p95 was 0.385%. These
exclude WMI/provider-side work. Collection wall time p95 was **15.456 seconds**,
so faster polling is not justified by the low own-CPU number alone. The median
CPU accounting gap was 23.96 percentage points; protected/inaccessible process
properties occurred in every sample. Process rankings are incomplete and
non-atomic. Keep the 30-second continuous cadence; use short focused traces for
transients and an observer-off/on comparison for end-to-end instrumentation cost.

## Resource and reuse candidates

Observed persistent-process CPU, normalized to 22 logical processors and the
full hour, included Explorer 5.07%, WMI providers 5.05%, DWM 3.85%, Terminal
2.84%, OneDrive 2.48%, the Process Lasso GUI 1.63% and its governor 1.22%.
These are candidates for role-specific investigation, not termination targets.
Compare GUI minimized/closed while keeping the governor running; inspect shell,
terminal and display update costs, redundant polling and sync overlap. Preserve
security, required synchronization and useful job completion.

Python, WSL, compilers and agents also used substantial resources and remain
productive work. Per-name peak private-byte sums reached WSL 18.46 GiB,
Chrome 7.55 GiB, PowerShell 5.74 GiB and Node 4.80 GiB. Their peaks occurred
at different times; do not sum them, infer tab counts or label them leaks.
Reuse owned workers, authenticated connections, browser contexts and indexes
where lifecycle and isolation permit; measure completed useful units before
changing concurrency or shared runtime priorities.

The short scheduling trace captured many Git generations missed by polling:
1,539 process rows contributed about 6.71% host CPU in that separate window.
This reinforces protecting useful Git work and qualifying polling rankings.

## Bounded media and trace evidence

An installed FFmpeg synthetic 720p/30-fps picture and 48-kHz tone produced a
30-second H.264/AAC file with 900 video frames, zero reported duplicates/drops,
and successful paced decode. The test used no microphone, camera, screen,
speaker or external traffic. Both NVIDIA GPU indices separately produced
30 valid H.264 frames through NVENC. These are software-path and hardware-codec
capability checks, not matched throughput/quality comparisons or proof of the
adapter selected by Zoom/Teams. Pacing intentionally limits throughput.

The private CPU.light file-mode ETL spans 75.232 seconds and is 780,140,544
bytes. xperf header statistics report zero lost events and buffers. Its process
context-switch analysis completed within a 90-second/2-GiB guard, peaking at
about 703 MiB working set. It attributed 87.14% of host capacity including idle
and emitted two thread-metadata warnings: quantitative attribution is partial.
No DPC/ISR driver cause, hard-fault backing-file cause or call glitch is proved.
WPR was verified stopped. WPA/xperf are installed under Windows Performance
Toolkit even though absent from PATH.

## Delivery Optimization and network follow-up

The existing WMI operational channel contained 84 canceled network-adapter
statistics queries from a generation-bound Delivery Optimization service host
in the trace window. Its direct CPU was low; shared WMI providers used more
CPU, but these observations do not attribute that cost solely to this client.
Microsoft explains that `0x80041032` can mean cancellation before full result
enumeration. [WMI cancellation documentation](https://learn.microsoft.com/en-us/troubleshoot/windows-client/system-management-components/wmi-activity-event-5858-logged-with-resultcode-0x80041032).

Supported Delivery Optimization API snapshots showed zero current transfer
usage and no foreground/background downloads at those instants. Accumulated
bytes, discovered peers and cached files eligible for upload are not current
throughput or established connection counts. Internet peering mode 3 was
configured. Keep sparse progress/policy snapshots and correlate provider cost
before considering supported bandwidth/peer settings; no service or policy
was changed. Avoid the `Get-DeliveryOptimizationLog -Flush` option, which stops
the service. [Monitoring](https://learn.microsoft.com/en-us/windows/deployment/do/waas-delivery-optimization-monitor),
[supported configuration](https://learn.microsoft.com/en-us/windows/deployment/do/delivery-optimization-configure).

Three separate five-second TCPv4 samples observed roughly 0.8–1.0 retransmitted
segments/sec. They are host aggregates, with no application attribution;
TCPv6 and UDP media were not covered. They do not establish meeting packet loss.

Proceed with the [three-workday plan](measurement-plan.md): actual Zoom/Teams
and sharing coverage; useful-work benchmarks; matched SmartTrim comparison;
verified live-media protection; then reuse/polling and topology/power/network
experiments. Extend missing scenarios or weak repeatability. Promote only with
call/UI quality preserved, useful throughput within the proposed 5% tolerance,
and no worse memory/refault behavior.
