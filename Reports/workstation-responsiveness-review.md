# Workstation responsiveness review

Host: dtm-p1gen7. Review date: 2026-10-09. Owner: Codex integration lane.
The user's immediate responsiveness priority supersedes additional local native
builds and fleet validation. The broader consolidation remains open.

## Applied changes

| Change | Actual evidence | Limit |
| --- | --- | --- |
| Completed the user's Docker stop | The exact surviving Desktop instance exited at 18:31 UTC; all six observed Desktop/backend instances were absent afterward. Fresh checks still find Desktop/backend absent. | A retained Docker service and older CLI process remain; the earlier zero-count label covered only Desktop/backend/build/dockerd. API queries timed out, so engine/container state remains unqualified. No Docker data was deleted and no WSL shutdown was issued. |
| Restored Process Lasso Governor | Started the signed installed executable at 18:12 UTC after finding the GUI without a Governor. | Presence is not complete policy efficacy. |
| Added recurring watchdog checks | Preserved the existing task XML and added indefinite five-minute checks while retaining delayed logon, principal and settings; report output uses D:. Natural runs at 18:20 and 18:45 UTC returned zero with the same responding Governor. | Actual missing-process recovery and post-reboot acceptance remain open. |
| Closed two failed Python stdin jobs | Two nonblocking stack observations, roughly 136 seconds apart, found the same REPL traceback formatter; exact shell parents remained and their launching ancestors were absent. Only the two retained Python instances were terminated. | Another measured Python instance could not be profiled because of partial-memory-read errors and was preserved. Active agents were preserved. |
| Repaired the local `python3` shell shim | Only first-argument `-` enables Python's classic REPL. Fresh stdin and `-c`/argument-with-spaces controls both exited zero. | Other launchers and the triggering PTY traceback failure were not qualified. |
| Reduced immediate contention | Four exact measured Python instances were lowered to Below Normal; the root's dedicated recovery browser closed normally. | No global process-name kill, affinity or priority rule was added. |
| Routed future owned build output to D: | A metadata-only child verified target, artifact, scratch and job-limit environment values. Two ignored root-owned build producers now respect caller targets or default to D:. | No build or cache migration ran; existing C: caches remain intact. |
| Started a SmartTrim-off trial | A real event at 18:45:46 UTC trimmed Node's resident set from 302 MB to 1 MB. At 18:49 UTC only `SmartTrimIsEnabled` changed from true to false, with exact backup and unchanged ACL/Governor. | Refault causality and matched-workload performance acceptance are open. |

Machine receipts and rollback material are retained outside Git under
`D:/pcai-relocation/responsiveness-immediate/r1`. The original INI has SHA256
`C6409149C2A4A9084628CF6CFFBB6E9726E915A5AA99D0780D765D03706DBE33`;
the one-key trial configuration has SHA256
`6E6AC0CD601AAB489B8F3269A2F5A296C96F4DACDA360B5AB7CC660BA4B3B10F`.
Restore only the changed key after checking for subsequent edits. The trial
does not change priority rules, clear standby caches or reduce the pagefile.
Bitsum documents live configuration adoption without restarting the app, and
explains that trimming active working sets can cause pages to be read back.
[Configuration adoption](https://bitsum.com/apps/process-lasso/docs/deployment/config-push-pull/),
[SmartTrim behavior](https://bitsum.com/smarttrim/).

## Measurements and remaining gaps

Short, unmatched observations moved from roughly 91% CPU and 5.6 GiB available
memory to 53% CPU/8.1 GiB at 18:23 UTC, 83%/10.6 GiB after Docker stopped, and
47%/11.9 GiB in a later three-sample window. C: queue length was zero in those
later samples, while pages-in remained variable. These are observations rather
than a sustained benchmark or proof that all stalls are fixed.

The SmartTrim-off observation collected all 48 five-second samples over about
four minutes. Mean CPU was 72.50%, available memory 11,451.56 MiB, commit 72.31%,
pages-in 10,174.30/s, C: read latency 0.339 ms and queue length 0.9375. The observer
used 2.0 CPU seconds, about 0.038% of total host capacity. The exact one-key INI
hash remained stable; the finite rotating-log readback showed no subsequent
trim events and the Governor remained present. Useful workload was not held
constant, so this does not establish a SmartTrim speedup or a paging cure.
The user subsequently reported that the workstation was more responsive.
This is operator feedback for the combined changes, not attribution to one setting.

The maintained registrar now defaults to delayed logon plus independent,
indefinite five-minute recovery checks and accepts an explicit report path.
`StartupDelaySeconds` delays the logon trigger and initial periodic start;
periodic checks can run earlier after future logons. Registration parameters
are named-only so standalone `--help` cannot become a task name. The canonical
Windows fixture passed all 15 cases with zero test, block or container failures,
zero skips/not-run, reconciled XML and restored prior global fixture state.
The same fixture detected 11 failures in the original. Mocked binding is separate
from actual task persistence and missing-Governor recovery. The fixture is
explicitly selected by Windows CI; current-head hosted acceptance remains open.

C: already had about 308 GiB free. Shared Cargo, rustup and sccache storage was
already linked to T:. Moving the historical private C: targets would recover
only about 5.9 GiB logical space, and the existing per-file PowerShell dry run
itself consumed substantial CPU. Its migration replacement remains unqualified;
no model, shared NuGet, pagefile or global temporary-directory move was applied.

Two per-tool Codex hooks launch nested PowerShell interpreters. Tool commands
now use `login:false` to avoid profile startup while retaining those hooks.
Installed Codex 0.161.0 supports hook command strings rather than a direct
executable/argument handler, so a runner optimization remains open. Hook trust
and fail-closed policy must be preserved. The path guard's terminal exception
handler currently fails open; that separate safety defect remains open.

The local minimal-profile import change was measured in three alternating
fresh-process pairs. Profile dot-source times were 654→475, 555→379 and 720→374 ms;
means were 643→410 ms. All six children preserved PATH/PSModulePath, command
resolution, roots and encoding, omitted only the expected accelerator module,
and exited with no pending custody. These are exploratory profile-load results;
end-to-end child wall timings were not retained. The small import-span change
was then applied atomically to P1's local canonical profile, with unchanged ACL
and protected original bytes. A fresh deployed-profile child passed the same
environment/mode controls and exited zero (452 ms profile interval). Interactive
import behavior, cloud shims and hooks were not changed. Private input/output,
rollback and receipts are in `D:/pcai-relocation/profile-startup/minimal-r1`.
The deployed profile SHA256 is
`513A193F9FE78A66F8AE09D082F49CA617F20D3063071AB180DC41458A8C6DC8`.

Next actions are a matched useful-workload comparison with trimming disabled,
actual watchdog recovery/post-reboot validation, remaining hook-runner
startup work, and root-owned build admission only after responsiveness is
stable. Current-head hosted PowerShell coverage remains below the required 85%
gate; scoped local passes do not waive it. Work connectivity, phone/desktop
credential reconciliation and complete fleet integration remain deferred.

The fresh 2026-10-10 12:36 UTC follow-up collected 12 samples over 60.09 seconds,
with all artifacts on D: under `D:/pcai-relocation/p1-workload-followup/r2`.
Median CPU was 34.73%, available memory 11,389 MiB and commit 78.42%; median
pages-in was 53.46/s, disk-transfer latency 0.068 ms and queue length zero.
The 95th percentiles were 55.08% CPU, 248.68 pages-in/s and 0.090 ms latency;
queue length remained zero. The observer used 14.94 CPU seconds, 1.13% of host
capacity, with collection times of 732 ms median and 1,577 ms at the 95th
percentile. Between 37 and 40 process-property reads were inaccessible, so this
is not a complete per-process attribution. The earlier unmatched capture had
median pages-in of about 2,337/s. Neither comparison establishes causality or
useful-throughput improvement. Commit remains close to the 80% build-admission
limit; further bounded native qualification is being prepared on Carbon,
whose fresh read-only snapshot showed 17,609 MiB available and 32% commit.

## Current workload and launcher follow-up

The October 10, 2026 capture started at 19:35:41 UTC and completed eight samples
in 120.12 seconds during agent integration. Median CPU was 37.85%, available
memory 10,786 MiB and commit 80.75%. Median pages-in was 62.47/s, aggregate disk
transfer latency 0.053 ms and sampled disk queue zero. The 95th percentiles were
57.55% CPU, 239.43 pages-in/s and 0.095 ms latency; queue remained zero. No
sampling gaps exceeded 150% of the interval. This window did not show sustained
disk thrashing; aggregate samples do not exclude a brief or drive-specific stall.
Pages-in include hard faults and do not establish pagefile swapping.

The observer consumed 0.464% of host CPU capacity; collection wall time reached
2,681 ms at the 95th percentile. Between 36 and 43 process-property reads were
inaccessible, leaving attribution incomplete. OneDrive, Python and System were
the leading observed CPU groups. Useful work and process names do not establish
waste or authorize stopping their sessions. The capture is unmatched to the
previous workload and does not establish improvement caused by a particular fix.
The exact summary is `D:/pcai-relocation/p1-responsiveness-profile-r1/summary.json`,
SHA256 `00E98A8D9EB9DDEB0549C4E32D91B09C4E9489DBD2D15AA1AF4EF3E3EE4B11FF`.

A separate read-only launch inventory observed 148 PowerShell processes, 138
with NoProfile. Their recognized script paths included 55 machine MCP launchers,
20 QMD launchers and 17 Serena launchers. Four Codex parents each had 12 machine
MCP wrappers. The current Codex configuration still uses PowerShell for
Filesystem, Context7 and GoogleWorkspace. The native launcher's earlier measured
CPU/private-memory gains therefore remain a deployment opportunity, rather than
an already realized reduction across these active processes. Current process
counts and private-memory totals alone do not justify terminating wrappers.
Inventory SHA256 is `5A978062D5B276A85A2C9099AEDE86B7AF33EF2B08C610237C27FDADDDF8051F`
at `D:/pcai-relocation/p1-runtime-launch-audit-r1/pwsh-launches.json`. No command
lines, credential values or environment secrets are stored in that inventory.

Owner: Codex integration lane. Next action: verify all three launch modes and
artifact/source binding, then prepare a reversible P1 pilot for new sessions.
Existing clients, QMD/Serena database workloads and active agent sessions remain
outside the pilot. Shared-server deployment requires separate multi-client,
configuration and lifecycle validation. No new configuration deployment, cache
purge, service restart or global memory trim was applied in this capture.