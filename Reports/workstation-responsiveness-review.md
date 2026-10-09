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
