# dtm-task-run: a hidden launcher for scheduled tasks that keeps the exit code

`dtm-task-run.exe` starts a scheduled task's real command with no window, captures its output,
enforces an optional timeout and exits with the child's own exit code. It exists because
`conhost.exe --headless`, the wrapper used to hide task windows, also hides the result: a task
whose script exits 3 still reports `LastTaskResult` 0x0. The owner wants Windows tasks to surface
errors for debugging and to report to agents where needed; a wrapper that turns every failure into
a success cannot do that.

Related: [TASK_REGISTRY.md](TASK_REGISTRY.md) (what every task is for and how it signals failure),
[task-registry.json](task-registry.json), [benchmark-results.json](benchmark-results.json).

## Install, build, test, run

```powershell
# Build and test (xUnit, 62 tests; CI runs the same through the dotnet-build job)
dotnet test Native\DtmTaskRun.Tests\DtmTaskRun.Tests.csproj -c Release

# Publish the shipped build (NativeAOT, 3.3 MB, no .NET runtime needed)
dotnet publish Native\DtmTaskRun\DtmTaskRun.csproj -c Release -r win-x64 -p:PublishAot=true -p:SelfContained=true -o out

# Deploy to the stable path the migrated tasks use, then register the event-log source once (elevated)
Copy-Item out\dtm-task-run.exe $env:USERPROFILE\.local\share\pc-ai\deployed\bin\dtm-task-run.exe
& $env:USERPROFILE\.local\share\pc-ai\deployed\bin\dtm-task-run.exe --install-eventlog-source

# Run (this is exactly what a scheduled task's action looks like)
dtm-task-run.exe --name Nightly-Backup --timeout 590 --notify-bus -- "C:\Program Files\PowerShell\7\pwsh.exe" -NoProfile -File C:\x\backup.ps1
```

Source: [`Native/DtmTaskRun`](../../Native/DtmTaskRun). Tests: [`Native/DtmTaskRun.Tests`](../../Native/DtmTaskRun.Tests).
The deployed copy and its SHA256 are recorded in `~\.local\share\pc-ai\deployed\MANIFEST.json`.

## Command line

```
dtm-task-run.exe --name <TaskId> [--timeout <sec>] [--notify-bus] -- <exe> [args...]
dtm-task-run.exe --install-eventlog-source | --version | --help
```

Everything after the first bare `--` belongs to the child and is passed through
`ProcessStartInfo.ArgumentList`, so quoting and child flags such as `--timeout` are never reinterpreted.

| Exit code | Meaning |
|---|---|
| the child's own code | normal; NTSTATUS-style codes (for example `0xC000013A`) pass through unchanged |
| 1460 (0x5B4) | `--timeout` elapsed; the child's process tree was killed (ERROR_TIMEOUT) |
| 87 | the launcher was started with bad arguments (ERROR_INVALID_PARAMETER) |
| 2, 193, 5 ... | the child could not be started: the Win32 error code (file not found, bad image, access denied) |
| 1359 | an unexpected failure inside the launcher after it started (ERROR_INTERNAL_ERROR) |

Environment: the child gets `DTM_TASK_NAME=<TaskId>`. `DTM_TASKS_DIR` and `DTM_AGENT_BUS_EXE` override the
data directory and the `agent-bus.exe` path (used by tests and dry runs).

## What it writes

All under `%LOCALAPPDATA%\dtm-tasks`. `<stem>` is the task id with every character that is invalid in a file
name, and every space, replaced by `_` (`PC-AI Process Lasso Governor Watchdog` becomes
`PC-AI_Process_Lasso_Governor_Watchdog`). [`Test-ScheduledTaskHealth.ps1`](../../Tools/Test-ScheduledTaskHealth.ps1)
reimplements the same rule.

| Path | Content |
|---|---|
| `logs\<stem>.log` | one line per event: `<ISO-8601 local time with offset> <message>`; child lines are tagged `[out]` / `[err]`; the launcher's own lines start with `dtm-task-run:`. Secrets are masked. |
| `logs\<stem>.log.1` | the previous generation. When the log is larger than 262144 bytes (256 KB) it is moved to `.1`, replacing the old `.1`, and a new file starts. This is the same rule `Update-BwArchive.ps1` uses. |
| `status\<stem>.json` | the latest run, written atomically (temp file in the same folder, then an overwriting move) |
| `state\<stem>.notified`, `.notify-failed` | timestamps that rate-limit agent-bus messages |

`status\<stem>.json`:

```json
{
  "schemaVersion": 1,
  "task": "dtm-task-run-selftest",
  "state": "completed",
  "pid": 34768,
  "startUtc": "2026-10-20T21:48:38.4476804Z",
  "endUtc": "2026-10-20T21:48:38.5387573Z",
  "durationMs": 90,
  "exitCode": 3,
  "exitCodeHex": "0x3",
  "timedOut": false,
  "cmdline": "cmd.exe /c \"echo selftest-stdout & echo selftest-stderr token=*** 1>&2 & exit 3\"",
  "stderrTail": [ "selftest-stderr token=***  " ],
  "host": "dtm-p1gen7",
  "logPath": "C:\\Users\\david\\AppData\\Local\\dtm-tasks\\logs\\dtm-task-run-selftest.log"
}
```

`state` is `running` while the child lives and `completed` afterwards. A status file stuck at `running` while the
task is not running means the launcher itself was killed (Task Scheduler stop, logoff); the health check reports it.
`stderrTail` is the last 20 stderr lines. `exitCodeHex` is the unsigned value Task Scheduler shows as `LastTaskResult`.

On a nonzero exit (after the status file is written):

* **Application event log**, source `dtm-task-run`: id 1001 failed, 1002 timed out, 1003 could not start; the text
  holds the task, exit code, duration, redacted command line, log path and stderr tail.
* **Agent bus**, only with `--notify-bus`: runs `agent-bus.exe send --from-agent dtm-task-run --to-agent all
  --topic task-failures --schema finding --thread-id <host>-tasks --tag host:<host> --tag repo:PC_AI --body "FINDING:
  task <id> failed exit=<n> (0x..) on <host>. status=<path> log=<path> ... SEVERITY: medium"`. The body carries
  ids, codes and file paths only, never stderr or the command line. A 5 second cap applies; at most one message per
  task per hour (`state\<stem>.notified`); after a failed attempt the launcher waits 10 minutes before trying again
  (`.notify-failed`). The exit code and status file are final before the bus is contacted, so a slow or dead bus
  delays the launcher's exit by at most 5 s and can never change the result.

Redaction (command line, log, status, event text): `name=value` or `name:value` where the name contains `token`,
`password`, `secret`, `credential`, `pwd` or ends in `key`; the argument after a sensitive flag (`--token VALUE`);
`Bearer` tokens and `Authorization` headers; JWT-, GitHub-, OpenAI-, Slack- and AWS-shaped tokens; `user:password@` in
URLs. This is best effort, which is why the bus message leaves stderr out entirely.

## Design

* **No window, ever.** The project is a `WinExe` (Windows GUI subsystem, verified by a test that reads the PE
  header), so the OS never allocates a console for the launcher. The child starts with `CreateNoWindow = true`,
  `UseShellExecute = false` and all three standard streams redirected (so a console child gets a hidden console).
* **The exit code is the child's.** The launcher returns what the child returned; only a timeout (1460) or its own
  start-up failures use other values.
* **Job Object with kill-on-close.** The child is assigned to a job. If Task Scheduler kills the launcher (time
  limit, `Stop-ScheduledTask`, logoff) the whole tree dies with it, instead of being orphaned and then duplicated by
  the next trigger (`MultipleInstancesPolicy = IgnoreNew` stops protecting you once the launcher is gone). The timeout
  also terminates the job. On a **normal** exit the kill-on-close flag is cleared first, because tasks such as
  `LspmuxServer` deliberately start a detached daemon and exit; that daemon survives exactly as it did under
  `conhost --headless` (checked end to end, see below).
* **Output is pumped, not buffered.** Two threads read stdout and stderr so a chatty child cannot block on a full
  pipe. After the child exits the launcher waits at most 3 s for the pipes; a detached descendant holding them open is
  logged and ignored.
* **Cheap for the common case.** Most runs are well under a second, so the start line and the `running` status are
  written only if the child prints something or is still alive after 1.5 s; otherwise the start and end lines go out
  in one append and only the final status is written.
* **Logging cannot hurt the task.** Every write opens, appends and closes with `FileShare.ReadWrite | Delete`; errors
  are swallowed. This matters because `Update-BwArchive.ps1` appends to the same file (its
  `bitwarden-archive-refresh.log` and the launcher's `Bitwarden-Archive-Refresh.log` are one NTFS file).
* **Event log without `SourceExists`.** That call throws for a limited token because it enumerates the Security
  log. The launcher calls `ReportEvent` directly and checks the source's registry key; an elevated run registers it,
  a limited run just writes (Event Viewer then shows the text with a "description cannot be found" prefix until
  `--install-eventlog-source` has been run once).
* **NativeAOT.** See the benchmark: the runtime start-up and JIT dominate a launcher's cost, so the shipped build is
  NativeAOT. `IsAotCompatible` keeps the trim/AOT analyzers on in every normal build.

## Benchmark

Method: [`Tools/Measure-TaskLauncherOverhead.ps1`](../../Tools/Measure-TaskLauncherOverhead.ps1). Wall time from
`Process.Start` to exit, 20 measured runs per variant after 2 discarded warm-up runs, variants interleaved
round-robin so noise falls on all of them equally, p95 by nearest rank. Two children: `pwsh -NoProfile -Command exit 0`
(the real task shape) and `cmd /c exit 0` (isolates the wrapper). **The host was busy** (CPU load 85-100% from other
agents, OneDrive and Defender throughout), so absolute numbers are inflated and noisy; read the differences. Two
complete runs are shown; raw data in [benchmark-results.json](benchmark-results.json).

Overhead over launching the child directly (median, ms):

| Variant | Size | `cmd /c exit 0` run 1 | run 2 | `pwsh ... exit 0` run 1 | run 2 |
|---|---|---|---|---|---|
| direct | n/a | 97 (baseline) | 133 (baseline) | 498 (baseline) | 628 (baseline) |
| `conhost.exe --headless` | in the OS | +23 | +24 | +19 | +66 |
| dtm-task-run, framework-dependent single file (ReadyToRun) | 353 KB | +188 | +327 | +274 | +418 |
| dtm-task-run, self-contained single file (ReadyToRun) | 75 MB | +157 | +290 | +222 | +318 |
| **dtm-task-run, NativeAOT (shipped)** | **3.3 MB** | **+46** | **+68** | **+69** | **+104** |

Median / p95 in ms for the same runs (run 1 | run 2):

| Variant | `cmd /c exit 0` median | p95 | `pwsh ... exit 0` median | p95 |
|---|---|---|---|---|
| direct | 97 \| 133 | 116 \| 233 | 498 \| 628 | 587 \| 923 |
| conhost --headless | 120 \| 157 | 134 \| 180 | 516 \| 694 | 646 \| 1245 |
| dtm-task-run FD | 285 \| 460 | 339 \| 1035 | 771 \| 1047 | 941 \| 1672 |
| dtm-task-run SC | 254 \| 422 | 325 \| 524 | 720 \| 946 | 976 \| 1470 |
| dtm-task-run AOT | 142 \| 201 | 155 \| 292 | 567 \| 733 | 653 \| 1578 |

Reading it: framework-dependent and self-contained builds pay for runtime start-up and JIT on every launch
(+160 to +420 ms under this load), and the 75 MB self-contained file buys nothing over the 353 KB one. NativeAOT is
about 4x cheaper. Against `conhost --headless` it costs roughly 25-50 ms more per launch, which buys the exit code,
the log, the status file, the timeout and the event/bus reporting. For tasks that run every 5 to 15 minutes this
is negligible. A .NET runtime is installed here (8, 9 and 10), so framework-dependent was a real option; it lost on
measured cost, not availability.

### No visible window

The same script registers a throwaway Task Scheduler task per variant (same interactive, limited principal as the
real tasks), starts it 3 times and polls `EnumWindows` every 10 ms for new visible top-level windows owned by
console-like processes, then deletes the task:

| Variant | Launches | Window sightings | Kind |
|---|---|---|---|
| direct (positive control) | 3 \| 3 | **3 \| 4** | `pwsh / ConsoleWindowClass` |
| conhost --headless | 3 \| 3 | 0 \| 0 | |
| dtm-task-run (FD, SC, AOT) | 3 \| 3 each | 0 \| 0 each | |

The direct variant is the positive control: the detector sees a console window when there is one, so the zeros
are meaningful.

## Evidence that it tells the truth

Same child (`cmd /c exit 3`), same principal, run by Task Scheduler on 2026-10-20:

| Action | `LastTaskResult` |
|---|---|
| `conhost.exe --headless cmd.exe /c "exit 3"` | **0x0** |
| `dtm-task-run.exe --name dtm-task-run-selftest --notify-bus -- cmd.exe /c "... exit 3"` | **0x3** (Task Scheduler event 201 return code 2147942403 = 0x80070003), Application event 1001 written, status JSON `exitCode 3`, finding posted to thread `dtm-p1gen7-tasks` |

The self-test task was deleted afterwards. A migrated production task shows the same thing:
`VigilLocalRunners-OnDemand` now reports `0x4` (every GitHub REST runner-inventory query timed out), which was
invisible as `0x0` behind `conhost`. And `headscale-db-offbox-pull` shows in its log that the headscale backup has
been failing since 2026-10-06 (ssh to the UDM times out) even though its script exits 0; see the known issues in
[TASK_REGISTRY.md](TASK_REGISTRY.md).

Daemon survival, run with the deployed binary and a script that starts a hidden `pwsh` sleeper and exits:

| Wrapper | Sleeper alive after the wrapper exited |
|---|---|
| conhost --headless | yes |
| dtm-task-run | yes |

## Migrating a task

[`Tools/Convert-TaskToLauncher.ps1`](../../Tools/Convert-TaskToLauncher.ps1) rewrites `conhost.exe --headless <cmd>`
to `dtm-task-run.exe --name <TaskName> [--timeout N] [--notify-bus] -- <cmd>`. The child command line is copied
into the action verbatim; triggers, principal, settings and history are untouched (`Set-ScheduledTask` updates in
place). It exports the task XML to `~\.local\state\task-backups\pre-launcher__<task>.xml` before and
`after-launcher__<task>.xml` after, and `-Revert` restores the action (command, arguments and working directory) from
the backup. `-DryRun` (or `--DryRun`, `-WhatIf`) prints each new action and changes nothing: no task, no backup
directory, no XML file. `-h` / `--help` print the script's help. The dry run, the rollback and the CLI switches are
tested against a mocked Task Scheduler in `Tests/Unit/Convert-TaskToLauncher.Tests.ps1`.

```powershell
.\Tools\Convert-TaskToLauncher.ps1 -TaskName '\LspmuxServer','\Bitwarden-Archive-Refresh' -NotifyBus '\Bitwarden-Archive-Refresh' -DryRun
```

Quoting: the launcher splits the copied line into arguments and quotes each one again for the child
(`ArgumentList`), while `conhost --headless` passed the line through unchanged. Ordinary quoted paths survive (all the
tasks below were checked), but a command that depends on unusual quoting, such as nested `cmd /c ""..." "...""`,
should be started once after conversion and its log read.

`--timeout` defaults to the task's own ExecutionTimeLimit minus 10 s, so a hang is reported by the launcher as 1460
with a status file and event instead of Task Scheduler killing it silently (`0x41306`). Tasks with no limit (daemons,
runners, `LspmuxServer`) get none. The launcher refuses a `--timeout` above 86400 s with exit 87 and never starts the
child (LEARNED_RULES 85), so a task whose limit is longer than that, including Task Scheduler's default `PT72H`, also
gets none and keeps relying on its own limit; an explicit `-TimeoutSeconds` above 86400 is refused by the converter. The scheduler agent's delays and limits were left exactly as they were; the XML
diff of every migrated task shows only the `<Actions>` element changed.

Migrated on 2026-10-20 (the reasons for each `--notify-bus` choice are in the registry's "Launcher decisions"):

| Task | `--timeout` | `--notify-bus` |
|---|---|---|
| Bitwarden-Archive-Refresh | 590 | yes (secrets) |
| Google-Workspace-Auth-Watch | 590 | yes (auth) |
| headscale-db-offbox-pull | 590 | yes (off-box backup) |
| VigilNet-Watchdog | 590 | yes (fleet network) |
| VigilLocalRunners-OnDemand | 230 | yes (runners) |
| Sam3TestbedRunner | none | yes (runner) |
| AgentBusHubWatchdog | 110 | no (circular: reports through the bus it checks) |
| PC-AI Process Lasso Governor Watchdog | 290 | no |
| PowerShell\ProfileLogSync | 290 | no |
| UDP Socket Monitor | 170 | no |
| LspmuxServer | none | no |

Still behind `conhost --headless`, deliberately not touched: `Codedev-*` (three tasks owned by the monorepo-index
lane), `AgentHub Windows Runner` and `RmeRunner`. The health check names them in its summary.

## Health check

[`Tools/Test-ScheduledTaskHealth.ps1`](../../Tools/Test-ScheduledTaskHealth.ps1) now reads the status files of tasks
started through the launcher and reports: a nonzero or timed-out last run (Failed), a launcher killed mid-run
(Failed), a status record older than the task's cadence times `-StaleFactor` or older than Task Scheduler's own last
run (Stalled). A missing file is informational (a freshly migrated task has not run yet) and a direct task is never
flagged for lacking one. With no `-StatusDir` it searches the current user's and every profile's
`dtm-tasks\status`, because the SYSTEM watchdog task (`PC-AI Scheduled Task Health`) has its own `%LOCALAPPDATA%`.
Tasks still behind `conhost --headless` are counted in the summary (`ConhostWrapped`). Pester:
[`Tests/Boot/ScheduledTaskHealthLauncher.Tests.ps1`](../../Tests/Boot/ScheduledTaskHealthLauncher.Tests.ps1).

## Limits to know about

* The launcher reports what the process returns. A script that logs an error and exits 0 still looks green to Task
  Scheduler; it shows up in the log and nowhere else (`vigil-net.ps1 backup` is the live example). Fix such scripts
  to exit nonzero.
* Child output is decoded as UTF-8; only the stderr tail is in the status file, all output is in the log.
* A launcher killed in the first 1.5 s of a run leaves the previous run's status in place. The health check
  catches this through the Task Scheduler last-run comparison.
* The bus post can take up to 5 s, during which the task still shows as running. The 5 s cap is tight on a loaded
  host: a cold `agent-bus.exe` took 3.2-3.5 s here at 100% CPU, and the `VigilLocalRunners-OnDemand` log shows one
  send killed at the cap (`bus notify Failed`, 5.4 s after the end line). If the hub accepted the message just
  before the CLI was killed, the 10-minute failure backoff allows one duplicate. The cap is a tuning knob
  (`BusNotifier.Timeout`), not a correctness issue: the exit code and status file are already final.
* A task someone stops on purpose (`Stop-ScheduledTask`, the runners that `VigilLocalRunners-OnDemand` stops) kills
  the launcher mid-run, so its status file stays `running` and the health check reports it as Failed ("launcher was
  killed mid-run"). That is correct for a hang and noisy for a deliberate stop; list such a task in `-Expected` if
  it matters.
* Some exit codes are soft by their script's own contract. `Invoke-LocalRunnerOnDemand.ps1` returns 4 for "GitHub
  query failed, nothing started", which happens whenever GitHub times out. The health check and the bus now report
  it (the bus at most hourly). Either the script should return 0 for that case or the owner can accept the noise.
* The event-log source needs one elevated `--install-eventlog-source`; the `Highest` tasks also register it on first
  use.
* The watchdog task `PC-AI Scheduled Task Health` runs the checkout copy of the health script; until this change is
  merged and that checkout updates, it does not read the status files.
