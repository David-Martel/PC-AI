# Gen9 scheduled task console repair

The `Dtm-HomeLab-Runtime` task on dtm-p1gen9 now runs through the existing
[windowless launcher](../Native/DtmTaskRun/Program.cs). The observation on
2026-11-04 captured no new terminal windows across a controlled run and three
natural scheduled runs. The task's seven existing listeners remained reachable;
the transport, video and dashboard launcher PIDs and start times stayed equal.

## Cause and scope

The interactive task started PowerShell directly every two minutes with
`-WindowStyle Hidden`. Native window enumeration captured brief Windows Terminal
windows at 16:35:49, 16:37:49 and 16:39:49 UTC. A controlled task start at
16:56:24 UTC reproduced a visible terminal for approximately one second and
returned task result zero. The script completed successfully. These observations
indicate that its PowerShell hiding option did not prevent the initial terminal
window from appearing.

The watchdog maintains a dashboard, video helper and SSH forwards. Its existing
command, arguments and working directory remain exact. Only the task's Exec
action changed. Task identity, interactive limited-user principal, triggers,
settings and Task Scheduler timeout compare equal outside the Actions XML node.
The action adds no bus notification or launcher timeout. Existing failure logging
and child exit-code propagation follow the [launcher contract](../docs/scheduled-tasks/LAUNCHER.md).

The launcher uses a Windows GUI subsystem and creates its child with
`CREATE_NO_WINDOW`; the [child transport](../Native/DtmTaskRun/ChildProcess.cs)
retains the raw command line. Its normal-completion path
[releases surviving daemons](../Native/DtmTaskRun/TaskRunner.cs), preserving the
watchdog's existing long-lived helpers.

## Qualification and custody

| Binding | Observed value |
| --- | --- |
| Accepted source tree | `b63b9e13573b325d8853f8f4f547276562bbf20c` |
| Launcher source's last change | `bbdb9daef6a355eaed5b83cb26bc6a0544d2bcd7` |
| Cohort SHA256 | `398468d7a3569c792ec7c049fc9195b86229212e39805a84d2675f1af2acd6b7` |
| Apphost SHA256 | `bc697fc67fc5ba9e0b5fcc5157f04a69ffc7fc3b8d529042e881ba5a603d34da` |
| Managed DLL SHA256 | `0a4ea8033d96992119c5e4e4667b9b34c65587de3206daecd144b13eddf49d01` |
| Before task XML SHA256 | `679c794737dc45ed6341e3251c277c8d35fe940dadfb259af2843f94a706f848` |
| After task XML SHA256 | `f4c079ed4b2852870c947c60524033977a36df40d41651c687e5863e77cf6965` |
| Build | SDK `11.0.100-rc.3.26522.148`, Release, `net8.0/win-x64`, framework-dependent |
| Effective runtime | `Microsoft.NETCore.App 10.0.14`, through the accepted `LatestMajor` roll-forward policy |

The new deployment contains the apphost, managed DLL, dependency and runtime
configuration files, plus symbols and XML documentation. All six files were
read back against the qualified manifest before the task changed. The apphost
hash alone does not identify this managed-code cohort. The build emitted the
preview-SDK notice; qualification applies to these exact artifacts and runtime.
No SDK, global configuration, service identity or existing deployment was replaced.

The native suite passed 77 tests, and the existing PowerShell converter and
measurement helpers passed 56 tests. Seven published-apphost controls covered
GUI subsystem detection, actual PowerShell `GetConsoleWindow() == 0`, literal
arguments, exit 23, timeout 1460 and cleanup, launcher-termination cleanup,
normal daemon survival, and refusal before execution. A console-subsystem
counterexample was detected. The daemon control's first outer-pipe harness
timed out despite successful launcher completion; a corrected process-wait
control verified exit zero and daemon survival, then cleaned up the owned daemon.
Both observations remain in the private qualification packet.

The installed `Set-ScheduledTask` cmdlet has no `WhatIf` parameter. A separate
data-only plan preceded the mutation. Independent review checked source and
artifact hashes, original task XML, raw argument preservation and the intended
action. Fresh definition/state/hash guards preceded `Set-ScheduledTask`; complete
XML comparison after excluding Actions established the unchanged task policy.
Original task XML and machine-specific paths remain in the private local packet.

## Post-change observation

From 17:11:26 to 17:16:27 UTC, 100 ms sampling found zero new visible console or
Windows Terminal windows. A controlled run and natural runs at 17:11:48,
17:13:48 and 17:15:48 UTC completed with result zero. The transient
`0x41301` sample means the task was running, followed by completion; it is not
a task failure. The launcher status also recorded `completed`, exit zero.

The observer used 7.58 CPU seconds over approximately 301 wall-clock seconds.
This measures the observer, not an application performance improvement. The
checks establish task launching and existing-listener liveness. Device operation
and application HIL remain separate qualification.

Rollback restores only the original Exec action after verifying the current
wrapped action and retained preimage. Keep triggers, principal and settings
intact. Keep the qualified cohort while any task references it. Follow-up after
the next real logon remains open in [the boot work list](../boot.TODO.md); no
reboot or logoff was initiated during active work.
