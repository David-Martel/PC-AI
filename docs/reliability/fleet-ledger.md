# Fleet failure ledger, health sweeper and bounded reconciliation

Code: [`Modules/PC-AI.Reliability`](../../Modules/PC-AI.Reliability/PC-AI.Reliability.psm1).
CLIs: [`Tools/SystemScripts/Reliability`](../../Tools/SystemScripts/Reliability/Invoke-FleetHealthSweep.ps1).
Tests: [`Tests/Unit/PC-AI.Reliability.Tests.ps1`](../../Tests/Unit/PC-AI.Reliability.Tests.ps1).
Example policy: [`fleet-reliability.policy.example.json`](../../Modules/PC-AI.Reliability/fleet-reliability.policy.example.json).

## Install, test, run

```powershell
# Test (PowerShell 7 and Windows PowerShell 5.1 both pass; nothing touches the network or the scheduler)
Invoke-Pester -Path Tests/Unit/PC-AI.Reliability.Tests.ps1

# Help, and a dry run that writes, sends and remediates nothing (all four CLIs accept -h/--help and -DryRun/--DryRun)
pwsh -NoProfile -File Tools/SystemScripts/Reliability/Invoke-FleetHealthSweep.ps1 --help
pwsh -NoProfile -File Tools/SystemScripts/Reliability/Invoke-FleetHealthSweep.ps1 -DryRun

# Run once, observe and record only (no playbooks)
pwsh -NoProfile -File Tools/SystemScripts/Reliability/Invoke-FleetHealthSweep.ps1 -NoPlaybooks

# Installing the S4U tasks is DEFERRED (see "Deferred to a follow-up PR"): only the plan can be printed.
pwsh -NoProfile -File Tools/SystemScripts/Reliability/Install-FleetReliabilityTasks.ps1 -DryRun

# Wrap a task so its runs are recorded semantically (exit 0 + "[ERROR]" in the log = failed)
# Use -Command: `pwsh -File` cannot pass an array to -ArgumentList (the wrapper refuses a quoted list).
pwsh -NoProfile -Command "& 'Tools/SystemScripts/Reliability/Invoke-FleetTask.ps1' -Name '\backups\pull' -FilePath pwsh.exe -ArgumentList '-NoProfile','-File','C:\jobs\pull.ps1' -LogPath C:\logs\pull.log; exit `$LASTEXITCODE"

```

Token provisioning (`Set-FleetDiscordToken`) is deferred as well. Discord delivery works today from
`DISCORD_BOT_TOKEN`, or from an existing token file.

CI: PC-AI's Windows `CI` workflow is disabled, so pull requests run only `linux-ci.yml` (`pester-lint`). It runs
the `Portable`-tagged blocks of this suite on Linux; the DPAPI, scheduler-priority and Windows-path cases run
locally on Windows (PowerShell 7 and 5.1).

The installer's dry run prints the tasks it would register and the revert commands. A real install will export
any task it replaces to `%ProgramData%\fleet-reliability\backups\` first.

## Why

A fleet audit found failures that nobody was told about:

- Tasks reported success while failing. A pull logged `integrity check did not pass` and returned `0x0`, an export
  exited 0 with `STALE`, and a logger exited 0 for ten days while doing nothing.
- Tasks never ran. Interactive tasks sat on a host with nobody logged on, and one required task was Disabled.
- Alerts never arrived. Every path logged `ALERT not sent: no bot token`, because the only transport needed an
  unlocked password vault, which unattended (S4U and boot) tasks never have.
- Runners and workflows went silent for days.
- Resources ran out: disk, Postgres connections, leaked MCP processes. One disk watchdog read a missing drive as
  0 GB and stopped every runner.

This module records each of those systematically, alerts once per incident through a transport that works
unattended, and fixes the few cases that are safe to fix automatically.

## The ledger (`fleet-ledger/1`)

There is one JSON object per line, UTF-8 without a BOM, in `%ProgramData%\fleet-reliability\ledger\ledger.jsonl`
(override with `FLEET_RELIABILITY_HOME`). The file rotates at 5 MiB (`FLEET_LEDGER_MAX_BYTES`) and keeps five old
files. Appends and rotation take one file lock, which works across SYSTEM, S4U and interactive sessions. The bash
emitter in sentinel-utils (`scripts/fleet/linux/fleet-ledger.sh`) writes the same shape.

| field | meaning |
|---|---|
| `schema` | `fleet-ledger/1` |
| `event_id` | `<host>-<yyyyMMddTHHmmssfffZ>-<8 hex>` |
| `ts`, `start`, `end` | ISO-8601 UTC |
| `host` | lowercase short host name |
| `source` | `task` `timer` `service` `runner` `job` `workflow` `disk` `postgres` `process` `cert` `sweeper` `playbook` `digest` `alert` |
| `name`, `check` | the subject (task path, drive, runner) and the detector (`run`, `task-stale`, `log-rule`, `disk-trend`, ...) |
| `kind` | `run` `observation` `incident-open` `incident-resolved` `incident-flapping` `incident-reminder` `remediation` `escalation` `digest` |
| `status` | semantic: `ok` `failed` `stale` `skipped` `never-ran` `degraded` `unknown` |
| `severity` | `info` `low` `medium` `high` `critical` |
| `exit_code`, `message`, `log_path` | what happened; messages never carry secrets |
| `incident_key` | `<host>\|<source>\|<name>\|<check>` |
| `remediation` | `{playbook, attempt, budget, outcome, detail}` or null |
| `details` | free-form |

`run` and `observation` events stay local. The other kinds go to agent-bus as
`--topic fleet-failures --thread-id fleet-failures --from-agent fleet-ledger-<host>`, with tags `host:`, `source:` and
`incident:<sha256(key)[0:12]>`. The full event rides in `--metadata`, so the bus is queryable fleet-wide.

- **Schema `finding`** is used for open, reminder, flapping and escalation. The body is
  `FINDING: [<status>] <host> <source>/<name> (<check>): <message>` followed by `SEVERITY: <LEVEL>`.
- **Schema `status`** is used for `RESOLVED:`, `REMEDIATION:` and `DIGEST:`.

Each forward also writes a local `alert`/`delivery` line (`bus=sent discord=sent`, `bus=spooled ...`, or
`failed` when nothing delivered), so an alert that never arrived is visible in the ledger.

If a send fails, the same request goes to the agent-bus offline spool (`spool\bus-spool.ndjson`). Each sweep
renames the spool under a lock, then replays it, so new lines are never deleted with the replayed file. A spool
that does not parse is quarantined.

## Incident state

`state\incidents.json` is shared by the wrapper and the sweeper, so a failure both of them see opens one incident.

- **Open:** a bad observation opens the incident once it has repeated `OpenAfter` times (default 1) and sends one
  alert.
- **While open:** nothing more is sent, except a reminder every 24 h, or at once if the severity rises.
- **Resolve:** after `ResolveAfter` consecutive good observations (1 for task runs, 2 for most sweeper checks),
  the incident sends one `RESOLVED` event.
- **Flapping:** three opens within 6 h send one `incident-flapping` alert. After that, transitions are recorded but
  not forwarded for 6 h.
- **`skipped` is neutral:** for example, runners down during a planned DRAIN that is still inside its time budget.
- **Vanished subjects:** an open incident whose subject disappeared (task deleted, orphan reaped) is resolved when
  its detector completes a sweep without seeing it. Keys that only the wrapper reports (a task outside
  `task_scope`) are exempt; they resolve only on the wrapper's next clean run. A `detector-error` clears only when
  that detector runs successfully.

## Detectors

Policy `task_scope` (a list of task-path wildcards) limits the task detectors to those tasks only, for a
staged rollout; empty means every local task.

| detector | fires on | resolves on |
|---|---|---|
| task result (`run`) | non-zero last result from [`Test-ScheduledTaskHealth.ps1`](../../Tools/Test-ScheduledTaskHealth.ps1), which is reused rather than re-derived; benign codes in `task_ok_codes` (IgnoreNew `0x800710E0`) do not count, and never mask a launcher failure | next good result |
| `task-stale` | no run within `task_stale_factor` (2) times its interval, or never ran | a run |
| `task-expected` | a task in `expect_enabled` that is Disabled, missing or never ran | enabled and ran |
| `task-interactive-no-session` | LogonType Interactive while its user has no desktop (it will not fire; convert to S4U) | a session, or S4U |
| `log-rule` | exit 0, but the newest log written since the last run matches a pattern (`[ERROR]`, `STALE`, `integrity check did not pass`) | a clean log |
| `disk-free` | below `min_free_gb`/`min_free_pct`; a missing **required** drive is a failure, never 0 GB | above floor |
| `disk-trend` | a least-squares slope over 6 h projects the floor within 24 h ("T: hits 15 GB in 7.5 h") | trend flattens |
| `runner-offline` | a self-hosted runner offline (gh API; run on one host with `fleet_checks`) | online |
| `drain-overrun` | a DRAIN announced on agent-bus that has lasted longer than `drain.max_minutes`; while the DRAIN state cannot be read it is `skipped` (neutral), so an open overrun is never resolved by a bus outage | a successful read showing the DRAIN ended |
| `drain-state-read` | the agent-bus read of the DRAIN lifecycle failed, timed out or returned unparseable output (offline runners are then reported, not excused) | a successful read |
| `workflow-no-runs` | an expected workflow's newest run is older than `max_age_days` | a new run |
| `pg-connections` | established connections reach `warn_pct` of `max_connections`; no database credentials needed | below |
| `orphan-process` | a process matching a pattern whose parent died or whose parent PID was reused | none left |
| `cert-expiry` | under 21 days (medium), under 7 days (high), or a chain error (critical) | renewed |
| `detector-error` | a detector that threw; it is reported, never read as healthy | the detector runs |

## Deferred to a follow-up PR

Review of this PR converged on two surfaces. Most findings from round 4 onward, and every P1 after round 6 but one,
were there:

- **automatic remediation:** the playbooks and cache pruning;
- **privileged setup:** ACL hardening of the elevated home and Discord token provisioning.

The ledger, spool, incidents, detectors, alerts and digest are complete and on. The two deferred surfaces refuse to
run:

- **Remediation:** `Invoke-FleetPlaybook` returns `deferred` and runs no action. It writes no budget state and no
  ledger line, and the incident stays open and alerts as usual. `Invoke-FleetCachePrune` throws.
- **Privileged setup:** `Set-FleetHomeAcl` and `Set-FleetDiscordToken` throw, and the installer exits 2 unless
  `-DryRun` is given. Nothing is changed.

A test imports the module in a fresh process. It proves that all four refuse, that the playbook's action never runs,
and that no file is created. The implementation stays in the tree behind `$script:FleetDeferred` (in the `.psm1`),
and the unit suite keeps testing it.

The follow-up PR turns these on. It must keep all of the following rules, each learned from a review finding here:

**Remediation**

1. Reserve the attempt durably before acting, with a terminating write; if the reservation fails, do not act.
2. Unreadable budget state means "do not act". Keep the file as evidence; never treat it as empty.
3. Keep one state file per playbook and incident, under that pair's lock.
4. A failed ledger write after the action must not unspend the budget.
5. Never follow a junction or symlink. Refuse a cache root that is, or sits under, a reparse point, and never
   descend into one. Windows PowerShell 5.1's `Get-ChildItem -Recurse` follows junctions.
6. Re-run only allowlisted tasks that are neither running nor disabled. Escalate exactly once per spent budget.

**Privileged setup**

1. Refuse reparse points at the home, its ancestors and its descendants before any change, including creating the
   directory.
2. Ownership must be SYSTEM, Administrators or the installer. Take ownership, or refuse when that fails.
3. Strip explicit entries for other identities, reset children to inherit only, then verify the whole tree; fail
   closed on any `icacls` error.
4. Never overwrite a secret in place. Write a new file that inherits the restricted ACL, verify it, move it over the
   old one, and verify the result.
5. `-DryRun` and `--help` must have no side effects, with tests that prove it.

## Playbooks: bounded, and narrow on purpose (deferred, see above)

Each playbook:

- runs under a non-blocking lock;
- has a budget per incident per window, default 1 per 24 h;
- backs off exponentially, starting at 30 min;
- writes a `remediation` event for every attempt;
- once the budget is spent, sends **one** `escalation` (finding, high severity) and stops acting.

| playbook | does | never does |
|---|---|---|
| `rerun-task` | `Start-ScheduledTask` once, only for tasks on the `rerun_tasks` allowlist (idempotent ones) | run a Running task, re-enable a Disabled one |
| `reap-orphans` | runs the configured reaper with `-Apply` (it kills only processes whose parent is dead, re-checking identity) | kill anything itself |
| `disk-low` | prunes files older than N days in policy-listed directories that pass `Test-FleetCacheDirAllowed`; runs `uv cache prune`; **requests** compaction from the storage lane over agent-bus | stop runners, set DRAIN, delete outside cache directories |

`Test-FleetCacheDirAllowed` accepts only an absolute path at least three levels deep that has a `cache` segment. It
rejects anything under Documents, Desktop, source trees, `.git`, `.ssh` or secrets. Nothing in this module reboots,
DRAINs, changes network configuration or deletes user data.

## Alert transport and its security tradeoff

agent-bus is the primary sink. It needs no secret beyond the bus client configuration that S4U tasks already
read.

Discord is the second transport. Its token is read, in order, from `DISCORD_BOT_TOKEN` and then from
`%ProgramData%\fleet-reliability\secrets\discord-bot-token.dpapi`. That file is **LocalMachine**-scope DPAPI with
fixed entropy, and its ACL allows only SYSTEM, Administrators and the owner.

- **Why LocalMachine:** CurrentUser DPAPI `Protect` fails with "Access is denied" in a passwordless logon. This was
  measured over an SSH key logon on a host with no interactive session. S4U is also passwordless.
- **Tradeoff:** any administrator or SYSTEM process on that host can decrypt the token. The file ACL is the real
  boundary.
- **Narrower alternative (not implemented):** a single-channel Discord webhook in place of the bot token. A webhook
  can only post to one channel, so a stolen one cannot read messages or act as the bot.

The token is never logged. It lives only in the process that sends.

## Operations

- **State:** `state\incidents.json` (incidents), `state\playbooks\<hash>.json` (one budget file per playbook and
  incident), `state\disk-samples.json` (48 h of
  samples) and `state\last-sweep.json` (summary).
- **Logs:** wrapped tasks write to `logs\<task>.log`. If a job leaves a background process holding its output
  pipe, the wrapper stops waiting after `FLEET_PIPE_DRAIN_MS` (10 s) and notes that the output was not fully
  captured. The tool's own `-LogPath` is still scanned.
- **Profiles:** paths that use `{profile}` resolve through the known folder, never through `USERPROFILE`, which
  agent sessions redirect.
- **Revert:** run the printed revert commands, or `Unregister-ScheduledTask -TaskPath '\fleet-reliability\'`. Then,
  optionally, remove `%ProgramData%\fleet-reliability`. The ledger is the only record there, so archive it first.
