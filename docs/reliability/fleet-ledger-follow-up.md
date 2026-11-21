# Fleet reliability: follow-up items from the #346 sweep

Tracked in the repository because issues are disabled on PC-AI. Linked from [fleet-ledger.md](fleet-ledger.md) and from the #346 PR reply. Tick an item when its fix merges.

Follow-up to #346 (fleet reliability ledger). The fail-closed class sweep for #346 had three sources:

- my sweep (23 rows);
- a read-only codex audit of `3278dbfc` (109 inputs, 89 marked "not fail-closed");
- the autopilot reviews.

#346 fixes the 23 approved rows, the P1 credential redaction, and two ledger data-loss items: forward-only retry for an alert nobody received, and a torn last line. This file tracks the rest of the codex table. Row numbers refer to that table (`#n`). Each item names its input and the wrong outcome.

The autopilot reviewer may cite this file for any row listed here.

## A. Cheap, concrete fixes (follow-up PR)

Validation (refuse at load, or fall back with a warning):

- [ ] #16 `task_stale_factor` has no upper bound: `100000` makes an overdue task fresh. Cap it at 100.
- [ ] #99 `fleet_checks: null` silently disables runner and workflow checks. Require a boolean.
- [ ] #13/#78 `alerts: null` silently disables Discord. Require an object.
- [ ] #67 `FLEET_LEDGER_MAX_BYTES` set to 0 or -1 rotates on every append. Enforce a 64 KiB minimum and warn.
- [ ] #10/#72 `FLEET_BUS_TIMEOUT_SEC='90%'` throws mid-send. Validate it and fall back to the default.
- [ ] #105 `Send-FleetDigest -SinceHours -1`. Add `[ValidateRange(1,8760)]`.
- [ ] #97 `-Detector typo` gives a healthy-looking sweep. Reject unknown detector names.
- [ ] #95 `-FailurePattern @()`, or an invalid regex such as `[`. Require non-empty, compile before launching, and record the run anyway.

Task detector:

- [ ] #19 A benign code with missing `Reasons` reads as benign. Require a `Last run` reason.
- [ ] #22 `task-expected` with unknown `State` or a garbage `LastRunTime` gives ok. Make it neutral, with a `task-report` failure.

Disk:

- [ ] #32 `TotalBytes = 0` gives a percentage of 100. Make it unknown.
- [ ] #34 A malformed sample with no `free` skews the slope. Drop invalid samples.

DRAIN:

- [ ] #38 A remembered `{}`, or a future `started_at`. Validate the file; treat a future time as Unknown.
- [ ] #39 "has not ended" matches `ended`, and messages with no timestamp are used. Match the exact lifecycle phrases; ignore untimestamped messages.
- [ ] #40 A future START gives negative minutes and stays within budget. Treat it as Unknown.

Workflow:

- [ ] #43 A future `created_at` gives a negative age, which reads ok. Make it neutral.

Outbox and ledger:

- [ ] #64 `{"event":{}}`, or a missing `forward`. Quarantine; never write a null-filled line.
- [ ] #68 Rotation moves without `-ErrorAction Stop` can overwrite an archive. Stop on error so the event stays queued.
- [ ] #12/#63/#76 Quarantine names can collide. Add a unique suffix.
- [ ] #74 The replay file name can repeat within one timestamp. Add a unique suffix.
- [ ] #101 The digest reads only `ledger.1` and the current file. Read every rotated file the window needs.

Sweep and wrapper:

- [ ] #89 The wrapper's effective policy leaked after a throw. Fixed in #346 (try/finally); verify and close.
- [ ] #15/#33 Overlapping sweeps share the report and samples. Add a non-blocking sweep lock; a second sweep exits "already running".
- [ ] #94 A descendant holding the output pipe gives truncated output that reads ok. Report output not fully captured (`OutputComplete`) as its own check.
- [ ] #29 A fresh zero-byte log after the run reads clean. Report it as an empty log.
- [ ] #50 An orphan check with no creation dates. Make it neutral instead of "no orphans".

## B. Accepted risk (no change planned)

| Row | Why accepted |
|---|---|
| #2, #3 | Clock moved backwards, or time-only timestamps. Clocks are NTP-synced; skew is out of scope. |
| #4, #5, #6, #14 | Stale `FLEET_HOST` / `FLEET_RELIABILITY_HOME` / `AGENT_BUS_EXE` overrides or redirected profiles. These are operator configuration, set by the installer, not inputs. |
| #9 | A child exiting 0 just after its deadline is accepted. It is a poll-granularity race of under a second. |
| #11 | A pipe fault labelled complete. This is .NET process I/O semantics, outside the module's control. |
| #18 | Removing a rule from policy resolves its keys. Removing a check is an owner action, and resolving it is intended. |
| #20 | A Stalled task's `run` axis reads ok. TSH sets Stalled only when the result axis is healthy. |
| #23, #35 | A task missing from `Get-ScheduledTask`, or a missing gh page. Both throw on failure; partial success is not reported by the source. |
| #24 | An unknown explorer owner gives "no session". This fails loud (degraded alert), not open. |
| #26, #27 | Log enumeration races, or a rule without a task. Documented: rules should name their task. |
| #30 | An optional drive unplugged resolves its incident. A vanished subject is intended; required drives fail closed. |
| #37 | The remembered DRAIN state is lost AND a START has aged out. Needs two independent losses; documented. |
| #42 | A runner row reporting online. The gh API is the source of truth. |
| #44, #46, #52, #31 | `"NaN"`, negative and blank thresholds. Now refused by policy validation in #346. |
| #45, #47, #48 | Postgres conf inference. Removed in #346 (policy-only limits, listener check, drift report). |
| #53 | One failing gh scope discards that detector's other observations. Detector-level error is the design; the error stays open. |
| #54 | A pipe in a name collides keys. Windows task names cannot contain `|`, and other names come from policy. |
| #56 | An older wrapper result submitted after a newer failure. Submissions are serialized by the scheduler; out-of-order delivery is not a supported input. |
| #58, #59, #104 | Hand-edited state fields (`state='opn'`, `consecutive_good=99`, a future `flapping_until`). The state file is written only by the module, and a corrupt file is now preserved and announced. |
| #62 | A wrapper's alert can wait for the next flush. Every sweep (15 min) flushes; the delay is bounded. |
| #65 | A policy scope removed while incidents are open. Owner action. |
| #66 | `kind='incident-resolved', status='failed'`. Only the module produces events. |
| #69 | A torn line. Fixed in #346. |
| #70 | An alert nobody received. Fixed in #346 (forward-only retry). |
| #71 | A long log path truncates the bus body. The body is sized to fit; Discord gets the same body. |
| #73, #75, #79 | agent-bus, gh or Discord exiting 0 without doing the work. Their success is trusted; a positive acknowledgement protocol is a separate design. |
| #91, #92 | An equal-size same-prefix log replacement, or BOM-less UTF-16. Rare; documented. |
| #96 | A comma list passed as one CLI argument. The CLI already refuses quoted lists; the rest is caller error. |
| #98 | An empty detector response clears that detector's `detector-error`. The detector ran successfully, so this is intended. |
| #102, #106 | Damaged or future-dated lines in the digest window. Unreadable lines are now counted (#346); future dates are out of scope. |
| #103 | Fleet digest beyond `--limit 500`. Fleet mode only; this host's open incidents come from the durable store. |
| #107 | A stale disk snapshot in the digest. Fixed in #346 (sweep age stated; degraded when stale). |

The full codex table is posted on #346. Rows not listed here were already fail-closed, or are covered by #346's 23 rows.
