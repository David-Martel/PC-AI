# Workstation process, startup, storage and coordination review

Date: 2026-10-04. Host: DTM-P1GEN7. Scope: bounded live observations and selected
reversible repairs. The maintained queue is in [boot.TODO.md](../boot.TODO.md) and
[optimization.TODO.md](../optimization.TODO.md); older dated reports are retained.
Raw process commands, task XML and credential-adjacent inventories remain in private
operator evidence, outside Git. No secrets or research data are included here.

## Live resource evidence

Six one-second CPU samples averaged 82.29% Processor Time (range 78.37–89.12%),
with 22 logical processors. Frequency-scaled Processor Utility is not the busy
percentage. The process CPU census did not reconcile about 35 percentage points;
non-atomic sampling, exited/new processes and protected/virtual accounting remain
unresolved. Do not attribute all host load to the listed processes.

Commit sampled approximately 85.9 GB of 106.8 GB (80.5%); available memory was
approximately 14 GB. Selected PowerShell and Node process private bytes totaled
about 9.1 GB. Multiple MCP generations were direct children of five live Codex
clients. No selected immediate parent was shown absent/reused; process counts
alone do not establish orphaned processes or crash/restart loops.

HapticService repeatedly used approximately one logical core during two sample
windows. Its executable identifies Wyvrn/Razer, not the Sensel touchpad. Active
Python work was attributable to live usability/Clarius owners. No priority change,
service stop or termination was performed on that evidence.

Docker had 18 running containers and one created container; aggregate sampled CPU
was approximately 0.60% in Docker's CPU units. Seven Discord MCP containers sampled
about 273 MiB combined and 0.01% CPU. This does not support Docker as the main cause
of host CPU pressure. Current Codex configuration already disables MCP_DOCKER;
older active client/container lifecycles require owner-confirmed closure.

Audit limitation: an early read-only helper argument error invoked wsl.exe without
arguments, which exited after about ten seconds. A default-distro start cannot be
excluded because its prior state was not captured. Subsequent discovery used the
correct bounded arguments; no intentional distro start/stop was requested.

## Startup and storage evidence

Four enabled vendor tasks point to absent programs. NI Package Manager and current
Lenovo Vantage services are installed; alternate installer binaries are not proven
equivalent actions. Preserve exports before disabling, retain definitions and use
Enable-ScheduledTask for rollback. Task last-result codes remain historical.

The startup optimizer recognized only legacy approval states 0x02/0x03 while
overwriting other states and padding short arrays during Apply. Most observed
records were 12 bytes and had unknown first bytes. A conservative state/schema
guard and complete byte preservation are required before use. No blanket startup
changes are justified by the current inventory or an unmeasured next-boot benefit.

Existing mount-health validation passed: three VHDs attached/healthy, three mount
task results zero and no FilterManager event 3 in the recent hour. Its boot report
predates the current boot. Selected volume dependencies:

| Volume | Live dependency | Observation |
| --- | --- | --- |
| C: | system/developer data | NTFS, approximately 11.3% free |
| D: | archive/shared-development storage | NTFS, approximately 21.0% free; DO-NOT-WIPE policy |
| T: | VM storage | ReFS, approximately 36.5% free |
| F: | T: cloud-cache VHD | NTFS healthy; registered Proton root absent |
| W: | D: shared-dev VHD | NTFS healthy, approximately 47.5% free |

The separate unattached D: cloud-cache VHD occupies about 1.82 TiB physically and
has a different identity from the live T: VHD. Custody and independent data parity
are unknown. Neither deletion nor compaction is authorized by age, size or a
healthy replacement mount. Preserve Proton's registration pending provider/data
reconciliation. Cloud virtual G:/J: volumes are not independent physical capacity.

Six one-second disk-counter samples showed zero sampled current queue depth.
C: averaged about 0.083 ms per transfer; the other queried disks were idle in that
window. This supports no sustained storage queue in this observation, not storage
throughput capability or a future-workload guarantee.

## Fleet and remote-access boundaries

The Windows AgentHub service is healthy on localhost:18400. Localhost:8400 refused
connections. This local instance is a separate island; the client configuration
intentionally prefers the ASUS fleet hub. Do not change fleet routing to make a
local health probe pass. Watchdog candidate-array support requires current owner
coordination and must retain hub-identity checks.

Existing Tailscale access successfully reached DTM-WORK using SSH with strict
verification of an existing key. No VPN or default route changes were needed.
A machine-local `dtm-work-vpn` include now provides repeatable access with that
existing key, explicit port, connection timeout and keepalives. Effective SSH
configuration and a live hostname command validated it; original LAN aliases
remain available. Compression is off; no transport throughput gain is claimed.
The remote SSH shell emits a missing canonical-profile message and duplicate
missing-inference-library warnings. Its unqualified Write-Warning resolves to a
PC-AI wrapper; aggregate module initialization also imports Evaluation twice.
Qualified warning dispatch is the minimal proposed fix, not yet deployed. The
local and remote OneDrive shim hashes differ, and no local canonical runtime is
installed on DTM-WORK. Preserve sync/source custody before profile replacement.
The Super-NUC's actual identity/owner is not yet established.

The public agentbus.dtmventures.com health request received a Cloudflare 403
challenge. The coordinator reports a deployed Worker, but machine-client access
is not validated. Preserve authenticated fleet coordination and resolve this with
the deployment owner instead of bypassing access policy or assuming backend failure.

ASUS/3066 deployment and 0060 GPU/HIL holds remain owner-controlled. This maintenance
pass does not claim fleet runtime acceptance, keyboard recovery, physical
Thunderbolt peer enumeration, next-boot reliability or measured performance gain.

## Repair and validation status

Four obsolete tasks were disabled after fresh action-path and non-running checks:
NIUpdateServiceCheckTask, NIUpdateServiceStartupTask, RNIdle Task and
Lenovo/Vantage/StartupFixPlan. Readback confirmed Enabled=false and otherwise
identical XML after normalizing Task Scheduler's omitted default Enabled=true.
NI and Lenovo services remained running. No task was unregistered/repointed.
Before/after XML hashes and exact Enable-ScheduledTask rollback commands are in
the private repair receipt. Their historical failure results are not reset.

Both baseline and per-command `codex mcp list --json` checks exited zero. The
following optional recipe disables four servers for a future bounded maintenance
session while retaining agent-bus and Context7. Choose it only when the task does
not require those four capabilities:

```powershell
codex -c mcp_servers.serena.enabled=false `
      -c mcp_servers.google-workspace.enabled=false `
      -c mcp_servers.playwright_direct.enabled=false `
      -c mcp_servers.qmd.enabled=false
```

The stored configuration SHA-256 remained unchanged. This validates effective
configuration, not a new session's resource footprint or a speedup. No current
client was restarted. Per-command dotted-path overrides and the server enabled
setting are supported by [OpenAI's CLI source](https://github.com/openai/codex/blob/main/codex-rs/utils/cli/src/config_override.rs)
and [MCP documentation](https://developers.openai.com/codex/mcp/).

The optimizer accepts only the legacy enabled 0x02 state with a complete 12-byte
record, skips unsupported layouts/states and clones the entire byte array before
changing its first byte. It creates one full original backup inside the existing
per-entry approval, before the first registry write. Apply/WhatIf produces neither
backup files/directories nor registry writes. No live Apply was run.

Post-format validation passed 105/105 cases (18 focused safety cases and 87 existing
structural cases), with zero skipped and 92.31% command coverage. The focused suite
executes the real script and mocks external registry/CIM calls; the earlier broken
implementation failed 13/15 initial cases. Repository analyzer settings yielded
zero warning/error findings. Independent review approved exact tested source/test
hashes and verified eight before/after task XML hashes. Publication and subsequent
runtime follow-ups are tracked separately in private receipts and the maintained
backlog. Private receipts are held under the operator's cache
load-maintenance-20261004 directory; exported XML provides task rollback custody.
