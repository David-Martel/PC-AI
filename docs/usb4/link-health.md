# USB4 link health on startup and reconnect

Windows gives a USB4 P2P adapter a new interface GUID, MAC, ifIndex and alias whenever the
cable path or hub port changes ([multi-hop routing](multi-hop-routing.md), "Why hub topologies
broke"). Addresses and routes that `pcai-tb route-plan` planned for one instance do not follow
the cable to the next. Nothing re-checked the links after a reboot, a resume or a replug, so a
half-configured /30 went unnoticed until a transfer failed.

[`Invoke-PcaiLinkHealth.ps1`](../../Tools/SystemScripts/Networking/Invoke-PcaiLinkHealth.ps1)
now re-validates the USB4 links against the recorded route plan from triggers, never from the
PowerShell profile.
[`Install-PcaiLinkHealthTrigger.ps1`](../../Tools/SystemScripts/Networking/Install-PcaiLinkHealthTrigger.ps1)
installs, updates and removes those triggers. It also installs the profile and tooling check,
`Tools/SystemScripts/Machine/Invoke-PcaiToolingHealth.ps1`, when that script is present (see
[the profile README](../../Config/PowerShell/Profile/README.md)). Tests:
[link health and triggers](../../Tests/Unit/PcaiLinkHealth.Tests.ps1), with
[source route](../../Tests/Unit/pcai-link-health-source-route-review.json) and
[payload publication](../../Tests/Unit/pcai-link-health-payload-review.json) reviews.

## Install, run, remove

```powershell
# Per user, no elevation needed (Windows tasks or systemd --user units):
pwsh -NoProfile -File Tools/SystemScripts/Networking/Install-PcaiLinkHealthTrigger.ps1 -DryRun
pwsh -NoProfile -File Tools/SystemScripts/Networking/Install-PcaiLinkHealthTrigger.ps1 -RunNow
# Run a check by hand (read-only with -DryRun):
pwsh -NoProfile -File Tools/SystemScripts/Networking/Invoke-PcaiLinkHealth.ps1 -DryRun
# Remove (with -Restore, re-register the task definitions saved before the first install):
pwsh -NoProfile -File Tools/SystemScripts/Networking/Install-PcaiLinkHealthTrigger.ps1 -Uninstall
```

The installer hashes the bytes it will publish, then stages them and their `manifest.json`
before moving the completed directory into `%LOCALAPPDATA%\PC-AI\health-bin\<version>` or
`~/.local/share/pcai/health-bin/<version>`. Linux uses UTF-8 with LF line endings, so an
equivalent Windows checkout selects the same Linux version. The manifest records the source
commit and per-file hashes at first publication. Reinstallation verifies the existing version
and reuses its bytes while refreshing the trigger definitions. The checkout can change branch
without changing what a trigger executes.

Task and service definitions go under `backup/` before replacement. The first install saves
the pre-install definition or a `.none` marker, and `-Restore` uses that baseline. If an
existing payload has different bytes, extra files, a bad manifest, or unexpected permissions,
the installer stops before re-registering triggers and leaves the directory available for
inspection. Check which tasks or units reference that version and save the evidence before
restoring a known-good payload.

## Triggers

| Platform | Unit | Fires on |
|---|---|---|
| Windows, per user (S4U, least privilege) | `\PC-AI\LinkHealth` | logon (+45 s); USB4 adapter arrival and removal; network connect and disconnect; resume; every 30 min |
| Windows, per user (S4U, least privilege) | `\PC-AI\ToolingHealth` | logon (+2 min); network connect (+1 min); every 4 h |
| Windows, SYSTEM (owner opt-in, `-System`) | `\PC-AI\LinkHealthSystem` | boot (+1 min) and the same events, in Apply mode |
| Linux, systemd --user | `pcai-link-health.timer`, `pcai-link-health-monitor.service` | user-manager start (+60 s), every 30 min, and any link or address change on a `thunderbolt-net` interface (`ip -o monitor link address`; one run 10 s after the first event of a burst, not delayed by unrelated veth churn) |
| Linux, systemd --user | `pcai-tooling-health.timer` | user-manager start (+2 min), every 4 h |

The Windows event sources were checked against real USB4NET plug history on dtm-p1gen9
(2026-11-21), not only against documentation:

| Log | Event ID | Meaning for USB4 |
|---|---|---|
| `Microsoft-Windows-Kernel-PnP/Configuration` | 400, 410 | `USB4\...PROT_USB4NET\...` configured, started (adapter arrival) |
| `Microsoft-Windows-Kernel-PnP/Device Management` | 1010 | `...PROT_USB4NET...` "surprise removed as it is reported as missing on the bus" (unplug) |
| `Microsoft-Windows-NetworkProfile/Operational` | 10000, 10001 | connect and disconnect, including the link's "Unidentified network" |
| `System`, `Microsoft-Windows-Power-Troubleshooter` | 1 | resume from sleep |

`Tcpip` 4201/4202 never appeared for these adapters and are not used. Event XPath cannot filter
on a substring of the device instance ID, so PnP events for other devices also start the task.
Three layers keep that cheap: a 15 s trigger delay, `IgnoreNew` for overlapping starts, and a
20 s minimum interval inside the check (`-MinIntervalSeconds`). The interval is a trailing-edge
debounce: a run that starts inside it waits out the rest and then observes, so a reconnect
seconds after a disconnect is still checked rather than dropped. A boot trigger needs
elevation, so the per-user task relies on logon; the owner's SYSTEM task adds boot.

The per-user tasks are registered with the S4U logon type: they run with the user's filtered
(least-privilege) token whether or not the user is logged on, in session 0 without a console
window, and without a stored password. That matters on hosts reached only over SSH (dtm-work,
the NUC, Gen7), which have no interactive session, so an interactive-token task there never
ran. S4U requires the "Log on as a batch job" right; where it is missing the installer falls
back to an interactive-token task and says so. An S4U token has no network credentials, which
these checks do not need (agent-bus, git over HTTPS and SSH keys are unaffected).

On Linux, user units start at boot only when linger is on (`loginctl show-user $USER -p Linger`).
minisforum-ms01 has linger enabled. Enabling it elsewhere is an owner command:
`sudo loginctl enable-linger <user>`.
The installer checks every `systemctl --user` call. If the user manager is unavailable or a
unit does not enable or start, it names the failed call and exits nonzero instead of reporting
the units as enabled.

## What the link check does

1. Lists present USB4/Thunderbolt adapters with GUID, MAC, ifIndex, link state, IPv4 addresses
   and forwarding (Linux: interfaces whose driver is `thunderbolt-net`).
2. Compares them with this host's block of the recorded plan
   (`%LOCALAPPDATA%\PC-AI\link-health\route-plan.json`, or `planPath` in `config.json`):

   | State | Meaning | Severity |
   |---|---|---|
   | `InSync` | Guards match; every planned address, route and forwarding setting is present | ok |
   | `Drift` | Guards match; some planned state is missing (reboot, or replug into the same port) | warn |
   | `NeedsRegeneration` | Guards match, but state the plan relies on and does not own (`already_present`, no apply line) is gone; the generated script cannot restore it, so nothing is applied and the plan must be regenerated | warn |
   | `LinkDown` | A planned adapter, or with no plan every adapter, is present but down (a USB4 P2P adapter exists only while a peer is attached) | warn |
   | `InstanceChanged` | The planned ifIndex names another adapter, the adapter moved, or it is gone | warn |
   | `Absent` | No USB4 adapter and no plan (for example ms01 today) | ok |
   | `Healthy`, `NotInPlan` | Adapters up, but there is no plan or the plan does not name this host | ok |

3. Resolves each Windows path's first edge from the plan. Before probing, it checks the live
   adapter's ifIndex, GUID and MAC, then asks Windows for the best route and source address
   to the planned destination. Both must match the first edge, including the planned next
   hop for a routed path. A mismatch suppresses both probes. The TCP probe binds the first
   edge's source address. After a successful probe, the watcher checks the route again and
   discards the result if it changed. Each reachability entry reports `RouteStatus`,
   `SourceAddress`, `InterfaceIndex` and `NextHop`. TCP 22 is informational because the
   peer's firewall may block the /30 source. See [peer SSH access](peer-ssh-access.md).
4. Reports PC-AI's own USB4 tasks (`\PC-AI\*Thunderbolt*`, `\PC-AI\*Usb4*`,
   `Rclone-*-Thunderbolt-*`) or `pcai-usb4-*` units. It restarts only an allow-listed,
   **enabled**, read-only observer whose last run failed. A scheduled task's own action must be
   known, pass no `-Apply` and not run `Invoke-ThunderboltStartup.ps1` (which applies the enrolled
   controller); otherwise it is reported as `MutatingAction` and left alone. A disabled task stays disabled:
   `\PC-AI\Usb4PeerNetwork` runs `-Apply` as SYSTEM, so enabling it is the owner's decision.
5. Reads the startup observer's receipt (`~/.machine/logs/usb4-boot/current.json`,
   [Windows boot](windows-boot.md)) and includes its state and age.
6. Writes `status.json` and `history.jsonl` under the status root and, when the fingerprint
   of the facts changes, posts one line to agent-bus (topic `status`, thread `machine-hygiene`,
   tags `repo:PC-AI` and `host:<name>`, sender `pcai-link-health-<host>`).

Exit codes: 0 when the severity is ok, 2 when there is something to report, 1 on an error.
If Windows cannot enumerate network adapters, the run exits 1 before comparing
routes, attempting recovery or replacing `status.json`. The saved observation
keeps its original timestamp. Use its age and the scheduler result together when
reading it after an error. A successful enumeration with zero USB4 adapters
still produces an ordinary observation. See the [adapter error test review](../../Tests/Unit/pcai-link-health-adapter-error-review.json).

### Drift and the auto-apply switch

The default is Observe. On `Drift` the check writes this host's guard, preflight and apply
lines from the plan to `pending-apply-<plan-id>.ps1` under the status root and prints the
command to run it elevated. It changes nothing itself.

Apply mode runs only in the SYSTEM task, and only when all of these hold:

- the process is elevated;
- `%ProgramData%\pcai-usb4\link-health\auto-apply.json` exists, is owned by Administrators or
  SYSTEM, is not a reparse point, grants no other account a write right, and reads
  `{"schema": "pcai.link-health-auto-apply/v1", "enabled": true, "planPath": "<path>"}`;
- the `planPath` file passes the same ownership checks (the check executes command text from
  the plan, so a user-writable plan or policy would be a privilege escalation);
- the status root `%ProgramData%\pcai-usb4\link-health` is itself admin-owned and grants Users
  only read access. `-System` creates it that way, because a directory created under
  `%ProgramData%` otherwise inherits "Users may create files", which would let a standard user
  plant `config.json` or pre-create `apply\`. The installer refuses an existing `pcai-usb4` or
  `link-health` that another account owns. Apply mode ignores a `config.json` that is not
  admin-owned. An elevated Apply run whose status root fails the check writes nothing there
  (no lock, status or review script, which a planted link could redirect), only prints, and
  reports a warning (exit 2). A root that passes but holds an entry that does not (a file
  planted before it was protected) counts as failing; `-System` refuses to install over such
  an entry and names it. `-System` checks the ancestors above `pcai-usb4` before it creates or
  re-permissions any directory, so a junction there cannot redirect those ACL changes. It makes
  Administrators the owner of every file it writes (client Windows otherwise leaves them owned by
  the installing user) and verifies each before it registers the task. Every ancestor of a trusted path, the volume root included, must be owned by
  Administrators, SYSTEM or TrustedInstaller, must not be a reparse point, and must not let
  another account delete, rename or re-permission it (letting Users create entries, as C:\
  and `%ProgramData%` do, is allowed);
- the interpreter is trusted too. The task runs `pwsh.exe` as SYSTEM before any script check,
  so `-System` registers only a `pwsh.exe` whose file, directory (no other account may add
  files to it) and ancestors pass the same checks. It tries the installer's own `$PSHOME`,
  then `Program Files\PowerShell\7`, and otherwise refuses, dry run included. A per-user or
  portable PowerShell fails: dtm-p1gen9's `C:\codedev\tools\PowerShell\7` grants
  Authenticated Users Modify, and that host has no machine-wide PowerShell 7 (checked
  2026-11-21), so `-System` refuses there until the MSI is installed. The NUC, dtm-work and
  Gen7 have the MSI install, owned by SYSTEM;
- the state is `Drift`, so every guard (ifIndex, MAC, GUID) still matches.

It then runs the plan's own guard and apply lines, which create state only when absent and
journal it (`%ProgramData%\pcai-usb4\route-plan\<plan-id>.journal`), and re-validates. Rollback
uses the plan's rollback lines as before.

**What auto-apply cannot cover.** When the adapter instance changed (new GUID after a hub-port
change), the plan's guards fail by design and nothing is applied. `route-plan` needs a fresh
observation of every host on the path, so the plan must be regenerated and replaced; the
check says so in its status and bus line. The switch is off by default and stays off until the
owner writes the policy file.

## Configuration

The check reads an optional `config.json` in its status root and refuses unknown keys.

| Key | Default |
|---|---|
| `planPath` | `<status root>/route-plan.json` |
| `busEnabled`, `busThread` | `true`, `machine-hygiene` |
| `restartAllowList` | `pcai-usb4-observer.service` (`\PC-AI\Configure-ThunderboltSwitchRoutes` is not listed: its deployment can run the startup coordinator, which applies) |
| `tcpProbePort` | `22` (0 disables) |

## Limits

- The plan is IPv4 and Windows-only, so Linux hosts get discovery, link state and reachability
  but no plan comparison or apply.
- A plan's observation TTL is not enforced for comparison or apply; the per-ifIndex MAC and GUID
  guards are what bind it to the hardware. Regenerate after any replug that changes them.
- The probe checks local egress and transport reachability. Use a pinned SSH host key
  and verify both directions for authenticated peer readiness.
- The Linux Pester list (`Tests/linux-ci-pester.txt`) does not include these files yet; the
  Windows CI runs all of `Tests/Unit`.
