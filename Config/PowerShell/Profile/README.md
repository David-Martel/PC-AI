# PowerShell Profile Bootstrap

`Microsoft.PowerShell_profile.ps1` is the sanitized, Git-backed bootstrap for
this workstation's PowerShell 7 profile system.

Deploy the same file to both possible user-document roots:

- `%USERPROFILE%\Documents\PowerShell\Microsoft.PowerShell_profile.ps1`
- `%USERPROFILE%\OneDrive\Documents\PowerShell\Microsoft.PowerShell_profile.ps1`

Run `Tools/SystemScripts/Machine/Install-PcaiProfileShim.ps1` (`-WhatIf` first) to do it: it writes the
source to each root that exists, copies every file it replaces to `~/.machine/backups/profiles` as
`<host>.<folder>.<file>.bak-<sha256 prefix>` with a UTC-stamped line in `receipts.jsonl`, and leaves a copy
that already matches alone. It never touches the private profile. Tests:
`Invoke-Pester -Path Tests/Unit/PcaiProfileShimInstall.Tests.ps1`.

The bootstrap intentionally contains no credentials or machine-specific secret
values. It delegates to the private machine-local canonical profile at
`%USERPROFILE%\.config\powershell\Microsoft.PowerShell_profile.ps1` and keeps
the local module root authoritative.

Do not copy the canonical profile into this public repository without first
decomposing it into sanitized public configuration and private local fragments.
After deployment, compare SHA-256 hashes of the Git source and both installed
copies, run PSScriptAnalyzer, and start a fresh interactive PowerShell session.

The public shim module-qualifies its built-in commands. Early command discovery
can otherwise import unrelated wildcard-export modules before the private
profile is reached. Keep the missing-private-profile warning visible: this
bootstrap is not a substitute for that private configuration.

Preserve the SSH noninteractive early-return guard when deploying. Warning or
profile output can corrupt command/subsystem transfer protocols. The guard
returns before environment/module initialization for SSH command, encoded
command and file invocations; normal local startup retains diagnostics.

Regression check:
`Invoke-Pester -Path Tests/Boot/ProfileBootstrap.Tests.ps1 -Output Detailed`.

The bootstrap removes any already-loaded PSReadLine module when console input
is redirected or `PCAI_DISABLE_PSREADLINE=1`. ConsoleHost then uses a global
`PSConsoleHostReadLine` function backed by `Console.ReadLine()`. The environment
switch is an explicit fallback for console-reader crashes; it costs PSReadLine
editing, prediction, and history features. Clear the switch and start a new
shell to restore normal behavior. Removal failures remain visible as warnings.
When the process variable is absent, the bootstrap reads the persisted User
setting directly, so a fresh shell launched by an older terminal parent still
honors it. An explicit process value, including `0`, takes precedence.

The October 2026 audit found the bundled and OneDrive PSReadLine 2.4.5 DLLs
identical. Selecting the bundled copy avoids cloud module loading but is not a
workaround for the PSReadLine console-input-loss exception. This mitigation does
not explain or repair a whole-PC hardware or kernel restart.

PowerShell documents the console reader extension at
<https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_psconsolehostreadline?view=powershell-7.6>.

Behavior checks:
`Invoke-Pester -Path Tests/Boot/ProfileBootstrap.Tests.ps1,Tests/Boot/CrashProfileSafety.Tests.ps1 -Output Detailed`.

## Startup fast paths and the WinGet hook

Measured on 2026-11-21 (median of three `pwsh -NoLogo -Command exit` runs): a Claude agent-home took
17-31 s and the NUC 1.7 s, because the bootstrap fell through to the PowerToys
`Import-Module Microsoft.WinGet.CommandNotFound` block that the installer appends after it. That
module is a cloud-only OneDrive file, and the import throws where winget is not installed.

- The bootstrap now ends every path in `return`, so the appended PowerToys block is unreachable.
- An interactive console shell gets a one-shot command-not-found hook that imports the WinGet
  module on the first unresolved command, and only when `winget` is on PATH. Shells started with
  `-NonInteractive`, redirected input or a non-console host do not get the hook.
- When `$HOME` contains `\agent-homes\` the bootstrap returns before touching the environment,
  the disk or the network. `PCAI_PROFILE_FAST=1` is the same opt-out for any other tool.
- A shell started with `-NonInteractive` still loads the private profile, because tools rely on
  its PATH ordering. Skipping it for `-NonInteractive` is an owner decision (see the PR).

`Microsoft.PowerShell_profile.minimal.ps1` is a deployable private profile for hosts that need only
PATH ordering and module roots (it does no imports, network, process or secret work, and keeps the
vault behind `Connect-MachineVault`). Copy it to `%USERPROFILE%\.config\powershell\`. Tests:
`Invoke-Pester -Path Tests/Boot/ProfileBootstrap.Tests.ps1,Tests/Boot/MinimalProfile.Tests.ps1`.

## Tooling health on logon and network change

[`Invoke-PcaiToolingHealth.ps1`](../../../Tools/SystemScripts/Machine/Invoke-PcaiToolingHealth.ps1)
checks that the profile chain still loads and that the tools and repositories it relies on are
what they should be. It runs from the `\PC-AI\ToolingHealth` scheduled task or the
`pcai-tooling-health.timer` user unit, never from the profile, so it cannot slow an interactive
shell. The trigger installer and its schedule are described in
[USB4 link health](../../../docs/usb4/link-health.md); tests are in
[PcaiToolingHealth.Tests.ps1](../../../Tests/Unit/PcaiToolingHealth.Tests.ps1).

```powershell
pwsh -NoProfile -File Tools/SystemScripts/Machine/Invoke-PcaiToolingHealth.ps1 -Quick -DryRun   # profile + shims, read-only
pwsh -NoProfile -File Tools/SystemScripts/Machine/Invoke-PcaiToolingHealth.ps1 -NoFastForward  # full pass, report only
```


- **Profile chain.** Every profile file exists and parses. The deployed shim copies are compared
  with `Config/PowerShell/Profile/Microsoft.PowerShell_profile.ps1` at the PC-AI checkout's
  `origin/HEAD` (line endings normalized). The full pass times a child `pwsh -Command 'exit 0'`
  that loads the chain, against a 5 s budget.
- **Tool shims.** Each tool resolves on PATH and prints a `--help` banner (all started
  concurrently). Shadowing copies further down PATH are listed, with a size and SHA-256
  comparison. A tool is identified by path and `--help`, never by `--version` alone, because
  `~/bin` and `~/.local/bin` shadow system commands at different versions.
- **Repositories.** Checkouts whose `origin` is dtm-claude, dtm-codex, PC-AI, google-access or
  agent-hub (configurable) are fetched. A checkout is fast-forwarded with `git merge --ff-only --no-overwrite-ignore`
  only when it is on its default branch, has no tracked changes, no merge, rebase,
  cherry-pick or index lock, and no local commits. A merge that would overwrite an ignored local file is refused
  (`FastForwardFailed`), since that file was never committed. Everything else (`OffDefault`, `Detached`,
  `Dirty`, `Busy`, `Ahead`, `Diverged`, `FetchFailed`) is reported, never changed. If
  someone switches branch or detaches HEAD between the last check and the merge, the merge
  can move that ref instead, even if they switch back afterwards. The merge carries a per-run
  tag in its reflog message (`GIT_REFLOG_ACTION`) and forces `core.logAllRefUpdates` on, so
  every ref it moves is recorded; the outcome comes from those records, not from what is
  checked out afterwards. Only the sampled branch moving is `FastForwarded`; nothing moving is
  `Changed`. Any other ref it moved is reported as `FastForwardMisapplied` (severity warn),
  with that ref's previous tip and the `git update-ref` / `git reset --keep` command that
  undoes it. Nothing is reset automatically: a reset acts on whatever is checked out when it
  runs, and the person may have switched again. Branch names compare case-sensitively. Linked
  worktrees that are merged, whose upstream is gone, prunable or detached, stale local
  branches and your open pull requests are listed. Nothing is deleted.

`-Quick` checks only the profile chain and the shims and does not run `--help`; it took about
1 s inside the script on dtm-p1gen9, where `gh`, `rclone` and `pwsh` each need 1.5-2 s just to
print `--help`. The scheduled full pass took 15-25 s with eight checkouts.

It writes `status.json` under `%LOCALAPPDATA%\PC-AI\tooling-health` (or
`~/.local/state/pcai/tooling-health`) and posts a one-line summary to agent-bus (thread
`machine-hygiene`, sender `pcai-tooling-health-<host>`) when the result changes or a checkout
was fast-forwarded. An optional `config.json` there overrides `repoNames`, `repoRoots`,
`repoPaths`, `tools`, `profileLoadBudgetSec`, `busEnabled` and `busThread`; unknown keys are
refused. Exit codes: 0 ok, 2 something to report, 1 a parse error or missing git or pwsh.
