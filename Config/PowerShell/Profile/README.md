# PowerShell Profile Bootstrap

`Microsoft.PowerShell_profile.ps1` is the sanitized, Git-backed bootstrap for
this workstation's PowerShell 7 profile system.

Deploy the same file to both possible user-document roots:

- `%USERPROFILE%\Documents\PowerShell\Microsoft.PowerShell_profile.ps1`
- `%USERPROFILE%\OneDrive\Documents\PowerShell\Microsoft.PowerShell_profile.ps1`

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
