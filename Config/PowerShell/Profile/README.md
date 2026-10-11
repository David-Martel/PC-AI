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
