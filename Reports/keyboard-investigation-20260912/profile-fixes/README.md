# Deployed local PowerShell profile repairs

The canonical `~/.config/powershell/Microsoft.PowerShell_profile.ps1` was repaired
in place on September 12, 2026. Both Documents/PowerShell loaders reference this
file. Its containing directory is not a Git repository; this folder preserves the
reviewed patch and reproducible isolated tests. The complete original profile
backup remains local under `.codex/research/keyboard-profile-fixes-20260912/` and
is deliberately excluded from publication.

## Fixed behavior

- Honor `PS_SKIP_PROFILE_ACCELERATOR=1` before importing the accelerator.
- Parse host arguments without executing or reinterpreting command payloads;
  `-NoExit -Command` can take the interactive path, while file/encoded/noninteractive
  invocations retain minimal startup.
- Separate loading and successful-load guards so terminating partial failure is
  retryable and concurrent/reentrant initialization is suppressed.
- Reload with **`. Import-Profile`** in the caller's scope. Ordinary `Import-Profile`
  now explains the required dot invocation instead of silently doing nothing.
- Mark the optional bootstrap loaded only after its file exists and executes
  successfully; restore the prior bootstrap context and error preference.
- Close only the profile-owned history writer and replace only its own exit
  subscription/job during reload. Preserve unrelated event subscribers.
- Capture explicit history state/bound commands for the exit callback. The actual
  PowerShell exit event did not forward registration MessageData, so the callback
  uses a closure verified by real child-process exit tests.

## Validation and deployment boundary

**32 tests passed**, zero failed or skipped. Tests extract the real profile AST
blocks/functions and replace external initialization boundaries; they do not
execute the complete credential/bootstrap pipeline. Owned/unowned exit fixtures
launch fresh `pwsh -NoProfile` children, verify writer closure only when owned and
history merging in both cases, with a ten-second process deadline. The standalone
exit smoke also completed with closure and merge markers.
The same suite against the original backup had ten passing and 22 failing cases.

The final file has SHA256
`d34d4e5a9b34940d97217e0a34ade3de5076aa5e1ac58c2015f9281fd60c3340`.
It parses without errors. Profile analyzer findings remain the two pre-existing
rules (empty catch and Unicode/no-BOM); no findings were added. Original encoding
and existing line-ending style were preserved. See the adjacent JSON evidence.

Example explicit validation against the deployed profile:

```powershell
$container = New-PesterContainer -Path ./ProfileFixes.Tests.ps1 -Data @{
    ProfilePath = "$HOME/.config/powershell/Microsoft.PowerShell_profile.ps1"
}
Invoke-Pester -Container $container -Output Detailed
```

The patch is machine-specific evidence, not an automatic installer. Validate the
base hash and review before applying it elsewhere. Existing user shells were not
reloaded; new profile-loading sessions read the repaired source. No full interactive
startup performance claim or keyboard cure is made. In particular, the subsequent
hard-press recovery of Down requires physical investigation independently of these
shell reliability fixes.

Provenance follow-up: the existing Documents/OneDrive sync plan preserves the
869-byte loaders, not the canonical profile body. No deployment sync was run.
Keep the reviewed patch and local backup until canonical-source ownership and
deployment coverage are reconciled. Real module initialization, credential startup,
live history/log contents and desktop input were outside these isolated tests.
