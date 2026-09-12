# Governor watchdog window — 2026-09-12

The existing `PC-AI Process Lasso Governor Watchdog` task uses an interactive
user principal. Its PowerShell action omitted `-WindowStyle Hidden`, allowing
a terminal window to appear at logon. The maintained registration script now
includes that argument, and the existing task was updated in place.

Before mutation, the complete task XML was saved in the workstation audit at
`C:/Users/david/audits/2026-09-12/processlasso-task-before.xml`. The update changed
only `Task.Actions.Exec.Arguments`. Comparing the complete resulting XML with
the original plus that single expected replacement passed. Triggers, principal,
execution limit, restart policy and enabled state were preserved. The task was
not started, and no Process Lasso process was restarted.

The real registration `-DryRun` displayed the hidden PowerShell action.
Repository-configured PSScriptAnalyzer reported zero warnings/errors. The
existing Pester dry-run case now asserts `-NoProfile -WindowStyle Hidden`
in the emitted action, so removing the hidden flag breaks the regression.
The global generic analyzer separately reports four pre-existing Write-Host
advisories excluded by this repository's analyzer configuration.

This change concerns the governor watchdog. It does not change the separate
AgentHub runner, VHD mount, cloud-client or Gemini updater task policies.
