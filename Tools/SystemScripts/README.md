# System Scripts

This folder is the repo-owned home for workstation scripts that were previously
spread across Task Scheduler actions, `C:\Scripts`, `~\.machine`,
`~\.local\bin`, `~\bin`, OneDrive PowerShell script folders, and selected
`unifi_api` startup helpers.

## Migration Tool

Use the migration script from the repo root:

```powershell
pwsh .\Tools\Migrate-SystemScriptsIntoRepo.ps1 -DryRun
pwsh .\Tools\Migrate-SystemScriptsIntoRepo.ps1 -Apply
```

The script supports `-h`, `--help`, `-DryRun`, and `--DryRun`. Apply mode
emits a JSON report under `Reports\system-script-migration-<timestamp>.json`.

## Scheduled Tasks Repointed

The 2026-04-30 migration repointed these PowerShell scheduled tasks to repo
paths:

| Task | Repo script |
|------|-------------|
| `\BW-Auto-Unlock` | `Tools\SystemScripts\Machine\Unlock-BwVault.ps1` |
| `\Bitwarden\Initialize-MachineSecrets` | `Tools\SystemScripts\Machine\Initialize-SecretsAtBoot.ps1` |
| `\DevEnvironmentStartup` | `Tools\SystemScripts\TaskScheduler\PowerShellScripts\Start-DevEnvironment.ps1` |
| `\Gemini-CLI-Update-stable` | `Tools\SystemScripts\TaskScheduler\Gemini\check-releases.ps1` |
| `\LspmuxServer` | `Tools\SystemScripts\LocalBin\Start-LspmuxServer.ps1` |
| `\PowerShell\ProfileLogSync` | `Tools\SystemScripts\Machine\Sync-ProfileLogs.ps1` |
| `\UDP Socket Monitor` | `Tools\SystemScripts\Machine\Monitor-UDPSockets.ps1` |
| `\UnifiUdmDriveStackStartup` | `Tools\SystemScripts\unifi_api\scripts\windows\Start-UDMDriveStack.ps1` |

`UnifiUdmDriveStackStartup` remains disabled until OneDrive has a clean
post-repair sync-health window.

## Source Groups

- `C-Scripts`: legacy WSL, Docker, VHD, startup, and registry-credential tools
  moved from `C:\Scripts`.
- `HomeRootArchive`: archived home-root scripts with historical registry,
  network, cloud-sync, GCP, npm, MCP, and encoding repair utilities.
- `LocalBin`: selected system-modifying scripts moved from `~\.local\bin`.
- `Machine`: selected scheduled-task scripts moved from `~\.machine`; secret
  modules, logs, and cache files intentionally remain outside this repo.
- `TaskScheduler`: scripts that are direct scheduled-task targets or companions.
- `UserBin`: selected system-modifying scripts moved from `~\bin`.
- `unifi_api`: UDM Windows and on-boot helper scripts required by the migrated
  UDM drive-stack task.

## Safety Rules

- Treat migrated scripts as production workstation automation, not scratch
  snippets.
- Prefer `-DryRun` and evidence capture before enabling or registering any
  startup/logon behavior.
- Do not force-add logs, cache files, secret material, or backup files that are
  ignored by git.
- Before enabling Task Scheduler usage for a migrated script, check whether it
  needs `-h`/`--help`, `-DryRun`, idempotency, structured logs, and nonzero exit
  codes for failure.
