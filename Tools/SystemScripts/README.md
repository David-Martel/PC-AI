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
- `Machine`: selected scheduled-task scripts and reviewed credential backend
  code. Credential values, private machine configuration, logs and cache files
  remain outside this repo.
- `TaskScheduler`: scripts that are direct scheduled-task targets or companions.
- `UserBin`: selected system-modifying scripts moved from `~\bin`.
- `unifi_api`: UDM Windows and on-boot helper scripts required by the migrated
  UDM drive-stack task.

## Bitwarden archive maintenance

`Machine/Update-BwArchive.ps1` is the canonical archive entrypoint; the installed
`~/.machine/Update-BwArchive.ps1` forwards to it. `-DryRun`, `-WhatIf`, and `--help`
return before vault access. A real run validates the unlocked session, protects
and verifies archive storage ACLs, then synchronizes and exports into private
staging before publishing the timestamped and latest archives. Native failures
and timeouts leave the previous latest archive intact.

Run the Windows fixture suite in `Tests/LocalMachine` after changing the archive
or installed machine credential modules. It does not access the real vault.
Plain JSON remains the compatibility default. Do not change scheduled jobs to
encrypted exports until their recovery procedure has been validated separately.

## Credential backend code and custody

`Machine/SecretBackendUtilities.ps1` and `Machine/SecretsTier.psm1` are the
canonical backend source. Imports are passive: they neither authenticate nor
register providers, read credentials, create storage or change existing ACLs.
Default bootstrap and cache storage is a protected child of the current user's
profile. Explicit storage overrides are authoritative and fail if their
namespace is unsafe; they do not silently fall back. Existing legacy cache
access is read-only and requires current-user DPAPI, metadata/schema validation
and retained file identity. Partial or invalid preferred storage prevents
rollback to legacy data.

Private publication requires qualified Windows NTFS, trusted directory ancestry
and ordinary private files. Recovery retains exact file/process custody when
cleanup cannot be confirmed. Its owner/group/DACL contract does not establish
SACL/audit preservation or atomic comparison against arbitrary writers.

`Tests/Unit/SecretBackendCustody.Tests.ps1` exercises synthetic data and real
owned child processes. It isolates environment and executable discovery itself,
restores the original environment and keeps recovery witnesses in protected
fixture storage outside the checkout. The standalone canonical gate must pass
before deployment; its result does not qualify live provider authentication.

`Machine/managed-files.json` lists these two backend code files. An installer
must reconcile that source manifest with the installed manifest and preserve
unrelated files, private configuration and active consumers. Do not replace an
existing broader installed manifest with this two-file source manifest.

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
