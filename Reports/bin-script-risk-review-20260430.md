# `~\bin` Script Risk Review - 2026-04-30

## Scope

This review covers scripts under `C:\Users\david\bin` and
`C:\Users\david\bin\scripts\home-root-archive` that can affect startup,
filesystem notifications, cloud sync, network configuration, process load, or
UI responsiveness. It extends the OneDrive/touchpad investigation and should be
read alongside `Reports\drive-performance-sync-risk-20260430.md`.

## Summary

No current scheduled task scan found an obvious active task invoking
`C:\Users\david\bin`, `Start-RAGRedis`, `RAG-Redis`, `dnsproxy`, Acrylic DNS, or
sccache. Startup folder inventory likewise did not show these `~\bin` scripts
directly. That means most `~\bin` risks are latent/manual risks, not proven
current boot causes.

The scripts still matter because several are designed to create logon tasks,
change DNS, start Docker/Redis/WSL workloads, kill or restart process trees,
write user environment variables, modify OneDrive-backed PowerShell modules, or
perform large file-copy/archive operations. Any one of those can indirectly
make touchpad glitches worse by adding CPU, disk, network, filter-driver, or
shell-notification pressure while OneDrive is already crash-looping and stuck
with queued changes.

## Highest-Risk Groups

### RAG Redis Startup And Health Scripts

Files:

- `C:\Users\david\bin\Setup-RAGRedisAutoStart.ps1`
- `C:\Users\david\bin\Start-RAGRedisNative.ps1`
- `C:\Users\david\bin\Test-RAGRedisHealth.ps1`
- Related module path:
  `C:\Users\david\Documents\PowerShell\Modules\PC-AI\Modules\PC-AI.Virtualization\Public\Get-WSLEnvironmentHealth.ps1`

Intent:

- Start a Redis/RAG MCP stack for local retrieval tooling.
- Support both Docker Redis and native Windows Redis.
- Register an at-logon scheduled task in `Setup-RAGRedisAutoStart.ps1`.
- Auto-recover RAG Redis from PC-AI virtualization health checks when
  `-AutoRecover` is used.

Risk:

- `Setup-RAGRedisAutoStart.ps1` writes `~\bin\Start-RAGRedis.ps1` and registers
  a logon task by default. It has `-Remove`, but no `-DryRun`, no `-Apply`, no
  structured JSON result, no event-log integration, and no boot-contention
  guard.
- `Start-RAGRedisNative.ps1` can stop `rag-server`, stop Docker containers,
  and start native Redis or Docker Compose workloads. It logs to a flat file
  but has no dry-run or Windows event trail.
- `Test-RAGRedisHealth.ps1` is named like a read-only test, but `-Fix` starts
  services, copies binaries from `W:\dropbox-local`, starts MCP processes, and
  can invoke Docker Compose.
- PC-AI virtualization health with `-AutoRecover` can start WSL, Docker, and
  RAG Redis. That is useful but should remain explicit and visible.

Touchpad/OneDrive relevance:

- Not a direct HID/I2C touchpad cause.
- High indirect risk if enabled at logon because Docker, WSL, Redis, and MCP
  startup compete with OneDrive, shell overlays, Defender, VHD mounts, and
  Process Lasso during the same boot window.

Recommended mitigations:

- Convert `Setup-RAGRedisAutoStart.ps1` to report-only by default and require
  `-Apply` for task/script creation or removal.
- Add `-DryRun`, `--DryRun`, `-h`, and `--help` to all three scripts.
- Add delayed logon registration, event-log source, transcript, JSON result,
  and explicit nonzero exit codes for degraded startup.
- Add `-NoDocker`, `-NoWSL`, `-NoProcessKill`, and `-MaxStartupDelaySeconds`
  style controls.
- Ensure any auto-start task uses low priority/background behavior where
  supported and starts only after VHD mount health and Process Lasso governor
  health pass.

### DNS Proxy And Acrylic DNS Scripts

Files:

- `C:\Users\david\bin\LocalDNSProxy.ps1`
- `C:\Users\david\bin\Install-AcrylicDNS.ps1`
- `C:\Users\david\bin\dns-proxy.bat`

Intent:

- Resolve local development domains such as `mcp.local` to localhost.
- Install or manage either AdGuard `dnsproxy` or Acrylic DNS.
- Set adapter DNS servers to localhost plus public fallback resolvers.

Risk:

- `LocalDNSProxy.ps1` removes/recreates a Windows service and changes DNS on
  all active adapters.
- `Install-AcrylicDNS.ps1` can install Acrylic via winget, write Acrylic hosts,
  start its service, flush DNS, and rewrite adapter DNS server lists.
- `dns-proxy.bat` can set the `dnsproxy` service startup type to Automatic or
  Manual.
- None of these scripts currently expose dry-run/report-only behavior or a
  captured pre-change DNS snapshot.

Touchpad/OneDrive relevance:

- Not a direct touchpad cause.
- Can worsen OneDrive and cloud-provider behavior if DNS service startup,
  adapter DNS changes, or failed local proxy resolution cause network stalls at
  logon.

Recommended mitigations:

- Add `-DryRun`, `-Apply`, `-RestoreFromSnapshot`, `-OutputJson`, and
  event-log warnings.
- Snapshot adapter DNS settings before mutation.
- Restrict changes to named adapters instead of all active adapters.
- Prefer a hosts-file or app-local resolver path for development domains when
  possible, rather than globally rewriting DNS.

### OneDrive/GCP Profile Consolidation Scripts

Files:

- `C:\Users\david\bin\scripts\home-root-archive\fix-gcp-profiles.ps1`
- `C:\Users\david\bin\scripts\home-root-archive\Initialize-GcpProfile.ps1`
- `C:\Users\david\bin\scripts\home-root-archive\gcp-profile-consolidation-script.ps1`
- `C:\Users\david\bin\_WrapperHelpers.psm1`

Intent:

- Prefer local PowerShell modules over OneDrive-backed modules.
- Repair or consolidate GCP profile tooling.
- Work around OneDrive locks on module manifests.

Risk:

- Archived GCP scripts remove or copy files in
  `C:\Users\david\OneDrive\Documents\PowerShell\Modules`.
- `gcp-profile-consolidation-script.ps1` removes an entire OneDrive-backed
  `GcpUtils` module after an interactive prompt and also mutates WSL files.
- `Initialize-GcpProfile.ps1` edits the current process module path and writes
  profile state under `~\.gcp`.
- `_WrapperHelpers.psm1` contains useful OneDrive-lock detection for CargoTools,
  but its existence also confirms that loading tooling from OneDrive-backed
  module paths is a known reliability hazard.

Touchpad/OneDrive relevance:

- Directly relevant to OneDrive sync churn, locks, and file-notification load.
- Indirectly relevant to touchpad/UI latency if shell, sync, and module loads
  are contending heavily.

Recommended mitigations:

- Treat OneDrive-backed PowerShell modules as mirrors only. Canonical active
  modules should live in non-OneDrive paths.
- Convert archived repair/consolidation scripts to dry-run/report-first before
  any future use.
- Replace recursive remove/copy operations with touched-path snapshots and
  restore manifests.
- Add explicit cloud-sync preflight checks and block writes inside sync roots
  unless a named override is supplied.

### Network Drive And Remote-Access Scripts

Files:

- `C:\Users\david\bin\scripts\home-root-archive\temp-fix-network-drives.ps1`
- `C:\Users\david\bin\scripts\home-root-archive\setup-server-winrm.ps1`
- `C:\Users\david\bin\scripts\home-root-archive\sync-to-gdrive.ps1`

Intent:

- Reduce mapped-drive delays by setting `ProviderFlags=1`.
- Configure WinRM for remote administration.
- Copy remote-setup scripts to Google Drive.

Risk:

- `temp-fix-network-drives.ps1` writes all `HKCU:\Network` mapped-drive keys
  without dry-run, backup, or targeting.
- `setup-server-winrm.ps1` can enable WinRM, modify firewall rules/settings,
  and generate registry import guidance.
- `sync-to-gdrive.ps1` copies files directly into a Google Drive sync root.

Touchpad/OneDrive relevance:

- Mapped-drive and remote-access scripts are not touchpad drivers, but network
  drive reconnect behavior at logon can contribute to Explorer or shell stalls.
- Google Drive writes can add cloud-provider notification and filter load.

Recommended mitigations:

- Add report-only mode, touched-key snapshots, and target-drive filtering to
  `temp-fix-network-drives.ps1`.
- Keep WinRM setup manual/admin-only with explicit `-Apply` and firewall
  rollback output.
- Avoid cloud-sync writes during OneDrive recovery windows.

### Heavy File, Archive, Build, And Toolchain Scripts

Files:

- `C:\Users\david\bin\scripts\home-root-archive\universal-archiver.ps1`
- `C:\Users\david\bin\scripts\home-root-archive\create-backup.ps1`
- `C:\Users\david\bin\scripts\home-root-archive\rust-build-optimizer.ps1`
- `C:\Users\david\bin\scripts\home-root-archive\rust-build-fast.ps1`
- `C:\Users\david\bin\sccache-manager.ps1`
- `C:\Users\david\bin\Setup-BuildEnvironment.ps1`
- `C:\Users\david\bin\Manage-DevTools.ps1`
- `C:\Users\david\bin\Update-DevUtilities.ps1`
- `C:\Users\david\bin\Install-CoreUtils.ps1`
- `C:\Users\david\bin\Install-CoreUtils-Direct.ps1`
- `C:\Users\david\bin\Fix-Winget.ps1`
- `C:\Users\david\bin\Fix-NPM-Issues.ps1`

Intent:

- Speed development workflows through caches, wrappers, coreutils, PATH/env
  setup, and package-manager repair.
- Archive or back up working trees.

Risk:

- `sccache-manager.ps1` can force-stop `sccache*` and `sccache-dist*`.
- `Setup-BuildEnvironment.ps1` persists user PATH and build/cache environment
  variables.
- `Fix-Winget.ps1` clears winget cache and resets sources without dry-run.
- `Fix-NPM-Issues.ps1` can persist the user `PYTHON` environment variable and
  update global node tooling.
- `Install-CoreUtils-Direct.ps1` installs direct replacements for common
  utilities into user bin paths and backs up/removes existing files.
- `universal-archiver.ps1` already has `-DryRun`, but heavy compression/copying
  can still saturate disk and Defender scanning if run over sync roots.

Touchpad/OneDrive relevance:

- These are low direct risk and medium indirect risk. During active builds or
  archive jobs they can consume CPU, memory, disk, and Defender/filter-driver
  bandwidth, which can make touchpad/UI latency visible.
- The benefits are real for developer workflows, but not boot-time benefits.
  They should not be invoked automatically during logon while OneDrive is
  unstable.

Recommended mitigations:

- Do not run build/archive/toolchain installers at logon.
- Add or verify `-DryRun`, `--DryRun`, `-h`, and `--help`.
- Require explicit `-Apply` for persistent PATH/env/tool installation changes.
- Add metrics for elapsed time, bytes processed, process priority, target
  paths, cache roots, and whether any path is inside a sync root.
- Use Process Lasso to keep heavy build/archive processes background-friendly,
  while keeping OneDrive, Explorer, Lenovo input services, and Process Governor
  protected from inappropriate trimming or priority changes.

## Process Lasso Interaction

Process Lasso is helpful only if the governor is running and policy protects
foreground/UI/input responsiveness. It cannot fix OneDrive file-state
corruption, DNS misconfiguration, or scripts that start heavyweight workloads at
logon. For these scripts:

- Keep `ProcessGovernor.exe` covered by the watchdog already deployed.
- Keep OneDrive, Explorer, ShellExperienceHost, TextInputHost, Lenovo input
  services, and Process Lasso itself out of aggressive trimming rules.
- Prefer background/low-priority policy for Docker, Redis, RAG servers, 7-Zip,
  robocopy, cargo/rustc/linker, sccache, winget, npm, and archive jobs.
- Capture Process Lasso log lines during any UI glitch capture so script-driven
  process boosts/restraints can be correlated with touchpad symptoms.

## Verdict

The apparent benefits of these scripts are mostly developer convenience and
local-service recovery, not boot responsiveness. Those benefits are worthwhile
when the scripts are invoked intentionally and observed. They are not worth the
risk as silent startup automation while OneDrive is unhealthy.

The strongest current touchpad hypothesis remains indirect UI starvation from
OneDrive/cloud-filter churn, not a `~\bin` script directly handling touchpad
events. The `~\bin` scripts should therefore be treated as risk amplifiers:
make them explicit, dry-run capable, logged, recoverable, and barred from
logon-time execution unless a validation report proves they are needed.
