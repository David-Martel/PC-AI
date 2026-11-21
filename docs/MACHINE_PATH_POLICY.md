# Machine PATH and binary-location policy

Install, test, run: `pwsh -NoProfile -File Tools/SystemScripts/Machine/Test-MachinePath.ps1` audits a host
(read only; exit 0 clean, 1 warnings, 2 errors). `Repair-MachinePath.ps1` shows the plan and, with `-Apply`, executes it.
Tests: `Invoke-Pester Tests/Unit/MachinePath.Tests.ps1`. The policy data is
[machine-path.policy.json](../Tools/SystemScripts/Machine/machine-path.policy.json); the code is
[MachinePath.psm1](../Tools/SystemScripts/Machine/MachinePath.psm1),
[Test-MachinePath.ps1](../Tools/SystemScripts/Machine/Test-MachinePath.ps1),
[Repair-MachinePath.ps1](../Tools/SystemScripts/Machine/Repair-MachinePath.ps1) and the PATH section of
[Sync-MachineEnvironment.ps1](../Tools/SystemScripts/Machine/Sync-MachineEnvironment.ps1), which now reads the same policy.
See also `~/.agents/RUST_TOOLING.md` for the cargo home.

## What the audit found (dtm-p1gen9, 2026-11)

1. **Two PATH compositions are in use.** Windows builds a new logon's PATH as machine entries then user entries. The
   owner's shells and agent sessions put user entries first. A command that exists in both a user bin directory and a
   machine directory therefore resolves differently depending on how the process started. Fourteen names did so
   (`gpg*`, `gcloud`, `gswin64c`, `cpack`, `ctest`, `usbipd`, `ykman`).
2. **The PATH is not what makes lookups slow.** Probing all 60 directories takes about 25 ms. A missing command in
   PowerShell costs 400 to 800 ms because of module auto-discovery over `PSModulePath` (9 roots, one with 90 modules). With
   `$PSModuleAutoLoadingPreference = 'None'` the same lookup takes 25 ms. `Resolve-MachineCommand` is a PATH-only
   lookup that takes 5 to 15 ms; use it for "is this tool installed" probes.
3. **Agent homes pollute the registry.** The dotnet SDK appends `<profile>\.dotnet\tools` to the user PATH on first run
   ([docs](https://learn.microsoft.com/dotnet/core/tools/global-tools): "added to the user's path when the SDK is first
   run"). In a Claude Code session that profile is the agent home, and 147 agent homes exist on that host.
   `DOTNET_ADD_GLOBAL_TOOLS_TO_PATH=false` (documented) switches it off; the policy requires it.
4. **Wrappers that use `%USERPROFILE%` break in agent homes** when the path exists only in the real profile. Thirteen
   wrappers in `~/bin` did.
5. **Within one directory the shells disagree.** `cmd.exe` runs `name.exe` before `name.cmd`; PowerShell runs `name.ps1`
   first. Ten names have both (`claude`, `cargo`, `gcloud`, `rust-analyzer`, ...). Intentional ones are listed in the policy.

## The policy

* **Real profile only.** Every tool resolves the profile from the registry (`ProfileList`), never from `USERPROFILE` or
  `HOME`, and writes absolute entries. `%USERPROFILE%` never appears in a PATH value.
* **User PATH order.** `{UserHome}\.local\bin` (self-updating installers: claude, uv, python shims), `{UserHome}\bin`
  (wrappers and Rust utilities), `CARGO_HOME\bin`, dotnet, go, agy, OpenSSH, PowerShell, `PC_AI\bin`, WinGet links, then
  everything else in its existing order. `.local\bin` stays ahead of `WindowsApps` so `python` never reaches the Store stub.
* **Cargo.** `CARGO_HOME\bin` is the canonical cargo bin directory. `~\.cargo\bin` is forbidden on PATH. The directory itself
  is left alone while Codex's MCP commands still point into it.
* **Intentional overrides** of Windows or machine commands are named in `intentionalShadows` with a reason: `bash` (Git
  bash beats the WSL launcher), the Rust `where`, `whoami`, `hostname`, `sort`, `more`, `expand`, `tree`, `cmd`, and the
  `python`, `docker`, `npm` wrappers. They only apply in user-first shells; that is accepted.
* **No pass-through wrappers.** A `.cmd` whose only job is to call the same-named program that is already on the machine
  PATH (`gpg.cmd`) adds a `cmd.exe` start (about 70 ms) and a second resolution order. It is archived.
* **Archive, never delete.** Superseded files move to `<bin>\.superseded\<name>-<sha12><ext>` after a reference search
  (live MCP configs, Codex, Gemini, VS Code, git and ssh config, hooks, `.machine`, scheduled tasks, services and running
  processes) finds nothing that names them.
* **Only a copy that would actually win is a twin.** An identical copy elsewhere on PATH justifies archiving a `~/bin` exe only
  when it is the first runnable candidate left, and never when it lives under the profile's `AppData` (WinGet Links,
  WindowsApps): the Codex sandbox users cannot read it, so the `~/bin` copy is the one they can run.
* **Per-host overrides** live in `hostOverrides` in the policy. dtm-work uses `keepOrder` and a machine-PATH addition because
  its services and sshd sessions never see the user PATH (it keeps `~/bin` and `~/.local/bin` in the machine PATH on purpose) and
  its `python.cmd` and `npm.cmd` wrappers are deliberate. A directory present in both scopes is informational, not a defect.
* **Hardlinks are left alone.** Rustup proxies and pdf-tools links are maintained by `Sync-MachineEnvironment.ps1`.

## Running it

```powershell
pwsh -NoProfile -File Tools/SystemScripts/Machine/Test-MachinePath.ps1 -Timing          # audit
pwsh -NoProfile -File Tools/SystemScripts/Machine/Repair-MachinePath.ps1                # plan
pwsh -NoProfile -File Tools/SystemScripts/Machine/Repair-MachinePath.ps1 -Apply -SkipArchive -SkipWrappers   # PATH only
pwsh -NoProfile -File Tools/SystemScripts/Machine/Repair-MachinePath.ps1 -Apply         # PATH, archive, wrappers
```

`-Apply` writes `<profile>\.machine\backups\path\<host>.{user,machine}.path.bak-<hash>` (first line `kind=...`, second line the
raw value), a before/after resolution baseline and a before/after `Get-Command -All` capture from a fresh `pwsh` and from
an agent-home environment, and prints every command whose winner changed. A command that stops resolving exits 2.

Machine PATH changes need an elevated session and `-IncludeMachine`. New sessions pick the change up; the script
broadcasts `WM_SETTINGCHANGE` for running Explorer-launched shells.

## Deferred on purpose

* The stray `~\.cargo` and `~\.rustup` homes keep being written by a context that lacks `CARGO_HOME`/`RUSTUP_HOME`. Codex's
  `shell_environment_policy.inherit = "core"` is the leading suspect. Pruning them needs the Codex MCP commands repointed first.
* Copies that differ between `~\bin` and `~\.local\bin` (`vigilclarius-*`) need an owner decision; modification time does not
  say which build is correct.
* `.ps1` wrappers that use `$HOME` are reported, not rewritten, because a textual rewrite is unsafe inside expressions.
* `PSModulePath` roots (OneDrive modules) drive the 500 ms missing-command cost and belong to the profile lane.
