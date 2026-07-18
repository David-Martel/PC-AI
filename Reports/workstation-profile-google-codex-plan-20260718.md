# Workstation Profile, Google Access, and Codex Hook Remediation

Date: 2026-07-18

Host: `dtm-p1gen7`

Scope: PowerShell 7 profile loading, Google Workspace and Google Cloud identity,
Codex hooks, CargoTools, and repository synchronization.

## Validated outcome

- PowerShell 7 starts from the OneDrive-backed registered profile, which is now
  a thin shim into the canonical machine-local `.config\powershell` profile.
- The default Google identity is Personal: `davidmartel07@gmail.com` on project
  `dtm-gemini-ai`.
- The named Business profile is `damartel@umich.edu` on `dtm-gemini-ai`; it is
  not automatically activated and still requires its own interactive gcloud
  login before project access can be claimed.
- Workspace OAuth is separate from gcloud and ADC. The Workspace token uses the
  repo-managed 12-scope contract, including OpenID and email identity, and the
  full CLI flow adds Gmail settings scope only when requested.
- Fresh Personal authorization completed. Strict status returned `ok=true`, the
  account matched, no required scope was missing, the token was refreshable,
  gcloud and ADC used the Personal account, and the ADC quota project matched.
- Bounded Drive and Tasks reads passed without exposing credential material.
- Codex event wrappers now emit only supported schema fields. RTK command
  rewriting is opt-in on Windows, so PowerShell variables, scriptblocks, and
  revision expressions such as `HEAD...@{upstream}` remain intact.
- CargoTools now recognizes both `sccache` and absolute `sccache.exe` wrapper
  paths. Stale global port overrides were removed so CargoTools owns dynamic
  port selection.

## Profile architecture

The registered profile is:

`C:\Users\david\OneDrive\Documents\PowerShell\Microsoft.PowerShell_profile.ps1`

It delegates to:

`C:\Users\david\.config\powershell\Microsoft.PowerShell_profile.ps1`

The shim does not rewrite `PSModulePath`. The canonical profile uses the local
module root under `Documents\PowerShell\Modules`, pins the Personal Workspace
account and credential path, and delegates Google authorization to
`C:\codedev\google-access`.

The full interactive profile exposes:

- `Invoke-GoogleWorkspaceTool`
- `Get-GoogleAccessAuthenticationStatus`
- `Update-GoogleAccessAuthentication`
- `Set-ActiveGcpProfile`

Automation running with `pwsh -NoProfile` should invoke repository scripts
directly rather than depending on interactive aliases.

## Google identity contract

| Profile | Account | Project | State |
|---|---|---|---|
| Personal | `davidmartel07@gmail.com` | `dtm-gemini-ai` | Default, authenticated, validated |
| Business | `damartel@umich.edu` | `dtm-gemini-ai` | Named secondary profile; login pending |

The auth refresh path now verifies the granted email before replacing the
credential, refuses a mismatched account, re-probes after renewal, treats token
persistence failure as refresh failure, and requires valid credentials with no
missing scopes before the Rust MCP reports bootstrap readiness.

No user-OAuth profile sets `GOOGLE_APPLICATION_CREDENTIALS` to OAuth client
metadata. Service-account activation remains explicit and separate.

## Codex and build-tool repairs

- PreToolUse parses input once and bypasses RTK unless
  `CODEX_ENABLE_RTK_HOOK=1`.
- PreToolUse, PostToolUse, PreCompact, SessionStart, and Stop wrappers filter
  native telemetry into the schema accepted for each event and remain silent
  on ordinary success.
- Generated Codex-home configuration uses those wrappers rather than invoking
  richer native telemetry binaries directly.
- Obsolete Codex feature keys were removed from both source configuration and
  template.
- Codex-home guidance now requires bounded specialist-agent delegation for
  independent research and verification lanes.
- The active and secondary Cargo configuration files no longer force a fixed
  `SCCACHE_SERVER_PORT`; the exact Google Rust gate selected port 4200.
- The Google raw-Cargo fallback explicitly passes
  `--config build.rustc-wrapper=""`, so it is genuinely cache-independent.

Existing Codex processes retain their startup hook snapshot. Restart them to
load the corrected wrappers.

## Validation evidence

- Workspace Python: 72 passed, 2 skipped.
- Focused auth regressions after formatting: 14 passed.
- Rust clippy: passed with warnings denied on first accelerated attempt.
- Rust nextest: 60/60 passed on first accelerated attempt.
- CargoTools focused Pester: 138/138 passed.
- `validate-hooks.ps1 -RunSmoke`: all schema and round-trip cases passed.
- `validate-codex-home.ps1`: passed.
- Fresh isolated Codex process: exact `P3_HOOK_OK` response.
- PowerShell analyzer: no error or warning in the two profile entry files.
- Fresh interactive PowerShell: Personal profile, account, project, and
  Workspace credential routing passed.
- Agent-bus strict client validation: localhost configuration passed.

## Retired source

The inactive `C:\Users\david\PowerShell-Profile-Template` directory was moved,
not deleted, to:

`C:\Users\david\.machine\archive\PowerShell-Profile-Template-retired-20260718`

It had no startup references, but its manual installer could overwrite the
validated profile and its timestamp-based sync logic did not recognize the
canonical `.config` source. The archive contains a retirement note and is
recoverable.

## Remaining operator work

- Authenticate `damartel@umich.edu` in the inactive gcloud `business`
  configuration, verify project access, then restore Personal `default` as the
  active configuration.
- Rotate the previously exposed Context7 key and the six redacted Google API
  keys at their providers. No key values are recorded in this report.
- Decide whether the full canonical profile belongs in a private repository or
  should be decomposed into a sanitized public bootstrap plus private
  machine-local fragments. Do not publish the current live 96 KB profile as-is.
- Investigate remote HTTP MCP initialization warnings separately; they are not
  hook-formatting failures.
- Restart existing Codex sessions.

## Safe revalidation

```powershell
Get-GoogleAccessAuthenticationStatus -Json
Set-GcpProfile personal
gcloud auth list
gcloud config get-value project

pwsh -NoLogo -NoProfile -File C:\codedev\dtm-codex\scripts\validate-hooks.ps1 `
  -CodexHome C:\Users\david\.codex -RunSmoke
```
