# Workstation profile, Google identity, and Codex integration plan

Date: 2026-07-18  
Host: `DTM-P1GEN7`  
Scope: research and planning; no profile, browser, OAuth, IAM, or MCP credential
mutation was performed in this phase.

## Desired identity model

| Layer | Desired identity | Desired project / role |
|---|---|---|
| Chrome default profile | `Personal` / `davidmartel07@gmail.com` | Default interactive Google profile |
| Chrome business profile | `Business` / `damartel@umich.edu` | Separate cookies, sync, and Workspace session |
| Google Workspace MCP/CLI | `davidmartel07@gmail.com` | OAuth client and enabled APIs in `dtm-gemini-ai` |
| gcloud CLI default config | `davidmartel07@gmail.com` | `dtm-gemini-ai` |
| Application Default Credentials | Authorized user for personal account | quota project `dtm-gemini-ai` |
| Vertex / Gemini tooling | Personal account or approved workload identity | `dtm-gemini-ai` |

Keep these layers explicit. Chrome profile selection, Workspace OAuth, gcloud
CLI credentials, and ADC are independent stores and must be validated
independently.

## Current state and findings

### P0 - Rotate exposed credentials before normalizing configuration

- The active Codex configuration contains a Context7 API key in command
  arguments. The current MCP listing can expose that argument without
  redaction.
- Secret-bearing Google OAuth/ADC artifacts exist in Antigravity/Gemini scratch,
  old, and generated-message directories. Do not copy their contents into issue
  trackers, reports, commits, or `agent-bus` messages.
- Treat affected API keys, refresh tokens, and OAuth client material as exposed.
  Rotation/revocation must precede deletion so recovery and attribution remain
  possible.

### P0 - Workspace OAuth is not usable

- `workspace-tool.cmd drive-list --limit 1` and `tasks-list --limit 1` both fail
  with `invalid_grant`: the refresh token is expired or revoked.
- All primary Workspace wrappers converge on
  `C:\Users\david\.google\workspace_creds.json`, but launch commands and
  descriptions are inconsistent across Codex, Claude, Gemini, and shared MCP
  snippets.
- The Rust catalog currently exposes 82 tools. Several docs and config comments
  still claim 35 or 68 tools.
- The stored Workspace grant has only the older core scopes. The desired 82-tool
  surface also needs an explicit decision for identity scopes, Apps Script,
  Photos, and Cloud/Vertex access. Request only the features that will actually
  be used.
- The current canonical `auth.py` also requests only those six legacy scopes.
  Rerunning `workspace-tool.cmd auth` before fixing the scope contract would
  reproduce an incomplete grant.

### P1 - Google browser identities do not match the requested profile model

- Chrome `Default` is the last-used profile and is signed in primarily as
  `davidmartel07@gmail.com`, but its displayed name is not `Personal`.
- `damartel@umich.edu` appears as a secondary account in the personal profile,
  which weakens cookie/session separation.
- Chrome `Profile 3` is named `Work` and is primarily associated with a
  different Umich identity. It must not be silently repurposed as the requested
  `Business` profile.
- An orphan `User Data\umich.edu` directory is not registered in Chrome's
  `info_cache`. Preserve it until bookmarks, cookies, extensions, and ownership
  are inventoried.

### P1 - Shell identity and Google environment defaults drift

- The interactive profile selects `business` through
  `C:\Users\david\.auth\shared\current-profile.txt`, even though the requested
  default is personal.
- The personal GCP environment points to `davidmartel07@gmail.com` and
  `dtm-gemini-ai`.
- The business GCP environment still names an obsolete business email/project;
  it must be redesigned for `damartel@umich.edu`. Do not assume the Umich
  account should use a personal project without confirming IAM and billing
  intent.
- Current gcloud state itself is already correct: the only active account is
  `davidmartel07@gmail.com`, the active configuration is `default`, the project
  is `dtm-gemini-ai`, ADC can mint a token, and ADC's quota project is
  `dtm-gemini-ai`.
- `damartel@umich.edu` currently has both project owner and viewer bindings on
  `dtm-gemini-ai`. The viewer binding is redundant, but removal should follow an
  explicit confirmation that owner access is intentional.

### P1 - PowerShell launch works, but the layering is internally inconsistent

- PowerShell 7 loads the OneDrive current-host profile at
  `C:\Users\david\OneDrive\Documents\PowerShell\Microsoft.PowerShell_profile.ps1`.
  That file dot-sources the canonical local profile at
  `C:\Users\david\.config\powershell\Microsoft.PowerShell_profile.ps1`.
- The OneDrive file is no longer a minimal shim: it also contains environment,
  module-path, fallback, and VIGIL helper logic. This increases sync conflict and
  split-brain risk.
- The shim names `C:\Users\david\.local\share\powershell\Modules`, which does
  not currently exist. The active module path still includes the OneDrive
  Modules tree. Persistent environment values and `.machine` documentation name
  still other profile/module layouts.
- The profile-rewritten PowerShell 7 module path cannot currently resolve
  Windows modules such as `Get-ScheduledTask` and `Get-AppxPackage`, although
  Windows PowerShell 5.1 can. Module-path repair needs compatibility regression
  tests, not just a path reorder.
- One local measurement was about 2.86 seconds with the profile versus 0.81
  seconds with `-NoProfile`. A traced run attributed most additional time to
  `ProfileUtilities`; another run was slower still. These are triage signals,
  not yet a statistically controlled benchmark.

### P2 - `pwsh.exe` resolution is mode-dependent and should be explicit

- Windows Terminal's default GUID selects the profile named `PowerShell`, whose
  command line is the unqualified `pwsh.exe`.
- A full interactive trace resolved two identical unsigned user-bin Rust
  wrappers before Microsoft PowerShell, while the final noninteractive evidence
  check resolved the signed Microsoft binary first. The disagreement proves
  mode-dependent PATH behavior rather than a universal hijack.
- Unqualified nested launches remain ambiguous even though the checked
  noninteractive launch is currently safe. Pinning the Terminal command makes
  intent and recovery deterministic.
- The Store Terminal settings also contain duplicate visible PowerShell and
  Ubuntu-family profiles. These are mostly usability drift, but they complicate
  troubleshooting.

### P1 - `dtm-codex` is not a safe full-redeployment source yet

- `C:\codedev\dtm-codex\scripts\apply-codex-home.ps1` establishes the intended
  canonical mapping: repository `AGENTS.md`, `templates\config.toml.tmpl`, and
  `templates\hooks.json.tmpl` deploy into `~\.codex`.
- The live Codex config and hooks materially exceed those templates. A full
  apply today would drop current Google Workspace MCP wiring, active MCP/plugin
  settings, authenticated HTTP agent-bus configuration, and two additional
  session-start hooks.
- Tracked `config\config.toml` and `hooks\hooks.json` contain stale Linux paths
  and behave as misleading mirrors; the deployment script does not consume
  them.
- The active config and repository template contain feature keys reported as
  removed by the installed Codex CLI. Remove them only after a version-pinned
  config validation pass.
- The active `cloudflare-api` state conflicts with local guidance saying its
  prerequisites are not ready.
- The local `dtm-codex` tracking ref is behind the live default branch. Sync it
  before committing the new specialist-agent guidance.
- `~\.codex\instructions.md` is not a current instruction surface unless
  `model_instructions_file` references it. Durable global behavior belongs in
  `~\.codex\AGENTS.md`; home-directory behavior belongs in
  `C:\Users\david\AGENTS.md`.
- The 73 TOML files under `~\.codex\agents` are not evidence of 73 usable named
  roles: they are not registered through documented `[agents.<name>]` entries.
  Curate and register only the roles that materially improve local workflows.
- `~\.codex\docs\AGENT_BUS.md` still describes port 8401 as the MCP endpoint,
  while the validated active route is `http://localhost:8400/mcp` and only port
  8400 was observed listening.

### Healthy baselines

- AgentHub is running and the strict same-machine validator is green. Functional
  client configuration uses `localhost:8400`, with authenticated access where
  required. Do not redesign the healthy bus while correcting documentation and
  client-command drift.
- The live Codex hook validator resolves all configured hook executables/scripts
  and its RTK smoke passes.
- Windows Terminal's default profile resolves to PowerShell 7 and starts in the
  user profile directory; the principal concern is unqualified executable
  resolution, not the selected GUID.

## Execution plan and TODO list

### Phase 0 - Contain and rotate exposed secrets

- [ ] Inventory secret-bearing files by path, owner, ACL, modification time,
  and credential type without printing values.
- [ ] Rotate the Context7 API key; replace literal command arguments with
  `CONTEXT7_API_KEY` environment injection or the approved secret store.
- [ ] Revoke exposed Google refresh tokens and create a new Desktop OAuth client
  in `dtm-gemini-ai` if the existing client provenance cannot be proven clean.
- [ ] Remove or quarantine secret-bearing Antigravity/Gemini scratch and
  generated-message artifacts after rotation and backup verification.
- [ ] Add a secret scan to `dtm-codex` validation and the Google tooling repo,
  covering generated/scratch directories as well as tracked files.
- [ ] Validate that `codex mcp list`, process command lines, logs, and bus
  messages no longer reveal API keys or tokens.

### Phase 1 - Establish the Google profile contract

- [ ] Back up Chrome `Local State`, `Bookmarks`, extension inventory, and the
  profile-directory map while Chrome is closed.
- [ ] Rename Chrome `Default` to `Personal`; verify its primary sync identity is
  `davidmartel07@gmail.com` and it remains the last-used/default interactive
  profile.
- [ ] Create a dedicated `Business` profile for `damartel@umich.edu`, or convert
  a profile only after confirming it contains no other primary identity.
- [ ] Remove the Umich secondary session from `Personal` after the dedicated
  Business profile is validated.
- [ ] Inventory the orphan `umich.edu` directory; archive or delete it only after
  proving it has no unique user data.
- [ ] Change the shared shell profile selector from `business` to `personal`.
- [ ] Rewrite the business GCP environment for `damartel@umich.edu`; eliminate
  obsolete account/project defaults and document whether this account should
  access `dtm-gemini-ai`.

Validation:

- [ ] Chrome launches into `Personal`; Gmail/Drive identity checks show the
  personal account.
- [ ] A separately launched `Business` profile shows only the intended Umich
  primary account.
- [ ] New PowerShell sessions report Personal by default; explicit Business
  selection is reversible and does not mutate gcloud's personal default config.

### Phase 2 - Reauthorize and normalize Google Workspace tooling

- [ ] Confirm the OAuth client belongs to `dtm-gemini-ai`, the consent screen is
  configured, required APIs are enabled, and the intended test/publishing model
  supports `davidmartel07@gmail.com`.
- [ ] Consolidate the scope contract in canonical source and tests before
  reauthentication; the current six-scope `auth.py` path cannot authorize the
  full catalog.
- [ ] Define and approve the least-privilege scope set for the selected feature envelope:
  core Workspace services, identity assertion, and only the requested Apps
  Script, Photos, and Cloud/Vertex capabilities.
- [ ] Back up credential metadata, then run the canonical interactive
  `workspace-tool.cmd auth` flow from the `Personal` Chrome profile.
- [ ] Normalize Codex, Claude, Gemini, and shared MCP snippets on
  `C:\Users\david\bin\workspace-rust-mcp.cmd` plus
  `GOOGLE_WORKSPACE_RS_TOKEN_PATH`.
- [ ] Update all tool-count and capability documentation to the generated
  catalog rather than hand-maintained numbers.
- [ ] Review `damartel@umich.edu` IAM; remove the redundant viewer binding only
  after confirming owner access is intended.

Validation:

- [ ] `auth_status` proves the authorized email is
  `davidmartel07@gmail.com`; `gcloud_status` proves project
  `dtm-gemini-ai` and healthy ADC without printing tokens.
- [ ] Bounded read-only checks pass for Gmail, Drive, Calendar, Tasks, Photos,
  and the selected Cloud/Vertex path.
- [ ] MCP initialize and `tools/list` pass from each configured client.
- [ ] If write validation is authorized, use disposable create/read/delete
  objects and record their IDs and cleanup results.

### Phase 3 - Make the PowerShell profile local-first and measurable

- [ ] Reduce the OneDrive profile to a minimal, parser-safe shim that sets only
  essential bootstrap values and dot-sources the local canonical profile.
- [ ] Move VIGIL aliases, fallback behavior, identity selection, and other
  workstation logic into local modules/profile code under `.config` or the
  appropriate maintained repo.
- [ ] Select one canonical local module root, create it, inventory/migrate
  modules from OneDrive, and retain a rollback manifest.
- [ ] Remove OneDrive module lookup from the normal hot path only after module
  import parity is proven; keep an explicit opt-in fallback.
- [ ] Reconcile persistent `POWERSHELL_PROFILE_ROOT`,
  `POWERSHELL_MODULES_PATH`, `.machine` docs, and runtime diagnostics with the
  same local-first layout.
- [ ] Profile `ProfileUtilities` imports and split eager startup work from lazy
  on-first-use functions.

Validation:

- [ ] Parser and PSScriptAnalyzer checks pass for shim and local profile.
- [ ] Run multiple warm/cold startup samples with and without profiles; record
  median and high-percentile latency in `PC_AI\Reports`.
- [ ] Exercise Bitwarden/BWS lazy initialization without printing secrets.
- [ ] Verify core aliases, module discovery, history/log flush, agent-bus token
  bootstrap, and `pwsh -NoProfile` recovery.
- [ ] Verify `Get-ScheduledTask` and `Get-AppxPackage` resolve in PowerShell 7,
  without breaking CargoTools, ProfileUtilities, or ProfileAccelerator.

### Phase 4 - Pin Windows Terminal to the trusted shell

- [ ] Change the default PowerShell profile command line to the signed
  `C:\Program Files\PowerShell\7\pwsh.exe` after signature and version checks.
- [ ] Inventory the user-bin `pwsh.exe` wrappers, their provenance, and their
  consumers. Rename/remove them or make their purpose explicit; do not leave
  them as transparent PATH shadows.
- [ ] Deduplicate visible PowerShell/Ubuntu profiles and hide obsolete generated
  entries after backing up `settings.json`.
- [ ] Preserve `%USERPROFILE%` as the Terminal starting directory; ensure the
  PowerShell profile does not override it unintentionally.

Validation:

- [ ] `Get-Command pwsh -All` resolves the trusted Microsoft binary first in a
  fresh Terminal session.
- [ ] Authenticode signature, version, startup directory, profile path, and
  interactive profile level match the expected values.

### Phase 5 - Reconcile Codex home with `dtm-codex`

- [ ] Rotate the exposed Context7 key before any config diff is published.
- [ ] Fetch/sync `David-Martel/dtm-codex`, move the current AGENTS change to a
  topic branch, and preserve the mandatory signed-commit/PR workflow.
- [ ] Generate a sanitized live-vs-template diff. Classify each active setting
  as portable default, Windows override, machine secret binding, or local-only
  state.
- [ ] Replace full-file overwrite deployment with a merge/overlay or a complete
  versioned Windows template plus dry-run diff and rollback.
- [ ] Make `config\config.toml` and `hooks\hooks.json` either generated fixtures
  with explicit platform labels or remove them from the operator path.
- [ ] Remove feature keys that the installed Codex reports as removed; add a
  version-aware config validation test.
- [ ] Reconcile the Cloudflare state with the documented prerequisite policy.
- [ ] Preserve the active Google Workspace MCP, authenticated HTTP agent-bus,
  presence hook, and signing hook in the canonical source.
- [ ] Register a small curated set of specialist roles with documented
  `[agents.<name>]` plus `config_file` entries; do not bulk-register all 73
  inherited TOML files.
- [ ] Correct agent-bus client docs so the backend/MCP binary, HTTP API, and any
  server-style `agent-bus-http.exe` behavior are not described as the same CLI.
- [ ] Reconcile the stale 8401 MCP documentation with the live multiplexed
  `http://localhost:8400/mcp` route.
- [ ] Make generated Workspace catalog counts the single source for AGENTS/docs
  capability summaries.

Validation:

- [ ] `apply-codex-home.ps1 -DryRun` shows no unintended removal of live MCP,
  plugin, hook, or project-trust settings.
- [ ] `validate-codex-home.ps1`, hook validation, and
  `validate-agent-client-configs.ps1 -Strict` pass.
- [ ] Start a fresh Codex session and verify global/home `AGENTS.md` precedence,
  specialist-agent availability, MCP health, presence, and signing hooks.

### Phase 6 - PC_AI ownership and ongoing drift checks

- [ ] Add a read-only workstation profile audit under `PC_AI\Tools` with JSON
  output and no secret values.
- [ ] Cover profile chain, module roots, PATH shadowing, Terminal default
  executable, identity selector, gcloud account/project, ADC quota project,
  Workspace auth state, Codex template drift, and agent-bus strict validation.
- [ ] Add Pester tests for dry-run/no-secret behavior and fixture-based drift
  detection.
- [ ] Link the audit from `boot.TODO.md` or `machine-reliability.TODO.md` after
  deciding which backlog owns ongoing workstation configuration integrity.
- [ ] Run the audit after profile, Terminal, Google auth, or Codex deployment
  changes and after major Windows/OneDrive updates.

## Guidance changes completed in this phase

Specialist-agent delegation guidance was added consistently to:

- `C:\Users\david\AGENTS.md`
- `C:\Users\david\.codex\AGENTS.md`
- `C:\codedev\dtm-codex\AGENTS.md`

The policy requires bounded independent lanes, read-only parallel discovery by
default, explicit agent-bus ownership before shared edits, primary-agent skill
and instruction reading, and verification of load-bearing specialist claims.

## Authoritative documentation consulted

- PowerShell profiles and `-NoProfile` diagnostics:
  https://learn.microsoft.com/powershell/scripting/learn/shell/creating-profiles
- Windows Terminal profile settings and startup-directory troubleshooting:
  https://learn.microsoft.com/windows/terminal/customize-settings/profile-general
  and https://learn.microsoft.com/windows/terminal/troubleshooting
- Chrome profile separation and renaming:
  https://support.google.com/chrome/answer/2364824?hl=en
- gcloud named configurations:
  https://docs.cloud.google.com/sdk/docs/configurations
- Google Workspace authentication and OAuth credentials:
  https://developers.google.com/workspace/guides/auth-overview and
  https://developers.google.com/workspace/guides/create-credentials
- Codex guidance, subagents, configuration, and hooks:
  https://developers.openai.com/codex/guides/agents-md,
  https://learn.chatgpt.com/docs/agent-configuration/subagents,
  https://developers.openai.com/codex/config-reference, and
  https://developers.openai.com/codex/config-advanced#hooks
