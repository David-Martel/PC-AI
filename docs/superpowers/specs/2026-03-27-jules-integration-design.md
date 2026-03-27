# Jules Integration Design for PC_AI

> **Date:** 2026-03-27
> **Status:** Draft
> **Author:** Claude Opus 4.6 + David Martel
> **Repo:** David-Martel/PC-AI

## Problem Statement

53 Jules sessions have completed on this repo producing code quality fixes, security patches, test expansions, and refactors. The current workflow is manual: sessions are created ad-hoc via the Jules web UI or one-off API calls, plans are reviewed manually, and patches are pulled individually. There is no systematic way to:

- Batch-dispatch targeted reviews across modules
- Have an LLM agent craft high-quality prompts based on repo analysis
- Iteratively review and refine Jules plans before approval
- Schedule recurring scans for code quality, security, and test gaps
- Track Jules session state programmatically

This design creates a full Jules integration stack: API wrapper, AGENTS.md optimization, GitHub Actions automation, jules-skills integration, batch review pipeline, and an LLM orchestration layer for intelligent dispatch and iterative plan review.

## Prior Art

- 53 completed Jules sessions (5 merged commits, 10+ patches applied)
- Jules CLI installed (`@google/jules` via npm)
- REST API previously used for `sendMessage` + `approvePlan` from Claude sessions
- AGENTS.md exists (254 lines) but lacks Jules-specific guidance
- 3 planned Jules reviews remain unchecked (PC-AI.Gpu, Sync-NvidiaDriverVersion, gpu/mod.rs)

---

## Section 1: API Wrapper — `Tools/Invoke-JulesSession.ps1`

### Purpose

Low-level PowerShell wrapper around the Jules REST API (v1alpha) and CLI, providing programmatic session lifecycle management consistent with the existing `Tools/` pattern.

### Interface — Full API Coverage

The wrapper exposes every Jules REST API endpoint and CLI command. Actions map 1:1 to the API surface.

#### Sources (Connected Repositories)

```powershell
# List all connected GitHub repos (GET /v1alpha/sources)
Invoke-JulesSession -Action ListSources [-PageSize 30] [-Filter "..."] [-Format Table|Json]

# Get details for a specific source (GET /v1alpha/sources/{sourceId})
Invoke-JulesSession -Action GetSource -SourceId "github-David-Martel-PC-AI"
```

#### Sessions (Core Lifecycle)

```powershell
# List sessions with filtering (GET /v1alpha/sessions)
Invoke-JulesSession -Action List [-State Completed|InProgress|Failed|AwaitingPlanApproval|AwaitingUserFeedback|Queued|Planning|Paused] [-PageSize 30] [-Filter "..."] [-Format Table|Json]

# Create a new session (POST /v1alpha/sessions)
Invoke-JulesSession -Action New -Prompt "..." [-Branch main] [-Source "sources/github/David-Martel/PC-AI"] [-Title "..."] [-RequirePlanApproval] [-AutomationMode None|AutoCreatePR]

# Get session details (GET /v1alpha/sessions/{sessionId})
Invoke-JulesSession -Action Status -SessionId <id> [-Format Table|Json]

# Approve a pending plan (POST /v1alpha/sessions/{sessionId}:approvePlan)
Invoke-JulesSession -Action Approve -SessionId <id>

# Send feedback/message to active session (POST /v1alpha/sessions/{sessionId}:sendMessage)
Invoke-JulesSession -Action Message -SessionId <id> -Prompt "..."

# Delete a session (DELETE /v1alpha/sessions/{sessionId})
Invoke-JulesSession -Action Delete -SessionId <id>
```

#### Activities & Artifacts

```powershell
# List all activities in a session (GET /v1alpha/sessions/{sessionId}/activities)
Invoke-JulesSession -Action ListActivities -SessionId <id> [-PageSize 30] [-Format Table|Json]

# Get a specific activity (GET /v1alpha/sessions/{sessionId}/activities/{activityId})
Invoke-JulesSession -Action GetActivity -SessionId <id> -ActivityId <activityId>

# Extract specific artifact types from a session's activities
Invoke-JulesSession -Action GetPlan -SessionId <id>          # Extract planGenerated event → plan steps
Invoke-JulesSession -Action GetPatch -SessionId <id>         # Extract changeSet artifacts → git patches (unidiffPatch, suggestedCommitMessage)
Invoke-JulesSession -Action GetBashOutput -SessionId <id>    # Extract bashOutput artifacts → command, output, exitCode
Invoke-JulesSession -Action GetMedia -SessionId <id>         # Extract media artifacts → mimeType, base64 data
```

Activity event types surfaced:
- `planGenerated` — Plan with steps (id, index, title, description)
- `planApproved` — Plan approval confirmation
- `userMessaged` / `agentMessaged` — Conversation messages
- `progressUpdated` — Title + description progress updates
- `sessionCompleted` / `sessionFailed` — Terminal states (failure includes reason)

#### CLI Operations

```powershell
# Pull patch locally via CLI (jules remote pull)
Invoke-JulesSession -Action Pull -SessionId <id>

# List repos via CLI (jules remote list --repo)
Invoke-JulesSession -Action ListRepos

# List sessions via CLI (jules remote list --session)
Invoke-JulesSession -Action ListSessionsCli

# Create session via CLI with parallel support (jules remote new)
Invoke-JulesSession -Action NewCli -Prompt "..." [-Repo "."] [-Parallel 3]

# Check CLI version (jules version)
Invoke-JulesSession -Action Version
```

#### Batch Operations (Compound)

```powershell
# Dispatch multiple sessions via API (iterates POST /v1alpha/sessions)
Invoke-JulesSession -Action Batch -Prompts @("prompt1","prompt2") [-RequirePlanApproval] [-AutomationMode AutoCreatePR]

# Dispatch multiple sessions via CLI with parallelism
Invoke-JulesSession -Action BatchCli -Prompts @("prompt1","prompt2") [-Parallel 3]
```

### API-to-Action Reference

| Jules API Endpoint | HTTP | Action |
|---|---|---|
| `/v1alpha/sources` | GET | `ListSources` |
| `/v1alpha/sources/{id}` | GET | `GetSource` |
| `/v1alpha/sessions` | GET | `List` |
| `/v1alpha/sessions` | POST | `New` |
| `/v1alpha/sessions/{id}` | GET | `Status` |
| `/v1alpha/sessions/{id}` | DELETE | `Delete` |
| `/v1alpha/sessions/{id}:approvePlan` | POST | `Approve` |
| `/v1alpha/sessions/{id}:sendMessage` | POST | `Message` |
| `/v1alpha/sessions/{id}/activities` | GET | `ListActivities` |
| `/v1alpha/sessions/{id}/activities/{aid}` | GET | `GetActivity` |
| (compound — filters activities) | — | `GetPlan`, `GetPatch`, `GetBashOutput`, `GetMedia` |

| Jules CLI Command | Action |
|---|---|
| `jules version` | `Version` |
| `jules remote list --repo` | `ListRepos` |
| `jules remote list --session` | `ListSessionsCli` |
| `jules remote new` | `NewCli` |
| `jules remote pull` | `Pull` |

### Authentication

- **API operations**: `$env:JULES_API_KEY` environment variable (required). Reads from `.env` file in repo root as fallback (consistent with financial repo pattern). Header: `X-Goog-Api-Key`.
- **CLI operations** (`Pull`, `NewCli`, `ListRepos`, `ListSessionsCli`, `Version`): Delegate to `jules` CLI binary (requires prior `jules login` for OAuth).
- Script validates key/CLI presence before calls, exits with actionable error message.

### API Details

- Base URL: `https://jules.googleapis.com/v1alpha/`
- Default source: `sources/github/David-Martel/PC-AI` (configurable via `-Source` parameter)
- All API calls use `Invoke-RestMethod` with proper error handling and retry on 429
- Pagination: `-PageSize` (1-100, default 30) and internal `pageToken` following for `List*` actions. `-All` switch to auto-paginate and return all results.
- Filtering: `-Filter` parameter passes AIP-160 filter expressions directly to the API

### Output

- `-Format Json` (default for programmatic use): Raw API response objects
- `-Format Table`: Formatted table (columns vary by action — e.g., List: SessionId, Title, State, Created, PR URL; ListSources: SourceId, Repo, Connected)
- Batch/dispatch results written to `.pcai/jules/sessions/<timestamp>.json`
- `GetPatch` outputs `.patch` files to `.pcai/jules/patches/` for direct `git apply`

### CMD Wrapper

`Tools/jules_api.cmd`:
```cmd
@echo off
pwsh -NoLogo -NoProfile -File "%~dp0Invoke-JulesSession.ps1" %*
```

### Error Handling

| HTTP Status | Message |
|---|---|
| 401 | "JULES_API_KEY invalid or expired. Regenerate at jules.google.com/settings" |
| 403 | "Repository not connected to Jules. Connect at jules.google.com" |
| 404 | "Session or activity not found. Verify the ID." |
| 429 | Exponential backoff retry (3 attempts, 2s/4s/8s delays) |
| 500 | "Jules server error. Retry later or check jules.google.com/status" |
| Network failure | "Network error reaching Jules API. Check connectivity." |
| Missing CLI | "jules CLI not found. Install: npm install -g @google/jules" |
| Missing API key | "JULES_API_KEY not set. Generate at jules.google.com/settings and set via: $env:JULES_API_KEY = '...'" |

---

## Section 2: AGENTS.md Enhancement

### Purpose

Optimize the existing AGENTS.md so Jules (running in an Ubuntu VM) has the best possible context for this Windows-first, multi-language repo.

### Changes

Add a new section after "Documentation and automation" (line 253):

```markdown
## Jules agent guidance

Jules runs in a short-lived Ubuntu VM. Adapt accordingly:

### Environment constraints
- Use `pwsh` (not `powershell`) for PowerShell commands
- Windows-only CIM/WMI cmdlets (`Get-CimInstance`, `Get-PnpDevice`) will fail — skip integration tests that require them
- Rust builds work natively; C# builds require `dotnet` (pre-installed)
- CUDA is not available in the VM — use `--no-default-features` for Rust crates that default to CUDA
- The `bin/pcai_inference.dll` FFI DLL is Windows-only — skip FFI integration tests

### What Jules should focus on
- Rust code quality: clippy compliance, Microsoft Pragmatic Rust Guidelines, error handling
- Unit test coverage: Rust `#[test]` and PowerShell Pester tests that don't require hardware
- Code review: dead code, unnecessary clones, unsafe blocks, missing docs
- Security: hardcoded values, Invoke-Expression usage, credential exposure
- Performance: unnecessary allocations in hot paths, benchmark suggestions

### What Jules should avoid
- Modifying CUDA/GPU-specific code without understanding SM 89/120 constraints
- Running full integration test suites (require Windows + GPU hardware)
- Changing FFI function signatures (breaks C# P/Invoke + PowerShell bindings)
- Refactoring the media pipeline tensor operations without benchmark data

### Plan approval expectations
All Jules sessions on this repo use `requirePlanApproval: true`. Plans are reviewed by an LLM orchestrator (Claude/Codex) against:
1. Does the plan modify files consistent with its stated goal?
2. Does it add or update tests?
3. Does it follow the benchmark-first principle (AGENTS.md §1)?
4. Are file modifications scoped (no unnecessary drive-by refactors)?

### Code conventions
- Rust: edition 2021, `cargo fmt` + `cargo clippy -- -D warnings`, see `[lints]` in Cargo.toml
- PowerShell: `PSScriptAnalyzerSettings.psd1` in repo root
- C#: .NET 8, nullable reference types enabled
- Commit messages: Conventional Commits (`feat:`, `fix:`, `refactor:`, `test:`, `chore:`)
- PR titles: under 70 characters, imperative mood

### File ownership (parallel sessions)
When multiple Jules sessions run in parallel, each session owns specific files. Do not modify files outside your assigned scope. The orchestrator manages the File Ownership Matrix.
```

### Size Impact

Adds ~50 lines. Total AGENTS.md: ~304 lines (above Jules' 150-line recommendation but justified by repo complexity — 7 Rust crates, 14 PowerShell modules, 3 C# projects, 2 build systems).

---

## Section 3: GitHub Actions — `.github/workflows/jules-review.yml`

### Purpose

Automated Jules dispatch via GitHub Actions for scheduled reviews, issue-triggered tasks, and CI failure remediation.

### Workflow Definition

```yaml
name: Jules Review
on:
  schedule:
    - cron: '0 6 * * 1'  # Weekly Monday 6am UTC
  issues:
    types: [labeled]
  workflow_dispatch:
    inputs:
      prompt:
        description: 'Jules task prompt'
        required: true
        type: string
      module:
        description: 'Target module (e.g., pcai_inference, PC-AI.Gpu)'
        required: false
        type: string

jobs:
  jules-issue:
    if: github.event_name == 'issues' && github.event.label.name == 'jules'
    runs-on: ubuntu-latest
    steps:
      - uses: google-labs-code/jules-invoke@v1
        with:
          prompt: |
            Fix the issue described in #${{ github.event.issue.number }}.
            Title: ${{ github.event.issue.title }}
            Body: ${{ github.event.issue.body }}
            Follow AGENTS.md conventions. Add tests. Use Conventional Commits.
          jules_api_key: ${{ secrets.JULES_API_KEY }}
          starting_branch: main

  jules-manual:
    if: github.event_name == 'workflow_dispatch'
    runs-on: ubuntu-latest
    steps:
      - uses: google-labs-code/jules-invoke@v1
        with:
          prompt: ${{ inputs.prompt }}
          jules_api_key: ${{ secrets.JULES_API_KEY }}
          starting_branch: main

  jules-weekly:
    if: github.event_name == 'schedule'
    runs-on: ubuntu-latest
    strategy:
      matrix:
        target:
          - name: pcai_inference
            prompt: "Review Native/pcai_core/pcai_inference/src/ for code quality, test gaps, and clippy compliance. Focus on error handling and unsafe blocks."
          - name: pcai_media
            prompt: "Review Native/pcai_core/pcai_media/src/ for unnecessary allocations, missing error context, and test coverage gaps."
          - name: pcai_core_lib
            prompt: "Review Native/pcai_core/pcai_core_lib/src/ for dead code, missing docs on public items, and potential panics."
      max-parallel: 3
    steps:
      - uses: google-labs-code/jules-invoke@v1
        with:
          prompt: ${{ matrix.target.prompt }}
          jules_api_key: ${{ secrets.JULES_API_KEY }}
          starting_branch: main
```

### Requirements

- `JULES_API_KEY` stored as GitHub Actions secret
- Jules GitHub App installed on David-Martel/PC-AI (already done)
- `jules` label created on the repo for issue-triggered dispatch

### Concurrency

Weekly scan dispatches up to 3 parallel sessions (within free tier limit of 5 concurrent). Each targets a different Rust crate to avoid file ownership conflicts.

---

## Section 4: Jules Skills Installation

### Purpose

Install the `automate-github-issues` skill from `google-labs-code/jules-skills` for structured issue triage and parallel dispatch.

### Installation

```bash
npx skills add google-labs-code/jules-skills --skill automate-github-issues --global
```

This installs:
- `scripts/` — TypeScript orchestration (Bun runtime, `@google/jules-sdk`, Octokit)
- `assets/` — GitHub Actions workflow templates (`fleet-dispatch.yml`, `fleet-merge.yml`)
- `resources/` — Architecture docs and troubleshooting guides

### Adaptation for PC_AI

The skill's analysis prompt needs customization for this repo:
- Add Rust-specific review criteria (Microsoft Pragmatic Guidelines, clippy lints)
- Add PowerShell-specific review criteria (PSScriptAnalyzer, Pester patterns)
- Add benchmark-first requirement to acceptance criteria
- Configure File Ownership Matrix to respect the Rust workspace crate boundaries

Customization goes in `.jules/skills/pcai-review/SKILL.md` (a repo-local custom skill directory, gitignored by default by the `skills` CLI — add to `.gitignore` if needed).

### Custom Skill: `pcai-review`

```
.jules/skills/pcai-review/
  SKILL.md          # Frontmatter + agent instructions for PC_AI code review
  resources/
    rust-guidelines.md    # Subset of Microsoft Pragmatic Rust Guidelines relevant to review
    module-map.md         # Module descriptions and ownership boundaries
  scripts/
    analyze-modules.ts    # Generate per-module review prompts with file-level context
```

The `SKILL.md` instructs Jules to:
1. Run `cargo clippy --all-targets -- -D warnings` and capture violations
2. Run `pwsh -Command "Invoke-ScriptAnalyzer -Path Modules/ -Recurse"` for PowerShell
3. Check test coverage gaps by comparing `*.rs` files to `*_test.rs` / `tests/*.rs`
4. Generate targeted fix prompts with file:line references
5. Respect the File Ownership Matrix for parallel dispatch

---

## Section 5: Batch Review Pipeline

### Purpose

Structured scripts to dispatch, monitor, and triage Jules reviews across all major modules.

### Components

#### `Tools/Invoke-JulesBatchReview.ps1`

Dispatches targeted Jules sessions for each major module:

```powershell
# Review all Rust crates
Invoke-JulesBatchReview -Scope RustCrates [-MaxSessions 5] [-RequirePlanApproval]

# Review all PowerShell modules
Invoke-JulesBatchReview -Scope PowerShellModules [-MaxSessions 5]

# Review specific modules
Invoke-JulesBatchReview -Modules pcai_inference,pcai_media,PC-AI.Gpu

# Review based on git diff (changed files since last release)
Invoke-JulesBatchReview -Scope ChangedSinceTag -Tag v0.2.0
```

**Module prompt templates** are stored in `Config/jules-review-prompts.json`:
```json
{
  "pcai_inference": {
    "prompt": "Review {files} for: error handling (no unwrap in non-test code), unsafe block documentation, HTTP endpoint test coverage, FFI boundary safety. Run cargo test --lib.",
    "files": "Native/pcai_core/pcai_inference/src/",
    "priority": "high"
  },
  "pcai_media": {
    "prompt": "Review {files} for: tensor allocation efficiency, VQ decode correctness, GGUF quantization path coverage. Check for unnecessary .clone() in hot loops.",
    "files": "Native/pcai_core/pcai_media/src/",
    "priority": "high"
  }
}
```

#### `Tools/Get-JulesPRStatus.ps1`

Triage dashboard for Jules-created PRs:

```powershell
# List all Jules PRs with status
Get-JulesPRStatus [-State open|closed|all] [-Format Table|Json]

# Output columns: PR#, Title, State, Mergeable, CI Status, Files Changed, Conflicts
```

Uses `gh pr list --author "jules[bot]"` and `gh pr view` for enrichment.

### Priority Order for Initial Batch

Based on repo state and planned work:

| Priority | Module | Rationale |
|----------|--------|-----------|
| 1 | `pcai_inference` | Core inference engine, highest complexity |
| 2 | `pcai_media` | Active development (GGUF, VQ decode issues) |
| 3 | `PC-AI.Gpu` | New module, 3 planned reviews already queued |
| 4 | `pcai_core_lib` | Shared library, wide blast radius |
| 5 | `PcaiNative` (C#) | 20 modules, P/Invoke boundary safety |
| 6 | `PC-AI.Hardware` | Mature but untested edge cases |
| 7 | `Deploy/rust-functiongemma-*` | Router and training pipeline |

---

## Section 6: LLM Orchestrator — `Tools/Invoke-JulesOrchestrator.ps1`

### Purpose

Intelligence layer that enables LLM agents (Claude, Codex, Gemini) to craft high-quality Jules prompts based on repo analysis and iteratively review/refine Jules plans before approval.

### Capability 1: Smart Dispatch

```powershell
# Analyze repo and generate optimal Jules prompts
Invoke-JulesOrchestrator -Action AnalyzeAndDispatch [-MaxSessions 5] [-DryRun]
```

**How it works:**

1. **Gather signals** — The script collects:
   - `cargo clippy --all-targets --message-format=json` output (lint violations)
   - `pwsh -Command "Invoke-ScriptAnalyzer -Path Modules/ -Recurse -Severity Warning,Error"` output
   - `git diff --stat HEAD~20` (recently changed files = higher review priority)
   - Test coverage gaps (`.rs` files without corresponding test files)
   - TODO/FIXME markers (`grep -rn "TODO\|FIXME\|HACK" Native/ Modules/`)
   - Open issues labeled `bug` or `enhancement`

2. **Generate analysis report** — Structured JSON output:
   ```json
   {
     "timestamp": "2026-03-27T...",
     "signals": {
       "clippy_violations": 12,
       "psscriptanalyzer_warnings": 5,
       "recently_changed_files": ["..."],
       "untested_modules": ["..."],
       "todo_markers": 34,
       "open_issues": 3
     },
     "recommended_sessions": [
       {
         "module": "pcai_inference",
         "priority": "high",
         "prompt": "...(generated with file-level context)...",
         "files_in_scope": ["src/http/mod.rs", "src/ffi/mod.rs"],
         "acceptance_criteria": ["All clippy warnings resolved", "New tests for uncovered paths"],
         "estimated_complexity": "medium"
       }
     ]
   }
   ```

3. **Dispatch** — Unless `-DryRun` is specified, creates Jules sessions with `requirePlanApproval: true` for each recommended session. Writes session IDs to `.pcai/jules/dispatch-<timestamp>.json`.

**Prompt generation principles:**
- Include specific file paths and line ranges, not vague module names
- Include the acceptance criteria Jules must meet
- Include what NOT to change (file ownership boundaries)
- Reference AGENTS.md conventions by section
- Keep each prompt under 2000 characters (Jules processes shorter prompts more reliably)

### Capability 2: Iterative Plan Review

```powershell
# Review all pending Jules plans
Invoke-JulesOrchestrator -Action ReviewPlans [-AutoApprove Low] [-Format Report]

# Review a specific session's plan
Invoke-JulesOrchestrator -Action ReviewPlan -SessionId <id>
```

**How it works:**

1. **Fetch pending plans** — Polls all sessions in `AWAITING_PLAN_APPROVAL` state via API

2. **Extract plan details** — For each session, fetches activities to get the `planGenerated` event containing plan steps

3. **Generate review report** — Structured assessment for each plan:
   ```json
   {
     "session_id": "...",
     "title": "...",
     "plan_steps": [...],
     "assessment": {
       "scope_check": "pass|warn|fail",
       "scope_notes": "Plan modifies 3 files, all within stated scope",
       "test_coverage": "pass|warn|fail",
       "test_notes": "Step 4 adds 2 new tests for the refactored function",
       "conventions": "pass|warn|fail",
       "conventions_notes": "Uses unwrap() in step 2 — should use expect() with context",
       "benchmark_impact": "none|low|high",
       "recommendation": "approve|feedback|reject",
       "feedback_message": "Step 2: replace .unwrap() with .expect(\"descriptive message\") per M-APP-ERROR guideline"
     }
   }
   ```

4. **Take action** — Based on assessment:
   - `approve`: Calls `Invoke-JulesSession -Action Approve`
   - `feedback`: Calls `Invoke-JulesSession -Action Message` with the feedback, then re-polls after Jules revises
   - `reject`: Logs rejection reason, does not approve (session times out naturally)

5. **Iteration loop** — After sending feedback, the orchestrator waits for Jules to revise the plan (polls every 30s, max 10 minutes), then re-evaluates. Maximum 3 feedback rounds per session to prevent infinite loops.

**Review criteria (encoded in script):**

| Criterion | Pass | Warn | Fail |
|-----------|------|------|------|
| Scope | All files in declared module | 1-2 files outside scope | >2 files outside scope or modifying FFI signatures |
| Tests | Adds/updates tests | No test changes but low-risk refactor | Removes tests or changes behavior without tests |
| Conventions | Follows Rust guidelines + PSAnalyzer | Minor style deviations | Uses unwrap in non-test code, adds unsafe without docs |
| Benchmark | N/A or adds benchmark | Touches hot path without benchmark | Removes or weakens existing benchmarks |

### Output Artifacts

All orchestrator runs write to `.pcai/jules/`:
```
.pcai/jules/
  dispatch-<timestamp>.json     # Session IDs from AnalyzeAndDispatch
  review-<timestamp>.json       # Plan review assessments
  analysis-<timestamp>.json     # Repo analysis signals
```

### Agent Bus Integration

When running in multi-agent mode, the orchestrator:
- Posts to agent bus topic `jules-orchestration` with session status updates
- Checks bus for ownership claims before dispatching to avoid conflicts with other agents
- Uses agent ID `jules-orchestrator` for presence

```powershell
# Example bus message on dispatch
agent-bus send --from-agent jules-orchestrator --to-agent all --topic jules-orchestration --body "Dispatched 3 sessions: pcai_inference (sid-abc), pcai_media (sid-def), PC-AI.Gpu (sid-ghi). Plan approval pending."
```

---

## File Inventory

| File | Type | Purpose |
|------|------|---------|
| `Tools/Invoke-JulesSession.ps1` | New | Low-level API wrapper |
| `Tools/jules_api.cmd` | New | CMD shim for quick CLI access |
| `Tools/Invoke-JulesBatchReview.ps1` | New | Module-by-module batch dispatch |
| `Tools/Get-JulesPRStatus.ps1` | New | Jules PR triage dashboard |
| `Tools/Invoke-JulesOrchestrator.ps1` | New | LLM orchestration: smart dispatch + plan review |
| `Config/jules-review-prompts.json` | New | Per-module review prompt templates |
| `.github/workflows/jules-review.yml` | New | GitHub Actions: scheduled + issue-triggered |
| `AGENTS.md` | Modified | Add Jules agent guidance section (~50 lines) |
| `.jules/skills/pcai-review/SKILL.md` | New | Custom Jules skill for PC_AI review |
| `.jules/skills/pcai-review/resources/module-map.md` | New | Module descriptions for skill context |
| `.jules/skills/pcai-review/scripts/analyze-modules.ts` | New | Module analysis script |

## Dependencies

| Dependency | Status | Action |
|------------|--------|--------|
| `@google/jules` CLI | Installed | None |
| `JULES_API_KEY` env var | Not set | Generate at jules.google.com/settings, add to `.env` and GitHub secrets |
| Jules GitHub App | Installed on repo | None |
| `@google/jules-sdk` (for skills) | Not installed | `npm install @google/jules-sdk` (only needed for skills scripts) |
| Bun runtime (for skills) | Not installed | `npm install -g bun` (only needed for skills scripts) |
| `jules` label on GitHub | Not created | `gh label create jules --description "Trigger Jules AI agent" --color 4285F4` |

## Out of Scope

- Execution monitoring (polling Jules during task execution)
- PR quality gate automation (auto-merge/reject based on diff analysis)
- Follow-up dispatch (auto-creating sessions based on completed session results)
- Jules MCP server connections (Linear, Supabase, etc.)
- Custom Jules environment snapshots or VM configuration
- Modifying the jules-skills `automate-github-issues` core logic

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| API rate limiting (v1alpha) | Sessions fail to create | Exponential backoff, batch limits, respect daily quota |
| Jules plan quality on complex Rust code | Low-value PRs | `requirePlanApproval: true` on all sessions, LLM review gate |
| File ownership conflicts in parallel sessions | Merge conflicts | File Ownership Matrix, crate-level scoping |
| Ubuntu VM can't run Windows-specific tests | False test failures | AGENTS.md guidance to skip CIM/FFI tests |
| API v1alpha breaking changes | Wrapper stops working | Pin to known-good endpoints, version check on startup |
| Jules daily task limit (15 free, 100 pro) | Can't dispatch full batch | Priority ordering, staged dispatch over multiple days |
