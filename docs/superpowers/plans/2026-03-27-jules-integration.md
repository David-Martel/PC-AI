# Jules Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a full Jules AI agent integration stack for PC_AI: API wrapper, AGENTS.md guidance, GitHub Actions automation, batch review pipeline, and LLM orchestration for smart dispatch and iterative plan review.

**Architecture:** PowerShell-first tooling (consistent with existing `Tools/`) wrapping the Jules REST API v1alpha and CLI. A config-driven prompt template system feeds the orchestrator. GitHub Actions provide scheduled and event-triggered automation. All Jules session state persists to `.pcai/jules/`.

**Tech Stack:** PowerShell 7+, Jules REST API v1alpha (`Invoke-RestMethod`), Jules CLI (`@google/jules`), GitHub Actions (`google-labs-code/jules-invoke@v1`), `gh` CLI for PR triage.

**Spec:** `docs/superpowers/specs/2026-03-27-jules-integration-design.md`

---

## File Structure

| File | Responsibility |
|------|---------------|
| `Tools/Invoke-JulesSession.ps1` | Low-level API/CLI wrapper (all 10 REST endpoints + 5 CLI commands) |
| `Tools/jules_api.cmd` | CMD shim delegating to Invoke-JulesSession.ps1 |
| `Tools/Invoke-JulesBatchReview.ps1` | Batch dispatch across modules using prompt templates |
| `Tools/Get-JulesPRStatus.ps1` | Jules PR triage dashboard via `gh` CLI |
| `Tools/Invoke-JulesOrchestrator.ps1` | LLM orchestration: smart dispatch + iterative plan review |
| `Config/jules-review-prompts.json` | Per-module review prompt templates |
| `.github/workflows/jules-review.yml` | Scheduled + issue-triggered + manual Jules dispatch |
| `AGENTS.md` | Modified: add Jules agent guidance section |
| `Tests/Unit/Invoke-JulesSession.Tests.ps1` | Pester tests for API wrapper |
| `Tests/Unit/Invoke-JulesOrchestrator.Tests.ps1` | Pester tests for orchestrator |

---

## Task 1: Jules API Wrapper Core — `Invoke-JulesSession.ps1`

**Files:**
- Create: `Tools/Invoke-JulesSession.ps1`
- Create: `Tools/jules_api.cmd`
- Test: `Tests/Unit/Invoke-JulesSession.Tests.ps1`

This is the foundation. All other tools depend on it.

- [ ] **Step 1: Write failing tests for API helper functions**

Create `Tests/Unit/Invoke-JulesSession.Tests.ps1`:

```powershell
#Requires -Modules @{ ModuleName = 'Pester'; ModuleVersion = '5.0.0' }

BeforeAll {
    $script:ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
    $script:ScriptPath  = Join-Path $script:ProjectRoot 'Tools' 'Invoke-JulesSession.ps1'
}

Describe 'Invoke-JulesSession' {

    Context 'Parameter validation' {
        It 'Throws when Action is missing' {
            { & $script:ScriptPath } | Should -Throw
        }
        It 'Throws when New action has no Prompt' {
            { & $script:ScriptPath -Action New } | Should -Throw '*Prompt*'
        }
        It 'Throws when Status action has no SessionId' {
            { & $script:ScriptPath -Action Status } | Should -Throw '*SessionId*'
        }
        It 'Throws when Approve action has no SessionId' {
            { & $script:ScriptPath -Action Approve } | Should -Throw '*SessionId*'
        }
        It 'Throws when Message action has no SessionId' {
            { & $script:ScriptPath -Action Message -Prompt 'test' } | Should -Throw '*SessionId*'
        }
        It 'Throws when Message action has no Prompt' {
            { & $script:ScriptPath -Action Message -SessionId 'abc' } | Should -Throw '*Prompt*'
        }
        It 'Throws when Delete action has no SessionId' {
            { & $script:ScriptPath -Action Delete } | Should -Throw '*SessionId*'
        }
        It 'Throws when Batch action has no Prompts' {
            { & $script:ScriptPath -Action Batch } | Should -Throw '*Prompts*'
        }
    }

    Context 'API key validation' {
        BeforeEach {
            $script:OrigKey = $env:JULES_API_KEY
            $env:JULES_API_KEY = $null
        }
        AfterEach {
            $env:JULES_API_KEY = $script:OrigKey
        }

        It 'Throws descriptive error when JULES_API_KEY is not set for API actions' {
            { & $script:ScriptPath -Action List } | Should -Throw '*JULES_API_KEY*'
        }
        It 'Does not require API key for CLI-only actions' {
            # Version action uses CLI, not API — should not throw about API key
            # (will fail if CLI not installed, but not about the key)
            { & $script:ScriptPath -Action Version } | Should -Not -Throw '*JULES_API_KEY*'
        }
    }

    Context 'URL construction' {
        BeforeAll {
            # Dot-source to access internal functions
            . $script:ScriptPath -Action '__test_load__' 2>$null
        }

        It 'Builds correct sessions list URL' {
            $url = Get-JulesApiUrl -Endpoint 'sessions'
            $url | Should -Be 'https://jules.googleapis.com/v1alpha/sessions'
        }
        It 'Builds correct session detail URL' {
            $url = Get-JulesApiUrl -Endpoint 'sessions' -Id 'sess-123'
            $url | Should -Be 'https://jules.googleapis.com/v1alpha/sessions/sess-123'
        }
        It 'Builds correct activities URL' {
            $url = Get-JulesApiUrl -Endpoint 'sessions' -Id 'sess-123' -Sub 'activities'
            $url | Should -Be 'https://jules.googleapis.com/v1alpha/sessions/sess-123/activities'
        }
        It 'Builds correct sources URL' {
            $url = Get-JulesApiUrl -Endpoint 'sources'
            $url | Should -Be 'https://jules.googleapis.com/v1alpha/sources'
        }
        It 'Builds correct approvePlan URL' {
            $url = Get-JulesApiUrl -Endpoint 'sessions' -Id 'sess-123' -Action 'approvePlan'
            $url | Should -Be 'https://jules.googleapis.com/v1alpha/sessions/sess-123:approvePlan'
        }
        It 'Builds correct sendMessage URL' {
            $url = Get-JulesApiUrl -Endpoint 'sessions' -Id 'sess-123' -Action 'sendMessage'
            $url | Should -Be 'https://jules.googleapis.com/v1alpha/sessions/sess-123:sendMessage'
        }
    }

    Context 'Request body construction' {
        BeforeAll {
            . $script:ScriptPath -Action '__test_load__' 2>$null
        }

        It 'Builds correct New session body with defaults' {
            $body = New-JulesSessionBody -Prompt 'Fix bug' -Source 'sources/github/David-Martel/PC-AI' -Branch 'main'
            $body.prompt | Should -Be 'Fix bug'
            $body.sourceContext.source | Should -Be 'sources/github/David-Martel/PC-AI'
            $body.sourceContext.githubRepoContext.startingBranch | Should -Be 'main'
            $body | ConvertTo-Json -Depth 5 | Should -Not -BeNullOrEmpty
        }
        It 'Includes requirePlanApproval when specified' {
            $body = New-JulesSessionBody -Prompt 'Fix bug' -Source 'sources/github/David-Martel/PC-AI' -Branch 'main' -RequirePlanApproval
            $body.requirePlanApproval | Should -BeTrue
        }
        It 'Includes automationMode when AutoCreatePR' {
            $body = New-JulesSessionBody -Prompt 'Fix bug' -Source 'sources/github/David-Martel/PC-AI' -Branch 'main' -AutomationMode 'AutoCreatePR'
            $body.automationMode | Should -Be 'AUTO_CREATE_PR'
        }
        It 'Includes title when specified' {
            $body = New-JulesSessionBody -Prompt 'Fix bug' -Source 'sources/github/David-Martel/PC-AI' -Branch 'main' -Title 'Bug fix session'
            $body.title | Should -Be 'Bug fix session'
        }
    }

    Context 'Output formatting' {
        BeforeAll {
            . $script:ScriptPath -Action '__test_load__' 2>$null
        }

        It 'Formats session list as table' {
            $sessions = @(
                @{ name = 'sessions/abc'; title = 'Test'; state = 'COMPLETED'; createTime = '2026-03-27T00:00:00Z'; outputs = @(@{ pullRequest = @{ url = 'https://github.com/pr/1' } }) }
            )
            $result = Format-JulesSessionTable -Sessions $sessions
            $result | Should -Not -BeNullOrEmpty
            $result[0].SessionId | Should -Be 'abc'
            $result[0].State | Should -Be 'COMPLETED'
            $result[0].PRUrl | Should -Be 'https://github.com/pr/1'
        }
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pwsh -Command "Invoke-Pester Tests/Unit/Invoke-JulesSession.Tests.ps1 -Output Detailed"`
Expected: All tests FAIL (script does not exist yet)

- [ ] **Step 3: Implement `Invoke-JulesSession.ps1`**

Create `Tools/Invoke-JulesSession.ps1` with:

1. **Script header** — `#Requires -Version 5.1`, synopsis/description/examples in comment-based help (follow `Get-BuildVersion.ps1` pattern)
2. **Parameters** — `[CmdletBinding()] param(...)` with:
   - `[Parameter(Mandatory)][ValidateSet('List','New','Status','Approve','Message','Delete','ListActivities','GetActivity','GetPlan','GetPatch','GetBashOutput','GetMedia','ListSources','GetSource','Pull','ListRepos','ListSessionsCli','NewCli','BatchCli','Batch','Version')][string]$Action`
   - `[string]$SessionId`, `[string]$ActivityId`, `[string]$SourceId`
   - `[string]$Prompt`, `[string[]]$Prompts`
   - `[string]$Branch = 'main'`, `[string]$Source = 'sources/github/David-Martel/PC-AI'`
   - `[string]$Title`, `[switch]$RequirePlanApproval`
   - `[ValidateSet('None','AutoCreatePR')][string]$AutomationMode = 'None'`
   - `[ValidateSet('Table','Json')][string]$Format = 'Json'`
   - `[ValidateSet('Completed','InProgress','Failed','AwaitingPlanApproval','AwaitingUserFeedback','Queued','Planning','Paused')][string]$State`
   - `[int]$PageSize = 30`, `[string]$Filter`, `[switch]$All`
   - `[string]$Repo = '.'`, `[int]$Parallel = 1`

3. **Internal helper functions** (accessible for testing via `__test_load__` action):
   - `Get-JulesApiUrl` — builds endpoint URLs
   - `New-JulesSessionBody` — builds POST body for session creation
   - `Invoke-JulesApi` — wraps `Invoke-RestMethod` with auth header, error handling, 429 retry (3 attempts, exponential backoff)
   - `Invoke-JulesCli` — wraps `jules` CLI calls with error handling
   - `Format-JulesSessionTable` — converts API response to table-friendly objects
   - `Get-JulesApiKey` — reads from `$env:JULES_API_KEY`, falls back to `.env` file
   - `Assert-RequiredParam` — validates required params per action

4. **Action dispatch** — `switch ($Action)` routing each action to the correct API call:
   - `List` → `GET /sessions` with optional `$State` filter and pagination
   - `New` → `POST /sessions` with body from `New-JulesSessionBody`
   - `Status` → `GET /sessions/$SessionId`
   - `Approve` → `POST /sessions/$SessionId:approvePlan`
   - `Message` → `POST /sessions/$SessionId:sendMessage` with `@{ prompt = $Prompt }`
   - `Delete` → `DELETE /sessions/$SessionId` via `Invoke-RestMethod -Method Delete`
   - `ListActivities` → `GET /sessions/$SessionId/activities`
   - `GetActivity` → `GET /sessions/$SessionId/activities/$ActivityId`
   - `GetPlan` → `ListActivities` then filter for `planGenerated` event, extract steps
   - `GetPatch` → `ListActivities` then filter artifacts with `changeSet`, write `.patch` files to `.pcai/jules/patches/`
   - `GetBashOutput` → `ListActivities` then filter artifacts with `bashOutput`
   - `GetMedia` → `ListActivities` then filter artifacts with `media`
   - `ListSources` → `GET /sources`
   - `GetSource` → `GET /sources/$SourceId`
   - `Pull` → `Invoke-JulesCli "remote pull --session $SessionId"`
   - `ListRepos` → `Invoke-JulesCli "remote list --repo"`
   - `ListSessionsCli` → `Invoke-JulesCli "remote list --session"`
   - `NewCli` → `Invoke-JulesCli "remote new --repo $Repo --session `"$Prompt`" $(if ($Parallel -gt 1) { "--parallel $Parallel" })"`
   - `Version` → `Invoke-JulesCli "version"`
   - `Batch` → Loop over `$Prompts`, call `New` for each, collect results to `.pcai/jules/sessions/<timestamp>.json`
   - `BatchCli` → Loop over `$Prompts`, call `NewCli` for each

5. **Output** — Format switch: `Json` returns raw objects, `Table` pipes through `Format-JulesSessionTable` (or action-specific formatter)

6. **`__test_load__` action** — When `$Action -eq '__test_load__'`, define internal functions in caller's scope and return (enables Pester dot-sourcing)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pwsh -Command "Invoke-Pester Tests/Unit/Invoke-JulesSession.Tests.ps1 -Output Detailed"`
Expected: All 14 tests PASS

- [ ] **Step 5: Create CMD wrapper**

Create `Tools/jules_api.cmd`:

```cmd
@echo off
REM Jules API wrapper — delegates to PowerShell
REM Usage: jules_api.cmd -Action List -Format Table
pwsh -NoLogo -NoProfile -File "%~dp0Invoke-JulesSession.ps1" %*
```

- [ ] **Step 6: Commit**

```bash
git add Tools/Invoke-JulesSession.ps1 Tools/jules_api.cmd Tests/Unit/Invoke-JulesSession.Tests.ps1
git commit -m "feat(jules): add Invoke-JulesSession API/CLI wrapper with full endpoint coverage"
```

---

## Task 2: Review Prompt Templates — `Config/jules-review-prompts.json`

**Files:**
- Create: `Config/jules-review-prompts.json`

- [ ] **Step 1: Create the prompt template config**

Create `Config/jules-review-prompts.json`:

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "description": "Per-module Jules review prompt templates for Invoke-JulesBatchReview",
  "modules": {
    "pcai_inference": {
      "prompt": "Review Native/pcai_core/pcai_inference/src/ for: error handling (no unwrap in non-test code, use expect with context), unsafe block documentation, HTTP endpoint test coverage, FFI boundary safety. Run: cargo test -p pcai_inference --no-default-features --features server,ffi --lib. Follow AGENTS.md conventions.",
      "files": "Native/pcai_core/pcai_inference/src/",
      "priority": "high",
      "tags": ["rust", "inference", "ffi"]
    },
    "pcai_media": {
      "prompt": "Review Native/pcai_core/pcai_media/src/ for: tensor allocation efficiency (no unnecessary clone in hot loops), VQ decode correctness, GGUF quantization path coverage, missing error context on Results. Run: cargo test -p pcai_media --no-default-features --lib. Follow AGENTS.md conventions.",
      "files": "Native/pcai_core/pcai_media/src/",
      "priority": "high",
      "tags": ["rust", "media", "gpu"]
    },
    "pcai_core_lib": {
      "prompt": "Review Native/pcai_core/pcai_core_lib/src/ for: dead code, missing docs on public items, potential panics (unwrap/expect in library code), Windows API safety. Run: cargo test -p pcai_core_lib --lib. Follow AGENTS.md conventions.",
      "files": "Native/pcai_core/pcai_core_lib/src/",
      "priority": "medium",
      "tags": ["rust", "library"]
    },
    "pcai_media_model": {
      "prompt": "Review Native/pcai_core/pcai_media_model/src/ for: model config correctness, serde safety (missing defaults, strict vs lenient parsing), public API surface docs. Run: cargo test -p pcai_media_model --lib. Follow AGENTS.md conventions.",
      "files": "Native/pcai_core/pcai_media_model/src/",
      "priority": "medium",
      "tags": ["rust", "model"]
    },
    "pcai_media_server": {
      "prompt": "Review Native/pcai_core/pcai_media_server/src/ for: axum handler error propagation, request validation, missing endpoint tests. Run: cargo test -p pcai_media_server --lib. Follow AGENTS.md conventions.",
      "files": "Native/pcai_core/pcai_media_server/src/",
      "priority": "medium",
      "tags": ["rust", "server"]
    },
    "pcai_perf_cli": {
      "prompt": "Review Native/pcai_core/pcai_perf_cli/src/ for: CLI argument validation, output format correctness, error handling. Run: cargo test -p pcai_perf_cli --lib. Follow AGENTS.md conventions.",
      "files": "Native/pcai_core/pcai_perf_cli/src/",
      "priority": "low",
      "tags": ["rust", "cli"]
    },
    "pcai_ollama_rs": {
      "prompt": "Review Native/pcai_core/pcai_ollama_rs/src/ for: HTTP client error handling, benchmark result parsing, timeout handling. Run: cargo test -p pcai_ollama_rs --lib. Follow AGENTS.md conventions.",
      "files": "Native/pcai_core/pcai_ollama_rs/src/",
      "priority": "low",
      "tags": ["rust", "ollama"]
    },
    "PC-AI.Gpu": {
      "prompt": "Review Modules/PC-AI.Gpu/ for: NVIDIA detection edge cases, registry JSON validation, software installation safety (WhatIf support), error handling on missing nvidia-smi. Run: pwsh -Command \"Invoke-Pester Tests/Unit/PC-AI.Gpu.Tests.ps1\". Follow AGENTS.md conventions.",
      "files": "Modules/PC-AI.Gpu/",
      "priority": "high",
      "tags": ["powershell", "gpu", "nvidia"]
    },
    "PC-AI.Hardware": {
      "prompt": "Review Modules/PC-AI.Hardware/ for: CIM query robustness, diagnostic report accuracy, error formatting, missing device handling edge cases. Run: pwsh -Command \"Invoke-Pester Tests/Unit/PC-AI.Hardware.Tests.ps1\". Follow AGENTS.md conventions.",
      "files": "Modules/PC-AI.Hardware/",
      "priority": "medium",
      "tags": ["powershell", "diagnostics"]
    },
    "PC-AI.Drivers": {
      "prompt": "Review Modules/PC-AI.Drivers/ for: driver version comparison logic, Thunderbolt peer discovery robustness, registry update safety. Run: pwsh -Command \"Invoke-Pester Tests/Unit/PC-AI.Drivers.Tests.ps1\". Follow AGENTS.md conventions.",
      "files": "Modules/PC-AI.Drivers/",
      "priority": "medium",
      "tags": ["powershell", "drivers"]
    },
    "PcaiNative": {
      "prompt": "Review Native/PcaiNative/ for: P/Invoke signature correctness (match Rust FFI exactly), null pointer handling, SafeHandle usage, resource disposal. This is C# .NET 8 code. Follow AGENTS.md conventions.",
      "files": "Native/PcaiNative/",
      "priority": "medium",
      "tags": ["csharp", "pinvoke", "ffi"]
    },
    "functiongemma": {
      "prompt": "Review Deploy/rust-functiongemma-core/src/ and Deploy/rust-functiongemma-runtime/src/ for: router logic correctness, GPU config handling, prompt template safety, LoRA loading edge cases. Run: cargo test -p rust-functiongemma-core --lib. Follow AGENTS.md conventions.",
      "files": "Deploy/rust-functiongemma-core/src/,Deploy/rust-functiongemma-runtime/src/",
      "priority": "low",
      "tags": ["rust", "router"]
    }
  }
}
```

- [ ] **Step 2: Validate JSON is well-formed**

Run: `pwsh -Command "Get-Content Config/jules-review-prompts.json | ConvertFrom-Json -Depth 5 | Select-Object -ExpandProperty modules | Get-Member -MemberType NoteProperty | Measure-Object | Select-Object -ExpandProperty Count"`
Expected: `12` (number of modules)

- [ ] **Step 3: Commit**

```bash
git add Config/jules-review-prompts.json
git commit -m "feat(jules): add per-module review prompt templates for batch dispatch"
```

---

## Task 3: Batch Review Pipeline — `Invoke-JulesBatchReview.ps1`

**Files:**
- Create: `Tools/Invoke-JulesBatchReview.ps1`

Depends on: Task 1 (Invoke-JulesSession.ps1), Task 2 (jules-review-prompts.json)

- [ ] **Step 1: Write failing test for batch dispatch**

Add to `Tests/Unit/Invoke-JulesSession.Tests.ps1` (or create a separate file):

```powershell
Describe 'Invoke-JulesBatchReview' {
    BeforeAll {
        $script:ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
        $script:ScriptPath = Join-Path $script:ProjectRoot 'Tools' 'Invoke-JulesBatchReview.ps1'
    }

    Context 'Parameter validation' {
        It 'Throws when no scope or modules specified' {
            { & $script:ScriptPath } | Should -Throw
        }
    }

    Context 'Prompt template loading' {
        It 'Loads and parses jules-review-prompts.json' {
            $config = Get-Content (Join-Path $script:ProjectRoot 'Config' 'jules-review-prompts.json') | ConvertFrom-Json -Depth 5
            $config.modules | Should -Not -BeNullOrEmpty
            $config.modules.pcai_inference.prompt | Should -Not -BeNullOrEmpty
            $config.modules.pcai_inference.priority | Should -Be 'high'
        }
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pwsh -Command "Invoke-Pester Tests/Unit/Invoke-JulesSession.Tests.ps1 -Output Detailed -Filter 'Invoke-JulesBatchReview'"`
Expected: Parameter validation test FAILS (script does not exist)

- [ ] **Step 3: Implement `Invoke-JulesBatchReview.ps1`**

Create `Tools/Invoke-JulesBatchReview.ps1` with:

```powershell
#Requires -Version 5.1
<#
.SYNOPSIS
    Batch dispatch Jules review sessions across PC_AI modules.
.DESCRIPTION
    Reads module definitions from Config/jules-review-prompts.json and dispatches
    targeted Jules sessions. Supports scoping by language, specific modules, or
    git diff since a tag.
.PARAMETER Scope
    Predefined scope: RustCrates, PowerShellModules, CSharp, All.
.PARAMETER Modules
    Comma-separated list of specific module names from the config.
.PARAMETER ChangedSinceTag
    Dispatch reviews only for modules with files changed since this git tag.
.PARAMETER MaxSessions
    Maximum number of sessions to dispatch (default: 5).
.PARAMETER RequirePlanApproval
    Require plan approval for all dispatched sessions (default: true).
.PARAMETER DryRun
    Show what would be dispatched without creating sessions.
.PARAMETER Format
    Output format: Table or Json (default: Table).
#>
[CmdletBinding(DefaultParameterSetName = 'ByScope')]
param(
    [Parameter(Mandatory, ParameterSetName = 'ByScope')]
    [ValidateSet('RustCrates', 'PowerShellModules', 'CSharp', 'All')]
    [string]$Scope,

    [Parameter(Mandatory, ParameterSetName = 'ByModule')]
    [string[]]$Modules,

    [Parameter(ParameterSetName = 'ByDiff')]
    [string]$ChangedSinceTag,

    [int]$MaxSessions = 5,
    [switch]$RequirePlanApproval = $true,
    [switch]$DryRun,
    [ValidateSet('Table', 'Json')][string]$Format = 'Table'
)

$ErrorActionPreference = 'Stop'
$configPath = Join-Path $PSScriptRoot '..' 'Config' 'jules-review-prompts.json'
$config = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json -Depth 5
$wrapperScript = Join-Path $PSScriptRoot 'Invoke-JulesSession.ps1'

# Resolve which modules to dispatch
$targetModules = switch ($PSCmdlet.ParameterSetName) {
    'ByScope' {
        $tagMap = @{
            'RustCrates'        = 'rust'
            'PowerShellModules' = 'powershell'
            'CSharp'            = 'csharp'
            'All'               = $null
        }
        $tag = $tagMap[$Scope]
        $config.modules.PSObject.Properties | Where-Object {
            -not $tag -or ($_.Value.tags -contains $tag)
        } | ForEach-Object { $_.Name }
    }
    'ByModule' { $Modules }
    'ByDiff' {
        $changedFiles = git diff --name-only "$ChangedSinceTag..HEAD" 2>$null
        $config.modules.PSObject.Properties | Where-Object {
            $files = $_.Value.files -split ','
            $files | Where-Object { $f = $_; $changedFiles | Where-Object { $_ -like "$f*" } }
        } | ForEach-Object { $_.Name }
    }
}

# Sort by priority, limit to MaxSessions
$priorityOrder = @{ 'high' = 1; 'medium' = 2; 'low' = 3 }
$sorted = $targetModules | Sort-Object {
    $mod = $config.modules.$_
    $priorityOrder[$mod.priority] ?? 99
} | Select-Object -First $MaxSessions

$results = @()
foreach ($modName in $sorted) {
    $mod = $config.modules.$modName
    if ($DryRun) {
        $results += [PSCustomObject]@{
            Module   = $modName
            Priority = $mod.priority
            Prompt   = $mod.prompt.Substring(0, [Math]::Min(80, $mod.prompt.Length)) + '...'
            Action   = 'DRY RUN'
        }
    }
    else {
        $sessionResult = & $wrapperScript -Action New -Prompt $mod.prompt -RequirePlanApproval:$RequirePlanApproval -AutomationMode AutoCreatePR -Format Json
        $results += [PSCustomObject]@{
            Module    = $modName
            Priority  = $mod.priority
            SessionId = ($sessionResult.name -split '/')[-1]
            State     = $sessionResult.state
        }
    }
}

# Persist results
if (-not $DryRun -and $results.Count -gt 0) {
    $outDir = Join-Path $PSScriptRoot '..' '.pcai' 'jules' 'sessions'
    New-Item -ItemType Directory -Path $outDir -Force | Out-Null
    $outPath = Join-Path $outDir "batch-$(Get-Date -Format 'yyyyMMdd-HHmmss').json"
    $results | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $outPath
    Write-Host "Results saved to: $outPath"
}

if ($Format -eq 'Table') { $results | Format-Table -AutoSize }
else { $results | ConvertTo-Json -Depth 5 }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pwsh -Command "Invoke-Pester Tests/Unit/Invoke-JulesSession.Tests.ps1 -Output Detailed"`
Expected: All tests PASS (including the new batch review tests)

- [ ] **Step 5: Commit**

```bash
git add Tools/Invoke-JulesBatchReview.ps1 Tests/Unit/Invoke-JulesSession.Tests.ps1
git commit -m "feat(jules): add batch review dispatch with priority ordering and dry-run support"
```

---

## Task 4: PR Triage Dashboard — `Get-JulesPRStatus.ps1`

**Files:**
- Create: `Tools/Get-JulesPRStatus.ps1`

- [ ] **Step 1: Implement `Get-JulesPRStatus.ps1`**

```powershell
#Requires -Version 5.1
<#
.SYNOPSIS
    List and triage Jules-created pull requests.
.DESCRIPTION
    Uses gh CLI to list PRs created by Jules, enriched with CI status,
    merge readiness, and conflict information.
.PARAMETER State
    PR state filter: open, closed, merged, all (default: open).
.PARAMETER Format
    Output format: Table or Json (default: Table).
#>
[CmdletBinding()]
param(
    [ValidateSet('open', 'closed', 'merged', 'all')][string]$State = 'open',
    [ValidateSet('Table', 'Json')][string]$Format = 'Table'
)

$ErrorActionPreference = 'Stop'

# Verify gh is available
if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
    throw 'gh CLI not found. Install: winget install GitHub.cli'
}

$ghState = if ($State -eq 'all') { '--state all' } elseif ($State -eq 'merged') { '--state merged' } else { "--state $State" }
$prs = gh pr list $ghState --author 'jules[bot]' --json number,title,state,mergeable,statusCheckRollup,changedFiles,headRefName,createdAt,url 2>$null | ConvertFrom-Json

if (-not $prs -or $prs.Count -eq 0) {
    Write-Host "No Jules PRs found with state: $State"
    return
}

$results = foreach ($pr in $prs) {
    $ciStatus = if ($pr.statusCheckRollup) {
        $states = $pr.statusCheckRollup | ForEach-Object { $_.status ?? $_.conclusion }
        if ($states -contains 'FAILURE') { 'FAILED' }
        elseif ($states -contains 'PENDING') { 'PENDING' }
        else { 'PASSED' }
    }
    else { 'NONE' }

    [PSCustomObject]@{
        'PR#'      = $pr.number
        Title      = $pr.title
        State      = $pr.state
        Mergeable  = $pr.mergeable
        CI         = $ciStatus
        Files      = $pr.changedFiles
        Branch     = $pr.headRefName
        Created    = $pr.createdAt
        URL        = $pr.url
    }
}

if ($Format -eq 'Table') { $results | Format-Table -AutoSize }
else { $results | ConvertTo-Json -Depth 5 }
```

- [ ] **Step 2: Smoke test locally**

Run: `pwsh -File Tools/Get-JulesPRStatus.ps1 -State all -Format Table`
Expected: Table output (may be empty if no Jules PRs exist yet — that's fine, verify no errors)

- [ ] **Step 3: Commit**

```bash
git add Tools/Get-JulesPRStatus.ps1
git commit -m "feat(jules): add PR triage dashboard via gh CLI"
```

---

## Task 5: LLM Orchestrator — `Invoke-JulesOrchestrator.ps1`

**Files:**
- Create: `Tools/Invoke-JulesOrchestrator.ps1`
- Test: `Tests/Unit/Invoke-JulesOrchestrator.Tests.ps1`

Depends on: Task 1 (Invoke-JulesSession.ps1), Task 2 (jules-review-prompts.json)

- [ ] **Step 1: Write failing tests for orchestrator**

Create `Tests/Unit/Invoke-JulesOrchestrator.Tests.ps1`:

```powershell
#Requires -Modules @{ ModuleName = 'Pester'; ModuleVersion = '5.0.0' }

BeforeAll {
    $script:ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
    $script:ScriptPath  = Join-Path $script:ProjectRoot 'Tools' 'Invoke-JulesOrchestrator.ps1'
}

Describe 'Invoke-JulesOrchestrator' {

    Context 'Parameter validation' {
        It 'Throws when Action is missing' {
            { & $script:ScriptPath } | Should -Throw
        }
        It 'Accepts AnalyzeAndDispatch action' {
            # DryRun should not require API key
            { & $script:ScriptPath -Action AnalyzeAndDispatch -DryRun } | Should -Not -Throw '*Action*'
        }
        It 'Throws when ReviewPlan has no SessionId' {
            { & $script:ScriptPath -Action ReviewPlan } | Should -Throw '*SessionId*'
        }
    }

    Context 'Signal gathering (AnalyzeAndDispatch -DryRun)' {
        It 'Produces analysis JSON with expected structure' {
            $result = & $script:ScriptPath -Action AnalyzeAndDispatch -DryRun -Format Json 2>$null
            $parsed = $result | ConvertFrom-Json -Depth 10 -ErrorAction SilentlyContinue
            # Analysis should have signals and recommended_sessions keys
            $parsed.signals | Should -Not -BeNullOrEmpty
            $parsed.recommended_sessions | Should -Not -BeNullOrEmpty
        }
    }

    Context 'Plan review criteria' {
        BeforeAll {
            . $script:ScriptPath -Action '__test_load__' 2>$null
        }

        It 'Scores a well-scoped plan as pass' {
            $plan = @{
                steps = @(
                    @{ title = 'Fix unwrap in http/mod.rs'; description = 'Replace unwrap with expect' }
                    @{ title = 'Add test for error path'; description = 'New test in http/mod.rs tests' }
                )
            }
            $assessment = Invoke-PlanReview -Plan $plan -Module 'pcai_inference' -Scope 'Native/pcai_core/pcai_inference/src/'
            $assessment.scope_check | Should -Be 'pass'
            $assessment.test_coverage | Should -Be 'pass'
        }

        It 'Scores a plan with no tests as warn' {
            $plan = @{
                steps = @(
                    @{ title = 'Refactor config parsing'; description = 'Simplify config.rs parsing logic' }
                )
            }
            $assessment = Invoke-PlanReview -Plan $plan -Module 'pcai_inference' -Scope 'Native/pcai_core/pcai_inference/src/'
            $assessment.test_coverage | Should -Be 'warn'
        }
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pwsh -Command "Invoke-Pester Tests/Unit/Invoke-JulesOrchestrator.Tests.ps1 -Output Detailed"`
Expected: All tests FAIL

- [ ] **Step 3: Implement `Invoke-JulesOrchestrator.ps1`**

Create `Tools/Invoke-JulesOrchestrator.ps1` with:

1. **Parameters:**
   - `[Parameter(Mandatory)][ValidateSet('AnalyzeAndDispatch','ReviewPlans','ReviewPlan')][string]$Action`
   - `[string]$SessionId`
   - `[int]$MaxSessions = 5`, `[switch]$DryRun`
   - `[ValidateSet('Table','Json','Report')][string]$Format = 'Json'`
   - `[int]$MaxFeedbackRounds = 3`, `[int]$PollIntervalSeconds = 30`, `[int]$PollTimeoutMinutes = 10`

2. **`AnalyzeAndDispatch` action:**
   - Gather signals by running:
     - `cargo clippy --manifest-path Native/pcai_core/Cargo.toml --all-targets --message-format=json 2>$null` — count `warning` level messages
     - `git diff --name-only HEAD~20` — map changed files to modules via `jules-review-prompts.json` file paths
     - `git grep -c "TODO\|FIXME\|HACK" -- "Native/" "Modules/"` — count markers per directory
     - `gh issue list --label bug,enhancement --json number,title --limit 10` — open issues
   - Build analysis JSON with `signals` and `recommended_sessions` (sorted by: clippy violations > recently changed > high priority)
   - If not `-DryRun`, call `Invoke-JulesSession -Action New` for each recommended session
   - Write results to `.pcai/jules/analysis-<timestamp>.json` and `.pcai/jules/dispatch-<timestamp>.json`

3. **`ReviewPlans` action:**
   - Call `Invoke-JulesSession -Action List -State AwaitingPlanApproval -Format Json`
   - For each session, call `Invoke-JulesSession -Action GetPlan -SessionId $id`
   - Run `Invoke-PlanReview` on each plan
   - Output assessment report

4. **`ReviewPlan` action:**
   - Get plan for `$SessionId`
   - Run `Invoke-PlanReview`
   - If recommendation is `approve`, call `Invoke-JulesSession -Action Approve`
   - If recommendation is `feedback`, call `Invoke-JulesSession -Action Message`, then poll for revised plan (loop up to `$MaxFeedbackRounds`)
   - If recommendation is `reject`, log and skip

5. **`Invoke-PlanReview` internal function:**
   - Takes plan steps, module name, and scope path
   - Checks each criterion from the spec:
     - **Scope**: Do step titles/descriptions reference files within scope? `pass` if all in-scope, `warn` if 1-2 outside, `fail` if >2 outside or mentions FFI signature changes
     - **Tests**: Do any steps mention "test" in title? `pass` if yes, `warn` if no but steps are low-risk (title contains "refactor", "rename", "format"), `fail` if steps remove tests or change behavior words without test steps
     - **Conventions**: Do steps mention "unwrap" without "expect"? Check for "unsafe" without "document" nearby. `pass`/`warn`/`fail`
     - **Benchmark**: Do steps reference hot-path files (generate.rs, attention, decode)? If so, do they mention "benchmark"? `none`/`low`/`high`
   - Returns `recommendation`: `approve` if all pass/warn, `feedback` if any fail with actionable message, `reject` if >2 fails

6. **`__test_load__` action** — expose `Invoke-PlanReview` for testing

- [ ] **Step 4: Run tests to verify they pass**

Run: `pwsh -Command "Invoke-Pester Tests/Unit/Invoke-JulesOrchestrator.Tests.ps1 -Output Detailed"`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add Tools/Invoke-JulesOrchestrator.ps1 Tests/Unit/Invoke-JulesOrchestrator.Tests.ps1
git commit -m "feat(jules): add LLM orchestrator with smart dispatch and iterative plan review"
```

---

## Task 6: AGENTS.md Enhancement

**Files:**
- Modify: `AGENTS.md` (append after line 253)

- [ ] **Step 1: Read current end of AGENTS.md to confirm insertion point**

Run: `pwsh -Command "(Get-Content AGENTS.md).Count"` — verify ~254 lines
Read lines 245-254 to confirm the last section is "Documentation and automation"

- [ ] **Step 2: Append Jules agent guidance section**

Add the following after the last line of `AGENTS.md`:

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
3. Does it follow the benchmark-first principle (AGENTS.md section 1)?
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

- [ ] **Step 3: Verify line count is reasonable**

Run: `pwsh -Command "(Get-Content AGENTS.md).Count"`
Expected: ~304 lines (254 original + ~50 new)

- [ ] **Step 4: Commit**

```bash
git add AGENTS.md
git commit -m "docs(agents): add Jules agent guidance section with VM constraints and review criteria"
```

---

## Task 7: GitHub Actions Workflow — `jules-review.yml`

**Files:**
- Create: `.github/workflows/jules-review.yml`

- [ ] **Step 1: Create the workflow file**

Create `.github/workflows/jules-review.yml`:

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

permissions:
  issues: read
  pull-requests: write
  contents: read

jobs:
  jules-issue:
    name: Jules Issue Fix
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
    name: Jules Manual Task
    if: github.event_name == 'workflow_dispatch'
    runs-on: ubuntu-latest
    steps:
      - uses: google-labs-code/jules-invoke@v1
        with:
          prompt: ${{ inputs.prompt }}
          jules_api_key: ${{ secrets.JULES_API_KEY }}
          starting_branch: main

  jules-weekly:
    name: Jules Weekly Review (${{ matrix.target.name }})
    if: github.event_name == 'schedule'
    runs-on: ubuntu-latest
    strategy:
      matrix:
        target:
          - name: pcai_inference
            prompt: "Review Native/pcai_core/pcai_inference/src/ for code quality, test gaps, and clippy compliance. Focus on error handling and unsafe blocks. Follow AGENTS.md conventions."
          - name: pcai_media
            prompt: "Review Native/pcai_core/pcai_media/src/ for unnecessary allocations, missing error context, and test coverage gaps. Follow AGENTS.md conventions."
          - name: pcai_core_lib
            prompt: "Review Native/pcai_core/pcai_core_lib/src/ for dead code, missing docs on public items, and potential panics. Follow AGENTS.md conventions."
      max-parallel: 3
    steps:
      - uses: google-labs-code/jules-invoke@v1
        with:
          prompt: ${{ matrix.target.prompt }}
          jules_api_key: ${{ secrets.JULES_API_KEY }}
          starting_branch: main
```

- [ ] **Step 2: Validate YAML syntax**

Run: `pwsh -Command "Get-Content .github/workflows/jules-review.yml -Raw | ConvertFrom-Yaml" 2>$null; echo $?`
Or: `python -c "import yaml; yaml.safe_load(open('.github/workflows/jules-review.yml'))" 2>$null; echo $?`

If no YAML parser available, visually confirm indentation is consistent (2-space).

- [ ] **Step 3: Create the `jules` label on GitHub**

Run: `gh label create jules --description "Trigger Jules AI agent" --color 4285F4 --repo David-Martel/PC-AI`
Expected: Label created (or "already exists" if it was created before)

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/jules-review.yml
git commit -m "ci(jules): add scheduled weekly review, issue-triggered, and manual dispatch workflows"
```

---

## Task 8: Update CLAUDE.md and Documentation References

**Files:**
- Modify: `CLAUDE.md` (add Jules tools to relevant sections)

- [ ] **Step 1: Add Jules workflow to CI/CD table in CLAUDE.md**

The `jules-review.yml` workflow was already added in the earlier CLAUDE.md audit. Verify it's present.

Run: `grep 'jules-review' CLAUDE.md`
Expected: Line containing `jules-review.yml`

- [ ] **Step 2: Add Jules tools to the Tools count and references**

If the Tools count in CLAUDE.md architecture tree needs updating (now +4 scripts: Invoke-JulesSession, jules_api.cmd, Invoke-JulesBatchReview, Get-JulesPRStatus, Invoke-JulesOrchestrator = 67 + 5 = 72), update it.

- [ ] **Step 3: Update CLAUDE.TODO.md header**

Update the coordinating agents line to reflect the new tooling:

Change: `Coordinating agents: Claude, Codex, Jules (53 sessions complete)`
To: `Coordinating agents: Claude, Codex, Jules (53 sessions complete, API wrapper + orchestrator active)`

- [ ] **Step 4: Add `.pcai/jules/` to .gitignore**

Verify `.pcai/` is already gitignored. If not, add:

```
# Jules session artifacts (API responses, patches, analysis)
.pcai/jules/
```

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md CLAUDE.TODO.md .gitignore
git commit -m "docs: update CLAUDE.md with Jules tooling, update TODO header"
```

---

## Task 9: Verify End-to-End (Manual Smoke Test)

This task is a checklist, not code. Run each command and verify no errors.

- [ ] **Step 1: Verify API wrapper loads without errors**

Run: `pwsh -Command "& Tools/Invoke-JulesSession.ps1 -Action Version"`
Expected: Jules CLI version output (e.g., `1.x.x`)

- [ ] **Step 2: Verify batch review dry-run**

Run: `pwsh -File Tools/Invoke-JulesBatchReview.ps1 -Scope RustCrates -DryRun -Format Table`
Expected: Table showing 7 Rust crate modules with DRY RUN action, sorted by priority

- [ ] **Step 3: Verify orchestrator dry-run**

Run: `pwsh -File Tools/Invoke-JulesOrchestrator.ps1 -Action AnalyzeAndDispatch -DryRun -Format Json`
Expected: JSON with `signals` and `recommended_sessions` keys

- [ ] **Step 4: Verify PR status (may be empty)**

Run: `pwsh -File Tools/Get-JulesPRStatus.ps1 -State all -Format Table`
Expected: Table output or "No Jules PRs found" message

- [ ] **Step 5: Verify all Pester tests pass**

Run: `pwsh -Command "Invoke-Pester Tests/Unit/Invoke-JulesSession.Tests.ps1, Tests/Unit/Invoke-JulesOrchestrator.Tests.ps1 -Output Detailed"`
Expected: All tests PASS

- [ ] **Step 6: Verify CMD wrapper**

Run: `Tools\jules_api.cmd -Action Version`
Expected: Same output as Step 1

---

## Dependency Graph

```
Task 1 (API Wrapper) ─────┬──> Task 3 (Batch Review)
                           ├──> Task 4 (PR Triage)
Task 2 (Prompt Templates) ─┤
                           └──> Task 5 (Orchestrator)
Task 6 (AGENTS.md) ────────── independent
Task 7 (GitHub Action) ────── independent
Task 8 (Docs Update) ──────── after Tasks 1-7
Task 9 (Smoke Test) ───────── after all
```

Tasks 1, 2, 6, 7 can run in parallel. Tasks 3, 4, 5 depend on Task 1+2. Task 8 is a cleanup pass. Task 9 is final verification.
