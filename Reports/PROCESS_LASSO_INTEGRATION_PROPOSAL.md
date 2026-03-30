# Process Lasso Integration Proposal

## Goal

Make Process Lasso configurable from `C:\codedev\PC_AI\` using the repo's existing architecture:

- Rust for fast collection, parsing, correlation, and scoring
- C# for typed interop and PowerShell-friendly DTOs
- PowerShell for orchestration, reporting, safe apply, and operator workflows

The source of truth should move into the repo. `prolasso.ini` becomes a compiled target, not the primary place humans edit policy.

## Why This Fits PC_AI

PC_AI already has the right execution pattern:

- `Native\pcai_core\pcai_perf_cli\src\main.rs` for fast JSON/worker-style native commands
- `Native\PcaiNative\PerformanceModule.cs` for typed C# bridges over native results
- `Release\PowerShell\PC-AI\Modules\PC-AI.Acceleration\Public\Get-ProcessesFast.ps1` for PowerShell exposure with native-first fallback logic
- `Tools\Collect-SystemPerformanceData.ps1` for operator-facing collection and report generation

Process Lasso should use the same pattern rather than introducing a separate utility stack.

## Current Machine Evidence

The current workstation state is exactly why the integration should be data-driven:

- Live Process Lasso data lives at:
  - `C:\ProgramData\ProcessLasso\config\prolasso.ini`
  - `C:\ProgramData\ProcessLasso\logs\processlasso.log`
- Recent log output is dominated by repeated `Set Efficiency Mode -> OFF` entries for `pwsh.exe`, `chrome.exe`, and `brave.exe`. That is signal for rule churn and reporting noise.
- Current process state still shows policy drift that should be detected automatically:
  - `dwm.exe` is running at `RealTime`
  - `vmmemWSL` is the largest sustained system workload
  - two `claude.exe` processes are consuming large private memory
  - `com.docker.backend.exe` and NVidia container services remain material background pressure
- Current counter sample remains elevated:
  - committed bytes in use: `69.23%`
  - pages/sec: `346.79`
  - processor utility: `123.22`

That means a useful Process Lasso integration cannot stop at static config diffing. It needs to answer:

1. Which rules are firing repeatedly?
2. Which rules are ineffective or no-ops?
3. Which processes are actually causing interactivity loss?
4. Which settings drift away from repo policy over time?

## Proposed Architecture

### 1. Rust: `pcai_core_lib::process_lasso`

Add a new native module responsible for fast, deterministic analysis.

Suggested responsibilities:

- Parse `prolasso.ini` into normalized typed structures
- Parse `processlasso.log` efficiently, including aggregation by action, process, and time window
- Correlate Process Lasso events with live process state and Windows counters
- Score policy effectiveness using deterministic heuristics
- Emit compact JSON suitable for PowerShell and LLM summarization

Suggested native outputs:

- `ProcessLassoSnapshot`
- `ProcessLassoRuleHitSummary`
- `ProcessLassoPolicyDriftReport`
- `ProcessLassoPressureCorrelationReport`
- `ProcessLassoRecommendationSet`

Suggested command additions to `pcai_perf_cli`:

- `pcai-perf process-lasso snapshot`
- `pcai-perf process-lasso analyze --since-minutes 60`
- `pcai-perf process-lasso diff --desired <json> --live <ini>`
- `pcai-perf process-lasso compile --profile <json>`

### 2. C#: `Native\PcaiNative\ProcessLassoModule.cs`

Add a dedicated managed bridge rather than overloading `PerformanceModule.cs`.

Suggested DTOs:

- `ProcessLassoSnapshotResult`
- `ProcessLassoEventSummary`
- `ProcessLassoRuleDiagnostic`
- `ProcessLassoRecommendation`
- `ProcessLassoCompileResult`

Responsibilities:

- Marshal compact native results to typed .NET objects
- Expose both typed and JSON entrypoints for PowerShell
- Provide safe file operations around backup, write, and validation
- Normalize repo config into a compiled overlay targeting `prolasso.ini`

### 3. PowerShell: new module `PC-AI.ProcessLasso`

Keep orchestration separate from generic acceleration helpers.

Suggested public commands:

- `Get-ProcessLassoSnapshot`
- `Get-ProcessLassoReport`
- `Test-ProcessLassoPolicy`
- `Compare-ProcessLassoPolicy`
- `New-ProcessLassoOverlay`
- `Set-ProcessLassoProfile`
- `Export-ProcessLassoMetrics`
- `Invoke-ProcessLassoGuidance`

Operator behavior:

- default to read-only analysis
- back up live config before any apply
- emit JSON and Markdown reports into `Reports\process-lasso\<timestamp>\`
- restart `ProcessGovernor.exe` and `ProcessLasso.exe` only after validation succeeds

## Repo-Driven Configuration Model

### Source Of Truth

Add repo-owned config files:

- `Config\process-lasso.schema.json`
- `Config\process-lasso.ai-dev-workstation.json`
- later: `Config\process-lasso.<machine-or-role>.json`

### Compile Model

The repo config should be higher level than raw `prolasso.ini`.

It should express:

- workload classes: interactive, build, inference, virtualization, services
- thresholds: commit pressure, paging, CPU pressure, rule-churn thresholds
- intent: protect interactive shells and browsers, preserve compile throughput, contain runaway background AI workloads
- compiler hints: preserve unknown keys, emit overlay vs full replacement, require backup

The compiler then translates repo policy into:

- `OutOfControlProcessRestraint`
- `MemoryManagement`
- `ProcessDefaults`
- `ProcessAllowances`
- `GamingMode`
- `PowerManagement`

This prevents hand-maintained `ini` drift and makes policy reviewable in git.

## Machine-Specific Policy Intent

This workstation is not a generic office desktop. It is a mixed interactive and batch box with routine:

- local AI and LLM workloads
- WSL and Docker backend load
- Rust compilation
- terminals and PowerShell-heavy orchestration
- browser-based research and monitoring

That means the policy model should explicitly separate:

### Interactive

- `chrome.exe`
- `brave.exe`
- `pwsh.exe`
- `windowsterminal.exe`
- `codex.exe`
- `code.exe`
- `explorer.exe`
- `dwm.exe`
- `zoom.exe`

Intent:

- keep out of Efficiency Mode
- keep out of ProBalance where appropriate
- preserve responsiveness
- detect if any of these are elevated above intended priority ceilings

### Build

- `cargo.exe`
- `rustc.exe`
- `cl.exe`
- `link.exe`
- `cmake.exe`
- `ninja.exe`
- `dotnet.exe`

Intent:

- allow throughput when foreground build activity is expected
- avoid permanent exemptions when the system is paging hard
- score whether compile jobs are starving interactivity

### AI / Inference

- `claude.exe`
- `ollama.exe`
- `python.exe` model runners
- `pcai_*`
- `llama-*`

Intent:

- allow sustained background work
- identify when these exceed configured memory or paging budgets
- recommend containment when they correlate strongly with degraded interactivity

### Virtualization / Backend

- `vmmemWSL`
- `wslservice.exe`
- `com.docker.backend.exe`
- `Docker Desktop.exe`

Intent:

- avoid forcing permanent high-performance mode
- correlate backend pressure with paging and foreground latency

### Service / Noise Floor

- `nvcontainer.exe`
- `NVDisplay.Container.exe`
- sync/telemetry helpers

Intent:

- detect when service overhead is excessive
- reduce invisible background tax

## Reports And Metrics

### Raw Artifacts

Per run, write:

- `snapshot.json`
- `live-processes.json`
- `rule-hit-summary.json`
- `policy-drift.json`
- `recommendations.json`
- `analysis.md`

under:

- `Reports\process-lasso\<yyyyMMdd_HHmmss>\`

### Core Metrics

Track metrics that actually help decisions:

- commit usage %
- pages/sec percentile and peaks
- top private-memory processes
- top CPU utility processes
- Process Lasso actions by type and process
- rule churn rate per hour
- SmartTrim action rate and no-action rate
- efficiency-mode toggles by process
- priority drift from desired policy
- no-op rule count, such as all-core CPU-set assignments

### Meaningful Derived Scores

Add deterministic scoring, for example:

- `interactive_responsiveness_risk`
- `background_pressure_score`
- `rule_noise_score`
- `policy_drift_score`
- `compile_throughput_protection_score`
- `ai_workload_containment_score`

## AI Guidance Layer

The LLM should not reason over raw logs directly. It should consume a compact structured bundle generated by deterministic analysis first.

Recommended flow:

1. Rust collector parses config, logs, counters, and process state.
2. C# bridge exposes a normalized summary object.
3. PowerShell writes a "search pin" style JSON packet.
4. Existing PC_AI LLM tooling summarizes findings and proposes adjustments.

Constraints for AI guidance:

- always include raw evidence references
- prefer recommendation bundles over freeform speculation
- require deterministic findings before apply suggestions
- distinguish "observed" from "inferred"

## Safe Apply Workflow

`Set-ProcessLassoProfile` should implement:

1. Load repo profile JSON
2. Compile overlay
3. Diff overlay against live `prolasso.ini`
4. Back up live config with timestamp
5. Apply overlay
6. Restart Process Lasso services/processes if needed
7. Re-sample counters and process state after a cooldown period
8. Emit before/after report

If validation fails, restore the previous backup automatically.

## Suggested File Layout

```text
Config/
  process-lasso.schema.json
  process-lasso.ai-dev-workstation.json

Native/
  PcaiNative/
    ProcessLassoModule.cs
  pcai_core/
    pcai_core_lib/
      src/process_lasso.rs
    pcai_perf_cli/
      src/main.rs

Release/PowerShell/PC-AI/Modules/
  PC-AI.ProcessLasso/
    PC-AI.ProcessLasso.psd1
    PC-AI.ProcessLasso.psm1
    Public/
      Get-ProcessLassoSnapshot.ps1
      Get-ProcessLassoReport.ps1
      Test-ProcessLassoPolicy.ps1
      New-ProcessLassoOverlay.ps1
      Set-ProcessLassoProfile.ps1
      Invoke-ProcessLassoGuidance.ps1

Tools/
  Invoke-ProcessLassoAnalysis.ps1
  Export-ProcessLassoPolicy.ps1
  Import-ProcessLassoPolicy.ps1
```

## Incremental Delivery Plan

### Phase 1: Read-Only Analysis

- parse live config and logs
- collect current counters and process state
- emit snapshot + drift + noise reports
- no mutation yet

### Phase 2: Repo Config + Compiler

- add JSON schema and workstation profile
- compile overlay `ini`
- validate generated output against current live config sections

### Phase 3: Safe Apply

- backup/diff/apply/restart flow
- before/after measurements
- rollback on validation failure

### Phase 4: AI Guidance

- integrate with existing PC_AI LLM routes
- generate explanation and recommendation bundles
- add profile tuning suggestions based on observed workloads

## Initial Recommendation

Start by implementing Phase 1 and Phase 2 only. That gives:

- visibility into why Process Lasso is helping or hurting
- git-reviewed policy management from `C:\codedev\PC_AI\`
- reproducible overlays instead of manual edits

Then add safe apply once the reports consistently reflect real machine behavior.
