# AGENTS.md

Current agent guidance for `PC_AI` as of April 30, 2026.

## Mission

`PC_AI` is no longer just a local diagnostics shell around an LLM. It is an
active PowerShell + Rust/C# platform for:

- deterministic Windows diagnostics and optimization
- native-first acceleration for expensive tooling paths
- local LLM inference, routing, and evaluation
- benchmark-driven performance work
- an emerging multimodal/media stack built around Janus-style models
- boot/logon, VHD, sync-provider, Process Lasso, and UI-responsiveness
  reliability work for this Windows workstation

Agents working in this repo should optimize for measurable improvement, not
just feature addition.

## Current priorities

### 1. Benchmark-first optimization of the PC-AI toolchain

The repo now has a real tooling benchmark path and agents should use it when
changing acceleration, search, context gathering, or other hot-path tooling.

Primary entrypoints:

- `Tests/Benchmarks/Invoke-PcaiToolingBenchmarks.ps1`
- `Config/pcai-tooling-benchmarks.json`
- `Reports/tooling-benchmarks/<timestamp>/`
- `Reports/TOOL_BACKEND_COVERAGE.md`

Current benchmark guidance from the repo:

- native `directory-manifest` is a clear win and should be preferred
- native `token-estimate` is also a meaningful improvement
- native `full-context` is promising but still moderate
- accelerated `fd` / `rg` remains the preferred fast path for generic
  file/content search until the Rust search layer catches up

Do not claim a performance win without either:

- a tooling benchmark run
- an evaluation baseline / regression comparison
- a targeted microbenchmark for the changed routine

### 2. Native-first parity and startup-cost reduction

The architecture direction is still Rust/C# first where that produces better
determinism, performance, or reuse. Current backlog themes pulled from
`TODO.md` and `optimization.TODO.md`:

- reduce cold import cost in `PC-AI.Acceleration`
- stop paying full module parse / dot-source costs on every startup
- add batched native manifest/search APIs instead of repeated point queries
- standardize C ABI contracts and JSON schemas
- improve cancellation, structured logging, and error translation across
  PowerShell -> C# -> Rust

When touching the acceleration layer, keep startup latency and cache behavior as
first-class constraints.

### 3. AO-media / AI-Media and native media stack uplift

The repo contains both:

- legacy/prototype media work in `AI-Media/`
- the canonical native media path in `Native/pcai_core/pcai_media_model/`,
  `Native/pcai_core/pcai_media/`, and `Native/pcai_core/pcai_media_server/`

Agents should treat the native `pcai_media*` crates as the long-term home of
the media agent, with `AI-Media/` mainly useful as prototype/reference code.

Current media-stack expectations:

- keep Janus/vision-generation changes aligned across Rust, C#, and PowerShell
- expand testing fixtures instead of relying only on ad hoc manual runs
- benchmark performance-sensitive tensor, attention, decode, and image pipeline
  routines
- prefer optimized implementations that stay measurable and testable
- avoid one-off prototype improvements that never land in the canonical native
  crates

The media backlog currently includes both fixture expansion and performance
cleanup. If you optimize media routines, add or update tests and document how to
reproduce the measurement.

### 4. Workstation boot, sync, and UI responsiveness hardening

The 2026-04-30 workstation investigation produced maintained tooling for VHD
startup, Filter Manager visibility, Process Lasso policy, OneDrive repair,
risky registry-script rollback, and `~\bin` script risk review. Treat this as a
first-class diagnostics surface, not scratch work.

Primary entrypoints:

- `boot.TODO.md`
- `Tools\Collect-BootDiagnostics.ps1`
- `Tools\Test-BootMountHealth.ps1`
- `Tools\Test-SyncProviderHealth.ps1`
- `Tools\Test-ProcessLassoBootSafety.ps1`
- `Tools\Apply-ProcessLassoUiSyncTuning.ps1`
- `Tools\Collect-DrivePerformanceSyncRisk.ps1`
- `Tools\Repair-OneDriveSync.ps1`
- `Tools\Migrate-SystemScriptsIntoRepo.ps1`
- `Tools\SystemScripts\README.md`
- `Tests\Boot\PersistentVHDX.Tests.ps1`
- `Tests\Boot\BootValidationTools.Tests.ps1`
- `Reports\boot-diagnostics\`
- `Reports\drive-performance-sync-risk\`
- `Reports\onedrive-repair-20260430.md`
- `Reports\bin-script-risk-review-20260430.md`

Operational rules:

- Preserve OneDrive, Filter Manager, Task Scheduler, VHD, and Process Lasso
  evidence before changing startup or registry policy.
- Keep Task Scheduler scripts and other workstation-mutating helpers under
  `Tools\SystemScripts` rather than scattered through `C:\Scripts`,
  `~\.machine`, `~\.local\bin`, `~\bin`, or OneDrive script folders.
- Write-capable session scripts must expose `-h`, `--help`, and non-mutating
  `-DryRun` behavior, and their tests must prove dry-run suppresses side
  effects including file writes, event-log writes, Task Scheduler mutation, and
  mount operations where applicable.
- Keep the UDM SMB auto-launch disabled until OneDrive has a clean sync-health
  window, unless the user explicitly asks to re-enable it.
- Do not add broad Explorer remote-change-notification or filesystem/cache
  registry tweaks as OneDrive fixes without Microsoft-supported rationale,
  rollback artifacts, and before/after metrics.
- Treat Process Lasso as a prioritization aid around user-mode contention. It
  cannot directly reprioritize kernel HID/I2C interrupt handling.

## Architecture quick map

- `PC-AI.ps1`: unified CLI entry point
- `Modules/`: PowerShell command surface
  - `PC-AI.LLM`: inference orchestration and router integration
  - `PC-AI.Evaluation`: evaluation, baselines, regressions, A/B testing
  - `PC-AI.Acceleration`: Rust CLI + native DLL acceleration
  - `PcaiMedia.psm1`: PowerShell media wrapper over native media bindings
- `Native/PcaiNative/`: C# bridge and native resolver layer
- `Native/pcai_core/`: Rust workspace
  - `pcai_inference`: llama.cpp / mistral.rs inference backends
  - `pcai_core_lib`: shared native acceleration surface
  - `pcai_media_model`: Janus-style model components
  - `pcai_media`: media pipeline + FFI exports
  - `pcai_media_server`: server wrapper for media APIs
- `AI-Media/`: older standalone/prototype multimodal workspace
- `Tests/`: Pester, evaluation, benchmark, and integration automation
- `Tools/`: repo-local diagnostics, repair, benchmarking, and workstation
  hardening scripts
- `Tools/SystemScripts/`: repo-owned Task Scheduler and workstation
  system-modification scripts migrated from local bin, machine, OneDrive, and
  UDM script folders
- `Reports/`: evidence snapshots and generated diagnostics; preserve dated
  artifacts that explain workstation or benchmark decisions

## Prompt and routing contracts

- Diagnose mode uses `DIAGNOSE.md` + `DIAGNOSE_LOGIC.md`
  - output must be valid JSON per `Config/DIAGNOSE_TEMPLATE.json`
  - findings must be evidence-first and tied to concrete report/log lines
- Chat mode uses `CHAT.md`
- Tool-routing schema lives in `Config/pcai-tools.json`
- Router scenarios/training data live under
  `Deploy/rust-functiongemma-train/examples/`

If a tool or native capability changes diagnostic behavior, update the prompt
contracts and routing/training assets in the same workstream.

## Build and runtime workflow

Recommended build entrypoint:

```powershell
.\Build.ps1
```

Useful variants:

```powershell
.\Build.ps1 -Component inference -EnableCuda
.\Build.ps1 -Component llamacpp -EnableCuda
.\Build.ps1 -Clean -Package -EnableCuda
```

Direct inference builds:

```powershell
cd Native\pcai_core\pcai_inference
.\Invoke-PcaiBuild.ps1 -Backend llamacpp -Configuration Release
.\Invoke-PcaiBuild.ps1 -Backend all -Configuration Release -EnableCuda
```

Version metadata comes from git via `Tools/Get-BuildVersion.ps1`.

## Benchmark and evaluation workflow

### Tooling/runtime benchmarks

Use when changing acceleration, native wrappers, repo search/context routines,
or startup-sensitive code:

```powershell
pwsh .\Tests\Benchmarks\Invoke-PcaiToolingBenchmarks.ps1 -Suite quick
pwsh .\Tests\Benchmarks\Invoke-PcaiToolingBenchmarks.ps1 -CaseId content-search,full-context
pwsh .\Tests\Invoke-AllTests.ps1 -Suite Benchmarks
```

### LLM evaluation and regressions

Use when changing inference, prompts, routing, model defaults, or evaluation
logic:

```powershell
pwsh .\Tests\Evaluation\Invoke-InferenceEvaluation.ps1 `
  -Backend llamacpp-bin `
  -ModelPath "C:\Models\tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf" `
  -Dataset diagnostic `
  -RunLabel local-smoke
```

Key support features already exist:

- baselines via `New-BaselineSnapshot`
- regression detection via `Test-ForRegression`
- evaluation artifacts under `.pcai/evaluation/runs/`

Agents should prefer baseline/regression comparisons over one-off anecdotes.

## Media-agent guidance

If the task touches the Janus/media path:

1. Check whether the change belongs in `AI-Media/` or should be ported directly
   into `Native/pcai_core/pcai_media_model` or `Native/pcai_core/pcai_media`.
2. Keep the FFI and PowerShell/C# wrapper implications in scope:
   `Native/PcaiNative/MediaModule.cs` and `Modules/PcaiMedia.psm1`.
3. Add or extend test fixtures for prompts, images, tensor transforms, async
   request handling, or model-loading edge cases.
4. Add a reproducible benchmark or profiling note for optimized routines.
5. Prefer improvements that can graduate from prototype code into the canonical
   native media crates.

High-value media work right now:

- broaden fixtures beyond basic constructor/FFI smoke coverage
- benchmark decode/attention/tensor hot paths
- tighten async request lifecycle and cancellation behavior
- align error/reporting behavior with the rest of the native stack

## Testing expectations

At minimum, choose the narrowest relevant validation path:

- Pester for PowerShell module behavior
- Rust unit/integration tests for native crates
- evaluation runs for inference or prompt behavior
- tooling benchmarks for hot-path acceleration changes
- baseline/regression comparisons for performance-sensitive work

Important active testing gaps:

- more native DLL availability and fallback coverage as new surfaces are added
- benchmark-backed regression tests for acceleration hot paths
- stronger fixture coverage for the media agent and multimodal routines
- clean post-reboot validation windows for OneDrive, VHD mount health, Process
  Lasso governor presence, and touchpad/UI glitch correlation

## Runtime diagnostics

Useful commands when the LLM or native stack is unhealthy:

- `Invoke-PcaiDoctor`
- `Get-PcaiServiceHealth`
- `Get-PcaiNativeStatus`
- `Get-PcaiCapabilities`

Useful commands when boot, sync, or UI responsiveness is unhealthy:

- `Tools\Collect-BootDiagnostics.ps1 -SinceMinutes 120 -PostRebootVerify`
- `Tools\Test-BootMountHealth.ps1 -SinceMinutes 60 -PassThru`
- `Tools\Test-SyncProviderHealth.ps1 -SinceMinutes 60 -PassThru`
- `Tools\Test-ProcessLassoBootSafety.ps1`
- `Tools\Collect-DrivePerformanceSyncRisk.ps1 -SinceMinutes 240`

Service endpoints:

- pcai-inference: `http://127.0.0.1:8080/health` and `/v1/models`
- FunctionGemma router: `http://127.0.0.1:8000/health` and `/v1/models`

## Outstanding repo work

These themes should be treated as live backlog, not stale notes:

- large-context offload ideas for `pcai_inference`
- versioned C ABI contract and shared native schemas
- cancellation/timeouts across the full host stack
- streaming/progress support for long native operations
- structured native logging and metrics
- expanded benchmarking coverage for startup, caching, and batched search
- media-agent fixture growth and performance tuning
- continued consolidation away from PowerShell-only implementations when the
  native path is clearly better
- OneDrive post-repair monitoring, stale task triage, cloud-root-on-VHD startup
  ordering, and `~\bin` script hardening from `boot.TODO.md`

## Documentation and automation

- `Tools/Invoke-DocPipeline.ps1 -Mode Full`
- `Tools/Invoke-DocPipeline.ps1 -Mode DocsOnly`
- `Tools/generate-auto-docs.ps1 -BuildDocs`
- `Tools/generate-tools-catalog.ps1`

Keep docs aligned with the real scripts, benchmarks, and active backlog. If the
repo gains a new benchmark, fixture suite, or native capability, update this
file along with the relevant README or module docs.

For workstation hardening changes, update `boot.TODO.md`, the relevant
`Reports\*.md` evidence summary, and any script comment-based help or Pester
coverage in the same workstream.

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
