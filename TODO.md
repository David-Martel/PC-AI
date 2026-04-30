# TODO

This is the active high-level backlog for `PC_AI`. Completed historical work
should be recorded in the relevant report, context snapshot, or specialized
ledger instead of left here as active work.

Last reconciled: 2026-04-30, after the boot/sync/Process Lasso/OneDrive tooling
pass and Dependabot/security cleanup.

## Recently Reconciled

- Prompt/tool-schema parity, module fallbacks, routed JSON enforcement, and
  bounded deterministic tool envelopes are complete.
- FunctionGemma health/model metadata, deterministic required-tool routing,
  GPU selection, and LoRA adapter load support are complete.
- Native `pcai_fs` consolidation, the capability registry, and native DLL
  availability/graceful fallback tests are complete.
- Boot/session tooling now has maintained VHD wrappers, Task Scheduler
  registration, Process Lasso policy/validation, OneDrive repair tooling,
  `-h`/`--help`, and `-DryRun` contract tests. The operational ledger is
  [boot.TODO.md](boot.TODO.md).
- Task Scheduler and selected system-modifying scripts from `C:\Scripts`,
  `~\.machine`, `~\.local\bin`, `~\bin`, OneDrive PowerShell script folders,
  and UDM startup folders are centralized under `Tools\SystemScripts`.
- Recent dependency-security work is merged to `main`; no open Dependabot PRs
  or alerts were present at the last validation.

## Active Priorities

### 1. OneDrive, Boot, And UI Responsiveness

- [ ] Monitor OneDrive after installer repair and reset until at least one clean
  60 minute `Tools\Test-SyncProviderHealth.ps1 -SinceMinutes 60 -PassThru` run
  shows no new OneDrive/FileSyncHelper WER events.
- [ ] Validate registry rollback after a clean reboot using
  `Tools\Collect-DrivePerformanceSyncRisk.ps1`, `Tools\Test-BootMountHealth.ps1`,
  `Tools\Test-SyncProviderHealth.ps1`, and
  `Tools\Test-ProcessLassoBootSafety.ps1`.
- [ ] Decode and triage stale/non-primary OneDrive scheduled task results,
  especially `0x8004EE04` and `267011`, before deleting or rewriting tasks.
- [ ] Decide whether Dropbox/Proton/other nonessential cloud providers should
  start after VHD mount health rather than during logon.
- [ ] Keep `UnifiUdmDriveStackStartup` disabled until OneDrive has a clean
  health window; then choose SMB+Rclone repair or an explicit rclone-only mode.
- [ ] Capture the next touchpad glitch immediately with OneDrive I/O, Process
  Lasso log lines, HID/I2C/Kernel-PnP events, top disk I/O, and sync-provider
  state.
- [ ] Harden or quarantine high-risk `~\bin` startup/network/archive/RAG scripts
  before any new boot/logon use; migrated copies now live under
  `Tools\SystemScripts`; see `Reports\bin-script-risk-review-20260430.md`.

### 2. Native-First Architecture

- [ ] Replace remaining PowerShell-only diagnostics with Rust/C# native backends
  where measurements justify the migration, starting with logs, inventory, and
  health checks.
- [ ] Define a versioned C ABI contract for all Rust DLL exports, including
  error codes, result structs, memory ownership, and free functions.
- [ ] Standardize native output schemas in a shared schema folder with explicit
  version pins.
- [ ] Centralize error translation across Rust, C#, and PowerShell so native
  failures become predictable PowerShell error records.
- [ ] Provide cancellation and timeout propagation across PowerShell, C#, Rust,
  HTTP servers, and long-running native operations.
- [ ] Add structured native logging and metrics, preferably ETW or append-only
  JSON lines with stable event names.

### 3. Acceleration And Startup Cost

- [ ] Update status/reporting and agent-facing tooling to use
  `Get-PcaiAccelerationProbe`, `Get-PcaiDirectCoreProbe`, or
  `Get-PcaiDirectTokenEstimate` when they only need scalar/status data.
- [ ] Split `PC-AI.Acceleration` into a thin loader plus nested command groups
  so import cost is not paid for every command surface.
- [ ] Benchmark import costs by file and command group, then add import-phase
  timing hooks for regression debugging.
- [ ] Extend compact/binary native result transport to full-context and
  telemetry entrypoints.
- [ ] Add true batched native file-search and directory-manifest APIs for
  multi-pattern/project-discovery workloads.
- [ ] Benchmark native search against `fd`, `rg`, and PowerShell by workload
  shape before changing preferred backend heuristics.

### 4. LLM, Evaluation, And Large Context

- [ ] Explore large-context offload for `pcai_inference` and FunctionGemma:
  KV-cache offload, chunked softmax attention, CUDA memory pool behavior, and
  GPUDirect Storage where hardware and drivers support it.
- [ ] Match FunctionGemma runtime chat-template behavior to training/evaluation
  assumptions.
- [ ] Port Python dataset/schema unit coverage into the Rust FunctionGemma
  training/runtime surface.
- [ ] Preserve evaluation baselines when prompts, routing, model defaults, or
  inference providers change.

### 5. Memory And RAG Integrations

- [ ] Integrate `rag-redis` from `W:\dropbox-local\rag-redis` with Redis
  endpoints `6379`/`6380` for tool memory and retrieval.
- [ ] Convert RAG Redis startup tooling to loud, delayed, recoverable automation
  before considering any logon/startup re-enablement.
- [ ] Evaluate SIMD distance kernels such as `simsimd` for local vector
  similarity.
- [ ] Add optional Postgres/MS SQL backed memory storage for long-term tool
  history.

### 6. UI, TUI, And Media

- [ ] Provide progress and streaming updates for long native operations.
- [ ] Expand fixtures for `AI-Media`, `pcai_media_model`, `pcai_media`,
  `PcaiNative.MediaModule`, and `Modules/PcaiMedia.psm1`.
- [ ] Add reproducible benchmarks for media decode, tensor transforms,
  attention, preprocessing, and async request lifecycle paths.
- [ ] Consolidate useful prototype-only `AI-Media/` behavior into the canonical
  native `pcai_media*` crates.

## Validation Anchors

- Boot/session tooling:
  `Invoke-Pester -Path .\Tests\Boot\PersistentVHDX.Tests.ps1,.\Tests\Boot\BootValidationTools.Tests.ps1`
- Native/Rust:
  `Import-Module CargoTools -Force; Test-BuildEnvironment -Detailed`
  followed by `Invoke-CargoWrapper check --llm-output`
- C# bridge:
  `dotnet build .\Native\PcaiNative\PcaiNative.csproj --no-restore`
- Tooling benchmarks:
  `pwsh .\Tests\Benchmarks\Invoke-PcaiToolingBenchmarks.ps1 -Suite quick`
- LLM evaluation:
  `pwsh .\Tests\Evaluation\Invoke-InferenceEvaluation.ps1 -Backend llamacpp-bin -Dataset diagnostic`
