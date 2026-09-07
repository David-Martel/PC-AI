# TODO

This is the active high-level backlog for `PC_AI`. Completed historical work
should be recorded in the relevant report, context snapshot, or specialized
ledger instead of left here as active work.

Last reconciled: 2026-09-07, after a CI/tooling repair pass. The previous
reconciliation was 2026-04-30.

## 2026-09-07 Reconciliation

Verified by running the checks, not by reading the ledger.

**Validation anchors** (the commands at the bottom of this file) — all
re-executed on 2026-09-07:

| Anchor | Result |
|---|---|
| Boot/session Pester (`Tests\Boot\*`) | 28 passed, 0 failed, 1 skipped (85s) |
| CargoTools `Test-BuildEnvironment -Detailed` | pass; one advisory — no Defender exclusions on `T:\RustCache` |
| C# bridge `dotnet build PcaiNative.csproj` | succeeds, 1 warning (unresolved XML cref `pcai_media_free_string`) |
| Rust MSRV | now declared: `rust-version = "1.85"` in `[workspace.package]`, verified by `cargo +1.85 check` |
| LLM evaluation | not run — needs a local GGUF model, not available unattended |

**Evidence freshness under `Reports\`** — the items in section 1 below name
four verification tools. Their newest evidence:

| Tool | Newest evidence | Age at reconciliation |
|---|---|---|
| `Test-SyncProviderHealth.ps1` | 2026-06-06 (`workstation-audit-20260606-124859\direct-pass-20260606\`) | 3 months |
| `Test-BootMountHealth.ps1` | 2026-06-06 (`workstation-audit-20260606-124859\boot-mount-health-refresh.txt`) | 3 months |
| `Test-ProcessLassoBootSafety.ps1` | 2026-06-06 (`workstation-audit-20260606-124859\process-lasso-boot-safety.txt`) | 3 months |
| `Collect-DrivePerformanceSyncRisk.ps1` | 2026-04-30 (`drive-performance-sync-risk\20260430-153629\`) | 4 months |

Corrected 2026-09-07: the Process Lasso row first read "2026-09-07
(`processlasso-governor-watchdog.json`) | current", which was wrong twice over.
That file is written by `Register-ProcessLassoGovernorWatchdog.ps1:109`, not by
`Test-ProcessLassoBootSafety.ps1`, and its `GeneratedAt` is 2026-08-19 even
though its mtime is 2026-09-07. **Use the `GeneratedAt` field inside these
artifacts, never the file mtime** -- several `Reports\` subtrees have uniform
mtimes from a bulk copy (the four `2026-06-19 22:23` directories all contain
artifacts whose internal timestamps read 2026-06-06). Getting this wrong makes
stale evidence look current, which is the one error mode this table exists to
prevent. All four tools are in fact 3-4 months stale; none is current.

`boot.TODO.md`'s 2026-06-06 block already records that
`Test-SyncProviderHealth.ps1 -SinceMinutes 60 -PassThru` passes for
OneDrive/GoogleDrive with only a stale iCloud warning, and that
`Test-BootMountHealth.ps1` passes with `PostRebootFailureCount = 0`. The first
two items in section 1 below were therefore satisfied as of 2026-06-06; they
are kept open only because the evidence has aged past a reboot cycle and
`Collect-DrivePerformanceSyncRisk.ps1` has not been re-run since April.

**Path references checked** — every `Tools\`/`Tests\`/`Modules\`/`Config\`/
`Deploy\` path named across the six ledgers was resolved with `Test-Path`
(47 paths). One is genuinely stale:

- `Tests\Evaluation\Invoke-FunctionGemmaEval.ps1`, a validation anchor in
  `llm.TODO.md`, does not exist and never has. The script is at
  `Tools\Invoke-FunctionGemmaEval.ps1` (also exported from
  `Modules\PC-AI.LLM\Public\`). Corrected in `llm.TODO.md`.

Two others resolved on closer inspection and are NOT defects:
`Tools\Collect-BinScriptRisk.ps1` is a *proposed* script inside an open
checkbox in `boot.TODO.md`, not a reference to something missing; and
`CargoTools` is present at both `%LOCALAPPDATA%\PowerShell\Modules\` and
`~\Documents\PowerShell\Modules\`, so `optimization.TODO.md`'s path is right.

**Superseded by fleet policy**: section 5's `rag-redis` integration items
(here and in `Deploy\rust-functiongemma-train\TODO.md`) conflict with the
workstation-level instruction that `rag-redis` is retired and must not be
reintroduced. Left in place but marked, pending an explicit decision.

**CI/tooling state**: five Weekly Maintenance checks and the Jules Review
workflow had been failing since at least 2026-08-24, and `release-cuda.yml` was
not valid YAML and had never run. Repaired in the 2026-09-07 pass; see
`git log --grep "never actually run"`.

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
- ~~Recent dependency-security work is merged to `main`; no open Dependabot PRs
  or alerts were present at the last validation.~~ **No longer true as of
  2026-09-07**: 6 Dependabot PRs are open (oldest 2026-06-23), GitHub reports 1
  high-severity alert on `main`, and `cargo audit` found 3 live RUSTSEC
  advisories — which had gone unseen because the Security Scan job could never
  reach its Cargo Audit step. The 3 advisories are fixed on this branch; the
  open PRs still need triage.

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

### 1a. CI And Test-Suite Debt (opened 2026-09-07)

Surfaced by the CI repair pass; each item is a real measurement, not a
suspicion.

- [ ] Reduce the Windows PowerShell 5.1 failure count. Baseline is 456 failed /
  888 total, recorded in `Tests\powershell-51-baseline.json`; the Weekly
  Maintenance job now gates against regression only. The bulk of it is 42 test
  files that cannot even load, because `PC-AI.LLM.psm1` and
  `PC-AI.Virtualization.psm1` declare `#Requires -PSEdition Core`. Decide per
  module: drop the Core-only requirement where PS7 features are not actually
  used, or tag those suites so 5.1 skips them instead of counting failures.
- [ ] Fix the pre-existing PowerShell 7 unit-test failures first documented in
  commit `15f999a` (broken test-to-module contracts in PC-AI.LLM logging and
  process-idle filtering). The WSL vsock bridge share of those is fixed
  (`Install-WSLVsockBridge.Tests.ps1` is now 20/20); the rest are not.
- [ ] Triage the 6 open Dependabot PRs (#50, #51, #52, #53, #62, #64) and the
  1 high-severity GitHub alert on `main`. #62 (quinn-proto) is already
  superseded by the lockfile update on this branch.
- [ ] Decide the fate of the two stale WIP-preservation PRs, #65
  (`preserve/local-work-20260819`) and #57 (`chore/land-wip-vllm-...`).
- [ ] `Tools\Invoke-DocPipeline.ps1` still reports the FunctionGemma router
  dataset step as an error locally: the CargoTools `cargo` shim's mandatory
  auto-fix phase invokes `cargo fix` with a stray `-` argument. CI is
  unaffected (no CargoTools on the runners), so this is a workstation-only
  break in `Build.ps1 -Component functiongemma-router-data`.
- [ ] `Portable CI (Linux)` takes far longer than the "~4-6 min" its comment
  claimed, because `cargo test --workspace` builds all seven crates. The
  60-minute cap added on 2026-09-07 is an upper bound, NOT a measurement:
  every dispatched run so far was cancelled by a newer push before it
  finished, the longest reaching 21+ minutes still inside `Rust Tests`. Let
  one run to completion, then set the cap from the real number and consider
  narrowing the workspace scope.
- [ ] **This is what actually holds the `CI Gate` red, and neither half is new.**
  Measured on run 34152812699 and confirmed identical on run 32283624611 from
  2026-08-19, before any of this session's work:
  - `Rust Tests >> Coverage Report` runs
    `cargo llvm-cov --no-default-features --features server,ffi --lib
    --fail-under-lines 70` against `pcai_inference`. Actual line coverage is
    **43.47%** (2741 regions, 1730 lines, 978 missed), so the step exits 1. The
    gap is concentrated in `http\mod.rs` (30.12% lines) and `ffi\mod.rs`
    (45.12%); `lib.rs`, `version.rs` and `backends\mod.rs` are all 91-100%.
    Either write tests for the HTTP and FFI surfaces or lower the threshold to
    a number that reflects reality --- but do not leave a threshold nobody
    intends to meet, because a permanently red gate gets ignored exactly like a
    permanently green one.
  - `PowerShell Tests >> Run Tests with Coverage` sets `Run.Exit = $true` and
    `CoveragePercentTarget = 85`, so it fails on the pre-existing PS7
    test-contract failures already tracked above.
  Every completed `ci.yml` run in the repo's history is a failure, across
  unrelated branches (dependabot, `feat/import-nukenul`), which is consistent
  with this being long-standing rather than branch-specific.
- [ ] git-guard's commit-time Rust gate can never PASS in this repo, only
  block. `qa_check_rust` in `git-guard/hooks/common/qa_gate.sh` runs
  `cd "$REPO_ROOT" && cargo fmt --all --check` (and the same for clippy), but
  PC_AI has no root `Cargo.toml` -- the workspaces are at
  `Native\pcai_core` and `Deploy\rust-functiongemma`. Cargo exits "could not
  find `Cargo.toml`", so with `.qa-gate.conf` set to `block` every commit
  touching a `.rs` file was rejected regardless of its quality, which pushes
  people toward `GIT_GUARD=0`. Both are now `off` in `.qa-gate.conf` with the
  reasoning recorded there; CI still enforces fmt and clippy properly under
  `working-directory: Native/pcai_core`. The real fix belongs upstream: teach
  `qa_check_rust` a `rust.dir` (or manifest-path) setting defaulting to
  `$REPO_ROOT`. git-guard is shared fleet infrastructure, so that change needs
  an ownership announcement before anyone makes it.
- [ ] The `Deploy\rust-functiongemma` workspace does not resolve at all.
  Its `[workspace] members` are `"../rust-functiongemma-runtime"`,
  `"../rust-functiongemma-train"` and `"../rust-functiongemma-core"`, and Cargo
  rejects members that are not hierarchically below the workspace root:
  `cargo metadata` fails outright, so `cargo fmt`/`cargo clippy`/`cargo build`
  cannot be run against that root. `Build.ps1` still points a build path at it
  (`Build.ps1:1190`). Either move the three crates under
  `Deploy\rust-functiongemma\` or drop the aggregating root and treat the
  three as independent crates. This is pre-existing and separate from the
  CargoTools `cargo fix` bug noted above.
- [ ] `release-cuda.yml` is now valid YAML but has still never executed — the
  repo has no tags at all. Cut a throwaway pre-release tag to prove the
  4-variant CUDA/CPU release path actually works end to end.

### 1b. Rust Lint Policy Backlog (opened 2026-09-07)

`[workspace.lints]` in `Native\pcai_core\Cargo.toml` had never been applied.
All seven crates declared `lints.workspace = true` *inside* their `[package]`
table; Cargo requires a top-level `[lints]` table, so it silently reported
`unused manifest key: package.lints` and dropped the entire policy. The
`clippy -D warnings` gate in `portable-ci.yml` was therefore only ever
enforcing clippy's built-in defaults.

The key is now in the right place and the gate is green, verified with a
positive control: an unused-lifetime canary fails the gate, and was reverted.
Because the policy had never run, roughly 960 accumulated warnings surfaced at
once. Rather than redden the gate or silently drop the policy, the lints are
split into an enforced set and a deferred backlog. Every count below is from a
single real clippy run, not an estimate.

Note that the enforced clippy *groups* (correctness, suspicious, style,
complexity, perf) overlap clippy's own defaults, so the practical enforcement
gained is the non-default rust lints plus `empty_drop`. The larger win is that
the policy now means what it says and the backlog is visible instead of silent.

- [ ] `clippy::pedantic` -- ~800 sites. Dominated by missing doc backticks
  (152), `must_use` candidates (105), and cast-precision warnings (~105).
  Re-enable in tranches, not all at once.
- [ ] `undocumented_unsafe_blocks` -- 59 sites needing `// SAFETY:` comments.
  Highest safety value in this backlog; do this one first.
- [ ] `missing_debug_implementations` -- 32 public types.
- [ ] `unsafe_op_in_unsafe_fn` -- 25 sites. These are edition-2024 E0133 hard
  errors, so clearing this one changes code, not just annotations.
- [ ] `clippy::cargo` -- 25 sites, all missing publish metadata
  (description/readme/keywords/categories). Low value while these crates stay
  unpublished; either add the metadata or close as won't-fix.
- [ ] `clone_on_ref_ptr` -- 21 sites.
- [ ] `redundant_imports` -- 18 sites.
- [ ] `map_err_ignore` -- 17 sites discarding the original error.
- [ ] `allow_attributes_without_reason` -- 6 sites.
- [ ] `trivial_numeric_casts` (2 sites) and `unused_result_ok` (1 site).

Two related defects were fixed in the same pass: repo `clippy.toml` declared
`msrv = "1.75.0"` while the workspace declares `rust-version = "1.85"`, and
clippy was silently using the lower one; and `clippy::string_to_string` was
still listed although upstream removed it in favour of `implicit_clone`.

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

- [ ] **DECISION NEEDED — conflicts with fleet policy.** Integrate `rag-redis`
  from `W:\dropbox-local\rag-redis` with Redis endpoints `6379`/`6380` for tool
  memory and retrieval. The workstation-level instructions state `rag-redis` is
  retired and must not be reintroduced; either drop these two items or record
  why PC_AI is an exception.
- [ ] Convert RAG Redis startup tooling to loud, delayed, recoverable automation
  before considering any logon/startup re-enablement. (Same decision as above.)
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
