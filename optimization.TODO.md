# Optimization TODO

This backlog is based on real measurements taken while optimizing the Codex
context toolkit against `PC-AI.Acceleration` on March 6, 2026.

## 2026-03-07 Windows + Intel Optimization Program

### Completed In This Pass

- [x] Added `Get-PcaiAccelerationProbe` to `PC-AI.Common` so status-only callers
  can inspect acceleration/module/native-tool presence without importing the
  full `PC-AI.Acceleration` module.
- [x] Added `Get-PcaiDirectCoreProbe` and `Get-PcaiDirectTokenEstimate` so
  PowerShell can P/Invoke `pcai_core_lib.dll` directly for scalar/status-style
  operations without routing through the C# bridge or JSON parsing.
- [x] Added dedicated startup benchmark cases for:
  - `acceleration-import`
  - `acceleration-probe`
  - `direct-core-probe`
- [x] Simplified `PC-AI.Acceleration.psm1` script discovery to use direct file
  enumeration instead of recursive pipeline-based dot-sourcing.
- [x] Extended `Measure-CommandPerformance` and
  `Invoke-PcaiToolingBenchmarks.ps1` so tooling benchmark reports now persist:
  - working set deltas
  - private memory deltas
  - managed heap deltas
  - managed allocation deltas
- [x] Fixed a clean-process probe bug in `Get-PcaiAccelerationProbe` where an
  empty repo-root candidate could abort `direct-core-probe` before it reached
  valid local paths.
- [x] Fixed the tooling benchmark runner so it measures the repo-local imported
  modules instead of accidentally resolving identically named commands from the
  installed umbrella `PC-AI` module on `PSModulePath`.
- [x] Synced the release-copy benchmark helper under
  `Release\PowerShell\PC-AI\Modules\PC-AI.Acceleration\` with the source
  helper so fresh benchmark runs no longer fault on missing memory fields.
- [x] Validated the new probe path and benchmark harness end to end.
- [x] Restored `PC-AI.Acceleration` manifest/module export parity so fresh
  imports of `PC-AI.Acceleration.psd1` once again export
  `Measure-CommandPerformance`, native search helpers, and hardware-report
  entrypoints correctly.
- [x] Hardened `Get-PcaiSharedCache.ps1` for clean-process `Set-StrictMode`
  runs so repo-root/runtime-config discovery no longer depends on preexisting
  shell globals.
- [x] Switched the benchmark harness to use `Resolve-TestRepoRoot` as the repo
  root source of truth and to resolve `Get-PcaiCapabilities` from the
  repo-imported acceleration module.
- [x] Added compact binary result transport for the first hot bulk search
  surfaces:
  - `pcai_find_files_compact`
  - `pcai_search_content_compact`
  - `PcaiNative.PcaiSearch.FindFiles(...)`
  - `PcaiNative.PcaiSearch.SearchContent(...)`
- [x] Removed PowerShell-side JSON parsing from the `Search-ContentFast`
  native path by consuming the typed C# bridge result directly.
- [x] Hardened the compact C# parser so malformed/truncated buffers now cleanly
  fall back to the JSON path instead of throwing into the caller.
- [x] Added overflow checks to Rust compact packers so path/line offsets and
  lengths no longer silently narrow to `u32`.
- [x] Added regression/integration coverage for:
  - benchmark-runner command shadowing and memory-field persistence
  - native compact file/content search via `PC-AI.Acceleration`
- [x] Reworked native `content-search` around the `ignore` parallel visitor
  pipeline instead of a single-threaded pre-walk plus Rayon over a cloned path
  list.
- [x] Removed the extra per-file binary probe and double-open behavior from the
  hot `content-search` path.
- [x] Added a byte-oriented no-context search fast path so unmatched lines no
  longer pay UTF-8 decoding/allocation cost before regex evaluation.
- [x] Reduced `content-search` cross-thread contention by batching matches per
  worker and merging them into the shared result buffer in chunks instead of
  pushing through a global mutex on every matched file.
- [x] Moved `file-search` glob matching ahead of `metadata()` calls so
  non-matching entries skip Windows metadata I/O entirely.
- [x] Fixed the `content-search` PowerShell benchmark baseline to measure the
  full hit set instead of `Select-String -List`, which was not workload
  equivalent to the native/accelerated paths.
- [x] Added fixture-backed parity validation that compares native
  `Search-ContentFast` hits against `Select-String` for the same fixture and
  pattern.

### Validation Snapshot

Validated via:

- `Import-Module .\Modules\PC-AI.Common\PC-AI.Common.psm1 -Force; Get-PcaiAccelerationProbe -IncludeToolPaths`
- `Import-Module .\Modules\PC-AI.Common\PC-AI.Common.psm1 -Force; Get-PcaiDirectCoreProbe`
- `pwsh -NoProfile -Command "& { .\Tests\Benchmarks\Invoke-PcaiToolingBenchmarks.ps1 -SkipCapabilities -CaseId @('runtime-config') -PassThru | ConvertTo-Json -Depth 8 }"`
- `pwsh -NoProfile -Command "& { .\Tests\Benchmarks\Invoke-PcaiToolingBenchmarks.ps1 -SkipCapabilities -CaseId @('acceleration-probe','direct-core-probe') -PassThru | ConvertTo-Json -Depth 8 }"`
- `pwsh -NoProfile -Command "& { .\Tests\Benchmarks\Invoke-PcaiToolingBenchmarks.ps1 -SkipCapabilities -CaseId @('acceleration-import','acceleration-probe','direct-core-probe') -PassThru | ConvertTo-Json -Depth 8 }"`

Observed startup numbers from:

- `Reports/tooling-benchmarks/20260307_215045/`
- `Reports/tooling-benchmarks/20260307_215044/`
- `Reports/tooling-benchmarks/20260307_215540/`

Measured on this machine:

- `runtime-config`: about `19.93ms` mean
  - working set delta: about `47514` bytes mean
  - private delta: about `819` bytes mean
  - managed heap delta: about `914155` bytes mean
  - managed allocation volume: about `908240` bytes mean
- `acceleration-import`: about `9692.61ms` mean in a fresh-host benchmark
- `acceleration-probe`: about `39.50ms` mean
  - working set delta: about `1164902` bytes mean
  - private delta: about `1077248` bytes mean
  - managed heap delta: about `1337085` bytes mean
  - managed allocation volume: about `1336630` bytes mean
- `direct-core-probe`: about `44.23ms` mean
  - working set delta: about `68813` bytes mean
  - private delta: about `819` bytes mean
  - managed heap delta: about `1326298` bytes mean
  - managed allocation volume: about `1322438` bytes mean
- Latest targeted follow-up benchmark run from
  `Reports/tooling-benchmarks/20260307_233435/` measured:
  - `acceleration-import`: about `1861.89ms` mean
  - `runtime-config`: about `71.66ms` mean
  - `direct-core-probe`: about `31.03ms` mean
  - `file-search`:
    - native compact path: about `21.77ms`
    - accelerated wrapper (`Find-FilesFast`): about `120.07ms`
    - baseline PowerShell: about `1478.91ms`
  - `content-search`:
    - native compact path: about `1722.48ms`
    - accelerated wrapper (`Search-ContentFast`): about `1776.51ms`
    - baseline PowerShell: about `1714.44ms`
- Latest Rust-search follow-up benchmark run from
  `Reports/tooling-benchmarks/20260308_001158/` measured:
  - `file-search`:
    - native compact path: about `31.48ms`
    - accelerated wrapper (`Find-FilesFast`): about `194.81ms`
    - baseline PowerShell: about `2151.65ms`
  - `content-search`:
    - native compact path: about `13.33ms`
    - accelerated wrapper (`Search-ContentFast`): about `26.91ms`
    - baseline PowerShell: about `1913.69ms`
  - This moved native `content-search` from near-parity with PowerShell into a
    roughly `143.56x` speedup band on the current repo benchmark, with the
    wrapper path still about `71.11x` faster than baseline despite
    PowerShell-side object shaping costs.
- Both probe paths are still about `218x` to `245x` cheaper than a full
  `PC-AI.Acceleration` import for status-only callers.
- `direct-core-probe` and `acceleration-probe` are currently in the same
  latency band, but the direct Rust path showed materially lower working-set
  and private-memory growth in the measured fresh-host run.
- The benchmark harness now clears the module query cache before accelerated
  search cases so repeated iterations measure work instead of 15-second
  session-cache hits.

Validation note:

- The benchmark runner was previously resolving
  `Measure-CommandPerformance` from the installed umbrella `PC-AI` module.
  That produced timing-only results and silently dropped memory/allocation
  fields. The runner now invokes the repo-local imported module explicitly.
- The quick suite still reports that native `file-search` does not support the
  current `-StatsOnly` benchmark path. That is an existing benchmark-surface
  gap, not a regression introduced by the probe work.

### Research Conclusions: Windows-Only + Intel-Focused

- [x] Rust on Windows should continue to target `*-pc-windows-msvc` as the
  canonical supported toolchain target.
- [x] Rust already has first-party compiler/runtime levers worth productizing
  here before any toolchain fork:
  - PGO (`-Cprofile-generate` / `-Cprofile-use`)
  - `target-cpu=native`
  - `lto=thin` or fat LTO where runtime wins justify build cost
  - lower `codegen-units` for runtime-critical crates
- [x] Intel oneAPI is viable for native C/C++ companion code on Windows, not as
  a direct replacement toolchain for managed .NET or PowerShell code.
- [x] C# acceleration work should focus on `.NET` runtime/deployment features
  such as `PublishAot`, `PublishReadyToRun`, and `TieredPGO`, with oneAPI used
  only through native libraries where justified.
- [x] PowerShell should remain the orchestration/control plane, with heavy work
  pushed into Rust/C# or other native helpers.
- [x] Intel libraries are relevant for the media/LLM stack on Windows:
  - oneDNN is a real candidate for tensor / transformer / inference kernels
  - oneMKL is a real candidate for BLAS/LAPACK/FFT/vector-math heavy work
  - both are better fits behind FFI than as direct PowerShell dependencies

### Multi-Stage Plan

#### Stage 1: Probe Adoption

- [ ] Update status/reporting scripts and agent-facing tooling to call
  `Get-PcaiAccelerationProbe` when they only need availability, manifest, DLL,
  or Rust-tool presence.
- [ ] Update status/reporting scripts and agent-facing tooling to call
  `Get-PcaiDirectCoreProbe` or `Get-PcaiDirectTokenEstimate` when they only
  need scalar Rust DLL checks or token counts and do not need the C# bridge.
- [ ] Keep full `Get-PcaiCapabilities` / `Get-PcaiNativeStatus` only for callers
  that need live C# bridge state, module coverage, or service details.
- [ ] Add a small Pester contract test around probe shape and path resolution.

#### Stage 2: Import-Latency Reduction

- [ ] Split `PC-AI.Acceleration` into a thin loader and nested command groups.
- [ ] Benchmark import costs per imported file / command group.
- [ ] Stop dot-sourcing the full public/private tree on every import.
- [ ] Add import-phase timing hooks so cold-start regressions are easier to pin
  down.

#### Stage 2.5: Benchmark Integrity And Memory Tracking

- [x] Carry working set, private bytes, managed heap delta, and managed
  allocation volume into the tooling benchmark JSON and Markdown reports.
- [x] Bind the tooling benchmark runner to the repo-local imported modules so
  measurements reflect the code under test instead of stale installed modules.
- [x] Add a regression test that asserts the benchmark JSON report schema still
  includes all memory/allocation fields.
- [x] Add a regression test that fails if the runner resolves hot-path
  acceleration commands from outside the repo under test.
- [x] Remove benchmark-case cache pollution for accelerated search timing by
  clearing the in-module query cache between benchmark iterations.

#### Stage 3: Native Search And Data-Fabric Throughput

- [x] Add `repr(C)` stats/result structs for hot-path native calls instead of
  returning JSON strings for everything.
- [x] Add binary buffer contracts for bulk payloads:
  - pointer + length
  - explicit free function
  - versioned ownership rules
- [x] Prioritize the first ABI redesigns around the heaviest JSON-returning
  search/context surfaces:
  - `pcai_find_files` / `PcaiSearch.FindFilesJson`
  - `pcai_search_content` / `PcaiSearch.SearchContentJson`
  - `pcai_query_full_context_json` and the telemetry JSON entrypoints
- [ ] Extend the same compact/binary treatment to `pcai_query_full_context_json`
  and the telemetry JSON entrypoints.
- [ ] Keep direct PowerShell → Rust only for scalar or POD-style surfaces that
  are cheap to P/Invoke from `Add-Type`.
- [ ] Prefer direct PowerShell → Rust first for already-struct-backed exports:
  - `pcai_find_files_stats`
  - `pcai_search_content_stats`
  - `pcai_collect_directory_manifest_stats`
  - `pcai_find_duplicates_stats`
- [ ] Keep the C# bridge for complex ownership-heavy payloads until the ABI is
  stabilized and benchmarked.
- [ ] Add a true batched native file-search API.
- [ ] Expand native directory-manifest adoption in callers that still fan out
  into repeated file scans.
- [x] Fix native `file-search` / `content-search` benchmark support in the
  harness by measuring the normal return path and discarding output instead of
  relying on the old `-StatsOnly` mismatch.
- [x] Replace the native `content-search` pre-collect + Rayon pattern with a
  direct `ignore` parallel-visitor pipeline and a byte-oriented no-context fast
  path.
- [ ] Apply the same per-thread batching / reservation pattern to native
  `file-search` and directory-manifest collectors so they stop taking a shared
  mutex on every matched file or manifest entry.
- [ ] Benchmark native vs `fd` / `rg` / PowerShell by workload shape before
  changing preferred backends.
- [ ] Finish removing mutex-heavy result collection in the remaining Rust
  search/manifest hot paths by using thread-local buffers plus bounded merge
  steps.

#### Stage 4: Media-Agent Optimization

- [ ] Expand fixture coverage for `AI-Media`, `pcai_media_model`, `pcai_media`,
  `PcaiNative.MediaModule`, and `Modules/PcaiMedia.psm1`.
- [ ] Add reproducible benchmarks for decode, tensor transforms, attention, and
  async media request lifecycle paths.
- [ ] Add Rust-native microbenchmarks for media preprocessing and autoregressive
  decode so copy-elimination work is measurable before changing kernels.
- [ ] Remove avoidable tensor/vector copies in `pcai_media` preprocessing,
  sampling, and CFG batch assembly before considering lower-level compiler
  tricks.
- [ ] Consolidate prototype-only optimizations from `AI-Media/` into the
  canonical native `pcai_media*` crates.

#### Stage 5: Windows + Intel Toolchain Strategy

- [ ] Add a Rust release-profile benchmark matrix for:
  - `lto = thin`
  - lower `codegen-units`
  - PGO on the hottest DLL/CLI crates
- [ ] Treat the current release profile in `Native/pcai_core/Cargo.toml` as
  build-speed oriented, not production-performance oriented.
- [ ] Keep Windows builds centered on the Rust `*-pc-windows-msvc` path for the
  hybrid repo unless a specific native library requires a different linker.
- [ ] Treat Intel oneAPI as a native-library acceleration option for Rust/C#,
  not as a replacement managed-code compiler.
- [ ] Evaluate oneMKL / oneDNN style FFI targets only for hotspots that
  dominate benchmark results.
- [ ] Validate any oneAPI experiment with the same tooling benchmark and
  evaluation harness before broad adoption.

## Observed Findings

- Cold import of `PC-AI.Acceleration` still costs about `12.44s` in a fresh
  `pwsh -NoProfile` process, even after removing eager native/tool
  initialization from module load.
- Cold `Get-CodexContextStatus.ps1` for `C:\Users\david` dropped from about
  `14.34s` to about `6.40s` on the default fast path. The detailed
  acceleration/native probe remains intentionally slower when explicitly
  requested.
- Context-toolkit hot paths improved substantially after switching from repeated
  file probes to a cached shallow inventory:
  - `Get-ProjectType`: about `42.96s warm` to about `0.98s`
  - `Get-KeyFiles`: about `6.13s warm` to about `0.21s`
  - `Get-ClaudeContextMatches`: about `2.44s warm` to about `0.23s` on first
    pass and about `0.015s` from cache
- `Find-FilesFast` now benefits from in-memory result caching in a warmed
  session:
  - first identical query: about `2211ms`
  - second identical query: about `46ms`
- `Search-ContentFast` now benefits from in-memory result caching in a warmed
  session:
  - first identical query: about `616ms`
  - second identical query: about `13ms`

## Implemented Now

- Made `PC-AI.Acceleration` lazy on module load.
- Added bounded in-memory query caching to `Find-FilesFast` and
  `Search-ContentFast`.
- Replaced expensive `Get-Item` result materialization in `Find-FilesFast` with
  lightweight path objects.
- Enabled native depth filtering in `Find-WithPcaiNative` instead of rejecting
  `MaxDepth > 0`.
- Centralized shallow file discovery in the Codex context toolkit with a small
  LRU-style cache.
- Narrowed Claude-context matching to project-specific names instead of broad
  `*context*` scans.
- Added a session-wide shared cache helper in `PC-AI.Common` so repeated
  metadata/config lookups can be reused across `PC-AI.Common`, `PC-AI.CLI`, and
  `PC-AI.LLM` instead of each module maintaining isolated cache state.
- Added stamp-based caching for `Get-PcaiRuntimeConfig`, keyed by the runtime
  config path and file metadata rather than TTL alone.
- Reused the shared cache for `PC-AI.CLI` command-map discovery and
  `PC-AI.LLM` settings/native-Ollama CLI resolution.
- Added a native directory-manifest API to the Rust/C#/PowerShell bridge so
  search/discovery workloads can benchmark a real end-to-end native surface,
  not only file/content point queries.
- Added a dedicated tooling benchmark runner driven by
  `Config/pcai-tooling-benchmarks.json`, with benchmark reports under
  `Reports/tooling-benchmarks/`.
- Expanded coverage reporting so backend parity now tracks Rust, C# bridge, and
  PowerShell surface availability per operation rather than only tool-schema
  name coverage.
- Added `Get-PcaiAccelerationProbe` to `PC-AI.Common` so callers can check
  acceleration/module/native-tool presence without importing the full
  `PC-AI.Acceleration` module.
- Added dedicated startup benchmark cases for:
  - `acceleration-import`
  - `acceleration-probe`
- Simplified `PC-AI.Acceleration.psm1` script discovery to use direct file
  enumeration instead of recursive pipeline-based dot-sourcing.

## Next Work

- Split `PC-AI.Acceleration` into a lightweight loader plus nested command
  groups.
  The current cold import cost suggests the remaining bottleneck is module
  parse/dot-source overhead rather than native initialization.
- Benchmark import costs per file and per nested module.
  Measure `Import-Module` for the root module, then for subsets such as:
  `Initialize-RustTools.ps1`, `Initialize-PcaiNative.ps1`,
  `Find-FilesFast.ps1`, and `Search-ContentFast.ps1`.
- Stop dot-sourcing every public/private file at root import time.
  Candidates:
  - move heavy functions behind autoloaded nested modules
  - generate a compact release loader that imports only requested command
    groups
  - consider a build step that emits a flattened optimized module file
- [x] Add a lightweight capability probe API that does not require full module
  import.
  Codex-style toolchains frequently need to know whether acceleration is
  installed, not to immediately load every command.
- Adopt the lightweight probe in status/reporting paths that do not need the
  full acceleration surface.
- Extend the shared cache helpers already added in `PC-AI.Common` to other
  frequently-called PowerShell modules.
  Likely targets:
  - `PC-AI.CLI`
  - `PC-AI.LLM`
  - any path-resolution or config-loading helpers called on every command
- Extend cache invalidation keyed by dependency stamps, not only TTL.
  Good candidates:
  - directory manifest cache keyed by root path + top-level last-write summary
  - config cache keyed by file mtime + file length
  - module capability cache keyed by module path + version
- Add a native directory-manifest API to the Rust/C#/DLL layer.
  Completed for the search bridge. Next step is to adopt it in more callers
  that currently fan out into repeated file/content scans.
- Add a true batched file-search API.
  Current workflows often need `README.md`, `AGENTS.md`, `Cargo.toml`,
  `package.json`, and several language signals at once. Repeated wrapper calls
  are the wrong shape for that workload.
- Benchmark native vs `fd` vs `Get-ChildItem` by workload shape.
  Separate:
  - tiny shallow trees
  - medium repo roots
  - large deep trees
  - repeated identical queries
  Use those results to drive heuristics rather than assuming `fd` always wins.
- Improve `Find-FilesFast` path handling further.
  Potential follow-ups:
  - skip object creation entirely for callers that only need strings
  - expose a `-RawPath` or `-AsString` mode
  - avoid extra existence checks when stdout is already authoritative
- Improve `Search-ContentFast` result handling further.
  Potential follow-ups:
  - cache parsed ripgrep JSON lines directly
  - add a raw JSON mode for internal callers
  - add a path-only fast mode for index builders and discovery tasks
- Add benchmark-backed Pester coverage for performance-sensitive workflows.
  Suggested tests:
  - cold import budget
  - repeated `Find-FilesFast` cache hit path
  - repeated `Search-ContentFast` cache hit path
  - context-toolkit style shallow inventory workloads
- Extend the new tooling benchmark matrix with additional native candidates that
  still route through PowerShell or C# only.
  Highest-value targets:
  - batched multi-pattern file search
  - disk-usage/directory-size workloads
  - model discovery shallow manifest mode
  - more direct Rust-backed telemetry collectors
- Add instrumentation switches for startup analysis.
  For example:
  - `Import-Module PC-AI.Acceleration -Verbose`
  - per-phase timestamps inside the loader
  - optional profiling output for command resolution, native load, and script
    dot-sourcing

## Suggested Ownership Order

1. Cut cold import cost for `PC-AI.Acceleration`.
2. Add batched manifest/search APIs in the native layer.
3. Reuse the same cache primitives in `PC-AI.Common`, `PC-AI.CLI`, and
   `PC-AI.LLM`.
4. Add regression benchmarks so future changes do not reintroduce startup
   latency.

## Module Install And Resolution

- Standardize on `%LOCALAPPDATA%\PowerShell\Modules` as the stable developer
  install root for PC_AI-owned modules and copied external dependencies such as
  `CargoTools`.
- Keep OneDrive-backed `Documents\PowerShell\Modules` as a source tree only.
  Runtime import paths should prefer explicit manifests or the stable local
  install root before falling back to generic `PSModulePath`.
- Add a lightweight `Get-PcaiModuleStatus` diagnostic that reports:
  - active `PSModulePath`
  - user/machine `PSModulePath` values
  - resolved manifest path for `CargoTools` and `PC-AI.Acceleration`
  - whether the current process still needs a restart to pick up a persisted
    user-level `PSModulePath` change
- Consider moving `CargoTools` source out of OneDrive entirely; until then,
  install by copy rather than junction so sync state cannot perturb builds.

## CargoTools Queueing And Cache Policy

- Update deployed wrapper scripts and bootstrap entrypoints so they prefer:
  - `CARGOTOOLS_MANIFEST` / explicit manifest overrides
  - `%LOCALAPPDATA%\PowerShell\Modules\CargoTools\CargoTools.psd1`
  - repo-local manifests when intentionally running from source
  - OneDrive source only as a final fallback
- Keep `SCCACHE_DIR`, `CARGO_HOME`, and `RUSTUP_HOME` shared under
  `T:\RustCache`, but leave `CARGO_TARGET_DIR` unset by default so Cargo uses
  normal project-local `target/` layouts unless shared mode is explicitly
  requested.
- Maintain a machine-wide queue for top-level Cargo invocations and report
  queue position/depth back to callers so contention looks like backpressure,
  not random build failure.
- Add regression coverage for:
  - empty-queue status reads
  - queued multi-process entry/exit
  - wrapper resolution preferring the stable install root over OneDrive
