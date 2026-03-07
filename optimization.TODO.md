# Optimization TODO

This backlog is based on real measurements taken while optimizing the Codex
context toolkit against `PC-AI.Acceleration` on March 6, 2026.

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
- Add a lightweight capability probe API that does not require full module
  import.
  Codex-style toolchains frequently need to know whether acceleration is
  installed, not to immediately load every command.
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
  The context toolkit benefits most from one pass that returns:
  - full path
  - relative path
  - file type
  - extension
  - mtime
  - optional depth
  This is more useful than repeated single-pattern file searches.
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
