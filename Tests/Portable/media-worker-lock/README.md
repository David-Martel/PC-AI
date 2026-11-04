# Media worker acquisition controls

From the [repository root](../../../README.md), using PowerShell:

```powershell
# Install Rust tooling prerequisites.
rustup component add rustfmt clippy
# Build the source-bound, std-only probe with two jobs.
$env:PCAI_MEDIA_FFI_SOURCE=(Resolve-Path Native/pcai_core/pcai_media/src/ffi/mod.rs); cargo build --manifest-path Tests/Portable/media-worker-lock/Cargo.toml --locked --offline --jobs 2 --target-dir .temp/media-worker-lock/build
# Test through the same discovery entrypoint as Linux CI's Python admission job.
uv run --no-project python -m unittest discover -s Tests/Portable -p test_media_worker_lock.py -v
# Run an explicit source and retain its executable/source receipt.
uv run --no-project python Tests/Portable/test_media_worker_lock.py --source Native/pcai_core/pcai_media/src/ffi/mod.rs --target-dir .temp/media-worker-lock/run --receipt .temp/media-worker-lock/receipt.json -v
```

Windows checks start with CargoTools `Test-BuildEnvironment -Detailed`; see
[machine tooling](../../../CLAUDE.md#git-hooks-lefthook). Select native Cargo
explicitly with `--cargo <absolute-cargo.exe>` when wrappers are present.
The fixture needs Rust 1.95 or newer and a native linker. It loads no model,
media DLL, GPU, camera or network service. It reports missing prerequisites as
failures. The existing [Python admission job](../../../.github/workflows/linux-ci.yml)
discovers these tests without another workflow or test framework.

## Contract

The asynchronous producer in
[the FFI module](../../../Native/pcai_core/pcai_media/src/ffi/mod.rs)
must return when model-state acquisition yields a poisoned mutex. `PoisonError`
already holds its guard; trying to lock that mutex again deadlocks. The repair
consumes the held guard, records the original internal failure and returns.
It preserves poison and changes no C ABI.

[The build script](build.rs) extracts the actual acquisition block between
maintained producer boundaries and compiles it unchanged with minimal state
types in [the probe](src/main.rs). Boundary drift fails the build. This exercises
the acquisition/error subpath, rather than the full media crate or pipeline.
The [Python supervisor](../test_media_worker_lock.py) starts one owned child per
scenario, bounds it to three seconds, and kills/reaps that exact child on timeout.
No process-name or external-PID termination is used. Each build receipt records
source, fixture and executable hashes. Distinct source inputs require separate
target directories; automatic discovery creates a private temporary target.

The healthy running/cancelled controls preserve both request states and their
count. The poisoned control checks the entry/return markers, original failure
text, unrelated cancellation and retained poison. The pre-repair source passes
the two healthy controls and times out in the poisoned control. The repaired
source passes all three. See the [hash-bound review register](../media-worker-lock-review.json)
for the exact source and selector disposition.

The mutex remains poisoned. Existing public polling can still return `-1`.
These controls do not establish public failure delivery, state recovery, full
FFI/model execution, installed binaries or hardware acceptance.
