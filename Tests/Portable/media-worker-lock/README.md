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

The asynchronous worker in
[the FFI module](../../../Native/pcai_core/pcai_media/src/ffi/mod.rs)
reaches the global request/state mutex only through `lock_state_recovering`,
which consumes the guard held by a poisoned mutex with `PoisonError::into_inner`
(`PoisonError` already holds its guard; locking that mutex again deadlocks).
The state lock is a short bookkeeping lock: the worker never holds it across
generation, encoding or the publish rename, so `pcai_media_poll_result` and
`pcai_media_cancel`, which take only this lock, stay responsive for every
request. Per-request cancel-versus-publish exclusion is the
`Running -> Publishing` state flip, not a held lock. The contract preserves
poison and changes no C ABI.

[The build script](build.rs) extracts `lock_state_recovering` and
`begin_async_request` verbatim from the maintained source, asserts that the
acquisition recovers with `into_inner` and does not panic, and asserts
structurally that no state lock is taken between `begin_async_request` and
`generate()` or between `begin_publish` and `finish_publish`. Boundary drift
fails the build. [The probe](src/main.rs) compiles the two extracted functions
with minimal state types. This exercises the acquisition/transition subpath,
rather than the full media crate or pipeline.
The [Python supervisor](../test_media_worker_lock.py) starts one owned child per
scenario, bounds it to three seconds, and kills/reaps that exact child on timeout.
No process-name or external-PID termination is used. Each build receipt records
source, fixture and executable hashes. Distinct source inputs require separate
target directories; automatic discovery creates a private temporary target.

The healthy controls start a pending request and refuse to start a cancelled
one, preserve both request states and their count, and require the state lock
to be free when the transition returns. The poisoned control poisons the
mutex, runs the transition under the 3 second deadline, and checks the
entry/return markers, that the pending request started, unrelated cancellation,
and retained poison. The marker names (`..._FAILURE_RECORDED` and so on) are
kept from the earlier design so the supervisor is unchanged; they now report
recovered state. A copy of the source that replaces `into_inner` with a panic,
or takes a global lock across the publish step, fails the build. See the
[hash-bound review register](../media-worker-lock-review.json) for the exact
source and selector disposition; its entries were re-bound to the current
sources by the change author and await independent re-review.

The mutex remains poisoned. Existing public polling can still return `-1`.
These controls do not establish public failure delivery, state recovery beyond
the transition, full FFI/model execution, installed binaries or hardware
acceptance.
