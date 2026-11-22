# FunctionGemma runtime and core validation

Install Rust stable, then run these commands from the repository root:

```sh
cargo build --locked --manifest-path Deploy/Cargo.toml -p rust-functiongemma-runtime
cargo test --locked --manifest-path Deploy/Cargo.toml -p rust-functiongemma-runtime
cargo test --locked --manifest-path Deploy/Cargo.toml -p rust-functiongemma-core --no-default-features
cargo run --locked --manifest-path Deploy/Cargo.toml -p rust-functiongemma-runtime -- --config Config/pcai-functiongemma.json
```

The [runtime manifest](../Deploy/rust-functiongemma-runtime/Cargo.toml) has empty
default features. The [core manifest](../Deploy/rust-functiongemma-core/Cargo.toml)
defaults to CUDA and FlashAttention, so the core command disables defaults.
Both commands use the [Deploy workspace](../Deploy/Cargo.toml) and run without
model weights or a GPU.
The [binary entrypoint](../Deploy/rust-functiongemma-runtime/src/main.rs) reads the
supplied [configuration](../Config/pcai-functiongemma.json). Although that file
selects the model engine, the default-feature [chat handlers](../Deploy/rust-functiongemma-runtime/src/handlers.rs)
fall back to heuristic routing when model support is absent. See the
[runtime README](../Deploy/rust-functiongemma-runtime/README.md) for model setup
and the [repository entrypoint](../README.md) for the broader build workflow.

The [FunctionGemma workflow](../.github/workflows/functiongemma-runtime.yml) runs
both locked test commands on Ubuntu and Windows for runtime or core changes,
including changes limited to the core or its Candle patch. Its 25-minute job bound limits stalled or cold
builds; reaching it fails the job and requires investigation. Revert the workflow
change to roll back this gate. Watch each matrix job's result and duration;
Linux execution alone does not establish Windows qualification.

The [HTTP tests](../Deploy/rust-functiongemma-runtime/tests/http_router.rs) exercise
JSON health/model metadata, ordinary and tool-call chat responses, streamed
completion framing and queue-full rejection over ephemeral loopback sockets.
Cargo also compiles the [authentication documentation
example](../Deploy/rust-functiongemma-runtime/src/auth.rs), which uses the public
[router factory](../Deploy/rust-functiongemma-runtime/src/lib.rs). This catches
examples that accidentally depend on private middleware APIs. The existing HTTP
tests use the default unauthenticated configuration; this gate does not establish
coverage of configured bearer-token rejection or acceptance.

### Health telemetry latency

The [health collector](../Deploy/rust-functiongemma-runtime/src/metrics.rs)
refreshes RAM and router-process counters every five seconds. A cold
`cpu_usage` is `null` until an interval is available. CPU percentages use one
logical core as 100%. If sampling fails initially, the response omits
`metrics`. A later failure keeps the last good snapshot until the next
scheduled refresh. The four `*_mb` values retain their keys and now represent
MiB. Older builds reported KiB under those keys.

Use the [loopback benchmark](../Tests/Benchmarks/measure_functiongemma_health.py)
with Python 3.11 or later. Stage the locked baseline and candidate binaries
separately and run the same command with each path.

```sh
python Tests/Benchmarks/measure_functiongemma_health.py <staged-runtime> --trials 3
```

Each trial launches a fresh CPU runtime and measures the first health request,
a cached request, and a request after the five-second refresh interval. The
receipt records each latency and the binary SHA-256. Windows loopback medians
for three debug launches per build against baseline `244fe612` were 1,285 ms
to 35 ms on first access and 1,352 ms to 53 ms on refresh. Cached responses
varied between runs, with medians of 2.25 ms and 23.48 ms in this sample.

The core selection exercises the [chat-template fixtures](../Deploy/rust-functiongemma-core/src/chat_template.rs),
[safetensors utilities](../Deploy/rust-functiongemma-core/src/safetensors_utils.rs)
and other CPU-only core tests. The runtime still uses its default heuristic
configuration. In a core build without CUDA, [runtime memory queries](../Deploy/rust-functiongemma-core/src/gpu.rs)
report metrics and pool controls as unavailable. The [model](../Deploy/rust-functiongemma-core/src/model.rs)
accepts full-precision weights and rejects an explicit 4-bit QLoRA request.
The trainer, model-backed HTTP integration, CUDA kernels,
production throughput and GPU memory limits require separate validation. Existing test rationale
and hash-bound review coverage remain subject to the repository's shared
[documentation and test standard](https://github.com/David-Martel/dtm-claude/blob/main/docs/reference/DOCUMENTATION_AND_TEST_STANDARD.md).
