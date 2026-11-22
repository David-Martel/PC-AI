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
