# FunctionGemma runtime validation

Install Rust stable, then run these commands from the repository root:

```sh
cargo build --locked --manifest-path Deploy/Cargo.toml -p rust-functiongemma-runtime
cargo test --locked --manifest-path Deploy/Cargo.toml -p rust-functiongemma-runtime
cargo run --locked --manifest-path Deploy/Cargo.toml -p rust-functiongemma-runtime -- --config Config/pcai-functiongemma.json
```

The [runtime manifest](../Deploy/rust-functiongemma-runtime/Cargo.toml) has empty
default features. These commands select only the runtime package in the
[Deploy workspace](../Deploy/Cargo.toml). They need no model weights or GPU.
The [binary entrypoint](../Deploy/rust-functiongemma-runtime/src/main.rs) reads the
supplied [configuration](../Config/pcai-functiongemma.json). Although that file
selects the model engine, the default-feature [chat handlers](../Deploy/rust-functiongemma-runtime/src/handlers.rs)
fall back to heuristic routing when model support is absent. See the
[runtime README](../Deploy/rust-functiongemma-runtime/README.md) for model setup
and the [repository entrypoint](../README.md) for the broader build workflow.

The [runtime workflow](../.github/workflows/functiongemma-runtime.yml) runs the
locked test command on Ubuntu and Windows for runtime source, tests, manifest,
lockfile and workflow changes. Its 25-minute job bound limits stalled or cold
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

The test command does not enable the optional model feature or select the
training/core packages. Model inference, CUDA kernels, throughput under production
load and GPU memory limits require separate validation. Existing test rationale
and hash-bound review coverage remain subject to the repository's shared
[documentation and test standard](https://github.com/David-Martel/dtm-claude/blob/main/docs/reference/DOCUMENTATION_AND_TEST_STANDARD.md).
