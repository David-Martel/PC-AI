# Rust FunctionGemma runtime

The FunctionGemma router provides OpenAI-compatible chat completions and tool calls.

## Install

Use the repository's Rust toolchain and the Windows MSVC or Linux system
toolchain. For model inference, place the model configuration, tokenizer, chat
template, and safetensors weights in a local directory. The repository
configuration uses `Models\functiongemma-270m-it` on Windows and
`Models/functiongemma-270m-it` on Linux. To use a different directory, set
`runtime.router_model_path` in the [router configuration](../../Config/pcai-functiongemma.json).

## Build

Run from the repository root.

```sh
cargo build --locked --manifest-path Deploy/Cargo.toml -p rust-functiongemma-runtime --features model
```

`model` compiles CPU inference without the CUDA toolkit or NVIDIA runtime
libraries. `cuda` includes `model` and enables CUDA inference, NVML discovery,
and flash attention. Building that profile requires a compatible CUDA toolkit.

```sh
cargo build --locked --manifest-path Deploy/Cargo.toml -p rust-functiongemma-runtime --features cuda
```

Building with no features provides the heuristic router. The
[PowerShell build](../../Build.ps1) selects the CPU model profile for
`-Component functiongemma` and the CUDA profile when `-EnableCuda` is also set.

## Test

```sh
cargo test --locked --manifest-path Deploy/Cargo.toml -p rust-functiongemma-runtime --features model -- --test-threads=1
```

The [CPU model test](tests/model_cpu.rs) generates local safetensors weights
and exercises the chat route over loopback HTTP. The separate Hugging Face
fallback smoke requires network access and is ignored by default.

## Run

```sh
cargo run --locked --manifest-path Deploy/Cargo.toml -p rust-functiongemma-runtime --features model -- --config Config/pcai-functiongemma.json
```

Set `runtime.router_engine` to `model` to enable inference. The default bind
address is `127.0.0.1:8000`. The server exposes `GET /health`, `GET /v1/models`,
and `POST /v1/chat/completions`.

The CPU model profile reports `cpu` even if a shared configuration supplies
CUDA preferences. With the CUDA profile, `runtime.router_device` still accepts
`auto`, `cpu`, and `cuda:N`. The runtime uses
[shared device selection](../rust-functiongemma-core/src/gpu.rs) for GPU
selection and telemetry. Set `runtime.router_kv_cache` to tune KV caching.

For the exact feature sets and CPU contract, see the
[manifest](Cargo.toml) and [device selection fixture](src/gpu.rs).
