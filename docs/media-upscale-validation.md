# Optional media upscale validation

Install Rust 1.99 and a native C/C++ linker. From the repository root, build and
test the optional CPU upscale path with the checked-in lockfile:

```sh
cargo check --locked --manifest-path Native/pcai_core/Cargo.toml -p pcai-media --features upscale
cargo test --locked --manifest-path Native/pcai_core/Cargo.toml -p pcai-media --features upscale upscale::tests
cargo test --locked --manifest-path Native/pcai_core/Cargo.toml -p pcai-media --features upscale --test upscale_cpu_inference
cargo clippy --locked --manifest-path Native/pcai_core/Cargo.toml -p pcai-media --features upscale --lib -- -D warnings
```

The [media manifest](../Native/pcai_core/pcai_media/Cargo.toml) enables ONNX Runtime
through `upscale`. The [loader](../Native/pcai_core/pcai_media/src/upscale.rs) creates
an explicit ORT environment before building its session with ort 2.0.0-rc.15.
The session retains the environment for its lifetime.

ORT builder errors retain the recoverable session builder. The loader converts
those errors to ordinary `ort::Error` before attaching `anyhow` context. This
preserves the ORT error while releasing the recoverable builder, whose error
otherwise fails `anyhow`'s Send/Sync bounds. The loader still rejects configuration
and model errors rather than silently continuing with different settings.

The [FFI error store](../Native/pcai_core/pcai_media/src/ffi/mod.rs) consumes owned
error messages. Upscale failure branches pass formatted strings directly into
that store, preserving diagnostic bytes and the existing thread-local CString
lifetime while avoiding unnecessary borrowed-string copies. Existing FFI tests
cover error text/code retrieval, null state, clearing and complete cause chains.

The focused tests check RGB tensor shape and normalized channel values, the
fixed scale factor, and a missing model path. The missing-model test creates its
own temporary directory and requires the load error to name the requested path
and retain an underlying cause. It exercises environment/session initialization
and file-load error reporting without a model fixture or GPU. Record independent,
hash-bound review of that selector before merging, using the repository's
existing review process.

ONNX Runtime's native library must be available to build and link these tests.
The manifest's `download-binaries` feature normally acquires it during the build.
Offline execution needs a matching pre-provisioned ORT library via
`ORT_LIB_LOCATION`; downloading or resolving dependencies alone is not a test.

The [CPU integration test](../Native/pcai_core/pcai_media/tests/upscale_cpu_inference.rs)
loads a small [ONNX Tile graph](../Native/pcai_core/pcai_media/tests/fixtures/tile-4x.onnx.pb)
through the public upscale pipeline. It checks the dimensions and every RGB pixel
of a non-square input after a real ORT session run. Use a known RealESRGAN model
for image quality, GPU, throughput and production memory qualification.

The [fixture generator](../Native/pcai_core/pcai_media/tests/fixtures/generate_tile_4x.py)
uses Python's standard library and the ONNX protobuf field layout. Run the
generator with a scratch output path to reproduce the fixture. Use `--check`
to compare the checked-in 180-byte file without writing it. Its SHA-256 is
`206823dd028a1db7966ed003d472d45b98e6bc11df092448c1260457fa13f661`.
[ONNX schema](https://github.com/onnx/onnx/blob/v1.13.1/onnx/onnx.proto)

```sh
uv run --no-project python Native/pcai_core/pcai_media/tests/fixtures/generate_tile_4x.py Native/pcai_core/pcai_media/tests/fixtures/tile-4x.onnx.pb --check
```

The [optional-feature workflow](../.github/workflows/media-upscale.yml) runs these
commands on Ubuntu and Windows with stable Rust. It triggers on Cargo config, media,
media-model, workspace manifest/lockfile and workflow changes. Each matrix job is
bounded to 30 minutes and two compiler jobs; a missing native binary, download
failure or linker error fails the job. Default-feature core/bootstrap checks do
not compile this module. Hosted results remain pending until that exact head runs;
local Linux execution does not establish Windows qualification.

Independent selector/source review is recorded in
[the review register](../Tests/Portable/media-upscale-review.json). The
[CPU graph review entry](../Tests/Portable/media-upscale-cpu-review.json) binds the
new selector, fixture and ORT dependency graph.

## Windows C++ runtime policy

The [root Cargo configuration](../.cargo/config.toml) supplies target-specific
`CXXFLAGS_x86_64_pc_windows_msvc=/MD` unless the caller already sets that variable.
The locked esaxx-rs dependency requests a static CRT unconditionally; downloaded
ORT and the default Rust MSVC target use the dynamic CRT. cc-rs applies environment
flags after dependency defaults, so the resulting C++ objects use the matching
dynamic runtime. This changes compilation, rather than suppressing linker mismatch
diagnostics. Linux compilers do not consume this Windows-target variable.

Invoke the documented commands from the repository root so Cargo discovers this
configuration. Callers overriding the target variable must preserve compatible
runtime flags; deliberately static-CRT builds need separately compatible native
dependencies and are not qualified by this CPU matrix. The initial Windows gate
failed with LNK2038 (esaxx MT versus ORT MD); keep that failed run as evidence and
require the corrected exact-head linked tests, not only `cargo check`.
