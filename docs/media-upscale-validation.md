# Optional media upscale validation

Install Rust 1.99 and a native C/C++ linker. From the repository root, build and
test the optional CPU upscale path with the checked-in lockfile:

```sh
cargo check --locked --manifest-path Native/pcai_core/Cargo.toml -p pcai-media --features upscale
cargo test --locked --manifest-path Native/pcai_core/Cargo.toml -p pcai-media --features upscale upscale::tests
cargo clippy --locked --manifest-path Native/pcai_core/Cargo.toml -p pcai-media --features upscale --lib -- -D warnings
```

The [media manifest](../Native/pcai_core/pcai_media/Cargo.toml) enables ONNX Runtime
through `upscale`. The [loader](../Native/pcai_core/pcai_media/src/upscale.rs) creates
an explicit ORT environment before building its session, as required by ort
2.0.0-rc.14. A session retains its environment internally; the local environment
handle need not outlive `load`.

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

These commands do not execute successful model inference, validate 4x image
quality, establish GPU acceleration, measure throughput or qualify production
memory limits. A separate inference run needs a known RealESRGAN ONNX model and
image fixtures, with expected output dimensions and finite pixel checks. The
loader's Rust example shows the intended [pipeline API](../Native/pcai_core/pcai_media/src/upscale.rs).

The [optional-feature workflow](../.github/workflows/media-upscale.yml) runs the
three commands above on Ubuntu and Windows with stable Rust. It triggers on media,
media-model, workspace manifest/lockfile and workflow changes. Each matrix job is
bounded to 30 minutes and two compiler jobs; a missing native binary, download
failure or linker error fails the job. Default-feature core/bootstrap checks do
not compile this module. Hosted results remain pending until that exact head runs;
local Linux execution does not establish Windows qualification.

Independent selector/source review is recorded in
[the review register](../Tests/Portable/media-upscale-review.json). Existing
unreviewed tests elsewhere remain outside that disposition.
