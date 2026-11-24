# pcai_core fuzzing

Coverage-guided fuzzing for parsers that read untrusted files. The crate is outside the
`pcai_core` workspace (its own `[workspace]` table) because it needs nightly and libFuzzer.
CI does not build it; run it by hand after changing a reader.

## Install, build, test, run

```powershell
rustup toolchain install nightly
cargo install cargo-fuzz --locked          # 0.13.2 was used

cd Native\pcai_core
cargo +nightly fuzz build gguf_meta        # build
# MSVC: libFuzzer links the ASan runtime; put its directory on PATH first, for example
#   $env:PATH += ";C:\Program Files\Microsoft Visual Studio\18\Insiders\VC\Tools\MSVC\<ver>\bin\Hostx64\x64"
# (without it the target exits with 0xc0000135, STATUS_DLL_NOT_FOUND)
cargo +nightly fuzz run gguf_meta fuzz\corpus\gguf_meta fuzz\seeds\gguf_meta -- -max_total_time=60 -max_len=4096
```

Pass `fuzz\corpus\gguf_meta` (ignored by git) first: libFuzzer writes new inputs to the first
directory, so putting `fuzz\seeds\gguf_meta` first fills the committed seeds with generated files.

## Targets

| Target | Code under test | Property |
|---|---|---|
| `gguf_meta` | `read_gguf_meta_from`, `read_gguf_layout_from` (metadata plus the tensor-info section) and `validate_gguf_layout_from` (declared offsets, alignment, file extent, for four file lengths) in [`gguf.rs`](../pcai_core_lib/src/preflight/gguf.rs) | never panics, overflows the stack or hangs; a successful parse also gives a memory estimate; a layout the validator accepts is one the size reader accepts, with the same header length |

The target includes `gguf.rs` with `#[path]` instead of depending on `pcai_core_lib`: that crate
is a `cdylib`, and libFuzzer's instrumentation then asks the DLL link for a `main` symbol
(`LNK2001` on MSVC). `gguf.rs` needs only `anyhow`, `serde` and `thiserror`.

The deterministic regressions for the same reader live in
[`pcai_core_lib/tests/gguf_hardening.rs`](../pcai_core_lib/tests/gguf_hardening.rs) (metadata) and
[`pcai_core_lib/tests/gguf_nvfp4.rs`](../pcai_core_lib/tests/gguf_nvfp4.rs) (tensor infos and sizes) and
[`pcai_core_lib/tests/gguf_layout.rs`](../pcai_core_lib/tests/gguf_layout.rs) (offsets, alignment, extent);
a fuzzer finding should become a test there before the fix. The seed `nvfp4_tensors` reaches the
tensor-info parser; a test in `gguf_nvfp4.rs` keeps it equal to that file's fixture.
