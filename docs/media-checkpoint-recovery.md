# Media checkpoint recovery

The Candle checkpoint recovery combines the trained-weight and metadata
preflight changes from `3920c5ab222c982af0b99c28c4e36d8f9f352405` with the
workspace on accepted main `0df84a8882ff42dcb283c0558b39d17895f6eac9`.
The six generation, hub, model and HTTP-server source files also match
committed `c12f356719fb30b5288365970488605e04cb19a5` after Git text conversion;
they are shared recovered work rather than additional unique changes beyond
that integration candidate.

Generation rejects missing, partial or misshapen required safetensors headers
before allocating the real model. Constructor metadata is collected through
scalar-backed broadcast views, with unsupported unchecked/presence-dependent
requests refused. Quantized generation still requires the non-LLM generation
weights; an unopened GGUF cannot bypass that requirement. Optional vision and
aligner readiness is distinct from generation readiness. HTTP generation
forwards request overrides and shares inference admission with understanding.

## Source disposition

| Preserved item | Reconciliation |
| --- | --- |
| `c66ad71` trained generation, complete weight loading, optional understanding, HTTP admission, CPU Flash Attention guard and bounded explicit GGUF fixture | Recover source and detecting controls; retain current-main ORT initialization and typed missing-model error test |
| `e860654` metadata preflight and partial 7B refusal | Recover six metadata controls and the exact partial-checkpoint control |
| `510d2b6` tokenizer feature/lock change | Preserve main's dependency graph and existing x64 MSVC `/MD` compatibility setting; do not replace the lockfile or disable trainer features as part of this checkpoint recovery |
| `86653c9` FFI/upscale strict lint corrections | FFI correction already present on main; retain the upscaler assertion cleanup with strict spatial/size validation |
| `3920c5a` optional-kernel performance comments | Recover qualified wording without a performance claim |
| WIP workspace Rust 1.95 floor | Already present in main; preserve `pcai_tb_cli` membership and Mistral 0.8.1 |
| WIP Process Lasso chunk parser | Already represented by main's `array_chunks` parser; no parser replacement |
| WIP generation `is_multiple_of` correction | Recover with the metadata source; also present in committed c12 |
| WIP root Clippy MSRV 1.95 | Superseded for this workspace by `Native/pcai_core/clippy.toml`; preserve root MSRV 1.85 for the separate Deploy workspace |

The upscaler requires the complete expected `[1, 3, height, width]` shape,
checked output sizing and representable image dimensions before pixel access.
Model-free tensor fixtures detect spatial mismatches and preserve RGB planes.
These checks retain the newer ONNX Runtime release and load-error behavior.

## Model-free validation

Run from `Native/pcai_core` after the CargoTools environment preflight. Use an
isolated `CARGO_TARGET_DIR` and two build jobs on this recovery lane:

```powershell
Invoke-CargoWrapper --raw test --locked --offline -j 2 -p pcai-media --no-default-features --lib metadata -- --test-threads=1
Invoke-CargoWrapper --raw test --locked --offline -j 2 -p pcai-media --no-default-features --lib generate::tests::test_load_rejects_nonempty_partial_7b_checkpoint_before_allocation -- --exact --test-threads=1
```

The `metadata` filter must execute six tests, not zero. The exact partial 7B
test creates a tiny owned archive containing one unused scalar tensor and a
7B configuration. Both ordinary and GGUF-configured loads must return
`IncompleteWeights` before weight allocation or GGUF opening. No trained model
download or GPU is needed. CPU library/server tests and strict all-target
Clippy supplement these detecting controls; optional backend compilation must
be recorded separately when it is not run.

No throughput improvement, trained generation quality, CUDA/Flash Attention,
cuDNN, ONNX Runtime execution, native FFI lifecycle or deployed bundle
qualification follows from these fixtures. Large model-construction and
explicit GGUF tests retain their admission requirements and ignored status.
Private execution receipts and source mappings live in the owned
`pcai-fleet-integration/candle` evidence packet.
