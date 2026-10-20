# Native and deployment dependency coherence

The [native Cargo workspace](../Native/pcai_core/Cargo.toml) must resolve the
optional [deployment training path](../Deploy/rust-functiongemma-train/Cargo.toml) even
when a particular build disables that feature. Updating deployment manifests can
therefore invalidate the native lockfile. Keep both workspaces' locks coherent;
do not remove `--locked` to make CI pass.

## Failure and correction

PR212's branch-only tree passed its isolated native checks. GitHub tested the merge
with newer deployment dependency updates instead. That merge and main
`6c4788114ec08929c4a94ad5e3a041417f3a9098` have the identical tree
`dc14b5ea4f427dd9e3fe2b973da01228a63e2916`. Both native workflows failed before
compilation because the native lock omitted deployment's new minijinja dependency
and retained the old training rand reference.

The [native lock correction](../Native/pcai_core/Cargo.lock) adds minijinja 3.0.0,
matching the [deployment lock](../Deploy/Cargo.lock), and uses the
already locked rand 0.10.2 for training. Existing package versions and checksums
remain intact; older minijinja references are explicitly disambiguated. This is
lock coherence, not a broad dependency upgrade.

Both [portable bootstrap](../.github/workflows/portable-bootstrap.yml) and
[USB4 lifecycle](../.github/workflows/usb4-lifecycle.yml) now include deployment
Cargo manifests and lock changes in their existing trigger filters. Four actual
deployment paths were excluded before the correction and included afterward.

## Validation

Run from the repository root using the CI toolchain and platform prerequisites:

```text
cargo test --locked --manifest-path Native/pcai_core/Cargo.toml -p pcai-tb
cargo clippy --locked --manifest-path Native/pcai_core/Cargo.toml -p pcai-tb --all-targets -- -D warnings
cargo build --locked --manifest-path Native/pcai_core/Cargo.toml -p pcai_core_lib --no-default-features
```

Pinned Rust/Cargo 1.99.1, isolated Linux qualification: 71 library and ten CLI tests
pass without ignored tests; strict Clippy and the core build pass. Actual `ip`,
OpenSSL development files and pkg-config are prerequisites. Their initial absence
was recorded and corrected only in the disposable QA image; product assertions
and readiness behavior were preserved. Final tests used read-only source,
`--locked`, offline dependency resolution and no network or host credentials.

The unchanged portable bootstrap script then loaded the freshly built native core
in a profile-free PowerShell container: fifteen Unicode cases passed in both the
parent and nested child, together with mismatch-refusal, equal-work and report-proof
controls. No native path was skipped. This debug verification artifact was not
deployed and establishes neither release performance nor acceleration gains.

Independent review verified all three executable/configuration file hashes and
the minimal lock delta. Windows hosted execution, optional training/CUDA behavior
and release deployment remain separate gates. The earlier branch-only evidence is
preserved in [the registry and boot-health review](pcai-tb-registry-health-review.md);
it does not substitute for merge-tree validation.
