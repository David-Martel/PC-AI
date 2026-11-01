# Candle archive security backport

Install: use the repository's Rust toolchain (Rust 1.95 or newer for the native workspace).
Build: `cargo check --locked --manifest-path Native/pcai_core/Cargo.toml -p pcai-media --no-default-features`.
Test: `cargo test --locked --manifest-path Tests/Portable/candle-zip-security/Cargo.toml`.
Run: `uv run --no-project python Tools/verify_candle_vendor.py` from the repository root.

The Deploy and native workspace locks previously selected ZIP 7.2.0 and 8.6.0
through three Candle versions. [GHSA-m665-v7pm-7grj](https://github.com/advisories/GHSA-m665-v7pm-7grj)
describes conflicting local/central ZIP headers and lists 9.0.5 as the first
patched version. Published Candle 0.9.2, 0.10.2 and 0.11.0 constrain older ZIP
major versions, so changing only a lockfile cannot resolve those constraints.

These directories preserve their published crate versions and licenses. In each
package only `Cargo.toml`, `src/npy.rs` and `src/pickle.rs` change: ZIP's minimum
becomes 9.1.3, fallible filename APIs propagate errors, and every archive read
explicitly enforces header consistency. ZIP 9's compatibility defaults do not
enable that strict option. PyTorch metadata enumeration validates every entry
before deciding whether to ignore non-tensor names. Lazy tensor readers recheck
headers when reopening the archive. Public Candle signatures stay unchanged.

[Provenance](candle-core-provenance.json) records original crates.io checksums,
every original/current file hash and the three [review patches](patches/).
The published library `Cargo.lock` in each package is deliberately omitted:
[Deploy](../Deploy/Cargo.lock) and [native](../Native/pcai_core/Cargo.lock) locks
are the authoritative application resolutions. No package source or license is
otherwise omitted. Exact bytes are preserved through `.gitattributes`.
The root [Ruff configuration](../pyproject.toml) preserves the gate's E/F selection and
exempts only E402/E501 in the three authenticated upstream `tests/pth.py`
fixture generators. Their published import order and long lines stay intact;
all other rules and authored Python remain checked.

The [portable contracts](../Tests/Portable/candle-zip-security/README.md) load
valid NPZ and PyTorch tensors and reject header disagreement across all three
versions. The [custody checks](../Tests/Portable/test_candle_vendor.py) detect
source, license, receipt, scope and file-set drift. The
[offline verifier](../Tools/verify_candle_vendor.py) hashes all 319 package files;
the pinned upstream checksums were independently checked during preparation.
It does not fetch crates or independently authenticate a rewritten manifest.

Strict rejection can refuse ambiguous archives previously accepted by these
loaders. These CPU software checks establish archive behavior and downstream
compilation; they do not qualify GPU execution, model quality, throughput or
already-installed native binaries. Remove these overrides only when every
locked upstream Candle version supplies equivalent patched ZIP constraints and
strict reader behavior, with these same controls passing.
