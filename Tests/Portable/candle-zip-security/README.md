# Candle ZIP archive contracts

Install: Rust and the repository's [audited vendor sources](../../../vendor/README.md).
Build: `cargo test --locked --manifest-path Tests/Portable/candle-zip-security/Cargo.toml --no-run`.
Test/run: `cargo test --locked --manifest-path Tests/Portable/candle-zip-security/Cargo.toml`.
From the repository root, also run `uv run --no-project python -m unittest discover -s Tests/Portable -p test_candle_vendor.py -v`.

The [test source](src/lib.rs) runs twelve named contracts against each locked
Candle version: 0.9.2, 0.10.2 and 0.11.0. Valid eager, named and lazy NPZ reads
retain names, shapes, dtype and numeric data. A real published PyTorch fixture
retains its metadata and 2-by-4 integer tensor. Non-tensor ZIP entries remain
valid empty PyTorch metadata. The negative cases change only a filename in one
local or central header, including mutations after a valid lazy index exists.
Both eager and reopened readers must reject conflicting headers.

`fixtures/test.pt` is a byte-identical copy of the MIT-licensed Candle 0.11.0
published `tests/test.pt`, also present in
[the audited package](../../../vendor/candle-core-0.11.0/tests/test.pt).
It requires neither Python/PyTorch installation nor a downloaded model. Tests
use owned temporary files, CPU tensors and unchanged runtime reader APIs.

The independent [review register](review.json) binds selectors and exact source
hashes. Windows preparation ran the identical 36 cases against untouched,
checksum-verified published sources: fifteen valid cases passed and all
twenty-one header-rejection cases failed. The strict backport passed all 36.
These are software archive controls; no GPU, performance or deployment claim
follows from them. [Hosted execution](../../../.github/workflows/portable-bootstrap.yml)
also verifies complete source custody before running this contract.
