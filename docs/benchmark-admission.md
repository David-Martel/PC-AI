# Benchmark admission and its limits

Install PowerShell 7 and UV. Build/run the managed bridge as described in
[portable bootstrap](portable-bootstrap.md). Run bounded functional acceptance:

```powershell
pwsh -NoLogo -NoProfile -File Tests/Portable/Test-BenchmarkRunner.ps1
pwsh -NoLogo -NoProfile -File Tests/Portable/Test-TokenBenchmarkParity.ps1
uv run --no-project python -m unittest discover -s Tests/Portable -p test_benchmark_admission.py -v
```

[The gate](../Tests/Invoke-BenchmarkGate.ps1) always requires fresh evidence. Missing
baseline/report, failed child, mismatched run/commit/suite/host, input fingerprint,
missing case/backend, duplicate row, invalid time and incomplete measurement all
fail. `-FailOnRegression` controls only whether a measured slowdown fails; it does
not permit absent evidence. Reports from previous runs are never substitutes.

Baselines now use schema2 and case/backend keys. Regenerate legacy baselines
explicitly with `-SaveBaseline` after reviewing the benchmark workload and platform.
Comparison requires the same host, OS description, architecture and configuration
hash. A different source commit is expected for regression measurement; runner and
configuration fingerprints bind each current report. Run IDs are random artifact
identities, not timestamp folders. Existing historical reports remain unchanged.

[The runner](../Tests/Benchmarks/Invoke-PcaiToolingBenchmarks.ps1) requires the
requested number of successful iterations and checks each of its three explicit
child PowerShell exits. Partial/all failures are not timings. It suppresses
cross-backend speedup ratios. Token estimates now undergo an untimed exact-output
check against a Unicode-scalar reference before either backend is timed. Uncapped
directory traversal checks aggregate counts and file bytes, not entry identities.
Capped traversal requires the bounded reference's entry count, so zero or partial
work cannot become a fast sample for a nonempty corpus. Its selected identities,
category counts and bytes remain unverified because traversal order can differ.
Search still has no equivalence proof; missing or nonboolean proof remains false
and an empty report never implies equivalence. Timing rows still use declared backend labels; they do not
attest the actual loaded native implementation. Native fallback/provenance and
untimed correctness parity remain required before an acceleration claim.

This is commit-labelled evidence, not full source/artifact attestation: dirty
source, loaded native binaries, CPU policy, system load and corpus contents are not
cryptographically bound by the gate. Fingerprints do not certify those inputs.
Zero-rounded measurements are rejected rather than treated as infinite speedup;
higher timing precision/batched microbenchmarks remain necessary for very fast
operations. Full native loading, GPU qualification and USB4 HIL remain separate.

[Admission tests](../Tests/Portable/test_benchmark_admission.py) run the real gate in
fresh processes with a synthetic runner. [Runner controls](../Tests/Portable/Test-BenchmarkRunner.ps1)
extract the actual runner function/child blocks and inject measurement failures;
they do not claim hardware benchmarking. Each selector states its rationale.
[Review records](../Tests/Portable/benchmark-admission-review.json) bind independent
review and failure controls to the source. The [host matrix](../.github/workflows/portable-bootstrap.yml)
runs these on Windows and Ubuntu without host configuration/deployment changes.

## Unicode parity and reconciliation

[Token controls](../Tests/Portable/Test-TokenBenchmarkParity.ps1) exercise the actual
reference function, case preflight and report expressions. Fifteen fixtures cover
empty input, ASCII, emoji, supplementary letters/numbers, nondecimal number
categories, combining marks, whitespace, CJK and joined emoji. UTF-8 byte lengths
are counted after scalar-aware segmentation matching the Rust heuristic. This is
an estimator parity check, not a model-tokenizer accuracy claim.

Set `PCAI_TOKEN_NATIVE_PATH` to a host-built core library for actual C ABI
comparison; otherwise that part explicitly reports SKIP. Both hosted OS jobs
build the locked no-default-feature core and supply the artifact. The local
preexisting artifact is useful for an FFI smoke check but is not source-attested.
The case-wiring control injects mismatched native output and verifies neither
timer runs; equal output executes both actual command closures. Separate controls
reject missing/truthy-string proof and exercise actual empty-report expressions.
Ratios remain suppressed even for the corrected token case pending full timing
and loaded-artifact provenance qualification.

PR200's former schema1 Pester fixtures target removed stale-report/path bypasses.
Their successor is the schema2 functional gate suite above; reconciliation must
retire those fixtures rather than restore bypass parameters to satisfy old tests.

The expanded twenty-minute CI job bound accommodates the cold locked Rust build
and functional controls; it changes no production runtime timeout.

[Token selector review](../Tests/Portable/token-parity-review.md) records the
mutation controls and hash-bound dispositions.

The token controls also run as a nested child script: hosted CI exposed a
function-lookup failure hidden by direct -File/global scope. The timed closure
now captures the actual estimator function body explicitly.

[Fixture succession](benchmark-fixture-succession.md) maps retired schema1 gate
selectors and preserves the two independent native fallback/status controls.

[Developer hook validation](developer-hook-validation.md) covers portable
manifest checks and the refreshed runner review bindings.
