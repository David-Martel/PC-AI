# Benchmark admission and its limits

Install PowerShell 7 and UV. Build/run the managed bridge as described in
[portable bootstrap](portable-bootstrap.md). Run bounded functional acceptance:

```powershell
pwsh -NoLogo -NoProfile -File Tests/Portable/Test-BenchmarkRunner.ps1
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
cross-backend speedup ratios because existing token and search cases do not prove
equivalent output. Timing rows still use declared backend labels; they do not
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
