# Token benchmark parity independent review

Install PowerShell7, UV and the host native core. Build the locked core without default accelerators as done in the [host workflow](../../.github/workflows/portable-bootstrap.yml). Run:

```powershell
$env:PCAI_TOKEN_NATIVE_PATH = '<absolute host pcai_core_lib artifact>'
pwsh -NoLogo -NoProfile -File Tests/Portable/Test-TokenBenchmarkParity.ps1
uv run --no-project python -m unittest discover -s Tests/Portable -p test_benchmark_admission.py -v
```

The [runner](../Benchmarks/Invoke-PcaiToolingBenchmarks.ps1) now uses Unicode scalars with L/N and whitespace grouping and UTF8 piece sizes, matching the Rust tokenizer contract on the reviewed corpus. Native output is compared before either timer. Missing/nonboolean proof and empty reports remain unverified; speedup ratios remain suppressed. Strict unique-run admission is retained.

Independent Ubuntu validation passed fifteen reference and fifteen real native FFI cases, mismatch refusal, strict boolean proof, actual token-case timer wiring, and actual report AST unknown/empty controls. The real FFI artifact was the existing `Native/pcai_core/target/debug/libpcai_core_lib.so`; this is runtime behavior evidence, not a fresh-build attestation. The strict gate's three functional selectors and fourteen negative modes passed in 14.174 seconds.

Two private source mutants failed as intended: removing the pre-timing parity call fails `Mismatched work reached timer`; removing Rune.IsNumber fails `Reference mismatch number-category-group: 3`. Review discovered the original twelve-case corpus failed to detect numeric grouping because its three-rune examples had coincident token totals. The added two-Roman-number, ASCII digit and mixed-letter/digit examples correct that gap. No production source was modified to run the mutants.

Runner SHA256 (LF-normalized): `75fa1e51be58544b2fb53481572b4ab87be25d61a69c1ec549c26e6cd0c60c3d`.
Test SHA256 (LF-normalized): `7bab85d6136f57d5e054985c9db32177b81d9f9981186526e0f7c29e6e18ce6b`.
The [selector register](token-parity-review.json) binds this review to exact source/test bytes. Later changes require re-review.

The CI job budget increases from ten to twenty minutes to accommodate an added cold locked native-core build on Windows/Ubuntu, not a production timeout change. Documentation records the rationale; hosted timing and cross-platform receipts remain pending. Code rollback is the commit revert. No host network or driver configuration changes occur. Workflow input coverage should include native tokenizer/lockfile changes so this validation runs when those inputs change.

These controls prove token output behavior on the corpus and fail-closed admission, not performance gains, actual timed backend provenance, arbitrary-Unicode equivalence, corpus/artifact attestation or USB4 HIL readiness. The timed wrapper may fall back; that remains a separate qualification.

## Closure-scope follow-up

Independent nested invocation reproduced the original79efe40 defect: the dynamic closure could not find `Get-ReferenceTokenEstimate`. Initially this was a nonterminating error that the timer fixture discarded, allowing misleading PASS output and exit0. The strengthened fixture now captures error records and throws, while the production closure captures the actual function body and sets Stop locally.

Current nested invocation passes all fifteen real FFI/reference cases, actual case/report controls, and its nested-child control. An exact private copy of79efe40 now fails exit1 with `Timed command produced an error record`. No production source was mutated for this control. Existing review history is retained in the register.

New LF-normalized runner SHA256: `ad6aeb27da85ab66fd89205b8e2a8ff16f5834163fde6f8831cf1a8da11f5eed`.
New test SHA256: `a9a49673590c5e8d65cdff69016effb1823948eb7d73ff068936f09442aea47b`.

The earlier review's direct-file invocation alone did not qualify nested module scope. This follow-up closes that test gap; it does not imply any acceleration benefit.
