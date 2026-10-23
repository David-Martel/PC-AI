# Native estimator and status selector review

Install PowerShell7 and run:

```powershell
pwsh -NoLogo -NoProfile -File Tests/Portable/Test-NativeEstimatorAndStatus.ps1
```

The [ported fixture](Test-NativeEstimatorAndStatus.ps1) invokes the actual [initializer functions](../../Modules/PC-AI.Acceleration/Private/Initialize-PcaiNative.ps1), forcing native availability false. It preserves all seven exact prior fallback input/expected-count assertions from the last two selectors in `fdee5ee:Tests/Unit/Invoke-BenchmarkGate.Tests.ps1`. Status assertions preserve RID agreement and CoreLibraryPath, WrapperPath and ArtifactHash field presence; filename/RID now compare against the actual host rather than accepting any platform regex. Both selectors passed independently on Ubuntu with no native initialization or hardware.

The six retired schema1 gate selectors are mapped in [fixture succession](../../docs/benchmark-fixture-succession.md) to the real schema2 subprocess admission tests; accepting stale reports is intentionally removed. This review does not approve restoring bypass switches. Canonical runner/gate Git blobs equal origin/main, and actual `Test-BenchmarkRunner.ps1` controls passed again. Only its two source-hash bindings were refreshed, with old records preserved in history.

The [register](native-estimator-status-review.json) binds exact test/source bytes. Test SHA256 (LF-normalized): `e14bb6aa4db0702e35ec21d88ddb1d9568dec3b774fce37c09c982058b61e9ac`. Initializer SHA256: `03864f1241a488de73987ee37648cf11e99cc585fa59827d1c86cdade30c9b12`.

This supports source integration of main with the loader branch, not general native deployment readiness. These are fallback/shape tests, not real FFI, GPU, all-export coverage, Unicode-scalar fallback correction, throughput evidence or USB4 HIL acceptance. The preserved fallback still uses its existing UTF16 regex; separate real Rust parity controls qualify the benchmark reference only. Windows hosted results remain pending. No production configuration or numerical limits were changed by the fixture port.

## Scalar fallback candidate evidence

Candidate observation: October 23, 2026 UTC, from base commit
`2c9c25da72b9f4f529a1707c800ff5ffa00013ad`. The preceding review remains a
historical qualification of its pinned bytes. This appended evidence is an
implementation report pending independent review; it does not refresh the
JSON register's source hashes or represent approval of the candidate.

The portable fixture retains the seven prior count assertions and all status
checks, and adds sixteen scalar assertions covering emoji, astral letters and
numbers, mixed letter/number runs, combining marks, whitespace, CJK and ZWJ.
Against the unchanged initializer, all twenty-three count cases executed:
eight scalar cases failed, the other fifteen passed, and status checks passed.
After initially applying the benchmark reference's `EnumerateRunes` grouping to
the PowerShell fallback, all twenty-three counts and status checks passed. Empty
input, UTF8-byte sizing, the nonempty BOS increment and native dispatch retain
their contracts. The C# native-unavailable result and native loader admission
were not changed.

Six isolated native-override admission controls passed. The maintained token
benchmark parity fixture passed its fifteen reference cases, mismatch/proof
controls, actual fake-native case/report controls and nested invocation.
Real native parity was explicitly skipped because `PCAI_TOKEN_NATIVE_PATH`
was unset. These results establish neither native DLL execution nor deployment.

A bounded local microbenchmark extracted the complete old/new estimator
functions, forced native availability false, warmed each sample thirty times,
then alternated timing order over four rounds of three hundred calls each.
ASCII short input produced four tokens in both versions, averaging 500.00/495.36
microseconds per old/new call. ASCII medium input (240 characters) produced 101
tokens in both, averaging 2089.95/4367.26 microseconds. The medium ASCII sample
therefore shows an approximately 2.09-fold elapsed-time cost for this scalar
implementation on this shared host; no speed improvement is claimed. Emoji
and mixed astral letter/number counts changed from 3/5 to the correct 2/2, so
their old/new timings are not equivalent-work comparisons. A scratch benchmark
preflight initially used an incorrect hand-calculated ASCII expected count;
its failure was preserved before correcting the expectation from 121 to 101.

The final candidate retains the exact original regex heuristic for text with
no UTF16 surrogate code units, selected by a single surrogate-range check;
only surrogate-containing text enters the scalar loop. Five additional cases
cover a mixed BMP letter/number group, isolated high/low surrogates, reversed
surrogates and an unpaired surrogate between word runs. Malformed sequences
are decoded to replacement scalars by `EnumerateRunes`. All twenty-eight
count assertions and retained status checks passed with this compatibility
path. The same bounded measurement was then repeated once: medium ASCII
old/new means were 2126.98/1991.03 microseconds, and short ASCII means were
563.47/498.16 microseconds. The observed medium-ASCII penalty was absent in
that repeat; shared-host variation and the bounded sample do not establish
a speed improvement. Scalar-only and final-candidate measurements are retained
separately. Override admission and the maintained benchmark parity controls
were repeated against the final candidate, with real native parity still skipped.

Raw RED/GREEN logs, the unchanged source preimage, source hashes, analyzer
receipts and the reproducing microbenchmark are private in the owned
`pcai-fleet-closeout/token-repair` evidence directory. Warning/Error analysis
reported eight existing warnings, zero errors and no changed diagnostics
against the baseline; whitespace validation passed. The JSON register requires an
independent reviewer to bind the final reviewed bytes before integration.

## Independent current-source review

Codex `pcai_token_review` independently reviewed the final source on October 23,
2026 UTC and found no critical or high findings. Its fresh Windows execution
passed all twenty-eight count assertions, the retained status checks, six
override controls and the maintained reference/case/report/nested controls.
An inverse replay using the preserved unchanged initializer reproduced the
eight supplementary-scalar failures. Fifteen additional malformed/category/
threshold/UInt64 checks and one thousand representative combinations passed.
Real native comparison remained explicitly skipped; no native ABI, deployment,
hardware or performance improvement is qualified.

Three appended register entries bind the current fallback, scalar and status
selectors. The two preceding register entries and candidate narrative remain
historical. Current LF-normalized initializer SHA256 is
`ca24f1cc699cb1d825efbd400f824cbd00a07aff0feaae5c49a85276f3109bc3`;
fixture SHA256 is
`f468e0d68b2645bdbeb26520303b44e17af32f80a12c4316ef464fa7a88a00e5`.
Private independent logs, inverse replay and before/after pins are retained in
`pcai-fleet-closeout/token-review`.
