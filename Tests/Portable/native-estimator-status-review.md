# Native estimator and status selector review

Install PowerShell7 and run:

```powershell
pwsh -NoLogo -NoProfile -File Tests/Portable/Test-NativeEstimatorAndStatus.ps1
```

The [ported fixture](Test-NativeEstimatorAndStatus.ps1) invokes the actual [initializer functions](../../Modules/PC-AI.Acceleration/Private/Initialize-PcaiNative.ps1), forcing native availability false. It preserves all seven exact prior fallback input/expected-count assertions from the last two selectors in `fdee5ee:Tests/Unit/Invoke-BenchmarkGate.Tests.ps1`. Status assertions preserve RID agreement and CoreLibraryPath, WrapperPath and ArtifactHash field presence; filename/RID now compare against the actual host rather than accepting any platform regex. Both selectors passed independently on Ubuntu with no native initialization or hardware.

The six retired schema1 gate selectors are mapped in [fixture succession](../../docs/benchmark-fixture-succession.md) to the real schema2 subprocess admission tests; accepting stale reports is intentionally removed. This review does not approve restoring bypass switches. Canonical runner/gate Git blobs equal origin/main, and actual `Test-BenchmarkRunner.ps1` controls passed again. Only its two source-hash bindings were refreshed, with old records preserved in history.

The [register](native-estimator-status-review.json) binds exact test/source bytes. Test SHA256 (LF-normalized): `e14bb6aa4db0702e35ec21d88ddb1d9568dec3b774fce37c09c982058b61e9ac`. Initializer SHA256: `03864f1241a488de73987ee37648cf11e99cc585fa59827d1c86cdade30c9b12`.

This supports source integration of main with the loader branch, not general native deployment readiness. These are fallback/shape tests, not real FFI, GPU, all-export coverage, Unicode-scalar fallback correction, throughput evidence or USB4 HIL acceptance. The preserved fallback still uses its existing UTF16 regex; separate real Rust parity controls qualify the benchmark reference only. Windows hosted results remain pending. No production configuration or numerical limits were changed by the fixture port.
