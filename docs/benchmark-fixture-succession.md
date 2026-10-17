# Benchmark fixture succession

Install PowerShell 7 and UV. Build the host core using the portable
bootstrap workflow; run the gate/runner/token controls documented in
[benchmark admission](benchmark-admission.md). Run the retained native-only tests
with `pwsh -NoProfile -File Tests/Portable/Test-NativeEstimatorAndStatus.ps1`.

The old schema1 gate suite accepted caller-supplied reports, short commit labels
and an AllowStaleReport bypass. Those are incompatible with current admission.
Six obsolete selectors are retired after their behavior is covered by the real
schema2 gate subprocess tests; bypass acceptance is intentionally removed.

| Former selector | Current protection |
| --- | --- |
| (a) runner failure | runner-failed and absent/partial iteration/child-exit controls |
| (b) missing baseline | AdmissionTests.test_missing_baseline_fails |
| (c) missing report | no-report mode; no report from another run accepted |
| (d) stale head with optional bypass | stale-head and stale-run always fail; bypass removed |
| (e) missing baseline case | missing-case, missing-backend, duplicate and empty reject |
| (f) valid report within threshold | fresh compatible evidence succeeds; regression fails |

[Schema2 gate selectors](../Tests/Portable/test_benchmark_admission.py) remain
functional tests of the actual gate in fresh processes. Their fixture runner is
explicitly synthetic. The source gate/runner are identical to accepted main,
preserving exact run, config/runner/host fingerprints and full case/backend checks.

The two separate native fallback and status metadata selectors were retained in
[Test-NativeEstimatorAndStatus.ps1](../Tests/Portable/Test-NativeEstimatorAndStatus.ps1),
not silently discarded with the obsolete gate fixtures. They are hardware-free
fallback/shape tests, not actual native ABI or all-export admission. Token native
parity and override loading have their own controls and qualification limits.

The old gate fixture source remains recoverable from Git history at fdee5ee.
No old latest-report/ReportPath/SkipRunner/AllowStaleReport behavior is restored.

[Retained selector review](../Tests/Portable/native-estimator-status-review.md)
records preserved assertions and current source hashes; no native deployment
qualification follows from these hardware-free fallback/status checks.

During normal integration, the existing Bash-expanded inline manifest hook
rejected /usr/bin/git.FullName instead of testing module files. The hook now
calls [a PowerShell file entrypoint](../Tools/Test-PcaiModuleManifests.ps1);
[positive/negative controls](../Tests/Portable/Test-ModuleManifestHook.ps1) require
real valid-manifest acceptance and missing-root-module rejection. The full
repository manifest set is validated separately in the portable CI matrix.
Rust formatting of the imported switch source was normalized to satisfy the
existing hook; no hook was bypassed.

[Manifest-hook selector review](../Tests/Portable/module-manifest-hook-review.md)
binds the positive and negative fixtures to the helper and hook configuration.
