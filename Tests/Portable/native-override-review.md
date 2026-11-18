# Native override admission review

Keep all nine selector records in the [hash-bound register](native-override-review.json). No critical/high blocker remains in this bounded change. The PowerShell initializer rejects a missing authoritative override before discovery or cached success; its availability wrapper re-enters that validation. The C# resolver throws on missing/unloadable authoritative overrides, preventing CLR fallback. Valid-path hot replacement still requires a fresh process.

Actual Ubuntu validation: six isolated [initializer controls](Test-NativeOverrideAdmission.ps1) pass; an isolated copy of the original initializer fails the first control with exit1. Three [resolver selectors](test_native_override_admission.py) pass without skips using private .NET8.0.32 and the review-built host .so. The tests compile the production resolver unchanged and invoke distinct Rust sentinel libraries in fresh processes. The original resolver fails all six negative override/path subcases when a competing default library is available. The real-core control invokes its magic export successfully; this does not validate all exports, operation semantics, release performance or GPU acceleration.

Each register row records protects/detects/needs, the test and production source breadcrumb, actual validation scope and normalized source hashes. The PowerShell group exercises three overrides in initially unloaded and stale-loaded states, and separately checks the availability wrapper within each child. Positive Rust sentinels protect against unconditional loader rejection. Failure of the original implementations is evidence that these assertions detect the repaired behavior; unexecuted original PowerShell selectors receive no inferred failure claim.

The [CI workflow](../../.github/workflows/native-override-admission.yml) uses Windows and Ubuntu hosts, read-only contents permission, .NET8 setup and locked no-default-feature host-core compilation. It supplies the actual artifact to the resolver controls and fails if that artifact or its build is missing. Its 25-minute disposable-job limit and subprocess bounds affect validation execution, not production resource limits; timeout is a visible failed check. Source revert is the rollback. Native prerequisites, full-workspace lock resolution and hosted runtime differences can still fail CI; local success does not predict all hosted results.

See [deployment boundaries](../../docs/native-override-admission.md). The earlier full managed-suite environment race, required-export admission and physical USB4 topology remain separate work. No production deployment, routing, firmware, session timeout or concurrency setting changed in this slice.

## Explicit bundle pairing

The [updated pairing selectors](../Unit/PcaiNativeArtifactPairing.Tests.ps1)
initially failed four controls against the original initializer on Windows.
They detected the ambient wrapper winning both explicit layouts and the
managed load boundary, and a split result incorrectly reported as co-located.
The corrected initializer passed all 12 pairing cases and six isolated
override-admission checks on profile-free Windows PowerShell 7.6.10. A separate
fresh-process smoke test reproduced the ambient-wrapper selection with an
inert wrapper and confirmed that the corrected initializer loaded the actual
win-x64 bridge beside the selected core.

The independent review accepted the flat and matching-runtime package
locations, the wrong-runtime negative case, and the existing core-only
fallback. The [review register](native-override-review.json) binds the
reviewed selector spans and initializer to their SHA-256 hashes.
