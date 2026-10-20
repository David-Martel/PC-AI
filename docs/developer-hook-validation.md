# Portable developer hook validation

Install PowerShell 7 and the repository Rust toolchain. Run the same manifest
entrypoint as the normal hook, then its functional controls and Rust formatting:

```powershell
pwsh -NoProfile -File Tools/Test-PcaiModuleManifests.ps1
pwsh -NoProfile -File Tests/Portable/Test-ModuleManifestHook.ps1
cargo fmt --manifest-path Native/pcai_core/Cargo.toml --all --check
```

The previous inline PowerShell hook placed `$_.FullName` inside shell double
quotes. On Bash, `$_` expanded before PowerShell saw it, producing an unrelated
`/usr/bin/git.FullName` path. [The hook](../lefthook.yml) now invokes a
[PowerShell file](../Tools/Test-PcaiModuleManifests.ps1) with no outer-shell
variables. It enumerates actual manifests, rejects an empty set and propagates
Test-ModuleManifest failures. Fourteen repository manifests passed on MS01.

[Controls](../Tests/Portable/Test-ModuleManifestHook.ps1) require a real valid
fixture to pass and a missing-root-module fixture to fail. The test CLI returns
zero only after all assertions and cleanup succeed; otherwise the expected
negative child's native exit code would leak into GitHub's PowerShell wrapper.
A deliberately always-successful helper still fails the rejection assertion.
[Review records](../Tests/Portable/module-manifest-hook-review.md) bind the
fixtures to helper/config hashes. Hosted Windows/Ubuntu jobs run these controls
and separately validate the full repository manifest set.

The imported USB4 switch source was formatted without semantic changes so the
existing Rust formatting hook passes. No hook was disabled. Two generic
benchmark runner review bindings were refreshed after actual runner controls;
[benchmark admission](benchmark-admission.md) remains unchanged at runtime.

These changes do not qualify native-loader exports, full managed concurrency,
physical USB4 topology, route Apply or performance gains. Native-loader source
continues through its separate draft PR.
