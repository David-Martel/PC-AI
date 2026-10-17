# Module manifest hook independent review

Install PowerShell7. No compilation is needed for this helper. Run:

```powershell
pwsh -NoProfile -File Tests/Portable/Test-ModuleManifestHook.ps1
pwsh -NoProfile -File Tools/Test-PcaiModuleManifests.ps1
```

The [production helper](../../Tools/Test-PcaiModuleManifests.ps1) moves PowerShell source out of the [lefthook command](../../lefthook.yml), preventing Bash expansion of `$_.FullName`. It retains recursive manifest validation with Stop semantics, refuses a missing/empty module inventory and validates each actual manifest. The new hook has no interpolated PowerShell variables for the outer shell to consume.

Independent execution passed both [functional selectors](Test-ModuleManifestHook.ps1): an owned valid fixture is accepted via a fresh child PowerShell `-File`, and a missing RootModule is specifically rejected with nonzero exit. Executing the exact new hook command through Bash independently validated all fourteen repository manifests. This is real command/manifest validation, not string-only inspection. Owned fixture cleanup occurs in finally.

The [register](module-manifest-hook-review.json) binds both selectors to the helper, hook and test bytes. No timeout/resource limits or production settings change. Rollback is the source commit revert, but restoring the old quoted inline command would restore the shell-expansion bug. Hosted Windows qualification remains separate.

The accompanying switch.rs changes were inspected and contain formatting/line-wrap changes only, with expressions, strings and numeric constants retained. No Rust behavioral test or HIL claim follows from that inspection; root owns the separate rustfmt result. No source files were edited by this reviewer.

## CLI exit-contract follow-up

The negative child intentionally exits nonzero; its LASTEXITCODE previously leaked through the GitHub wrapper even after all assertions passed. An explicit test-script `exit 0` now follows assertions and finally cleanup. Independent exact-style wrapper invocation passes both fixture controls plus fourteen actual manifests and exits0.

A private copy using an always-success helper fails specifically with `Broken manifest was not specifically rejected` and exits1. Thus the final success exit does not bypass failed assertions. Production helper and hook source are unchanged. Prior review records remain in history. Updated test SHA256 (LF-normalized): `c8432368f42fccd6b92b3986cdffb38fb11a459b3e1cf6d7862796945eee7277`.
