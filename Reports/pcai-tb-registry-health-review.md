# Explicit boot-health profile and install registry review

Install/build/test/run scope: this repair changes maintained source only. It installs
no module, binary, task or mount. Build the native CLI through the repository's
normal workflow; run the focused software checks below before a separately reviewed
deployment. See [boot TODO](../boot.TODO.md), [health script/help](../Tools/Test-BootMountHealth.ps1),
[Pester cases](../Tests/Boot/BootMountHealthProfiles.Tests.ps1) and
[resolver implementation/tests](../Native/pcai_core/pcai_tb_cli/src/configure.rs).

## Behavior and bounded evidence

The PR212 parent is f82377828ee0a0237ee2254e6d381513f2de32d3. Its health script
selected RustCache whenever that VHDX existed and cloud-cache-disk did not. The
actual full-script negative control returned Pass despite all three default disks
being absent. Health expectations now come from explicit `-Profile Default`,
`RustCacheDevDrive` or `All`; absence never selects a smaller contract. Existing
RustCache-only callers must deliberately pass `-Profile RustCacheDevDrive`.
The default remains the three legacy mounts. The result reports the profile.

The resolver retains explicit-registry, repository, then default-candidate order
and existing JSON/registry validation errors. Its private candidate-list helper
allows deterministic tests without changing process cwd/home. The new test embeds
a maintained registry fixture at build time and creates private temporary files.
It covers repository precedence, missing-repository fallback, explicit precedence,
all-missing None, missing explicit error and malformed repository error.

Actual isolated Linux software QA: four Pester cases pass, zero failures/skips;
all five configure-module Rust tests pass; strict locked offline Clippy for pcai-tb
lib/tests passes. Pester executes the full script with synthetic OS inventory at
Windows command boundaries; it does not inspect or mount real disks. The original
same Pester selector fails the false-Pass assertion. The original Rust resolver
test also fails when its compiled binary runs from /tmp with no checkout mounted;
the replacement passes in the same isolated environment. This proves the ambient
fixture dependency and its removal, not Windows native deployment.

Reproduce maintained selectors from an admitted development environment:

```powershell
Invoke-Pester ./Tests/Boot/BootMountHealthProfiles.Tests.ps1 -Output Detailed
pwsh -NoProfile -File ./Tools/Test-BootMountHealth.ps1 -Profile RustCacheDevDrive -DryRun -PassThru
```

The second command queries actual host health and requires separate host admission;
it was not run live for this repair. Rust commands from Native/pcai_core:

```bash
cargo test --locked -p pcai-tb --lib configure::tests
cargo clippy --locked -p pcai-tb --lib --tests -- -D warnings
```

Recorded QA used disposable containers, read-only source, 2 CPUs, 2 GiB for
PowerShell and 4 GiB for Rust, no GPU/devices/engine socket or host credentials.
Pester 5.7.1 and Rust 1.99.0; dependency population preceded network-disabled QA.
Logs and failed controls are retained in the owner's private evidence directory
`fleet-unification/resume-review/pcai212-repair/`. No acceleration, full-workspace,
Windows boot persistence, TB link or deployed-binary qualification is claimed.

## Hash-bound independent review register

Independent Codex reviewer `benchmark_context` inspected all five selectors,
detecting controls and implementation at base `f82377828ee0a0237ee2254e6d381513f2de32d3`
plus these exact source hashes on 2026-10-19. Disposition: keep. The reviewer did
not author the repair. The false healthy result is now rejected; registry
precedence and error handling remain while ambient fixtures are removed. Hash normalization follows repository attributes: PowerShell uses CRLF and Rust uses LF, including selector slices; normalize canonical Git LF blobs to that declared EOL before hashing. Hashes bind those exact UTF-8 bytes; selector hashes bind the
comment plus It block until the next rationale (last block excludes Describe's
closing brace), or the Rust rationale/test up to the next #[test].

Health script SHA256: `585cda28900e0a81a580dacab3f61364b8beb62c6ae1107d7dda3a1013ec439d`.

Pester file SHA256: `17cdb740e3c0d65c5d6218c74f1867613040353b07ddf4b674759a52ba5f1bd5`.

Rust file SHA256: `4121232704b2c3a6ba9cd78a0f78f65450562215b01019d28dfc9bca9dea9680`.

| Selector | Selector SHA256 | Independent review |
| --- | --- | --- |
| `keeps all default disks required when only RustCache exists` | `557833e4228eeb800556fdd2a8ca518c930d1956b6f168fdef4650f5e3f6f3a4` | Keep — independent Codex benchmark_context, 2026-10-19 |
| `accepts RustCache alone only with the explicit profile` | `2963d84a64ef05a727864a5c43c1d774b7212b02157a4fa48f060964455633e2` | Keep — independent Codex benchmark_context, 2026-10-19 |
| `reports missing RustCache under the explicit profile` | `ea94bcd9035dac3002af4327e09adc1a757c824a1c9f3341aee344db48e65e9f` | Keep — independent Codex benchmark_context, 2026-10-19 |
| `retains missing default disks under All` | `aaae02a4fa30f0afbc5c8fa29253fca29759a8e41ba81cac0824d511a0e42026` | Keep — independent Codex benchmark_context, 2026-10-19 |
| `configure::tests::test_resolve_install_registry_source_hermetic_precedence_and_errors` | `e19ae0b818407515eb8dd806c7511da8a6dbabf694e303c3f27ebcb8214bf334` | Keep — independent Codex benchmark_context, 2026-10-19 |
