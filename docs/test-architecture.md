# Testing and runtime architecture

PC-AI has one Pester execution engine, explicit test cohorts and separate
runtime qualification. Choose the smallest relevant cohort during development;
run the complete acceptance cohort before treating the repository as qualified.

## Start with a plan

From the selected checkout in PowerShell 7:

```powershell
# Pure configuration: no Pester import, Pester discovery, execution or writes.
./Tests/.pester.ps1 -Profile WindowsCI -Plan
./Tests/.pester.ps1 -Profile Portable -Plan

# One real producer's tests, without discovering unrelated files.
./Tests/.pester.ps1 -Path Tests/Unit/PcaiEvaluationScores.Tests.ps1
./Tests/.pester.ps1 -Path Tests/Unit/PesterChildCoverage.Tests.ps1 -Tag CoverageReport

# A named test inside a selected file.
./Tests/.pester.ps1 -Path Tests/Unit/PesterChildCoverage.Tests.ps1 -FullName '*Saved command-gap reporting*'

# Full Windows acceptance: fixed scope, all Modules, unchanged 85% target.
./Tests/.pester.ps1 -Profile WindowsCI -CI -OutputDirectory TestResults/windows
```

`Tests/TestProfiles.psd1` owns cohort paths. The Windows workflow consumes its
same ordered 19-path WindowsCI cohort. That profile refuses narrowing filters.
`-Plan` may enumerate legacy filename globs, but never executes Pester discovery.
`-Path` selects literal test files inside Tests; `-FullName` narrows their cases.
These focused selections are diagnostics, not full acceptance.

| Entrypoint | Purpose |
|---|---|
| `Tests/.pester.ps1` | Canonical execution, planning, result validation, source-bound coverage |
| `Tests/Invoke-AllTests.ps1` | Compatibility adapter preserving its original flat four-suite All cohort; benchmark dispatch |
| `Tests/Invoke-PortableTests.ps1` | Multi-language portable checks; Pester delegates to the canonical Portable profile |
| `Tests/coverage-report.ps1` | Saved source-bound missed-command ranking, or explicitly unmeasured inventory |
| `Tests/Invoke-BenchmarkGate.ps1` | Actual-run baseline/regression admission with source/config/host/harness identities |

The two historical All meanings remain explicit: `.pester.ps1 -Type All`
selects Unit and Integration; `Invoke-AllTests.ps1 -Suite All` also includes
E2E and Functional flat-file cohorts. The latter's coverage option now measures
all Modules, replacing its misleading two-wrapper denominator. Neither is a
substitute for WindowsCI. Existing narrow/dedicated native, process-capture,
model, media and hardware qualifications retain their own entrypoints.

## Add coverage where commands are actually missed

```powershell
./Tests/coverage-report.ps1 -CoverageSnapshotPath TestResults/windows/source-bound-coverage.json -OutputPath Reports/coverage-gaps.md
```

The reporter reuses the maintained source-hash and exact command-union checks.
Only actual Passed child attachments can add hits; repeats cannot inflate the
numerator. It reads saved evidence rather than repeating a full instrumented
suite. Changed source bytes fail validation. Its table ranks missed commands
per source, and reports the unique-hit deficit against the unchanged 85% target.
Without a snapshot it labels file counts as inventory, never measured coverage.
Local checkout coverage packets capture the selected checkout's Git identity;
unversioned source capsules require separately retained source provenance.

Choose tests by source behavior and the corresponding consumer boundary:

- Pure algorithms: parameterized exact values, units, boundaries and invalid
  inputs. The metric-calculator additions execute all eight real calculators,
  including missing metadata and zero duration, with no inference/model calls.
- Orchestration: execute the real producer; replace only external HTTP, device,
  registry or process I/O, asserting exact calls, failure propagation and cleanup.
- Native bindings: compile and execute actual ABI/library contracts in admitted
  environments. Mocked transport and PowerShell fallback success are separate.
- Installation: preserve target bytes/configuration, verify source closure and
  readback, then execute a fresh process against that exact selected release.

A targeted metric-definition observation improved from 6/27 commands to 27/27
with 16 added cases (28 passing total). This does not establish repository-wide
coverage. The prior full source-bound result remains 16,032/24,309 (65.950882%),
4,631 unique hits below 85%, until a new complete run measures the revised head.
Large prior gaps include inference, peer registration, shared cache, LLM helpers
and service-host orchestration; filename similarity is not evidence of coverage.

## Keep the application layers understandable

The CLI and aggregate module compose the public PowerShell producers. Common
owns shared path/config/cache helpers; feature modules own their commands.
Expensive native operations use the existing C# ownership/resolver bridge and
Rust Core/inference/media crates. Scalar native probes and batched operations
should reuse those contracts rather than add another discovery or execution path.

This consolidation changes test infrastructure and documents ownership. It does
not introduce another daemon, service framework, dependency graph or eager
application import. Existing standalone Common/cache bootstraps remain ahead of
runtime configuration discovery, preventing aggregate reentry. Split loaders or
replace PowerShell algorithms only after startup/hot-path benchmarks demonstrate
an advantage and public consumer/parity tests protect the contract.

## Benchmark comparable work

The token/manifest timing path records useful-command time separately from
selection, validation, inner observer and retained guard-inclusive mean. CPU and
memory deltas are process-wide observations, include observer/concurrent-thread
work, and are not exclusive backend allocation or peak-memory measurements.

Baselines bind the exact benchmark runner SHA as well as suite, configuration,
host and case/backend inventory. Unbound or changed-harness baselines require
explicit reviewed regeneration; a changed timing boundary cannot manufacture a
speedup by comparing it with an older clock. Incompatible baselines are rejected
before starting an expensive benchmark, with post-run fingerprints retained to
catch drift. Synthetic timing controls prove admission, not native acceleration.

## Integration and deployment boundaries

Source, targeted tests, full coverage, compiled artifacts, installed bytes and
runtime acceptance are separate claims. A sealed source capsule must preserve
Tests, Tools, Modules, Config and their relative layout; the module installer
alone does not package configuration or qualify runtime behavior.

Install by host role using maintained copy/stage/readback tooling and explicit
release selection. Keep credentials, peer enrollment and machine policies local.
Preserve dirty WORK/Gen9 trees, loaded assemblies, cloud mounts, active agents and
hardware sessions. NUC retains its two-job cap; ARM/Linux require their own native
artifacts. Exact Windows x64 bundles do not qualify Linux or ARM. A source-only
capsule can run portable PowerShell contracts without asserting native readiness.

## Reviewed unfinished work

Historical PR #182 is closed without merge. Its preserved integration work is
tracked through draft PR #233; do not continue a nonexistent open-PR conflict
resolution. The prior full Windows gate still has one fixture setup failure:
PcaiFdReview is missing before the search producer is called. Selected discovery
and isolated passes do not establish its full-runtime cause or resolution.

Remaining owners and next actions:

| Work | Owner and next action |
|---|---|
| Full Windows acceptance and 85% | Integration lane: run the frozen current head; diagnose retained real failures; target missed producer branches |
| New benchmark timing | Benchmark lane: preserve harness identity, scopes and detecting controls; qualify real host/backend measurements separately |
| Standalone deployment | Host owners: verify signed source/config closure, role probes and fresh-process readback before default adoption |
| Optional inference/media/GPU | Backend owners: build and test enabled backends; CPU greens do not qualify them |
| Startup/import/batched search | Acceleration owner: measure cold/warm work before simplifying loaders or adding native APIs |
| Hardware/network/device acceptance | Host/operator owners: retain explicit physical/runtime admission and original evidence |