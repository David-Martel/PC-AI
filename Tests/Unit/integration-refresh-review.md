# Acceptance refresh fixture compatibility

The acceptance refresh combines the draft test runner with accepted source
without changing the WindowsCI cohort or its 85 percent all-Modules gate.
Run the complete cohort through [the canonical engine](../.pester.ps1), as
described in [the test architecture](../../docs/test-architecture.md).

The `fix/fleet-acceptance-refresh` candidate retains integration source
`90d47fda195bc5ad27604f23cf4896ff9db1067c` and accepted main
`e7aa85df7e329c7cc0ec73f7bcf9a9d715b0a8c2`. Its focused Windows rerun passed
244 cases with zero failures, skips or unrun cases in 86.09 seconds.

The complete WindowsCI run at signed source
`6bf800ffb552811c2682b67d7f2a9306e5a244af` executed all 4,180 discovered cases:
4,140 passed, zero failed, 40 retained prerequisite skips and zero unrun cases.
Execution took 787.04 seconds. All 3,145 captured source inputs retained their
hashes and HEAD remained unchanged. The earlier holder-fixture and syntax
failures did not reproduce in this complete run.

Acceptance remains **HOLD**. Parent-only Pester coverage was 15,326/24,318
commands (63.02 percent). The source-bound union with successful child evidence
was 16,008/24,318 (65.8277818899581 percent), so the unchanged 85 percent gate
failed. Reaching that target requires another 4,663 unique command hits. The
older 66.0126 percent receipt remains historical. Saved gap ranking identifies
inference, peer registration, shared cache and LLM helpers among the largest
remaining missed-command groups; filename counts do not establish coverage.

The first hidden-host invocation omitted `-NonInteractive` and stopped
advancing at the missing-mandatory-Action refusal fixture. Only that owned
child was terminated, and its incomplete evidence was retained. The completed
run used `-NonInteractive`, retained the same source, cohort and per-test
timeouts, and passed that suite. Both invocation records and the complete
source-bound result remain under the operator's
`pcai-fleet-integration/r3/integration` evidence directory.

The [evaluation endpoint fixture](EvaluationSuiteType.Tests.ps1) retains its
three real-consumer endpoint assertions. Its discovery double now accepts the
command-type filter used by timeout introspection, and its transport double
accepts the operation timeout supported by current PowerShell. These changes
preserve the production timeout boundary while keeping transport synthetic.

The [module publication fixture](../Boot/DevModuleDeployment.Tests.ps1) now
creates its second-target collision after Alpha has actually been published.
It proves the new Alpha bytes were active at that boundary, then verifies the
complete old tree, unrelated target file, retained staged reports and a
RolledBack receipt with no secondary errors. This replaces assumptions about
the retired staging layout and a collision that the current installer rejects
during preflight. The junction case retains its refusal and outside-target
byte checks and expects the current earlier linked-payload guard.

The initial combined run executed 244 cases: 239 passed and five failed. The
three endpoint failures rejected the old double's unsupported CommandType
argument. The two publication failures exposed the old error/layout
assumptions. The refresh does not weaken a production assertion, skip a case,
increase a deadline or lower coverage. Raw red and subsequent run evidence is
retained under the operator's `pcai-fleet-integration/r3/integration` directory.

The [refresh register](integration-refresh-review.json) records the current
selector/source bindings and the independent KEEP disposition for all three
refreshed groups at `6bf800f`. Disabling only prior-installation recovery made
the actual rollback selector fail after its passing baseline. A copied endpoint
control passed all three cases with inbox Utility cmdlet metadata available;
restoring bare hashtable URL interpolation then failed both HTTP cases while
Ollama remained passing. The endpoint review covers the canonical module tree
and records that metadata prerequisite rather than claiming arbitrary copied
module layouts. Earlier
[endpoint](evaluation-runtime-review.json) and
[report-custody](../Boot/module-report-custody-review.json) records retain their
historical bindings; their prior passing receipts do not qualify this refresh.

## Required CI artifacts

Post-run review found that a multi-path upload can succeed when JSON sidecars
exist but the coverage XML is absent. The always-running presence check in
[the workflow](../../.github/workflows/ci.yml) now requires all three real
files before the blocking upload. The
[actual-step fixture](PowerShellCoverageArtifacts.Tests.ps1) passes five
controls: complete packet, each missing member with the other two present,
and a directory substituted for XML. Before the guard existed, the fixture
failed its step-discovery setup. All 17 workflows and 108 embedded PowerShell
steps pass the narrow syntax check after the fix. The
[artifact review register](coverage-artifact-review.json) records this
separate follow-up; it does not relabel the full run as validation of later
workflow edits or as 85 percent acceptance.
