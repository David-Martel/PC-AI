# Acceptance refresh fixture compatibility

The acceptance refresh combines the draft test runner with accepted source
without changing the WindowsCI cohort or its 85 percent all-Modules gate.
Run the complete cohort through [the canonical engine](../.pester.ps1), as
described in [the test architecture](../../docs/test-architecture.md).

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
selector/source bindings. Its disposition remains unreviewed until an
independent reviewer examines this exact change. Earlier
[endpoint](evaluation-runtime-review.json) and
[report-custody](../Boot/module-report-custody-review.json) records retain their
historical bindings; their prior passing receipts do not qualify this refresh.
