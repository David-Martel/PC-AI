# Evaluation smoke lifecycle review

The [workflow](../../.github/workflows/evaluation-smoke.yml) now passes dispatch
inputs through environment values, starts the Windows mock with a hidden window,
and avoids assigning PowerShell's read-only PID variable during cleanup.
The [three controls](EvaluationSmokeWorkflow.Tests.ps1) exercise the maintained
run blocks with process/evaluation calls mocked or inspected through the AST.
The [review register](evaluation-smoke-review.json) retains all three selectors
at their normalized source and test hashes.

Independent fresh-process controls on PowerShell 7.6.7 and Pester 5.7.1 produced
zero passes and three failures against the original workflow, then three passes
and zero failures against the corrected workflow. Neither run skipped a case.
The original cleanup fails on the read-only variable. The original evaluation
run block retains unresolved dispatch expressions and cannot be parsed as the
required source boundary. The original launcher lacks the hidden-window option.
The corrected evaluation block forwards an adversarial dataset value as one
captured argument. This control does not reproduce GitHub expression expansion.

Normalized SHA256 bindings are workflow
`624c588369a41f220680d1a71d98050575b9a6b552fdd41a1576c1d5a15ae73d`
and tests `39382eb2ca20b510da585f620193d222ea02edb9f8839550d22fd090b30b5df6`.
The register identifies the private raw control receipts by SHA256.

The actual [evaluation entry point](../Evaluation/Invoke-InferenceEvaluation.ps1)
currently exposes a separate pre-existing EvaluationSuite type-visibility
failure. These three lifecycle fixes neither resolve nor qualify that runtime
path. No inference success, model quality, installed-consumer acceptance or
aggregate coverage follows from the focused controls.
