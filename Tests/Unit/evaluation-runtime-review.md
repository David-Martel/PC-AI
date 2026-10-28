# Evaluation consumer and endpoint regression review

The [maintained fixture](EvaluationSuiteType.Tests.ps1) exercises real evaluation
classes in fresh, profile-free PowerShell children. Its two suite cases import the
real manifest, construct a suite, add a real diagnostic dataset case, and pass the
suite through the typed result consumer. Quoting the
[OutputType name](../../Modules/PC-AI.Evaluation/Public/New-EvaluationSuite.ps1)
keeps metadata from resolving the module-private class in the calling scope.
The returned object remains a real `EvaluationSuite`.

Three endpoint cases exercise the real
[single-case producer](../../Modules/PC-AI.Evaluation/Private/Invoke-SingleTestCase.ps1)
for HTTP completion, HTTP chat fallback and Ollama HTTP fallback. Only transport
and discovery of the optional Ollama command are doubled inside the exact script
module. Each case checks the complete URI sequence and returned response/error.
Subexpressions evaluate the configured base URL before appending its endpoint;
bare interpolation otherwise sends a relative hashtable string to transport.
No real Ollama, native backend, model or network is required by these five cases.

Run with PowerShell 7 and Pester 5:

```powershell
Invoke-Pester -Path Tests/Unit/EvaluationSuiteType.Tests.ps1 -Tag Portable
```

Each child has its own module state, native-disable/memory-cache environment and
30-second deadline. The fixture leaves the parent environment intact. The optional
`EvaluationModuleRoot` parameter permits the same fixture to test an isolated
baseline without replacing maintained source. The
[review register](evaluation-runtime-review.json) binds each selector to LF-normalized
source, selector and whole-test hashes. The independent Windows fleet reviewer
retains all five cases at those exact hashes and independently observed five
passes with zero failures, skips or unrun cases. This review covers the fixture
and the four changed production lines.

At the initial main baseline, all five cases fail for the intended reasons:
two cannot resolve `EvaluationSuite` from OutputType metadata and three receive
nonabsolute request URIs. With only the metadata fix, two pass and three fail;
with all four corrected source lines, Windows and Linux pass five cases with zero
failures, skips or unrun cases. Linux used ASUS, Ubuntu 24.04.5 LTS, PowerShell
7.6.9 and Pester 5.7.1. Windows used PowerShell 7.6.7 and Pester 5.7.1. An initial
isolated Linux copy lacked a repository marker and failed root discovery; copying
the unchanged marker corrected that fixture, and the initial receipt is retained.

The separate HTTP mock smoke executed three diagnostic cases with `Errors: 0`,
child exit zero, and its owned server stopped afterward. Both per-backend and
aggregate JSON summaries were retained and the aggregate counts were explicitly
validated. This supports execution and output plumbing only; the mock scores
do not qualify model quality, GPU behavior, real inference or native performance.
External raw receipts are under the operator's fleet integration evidence root,
`r2/evaluation-runtime-verification/`: `full-baseline.clixml`,
`linux-receipt.json`, `mock-summary-validation.json`, the original mock log and
summary artifacts. Portable cohort qualification for the
[Linux CI allowlist](../linux-ci-pester.txt) is recorded separately in that root.
With the new fixture first, the existing allowlisted cohort passed 338 cases
with zero failures, 33 explicit skips and five unrun non-Portable cases. The new
five cases passed without skips. The first source archive omitted Git metadata,
so its benchmark fixture could not bind a source commit; adding isolated Git
metadata at the archive's actual baseline corrected that prerequisite. Both
incomplete-fixture receipts remain in external custody.
