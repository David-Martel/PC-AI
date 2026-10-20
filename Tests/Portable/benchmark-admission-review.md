# PC-AI benchmark admission independent review

Scope: live `fix/benchmark-admission` edits to the gate, runner and portable functional tests. Source reviewed read-only; foreign work preserved.

## High priority

1. Runner child PowerShell commands do not inspect native exit status. The acceleration import, thunderbolt and network-discovery cases can count failed child executions as successful timing iterations. Reproduced using the real `Measure-CommandPerformance` with a child `pwsh` exiting 23: parent returned `Iterations=1`, `Mean=121.34`, `Tool=Measure-Command`, exit 0. The new iteration-count check alone cannot detect this. Require explicit nonzero-child throws and functional regression coverage before claiming measurement completeness.

2. Provenance is commit-labelled, not fully source-bound: HEAD remains unchanged by dirty source/config files and separately compiled native binaries. Record relevant source/config/artifact hashes and dirty state, or explicitly limit the claim. Output-equivalence remains explicitly false, correctly preventing the removed speedup field from implying validated parity.

## Verified improvements

The three portable admission test selectors passed on Ubuntu in 12.543 seconds. They invoke actual PowerShell subprocesses against copied gate code, exercise successful admission plus slowdown rejection, and reject runner failure, missing report, stale run/head, foreign host, missing case/backend, empty data, invalid timing, false/string completeness and duplicates. Missing baseline is rejected. These are functional gate tests, not performance or full runner qualification.

The gate no longer searches old report directories or silently approves absent evidence. Each case/backend must match the reviewed baseline; no fastest-backend substitution remains. No live routing, drivers, pool/timeouts or host configuration changed in this patch.

## Remaining qualification

Add malformed baseline shape/numeric edge cases and native artifact identity. Do not advertise timing comparisons as algorithmic equivalence, sustained acceleration or hardware readiness. Windows functional receipt remains necessary. Gate threshold remains 15 percent by default; new bound only rejects negative or above-1000 input, not a change to the default acceptance margin.

## Corrected-head re-review

The child-exit finding is resolved in all three runner child blocks. Import failures now terminate the child, and module paths are escaped. The current runner controls passed: absent/partial measurements rejected, complete accepted, each actual AST-extracted child block rejected a failing child. Against the original HEAD runner the same control exits 1 with `Iteration completeness oracle failed for 0`, establishing detection of the previous implementation.

Admission suite rerun: three selectors PASS in 25.147 seconds, including 14 rejection modes. Config and runner SHA256 checks now bind the report; docs explicitly preserve the dirty-source/native-binary limitations. No remaining high-priority issue in this bounded patch was found. Hosted Windows qualification and actual backend output parity remain separate deliverables.

Reviewed SHA256:

- gate: `40d14f762c2df9a9e3274e03a515bf53b2d4c4804e00fd136746ab0f86fd3ede`
- runner: `81690ff103a715c4a61ed734d719217ebf8ce2d814d58b491b13ed8b918c6545`
- admission tests: `25813d1b7ba441c8cad38fadbf1a0899ff5414fb37c6ff527ea73e9c75406abf`
- runner controls: `acb967f7c434d2b8bf7b2abe111fb0b357f51bdeea59912011bdc78848778bf9`
- workflow: `131e130be68e81ef307b16a732b1367b5164792b48ec629491bbd32f3f8873d1`
- admission docs: `8a145e755bcca736e30ce07dc3000522989facb62340f2da0d54eb38cebffc6f`

The new workflow keeps contents read-only, its existing 10-minute bound and Windows/Ubuntu matrix, adding pinned UV installation and functional controls. It does not change production resource limits. At review time the documentation review-record link awaits root generation of its JSON record before commit.

## Capability-only compatibility follow-up

Reviewed the two-line `-SkipPerformance` compatibility adjustment: required-case iteration is suppressed only for capability-only mode, whose report now explicitly sets `MeasurementsComplete=false`. Normal measurement admission remains strict and the gate rejects this incomplete report. The AST runner functional controls passed again. Root is separately running the full capability-only invocation; this review does not substitute for that receipt. Final runner SHA256: `735bd462eedd37f701b056ec276afed3256cdb5b5fcd209a25eb311756113abb`. This supersedes the earlier runner hash only; no new issue found.

## Python formatting follow-up

Re-reviewed reformatted Python admission tests: fixture values, 14 rejection modes and assertions are unchanged. Multiline embedded PowerShell preserves hashtable semantics, and explicit `check=False` retains subprocess.run's previous default needed to inspect expected failures. Final Python test SHA256: `d4cb5f2ad70671951bfef621dfa53cdb212ae065f81d183b5adc5673c706ce87`. This supersedes the earlier Python test hash only. No production change is introduced by this formatting update.

Final reformatted test rerun: 3 selectors PASS, 14 negative modes retained, 24.631 seconds.
