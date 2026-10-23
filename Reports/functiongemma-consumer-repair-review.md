# FunctionGemma consumer repair review

The exported `Invoke-FunctionGemmaDataset` wrapper walked two parents from
`Modules/PC-AI.LLM/Public`, selecting `Modules/Tools` instead of repository
`Tools`. The repair walks three parents. Parameters, missing-helper behavior,
exception propagation and `PSBoundParameters` forwarding remain unchanged.

Eight independently reviewed contracts now run through normal
`Tests/Unit` discovery. They copy the exact selected wrapper into an owned
TestDrive layout with an inert parameter ledger and a throwing wrong-path
helper. The first control reproduces the captured predecessor's incorrect
route. Remaining controls verify the correct route, omitted defaults, both
native switches, every path/count parameter, explicit false switches, missing
helper refusal and exception propagation.

The immutable predecessor fixture has a path-specific `-text -eol` rule and
`whitespace=cr-at-eol` to recognize its preserved CRLF without changing bytes.
Its raw SHA256 is
`16D2A190301204A590193021917685CE195E1CC35CEB085C0F78FBAB356BD957`.
Ordinary PowerShell files retain the repository's normal text handling.
The test defaults to the actual canonical wrapper; it has no dependency on a
private preparation packet. Setup and teardown bind the selected source,
fixture and test. Endpoint hashes cannot exclude a transient edit later restored.

## Actual validation

The private candidate passed eight controls. The maintained test then passed
eight controls through `Run.Path`, with no source/container override:
zero failures, skips, NotRun, failed blocks or failed containers, and stable
source hashes. The maintained Pester run took 6.31 seconds. Its owned fresh
process exited zero; the parent retained the original process handle and
confirmed closure and disposal. Production PSScriptAnalyzer reported no
warnings or errors. These timings describe validation, not a speedup benchmark.

[The review registry](../Tests/Fixtures/FunctionGemmaDatasetWrapper.TestReview.json)
binds all eight complete assertion commands, their rationales, source hashes
and actual result. The canonical result SHA256 is
`088656756FD85A611F48DAD9EEE3928304F1E514E4B4AB7ECE1C693046C91E82`.
The test author and independent root reviewer are different agents.

## Remaining consumer gaps

| State | Evidence | Owner and next action |
|---|---|---|
| OPEN | `NativeOnly` alone currently misses the helper's native branch and reaches the CLI. A private, independently reviewed one-condition repair is prepared. | Integration owner: qualify the actual helper route before adoption. |
| OPEN | A CPU core may omit the optional dataset export. A private catch confined to that export is prepared; parsing and mandatory buffer release stay outside the catch. | Integration owner: build the current managed source and qualify feature-off, feature-on and error/buffer ownership fixtures. |
| OPEN | The current shared resolver selects explicit inference/media bundles but omits the core DLL route. | Native owner: preserve current resolver registration and qualify core bundle selection. |
| NOT_TESTED | These copied-wrapper controls do not import the full module, load native DLLs, run builds or generate datasets. | Integration owner: verify the full paired consumer pipeline separately. |
| FAIL | Published predecessor head `2648702` passed 2,578 PowerShell tests but source-bound coverage was 13,451/21,116, or 63.7005%, below the unchanged 85% gate. | Test owners: exercise genuine uncovered behavior; do not waive the gate or credit private tests without hosted evidence. |

No native runtime, full-module, coverage increase or performance acceptance is
claimed by the wrapper repair. The additional native/helper proposals remain
private and uninstalled at this review cutoff, October 10, 2026, 04:09 UTC.
