# Evaluation public contract validation

On October 11, 2026, both the preloaded-module and fresh-process runs passed all fourteen production-interface cases with zero failures, skips or cases not run. The preloaded run preserved its original module and eight state fields: five original object references and three string values. It also restored both suite environment variables. The fresh-process run restored those environment variables and left zero Evaluation root or validator modules loaded. Both verified sixty source pins.

The earlier run passed nine and failed five cases. Four failures came from a BeforeEach environment-loop variable overwriting the parameterized case Name in PowerShell's case-insensitive scope. The fifth exposed a production contract: missing-baseline handling emitted one null success-pipeline item. The fix retains the original diagnostic and returns without that item. All fourteen original case bodies and assertions are preserved.

The tests exercise the real Evaluation module, orchestration, owned temporary persistence and regression consumers. Only Invoke-RestMethod at the HTTP boundary is mocked. These results do not qualify a live HTTP server, inference model, native implementation or performance improvement. The narrow instrumented run executed 143 of 179 commands across three files (79.89%); this is separate from the required repository-wide 85% CI gate, which remains failing at the earlier published head.

Run the public tests from the repository in PowerShell with Pester 5.7.1 installed:

```powershell
$env:PCAI_CACHE_PROVIDER = 'memory'
Import-Module Pester -RequiredVersion 5.7.1
Invoke-Pester -Path ./Tests/Unit/EvaluationPublicContracts.Tests.ps1 -Output Detailed
```

Test rationale and source breadcrumbs appear above each selector in [the public controls](../Tests/Unit/EvaluationPublicContracts.Tests.ps1). The production correction is in [Test-ForRegression](../Modules/PC-AI.Evaluation/Public/Test-ForRegression.ps1). Only thirty-two rationale comments were added after execution. Removing those exact added lines reproduces the whole executed source, and all eight complete It commands exactly match it.

## Review bindings

Source reviewed by the distinct Codex source-qualification lane before exact adoption. Recorded hashes use SHA256; complete selector hashes cover each It command, including parameterized data and assertions.

- ExecutedTestSHA256: `6572E4FF87C1EB46F64AE5D57273B04C1CA62ABAAD5B5B79A334B382C1B4F02D`.
- PublishedTestSHA256: `02D708D3000D6B87CC90DFD55B90C627330D6C3EA7436B4AD3A1BFFAF2FF7EBC`.
- ProductionSHA256: `183DB157C54D695D6D7147BDC0C6CA2A440239EED6910C4262B5E667418A40CC`.
- SourcePeerSHA256: `9CC4C1F534C5236B99BB18598B3291E1A244F7CDA4F23D4E89F613624E94F3F1`.
- AmbientCountsSHA256: `BF47DC279B4000E5AB793827EA633C8A098AFA6E913C2CF3F5FC79859811B571`.
- ColdCountsSHA256: `B6A19BE405D0EBAAB7DC7CB02D3AE7B30B9D59852B6060B8E7AC929EDFCF2277`.

| Selector | Cases | Complete It SHA256 |
|---|---:|---|
| persists an actual completed HTTP evaluation and consumes its on-disk baseline in reports | 1 | `89F7DDC59D25231128102225543E5CD68D23A7C4C3F4A86FEFF30394C1CD7058` |
| refuses a nonfinite aggregate from individually finite completed observations before publishing a baseline | 1 | `E4C79FDC5DE4369044924B74FBDD8082E9A569B6569A0D86F63D61C2EC1DC511` |
| reads a legacy scalar metric baseline without rewriting its physical representation | 1 | `81EB31F64D5A7B650A611AA599F914A379E691FD5A1EE42CE9581399B7858D10` |
| rejects a baseline without a metric dictionary: <Shape> | 3 | `E26D4EA3417E9139189A83DA66A956E199C348225D460C39BEAC429FD23334B7` |
| refuses a finite input pair whose relative summary exceeds the numerical domain: <Kind> | 2 | `C6FFF083BF7A64821DDA988B0986E03140C0AD0230082DDDBE078086AD4E7DB8` |
| refuses an unsafe regression reference name before loading another path: <Kind> | 4 | `B3EC02B93CE1147DF589376E99C477BFED93EE6D18A68AFFDC27BCD7FECD4613` |
| returns no successful comparison after a missing reference under nonterminating error policy | 1 | `3E6A7FAB13E90F6E6EEA46444617268192C28C634764A34C315A5FB6D25988F0` |
| reports absent current suite without fabricating a successful summary | 1 | `40B17CA814B67A090D07048C9AA6C1B928388B5ED2F2C08A16F1FF0D96B96336` |

Physical evidence is retained locally under `.pcai/integration/evaluation-public-contracts-actual-r5/` (preserved failures), `evaluation-public-contracts-actual-r6/`, `evaluation-public-contracts-cold-actual-r6/`, `evaluation-public-contracts-source-peer-r6/` and `evaluation-public-contracts-publication-r1/`. The original child exit codes were zero for both R6 runs; this does not expose or establish custody of the tool adapter's internal native handles.
