# LLM strict collection and optional metadata boundaries

The file/content search fallbacks and two Ollama response adapters now accept the tested empty, singleton and optional metadata shapes under StrictMode Latest. Final source-bound local regression executed **128 passed, 0 failed, 0 skipped, 0 not run**: 39 focused controls, 71 existing LLM contracts, eight progress custody controls and ten streaming deadline controls. PSScriptAnalyzer 1.25.0 reported zero findings on all seven changed scripts/tests. This is local fixture evidence; current-head CI, actual native/provider behavior, coverage and performance remain separate gates.

## Narrow changes

File discovery and result limiting explicitly capture arrays; content discovery also captures an array. Empty selected files produce `TotalSize = 0`, while nonempty sums retain their prior calculation. Eight controls call the actual public search wrapper with real tiny D-drive files, exact paths, lengths, line contents and input hashes. Native availability is disabled only at the boundary; no copied search algorithm is used. Duplicate search fallback collection behavior remains OPEN outside this change.

The native chat adapter and public generation formatter check optional raw, tools and timing properties before reading them. Missing/null tools become empty arrays and absent duration remains null. Explicit zero duration/settings and supplied metadata identity are preserved. Required content/model failures remain detected under StrictMode. Real generic `Dictionary[string,object]` and `OrderedDictionary` responses use string `PSBase.Keys`, ordinal-ignore-case matching and indexing by the actual matching key. More than one case-distinct matching key is rejected as ambiguous. This avoids the generic dictionary's unsupported PowerShell `.Contains(name)` invocation and its adapted `Keys` member shadowing. It does not promise arbitrary dictionary/object implementations or change the required payload dialect.

Only `Invoke-OllamaNativeChat` changed in the complete private helper. The other 22 function extents and all text outside this extent are exact to the published predecessor, including both streaming helpers and the progress routine. Only file/content fallback function extents changed in the search file. There are no new production helper functions, FFI contracts, background jobs or endpoint calls.

Two existing fixture expressions were corrected without weakening their assertions. Successful diagnosis checks optional `Error` presence while retaining the no-error assertion. The progress declaration guard handles null parse-error reports and null `EndBlock.Traps`, retaining actual parser-error, executable statement and trap refusal. Direct parser metadata showed a valid helper has an empty nonnull ParseError array and **null Traps**; the historical undifferentiated `.Count` error must not be attributed solely to null parse errors. All eight progress test bodies execute and remain exact; all 169 existing LLM contract and 58 progress `Should` commands remain exact.

## Preserved failures and qualification

Evidence lives under `D:/pcai-relocation/llm-strict-boundary-*`; private source custody and final inventory live under `.pcai/integration/llm-strict-boundary-repair-r1/`.

| Run | Actual outcome | Interpretation |
|---|---|---|
| baseline-r1 | 9 pass / 17 fail of 26 | Includes new search mock-registration faults; preserved, not a clean production detector. |
| baseline-r2 | 12 pass / 14 fail of 26 | Corrected actual-wrapper detectors; input bytes/source stable. |
| candidate-r1 | 23 pass / 3 fail of 26 | Remaining declaration-guard null Traps failures preserved. |
| candidate-r2 | 27 pass | Adds a genuine parsed trap-only refusal control. |
| regression-r1 | Setup failure / 0 tests | Private runner attempted unsupported Pester configuration `+=`; preserved without execution credit. |
| regression-r2 | 116 pass | Source and receipt retained before dictionary extension. |
| dictionary-baseline-r1 | 27 pass / 12 fail of 39 | Test constructor also suffered a deliberate `Keys` shadow; preserved. |
| dictionary-baseline-r2 | 29 pass / 10 fail of 39 | Corrected constructor, genuine typed dictionary/casing/ambiguity detectors. |
| dictionary-candidate-r1 | 39 pass | Twelve additive dictionary cases; original 27 bodies unchanged. |
| regression-r3 | 128 pass | Final exact sources; all five suites under StrictMode Latest. |

Final regression original tool session 60678 completed exit 0 using PowerShell 7.6.6, .NET 10.0.12 and explicit Program Files Pester 5.7.1. Receipt SHA256 is `15F825CEA2CC61FFBD189824F3D5801BC1EA20389CD4AE30DCB5451813A82CDC`; XML is `A71367E46076FA0B8EAB6E694E48B22321300AEF512AFEFDF6DFF24E291AC263`. It records zero pending streaming fixtures, no first/secondary failure and zero source drift across 12 inputs. All eight search input inventories remain unchanged. Lint original session 98395 completed exit 0; receipt SHA256 is `18CF6BF1768113F995E97419F881D36159026FFB457D8908F114B098A5DDF327`.

The seven final code/test hashes and every historical receipt/XML are pinned in private `ready.json`; `source-equivalence-r2.json` records exact preserved extents and assertions. Helper SHA256 is `7D170226AC97687AEF46B98D60A11DAEEB0E38B9AD2FDD2E194742FF5B46217A`, generation formatter `F74B881FB4DAC0F4761E35FA95BDD675AA5E1A73AD2BFC4DF6BEF1C17CDC857A`, and search `5125A02F1C5E9241822BE6E130A2EBBBFC903873E55664EA04801AC8EFFF58BF`.

These controls mock provider/native transport boundaries, not the production parsing/formatting/search logic. The inherited streaming tests use owned inert loopback fixtures; final XML execution and zero pending registry establish ordinary completion, not newly injected IDisposable, interruption or generic process custody fault acceptance. No model, vendor endpoint, native ABI, deployment, numerical acceleration or global 85% coverage acceptance is claimed. Ordinary-mode reruns were not added because this repair addresses the observed StrictMode boundary and the full relevant strict regression passed.
