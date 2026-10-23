# LLM progress job custody

Reviewed October 10, 2026. Owner: root/integration; the canonical eight-case
inert qualification and independent receipt review passed. This review concerns
`Invoke-OpenAIChatWithProgress` in
[`LLM-Helpers.ps1`](../Modules/PC-AI.LLM/Private/LLM-Helpers.ps1).

An exception after job creation previously skipped removal and progress
finalization. The prepared repair retains the original job object, attempts
stop when its state is active, attempts removal independently, and completes
progress in `finally`. It preserves the primary error and attaches secondary
cleanup errors to `Exception.Data['PcaiProgressCleanupErrors']`. When removal
is unconfirmed, `Exception.Data['PcaiProgressOwnedJob']` retains that exact job
for caller recovery. A cleanup-only error terminates instead of returning a
successful response. Do not serialize job contents, API credentials or complete
exception Data dictionaries into ordinary logs.

Only this function changed. The other 22 helper functions and their order are
unchanged. The helper consists of 23 function declarations; the maintained
suite verifies that structure before dot-sourcing it and checks the loaded
function's canonical path and entire AST function extent. The typed `InertJob` fixture
retains its private qualification bytes and `PcaiProgressCustodyR1` namespace.
It creates no process, runspace or registered job. Mocked job, HTTP, wait and
progress boundaries prevent a real request or payload in these fixtures.

## Evidence and source bindings

The original six-case qualification produced **2 PASS / 4 FAIL**, with zero
skips or not-run selectors and stable source. The private successor produced
**8 PASS / 0 FAIL**, with zero skips, not-run selectors or failed containers,
and stable source. Independent audit `E0FF42AC…FE9DD2C` accepted only those
private synthetic contracts. Neither result establishes a real job's closure.

The maintained canonical successor passed **8 PASS / 0 FAIL**, with zero
skips, not-run selectors or failed blocks/containers. Actual canonical file
binding and all source/runtime pins remained stable. The parent recorded
original PID95184, exit0, confirmed closure/disposal and no pending custody.
Independent receipt review `E1089D45…EF72F5` accepted that exact finite scope.

The first canonical run remains **7 PASS / 1 FAIL**: its first binding detector
compared the loaded entire function extent (4464 bytes) with only its body
(4425 bytes). Canonical file binding and source stability passed. The successor
changes only that assertion to compare both entire function extents; it removes
no assertion. Original result, XML, logs, source and closed parent receipt are
preserved. The six original detecting cases and two additional cleanup-only
cases remain; the other seven complete `It` bodies are exact private copies.

| Artifact | SHA-256 |
| --- | --- |
| Original helper | `5FCA017A9C72F303F8F05655F7203403F813F600F2510CCF8CC41596A3CB9D2A` |
| Qualified canonical helper, inert scope | `339B481CE815B3C3AF22011F056C8AB6CBE2E5051806CFCDA026FAA0551A0571` |
| Maintained successor suite | `B68356511F920D07E238E5A1DE593F6251A0EFF7A0DB7C550880BC7AD3129F21` |
| Historical first canonical suite | `20E2059949045C9C330A358E3B176F6DC3B39F1F7BA97B75B9B45709A64B055D` |
| Typed fixture | `FDF5B9BEAFD13DB429CB78BEC11AE2F07E9E45D49E027D6C179843648D7FC11B` |
| Original actual six-case result | `934F9D6791963729C4C27C5E5AB89AAB8843166D1CC65CA6BA5B3729637BC49D` |
| Private actual eight-case result | `E058A3AE3B55EF6282418930076AAF88B6A856768AFBAA3FA954EAB9A1702DE8` |
| Independent private eight-case audit | `E0FF42AC47573D00F38F288BBB85E00D5F3F36DE3524B6A12F0C9AEF2FE9DD2C` |
| First canonical result, 7 PASS / 1 FAIL | `BD4B41B064F6F992DD73E3F8FD5BD4CEC178CFE11C6C5E247C4296E45C44B79F` |
| Canonical successor result, 8 PASS | `7D671751AE35E86B6AF500B14E9A4B3B336D894CF61A2ED8924CBE64A98FF874` |
| Canonical successor NUnit XML | `2C3FAE78AC991A82C463BF6BC1AFA61A64356A68F243DB12B261211C957E494C` |
| Canonical successor parent custody | `FD4439B5208FE8D85329C49A1544009D9DA679FB897303618B9B49F46C6E2CD5` |
| Independent canonical successor review | `E1089D455526C687142B7D7D28C3E7EA676EED05B4D11D5C60AE1D7838EF72F5` |

The exact results are retained under private
`.pcai/integration/llm-progress-custody-repair-r1/actual-predecessor-r1/` and
`llm-progress-custody-repair-r2/actual-candidate-r1/`; the independent audit is
`llm-progress-custody-peer-r1/actual-private8-review.json`. Canonical source,
preimages and selector hashes are retained in
`.pcai/integration/llm-progress-canonical-prep-r1/source-preparation.json`.
The canonical failure remains in `llm-progress-canonical-qualification-r1/`;
successor source, preimage and actual receipts are in
`llm-progress-canonical-qualification-r2/`. The independent successor readback
is `llm-progress-canonical-peer-r1/actual-canonical-r2-review.json`. Execution
used physical PowerShell7.6.6 and Pester5.7.1; minimum-target execution remains
untested.

## Maintained selectors and rationale

Selectors are in
[`LLMProgressJobCustody.Tests.ps1`](../Tests/Unit/LLMProgressJobCustody.Tests.ps1).
All eight passed the source-bound canonical successor and distinct receipt
review; private passes remain historical. Each breadcrumb is the ordinal in
`source-preparation.json`; successor readiness and peer review bind the single
first-case correction and exact other seven bodies to suiteB6835651.

| Ordinal / selector purpose | Protects | Detects | Needs / breadcrumb |
| --- | --- | --- | --- |
| 1. Completed response | Useful output and canonical function binding | Response corruption, wrong removal target, unnecessary stop or HTTP execution | Canonical file/entire-function comparison plus response and exact mock counts; row 1 |
| 2. Model lookup failure | Primary cause and created-job custody | Lost cause or missing stop/removal after lookup throws | Reference identity and exact cleanup/finalization counts; row 2 |
| 3. Receive failure | Completed-job cleanup | Missing removal/finalization or replacement of receive error | Original exception identity, removal once, no stop; row 3 |
| 4. Poll cancellation | Active owned job | Cleanup omitted after the polling wait throws | Cancellation identity, exact stop/removal, no receive; row 4 |
| 5. Two cleanup faults | Primary error with both secondary causes | Early cleanup abort, lost primary or missing secondary evidence | `Stop,Remove` order and both failure markers; row 5 |
| 6. Job creation failure | Unrelated job | Cleanup without an owned job or further model/HTTP work | Zero cleanup/receive/model/HTTP calls and foreign job still Running; row 6 |
| 7. Removal-only fault | Exact unremoved-job reference | Successful response despite failed removal or lost recovery custody | Exact primary/secondary/job reference identities and cleanup counts; row 7 |
| 8. Finalization-only fault | Completed job with honest failure | Hidden progress error or unnecessary retained job after removal | Exact failure identity, one secondary cause and no retained-job key; row 8 |

## Remaining gates

**OPEN:** actual PowerShell job stop/removal and termination, cancellation and
wall-clock deadlines, admission of a new request after retained-job cleanup,
caller handling of retained references, nonterminating `Start-Job` failures,
SSE/streaming and native timeout parity. Stop/removal attempts are not closure
proof. Retaining a reference is not a pending registry or admission guard.
The private parent qualified its normal finite path only; its timeout/drain
and unconfirmed-custody paths remain unqualified for arbitrary payload reuse.

No network, native backend, minimum PS7.0 runtime, fleet installation or
installed-profile acceptance is established. No performance benchmark or
coverage credit is claimed; the global 85% floor remains unchanged. Root owns
normal integration checks and current-head CI after the reviewed commit.
Real-job qualification requires a separately admitted
bounded workload and exact owned-job recovery evidence.
