# PC-AI fleet integration review

Evidence cutoff: October 10, 2026 UTC. Owner: Codex integration lane. Live receipts and
private build outputs remain under `.pcai/integration/`; private profile and hosts
originals remain outside Git. This report records verified work and outstanding
gates rather than declaring a fleet clean before those gates finish.

## Current acceptance readback (23:17 UTC)

PC-AI PR 182's published `da3d9af` passes 2,727 PowerShell tests with 39 skips and
no test failures, but CI run `38088936606` fails the unchanged coverage gate:
14,022 of 21,324 commands, 65.7569%, leaving 4,104 additional distinct covered
commands to reach 85%. Rust Guidelines `38088936595` and NVIDIA software validation
`38088936592` pass. Those checks do not establish hardware or deployment acceptance.

The independent shared-cache repair now passes eleven controls with one explicit
Redis skip; its seven detecting baseline failures and source-bound selector register
are documented in [the cache review](shared-cache-correctness-review.md). No measured
performance or new coverage credit follows from those narrow controls.

Related dtm-codex PR 39 merged as `62985c7` after native CI `38093486222` passed its
paired build and 26 maintained contracts and portable CI `38093489141` passed at
`56783cb`. Local main fast-forwarded to the merge, its tree equals the reviewed head,
and the integrated owned branch was retired normally. Foreign untracked files remain
preserved. The Carbon runner is running; the earlier startup qualification failure
remains a failure despite successful subsequent CI. Merge-head CI is a separate gate.

Production MCP deployment remains open. The genuine R5 client read reached the fixture,
but its progress event failed the old validator. Thirty-four managed detecting controls
qualify the successor validator only. The R6 prerequisite check then failed before
launch because exact JSON integers decode as Int64 while new checks required Int32;
the frozen failed packet is preserved while a typed successor is prepared.

The isolated uutils `mv` package Clippy run failed with exit 101 on Rust 1.99's
`map_unwrap_or` diagnostic in `winutils/shared/winpath/src/volume.rs`. All original
process/job/drain resources and receipt writers closed; no later test phase ran.
A minimal separately frozen successor is being prepared without changing the failed
candidate or protected T: work. Metadata acquisition had passed independently; that
success never counted as compilation or lint acceptance.

Milly retains about 580 GiB internal free after the verified migration. Its reconnected
SSD negotiates 10 Gb/s; the bounded 256 MiB direct-read sample measured 416.26 MiB/s.
Cold boot/removal, Windows access to the native image, restore and physical second-slot
vacancy remain open. dtm-work connectivity and credential reconciliation remain open.

## Historical acceptance readback (07:48 UTC)

At 07:48 UTC, PR 182's published head remains `c6f5222`; local reviewed commits
have advanced to signed `8bede69` and have not yet been pushed. Published head CI run
`37888041832` failed three PowerShell tests and measured
37.33% coverage against the unchanged 85% target. Rust, .NET, lint and CPU runtime
jobs passed; downstream integration was skipped. The source-equivalent local
run passed 1,751 cases, failed three and skipped 53, with 38.347% coverage. Six
ignored Archive files explain its larger command inventory; they remain preserved.
No complete CI, native-pair, deployment or fleet-clean claim follows from these
partial results. Commit `5b4a858` binds native probes to manifest ancestry and
passes 14 focused tests. Commit `245282d` captures the original owner/group/DACL
before replacement and restores through an owned handle, detecting distinct
same-byte later files. The 93-case config gate and separate 46-case C:/D:
controls pass without skips; SACL/audit preservation is outside that contract.
The successor frozen signed `8bede69` checkout on D: now passes all 1,844 tests
with 59 explicit skips, no failed containers or source changes. Exact current
CI selection and recursive module coverage measured 8,391 of 20,154 commands:
41.634%, still below 85%. This clean checkout contains tracked source rather than
the previous local ignored Archive files; no coverage-path exclusion or target
waiver was introduced. Its test success does not clear the coverage gate.
Substantial uncovered module behavior remains active work.

Baseline regression review also reproduced a false-green performance comparison:
snapshots stored run summaries instead of the metric means consumed by the
comparison. The reviewed repair persists real distributions and separate summaries;
two synthetic save/read tests now detect both +20% latency and -20% throughput.
Live inference qualification and additional baseline admission checks remain separate.
Further detecting fixtures found integer Math overloads rounding fractional
evaluation scores. Eight genuine failures and four controls became twelve passes
after selecting floating-point overloads. The full 46-case algorithm, aggregation,
provider and baseline union passes without skips; source hashes remained unchanged.
Two informational or constructor-only cases were replaced by real empty/mixed
result aggregation assertions. Exact safe evaluation/provider suites are being
added to CI without changing the production coverage universe or 85% target.
Separate A/B detecting fixtures reproduced ten failures, including false
significance for two overlapping two-element samples. The local successor uses
the two-sided Student t probability for Welch inference, validates alpha and
finite observations, and refuses degenerate or unrepresentable summaries.
Four published SciPy references and three analytic values qualify the numerical
routine. An additional real overflow-tail and relative-summary failure were
repaired. Independent review then reproduced multiple simultaneous effect-size
labels from overlapping switch predicates. Seven public boundary fixtures detect
that defect; the committed repair terminates the first matching predicate.
The complete 85-case evaluation/provider/baseline/statistics union passes without
skips, parser/analyzer findings or source changes. Independent review admitted
commit `8bede69`; remote publication remains pending. Effect size retains its
existing equal-weight RMS standard-deviation convention. These are software
checks, not population or clinical acceptance.

CargoTools caller-selection qualification now passes 18 fresh PowerShell7/5.1
controls plus four actual offline native exit0/101 controls on its frozen
successor. A real UniFi Windows wheel attempt separately failed because the
canonical wrapper mixed cargo JSON stdout and its scalar preflight status. The
underlying cargo step reported success; the wrapper rejected its output array.
No DLL, wheel or consumer update follows from that attempt. The status boundary
is being repaired with real child-process stream controls before another build.
Those controls also reproduced rust-analyzer output-routing and consumption of
child arguments after Cargo's separator. The frozen successor passes 96 focused
checks plus fresh Windows PowerShell 5.1 controls; its full repository gate is
running. Successful native output and scalar status now use separate channels.

The credential publisher remains private and uninstalled. Qualified process
custody does not establish file-publication privacy: a real replacement exposed
synthetic secret bytes under an Everyone-readable predecessor DACL before
final tightening. A second real detecting fixture replaced the initially private
original through a parent with foreign mutation authority, exposing synthetic
bytes before displaced-file rejection. The private successor admits owner/DACL
and existing namespace authority before staging secret bytes. It passes 29
publication controls; full legacy/durable qualification and review remain pending.
Read-only ACL metadata predicts deliberate refusal of this host's default TEMP,
`.machine` and `.local/share` staging roots. A protected child directly beneath
the admitted user-profile root is being qualified. No live ACL repair or secret
publication occurred; exact prior failures and compatibility losses are retained.

Carbon's new private CPU Core/CLI build uses absolute, hash-pinned Rust1.99
tools and two jobs after all 2,437 extracted files matched the signed frozen
archive. Core/CLI qualification passed 122 tests with two explicit ignored cases,
strict Clippy and release compilation; every source hash remained unchanged.
The positive-CPU real busy-child test passed within its unchanged deadline.
Transferred DLL/CLI hashes matched on P1. A private D: build using SDK10.0.401
and its latest compiler passed 26 managed/native-parser tests, with one media
fixture explicitly skipped. Fresh PowerShell worker/DLL/bridge pairing passed
47 cases, including all three native consumer tests; one filesystem short-alias
control skipped because the selected filesystem did not provide aliases. The
native version export matches `0.1.0+37743c1`, and all source hashes stayed fixed.
On the matched twelve-small-file SHA256 benchmark (five measurements per backend),
PowerShell averaged 5.57 ms, the persistent worker 29.36 ms and the direct CLI
91.79 ms. All backends validated the exact unique path/hash set. This workload
favours PowerShell; the worker improves process-start cost relative to the CLI
but does not beat the baseline. The observation does not establish a large-file
or cross-host policy. CUDA, model execution and consumer installation remain
separate gates.
Work remains unreachable on the latest established routes; route investigation
continues through its exclusive owner without changing its preserved checkout.

Git-guard's current reviewed source is merged main `e46d408`, with both PR and
merged-main CI passing. Signed immutable `v0.2.11` is normally installed on P1;
installed engine hashes and reject/allow controls passed. Historical v0.2.9
observations below describe prior stages, not the current installed version.
Fresh hygiene reports and dry runs now work; retirement still requires individual
ownership, exact preservation and active-writer checks.

## Reviewed integrations

PC-AI PRs 178, 179 and 180 are merged. Current main `b92e5f6` passes its complete
CI workflow. The workload documentation's archived launch and summary hashes were
verified locally. Dependency lock updates were reviewed together with consumers.

CargoTools PRs 12 and 13 are merged; main and origin are identical at `01b4c8c`.
Explicit preflight disable is honored at every quality boundary. The telemetry
setting now uses a value accepted by cargo-binstall while retaining user override
semantics. Local full validation passed 644 tests, with five skipped and nine
opt-in tests not run. Both reviewed PR heads and merged main passed hosted CI.
The independent T: checkout has the same main and retains its untracked test
report with unchanged hash. Retired source branches have verified Git-bundle custody.
Consumer discovery also found an independent installed Git checkout under the
local Documents module root. Its complete refs and unique wrapper backup were
preserved outside Git before a normal fast-forward to `01b4c8c`. The old branch's
tree matched the merged predecessor and was retired by an exact-head deletion.
A fresh PowerShell import selects that clean updated checkout; the normal
installed wrapper reports Rust/Cargo 1.99.0. A copy installation correctly refused
to overwrite this nested Git checkout.

Linked dtm-codex PRs 34 and 35 are merged; main and origin are identical at
`af4fac4`. The deployment-policy claim was corrected. Actual merged-main failure
logs then exposed process-custody and fixture-deadline defects: the probe now waits
for child/helper closure and fails closed when termination is unconfirmed. All
16 real process tests passed on Windows and Linux; reviewed-head and merged-main
CI passed. The 489 imported, untracked skills retain their paths, lengths and hashes.

Linked git-guard PRs 45, 46 and 47 are merged. Secret scanning now covers rename
diffs; the rule catalogue exposes witnessed enforcement gaps rather than hiding
them; structured-data validation parses format bytes instead of the host locale.
The latter's real-gate regression reproduced twelve failures before the fix and
passed all sixteen cases afterward. PR 48 released signed immutable v0.2.8;
the installed release's native links and actual reject/allow hook probes passed.
The first installer attempt preserved the previous release when MSYS copied a
directory instead of creating a link. PR 49 fixes that defect and restores prior
hooks/docs when late native-link creation fails: 43 actual Windows fixtures
passed with no skips, and both full CI jobs passed. Its source merged as
`c42347d`. PRs 50 and 51 completed v0.2.9 and corrected the Docker CI runner
selection. Main and origin match `46cd373`; its current-head CI passed. Signed
immutable tag v0.2.9 was verified and installed through the normal Windows
installer. Actual unsafe-commit rejection and clean-commit admission passed.
The predecessor release, overlays and retired local/remote ref tips retain
verified bundle/hash custody. A later live writer has new hygiene implementation
work in this repository; that work is protected while ownership is reconciled.
The reviewed installed v0.2.9 does not implement the newly required
`hygiene report/drain` commands, and installed QA guidance lacks section 12.
Its help output with exit zero is not a hygiene report. Qualification and
installation of the owner's reviewed successor must precede automated drain.
Independent review of PR 53 reproduced deletion of a concurrently changed
ancestor branch despite the supplied expected tip. The finding is recorded on
the PR with a real Git reproduction; active source edits remain protected and
no automated drain apply has been admitted.

## Repaired behavior

- The module installer verifies staged bytes, preserves existing installations and
  unrelated module-path roots, and provides genuine dry-run behavior. Copies are
  the default; live junctions require an explicit selection.
  It inventories all fifteen development modules, including the paired flat
  Media/Inference manifests. Sparse copies, incomplete explicit bundles and
  unsupported flat junction publication fail closed. All 328 discovered module
  contracts and a private fifteen-module install/consumer witness passed.
- The profile shim derives roots from the current consumer's environment and
  rejects a self-referencing canonical path. The canonical p1 profile now honors
  the module-path skip flag, resolves its developer module root consistently and
  avoids secret-backend fallback in minimal sessions. Original and actual displaced
  bytes have private hash-bound custody; concurrent replacement fixtures passed.
- Explicit native bundles select the intended Rust/C# pair, reject incomplete
  bundles and prevent reuse of a different loaded managed assembly. Media native
  resolution also honors the selected bundle. A negative native operation error
  no longer falsely marks the media DLL unavailable.
- Standalone media imports now honor explicit bundle selection, canonicalize
  relative paths consistently with the native resolver, and reject reuse of a
  different loaded managed assembly. Five fixtures and a fresh-process actual
  media DLL probe passed.
- Provider settings reject unsupported fallback entries before mutation. Failed
  persistence restores the prior memory settings. Filtered configuration tests
  now preserve the real repository configuration; 46 fixtures and a separate
  Reset-only run passed.
- CUDA media builds require optional cuDNN and FlashAttention only when requested.
  Native and aggregate publications allocate stable revisions and retain earlier
  payloads, ZIPs and evidence. Build clean retains published packages/deployments.
  Timestamp-bearing versions remain metadata rather than artifact path names.
  Benchmark CLI comma-separated case selectors now work as documented.
- The C# bridge passes a strict Release build with zero warnings and errors.
- Device diagnostics consume the native `config_error_code` and `pnp_class`
  fields. Malformed or fractional codes discard the complete native result and
  fall back to CIM. Ten adversarial fixtures passed; an actual native/CIM run
  agreed on all 588 devices and three error devices. Seventeen adversarial array
  and row fixtures now pass; an actual Windows PowerShell 5.1 run qualified eleven
  JSON shapes without dropping the module's supported consumer version.
  The capability catalogue
  uses the exported token-estimation command. The advertised Defender helper
  and evaluation-suite help now resolve through their actual public contracts.
- Optional media upscaling absent from a CPU-only DLL produces a clear feature
  availability error. An actual C# integration test exercised that missing
  export without producing a file. No FFI signature or optional export was faked.
- Legacy Thunderbolt entrypoints delegate explicit static intent through the
  maintained adapter/global-address/route guards. Plan, help, dry-run and WhatIf
  remain nonmutating; unrelated addresses and ambiguous state refuse before
  assignment. All 144 affected fixtures passed, with zero analyzer issues.
  Actual network Apply was not executed.
- Owned Windows process cleanup confirms exit across redundant termination races,
  independently attempts handle closure and preserves the original error with
  exact-handle custody when cleanup remains unconfirmed. Explicit GUID retry and
  WhatIf preserve that custody; replacement launches are blocked meanwhile.
  Forty-six focused fixtures passed without skips, including a real failed
  native close, blocked stdin and descendant termination after normal parent exit.
  Cleanup waits at most two additional seconds; loaded older helpers require a
  fresh PowerShell process rather than silently retaining old behavior.
- API tests write generated reports under their private test directory and verify
  that tracked reports are unchanged. The maintained report now accounts for all
  twenty-two standalone public exports and uses repository-relative paths.
  Its inventory reports 262 functions and 184 functions lacking comment help.
  The C#/Rust comparison explicitly describes its limited core-import scope;
  it does not qualify every optional Media/Inference export.
  Flat wrappers are explicitly included in the coverage configuration.

## Verification and measurements

Local tools: PowerShell 7.6.6, Rust 1.99.0 and .NET SDK 10.0.401. The bridge retains
its .NET 8 consumer target. The core Release build passed 59 unit tests and its
documentation test; 18 C# tests passed against that explicitly selected fresh DLL.
Three PowerShell native-search integration tests passed without skips. Twenty-four
profile repair fixtures and seventeen installer fixtures passed.
The newer Pester release exposed missing default mock handling in four race
fixtures; the correction keeps real original/displaced-file hashing and all race
assertions. A current Pester 6.2 run passed all 102 profile, installer, provider,
media-selection and artifact fixtures without skips. Its persisted manifest binds
the tested source hashes. CI then exposed four legacy media-loader fixtures that
still assumed the former warning/path contract. Corrected real invalid-assembly
tests and fresh-process successful loading passed all 99 affected media fixtures;
missing-file cases also assert that they did not pass through mock exceptions.
Subsequent hosted CI exposed a false-positive in two successful-loading fixtures:
their assertion loaded the fixture assembly itself. Fresh child tests now select
the actual project root and only inspect assemblies the initializer already
loaded. A separate fresh producer and loader passed 94 affected fixtures locally
under CI's Pester 5.9.1; hosted failures remained. Printed current-head diagnostics
then proved that Windows 8.3 paths and .NET-expanded assembly locations referred
to the same DLL but were incorrectly classified as different bundles. This is
a real consumer canonicalization defect, and its production fix and exact-head
CI remain required. Review also found a stale assembly variable introduced while
extracting discovery; a same-loaded-bridge StrictMode witness is required with
its repair. The reviewed nine-file repair now passes all 469 focused cases under
Pester 5.9.1 without skips, including actual short-directory and short-leaf
selection, real Core calls, same-bridge StrictMode reuse and rejection of copied
identical-byte DLLs from a foreign root. Those runtime checks bind the earlier
qualified DLL pair, not the pending new sampler. Current-head hosted CI is still
required; the measured loader coverage was 32.08%, not its displayed target.
Restoring discovery of parameterized module contracts exposed three further
public-surface defects. After correction, all 286 module-loading contracts pass
under both Pester versions, with no skipped or undiscovered cases. The later
standalone export/install correction expanded this to 328 actual contracts. The bounded
suite is now included in the CI configuration.
Ten artifact-publication fixtures passed, including repeated aggregate publication,
ZIP collision and clean preservation. The actual media DLL remains
available after an intentionally induced negative native error.

Matched minimal-profile startup probes before and after the reviewed patch measured
median profile execution of 958 ms and 605 ms respectively, across three samples
each. Whole-process medians were 1816 ms and 1798 ms. These small local samples,
collected during builds, establish the observed patch result rather than a fleet
throughput claim. Tooling reruns with the fresh bundle are preserved; differing
traversal policies and output limits require parity checks before speedup claims.

## Fleet and remaining gates

The October 9 bus reconciliation is maintained in
[agent-bus-pcai-findings.md](agent-bus-pcai-findings.md), including exact message
IDs, retractions and acceptance boundaries. In particular, the published
integration head's green badge concealed 53.13% coverage against the required
85% and an incomplete 18-file inventory. Signed `5b766013` repairs enforcement
and recursive selection with eighteen actual passing tests; final-source measured
coverage remains required. Signed `2b6f95b` repairs the 48-byte compact managed
process layout with eight passing direct-parser tests. Signed `092565f` preserves
deep configuration and actual displaced-file custody with 82 passing Windows
tests. POSIX configuration publication remains explicitly fail-closed.

The profile/worker successor has 60 portable passes and three explicit pending
native-pair cases, including 24 exact-custody tests. Work's later unexpected
outage prevents final source acceptance; earlier successful native/profile/auth
receipts remain historical. New Rust compact serialization and the managed
bridge must be qualified together before publication. UniFi's exact SSH/MCP
strict Clippy completed successfully; its broader unchanged-deadline Python
gate remains separate and is running with claimed D: fixture storage. These
observations supersede pending-status descriptions below only for the precise
gates stated here.

| Surface | Current evidence | Owner and next action |
| --- | --- | --- |
| dtm-carbon-two | CPU-only Intel graphics. Earlier receipts reported Rust 1.99.0, a private SDK, 59 Core tests, one documentation test, strict Clippy, 17 managed tests and 18 native fixtures. Current owner verification at 05:00 UTC reports SDK 10.0.401 absent, so that older SDK observation is not current. Corrected archive custody is verified; current Windows native and managed qualification are pending. Canonical profile has an intentional alternate structure; the p1 repair planner fails closed. | Root/fleet: qualify existing Rust Core/CLI independently of the absent managed SDK, then pair on a qualified Windows SDK host. Preserve profile/auth, junctions, private overlays and untracked build script. Configured model assets are absent; shared GPU defaults need host-specific overrides before AI execution. |
| dtm-work | Strict SSH reached DTM-WORK after the peer returned online. PowerShell 7.6.6, Rust 1.99.0, SDK 10.0.401, approximately 112 GiB free RAM and Quadro RTX 4000 were verified. The actual `C:/codedev/pc-ai` checkout retains its sixteen tracked and three untracked source changes. The canonical profile, fourteen package aliases and PATH repair passed consumer/readback checks with original custody; actual Bitwarden session reuse and configured GPG signing/tamper rejection passed. The private CPU inference build generated four tokens from the existing TinyLlama model and reaped its owned server. The Network key's Site Manager 401 remains a separate credential scope result. | Root/fleet: reconcile preserved unique checkout source and final-head deployment through Work's exclusive owner. Admit the credential helper only after failure-custody repair and provenance review. The ADB-authorized, locked Pixel did not need an account or network policy change. |
| Local name resolution | Removed one obsolete Headscale IPv6 hosts entry and a conflicting bare dtm-work LAN token; preserved radius LAN alias and tailnet mapping with original-byte custody. | Fleet lane: verify route retries; this does not prove endpoint recovery. |
| Candle/media PR 156 | Corrected Candle 0.11 union passed 125 media, 58 model, 18 server and 23 documentation tests; three model cases explicitly require absent real-model fixtures. Default CPU media tests passed 119 cases. Optional CPU/NVML/upscale Clippy passed. A fresh CPU DLL exposes all thirteen base exports; actual PowerShell initialization, missing-model error, async unknown-request status and shutdown checks passed. | Dependency/root: CPU source integrated; finish bounded GPU compilation and deterministic CPU/GPU comparisons before GPU publication. Full trained-model quality and the three absent-fixture tests remain distinct gaps. |
| Mistral backend | SDK/core 0.8.1 alignment and the merged graph passed 142 tests without skips. Work's real production Release build passed the same 142 tests; actual CPU model loading, enumeration, four-token generation and HTTP 400/422/404 rejection passed with its owned server reaped. The binary binds source `245e742`, not the later final integration head. Full locked metadata repaired four optional Candle lock entries lost during textual merge. | Root/fleet: retain the successful private model receipt and qualify final-head metadata/deployment. The source-only Work archive and private build did not alter its checkout or defaults. |
| Linked dtm-codex | PRs 34/35 integrated with current-head and merged-main CI; completed branches retired after preservation. | Root: source deployment/consumer validation if this host consumes the changed launcher; keep imported skills under their existing custody. |
| Existing WIP and private evidence | The three untracked private boot/Thunderbolt evidence files now have verified original and copy custody outside Git under stable recovered-evidence revisions. Their mapping is `.pcai/integration/preserved-evidence-migration.json`; historical inventory observations retain their original paths. SQLite/Python compilation caches remain in place with specific generated-cache exclusions. Carbon build script and private overlays remain preserved. | Root/fleet: refresh final inventory and cross-machine equivalence after source integration; no reset or forced cleanup. |

Work's unique benchmark/profile and native-sampler changes have separate source
custody. The preserved pre-change performance binary built successfully and its
actual worker measurements record child and observer interval CPU; the candidate
port must establish output parity before any speedup claim. Windows PDH sampling
exceeded the concurrent test deadline, while a separate serial baseline probe
completed in nine seconds. The candidate replaces Windows global CPU sampling
with checked GetSystemTimes deltas and preserves unknown values on API failure
or multiple processor groups. Native gates and the rebuilt pair remain required.
The new source's actual concurrent Core process gate returned ten passes, one
failure and one ignored case after 452 seconds; it failed to observe positive
child CPU. A serial probe of the same executable also exceeded its unchanged
ten-second guard and confirmed closure of its owned process. These failures are
preserved, and equivalent existing gates are being handed to Work rather than
increasing deadlines. CLI tests, strict Clippy and fresh paired runtime checks
remain separate requirements.
The CLI JSON null value retains process rows in the actual C# DTO; serializing
the raw C# NaN field with strict System.Text.Json remains unsupported.
The inherited native total-thread value is still a process-count approximation;
it has not been qualified as a real thread count.
Work's credential adapter passed 92 private fixtures and actual vault/session
reuse. Independent source review nevertheless found unconditional Process
disposal after unconfirmed termination or stream closure. Its owner is repairing
exact-handle and pipe custody before maintained source admission. The existing
version-one managed-file schema is verified; credential values and private
machine configuration remain outside Git.
A temporary owned-compiler priority experiment was overwritten by the governor
and established no useful-throughput improvement. No persistent policy changed.
UniFi's reviewed
source fixes sparse/malformed Protect telemetry and boot timestamp units. Its
full Python run passed 4,123 tests with 131 explicit skips and 83.10% coverage.
Actual Windows SSH/MCP unit/doc tests passed seventy cases without skips.
Its freshly compiled executable passed all thirty-two MCP protocol tests after
the malformed-argument fixture was corrected to require the exact current SDK
tool-error response and successful real-handler recovery. Strict Clippy, builder
publication repairs, current-head CI and PR 71 integration remain required.

Outstanding broader platform work remains in `TODO.md`, `optimization.TODO.md`
and `boot.TODO.md`, including cancellation/schema parity, media fixture expansion,
startup latency and post-reboot workstation validation. CI and offline/native checks
do not establish real-model quality, GPU performance or a clean post-reboot window.
The earlier p1/Carbon inventory found no configured AI/media service listeners;
Work's runtime inventory is being collected separately.
P1 has two NVIDIA GPUs and existing Media/Inference DLL consumers: a persistent
Core/C#-only explicit bundle would suppress those surfaces and must not be deployed
as its complete runtime. Carbon's CPU-only requirements differ. No GPU/model
quality claim follows from the CPU builds or FFI load checks.

## Historical reviewed work and remaining gates (11:52 UTC)

Readback: October 9, 2026, 11:52 UTC. This remains an active integration record.
PC_AI's reviewed local repairs include signed `1762ff95` and `ff43162`; the published PR 182 head is still
`c6f5222`. Current-source CI, merged-main equivalence, deployment and fleet
acceptance remain open. Earlier sections retain historical evidence and must
not be read as current dispositions.

The completed whole-repository run on frozen `e5cb5fb` passed 2,322 tests,
with zero failures, 59 explicit skips and no unrun cases or failed containers.
It executed 12,053 of 20,620 recursively selected module commands:
58.452958%, **FAIL** against the unchanged 85% requirement. Independent
readback confirms all 2,462 tracked files unchanged, 236 module source files,
235 instrumented files and untouched primary configuration/log bytes.
The original driver's source count was corrupted by a shared PowerShell
variable; independent XML/file verification corrects that metadata without
changing the run, denominator or threshold. The older `8bede69` run remains
preserved. Focused results below cannot be summed to clear the whole-repo gate.

Reviewed, signed repairs now include baseline lifecycle (`ad8901f`, 105 cases),
evaluation/provider contracts (`ca86eb0`, 134 cases), evaluation project-root
and judge admission (`88cb1f8`, 175-case union), native Network exit-status and
WhatIf behavior (`610ed65`, 64 cases), and WSL version/service status handling
(`9b35703`, 65 cases). These are deterministic software contracts. The private
HTTP timeout witness establishes a stalled-read error, not a whole-request
deadline. They do not qualify live inference, NIC changes or machine defaults.

Signed GPU discovery/admission fixes (`6c88259`) pass 65 cases, including
numeric CUDA/cuDNN versions, download URL admission and WhatIf dispatch.
The predecessor produced five failures in eight cases. The backup helper
already inherited WhatIf; the detecting assertion proves unnecessary dispatch,
not an actual backup write. No live GPU, driver or registry setting changed.

Signed PowerShell proxy custody fixes (`87fee7d`) pass 95 cases. Actual owned
children and synthetic state reproduce PID-only termination, lost state after
refusal, publication races and recovery-getter failures before the repairs.
Strong process references, executable paths and UTC creation ticks now govern
admission; unresolved children and displaced state remain recoverable. The
Windows file identity witness includes distinct same-byte replacement files.
This is not an arbitrary-writer compare-and-swap guarantee.

A later cross-language witness found that PowerShell accepted complete cached
identity fields even when C# recovery explicitly marked them incomplete.
Three of four detecting cases failed. A minimal guard rejects true or invalid
recovery markers; the false-marker positive control still admits exact owned
identity. The signed `891fbf58` successor passes the 99-case canonical
virtualization/custody union, with no skips and zero parser/analyzer findings.

Signed C# ServiceHost repair `1762ff95` preserves executable/creation-time
identity and unknown PowerShell state fields, refuses incomplete entries,
retains unresolved children and validates original/staged file identity and
bytes. All 34 maintained custody tests pass; nine TUI tests and eight safe CLI
checks pass separately on the exact private SDK 10.0.401 build. Independent
readback matched 30 source/evidence bindings and all 43 raw TRX cases. Actual
archived CLI failures, staged-file replacement failures and a later null-array
entry failure are preserved. The null-entry repair rejects the complete ledger
before any process effects. Explicit-project format verification with the
actual repository editorconfig reports zero changes for both affected projects;
the hook's generic suppressed diagnostic does not establish a source defect.
No live proxy, installed service or arbitrary-writer compare-and-swap is qualified.

Signed search fixes (`ff1791c`) pass 22 cases using actual installed ripgrep
and private files: literal patterns, bounded content results, per-file log
counts and command failure propagation. Fresh child controls also cover
explicit `NoIgnore:false` and an unfiltered native-helper invocation. Their
synthetic native availability proves routing, not DLL execution. The earlier
matched-output regex benchmark is bound to its recorded source and supports
no general speedup claim.

Signed LLM fixes (`e5cb5fb`) pass 220 canonical cases without skips; the
unchanged logging fixture passes 21 additional cases on 39 hash-verified
private module copies. Real repository logs and configuration remain intact.
Repairs preserve explicit zero options and cancellation, honor initial requests
plus retries, parse finite metrics and embedded JSON, preserve grounded
diagnosis under strict callers and label offline knowledge honestly.
Module-focused coverage is 61.591%, not repository acceptance. Live inference,
full-stack cancellation and retrieval remain separate gaps.

Credential helpers and maintained synthetic fixtures are signed `900a533`.
The canonical 69-case gate passes with exact source/environment readback and
no pending owned process. Separate publication, namespace, legacy and durable
custody gates remain source-bound evidence. Root has not installed or
authenticated this backend. The reviewed two-file payload proposes a nineteen-entry
manifest preserving all eighteen existing entries and their order/properties.
Actual passive staged discovery preserves all 32 legacy callable names; separate
metadata publication and aggregate rollback qualification are pending. The actual
scheduled boot action targets an existing canonical repository script. Five missing
installed copies do not establish that this action is broken. Legacy cache-manifest
ACL compatibility remains unqualified; no private content or live ACL was changed.
Owner/group/DACL qualification excludes SACL/audit.

CargoTools PR 14 merged normally to `e708371`, with verified main/origin
equivalence and custody before retiring the owned branch. Main CI passed 751
cases with eleven skips and thirteen explicitly unrun; current local focused
qualification passed 164. Installed physical consumers were reconciled with
preservation of private files and unrelated wrappers. Fresh PowerShell 7/5.1
checks pass all eight consumer cases, and the installed wrapper returns scalar
Int32 zero separately from useful compiler JSON frames. The T: clone was preserved
and fast-forwarded normally to the same main/origin revision. Its unique historical
TEST_RESULTS.md remains untracked and is independently copied/hash-verified on D:;
that checkout is not claimed clean.

UniFi PR 71 merged normally to `b301be1`; its current-main CI passed the
unchanged 83% coverage gate. Equivalent dependent PRs were reconciled after
preservation. The real Windows native attempt passed 147 mandatory cases,
built its ABI3 wheel on D: and exercised the exact staged PYD; no live Python
consumer was replaced. Reviewed dependency PRs 80, 81 and 82 merged with
passing exact-head CI and bounded private consumer checks. PR 83 merged after
fresh updated-head CI; merged-main run 37918247803 passed all three jobs. Seven fresh
security alerts are actually fixed. A narrowly scoped Rust TLS lock update is
under frozen-graph/resource qualification. Its changed graph passes 222 Rust library
cases with two explicit live Credential Manager skips; the freshly built exact MCP
binary passes all 32 protocol cases. MCP binary unit, Tauri and strict lint gates
remain pending; earlier native passes do not qualify that changed consumer graph.

GitGuard scanner PR 58 merged normally to `8e24529`; exact-head and merged-main
CI passed. Private GNU/uutils predecessor/candidate fixtures agree on all 52
shared cases, and seven extra controls verify matcher failures block commits.
A separate balanced negative-corpus benchmark measured GNU 39.319 to 9.036
seconds and uutils 66.305 to 12.975 seconds. Instrumented invocation counts
fell from 400 to 40; those observations were excluded from timing. This is
bounded scanner evidence, not a fleet speedup. Installed 0.2.12 remains intact.
Release PR 59 is preparing 0.2.13 and current Rust CI, with Windows installer
fixture corrections under qualification. Focused installer checks pass 69 cases
with one explicit Windows Unix-mode skip. The filtered 37-case source suite has
additional Windows harness failures and is not accepted as green. The earlier
unfiltered runner unexpectedly started Ubuntu and completed a Docker image build
before its owned process was stopped. The partial result, image identity and
unknown predecessor identity are retained; no distro, daemon or image was removed.
No hook bypass or installed scanner replacement is claimed.

The owned build-cache relocation verified 3,504 files / 1,819,130,433 bytes at
`D:/pcai-relocation/work-perf-target/r1` before source retirement. A separate
30-sample, 30.95-second observation found mean C: transfer latency 0.333 ms
and queue length 0.033, peaking at one. Observer CPU was 2.16 seconds, about
6.97% of one core. Host-wide paging remained observable without process or
pagefile attribution. This window does not prove relocation cured thrashing.
New owned build/cache/temp outputs use D: on P1. Carbon has no D:/T: volume and
selects its private healthy C: storage from its own inventory.

A current fleet-owner message records stale compiled Cargo output admitted
after an archive extraction at an existing source path: source hashes alone
did not force recompilation when timestamps matched older cache metadata.
Its corrected qualification uses a distinct owned target and requires actual
compilation. PC_AI's proposed inference/media offload therefore binds source,
toolchain, features and target fingerprint explicitly. Rust 1.99 is selected
for this latest-toolchain qualification; the workspace's 1.95 is a dependency
floor, not a toolchain pin. No existing consumer or machine default is changed.
The owner's actual Carbon readback found the named 1.99 toolchain incomplete:
cargo/clippy exist but rustc/rustdoc do not. A complete existing stable provider
was independently identified as 1.99 and admitted for a pinned private proposal,
without toolchain/default writes. CargoTools preflight also reached installed
PC_AI and a live cache despite private environment selection; that actual failure
is retained as a per-host isolation gap. Offload remains held on active-owner
coordination and fresh resource/tool/source admission.

The full run also exposed a test isolation defect: the existing Network unit
fixture mocked Invoke-Command while production called wsl directly. Read-only
WSL network queries escaped the intended boundary. A new actual detecting
corpus fails ten of fifteen cases, including an owned native child printing
HTTP 200 while exiting 7. Signed repair `3e69756` passes the 112-case canonical
union with twelve retained prerequisite skips; existing assertion texts are
preserved and direct native boundaries are isolated. No real WSL/network
operation is part of that gate.

Signed PATH repair `ff43162` moves actual backup writes inside ShouldProcess
and uses literal directory-component matching. Its predecessor failed five of
eight actual cases, including backup writes under WhatIf. The successor passes
30 canonical cases and eight PowerShell 5.1 controls, with actual environment
hashes unchanged. Signed ServiceHealth repair `a78669c` passes 87 canonical
cases and 22 PowerShell 5.1 controls; it rejects failed native stdout, exact-name
lookalikes and malformed bridge counts. Existing assertions are preserved.
Backup-error admission, native command deadlines and live machine policy remain open.

The corrected linked inventory includes the authoritative agent-bus path and its
active/locked worktrees. It observes thirteen registered locations across nine
groups using cached refs; the obsolete agent-hub path remains a preserved unknown
historical location. Its owner's staged files and worktree custody are protected.
This bounded known-location refresh is not complete filesystem discovery or
publication/merge evidence. A nested-import caller-removal failure reported by
another agent also remains a source-bound provider integration lead.

Work remains unreachable through investigated existing routes. Its original
checkout and exclusive recovery ownership are preserved. The owner has an
unlocked credential backend and is investigating existing UDM routes rather
than requiring the user to identify secrets. Its latest reported actual probe
at 10:04 UTC found zero connections across twelve configured Work/NUC/UDM TCP
routes; the saved strict Work SSH route failed before authentication. Tailnet
Work remains offline. Home DHCP/ARP and power state remain unknown; USB phone
access is a coordinated lead rather than a completed credential recovery.
Main integration, current-head CI, real full-repository coverage, required
runtime/model/boot checks, remote readback and fresh active-writer/preservation
checks remain open. No fleet-clean or complete deployment claim is made.

## Current reviewed work and remaining gates

Update cutoff: 2026-10-09T17:31:01.5314121Z. Historical evidence above remains byte-for-byte preserved.
Reviewed signedG source revision `20e0ab8b39edbd68eb955ce1a48566bf21c6e21b` commits root's
benchmark source29404E22/fixtureBC97CB2B after signed capability0b137. At readback
working tree is clean, ahead2 of origin `2d0e6e28f0304ec4e85851163d1c81425d98d110`.
Publication, exact-successor CI and PR182 integration remain **OPEN**; refresh live
refs before promotion because root owns concurrent publication.

[Published2d0 CI37956463393](https://github.com/David-Martel/PC-AI/actions/runs/37956463393)
uses merge checkout981c52fd95d70fe0637e3e909007b8f36363d6ff:2,503PASS0FAIL39SKIP0NotRun,
**FAIL85** at60.2224651803964%,12,669 covered/21,037 commands,8,368 missed,235 classes.
Actual JaCoCo artifact11629795056 agrees. At the same denominator5,213 additional
covered commands would be needed; this projection is not achieved coverage.
No separate NUnit artifact establishes serialized block/container counts.
Rust format/check/Clippy/tests, .NET, PowerShell lint/security and CPU builds pass;
CI Gate fails/integration skips. NVIDIA and Rust Guidelines37956463379 succeed on2d0.
Official verified ripgrep15.2.0 bootstrap passes. Earlier fceHTTP504 occurred before
tests;26360.7426%/one VSock failure remain historical. Codecov protected-branch
upload needs a token; action-step success does not resolve that reporting gap.

Signed0b137 capability source7F43/fixture968 has canonical19PASS0FAIL0SKIP0NotRun,
Pester5.9.1, stable source and zero failed blocks/containers (receipt9F32/XML51B3).
Original11PASS8FAIL and private successor19PASS remain preserved. Its inert native
fixture does not qualify live loaded-parent behavior or meet global85.
Root's benchmark successor fixes the actual StrictMode empty-Sum failure:
private original37PASS1FAIL and successor38PASS remain separate. Fresh canonical38
actually passes38/0FAIL/0SKIP/0NotRun, stable source29404/fixtureBC97 and inert
transports, receiptCBFA0CD5021E9A90590BA6D3E3D5095A1CAD0F7DD0EA1BDCEEDAD7937B104F98,
XML9848B43F478766463A12DFBAE81D780A1059F7D293FD89F3C9209DA8D3584088.
These contracts establish no native speedup or global coverage acceptance.
Warm-r1 actually93PASS1FAIL remains detecting evidence. Actual warm-r2 passes94
parent tests (fixture19/38/25/3/9) and **separate63 child** tests, zero failure/skip/
NotRun/blocks/containers. Receipt114DD28F5B32ED7FA68A72912532662E0C7A8D61CB58A4DA24FD92E6A7827426
and parent XML3E0163CFF574B78730D4A3ADD146F9080C7511E8144325F029CE223CC9225153
agree. All30 live source and27 artifact bindings independently match; source stable,
native before/preload/after empty. Known managed bridge metadata preload and inert
child transport do not qualify native APIs. No union coverage/global85 claim follows.

Signed fce streams/action preferences qualify19 controls; canonical Network/VSock160
passes/one admin-prerequisite skip. Signed f731 requires actual useful-output parity
and native availability before speedup reporting; five native cases are not run.
Token fallback, manifest order/races, reparse semantics and generic parity remainOPEN.
Signed2d0 Performance37 parent and separate63 child pass/Pester5.9.1/source stable/
analyzer0/no native mappings. Desktop5.1 Common import, Linux/ARM Performance,
DLL activation and throughput remain unaccepted. Earlier frozen377 Rust/C# pair
retains exact-source qualification. Carbon additionalCPU8 remains **RAMHOLD0/8**;
B13 MSVC passes separately. Credential32 synthetic/15 guard controls do not establish
live installation, namespace recovery, boot/session or master-database reconciliation.
Model/inference/media/GPU/ARM/Linux/HIL gates remain separate.

Root owns Work/UDM/phone/Bitwarden routes after handoff. Existing cached desktop
Bitwarden is read-only unlocked/metadata31, nonroot Pixel ADB/master app confirmed;
physical phone unlock reply remains pending. Root's authenticated current DTM-home
UDM SE is online9c:05:d6:32:83:81, OS5.1.33/Network10.6.106, ordinary cloud Network
dashboard accessed. The obsolete UDM Pro offline since October2024 is separate.
Work interface history last connected October8 at23:19:10.10.15.215/f8:f2:1e:c6:f9:1f
on aggregation-pro Port1, and192.168.1.133/0c:9d:92:1f:e6:ef on lab PoE-pro Port2.
These are last-known device bindings; actual Work console/SSH/source equivalence
remain **OPEN**, Tailnet100.64.0.2 offline. Actual root read-only terminal commands
now show hostnameudmpro-se, route192.168.1.133 via br0/source192.168.1.1,
neighbors192.168.1.133/0c:9d:92:1f:e6:ef on br0 and10.10.15.215/f8:f2:1e:c6:f9:1f
on br15 bothREACHABLE. ScreenshotA8DC9FA1732CCFDD2F7758BDA0B79E9D52FB1DBC7F9743ABB28D710FD5DFA583
independently visually read confirms current gateway neighbor bindings, not Work
SSH/repo acceptance. Submitted ping/client-location probes await command readback;
no SSH or WoL success is established. P1 self100.64.0.1 and Carbon.6 online;
old ASUS.3/Sparks.4/.5 offline. Local P1 LAN192.168.50.42/gateway.1.
Safe CF metadata642A records3GET200/proxied UDM A162.193.10.247, dtm-cf-rdp DOWN,
fleet-atlas healthy, parked Headscale DOWN. Maintained WAN443->Caddy10443->Headscale8085
architecture expects Headscale HTML. No route correction, Access/provider/database/
global writes or published credential values are claimed. Seven-app email/group
policy/no service policy remains separate read-only evidence.

Cold bus process-only `http://localhost:18480` works; ASUS LAN disappearance/public401
and warmMCP401 remain separate. Owner five-host committed cutover is not independent
whole-fleet acceptance. UniFi final private R9 synthetic protocol now33PASS0FAIL/
0ERROR/0SKIP, producer exit0, sourceStable=true, source98/importedclosure30:
qualificationD254/XMLDF920. This follows preserved R8 actual2PASS30FAIL, cargo command
array-discovery and JSON exception-harness faults. Real file-sharing and caller
controls execute; copied PE/native605 never execute in this protocol.
NormalR5 bare-Cargo zero-body loaderFAIL, earlier false green and external-manifest603
predecessor remain separate. Native observer is a private reviewed proposal only;
RAM below6GiB and unqualified observer/runtime gate mean **NO native GO**.
UniFi primary/lock/TLS alert117 remain unintegrated by synthetic controls.

Preserve WS5dirty4/ignored55, Work19 unique paths and five partial dependency PRs.
Earlier mainFFb92/origin equivalence and exact-bundled owned docs-branch retirement
do not integrate PR182 or authorize foreign cleanup. GitGuard59/CargoTools14 and
installation receipts remain host-specific. Verified3,504 files/1,819,130,433 bytes
(1.694GiB) relocated under owned D: custody do not establish causal C-thrashing relief.
Actual85, exact-head CI/review, remote equivalence, native/consumer/profile/boot/HIL
acceptance and fresh writer/ignored-work checks still gate clearance. No waiver,
hook bypass, force/reset or foreign cleanup requested.

Root readback cutoff: 2026-10-09T17:43:19.0154217Z.

### Root route and resource readback

This readback supersedes the pending route and observer statements above.
The gateway's two-packet ping checks received no responses from either Work IP.
All eight TCP checks (31415, 22, 3389 and 5985 on both interfaces) returned failure.
The verified br0 address is192.168.1.1/24; its neighbor MAC bindings remain distinct
from host-service availability. Headscale reports Work100.64.0.2 offline, last seen
2026-10-09T03:15:47Z. One authorized Wake-on-LAN packet was sent from192.168.1.1
to192.168.1.255:9 for motherboard0c:9d:92:1f:e6:ef: actual102-byte send succeeded.
Follow-up SSH31415/RDP3389 checks on both IPs still failed and Work remained offline.
This is packet-send evidence, not a successful wake or remote source acceptance.
Root visually read retained gateway screenshots r3/r7/r8/r9 under private route
evidence; no firewall, switch, router, credential or BIOS setting was changed.

The user opened and unlocked the Pixel Bitwarden vault. Root observed the normal
vault list and navigated Settings; account/server/sync equivalence and database
reconciliation are still OPEN. The corrected ordinary MAIN/LAUNCHER intent returned
success; the earlier Error3 cause remains unproven. No password/TOTP/database export,
credential modification or private Android data extraction is part of this receipt.

The private native observer now passes seven actual bounded controls, receipt
E3C41C8574070E76E92509E134F178220F1ABB618FC3B898C85778134D723390.
Exit0/nonzero9 and actual child/grandchild RAM-floor/deadline/observation-error
controls confirm retained parent exit, empty owned job and drained output. Log
collision starts no child; synthetic empty-registry refusal proves refusal only.
Earlier path-normalization and three-second startup-control failures are preserved;
the successor ten-second fixture retains its readiness/closure assertions. Production
3600-second deadline is unchanged. This does not qualify real native605: available
RAM3.62GiB and foreign Cargo/Rust producers prevented admission at17:35; fresh17:42
readback has5.30GiB and no observed Cargo/Rust producer, still below the6GiB gate.
No foreign process was stopped. User-authorized additional C: storage work is under
fresh resource/custody audit; no pagefile, cache-policy or system mutation is claimed.

## Accepted launcher measurements and current validation gates

Readback cutoff: October 10, 2026, 15:20 UTC. This addendum supersedes earlier
pending statements only for the evidence below; historical receipts stay preserved.

Independent review accepted twelve cold filesystem MCP sessions on DTM-P1GEN7:
three serialized A-B-B-A blocks, six maintained PowerShell launchers and six paired
native launchers, using the same Rust backend and a private 26-byte read-only
fixture per session. Every session completed initialization, the full tool
catalogue, exact file read and exact allowed-root check. Canonical catalogue
fingerprints agreed despite different raw object-key ordering. All twelve original
bridge operations and the enclosing stage closed with drained streams, disposed
original handles and no pending custody or failures. Evidence is retained in
`D:/pcai-relocation/mcp-live-abba-actual-r1/`: owner SHA256 `F85D96C7…`, stage
`9D32F281…`; independent acceptance is
`.pcai/integration/mcp-live-abba-actual-peer-r1/review.json`, SHA256
`AB564714CC9B0C72A32033C705FF321178CAEFF3DC50BD637745A1F69B221B78`.

| Median across six sessions per launcher | PowerShell | Native | Observed difference |
| --- | ---: | ---: | ---: |
| Original launcher lifetime CPU | 1,382.81 ms | 195.31 ms | 85.9% lower |
| Maximum sampled launcher private memory | 38.945 MiB | 8.447 MiB | 78.3% lower |
| Startup through initialization | 2,225.13 ms | 2,079.76 ms | 6.5% lower median; mixed block directions |
| Tiny read round trip | 83.02 ms | 94.67 ms | 14.0% higher median; mixed block directions |

Launcher CPU and sampled private memory were lower in every block. Startup ranges
overlapped; the native read was slower in two of three blocks. Backend CPU medians
were equal at 39.06 ms, only a few Windows accounting ticks. Sparse nominal 100 ms
samples had gaps exceeding one second, so their maxima are not exact lifetime
private-memory peaks. OS peak pagefile and working-set fields remain separate
metrics and do not establish actual disk swapping. Startup excludes prelaunch
setup. Observer CPU includes protocol, CIM discovery and evidence work; its
pre-result snapshots exclude final serialization, flush and disposal. Full
observer lifetime CPU was not measured, and no overhead was subtracted. These
descriptive whole-launcher results establish no fleet-wide speedup, sustained or
concurrent workload benefit, deployment, consumer configuration or shared-server
reuse. Broader HOLD, blocked-reader, caller-shutdown and recovery gates remain open.

The private Process Lasso parser candidate passed all fourteen unchanged detecting
cases after the preserved canonical baseline produced nine passes and five
failures. Independent private acceptance is
`.pcai/integration/process-lasso-parser-candidate-actual-peer-r1/review.json`.
The maintained function now matches candidate SHA256
`77696D022CD4EF0EF4D200ACC1B778F21C2B6BB6F058ECC73887E540F21767F7`:
BOM-aware config reading and explicit collection at the two comma-list consumers.
Maintained unit execution passed all fourteen cases in PowerShell 7.6.6;
Windows PowerShell 5.1 discovered fourteen explicit skips and zero passes.
The first adapter trial's thirteen passes and one literal-null mock failure
remain preserved; replacing the mock's literal null with no output retained all
fourteen assertion bodies. Independent maintained acceptance is
`.pcai/integration/process-lasso-unit-adapter-peer-r3/review.json`.
Its unit adapter forces only native-command discovery absent; that proof remains
distinct from the private fresh-process run with no mocks. Native execution,
live Process Lasso policy and performance are not qualified by these parser tests.

Further StrictMode validation reproduced thirteen failures in fourteen Evaluation
dependency checks. Missing optional configuration, undefined or stale DLL paths,
wildcard paths, dependency-named directories and malformed optional values were
handled incorrectly. The repair passed the same fourteen real-file controls;
Windows PowerShell 5.1 explicitly skipped all fourteen. Tiny DLL/EXE files are
availability markers and were never loaded or launched. Independent acceptance:
`.pcai/integration/evaluation-dependency-actual-peer-r1/review.json`, SHA256
`92913FA92EE555D6F400490B8625EF338ADDE4D5C42CC8D6FEDF859049D81FD3`.

At 15:37 UTC, the combined working-tree check passed all 375 tests with no
test, block or container failures: parser, acceleration wrappers, real ripgrep/fd
contracts and module loading. The preceding 168-pass/207-failure run remains
preserved. Its separate leading-hyphen fixture failure came from the local
ancestor `.rgignore` excluding log files; the fixture now explicitly uses
`-NoIgnore`, with unchanged expected bytes and assertions. Production search
and the user's ignore configuration are unchanged. Actual evidence is in
`D:/pcai-relocation/process-lasso-maintained-combined-r3/`. Coverage was not
measured by this bounded check; published-current-head CI and the unchanged 85%
coverage gate remain separate requirements.

Milly's read-only persistence audit at 14:56–14:59 UTC found active ordered mounts,
the matching root-owned guard, the original study path traversable by `yayuanli`
and five zero Btrfs device-error counters. Root available space was
623,061,983,232 bytes; both SSD links negotiated UAS at 10 Gb/s. The retained
internal Python interpreter remains a dependency of the migrated study environment;
no application was executed. The initial collection's SSH exit127 remains a
failure; its corrected bounded follow-up exited0. See
`.pcai/integration/milly-mount-persistence-review-r1/review.json` and the maintained
[storage review](milly-storage-and-expansion-review.md). Cold boot, physical drive
removal, Windows access, independent backup/restore and application/HIL checks
remain untested; this audit supplies no additional deletion or unmount authority.
