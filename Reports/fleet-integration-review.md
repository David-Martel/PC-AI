# PC-AI fleet integration review

Evidence cutoff: October 9, 2026 UTC. Owner: Codex integration lane. Live receipts and
private build outputs remain under `.pcai/integration/`; private profile and hosts
originals remain outside Git. This report records verified work and outstanding
gates rather than declaring a fleet clean before those gates finish.

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

Readback: October 9, 2026, 13:18 UTC. Root's current signed PC_AI head is
`3f1df889620c10e9720d77e695da9dd524668793`; published PR 182 remains
`c8e591e2edfd2d4c059e9433e77a79fc999eb796`. Integration remains active.

The actual published-head [CI run 37931175131](https://github.com/David-Martel/PC-AI/actions/runs/37931175131)
completed with 2,373 passing tests, four failures and 58 skips. Coverage is
58.58% of 20,713 commands, **FAIL** against the unchanged 85% requirement.
Rust format, check/Clippy and tests, .NET build, PowerShell lint, security scan,
CPU deployment and llama.cpp CPU build passed. Downstream integration was
skipped. All four failures occurred in the inference configuration custody
fixture because the Windows runner had no `T:` drive: candidate construction
threw before reaching a valid repository-local executable. The ripgrep and
scanner readiness repairs passed in this actual hosted run.

Signed `3f1df88` constructs candidate paths with `System.IO.Path.Combine`,
preserving the existing search order and subsequent existence checks. The
maintained configuration/virtualization union passes all 70 cases. A controlled
private variant changes only the three literal cache prefixes to a currently
absent drive: the predecessor passes one case and fails four; the successor
passes all five. This is an explicit source variant, not unchanged-source
no-`T:` qualification. An attempted process-local drive-removal preflight was
invalid and is preserved separately; no physical drive or mapping changed.
The tiny owned native JSON reader qualifies configuration custody, not real
model inference. New-head hosted CI remains required.

Signed `d82af157` also closes a false-green validation boundary: a genuine
Pester `AfterAll` failure left one passing test, zero failed tests/containers
and 100% coverage, which the predecessor gate accepted. The gate and standalone
test runner now reject failed blocks. All 19 maintained gate cases pass without
skips or parser/analyzer findings. The original fixture failure and a later
runner-only empty-array accounting error remain preserved. Test selection,
coverage exclusions and the 85% target are unchanged.

The VSock successor remains private. It refuses overwriting existing backups,
captures registry absence and original value types, stops on capture/write
failures, checks actual native exit status and reports restore errors. Its 24
cases pass separately on actual PowerShell 7 and Windows PowerShell 5.1. The
original six failures and the additional four predecessor failures are retained.
The existing Network fixture's 93 unaffected assertions remain in order; its
11 administrator cases must genuinely execute in the proposed union rather
than stay skipped by a discovery-scope error. Complete legacy-union and restore
admission review precede tracked integration. No live registry/netsh/WSL tuning
or performance improvement is qualified.

Credential deployment also remains private. An actual second aggregate could
publish while the first held only the manifest lock; recovery also changed
inherited original metadata. The revised target-set lock precedes all capture
and copy operations, and guarded success/recovery retains original metadata.
All 26 private synthetic cases pass; the detecting 21-pass/two-failure run is
preserved. Explicit synthetic `C:` targets with `D:` copied custody are now under
qualification. Original/displaced file identities and external-copy identities
remain distinct. Installed code, authentication, caches, sessions, live ACLs and
tasks are unchanged. Legacy cache privacy compatibility and a reviewed live
installation window remain open.

GitGuard PR 59 is signed and clean at `660e2d55`; all 13 reviewed source hashes
match its primary Windows run: 812 passes, zero failures, 24 skips. The local
optional-backend case was not run. Its actual current-head Linux native and
Docker CI both passed in run 37935484416. The PR merged normally to `97e64048`;
the primary checkout is clean and equals origin/main. Merged-main CI, release
and immutable installed-version transition remain separate. Installed version
0.2.12 remains unchanged.

UniFi's private updated Rust TLS graph passes 222 library tests with two live
Credential Manager ignores, 30 MCP unit tests and 32 actual protocol cases,
as separate runs. Its Tauri check reached strict Clippy and failed on eight
existing findings in six source files. A finite private six-file repair preserves
the public frontend command arguments and is awaiting normal qualification.
The TLS lock is not promoted and alert 117 remains open. Earlier dependency
PRs 80–83 retain passing merged-main CI.

New build, cache, temporary and test outputs remain on qualified `D:` storage.
The independently verified relocation preserves 3,504 files/1,819,130,433 bytes.
`F:` is file-backed and `W:` was absent in the recorded inventory; those paths
are not assumed suitable on other machines. The bounded disk observation did
not establish sustained `C:` thrashing or a relocation cure. Carbon's dynamic
storage/provider admission remains its exclusive owner's work; its additional
eight inference/media CPU stages have not yet run. Work remains inaccessible
through the last actually tested configured routes; power state is unknown.
Exact final-head coverage/CI, real native/model/profile/boot acceptance, remote
readback and fresh preservation/active-writer checks still gate fleet clearance.
