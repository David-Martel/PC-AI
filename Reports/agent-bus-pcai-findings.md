# PC_AI agent-bus findings and integration custody

Evidence cutoff: October 9, 2026, 14:05 UTC. This is an active integration
record; it does not certify a clean fleet, complete coverage or final deployment.
Root owner: `codex-p1-pcai-integration`.

## Evidence scope

The live 1,440-minute query reached its 500-message limit: its actual returned
interval was 01:36:42–03:40:36 UTC, with 98 related messages. Separate seven-day
queries for both repository tag spellings returned 71 unique messages spanning
October 2–9. Sender-specific Clarius evidence supplements these reads. The
PostgreSQL export returned only 111 messages and ended October 8 at 20:37 UTC;
it is sparse history, not a complete live-bus export. Private original bodies,
IDs and capture boundaries remain under `.pcai/integration/agent-bus-*`.
The separate 60-minute refresh captured at 07:18 UTC records current ownership
and continuing Work route failures. Historical table entries below retain their
original reproduction boundaries; the current successor readback follows them.

## Findings, disposition and remaining acceptance

| Failure mode and originating evidence | Verified disposition | Owner / next action |
| --- | --- | --- |
| CI accepted 53.13% coverage against 85%, with 1,532 passes and 52 skips. Carbon message `01a11ea3-e116-71b3-889e-a2abb5d990b7`. | Confirmed false-green admission defect. Pester `Run.Exit` terminated before caller assertions. Signed commit `5b766013` captures results, explicitly enforces the existing target, rejects absent/invalid coverage, zero tests and failed containers. Eighteen actual tests passed, including fresh-process below-target failure and full-coverage success. | Root: rerun immutable final-head CI and repair any resulting real deficits; keep PR 182 held until genuinely green. |
| Coverage globs selected only 18 files despite recursive module source. Carbon messages `01a11eae-1dd7-7268-9d4d-ef839aa5828c` and correction `01a11eaf-29fe-77ba-bba0-c373d7a3142b`. | Confirmed inventory defect. The first 237 count included three test files; corrected production inventory is 234. Directory-based recursive coverage now includes nested Public/Private functions and flat wrappers. Actual pinned-Pester inventory tests verify the selection. No target or denominator exemption was added. | Root/Carbon: measure full final-source coverage; the separate 108-pass, 94.67% four-file fixture run is useful bounded evidence, not overall coverage. |
| Installed disk tool emitted no output for 300 seconds under paging. Clarius message `01a11ebc-40ed-7793-8f86-25e23f57d3fd`. | Reported runtime failure; installed executable independently identified as 476,672 bytes, March 11 build, SHA256 `3570F1D40DA4661D562BD54A37D33281BC21A89AA064AE72EB7F99D2CC1BE160`. Current disk source still uses recursive traversal and logical `metadata.len`, discards traversal errors, and lacks an MFT/USN path or partial-result deadline. This historical binary is not the pending native candidate. | Root/native: qualify fresh paired output, bounded cancellation and useful-byte/allocated-byte semantics; preserve fixtures and benchmark before adding an MFT path. Do not infer speedup from different traversal policies. |
| The disk consumer swallowed fatal CLI/worker errors and started another backend scan. Root/profile actual production-pipeline fixtures. | Confirmed: predecessor controls passed while five timeout/custody cases failed with a downstream scan. Signed commit `dfd9390` rejects timeout, cancellation, bundle/protocol corruption, truncated or malformed frames and pending custody, preserving original errors and exact retained objects. Actual 41/41 tests passed without skips, including the unchanged seven predecessor/control cases, real frame/bundle checks and legitimate unavailable/legacy fallback. | Root: finish final coverage/native qualification. Default native, dust and parallel scans still have no overall deadline; this repair does not bound the entire disk API. |
| Native process compact layout disagreed with its managed reader, and JSON fallback concealed it. Independent root review while reconciling reported sampler work. | Confirmed: C# packed entries at 44 bytes versus Rust's 48, shifting memory from offset 40. Signed commit `2b6f95b` corrects layout and validates counts before allocations. Eight direct parser tests passed; predecessor had two actual failures. Signed `37743c1` initializes Rust padding and reconciles the sampler/CLI; its exact native qualification is pending. | Root/Carbon/profile lane: compile/test the immutable exact-source archive and fresh managed bridge, then run the real native pair; historical DLLs do not qualify new bytes. |
| Failed worker/process/pipe disposal lost exact custody; a later script invocation could launch a replacement despite failed closure. Profile lane detecting fixtures. | Confirmed predecessors failed detecting tests. Signed `b249cca` roots exact state before process launch, preserves origin across retries, distinguishes active state from pending failed closure, and refuses unknown registry schema. Focused 24/24 and combined portable 60/60 tests passed; three actual-native pair cases are explicitly skipped. Independent review admitted the documented child/runspace contract. | Profile/root: complete actual native parity/cancellation qualification. |
| Hash benchmarks accepted duplicated correct rows while omitting expected files. Independent dependency review. | Confirmed in the real predecessor case pipelines on both transports, with four detecting failures. Signed `b249cca` requires the exact unique normalized expected path set and valid correct SHA256s. The updated complete benchmark suite passed 24 tests separately from the older 60-pass union; no invented combined-run or speedup claim. | Root/benchmark: measure the qualified actual native pair with matching useful output. |
| Configuration serialization truncated unrelated deep settings; non-cooperating replacement races and partial Windows replacement effects required custody. Dependency lane detecting fixtures. | Signed `092565f` preserves captured originals/displaced bytes, rejects serialization warnings and publishes memory only after success. Historical focused 82/82 tests passed. The later full run exposed a genuine ACL recovery failure: moving an original into a protected custody parent can change its inherited/protected descriptor before the recovery code captures it. A private D: reproduction fails the unchanged strict SDDL assertion; its C: control passes. | Profile/root: capture metadata through the exact original handle before publication, bind it to displaced identity, and qualify strict C:/D: recovery before admitting a successor. POSIX publication remains a separate compatibility gap. |
| Installed CargoTools looked loaded inside a helper but its commands were unavailable to the caller, permitting old-module autoload. Carbon import review and subsequent retractions. | Confirmed import-scope/provenance defect. The alleged source-01b raw-pin/List-copy defect was retracted by `01a11ea9-4628-76f2-98da-e51fd030084c` and `01a11eaa-c53b-747b-b508-7fce5249ed8f`: direct source raw/stable controls passed. A repaired alias check exposed another actual failure: loaded 0.8 commands were accepted after the manifest at the same path became 0.9 (`01a11ee3-6cc7-7311-bdd4-adc945ce5f88`). Both private predecessors remain held. | Carbon/root: require selected-manifest, loaded-module and actual caller-command version equality along with alias/provenance controls. Preserve stale-module state without forced unloading; admit only the qualified successor. |
| Legitimate OneDrive Cloud reparse metadata was rejected by the development-module installer. Carbon `01a11eac-ac1e-737f-a4d8-a8bbc9c33930`. | Confirmed on Carbon and P1. Signed commit `a1f0d79` recognizes the bounded Cloud-tag family while retaining no-follow checks and junction/unknown-tag rejection. Actual primary 71/71 tests passed, including existing transactions, real P1 Cloud metadata and non-mutating dry-run. Removing only OPEN_REPARSE_POINT caused the real-junction test to fail as expected. The bootstrap is unchanged. | Root/Carbon: qualify guarded consumers; no Cloud-backed publication is implied by metadata-only acceptance. |
| Process Lasso UTF-16 configuration and memory-pressure visibility. Clarius `01a11ebc-40ed-7793-8f86-25e23f57d3fd`. | Current Rust loader already handles UTF-16LE/BE BOMs; an absent-decoder fix is unwarranted. Full sections preserve memory-priority rules, while the typed summary's policy visibility needs review. Clarius reports persistent governor rules and a backup; root has not changed that foreign-owned policy. | Root/boot: retain existing evidence and measure intervals/observer overhead before tuning. Reported process totals do not by themselves prove waste or unnecessary duplicate consumers. |
| C: paging and competing native jobs caused genuine runtime and unchanged-deadline fixture failures. Clarius `01a11dc5-fbf7-77cb-8960-410e2964212f`, plus owned build receipts. | Root observed 0.51 GiB free RAM, later 1.86 GiB after the obsolete owned cache warm stopped and UniFi Clippy completed. D: is a separate healthy NVMe; F: is a file-backed VHD and W: is absent. Only the released root-owned 1.694 GiB build cache is admitted for verified relocation. | Root/closeout: use direct D: targets and unchanged test deadlines. Do not attribute all pressure to one process or equate cache movement with fixing RAM paging. |
| Work became unreachable after earlier successful build/profile/auth qualification. Fleet-owner receipts; Clarius `01a11ebc-6137-7620-9597-9493a1052136`. | Historical strict SSH, real CPU inference and profile/alias readbacks remain valid for their recorded source. The later outage does not prove those changes caused it and prevents final exact-source Work acceptance. | Exclusive fleet owner: investigate existing routes, preserve Work's 19 unique source paths and complete final-head validation when access returns. |
| GitOps monitor reports `inspection=Unavailable` with workflow/upstream counts. Repeated tagged bus messages. | Such reports are not CI success, upstream parity or clean-checkout evidence. Direct GitHub run logs and exact Git refs are used for acceptance. | Monitor owner/root: investigate inspection blindness and reconcile fresh direct inventory before cleanup. |
| Jules review-only sessions produced pull-request outputs. Ten historical tagged reports from `claude-asus-jules`. | Historical policy tripwire, not proof those changes were admitted. Current open PR disposition must be independently reconciled with exact session/output identities. | Root/review owner: verify history and PR custody before merge or closure; do not dispatch new overlapping sessions. |
| Installed git-guard 0.2.10 could delete a branch checked out at an unchanged tip or miss late ignored work. Root actual detecting fixtures against immutable `5492c4f`. | Confirmed. Reviewed PR 55 merged at `e46d408` after 74/74 hygiene tests and both PR and merged-main CI jobs passed. Signed immutable 0.2.11 is published and normally installed on P1; installed engine SHA256 matches the reviewed source. Installed BLOCK/clean-commit controls passed. The owned integrated branch was retired normally. | Root/fleet: use fresh report/dry-run plus individual custody and preservation before cleanup. No atomic exclusion against arbitrary external writers is claimed. |
| Native source archive admission used working-tree hashes for two differently serialized ZIP entries. Root manifest r1; Carbon raw-hash rejection. | Root corrected this error, preserving r1. Successor r2 records actual ZIP hashes separately from working-tree hashes. Complete supplement inventories all 2,437 raw paths and lengths; an independently regenerated archive from signed `37743c1` reproduces the entire original ZIP SHA256. Git attributes explain distinct stored canonical blobs and raw archive serialization; they are labeled separately. | Root/Carbon: require exact raw extraction binding before compilation. Original ZIP `3FD891...` is unchanged; normalized text equality alone does not qualify compiled bytes. |
| Credential helper retained failed process/task custody but permitted a subsequent invocation to launch another child. Root review of immutable helper `8EDCEC...`, message `01a11ef7-df07-7559-a6ac-6dbebf360220`. | Genuine first-child-pending/second-child-marker failure reproduced against the predecessor. Private successor `68B7C7...` blocks the second admission and preserves exact recovery. Independent pinned-Pester qualification passed 100 cases after one documented private fixture scope correction; the original 99-pass/1-fail result is retained. Twelve maintained synthetic custody/import/manifest fixtures passed, including real blocked stdin task retention. | Dependency/root: finish source review and the separate credential-file publication gap before deployment. Preserve private credentials, installed eighteen-file manifest and original receipts; no installed backend changes. |
| Corrected full coverage exposed previously unexecuted failures. Exact PR 182 head `c6f5222`, CI run `37888041832`, and local source-equivalent recursive run. | CI: 1,746 passes, three failures, 58 skips, 37.33% coverage against 85%. Local: 1,751 passes, three failures, 53 skips, 38.347% against 85%; all 241 captured source hashes remained unchanged. Six ignored Archive files add 454 local commands; CI analyzes the 234 maintained source files. Two manifest-root failures now pass in a focused 14-case root/copied-consumer run; the third is the ACL failure above. | Root: retain the 85% gate, repair real behavior and qualify additional meaningful fixtures. Integration CI was skipped after the failing job; other passed jobs do not establish complete acceptance. |
| Baseline snapshots saved the run summary in the metric-distribution field, concealing real performance regressions. Root detecting production save/read fixtures. | The unchanged predecessor produced two genuine failures: missing persisted latency means and a false no-regression result for latency +20% / throughput -20%. The reviewed successor stores actual distributions and a separate summary; both save-and-compare tests pass. Initial fixture binding/type errors are retained separately and are not the production reproduction. | Root: integrate the reviewed successor and broaden baseline lifecycle/zero-baseline validation. This synthetic producer fixture does not qualify a live inference backend. |
| Evaluator fractional scores were silently rounded by integer-first Math overloads. Root strengthened repetition and numerical score fixtures. | Actual `Math.Max(0, 0.72)` returns 1 on this PowerShell host; floating-point `0.0` retains 0.72. The same defect rounded throughput, memory, accuracy, toxicity and groundedness. Correctly bound predecessor fixtures produced eight genuine failures and four passing controls. Reviewed floating-point overload repairs pass all twelve cases and the complete 46-case evaluation/provider/baseline union without skips or source changes. | Root: integrate the reviewed source and safe exact evaluation/provider suites into CI. These are deterministic software contracts, not clinical validation or live inference acceptance. Invalid/nonfinite metrics and the separate A/B probability formula remain open. |
| Credential-file publication changed original permissions before a failing replacement. Independent dependency fixture against private helper `68B7C7...`. | A real staged-file disappearance caused the unmodified replacement to fail with HRESULT 0x80070002 after Set-Acl. Synthetic original bytes remained unchanged, but strict original SDDL changed and added SYSTEM/Administrators access. The detecting case failed; the separate later-writer preservation control passed. | Dependency/root: preserve exact original metadata before mutation, retain displaced custody and qualify guarded recovery on synthetic private files before canonical deployment. No live credential files were modified. |
| A/B inference used a linear formula rather than a Student t probability, declaring overlapping two-element samples significant. Root actual numerical fixtures. | Original twelve cases produced two passes and ten genuine failures. A two-sided Welch successor passes32 statistics cases, including analytic values, four published SciPy references, zero/nonfinite/alpha controls and common rescaling. Two additional detecting cases caught a representable Cauchy tail lost to squaring overflow and an infinite relative summary; both now pass. The actual combined union passes78 without skips or source drift. | Root/review: complete independent review and signed integration. No claim about independent sampling assumptions, model quality or clinical acceptance follows from algorithm tests. |
| CargoTools returned cargo JSON stdout together with preflight exit0, causing a successful native build step to be rejected. Actual UniFi Windows wheel attempt and private real-child witness. | The attempt failed before DLL/wheel publication with unchanged source and consumers. Both actual exit0 and101 witnesses return mixed arrays in the predecessor. Caller-selection22-case qualification is separate and cannot repair this stream/status failure. | Profile/root: preserve scalar status, replay useful stdout once, qualify blocking/nonblocking/RA controls and retry the actual build with normal preflight. |
| Credential replacement merged a broad predecessor ACL into newly published secret bytes before tightening. Dependency real-Replace synthetic Everyone-read witness. | Private predecessor `D348886...` failed the detecting privacy assertion; actual replacement occurred once. Its earlier27/100 passes do not cover this boundary. Admission-only successor refuses unsupported foreign/null allow ACLs before custody or secret staging. | Dependency/root: independently review fail-closed admission, disposition compatibility failures honestly, and qualify private inherited success/recovery separately before canonical source or installation. |

## Historical successor readback (before 09:20 UTC)

The config ACL successor is signed commit `245282d`: 93 combined checks and
46 per-volume C:/D: controls pass, including exact descriptor and distinct
same-byte file identity cases. Its raw metadata scope is owner/group/DACL,
excluding SACL/audit. The score/baseline repair is signed `3029ffe`.

The A/B successor is independently reviewed, signed `8bede69`. Seven additional
detecting fixtures reproduced multiple effect labels from overlapping switch
predicates. It now passes the full 85-case evaluation/provider/baseline/statistics
union without skips, source changes or parser/analyzer findings. Exact earlier
RED runs remain preserved. Equal-weight RMS effect size and independent sampling
assumptions remain explicit; the software gate is not population acceptance.

The frozen CargoTools successor passes 96 focused native-stream, routing,
separator, caller-provenance and existing contract checks plus actual Windows
PowerShell 5.1 controls. Full repository QA is running. A separate read-only
witness also reproduced an empty-separator null AddRange failure in LLM output
argument construction; its narrowly scoped detecting repair follows that freeze.
Neither focused success qualifies the previously failed real UniFi wheel build.

A real credential-parent substitution fixture exposed synthetic secret bytes
under a broad replacement DACL before displaced-file rejection. The private
guarded successor now admits existing parent/ancestor authority before secret
staging. Its publication 29, durable 12 and supported-profile legacy 100 gates
pass separately; original broad-input incompatibility remains recorded. Eleven
namespace controls also pass. Read-only live metadata admits the profile and
`.bwdata`, while refusing default TEMP, `.machine`, `.local/share` and D: temp.
Consumer-path integration is under review; no live ACL repair, authentication,
credential publication or installation occurred.

Carbon's private Windows CPU build verified all 2,437 archive source files and
pinned Rust1.99 tools. It passed 122 tests, with two explicitly ignored cases,
strict Clippy and release compilation, without source changes. The real
positive-CPU busy-child check passed within its original deadline. Transferred
DLL/CLI hashes matched. The fresh managed build on P1 D: with SDK10.0.401/latest
compiler passed 26 tests, with one explicit media-fixture skip. Actual paired
PowerShell qualification passed 47 cases, including all three native consumer
cases; one filesystem short-alias case skipped because aliases were unavailable.
The native export reports `0.1.0+37743c1`. The matched twelve-small-file SHA256
benchmark passed useful-output validation on all three transports. Five samples
per backend averaged 5.57 ms for PowerShell, 29.36 ms for the persistent worker
and 91.79 ms for the direct CLI. PowerShell wins this fixture; no broader native
speedup, large-file or fleet-performance claim follows. Hash-bound source,
artifacts and receipts are recorded in `Reports/native-pair-qualification.json`.
No installed consumer or machine default changed. Latest Work route attempts
remain unsuccessful and preserve its recovery ownership and unique checkout.

## Storage and operational boundaries

The user's temporary-storage instruction authorizes reviewed off-C staging.
The selected build-cache mover requires PowerShell 7.3 or later, owner release, fresh slash-normalized
process checks, contained ordinary paths, no Git-store ancestry and complete
SHA256 copies. It records durable outside-source custody before deleting only
matching source files; directory retirement is empty-only. Existing receipts or
changed source bytes cause preservation. Its dry-run and WhatIf modes write
nothing. Its thirteen actual pinned-Pester tests and independent review passed.
The actual relocation completed: 3,504 files / 1,819,130,433 bytes were verified
again at `D:/pcai-relocation/work-perf-target/r1`, and the original source was
retired. The outside-source manifest is `r1.relocation.json`, SHA256
`6C7A086F882A3AD9D7CBF91FF9EBEBC7271EF1ECCEDE12BB2812E2A212C8F7E8`.
The earlier dry-run total differed by 20 bytes; the actual capture and readback
agree. No claim of immutability between those observations or paging improvement
is made. Future P1 builds should select qualified D: paths directly; other hosts
must select storage from their own volume and workload inventory. Historical warm receipts and
producer files retain their original provenance.

Clarius's pagefile, cloud-client, WSL and Home Assistant migration proposals are
recorded plans, not completed actions by this lane. Live Models, cloud roots,
foreign caches, services and persistent machine defaults remain under their
specific consumer/owner qualification. Original model/boot evidence and active
WIP are retained. Final integration still requires exact-source native gates,
real coverage, reviewed PR disposition, merged-main CI, fleet readback and fresh
preservation/active-writer checks before branch or worktree retirement.

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

Root readback cutoff: 2026-10-09T17:43:18.7767673Z.

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
