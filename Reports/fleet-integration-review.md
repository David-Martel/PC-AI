# PC-AI fleet integration review

## Published head and remaining validation (14:16 UTC)

Signed commit `5180c160fac42b510d05de971e83d3921b035b99` is published on
`fix/fleet-runtime-integration`; PR182 remains a draft. Independent published
delivery review verified all24 payloads against raw HEAD blobs and the preserved
checkout, with48 cached text/EOL settings unset. Receipt:
`.pcai/integration/cache-corrective-published-delivery-peer-r1/review.json`.

Completed CI run38144048210 passed2,848 tests, skipped40 and failed no tests.
Its source-bound merged coverage is14,560/21,817 commands (66.736948%). The
unchanged85% gate failed; the current inventory needs3,985 additional genuine
hits. Parent-only Pester coverage and merged coverage have different numerators;
neither is an acceptance waiver. Physical artifacts are retained under
`D:/pcai-relocation/cache-corrective-current-ci-r1/`. Rust, C#, formatting,
security and CPU build jobs passed in that run; integration was skipped. The
separate Rust-guidelines workflow remains in progress at this cutoff.

The Context7 HTTP-header helper passed independent Carbon build review: six
serialized offline operations and eight compiled synthetic groups, with exact
source/DLL/PDB pairing. Review:
`.pcai/integration/context7-header-execution-actual-peer-r1/review.json`.
This establishes compilation and synthetic behavior only. Real credentials,
Codex consumption, connection reuse, loaded helper CLR and matched performance
remain unqualified. Production MCP configuration is unchanged.

Timeout harness R2 failed before any child or HTTP operation because an empty
original operation list was collapsed by PowerShell expression enumeration.
Reviewed R3 preserves that reference, but its first child exited1 at the stdin
readiness guard before the HTTP actor. All ten native closure flags and both
owner writers closed; zero native custody remains pending. No HTTP case passed,
and no production REST timeout change is adopted. Both failures and original
errors are preserved; R3 files are under
`D:/pcai-relocation/llm-timeout-failed-actual-r3/`.

New public Evaluation controls remain unpublished. A real manifest import
returns both the validator and root module objects. Independent source review
also found cleanup could miss those new objects if selection fails before
ownership is recorded. Private correction and genuine execution remain open;
the fourteen case bodies are preserved. Source HOLD:
`.pcai/integration/evaluation-public-contracts-source-peer-r3/review.json`.

Fresh Milly readback confirms both external SSDs use UAS at10Gb/s, the migrated
study filesystem is mounted, and the internal filesystem has about580GiB free.
The study owner can resolve the original recording path and read its server
file. Cold boot, removal/reattachment, backup restoration and application/HIL
acceptance remain separate gates.

Agent-bus findings remain evidence leads with explicit failures: the Clarius
consumer lane reported native exit101 and a second kernel operation NOT_RUN,
with owned resources retired; the vigil-system mirror reported non-fast-forward
divergence and preserved foreign branches. Neither establishes PC-AI runtime
success or authorizes deleting another lane's work. Each repository's owner
must reconcile its exact source and runtime evidence before integration.

Evidence cutoff: October 11, 2026, 13:22 UTC. Owner: Codex integration lane. Live receipts and
private build outputs remain under `.pcai/integration/`; private profile and hosts
originals remain outside Git. This report records verified work and outstanding
gates rather than declaring a fleet clean before those gates finish.

## Current public controls and Git delivery

The three current public suites passed50 tests with no failures, one explicit
Redis skip and no unrun tests under PowerShell7.6.6/Pester5.7.1. They exercise
cache semantics, real managed file enumeration and the canonical LLM request
builder; HTTP, health, native Ollama and router boundaries remain mocked.
The six instrumented canonical source leaves executed665 of2,146 commands
(30.99% locally). This narrow denominator supplies no repository-wide CI or
live-inference acceptance. Source pins for the three tests and six measured
leaves were checked before and after. Other imported collaborators were not
contemporaneously pinned by this runner. Independent source/actual review:
`.pcai/integration/llm-provider-routing-public-peer-r3/review.json`; physical
results: `.pcai/integration/current-public-cache-enumeration-routing-actual-r2/`.

The disk-usage fallback now checks `Top` before probing native availability.
The one-line guard preserves positive-Top behavior and avoids that probe for
Top0. Eighteen public enumeration controls and four matching ABBA row sets
passed; the unchanged output assertions establish parity, not a speedup.
Independent review is
`.pcai/integration/file-enumeration-public-controls-peer-r2/review.json`.

All22 current register-bound files, the register itself and the public peer
receipt retain identical SHA256 and lengths in the working tree, staged index
and preserved Windows checkout. All48 cached text/EOL observations are unset.
The old23-path delivery review retains its original predecessor scope. Current
independent review is
`.pcai/integration/cache-corrective-git-delivery-peer-r1/review.json`;
checkout evidence is `D:/pcai-relocation/cache-corrective-git-delivery-actual-r1/`.

The first current-cache execution wrapper failed before the maintained entry:
Windows Join-Path emitted backslashes while its child compared a forward-slash
literal. The original FAIL, source-review miss and exact three output files
remain preserved in
`D:/pcai-relocation/maintained-current-cache-failed-actual-r1/`.
A reviewed successor normalizes the producer before JSON and retains the whole
strict child guard. Its actual Carbon run passed18 capture controls and21
full-module cache cases with one Redis skip; root and child closure checks
returned successfully. Independent actual review accepts the physical XML rows,
source ancestry and Root-attributed original-object closure. The review is
`.pcai/integration/maintained-current-cache-execution-actual-peer-r2/review.json`.
These results concern current source, unlike the earlier10:40 predecessor.
The retained actual files are
`D:/pcai-relocation/maintained-current-cache-actual-r2/`.

The genuine runtime attribution comparison passed the same25 whole controls
in both phases. Resolving the fixture module path before parsing restored313
hits out of the unchanged760-command inventory; its lexical-path baseline
recorded0. Within InstallRuntimePair,147 of191 commands executed and44 remained
missed. The candidate's41.1842% is a local diagnostic, below85%, with no global
CI credit. Every paired verbatim/span identity matched, without lost hits.
Independent actual review:
`.pcai/integration/runtime-integrity-coverage-attribution-execution-actual-peer-r6/review.json`.
Root then adopted exactly the reviewed one-line ProviderPath correction over
the verified whole preimage; all eighteen test bodies/25 expanded controls
remain unchanged. Both earlier timestamp/DTD failures remain preserved. No
production resolver code or XML exporter was changed by this fixture repair.

A separate source audit found eight REST calls without a finite stalled-body
read timeout. The existing TimeoutSec setting bounds the request/header stage;
OperationTimeoutSeconds separately bounds gaps between body reads. A proposed
feature gate preserves PowerShell7.0 compatibility and existing timeout values.
This is a per-read stall bound, not a total-call deadline. No production timeout
edit or loopback behavior is qualified yet; see
`.pcai/integration/llm-http-timeout-gap-source-r1/review.json`.

## Earlier cache corrective validation and predecessor CI (12:28 UTC)

The reviewed predecessor was signed and published as `b1cc1a1e2005343f80fc0ec290068f9ec0d748cc`.
PR182 remains a draft. Its completed CI run38133318595 passed 2,809 tests with
40 skips and no test failures, but executed only14,209 of21,769 instrumented
commands (65.271716%). The unchanged85% gate failed; reaching it requires4,295
additional genuine command hits against that inventory. The other completed
Rust, C#, security and formatting checks passed. These are predecessor-head
results, not acceptance of the subsequent working-tree repair.

Additional cache controls reproduced nine defects against the unchanged
predecessor. The repaired helper preserves empty, singleton and nested array
shape and supported mutable copies; records setter expiry so a later getter
cannot extend it; and returns a miss without deleting an external record that
another writer may have replaced. Legacy records without expiry retain legacy
age behavior. The corrected ordinary unit fixture forces memory before import,
restores six original process environment values, and makes live Redis testing
an explicit opt-in with bounded probing and a GUID-owned prefix.

The maintained public cache suite passed21 cases and explicitly skipped the
live Redis case. Frozen predecessor execution passed12 and failed9 with the
same skip. An initial positive-run container syntax error remains preserved;
its corrected successor has no failures. Actual evidence is
`.pcai/integration/cache-correctness-root-actual-r1/`; distinct source/test review
is `.pcai/integration/cache-correctness-regression-tests-peer-r2/`. No live Redis,
native process capture, fleet installation or equivalent-semantics performance
acceptance is claimed by these cache results.

The three source registers now bind the current helper and22 expanded cache
controls. `Tests/ProcessCapture/source-review.json` supplies the independently
authored portable source approval without requiring private checkout evidence.
It remains outside the register to avoid circular hashes and has its own raw
Git attribute. Fresh maintained eighteen-capture and full-module execution
against these new pins is pending; the10:40 receipts below retain their original
predecessor scope.

A separate Carbon attribution diagnostic failed before Pester began: default
PowerShell JSON decoding converted an ISO timestamp into `DateTime`, causing
the unchanged exact identity guard to reject it. The original failure, all five
output files and clean native/local closure are retained in
`D:/pcai-relocation/carbon-attribution-failed-actual-r4/`. A separately reviewed
successor must preserve timestamp strings rather than weaken that guard. No
test or coverage credit resulted from the failed diagnostic. Its string-date
successor admitted the original identity and ran the baseline:25 tests passed,
with0 executed commands of760 analyzed. The diagnostic then failed because
Pester's XML exporter appended an empty internal DTD subset `[]` to the pinned
external declaration. All native and local resources closed; the original
failure remained sticky. Its106 output files and two readbacks are preserved in
`D:/pcai-relocation/carbon-attribution-failed-actual-r5/`. The candidate did not
run, so the pair, attribution repair and CI coverage remain unaccepted. The XML
reader successor must allow only the observed exact empty-subset form while
retaining entity refusal and disabled external resolution.

The private native MCP probe compiled on Carbon using the admitted .NET8 SDK;
distinct review accepts source/DLL/PDB pairing only. The new Context7 HTTP
header helper currently has source approval only. Its compilation, genuine
Codex client behavior and matched performance comparison remain pending.
No current MCP registration or another agent's environment was changed.

The bounded P1 workload profile observed median commit86.916%, above the
qualification budget; heavyweight compilation remains offloaded. Observer CPU
was1.876%. Aggregate disk latency and process counts do not establish a
C:-specific bottleneck or wasted memory. Useful agent, sync and study processes
remain preserved. Milly's latest read-only check confirms both SSD links at
UAS10Gb/s, about580GiB internal free and owner-readable migrated study paths.
Compression previously recovered13.84GiB. M.2 vacancy and cold-boot, removal,
portable-access and independent restore checks remain open.

## Earlier cache adoption and delivery (10:40 UTC)

The predecessor maintained cache entrypoint passed all eighteen capture controls and
full-module QA on Carbon. Full-module import exposes twenty exports; import and
memory paths leave the native capture type unloaded. Eleven existing cache tests
pass, with the original Redis-dependent skip. The corrected cancellation control
waits for the owned child to signal readiness before cancelling; it preserves
assignment-before-resume and closure checks. The earlier seventeen-pass/one-fail
prelaunch cancellation remains a failure. Distinct actual review binds the source,
physical XML identities and original parent/child closeout. Evidence is
`.pcai/integration/canonical-shared-cache-maintained-defaults-actual-peer-r2/`.

Root copied the exact seventeen reviewed additions/replacement into canonical
paths and verified all twenty-three source/destination pairs. Six existing source
bodies remain physically unchanged. Source registers, dedicated qualification
entrypoints and the lazy C# capture implementation are included; default Unit/All
CI discovery remains unchanged. This is canonical source adoption, with commit,
current-head CI and deployed-user acceptance still pending. Evidence is
`.pcai/integration/cache-canonical-adoption-execution-r1/`.

A genuine fresh Windows Git checkout reproduced a delivery defect: existing
text/EOL rules changed nineteen of the twenty-three pinned files. Targeted raw
attributes and explicit renormalization preserve all twenty-three byte-for-byte
in the successor checkout. The first successor attempt reused cached index blobs
and failed; its failure and original error remain preserved. The successful
roundtrip is in `D:/pcai-relocation/cache-git-roundtrip-r2/`. Distinct actual review accepts that delivery proof. Root then applies the exact
canonical attributes and verifies all twenty-three raw staged Git blobs against
physical pinned sources. Five unchanged bodies require raw restaging; their EOL
diffs do not change code. Commit and current-head CI remain pending.
No test or source-register hashes are relaxed to accept transformed bytes.

The native MCP client pilot R3 passes distinct source review after preserving
original secondary errors, HOLD recovery custody and matching its mutex binding.
This remains source acceptance: actual private probe build, live consumer trials,
matched measurements and deployment are open. Evidence is
`.pcai/integration/native-machine-mcp-client-pilot-source-peer-r3/`.

An independently reviewed one-line runtime-test loader candidate resolves the
module path before ParseFile. Existing lexical ../../ paths survive in the
function AST, while coverage uses canonical file paths. All eighteen whole test
bodies, representing twenty-five expanded cases, remain unchanged. Attribution
recovery is a hypothesis pending genuine instrumented baseline/candidate runs;
no coverage hits are credited. Source evidence is
`.pcai/integration/runtime-integrity-coverage-attribution-source-peer-r1/`.
Published head `6673473` still fails coverage at 14,216/21,621 commands
(65.750890%), requiring 4,162 additional genuine hits for the unchanged 85% gate.

## Earlier acceptance and integration observations (09:39 UTC)

Published head `6673473` passes 2,809 PowerShell tests with zero failures, 40 skips
and zero not-run cases. CI run `38122837560` still fails the unchanged coverage
gate: the source-bound parent/child union is 14,216/21,621 commands, or
65.750890%, leaving 4,162 genuine additional hits required for 85%. Independent
reconstruction verifies all 239 module hashes, exact merge parents and every
command identity. Its production module tree is unchanged from `a3123dd`.
The three new disk cases genuinely cover seven previously missed disk-pipeline
commands. Nine other gains and two losses occur in unchanged runtime-sensitive
paths; the net fourteen-hit increase is not wholly attributed to the new cases.
Rust tests, strict
Clippy, formatting, .NET build, CPU runtime/build,
PowerShell lint and security pass; downstream integration is skipped. NVIDIA
and Guidelines pass in separate exact-head runs `38122837566` and `38122837590`.
The fresh PR check rollup confirms head `6673473`; these are not jobs inside the
main run. Exact private evidence is
`.pcai/integration/head-6673473-coverage-readback-r1/`. Prior-head receipts remain
preserved. Local and origin integration heads match; PR 182 remains draft/open.

The metadata successor now passes all twelve owned C:/D: controls, including
sharing refusal, historical and current-parent descriptor restoration, byte
mismatch refusal and later-writer refusal. Independent review verifies the exact
ordered controls, all six restored full descriptors, 98 closed child resources,
unchanged source bindings and normal original outer closure. Root confirms its
original input, result and closeout handles closed after return. Earlier raw
failures and their original error objects remain unchanged. This accepts the
metadata controls only. Evidence is
`.pcai/integration/metadata-writer-twelve-actual-peer-r4/`.

The repaired private registration transaction now passes all 22 unchanged
controls: four non-mutating entrypoints, then nine owned C: and nine owned D:
cases. Independent review verifies six exact install/rollback round trips,
18 security readbacks, unchanged detecting bodies and 422 stable physical pins.
Root verifies all 27 original control states, 435 original writers and its own
closeout writer closed after return. The two preparation failures, earlier raw
failure and original error references remain preserved. This accepts owned
fixtures only; new-session registration, live MCP use, exceptional disposal and
caller shutdown remain open. Evidence is
`.pcai/integration/mcp-filesystem-transaction-actual-peer-r5/`.

Carbon's separately admitted native host build and qualification now pass with
independent actual review. SDK 8.0.425 is selected for the net8.0 target; this is
an installed compatible toolchain, not a latest-SDK claim. All three actual
SDK/restore/build commands close with exit zero; 15 output artifacts retain exact
source and portable-PDB pairing. All 22 resolver selectors and four real relay
groups pass. The probe loads shared CLR 10.0.9 while PowerShell uses bundled CLR
10.0.12. The 208 output files (1,373,580 bytes) reach the D: receiver with exact
hashes; 316 physical pin references remain stable. Root observes original build,
test and inventory writer closure in their retained owner sessions after return.
The qualified outer Supervisor4 remains distinct from the unchanged inner
Supervisor2 build stage. Inner HOLD recovery access, canonical adapter adoption,
deployment, live consumers and performance remain open. Evidence is
`.pcai/integration/carbon-mcp-current-actual-peer-r1/`.

The complete maintained disk public-API unit file now passes all 37 cases on
Carbon with no skips or failures. Three new cases exercise minimum-size equality,
name/count ordering before Top and successful empty filtered output. The public
function runs unchanged; synthetic worker rows enter through the mocked transport.
Independent review verifies all 17 pre-existing whole test declarations, the
physical XML's exact 37 identities and 31 stable source/evidence pins. The original
outer process exits zero in 11.278 seconds with all ten closure flags; Root checks
the original receipt handles and empty custody registry after return. The first
runner's slash-comparison failure occurred before Pester import and remains
preserved. This qualifies the public pipeline tests, not a native scan or speedup.
Independent current-head coverage reconstruction now separately verifies seven
additional hosted disk-command hits. Evidence is
`.pcai/integration/diskusage-public-controls-actual-peer-r2/` and
`.pcai/integration/head-6673473-coverage-readback-r1/`.

The shared-cache projection now passes all eighteen maintained controls on Carbon.
Independent actual review verifies all 24 transferred/output artifacts, eighteen
physical UTF-8 XML identities with no failures/skips, nine real captures and 113
stable references. Root's same-original checks confirm outer process, two result
writers, identity, XML reader/stream and the separate inventory writer closed.
The prior count/encoding failures and retracted separator suspicion remain
preserved. Evidence is
`.pcai/integration/canonical-shared-cache-maintained-actual-peer-r2/`.

Full-module QA source R1 failed independent review because a generic counter
assertion preceded preservation of original failed-test errors. Additive R2 moves
the original error collection before that gate and admits only matching CACHECCB
ledgers. Its eight module files and complete existing cache test remain identical;
all 74 source references are stable. The first Root input attempt then stopped
before input IO/native launch because a prerequisite helper was not imported;
its original error and failure receipt are retained. A fresh corrected admission
completes actual module/memory/cold/warm validation on Carbon: full strict module
import exposes twenty exports, import and memory paths leave the native capture
type unloaded, six internal captures settle, and the complete cache unit file
passes eleven tests with its original Redis-dependent skip. Original outer
closure takes 91.449 seconds; the stopped owner takes 91.933 seconds. There is
no pending custody; all original result,
inventory and identity handles close after return. The Root readback retains the
first admission error and separately records its refused 65-character manifest
hash transcription before correction from the original owner's physical hash.
Independent actual review now accepts this exact full-module scope: all thirty-four
original artifact tuples and 116 evidence pins are stable; Root also verifies the
four review leaves, giving 120 physical pins without drift. Maintained default
adapter publication and canonical adoption are still pending at this cutoff.
The first concrete maintained-parent source review finds a write-error gap:
input flush/disposal can save a sticky original error yet leave complete bytes;
without a post-settlement error gate, the parent could still hash the input and
launch the child. That source remains unexecuted and preserved. An additive
successor must refuse launch after any original input-write failure; inert NoIO
checks alone did not exercise this branch. Additive R2 supplies that gate and
passes distinct source review. Its first actual maintained-entrypoint run then
fails one genuine cancellation control: 17 pass, one fails, none are skipped.
The fixed one-second cancellation timer fires before the public wrapper reaches
native process creation. The retained native result reports cancellation at the
first Start admission, a null child PID and one millisecond of native execution;
there is no evidence of leaked inner custody. The test still requires assignment
before resume and therefore correctly refuses a prelaunch cancellation as proof
of cancelling a real child. Preserve this FAIL; the successor must synchronize
cancellation with owned child readiness while retaining every closure assertion.
The second module-QA child is not launched after this failure. Root's outer
operation exits one with all ten closure flags, stopped clocks, zero pending
custody and its original result writer closed. The original error reference is
unchanged. All nine failed-run artifacts and the original readback arrive on D:
with exact hashes. Evidence is
`D:/pcai-relocation/carbon-maintained-defaults-failed-actual-r1/` and
`.pcai/integration/cache-cancellation-failure-source-review-r1/`.
The eight descriptive measurements
include 525.604-ms full import and 836.787-ms cold capture; warm captures range
from 1,248.336 to 1,561.409 ms. These include fresh child launch and observer
work, have no matched baseline and establish neither a speedup nor RSS savings.
Evidence is `.pcai/integration/canonical-shared-cache-module-qa-source-peer-r2/`,
`.pcai/integration/canonical-shared-cache-module-qa-actual-peer-r2/` and
`D:/pcai-relocation/carbon-cache-module-qa-actual-r2/`.

Five synthetic parser controls and seven genuine parser refusals pass in 308 ms
after repairing the stopwatch and literal long-help handling. All 17 original
writers, the identity handle and Root's separate receipt writer close; the stopped
duration is checked after final custody settlement. R1's final-duration refusal
and Root observer mistakes remain preserved. Both literal `-h` and `--help` now
return usage without creating new outputs or replacing original objects. These
inert marker tests do not execute native images or establish recovery behavior.
Evidence is `.pcai/integration/carbon-mcp-adapter-parser-controls-actual-peer-r3/`.

The first inherited-pipe input helper refused admission before creating any input,
owner, child or target directory. It incorrectly read per-input error/writer
fields from Native19 records that contain only Phase/Path/SHA256; the originals
belong to the enclosing registry. A second assumed closeout flag also does not
exist. The failed helper, exact original error references and closed readback
writer are retained. Additive source successors correct the actual registry
schema. The separately admitted matched inherited-pipe pair now passes all four
controls with independent physical review. The existing capture takes 30,523 ms
despite its requested 10-second timeout; the bounded native candidate takes
3,025 ms including initialization, with a 1,238-ms operation and 3-ms cleanup.
Its original readers, job and handles close, both outer operations close, and
Root verifies original input, phase and closeout writer closure after return.
The existing capture's inner handle state remains unexposed. This one tiny pair
is descriptive evidence, not a sustained fleet benchmark or canonical adoption.
The actual peer verifies 104 stable pins and preserves the historical failure.
Evidence is `.pcai/integration/inherited-pipe-capture-actual-peer-r1/`;
failure evidence remains
`D:/pcai-relocation/inherited-input-preparation-failure-r1/readback.json`.

Fresh Milly readback confirms approximately 580 GiB free internally, both USB
storage links at UAS 10 Gb/s, the WDC serial `1830D0800483`, and the study path
resolving to its native SSD image. Compression accounting below remains the
completed 13.84 GiB measurement; it is not a new throughput measurement. The two
nonsecret Clarius failure artifacts have reached the agreed ASUS evidence
directory with exact hashes and no overwrites. Existing UDM SSH times out before
the known-MAC Work lease lookup. A fresh route readback sends the configured UDM
address through P1's default LAN gateway; the current tailnet advertises no route
to that subnet and still lists Work offline. Work access remains unresolved. ADB again lists
no device, so the phone vault has not been read.

P1's 08:24 UTC bounded profile records three samples across both rotated parts
over 15.086 seconds: median CPU 47.53%, available memory 12,857 MiB and committed
memory 85.286%. Aggregate sampled disk queue is zero; disk latency P95 is
0.105 ms. This is a short aggregate observation, not a C:-specific measurement
or proof of sustained absence of swapping. The observer consumes 7.406 CPU
seconds, 2.23% mean host CPU. Heavy local build admission remains closed.
Evidence is `D:/pcai-relocation/p1-workload-current-r2/`.
The earlier bounded eight-second observation reports 21.3 GiB resident
pagefile usage; these are distinct metrics and do not prove disk swapping.
The busy Python process belongs to `poc_gesture_audiogram.py`, not an
identified MCP server. C: and D: are separate healthy NVMe disks, with about
269 and 1,069 GiB free. No process termination, memory trimming or pagefile change
is made from that observation. Carbon's fresh unit-test preparation has 16,444 MiB
available, 36% committed memory and no active compilers. It has no D: drive or
dtm-codex checkout; stable SDK 10.0.401 is absent. Build offloading must use its
actual paths and selected installed toolchain. The staged native MCP pilot is
older than the current merged native host and cannot be relabeled as that release.

The 08:29 UTC bounded MCP inventory identifies sixty active machine-launcher
PowerShell processes: twenty-one Context7, twenty-one Filesystem and eighteen
Google Workspace. Their snapshot private memory is 3,057,934,336 bytes (2.85 GiB);
process count and resident/private snapshots do not establish waste or savings.
Current Codex and Claude both use PowerShell launchers for Context7 and
Filesystem; Claude's Workspace registration is native. No active native machine
host or executable at either expected install path is found. Its current design
owns one backend per client and is not a shared broker. Thirty-nine additional
configured PowerShell wrappers remain. Serena and QMD have guarded database
startup paths; this does not establish that Docker restarted. Existing launchers
already use NoProfile, so removing full profile bootstrap is not the proposed
gain. A concrete private current-source native pilot adapter, graceful EOF client
and sequential 36-session comparison runner are sealed for review. The review
finds secondary errors converted to strings and later cleanup causes suppressed
after HOLD. Original error objects must remain available before any actual
qualification. No live registration or active client is replaced.
Evidence is `.pcai/integration/mcp-consumer-reuse-inventory-r1/`.

The agent-bus failure intake distinguishes local schema auto-fit warnings from
delivery failure: the authoritative hub and database are healthy, and current
requests are acknowledged at that earlier cutoff. A later Root presence renewal
fails because the local private durable outbox cannot be opened, while bus reads
still work. The generic CLI error does not identify whether this is contention,
scope, corruption or another IO failure. Read-only native outbox status passes
with zero pending requests, and subsequent Root presence/claim renewals succeed
without journal or lock changes. Exact historical cause remains unknown. Source
review identifies loss of the typed open-error cause, so safe diagnostic categories
and bounded fixture qualification are a remaining related-tooling gap. Reported PC-AI gaps include canonical Supervisor2
managed-child HOLD access, legacy capture handles that are not exposed, unbounded
shared-cache capture not yet replaced in canonical source, unavailable live
GitOps billing and the current coverage shortfall. The accepted Clarius failure
map and caller refinements were forwarded to the active ASUS owners. Original
Milly command/source/feature provenance, selected adapter projection and power-off
scan-timeout handling remain open; migration causality is unproved. Evidence is
`.pcai/integration/agent-bus-failure-readback-r1/` and the preserved exact intake
`.pcai/integration/clarius-milly-command-intake-r1/`.

Later bus messages report successful backup commands followed by fatal restic
integrity-check failures on both Sparks. These are reported failures, not a Root
diagnosis or accepted backup restore; backup validation remains open. The active
fleet lane receives this distinction. Root separately verifies all nineteen
declared public Clarius handoff leaves against the physical delivery manifest
`A1CE38511B1D30EE8A89C8230EEAB80654307F1B9DC44493338EBCEC4B89C0FA`
and acknowledges delivery intake only. A separately acquired public ASUS finding
matches SHA256 `15AE3B40CDD135E7EE535604E429562EE890F1FAF21FB5B2FB3C7D8B9EAFA475`
and fills the recorded historical invocation: the bench3 debug helper runs
`connect`, profile `lab_0546`, connection `milly_vigil_ap`, zero hold seconds and
operator-session output. Fresh remote hashes match all three original failure
logs. These records retain primary GATT-read and secondary disconnect deadline
failures; they do not bind the original binary bytes, build features, effective
environment or selected adapter. Numeric native exit, launch timestamps and cwd
remain unproven. No radio retry or wireless migration causality follows. Evidence
is `.pcai/integration/agent-bus-outbox-diagnostics-r1/`; Root verifies its four
owned and nine external pins. Adoption and device acceptance remain open. The fleet lane also
reports a stuck original P1 Utils fetch; Root preserves that custody and does
not replay it or advance the checkout.

The repaired GitOps monitor's first live run reports run/workflow inspection
available while billing remains unavailable. Owner lookup succeeds; the billing
request exits one. Quota stays Unknown, and no billing permission or successful
live usage response is inferred.

The GitOps workflow monitor called the retired product-specific billing API,
making overall inspection unavailable despite available run/workflow metadata.
The reviewed repair resolves the repository's User/Organization owner and
requests current UTC-month consolidated usage through the
[supported API](https://docs.github.com/en/rest/billing/usage). Missing access,
owner mismatches and malformed metadata remain unavailable; workflow inspection
availability is reported separately. Usage contains no included Actions allowance,
so quota remains Unknown. No quantities, prices or charges enter reports or
console output. Eight new executable controls fail against the preserved
predecessor; all 23 focused controls pass after repair with no skips. Existing
behavior assertions remain, with three request-count expectations increased for
the bounded owner lookup. Analysis and parsing report zero findings. Root reviews
the exact diff and physical result files; live billing access remains unqualified.

Fresh Carbon SSH and fetch confirm main `b92e5f6` matches origin/main, with the
untracked private build script preserved. PowerShell 7.6.6 and Rust 1.99.0 are
installed; SDK 10.0.401 is absent, so the P1 toolchain cannot simply be assumed on
Carbon. Its four linked review worktrees exceed the three-worktree limit. The
`pc-ai-current` tip `3b7600e` is an ancestor of main; review tips `45e7c3a`,
`7a08bec` and `8cf282f` are not. The `45e7c3a` checkout retains three modified
media/test files and a unique untracked test. A Codex process and retained
PowerShell processes are alive. No worktree is retired from clean status or age
alone; unique source and active custody still require reconciliation.

The real native Bitwarden version attempt exits zero and settles its process,
tasks, streams, cancellation object and handle, but the strict adapter rejects
123 stderr bytes. The outer child exits one; original owner and closeout writers
are closed, and the raw FAIL is preserved. Distinct review confirms settlement
without accepting a version or credentials. Official CLI startup source gives a
missing-data-file warning matching that byte count for the isolated path; this
is an inference because raw stderr was not retained. A fresh source successor
prepares an empty public data file before version discovery while keeping the
zero-stderr contract. The separately admitted successor now passes actual native
version discovery: public version 2026.9.1, nine stdout bytes and zero stderr
bytes, engine exit zero with all seven resource-closure flags. Its outer child
exits zero in 5.250 seconds with all ten closure flags, and original owner,
input and closeout writers are closed after return. Independent artifact review
accepts only public version discovery and confirms every immutable source pin.
The earlier FAIL remains unchanged. No vault operation has run.

Cache R5 again passes all 29 runnable controls with the one explicit Redis skip,
but fails its report validator before XML production: a Pester container has
Blocks and no Tests property, and the traversal reads the missing property under
StrictMode. All original outer closure flags and writer closures pass; earlier
cache and Bitwarden error objects remain unchanged. The failed receipt is
retained. R6 uses typed container/block traversal while keeping all test bodies,
30 source identities, grouping and physical XML validation unchanged. Its fresh
actual run passes 29 controls with one declared Redis skip, verifies the physical
UTF-8 XML's exact 30 identities and grouping order, and closes all nine capture
operations plus the original outer process and writers. The outer run takes
30.322 seconds. Independent actual review binds source, runtime, original
closeout and the physical report. Native zero-byte/inherited-pipe fixtures,
canonical adoption, real Redis and performance remain separate open gates.

The first offline fixture builder remains unexecuted after independent source
review identifies inherited PreBuildEvent/PostBuildEvent properties that the
pinned SDK executes. Independent review accepts the minimal successor's
rejection of these inputs and empty global properties for both restore/build.
The actual ZeroByte run restores and compiles successfully with zero compiler
warnings/errors, but its encloser remains FAIL: the Build host-trace read races
an open file writer. All three original commands exit zero and close; the outer
child exits one with all ten closure flags and closed original writers. The
review also finds a case-insensitive variable collision: the trace parser
overwrites its Path parameter with the loaded hostfxr path, mislabeling successful
SDK/Restore trace receipts. Physical traces select hostfxr 10.0.12. Five output
files exist with the expected net8.0/runtime8.0.31 metadata. Independent review
accepts their exact source, portable-PDB pairing and physical trace lineage for
separately admitted test-only execution; it preserves the failed encloser and
does not accept fixture behavior. A minimal successor corrects the variable
collision and retries only transient sharing/lock errors within a five-second
retry-decision budget; this is not a hard deadline on an individual file read.
The separately admitted InheritedPipe build passes SDK, restore and build with
all three commands selecting hostfxr 10.0.12 and exiting zero. Its outer child
exits zero with all ten closure flags; original input, owner and closeout writers
close. Independent review verifies all five artifacts, unchanged source and
DLL/PDB pairing. This remains build-only acceptance; the separate paired
inherited-pipe controls above provide the subsequent bounded behavior evidence.
The successful read-only SDK inventory is not build, package,
hostfxr-selection or native-fixture acceptance.

Separate baseline/candidate native validation passes all 19 ZeroByte controls
in both runs, with exact physical UTF-8 XML identities and all capture, outer
process and original writer resources closed. Independent review accepts
compatibility only: the TRUE/zero-byte fault is NOT_REPRODUCED. These passes do
not demonstrate a detected defect or qualify inherited pipes, Redis, performance
or canonical adoption. The earlier failed build receipt remains unchanged.

Original metadata session 35491 now passes its own five retained-child supervisor
controls. Physical records show HOLD retention, observation-only retry, refused
new launch while custody is pending, and eventual closure of the same originals.
The timeout and injected stream-overflow causes remain recorded; the overflow
control keeps DrainsSucceeded false and its secondary drain error. Its physical
log is exactly 32 KiB even though the faulted operation's byte-count field remains
zero. Independent review verifies 67 stable pins and attributes Root's same-session
post-return handle checks. All three earlier metadata FAIL objects and their
original first/secondary error references remain unchanged. The twelve metadata
writer controls and persistent MCP deployment remain separate open gates.

An actual uncapped directory-manifest benchmark initially fails the FileCount
comparison. The tree has exactly 391 file symbolic links and one directory
symbolic link; native statistics count these among entries rather than ordinary
files/directories. The PowerShell baseline now follows that classification,
preserves ordinary hardlinks, and validates the derived other-entry count while
retaining all four exact uncapped statistic comparisons. Two real link fixtures
fail against the preserved predecessor; the corrected contract suite passes all
40 cases without skips. Script analysis reports zero findings. The fresh native
and PowerShell benchmark passes aggregate parity at depth three without a result
cap: native mean 97.90 ms versus 1,368.48 ms, 13.98 times faster in this five-run
window. Mean managed allocations are 646.82 KiB versus 303,046.91 KiB. Loaded
Rust and C# DLL hashes remain unchanged. Independent review accepts the complete
source change, genuine predecessor failures, physical 40-pass result and current
aggregate comparison. Entry identities, full JSON link
classification, whole-system responsiveness and other tooling remain separate
claims. Capped manifests and different token heuristics still receive no
qualified speedup comparison.

Published signed head `ad25aee` adds the manifest benchmark correction. Its
[CI run 38113748451](https://github.com/David-Martel/PC-AI/actions/runs/38113748451)
passes 2,798 PowerShell cases with zero failures, 40 skips and zero not-run cases
in 464.09 seconds. Coverage still fails the unchanged 85% gate at 14,209/21,621
commands, 65.718514%. Rust tests, strict Clippy, format, .NET, CPU inference,
security and PowerShell lint pass; integration tests are skipped and the aggregate
gate fails. NVIDIA validation passes at the same head; Rust Guidelines remains
in progress at this cutoff. Local and origin integration heads match. Independent
current-head artifact reconstruction is underway; predecessor artifacts are not
substitutes. PR 182 remains draft and unmerged.

Preceding signed head `83b5fe8` adds the qualified runtime DLL repair. Its exact
[CI run 38105878561](https://github.com/David-Martel/PC-AI/actions/runs/38105878561)
passes 2,796 PowerShell cases, with zero failures, 40 skips and zero not-run cases
in 435.19 seconds, but fails the unchanged 85% coverage gate at 14,209/21,621
commands, 65.718514%. Root reconstructs the same union from the downloaded
source-bound artifact against all 239 module hashes in CI's exact merge commit:
13,761 parent commands plus 448 child commands. Reaching 85% at this denominator
requires 4,169 additional genuine hits. Distinct review independently reproduces
the same parent, child and union identities and hashes. Rust tests, strict
Clippy, format, .NET, CPU inference, security and PowerShell lint pass. NVIDIA
validation and the separate Rust Guidelines run pass at the same published head. The
coverage failure skips integration tests and fails the aggregate CI gate. Local
and origin integration-branch heads match; PR 182 remains draft and unmerged.
The preceding independently reconstructed coverage evidence follows.

Published `908ac1a` passes 2,771 PowerShell cases with zero failures and 40 skips
in [CI run 38102938733](https://github.com/David-Martel/PC-AI/actions/runs/38102938733).
The unchanged 85% gate fails at 14,200/21,342 commands, 66.535470%, requiring
3,941 additional covered commands at the same denominator. Raw JaCoCo has
13,752 covered commands; 448 source-bound child hits contribute to the gate.
The uploaded artifact now includes the complete source-bound parent and child
inventories. Root and an independent reviewer reproduce the exact command union
against all 239 module hashes from CI's exact merge commit, without substituting
local CRLF files. The same denominator requires 3,941 additional hits to reach
85%; no threshold or denominator was reduced. Rust Guidelines run `38102938734`
has completed successfully. The new head above has its own current-head result;
the older reconstruction does not substitute for it.

The workflow now captures source hashes before measurement and uploads the full
parent command snapshot and result-bearing child receipts before enforcing the
gate. The existing measured-child control reproduces the command union and gate
from a JSON roundtrip; zero coverage still fails. A strict run exposed three
summary-only caller failures in optional child-receipt handling. The repair keeps
those assertions unchanged and passes all 35 coverage controls, with no skips,
block failures or source drift. Normal original-job closure takes 25.199 seconds;
the preserved first run has 32 passes and three failures. Actual receipts are in
`D:/pcai-relocation/coverage-artifact-qualification-r2/`, including the original
closeout. This qualifies artifact transport and caller compatibility, not new
module coverage, a published-head CI pass or a reduced target.

Gitguard native inventory passes eight tiny controls and twelve descriptive
ABBA measurements: native median 187 ms, legacy median 3,062.5 ms. The separate
first real-repository attempt passes two existing-file positive controls, then
refuses an existing reserved-name junction before legacy comparison. Its corrected
successor preserves regular file bytes, directory metadata and raw reparse target
metadata without following the target. It passes both positives and the full
24-candidate/14-scope repository comparison with every original operation closed;
the outer run takes 65.859 seconds. An independent reviewer verifies the exact
candidate inventories and preservation. The prior failed result is preserved
in `D:/pcai-relocation/gitguard-windows-inventory-repository-qualification-actual-r1/`
and independently reviewed under the corresponding actual-peer namespace.
The successful successor is in the corresponding `actual-r2` directory. No new
hook is installed; tiny timings establish no whole-repository speedup.

Runtime DLL selectors reproduce integrity and alias-coordination failures:
the 16-case predecessor has strict 7 passes/9 failures and ordinary 8 passes/8
failures, without skips or source drift. The first private repair was blocked
before execution because failed rollback could lose backups, path deletion could
race a replacement, later variants could hide settlement failure, resource/error
custody was incomplete, and empty variants added GPU-discovery overhead. The
reviewed repair retains recovery data, refuses unresolved resources before another
activation, validates both file digests and handles generic/ordered dictionaries
without the prior overload-binding defect. Eight new dictionary controls reproduce
one pass/seven failures on R2. R3 passes all 25 controls in StrictMode and ordinary
PowerShell, without skips or setup failures. Independent review binds all three
XML results, 228 fixture pins and the original closed process. The exact proposed
module and tests are adopted; a separate run using the canonical default path
passes 25 integrity controls and all 43 existing inference consumer tests. These
are file and software contracts; live native inference, GPU behavior, crash
atomicity and exceptional OS-disposal recovery remain unqualified. Native
inference is currently disabled, so the original defects were latent rather than
evidence that a live DLL was corrupted.

The first Bitwarden public-controller run completes with two passes and thirteen
failures, with its original process, streams and writers closed. It exposes a
duplicate `help` alias and object parameters that unwrap caller references before
process allocation. Both are repaired in a private successor; exact parameter
binding probes preserve the old failure and new reference identity. Independent
source review also catches duplicate external pins that would reject its enclosure
before execution. The packaging-only successor retains every control body and
passes all fifteen public controls without skips or not-run cases. Independent
actual review verifies 104 source-bound assertions, 98 stable pins, eleven original
engine closure rows and the closed outer operation. Root separately verifies
original owner, input and closeout writers closed, with no native custody pending.
The packaging failure and original actual failures remain preserved. These public
controls run no vault, credential or real Bitwarden executable. A real native
version-only canary remains source preparation, not credential acceptance.

The metadata-writer probe initially fails at admission before allocating a child:
its validator uses PowerShell's automatic `Input` variable as a parameter. The
same complete function reproduces `ArgumentException: index`; renaming only that
parameter and its references passes the inert metadata check. Three original
read streams close with no native custody remaining. The repaired admission then
launches a real isolated native probe: C_SharingRed passes, followed by AccessDenied
when C_HistoricalPrivateParent reopens its replaced target before restoration.
Ten later controls do not run. All fourteen reported child resource slots and all
ten original outer closure flags close; Root owner/input/closeout writers are also
closed. Independent actual review preserves the first failure and secondary
diagnostics. Later read-only observations find an empty target DACL, the exact
historical eight-ACE displaced descriptor, and a noninheritable private parent.
The fixture omitted the maintained helper's Directory switch at its two directory
call sites. A minimal successor corrects only those calls and advances execution
paths to unused locations, retaining every descriptor, byte, identity, sharing
and ordering assertion. Its actual run passes eight C:/D: controls, including exact
historical descriptor restoration, wrong-byte refusal and later-path writer
refusal. Current-parent setup then fails its unchanged exact-descriptor assertion:
the captured directory has 12 ACEs/488 bytes, while the fixture clone has 11/452.
Three later controls remain not run. The original outer job and all owner, input
and successful closeout writers are closed with no native custody pending. A
failed first Root closeout helper remains preserved separately; it failed before
writer allocation. The raw-directory setup proposal passes independent source review;
neither the eight controls nor that proposal qualifies current-parent restoration,
MCP registration repair or deployment.

Shared-cache process capture still waits for exit before reading either pipe and
uses unbounded post-exit reads. The minimal private proposal preserves all current
cache functions except that capture extent, reuses the existing bounded C# job
controller and adds final deadline accounting after bounded decoding. Independent source review detects
false EOF after successful zero-byte native pipe reads and an outer supervisor
that terminates a child reporting managed resource HOLD. Source-reviewed successors
address both defects. Five genuine supervisor controls pass: two ordinary exits,
marker HOLD, markerless timeout HOLD and markerless output-overflow HOLD. Each
failure stays sticky, blocks another launch, and closes only after the fixture
settles its own original resources. Same-original post-return checks confirm all
seven receipt writers closed, no first or secondary owner error and no native
custody pending; independent actual review accepts only those five controls.
Partial launch, descendant and shutdown paths remain unqualified. The cache
encloser also fixes inherited Redis settings before Pester; its predecessor could
call an external provider despite the probe skip. The first isolated run has
29 passing rows and one declared Redis skip, but incorrectly treats Pester's skip
reason as a failure. Its original resources close and its failed result remains
preserved. A narrow successor retains the exact typed skip reason and passes all
29 ordinary rows, with one declared skip and no not-run cases. Independent actual
review then rejects its XML: UTF-8 physical bytes declare UTF-16. Passing test
observations and original closure remain valid; export acceptance is HOLD.
The source-reviewed XML successor checks the closed physical byte hash, XML
encoding, all 30 ordered results and original parser disposal before PASS.
Its actual run preserves 29 passes and the declared skip but fails the export
validator's assumption that XML order equals the flat Pester result order.
Independent physical review proves all 30 XML identities and results are correct:
the pinned exporter groups parameterized cases by GroupId. Original resources
close; the failure and its first/secondary causes remain preserved. A minimal
successor must validate the genuine exporter projection and complete case
identity mapping without changing test assertions.
Genuine inherited-pipe
qualification, canonical test defaults and cold/warm measurements remain gates
before adoption;
there is no production speedup or shared-server reuse claim yet.

The separately admitted read-only .NET inventory completes in 76.175 seconds,
reporting 4,040 files, 768 directories and 531,047,114 bytes for the selected
SDK 8.0.425/runtime 8.0.31 inputs. The original outer operation exits zero with
all ten closure flags true; original owner and separate Root receipt writers
close without first or secondary errors. Independent actual review reproduces
the complete record counts and totals, physical receipt hashes and closure
evidence. The reviewer does not rehash the underlying SDK tree; those hashes and
double-set checks are the sealed observer's successful-path evidence.
This inventory provides no offline build, package, native fixture or toolchain
deployment acceptance and does not change the installed default SDK.

The latest bounded workstation sample covers 60.121 seconds and twelve samples:
median CPU 36.018%, median available RAM about 15.3 GiB, commitment 81.387%,
disk latency median 0.063 ms/P95 1.021 ms and disk queue median zero/P95 one.
Observer host CPU averages 1.171%. There is no sustained disk thrashing in this
window; it does not establish lasting responsiveness or cause. Hard-fault counters,
resident pagefile usage and process counts do not prove waste or active swapping.
Raw evidence is `D:/pcai-relocation/p1-resource-live-r6/`; no useful agent priorities,
memory trimming or live processes are changed by this observation.

The media repair remains uncompiled and untested. A proposed 1,074-file public
cache seed uses about 118 MB instead of copying over 27,000 cache leaves. Source
review caught an unbound argument guard and an incorrect readiness token before
copying. R2 fixes both, but its below-80% commit gate is not met by the fresh 81%
sample. Copy, archive-byte verification and actual locked offline resolution are
still unexecuted. The seed is not an offline-qualified build cache; expanded-cache
inventory and the unchanged build limits remain separate requirements.

Milly's fresh read-only storage check still shows approximately 580 GiB
free internally, the migrated SSD at UAS 10 Gb/s and all five Btrfs device-error
counters at zero. Shared NTFS free space and sparse-image free space are not
additive. The Btrfs mount enables transparent zstd:3 compression; this configuration
alone does not establish the whole study's achieved compression ratio. A fresh
read-only native extent scan of the resolved study tree reproduces the completed
migration's exact 3,763-file accounting: 557,246,863,357 disk bytes versus
572,110,004,663 uncompressed bytes, saving about 13.84 GiB. The prior complete WAV
group accounts for about 13.71 GiB of those savings; MP4 savings are negligible.
A second attached
SK Hynix SSD (serial AYCCN03781CB9190E) has about 953 GiB available on its existing
FAT filesystem and retains about 168 MiB of unique calibration/audio files. No
second-drive files are moved or deleted. Its file-size and Unix-metadata limits
need a consumer-specific plan before using it for large study data. Storage
migration is verified; additional study compression, cold boot, physical removal, Windows access and
restore/application checks remain separate. Dtm-work access and credential
adapter qualification remain open; the fresh ADB enumeration has no device.

## Follow-up software qualification (01:06 UTC)

The NVIDIA coordinator now returns non-success after a declined download or
backup, before reachability, staging or vendor delegation. Valid existing-file
download-only and already-current no-op results remain successful. CUDA JSON
discovery handles optional component/version fields under StrictMode; optional
registry metadata preserves null values and supplied array shape. Strict and
ordinary runs each pass 102 controls: 65 existing, 19 coordinator and 18 metadata.
The unchanged predecessor fails seven coordinator and four metadata controls;
an intermediate array-shape regression also remains preserved. Independent review
is `.pcai/integration/nvidia-repairs-actual-peer-r2/review.json`. This is software
contract acceptance; no vendor installation, hardware, performance or coverage
acceptance follows from it.

Private Coreutils candidate tests pass all twelve Windows correctness controls;
the baseline reproduces three failures with one positive control. Root confirms
unchanged fixture ACLs, no owned registry residue and closed original resources.
Independent review is `.pcai/integration/coreutils-windows-correctness-actual-peer-r2/review.json`.
Five compiler warnings and the earlier 86 Clippy diagnostics remain open. The
canonical T: checkout and its fourteen foreign WIP files remain untouched.

The protected metadata keeper probe passes two controls. Full writer qualification
still fails after six passing rows: R3 loses the legacy fixture's inheritance flag
during setup; R4 reproduces the exact setup but fails after actual replacement.
Fixture-parent inheritance remains under investigation. Root withdrew an earlier
supplementary descriptor read that used a missing shortened path without checking
an invalid handle. A fresh valid-path/handle read confirms R3 retains eight ACEs;
R4 has three after replacement. The correction and closed original resources are
in `D:/pcai-relocation/p1-native-filesystem-writer-controls-root-failure-r4/root-global-closeout-r1.json`.
The live MCP configuration and stable runtime remain unchanged.

Published documentation head `bb8fc3c` has a failed PowerShell Tests job in CI
`38099343875`; current-head coverage and PR 182 integration remain open. The
untested media mutex repair is excluded from the NVIDIA repair commit.

## Current acceptance readback (00:34 UTC)

Published `4bef994` passes 2,734 PowerShell cases with zero failures and 40 skips
in [CI run 38095401802](https://github.com/David-Martel/PC-AI/actions/runs/38095401802).
The unchanged 85% gate fails at 14,052/21,331 distinct commands, 65.875955%,
requiring 4,080 further covered commands at that denominator. Uploaded JaCoCo
contains 13,604 covered commands; the maintained gate adds 448 source-bound child
commands. The independent review did not recompute the unavailable child inventories.
The PR checkout is merge `39f4c617`, with published head `4bef994` as a parent.
Rust Guidelines `38095401792` completed successfully on the same published head:
452 tests pass, four are ignored and none fail across five distinct result rows.
The media-model row takes 2,222.72 seconds; its 58 passes and three ignored tests
explain most of the long execution. Six allowed dependency-audit warnings remain;
success does not establish zero vulnerabilities. NVIDIA software validation passes.
The terminal receipt is `.pcai/integration/current-ci-runtime-gap-review-r1/terminal-summary-r2.json`.
PR 182 stays draft because the coverage gate and fleet acceptance remain open.

dtm-codex PR 40 merged as `3e2b331`. Its exact merge-head
[native run 38095723844](https://github.com/David-Martel/dtm-codex/actions/runs/38095723844)
and portable quality run `38095723836` pass. Independent review verifies all 22
selected contracts and four raw relay groups, source bindings and original closure.
The expected timeout now returns exit 1 normally in 1.264 seconds without a harness
kill; the earlier failed crash-exit trial remains preserved. Local main matches the
merge and the reviewed source tree; only the integrated owned branch was retired.

The fresh R7 filesystem canary passes: one real client read returns the exact
27-byte Unicode fixture and all original custody closes. This qualifies that read;
persistent registration remains gated on exact installer source and owned C:/D:
writer controls. The first installer proposal is held because its fixture gate did
not bind the delegated transaction helper; a separately frozen successor adds that
binding. Actual writer controls then fail on the first C: install with sharing
violation 32: its replacement data handle remains open across `File.Replace`.
All original resources close, and no live configuration is changed. A separate
metadata-only keeper permits replacement, but its initial unprotected sample
changes the ACL auto-inheritance flag; that result remains FAIL. The protected
successor also detects lost inheritance state before replacement. The next probe
must use the maintained private-stage and recovery-ACL sequence exactly, and the
full C:/D: inherited and legacy-deny controls remain mandatory before registration.
Active clients and authentication remain outside the proposed change.

The Rust winpath lint repair passes its former dependency point, but successor
Clippy fails with 86 winutils-core diagnostics: 22 dependency-version policy
warnings and 64 located code warnings. No package tests followed that failure.
Review separately confirms the pathname-as-HANDLE ACL query and missing descriptor
free, and identifies a source-level registry buffer-capacity risk; private source
repairs and detecting controls are frozen. A readonly admission failure exposed
the automatic PowerShell `$input` parameter collision; its successor renames that
parameter and preserves all production Rust and twelve controls. Owned file and
directory fixtures now pass preparation. Separate Root checks of selected ACLs
match their retained expectations, and Root observes zero owned HKCU residue.
Baseline native tests are running separately with two
jobs, offline dependencies and D: outputs. Neither lint counts nor source review
establish a reproduced memory overwrite or passing native correctness tests.

The Gitguard inventory build successor replaces the reserved HOME assignment and
passes its three offline SDK/build commands in 7.518 seconds with zero compiler
warnings or errors. Independent source and actual reviews bind the exact five
outputs; all ten outer resources and original writers close. The earlier HOME
failure remains preserved. Real inventory parity, fallback checks and timings
are still pending; no hook installation or performance acceptance follows.

A separate media-worker source review identifies a poisoned-mutex deadlock: the
error owns a guard while the worker attempts to acquire the same lock again.
The claimed repair uses that held guard and retains the original error wording.
Its two detecting fixtures require bounded predecessor/candidate execution;
compilation, actual deadlock detection, narrow regressions and lint remain NOT_RUN.
The successful published CI precedes this repair and does not qualify it.

Fresh Milly readback still shows 622,824,415,232 internal free bytes (about 580 GiB)
and the WDC SSD at UAS 10 Gb/s. Shared NTFS and its contained sparse Btrfs image
have separate, non-additive free-space accounting. Physical second-slot vacancy,
cold boot/removal and Windows access to the native image remain open. dtm-work and
credential reconciliation remain open; fresh local ADB currently lists no device.

## Historical acceptance readback (23:17 UTC)

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

## Source, runtime and route findings at the preceding cutoff

Review cutoff: October 11, 2026, 19:20 UTC. Owner: Codex integration lane.
These observations supersede pending statements above only for the named controls.
The fleet and repository consolidation remain incomplete.

Published head `a39431fd23fc845dd84a8338b79b8c620f0c4dc1` equals its origin
integration branch. Its PR CI run `38163641926` tests synthetic merge checkout
`bcf049f8151db5e03741e3668d652d80cedbbb1e`, whose second parent is that exact
published head. It passes 2,862 tests and skips
40, but fails the unchanged 85% coverage gate: 14,583/21,816 commands,
66.845435%, with 3,961 additional genuine covered commands required. Rust,
C#, PowerShell lint and security jobs pass; integration is skipped. Draft PR
#182 remains open and cannot yet be merged. This is not current-head acceptance
for subsequent working-tree changes.

The working tree adds 29 detecting unit cases: eight runtime selection/bundle
boundaries, four process-capture prelaunch refusals and seventeen prompt
enrichment cases. This unit-test slice's sole production change extracts the existing native
dashboard call into a private boundary while preserving enrichment formatting
and fallback. Real-file selection tests never load marker DLLs; prompt tests
mock the native boundary and do not prove a native dashboard execution.
Independent source reviews and a fresh five-file aggregate run are retained in
`.pcai/integration/` and
`D:/pcai-relocation/prompt-cache-runtime-aggregate-tests-r1/`: 79 pass, zero
failures, one existing opt-in Redis skip, zero block/container failures.
Focused coverage is not credited to the published CI gate.

Carbon's native Filesystem pilot passes six actual controls: initialization and
full catalogue, exact owned read, outside-root read denial, read-only write
denial, actual CLR observation and graceful EOF shutdown. Original post-return
closure was observed by the retained Root session; physical receipt review alone
does not establish live object custody. Evidence:
`D:/pcai-relocation/filesystem-native-standalone-carbon-r2/` and
`.pcai/integration/filesystem-native-standalone-actual-peer-r2/review.json`.
The earlier reserved-HOME prelaunch failure remains preserved. Current consumers
permit writes, so a fresh writable pilot is required before adoption. The
installed backend's `read_text_file` catalogue differs from local source's
`read_file`; installer output paths also differ from launcher paths. Rebuild
lineage, effective roots/permissions, new-session consumers and shared-server
reuse remain open. No active wrapper or registration has been replaced.

Four actual media phases pass on Carbon: selected CPU FFI baseline/candidate
builds and the same one unskipped healthy-state Rust test for each. The baseline
poison control fails: its ten-second deadline retires the actual Rust process,
but the outer catch retains a different ErrorRecord for the same exception and
the exact-original acceptance guard correctly refuses it. Candidate poison has
not run. The failed original owner and errors are retained; a separately sealed
source correction is under review. No media acceptance or canonical FFI
integration follows from the four passing phases. Actual evidence is in
`D:/pcai-relocation/media-carbon-preparation-r1/actual-r5/`.

All three bounded HTTP/DNS processes close, but both HTTP cases remain failures.
Cold localhost fails before server acceptance. Cold IPv4 completes the request
and exact `owned-complete` response, then fails because one original client
error was captured. DNS alone succeeds in 15 ms and returns IPv6 loopback before
IPv4 loopback. That resolver order does not establish the HTTP endpoint or the
cause of either failure. Keep the one-second client contract and retained errors;
diagnose the IPv4 error before any production repair. Evidence:
`D:/pcai-relocation/native-mcp-http-carbon-transfer-r2/actual/`.

The ordinary Gitguard candidate is rejected for deployment. Sixteen matched
measurements show median candidate/baseline ratios of 1.8648 on the small fixture
and 1.8936 on the wider fixture, respectively 86.5% and 89.4% slower. Installed
Gitguard remains unchanged. Separate positive correctness remains open;
`D:/pcai-relocation/gitguard-carbon-staging-r1/actual-r2/result.json` retains
the actual observations rather than a performance claim inferred from source.

Milly's 18:23 UTC storage readback confirms 622,791,888,896 bytes free internally,
35% used, active ordered NTFS/Btrfs mounts and five zero Btrfs error counters.
Outer SSD and nested filesystem free space are not additive. Study migration
and lossless compression are complete; cold boot, backup restoration, Windows
access, application acceptance and physical M.2 slot vacancy remain untested.

Authenticated Cloudflare readback resolves Work's SSH hostname to tunnel
`5242df17-2784-4679-951a-e01388c18f88`, currently disconnected with zero
connections. Retained Root tool observations show the UDM hostname reaching an
Access login; existing Site Manager
keys return 401. No access policy or unrelated service-token credential was used
to work around this. Exact route evidence is in
`D:/pcai-relocation/cloudflare-route-mapping-current-r1/result.json`.
Super-NUC crash evidence reported on agent-bus has not been independently read:
the bounded SFTP attempt times out during banner exchange in retained Root tool
observations, without a separate physical result receipt. The reported Eko soak
without a freeze is a lead, not causal attribution of this workstation's stalls.

## Current validation and deployment gaps

Review cutoff: October 11, 2026, 20:38 UTC. Owner: Codex integration lane.
Earlier receipts above retain their original states; the following findings
supersede only the named pending items.

Predecessor published head `71ed3e9c5b981d58b57e526d2e8c1485c1e26c69`, CI `38168895722`,
fails with 2,870 passes, 21 failures and 40 skips. Synthetic checkout
`3acd84509c64bfce7c53db8e37372b70046b739d` binds that exact PR head. Three
test consumers still expected 23 helper declarations after the reviewed
dashboard extraction added a 24th. The repair checks the exact 24-name multiset
and rejects same-count substitutions or duplicates. It retains parser and
security checks. A fresh six-file run passes all 143 cases with no skips or
container errors. This focused result does not satisfy hosted CI or its
unchanged 85% gate; the failed published run records 13,978/21,817 commands
(64.069304%). Evidence: `D:/pcai-relocation/current-head-ci-71ed3e9-r1/` and
`D:/pcai-relocation/llm-declaration-endpoint-tests-r1/`.

The correction is signed and published as
`72ee00d74e1870d2623ac2ee672f58585529fd51`, with the required Codex trailers.
Local and origin integration tips agree. Its CI `38172830775` is in progress;
neither focused validation nor predecessor receipts establish current-head CI.

The actual cold IPv4 error is now attributed to an absent optional
`Resolve-PcaiPath` lookup recording a CommandNotFoundException. Non-recording
command discovery fixes that specific defect while preserving actual resolver
exceptions. Four standalone child controls and the 143-case aggregate pass.
Eight matched lookup-only measurements reduce the median for 25 calls from
1,100.545 ms to 325.647 ms (70.4% lower). This measures missing optional
command discovery, not profile startup or end-to-end throughput.
Evidence: `D:/pcai-relocation/endpoint-optional-discovery-benchmark-r2/`.

A fresh repaired IPv4 client completes the exact owned response in 58 ms,
captures zero errors and closes its original process and tracked resources.
The separate fresh localhost client still fails with a timeout before any
request reaches the listener. Both original parent cases close without pending
native custody; historical failures remain failures. No relaxed timeout,
prewarmed client or general transport acceptance is claimed. Evidence:
`D:/pcai-relocation/http-r4-carbon-actual-r1/`.

The writable native Filesystem pilot passes nine actual catalogue, Unicode
read/write/overwrite, outside-root refusal, CLR and graceful-EOF controls.
Source and physical-receipt review do not qualify live configured consumers.
The first reversible transaction fails its exact ACL check; its second source
revision passes independent 20 transaction and 14 custody/ACL controls, but
still omits audit ACL reads. A successor must capture and verify the SACL.
The Codex consumer source also needs detecting controls for reused tool IDs
and unrelated authentication denial. Root additionally found its capacity
check using a nonexistent CIM property; actual committed memory is 89%, so the
unchanged below-85% admission must refuse rather than silently compare null.
These are deployment blockers. No live MCP registration has changed.

The media R7 poison pair reproduces the baseline noncompletion under its owned
deadline and passes the candidate's one real Rust test with exit zero. This
supports the narrow held-lock self-deadlock repair. Independent FFI review
finds unconditional Failed insertion can overwrite terminal/cancelled states
or resurrect a missing ID; public poison polling and cancellation races remain
open. Preserve the foreign canonical media edit and require detecting fixtures
and custody before integration. Evidence:
`D:/pcai-relocation/media-carbon-preparation-r1/actual-r7/` and
`.pcai/integration/media-ffi-source-review-r1/review.json`.

Milly's fresh readback confirms 622,685,741,056 bytes available internally,
35% root use, both external UAS links at 10 Gb/s and all five Btrfs device-error
counters at zero. The native data mount retains zstd:3 compression. The separate
SKHynix FAT drive is unchanged. Lenovo documents two M.2 2280 PCIe 4.0 x4 slots
for the observed ThinkPad P16 Gen 2; software enumerates one internal NVMe.
Physical second-slot vacancy and installation parts remain unverified. See
[the storage review](milly-storage-and-expansion-review.md).

The fresh bounded agent-bus read retains 500 messages covering approximately
06:53 to 20:35 UTC; it is not an exhaustive 24-hour history. Other-agent leads
include Super-NUC AX211/BLE freeze correlation and its later no-freeze soak.
The requested exact crash-source transfer is pending; neither lead establishes
causality. Repeated GitOps Unavailable messages are independently traced to
billing and code-scanning inspections returning HTTP404 while workflow/run
inspection remains available. Do not reinterpret zero returned upstream items
as a successful security inspection. Evidence:
`D:/pcai-relocation/agent-bus-current-findings-r1/messages.json` and
`Reports/gitops/monitors-latest-David-Martel_PC-AI.json`.

## Current reviewed integration state

Validation cutoff: October 11, 2026, 21:45 UTC. Earlier receipts remain historical.

The published 24ae406 head completed CI38173326731 with2,897 PowerShell passes,
zero failures and40 skips. Coverage14,623/21,817 (67.025714%) fails the unchanged
85% floor. Other applicable .NET, lint, security, Rust and CPU deployment jobs
passed. The source-aware coverage review is retained at
`.pcai/integration/ci-coverage-current-risk-review-r2/review.json`.

PR183 head629196a completed both current workflows successfully. A read-only
merge with24ae406 still produces eight conflicts. This workstream adopts only
the reviewed installer and deployment tests; it does not merge the remaining
profile, runtime, native, telemetry or CI changes or retire the active branch.
The exact installer/test candidates passed54 isolated controls without skips
and an independent source review. The validator also accepted all13 maintained
directory modules. Stage verification precedes publication, and the genuine
late-failure test witnesses Alpha's new payload before Beta fails and the
original payloads/environment are restored. No live module deployment occurred.
Evidence: `D:/pcai-relocation/pr183-installer-fixture-r4/` and
`.pcai/integration/installer-adoption-peer-r1/review.md`.

Private telemetry validation now includes16 passing focused controls and a
same-assertion behavioral comparison: the original collector fails all seven
selected cases; the candidate passes all seven. This independently detects
missing-value coercion and lost write-failure reporting rather than a fixture
setup defect. Evidence:
`D:/pcai-relocation/pr183-crash-telemetry-baseline-detection-actual-r1/`.
Real CIM units, observer overhead and deadline acceptance remain open. The
five nullable public C# metric accessor changes require compiled-consumer
compatibility qualification; declared project version remains0.1.0.

The native filesystem transaction is now qualified by34 owned controls plus
actual nonempty SACL preservation. Earlier tool-ID/auth-denial fixture findings
were corrected and revalidated. The genuine configured Codex consumer pilot
remains gated by about89% local commitment, above its unchanged85% limit.
No live MCP configuration, native-host registration or shared-server reuse
has been adopted. Transaction qualification does not establish consumer or
fleet qualification.

Media terminal-state admission R1 refused before compilation: an older retained
cause is a PropertyNotFoundException, while R1 expected ErrorRecord. Both
original failures remain retained. R2 source admission corrects that specific
assumption and preserves exact original references. Its matched new builds
and terminal-state detectors remain pending; public polling, cancellation,
missing-ID and detached-worker lifecycle findings are still open. The foreign
canonical media edit remains untouched.

Milly's bounded direct read measured457 MB/s with the existing GNU dd command;
internal available space remains about580 GiB. uutils dd's direct-read failure
matches an upstream buffer-alignment report. See the
[storage review](milly-storage-and-expansion-review.md) for measurements,
source attribution and remaining physical/application acceptance gaps.

P1's live Python stack was inspected without blocking or mutating the target:
it belongs to the vigil-audio gesture-audiogram/MediaPipe pose path with an
active Claude ancestor. The process remains preserved while owner disposition
is pending. Evidence:
`D:/pcai-relocation/responsiveness-live-python-r3/result.json`.
The source, runtime, coverage, cleanup and fleet gaps above remain open; this
is not a clean-main or cross-machine synchronization closeout.

## Current integration and qualification update

Observed cutoff: 2026-10-11 23:26 UTC on dtm-p1gen7. Earlier sections and failed
receipts remain preserved. The local head is `80897aab93751bccf9ce1147fa0f23b2cdede818`;
the pending actual merge incorporates `67223a05440973a347950c4bb7205c38424ea090`.
All nine conflicts have reviewed resolutions and no unmerged index entries.
This is an uncommitted merge at this cutoff, not main/origin equivalence.

- Main/PR183/PR184 source is integrated with the runtime branch's explicit 85%
  coverage enforcement, Rust metadata, installer rollback, profile roots and
  Common discovery/cache guards. Publisher-local cleanup and bus statements in
  `docs/windows-fleet-closeout.md` now explicitly identify their historical,
  unrecorded-host scope rather than describing this machine.
- The first mixed run recorded 151 passes, three failures and four skips.
  Profile warning observation, actual true-body delegation and null-versus-empty
  environment restoration were repaired without changing production profile
  behavior. The second mixed run records 171 passes, zero failures, four skips
  and zero not-run tests. It includes real Windows CIM integration and owned SSH
  profile child controls. The four skips require explicit Redis/native media
  opt-ins; they are not passes. Evidence is under
  `D:/pcai-relocation/main-integration-current-review-r3/mixed-r2`.
- The first live-CIM failure was a null memory commit limit. Its original sample
  was removed by the test fixture, leaving the provider cause OPEN. A later
  bounded provider query and retained complete collector capture returned valid
  counters without provider errors. The unchanged positive assertion now includes
  recorded provider-error diagnostics for future failures. These later successes
  do not erase the original failure or prove a performance improvement.
- Main's crash telemetry still converted three nullable secondary thermal fields
  to zero and could stop cleanup after its first disposal failure. The reviewed
  repair preserves main's unsigned energy getter and all existing control bodies,
  independently attempts remaining cleanup, and retains the original first error.
  Identical added assertions produce seven genuine old-source failures and seven
  candidate passes; focused candidate controls pass 17 tests. The first comparison
  runner incorrectly rejected 25 deliberately filtered tests; its failure is
  retained, and the successor checks the exact filtered population. These counts
  overlap with mixed validation and confer no global coverage credit.
- Carbon's four original media phases are closed: both CPU/FFI builds returned
  zero, the baseline poison test failed naturally at the expected terminal-state
  assertion, and the candidate passed all six ordered local helper cases. The
  distinct physical-receipt review supports only that narrow detector. General
  poisoned polling, cancellation resurrection, blocking mutex use and detached
  worker/shutdown ownership remain OPEN.
- Carbon's separate C# compatibility reproduction compiled six fixture artifacts
  successfully. The unchanged consumer passes against the old DTO and fails
  against the nullable DTO with five genuine `MissingMethodException` results.
  All 113 outputs were individually read back and verified. Independent review
  confirms that current main's DTO bodies match the breaking fixture. Production
  metrics deployment is HOLD pending an additive API and actual consumer/artifact
  qualification; compilation alone does not establish binary compatibility.
- All eight metrics sessions, twelve metadata resources and four media phases
  passed original-session closure checks. Carbon's original SSH parent and all
  previous owner/failure objects remain preserved; Sup4 and B897 were clear at
  final return. No forced termination was credited as a test pass.
- The foreign media preimage `1FC930274B372793E6A34151486A08F07E0A60E3F3E3111BB32A2E3EE119D334`
  is preserved. Two whole, reversible formatting changes yield current working
  hash `028C568349DB1DE93385E26C0A7924986B3AEE14947A2A5E1B3358E0A860BE5A`.
  Its code remains unstaged and its broad async integration remains HOLD.
- Published head `fac1d1a` still fails explicit coverage at 14,616/21,817 commands
  (66.9936% versus 85%), despite 2,897 PowerShell passes and successful other
  applicable checks. These old-head CI results do not qualify the pending merge.

Milly's completed lossless migration preserves its Linux paths and frees the
internal disk; the reconnected WDC SSD uses the faster 10 Gb/s UAS connection.
The bounded read check is separate from an application benchmark or restore test.
P1's last measured commit pressure still blocks the configured native MCP pilot.
Work access, master-vault reconciliation, optional backend/runtime qualification,
remaining PR/worktree disposition, exact-current-head CI and fleet equivalence
remain OPEN. Root owns the next source/consumer validation and publication;
no current clean-main, cross-machine synchronization or performance gain is claimed.

## Current CI and Docker startup guard

Cutoff: October 12, 2026, 00:38 UTC. Published head
`07599c69f7558b94b4608d40245827d634c49dd0` repairs three detecting test
inventories for the merged `Invoke-PcaiFleetToolCall` helper. The production
helper is unchanged. The earlier combined run had four streaming timeouts;
its successor passes all 51 focused controls with original fixture resources
closed. The transient timeout cause remains OPEN. CI for this head records
3,078 passes, zero failures, 41 skips and zero not-run tests in 546.2 seconds.
Its unchanged 85% coverage gate still fails at 14,647/21,928 commands
(66.795877%). Other applicable main jobs and NVIDIA validation succeed; Rust
Guidelines remains pending at this update. These results bind the published
head's synthetic PR merge `e94ec7a50cf8663f547f99b41dd8a735ae03aff3`.
The prior head's 23 PowerShell failures and lower coverage remain preserved.

The maintained [Serena launcher](../Tools/SystemScripts/SerenaLauncher/README.md)
now refuses unavailable local PostgreSQL before Docker discovery. Managed
PostgreSQL startup requires explicit `-AllowManagedPostgresStartup`. The previous
installed launcher is retained on D:; the exact reviewed successor was installed
using atomic replacement and hash readback. Seventeen focused controls and four
installed help/dry-run invocations pass. No ordinary backend, Docker, client
configuration or running-client restart was invoked. Native stderr sanitation,
effective Rust database configuration and full backend readiness remain OPEN.
The failed first installer is preserved; its null-string replacement and literal
help binding were corrected in a separate successor.

The bounded P1 profile still observed about 90% commit pressure. The MCP lineage
audit found 262 configured matches anchored to live clients, including Warp;
none is admitted for orphan termination. Five small PowerShell processes are
source-associated with VIGIL runner tooling. Its Docker startup path already
requires `-StartDocker`; the literal flag is absent from the five observed
invocations and configured task. Abbreviated or dynamic binding, loaded source,
runtime ownership and duplicate-instance custody require owner reconciliation;
no default Docker-startup defect is established for these runner processes.
Hard faults alone do not establish pagefile swapping, and no measured system
responsiveness improvement is claimed by these source or process checks.

Carbon received 86 additive metrics source files with complete byte/hash
readback. The first additive validation stopped during preflight: its preserved
original-object graph exceeded a 4,096-object bound. It recorded zero compiler
or consumer sessions, zero metadata resources and no output namespace. The
original first error is retained; the identity, readers and writers are closed,
and Sup4/B897 are clear. Its 499.6-second duration is whole preflight/cleanup time,
not an attributed routine benchmark. A separate source successor is required;
compatibility and full production-project validation remain pending.
All 39 cached package archives match their archive sidecars, and
their metadata content hashes match the existing restore records. The earlier
comparison mistakenly treated these distinct hash domains as interchangeable;
that finding is superseded and preserved in the source preparation receipts.
Fresh offline restore and package authenticity remain separate checks. The
binary compatibility deployment hold remains in effect.

## Integration and measured limits

Root observation cutoff: 2026-10-12 01:48 UTC. Last published head `2b0253d`
completed 3,095 PowerShell tests with zero failures, 41 skips and zero not-run
tests, but its genuine source-bound command coverage is 14,647 / 21,928
(66.7959%), below the unchanged 85% requirement. The 240 module source bindings
were independently verified. Another 3,992 distinct command hits are needed;
earlier behavioral passes cannot be counted again as coverage. Main's older
wildcard scope and absence of an explicit coverage assertion do not establish
equivalent coverage. All other applicable main jobs and the separate Rust
guidelines and NVIDIA workflows passed; integration tests were skipped.

Upstream USB descriptor PR #185 merged as main `7f8c250`. Root merged those five
disjoint files through normal hooks into signed local `ebee823`, preserving the
three existing dirty files byte-for-byte. This local merge is not yet published
or qualified by current-head CI. Foreign media `028C` remains unstaged. Carbon
has four linked worktrees, including dirty media work; local Candle has 32
unpublished commits and edits. None was retired.

Milly's live read-only storage check at 01:46 UTC confirms about 580 GiB internal
free space, WDC serial `1830D0800483` at 10,000 Mb/s with UAS, and accessible
original `av_recording` paths through the migrated Btrfs storage. All five
Btrfs device error counters are zero. Outer NTFS free space and inner Btrfs free
space are not additive. This is mount/access evidence, not cold-boot, restore,
Windows-access or physical M.2-slot qualification.
The earlier stdin transport's final machine-guidance loop failed parsing after
the storage records were emitted; that EXIT2 receipt is preserved. A normalized,
newline-terminated source retry returned EXIT0 with the same storage findings.

Metrics actor R2 failed preflight on the complete 10,230-node snapshot because
generic `Dictionary<string,string>` does not expose the same PowerShell
`Contains` overload as a Hashtable. It launched zero sessions, opened no
metadata resources and created no actual output namespace. Its two readers
and identity closed; elapsed preflight/cleanup was 37.316817 seconds. Both
failed owners and their original first errors remain retained. Distinctly
reviewed source R3 fixes interface dispatch while preserving every original
item/reference comparison and both failed-owner custody checks. Its actual
19-stage validation completed in original Carbon session 92653 with fresh
capacity and all 4,040 runtime pins. Twelve restore/build actors returned zero;
the original getter and initializer each passed against baseline and repaired
assemblies and genuinely failed against the held incompatible assembly. The
nullable wire control passed. All 19 sessions closed without force, six paired
metadata artifacts were inspected, and the original owner has no first error,
secondary error or pending local work. Physical readback verified all 251 output
pins (1,789,231 bytes); the result SHA256 is
`26DCBC13B17409906CA4F438E5E631C5444DB5F7786B70710252291733C157A3`.
This validates isolated compatibility controls. The full production assembly,
its actual native boundary and publish restore graph remain unqualified;
deployment remains HOLD. The existing 39-package cache does not contain the
runtime/host/crossgen packages needed to establish ReadyToRun publish closure.

The GitOps bus message now reports workflow metadata, workflow problems,
billing and upstream metadata separately. Twenty-six bounded fixture tests
passed, including realistic partial failures and privacy controls, and distinct
review accepted the two-file change. No live billing, bus delivery or performance
acceptance follows from those fixture tests. The first simultaneous native bus
claim refused to open its private outbox; a sequential retry succeeded. The
typed open cause remains unknown; no journal or lock was changed.

Three cleanup PATH existence checks now use literal paths. The same detecting
controls produced six passes and nine genuine failures on the original source,
then 15 passes on the candidate. Fifteen selected legacy PATH tests also passed;
11 unrelated tests were not run. Distinct review verified complete byte inverses
and accepted the exact three substitutions. Persistent PATH was unchanged.
The fixtures restore three tracked process variables, while TEMP/TMP changes
are confined to disposable test processes. Existing backup/log operations
precede ShouldProcess and are fixture-stubbed; these tests do not qualify all
filesystem IO under WhatIf. No global CI coverage credit or speedup is claimed.
The fresh agent-bus read retained both ambiguous monitor messages and the
unanswered runner custody request; it supplied no new authority to stop runners.

P1's memory-compression trial did not establish a performance gain. A six-sample
30.12-second baseline was captured on D:. `Enable-MMAgent -MemoryCompression`
returned but three legacy MMAgent readings became true. Legacy disable calls
returned "request not supported"; disabling compression returned and restored
that flag. Stopping SysMain restored all six saved MMAgent readings, with its
Automatic startup mode unchanged. Initial SysMain service state was not directly
captured, so complete original service-state equivalence is OPEN. Compression
remains disabled, and no enabled after-window was accepted. Evidence and failed
attempts are retained in `D:/pcai-relocation/p1-memory-compression-trial-r1`.

Fresh configured Work SSH still times out; the installed Android SDK ADB command
currently lists zero devices. Earlier successful Work or Pixel observations are
historical and do not establish current connectivity or vault reconciliation.
