# PC-AI fleet integration review

Evidence cutoff: October 8, 2026. Owner: Codex integration lane. Live receipts and
private build outputs remain under `.pcai/integration/`; private profile and hosts
originals remain outside Git. This report records verified work and outstanding
gates rather than declaring a fleet clean before those gates finish.

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
`c42347d`; a separate v0.2.9 release is being prepared without changing v0.2.8.

## Repaired behavior

- The module installer verifies staged bytes, preserves existing installations and
  unrelated module-path roots, and provides genuine dry-run behavior. Copies are
  the default; live junctions require an explicit selection.
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
  agreed on all 588 devices and three error devices. The capability catalogue
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
loaded. All 99 fixtures pass under both Pester 6.2 and CI's Pester 5.9.1.
Restoring discovery of parameterized module contracts exposed three further
public-surface defects. After correction, all 286 module-loading contracts pass
under both Pester versions, with no skipped or undiscovered cases. The bounded
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

| Surface | Current evidence | Owner and next action |
| --- | --- | --- |
| dtm-carbon-two | CPU-only Intel graphics; Rust stable 1.99.0 and private SDK 10.0.401 verified. Core passed 59 unit tests, one documentation test and strict Clippy. Managed candidate passed 17 tests and 18 independent native fixtures. Alternating benchmark pairs show no material candidate regression or speedup. Canonical profile has an intentional alternate structure; the p1 repair planner fails closed. | Root/fleet: coordinated final-head stamped rebuild and CPU deployment preserving profile/auth, junctions, private overlays and the untracked build script. Configured model assets are absent; shared GPU defaults need host-specific overrides before AI execution. |
| dtm-work | Strict SSH reached DTM-WORK after the peer returned online. Work has PowerShell 7.6.6, approximately 112 GiB free RAM and a Quadro RTX 4000. The actual checkout is `C:/codedev/pc-ai`, with sixteen tracked changes and three untracked source files retained. Its profile shim selects a missing canonical core; the surviving OneDrive profile and WinGet executables require consumer-path repair. Real Bitwarden/rclone package executables run, while their WinGet links fail with an untrusted-mount-point error in SSH. Vault bootstrap/cloud sync and the existing Cloudflare management credential succeeded; the Network key's Site Manager 401 remains a separate scope result. | Root/fleet: preserve exact Work WIP bytes and refs, reconcile unique source changes and repair profile/tool paths with original custody before host-specific deployment. The USB Pixel is ADB-authorized and locked; installed credential/network apps were identified without logging credentials or changing account/network policy. |
| Local name resolution | Removed one obsolete Headscale IPv6 hosts entry and a conflicting bare dtm-work LAN token; preserved radius LAN alias and tailnet mapping with original-byte custody. | Fleet lane: verify route retries; this does not prove endpoint recovery. |
| Candle/media PR 156 | Corrected Candle 0.11 union passed 125 media, 58 model, 18 server and 23 documentation tests; three model cases explicitly require absent real-model fixtures. Default CPU media tests passed 119 cases. Optional CPU/NVML/upscale Clippy passed. A fresh CPU DLL exposes all thirteen base exports; actual PowerShell initialization, missing-model error, async unknown-request status and shutdown checks passed. | Dependency/root: CPU source integrated; finish bounded GPU compilation and deterministic CPU/GPU comparisons before GPU publication. Full trained-model quality and the three absent-fixture tests remain distinct gaps. |
| Mistral backend | SDK/core 0.8.1 alignment and the merged dependency graph passed 140 library and two CLI tests with no skips, including CPU selection, invalid device ordinals and empty responses. A full locked metadata check repaired four optional Candle lock entries lost during an otherwise clean textual merge. | Root/fleet: finish production Release binaries and isolated CPU model smoke. Private source-only Work build archive has an explicit version/environment manifest; it does not alter Work's checkout or defaults. |
| Linked dtm-codex | PRs 34/35 integrated with current-head and merged-main CI; completed branches retired after preservation. | Root: source deployment/consumer validation if this host consumes the changed launcher; keep imported skills under their existing custody. |
| Existing WIP and private evidence | The three untracked private boot/Thunderbolt evidence files now have verified original and copy custody outside Git under stable recovered-evidence revisions. Their mapping is `.pcai/integration/preserved-evidence-migration.json`; historical inventory observations retain their original paths. SQLite/Python compilation caches remain in place with specific generated-cache exclusions. Carbon build script and private overlays remain preserved. | Root/fleet: refresh final inventory and cross-machine equivalence after source integration; no reset or forced cleanup. |

Work's unique benchmark/profile and native-sampler changes have separate source
custody. The preserved pre-change performance binary built successfully and its
actual worker measurements record child and observer interval CPU; the candidate
port must establish output parity before any speedup claim. UniFi's reviewed
source fixes sparse/malformed Protect telemetry and boot timestamp units. Its
full Python run passed 4,115 tests with 131 explicit skips and 83.10% coverage;
actual Windows SSH/MCP compilation and current-head hosted CI remain required.

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
