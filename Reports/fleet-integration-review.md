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
passed all sixteen cases afterward. Installed hooks retain their qualified v0.2.7
release while source-main postmerge checks and branch custody finish separately.

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
| Candle/media PR 156 | Earlier CPU tests passed 110 media and 58 model tests. Actual optional linking exposed a static/dynamic CRT conflict in an unused C++ tokenizer trainer; the corrected feature graph passes all-target optional checks without dependency-version churn. A nonempty partial checkpoint could still allocate before failing; constructor/header metadata preflight is being added. | Dependency lane: finish actual corrected union tests, strict Clippy, fresh FFI build and reviewed integration. Real trained-model/GGUF/GPU execution remains separate. |
| Mistral backend | SDK/core 0.8.1 alignment and actual optional compilation completed. Nine backend tests passed, including empty responses, exact token/finish reporting and device ordinals without CPU fallback. | Root: complete the two CLI configuration tests, final-head runtime build and isolated CPU model smoke. |
| Linked dtm-codex | PRs 34/35 integrated with current-head and merged-main CI; completed branches retired after preservation. | Root: source deployment/consumer validation if this host consumes the changed launcher; keep imported skills under their existing custody. |
| Existing WIP and private evidence | Local SQLite index, boot XMLs, Thunderbolt backup receipt and Python cache retained; Carbon build script and private overlays retained. | Root/fleet: review custody and consumer references before any retirement; no reset or forced cleanup. |

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
