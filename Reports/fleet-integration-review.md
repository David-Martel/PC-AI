# PC-AI fleet integration review

Evidence cutoff: October 8, 2026. Owner: Codex integration lane. Live receipts and
private build outputs remain under `.pcai/integration/`; private profile and hosts
originals remain outside Git. This report records verified work and outstanding
gates rather than declaring a fleet clean before those gates finish.

## Reviewed integrations

PC-AI PRs 178, 179 and 180 are merged. Current main `b92e5f6` passes its complete
CI workflow. The workload documentation's archived launch and summary hashes were
verified locally. Dependency lock updates were reviewed together with consumers.

CargoTools PR 13 now honors explicit preflight disable at every quality boundary,
including reordered flags and environment settings. Local full validation passed
644 tests, with five skipped and nine opt-in tests not run. Exact-head hosted CI
is still queued at this cutoff. PR 12 remains pending its own telemetry validation.

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
- CUDA media builds require optional cuDNN and FlashAttention only when requested.
  Build and tooling reports allocate stable revisions and retain earlier evidence.
  Benchmark CLI comma-separated case selectors now work as documented.
- The C# bridge passes a strict Release build with zero warnings and errors.

## Verification and measurements

Local tools: PowerShell 7.6.6, Rust 1.99.0 and .NET SDK 10.0.401. The bridge retains
its .NET 8 consumer target. The core Release build passed 59 unit tests and its
documentation test; 17 C# tests passed against that explicitly selected fresh DLL.
Three PowerShell native-search integration tests passed without skips. Twenty
profile repair fixtures and seventeen installer fixtures passed; twenty-one bundle,
shim, artifact and build-feature fixtures passed. The actual media DLL remains
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
| dtm-carbon-two | Reachable; CPU-only Intel graphics. Rust stable updated to 1.99.0, pinned legacy toolchain preserved. Verified SDK 10.0.401 staged privately. | Fleet lane: finish exact-source CPU/native/managed checks, then coordinated default deployment preserving profile/auth and runtime overlays. |
| dtm-work | Gateway alive; Headscale peer offline. No dtmventures LAN subnet route from either available machine. Existing direct, LAN and UDM jump routes remain unavailable. | Fleet/access lanes: restore an existing authenticated endpoint route; remote source, users and deployment remain unverified. |
| Local name resolution | Removed one obsolete Headscale IPv6 hosts entry and a conflicting bare dtm-work LAN token; preserved radius LAN alias and tailnet mapping with original-byte custody. | Fleet lane: verify route retries; this does not prove endpoint recovery. |
| Candle/media PR 156 | Fresh CPU tests pass 110 media and 58 model tests. Missing/partial weights reject before use; optional vision fallback and CPU FlashAttention guards are exercised. | Dependency lane: optional feature checks, reviewed commit/CI/integration. Real GGUF and GPU inference require representative assets/hardware. |
| Mistral backend | SDK/core upgrade is aligned; selected device is passed into both builders and empty responses return an error. | Root: complete actual optional-feature compilation and response tests before integration. |
| Linked dtm-codex PR 34 | False universal policy-deployment claim corrected; source/instruction checks pass, remote review-thread issue addressed. | Root: current-head CI and guarded integration, preserving 489 imported skill files. |
| Existing WIP and private evidence | Local SQLite index, boot XMLs, Thunderbolt backup receipt and Python cache retained; Carbon build script and private overlays retained. | Root/fleet: review custody and consumer references before any retirement; no reset or forced cleanup. |

Outstanding broader platform work remains in `TODO.md`, `optimization.TODO.md`
and `boot.TODO.md`, including cancellation/schema parity, media fixture expansion,
startup latency and post-reboot workstation validation. CI and offline/native checks
do not establish real-model quality, GPU performance or a clean post-reboot window.
