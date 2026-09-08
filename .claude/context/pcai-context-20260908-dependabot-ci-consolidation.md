# PC-AI Context — Dependabot Consolidation, CI Repair, Windows-Only Scope

**ID:** `ctx-pcai-dependabot-ci-20260908` · **Date:** 2026-09-08
**Branch:** `chore/repo-hygiene-v2-20260908` · **Head:** `bd8afe3`
**Main:** `2b916b4` (PR #58 merged)

## One-line state
Merged PR #58; consolidated a 9-PR dependabot backlog into **#71**; salvaged #57's real
work into **#72**; corrected my own bad change in **#74**. Repaired CI triggers that had
**never fired**, deleted the Linux gate after the owner confirmed cross-platform is a
non-goal, and added the **first .NET build gate** the repo has ever had. Three PRs open
and green-or-pending; ten dependabot PRs staged for closure.

## Owner constraints established this session (durable)
1. **Windows-only by design.** Cross-platform support was never intended. Linux is in
   scope *only* where a Docker/WSL build/test/deploy requirement needs it — never as a
   portability target. Recorded in `CLAUDE.md` under **Platform scope**.
2. Use latest Rust/build tooling compatible with the local **sccache** server.

## What shipped

### PR #71 — deps + CI (`8637cdd` → `65fb022`)
- Five Cargo bumps re-resolved against current main: `log` 0.4.34, `memmap2` 0.9.11,
  `mimalloc` 0.1.52, `windows` 0.58→**0.62.2**, `tokenizers` 0.22→**0.23.2**.
  Supersedes #69/#49/#52/#53/#70.
- `System.Text.Json` → 10.0.11 in **both** PcaiNative *and* PcaiServiceHost (the latter
  had drifted to 10.0.7, unwatched). Supersedes #64.
- **Push triggers fixed.** `ci.yml`, `rust-guidelines.yml`, `nvidia-validation.yml` all
  declared `push: branches: [develop]`. **`develop` does not exist** — so none had ever
  run on a push and nothing validated `main` after a merge. Now `[main]`; `develop`
  removed everywhere rather than left as a dead reference.
- **`portable-ci.yml` deleted.** Linux job running the whole PowerShell suite; red on
  every branch since forever. No Rust coverage lost — `rust-guidelines.yml` already runs
  the same workspace-wide fmt/clippy/test on `windows-latest`, and `Native/pcai_core/Cargo.toml`
  has no `[package]` table so root-level cargo covers all seven members.
- **New `dotnet-build` job** — the repo had *no* .NET build gate at all. Compiles all five
  csproj and runs both test projects; wired into the CI Gate's `needs`. Discovers projects
  with a positive control (fewer than five found is a hard error).
- `dependabot.yml`: grouping added (minor+patch → one PR per ecosystem), coverage extended
  from 2 to 5 .NET projects, whole candle stack `ignore`d.

### PR #72 — salvage from #57 (`1598e75`, `f842260`, `ca26bde`)
- vLLM compose: caches pinned to `T:\Models`, explicit vLLM cache mount, model mount `:ro`.
- `Tools/Invoke-CleanupArchive.ps1` (rclone archive tool) + manifest.
- **Six review fixes, one a data-loss bug** — see Findings below.

### PR #74 — evidence collectors (`9c8042b`, `bd8afe3`)
Replaces **#73**, which I got wrong. Fixes the *producers*, leaves artifacts untouched.

## Findings (each verified, not inferred)

| # | Finding | Evidence |
|---|---|---|
| 1 | **`Deploy` workspace does not compile on main.** 10 errors in `rust-functiongemma-core` (`model.rs:91,149-151,206-209`). `qlora-rs = "1.0.5"` caret range drifted to **1.3.0**, which moved to candle 0.11 — crate mixes candle 0.9.2 and 0.11.0 types. `--precise 1.0.5` doesn't help (qlora-rs 1.0.5 fails on its own, 9 errors). | Baseline run on **unmodified main** gives identical 10 errors |
| 2 | **PR #66 would not compile.** Bumps `candle-transformers` alone → two semver-incompatible `Tensor` types in one crate; vendored kernels pinned 0.9.2 | Lockfile dependent analysis |
| 3 | **Data loss in the archive tool.** Remote key was leaf filename alone; same-named sources overwrote each other while each verified against its *own* upload and both were deleted | Code read + reproduced in test harness |
| 4 | **`PcaiServiceHost.csproj` copied `pcai_core_lib.dll` with no `Exists` condition** (unlike PcaiNative). Only built on machines that had run the Rust build first | CI MSB3030; verified fix both directions |
| 5 | **Locale-dependent event classification.** Matching `"No events were found"` misclassifies a successful empty query as a failure on non-English Windows | Triggered the case: `NoMatchingEventsFound,Microsoft.PowerShell.Commands.GetWinEventCommand` |
| 6 | **`ConvertTo-Json` on an empty pipeline emits nothing**, not `[]` — the mechanism behind seven zero-byte artifacts. Both obvious fixes are wrong | `@()\|ConvertTo-Json -AsArray`→`''`; `-InputObject @() -AsArray`→`[[]]`; correct is `,$rows \| ConvertTo-Json` |
| 7 | **`cl.exe` on PATH is a `~/bin` shim**, so nvcc dies "Failed to preprocess host compiler properties" until run through `vcvars64.bat` | Reproduced directly |
| 8 | **Merge blocker is review threads, not checks.** Ruleset: `required_review_thread_resolution: true`, `required_approving_review_count: 0`, **no `required_status_checks`** | `gh api .../rules/branches/main` |

## Decisions
- **dec-001 — merge method.** #58 merged as a **merge commit**, deviating from CLAUDE.md's
  squash convention, to preserve 55 commits of root-cause analysis. Flagged to owner.
  Later PRs follow the squash convention.
- **dec-002 — delete `portable-ci.yml` rather than fix 24 test files.** 144 Windows-specific
  constructs across 24 files; the tests are correct and were mistagged `Portable`.
- **dec-003 — do not rewrite evidence artifacts.** TODO.md already said the fix belongs in
  the producer. `[]` asserts "query succeeded, found nothing", which a possibly-failed
  capture cannot support. #73 closed, #74 opened.
- **dec-004 — exclude #67/#68.** Their workspace does not build, so the bumps cannot be gated.
- **dec-005 — do not touch the ruleset or delete remote branches.** Owner's call.

## Agent Work Registry
| Agent | Task | Files | Status |
|---|---|---|---|
| (main session) | All work this session | see PRs #71/#72/#74 | In flight |
| Copilot (review bot) | Found tokenizers manifest drift, 3 archive-tool bugs, locale bug | — | Addressed + resolved |
| chatgpt-codex-connector (review bot) | Found .NET gate gap, archive-tool data loss, dry-run mutation, help switches, audit-trail ordering, locale bug | — | Addressed + resolved |

## Blockers / open decisions for the owner
1. **The CI Gate is still not binding.** No `required_status_checks` on ruleset 17356124 —
   any PR can merge red. Fixing the triggers did *not* fix this; they are separate problems.
   Adding it has lockout potential, so it needs an explicit decision.
2. **Two orphan branches, provably redundant, not deleted.** `preserve/local-work-20260819`
   (zero unique content) and `feat/import-nukenul` (re-imports `6765c50`, which is the HEAD
   of the standalone `C:\codedev\nukenul`, already pushed).
3. **`Deploy` workspace repair** — needs a coordinated candle decision (pin qlora-rs to a
   candle-0.9.2-compatible release, or move the whole stack to 0.11 and re-vendor kernels).
4. **`Reports/workstation-profile-google-codex-plan-20260718.md`** — #57's 369-line plan vs
   main's 145-line validated-outcome doc, same filename, different documents. Keep #57's
   branch until decided.
5. `Tests/Invoke-PortableTests.ps1` is now vestigial (nothing references it) — remove or keep?

## Roadmap
**Immediate:** merge #71 → #72 → #74; run the closure script for #49/#52/#53/#64/#66/#67/#68/#69/#70; close #57 after #72.
**Next:** decide items 1-5 above. Confirm a **push-event** ci.yml run actually appears on
main after #71 lands — silent non-firing is the exact failure mode being fixed.
**Tech debt:** `integration-tests` runs 1 of ~23 suites; `maintenance.yml` checks one crate;
374 Windows-PowerShell-5.1 failures (baseline, gated against regression only).

## Validation anchors re-run this session
```
cargo fmt --all --check                                        clean
cargo clippy --workspace --all-targets -D warnings             clean
cargo test --workspace --features server,ffi --lib             229 passed, 0 failed
cargo check -p pcai-media -p pcai-media-model -p pcai-media-server  clean
dotnet build (5 projects, Release)                             5/5 succeeded
dotnet test  (2 projects)                                      9 + 6 passed
Invoke-CleanupArchive harness                                  7/7 pass
Collect-RemainingEventSources (temp dir)                       7 files, 0 empty, 3 valid []
```
