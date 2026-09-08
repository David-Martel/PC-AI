# LATEST CONTEXT

**Pointer →** [`pcai-context-20260908-dependabot-ci-consolidation.md`](pcai-context-20260908-dependabot-ci-consolidation.md)
**ID:** `ctx-pcai-dependabot-ci-20260908` · **Date:** 2026-09-08 · **Main:** `2b916b4`

## One-line state
PR #58 merged. A 9-PR dependabot backlog consolidated into **#71**; #57's real work
salvaged into **#72**; my own bad change corrected in **#74**. CI push triggers that had
**never fired** repaired, `portable-ci.yml` deleted (owner confirmed cross-platform is a
non-goal), and the repo's **first .NET build gate** added.

## Owner constraint established 2026-09-08 (durable)
**This repo is Windows-only by design.** Cross-platform support was never intended. Linux
is in scope *only* where a Docker/WSL build/test/deploy requirement needs it — never as a
portability target. Do not "fix" tests to run on Linux; a Linux job running the PowerShell
suite is a misconfiguration, not coverage. Recorded in `CLAUDE.md` → **Platform scope**.

## Resume here
1. Merge **#71** → **#72** → **#74** (in that order; #72/#74 branched before the
   `portable-ci.yml` deletion — neither touches that file, so a merge cannot resurrect it).
2. Run the closure script for the superseded dependabot PRs:
   `#49 #52 #53 #64 #69 #70` (superseded by #71), `#66` (would not compile),
   `#67 #68` (Deploy workspace does not build). Close **#57** after #72 merges.
3. **Confirm a push-event `ci.yml` run actually appears on `main`** after #71 lands —
   silent non-firing is the exact failure mode being fixed.

## Open decisions for the owner (do NOT do unprompted)
- **The CI Gate is still not binding.** Ruleset 17356124 has **no `required_status_checks`**,
  so any PR can merge red. Fixing the triggers did not fix this — separate problems.
- **Two orphan branches, provably redundant, not deleted:** `preserve/local-work-20260819`
  (zero unique content) and `feat/import-nukenul` (re-imports `6765c50`, already the HEAD of
  the standalone, pushed `C:\codedev\nukenul`).
- **`Deploy` workspace does not compile on `main`** — `qlora-rs` drifted from the declared
  `1.0.5` caret range to `1.3.0`, which moved to candle 0.11, so the crate mixes candle
  0.9.2 and 0.11.0 types. Needs a coordinated candle decision, not a pin.
- `Reports/workstation-profile-google-codex-plan-20260718.md` — #57's 369-line *plan* vs
  main's 145-line *validated outcome*, same filename, different documents. Keep #57's
  branch until decided.

## Gotcha worth remembering
`gh pr merge` failing with *"the base branch policy prohibits the merge"* here means
**unresolved review threads** (`required_review_thread_resolution: true`), not failing
checks. Copilot and `chatgpt-codex-connector[bot]` post threads automatically — and they
found three genuine defects this session, including a data-loss bug.
