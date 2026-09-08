# PC-AI Context — CI Restoration & Tech-Debt Ledger

**ID:** `ctx-pcai-ci-restoration-20260718` · **Date:** 2026-07-18
**Branch:** `fix/ci-gate-pslint-clippy` → PR #58 (`MERGEABLE`) · **Head:** `15f999a`

## One-line state
Re-enabled GitHub Actions (off ~3 weeks), root-caused **7 distinct pre-existing CI gate
failures** and fixed all. Landed WIP (PR #57), aligned git/QA config, provisioned
git-guard repo gate. PR #58 carries the final two fixes (19 clippy `-D warnings` lints +
Pester `Run.Path`). Full ledger: `Reports/ci-restoration-and-techdebt-20260718.md`.

## What shipped (commits on fix/ci-gate-pslint-clippy)
- `67f6467` build: `--locked/--frozen` before cargo `--` separator
- `37993d5` build: route `--`-separated cargo cmds past pwsh -File wrapper
- `7e013fe` lint: PowerShell parse errors (`$var:` interp, malformed foreach)
- `18b6dd3` lint: 21 PSScriptAnalyzer findings (ShouldProcess/IEX/plaintext-pw)
- `ed4251d` qa: `.qa-gate.conf` git-guard repo Rust gate
- `c873fd2` deps: candle-core/transformers/flash-attn 0.9→0.10 (media tree)
- `7515e03` lint: 19 workspace clippy `-D warnings` (both gates exit 0, re-verified)
- `15f999a` test: Pester Run.Path → Tests/Unit,Tests/Integration

## Verified green locally (exit 0)
- `clippy --all-targets --no-default-features --features server,ffi -- -D warnings`
- `clippy --workspace --all-targets --no-deps -- -D warnings -A clippy::type_complexity`

## Key discoveries
- **CI is NOT a merge gate.** Ruleset `dtm-default-protection` (id 17356124) = signed
  commits + PR (0 approvals), **no required_status_checks**. Red CI ≠ blocked.
- **65 pre-existing PS Unit-test failures** surfaced (not caused) by the Pester fix.
  Attribution proven: identical counts repo-root vs Tests/ cwd; all use `$PSScriptRoot`.
  Clusters: LLM-Logging 21/21, WSL vsock bridge 13/20, process-idle ~5. Tracked debt.
- **Stray corruption caught:** `Config/llm-config.json` was gutted (37 lines) by an
  unknown prior process — reverted; HEAD intact.

## Resume here (awaiting user go-ahead — nothing auto-merged)
1. **Merge sequence:** #58 → #57 (rebase) → Dependabot safe batch #56/#50/#49/#52 →
   #51 tokenizers & #53 windows 0.58→0.62 **gated behind local `cargo build`**.
2. **65-test repair** as separate `test/repair-unit-contracts` PR — start with
   LLM-Logging (whole file broken). NOT started; needs scope confirmation (no
   reward-hacked/silenced tests).
3. **Latent:** `unused manifest key: package.lints` in 3 Cargo.toml (should be `[lints]`).
4. **Backlog (not started):** 126 TODO-ledger items; Gemini/vLLM + google-access
   grand-integration asks remain design-stage.

Full detail: `Reports/ci-restoration-and-techdebt-20260718.md`
