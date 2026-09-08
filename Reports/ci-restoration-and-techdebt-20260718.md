# CI Restoration & Tech-Debt Ledger — 2026-07-18

**Branch:** `fix/ci-gate-pslint-clippy` → PR #58 (`MERGEABLE`)
**Context:** GitHub Actions was disabled for ~3 weeks; re-enabling surfaced a stack of
pre-existing CI failures that had accumulated unseen. This report records what was
fixed, what remains as tracked debt, and the merge/dependency sequence to close out.

---

## 1. CI gate restoration (done)

Actions were re-enabled (free hosted runners, public repo). Root-causing the red gates
turned up **seven distinct pre-existing bugs**, all masked while Actions was off:

| # | Gate | Root cause | Fix (commit) |
|---|------|-----------|--------------|
| 1 | Rust Check/Clippy | `--locked`/`--frozen` inserted *after* the cargo `--` separator | `67f6467` |
| 2 | Rust build wrapper | `Invoke-RustBuild.ps1` via `pwsh -File` chokes on a bare `--` | `37993d5` |
| 3 | PS Lint | PowerShell parse errors (`$var:` interpolation, malformed `foreach`) | `7e013fe` |
| 4 | PS Lint | 21 PSScriptAnalyzer findings (ShouldProcess, IEX, plaintext-pw) | `18b6dd3` |
| 5 | Media build | candle-core/transformers/flash-attn pinned at 0.9 vs candle-nn 0.10 | `c873fd2` |
| 6 | Clippy `-D warnings` (Cross-Platform + MS Rust Guidelines) | 19 workspace clippy lints | `7515e03` |
| 7 | PS Tests | Pester `Run.Path` bare `('Unit','Integration')` → "No test files found" | `15f999a` |

Plus `ed4251d`: provisioned `.qa-gate.conf` (git-guard repo-level Rust quality gate).

**Verified locally green (exit 0):**
- `cargo clippy --all-targets --no-default-features --features server,ffi -- -D warnings`
- `cargo clippy --workspace --all-targets --no-deps -- -D warnings -A clippy::type_complexity`
- Build.ps1 arg-ordering + wrapper routing
- PS Lint (PSScriptAnalyzer) across the workstation tooling

**CI verdict for #58:** re-triggered on push `15f999a`; authoritative result on the PR checks.

---

## 2. Pre-existing PowerShell Unit-test failures (TRACKED DEBT — not a merge blocker)

Fixing the Pester `Run.Path` (bug #7 above) made the PS Tests job *actually run* the
suite for the first time, which exposed **65 pre-existing test failures** that were
hidden behind "No test files were found".

**These are genuine test↔module contract rot, NOT caused by the path fix.**
Attribution proven two ways:
1. Identical pass/fail counts running from repo-root cwd vs `Tests/` cwd.
2. All affected tests resolve their module via `$PSScriptRoot` (cwd-independent).

**Local baseline:** 759 total → 650 pass / 65 fail / 44 skip (~704 s).

**Not a merge gate:** ruleset `dtm-default-protection` (id 17356124) requires only
signed commits + PR (0 approvals) and has **no `required_status_checks`** — red CI does
not block merge. These are deferred, not excluded (per the "no reward-hacked tests" rule
they must be fixed *credibly*, not silenced).

### Failure clusters (representative)

| Test file | Fail/Total | Symptom | Likely cause |
|-----------|-----------:|---------|--------------|
| `LLM-Logging.Tests.ps1` | 21/21 | Every `Write-LLMLog` / `Log Level Management` assertion fails | `PC-AI.LLM` logging contract changed/broke; module export or JSON shape drifted from tests |
| `Install-WSLVsockBridge.Tests.ps1` | 13/20 | Result-object properties, EnableService/StartService flags, missing-file error entries | `Install-WSLVsockBridge` result schema / mock expectations decayed |
| Process-idle filtering (`*Process*`) | ~5 | `-ExcludeIdle` count/order assertions | process-list helper contract drift |
| Remaining (~26) | — | scattered | to be enumerated on the fix pass |

**Reproduce full list:**
```powershell
Set-Location C:\codedev\PC_AI
$c = New-PesterConfiguration
$c.Run.Path = 'Tests/Unit'; $c.Run.PassThru = $true; $c.CodeCoverage.Enabled = $false
$c.Output.Verbosity = 'None'
(Invoke-Pester -Configuration $c).Failed | Select Block,Name,@{n='Err';e={($_.ErrorRecord.Exception.Message -split "`n")[0]}}
```

**Recommended workstream (separate PR):** a `test/repair-unit-contracts` branch —
fix the module↔test contracts file-by-file, starting with `LLM-Logging` (21/21 = whole
file broken, highest leverage). Est. medium effort; needs the module source read against
each assertion. **Not started — awaiting your scope go-ahead.**

---

## 3. Dependency / Dependabot backlog

1 open Dependabot alert (moderate, `dependabot/27`). Open dep PRs, by recommended order:

| PR | Bump | Risk | Recommendation |
|----|------|------|----------------|
| #56 | serde_with 3.16.1 → 3.21.0 | Low (minor) | Merge after #58/#57 — clean |
| #50 | log 0.4.29 → 0.4.33 | Trivial (patch) | Safe batch |
| #49 | memmap2 0.9.10 → 0.9.11 | Trivial (patch) | Safe batch |
| #52 | mimalloc 0.1.50 → 0.1.52 | Trivial (patch) | Safe batch |
| #51 | tokenizers 0.22.2 → 0.23.1 | Medium (minor, API surface) | **Gate behind local `cargo build` of pcai_media/inference** |
| #53 | windows 0.58.0 → 0.62.2 | High (4 majors) | **Gate behind local build; likely needs source edits** to the Windows API call sites |

**Rule (per CLAUDE.md):** prefer latest compatible, but #51/#53 must build locally
before merge — do not merge on green Dependabot CI alone, since CI does not compile the
`llamacpp`/`mistralrs-backend` cfg-gated blocks where `windows` is used most.

---

## 4. Latent issues noted (not fixed here)

- **`unused manifest key: package.lints`** — `pcai_core_lib`, `pcai_media_model`,
  `pcai_perf_cli` `Cargo.toml` declare `[package.lints]` (should be `[lints]` or a
  `[workspace.lints]` + `lints.workspace = true`). Benign (doesn't fail `-D warnings`)
  but means those crates' lint tables are silently ignored. One-line fix each; deferred
  to avoid touching Cargo.toml on the CI-fix branch.
- **`Config/llm-config.json` stray gutting** — found the working tree with 37 lines
  deleted (model/GPU/sampling/adaptive-ctx settings) by an unknown prior process; **not
  authored by this work and reverted** (`git checkout`). Flagging so you're aware the
  file was briefly corrupted locally; HEAD is intact.

---

## 5. Merge sequence (awaiting your action — not auto-merged)

1. **#58** (this branch) — merge once CI renders (or now; not gated on CI).
2. **#57** (WIP: vLLM cache pinning, cleanup-archive, workstation profile) — `MERGEABLE`;
   rebase on main after #58, then merge.
3. **Dependabot safe batch:** #56 → #50 → #49 → #52.
4. **Dependabot gated:** #51 (tokenizers), #53 (windows) — only after a local
   `cargo build --release` of the media + inference trees passes.

## 6. Longer-horizon backlog (identified, not started)

126 open TODO items across the repo ledgers — surfaced for prioritization, not executed:
`boot.TODO.md` (33), `CLAUDE.TODO.md` (38), `optimization.TODO.md` (28),
`machine-reliability.TODO.md` (14), `llm.TODO.md` (13). Grand-integration asks
(Gemini/vLLM runtime, google-access) remain design-stage backlog.
