# Doc-Automation & Repo-Analysis Tooling — Evaluation & Reconciliation

**Date:** 2026-06-06 · **Scope:** the tooling that automates repo analysis and documentation, its
functionality/relevance/deficiencies, plus CLAUDE.md accuracy drift. Read-only evaluation; no pipeline runs mutated state.

## 1. Tooling status

| Tool | Functional? | What it does | Deficiencies needing reconciliation/validation |
|---|---|---|---|
| `Tools\validate-doc-accuracy.ps1` | **Works** (exit 0) | 3 checks: TOOLS.md vs `Config\pcai-tools.json`; training_data.jsonl tool-call validity; DOC_STATUS self-ref pollution | **Found real drift** (schema 31 tools vs TOOLS.md 28). Always returns `Continue` → no non-zero exit, **so CI cannot gate on it**. The DOC_STATUS-pollution check is now **dead signal** (reports 0/0/0; the pollution it was written for is gone). Hardcoded paths. |
| `Tools\generate-auto-docs.ps1` | Static-analyzed (no dry-run; always writes) | Unified generator: ast-grep doc-status + tool-coverage, PS module index, C# XML doc inventory, Rust `cargo doc` index, `AUTO_DOCS_SUMMARY.md` | **No read-only/dry-run mode.** C#/Rust doc *building* gated behind `-BuildDocs` (default off). Hardcoded external path `C:\codedev\nukenul\nuker_core`. **Duplicates** DocPipeline's PS/Rust steps → two code paths to keep in sync. |
| `Tools\Invoke-DocPipeline.ps1` | Reads clean; not executed | Master orchestrator (Full/DocsOnly/TrainingOnly/Validate); 10 steps incl. Litho extract + doc-gen + training data | **Litho steps are dead on this machine:** they look for a binary named `litho`/`~\bin\litho.exe` which **does not exist** — the real binary is **`deepwiki-rs.exe`** (different name/CLI), so both Litho steps always silently `Skip`. **Output-path mismatch:** writes `docs\auto\litho` but `litho.toml` targets `docs/auto/deepwiki_litho_docs`. Litho doc-gen uses `--provider codex` (repo standard is Ollama/openai). TrainingData step warns-skips without CUDA/`-UseNativeRouter`. |
| `.litho\litho.toml` (deepwiki-rs) | Config valid; **never run for this repo** | C4 doc config: provider `openai`@`127.0.0.1:8085/v1`, model `qwen2.5-coder`, 7 categories | `deepwiki-rs.exe` **is present** (PATH + `C:\codedev\.claude\tools\bin\`), but **`docs/auto/` does not exist** → docs stale-by-absence. Config endpoint **port 8085** vs the 11434 Ollama port referenced elsewhere — verify which the local server listens on before running. |
| `Reports\` generators (`update-doc-status.ps1`, `generate-functiongemma-tool-docs.ps1`, `generate-api-signature-report.ps1`, `generate-tools-catalog.ps1`, `prepare-functiongemma-router-data.ps1`) | Present | Produce DOC_STATUS/TOOLS/API_SIGNATURE/TOOLS_CATALOG + router data | **All outputs 2–4 months stale** (mtimes Jan–Mar 2026 vs 2026-06-06). Pipeline not re-run recently → reports don't reflect current code. |

## 2. Reconciliation actions (priority order)

1. **Re-run the doc pipeline** (`.\Tools\Invoke-DocPipeline.ps1 -Mode DocsOnly`) to refresh the 2–4-month-stale `Reports/` outputs. *(Validation gap: confirm it completes without the Litho steps erroring.)*
2. **Fix TOOLS.md drift:** `Config\pcai-tools.json` has **31** tools; `Deploy\rust-functiongemma\TOOLS.md` documents **28** — missing `pcai_analyze_image`, `pcai_generate_image`, `pcai_media_status`. Re-run `Tools\generate-functiongemma-tool-docs.ps1`.
3. **Repair the Litho integration** in `Invoke-DocPipeline.ps1`: point at `deepwiki-rs.exe` (not `litho`), align output path to `docs/auto/deepwiki_litho_docs`, and use the Ollama/openai provider args from `CLAUDE.md` instead of `--provider codex`. Then generate the never-produced C4 docs.
4. **Make `validate-doc-accuracy.ps1` CI-gateable:** return a non-zero exit on drift, and retire/replace the dead DOC_STATUS-pollution check with a live signal.
5. **De-duplicate** `generate-auto-docs.ps1` vs `Invoke-DocPipeline.ps1` (one should call the other) to stop two code paths drifting.

## 3. CLAUDE.md accuracy — fixed this session
- "89 utility scripts" → "~81 top-level (245+ incl. subfolders)". ✅ corrected
- `.github/workflows` table: added `portable-ci.yml` + `jules-review.yml` (11 actual, 9 were listed). ✅ corrected
- Litho docs: added a "not yet generated / wrong binary name" status note. ✅ corrected

**Verified accurate (kept as-is):** 7-crate `pcai_core` workspace; per-file Rust test counts (lib 5, config 8, backends/mod 7, http/mod 21, ffi/mod 17, version 3 — all exact); "65+ in pcai_inference" (actual 67); PcaiNative ~20 modules (21 .cs); PcaiChatTui/PcaiServiceHost/nukenul all exist; Tests "80+" holds (74 Pester + 80 Rust). Minor/deferred: PcaiNative "20"→21; Modules `Archive\` subdir undocumented.
