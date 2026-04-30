# PC_AI Context: Jules Integration

> **Context ID:** ctx-pcai-20260327-jules-integration
> **Created:** 2026-03-27T13:20:00Z by claude-opus-4.6
> **Branch:** main @ 8542859
> **Schema Version:** 2.0

## Summary

Built a full Jules AI agent integration stack for PC_AI: API wrapper (1002 lines, all 10 REST endpoints + 5 CLI commands), batch review pipeline, PR triage dashboard, LLM orchestrator (smart dispatch + iterative plan review), GitHub Actions workflow (weekly/issue/manual), AGENTS.md Jules guidance section, and per-module review prompt templates (12 modules). Also audited and improved both PC_AI/CLAUDE.md and codedev/CLAUDE.md.

## Recent Changes

| File | Change |
|------|--------|
| `Tools/Invoke-JulesSession.ps1` | NEW: Full Jules REST API + CLI wrapper (1002 lines) |
| `Tools/jules_api.cmd` | NEW: CMD shim for wrapper |
| `Tools/Invoke-JulesBatchReview.ps1` | NEW: Batch dispatch across modules (102 lines) |
| `Tools/Get-JulesPRStatus.ps1` | NEW: PR triage dashboard via gh CLI (76 lines) |
| `Tools/Invoke-JulesOrchestrator.ps1` | NEW: LLM orchestration - smart dispatch + plan review (393 lines) |
| `Config/jules-review-prompts.json` | NEW: Per-module review prompts for 12 modules |
| `.github/workflows/jules-review.yml` | NEW: Weekly scan + issue trigger + manual dispatch |
| `Tests/Unit/Invoke-JulesSession.Tests.ps1` | NEW: 18 Pester tests (all pass) |
| `AGENTS.md` | MODIFIED: Added Jules agent guidance section (+40 lines) |
| `CLAUDE.md` | MODIFIED: Added PC-AI.Gpu module, nvidia-validation.yml, updated counts |
| `CLAUDE.TODO.md` | MODIFIED: Updated header with Jules orchestrator status |
| `docs/superpowers/specs/2026-03-27-jules-integration-design.md` | NEW: Design spec (632 lines) |
| `docs/superpowers/plans/2026-03-27-jules-integration.md` | NEW: Implementation plan (9 tasks) |

## Key Decisions

### dec-001: PowerShell-first tooling
- **Decision:** All Jules tools are PowerShell scripts in `Tools/` (not TypeScript/Node)
- **Rationale:** Consistent with existing repo pattern (69 scripts in Tools/), PS 7+ is the primary shell
- **Alternatives:** TypeScript with jules-sdk, Python wrapper, bash scripts

### dec-002: API key resolution chain
- **Decision:** 4-source fallback: `$env:JULES_API_KEY` > `~/.machine/*.json` > `.env` > `bw get notes`
- **Rationale:** User requested bw CLI and ~/.machine/* support alongside env var
- **Alternatives:** Single env var only

### dec-003: Jules CLI assumed broken
- **Decision:** API-only focus, CLI actions present but not relied upon
- **Rationale:** User reported Jules CLI is broken; all critical functionality uses REST API
- **Alternatives:** CLI-first approach

### dec-004: Plan approval gate on all sessions
- **Decision:** `requirePlanApproval: true` on every session, reviewed by LLM orchestrator
- **Rationale:** Prevents low-value PRs, enforces benchmark-first and convention compliance
- **Alternatives:** Auto-approve low-risk sessions

### dec-005: AGENTS.md over .jules/ directory
- **Decision:** Jules guidance in AGENTS.md, not a separate config directory
- **Rationale:** Jules reads AGENTS.md natively; no .jules/ directory standard exists
- **Alternatives:** Custom .jules/ config directory

## Agent Work Registry

| Agent | Task | Files | Status | Notes |
|-------|------|-------|--------|-------|
| jules-docs-research | Research Jules API/CLI | none | Complete | Comprehensive API docs compiled |
| jules-repo-state | Explore existing Jules integration | none | Complete | Found 53 sessions, 5 commits, CLI installed |
| jules-skills-research | Research jules-skills repo | none | Complete | SKILL.md format, 2 skills documented |
| task1-api-wrapper | Invoke-JulesSession.ps1 | Tools/*.ps1, Tests/ | Complete | 1002 lines, 18/18 tests pass |
| task2-config | jules-review-prompts.json | Config/ | Complete | 12 modules, validated |
| task3-batch | Invoke-JulesBatchReview.ps1 | Tools/ | Complete | 102 lines, dry-run verified |
| task4-pr-triage | Get-JulesPRStatus.ps1 | Tools/ | Complete | 76 lines, working |
| task5-orchestrator | Invoke-JulesOrchestrator.ps1 | Tools/ | Complete | 393 lines, dry-run verified |
| task6-agents-md | AGENTS.md update | AGENTS.md | Complete | +40 lines Jules guidance |
| task7-workflow | jules-review.yml | .github/workflows/ | Complete | 70 lines, 3 jobs |

## Orchestrator Signal Analysis (from dry-run)

- **Clippy violations:** 0 (clean)
- **Changed modules (last 20 commits):** pcai_media, pcai_media_model, pcai_core_lib, pcai_inference, PcaiNative, functiongemma, pcai_media_server, pcai_perf_cli, pcai_ollama_rs
- **TODO/FIXME markers:** 0
- **Open issues:** 0

## Next Steps

### Immediate
1. Commit-cluster the Jules integration files
2. Set up JULES_API_KEY in GitHub Actions secrets
3. Create `jules` label on GitHub repo
4. Run first batch review: `Invoke-JulesBatchReview -Scope RustCrates -MaxSessions 3`
5. Review pending Jules plans via orchestrator
6. Review and merge worthwhile Jules recommendations from prior 53 sessions

### This Week
7. Install `automate-github-issues` skill from jules-skills
8. Create custom `pcai-review` Jules skill
9. Monitor weekly scheduled review runs (Monday 6am UTC)

### Tech Debt
- Jules CLI broken — re-evaluate when Google fixes it
- `Set-StrictMode -Version Latest` removed from 2 scripts due to `.Count` on null arrays — investigate proper null-safe patterns
- Orchestrator's `Invoke-PlanReview` uses string matching on plan step titles — could use more sophisticated analysis

## Validation

- All 18 Pester tests pass
- All 6 smoke tests pass (API list, batch dry-run, PR triage, orchestrator dry-run, tests, CMD wrapper)
- Live API call confirmed working (sessions listed from Jules)
- `.pcai/` already gitignored
