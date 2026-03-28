# Local LLM Quality Assessment — March 28, 2026

## Task-Model Matrix (validated on real PC_AI code)

| Task | Best Model | tok/s | Quality | Notes |
|------|-----------|-------|---------|-------|
| **Quick triage** | 3B | 115 | 6/10 | JSON findings, 10 files in 7s |
| **Documentation** | 7B | 64 | 8/10 | Correct types, P/Invoke examples |
| **Code review** | 7B | 64 | 7/10 | Specific bugs with line numbers |
| **Complex analysis** | 30B MoE | 72 | 9/10 | Production-quality, hits token limits |
| **CI pipeline** | 7B | 62 | 7/10 | Identifies missing steps, parallelization |
| **Test generation** | 7B | 64 | 6/10 | Correct structure, sometimes generic |

## Synergistic Workflow (Local + Cloud)

### Layer 1: Local LLM Triage (seconds)
- 3B scans 10 files in 7.4 seconds
- Produces JSON-structured findings
- Catches obvious issues: missing error handling, unused variables

### Layer 2: Local LLM Deep Review (minutes)
- 7B reviews individual files with specific findings
- Generates documentation with correct FFI types
- 64 tok/s sustained, ~20s per file

### Layer 3: MoE Quality Analysis (minutes)
- 30B MoE produces frontier-quality analysis at 72 tok/s
- Handles complex multi-concept prompts
- Quality comparable to small cloud models

### Layer 4: Cloud LLM (Claude/Codex/Jules)
- Multi-file architecture review
- Verified fixes with compilation checks
- PR creation with full test coverage
- Human-level code review with context

## Quality Scores (Documentation Generation)

| Model | Structure | Type Accuracy | Safety Docs | Examples | Overall |
|-------|-----------|--------------|-------------|----------|---------|
| 3B | Correct | Generic | Boilerplate | Wrong types | **6/10** |
| 7B | Natural | **Correct** | Module-specific | **Correct P/Invoke** | **8/10** |
| 30B MoE | Rich | Correct | Specific | Detailed | **9/10** |

## Recommendation

Use `Invoke-LocalLLMReview.ps1` with task-appropriate models:
- `-Task Triage` → 3B (speed priority, feed findings to Jules/Claude)
- `-Task CodeReview` → 7B (quality priority, human-reviewable)
- `-Task Documentation` → 7B (correct types matter for FFI/P/Invoke)
- Complex analysis → qwen3:30b via direct Ollama API
