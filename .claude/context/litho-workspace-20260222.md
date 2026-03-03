# litho-workspace Implementation Context

**Context ID**: ctx-litho-workspace-20260222
**Created**: 2026-02-22T20:50:00-05:00
**Branch**: main @ 58f4dc7
**Repository**: C:\codedev\litho-workspace

## Summary

Complete implementation of litho-workspace, a 6-crate Rust monorepo that restructures deepwiki-rs into a modern codebase documentation toolkit. Replaces 6,889 lines of regex-based language processors with tree-sitter AST parsing, and 14 LLM round-trips with a single codex-cli invocation. All 125 tests pass, zero warnings.

## Architecture

```
litho-workspace/
├── crates/
│   ├── litho-core/        # Shared config + types (LithoConfig, ExtractedCodebase)
│   ├── litho-extract/     # Tree-sitter AST extraction (Rust/TS/Python/C#)
│   ├── litho-codex/       # codex-cli exec provider (single invocation)
│   ├── litho-cli/         # clap CLI: `litho extract` + `litho generate`
│   ├── litho-generator/   # Legacy deepwiki-rs pipeline (imported, ~3,684 lines)
│   └── litho-book/        # Web documentation reader (axum, imported)
├── .cargo/config.toml     # CC_x86_64_pc_windows_msvc=cl.exe (critical)
├── .github/workflows/ci.yml  # Build/test/clippy + tag-based release
└── docs/plans/            # Design doc + implementation plan
```

### Dependency Flow (strictly layered)
```
litho-core (no deps on siblings)
  └─> litho-extract (depends on core)
        └─> litho-codex (depends on core + extract)
              └─> litho-cli (depends on all three)
litho-generator (standalone, legacy)
litho-book (standalone, imported)
```

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| AST parsing | tree-sitter 0.26 | In-process, no external tools, 4 language grammars |
| Doc generation | codex-cli exec | Single subprocess, --full-auto, replaces 14 LLM calls |
| Workspace edition | Rust 2024, resolver 3 | Latest stable, let-chains for cleaner extractors |
| API defaults | Empty string (no URL) | Security: no hardcoded Chinese server defaults |
| CI platform | windows-latest only | Primary target, tree-sitter C compilation needs MSVC |
| Extract functions | Synchronous | No async I/O in extraction, avoids blocking tokio runtime |

## Security Fixes Applied

1. **modelscope.cn** default removed from litho-generator config.rs — uses `LITHO_API_BASE_URL` env var
2. **bigmodel.cn** hardcoded URL removed from litho-book server.rs — uses `LITHO_BOOK_LLM_API_URL` env var
3. **GLM-4.7-Flash** hardcoded model removed — uses `LITHO_BOOK_LLM_MODEL` env var

## Build Requirements

- **MSVC**: Required for tree-sitter C grammar compilation
- **vcvarsall.bat**: Must be called before cargo commands (sets INCLUDE, LIB, VCINSTALLDIR)
- **.cargo/config.toml**: Sets CC_x86_64_pc_windows_msvc=cl.exe to prevent Strawberry Perl gcc interference
- **Batch file approach**: Use `check.bat`/`test.bat` pattern with vcvarsall for reliable builds

## Commit History

```
58f4dc7 fix: remove hardcoded bigmodel.cn URL, dead code, and dependency issues
364701f chore: update Cargo.lock after thiserror upgrade
365e512 ci: add build + test + release workflow
c5f5d95 fix(litho-book): upgrade thiserror 1.0 → 2.0 to match workspace
97828a7 feat: import deepwiki-rs + litho-book as litho-generator + litho-book crates
559a54e feat(litho-cli): wire clap subcommands for extract and generate
4272392 feat(litho-codex): codex-cli exec provider with prompt templates
e0a6fed chore: remove temp build script
3c6bd51 fix: target-specific CC override for MSVC, remove unused JS dep
4364be0 feat(litho-extract): tree-sitter AST extraction for Rust/TS/Python/C#
2ca515c feat(litho-core): config with safe defaults + shared types
81dc94c feat: scaffold litho-workspace Cargo workspace with 4 crates
```

## Agent Work Registry

| Agent | Task | Files | Status |
|-------|------|-------|--------|
| rust-pro | litho-core config+types | crates/litho-core/ | Complete |
| rust-pro | litho-extract tree-sitter | crates/litho-extract/ | Complete |
| rust-pro | litho-codex provider | crates/litho-codex/ | Complete |
| rust-pro | litho-cli wire + E2E | crates/litho-cli/ | Complete |
| rust-pro | Import legacy code | crates/litho-generator/, litho-book/ | Complete |
| powershell-pro | PC_AI doc pipeline | Tools/Invoke-DocPipeline.ps1 | Complete (not committed in PC_AI) |
| deployment-engineer | CI/CD workflow | .github/workflows/ci.yml | Complete |
| superpowers:code-reviewer | Final review | All crates | Complete, fixes committed |
| rust-pro | Review fix implementation | 8 files | Complete (58f4dc7) |

## E2E Validation

Tested against PC_AI codebase:
- **664 files** extracted
- **96,976 LOC** counted
- **5 languages** detected (Rust, TypeScript, Python, C#, PowerShell)
- **738.1 KB** JSON output

## Known Issues / Future Work

1. **litho-generator not wired to litho-extract** — legacy regex pipeline preserved, could delegate to tree-sitter
2. **CI only tests Windows** — litho-book has cfg(unix) code untested in CI
3. **No JavaScript extractor** — tree-sitter-javascript dep was removed; TS extractor handles .tsx/.ts
4. **Chinese comments in litho-book** — inherited from upstream, should be localized
5. **`is_privileged()` on Windows** — returns true unconditionally in litho-book
6. **`node_text` helper duplicated** — same function in all 4 extractors, could extract to mod.rs
7. **Complexity heuristic** — counts keywords in strings/comments, not AST nodes

## PC_AI Integration (Uncommitted)

`Tools/Invoke-DocPipeline.ps1` has two new functions:
- `Invoke-LithoExtraction` — runs `litho extract` against PC_AI, outputs Reports/LITHO_EXTRACT.json
- `Invoke-LithoDocGeneration` — runs `litho generate`, outputs docs/auto/litho/
- Binary resolution: `Get-Command litho` fallback `$env:USERPROFILE\bin\litho.exe`
