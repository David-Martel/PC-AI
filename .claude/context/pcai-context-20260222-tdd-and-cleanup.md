# PC_AI Context: TDD Unit Tests, Async Refactor & Ignore Cleanup

**Context ID:** ctx-pcai-20260222b
**Created:** 2026-02-22
**Branch:** main @ 99ed9ac

## State Summary

Building on the FFI torture test completion (ctx-pcai-20260222), this session:
- Added 42 Rust unit tests across pcai_inference (TDD plan `starry-jumping-rocket` complete)
- Corrected CUDA build environment: CUDA 13.1 now works with updated Rust crate dependencies
- Updated .gitignore/.rgignore for alternate target dirs and training checkpoints
- Previous session committed C# async refactor and PS improvements (99ed9ac)

## CUDA Build Status

**Current:** CUDA 13.1 with updated Rust crate dependencies (working)
**Previous workaround:** CUDA 12.6 via `.cargo/config.toml` env overrides (no longer needed as primary)
**Env overrides still available** in `.cargo/config.toml` for version pinning if needed

## Recent Changes (Uncommitted)

| File | Change | Cluster |
|------|--------|---------|
| `.gitignore` | Add `**/target-*/`, training checkpoints patterns | chore |
| `.rgignore` | Add `target-*` intermediate exclusions with binary negations | chore |
| `src/lib.rs` | +39 lines: 5 Error enum unit tests | test |
| `src/config.rs` | +87 lines: 8 config serde/file I/O tests | test |
| `src/backends/mod.rs` | +80 lines: 7 request/response/FinishReason tests | test |
| `src/http/mod.rs` | +171 lines: 18 tests (chat prompt, tokens, stop, chunks, StopTracker) + f64→f32 cast fix | test |
| `src/ffi/mod.rs` | +36 lines: 4 FFI edge case tests | test |
| `Invoke-PcaiBuild.ps1` | Style: single quotes, whitespace normalization | style |
| `CSharp_RustDLL.md` | Markdown indent formatting | docs |

## Recent Changes (Committed in 99ed9ac)

| File | Change |
|------|--------|
| `InferenceBackend.cs` | IsAvailable → CheckAvailabilityAsync() |
| `Program.cs` | Async call sites updated |
| `ToolExecutor.cs` | +67 lines new tool execution logic |
| `Initialize-PcaiNative.ps1` | Extracted wrapper functions |
| `Get-UnifiedHardwareReportJson.ps1` | Uses new wrappers |
| `Set-LLMProviderOrder.ps1` | Fixed -Args → -ServerArgs |
| `Start-HVSockProxy.ps1` | Relative config path, List<> |
| `update-doc-status.ps1` | +45 lines improvements |

## Key Decisions

### dec-001: CUDA 13.1 Compatibility
- **Decision:** Updated Rust crate dependencies to support CUDA 13.1 directly
- **Previous:** Used CUDA 12.6 env overrides as workaround for nvcc 13.1 preprocessor bug
- **Status:** CUDA 13.1 now works; .cargo/config.toml overrides retained for flexibility

### dec-002: Ignore File Strategy
- **Decision:** `.gitignore` blocks `**/target-*/` (alternate build dirs), `.rgignore` excludes intermediates but keeps final binaries searchable
- **Rationale:** LLM agents using rg/grep need to find DLL/EXE paths for build verification and path discovery

### dec-003: Unit Test Scope
- **Decision:** Test pure logic only with `--no-default-features --features server,ffi`
- **Rationale:** Backend tests (llamacpp/mistralrs inference) require real models; pure logic tests are self-contained

## Test Coverage

### Rust Unit Tests (pcai_inference)
| Module | Tests | Coverage |
|--------|-------|----------|
| `lib.rs` | 5 | Error Display + From conversions |
| `config.rs` | 8 | Serde roundtrip, file I/O, defaults |
| `backends/mod.rs` | 7 | Request/Response serde, FinishReason |
| `http/mod.rs` | 21 (3 existing + 18 new) | Chat prompt, tokens, stop sequences, chunks, StopTracker |
| `ffi/mod.rs` | 17 (13 existing + 4 new) | FFI edge cases |
| **Total** | **58** | Up from 19 baseline |

### PowerShell FFI Tests
- Integration: 28 tests (23 pass, 5 skip)
- Stress: 21 tests (20 pass, 1 skip)

## Agent Registry

| Agent | Task | Status |
|-------|------|--------|
| context-restore | Analyzed repo state, 5 plans, uncommitted work | Complete |
| (this session) | Updated ignore files, memory, context | Complete |

## Recommended Next Steps

1. **Commit the clustered changes** (3 groups: chore, test, style/docs)
2. **Run `cargo test --no-default-features --features server,ffi --lib`** to verify all 58 tests pass
3. **Security audit** FFI boundary (pcai_free_string null safety, string lifetime ownership)
4. **Continue wondrous-wishing-gadget plan** — migrate high-value PS scripts to Rust DLL
5. **CI pipeline** — add unit test step to GitHub Actions workflow
