# Testing Context Slice - PC_AI

**Updated**: 2026-01-28
**For Agents**: test-runner, test-automator

## Test Coverage Summary

| Component | Current | Target | Framework |
|-----------|---------|--------|-----------|
| CargoTools (PowerShell) | ~20% (16/17) | 85% | Pester 5.7 |
| rust-functiongemma-train | Minimal | Full | Rust #[test] |
| rust-functiongemma-runtime | 1 test | Full | Rust #[test] |
| pcai_core_lib (Rust) | Good | Maintain | Rust #[test] |
| PC-AI.Common | None | Basic | Pester |

## CargoTools Pester Tests

**Location**: `C:\Users\david\OneDrive\Documents\PowerShell\Modules\CargoTools\Tests\`

### Existing Tests (16 passing)
```
Describe Resolve-RustAnalyzerPath
  Context When RUST_ANALYZER_PATH is set
    [+] Should return the environment variable path if valid
    [+] Should skip invalid RUST_ANALYZER_PATH and continue resolution
  Context When using rustup toolchain
    [+] Should find rust-analyzer in stable toolchain
  Context Validation
    [+] Should reject empty files (0-byte shims)

Describe Test-RustAnalyzerSingleton
  Context When no rust-analyzer is running
    [+] Should report NotRunning status
  Context Lock file validation
    [+] Should detect stale lock files
  Context Memory threshold
    [+] Should accept custom warning threshold

Describe Get-RustAnalyzerMemoryMB
  [+] Should return 0 when no rust-analyzer is running
  [+] Should return positive value when rust-analyzer is running

Describe Invoke-RustAnalyzerWrapper
  Context Help
    [+] Should display help with --help flag
  Context Environment variables
    [+] Should set RA_LRU_CAPACITY via Initialize-CargoEnv
    [+] Should set CHALK_SOLVER_MAX_SIZE via Initialize-CargoEnv
    [+] Should set RA_PROC_MACRO_WORKERS via Initialize-CargoEnv

Describe System-wide shim
  Context C:\Users\david\bin\rust-analyzer.cmd
    [+] Should exist
    [+] Should resolve before any rust-analyzer.exe in PATH

Describe Integration Tests
  [!] Should enforce singleton via lock file (SKIPPED)
  [+] Should have lock file directory accessible
```

### Missing Test Coverage
- `Invoke-CargoRoute` - Route selection logic
- `Invoke-CargoWrapper` - Build orchestration
- `Invoke-CargoWsl` - WSL execution
- `Invoke-CargoDocker` - Docker execution
- `Format-CargoOutput` - JSON envelope generation
- `Format-CargoError` - Error parsing
- `ConvertTo-LlmContext` - Context extraction
- `Get-RustProjectContext` - Project analysis
- `Initialize-CargoEnv` - Full environment setup
- `Start-SccacheServer` - Sccache lifecycle
- `Stop-SccacheServer` - Cleanup

## Rust Tests

### rust-functiongemma-train
**Run**: `cargo test -p rust-functiongemma-train`
```rust
// tests/integration_test.rs
#[test]
fn test_schema_loading() { ... }
```

**Missing**:
- Trainer unit tests
- Dataset generation tests
- Eval metric tests

### rust-functiongemma-runtime
**Run**: `cargo test -p rust-functiongemma-runtime`
```rust
// tests/http_router.rs
#[tokio::test]
async fn test_health_endpoint() { ... }
```

**Missing**:
- Chat completions endpoint tests
- Tool call parsing tests
- Model loading tests

## Test Commands

```powershell
# CargoTools Pester
cd "C:\Users\david\OneDrive\Documents\PowerShell\Modules\CargoTools"
Invoke-Pester -Path ./Tests -Output Detailed

# Rust FunctionGemma
.\Tools\Invoke-RustBuild.ps1 -Path Deploy\rust-functiongemma-train test
.\Tools\Invoke-RustBuild.ps1 -Path Deploy\rust-functiongemma-runtime test

# pcai_core_lib
cargo test -p pcai_core_lib --all-features
```
