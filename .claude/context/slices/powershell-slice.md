# PowerShell Context Slice - PC_AI

**Updated**: 2026-01-28
**For Agents**: powershell-pro, test-automator

## CargoTools Module

**Path**: `C:\Users\david\OneDrive\Documents\PowerShell\Modules\CargoTools\`
**Version**: 0.4.0
**Tests**: 16/17 passing (~20% coverage)

### Public Functions
| Function | Purpose |
|----------|---------|
| `Invoke-CargoRoute` | Route cargo commands |
| `Invoke-CargoWrapper` | Cargo build wrapper |
| `Invoke-CargoWsl` | WSL cargo execution |
| `Invoke-CargoDocker` | Docker cargo execution |
| `Invoke-CargoMacos` | macOS cross-compilation |
| `Invoke-RustAnalyzerWrapper` | Singleton rust-analyzer |
| `Test-RustAnalyzerHealth` | Health check with LLM output |

### LLM Helper Functions
| Function | Purpose |
|----------|---------|
| `Format-CargoOutput` | JSON envelope formatting |
| `Format-CargoError` | Error context for debugging |
| `ConvertTo-LlmContext` | Token-optimized context |
| `Get-RustProjectContext` | Project structure extraction |
| `Get-CargoContextSnapshot` | Environment capture |

### Recent Fixes Applied
1. **Mutex acquisition** - Now uses `WaitOne()` properly
2. **TOCTOU race** - Atomic `FileMode::CreateNew`
3. **Hardcoded paths** - Dynamic `Resolve-CacheRoot`

### Test Coverage Gaps
- [ ] `Invoke-CargoRoute` - No tests
- [ ] `Invoke-CargoWrapper` - No tests
- [ ] `Invoke-CargoWsl` - No tests
- [ ] `Invoke-CargoDocker` - No tests
- [ ] `Format-CargoOutput` - No tests
- [ ] `Format-CargoError` - No tests

### Architecture Issues
- `Environment.ps1` has 7 concerns (violates SRP):
  1. MSVC environment setup
  2. Cache root resolution
  3. Sccache management
  4. Linker configuration
  5. Rust-analyzer path resolution
  6. Rust-analyzer memory helpers
  7. Build job optimization

**Recommendation**: Split into `MsvcEnvironment.ps1`, `CacheManagement.ps1`, `SccacheHelpers.ps1`, `RustAnalyzerHelpers.ps1`

## PC-AI Modules

| Module | Path | Status |
|--------|------|--------|
| PC-AI.CLI | `Modules/PC-AI.CLI/` | Working |
| PC-AI.LLM | `Modules/PC-AI.LLM/` | Working |
| PC-AI.Common | `Modules/PC-AI.Common/` | New, needs tests |

## Scripts
| Script | Purpose |
|--------|---------|
| `Scripts/Test-RustAnalyzerHealth.ps1` | Standalone health check |
| `Scripts/Cleanup-RustAnalyzer.ps1` | Process cleanup |
| `Scripts/Test-CargoToolsModule.ps1` | Module validation |
| `Scripts/Run-CargoToolsTests.ps1` | Pester test runner |
