# Cross-Platform Test Harness Design (Sub-project A)

> **Date:** 2026-03-27
> **Status:** Approved
> **Author:** Claude Opus 4.6 + David Martel
> **Repo:** David-Martel/PC-AI
> **Part of:** 4-part testing/benchmarking/LLM optimization initiative (A → B → C → D)

## Problem Statement

Remote AI agents (Jules, Codex, Copilot) run in Linux VMs without GPU access. Currently, most of the PC_AI test suite requires Windows APIs (CIM/WMI), native FFI DLLs, or CUDA — making 70%+ of tests unrunnable by remote agents. This limits their contributions to surface-level code review rather than verified quality improvements.

## Design

### Test Tiers

Every test is tagged with its platform requirements:

| Tier | Tag | Runs On | Content |
|------|-----|---------|---------|
| `Portable` | No Windows/GPU deps | Linux + Windows | Rust unit/clippy, C# build/test, PSScriptAnalyzer, Pester unit tests without CIM |
| `Windows` | Requires Windows APIs | Windows CI only | CIM/WMI tests, FFI integration, native DLL loading |
| `Gpu` | Requires CUDA | Windows + GPU only | Media pipeline, CUDA kernels, inference E2E |

### Component 1: `Tests/Invoke-PortableTests.ps1`

Single entry point for all portable tests. Runs on Linux or Windows.

**Sections:**
1. **Rust compilation + tests**: `cargo test --workspace --no-default-features --features server,ffi` (excludes CUDA features)
2. **Rust lint**: `cargo clippy --workspace --all-targets --no-deps -- -D warnings -A clippy::type_complexity`
3. **Rust format**: `cargo fmt --all --check`
4. **C# build**: `dotnet build Native/PcaiNative/PcaiNative.csproj` (verifies P/Invoke signatures compile)
5. **C# tests**: `dotnet test Native/PcaiNative.Tests/PcaiNative.Tests.csproj` (signature verification)
6. **Pester portable**: `Invoke-Pester -Tag Portable` on `Tests/Unit/`
7. **PSScriptAnalyzer**: Lint all `Modules/**/*.ps1` with `PSScriptAnalyzerSettings.psd1`

**Parameters:**
- `-Section <string[]>` — Run specific sections (e.g., `-Section Rust,CSharp`)
- `-Format <Table|Json|JUnit>` — Output format (JUnit for CI integration)
- `-FailFast` — Stop on first failure

**Output:** `Tests/Results/portable-<timestamp>.json` with:
```json
{
  "timestamp": "ISO-8601",
  "platform": "linux|windows",
  "sections": {
    "rust_test": { "passed": 65, "failed": 0, "skipped": 2, "duration_ms": 42000 },
    "rust_clippy": { "warnings": 0, "errors": 0 },
    "rust_fmt": { "passed": true },
    "csharp_build": { "passed": true, "warnings": 3 },
    "csharp_test": { "passed": 11, "failed": 0 },
    "pester": { "passed": 45, "failed": 0, "skipped": 12 },
    "psscriptanalyzer": { "violations": 0 }
  },
  "overall": "PASS"
}
```

### Component 2: `Tests/Invoke-QualityGate.ps1`

A/B code quality comparison between two git refs. This is how remote agents prove their changes improve quality.

**Parameters:**
- `-BaseRef <string>` — Base reference (default: `origin/main`)
- `-HeadRef <string>` — Head reference (default: `HEAD`)
- `-Format <Table|Json|Markdown>` — Output format
- `-FailOnRegression` — Exit non-zero if any metric regresses

**How it works:**
1. Stash current changes (if any)
2. Checkout `$BaseRef`, run quality metrics, save as `baseline`
3. Checkout `$HeadRef`, run quality metrics, save as `current`
4. Compute deltas for each metric
5. Output structured report

**Metrics compared:**
- Rust clippy warning count (per crate)
- Rust test pass count (per crate)
- C# build warning count
- C# test pass count
- Pester test pass count (portable tag)
- PSScriptAnalyzer violation count (per severity)

**Output:** `Tests/Results/quality-gate-<timestamp>.json` with:
```json
{
  "base_ref": "origin/main",
  "head_ref": "HEAD",
  "deltas": {
    "rust_clippy_warnings": { "base": 5, "head": 3, "delta": -2, "status": "improved" },
    "rust_tests_passing": { "base": 65, "head": 67, "delta": +2, "status": "improved" },
    "pester_passing": { "base": 45, "head": 47, "delta": +2, "status": "improved" },
    "psscriptanalyzer_violations": { "base": 0, "head": 0, "delta": 0, "status": "unchanged" }
  },
  "overall": "IMPROVED",
  "regressions": []
}
```

### Component 3: Test Tag Convention

Add `-Tag` to all Pester `Describe` blocks:

**Portable tests** (run on Linux): Tests that only use PowerShell logic, mocked dependencies, or cross-platform cmdlets.
```powershell
Describe 'Get-JulesApiUrl' -Tag 'Unit', 'Portable' {
```

**Windows tests** (require CIM/WMI/FFI): Tests that call Get-CimInstance, load native DLLs, or use Windows-only APIs.
```powershell
Describe 'Get-DeviceErrors' -Tag 'Unit', 'Windows' {
```

**GPU tests** (require CUDA): Tests that exercise GPU code paths.
```powershell
Describe 'Initialize-PcaiMediaFFI' -Tag 'Integration', 'Gpu' {
```

**Tagging rules:**
- Every `Describe` block MUST have at least one tier tag (`Portable`, `Windows`, or `Gpu`)
- Tests default to `Portable` unless they use Windows-specific APIs
- Tests that mock CIM/WMI results (not calling real CIM) are `Portable`
- Tests that call real CIM/WMI or load DLLs are `Windows`

### Component 4: `.github/workflows/portable-ci.yml`

New GitHub Actions workflow running on `ubuntu-latest`:

```yaml
name: Portable CI (Linux)
on:
  pull_request:
    branches: [main, develop]
  workflow_dispatch: {}

jobs:
  portable-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with: { components: 'rustfmt, clippy' }
      - uses: actions/setup-dotnet@v4
        with: { dotnet-version: '8.0.x' }
      - name: Install PowerShell modules
        shell: pwsh
        run: Install-Module -Name Pester -Force -Scope CurrentUser; Install-Module -Name PSScriptAnalyzer -Force -Scope CurrentUser
      - name: Run portable tests
        shell: pwsh
        run: ./Tests/Invoke-PortableTests.ps1 -Format JUnit -FailFast
      - name: Quality gate (vs base branch)
        shell: pwsh
        run: ./Tests/Invoke-QualityGate.ps1 -BaseRef origin/${{ github.base_ref }} -Format Markdown -FailOnRegression
      - name: Upload results
        uses: actions/upload-artifact@v4
        with:
          name: portable-test-results
          path: Tests/Results/
```

### Component 5: `Native/PcaiNative.Tests/`

xUnit test project for C# P/Invoke signature verification:

**Tests cover:**
- Every `[DllImport]` declaration compiles with correct marshaling attributes
- Struct layouts match Rust FFI equivalents (size, alignment)
- Enum values match Rust constants
- SafeHandle types are used where required (not IntPtr for owned pointers)
- Static-pointer functions (pcai_core_version, etc.) use IntPtr, NOT SafeHandle

**Does NOT test:**
- Actual DLL loading (requires Windows + native binary)
- Runtime FFI behavior (that's the Integration tier)

### Files Created/Modified

| File | Type | Purpose |
|------|------|---------|
| `Tests/Invoke-PortableTests.ps1` | New | Cross-platform test runner |
| `Tests/Invoke-QualityGate.ps1` | New | A/B quality comparison |
| `.github/workflows/portable-ci.yml` | New | Linux CI workflow |
| `Native/PcaiNative.Tests/PcaiNative.Tests.csproj` | New | C# test project |
| `Native/PcaiNative.Tests/PInvokeSignatureTests.cs` | New | P/Invoke verification |
| `Tests/Unit/*.Tests.ps1` (existing) | Modified | Add tier tags to Describe blocks |

### Out of Scope

- Mocking Windows CIM/WMI on Linux (fragile, low ROI)
- GPU simulation or CUDA mocking
- Full E2E test portability
- Docker-based test environments

### Success Criteria

1. `Invoke-PortableTests.ps1` runs cleanly on both Linux and Windows
2. `Invoke-QualityGate.ps1` correctly detects quality regressions/improvements
3. `portable-ci.yml` passes on GitHub Actions ubuntu-latest
4. Jules can run portable tests in its Ubuntu VM
5. Every existing Pester test has a tier tag
