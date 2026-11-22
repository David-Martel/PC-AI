# PC_AI Testing Framework

Comprehensive Pester 5.x testing framework for PC_AI diagnostics and optimization system.

## Overview

This testing framework provides unit and integration tests with 85% coverage target for all PC_AI modules.

### Framework Components

```
Tests/
├── PesterConfiguration.psd1      # Pester 5.x configuration
├── .pester.ps1                   # Test runner script
├── Fixtures/
│   └── MockData.psm1             # Mock data factory functions
├── Unit/                         # Fast, isolated unit tests
│   ├── PC-AI.Hardware.Tests.ps1
│   ├── PC-AI.Virtualization.Tests.ps1
│   ├── PC-AI.USB.Tests.ps1
│   ├── PC-AI.Network.Tests.ps1
│   ├── PC-AI.Performance.Tests.ps1
│   ├── PC-AI.Cleanup.Tests.ps1
│   └── PC-AI.LLM.Tests.ps1
└── Integration/                  # End-to-end workflow tests
    ├── ReportGeneration.Tests.ps1
    └── ModuleLoading.Tests.ps1
```

## Quick Start

### Prerequisites

```powershell
# Install Pester 5.x or later
Install-Module -Name Pester -MinimumVersion 5.0.0 -Force -Scope CurrentUser

# Verify installation
Get-Module Pester -ListAvailable
```

### Running Tests

```powershell
# Navigate to test directory
cd C:\codedev\PC_AI\Tests

# Run all tests
.\\.pester.ps1

# Run only unit tests (fast)
.\\.pester.ps1 -Type Unit

# Run integration tests
.\\.pester.ps1 -Type Integration

# Run with code coverage
.\\.pester.ps1 -Coverage

# Run in CI mode (exit codes, XML output)
.\\.pester.ps1 -CI

# Run specific tag
.\\.pester.ps1 -Tag Hardware,Fast

# Run the tooling benchmark suite
.\Invoke-AllTests.ps1 -Suite Benchmarks
```

### Running Tooling Benchmarks

```powershell
# Quick tooling benchmark sweep
.\Benchmarks\Invoke-PcaiToolingBenchmarks.ps1 -Suite quick

# Full default tooling benchmark sweep
.\Benchmarks\Invoke-PcaiToolingBenchmarks.ps1

# Targeted heavy cases only
.\Benchmarks\Invoke-PcaiToolingBenchmarks.ps1 -CaseId content-search,full-context
```

### Validated Findings

Results validated on March 7, 2026:

- Native `directory-manifest` is currently the strongest native search/discovery
  win: about `28.55 ms` native vs about `1161.75 ms` PowerShell on the
  `PC_AI` repo root.
- Native `token-estimate` is about `2.9x` faster than the PowerShell baseline.
- Native `full-context` is faster than the PowerShell baseline, but not yet by
  an order of magnitude.
- Accelerated `fd`/`rg` search remains the preferred fast path for generic
  file/content search on this machine.
  - `file-search`: about `8.7 ms` accelerated vs about `104.07 ms` native vs
    about `3435.82 ms` PowerShell
  - `content-search`: about `11.58 ms` accelerated vs about `3283.99 ms`
    native vs about `4734.92 ms` PowerShell

Use the benchmark reports to decide backend preference by workload shape rather
than assuming the native path is always fastest.

## Test Organization

### Unit Tests

Fast, isolated tests with mocked dependencies:

- **PC-AI.Hardware.Tests.ps1** - Device errors, disk health, USB status, network adapters
- **PC-AI.Virtualization.Tests.ps1** - WSL, Hyper-V, Docker diagnostics
- **PC-AI.USB.Tests.ps1** - USB device listing, WSL mounting, usbipd integration
- **PC-AI.Network.Tests.ps1** - Network diagnostics, WSL connectivity, VSock performance
- **PC-AI.Performance.Tests.ps1** - Disk space, process monitoring, resource tracking
- **PC-AI.Cleanup.Tests.ps1** - PATH cleanup, duplicate detection, temp file removal
- **PC-AI.LLM.Tests.ps1** - Ollama integration, LLM chat, PC diagnosis

### Integration Tests

End-to-end workflows with multiple modules:

- **ReportGeneration.Tests.ps1** - Complete diagnostic report generation and LLM analysis
- **ModuleLoading.Tests.ps1** - Module loading, exports, dependencies, interoperability

### Test Tags

| Tag | Purpose |
|-----|---------|
| `Unit` | Fast, isolated unit tests |
| `Integration` | Multi-module integration tests |
| `E2E` | End-to-end workflow tests |
| `Fast` | Tests that run in <1 second |
| `Slow` | Tests that may take >1 second |
| `RequiresAdmin` | Tests requiring Administrator privileges |
| `Hardware`, `Virtualization`, etc. | Module-specific tests |

### Running Tagged Tests

```powershell
# Fast unit tests only
.\\.pester.ps1 -Tag Fast

# All hardware-related tests
.\\.pester.ps1 -Tag Hardware

# Exclude slow tests
.\\.pester.ps1 -ExcludeTag Slow,RequiresAdmin
```

## Mock Data

The `Fixtures/MockData.psm1` module provides factory functions for consistent test data:

### Device Mocks
```powershell
New-MockPnPEntity -Name "USB Device" -ConfigManagerErrorCode 43
Get-MockDevicesWithErrors      # Pre-configured error scenarios
Get-MockDevicesHealthy         # Healthy system
```

### Disk Health Mocks
```powershell
Get-MockDiskSmartOutput -Health Healthy   # All OK
Get-MockDiskSmartOutput -Health Warning   # Pred Fail
Get-MockDiskSmartOutput -Health Failed    # Bad sectors
```

### Event Log Mocks
```powershell
New-MockWinEvent -Id 7 -Level Error -Message "Bad block"
Get-MockDiskUsbEvents -ErrorType Mixed    # Disk + USB errors
Get-MockDiskUsbEvents -ErrorType None     # Clean logs
```

### Network Mocks
```powershell
New-MockNetworkAdapter -Name "Ethernet" -Status "Connected"
Get-MockNetworkAdapters -IncludeVirtual   # Physical + virtual
```

### WSL Mocks
```powershell
Get-MockWSLOutput -Command Status   # wsl --status
Get-MockWSLOutput -Command List     # wsl -l -v
Get-MockWSLOutput -Command IpAddr   # wsl ip addr show
```

### LLM Mocks
```powershell
Get-MockOllamaResponse -Type Success    # Successful API response
Get-MockOllamaResponse -Type Error      # Model not found error
Get-MockOllamaResponse -Type ModelList  # List of models
```

### USB Mocks
```powershell
Get-MockUsbIpdOutput               # usbipd list output
```

### Disk Space Mocks
```powershell
Get-MockDiskSpace -Status Healthy   # 70%+ free space
Get-MockDiskSpace -Status LowSpace  # <10% free
Get-MockDiskSpace -Status Critical  # <5% free
```

## Test assessment and gates

`Tests/coverage-report.ps1` measures the suite and exits 1 when a gate fails. Run it with `pwsh Tests/coverage-report.ps1 -Tier Portable`
(about 7 minutes: it runs `Tests/Unit` with the `Portable` tag and measures coverage over every `Modules/**/*.ps1|psm1` file) or with
`-NoRun` for the static part only (seconds). It writes `.pcai/assessment/test-assessment.json` (gitignored) and nothing else in the
repository. `-Tier Full` runs the path list `ci.yml` uses and needs its own baseline entry.

The JSON holds the test inventory (Pester `It` blocks per directory, Rust and C# test attributes), which runners and workflows reach each
`*.Tests.ps1` file (files nothing reaches are orphans), measured Pester coverage overall and per module, and quality counts found through
the PowerShell AST: `It` blocks with no assertion, with only `Should -Not -Throw`, or with a constant input (`$true | Should -BeTrue`).

The thresholds live in `Config/test-baseline.json` and only ratchet. They start at what was measured, not at the 85% target
(`target_coverage_pct`), and `-UpdateBaseline` can raise the coverage minimums and lower the violation counts, the orphan allowlist and
the known-failure budget, never the reverse. Known failures are recorded per test file, so a new failing file fails the gate even while
the recorded ones are tracked elsewhere.

All runners (`Invoke-PortableTests.ps1`, `Invoke-AllTests.ps1`, `.pester.ps1`, `test.ps1`, `Invoke-QualityGate.ps1`) judge a Pester run with
`Tests/Helpers/Get-PesterRunVerdict.ps1`: a test file that does not parse, a failed `BeforeAll`, or a run that executed nothing fails,
whatever `FailedCount` says. `Tests/Unit/TestAssessment.Tests.ps1` plants each of those failures and asserts it is reported.

## Code Coverage

### Coverage Requirements

- **Target**: 85% code coverage minimum (the enforced floor is the ratcheted `Config/test-baseline.json` value; see above)
- **Output Format**: JaCoCo XML for CI integration
- **Excluded**: Test files themselves

### Running Coverage Analysis

```powershell
# Generate coverage report
.\\.pester.ps1 -Coverage

# Output: coverage.xml (JaCoCo format)
```

### Coverage Report Interpretation

The coverage report shows:
- **Commands Analyzed**: Total testable commands in codebase
- **Commands Executed**: Commands covered by tests
- **Coverage %**: (Executed / Analyzed) × 100
- **Missed Commands**: Specific lines not covered

### Improving Coverage

1. Run coverage analysis:
   ```powershell
   .\\.pester.ps1 -Coverage
   ```

2. Review missed commands in output

3. Add tests for uncovered code paths:
   - Error handling branches
   - Edge cases
   - Parameter validation

## CI/CD Integration

### GitHub Actions

```yaml
name: Test

on: [push, pull_request]

jobs:
  test:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v3

      - name: Install Pester
        run: Install-Module -Name Pester -MinimumVersion 5.0.0 -Force

      - name: Run Tests
        run: |
          cd Tests
          .\\.pester.ps1 -CI -Coverage

      - name: Upload Test Results
        uses: actions/upload-artifact@v3
        with:
          name: test-results
          path: Tests/test-results.xml

      - name: Upload Coverage
        uses: actions/upload-artifact@v3
        with:
          name: coverage
          path: Tests/coverage.xml
```

### Azure DevOps

```yaml
steps:
- task: PowerShell@2
  displayName: 'Install Pester'
  inputs:
    targetType: 'inline'
    script: 'Install-Module -Name Pester -MinimumVersion 5.0.0 -Force'

- task: PowerShell@2
  displayName: 'Run Tests'
  inputs:
    targetType: 'filePath'
    filePath: 'Tests/.pester.ps1'
    arguments: '-CI -Coverage'

- task: PublishTestResults@2
  displayName: 'Publish Test Results'
  inputs:
    testResultsFormat: 'NUnit'
    testResultsFiles: 'Tests/test-results.xml'

- task: PublishCodeCoverageResults@1
  displayName: 'Publish Coverage'
  inputs:
    codeCoverageTool: 'JaCoCo'
    summaryFileLocation: 'Tests/coverage.xml'
```

## Test Development Guidelines

### Writing Unit Tests

```powershell
Describe "Function-Name" -Tag 'Unit', 'ModuleName', 'Fast' {
    Context "When condition is true" {
        BeforeAll {
            # Setup mocks
            Mock Get-SomeData { Get-MockData } -ModuleName ModuleName
        }

        It "Should perform expected action" {
            $result = Function-Name -Parameter "value"
            $result | Should -Not -BeNullOrEmpty
        }

        It "Should call dependent function" {
            Function-Name -Parameter "value"

            Should -Invoke Get-SomeData -ModuleName ModuleName -Times 1
        }
    }

    Context "When error occurs" {
        BeforeAll {
            Mock Get-SomeData { throw "Error" } -ModuleName ModuleName
        }

        It "Should handle errors gracefully" {
            { Function-Name -ErrorAction Stop } | Should -Throw
        }
    }
}
```

### Best Practices

1. **Use Descriptive Names**: Test names should clearly state what is being tested
2. **Arrange-Act-Assert**: Structure tests with clear setup, execution, and verification
3. **Mock External Dependencies**: Isolate unit tests from system state
4. **Test Edge Cases**: Include error conditions, empty inputs, invalid parameters
5. **Tag Appropriately**: Use tags for easy filtering (Fast, Slow, RequiresAdmin)
6. **BeforeAll for Setup**: Use BeforeAll for test setup, not BeforeEach unless necessary
7. **AfterAll for Cleanup**: Clean up modules and test artifacts

### Mock Guidelines

- Mock at module boundary using `-ModuleName` parameter
- Use factory functions from `MockData.psm1` for consistency
- Return realistic data structures matching actual commands
- Include both success and failure scenarios

## Troubleshooting

### Pester Version Issues

```powershell
# Remove old Pester versions
Get-Module Pester -ListAvailable | Where-Object Version -lt 5.0.0 | Uninstall-Module

# Install Pester 5.x
Install-Module -Name Pester -MinimumVersion 5.0.0 -Force -SkipPublisherCheck
```

### Module Loading Failures

```powershell
# Check module manifests
Get-ChildItem ..\Modules\*\*.psd1 | ForEach-Object {
    Test-ModuleManifest $_.FullName
}

# Manually import module for debugging
Import-Module ..\Modules\PC-AI.Hardware\PC-AI.Hardware.psd1 -Force -Verbose
```

### Mock Not Working

```powershell
# Verify mock is in correct scope
Mock Get-Something { "mocked" } -ModuleName PC-AI.ModuleName

# Check mock was called
Should -Invoke Get-Something -ModuleName PC-AI.ModuleName -Times 1
```

### Test Isolation Issues

```powershell
# Ensure BeforeAll/AfterAll are used correctly
BeforeAll {
    # Setup runs once per Describe/Context
}

AfterAll {
    # Cleanup runs once after all tests
    Remove-Module PC-AI.* -Force -ErrorAction SilentlyContinue
}
```

### Coverage Not Generating

```powershell
# Ensure paths are correct in PesterConfiguration.psd1
CodeCoverage = @{
    Enabled = $true
    Path = @('..\Modules\PC-AI.*\**\*.ps1')
    ExcludeTests = $true
}
```

## Performance Benchmarks

The dedicated tooling benchmark runner writes reports under
`Reports\tooling-benchmarks\<timestamp>\` and records:

- backend coverage (`Rust`, `C# bridge`, `PowerShell surface`)
- schema coverage snapshots from `Tools\update-tool-coverage.ps1`
- timing comparisons across native, accelerated, and PowerShell baselines
- per-iteration memory deltas for working set, private bytes, managed heap
  delta, and managed allocation volume

Current validated entrypoints:

- `token-estimate`
- `directory-manifest`
- `file-search`
- `content-search`
- `full-context`
- `runtime-config`
- `command-map`

Expected test execution times:

| Test Suite | Duration |
|------------|----------|
| Unit Tests (All) | ~10-15s |
| Unit Tests (Fast only) | ~3-5s |
| Integration Tests | ~15-20s |
| Full Suite | ~25-35s |
| Full Suite + Coverage | ~40-60s |

## Contributing

When adding new functionality:

1. Write tests first (TDD approach)
2. Ensure 85%+ coverage for new code
3. Add appropriate tags (Unit, Fast/Slow, module name)
4. Update MockData.psm1 if new mock data needed
5. Run full suite before committing: `.\\.pester.ps1 -CI -Coverage`

## Resources

- [Pester Documentation](https://pester.dev)
- [Pester 5.x Migration Guide](https://pester.dev/docs/migrations/v3-to-v4)
- [PowerShell Testing Best Practices](https://pester.dev/docs/usage/mocking)
- PC_AI Documentation: `../CLAUDE.md`, `../DIAGNOSE.md`

## License

This testing framework is part of the PC_AI project and follows the same license terms.

## Native resolver admission

The [retained partial/full-symbol resolver controls](Portable/native-resolver-admission-review.md)
run in the existing native managed test project. They detect native calls or cached
handles from a core that failed required-export admission. Synthetic symbol
fixtures do not qualify the production Rust core or a deployed native bundle.

## Local CI mirror

GitHub-hosted runners are disabled for this account, so `ci.yml` never runs and its failures stay
invisible. `Tests/Invoke-PcaiCi.ps1` runs the same commands locally:

```powershell
pwsh Tests/Invoke-PcaiCi.ps1 -Plan              # list the Quick steps
pwsh Tests/Invoke-PcaiCi.ps1                    # Quick: credential scan, workflow syntax, changed-file lint, rustfmt (about 40 s)
pwsh Tests/Invoke-PcaiCi.ps1 -Tier Full -FailFast   # adds full lint (about 12 min), check, clippy, unit tests, Pester Unit and Boot
pwsh Tests/Invoke-PcaiCi.ps1 -Only RustCheck    # any step by id, whatever the tier
```

Logs and `pcai-ci-last.json` go to `.pcai/build/logs/pcai-ci/`. `Tests/Unit/Invoke-PcaiCi.Tests.ps1`
fails when a step's command or the Pester path list no longer appears in `ci.yml`, so the mirror
cannot drift silently. Quick uses `Tools/Test-ChangedPowerShellLint.ps1`, because PSScriptAnalyzer
keeps static state and cannot be parallelized in one process (splitting across eight processes was
only 1.4x faster: 701 s to 492 s). Its first full run also found a real lint failure on `main`
(`Tests/Portable/Test-Usb4RouteOwnership.ps1`, PSShouldProcess) that no CI run could report.
