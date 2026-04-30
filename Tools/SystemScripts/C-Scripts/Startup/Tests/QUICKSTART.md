# Pester Test Framework - Quick Start Guide

## Installation (First Time Setup)

### 1. Install Pester 5.x

```powershell
# Install Pester module
Install-Module -Name Pester -Force -SkipPublisherCheck -Scope CurrentUser

# Verify installation
Get-Module -Name Pester -ListAvailable
```

### 2. Validate Test Environment

```powershell
cd C:\Scripts\Startup\Tests
.\Validate-TestEnvironment.ps1
```

This will check:
- PowerShell version
- Pester installation
- Test files present
- Helper module works
- Directory structure
- Permissions

## Running Tests

### Basic Test Execution

```powershell
# Run all tests
.\Run-Tests.ps1

# Run with detailed output
.\Run-Tests.ps1 -ShowTestDetails

# Run specific test file
.\Run-Tests.ps1 -TestPath .\Example.Tests.ps1
```

### Advanced Test Execution

```powershell
# Run with code coverage
.\Run-Tests.ps1 -CodeCoverage

# Generate XML report for CI/CD
.\Run-Tests.ps1 -OutputFormat NUnitXml

# Run with coverage and report
.\Run-Tests.ps1 -CodeCoverage -OutputFormat JUnitXml

# Set minimum coverage threshold
.\Run-Tests.ps1 -CodeCoverage -MinimumCoverage 90
```

### Filtering Tests

```powershell
# Run only unit tests (if tagged)
.\Run-Tests.ps1 -Tags 'Unit'

# Exclude integration tests
.\Run-Tests.ps1 -ExcludeTags 'Integration'

# Fail on skipped tests
.\Run-Tests.ps1 -FailOnSkipped
```

## Test Development Workflow

### 1. Test-Driven Development (TDD)

```powershell
# 1. Write a failing test
# 2. Run tests to confirm failure
.\Run-Tests.ps1 -ShowTestDetails

# 3. Implement the feature
# 4. Run tests to confirm success
.\Run-Tests.ps1

# 5. Refactor if needed
# 6. Run tests to ensure no regression
.\Run-Tests.ps1 -CodeCoverage
```

### 2. Creating New Tests

Use the example test as a template:

```powershell
# Copy example test
Copy-Item .\Example.Tests.ps1 .\MyNewFeature.Tests.ps1

# Edit the new test file
code .\MyNewFeature.Tests.ps1

# Run just your new test
.\Run-Tests.ps1 -TestPath .\MyNewFeature.Tests.ps1
```

### 3. Using Test Helpers

```powershell
# In your test file's BeforeAll block:
BeforeAll {
    Import-Module (Join-Path $PSScriptRoot 'TestHelpers.psm1') -Force

    # Now you can use helper functions
    $mockService = New-MockService -Name 'MyService' -Status 'Running'
    $mockProcess = New-MockProcess -ProcessName 'MyApp'
    $retryScenario = New-RetryScenario -FailCount 2
}
```

## Understanding Test Results

### Console Output

```
TEST RESULTS SUMMARY
================================================================================

Total Tests:    42
Passed:         40
Failed:         0
Skipped:        2
Not Run:        0

Execution Time: 00:12.345
```

### Test Result Files

After running with `-OutputFormat`:
- `TestResults/TestResults.NUnitXml.xml` - NUnit format
- `TestResults/TestResults.JUnitXml.xml` - JUnit format
- `TestResults/coverage.xml` - JaCoCo coverage (if -CodeCoverage)

### Code Coverage Report

```
CODE COVERAGE SUMMARY
================================================================================

Commands Analyzed:  150
Commands Executed:  135
Commands Missed:    15

Coverage:           90.0% ✓
```

## Common Tasks

### Run Tests Before Commit

```powershell
# Full validation
.\Run-Tests.ps1 -CodeCoverage -MinimumCoverage 85 -FailOnSkipped

# If tests pass, safe to commit
# If tests fail, fix issues first
```

### Debug Failing Tests

```powershell
# Run with detailed output
.\Run-Tests.ps1 -ShowTestDetails

# Run specific failing test
.\Run-Tests.ps1 -TestPath .\Start-WSLDockerServices.Tests.ps1

# Check for specific test within file
# Open test file and add -Skip to other tests temporarily
```

### Update Tests After Code Changes

```powershell
# 1. Update test expectations
code .\Start-WSLDockerServices.Tests.ps1

# 2. Run tests to verify
.\Run-Tests.ps1 -ShowTestDetails

# 3. Check coverage
.\Run-Tests.ps1 -CodeCoverage

# 4. Commit both code and test changes together
```

## Troubleshooting

### Tests Are Skipped

**Cause:** Script under test doesn't exist yet

**Solution:** This is normal for TDD. Tests will run once you create the script.

```powershell
# Create the script
New-Item C:\Scripts\Startup\Start-WSLDockerServices.ps1 -ItemType File

# Add function stubs
# Re-run tests
```

### Pester Version Errors

**Cause:** Wrong Pester version installed

**Solution:**
```powershell
# Uninstall old versions
Get-Module Pester -ListAvailable | Uninstall-Module -Force

# Install latest Pester 5.x
Install-Module -Name Pester -Force -SkipPublisherCheck
```

### Mock Not Working

**Cause:** Mock is in wrong scope or not properly defined

**Solution:**
```powershell
# Put mocks in BeforeAll/BeforeEach
BeforeAll {
    Mock Get-Service { New-MockService -Name 'Test' }
}

# Verify mock with Should -Invoke
Should -Invoke Get-Service -Times 1 -Exactly
```

### Coverage Shows 0%

**Cause:** Script path is incorrect or not being executed

**Solution:**
```powershell
# Verify script path
$scriptPath = 'C:\Scripts\Startup\Start-WSLDockerServices.ps1'
Test-Path $scriptPath

# Ensure script is dot-sourced in tests
. $scriptPath
```

## Best Practices Checklist

- [ ] All tests pass before committing code
- [ ] Code coverage is above minimum threshold (85%)
- [ ] New features have corresponding tests
- [ ] Tests are independent (can run in any order)
- [ ] External dependencies are mocked
- [ ] Test names clearly describe what is being tested
- [ ] No hardcoded paths or values
- [ ] Tests clean up after themselves
- [ ] Integration tests are tagged separately
- [ ] Documentation is updated with code changes

## Next Steps

1. **Read the full documentation:** `README.md`
2. **Study the examples:** `Example.Tests.ps1`
3. **Review helper functions:** `TestHelpers.psm1`
4. **Start writing tests:** Use TDD approach
5. **Integrate with CI/CD:** Add to your pipeline

## Additional Resources

- **Pester Documentation:** https://pester.dev
- **Test Helper Functions:** See `TestHelpers.psm1`
- **Example Tests:** See `Example.Tests.ps1`
- **Main Test Suite:** See `Start-WSLDockerServices.Tests.ps1`

## Support

For issues or questions:
1. Check `README.md` for detailed documentation
2. Review `Example.Tests.ps1` for patterns
3. Run `Validate-TestEnvironment.ps1` to check setup
4. Consult Pester documentation at pester.dev

---

**Remember:** Tests are code. Maintain them with the same care as production code.
