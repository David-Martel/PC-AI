# WSL/Docker Services Test Suite

Comprehensive Pester 5.x test framework for `Start-WSLDockerServices.ps1`.

## Quick Start

```powershell
# Run all tests
.\Run-Tests.ps1

# Run with detailed output
.\Run-Tests.ps1 -ShowTestDetails

# Run with code coverage
.\Run-Tests.ps1 -CodeCoverage

# Generate XML report
.\Run-Tests.ps1 -OutputFormat NUnitXml
```

## Test Structure

### Test Files
- **Start-WSLDockerServices.Tests.ps1** - Main test suite
- **Run-Tests.ps1** - Test runner with reporting
- **TestHelpers.psm1** - Reusable test utilities

### Test Categories

#### Unit Tests
Tests for individual phase functions:
- `Start-LxssManager` - LxssManager service management
- `Initialize-WSL` - WSL initialization and verification
- `Initialize-DNS` - DNS configuration
- `Start-HyperVBridges` - Hyper-V networking services
- `Start-DockerDesktop` - Docker Desktop startup
- `Wait-DockerEngine` - Docker engine readiness
- `Start-RAGRedis` - Redis container management

#### Integration Tests
End-to-end workflow tests:
- Full startup sequence
- Phase ordering
- Error propagation
- Cleanup operations

#### Performance Tests
- Timeout handling
- Retry logic timing
- Resource usage

## Test Scenarios

### Happy Path
All services start successfully on first attempt.

### Retry Scenarios
Services fail initially but succeed within retry limit:
- Service temporarily unavailable
- Network not ready
- Process slow to start

### Error Scenarios
Failures that should abort execution:
- Service not found
- Max retries exceeded
- Missing executables
- Permission errors

### Already Running
Services already in desired state:
- Skip redundant operations
- Verify state without changes

## Running Tests

### Basic Execution

```powershell
# All tests
.\Run-Tests.ps1

# Specific test file
.\Run-Tests.ps1 -TestPath .\Start-WSLDockerServices.Tests.ps1

# With verbose output
.\Run-Tests.ps1 -ShowTestDetails
```

### Advanced Options

```powershell
# Code coverage with 90% minimum
.\Run-Tests.ps1 -CodeCoverage -MinimumCoverage 90

# Export to multiple formats
.\Run-Tests.ps1 -OutputFormat JUnitXml

# Filter by tags
.\Run-Tests.ps1 -Tags 'Unit' -ExcludeTags 'Slow'

# Fail on skipped tests
.\Run-Tests.ps1 -FailOnSkipped
```

## Writing New Tests

### Test Template

```powershell
Describe 'My-Function' {

    BeforeAll {
        # Setup mocks
        Mock External-Command { return 'mocked' }
    }

    Context 'Happy Path' {

        It 'Should do something' {
            # Arrange
            $input = 'test'

            # Act
            $result = My-Function -Input $input

            # Assert
            $result | Should -Be 'expected'
            Should -Invoke External-Command -Times 1
        }
    }

    Context 'Error Scenarios' {

        It 'Should throw on invalid input' {
            { My-Function -Input $null } | Should -Throw
        }
    }
}
```

### Best Practices

1. **Use BeforeAll/AfterAll** for setup/cleanup
2. **Mock external dependencies** - Never call real services in tests
3. **Test one thing per test** - Keep It tests focused
4. **Use descriptive names** - "Should start service when stopped"
5. **Follow AAA pattern** - Arrange, Act, Assert
6. **Verify mock invocations** - Ensure functions call dependencies correctly

### Helper Functions

The test suite includes helper functions:

```powershell
# Create mock service objects
$service = New-MockService -Name 'LxssManager' -Status 'Running'

# Create mock process objects
$process = New-MockProcess -ProcessName 'Docker Desktop' -Id 1234

# Create retry scenario
$retryLogic = New-RetryScenario -FailCount 2 -SuccessValue $true

# Verify retry count
Assert-RetryCount -ExpectedCount 3 -Message "Should retry 3 times"
```

## Continuous Integration

### CI Pipeline Integration

```yaml
# GitHub Actions example
- name: Run Pester Tests
  shell: pwsh
  run: |
    cd C:\Scripts\Startup\Tests
    .\Run-Tests.ps1 -CodeCoverage -OutputFormat JUnitXml -FailOnSkipped

- name: Publish Test Results
  uses: dorny/test-reporter@v1
  with:
    name: Pester Tests
    path: C:\Scripts\Startup\Tests\TestResults\*.xml
    reporter: jest-junit
```

### Azure DevOps

```yaml
- task: PowerShell@2
  displayName: 'Run Pester Tests'
  inputs:
    filePath: 'C:\Scripts\Startup\Tests\Run-Tests.ps1'
    arguments: '-CodeCoverage -OutputFormat NUnitXml'

- task: PublishTestResults@2
  inputs:
    testResultsFormat: 'NUnit'
    testResultsFiles: '**/TestResults.NUnitXml.xml'
```

## Test Results

### Output Locations
- **Console** - Immediate feedback
- **TestResults/** - XML reports
  - `TestResults.NUnitXml.xml` - NUnit format
  - `TestResults.JUnitXml.xml` - JUnit format
  - `coverage.xml` - JaCoCo code coverage

### Reading Results

#### Console Output
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

#### Code Coverage
```
CODE COVERAGE SUMMARY
================================================================================

Commands Analyzed:  150
Commands Executed:  135
Commands Missed:    15

Coverage:           90.0% ✓
```

## Troubleshooting

### Common Issues

#### Pester Version Mismatch
```powershell
# Install Pester 5.x
Install-Module -Name Pester -Force -SkipPublisherCheck -Scope CurrentUser
Import-Module Pester -Force
```

#### Mocks Not Working
- Ensure mocks are in `BeforeAll` blocks
- Verify mock scope (It, Context, Describe)
- Check parameter filters match exactly

#### Tests Skipped
Tests are skipped when the script under test doesn't exist yet.
This is expected during TDD - write tests first, implement later.

#### Code Coverage Shows 0%
- Verify script path is correct
- Ensure script is being dot-sourced
- Check script has executable code (not just functions)

## Coverage Goals

### Minimum Requirements
- **Overall Coverage**: 85%
- **Function Coverage**: 90% of all functions tested
- **Branch Coverage**: 80% of conditional branches

### Critical Paths
These must have 100% coverage:
- Error handling and retries
- Service state validation
- Docker engine readiness checks
- Resource cleanup

## Maintenance

### Regular Tasks
- Update tests when script changes
- Review skipped tests
- Maintain mock accuracy
- Update coverage goals

### When to Update Tests
- Adding new functions
- Changing function signatures
- Modifying retry logic
- Adding error handling
- Changing external dependencies

## Additional Resources

- [Pester Documentation](https://pester.dev)
- [Pester Quick Start](https://pester.dev/docs/quick-start)
- [Mocking in Pester](https://pester.dev/docs/usage/mocking)
- [Code Coverage](https://pester.dev/docs/usage/code-coverage)

## License

Same as parent project.
