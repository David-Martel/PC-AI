# CI/CD Integration Guide

Integration examples for running Pester tests in various CI/CD pipelines.

## GitHub Actions

### Basic Workflow

```yaml
name: PowerShell Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: windows-latest

    steps:
    - name: Checkout code
      uses: actions/checkout@v3

    - name: Install Pester
      shell: pwsh
      run: |
        Install-Module -Name Pester -Force -SkipPublisherCheck -Scope CurrentUser
        Import-Module Pester

    - name: Run Pester Tests
      shell: pwsh
      run: |
        cd C:\Scripts\Startup\Tests
        .\Run-Tests.ps1 -CodeCoverage -OutputFormat NUnitXml -MinimumCoverage 85

    - name: Publish Test Results
      uses: dorny/test-reporter@v1
      if: always()
      with:
        name: Pester Tests
        path: C:\Scripts\Startup\Tests\TestResults\*.xml
        reporter: dotnet-nunit

    - name: Upload Coverage
      uses: codecov/codecov-action@v3
      if: always()
      with:
        files: C:\Scripts\Startup\Tests\TestResults\coverage.xml
        flags: powershell
```

### Advanced Workflow with Matrix

```yaml
name: PowerShell Tests (Matrix)

on: [push, pull_request]

jobs:
  test:
    strategy:
      matrix:
        os: [windows-latest, windows-2019]
        pwsh: ['7.2', '7.3', '7.4']

    runs-on: ${{ matrix.os }}

    steps:
    - uses: actions/checkout@v3

    - name: Setup PowerShell ${{ matrix.pwsh }}
      uses: actions/setup-powershell@v1
      with:
        powershell-version: ${{ matrix.pwsh }}

    - name: Install Dependencies
      shell: pwsh
      run: |
        Install-Module -Name Pester -MinimumVersion 5.0 -Force -SkipPublisherCheck
        Import-Module Pester

    - name: Validate Environment
      shell: pwsh
      run: |
        cd C:\Scripts\Startup\Tests
        .\Validate-TestEnvironment.ps1

    - name: Run Tests
      shell: pwsh
      run: |
        cd C:\Scripts\Startup\Tests
        .\Run-Tests.ps1 -CodeCoverage -OutputFormat JUnitXml

    - name: Publish Results
      uses: EnricoMi/publish-unit-test-result-action/composite@v2
      if: always()
      with:
        files: C:\Scripts\Startup\Tests\TestResults\*.xml
```

## Azure DevOps

### Azure Pipelines YAML

```yaml
trigger:
  branches:
    include:
    - main
    - develop

pool:
  vmImage: 'windows-latest'

variables:
  testDirectory: 'C:\Scripts\Startup\Tests'

steps:
- task: PowerShell@2
  displayName: 'Install Pester'
  inputs:
    targetType: 'inline'
    script: |
      Install-Module -Name Pester -MinimumVersion 5.0 -Force -SkipPublisherCheck -Scope CurrentUser
      Import-Module Pester
      Get-Module Pester -ListAvailable

- task: PowerShell@2
  displayName: 'Validate Test Environment'
  inputs:
    filePath: '$(testDirectory)\Validate-TestEnvironment.ps1'
  continueOnError: false

- task: PowerShell@2
  displayName: 'Run Pester Tests'
  inputs:
    filePath: '$(testDirectory)\Run-Tests.ps1'
    arguments: '-CodeCoverage -OutputFormat NUnitXml -MinimumCoverage 85'
    errorActionPreference: 'stop'
    failOnStderr: true

- task: PublishTestResults@2
  displayName: 'Publish Test Results'
  condition: always()
  inputs:
    testResultsFormat: 'NUnit'
    testResultsFiles: '$(testDirectory)\TestResults\*.xml'
    failTaskOnFailedTests: true
    testRunTitle: 'PowerShell Pester Tests'

- task: PublishCodeCoverageResults@1
  displayName: 'Publish Code Coverage'
  condition: always()
  inputs:
    codeCoverageTool: 'JaCoCo'
    summaryFileLocation: '$(testDirectory)\TestResults\coverage.xml'
    reportDirectory: '$(testDirectory)\TestResults'
    failIfCoverageEmpty: true
```

### Azure Pipelines with Quality Gates

```yaml
stages:
- stage: Test
  displayName: 'Test Stage'
  jobs:
  - job: UnitTests
    displayName: 'Unit Tests'
    steps:
    - task: PowerShell@2
      displayName: 'Install Pester'
      inputs:
        targetType: 'inline'
        script: |
          Install-Module Pester -Force -SkipPublisherCheck

    - task: PowerShell@2
      displayName: 'Run Unit Tests'
      inputs:
        filePath: 'C:\Scripts\Startup\Tests\Run-Tests.ps1'
        arguments: '-Tags Unit -CodeCoverage -OutputFormat NUnitXml'

    - task: PublishTestResults@2
      condition: always()
      inputs:
        testResultsFormat: 'NUnit'
        testResultsFiles: '**/TestResults.NUnitXml.xml'
        failTaskOnFailedTests: true

  - job: IntegrationTests
    displayName: 'Integration Tests'
    dependsOn: UnitTests
    steps:
    - task: PowerShell@2
      displayName: 'Run Integration Tests'
      inputs:
        filePath: 'C:\Scripts\Startup\Tests\Run-Tests.ps1'
        arguments: '-Tags Integration -OutputFormat NUnitXml'

- stage: QualityGate
  displayName: 'Quality Gate'
  dependsOn: Test
  jobs:
  - job: CoverageCheck
    displayName: 'Coverage Threshold'
    steps:
    - task: PowerShell@2
      inputs:
        targetType: 'inline'
        script: |
          # Check coverage threshold
          $coverage = Import-Clixml 'TestResults\coverage.xml'
          if ($coverage.CoveragePercent -lt 85) {
            throw "Coverage $($coverage.CoveragePercent)% below threshold 85%"
          }
```

## GitLab CI

### .gitlab-ci.yml

```yaml
image: mcr.microsoft.com/powershell:latest

stages:
  - validate
  - test
  - report

variables:
  TEST_DIR: "C:/Scripts/Startup/Tests"

before_script:
  - pwsh -Command "Install-Module -Name Pester -Force -SkipPublisherCheck"

validate:
  stage: validate
  script:
    - pwsh -File "${TEST_DIR}/Validate-TestEnvironment.ps1"
  only:
    - merge_requests
    - main

test:unit:
  stage: test
  script:
    - pwsh -File "${TEST_DIR}/Run-Tests.ps1" -Tags Unit -OutputFormat JUnitXml
  artifacts:
    reports:
      junit: "${TEST_DIR}/TestResults/*.xml"
    paths:
      - "${TEST_DIR}/TestResults/"
    expire_in: 1 week

test:integration:
  stage: test
  script:
    - pwsh -File "${TEST_DIR}/Run-Tests.ps1" -Tags Integration -OutputFormat JUnitXml
  artifacts:
    reports:
      junit: "${TEST_DIR}/TestResults/*.xml"

test:coverage:
  stage: test
  script:
    - pwsh -File "${TEST_DIR}/Run-Tests.ps1" -CodeCoverage -MinimumCoverage 85
  coverage: '/Coverage:\s+(\d+\.\d+)%/'
  artifacts:
    reports:
      coverage_report:
        coverage_format: cobertura
        path: "${TEST_DIR}/TestResults/coverage.xml"

report:
  stage: report
  script:
    - pwsh -Command "Write-Host 'All tests passed!'"
  dependencies:
    - test:unit
    - test:integration
    - test:coverage
  only:
    - main
```

## Jenkins

### Jenkinsfile

```groovy
pipeline {
    agent {
        label 'windows'
    }

    environment {
        TEST_DIR = 'C:\\Scripts\\Startup\\Tests'
    }

    stages {
        stage('Setup') {
            steps {
                powershell '''
                    Install-Module -Name Pester -MinimumVersion 5.0 -Force -SkipPublisherCheck
                    Import-Module Pester
                '''
            }
        }

        stage('Validate') {
            steps {
                powershell '''
                    & "${env:TEST_DIR}\\Validate-TestEnvironment.ps1"
                '''
            }
        }

        stage('Unit Tests') {
            steps {
                powershell '''
                    & "${env:TEST_DIR}\\Run-Tests.ps1" `
                        -Tags 'Unit' `
                        -OutputFormat NUnitXml `
                        -CodeCoverage
                '''
            }
        }

        stage('Integration Tests') {
            steps {
                powershell '''
                    & "${env:TEST_DIR}\\Run-Tests.ps1" `
                        -Tags 'Integration' `
                        -OutputFormat NUnitXml
                '''
            }
        }

        stage('Publish Results') {
            steps {
                nunit testResultsPattern: "${env:TEST_DIR}\\TestResults\\*.xml"

                publishHTML([
                    allowMissing: false,
                    alwaysLinkToLastBuild: true,
                    keepAll: true,
                    reportDir: "${env:TEST_DIR}\\TestResults",
                    reportFiles: 'coverage.html',
                    reportName: 'Code Coverage'
                ])
            }
        }
    }

    post {
        always {
            archiveArtifacts artifacts: "${env:TEST_DIR}\\TestResults\\**/*", allowEmptyArchive: true
        }

        failure {
            emailext (
                subject: "Failed Pipeline: ${currentBuild.fullDisplayName}",
                body: "Pipeline failed. Check console output.",
                recipientProviders: [developers()]
            )
        }
    }
}
```

## TeamCity

### PowerShell Build Step

```powershell
# Install Pester
Install-Module -Name Pester -Force -SkipPublisherCheck -Scope CurrentUser

# Run tests
& "C:\Scripts\Startup\Tests\Run-Tests.ps1" `
    -CodeCoverage `
    -OutputFormat NUnitXml `
    -MinimumCoverage 85

# Exit with test result
if ($LASTEXITCODE -ne 0) {
    exit 1
}
```

### TeamCity Configuration

1. **Build Step 1:** Install Pester
   - Type: PowerShell
   - Script: `Install-Module -Name Pester -Force -SkipPublisherCheck`

2. **Build Step 2:** Run Tests
   - Type: PowerShell
   - Script file: `C:\Scripts\Startup\Tests\Run-Tests.ps1`
   - Script parameters: `-CodeCoverage -OutputFormat NUnitXml`

3. **Build Feature:** XML Report Processing
   - Report type: NUnit
   - Report paths: `C:\Scripts\Startup\Tests\TestResults\*.xml`

## CircleCI

### .circleci/config.yml

```yaml
version: 2.1

orbs:
  win: circleci/windows@5.0

jobs:
  test:
    executor:
      name: win/default
      shell: powershell.exe

    steps:
      - checkout

      - run:
          name: Install Pester
          command: |
            Install-Module -Name Pester -Force -SkipPublisherCheck
            Import-Module Pester

      - run:
          name: Run Tests
          command: |
            cd C:\Scripts\Startup\Tests
            .\Run-Tests.ps1 -CodeCoverage -OutputFormat JUnitXml

      - store_test_results:
          path: C:\Scripts\Startup\Tests\TestResults

      - store_artifacts:
          path: C:\Scripts\Startup\Tests\TestResults

workflows:
  version: 2
  test:
    jobs:
      - test
```

## Local Pre-Commit Hook

### .git/hooks/pre-commit (PowerShell)

```powershell
#!/usr/bin/env pwsh

Write-Host "Running Pester tests..." -ForegroundColor Cyan

try {
    $result = & "C:\Scripts\Startup\Tests\Run-Tests.ps1" `
        -CodeCoverage `
        -MinimumCoverage 85 `
        -ErrorAction Stop

    if ($LASTEXITCODE -ne 0) {
        Write-Host "Tests failed. Commit aborted." -ForegroundColor Red
        exit 1
    }

    Write-Host "All tests passed!" -ForegroundColor Green
    exit 0

} catch {
    Write-Host "Error running tests: $_" -ForegroundColor Red
    exit 1
}
```

## Docker Integration

### Dockerfile for Testing

```dockerfile
FROM mcr.microsoft.com/powershell:latest

# Install Pester
RUN pwsh -Command "Install-Module -Name Pester -Force -SkipPublisherCheck"

# Copy test files
WORKDIR /tests
COPY Tests/ .

# Run tests
CMD ["pwsh", "-File", "Run-Tests.ps1", "-CodeCoverage", "-OutputFormat", "NUnitXml"]
```

### docker-compose.yml

```yaml
version: '3.8'

services:
  pester-tests:
    build:
      context: .
      dockerfile: Dockerfile
    volumes:
      - ./Tests:/tests
      - ./TestResults:/tests/TestResults
    environment:
      - MINIMUM_COVERAGE=85
```

## Coverage Badge Integration

### Codecov Integration

```yaml
# In GitHub Actions
- name: Upload to Codecov
  uses: codecov/codecov-action@v3
  with:
    files: ./TestResults/coverage.xml
    flags: powershell
    name: powershell-coverage
```

### Coveralls Integration

```yaml
- name: Upload to Coveralls
  uses: coverallsapp/github-action@v2
  with:
    github-token: ${{ secrets.GITHUB_TOKEN }}
    path-to-lcov: ./TestResults/coverage.xml
```

## Best Practices for CI/CD

1. **Always install specific Pester version** to ensure consistency
2. **Fail fast** - Stop pipeline on first test failure
3. **Separate unit and integration tests** for faster feedback
4. **Archive test results** for debugging failed builds
5. **Set coverage thresholds** and enforce them
6. **Use parallel execution** when possible
7. **Cache Pester module** to speed up builds
8. **Run tests on multiple OS versions** to ensure compatibility

## Troubleshooting CI/CD Issues

### Tests Pass Locally But Fail in CI

- Check PowerShell version differences
- Verify environment variables are set
- Look for path differences (case sensitivity, separators)
- Check for missing dependencies

### Coverage Not Uploading

- Verify coverage.xml file is generated
- Check file path in upload step
- Ensure coverage format matches service expectation
- Verify authentication tokens are set

### Slow Test Execution

- Use parallel test execution
- Cache Pester module installation
- Skip integration tests in PR builds
- Use test filtering to run only affected tests

---

**Remember:** CI/CD integration ensures tests run consistently across all environments and prevent regressions from reaching production.
