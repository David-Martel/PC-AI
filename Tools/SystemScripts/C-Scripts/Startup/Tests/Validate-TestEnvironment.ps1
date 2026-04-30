<#
.SYNOPSIS
    Validates the Pester test environment setup

.DESCRIPTION
    Checks that all required components are installed and configured correctly
    for running the WSL/Docker Services test suite.

.EXAMPLE
    .\Validate-TestEnvironment.ps1

.EXAMPLE
    .\Validate-TestEnvironment.ps1 -Verbose
#>

[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Continue'

# Configuration
$requiredPesterVersion = [version]'5.0.0'
$testDirectory = $PSScriptRoot
$scriptUnderTest = Join-Path (Split-Path $testDirectory -Parent) 'Start-WSLDockerServices.ps1'

# Results tracking
$script:ValidationResults = @{
    Passed  = @()
    Failed  = @()
    Warnings = @()
}

function Write-ValidationResult {
    param(
        [string]$Test,
        [bool]$Success,
        [string]$Message = '',
        [switch]$Warning
    )

    $symbol = if ($Success) { '[PASS]' } else { '[FAIL]' }
    $color = if ($Success) { 'Green' } elseif ($Warning) { 'Yellow' } else { 'Red' }

    $output = "$symbol $Test"
    if ($Message) {
        $output += " - $Message"
    }

    Write-Host $output -ForegroundColor $color

    if ($Success) {
        $script:ValidationResults.Passed += $Test
    } elseif ($Warning) {
        $script:ValidationResults.Warnings += $Test
    } else {
        $script:ValidationResults.Failed += $Test
    }
}

Write-Host ""
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "Pester Test Environment Validation" -ForegroundColor Cyan
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host ""

#region PowerShell Version Check

Write-Host "Checking PowerShell Version..." -ForegroundColor Cyan

$psVersion = $PSVersionTable.PSVersion
$minVersion = [version]'5.1'

if ($psVersion -ge $minVersion) {
    Write-ValidationResult -Test "PowerShell Version" -Success $true `
        -Message "v$psVersion (minimum: v$minVersion)"
} else {
    Write-ValidationResult -Test "PowerShell Version" -Success $false `
        -Message "v$psVersion - Requires v$minVersion or higher"
}

#endregion

#region Pester Module Check

Write-Host ""
Write-Host "Checking Pester Module..." -ForegroundColor Cyan

try {
    $pesterModule = Get-Module -Name Pester -ListAvailable |
        Sort-Object Version -Descending |
        Select-Object -First 1

    if ($pesterModule) {
        if ($pesterModule.Version -ge $requiredPesterVersion) {
            Write-ValidationResult -Test "Pester Module Installed" -Success $true `
                -Message "v$($pesterModule.Version) (minimum: v$requiredPesterVersion)"

            # Check if Pester can be imported
            try {
                Import-Module Pester -Force -MinimumVersion $requiredPesterVersion -ErrorAction Stop
                Write-ValidationResult -Test "Pester Module Import" -Success $true
            } catch {
                Write-ValidationResult -Test "Pester Module Import" -Success $false `
                    -Message $_.Exception.Message
            }
        } else {
            Write-ValidationResult -Test "Pester Module Version" -Success $false `
                -Message "v$($pesterModule.Version) - Requires v$requiredPesterVersion or higher"
            Write-Host "  Install with: Install-Module -Name Pester -Force -SkipPublisherCheck" -ForegroundColor Yellow
        }
    } else {
        Write-ValidationResult -Test "Pester Module Installed" -Success $false `
            -Message "Not found"
        Write-Host "  Install with: Install-Module -Name Pester -Force -SkipPublisherCheck" -ForegroundColor Yellow
    }
} catch {
    Write-ValidationResult -Test "Pester Module Check" -Success $false `
        -Message $_.Exception.Message
}

#endregion

#region Test Files Check

Write-Host ""
Write-Host "Checking Test Files..." -ForegroundColor Cyan

$requiredFiles = @{
    'Start-WSLDockerServices.Tests.ps1' = 'Main test suite'
    'Run-Tests.ps1'                      = 'Test runner script'
    'TestHelpers.psm1'                   = 'Test helper module'
    'README.md'                          = 'Test documentation'
}

foreach ($file in $requiredFiles.GetEnumerator()) {
    $filePath = Join-Path $testDirectory $file.Key
    $exists = Test-Path $filePath
    $message = if ($exists) { "Found: $filePath" } else { "Missing: $filePath" }

    Write-ValidationResult -Test $file.Value -Success $exists -Message $message
}

#endregion

#region Test Helper Module Check

Write-Host ""
Write-Host "Checking Test Helper Module..." -ForegroundColor Cyan

$helperModulePath = Join-Path $testDirectory 'TestHelpers.psm1'

if (Test-Path $helperModulePath) {
    try {
        Import-Module $helperModulePath -Force -ErrorAction Stop

        # Check for expected functions
        $expectedFunctions = @(
            'New-MockService',
            'New-MockProcess',
            'New-RetryScenario',
            'Assert-RetryCount',
            'Measure-ExecutionTime'
        )

        $allFunctionsPresent = $true
        foreach ($funcName in $expectedFunctions) {
            if (Get-Command $funcName -ErrorAction SilentlyContinue) {
                Write-Verbose "Found helper function: $funcName"
            } else {
                Write-ValidationResult -Test "Helper function: $funcName" -Success $false `
                    -Message "Not exported from module"
                $allFunctionsPresent = $false
            }
        }

        if ($allFunctionsPresent) {
            Write-ValidationResult -Test "Test Helper Functions" -Success $true `
                -Message "All $($expectedFunctions.Count) helper functions available"
        }

        # Test a helper function
        try {
            $mockService = New-MockService -Name 'Test' -Status 'Running'
            if ($mockService.Name -eq 'Test') {
                Write-ValidationResult -Test "Helper Function Execution" -Success $true `
                    -Message "New-MockService works correctly"
            }
        } catch {
            Write-ValidationResult -Test "Helper Function Execution" -Success $false `
                -Message $_.Exception.Message
        }

    } catch {
        Write-ValidationResult -Test "Test Helper Module Import" -Success $false `
            -Message $_.Exception.Message
    }
} else {
    Write-ValidationResult -Test "Test Helper Module" -Success $false `
        -Message "File not found"
}

#endregion

#region Script Under Test Check

Write-Host ""
Write-Host "Checking Script Under Test..." -ForegroundColor Cyan

if (Test-Path $scriptUnderTest) {
    Write-ValidationResult -Test "Script Exists" -Success $true `
        -Message $scriptUnderTest

    # Try to parse the script
    try {
        $null = [System.Management.Automation.PSParser]::Tokenize(
            (Get-Content $scriptUnderTest -Raw),
            [ref]$null
        )
        Write-ValidationResult -Test "Script Syntax Valid" -Success $true
    } catch {
        Write-ValidationResult -Test "Script Syntax Valid" -Success $false `
            -Message $_.Exception.Message
    }
} else {
    Write-ValidationResult -Test "Script Exists" -Success $false `
        -Message "Not found (tests will be skipped)" -Warning
    Write-Host "  This is OK if you're practicing TDD" -ForegroundColor Yellow
}

#endregion

#region Directory Structure Check

Write-Host ""
Write-Host "Checking Directory Structure..." -ForegroundColor Cyan

$expectedDirs = @{
    'Tests'       = $testDirectory
    'TestResults' = (Join-Path $testDirectory 'TestResults')
}

foreach ($dir in $expectedDirs.GetEnumerator()) {
    $exists = Test-Path $dir.Value

    if ($dir.Key -eq 'TestResults' -and -not $exists) {
        Write-ValidationResult -Test "$($dir.Key) Directory" -Success $true `
            -Message "Will be created on first test run" -Warning
    } else {
        Write-ValidationResult -Test "$($dir.Key) Directory" -Success $exists `
            -Message $dir.Value
    }
}

#endregion

#region Permissions Check

Write-Host ""
Write-Host "Checking Permissions..." -ForegroundColor Cyan

try {
    $testFile = Join-Path $testDirectory "test_write_$(Get-Random).tmp"
    "test" | Out-File $testFile -ErrorAction Stop
    Remove-Item $testFile -ErrorAction Stop

    Write-ValidationResult -Test "Write Permissions" -Success $true `
        -Message "Can write to test directory"
} catch {
    Write-ValidationResult -Test "Write Permissions" -Success $false `
        -Message $_.Exception.Message
}

#endregion

#region Quick Test Run

Write-Host ""
Write-Host "Performing Quick Test Run..." -ForegroundColor Cyan

try {
    # Run example tests
    $exampleTestPath = Join-Path $testDirectory 'Example.Tests.ps1'

    if (Test-Path $exampleTestPath) {
        $config = New-PesterConfiguration
        $config.Run.Path = $exampleTestPath
        $config.Output.Verbosity = 'None'
        $config.Run.PassThru = $true

        $result = Invoke-Pester -Configuration $config

        if ($result.FailedCount -eq 0) {
            Write-ValidationResult -Test "Example Tests" -Success $true `
                -Message "Passed: $($result.PassedCount), Failed: $($result.FailedCount)"
        } else {
            Write-ValidationResult -Test "Example Tests" -Success $false `
                -Message "Passed: $($result.PassedCount), Failed: $($result.FailedCount)"
        }
    } else {
        Write-ValidationResult -Test "Example Tests" -Success $false `
            -Message "Example.Tests.ps1 not found" -Warning
    }
} catch {
    Write-ValidationResult -Test "Quick Test Run" -Success $false `
        -Message $_.Exception.Message
}

#endregion

#region Summary

Write-Host ""
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "Validation Summary" -ForegroundColor Cyan
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host ""

$totalTests = $script:ValidationResults.Passed.Count +
              $script:ValidationResults.Failed.Count +
              $script:ValidationResults.Warnings.Count

Write-Host "Total Checks:  $totalTests" -ForegroundColor Cyan
Write-Host "Passed:        $($script:ValidationResults.Passed.Count)" -ForegroundColor Green
Write-Host "Failed:        $($script:ValidationResults.Failed.Count)" -ForegroundColor Red
Write-Host "Warnings:      $($script:ValidationResults.Warnings.Count)" -ForegroundColor Yellow
Write-Host ""

if ($script:ValidationResults.Failed.Count -eq 0) {
    Write-Host "[PASS] Environment is ready for testing!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Cyan
    Write-Host "  1. Run tests: .\Run-Tests.ps1" -ForegroundColor White
    Write-Host "  2. View example tests: .\Example.Tests.ps1" -ForegroundColor White
    Write-Host "  3. Read documentation: .\README.md" -ForegroundColor White
    exit 0
} else {
    Write-Host "[FAIL] Environment setup incomplete" -ForegroundColor Red
    Write-Host ""
    Write-Host "Failed checks:" -ForegroundColor Red
    foreach ($fail in $script:ValidationResults.Failed) {
        Write-Host "  - $fail" -ForegroundColor Red
    }

    Write-Host ""
    Write-Host "Please resolve the issues above before running tests." -ForegroundColor Yellow
    exit 1
}

#endregion
