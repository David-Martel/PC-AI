<#
.SYNOPSIS
    Test runner for Start-WSLDockerServices.ps1 Pester tests

.DESCRIPTION
    Executes Pester tests for the WSL/Docker startup script and generates
    comprehensive test reports. Supports multiple output formats and coverage analysis.

.PARAMETER TestPath
    Path to specific test file. Defaults to all tests in the Tests directory.

.PARAMETER OutputFormat
    Test output format: NUnitXml, JUnitXml, or NUnit2.5
    Default: None (console output only)

.PARAMETER CodeCoverage
    Enable code coverage analysis

.PARAMETER ShowTestDetails
    Show detailed test output (Diagnostic verbosity)

.PARAMETER FailOnSkipped
    Treat skipped tests as failures

.PARAMETER Tags
    Run only tests with specific tags

.PARAMETER ExcludeTags
    Exclude tests with specific tags

.EXAMPLE
    .\Run-Tests.ps1
    Run all tests with default settings

.EXAMPLE
    .\Run-Tests.ps1 -ShowTestDetails
    Run all tests with detailed output

.EXAMPLE
    .\Run-Tests.ps1 -CodeCoverage -OutputFormat NUnitXml
    Run tests with code coverage and XML output

.EXAMPLE
    .\Run-Tests.ps1 -Tags 'Unit' -ExcludeTags 'Integration'
    Run only unit tests, excluding integration tests

.NOTES
    Requires Pester 5.x
    Install with: Install-Module -Name Pester -Force -SkipPublisherCheck
#>

[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string]$TestPath,

    [Parameter()]
    [ValidateSet('NUnitXml', 'JUnitXml', 'NUnit2.5')]
    [string]$OutputFormat,

    [Parameter()]
    [switch]$CodeCoverage,

    [Parameter()]
    [switch]$ShowTestDetails,

    [Parameter()]
    [switch]$FailOnSkipped,

    [Parameter()]
    [string[]]$Tags,

    [Parameter()]
    [string[]]$ExcludeTags,

    [Parameter()]
    [int]$MinimumCoverage = 80
)

#Requires -Modules Pester

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# Script configuration
$scriptRoot = $PSScriptRoot
$testsDirectory = $scriptRoot
$scriptUnderTest = Join-Path (Split-Path $scriptRoot -Parent) 'Start-WSLDockerServices.ps1'
$outputDirectory = Join-Path $scriptRoot 'TestResults'

# ANSI color codes for better console output
$colors = @{
    Success = "`e[32m"  # Green
    Failure = "`e[31m"  # Red
    Warning = "`e[33m"  # Yellow
    Info    = "`e[36m"  # Cyan
    Reset   = "`e[0m"   # Reset
}

function Write-ColorOutput {
    param(
        [string]$Message,
        [string]$Color = 'Info'
    )

    if ($colors.ContainsKey($Color)) {
        Write-Host "$($colors[$Color])$Message$($colors.Reset)"
    } else {
        Write-Host $Message
    }
}

function Test-PesterVersion {
    <#
    .SYNOPSIS
        Validates Pester version is 5.x or higher
    #>

    $pesterModule = Get-Module -Name Pester -ListAvailable | Sort-Object Version -Descending | Select-Object -First 1

    if (-not $pesterModule) {
        Write-ColorOutput "Pester module not found. Installing..." -Color Warning
        Install-Module -Name Pester -Force -SkipPublisherCheck -Scope CurrentUser
        Import-Module Pester -Force
        $pesterModule = Get-Module -Name Pester
    }

    if ($pesterModule.Version.Major -lt 5) {
        Write-ColorOutput "ERROR: Pester 5.x or higher required. Found version $($pesterModule.Version)" -Color Failure
        Write-ColorOutput "Install with: Install-Module -Name Pester -Force -SkipPublisherCheck" -Color Info
        throw "Incompatible Pester version"
    }

    Write-ColorOutput "Using Pester version $($pesterModule.Version)" -Color Success
}

function Initialize-TestEnvironment {
    <#
    .SYNOPSIS
        Prepares the test environment
    #>

    # Create output directory if it doesn't exist
    if (-not (Test-Path $outputDirectory)) {
        New-Item -Path $outputDirectory -ItemType Directory -Force | Out-Null
        Write-ColorOutput "Created test results directory: $outputDirectory" -Color Info
    }

    # Clean old test results
    Get-ChildItem -Path $outputDirectory -Filter "*.xml" | Remove-Item -Force
    Write-ColorOutput "Cleaned old test results" -Color Info
}

function New-PesterConfiguration {
    <#
    .SYNOPSIS
        Creates Pester configuration object
    #>

    $config = New-PesterConfiguration

    # Test discovery
    if ($TestPath) {
        $config.Run.Path = $TestPath
    } else {
        $config.Run.Path = $testsDirectory
    }

    # Output configuration
    if ($ShowTestDetails) {
        $config.Output.Verbosity = 'Detailed'
    } else {
        $config.Output.Verbosity = 'Normal'
    }

    # Code coverage configuration
    if ($CodeCoverage) {
        if (Test-Path $scriptUnderTest) {
            $config.CodeCoverage.Enabled = $true
            $config.CodeCoverage.Path = $scriptUnderTest
            $config.CodeCoverage.OutputPath = Join-Path $outputDirectory 'coverage.xml'
            $config.CodeCoverage.OutputFormat = 'JaCoCo'
        } else {
            Write-ColorOutput "WARNING: Script under test not found: $scriptUnderTest" -Color Warning
            Write-ColorOutput "Code coverage disabled" -Color Warning
            $config.CodeCoverage.Enabled = $false
        }
    }

    # Test result export
    if ($OutputFormat) {
        $config.TestResult.Enabled = $true
        $config.TestResult.OutputFormat = $OutputFormat
        $config.TestResult.OutputPath = Join-Path $outputDirectory "TestResults.$OutputFormat.xml"
    }

    # Filter configuration
    if ($Tags) {
        $config.Filter.Tag = $Tags
    }

    if ($ExcludeTags) {
        $config.Filter.ExcludeTag = $ExcludeTags
    }

    # Behavior configuration
    $config.Run.Exit = $false
    $config.Run.PassThru = $true

    return $config
}

function Format-TestResults {
    <#
    .SYNOPSIS
        Formats and displays test results
    #>
    param(
        [object]$Results
    )

    Write-Host ""
    Write-Host "=" * 80
    Write-ColorOutput "TEST RESULTS SUMMARY" -Color Info
    Write-Host "=" * 80

    # Overall statistics
    Write-Host ""
    Write-ColorOutput "Total Tests:    $($Results.TotalCount)" -Color Info
    Write-ColorOutput "Passed:         $($Results.PassedCount)" -Color Success
    Write-ColorOutput "Failed:         $($Results.FailedCount)" -Color Failure
    Write-ColorOutput "Skipped:        $($Results.SkippedCount)" -Color Warning
    Write-ColorOutput "Not Run:        $($Results.NotRunCount)" -Color Warning

    # Duration
    $duration = $Results.Duration
    Write-Host ""
    Write-ColorOutput "Execution Time: $($duration.ToString('mm\:ss\.fff'))" -Color Info

    # Failed tests detail
    if ($Results.FailedCount -gt 0) {
        Write-Host ""
        Write-ColorOutput "FAILED TESTS:" -Color Failure
        Write-Host "-" * 80

        foreach ($test in $Results.Tests | Where-Object { $_.Result -eq 'Failed' }) {
            Write-ColorOutput "  × $($test.ExpandedName)" -Color Failure
            if ($test.ErrorRecord) {
                Write-ColorOutput "    Error: $($test.ErrorRecord.Exception.Message)" -Color Failure
            }
        }
    }

    # Skipped tests detail
    if ($Results.SkippedCount -gt 0) {
        Write-Host ""
        Write-ColorOutput "SKIPPED TESTS:" -Color Warning
        Write-Host "-" * 80

        foreach ($test in $Results.Tests | Where-Object { $_.Result -eq 'Skipped' }) {
            Write-ColorOutput "  ○ $($test.ExpandedName)" -Color Warning
            if ($test.Skip) {
                Write-ColorOutput "    Reason: $($test.Skip)" -Color Warning
            }
        }
    }

    Write-Host ""
    Write-Host "=" * 80
}

function Format-CodeCoverageResults {
    <#
    .SYNOPSIS
        Formats and displays code coverage results
    #>
    param(
        [object]$Results
    )

    if (-not $Results.CodeCoverage) {
        return
    }

    $coverage = $Results.CodeCoverage
    $coveredPercent = 0

    if ($coverage.NumberOfCommandsAnalyzed -gt 0) {
        $coveredPercent = [math]::Round(
            ($coverage.NumberOfCommandsExecuted / $coverage.NumberOfCommandsAnalyzed) * 100,
            2
        )
    }

    Write-Host ""
    Write-Host "=" * 80
    Write-ColorOutput "CODE COVERAGE SUMMARY" -Color Info
    Write-Host "=" * 80
    Write-Host ""

    Write-ColorOutput "Commands Analyzed:  $($coverage.NumberOfCommandsAnalyzed)" -Color Info
    Write-ColorOutput "Commands Executed:  $($coverage.NumberOfCommandsExecuted)" -Color Info
    Write-ColorOutput "Commands Missed:    $($coverage.NumberOfCommandsMissed)" -Color Info
    Write-Host ""

    if ($coveredPercent -ge $MinimumCoverage) {
        Write-ColorOutput "Coverage:           $coveredPercent% ✓" -Color Success
    } else {
        Write-ColorOutput "Coverage:           $coveredPercent% (minimum: $MinimumCoverage%)" -Color Failure
    }

    # Missed commands detail
    if ($coverage.MissedCommands.Count -gt 0) {
        Write-Host ""
        Write-ColorOutput "MISSED COMMANDS:" -Color Warning
        Write-Host "-" * 80

        $grouped = $coverage.MissedCommands | Group-Object -Property File

        foreach ($group in $grouped) {
            Write-ColorOutput "  File: $($group.Name)" -Color Info
            foreach ($cmd in $group.Group | Select-Object -First 10) {
                Write-ColorOutput "    Line $($cmd.Line): $($cmd.Command)" -Color Warning
            }

            if ($group.Count -gt 10) {
                Write-ColorOutput "    ... and $($group.Count - 10) more" -Color Warning
            }
        }
    }

    Write-Host ""
    Write-Host "=" * 80
}

function Test-BuildSuccess {
    <#
    .SYNOPSIS
        Determines if test run should be considered successful
    #>
    param(
        [object]$Results
    )

    $success = $true
    $reasons = @()

    # Check for test failures
    if ($Results.FailedCount -gt 0) {
        $success = $false
        $reasons += "Failed tests: $($Results.FailedCount)"
    }

    # Check for skipped tests if configured
    if ($FailOnSkipped -and $Results.SkippedCount -gt 0) {
        $success = $false
        $reasons += "Skipped tests: $($Results.SkippedCount)"
    }

    # Check code coverage if enabled
    if ($CodeCoverage -and $Results.CodeCoverage) {
        $coveredPercent = 0
        if ($Results.CodeCoverage.NumberOfCommandsAnalyzed -gt 0) {
            $coveredPercent = [math]::Round(
                ($Results.CodeCoverage.NumberOfCommandsExecuted / $Results.CodeCoverage.NumberOfCommandsAnalyzed) * 100,
                2
            )
        }

        if ($coveredPercent -lt $MinimumCoverage) {
            $success = $false
            $reasons += "Code coverage below minimum: $coveredPercent% < $MinimumCoverage%"
        }
    }

    return @{
        Success = $success
        Reasons = $reasons
    }
}

# Main execution
try {
    Write-ColorOutput "WSL/Docker Services Test Runner" -Color Info
    Write-ColorOutput "================================" -Color Info
    Write-Host ""

    # Validate environment
    Write-ColorOutput "Validating test environment..." -Color Info
    Test-PesterVersion
    Initialize-TestEnvironment

    # Show configuration
    Write-Host ""
    Write-ColorOutput "Test Configuration:" -Color Info
    Write-ColorOutput "  Test Path:       $testsDirectory" -Color Info
    Write-ColorOutput "  Script Under Test: $(if (Test-Path $scriptUnderTest) { 'Found' } else { 'Not Found' })" -Color $(if (Test-Path $scriptUnderTest) { 'Success' } else { 'Warning' })
    Write-ColorOutput "  Code Coverage:   $CodeCoverage" -Color Info
    Write-ColorOutput "  Output Format:   $(if ($OutputFormat) { $OutputFormat } else { 'Console Only' })" -Color Info
    if ($Tags) {
        Write-ColorOutput "  Tags:            $($Tags -join ', ')" -Color Info
    }
    if ($ExcludeTags) {
        Write-ColorOutput "  Exclude Tags:    $($ExcludeTags -join ', ')" -Color Info
    }

    Write-Host ""
    Write-ColorOutput "Running tests..." -Color Info
    Write-Host ""

    # Create configuration and run tests
    $pesterConfig = New-PesterConfiguration
    $results = Invoke-Pester -Configuration $pesterConfig

    # Display results
    Format-TestResults -Results $results

    if ($CodeCoverage -and $results.CodeCoverage) {
        Format-CodeCoverageResults -Results $results
    }

    # Export detailed results if requested
    if ($OutputFormat) {
        Write-Host ""
        Write-ColorOutput "Test results exported to:" -Color Info
        Write-ColorOutput "  $($pesterConfig.TestResult.OutputPath.Value)" -Color Success

        if ($CodeCoverage -and $results.CodeCoverage) {
            Write-ColorOutput "Coverage results exported to:" -Color Info
            Write-ColorOutput "  $($pesterConfig.CodeCoverage.OutputPath.Value)" -Color Success
        }
    }

    # Determine success
    $buildResult = Test-BuildSuccess -Results $results

    Write-Host ""
    if ($buildResult.Success) {
        Write-ColorOutput "✓ TEST RUN SUCCESSFUL" -Color Success
        exit 0
    } else {
        Write-ColorOutput "✗ TEST RUN FAILED" -Color Failure
        Write-Host ""
        Write-ColorOutput "Failure Reasons:" -Color Failure
        foreach ($reason in $buildResult.Reasons) {
            Write-ColorOutput "  • $reason" -Color Failure
        }
        exit 1
    }

} catch {
    Write-ColorOutput "ERROR: Test execution failed" -Color Failure
    Write-ColorOutput "  $($_.Exception.Message)" -Color Failure
    Write-Host ""
    Write-Host $_.ScriptStackTrace

    exit 1
} finally {
    Write-Host ""
}
