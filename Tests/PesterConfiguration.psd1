@{
    Run = @{
        Path = @(
            'Unit'
            'Integration'
        )
        Exit = $false
        PassThru = $true
    }

    CodeCoverage = @{
        Enabled = $true
        OutputFormat = 'JaCoCo'
        OutputPath = 'TestResults/coverage.xml'
        Path = @(
            './Modules/**/*.ps1'
            './Modules/**/*.psm1'
        )
        ExcludeTests = $true
        RecursePaths = $true
        CoveragePercentTarget = 85
    }

    TestResult = @{
        Enabled = $false  # Enable via .pester.ps1 -CI flag
        OutputFormat = 'NUnitXml'
        OutputPath = 'test-results.xml'
        TestSuiteName = 'PC_AI_Test_Suite'
    }

    Output = @{
        Verbosity = 'Detailed'  # Detailed for local dev, Normal for CI
        StackTraceVerbosity = 'Filtered'
        CIFormat = 'Auto'
    }

    Filter = @{
        Tag = @()
        ExcludeTag = @()
        Line = @()
    }

    Should = @{
        ErrorAction = 'Stop'
    }

    Debug = @{
        ShowFullErrors = $true
        WriteDebugMessages = $false
        WriteDebugMessagesFrom = @()
        ShowNavigationMarkers = $false
    }
}
