@{
    Run = @{
        # Paths are resolved from the repo root (the CI working directory and
        # the cwd assumed by CodeCoverage.Path './Modules/**' below). Bare
        # 'Unit'/'Integration' resolved to ./Unit at the repo root and made
        # Pester find zero test files ("No test files were found" -> exit 1).
        Path = @(
            'Tests/Unit'
            'Tests/Integration'
        )
        Exit = $false
        PassThru = $true
    }

    CodeCoverage = @{
        Enabled = $true
        OutputFormat = 'JaCoCo'
        OutputPath = 'TestResults/coverage.xml'
        # A directory plus RecursePaths, not './Modules/**/*.ps1'. Pester treats '**' like '*', so the glob
        # analysed only files two levels down (17 of 241 module files, measured 2026-11). Tests/coverage-report.ps1
        # fails when fewer than 95% of the module files are analysed, so this cannot silently narrow again.
        Path = @(
            './Modules'
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
