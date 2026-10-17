#Requires -Version 7.0

BeforeAll {
    $script:RepoRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
    $script:GateScript = Join-Path $script:RepoRoot 'Tests\Invoke-BenchmarkGate.ps1'
    $script:CurrentGitHead = (git -C $script:RepoRoot rev-parse --short HEAD).Trim()

    function Invoke-GateProcess {
        param(
            [string[]]$Arguments
        )
        $output = & pwsh -NoProfile -File $script:GateScript @Arguments 2>&1 | Out-String
        return [PSCustomObject]@{
            ExitCode = $LASTEXITCODE
            Output   = $output
        }
    }

    function New-TestBaselineFile {
        param(
            [string]$Path,
            [string]$GitHash = $script:CurrentGitHead,
            [hashtable]$Cases = @{
                'token-estimate' = @{ mean_ms = 10.0; median_ms = 10.0; iterations = 5 }
                'file-search'    = @{ mean_ms = 20.0; median_ms = 20.0; iterations = 5 }
            }
        )
        $dir = Split-Path $Path -Parent
        if (-not (Test-Path -LiteralPath $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
        }
        [ordered]@{
            created_at = (Get-Date -Format 'o')
            git_hash   = $GitHash
            git_branch = 'test'
            suite      = 'quick'
            cases      = $Cases
        } | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $Path -Encoding UTF8
    }

    function New-TestReportFile {
        param(
            [string]$Path,
            [string]$GitHead = $script:CurrentGitHead,
            [bool]$EquivalenceVerified = $true,
            [array]$Results = @(
                @{ CaseId = 'token-estimate'; Backend = 'native'; MeanMs = 10.5; MedianMs = 10.4; Iterations = 5 }
                @{ CaseId = 'file-search'; Backend = 'native'; MeanMs = 20.5; MedianMs = 20.2; Iterations = 5 }
            )
        )
        $dir = Split-Path $Path -Parent
        if (-not (Test-Path -LiteralPath $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
        }
        [ordered]@{
            Generated           = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss')
            RepoRoot            = $script:RepoRoot
            GitHead             = $GitHead
            Suite               = 'quick'
            EquivalenceVerified = $EquivalenceVerified
            Results             = $Results
        } | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $Path -Encoding UTF8
    }
}

Describe 'Invoke-BenchmarkGate Fail-Closed Contract' -Tag 'Unit', 'BenchmarkGate', 'Portable' {
    It '(a) fails with exit code 1 when the benchmark runner exits non-zero even if a report exists' {
        $workDir = Join-Path $TestDrive 'case-a'
        $baselinePath = Join-Path $workDir '.pcai\benchmarks\baseline.json'
        $reportPath = Join-Path $workDir 'Reports\tooling-benchmarks\20260101_000000\tooling-benchmark-report.json'
        $runnerPath = Join-Path $workDir 'failing-runner.ps1'

        New-TestBaselineFile -Path $baselinePath
        New-TestReportFile -Path $reportPath -GitHead $script:CurrentGitHead
        Set-Content -LiteralPath $runnerPath -Value 'Write-Error "Simulated runner failure"; exit 2' -Encoding UTF8

        $res = Invoke-GateProcess -Arguments @(
            '-RepoRoot', $workDir,
            '-BaselinePath', $baselinePath,
            '-RunnerPath', $runnerPath,
            '-FailOnRegression'
        )

        $res.ExitCode | Should -Be 1
        $res.Output | Should -Match 'Benchmark runner failed with exit code 2'
    }

    It '(b) fails with exit code 1 when the baseline file is missing' {
        $workDir = Join-Path $TestDrive 'case-b'
        $missingBaseline = Join-Path $workDir '.pcai\benchmarks\missing-baseline.json'

        $res = Invoke-GateProcess -Arguments @(
            '-RepoRoot', $workDir,
            '-BaselinePath', $missingBaseline,
            '-SkipRunner',
            '-FailOnRegression'
        )

        $res.ExitCode | Should -Be 1
        $res.Output | Should -Match 'Baseline file not found'
    }

    It '(c) fails with exit code 1 when the benchmark report file is missing' {
        $workDir = Join-Path $TestDrive 'case-c'
        $baselinePath = Join-Path $workDir '.pcai\benchmarks\baseline.json'
        New-TestBaselineFile -Path $baselinePath

        $res = Invoke-GateProcess -Arguments @(
            '-RepoRoot', $workDir,
            '-BaselinePath', $baselinePath,
            '-SkipRunner',
            '-FailOnRegression'
        )

        $res.ExitCode | Should -Be 1
        $res.Output | Should -Match 'No benchmark report found'
    }

    It '(d) fails with exit code 1 when report GitHead is stale/foreign unless -AllowStaleReport is passed' {
        $workDir = Join-Path $TestDrive 'case-d'
        $baselinePath = Join-Path $workDir '.pcai\benchmarks\baseline.json'
        $reportPath = Join-Path $workDir 'Reports\tooling-benchmarks\20260101_000000\tooling-benchmark-report.json'

        New-TestBaselineFile -Path $baselinePath
        New-TestReportFile -Path $reportPath -GitHead 'deadbeef0'

        $strictRes = Invoke-GateProcess -Arguments @(
            '-RepoRoot', $workDir,
            '-BaselinePath', $baselinePath,
            '-ReportPath', $reportPath,
            '-SkipRunner',
            '-FailOnRegression'
        )
        $strictRes.ExitCode | Should -Be 1
        $strictRes.Output | Should -Match 'does not match current HEAD'

        $allowedRes = Invoke-GateProcess -Arguments @(
            '-RepoRoot', $workDir,
            '-BaselinePath', $baselinePath,
            '-ReportPath', $reportPath,
            '-SkipRunner',
            '-AllowStaleReport',
            '-FailOnRegression'
        )
        $allowedRes.ExitCode | Should -Be 0
    }

    It '(e) fails with exit code 1 when a baseline benchmark case is missing from the current report' {
        $workDir = Join-Path $TestDrive 'case-e'
        $baselinePath = Join-Path $workDir '.pcai\benchmarks\baseline.json'
        $reportPath = Join-Path $workDir 'Reports\tooling-benchmarks\20260101_000000\tooling-benchmark-report.json'

        New-TestBaselineFile -Path $baselinePath
        New-TestReportFile -Path $reportPath -GitHead $script:CurrentGitHead -Results @(
            @{ CaseId = 'token-estimate'; Backend = 'native'; MeanMs = 10.1; MedianMs = 10.0; Iterations = 5 }
        )

        $res = Invoke-GateProcess -Arguments @(
            '-RepoRoot', $workDir,
            '-BaselinePath', $baselinePath,
            '-ReportPath', $reportPath,
            '-SkipRunner',
            '-FailOnRegression'
        )

        $res.ExitCode | Should -Be 1
        $res.Output | Should -Match 'baseline case\(s\) missing from current report'
    }

    It '(f) exits 0 when report is fresh, matches GitHead, covers all baseline cases, and is within threshold' {
        $workDir = Join-Path $TestDrive 'case-f'
        $baselinePath = Join-Path $workDir '.pcai\benchmarks\baseline.json'
        $reportPath = Join-Path $workDir 'Reports\tooling-benchmarks\20260101_000000\tooling-benchmark-report.json'

        New-TestBaselineFile -Path $baselinePath
        New-TestReportFile -Path $reportPath -GitHead $script:CurrentGitHead

        $res = Invoke-GateProcess -Arguments @(
            '-RepoRoot', $workDir,
            '-BaselinePath', $baselinePath,
            '-ReportPath', $reportPath,
            '-SkipRunner',
            '-FailOnRegression',
            '-Format', 'Json'
        )

        $res.ExitCode | Should -Be 0
        $parsed = $res.Output | ConvertFrom-Json
        $parsed.regression_count | Should -Be 0
        $parsed.missing_count | Should -Be 0
    }
}

Describe 'Equal-Work PowerShell Tokenizer Equivalence' -Tag 'Unit', 'BenchmarkGate', 'Portable' {
    BeforeAll {
        $initScript = Join-Path $script:RepoRoot 'Modules\PC-AI.Acceleration\Private\Initialize-PcaiNative.ps1'
        . $initScript
        function script:Test-PcaiNativeAvailable { return $false }
    }

    It 'matches exact Rust tokenizer.rs token counts across empty, ASCII, UTF-8, long words, and punctuation' {
        Invoke-PcaiNativeEstimateTokens -Text '' | Should -Be ([uint64]0)
        Invoke-PcaiNativeEstimateTokens -Text 'Hello world' | Should -Be ([uint64]4)
        Invoke-PcaiNativeEstimateTokens -Text 'This is a test.' | Should -Be ([uint64]9)
        Invoke-PcaiNativeEstimateTokens -Text '你好世界' | Should -Be ([uint64]4)
        Invoke-PcaiNativeEstimateTokens -Text 'internationalization' | Should -Be ([uint64]6)
        Invoke-PcaiNativeEstimateTokens -Text 'abcdefghijklm' | Should -Be ([uint64]5)
        Invoke-PcaiNativeEstimateTokens -Text 'a,b;c!' | Should -Be ([uint64]7)
    }

    It 'exposes cross-platform RID, library filename, and provenance fields in Get-PcaiNativeStatus' {
        $rid = Get-PcaiNativeRuntimeIdentifier
        $rid | Should -Match '^(win|linux|osx)-(x64|arm64|x86)$'

        $libFile = Get-PcaiNativeLibraryFileName -LibraryName 'pcai_core_lib'
        $libFile | Should -Match '^(pcai_core_lib\.dll|libpcai_core_lib\.(so|dylib))$'

        $status = Get-PcaiNativeStatus
        $status.RuntimeIdentifier | Should -Be $rid
        $status.PSObject.Properties.Name | Should -Contain 'CoreLibraryPath'
        $status.PSObject.Properties.Name | Should -Contain 'WrapperPath'
        $status.PSObject.Properties.Name | Should -Contain 'ArtifactHash'
    }
}

