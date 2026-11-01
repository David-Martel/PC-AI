#Requires -Version 7.0

BeforeAll {
    $script:AdmissionModule=Import-Module (Join-Path $PSScriptRoot '../../Modules/PC-AI.Evaluation/PC-AI.Evaluation.psd1') -Force -PassThru | Where-Object Name -eq PC-AI.Evaluation
    $script:SavedAdmissionCase=$env:PCAI_BASELINE_ADMISSION_CASE
    function New-AdmissionSuite {
        param([double]$Latency=10.0)
        $suite=New-EvaluationSuite -Name 'admission-fixture' -Metrics @('latency','throughput')
        foreach($value in @(1,2)) {
            $result=[Activator]::CreateInstance($suite.Results.GetType().GetGenericArguments()[0])
            $result.TestCaseId="case-$value";$result.Status='pass';$result.OverallScore=0.9
            $result.Duration=[TimeSpan]::FromMilliseconds(10)
            $result.Metrics=@{latency=$Latency;throughput=5.0}
            $suite.Results.Add($result)
        }
        return $suite
    }
    function Save-AdmissionReference {
        param([object]$Mean=10.0,[switch]$Missing)
        [void][IO.Directory]::CreateDirectory((Join-Path $TestDrive 'baselines'))
        $metrics=if($Missing){@{unrelated=@{Mean=10.0}}}else{@{latency=@{Mean=$Mean};throughput=@{Mean=5.0}}}
        @{Timestamp='2026-01-01T00:00:00Z';Metrics=$metrics}|ConvertTo-Json -Depth 5|Set-Content -LiteralPath (Join-Path $TestDrive 'baselines/reference.json')
    }
}
AfterAll {
    $env:PCAI_BASELINE_ADMISSION_CASE=$script:SavedAdmissionCase
    Remove-Module PC-AI.Evaluation -Force -ErrorAction SilentlyContinue
}

Describe 'Baselines refuse unqualified completed-run evidence before persistence' {
    BeforeEach {
        & $script:AdmissionModule {param($root) $script:EvaluationConfig.BaselinePath=$root;$script:Baselines=@{}} (Join-Path $TestDrive 'baselines')
        $env:PCAI_BASELINE_ADMISSION_CASE='valid'
        Mock Invoke-EvaluationSuite -ModuleName PC-AI.Evaluation {
            param($Suite)
            $case=$env:PCAI_BASELINE_ADMISSION_CASE
            if($case -eq 'empty'){$Suite.Results.Clear()}
            foreach($result in $Suite.Results){
                if($case -eq 'error'){$result.Status='error';$result.ErrorMessage='synthetic backend failure'}
                if($case -eq 'nonfinite'){$result.Metrics.latency=[double]::NaN}
                if($case -eq 'missing'){$result.Metrics.Remove('throughput')}
            }
            $summary=$Suite.GetSummary();$summary.Cancelled=$case -eq 'cancelled';return $summary
        }
    }
    It 'refuses <Case> evidence and leaves disk and in-memory baseline stores empty' -TestCases @(
        @{Case='empty'},@{Case='cancelled'},@{Case='error'},@{Case='nonfinite'},@{Case='missing'}
    ) {
        param($Case)
        $env:PCAI_BASELINE_ADMISSION_CASE=$Case
        $suite=New-AdmissionSuite
        {New-BaselineSnapshot -Name rejected -Suite $suite}|Should -Throw
        Test-Path -LiteralPath (Join-Path $TestDrive 'baselines/rejected.json')|Should -BeFalse
        (& $script:AdmissionModule {$script:Baselines.Count})|Should -Be 0
    }
    It 'rejects a parent-traversal name before invoking the inference boundary' {
        $suite=New-AdmissionSuite
        {New-BaselineSnapshot -Name '../escaped' -Suite $suite}|Should -Throw
        Should -Invoke Invoke-EvaluationSuite -ModuleName PC-AI.Evaluation -Times 0 -Exactly
        Test-Path -LiteralPath (Join-Path $TestDrive 'escaped.json')|Should -BeFalse
    }
    It 'retains a completed finite low-scoring run as valid reference evidence' {
        $suite=New-AdmissionSuite
        foreach($result in $suite.Results){$result.Status='fail';$result.OverallScore=0.2}
        $baseline=New-BaselineSnapshot -Name measured-reference -Suite $suite
        $baseline.Metrics.latency.Mean|Should -Be 10.0
        $baseline.Summary.Failed|Should -Be 2
        $baseline.TestCount|Should -Be 2
        (Get-Content (Join-Path $TestDrive 'baselines/measured-reference.json') -Raw|ConvertFrom-Json).Metrics.throughput.Mean|Should -Be 5.0
    }
    It 'publishes a bracketed filename literally and preserves the matching foreign leaf' {
        [void][IO.Directory]::CreateDirectory((Join-Path $TestDrive 'baselines'))
        $foreign=Join-Path $TestDrive 'baselines/latencyr.json'
        [IO.File]::WriteAllText($foreign,'foreign-reference')
        $suite=New-AdmissionSuite
        New-BaselineSnapshot -Name 'latency[r]' -Suite $suite | Out-Null
        [IO.File]::ReadAllText($foreign)|Should -BeExactly 'foreign-reference'
        Test-Path -LiteralPath (Join-Path $TestDrive 'baselines/latency[r].json')|Should -BeTrue
    }
}

Describe 'Regression comparisons refuse absent or numerically undefined evidence' {
    BeforeEach {& $script:AdmissionModule {param($root) $script:EvaluationConfig.BaselinePath=$root} (Join-Path $TestDrive 'baselines')}
    It 'rejects threshold <Case> before comparison' -TestCases @(
        @{Case='negative';Value=-0.1},@{Case='NaN';Value=[double]::NaN},@{Case='positive infinity';Value=[double]::PositiveInfinity}
    ) {
        param($Value)
        Save-AdmissionReference
        $suite=New-AdmissionSuite
        {Test-ForRegression -BaselineName reference -Suite $suite -Threshold $Value}|Should -Throw
    }
    It 'refuses zero baseline means rather than reporting infinite percent or false success' -TestCases @(@{Value=0.0},@{Value=1.0}) {
        param($Value)
        Save-AdmissionReference -Mean 0.0
        $suite=New-AdmissionSuite -Latency $Value
        {Test-ForRegression -BaselineName reference -Suite $suite}|Should -Throw
    }
    It 'refuses a baseline with no matching current metrics' {
        Save-AdmissionReference -Missing
        $suite=New-AdmissionSuite
        {Test-ForRegression -BaselineName reference -Suite $suite}|Should -Throw
    }
    It 'refuses a nonfinite persisted mean' {
        Save-AdmissionReference -Mean ([double]::PositiveInfinity)
        $suite=New-AdmissionSuite
        {Test-ForRegression -BaselineName reference -Suite $suite}|Should -Throw
    }
    It 'refuses a nonfinite current metric' {
        Save-AdmissionReference
        $suite=New-AdmissionSuite -Latency ([double]::NaN)
        {Test-ForRegression -BaselineName reference -Suite $suite}|Should -Throw
    }
    It 'refuses a nonfinite current score even when every metric is finite' {
        Save-AdmissionReference
        $suite=New-AdmissionSuite;$suite.Results[0].OverallScore=[double]::NaN
        {Test-ForRegression -BaselineName reference -Suite $suite}|Should -Throw
    }
    It 'loads the literal bracketed reference rather than a matching foreign leaf' {
        Save-AdmissionReference
        $root=Join-Path $TestDrive 'baselines'
        [IO.File]::Copy((Join-Path $root 'reference.json'),(Join-Path $root 'latency[r].json'),$false)
        [IO.File]::WriteAllText((Join-Path $root 'latencyr.json'),'foreign-reference')
        $suite=New-AdmissionSuite -Latency 5.0
        $comparison=Test-ForRegression -BaselineName 'latency[r]' -Suite $suite
        $comparison.HasRegressions|Should -BeFalse
        $comparison.Improvements[0].Change|Should -Be ([double]-50.0)
    }
    It 'refuses empty current results' {
        Save-AdmissionReference
        $suite=New-AdmissionSuite;$suite.Results.Clear()
        {Test-ForRegression -BaselineName reference -Suite $suite}|Should -Throw
    }
    It 'retains zero threshold and reports an actual finite latency improvement' {
        Save-AdmissionReference
        $suite=New-AdmissionSuite -Latency 5.0
        $comparison=Test-ForRegression -BaselineName reference -Suite $suite -Threshold 0.0
        $comparison.HasRegressions|Should -BeFalse
        $comparison.Improvements.Count|Should -Be 1
        $comparison.Improvements[0].Change|Should -Be ([double]-50.0)
    }
}
