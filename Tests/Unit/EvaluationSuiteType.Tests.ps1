#Requires -Version 7.0

param([string]$EvaluationModuleRoot)

BeforeAll {
    $script:EvaluationModuleRoot = if ($EvaluationModuleRoot) { $EvaluationModuleRoot } else {
        [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../../Modules/PC-AI.Evaluation'))
    }
    $script:EvaluationManifest = Join-Path $script:EvaluationModuleRoot 'PC-AI.Evaluation.psd1'
    $script:EvaluationScriptModule = Join-Path $script:EvaluationModuleRoot 'PC-AI.Evaluation.psm1'
    $script:EvaluationProbePwsh = (Get-Command pwsh -CommandType Application | Select-Object -First 1).Source
    $script:SuiteChild = Join-Path $TestDrive 'suite-consumer.ps1'
    [IO.File]::WriteAllText($script:SuiteChild, @'
param([string]$ModulePath, [int]$Reload)
$ErrorActionPreference = 'Stop'
Import-Module $ModulePath -Force -WarningAction SilentlyContinue
if ($Reload) { Import-Module $ModulePath -Force -WarningAction SilentlyContinue }
$suite = New-EvaluationSuite -Name 'fresh-consumer' -Metrics @('latency', 'throughput')
$cases = @(Get-EvaluationDataset -Name diagnostic)
if (-not $cases.Count) { throw 'Diagnostic dataset is empty' }
$suite.AddTestCase($cases[0])
$summary = Get-EvaluationResults -Suite $suite
$report = [ordered]@{
    Name = $suite.Name
    Type = $suite.GetType().Name
    MetricCount = $suite.Metrics.Count
    CaseCount = $suite.TestCases.Count
    TotalTests = $summary.TotalTests
}
Write-Output ('PCAI_EVAL_PROBE_JSON=' + ($report | ConvertTo-Json -Compress))
'@)

    $script:EndpointChild = Join-Path $TestDrive 'endpoint-consumer.ps1'
    [IO.File]::WriteAllText($script:EndpointChild, @'
param([string]$ModulePath, [string]$Endpoint, [string]$Backend)
$ErrorActionPreference = 'Stop'
$module = @(Import-Module $ModulePath -Force -PassThru -WarningAction SilentlyContinue |
    Where-Object { $_.Path -eq $ModulePath })
if ($module.Count -ne 1) { throw 'The exact evaluation script module was not selected.' }
$report = & $module[0] {
    param($Endpoint, $Backend)
    $script:EvaluationConfig.HttpBaseUrl = 'http://localhost:39991'
    $script:EvaluationConfig.OllamaBaseUrl = 'http://localhost:39992'
    $script:EndpointCalls = [Collections.Generic.List[string]]::new()
    function script:Get-Command {
        [CmdletBinding()]
        param([string]$Name)
        if ($Name -eq 'Send-OllamaRequest') { return $null }
        Microsoft.PowerShell.Core\Get-Command @PSBoundParameters
    }
    function script:Invoke-RestMethod {
        param([uri]$Uri, [string]$Method, [string]$Body, [string]$ContentType, [int]$TimeoutSec)
        $script:EndpointCalls.Add($Uri.OriginalString)
        if (-not $Uri.IsAbsoluteUri -or $Uri.Scheme -notin @('http', 'https')) {
            throw 'The request did not reach the transport as an absolute HTTP URI.'
        }
        if ($Endpoint -eq 'chat fallback' -and $Uri.AbsolutePath -eq '/v1/completions') {
            return @{ choices = @(@{ text = ''; message = $null }) }
        }
        return @{ choices = @(@{ text = 'transport-response'; message = @{ content = 'transport-response' } }); response = 'transport-response' }
    }
    $case = New-EvaluationTestCase -Id 'endpoint-contract' -Prompt 'hello'
    $result = Invoke-SingleTestCase -TestCase $case -Backend $Backend -Model 'fixture-model'
    [ordered]@{
        ModulePath = $ExecutionContext.SessionState.Module.Path
        ResultType = $result.GetType().Name
        ErrorMessage = $result.ErrorMessage
        Response = $result.Response
        Uris = @($script:EndpointCalls)
    }
} $Endpoint $Backend
Write-Output ('PCAI_EVAL_PROBE_JSON=' + ($report | ConvertTo-Json -Compress))
'@)

    function Invoke-EvaluationConsumerProbe {
        param([string]$ScriptPath, [string[]]$Arguments)
        $start = [Diagnostics.ProcessStartInfo]::new()
        $start.FileName = $script:EvaluationProbePwsh
        $start.UseShellExecute = $false
        $start.RedirectStandardOutput = $true
        $start.RedirectStandardError = $true
        $start.WorkingDirectory = $TestDrive
        $start.Environment['PCAI_DISABLE_NATIVE'] = '1'
        $start.Environment['PCAI_CACHE_PROVIDER'] = 'memory'
        foreach ($argument in @('-NoLogo', '-NoProfile', '-File', $ScriptPath) + $Arguments) {
            [void]$start.ArgumentList.Add($argument)
        }
        $process = [Diagnostics.Process]::new()
        $process.StartInfo = $start
        try {
            if (-not $process.Start()) { throw 'Evaluation consumer did not start.' }
            $stdout = $process.StandardOutput.ReadToEndAsync()
            $stderr = $process.StandardError.ReadToEndAsync()
            if (-not $process.WaitForExit(30000)) {
                $process.Kill($true)
                [void]$process.WaitForExit(5000)
                throw 'Evaluation consumer exceeded 30 seconds.'
            }
            if (-not [Threading.Tasks.Task]::WaitAll([Threading.Tasks.Task[]]@($stdout, $stderr), 5000)) {
                throw 'Evaluation consumer output did not drain within five seconds.'
            }
            $errorText = $stderr.GetAwaiter().GetResult()
            $process.ExitCode | Should -Be 0 -Because $errorText
            $json = @($stdout.GetAwaiter().GetResult() -split '\r?\n' |
                    Where-Object { $_.StartsWith('PCAI_EVAL_PROBE_JSON=') })
            $json.Count | Should -Be 1
            return ($json[0].Substring('PCAI_EVAL_PROBE_JSON='.Length) | ConvertFrom-Json)
        }
        finally { $process.Dispose() }
    }
}

Describe 'Evaluation suite consumer type binding' -Tag 'Unit', 'Portable' {
    # Protects: Public suite construction and typed dataset/result calls work in a fresh consumer process.
    # Detects: OutputType metadata cannot resolve a class defined inside the importing script module.
    # Needs: PowerShell 7, Pester 5 and TestDrive; real manifest, classes and dataset, no inference or network.
    # Breadcrumb: Modules/PC-AI.Evaluation/Public/New-EvaluationSuite.ps1 OutputType metadata.
    It 'constructs and consumes a real suite after <ImportMode>' -TestCases @(
        @{ ImportMode = 'first import'; Reload = 0 }
        @{ ImportMode = 'forced reload'; Reload = 1 }
    ) {
        param($ImportMode, $Reload)
        $result = Invoke-EvaluationConsumerProbe -ScriptPath $script:SuiteChild -Arguments @(
            '-ModulePath', $script:EvaluationManifest, '-Reload', [string]$Reload
        )
        $result.Name | Should -Be 'fresh-consumer'
        $result.Type | Should -Be 'EvaluationSuite'
        $result.MetricCount | Should -Be 2
        $result.CaseCount | Should -Be 1
        $result.TotalTests | Should -Be 0
    }
}

Describe 'Evaluation endpoint construction' -Tag 'Unit', 'Portable' {
    # Protects: Completion, chat fallback and Ollama fallback use configured complete endpoint URLs.
    # Detects: Bare hashtable interpolation reaches the transport as a relative System.Collections.Hashtable URI.
    # Needs: PowerShell 7, Pester 5 and fresh children; exact real psm1/classes with only transport/discovery doubles.
    # Breadcrumb: Modules/PC-AI.Evaluation/Private/Invoke-SingleTestCase.ps1 request URI construction.
    It 'uses the configured base URL for <Endpoint>' -TestCases @(
        @{ Endpoint = 'completion'; Backend = 'http'; ExpectedUris = @('http://localhost:39991/v1/completions') }
        @{ Endpoint = 'chat fallback'; Backend = 'http'; ExpectedUris = @('http://localhost:39991/v1/completions', 'http://localhost:39991/v1/chat/completions') }
        @{ Endpoint = 'Ollama fallback'; Backend = 'ollama'; ExpectedUris = @('http://localhost:39992/api/generate') }
    ) {
        param($Endpoint, $Backend, $ExpectedUris)
        $result = Invoke-EvaluationConsumerProbe -ScriptPath $script:EndpointChild -Arguments @(
            '-ModulePath', $script:EvaluationScriptModule, '-Endpoint', $Endpoint, '-Backend', $Backend
        )
        $result.ModulePath | Should -Be $script:EvaluationScriptModule
        $result.ResultType | Should -Be 'EvaluationResult'
        $result.ErrorMessage | Should -BeNullOrEmpty
        $result.Response | Should -Be 'transport-response'
        @($result.Uris).Count | Should -Be $ExpectedUris.Count
        ($result.Uris -join '|') | Should -BeExactly ($ExpectedUris -join '|')
    }
}
