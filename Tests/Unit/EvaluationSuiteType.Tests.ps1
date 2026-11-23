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
        if ($Endpoint -eq 'empty completion') {
            return @{ choices = @(@{ text = ' '; message = @{ content = ' ' } }); response = ' ' }
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

    $script:AuthChild = Join-Path $TestDrive 'auth-consumer.ps1'
    [IO.File]::WriteAllText($script:AuthChild, @'
param([string]$ModulePath, [string]$Backend, [string]$HttpBase, [string]$ConfiguredBase, [string]$Token, [int]$Fallback)
$ErrorActionPreference = 'Stop'
if ($Token) { $env:PCAI_INFERENCE_TOKEN = $Token } else { Remove-Item Env:PCAI_INFERENCE_TOKEN -ErrorAction SilentlyContinue }
$module = @(Import-Module $ModulePath -Force -PassThru -WarningAction SilentlyContinue |
    Where-Object { $_.Path -eq $ModulePath })
if ($module.Count -ne 1) { throw 'The exact evaluation script module was not selected.' }
$report = & $module[0] {
    param($HttpBase, $ConfiguredBase, $Backend, $Fallback)
    $script:EvaluationConfig.HttpBaseUrl = $HttpBase
    $script:EvaluationConfig.PcaiInferenceUrl = $ConfiguredBase
    $script:EvaluationConfig.OllamaBaseUrl = 'http://localhost:39992'
    $script:AuthCalls = [Collections.Generic.List[object]]::new()
    function script:Get-Command {
        [CmdletBinding()]
        param([string]$Name)
        if ($Name -eq 'Send-OllamaRequest') { return $null }
        Microsoft.PowerShell.Core\Get-Command @PSBoundParameters
    }
    function script:Invoke-RestMethod {
        param([uri]$Uri, [string]$Method, [string]$Body, [string]$ContentType, [int]$TimeoutSec, [hashtable]$Headers)
        $script:AuthCalls.Add([ordered]@{
                Uri = $Uri.OriginalString
                Authorization = if ($Headers) { $Headers['Authorization'] } else { $null }
            })
        if ($Fallback -and $Uri.AbsolutePath -eq '/v1/completions') {
            return @{ choices = @(@{ text = ''; message = $null }) }
        }
        return @{ choices = @(@{ text = 'transport-response'; message = @{ content = 'transport-response' } }); response = 'transport-response' }
    }
    $case = New-EvaluationTestCase -Id 'auth-contract' -Prompt 'hello'
    $result = Invoke-SingleTestCase -TestCase $case -Backend $Backend -Model 'fixture-model'
    [ordered]@{ ErrorMessage = $result.ErrorMessage; Calls = @($script:AuthCalls) }
} $HttpBase $ConfiguredBase $Backend $Fallback
Write-Output ('PCAI_EVAL_PROBE_JSON=' + ($report | ConvertTo-Json -Compress -Depth 5))
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

    # Protects: acceptance requires actual generated text across backends.
    # Detects: accepting HTTP or Ollama responses carrying only whitespace.
    # Needs: PowerShell 7 and a fresh module consumer with local transport mocks.
    # Breadcrumb: Invoke-SingleTestCase.ps1 completion and chat fallback.
    It 'records a case error for empty <Backend> output' -TestCases @(
        @{ Backend = 'http'; ExpectedUris = 'http://localhost:39991/v1/completions|http://localhost:39991/v1/chat/completions' }
        @{ Backend = 'ollama'; ExpectedUris = 'http://localhost:39992/api/generate' }
    ) {
        param($Backend, $ExpectedUris)
        $result = Invoke-EvaluationConsumerProbe -ScriptPath $script:EndpointChild -Arguments @(
            '-ModulePath', $script:EvaluationScriptModule, '-Endpoint', 'empty completion', '-Backend', $Backend
        )
        $result.Response | Should -BeNullOrEmpty
        $result.ErrorMessage | Should -BeExactly "Backend '$Backend' returned an empty completion."
        ($result.Uris -join '|') | Should -BeExactly $ExpectedUris
    }
}

Describe 'Evaluation requests to a token-protected pcai-inference server' -Tag 'Unit', 'Portable' {
    # Protects: evaluation runs against a server that has a bearer token (required for any non-loopback bind).
    # The harness probes the open /health, then POSTs to the protected /v1/* routes; without the header every
    # case would get 401. Also protects the credential: the token must reach only the pcai-inference origin.
    # Detects: the header being dropped (401 on every case), or sent to every origin, such as an Ollama or
    # LM Studio endpoint that the 'http' backend was pointed at.
    # Needs: PowerShell 7, Pester 5 and fresh children; exact real psm1 with only the transport doubled.
    # Breadcrumb: Modules/PC-AI.Evaluation/Private/Invoke-SingleTestCase.ps1, PC-AI.Common Get-PcaiOriginBearerHeader.
    It '<Name>' -TestCases @(
        @{ Name = 'sends the token to the configured pcai-inference origin'; Backend = 'http'; HttpBase = 'http://localhost:39991'; Configured = 'http://localhost:39991'; Token = 'eval-token'; Fallback = 0
            Expected = @('Bearer eval-token') }
        @{ Name = 'sends it on the chat fallback request as well'; Backend = 'http'; HttpBase = 'http://localhost:39991'; Configured = 'http://localhost:39991'; Token = 'eval-token'; Fallback = 1
            Expected = @('Bearer eval-token', 'Bearer eval-token') }
        @{ Name = 'does not send it when the http backend targets another origin (an ollama-style URL)'; Backend = 'http'; HttpBase = 'http://localhost:39992'; Configured = 'http://localhost:39991'; Token = 'eval-token'; Fallback = 0
            Expected = @($null) }
        @{ Name = 'does not send it when only the port differs from the configured origin'; Backend = 'http'; HttpBase = 'http://localhost:39990'; Configured = 'http://localhost:39991'; Token = 'eval-token'; Fallback = 1
            Expected = @($null, $null) }
        @{ Name = 'sends it to a compiled backend the harness started'; Backend = 'llamacpp-bin'; HttpBase = 'http://localhost:39995'; Configured = 'http://localhost:39991'; Token = 'eval-token'; Fallback = 0
            Expected = @('Bearer eval-token') }
        @{ Name = 'never sends it on the ollama backend'; Backend = 'ollama'; HttpBase = 'http://localhost:39991'; Configured = 'http://localhost:39991'; Token = 'eval-token'; Fallback = 0
            Expected = @($null) }
        @{ Name = 'sends nothing when no token is configured'; Backend = 'http'; HttpBase = 'http://localhost:39991'; Configured = 'http://localhost:39991'; Token = ''; Fallback = 1
            Expected = @($null, $null) }
    ) {
        param($Name, $Backend, $HttpBase, $Configured, $Token, $Fallback, $Expected)
        $result = Invoke-EvaluationConsumerProbe -ScriptPath $script:AuthChild -Arguments @(
            '-ModulePath', $script:EvaluationScriptModule, '-Backend', $Backend, '-HttpBase', $HttpBase,
            '-ConfiguredBase', $Configured, '-Token', $Token, '-Fallback', [string]$Fallback
        )
        $result.ErrorMessage | Should -BeNullOrEmpty
        $seen = @($result.Calls | ForEach-Object { $_.Authorization })
        $seen.Count | Should -Be $Expected.Count
        for ($i = 0; $i -lt $Expected.Count; $i++) { $seen[$i] | Should -Be $Expected[$i] }
    }
}
