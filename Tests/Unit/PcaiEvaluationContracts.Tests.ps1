#Requires -Version 7.0

BeforeAll {
    $script:manifestPath=Join-Path $PSScriptRoot '../../Modules/PC-AI.Evaluation/PC-AI.Evaluation.psd1'
    Import-Module $manifestPath -Force -WarningAction SilentlyContinue
    $script:module=Get-Module -Name PC-AI.Evaluation
    # Pester exposes canonical parameter names, including aliases normalized by the host runtime.
    $script:httpTimeoutParameter=(Get-Command Invoke-RestMethod).Parameters.Values|
        Where-Object { $_.Name -eq 'TimeoutSec' -or $_.Aliases -contains 'TimeoutSec' }|
        Select-Object -ExpandProperty Name
    $script:hasOperationTimeout=(Get-Command Invoke-RestMethod -CommandType Cmdlet).Parameters.ContainsKey('OperationTimeoutSeconds')
    function New-ContractSuite {
        param([int]$Count=1)
        $suite=New-EvaluationSuite -Name 'contract-suite' -Metrics @('accuracy')
        $suite.Metrics[0].Calculator={param($result) if($result.Response -eq 'expected'){1.0}else{0.0}}
        foreach($index in 1..$Count){$suite.AddTestCase((New-EvaluationTestCase -Id "case-$index" -Prompt "prompt-$index" -ExpectedOutput 'expected'))}
        return $suite
    }
    function New-ContractResult {
        param($Suite,[string]$Id,[string]$Status,[double]$Score,[double]$Metric=0)
        $result=[Activator]::CreateInstance($Suite.Results.GetType().GetGenericArguments()[0])
        $result.TestCaseId=$Id;$result.Status=$Status;$result.OverallScore=$Score
        $result.Response='expected';$result.Duration=[timespan]::FromMilliseconds(12)
        $result.Metrics=@{accuracy=$Metric}
        if($Status -eq 'error'){$result.ErrorMessage='fixture provider failure'}
        $Suite.Results.Add($result)
    }
}
AfterAll { Remove-Module PC-AI.Evaluation -Force -ErrorAction SilentlyContinue }

Describe 'Evaluation public data and results contracts' {
    It 'makes every advertised manifest command callable from the actual module' {
        $manifest=Import-PowerShellDataFile $manifestPath
        $manifest.FunctionsToExport.Count|Should -BeGreaterThan 0
        foreach($name in $manifest.FunctionsToExport){(Get-Command -Name $name -Module PC-AI.Evaluation -ErrorAction Stop).ModuleName|Should -BeExactly 'PC-AI.Evaluation'}
    }
    It 'retains explicit test identity prompt context and tags' {
        $test=New-EvaluationTestCase -Id 'fixture-id' -Prompt 'literal prompt' -Category 'fixture' -ExpectedOutput 'expected' -Context @{context='source'} -Tags @('one','two')
        $test.Id|Should -BeExactly 'fixture-id';$test.Prompt|Should -BeExactly 'literal prompt'
        $test.Context.context|Should -BeExactly 'source';$test.Tags|Should -Be @('one','two')
    }
    It 'includes each requested/default metric once' {
        $suite=New-EvaluationSuite -Name 'defaults' -Metrics @('latency','accuracy') -IncludeDefaultMetrics
        $suite.Metrics.Name|Should -Be @('latency','accuracy','throughput','memory')
    }
    It 'rejects whitespace-only <Field> before producing a usable test' -ForEach @(@{Field='Id'},@{Field='Prompt'}) {
        $parameters=@{Id='valid';Prompt='valid'};$parameters[$Field]='   '
        {New-EvaluationTestCase @parameters -ErrorAction Stop}|Should -Throw
    }
    It 'round-trips multiple synthetic dataset cases with nested context and tags' {
        $file=Join-Path $TestDrive 'dataset.json'
        $cases=@(New-EvaluationTestCase -Id first -Prompt 'prompt one' -Category fixture -ExpectedOutput expected -Context @{context='grounded';nested=@{flag=$true}} -Tags @('a','b');New-EvaluationTestCase -Id second -Prompt 'prompt two')
        Export-EvaluationDataset -TestCases $cases -Path $file
        $read=@(Import-EvaluationDataset -Path $file)
        $read.Count|Should -Be 2;$read.Id|Should -Be @('first','second')
        $read[0].Context.context|Should -BeExactly 'grounded';$read[0].Context.nested.flag|Should -BeTrue
        $read[0].Tags|Should -Be @('a','b');$read[1].Prompt|Should -BeExactly 'prompt two'
    }
    It 'refuses syntactically invalid dataset JSON' {
        $file=Join-Path $TestDrive 'invalid.json';[IO.File]::WriteAllText($file,'{broken-json')
        {Import-EvaluationDataset -Path $file -ErrorAction Stop}|Should -Throw
    }
    It 'refuses a dataset record without its required prompt' {
        $file=Join-Path $TestDrive 'missing-prompt.json'
        [IO.File]::WriteAllText($file,'[{"id":"fixture","category":"fixture","expected":"expected","context":{},"tags":[]}]')
        {Import-EvaluationDataset -Path $file -ErrorAction Stop}|Should -Throw
    }
    It 'emits no partial dataset when a later record is malformed' {
        $file=Join-Path $TestDrive 'partial-dataset.json'
        [IO.File]::WriteAllText($file,'[{"id":"valid","prompt":"valid"},{"id":"bad","prompt":"   "}]')
        $observed=[Collections.Generic.List[object]]::new()
        {Import-EvaluationDataset -Path $file -ErrorAction Stop|ForEach-Object{$observed.Add($_)}}|Should -Throw
        $observed.Count|Should -Be 0
    }
    It 'reports an unknown dataset instead of producing empty successful cases' {
        {Get-EvaluationDataset -Name (Join-Path $TestDrive 'missing.json') -ErrorAction Stop}|Should -Throw '*Dataset not found*'
    }
    It 'aggregates pass fail and provider errors into an exact summary' {
        $suite=New-ContractSuite
        New-ContractResult $suite 'passed' pass 1 1
        New-ContractResult $suite 'failed' fail 0 0
        New-ContractResult $suite 'error' error 0 0
        $summary=Get-EvaluationResults -Suite $suite
        $summary.TotalTests|Should -Be 3;$summary.Passed|Should -Be 1;$summary.Failed|Should -Be 1;$summary.Errors|Should -Be 1
        $summary.PassRate|Should -Be 33.33;$summary.AverageScore|Should -Be 0.3333;$summary.AverageLatency|Should -Be 12
    }
    It 'returns failing/error IDs and excludes successful results' {
        $suite=New-ContractSuite
        New-ContractResult $suite 'passed' pass 1 1;New-ContractResult $suite 'failed' fail 0 0;New-ContractResult $suite 'error' error 0 0
        $failures=@(Get-EvaluationResults -Suite $suite -Format failures)
        $failures.TestId|Should -Be @('failed','error');$failures[1].Error|Should -BeExactly 'fixture provider failure'
    }
    It 'aggregates observed numeric metrics while excluding missing metric values' {
        $suite=New-ContractSuite
        New-ContractResult $suite one pass 1 1;New-ContractResult $suite two pass 1 3
        $empty=[Activator]::CreateInstance($suite.Results.GetType().GetGenericArguments()[0]);$empty.Metrics=@{};$suite.Results.Add($empty)
        $stats=(Get-EvaluationResults -Suite $suite -Format metrics).accuracy
        $stats.Mean|Should -Be 2;$stats.Min|Should -Be 1;$stats.Max|Should -Be 3;$stats.StdDev|Should -Be 1.4142
    }
    It 'returns detailed response previews bounded to the documented output field' {
        $suite=New-ContractSuite;New-ContractResult $suite one pass 1 1
        $suite.Results[0].Response='x'*150
        $detail=Get-EvaluationResults -Suite $suite -Format detailed
        $detail.TestId|Should -BeExactly 'one';$detail.Response.Length|Should -Be 103;$detail.Response.Substring(0,100)|Should -BeExactly ('x'*100)
    }
}

Describe 'Evaluation orchestration with actual synthetic files and mocked HTTP providers' {
    BeforeEach {
        $script:output=Join-Path $TestDrive ([guid]::NewGuid().ToString('N'))
        $script:requests=[Collections.Generic.List[object]]::new()
        & $module {
            $script:EvaluationConfig.ProgressLogPath=$null;$script:EvaluationConfig.EventsLogPath=$null;$script:EvaluationConfig.StopSignalPath=$null
            $script:EvaluationConfig.EmitStructuredMessages=$false;$script:EvaluationConfig.ProgressMode='silent'
            $script:CompiledServerProcess=$null;$script:CompiledServerConfigPath=$null
            Remove-Item Function:Send-OllamaRequest -ErrorAction SilentlyContinue
        }
        Mock -ModuleName PC-AI.Evaluation Get-PcaiArtifactsRoot {$script:output}
        $script:context=New-PcaiEvaluationRunContext -OutputRoot $output -RunLabel 'contract' -SuiteName 'contract-suite' -Backend http
        Mock -ModuleName PC-AI.Evaluation Invoke-RestMethod {
            if($Method -eq 'Get'){return [pscustomobject]@{status='ok'}}
            $script:requests.Add([pscustomobject]@{Uri=$Uri;Timeout=(Get-Variable -Name $script:httpTimeoutParameter -ValueOnly -ErrorAction Stop);OperationTimeout=if($script:hasOperationTimeout){Get-Variable -Name OperationTimeoutSeconds -ValueOnly -ErrorAction SilentlyContinue}else{$null};Body=($Body|ConvertFrom-Json -AsHashtable)})
            return [pscustomobject]@{choices=@([pscustomobject]@{text='expected'})}
        }
    }
    It 'forwards selected endpoint prompt limits timeout and model into HTTP requests' {
        $suite=New-ContractSuite
        $summary=Invoke-EvaluationSuite -Suite $suite -Backend http -BaseUrl 'http://fixture.invalid:48080' -Model 'selected-model' -MaxTokens 37 -Temperature 0.25 -RequestTimeoutSec 23 -RunContext $context -ProgressMode silent
        $summary.Passed|Should -Be 1;$requests.Count|Should -Be 1
        $requests[0].Uri|Should -BeExactly 'http://fixture.invalid:48080/v1/completions'
        $requests[0].Timeout|Should -Be 23;$requests[0].Body.prompt|Should -BeExactly 'prompt-1'
        if($hasOperationTimeout){$requests[0].OperationTimeout|Should -Be 23}
        $requests[0].Body.max_tokens|Should -Be 37;$requests[0].Body.temperature|Should -Be 0.25
        $requests[0].Body.model|Should -BeExactly 'selected-model'
    }
    It 'accepts a chat-shaped first response without making a redundant request' {
        Mock -ModuleName PC-AI.Evaluation Invoke-RestMethod {if($Method -eq 'Get'){return @{status='ok'}};return @{choices=@(@{text='';message=@{content='expected'}})}}
        $suite=New-ContractSuite
        (Invoke-EvaluationSuite -Suite $suite -Backend http -BaseUrl 'http://fixture.invalid' -RunContext $context -ProgressMode silent).Passed|Should -Be 1
        $suite.Results[0].Response|Should -BeExactly 'expected'
        Should -Invoke -ModuleName PC-AI.Evaluation Invoke-RestMethod -Times 1 -Exactly -ParameterFilter {$Method -eq 'Post'}
    }
    It 'sends the explicitly selected model in the completion request body' {
        $suite=New-ContractSuite
        $null=Invoke-EvaluationSuite -Suite $suite -Backend http -BaseUrl 'http://fixture.invalid' -Model 'body-selected-model' -RunContext $context -ProgressMode silent
        $requests.Count|Should -Be 1
        $requests[0].Body.model|Should -BeExactly 'body-selected-model'
    }
    It 'omits unsupported operation timeout parameters on older cmdlet contracts' {
        Mock -ModuleName PC-AI.Evaluation Get-Command { [pscustomobject]@{Parameters=@{TimeoutSec=$null}} } -ParameterFilter {$Name -eq 'Invoke-RestMethod' -and $CommandType -eq 'Cmdlet'}
        $suite=New-ContractSuite
        (Invoke-EvaluationSuite -Suite $suite -Backend http -BaseUrl 'http://fixture.invalid' -RequestTimeoutSec 11 -RunContext $context -ProgressMode silent).Passed|Should -Be 1
        $requests[0].Timeout|Should -Be 11;$requests[0].OperationTimeout|Should -BeNullOrEmpty
        Should -Invoke -ModuleName PC-AI.Evaluation Get-Command -Exactly -Times 1 -ParameterFilter {$Name -eq 'Invoke-RestMethod' -and $CommandType -eq 'Cmdlet'}
    }
    It 'uses the chat endpoint after a valid completion has no usable text' {
        Mock -ModuleName PC-AI.Evaluation Invoke-RestMethod {
            if($Method -eq 'Get'){return @{status='ok'}}
            if($Uri -like '*/v1/completions'){return @{choices=@(@{text='';message=$null})}}
            return @{choices=@(@{message=@{content='expected'}})}
        }
        $suite=New-ContractSuite
        (Invoke-EvaluationSuite -Suite $suite -Backend http -BaseUrl 'http://fixture.invalid' -Model 'chat-selected-model' -RunContext $context -ProgressMode silent).Passed|Should -Be 1
        Should -Invoke -ModuleName PC-AI.Evaluation Invoke-RestMethod -Times 1 -Exactly -ParameterFilter {$Uri -eq 'http://fixture.invalid/v1/chat/completions' -and ($Body|ConvertFrom-Json).messages[0].content -eq 'prompt-1' -and ($Body|ConvertFrom-Json).model -eq 'chat-selected-model'}
    }
    It 'records a provider error and still evaluates the following case' {
        Mock -ModuleName PC-AI.Evaluation Invoke-RestMethod {
            if($Method -eq 'Get'){return @{status='ok'}}
            if(($Body|ConvertFrom-Json).prompt -eq 'prompt-1'){throw 'fixture provider failure'}
            return @{choices=@(@{text='expected'})}
        }
        $suite=New-ContractSuite 2
        $summary=Invoke-EvaluationSuite -Suite $suite -Backend http -BaseUrl 'http://fixture.invalid' -Model 'requested-model' -RunContext $context -ProgressMode silent
        $summary.TotalTests|Should -Be 2;$summary.Errors|Should -Be 1;$summary.Passed|Should -Be 1
        $suite.Results.TestCaseId|Should -Be @('case-1','case-2')
        $suite.Results[0].ErrorMessage|Should -BeExactly 'fixture provider failure'
        $suite.Results[0].Model|Should -BeExactly 'requested-model'
        (Get-EvaluationRunState).Completed|Should -Be 2
    }
    It 'rejects malformed provider choices instead of marking the case successful' {
        Mock -ModuleName PC-AI.Evaluation Invoke-RestMethod {if($Method -eq 'Get'){return @{status='ok'}};return @{unexpected='synthetic'}}
        $suite=New-ContractSuite
        $summary=Invoke-EvaluationSuite -Suite $suite -Backend http -BaseUrl 'http://fixture.invalid' -RunContext $context -ProgressMode silent
        $summary.Errors|Should -Be 1;$summary.Passed|Should -Be 0
        $suite.Results[0].Response|Should -BeNullOrEmpty;$suite.Results[0].ErrorMessage|Should -Not -BeNullOrEmpty
    }
    It 'logs failed initialization cleans up and makes no inference request' {
        Mock -ModuleName PC-AI.Evaluation Invoke-RestMethod {throw 'fixture endpoint unavailable'}
        $suite=New-ContractSuite
        (Invoke-EvaluationSuite -Suite $suite -Backend http -BaseUrl 'http://fixture.invalid' -RunContext $context -ProgressMode silent -WarningAction SilentlyContinue)|Should -BeNullOrEmpty
        $suite.Results.Count|Should -Be 0
        Should -Invoke -ModuleName PC-AI.Evaluation Invoke-RestMethod -Times 0 -Exactly -ParameterFilter {$Method -eq 'Post'}
        $events=@(Get-Content $context.EventsLogPath|ForEach-Object{$_|ConvertFrom-Json})
        $events.type|Should -Be @('start','backend_error','backend_stop')
    }
    It 'returns cancelled zero-completion summary when a real stop file already exists' {
        Stop-EvaluationRun -StopSignalPath $context.StopSignalPath|Should -BeTrue
        $suite=New-ContractSuite 2
        $summary=Invoke-EvaluationSuite -Suite $suite -Backend http -BaseUrl 'http://fixture.invalid' -RunContext $context -ProgressMode silent
        $summary.Cancelled|Should -BeTrue;$summary.TotalTests|Should -Be 0
        (Get-EvaluationRunState).Completed|Should -Be 0
        Should -Invoke -ModuleName PC-AI.Evaluation Invoke-RestMethod -Times 0 -Exactly -ParameterFilter {$Method -eq 'Post'}
        (Get-Content $context.SummaryPath -Raw|ConvertFrom-Json).Cancelled|Should -BeTrue
    }
    It 'honors a stop request after the first actual case and retains that result' {
        Mock -ModuleName PC-AI.Evaluation Invoke-RestMethod {
            if($Method -eq 'Get'){return @{status='ok'}}
            $null=Stop-EvaluationRun -StopSignalPath $script:context.StopSignalPath
            return @{choices=@(@{text='expected'})}
        }
        $suite=New-ContractSuite 2
        $summary=Invoke-EvaluationSuite -Suite $suite -Backend http -BaseUrl 'http://fixture.invalid' -RunContext $context -ProgressMode silent
        $summary.Cancelled|Should -BeTrue;$summary.TotalTests|Should -Be 1
        $suite.Results[0].TestCaseId|Should -BeExactly 'case-1';(Get-EvaluationRunState).Completed|Should -Be 1
        Should -Invoke -ModuleName PC-AI.Evaluation Invoke-RestMethod -Times 1 -Exactly -ParameterFilter {$Method -eq 'Post'}
    }
    It 'cannot produce a passing verdict when a requested metric calculator throws' {
        $suite=New-EvaluationSuite -Name 'metric-error' -Metrics @('latency','accuracy')
        $suite.Metrics[1].Calculator={throw 'fixture metric failure'}
        $suite.AddTestCase((New-EvaluationTestCase -Id fixture -Prompt prompt))
        $summary=Invoke-EvaluationSuite -Suite $suite -Backend http -BaseUrl 'http://fixture.invalid' -RunContext $context -ProgressMode silent -WarningAction SilentlyContinue
        $summary.Passed|Should -Be 0;$suite.Results[0].Status|Should -BeExactly 'error'
        $summary.Errors|Should -Be 1
        $suite.Results[0].ErrorMessage|Should -Match 'accuracy.*fixture metric failure'
        @(Get-Content $context.EventsLogPath|ForEach-Object{$_|ConvertFrom-Json}|Where-Object type -EQ metric_error).Count|Should -Be 1
    }
    It 'gives two contexts distinct output directories even under an identical clock label' {
        Mock -ModuleName PC-AI.Evaluation Get-Date {'contract-clock'} -ParameterFilter {$Format -eq 'yyyyMMdd_HHmmss'}
        $first=New-PcaiEvaluationRunContext -OutputRoot $output -RunLabel same
        $second=New-PcaiEvaluationRunContext -OutputRoot $output -RunLabel same
        $first.RunDir|Should -Not -BeExactly $second.RunDir
        $first.RunId|Should -Match '^same-[a-f0-9]{32}$'
        Test-Path -LiteralPath $first.RunDir -PathType Container|Should -BeTrue
        Test-Path -LiteralPath $second.RunDir -PathType Container|Should -BeTrue
        { [datetime]::Parse($first.CreatedUtc) }|Should -Not -Throw
    }
    It 'forwards explicit Ollama HTTP options and the selected model' {
        Mock -ModuleName PC-AI.Evaluation Get-Command {$null} -ParameterFilter {$Name -eq 'Send-OllamaRequest'}
        Mock -ModuleName PC-AI.Evaluation Invoke-RestMethod {
            if($Method -eq 'Get'){return @{models=@()}}
            $script:requests.Add([pscustomobject]@{Uri=$Uri;Timeout=(Get-Variable -Name $script:httpTimeoutParameter -ValueOnly -ErrorAction Stop);OperationTimeout=if($script:hasOperationTimeout){Get-Variable -Name OperationTimeoutSeconds -ValueOnly -ErrorAction SilentlyContinue}else{$null};Body=($Body|ConvertFrom-Json -AsHashtable)})
            return @{response='expected'}
        }
        $suite=New-ContractSuite
        $summary=Invoke-EvaluationSuite -Suite $suite -Backend ollama -BaseUrl 'http://fixture.invalid:49000' -Model 'selected-ollama' -NumCtx 4096 -NumThread 3 -TopP 0.8 -Seed 17 -RequestTimeoutSec 19 -RunContext $context -ProgressMode silent
        $summary.Passed|Should -Be 1;$requests.Count|Should -Be 1
        $requests[0].Uri|Should -BeExactly 'http://fixture.invalid:49000/api/generate'
        $requests[0].Timeout|Should -Be 19;$requests[0].Body.model|Should -BeExactly 'selected-ollama'
        if($hasOperationTimeout){$requests[0].OperationTimeout|Should -Be 19}
        $requests[0].Body.options.num_ctx|Should -Be 4096;$requests[0].Body.options.num_thread|Should -Be 3
        $requests[0].Body.options.top_p|Should -Be 0.8;$requests[0].Body.options.seed|Should -Be 17
    }
    Context 'Optional Ollama consumer command boundary' {
        BeforeEach {
            & $module {
                function script:Send-OllamaRequest {
                    param($Prompt,$Model,$MaxTokens,$Temperature,$TimeoutSeconds,$MaxRetries,$NumCtx,$NumThread,$TopP,$TopK,$RepeatLastN,$RepeatPenalty,$TfsZ,$Seed)
                    $script:observedOllama=@{}+$PSBoundParameters
                    return @{Response='expected';Model=$Model}
                }
            }
        }
        AfterEach { & $module {Remove-Item Function:Send-OllamaRequest -ErrorAction SilentlyContinue} }
        It 'forwards selected model timeout and explicit sampling through the installed-command path' {
            $suite=New-ContractSuite
            $summary=Invoke-EvaluationSuite -Suite $suite -Backend ollama -BaseUrl 'http://fixture.invalid' -Model 'helper-model' -NumCtx 2048 -TopP 0.9 -RequestTimeoutSec 13 -RunContext $context -ProgressMode silent
            $summary.Passed|Should -Be 1
            $observed=& $module {$script:observedOllama}
            $observed.Prompt|Should -BeExactly 'prompt-1';$observed.Model|Should -BeExactly 'helper-model'
            $observed.TimeoutSeconds|Should -Be 13;$observed.MaxRetries|Should -Be 1
            $observed.NumCtx|Should -Be 2048;$observed.TopP|Should -Be 0.9
            $suite.Results[0].Model|Should -BeExactly 'helper-model'
        }
        It 'preserves provider defaults rather than forwarding unrequested zero options' {
            $suite=New-ContractSuite
            $null=Invoke-EvaluationSuite -Suite $suite -Backend ollama -BaseUrl 'http://fixture.invalid' -Model 'helper-model' -RunContext $context -ProgressMode silent
            $observed=& $module {$script:observedOllama}
            foreach($name in @('NumCtx','NumThread','TopP','TopK','RepeatLastN','RepeatPenalty','TfsZ','Seed')){$observed.ContainsKey($name)|Should -BeFalse}
        }
        It 'retains explicitly requested zero sampling values' {
            $suite=New-ContractSuite
            $null=Invoke-EvaluationSuite -Suite $suite -Backend ollama -BaseUrl 'http://fixture.invalid' -Model 'helper-model' -TopK 0 -Seed 0 -RunContext $context -ProgressMode silent
            $observed=& $module {$script:observedOllama}
            $observed.ContainsKey('TopK')|Should -BeTrue;$observed.TopK|Should -Be 0
            $observed.ContainsKey('Seed')|Should -BeTrue;$observed.Seed|Should -Be 0
            $observed.ContainsKey('NumCtx')|Should -BeFalse
        }
        It 'retains the actual model reported by the provider over the requested fallback' {
            & $module { function script:Send-OllamaRequest { param($Prompt,$Model,$MaxTokens,$Temperature,$TimeoutSeconds,$MaxRetries) return @{Response='expected';Model='provider-actual-model'} } }
            $suite=New-ContractSuite
            $null=Invoke-EvaluationSuite -Suite $suite -Backend ollama -BaseUrl 'http://fixture.invalid' -Model 'requested-model' -RunContext $context -ProgressMode silent
            $suite.Results[0].Model|Should -BeExactly 'provider-actual-model'
        }
    }
}
