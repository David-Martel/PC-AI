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
    $script:repoRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
    $script:rootProbe=Join-Path $TestDrive 'evaluation-root-consumer.ps1'
    @'
param([string]$Installed,[string]$Root)
$ErrorActionPreference='Stop'
$env:PCAI_ROOT=if($Root){$Root}else{$null}
$env:PCAI_ARTIFACTS_ROOT=$null
$env:PCAI_CACHE_PROVIDER='memory'
Import-Module (Join-Path $Installed 'PC-AI.Evaluation/PC-AI.Evaluation.psd1') -Force -WarningAction SilentlyContinue
$module=Get-Module PC-AI.Evaluation
$result=& $module {
    [ordered]@{Root=Get-PcaiProjectRoot;Artifacts=Get-PcaiArtifactsRoot;OllamaModel=$script:EvaluationConfig.OllamaModel;OllamaUrl=$script:EvaluationConfig.OllamaBaseUrl}
}
'EVALUATION-ROOT:' + ($result|ConvertTo-Json -Compress)
'@|Set-Content -LiteralPath $script:rootProbe
    function New-EvaluationRootFixture {
        $root=Join-Path $TestDrive ([guid]::NewGuid().ToString('N'))
        $null=New-Item -ItemType Directory -Path (Join-Path $root 'Config')
        '# Synthetic genuine PC-AI configuration root'|Set-Content -LiteralPath (Join-Path $root 'PC-AI.ps1')
        '{"ollama":{"model":"evaluation-root-unique-model","base_url":"http://evaluation-root.invalid:19432"}}'|Set-Content -LiteralPath (Join-Path $root 'Config/llm-config.json')
        return (Get-Item -LiteralPath $root).FullName
    }
    function Copy-EvaluationConsumer {
        param([string]$Destination,[switch]$WithoutCommon)
        $null=New-Item -ItemType Directory -Path $Destination -Force
        Copy-Item -LiteralPath (Join-Path $repoRoot 'Modules/PC-AI.Evaluation') -Destination $Destination -Recurse
        if(-not $WithoutCommon){
            $common=Join-Path $Destination 'PC-AI.Common/Public'
            $null=New-Item -ItemType Directory -Path $common -Force
            Copy-Item -LiteralPath (Join-Path $repoRoot 'Modules/PC-AI.Common/Public/Get-PcaiRuntimeConfig.ps1') -Destination $common
        }
    }
    function Invoke-EvaluationRootProbe {
        param([string]$Installed,[string]$Root)
        $output=& (Get-Command pwsh).Source -NoLogo -NoProfile -File $script:rootProbe $Installed $Root 2>&1
        $exitCode=$LASTEXITCODE
        if($exitCode){$output|ForEach-Object{Write-Host $_}}
        $exitCode|Should -Be 0
        $record=@($output|Where-Object{$_ -is [string] -and $_.StartsWith('EVALUATION-ROOT:')})
        $record.Count|Should -Be 1
        return $record[0].Substring('EVALUATION-ROOT:'.Length)|ConvertFrom-Json
    }
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

Describe 'Evaluation root selection without a profile or provider calls' {
    BeforeEach {
        $script:environmentBefore=@{}
        foreach($name in @('PCAI_ROOT','PCAI_ARTIFACTS_ROOT')){
            $value=Get-Item -LiteralPath "Env:$name" -ErrorAction SilentlyContinue
            $script:environmentBefore[$name]=@{Exists=$null -ne $value;Value=$value.Value}
            [Environment]::SetEnvironmentVariable($name,$null,'Process')
        }
    }
    AfterEach {
        foreach($name in $environmentBefore.Keys){
            [Environment]::SetEnvironmentVariable($name, $(if($environmentBefore[$name].Exists){$environmentBefore[$name].Value}else{$null}), 'Process')
        }
    }
    It 'finds the real source checkout rather than the Modules folder' {
        & $module {Get-PcaiProjectRoot}|Should -BeExactly (Get-Item -LiteralPath $repoRoot).FullName
    }
    It 'returns the already-existing actual checkout artifact directory without creating a substitute' {
        $expected=Join-Path (Get-Item -LiteralPath $repoRoot).FullName '.pcai'
        Test-Path -LiteralPath $expected -PathType Container|Should -BeTrue
        & $module {Get-PcaiArtifactsRoot}|Should -BeExactly $expected
    }
    It 'selects a genuine explicit machine root' {
        $selected=New-EvaluationRootFixture
        $env:PCAI_ROOT=$selected
        & $module {Get-PcaiProjectRoot}|Should -BeExactly $selected
    }
    It 'creates the explicitly selected synthetic artifact directory after root admission' {
        $selected=New-EvaluationRootFixture
        $env:PCAI_ROOT=$selected
        $artifact=Join-Path $TestDrive ('selected-artifacts-'+[guid]::NewGuid().ToString('N'))
        $env:PCAI_ARTIFACTS_ROOT=$artifact
        & $module {Get-PcaiArtifactsRoot}|Should -BeExactly $artifact
        Test-Path -LiteralPath $artifact -PathType Container|Should -BeTrue
        Test-Path -LiteralPath (Join-Path $selected '.pcai')|Should -BeFalse
    }
    It 'refuses an invalid explicit root before creating artifacts: <Kind>' -ForEach @(
        @{Kind='missing directory'},@{Kind='missing entrypoint'},@{Kind='missing config'},@{Kind='non-filesystem provider'},@{Kind='whitespace'}
    ) {
        $selected=New-EvaluationRootFixture
        $env:PCAI_ROOT=switch($Kind){
            'missing directory' {Join-Path $TestDrive 'absent-root'}
            'missing entrypoint' {Remove-Item -LiteralPath (Join-Path $selected 'PC-AI.ps1');$selected}
            'missing config' {Remove-Item -LiteralPath (Join-Path $selected 'Config/llm-config.json');$selected}
            'non-filesystem provider' {'Env:PCAI_ROOT'}
            'whitespace' {'   '}
        }
        $artifactPath=Join-Path $TestDrive ('refused-artifacts-'+[guid]::NewGuid().ToString('N'))
        $env:PCAI_ARTIFACTS_ROOT=$artifactPath
        {& $module {Get-PcaiArtifactsRoot}}|Should -Throw
        Test-Path -LiteralPath $artifactPath|Should -BeFalse
    }
    It 'cannot hide a changed invalid override behind an earlier successful discovery' {
        $env:PCAI_ROOT=New-EvaluationRootFixture
        $null=& $module {Get-PcaiProjectRoot}
        $env:PCAI_ROOT=Join-Path $TestDrive 'changed-invalid-root'
        {& $module {Get-PcaiProjectRoot}}|Should -Throw
    }
    It 'uses default checkout artifact paths in a fresh copied checkout' {
        $selected=New-EvaluationRootFixture
        $installed=Join-Path $selected 'Modules'
        Copy-EvaluationConsumer -Destination $installed
        $result=Invoke-EvaluationRootProbe -Installed $installed
        $result.Root|Should -BeExactly $selected
        $result.Artifacts|Should -BeExactly (Join-Path $selected '.pcai')
        Test-Path -LiteralPath (Join-Path $installed '.pcai')|Should -BeFalse
    }
    It 'uses the selected machine root and actual configuration from a detached copied consumer' {
        $selected=New-EvaluationRootFixture
        $installed=Join-Path $TestDrive ('detached-'+[guid]::NewGuid().ToString('N'))
        Copy-EvaluationConsumer -Destination $installed
        $result=Invoke-EvaluationRootProbe -Installed $installed -Root $selected
        $result.Root|Should -BeExactly $selected
        $result.Artifacts|Should -BeExactly (Join-Path $selected '.pcai')
        $result.OllamaModel|Should -BeExactly 'evaluation-root-unique-model'
        $result.OllamaUrl|Should -BeExactly 'http://evaluation-root.invalid:19432'
        Test-Path -LiteralPath (Join-Path $installed '.pcai')|Should -BeFalse
    }
    It 'finds genuine source ancestry when the optional Common helper is absent' {
        $selected=New-EvaluationRootFixture
        $installed=Join-Path $selected 'Modules'
        Copy-EvaluationConsumer -Destination $installed -WithoutCommon
        $result=Invoke-EvaluationRootProbe -Installed $installed
        $result.Root|Should -BeExactly $selected
        $result.Artifacts|Should -BeExactly (Join-Path $selected '.pcai')
    }
    It 'refuses a detached consumer without Common or a genuine configured root' {
        $installed=Join-Path $TestDrive ('unconfigured-'+[guid]::NewGuid().ToString('N'))
        Copy-EvaluationConsumer -Destination $installed -WithoutCommon
        $output=& (Get-Command pwsh).Source -NoLogo -NoProfile -File $script:rootProbe $installed '' 2>&1
        $LASTEXITCODE|Should -Not -Be 0
        ($output -join "`n")|Should -Match 'Set PCAI_ROOT'
        Test-Path -LiteralPath (Join-Path $installed '.pcai')|Should -BeFalse
    }
}

Describe 'Evaluation regression reports use actual isolated reference files' {
    BeforeEach {
        $script:references=Join-Path $TestDrive ([guid]::NewGuid().ToString('N'))
        $null=New-Item -ItemType Directory -Path $references
        $script:baselineBefore=& $module {$script:EvaluationConfig.BaselinePath}
        & $module {param($path) $script:EvaluationConfig.BaselinePath=$path} $references
        '{"Timestamp":"2026-01-01T00:00:00Z","Metrics":{"accuracy":{"Mean":1.0}}}'|Set-Content -LiteralPath (Join-Path $references 'reference.json')
        '{"Timestamp":"2026-01-01T00:00:00Z","Metrics":{"accuracy":{"Mean":0.5}}}'|Set-Content -LiteralPath (Join-Path $references 'unchanged.json')
        'not a baseline'|Set-Content -LiteralPath (Join-Path $references 'ignore.txt')
        $script:reportSuite=New-ContractSuite
        New-ContractResult $reportSuite fixture fail 0.5 0.5
    }
    AfterEach { & $module {param($path) $script:EvaluationConfig.BaselinePath=$path} $baselineBefore }
    It 'discovers only JSON references and aggregates independently expected regression counts' {
        $before=@(Get-ChildItem -LiteralPath $references -File|Get-FileHash|Select-Object Path,Hash)|ConvertTo-Json -Compress
        $report=Get-RegressionReport -Suite $reportSuite
        $report.BaselinesCompared|Should -Be 2
        ($report.Reports.BaselineName|Sort-Object)|Should -Be @('reference','unchanged')
        $report.Summary.TotalRegressions|Should -Be 1;$report.Summary.TotalImprovements|Should -Be 0
        ($report.Reports|Where-Object BaselineName -EQ reference).Regressions[0].Change|Should -Be -50
        (@(Get-ChildItem -LiteralPath $references -File|Get-FileHash|Select-Object Path,Hash)|ConvertTo-Json -Compress)|Should -BeExactly $before
    }
    It 'honors the explicit requested baseline subset' {
        $report=Get-RegressionReport -Suite $reportSuite -BaselineNames unchanged
        $report.BaselinesCompared|Should -Be 1;$report.Reports[0].BaselineName|Should -BeExactly 'unchanged'
        $report.Summary.TotalRegressions|Should -Be 0
    }
    It 'does not return a successful report when an explicitly requested reference is absent' {
        {Get-RegressionReport -Suite $reportSuite -BaselineNames missing -ErrorAction Stop}|Should -Throw
    }
}

Describe 'Evaluation progress and structured event consumers' {
    BeforeEach { $script:stateBefore=& $module {@{Config=@{}+$script:EvaluationConfig;State=$script:EvaluationRunState}} }
    AfterEach { & $module {param($state) $script:EvaluationConfig=$state.Config;$script:EvaluationRunState=$state.State} $stateBefore }
    It 'throttles repeated stream updates but persists a later completed observation' {
        $path=Join-Path $TestDrive 'progress/updates.log'
        & $module {param($path) $script:EvaluationConfig.ProgressMode='stream';$script:EvaluationConfig.ProgressLogPath=$path;$script:EvaluationConfig.ProgressIntervalSeconds=2;$script:EvaluationRunState=@{LastProgressUtc=$null}} $path
        $script:progressClock=[datetime]'2026-01-01T00:00:00Z'
        Mock -ModuleName PC-AI.Evaluation Get-Date {$script:progressClock}
        & $module {Write-EvaluationProgress -Completed 1 -Total 4 -TestCaseId first -Elapsed ([timespan]::FromSeconds(10))}
        $script:progressClock=$progressClock.AddSeconds(1)
        & $module {Write-EvaluationProgress -Completed 2 -Total 4 -TestCaseId throttled -Elapsed ([timespan]::FromSeconds(11))}
        $script:progressClock=$progressClock.AddSeconds(2)
        & $module {Write-EvaluationProgress -Completed 3 -Total 4 -TestCaseId later -Elapsed ([timespan]::FromSeconds(13))}
        @(Get-Content -LiteralPath $path)|Should -Be @('progress=1/4 (25%) elapsed=00:00:10 test=first','progress=3/4 (75%) elapsed=00:00:13 test=later')
    }
    It 'emits one valid structured event matching the actual persisted event' {
        $path=Join-Path $TestDrive 'events/output.jsonl'
        & $module {param($path) $script:EvaluationConfig.ProgressMode='silent';$script:EvaluationConfig.EventsLogPath=$path;$script:EvaluationConfig.EmitStructuredMessages=$true} $path
        $event=& $module {Write-EvaluationEvent -Type fixture -Message 'synthetic consumer event' -Level warn -Data @{completed=2;runId='fixture-run'}}
        $event|Should -BeExactly (Get-Content -LiteralPath $path)
        $parsed=$event|ConvertFrom-Json
        $parsed.type|Should -BeExactly 'fixture';$parsed.level|Should -BeExactly 'warn'
        $parsed.data.completed|Should -Be 2;$parsed.data.runId|Should -BeExactly 'fixture-run'
        {[datetime]::Parse($parsed.ts)}|Should -Not -Throw
    }
}

Describe 'Evaluation judge contracts at an isolated provider boundary' {
    BeforeEach {
        $script:judgeConfigBefore=& $module {@{}+$script:EvaluationConfig}
        & $module {
            $script:judgeSynthetic='{"accuracy":8,"overall":8,"reasoning":"synthetic judgment"}'
            $script:judgeFailure=$false
            function script:Invoke-PcaiGenerate {
                param($Prompt,$MaxTokens,$Temperature)
                $script:judgeObserved=@{}+$PSBoundParameters
                if($script:judgeFailure){throw 'synthetic judge provider failure'}
                return $script:judgeSynthetic
            }
        }
    }
    AfterEach { & $module {param($config) Remove-Item Function:Invoke-PcaiGenerate -ErrorAction SilentlyContinue;$script:EvaluationConfig=$config} $judgeConfigBefore }
    It 'forwards the actual question context reference and selected criteria to local judging' {
        $result=Invoke-LLMJudge -Response 'synthetic response' -Question 'synthetic question' -Context 'synthetic context' -ReferenceAnswer 'synthetic reference' -Criteria accuracy
        $observed=& $module {$script:judgeObserved}
        foreach($text in @('synthetic response','synthetic question','synthetic context','synthetic reference','- accuracy:')){$observed.Prompt|Should -Match ([regex]::Escape($text))}
        $observed.Prompt|Should -Not -Match '\- safety:'
        $result.accuracy|Should -Be 8;$result.overall|Should -Be 8
        $result.raw_response|Should -Match 'synthetic judgment'
    }
    It 'returns an explicit provider error instead of a numeric verdict after failure' {
        & $module {$script:judgeFailure=$true}
        $result=Invoke-LLMJudge -Response synthetic -Question synthetic -Criteria accuracy
        $result.error|Should -BeExactly 'synthetic judge provider failure'
        $result.ContainsKey('overall')|Should -BeFalse
    }
    It 'retains unparseable provider text with an explicit parse error' {
        & $module {$script:judgeSynthetic='no structured judgment'}
        $result=Invoke-LLMJudge -Response synthetic -Question synthetic -Criteria accuracy
        $result.error|Should -Not -BeNullOrEmpty;$result.raw_response|Should -BeExactly 'no structured judgment'
        $result.ContainsKey('overall')|Should -BeFalse
    }
    It 'rejects a malformed judgment despite containing JSON braces' {
        & $module {$script:judgeSynthetic='{broken-json}'}
        $result=Invoke-LLMJudge -Response synthetic -Question synthetic -Criteria accuracy
        $result.error|Should -Not -BeNullOrEmpty;$result.ContainsKey('overall')|Should -BeFalse
    }
    It 'rejects judgment schema outside its requested rating contract: <Kind>' -ForEach @(
        @{Kind='out-of-range rating';Body='{"accuracy":999,"overall":999,"reasoning":"synthetic"}'},
        @{Kind='missing requested criterion';Body='{"overall":8,"reasoning":"synthetic"}'},
        @{Kind='boolean rating';Body='{"accuracy":true,"overall":8,"reasoning":"synthetic"}'},
        @{Kind='string rating';Body='{"accuracy":"8","overall":8,"reasoning":"synthetic"}'},
        @{Kind='null rating';Body='{"accuracy":null,"overall":8,"reasoning":"synthetic"}'},
        @{Kind='missing overall';Body='{"accuracy":8,"reasoning":"synthetic"}'},
        @{Kind='empty reasoning';Body='{"accuracy":8,"overall":8,"reasoning":"   "}'},
        @{Kind='nonstring reasoning';Body='{"accuracy":8,"overall":8,"reasoning":8}'},
        @{Kind='nested rating shape';Body='{"accuracy":{"score":8},"overall":8,"reasoning":"synthetic"}'},
        @{Kind='nonfinite numeric rating';Body='{"accuracy":1e400,"overall":8,"reasoning":"synthetic"}'},
        @{Kind='array judgment shape';Body='[{"accuracy":8,"overall":8,"reasoning":"synthetic"}]'},
        @{Kind='array with a quoted closing delimiter';Body='["]",{"accuracy":8,"overall":8,"reasoning":"synthetic"}]'},
        @{Kind='array with an escaped quote and closing delimiter';Body='["\"]",{"accuracy":8,"overall":8,"reasoning":"synthetic"}]'}
    ) {
        & $module {param($body) $script:judgeSynthetic=$body} $Body
        $result=Invoke-LLMJudge -Response synthetic -Question synthetic -Criteria accuracy
        $result.error|Should -Not -BeNullOrEmpty
        $result.ContainsKey('overall')|Should -BeFalse
    }
    It 'retains valid boundary and fractional ratings for all requested criteria' {
        & $module {$script:judgeSynthetic='{"accuracy":1,"relevance":10,"overall":7.5,"reasoning":"synthetic valid ratings"}'}
        $result=Invoke-LLMJudge -Response synthetic -Question synthetic -Criteria accuracy,relevance
        $result.ContainsKey('error')|Should -BeFalse
        $result.accuracy|Should -Be 1;$result.relevance|Should -Be 10;$result.overall|Should -Be 7.5
    }
    It 'retains a valid fenced JSON object response' {
        & $module {$script:judgeSynthetic='```json'+"`n"+'{"accuracy":8,"overall":8,"reasoning":"synthetic fenced judgment"}'+"`n"+'```'}
        $result=Invoke-LLMJudge -Response synthetic -Question synthetic -Criteria accuracy
        $result.ContainsKey('error')|Should -BeFalse;$result.accuracy|Should -Be 8
        $result.raw_response|Should -Match 'synthetic fenced judgment'
    }
    It 'retains an earlier-supported explanatory object prefix: <Prefix>' -ForEach @(
        @{Prefix='Here is the judgment:'},@{Prefix='Evaluation [synthetic annotation]:'}
    ) {
        & $module {param($prefix) $script:judgeSynthetic=$prefix+"`n"+'{"accuracy":8,"overall":8,"reasoning":"synthetic explanatory judgment"}'} $Prefix
        $result=Invoke-LLMJudge -Response synthetic -Question synthetic -Criteria accuracy
        $result.ContainsKey('error')|Should -BeFalse;$result.accuracy|Should -Be 8
    }
    It 'validates the HTTP fallback response and sends the actual selected endpoint and criteria' {
        & $module {$script:EvaluationConfig.OllamaBaseUrl='http://judge-provider.invalid:19433'}
        Mock -ModuleName PC-AI.Evaluation Get-Command {$null} -ParameterFilter {$Name -eq 'Invoke-PcaiGenerate'}
        $script:judgeHttpObserved=$null
        Mock -ModuleName PC-AI.Evaluation Invoke-RestMethod {
            $script:judgeHttpObserved=@{Uri=$Uri;Method=$Method;Body=$Body|ConvertFrom-Json -AsHashtable}
            return @{response='{"accuracy":6,"overall":6,"reasoning":"synthetic HTTP judgment"}'}
        }
        $result=Invoke-LLMJudge -Response 'HTTP synthetic response' -Question 'HTTP synthetic question' -Criteria accuracy
        $result.accuracy|Should -Be 6;$result.ContainsKey('error')|Should -BeFalse
        $judgeHttpObserved.Uri|Should -BeExactly 'http://judge-provider.invalid:19433/api/generate'
        $judgeHttpObserved.Method|Should -BeExactly 'Post'
        $judgeHttpObserved.Body.prompt|Should -Match 'HTTP synthetic question'
        $judgeHttpObserved.Body.prompt|Should -Match '\- accuracy:'
        $judgeHttpObserved.Body.stream|Should -BeFalse
        Should -Invoke -ModuleName PC-AI.Evaluation Invoke-RestMethod -Exactly -Times 1
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
