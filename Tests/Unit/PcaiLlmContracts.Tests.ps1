# External providers, native functions, hardware and publication are replaced
# before invocation. Files used by fallback search are private TestDrive data.
BeforeAll {
    $script:LlmRoot = Join-Path $PSScriptRoot '../../Modules/PC-AI.LLM'
    foreach ($file in Get-ChildItem (Join-Path $script:LlmRoot 'Private') -File -Filter '*.ps1') { . $file.FullName }
    foreach ($file in Get-ChildItem (Join-Path $script:LlmRoot 'Public') -File -Filter '*.ps1') { . $file.FullName }
    function Test-PcaiNativeAvailable { throw 'Unmocked native availability forbidden' }
    function Invoke-PcaiNativeFileSearch { param($Pattern, $Path, $MaxResults) throw 'Unmocked native search forbidden' }
    function Invoke-PcaiNativeDuplicates { param($Path, $MinimumSize) throw 'Unmocked native search forbidden' }
    function Invoke-PcaiNativeContentSearch { param($Pattern, $Path, $FilePattern, $MaxResults, $ContextLines) throw 'Unmocked native search forbidden' }
    function Invoke-PcaiNativeSystemInfo { throw 'Unmocked native system query forbidden' }
    function Get-CimInstance { [CmdletBinding()] param($ClassName, $Filter) throw 'Unmocked hardware query forbidden' }
    function Get-PcaiDependencyStamp { param($InputObject) return 'fixture-stamp' }
    function Get-PcaiSharedCacheEntry { param($Namespace, $Key, $DependencyStamp) return $null }
    function Set-PcaiSharedCacheEntry { param($Namespace, $Key, $Value, $DependencyStamp, $TtlSeconds) throw 'Unmocked cache publication forbidden' }
    function New-LlmContractConfig {
        return @{
            ProjectRoot = $TestDrive; ConfigPath = (Join-Path $TestDrive 'fixture-config.json'); ToolsPath = 'fixture-tools.json'
            DefaultModel = 'fixture-default'; DefaultTimeout = 7; ProviderOrder = @('ollama', 'pcai-inference')
            PcaiInferenceApiUrl = 'http://inference.invalid:18080'; PcaiInferenceModel = 'fixture-inference'
            OllamaApiUrl = 'http://ollama.invalid:11434'; OllamaToolModel = 'fixture-tools'; OllamaSummaryModel = 'fixture-summary'
            RouterApiUrl = 'http://router.invalid:8000'; RouterModel = 'fixture-router'
            VLLMApiUrl = 'http://vllm.invalid:8001'; VLLMModel = 'fixture-vllm'; LMStudioApiUrl = 'http://studio.invalid:1234'
        }
    }
}

Describe 'LLM native-search public schemas and private-file fallbacks' -Tag 'Unit', 'LLM', 'Portable' {
    BeforeEach {
        Mock Get-Module { [pscustomobject]@{ Name = 'PC-AI.Acceleration' } } -ParameterFilter { $Name -eq 'PC-AI.Acceleration' }
        Mock Test-PcaiNativeAvailable { $false }
        Mock Import-Module { throw 'Unexpected acceleration import' }
        Mock Invoke-PcaiNativeFileSearch { throw 'Unexpected native file search' }
        Mock Invoke-PcaiNativeContentSearch { throw 'Unexpected native content search' }
        Mock Invoke-PcaiNativeDuplicates { throw 'Unexpected native duplicate search' }
    }
    It 'searches private files with the managed backend and preserves output/provenance' {
        [IO.File]::WriteAllText((Join-Path $TestDrive 'first.ps1'), 'fixture')
        [IO.File]::WriteAllText((Join-Path $TestDrive 'ignore.txt'), 'other')
        $result = Invoke-NativeSearch -Operation Files -Path $TestDrive -Pattern '*.ps1'
        $result.Engine | Should -Be 'PowerShell'
        $result.Status | Should -Be 'Success'
        $result.FilesMatched | Should -Be 1
        @($result.Files)[0].Path | Should -Be (Join-Path $TestDrive 'first.ps1')
        $result.SearchPath | Should -Be (Get-Item $TestDrive).FullName
        $result.TotalDurationMs | Should -BeGreaterOrEqual 0
        Should -Invoke Invoke-PcaiNativeFileSearch -Exactly -Times 0
    }
    It 'returns bounded private content matches and honors file selection' {
        [IO.File]::WriteAllLines((Join-Path $TestDrive 'fixture.log'), @('safe', 'error one', 'error two'))
        [IO.File]::WriteAllText((Join-Path $TestDrive 'ignored.txt'), 'error excluded')
        $result = Invoke-NativeSearch -Operation Content -Path $TestDrive -Pattern 'error' -FilePattern '*.log' -MaxResults 1
        $result.TotalMatches | Should -Be 1
        @($result.Matches)[0].LineNumber | Should -Be 2
        @($result.Matches)[0].Line | Should -Be 'error one'
    }
    It 'detects actual private duplicate bytes without grouping a different same-size file' {
        foreach ($leaf in 'a.bin', 'b.bin') { [IO.File]::WriteAllText((Join-Path $TestDrive $leaf), 'same') }
        [IO.File]::WriteAllText((Join-Path $TestDrive 'c.bin'), 'else')
        $result = Invoke-NativeSearch -Operation Duplicates -Path $TestDrive -MinimumSize 1 -MaxResults 0
        $result.DuplicateGroupCount | Should -Be 1
        $result.DuplicateFiles | Should -Be 1
        $result.WastedBytes | Should -Be 4
        @($result.Groups)[0].Paths | Should -HaveCount 2
    }
    It 'formats native file results while forwarding the requested limit and pattern' {
        Mock Test-PcaiNativeAvailable { $true }
        Mock Invoke-PcaiNativeFileSearch {
            [pscustomobject]@{ Status = 'Success'; Pattern = '*.ps1'; FilesScanned = 8; FilesMatched = 1; TotalSize = 1024; ElapsedMs = 2; Truncated = $true; Files = @([pscustomobject]@{ Path = 'fixture.ps1'; Size = 1024; ReadOnly = $false }) }
        }
        $result = Invoke-NativeSearch -Operation Files -Path $TestDrive -Pattern '*.ps1' -MaxResults 1 -AsJson | ConvertFrom-Json
        $result.Engine | Should -Be 'Native/Rust'
        $result.FilesMatched | Should -Be 1
        $result.Truncated | Should -BeTrue
        $result.Files[0].SizeKB | Should -Be 1
        Should -Invoke Invoke-PcaiNativeFileSearch -Exactly -Times 1 -ParameterFilter { $Pattern -eq '*.ps1' -and $MaxResults -eq 1 -and $Path -eq (Get-Item $TestDrive).FullName }
    }
    It 'formats native content results and forwards context/file selection' {
        Mock Test-PcaiNativeAvailable { $true }
        Mock Invoke-PcaiNativeContentSearch {
            [pscustomobject]@{ Status = 'Success'; Pattern = 'error'; FilePattern = '*.log'; FilesScanned = 2; FilesMatched = 1; TotalMatches = 1; ElapsedMs = 2; Truncated = $false; Matches = @([pscustomobject]@{ Path = 'fixture.log'; LineNumber = 2; Line = ' error '; Before = @('before'); After = @('after') }) }
        }
        $result = Invoke-NativeSearch -Operation Content -Path $TestDrive -Pattern error -FilePattern '*.log' -MaxResults 3 -ContextLines 1
        @($result.Matches)[0].Line | Should -Be 'error'
        @($result.Matches)[0].Context.Before[0] | Should -Be 'before'
        Should -Invoke Invoke-PcaiNativeContentSearch -Exactly -Times 1 -ParameterFilter { $FilePattern -eq '*.log' -and $ContextLines -eq 1 -and $MaxResults -eq 3 }
    }
    It 'reports a native operation error without scanning through a managed fallback' {
        Mock Test-PcaiNativeAvailable { $true }
        Mock Invoke-PcaiNativeFileSearch { throw 'Synthetic native refusal' }
        Mock Invoke-PowerShellFileSearch { throw 'Unexpected fallback after operation error' }
        $result = Invoke-NativeSearch -Operation Files -Path $TestDrive -Pattern '*.ps1'
        $result.Status | Should -Be 'Error'
        $result.Error | Should -Match 'Synthetic native refusal'
        Should -Invoke Invoke-PowerShellFileSearch -Exactly -Times 0
    }
    It 'rejects nonexistent roots before running any search' {
        { Invoke-NativeSearch -Operation Files -Path (Join-Path $TestDrive absent) -Pattern '*' } | Should -Throw '*Invalid path*'
        Should -Invoke Invoke-PcaiNativeFileSearch -Exactly -Times 0
    }
    It 'reports the missing required pattern as a structured operation error' {
        (Invoke-NativeSearch -Operation Content -Path $TestDrive).Error | Should -Match 'Pattern is required'
    }
}

Describe 'LLM public generation parameter and error contracts' -Tag 'Unit', 'LLM', 'Portable' {
    BeforeEach {
        $script:ModuleConfig = New-LlmContractConfig
        Mock Test-OllamaConnection { $true }
        Mock Get-OllamaModels { [pscustomobject]@{ Name = 'fixture-default' } }
        Mock Invoke-OllamaGenerate { [pscustomobject]@{ message = @{ content = 'fixture answer' }; model = 'fixture-model'; raw = @{ timing = @{ totalDurationNs = 3 } }; ToolCalls = @(); ExecutedTools = @() } }
        Mock Start-Sleep { throw 'Unexpected retry delay' }
        Mock Write-Warning {}
    }
    It 'forwards explicit zero and negative sampling options along with model/tool limits' {
        $result = Send-OllamaRequest -Prompt 'fixture prompt' -Model fixture-default -System 'fixture system' -MaxRetries 1 -Temperature 0 -MaxTokens 32 -NumCtx 4096 -NumThread 2 -TopP 0 -TopK 1 -RepeatLastN -1 -RepeatPenalty 0 -TfsZ 0 -Seed 0 -EnableTools -Stream -TimeoutSeconds 9
        $result.Response | Should -Be 'fixture answer'
        $result.Model | Should -Be 'fixture-model'
        $result.ToolCalls | Should -HaveCount 0
        Should -Invoke Invoke-OllamaGenerate -Exactly -Times 1 -ParameterFilter {
            $TopP -eq 0 -and $Seed -eq 0 -and $RepeatPenalty -eq 0 -and $TfsZ -eq 0 -and $RepeatLastN -eq -1 -and $NumCtx -eq 4096 -and $NumThread -eq 2 -and $MaxTokens -eq 32 -and $TimeoutSeconds -eq 9 -and $Stream -and $EnableTools -and $System -eq 'fixture system'
        }
    }
    It 'rejects an unreachable runner before model enumeration or generation' {
        Mock Test-OllamaConnection { $false }
        { Send-OllamaRequest -Prompt fixture -MaxRetries 1 } | Should -Throw '*Cannot reach*'
        Should -Invoke Get-OllamaModels -Exactly -Times 0
        Should -Invoke Invoke-OllamaGenerate -Exactly -Times 0
    }
    It 'retries one transient failure and returns the successful useful response' {
        $script:GenerationAttempts = 0
        Mock Start-Sleep {}
        Mock Invoke-OllamaGenerate {
            $script:GenerationAttempts++
            if ($script:GenerationAttempts -eq 1) { throw 'Synthetic transient failure' }
            [pscustomobject]@{ message = @{ content = 'recovered answer' }; model = 'fixture-model'; raw = @{} }
        }
        (Send-OllamaRequest -Prompt fixture -MaxRetries 2).Response | Should -Be 'recovered answer'
        Should -Invoke Invoke-OllamaGenerate -Exactly -Times 2
        Should -Invoke Start-Sleep -Exactly -Times 1
    }
}

Describe 'LLM native request serialization and private cleanup' -Tag 'Unit', 'LLM', 'Portable' {
    BeforeEach {
        $script:ModuleConfig = New-LlmContractConfig
        $script:CapturedPayload = $null
        $script:CapturedRequestPath = $null
        Mock Set-Content { $script:CapturedPayload = $Value | ConvertFrom-Json; $script:CapturedRequestPath = $Path }
        Mock Test-Path { $Path -eq $script:CapturedRequestPath }
        Mock Remove-Item {}
        Mock Invoke-OllamaNativeCli { [pscustomobject]@{ ok = $true; content = 'fixture answer'; model = 'fixture-tools'; toolCalls = @(); executedTools = @(); timing = @{ totalDurationNs = 8 } } }
    }
    It 'serializes tool messages and chooses the configured tool model without publishing a request file' {
        $messages = @([pscustomobject]@{ role = 'assistant'; content = 'fixture'; tool_calls = @('{"id":"t1","function":{"name":"fixture"}}') })
        $result = Invoke-OllamaNativeChat -Messages $messages -EnableTools -NumCtx 4096 -TopP 0 -Seed 0
        $script:CapturedPayload.model | Should -Be 'fixture-tools'
        $script:CapturedPayload.enableTools | Should -BeTrue
        $script:CapturedPayload.messages[0].toolCalls[0].id | Should -Be 't1'
        $script:CapturedPayload.topP | Should -Be 0
        $script:CapturedPayload.seed | Should -Be 0
        $result.message.content | Should -Be 'fixture answer'
        Should -Invoke Invoke-OllamaNativeCli -Exactly -Times 1 -ParameterFilter { $Arguments[0] -eq 'chat' -and $Arguments[1] -eq '--request-file' -and $Arguments[2] -eq $script:CapturedRequestPath }
        Should -Invoke Remove-Item -Exactly -Times 1 -ParameterFilter { $Path -eq $script:CapturedRequestPath -and $Force }
        [IO.File]::Exists($script:CapturedRequestPath) | Should -BeFalse
    }
    It 'attempts exact request-file cleanup when the native provider rejects the request' {
        Mock Invoke-OllamaNativeCli { [pscustomobject]@{ ok = $false; error = 'Synthetic request rejection' } }
        { Invoke-OllamaNativeChat -Messages @(@{ role = 'user'; content = 'fixture' }) } | Should -Throw '*Synthetic request rejection*'
        Should -Invoke Remove-Item -Exactly -Times 1 -ParameterFilter { $Path -eq $script:CapturedRequestPath }
    }
}

Describe 'LLM provider selection and endpoint mappings' -Tag 'Unit', 'LLM', 'Portable' {
    BeforeEach {
        $script:ModuleConfig = New-LlmContractConfig
        Mock Get-CachedProviderHealth { $true }
        Mock Invoke-OllamaChat { [pscustomobject]@{ message = @{ content = 'ollama answer' } } }
        Mock Invoke-OpenAIChat { [pscustomobject]@{ message = @{ content = 'openai answer' } } }
        Mock Invoke-OpenAIChatWithProgress { [pscustomobject]@{ message = @{ content = 'progress answer' } } }
    }
    It 'normalizes legacy provider aliases and preserves order while avoiding duplicate health probes' {
        $script:ModuleConfig.ProviderOrder = @('pcai-native', 'ollama', 'functiongemma')
        Mock Get-CachedProviderHealth { $Provider -eq 'vllm' }
        $result = Invoke-LLMChatWithFallback -Messages @(@{ role = 'user'; content = 'fixture' }) -TimeoutSeconds 30
        $result.Provider | Should -Be 'vllm'
        Should -Invoke Get-CachedProviderHealth -Exactly -Times 1 -ParameterFilter { $Provider -eq 'ollama' -and $TimeoutSeconds -eq 10 }
        Should -Invoke Invoke-OpenAIChat -Exactly -Times 1 -ParameterFilter { $Model -eq 'fixture-vllm' -and $ApiUrl -eq 'http://vllm.invalid:8001' -and $TimeoutSeconds -eq 30 }
    }
    It 'uses explicit providers and keeps progress requests on the selected endpoint' {
        $result = Invoke-LLMChatWithFallback -Messages @(@{ role = 'user'; content = 'fixture' }) -Provider vllm -ShowProgress -Model custom -ProgressIntervalSeconds 2
        $result.Provider | Should -Be 'vllm'
        $result.message.content | Should -Be 'progress answer'
        Should -Invoke Invoke-OpenAIChatWithProgress -Exactly -Times 1 -ParameterFilter { $Model -eq 'custom' -and $ApiUrl -eq 'http://vllm.invalid:8001' -and $ProgressIntervalSeconds -eq 2 }
        Should -Invoke Invoke-OllamaChat -Exactly -Times 0
    }
    It 'fails explicitly when every configured provider is unhealthy' {
        Mock Get-CachedProviderHealth { $false }
        { Invoke-LLMChatWithFallback -Messages @(@{ role = 'user'; content = 'fixture' }) } | Should -Throw '*No LLM providers*'
        Should -Invoke Invoke-OllamaChat -Exactly -Times 0
        Should -Invoke Invoke-OpenAIChat -Exactly -Times 0
    }
    It 'resolves private HVSock mapping and preserves unmapped explicit URLs' {
        $mapping = Join-Path $TestDrive 'hvsock.conf'
        [IO.File]::WriteAllLines($mapping, @('# fixture mappings', 'ollama:guid:localhost:11435', 'vllm:guid:localhost:8002'))
        Resolve-PcaiEndpoint -ApiUrl 'auto' -ProviderName ollama -ConfigPath $mapping | Should -Be 'http://localhost:11435'
        Resolve-PcaiEndpoint -ApiUrl 'hvsock://vllm' -ConfigPath $mapping | Should -Be 'http://localhost:8002'
        Resolve-PcaiEndpoint -ApiUrl 'https://explicit.invalid' -ConfigPath $mapping | Should -Be 'https://explicit.invalid'
    }
    It 'uses the selected host provider default when an auto mapping is absent' {
        Resolve-PcaiEndpoint -ApiUrl auto -ProviderName lmstudio -ConfigPath (Join-Path $TestDrive absent) | Should -Be 'http://studio.invalid:1234'
    }
}

Describe 'LLM public system-information hardware boundaries' -Tag 'Unit', 'LLM', 'Portable' {
    BeforeEach {
        Mock Import-Module { throw 'Unexpected hardware module import' }
        Mock Get-Command { $null } -ParameterFilter { $Name -in @('nvidia-smi', 'Get-PcaiServiceHealth') }
        Mock Invoke-PcaiNativeSystemInfo { throw 'Synthetic unavailable native system information' }
        Mock Get-CimInstance { throw 'Unexpected hardware class' }
    }
    It 'uses CIM OS metadata after native summary failure without exposing unrelated properties' {
        Mock Get-CimInstance { [pscustomobject]@{ Caption = 'Fixture OS'; Version = '10.0'; BuildNumber = 'fixture-build'; OSArchitecture = '64-bit'; SecretUnrelated = 'excluded' } } -ParameterFilter { $ClassName -eq 'Win32_OperatingSystem' }
        $result = Get-SystemInfoTool -Category OS | ConvertFrom-Json
        $result.Caption | Should -Be 'Fixture OS'
        $result.PSObject.Properties.Name | Should -Not -Contain SecretUnrelated
        Should -Invoke Invoke-PcaiNativeSystemInfo -Exactly -Times 1
    }
    It 'normalizes storage and network aliases and uses the selected PNP identity for driver queries' -ForEach @(
        @{ Category = 'DiskDrive'; Class = 'Win32_DiskDrive'; Label = 'Fixture Disk' },
        @{ Category = 'Net'; Class = 'Win32_NetworkAdapter'; Label = 'Fixture Network' }
    ) {
        Mock Get-CimInstance { [pscustomobject]@{ Model = $Label; Name = $Label; PNPDeviceID = 'FIXTURE\DEVICE'; InterfaceType = 'fixture' } } -ParameterFilter { $ClassName -eq $Class }
        Mock Get-CimInstance { [pscustomobject]@{ DriverVersion = '1.2.3'; Manufacturer = 'Fixture Provider' } } -ParameterFilter { $ClassName -eq 'Win32_PnPEntity' }
        $result = Get-SystemInfoTool -Category $Category -Detail DriverVersion | ConvertFrom-Json
        $result.Driver | Should -Be '1.2.3'
        Should -Invoke Get-CimInstance -Exactly -Times 1 -ParameterFilter { $ClassName -eq 'Win32_PnPEntity' -and $Filter -eq "DeviceID = 'FIXTURE\\DEVICE'" }
    }
    It 'normalizes the display alias without invoking the NVIDIA CLI' {
        Mock Get-CimInstance { [pscustomobject]@{ Name = 'Fixture GPU'; DriverVersion = '572.83' } } -ParameterFilter { $ClassName -eq 'Win32_VideoController' }
        $result = Get-SystemInfoTool -Category Display -Detail DriverVersion | ConvertFrom-Json
        $result.DriverVersion | Should -Be '572.83'
        $result.Name | Should -Be 'Fixture GPU'
    }
    It 'queries the requested media and HID classes with driver metadata intact' -ForEach @(
        @{ Category = 'Media'; Class = 'Win32_SoundDevice' },
        @{ Category = 'HIDClass'; Class = 'Win32_PnPEntity' }
    ) {
        Mock Get-CimInstance { [pscustomobject]@{ Name = 'Fixture Device'; DriverVersion = '1.2'; Manufacturer = 'Fixture Provider'; Status = 'OK' } } -ParameterFilter { $ClassName -eq $Class }
        $result = Get-SystemInfoTool -Category $Category -Detail DriverVersion | ConvertFrom-Json
        $result.DriverVersion | Should -Be '1.2'
        $result.Manufacturer | Should -Be 'Fixture Provider'
    }
    It 'reports a failed hardware query as structured JSON with the requested category' {
        $result = Get-SystemInfoTool -Category BIOS | ConvertFrom-Json
        $result.Category | Should -Be BIOS
        $result.Error | Should -Match 'Unexpected hardware class'
    }
}

Describe 'LLM diagnosis collection without provider or config changes' -Tag 'Unit', 'LLM', 'Portable' {
    BeforeEach {
        $script:ModuleConfig = New-LlmContractConfig
        Mock Invoke-NativeSearch { [pscustomobject]@{ Status = 'Success'; Summary = 'fixture useful summary'; FilesMatched = 1 } }
        Mock Invoke-LLMChatWithFallback { throw 'Unexpected inference request' }
        Mock Set-Content { throw 'Unexpected report publication' }
        Mock Write-Host {}
    }
    It 'collects each maintained analysis without inference or report publication when explicitly skipped' -ForEach @(
        @{ Analysis = 'Quick'; ExpectedCalls = 2 }, @{ Analysis = 'Full'; ExpectedCalls = 3 },
        @{ Analysis = 'Duplicates'; ExpectedCalls = 2 }, @{ Analysis = 'Storage'; ExpectedCalls = 3 }
    ) {
        $before = $script:ModuleConfig | ConvertTo-Json -Depth 5 -Compress
        $result = Invoke-SmartDiagnosis -Path $TestDrive -AnalysisType $Analysis -SkipLLMAnalysis -SaveReport
        $result.LLMAnalysis | Should -Be Skipped
        $result.AnalysisType | Should -Be $Analysis
        $result.DiagnosticSummary | Should -Match 'fixture useful summary'
        ($script:ModuleConfig | ConvertTo-Json -Depth 5 -Compress) | Should -BeExactly $before
        Should -Invoke Invoke-NativeSearch -Exactly -Times $ExpectedCalls
        Should -Invoke Invoke-LLMChatWithFallback -Exactly -Times 0
        Should -Invoke Set-Content -Exactly -Times 0
    }
    It 'returns useful partial collection when a later search fails' {
        Mock Invoke-NativeSearch { throw 'Synthetic later collection failure' } -ParameterFilter { $Operation -eq 'Files' }
        Mock Write-Warning {}
        $result = Invoke-SmartDiagnosis -Path $TestDrive -AnalysisType Quick -SkipLLMAnalysis
        $result.RawResults.Duplicates.Summary | Should -Be 'fixture useful summary'
        $result.DiagnosticSummary | Should -Match 'fixture useful summary'
        Should -Invoke Write-Warning -Exactly -Times 1
    }
}

Describe 'LLM model metadata and prompt grounding schemas' -Tag 'Unit', 'LLM', 'Portable' {
    BeforeEach {
        $script:ModuleConfig = New-LlmContractConfig
        Mock Resolve-PcaiEndpoint { $ApiUrl }
        Mock Invoke-RestMethod { throw 'Unmocked HTTP request forbidden' }
    }
    It 'chooses the requested provider model and preserves length and ownership metadata' {
        Mock Invoke-RestMethod { [pscustomobject]@{ data = @(
            [pscustomobject]@{ id = 'other'; max_model_len = 1024; root = 'other-root'; owned_by = 'other' },
            [pscustomobject]@{ id = 'requested'; max_model_len = 8192; root = 'fixture-root'; owned_by = 'fixture-owner' }
        ) } }
        $result = Get-VLLMModelInfo -ApiUrl 'http://vllm.invalid' -ModelName requested -TimeoutSeconds 3
        $result.Id | Should -Be requested
        $result.MaxModelLen | Should -Be 8192
        $result.OwnedBy | Should -Be 'fixture-owner'
        Should -Invoke Invoke-RestMethod -Exactly -Times 1 -ParameterFilter { $Uri -eq 'http://vllm.invalid/v1/models' -and $TimeoutSec -eq 3 -and $Method -eq 'Get' }
    }
    It 'returns unavailable model metadata after provider failure rather than inventing a loaded model' {
        Get-VLLMModelInfo -ApiUrl 'http://vllm.invalid' -ModelName requested | Should -BeNullOrEmpty
    }
    It 'builds prompt metadata and grounding text from mocked OS and file boundaries' {
        Mock Get-Command { $null } -ParameterFilter { $Name -eq 'Initialize-PcaiNative' }
        Mock Get-CimInstance { [pscustomobject]@{ Caption = 'Fixture OS' } } -ParameterFilter { $ClassName -eq 'Win32_OperatingSystem' }
        Mock Test-Path { $true }
        Mock Get-Content { 'fixture system prompt' } -ParameterFilter { $Path -like '*DIAGNOSE.md' }
        Mock Get-Content { 'fixture decision rules' } -ParameterFilter { $Path -like '*DIAGNOSE_LOGIC.md' }
        Mock Get-Content { '{"fixture":true}' } -ParameterFilter { $Path -like '*DIAGNOSE_TEMPLATE.json' }
        $result = Get-LLMPromptContext -AnalysisType Full
        $result.Metadata.analysis_type | Should -Be Full
        $result.Metadata.os_version | Should -Be 'Fixture OS'
        $result.Prompts.System | Should -Be 'fixture system prompt'
        $result.Prompts.Logic | Should -Be 'fixture decision rules'
        ($result.MetadataJson | ConvertFrom-Json).os_version | Should -Be 'Fixture OS'
    }
}

Describe 'LLM OpenAI-compatible provider request schemas' -Tag 'Unit', 'LLM', 'Portable' {
    BeforeEach {
        $script:ModuleConfig = New-LlmContractConfig
        Mock Resolve-PcaiEndpoint { $ApiUrl }
        Mock Invoke-RestMethod { throw 'Unexpected HTTP request' }
    }
    It 'forwards endpoint, messages, token limit and timeout then preserves useful completion and provider usage' {
        Mock Invoke-RestMethod {
            [pscustomobject]@{ choices = @([pscustomobject]@{ message = @{ content = 'fixture useful answer' } }); usage = @{ prompt_tokens = 4; completion_tokens = 2 } }
        }
        $result = Invoke-OpenAIChat -Messages @(@{ role = 'user'; content = 'fixture question' }) -Model fixture-model -Temperature 0 -MaxTokens 32 -TimeoutSeconds 3 -ApiUrl 'https://provider.invalid'
        $result.message.content | Should -Be 'fixture useful answer'
        $result.raw.usage.completion_tokens | Should -Be 2
        Should -Invoke Invoke-RestMethod -Exactly -Times 1 -ParameterFilter {
            $payload = $Body | ConvertFrom-Json
            $Uri -eq 'https://provider.invalid/v1/chat/completions' -and $Method -eq 'Post' -and $TimeoutSec -eq 3 -and
            $payload.model -eq 'fixture-model' -and $payload.messages[0].content -eq 'fixture question' -and
            $payload.temperature -eq 0 -and $payload.max_tokens -eq 32 -and $payload.stream -eq $false
        }
    }
    It 'retains an explicit empty choice list without fabricating response text' {
        Mock Invoke-RestMethod { [pscustomobject]@{ choices = @() } }
        $result = Invoke-OpenAIChat -Messages @(@{ role = 'user'; content = 'fixture' }) -Model fixture-model -ApiUrl 'https://provider.invalid'
        $result.message.content | Should -BeNullOrEmpty
        $result.raw.choices | Should -HaveCount 0
    }
    It 'preserves a provider failure instead of converting it to a useful completion' {
        Mock Write-Error {}
        { Invoke-OpenAIChat -Messages @(@{ role = 'user'; content = 'fixture' }) -Model fixture-model -ApiUrl 'https://provider.invalid' } | Should -Throw '*Unexpected HTTP request*'
    }
}

Describe 'LLM public provider-health status schema' -Tag 'Unit', 'LLM', 'Portable' {
    BeforeEach {
        $script:ModuleConfig = New-LlmContractConfig
        Mock Resolve-OllamaNativeCliPath { 'fixture-runner.exe' }
        Mock Test-OllamaConnection { $true }
        Mock Test-PcaiInferenceConnection { $false }
        Mock Get-OllamaModels { [pscustomobject]@{ Name = 'fixture-default' } }
        Mock Test-OpenAIConnection { $true }
        Mock Test-LMStudioConnection { $true }
        Mock Get-VLLMModelInfo { [pscustomobject]@{ Id = 'fixture-vllm'; MaxModelLen = 4096 } }
        Mock Get-VLLMMetricsSnapshot { [pscustomobject]@{ TokensTotal = 12.5 } }
    }
    It 'preserves health compatibility fields while keeping unavailable inference separate from healthy Ollama' {
        $result = Get-LLMStatus
        $result.Ollama.Available | Should -BeTrue
        $result.Ollama.ModelsLoaded | Should -Contain 'fixture-default'
        $result.Ollama.CliPath | Should -Be 'fixture-runner.exe'
        $result.PcaiInference.Available | Should -BeFalse
        $result.PcaiInference.AvailableModels | Should -HaveCount 0
        $result.ActiveProvider | Should -Be ollama
        $result.ActiveModel | Should -Be 'fixture-default'
        $result.VLLM | Should -BeNullOrEmpty
        Should -Invoke Get-VLLMModelInfo -Exactly -Times 0
    }
    It 'checks requested optional providers and forwards the exact selected endpoint/model for metadata' {
        $result = Get-LLMStatus -IncludeLMStudio -IncludeVLLM -TestConnection
        $result.LMStudio.ApiConnected | Should -BeTrue
        $result.VLLM.ApiConnected | Should -BeTrue
        $result.VLLM.ModelInfo.MaxModelLen | Should -Be 4096
        $result.VLLM.Metrics.TokensTotal | Should -Be 12.5
        Should -Invoke Get-VLLMModelInfo -Exactly -Times 1 -ParameterFilter { $ApiUrl -eq 'http://vllm.invalid:8001' -and $ModelName -eq 'fixture-vllm' }
    }
    It 'retains actionable runner-resolution failure in recommendations' {
        Mock Resolve-OllamaNativeCliPath { throw 'Synthetic selected-runner absence' }
        Mock Test-OllamaConnection { $false }
        $result = Get-LLMStatus
        $result.Recommendations -join '|' | Should -Match 'Synthetic selected-runner absence'
        $result.Ollama.ApiConnected | Should -BeFalse
        $result.Ollama.Models | Should -HaveCount 0
    }
}

Describe 'LLM read-only detecting controls' {
    BeforeEach {
        $script:ModuleConfig = @{ ProjectRoot = $TestDrive; DefaultModel = 'fixture-model'; DefaultTimeout = 7; ToolsPath = 'fixture-tools.json' }
        Mock Write-Warning {}
        Mock Get-Command { $null } -ParameterFilter { $Name -eq 'Initialize-PcaiNative' }
    }
    It 'preserves explicit zero sampling options through the actual public generation adapter' {
        Mock Test-OllamaConnection { $true }
        Mock Get-OllamaModels { [pscustomobject]@{ Name = 'fixture-model' } }
        $script:CapturedRequest = $null
        Mock Set-Content { $script:CapturedRequest = $Value | ConvertFrom-Json }
        Mock Test-Path { $false }
        Mock Invoke-OllamaNativeCli { [pscustomobject]@{ ok = $true; content = 'fixture answer'; model = 'fixture-model' } }
        Send-OllamaRequest -Prompt fixture -MaxRetries 1 -TopP 0 -Seed 0 -RepeatPenalty 0 -TfsZ 0 | Out-Null
        $script:CapturedRequest.PSObject.Properties.Name | Should -Contain seed
        $script:CapturedRequest.PSObject.Properties.Name | Should -Contain topP
        $script:CapturedRequest.PSObject.Properties.Name | Should -Contain repeatPenalty
        $script:CapturedRequest.PSObject.Properties.Name | Should -Contain tfsZ
    }
    It 'makes an initial provider request when zero retries are requested' {
        Mock Test-OllamaConnection { $true }
        Mock Get-OllamaModels { [pscustomobject]@{ Name = 'fixture-model' } }
        Mock Invoke-OllamaGenerate { [pscustomobject]@{ message = @{ content = 'fixture answer' }; model = 'fixture-model'; raw = @{} } }
        (Send-OllamaRequest -Prompt fixture -MaxRetries 0).Response | Should -Be 'fixture answer'
        Should -Invoke Invoke-OllamaGenerate -Exactly -Times 1
    }
    It 'preserves cancellation rather than retrying it and replacing its cause' {
        Mock Test-OllamaConnection { $true }
        Mock Get-OllamaModels { [pscustomobject]@{ Name = 'fixture-model' } }
        Mock Invoke-OllamaGenerate { throw [OperationCanceledException]::new('Synthetic caller cancellation') }
        Mock Start-Sleep {}
        $caught = $null
        try { Send-OllamaRequest -Prompt fixture -MaxRetries 3 | Out-Null } catch { $caught = $_ }
        $caught.Exception | Should -BeOfType [OperationCanceledException]
        Should -Invoke Invoke-OllamaGenerate -Exactly -Times 1
        Should -Invoke Start-Sleep -Exactly -Times 0
    }
    It 'parses actual provider metrics and filters other model labels without inventing zero totals' {
        Mock Resolve-PcaiEndpoint { $ApiUrl }
        Mock Invoke-RestMethod {
            @'
# private fixture only
vllm:prompt_tokens_total{model_name="wanted"} 12.5
vllm:generation_tokens_total{model_name="wanted"} 2e1
vllm:prompt_tokens_total{model_name="other"} 999
vllm:request_success_total{model_name="wanted",finished_reason="stop"} 2
vllm:request_success_total{model_name="wanted",finished_reason="length"} 3
vllm:num_requests_running{model_name="wanted"} 1
vllm:num_requests_waiting{model_name="wanted"} 4
vllm:kv_cache_usage_perc{model_name="wanted"} 0.25
'@
        }
        $result = Get-VLLMMetricsSnapshot -ApiUrl 'http://fixture.invalid' -ModelName wanted
        $result.PromptTokensTotal | Should -Be 12.5
        $result.GenerationTokensTotal | Should -Be 20
        $result.RequestSuccessTotal | Should -Be 5
        $result.NumRequestsRunning | Should -Be 1
        $result.KVCacheUsagePerc | Should -Be 0.25
    }
    It 'extracts an embedded JSON object from explanatory model text' {
        $result = ConvertFrom-LLMJson -Content 'Analysis follows: {"status":"ok","value":3} End of answer.' -Strict
        $result.status | Should -Be ok
        $result.value | Should -Be 3
    }
    It 'retains a successful diagnosis answer when the optional native assembly is unavailable' {
        Mock Invoke-NativeSearch { [pscustomobject]@{ Summary = 'private fixture summary' } }
        Mock Get-LLMPromptContext { [pscustomobject]@{ SystemPrompt = 'fixture system' } }
        Mock Invoke-LLMChatWithFallback { [pscustomobject]@{ message = @{ content = '{"status":"ok"}' } } }
        Mock Write-Host {}
        $result = Invoke-SmartDiagnosis -Path $TestDrive -AnalysisType Quick
        $result.LLMAnalysis.status | Should -Be ok
        $result.NativeEngineUsed | Should -BeFalse
        $result.Error | Should -BeNullOrEmpty
    }
    It 'passes actual maintained prompt grounding into the diagnosis provider request' {
        Mock Invoke-NativeSearch { [pscustomobject]@{ Summary = 'private fixture summary' } }
        Mock Get-CimInstance { [pscustomobject]@{ Caption = 'Fixture OS' } }
        Mock Test-Path { $true }
        Mock Get-Content { 'fixture grounding' } -ParameterFilter { $Path -like '*DIAGNOSE.md' }
        Mock Get-Content { 'fixture logic' } -ParameterFilter { $Path -like '*DIAGNOSE_LOGIC.md' }
        Mock Get-Content { '{}' } -ParameterFilter { $Path -like '*DIAGNOSE_TEMPLATE.json' }
        Mock Invoke-LLMChatWithFallback { [pscustomobject]@{ message = @{ content = '{"status":"ok"}' } } }
        Mock Write-Host {}
        Invoke-SmartDiagnosis -Path $TestDrive -AnalysisType Quick | Out-Null
        Should -Invoke Invoke-LLMChatWithFallback -Exactly -Times 1 -ParameterFilter { $Messages[0].content -eq 'fixture grounding' }
    }
    It 'labels offline knowledge as offline instead of claiming retrieved official documentation' {
        $result = Invoke-DocSearch -Query 'ConfigManagerErrorCode 31' -Source Microsoft | ConvertFrom-Json
        $result.Results[0].Title | Should -Not -Match 'Official'
    }
}

Describe 'LLM retry counts and exact cancellation custody' -Tag 'Unit', 'LLM', 'Portable' {
    BeforeEach {
        $script:ModuleConfig = New-LlmContractConfig
        Mock Test-OllamaConnection { $true }
        Mock Get-OllamaModels { [pscustomobject]@{ Name = 'fixture-default' } }
        Mock Invoke-OllamaGenerate { throw 'Synthetic exhausted request' }
        Mock Start-Sleep {}
        Mock Write-Warning {}
    }
    It 'makes the initial attempt plus exactly the requested retry count on exhaustion' -ForEach @(
        @{ Retries = 0; Attempts = 1 }, @{ Retries = 1; Attempts = 2 }, @{ Retries = 3; Attempts = 4 }
    ) {
        { Send-OllamaRequest -Prompt fixture -MaxRetries $Retries } | Should -Throw "*after $Attempts attempts*"
        Should -Invoke Invoke-OllamaGenerate -Exactly -Times $Attempts
        Should -Invoke Start-Sleep -Exactly -Times $Retries
    }
    It 'uses the documented default three retries after the first request' {
        { Send-OllamaRequest -Prompt fixture } | Should -Throw '*after 4 attempts*'
        Should -Invoke Invoke-OllamaGenerate -Exactly -Times 4
        Should -Invoke Start-Sleep -Exactly -Times 3
    }
    It 'preserves an exact wrapped cancellation cause without warning or retry' {
        $script:CancellationCause = [OperationCanceledException]::new('Synthetic wrapped cancellation')
        Mock Invoke-OllamaGenerate { throw [Exception]::new('Fixture wrapper', $script:CancellationCause) }
        $caught = $null
        try { Send-OllamaRequest -Prompt fixture } catch { $caught = $_ }
        [object]::ReferenceEquals($caught.Exception.GetBaseException(), $script:CancellationCause) | Should -BeTrue
        Should -Invoke Invoke-OllamaGenerate -Exactly -Times 1
        Should -Invoke Start-Sleep -Exactly -Times 0
        Should -Invoke Write-Warning -Exactly -Times 0
    }
}

Describe 'LLM finite provider metrics and transport refusal' -Tag 'Unit', 'LLM', 'Portable' {
    BeforeEach {
        $script:ModuleConfig = New-LlmContractConfig
        Mock Resolve-PcaiEndpoint { $ApiUrl }
        Mock Invoke-RestMethod { throw 'Synthetic metrics unavailable' }
    }
    It 'returns unavailable metrics after HTTP failure instead of zero acceptance' {
        Get-VLLMMetricsSnapshot -ApiUrl 'http://fixture.invalid' | Should -BeNullOrEmpty
    }
    It 'rejects malformed and nonfinite recognized samples' -ForEach @(
        @{ Value = 'NaN' }, @{ Value = '+Inf' }, @{ Value = '1e9999' }, @{ Value = '1..2' }, @{ Value = 'bad-value' }
    ) {
        Mock Invoke-RestMethod { "vllm:prompt_tokens_total $Value" }
        Get-VLLMMetricsSnapshot -ApiUrl 'http://fixture.invalid' | Should -BeNullOrEmpty
    }
    It 'rejects unsupported-only input rather than fabricating a measured zero snapshot' {
        Mock Invoke-RestMethod { 'vllm:unsupported_fixture_metric 8' }
        Get-VLLMMetricsSnapshot -ApiUrl 'http://fixture.invalid' | Should -BeNullOrEmpty
    }
    It 'preserves genuine measured zeros and parses CRLF/model labels and scientific notation' {
        Mock Invoke-RestMethod { "vllm:prompt_tokens_total{model_name=`"fixture`"} 0`r`nvllm:generation_tokens_total{model_name=`"fixture`"} 2.5e1`r`n" }
        $result = Get-VLLMMetricsSnapshot -ApiUrl 'http://fixture.invalid' -ModelName fixture
        $result.PromptTokensTotal | Should -Be 0
        $result.GenerationTokensTotal | Should -Be 25
        $result.TokensTotal | Should -Be 25
    }
    It 'rejects finite input samples whose aggregate overflows' -ForEach @(
        @{ Payload = "vllm:prompt_tokens_total 1e308`nvllm:generation_tokens_total 1e308" },
        @{ Payload = "vllm:request_success_total{finished_reason=`"stop`"} 1e308`nvllm:request_success_total{finished_reason=`"length`"} 1e308" }
    ) {
        Mock Invoke-RestMethod { $Payload }
        Get-VLLMMetricsSnapshot -ApiUrl 'http://fixture.invalid' | Should -BeNullOrEmpty
    }
    It 'parses provider fractional values using invariant numeric rules' {
        Mock Invoke-RestMethod { 'vllm:kv_cache_usage_perc 0.25' }
        $previousCulture = [Globalization.CultureInfo]::CurrentCulture
        try {
            [Globalization.CultureInfo]::CurrentCulture = [Globalization.CultureInfo]::GetCultureInfo('fr-FR')
            (Get-VLLMMetricsSnapshot -ApiUrl 'http://fixture.invalid').KVCacheUsagePerc | Should -Be 0.25
        } finally { [Globalization.CultureInfo]::CurrentCulture = $previousCulture }
    }
}

Describe 'LLM JSON and offline provenance compatibility' -Tag 'Unit', 'LLM', 'Portable' {
    It 'parses nested, quoted-brace and array JSON without including surrounding explanation' -ForEach @(
        @{ Payload = 'prefix {"status":"ok","nested":{"value":3},"text":"} [ quoted"} suffix'; Property = 'status'; Expected = 'ok' },
        @{ Payload = 'prefix [{"status":"ok"}] suffix'; Property = 'status'; Expected = 'ok' },
        @{ Payload = '```json{"status":"ok"}```'; Property = 'status'; Expected = 'ok' }
    ) {
        $result = ConvertFrom-LLMJson -Content $Payload -Strict
        @($result)[0].$Property | Should -Be $Expected
    }
    It 'retains malformed strict JSON rejection and non-strict raw-content compatibility' {
        Mock Write-Warning {}
        { ConvertFrom-LLMJson -Content 'prefix {bad json} suffix' -Strict } | Should -Throw '*Failed to parse JSON*'
        ConvertFrom-LLMJson -Content 'prefix {bad json} suffix' | Should -BeExactly 'prefix {bad json} suffix'
    }
    It 'does not imply web retrieval for an offline knowledge fragment' {
        $result = Invoke-DocSearch -Query 'ConfigManagerErrorCode 31' -Source Microsoft | ConvertFrom-Json
        $result.RetrievalPerformed | Should -BeFalse
        $result.Results[0].Provenance | Should -Be OfflineKnowledge
        $result.Source | Should -Be Microsoft
        $result.Url | Should -Match '^https://learn.microsoft.com/'
    }
}

Describe 'LLM diagnosis optional fields under strict callers' -Tag 'Unit', 'LLM', 'Portable' {
    BeforeEach {
        $script:ModuleConfig = New-LlmContractConfig
        Mock Invoke-NativeSearch { [pscustomobject]@{ Summary = 'fixture summary'; ElapsedMs = 2; Engine = 'PowerShell' } }
        Mock Invoke-LLMChatWithFallback { [pscustomobject]@{ message = @{ content = '{"status":"ok"}' } } }
        Mock ConvertFrom-LLMJson { [pscustomobject]@{ status = 'ok' } }
        Mock Write-Host {}
        Mock Write-Warning {}
    }
    It 'retains legacy SystemPrompt-only context under an isolated strict scope' {
        Mock Get-LLMPromptContext { [pscustomobject]@{ SystemPrompt = 'fixture legacy grounding' } }
        $result = & {
            Set-StrictMode -Version Latest
            try { Invoke-SmartDiagnosis -Path $TestDrive -AnalysisType Quick }
            finally { Set-StrictMode -Off }
        }
        $result.LLMAnalysis.status | Should -Be ok
        Should -Invoke Invoke-LLMChatWithFallback -Exactly -Times 1 -ParameterFilter { $Messages[0].content -eq 'fixture legacy grounding' }
    }
    It 'retains successful diagnosis with mixed collection entries whose optional Engine is absent' {
        Mock Get-LLMPromptContext { [pscustomobject]@{ Prompts = @{ System = 'fixture maintained grounding' } } }
        Mock Invoke-NativeSearch { [pscustomobject]@{ Summary = 'legacy fixture summary'; ElapsedMs = 2 } } -ParameterFilter { $Operation -eq 'Files' }
        Mock Build-DiagnosticSummary { 'fixture collected summary' }
        $result = & {
            Set-StrictMode -Version Latest
            try { Invoke-SmartDiagnosis -Path $TestDrive -AnalysisType Quick }
            finally { Set-StrictMode -Off }
        }
        $result.LLMAnalysis.status | Should -Be ok
        $result.NativeEngineUsed | Should -BeFalse
    }
    It 'formats incomplete dictionary entries under strict callers without inventing metrics' {
        $summary = & {
            Set-StrictMode -Version Latest
            try { Build-DiagnosticSummary -DiagnosticData @{ Results = @{ Legacy = @{ Summary = 'fixture summary' } } } }
            finally { Set-StrictMode -Off }
        }
        $summary | Should -Match 'fixture summary'
        $summary | Should -Not -Match 'Elapsed: 0'
    }
}
