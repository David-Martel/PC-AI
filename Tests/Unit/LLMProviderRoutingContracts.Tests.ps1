#Requires -Version 7.0
<#
Exercises normally imported routing, history and OpenAI request construction.
Only health, HTTP, native Ollama invocation and router boundaries are mocked.
No server, credentials, model, native library or progress child is used.
#>
BeforeAll {
    $script:RoutingEnvironment = @{}
    foreach ($name in @('PCAI_CACHE_PROVIDER', 'PCAI_REDIS_KEY_PREFIX')) {
        $script:RoutingEnvironment[$name] = [Environment]::GetEnvironmentVariable($name)
    }
    $env:PCAI_CACHE_PROVIDER = 'memory'
    $env:PCAI_REDIS_KEY_PREFIX = 'pcai-routing-owned-' + [guid]::NewGuid().ToString('N') + ':'
    Import-Module (Join-Path $PSScriptRoot '../../Modules/PC-AI.LLM/PC-AI.LLM.psd1') -Force -ErrorAction Stop
}

AfterAll {
    foreach ($entry in $script:RoutingEnvironment.GetEnumerator()) {
        [Environment]::SetEnvironmentVariable($entry.Key, $entry.Value)
    }
}

Describe 'Normal LLM provider routing and wire contracts' -Tag 'Unit', 'LLM', 'Portable' {
    BeforeEach {
        InModuleScope PC-AI.LLM {
            $script:RoutingSavedConfig = $script:ModuleConfig
            $script:ModuleConfig = $script:ModuleConfig.Clone()
            $script:ModuleConfig.DefaultModel = 'owned-default-model'
            $script:ModuleConfig.VLLMModel = 'owned-vllm-model'
            $script:ModuleConfig.ProviderOrder = @('pcai-inference', 'vllm', 'lmstudio')
            $script:ModuleConfig.PcaiInferenceApiUrl = 'https://pcai.invalid'
            $script:ModuleConfig.VLLMApiUrl = 'https://vllm.invalid'
            $script:ModuleConfig.LMStudioApiUrl = 'https://studio.invalid'
            $script:ModuleConfig.OllamaApiUrl = 'https://ollama.invalid'
        }
        $script:RoutingCalls = [Collections.Generic.List[object]]::new()
        Mock Get-CachedProviderHealth -ModuleName PC-AI.LLM { $true }
        Mock Invoke-RestMethod -ModuleName PC-AI.LLM {
            param($Uri, $Method, $Body, $Headers, $ContentType, $TimeoutSec, $ConnectionTimeoutSeconds)
            $script:RoutingCalls.Add(@{
                Uri = [string]$Uri; Method = [string]$Method
                Body = $Body | ConvertFrom-Json -Depth 20
                # Pester binds the cmdlet's canonical parameter name. In newer
                # PowerShell, TimeoutSec is an alias for ConnectionTimeoutSeconds.
                Headers = $Headers; ContentType = $ContentType
                Timeout = if ($null -ne $TimeoutSec) { $TimeoutSec } else { $ConnectionTimeoutSeconds }
            })
            [pscustomobject]@{ choices = @([pscustomobject]@{ message = [pscustomobject]@{ content = 'wire response' } }) }
        }
        Mock Invoke-OllamaNativeChat -ModuleName PC-AI.LLM { throw 'Unexpected native Ollama request.' }
        Mock Invoke-OpenAIChatWithProgress -ModuleName PC-AI.LLM { throw 'Unexpected progress child.' }
        Mock Invoke-OpenAIChatStream -ModuleName PC-AI.LLM { throw 'Unexpected streaming transport.' }
        Mock Invoke-LLMChatRouted -ModuleName PC-AI.LLM { throw 'Unexpected router boundary.' }
    }

    AfterEach {
        InModuleScope PC-AI.LLM {
            $script:ModuleConfig = $script:RoutingSavedConfig
            Remove-Variable RoutingSavedConfig -Scope Script -ErrorAction SilentlyContinue
        }
    }

    # Protects: explicit provider endpoint and request identity. Detects: wrong route/model or lost options.
    # Needs: canonical public call and captured HTTP boundary. Breadcrumb: Invoke-LLMChat -> real OpenAI builder.
    It 'sends an explicit <Selected> request with preserved message and options' -ForEach @(
        @{ Selected = 'pcai-inference'; Endpoint = 'https://pcai.invalid/v1/chat/completions' }
        @{ Selected = 'vllm'; Endpoint = 'https://vllm.invalid/v1/chat/completions' }
        @{ Selected = 'lmstudio'; Endpoint = 'https://studio.invalid/v1/chat/completions' }
    ) {
        $result = Invoke-LLMChat -Message 'owned message' -Model 'requested-model' -Provider $Selected -Temperature 0.25 -MaxTokens 17 -TimeoutSeconds 23
        $result.Response | Should -Be 'wire response'
        $result.RawResponse | Should -Be 'wire response'
        $result.Model | Should -Be 'requested-model'
        $result.History.Count | Should -Be 2
        $result.History[0].role | Should -Be 'user'
        $result.History[1].role | Should -Be 'assistant'
        InModuleScope PC-AI.LLM -Parameters @{ ExpectedEndpoint = $Endpoint; Calls = $script:RoutingCalls } {
            $Calls.Count | Should -Be 1
            $call = $Calls[0]
            $call.Uri | Should -Be $ExpectedEndpoint
            $call.Method | Should -Be 'Post'
            $call.ContentType | Should -Be 'application/json'
            $call.Timeout | Should -Be 23
            $call.Headers.Count | Should -Be 0
            $call.Body.model | Should -Be 'requested-model'
            $call.Body.temperature | Should -Be 0.25
            $call.Body.max_tokens | Should -Be 17
            $call.Body.stream | Should -BeFalse
            $call.Body.messages.Count | Should -Be 1
            $call.Body.messages[0].content | Should -Be 'owned message'
        }
        Should -Invoke Get-CachedProviderHealth -ModuleName PC-AI.LLM -Times 1 -Exactly -ParameterFilter { $Provider -eq $Selected -and $TimeoutSeconds -eq 10 }
        Should -Invoke Invoke-OllamaNativeChat -ModuleName PC-AI.LLM -Times 0 -Exactly
    }

    # Protects: caller-owned history and ordering. Detects: missing system/previous messages or in-place append.
    # Needs: two preexisting history rows. Breadcrumb: real public conversation construction.
    It 'preserves caller history while placing system and new messages in order' {
        $history = @(@{ role = 'user'; content = 'previous question' }, @{ role = 'assistant'; content = 'previous reply' })
        $before = $history | ConvertTo-Json -Depth 5 -Compress
        $result = Invoke-LLMChat -Message 'next question' -System 'system rule' -History $history -Provider vllm
        ($history | ConvertTo-Json -Depth 5 -Compress) | Should -Be $before
        ($result.History.role -join ',') | Should -Be 'system,user,assistant,user,assistant'
        ($script:RoutingCalls[0].Body.messages.content -join '|') | Should -Be 'system rule|previous question|previous reply|next question'
        @($script:RoutingCalls[0].Body.PSObject.Properties.Name) | Should -Not -Contain 'max_tokens'
    }

    # Protects: reachable-provider fallback. Detects: request to unhealthy provider or incorrect timeout cap.
    # Needs: mixed health boundary; request builder remains canonical. Breadcrumb: public auto route.
    It 'skips an unhealthy first provider and sends only to the next reachable endpoint' {
        Mock Get-CachedProviderHealth -ModuleName PC-AI.LLM { $Provider -ne 'pcai-inference' }
        $result = Invoke-LLMChat -Message 'fallback request' -Provider auto -TimeoutSeconds 4
        $result.Response | Should -Be 'wire response'
        $script:RoutingCalls.Count | Should -Be 1
        $script:RoutingCalls[0].Uri | Should -Be 'https://vllm.invalid/v1/chat/completions'
        $script:RoutingCalls[0].Timeout | Should -Be 4
        Should -Invoke Get-CachedProviderHealth -ModuleName PC-AI.LLM -Times 2 -Exactly -ParameterFilter { $TimeoutSeconds -eq 4 }
    }

    # Protects: useful unavailable-provider failure. Detects: false success or transport despite failed health.
    # Needs: all health checks negative. Breadcrumb: configured fallback exhaustion.
    It 'fails explicitly without transport when all configured providers are unavailable' {
        Mock Get-CachedProviderHealth -ModuleName PC-AI.LLM { $false }
        { Invoke-LLMChat -Message 'unavailable request' -Provider auto } | Should -Throw '*No LLM providers are reachable*'
        Should -Invoke Invoke-RestMethod -ModuleName PC-AI.LLM -Times 0 -Exactly
        Should -Invoke Get-CachedProviderHealth -ModuleName PC-AI.LLM -Times 3 -Exactly
    }

    # Protects: legacy aliases and uniqueness. Detects: duplicate or invalid health probes.
    # Needs: alias, duplicate and unsupported entries. Breadcrumb: real helper's normalization.
    It 'normalizes aliases and removes duplicates before probing providers' {
        Mock Get-CachedProviderHealth -ModuleName PC-AI.LLM { $Provider -eq 'vllm' }
        InModuleScope PC-AI.LLM {
            $script:ModuleConfig.ProviderOrder = @('pcai-native', 'ollama', 'unknown', 'functiongemma', 'vllm')
            $result = Invoke-LLMChatWithFallback -Messages @(@{ role = 'user'; content = 'alias request' })
            $result.Provider | Should -Be 'vllm'
        }
        $script:RoutingCalls[0].Body.model | Should -Be 'owned-vllm-model'
        Should -Invoke Get-CachedProviderHealth -ModuleName PC-AI.LLM -Times 2 -Exactly
        Should -Invoke Get-CachedProviderHealth -ModuleName PC-AI.LLM -Times 1 -Exactly -ParameterFilter { $Provider -eq 'ollama' }
        Should -Invoke Get-CachedProviderHealth -ModuleName PC-AI.LLM -Times 1 -Exactly -ParameterFilter { $Provider -eq 'vllm' }
    }

    # Protects: explicit native boundary. Detects: wrong options or unintended HTTP fallback.
    # Needs: canonical Ollama wrapper with native invocation mocked. Breadcrumb: public ollama route.
    It 'forwards Ollama options once without using an HTTP endpoint' {
        Mock Invoke-OllamaNativeChat -ModuleName PC-AI.LLM {
            [pscustomobject]@{ message = [pscustomobject]@{ content = 'native boundary response' }; model = 'native-model'; total_duration = 123 }
        }
        $result = Invoke-LLMChat -Message 'ollama request' -Provider ollama -Model 'native-model' -Temperature 0.4 -MaxTokens 9 -TimeoutSeconds 7
        $result.Response | Should -Be 'native boundary response'
        $result.Model | Should -Be 'native-model'
        $result.TotalDuration | Should -Be 123
        Should -Invoke Invoke-OllamaNativeChat -ModuleName PC-AI.LLM -Times 1 -Exactly -ParameterFilter { $Model -eq 'native-model' -and $Temperature -eq 0.4 -and $MaxTokens -eq 9 -and $TimeoutSeconds -eq 7 -and $Messages[0].content -eq 'ollama request' }
        Should -Invoke Invoke-RestMethod -ModuleName PC-AI.LLM -Times 0 -Exactly
    }

    # Protects: router delegation. Detects: missing mode/options or accidental second inference request.
    # Needs: canonical public dispatch and router boundary. Breadcrumb: UseRouter noninteractive branch.
    It 'delegates router mode and options without direct provider transport' {
        Mock Invoke-LLMChatRouted -ModuleName PC-AI.LLM { [pscustomobject]@{ Response = 'router response'; Route = 'fixture' } }
        $result = Invoke-LLMChat -Message 'routed request' -UseRouter -RouterMode diagnose -Provider lmstudio -Model 'routed-model' -TimeoutSeconds 19 -Temperature 0.6
        $result.Response | Should -Be 'router response'
        Should -Invoke Invoke-LLMChatRouted -ModuleName PC-AI.LLM -Times 1 -Exactly -ParameterFilter { $Message -eq 'routed request' -and $Mode -eq 'diagnose' -and $Provider -eq 'lmstudio' -and $Model -eq 'routed-model' -and $TimeoutSeconds -eq 19 -and $Temperature -eq 0.6 }
        Should -Invoke Invoke-RestMethod -ModuleName PC-AI.LLM -Times 0 -Exactly
        Should -Invoke Get-CachedProviderHealth -ModuleName PC-AI.LLM -Times 0 -Exactly
    }

    # Protects: response normalization without invented content. Detects: index error or stale response reuse.
    # Needs: legitimate empty choice array. Breadcrumb: real OpenAI response mapping.
    It 'returns a null response for an empty completion array without fabricating text' {
        Mock Invoke-RestMethod -ModuleName PC-AI.LLM { [pscustomobject]@{ choices = @() } }
        $result = Invoke-LLMChat -Message 'empty completion' -Provider lmstudio
        $result.Response | Should -BeNullOrEmpty
        $result.History.Count | Should -Be 2
        $result.History[1].content | Should -BeNullOrEmpty
    }

    # Protects: transport error visibility. Detects: false fallback/success after a failed request.
    # Needs: healthy selection followed by failed HTTP boundary. Breadcrumb: real request catch.
    It 'surfaces a selected-provider transport failure without returning a false success' {
        Mock Invoke-RestMethod -ModuleName PC-AI.LLM { throw 'owned transport failure' }
        { Invoke-LLMChat -Message 'failed request' -Provider vllm -ErrorAction Stop } | Should -Throw '*owned transport failure*'
        Should -Invoke Invoke-RestMethod -ModuleName PC-AI.LLM -Times 1 -Exactly
    }
}
