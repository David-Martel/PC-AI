# Tests/Unit/ProviderHealthCache.Tests.ps1
BeforeAll {
    $ModuleRoot = Join-Path $PSScriptRoot '..\..\Modules\PC-AI.LLM'
    . (Join-Path $ModuleRoot 'Private\ProviderHealthCache.ps1')
}

Describe 'ProviderHealthCache' -Tag 'Unit', 'Portable' {
    BeforeEach {
        Reset-ProviderHealthCache
    }

    Context 'Cache operations' {
        It 'Should cache health check results' {
            Set-ProviderHealthCache -Provider 'ollama' -IsHealthy $true
            $result = Get-ProviderHealthCache -Provider 'ollama'
            $result.IsHealthy | Should -BeTrue
        }

        It 'Should cache FunctionGemma health results' {
            Set-ProviderHealthCache -Provider 'functiongemma' -IsHealthy $true
            $result = Get-ProviderHealthCache -Provider 'functiongemma'
            $result.IsHealthy | Should -BeTrue
        }

        It 'Should return null for uncached providers' {
            $result = Get-ProviderHealthCache -Provider 'unknown'
            $result | Should -BeNull
        }

        It 'Should expire cache after TTL' {
            Set-ProviderHealthCache -Provider 'vllm' -IsHealthy $true
            # Simulate time passing by manipulating the cache
            $script:ProviderHealthCache.Results['vllm'].CachedAt = (Get-Date).AddSeconds(-60)
            $result = Get-ProviderHealthCache -Provider 'vllm'
            $result | Should -BeNull
        }
    }
}
