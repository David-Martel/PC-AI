<#
.SYNOPSIS
    Exercises provider-order persistence using isolated, real JSON files.
#>

BeforeAll {
    $modulePath = Join-Path $PSScriptRoot '../../Modules/PC-AI.LLM/PC-AI.LLM.psd1'
    $script:RepositoryConfigPath = Join-Path $PSScriptRoot '../../Config/llm-config.json'
    $script:RepositoryConfigHash = (Get-FileHash -LiteralPath $script:RepositoryConfigPath).Hash
    Import-Module $modulePath -Force -ErrorAction Stop
    $script:OriginalModuleConfig = InModuleScope PC-AI.LLM { $script:ModuleConfig.Clone() }
}

AfterAll {
    InModuleScope PC-AI.LLM -Parameters @{ Original = $script:OriginalModuleConfig } {
        $script:ModuleConfig = $Original
    }
    (Get-FileHash -LiteralPath $script:RepositoryConfigPath).Hash | Should -Be $script:RepositoryConfigHash
    Remove-Module PC-AI.LLM -Force -ErrorAction SilentlyContinue
}

Describe 'Set-LLMProviderOrder' -Tag 'Unit', 'LLM', 'Fast', 'Portable' {
    BeforeEach {
        $script:ConfigPath = Join-Path $TestDrive 'llm-config.json'
        @'
{"fallbackOrder":["ollama"],"providers":{"ollama":{"defaultModel":"machine-model","timeout":777}},"ollama":{"num_gpu":2,"num_ctx":16384,"tool_model":"machine-tools"},"machineSetting":{"preserve":true}}
'@ | Set-Content -LiteralPath $script:ConfigPath -Encoding utf8NoBOM
        InModuleScope PC-AI.LLM -Parameters @{ ConfigPath = $script:ConfigPath; Original = $script:OriginalModuleConfig } {
            $script:ModuleConfig = $Original.Clone()
            $script:ModuleConfig.ConfigPath = $ConfigPath
            $script:ModuleConfig.ProjectConfigPath = $ConfigPath
            $script:ModuleConfig.ProviderOrder = @('ollama')
        }
        $script:BeforeHash = (Get-FileHash -LiteralPath $script:ConfigPath).Hash
    }

    It 'updates disk and memory while preserving machine configuration and BOM-free JSON' {
        $result = Set-LLMProviderOrder -Order @('pcai-inference', 'ollama')
        $config = Get-Content -LiteralPath $script:ConfigPath -Raw | ConvertFrom-Json
        $result.Success | Should -BeTrue
        $result.Order -join ',' | Should -Be 'pcai-inference,ollama'
        $result.ConfigPath | Should -Be $script:ConfigPath
        $config.fallbackOrder -join ',' | Should -Be 'pcai-inference,ollama'
        $config.providers.ollama.defaultModel | Should -Be 'machine-model'
        $config.providers.ollama.timeout | Should -Be 777
        $config.ollama.num_gpu | Should -Be 2
        $config.ollama.num_ctx | Should -Be 16384
        $config.ollama.tool_model | Should -Be 'machine-tools'
        $config.machineSetting.preserve | Should -BeTrue
        [System.IO.File]::ReadAllBytes($script:ConfigPath)[0] | Should -Be 123
        InModuleScope PC-AI.LLM { $script:ModuleConfig.ProviderOrder -join ',' } | Should -Be 'pcai-inference,ollama'
    }

    It 'supports every runtime provider and legacy alias' {
        $order = @('ollama', 'pcai-inference', 'vllm', 'lmstudio', 'pcai-native', 'functiongemma')
        Set-LLMProviderOrder -Order $order | Out-Null
        (Get-Content -LiteralPath $script:ConfigPath -Raw | ConvertFrom-Json).fallbackOrder -join ',' | Should -Be ($order -join ',')
    }

    It 'adds fallbackOrder when absent' {
        '{"providers":{}}' | Set-Content -LiteralPath $script:ConfigPath -Encoding utf8NoBOM
        Set-LLMProviderOrder -Order @('vllm', 'ollama') | Out-Null
        (Get-Content -LiteralPath $script:ConfigPath -Raw | ConvertFrom-Json).fallbackOrder -join ',' | Should -Be 'vllm,ollama'
    }

    It 'rejects <Label> without changing disk or memory' -TestCases @(
        @{ Label = 'unknown provider'; Order = @('invalid') }
        @{ Label = 'mixed known and unknown providers'; Order = @('ollama', 'invalid') }
        @{ Label = 'whitespace provider'; Order = @('ollama', ' ') }
        @{ Label = 'empty provider'; Order = @('ollama', '') }
        @{ Label = 'null order'; Order = $null }
        @{ Label = 'empty order'; Order = @() }
    ) {
        param($Order)
        { Set-LLMProviderOrder -Order $Order -ErrorAction Stop } | Should -Throw
        (Get-FileHash -LiteralPath $script:ConfigPath).Hash | Should -Be $script:BeforeHash
        InModuleScope PC-AI.LLM { $script:ModuleConfig.ProviderOrder -join ',' } | Should -Be 'ollama'
    }

    It 'leaves disk and memory unchanged under WhatIf' {
        Set-LLMProviderOrder -Order @('vllm') -WhatIf
        (Get-FileHash -LiteralPath $script:ConfigPath).Hash | Should -Be $script:BeforeHash
        InModuleScope PC-AI.LLM { $script:ModuleConfig.ProviderOrder -join ',' } | Should -Be 'ollama'
    }

    It 'does not change memory or create files when the config is missing' {
        $missingPath = Join-Path $TestDrive 'missing-llm-config.json'
        InModuleScope PC-AI.LLM -Parameters @{ ConfigPath = $missingPath } {
            $script:ModuleConfig.ConfigPath = $ConfigPath
            $script:ModuleConfig.ProjectConfigPath = $ConfigPath
        }
        { Set-LLMProviderOrder -Order @('pcai-inference') -ErrorAction Stop } | Should -Throw "Config file not found: $missingPath"
        Test-Path -LiteralPath $missingPath | Should -BeFalse
        InModuleScope PC-AI.LLM { $script:ModuleConfig.ProviderOrder -join ',' } | Should -Be 'ollama'
    }

    It 'leaves malformed JSON and memory unchanged' {
        '{broken' | Set-Content -LiteralPath $script:ConfigPath -Encoding utf8NoBOM
        $before = (Get-FileHash -LiteralPath $script:ConfigPath).Hash
        { Set-LLMProviderOrder -Order @('pcai-inference') -ErrorAction Stop } | Should -Throw
        (Get-FileHash -LiteralPath $script:ConfigPath).Hash | Should -Be $before
        InModuleScope PC-AI.LLM { $script:ModuleConfig.ProviderOrder -join ',' } | Should -Be 'ollama'
    }

    It 'does not publish the new memory order when a real file write fails' {
        $fixtureJson = Get-Content -LiteralPath $script:ConfigPath -Raw
        Mock Get-Content { $FixtureJson } -ModuleName PC-AI.LLM
        # Bypass only the read to reach the real writer against an exclusive lock.
        $handle = [System.IO.File]::Open($script:ConfigPath, 'Open', 'ReadWrite', 'None')
        try {
            $failure = { Set-LLMProviderOrder -Order @('pcai-inference') -ErrorAction Stop } | Should -Throw -PassThru
            $failure.Exception.GetBaseException() | Should -BeOfType ([System.IO.IOException])
            $failure.Exception.Message | Should -Match 'WriteAllText'
            InModuleScope PC-AI.LLM { $script:ModuleConfig.ProviderOrder -join ',' } | Should -Be 'ollama'
        } finally {
            $handle.Dispose()
        }
        (Get-FileHash -LiteralPath $script:ConfigPath).Hash | Should -Be $script:BeforeHash
    }
}
