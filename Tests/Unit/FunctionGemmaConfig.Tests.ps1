#Requires -Version 7.0
#Requires -Modules Pester

Describe 'FunctionGemma runtime config' -Tag 'Unit', 'FunctionGemma', 'Portable' {
    BeforeAll {
        $script:RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
        $script:ConfigPath = Join-Path $script:RepoRoot 'Config\pcai-functiongemma.json'
        $script:RuntimeReadmePath = Join-Path $script:RepoRoot 'Deploy\rust-functiongemma-runtime\README.md'
        $script:Config = Get-Content -LiteralPath $script:ConfigPath -Raw | ConvertFrom-Json
    }

    It 'uses an existing repo-local model directory' {
        $modelPath = $script:Config.runtime.router_model_path
        $modelPath | Should -Not -BeNullOrEmpty
        [System.IO.Path]::IsPathFullyQualified($modelPath) | Should -BeFalse

        $resolvedModelPath = Join-Path $script:RepoRoot $modelPath
        Test-Path -LiteralPath $resolvedModelPath -PathType Container | Should -BeTrue
        Test-Path -LiteralPath (Join-Path $resolvedModelPath 'config.json') -PathType Leaf | Should -BeTrue
        Test-Path -LiteralPath (Join-Path $resolvedModelPath 'tokenizer.json') -PathType Leaf | Should -BeTrue
        Test-Path -LiteralPath (Join-Path $resolvedModelPath 'model.safetensors') -PathType Leaf | Should -BeTrue
    }

    It 'uses an existing tool schema path' {
        $toolsPath = $script:Config.runtime.tools_path
        $toolsPath | Should -Not -BeNullOrEmpty
        Test-Path -LiteralPath (Join-Path $script:RepoRoot $toolsPath) -PathType Leaf | Should -BeTrue
    }

    It 'does not document the stale user-profile repo path' {
        $readme = Get-Content -LiteralPath $script:RuntimeReadmePath -Raw
        $readme | Should -Not -Match 'C:\\Users\\david\\PC_AI\\Models\\functiongemma-270m-it'
        $readme | Should -Match 'Models\\functiongemma-270m-it'
    }
}
