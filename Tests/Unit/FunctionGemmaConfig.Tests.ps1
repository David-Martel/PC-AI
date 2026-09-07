#Requires -Version 7.0
#Requires -Modules Pester

Describe 'FunctionGemma runtime config' -Tag 'Unit', 'FunctionGemma', 'Portable' {
    # Resolved at DISCOVERY time, because -Skip is evaluated while Pester builds
    # the tree -- anything set in BeforeAll does not exist yet at that point.
    $discoveryRepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
    $discoveryModelPath = Join-Path $discoveryRepoRoot (
        (Get-Content -LiteralPath (Join-Path $discoveryRepoRoot 'Config\pcai-functiongemma.json') -Raw |
            ConvertFrom-Json).runtime.router_model_path)
    $modelPresent = Test-Path -LiteralPath $discoveryModelPath -PathType Container

    BeforeAll {
        $script:RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
        $script:ConfigPath = Join-Path $script:RepoRoot 'Config\pcai-functiongemma.json'
        $script:RuntimeReadmePath = Join-Path $script:RepoRoot 'Deploy\rust-functiongemma-runtime\README.md'
        $script:Config = Get-Content -LiteralPath $script:ConfigPath -Raw | ConvertFrom-Json
    }

    # The config contract itself holds everywhere and must never be skipped.
    It 'declares a relative repo-local model path' {
        $modelPath = $script:Config.runtime.router_model_path
        $modelPath | Should -Not -BeNullOrEmpty
        [System.IO.Path]::IsPathFullyQualified($modelPath) | Should -BeFalse
    }

    # The weights are a different matter: Models/ is gitignored and nothing under
    # it is tracked, so this directory cannot exist in a fresh checkout. Asserting
    # its presence made the test guaranteed-red on CI and it always had been --
    # invisible only because this job never got far enough to run. An absent
    # optional model must read as "not exercised", not as "broken".
    It 'has the router model weights on disk' -Skip:(-not $modelPresent) {
        $resolvedModelPath = Join-Path $script:RepoRoot $script:Config.runtime.router_model_path
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
