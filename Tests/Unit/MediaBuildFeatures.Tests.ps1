#Requires -Version 7.0
#Requires -Modules @{ ModuleName = 'Pester'; ModuleVersion = '5.0.0' }

BeforeAll {
    $buildScript = Join-Path (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent) 'Build.ps1'
    $tokens = $null
    $errors = $null
    $ast = [Management.Automation.Language.Parser]::ParseFile($buildScript, [ref]$tokens, [ref]$errors)
    if ($errors.Count) { throw 'Build.ps1 must parse before testing its actual feature selector.' }
    $selector = $ast.Find({ param($node)
        $node -is [Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq 'Get-MediaBuildFeatures'
    }, $true)
    . ([scriptblock]::Create($selector.Extent.Text))
}

Describe 'Host-qualified media build features' {
    It 'builds CPU without introducing a GPU dependency' {
        $features = Get-MediaBuildFeatures -Cuda $false -Cudnn $false -FlashAttention $false
        $features.Library | Should -Be @('ffi', 'upscale')
        @($features.Server).Count | Should -Be 0
    }

    It 'builds plain CUDA without requiring optional kernel libraries' {
        $features = Get-MediaBuildFeatures -Cuda $true -Cudnn $false -FlashAttention $false
        $features.Library | Should -Be @('ffi', 'upscale', 'cuda', 'nvml')
        $features.Server | Should -Be @('cuda', 'nvml')
    }

    It 'propagates each explicit optional kernel to library and server' -ForEach @(
        @{ Cudnn = $true; Flash = $false; Expected = 'cudnn'; Absent = 'flash-attn' }
        @{ Cudnn = $false; Flash = $true; Expected = 'flash-attn'; Absent = 'cudnn' }
    ) {
        $features = Get-MediaBuildFeatures -Cuda $true -Cudnn $Cudnn -FlashAttention $Flash
        $features.Library | Should -Contain $Expected
        $features.Server | Should -Contain $Expected
        $features.Library | Should -Not -Contain $Absent
        $features.Server | Should -Not -Contain $Absent
    }

    It 'rejects optional GPU kernels for CPU builds' -ForEach @(
        @{ Cudnn = $true; Flash = $false }, @{ Cudnn = $false; Flash = $true }
    ) {
        { Get-MediaBuildFeatures -Cuda $false -Cudnn $Cudnn -FlashAttention $Flash } | Should -Throw '*require CUDA*'
    }
}
