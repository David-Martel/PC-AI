#Requires -Version 7.0

BeforeAll {
    $script:ProjectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
    $script:BuildLogsDir = $TestDrive
    function Write-BuildStep {}
    function Invoke-RustBuildCommand {
        param([string]$Path, [string[]]$CargoArgs, [string]$LogFile)
        $script:CapturedCargoArgs = $CargoArgs
        return $true
    }
    $parseErrors = $null
    $buildAst = [System.Management.Automation.Language.Parser]::ParseFile(
        (Join-Path $script:ProjectRoot 'Build.ps1'), [ref]$null, [ref]$parseErrors
    )
    if ($parseErrors) { throw ($parseErrors | Out-String) }
    $operation = $buildAst.Find({
            param($node)
            $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
            $node.Name -eq 'Invoke-FunctionGemmaOperation'
        }, $false)
    . ([scriptblock]::Create($operation.Extent.Text))
    $profile = $buildAst.Find({
            param($node)
            $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
            $node.Name -eq 'Get-FunctionGemmaCargoArguments'
        }, $false)
    . ([scriptblock]::Create($profile.Extent.Text))
}

Describe 'FunctionGemma build and test profiles' -Tag 'Unit', 'Portable' {
    # Protects: both build and requested tests select the same explicit runtime and trainer device profile.
    # Detects: CUDA pulled into CPU builds, heuristic-only packaged runtime, or loss of the explicit CUDA route.
    # Needs: PowerShell and Pester 5; no Cargo or GPU. Breadcrumb: Build.ps1 Get-FunctionGemmaCargoArguments.
    It 'uses <Project> <Mode> features for both builds and tests' -ForEach @(
        @{ Project = 'runtime'; Mode = 'CPU'; WithCuda = $false; Features = @('--features', 'model') },
        @{ Project = 'runtime'; Mode = 'CUDA'; WithCuda = $true; Features = @('--features', 'cuda') },
        @{ Project = 'trainer'; Mode = 'CPU'; WithCuda = $false; Features = @() },
        @{ Project = 'trainer'; Mode = 'CUDA'; WithCuda = $true; Features = @('--features', 'cuda,flash-attn') }
    ) {
        foreach ($command in @('build', 'test')) {
            $expected = @($command, '--locked', '--no-default-features', '--release') + $Features
            $actual = Get-FunctionGemmaCargoArguments -Command $command -Project $Project -Configuration Release -EnableCuda:$WithCuda
            ($actual -join '|') | Should -BeExactly ($expected -join '|')
        }
    }
}

Describe 'FunctionGemma build operation routing' -Tag 'Unit', 'Portable' {
    # Protects: locked CPU preparation and existing train/eval defaults.
    # Detects: reactivating CUDA for data preparation or changing forwarded arguments.
    # Needs: PowerShell and Pester 5; the Cargo wrapper is a capture double.
    # Breadcrumb: Build.ps1 Invoke-FunctionGemmaOperation.
    It 'selects Cargo arguments for <Operation>' -ForEach @(
        @{ Operation = 'router-data'; Expected = @('run', '--locked', '--no-default-features', '--', 'prepare-router') }
        @{ Operation = 'token-cache'; Expected = @('run', '--locked', '--no-default-features', '--', 'prepare-cache') }
        @{ Operation = 'train'; Expected = @('run', '--') }
        @{ Operation = 'eval'; Expected = @('run', '--') }
    ) {
        $arguments = @('--fixture', 'value with spaces')
        $result = Invoke-FunctionGemmaOperation -Operation $Operation -OperationArgs $arguments
        $result.Success | Should -BeTrue
        ($script:CapturedCargoArgs -join '|') | Should -BeExactly (($Expected + $arguments) -join '|')
    }
}
