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
