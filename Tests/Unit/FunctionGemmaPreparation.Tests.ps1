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
    $script:BuildAst = $buildAst
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
            $expected = @($command, '--no-default-features', '--release') + $Features
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

Describe 'FunctionGemma build entry-point forwarding' -Tag 'Unit', 'Portable', 'FunctionGemmaForwarding' {
    BeforeAll {
        foreach ($name in 'Invoke-FunctionGemmaBuild', 'Invoke-PostBuildTests', 'Add-CargoDependencyFlags') {
            $definition = $script:BuildAst.Find({
                param($node)
                $node -is [Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq $name
            }, $false)
            if (-not $definition) { throw "Missing build function $name" }
            . ([scriptblock]::Create($definition.Extent.Text))
        }

        $statements = $script:BuildAst.EndBlock.Statements
        $targetSelection = $statements | Where-Object {
            $_ -is [Management.Automation.Language.AssignmentStatementAst] -and
                $_.Left.Extent.Text -eq '$buildTargets'
        }
        $postTests = $statements | Where-Object {
            $_ -is [Management.Automation.Language.IfStatementAst] -and
                $_.Clauses[0].Item1.Extent.Text -eq '$SkipTests'
        }
        if (@($targetSelection).Count -ne 1 -or @($postTests).Count -ne 1) {
            throw 'Expected one component selector and one post-build test gate.'
        }
        $start = $targetSelection.Extent.StartOffset
        $length = $postTests.Extent.EndOffset - $start
        $script:ForwardingDispatch = [scriptblock]::Create($script:BuildAst.Extent.Text.Substring($start, $length))

        function Write-BuildStep {}
        function Write-BuildPhase {}
        function Write-BuildResult {}
        function New-ForwardingFixtureResult {
            @{ Success = $true; Duration = [timespan]::Zero; Artifacts = @() }
        }
        function Invoke-InferenceBuild { New-ForwardingFixtureResult }
        function Invoke-MediaBuild { New-ForwardingFixtureResult }
        function Invoke-TuiBuild { New-ForwardingFixtureResult }
        function Invoke-DotnetComponentBuild { New-ForwardingFixtureResult }
        function Invoke-NukeNulBuild { New-ForwardingFixtureResult }
        function Resolve-CargoOutputDirectory { $script:ForwardingCargoOutput }
        function Resolve-PowerShellExecutable { 'Invoke-ForwardingMediaFixture' }
        function Invoke-ForwardingMediaFixture { $global:LASTEXITCODE = 0 }
        function Invoke-RustBuildCommand {
            param([string]$Path, [string[]]$CargoArgs)
            $crate = Split-Path $Path -Leaf
            if ($crate -in 'rust-functiongemma-runtime', 'rust-functiongemma-train') {
                $effective = Add-CargoDependencyFlags -CargoArgs $CargoArgs
                $script:ForwardingCalls.Add((@($crate) + $effective -join '|'))
            }
            return $true
        }
    }

    # Protects: both component dispatch and requested post-build tests forward the CUDA and dependency settings.
    # Detects: dropped forwarding at either top-level or helper call, loss of FunctionGemma from all,
    # invoking tests despite SkipTests, or replacing frozen or update with locked.
    # Needs: PowerShell and Pester 5. Actual Build.ps1 dispatch calls inert Cargo and media fixtures.
    # Breadcrumb: Build.ps1 component selection, Invoke-FunctionGemmaBuild and Invoke-PostBuildTests.
    It 'forwards <Mode> and <Policy> through <Target> with SkipTests=<SkipRequested>' -ForEach @(
        @{ Target = 'functiongemma'; Mode = 'CPU';  WithCuda = $false; SkipRequested = $false; Policy = 'locked'; Flag = '|--locked' }
        @{ Target = 'functiongemma'; Mode = 'CUDA'; WithCuda = $true;  SkipRequested = $false; Policy = 'locked'; Flag = '|--locked' }
        @{ Target = 'all';           Mode = 'CPU';  WithCuda = $false; SkipRequested = $false; Policy = 'locked'; Flag = '|--locked' }
        @{ Target = 'all';           Mode = 'CUDA'; WithCuda = $true;  SkipRequested = $false; Policy = 'locked'; Flag = '|--locked' }
        @{ Target = 'all';           Mode = 'CUDA'; WithCuda = $true;  SkipRequested = $true;  Policy = 'locked'; Flag = '|--locked' }
        @{ Target = 'functiongemma'; Mode = 'CPU';  WithCuda = $false; SkipRequested = $false; Policy = 'frozen'; Flag = '|--frozen' }
        @{ Target = 'all';           Mode = 'CUDA'; WithCuda = $true;  SkipRequested = $false; Policy = 'update'; Flag = '' }
    ) {
        $script:ProjectRoot = Join-Path $TestDrive ([guid]::NewGuid().ToString('N'))
        $script:BuildLogsDir = Join-Path $script:ProjectRoot 'logs'
        $script:BuildArtifactsDir = Join-Path $script:ProjectRoot 'artifacts'
        $script:ForwardingCargoOutput = Join-Path $script:ProjectRoot 'cargo'
        foreach ($relative in @('Deploy/rust-functiongemma-runtime', 'Deploy/rust-functiongemma-train', 'cargo')) {
            [void][IO.Directory]::CreateDirectory((Join-Path $script:ProjectRoot $relative))
        }
        $script:ForwardingCalls = [Collections.Generic.List[string]]::new()
        $Component = $Target
        $EnableCuda = $WithCuda
        $Configuration = 'Release'
        $RunTests = $true
        $SkipTests = $SkipRequested
        $priorExitCode = $global:LASTEXITCODE
        $priorStrategy = $script:DependencyStrategy
        try {
            $script:DependencyStrategy = $Policy
            . $script:ForwardingDispatch
        } finally {
            $script:DependencyStrategy = $priorStrategy
            $global:LASTEXITCODE = $priorExitCode
        }

        $failedCount | Should -Be 0
        $results['functiongemma'].Success | Should -BeTrue
        $commands = if ($SkipRequested) { @('build') } else { @('build', 'test') }
        $expected = foreach ($command in $commands) {
            $runtimeFeature = if ($WithCuda) { 'cuda' } else { 'model' }
            $trainerFeatures = if ($WithCuda) { '|--features|cuda,flash-attn' } else { '' }
            "rust-functiongemma-runtime|$command|--no-default-features|--release|--features|$runtimeFeature$Flag"
            "rust-functiongemma-train|$command|--no-default-features|--release$trainerFeatures$Flag"
        }
        ($script:ForwardingCalls -join [Environment]::NewLine) |
            Should -BeExactly ($expected -join [Environment]::NewLine)
    }
}
