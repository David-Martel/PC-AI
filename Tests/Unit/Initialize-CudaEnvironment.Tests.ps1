#Requires -Version 7.0
#Requires -Modules Pester

Describe 'Initialize-CudaEnvironment' -Tag 'Unit', 'Cuda', 'Portable' {
    BeforeAll {
        $script:RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
        $script:ScriptPath = Join-Path $script:RepoRoot 'Tools\Initialize-CudaEnvironment.ps1'
        $script:Content = Get-Content -LiteralPath $script:ScriptPath -Raw
        . $script:ScriptPath

        function New-ToolkitFixture([string]$Name) {
            $root = Join-Path $TestDrive $Name
            $null = New-Item -ItemType Directory -Path (Join-Path $root 'bin'), (Join-Path $root 'include')
            $null = New-Item -ItemType File -Path (Join-Path $root 'bin/nvcc.exe'), (Join-Path $root 'include/cuda_runtime.h')
            return $root
        }
    }

    It 'prefers CUDA v13.1 before v13.2 for current Rust cudarc compatibility' {
        $script:Content | Should -Match "@\('v13\.1', 'v13\.2'"
    }

    It 'documents CUDA v13.1 as the repository default' {
        $script:Content | Should -Match 'Preferred default for this repository is CUDA v13\.1'
    }

    # Protects: a complete toolkit must support the whole explicitly requested build matrix.
    # Detects: compiling against the preferred toolkit even when it lacks Blackwell code generation.
    # Needs: inert toolkit directories and separate stubbed nvcc and ptxas lists. Breadcrumb: Select-CudaToolkit.
    It 'chooses a complete toolkit supporting every requested code target' {
        $older = New-ToolkitFixture 'preferred-toolkit'
        $newer = New-ToolkitFixture 'compatible-toolkit'
        Mock Get-NvccGpuCodes { 'sm_89' } -ParameterFilter { $CompilerPath -like '*preferred-toolkit*' }
        Mock Get-NvccGpuCodes { 'sm_89'; 'sm_120' } -ParameterFilter { $CompilerPath -like '*compatible-toolkit*' }
        $null = New-Item -ItemType File -Path (Join-Path $newer 'bin/ptxas.exe')
        Mock Get-PtxasGpuCodes { 'sm_120a'; 'sm_120f' } -ParameterFilter { $AssemblerPath -like '*compatible-toolkit*' }

        $choice = Select-CudaToolkit -Candidates $older, $newer -TargetComputeCaps '8.9; 120,sm_120a,120f'

        $choice.Path | Should -BeExactly $newer
        ($choice.TargetCodes -join ',') | Should -BeExactly 'sm_89,sm_120,sm_120a,sm_120f'
        ($choice.Notes -join ' ') | Should -Match 'Unsupported CUDA targets sm_120, sm_120a, sm_120f'
        Should -Invoke Get-PtxasGpuCodes -Times 1 -Exactly
    }

    # Protects: unsupported selected targets fail without CPU or generic architecture substitution.
    # Detects: considering sm_120 code sufficient for an explicit sm_120a request.
    # Needs: an inert toolkit whose assembler cannot confirm the exact requested suffix. Breadcrumb: Select-CudaToolkit.
    It 'rejects a generic Blackwell toolkit for an architecture-specific target' {
        $generic = New-ToolkitFixture 'generic-toolkit'
        $null = New-Item -ItemType File -Path (Join-Path $generic 'bin/ptxas.exe')
        Mock Get-NvccGpuCodes { 'sm_120'; 'sm_120a' }
        Mock Get-PtxasGpuCodes { 'sm_120f' }

        $choice = Select-CudaToolkit -Candidates $generic -TargetComputeCaps '120a'

        $choice.Path | Should -BeNullOrEmpty
        ($choice.Notes -join ' ') | Should -Match 'Unsupported CUDA targets sm_120a.+ptxas did not list them'
    }

    # Protects: development-shell discovery works without probing optional architecture targets.
    # Detects: starting nvcc when the caller has asked only for a complete toolkit.
    # Needs: inert paths without an executable compiler. Breadcrumb: Select-CudaToolkit.
    It 'keeps candidate order when no target architecture is requested' {
        $first = New-ToolkitFixture 'default-toolkit'
        Mock Get-NvccGpuCodes { throw 'Unexpected nvcc probe' }

        (Select-CudaToolkit -Candidates $first).Path | Should -BeExactly $first
        Should -Invoke Get-NvccGpuCodes -Times 0 -Exactly
    }

    # Protects: invalid explicit target declarations stop before compiler discovery.
    # Detects: dropping the bad token and building for a partial target list.
    # Needs: an inert toolkit. Breadcrumb: Select-CudaToolkit.
    It 'rejects a malformed target list before probing nvcc' {
        $fixture = New-ToolkitFixture 'unneeded-toolkit'
        Mock Get-NvccGpuCodes { throw 'Unexpected nvcc probe' }

        { Select-CudaToolkit -Candidates $fixture -TargetComputeCaps '89,sm_12x' } |
            Should -Throw '*Invalid CUDA compute target*'
        Should -Invoke Get-NvccGpuCodes -Times 0 -Exactly
    }

    # Protects: bare nvcc and CUDA 13 runtime libraries resolve from the admitted CUDA toolkit.
    # Detects: leaving the selected toolkit behind older PATH entries or treating a substring as an exact match.
    # Needs: owned fixture directories and process-scoped PATH, restored on exit.
    # Breadcrumb: Initialize-CudaEnvironment preferred CUDA bin PATH assignment.
    It 'prioritizes the selected CUDA bin already present in PATH' {
        $selected = New-ToolkitFixture 'selected-toolkit'
        $older = Join-Path (New-ToolkitFixture 'older-toolkit') 'bin'
        $bin = Join-Path $selected 'bin'
        $nearMatch = "$bin-extras"
        $ast = [Management.Automation.Language.Parser]::ParseInput($script:Content, [ref]$null, [ref]$null)
        $bindings = @($ast.FindAll({
                param($node)
                $node -is [Management.Automation.Language.AssignmentStatementAst] -and
                $node.Left.Extent.Text -eq '$pathUpdated' -and $node.Right.Extent.Text -match 'Join-Path \$selected ''bin'''
            }, $true))
        $bindings.Count | Should -Be 1
        $bind = [scriptblock]::Create($bindings[0].Extent.Text)
        $originalPath = $env:PATH

        try {
            $env:PATH = @($older, $nearMatch, $bin, $bin) -join ';'
            $pathUpdated = $false
            . $bind
            $pathUpdated | Should -BeTrue
            $expectedPath = @($bin, $older, $nearMatch) -join ';'
            $env:PATH | Should -BeExactly $expectedPath

            $pathUpdated = $false
            . $bind
            $pathUpdated | Should -BeFalse
            $env:PATH | Should -BeExactly $expectedPath

            $runtimeBin = Join-Path $bin 'x64'
            $null = New-Item -ItemType Directory -Path $runtimeBin
            . $bind
            $pathUpdated | Should -BeTrue
            $env:PATH | Should -BeExactly (@($bin, $runtimeBin, $older, $nearMatch) -join ';')
        } finally {
            $env:PATH = $originalPath
        }
    }

    # Protects: a selected compiler without a runtime directory keeps unrelated PATH entries in order.
    # Detects: promoting an older toolkit's CUDA runtime ahead of unrelated entries.
    # Needs: Pester TestDrive directory fixtures and process PATH, restored even on failure.
    # Breadcrumb: Tools/Initialize-CudaEnvironment.ps1 Set-PreferredCudaBinPath.
    It 'leaves a foreign CUDA runtime in place when the selected toolkit has none' {
        $selectedBin = Join-Path (New-ToolkitFixture 'compiler-only') 'bin'
        $foreignBin = Join-Path (New-ToolkitFixture 'older-with-runtime') 'bin'
        $foreignRuntime = Join-Path $foreignBin 'x64'
        $null = New-Item -ItemType Directory -Path $foreignRuntime
        $systemBin = Join-Path $TestDrive 'other-tools'
        $originalPath = $env:PATH

        try {
            $env:PATH = @($systemBin, $foreignBin, $foreignRuntime) -join ';'
            Set-PreferredCudaBinPath $selectedBin | Should -BeTrue
            $env:PATH | Should -BeExactly (@($selectedBin, $systemBin, $foreignBin, $foreignRuntime) -join ';')
        } finally {
            $env:PATH = $originalPath
        }
    }

    # Protects: suffixed CUDA targets require the admitted toolkit's own assembler.
    # Detects: accepting an advertised numeric code with no sibling ptxas.
    # Needs: a complete numeric-only toolkit fixture and mocked compiler output.
    # Breadcrumb: Select-CudaToolkit architecture-specific target admission.
    It 'requires the same toolkit ptxas only for suffixed targets' {
        $fixture = New-ToolkitFixture 'toolkit-without-ptxas'
        Mock Get-NvccGpuCodes { 'sm_120' }
        Mock Get-PtxasGpuCodes { throw 'Unexpected assembler probe' }

        (Select-CudaToolkit -Candidates $fixture -TargetComputeCaps '120').Path | Should -BeExactly $fixture
        $choice = Select-CudaToolkit -Candidates $fixture -TargetComputeCaps '120f'
        $choice.Path | Should -BeNullOrEmpty
        ($choice.Notes -join ' ') | Should -Match 'ptxas is missing for CUDA targets sm_120f'
        Should -Invoke Get-PtxasGpuCodes -Times 0 -Exactly
    }

    # Protects: only the wrapped assembler allowed-values list certifies a suffix.
    # Detects: reading other options as capabilities or stopping before SM values after LTO entries.
    # Needs: synthetic ptxas help text with a PowerShell function standing in for the executable.
    # Breadcrumb: Get-PtxasGpuCodes allowed-values parser.
    It 'reads exact suffixed code names from wrapped assembler help' {
        function Get-FixturePtxasHelp {
            $global:LASTEXITCODE = 0
            '  --fixture-other <name>'
            "    Allowed values for this option: 'sm_121a'."
            "    Example only: 'sm_121f'"
            '  --gpu-name <name> (-arch)'
            "    Allowed values for this option:  'compute_100','compute_100a','compute_100f',"
            "      'compute_120','compute_120a','compute_120f','compute_89','compute_90a','lto_100','lto_100a',"
            "      'lto_100f','lto_120','lto_120a','lto_120f','lto_89','lto_90a','sm_120',"
            "      'sm_120a','sm_120f','sm_89','sm_90a'."
            "    Default value: 'sm_89'."
        }

        $oldExitCode = $global:LASTEXITCODE
        try {
            (@(Get-PtxasGpuCodes -AssemblerPath 'Get-FixturePtxasHelp') -join ',') |
                Should -BeExactly 'sm_120a,sm_120f,sm_90a'
        } finally {
            $global:LASTEXITCODE = $oldExitCode
        }
    }
    # Protects: toolkit admission rejects native tool errors even when stdout appears valid.
    # Detects: trusting output from a failing probe or losing the distinct empty-output diagnosis.
    # Needs: Windows cmd.exe, an isolated toolkit and the exact real probe bodies.
    # Breadcrumb: Select-CudaToolkit, Get-NvccGpuCodes and Get-PtxasGpuCodes.
    It 'rejects invalid native capability probe <Case>' -Skip:(-not $IsWindows) -ForEach @(
        @{
            Case = 'nvcc nonzero with valid stdout'; Target = '89'
            Nvcc = @('echo sm_89', 'exit /b 7')
            ExpectedReason = 'nvcc could not list GPU targets:'
        },
        @{
            Case = 'nvcc success with empty stdout'; Target = '89'
            Nvcc = @('exit /b 0')
            ExpectedReason = 'nvcc could not list GPU targets:'
        },
        @{
            Case = 'ptxas nonzero with valid stdout'; Target = '120a'
            Nvcc = @('echo sm_120', 'exit /b 0')
            Ptxas = @(
                'echo --gpu-name GPU (-arch)',
                "echo     Allowed values for this option: 'compute_120a','lto_120a','sm_120','sm_120a'.",
                'exit /b 9'
            )
            ExpectedReason = 'Unsupported CUDA targets sm_120a in .+ \(ptxas did not list them\)'
        }
    ) {
        $nativeControl = New-ToolkitFixture $Case
        $script:NativeNvccProbe = (Get-Command Get-NvccGpuCodes -CommandType Function).ScriptBlock
        $script:NativePtxasProbe = (Get-Command Get-PtxasGpuCodes -CommandType Function).ScriptBlock
        $nvccCmd = Join-Path $nativeControl 'bin/nvcc.cmd'
        Set-Content -LiteralPath $nvccCmd -Value (@('@echo off') + $Nvcc) -Encoding ascii
        if ($Ptxas) {
            $null = New-Item -ItemType File -Path (Join-Path $nativeControl 'bin/ptxas.exe')
            Set-Content -LiteralPath (Join-Path $nativeControl 'bin/ptxas.cmd') -Value (@('@echo off') + $Ptxas) -Encoding ascii
        }

        # The selector checks the toolkit's .exe paths. Substitute only the filename passed
        # to the captured native probe function so its launch, parsing, and exit checks run.
        Mock Get-NvccGpuCodes {
            & $script:NativeNvccProbe -CompilerPath ([IO.Path]::ChangeExtension($CompilerPath, 'cmd'))
        }
        Mock Get-PtxasGpuCodes {
            & $script:NativePtxasProbe -AssemblerPath ([IO.Path]::ChangeExtension($AssemblerPath, 'cmd'))
        }
        $PSNativeCommandUseErrorActionPreference = $false
        $choice = Select-CudaToolkit -Candidates $nativeControl -TargetComputeCaps $Target

        $choice.Path | Should -BeNullOrEmpty
        ($choice.Notes -join ' ') | Should -Match $ExpectedReason
        Should -Invoke Get-NvccGpuCodes -Times 1 -Exactly -ParameterFilter {
            [IO.Path]::GetFileName($CompilerPath) -eq 'nvcc.exe'
        }
        $assemblerCalls = if ($Ptxas) { 1 } else { 0 }
        Should -Invoke Get-PtxasGpuCodes -Times $assemblerCalls -Exactly
    }
}

Describe 'PC-AI CUDA build admission' -Tag 'Unit', 'Cuda', 'Portable' {
    BeforeAll {
        $repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
        $errors = $null
        $cudaAst = [Management.Automation.Language.Parser]::ParseFile(
            (Join-Path $repoRoot 'Native/pcai_core/pcai_inference/Invoke-PcaiBuild.ps1'), [ref]$null, [ref]$errors)
        if ($errors.Count) { throw ($errors | Out-String) }
        $cudaInit = $cudaAst.Find({
                param($node)
                $node -is [Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq 'Initialize-BuildEnvironment'
            }, $true)
        $cudaBranch = $cudaInit.Body.EndBlock.Statements | Where-Object {
            $_ -is [Management.Automation.Language.IfStatementAst] -and
            $_.Extent.Text -match '^if \(\$EnableCuda' -and
            $_.Extent.Text.Contains('Initialize-CudaEnvironment -Quiet')
        }
        if (@($cudaBranch).Count -ne 1) { throw 'The explicit CUDA selector was not found.' }
        $script:CudaProbe = [scriptblock]::Create($cudaBranch.Extent.Text)

        $tailStart = $cudaAst.EndBlock.Statements | Where-Object {
            $_ -is [Management.Automation.Language.AssignmentStatementAst] -and
            $_.Left.Extent.Text -eq '$previousMistralCudaTarget'
        }
        $tailEnd = $cudaAst.EndBlock.Statements | Where-Object {
            $_ -is [Management.Automation.Language.AssignmentStatementAst] -and
            $_.Left.Extent.Text -eq '$totalDuration'
        }
        if (@($tailStart).Count -ne 1 -or @($tailEnd).Count -ne 1) {
            throw 'The CUDA build tail was not found.'
        }
        $tailOffset = $tailStart.Extent.StartOffset
        $tailLength = $tailEnd.Extent.StartOffset - $tailOffset
        $script:CudaBuildTail = [scriptblock]::Create($cudaAst.Extent.Text.Substring($tailOffset, $tailLength))
    }

    # Protects: toolkit admission follows each backend's own target override.
    # Detects: a bootstrap CUDAARCHS default vetoing a valid legacy declaration or another backend's flags being ignored.
    # Needs: real extracted CUDA branch and real toolkit selection with an inert compiler reply.
    # Breadcrumb: Invoke-PcaiBuild.ps1 Initialize-BuildEnvironment CUDA selection.
    It 'probes the effective CUDA declaration for <Case>' -ForEach @(
        @{ Case = 'legacy GGML with stale CMake defaults'; TargetBackend = 'llamacpp'; Ggml = '89'; Cmake = '75;80;86;89;120'; Expected = 'sm_89' },
        @{ Case = 'legacy PCAI override'; TargetBackend = 'llamacpp'; Pcai = '89'; Plural = '120'; Expected = 'sm_89' },
        @{ Case = 'legacy plural override'; TargetBackend = 'llamacpp'; Plural = '89'; Cmake = '120'; Expected = 'sm_89' },
        @{ Case = 'legacy automatic compiler selection'; TargetBackend = 'llamacpp'; Cmake = '120'; Expected = '' },
        @{ Case = 'mistral singular target'; TargetBackend = 'mistralrs'; Singular = '89'; Ggml = '120'; Cmake = '120'; Expected = 'sm_89' },
        @{ Case = 'both backends'; TargetBackend = 'all'; Ggml = '89'; Cmake = '86'; Expected = 'sm_86,sm_89' }
    ) {
        $ToolsDir = Join-Path $TestDrive $Case
        $null = New-Item -ItemType Directory -Path (Join-Path $ToolsDir 'bin'), (Join-Path $ToolsDir 'include')
        $null = New-Item -ItemType File -Path (Join-Path $ToolsDir 'bin/nvcc.exe'), (Join-Path $ToolsDir 'include/cuda_runtime.h')
        Copy-Item -LiteralPath (Join-Path $PSScriptRoot '../../Tools/Initialize-CudaEnvironment.ps1') -Destination (Join-Path $ToolsDir 'RealToolkit.ps1')
        @'
. (Join-Path $PSScriptRoot 'RealToolkit.ps1')
function Get-NvccGpuCodes { param([string]$CompilerPath) 'sm_89'; 'sm_86' }
function Initialize-CudaEnvironment {
    param([switch]$Quiet, [string]$TargetComputeCaps)
    $selection = Select-CudaToolkit -Candidates $PSScriptRoot -TargetComputeCaps $TargetComputeCaps
    if (-not $selection.Path) { throw ($selection.Notes -join '; ') }
    throw "Fixture accepted: [$(($selection.TargetCodes | Sort-Object) -join ',')]"
}
'@ | Set-Content -LiteralPath (Join-Path $ToolsDir 'Initialize-CudaEnvironment.ps1')

        $values = @{
            GGML_CUDA_ARCH_LIST = $Ggml; PCAI_CUDA_ARCH_LIST = $Pcai
            CUDA_COMPUTE_CAPS = $Plural; CUDA_COMPUTE_CAP = $Singular; CUDAARCHS = $Cmake
        }
        $original = @{}
        $Backend = $TargetBackend
        $EnableCuda = $true
        try {
            foreach ($name in $values.Keys) {
                $original[$name] = [Environment]::GetEnvironmentVariable($name)
                [Environment]::SetEnvironmentVariable($name, $values[$name])
            }
            { & $script:CudaProbe } | Should -Throw -ExpectedMessage "Fixture accepted: $([WildcardPattern]::Escape("[$Expected]"))"
        } finally {
            foreach ($name in $original.Keys) {
                [Environment]::SetEnvironmentVariable($name, $original[$name])
            }
        }
    }

    # Protects: the legacy GGML build receives only numeric architecture targets.
    # Detects: dropping the suffix guard and letting GGML silently discard that target.
    # Needs: the actual CUDA branch and a helper that must never probe a compiler.
    # Breadcrumb: Invoke-PcaiBuild.ps1 Initialize-BuildEnvironment GGML suffix guard.
    It 'rejects an architecture suffix in legacy GGML input <Target>' -ForEach @(
        @{ Target = '89,120a' },
        @{ Target = '89;12.0f' }
    ) {
        $ToolsDir = $TestDrive
        @'
function Initialize-CudaEnvironment { throw 'Compiler probe must not run' }
'@ | Set-Content -LiteralPath (Join-Path $ToolsDir 'Initialize-CudaEnvironment.ps1')
        $Backend = 'llamacpp'
        $EnableCuda = $true
        $original = [Environment]::GetEnvironmentVariable('GGML_CUDA_ARCH_LIST')
        try {
            $env:GGML_CUDA_ARCH_LIST = $Target
            { & $script:CudaProbe } | Should -Throw 'The legacy llama-rs build accepts numeric CUDA targets only*'
        } finally {
            [Environment]::SetEnvironmentVariable('GGML_CUDA_ARCH_LIST', $original)
        }
    }

    # Protects: the child gets the selected target, compiler and runtime DLL path after optional library setup.
    # Detects: a lost GGML override, a missing CUDA 13 runtime path or another toolkit taking precedence on PATH.
    # Needs: actual CUDA branch and PATH helper, inert toolkit and optional library fixtures.
    # Breadcrumb: Invoke-PcaiBuild.ps1 Initialize-BuildEnvironment process exports.
    It 'preserves the GGML override and compiler after optional CUDA library setup' {
        $ToolsDir = Join-Path $TestDrive 'admitted-toolkit'
        $cudaBin = Join-Path $ToolsDir 'bin'
        $runtimeBin = Join-Path $cudaBin 'x64'
        $cudnnBin = Join-Path $TestDrive 'optional-cudnn/bin'
        $tensorRtBin = Join-Path $TestDrive 'optional-tensorrt/bin'
        $null = New-Item -ItemType Directory -Path $runtimeBin, $cudnnBin, $tensorRtBin
        $null = New-Item -ItemType File -Path (Join-Path $runtimeBin 'cudart64_13.dll')
        Copy-Item -LiteralPath (Join-Path $PSScriptRoot '../../Tools/Initialize-CudaEnvironment.ps1') `
            -Destination (Join-Path $ToolsDir 'RealToolkit.ps1')
        @'
. (Join-Path $PSScriptRoot 'RealToolkit.ps1')
function Initialize-CudaEnvironment {
    param([switch]$Quiet, [string]$TargetComputeCaps)
    if ($TargetComputeCaps.Trim(',') -ne '89') { throw 'Override was not admitted' }
    $null = Set-PreferredCudaBinPath (Join-Path $PSScriptRoot 'bin')
    @{ Found = $true; CudaPath = $PSScriptRoot }
}
'@ | Set-Content -LiteralPath (Join-Path $ToolsDir 'Initialize-CudaEnvironment.ps1')
        function Initialize-CudnnEnvironment {
            param([string]$CudaPath)
            $env:PATH = "$cudnnBin;$env:PATH"
        }
        function Initialize-TensorRtEnvironment { $env:PATH = "$tensorRtBin;$env:PATH" }

        $envNames = @('PATH', 'GGML_CUDA_ARCH_LIST', 'PCAI_CUDA_ARCH_LIST', 'LLAMA_CUDA', 'GGML_CUDA',
            'RUSTFLAGS', 'CUDA_NVCC_FLAGS', 'NVCCFLAGS', 'CMAKE_ARGS', 'CL', '_CL_', 'LIB', 'INCLUDE')
        $original = @{}
        foreach ($name in $envNames) { $original[$name] = [Environment]::GetEnvironmentVariable($name) }
        $Backend = 'llamacpp'
        $Configuration = 'Release'
        $EnableCuda = $true
        try {
            $env:GGML_CUDA_ARCH_LIST = $null
            $env:PCAI_CUDA_ARCH_LIST = '89'
            $priorPath = Join-Path $TestDrive 'existing-selection/bin'
            $env:PATH = @($priorPath, $runtimeBin, $cudaBin, $runtimeBin) -join ';'
            $null = & $script:CudaProbe
            $env:GGML_CUDA_ARCH_LIST | Should -BeExactly '89'
            $env:PATH | Should -BeExactly (@($cudaBin, $runtimeBin, $tensorRtBin, $cudnnBin, $priorPath) -join ';')
        } finally {
            foreach ($name in $envNames) { [Environment]::SetEnvironmentVariable($name, $original[$name]) }
        }
    }

    # Protects the Mistral build from probing one target and asking cudaforge to compile for another.
    # Detects dropped overrides, unqualified auto-suffixes and unsupported suffixes reaching Cargo.
    # Needs the actual CUDA branch, toolkit selector and inert advertised compiler and assembler targets.
    # Breadcrumb: Invoke-PcaiBuild.ps1 Initialize-BuildEnvironment Mistral target selection.
    It 'hands Mistral the admitted architecture for <Case>' -ForEach @(
        @{ Case = 'Ada plural shorthand'; Plural = '8.9'; Expected = '89' },
        @{ Case = 'Ada CMake fallback'; Cmake = 'compute_89'; Expected = '89' },
        @{ Case = 'Ada singular preference'; Singular = 'sm_89'; Plural = '120'; Expected = '89' },
        @{ Case = 'Blackwell numeric requires assembler'; Plural = '120'; ExpectedError = '*Unsupported CUDA targets sm_120a*' },
        @{ Case = 'Blackwell numeric with assembler'; Plural = '120'; AssemblerA = $true; Expected = '120' },
        @{ Case = 'Blackwell exact suffix'; Singular = 'sm_120a'; AssemblerA = $true; Expected = '120a' },
        @{ Case = 'Blackwell unsupported family suffix'; Plural = '120f'; ExpectedError = '*Mistral CUDA cannot compile target 120f*' },
        @{ Case = 'Native automatic'; Expected = '' },
        @{ Case = 'Native multiple candidates'; Plural = '89;120'; Expected = '' }
    ) {
        $ToolsDir = Join-Path $TestDrive $Case
        $null = New-Item -ItemType Directory -Path (Join-Path $ToolsDir 'bin'), (Join-Path $ToolsDir 'include')
        $null = New-Item -ItemType File -Path (Join-Path $ToolsDir 'bin/nvcc.exe'),
            (Join-Path $ToolsDir 'bin/ptxas.exe'), (Join-Path $ToolsDir 'include/cuda_runtime.h')
        if ($AssemblerA) { $null = New-Item -ItemType File -Path (Join-Path $ToolsDir 'supports-120a') }
        Copy-Item -LiteralPath (Join-Path $PSScriptRoot '../../Tools/Initialize-CudaEnvironment.ps1') `
            -Destination (Join-Path $ToolsDir 'RealToolkit.ps1')
        @'
. (Join-Path $PSScriptRoot 'RealToolkit.ps1')
function Get-NvccGpuCodes { param([string]$CompilerPath) 'sm_89'; 'sm_120' }
function Get-PtxasGpuCodes {
    param([string]$AssemblerPath)
    'sm_120f'
    if (Test-Path -LiteralPath (Join-Path $PSScriptRoot 'supports-120a')) { 'sm_120a' }
}
function Initialize-CudaEnvironment {
    param([switch]$Quiet, [string]$TargetComputeCaps)
    $choice = Select-CudaToolkit -Candidates $PSScriptRoot -TargetComputeCaps $TargetComputeCaps
    if (-not $choice.Path) { return @{ Found = $false; Notes = $choice.Notes } }
    if (-not $env:CUDAARCHS) { $env:CUDAARCHS = '75;80;86;89;120' }
    @{ Found = $true; CudaPath = $choice.Path }
}
'@ | Set-Content -LiteralPath (Join-Path $ToolsDir 'Initialize-CudaEnvironment.ps1')
        function Resolve-MsvcClPath { return $null }
        function Initialize-CudnnEnvironment { param([string]$CudaPath) }
        function Initialize-TensorRtEnvironment {}

        $names = @('PATH', 'CUDA_COMPUTE_CAP', 'CUDA_COMPUTE_CAPS', 'CUDAARCHS', 'LLAMA_CUDA',
            'GGML_CUDA', 'RUSTFLAGS', 'CUDA_NVCC_FLAGS', 'NVCCFLAGS', 'CMAKE_ARGS', 'CL', '_CL_',
            'LIB', 'INCLUDE', 'CMAKE_CUDA_COMPILER', 'NVCC_CCBIN')
        $original = @{}
        foreach ($name in $names) { $original[$name] = [Environment]::GetEnvironmentVariable($name) }
        $originalScriptMsvcClPath = $script:MsvcClPath
        $Backend = 'mistralrs'
        $Configuration = 'Release'
        $EnableCuda = $true
        try {
            $script:MsvcClPath = $null
            [Environment]::SetEnvironmentVariable('CUDA_COMPUTE_CAP', $Singular)
            [Environment]::SetEnvironmentVariable('CUDA_COMPUTE_CAPS', $Plural)
            [Environment]::SetEnvironmentVariable('CUDAARCHS', $Cmake)
            if ($ExpectedError) {
                { & $script:CudaProbe } | Should -Throw -ExpectedMessage $ExpectedError
            } else {
                $null = & $script:CudaProbe
                if ($Expected) {
                    $env:CUDA_COMPUTE_CAP | Should -BeExactly $Expected
                } else {
                    $env:CUDA_COMPUTE_CAP | Should -BeNullOrEmpty
                }
            }
        } finally {
            $script:MsvcClPath = $originalScriptMsvcClPath
            foreach ($name in $names) { [Environment]::SetEnvironmentVariable($name, $original[$name]) }
        }
    }

    # Protects: subsequent same-process builds see the caller's original Mistral target.
    # Detects: normalized targets leaking from either a successful build or a build that throws.
    # Needs: Pester, the actual top-level build tail and inert initialization, build and copy commands.
    # Breadcrumb: Native/pcai_core/pcai_inference/Invoke-PcaiBuild.ps1 snapshot and finally cleanup.
    It 'restores the inherited Mistral CUDA target after <Case>' -ForEach @(
        @{ Case = 'success with a caller target'; CallerTarget = 'sm_89'; BuildTarget = '89'; BuildFails = $false },
        @{ Case = 'failure with a caller target'; CallerTarget = 'sm_89'; BuildTarget = '89'; BuildFails = $true },
        @{ Case = 'success without a caller target'; CallerTarget = $null; BuildTarget = '120'; BuildFails = $false },
        @{ Case = 'failure without a caller target'; CallerTarget = $null; BuildTarget = '120'; BuildFails = $true }
    ) {
        $Backend = 'mistralrs'
        $Configuration = 'Release'
        $steps = [System.Collections.Generic.List[string]]::new()
        function Initialize-BuildEnvironment { $env:CUDA_COMPUTE_CAP = $BuildTarget }
        function Clear-IncompleteCmakeConfig {}
        function Invoke-Build {
            param([string]$BackendName)
            $steps.Add($BackendName)
            $env:CUDA_COMPUTE_CAP | Should -BeExactly $BuildTarget
            if ($BuildFails) { throw 'CUDA fixture build failed' }
        }
        function Copy-CompiledArtifacts {
            param([string]$Configuration, [string]$BackendBuilt)
            $steps.Add("$BackendBuilt/$Configuration")
        }
        $originalTarget = [Environment]::GetEnvironmentVariable('CUDA_COMPUTE_CAP')
        try {
            $env:CUDA_COMPUTE_CAP = $CallerTarget
            if ($BuildFails) {
                { & $script:CudaBuildTail } | Should -Throw 'CUDA fixture build failed'
                ($steps -join ',') | Should -BeExactly 'mistralrs'
            } else {
                & $script:CudaBuildTail
                ($steps -join ',') | Should -BeExactly 'mistralrs,mistralrs/Release'
            }
            if ($null -eq $CallerTarget) {
                $env:CUDA_COMPUTE_CAP | Should -BeNullOrEmpty
            } else {
                $env:CUDA_COMPUTE_CAP | Should -BeExactly $CallerTarget
            }
        } finally {
            [Environment]::SetEnvironmentVariable('CUDA_COMPUTE_CAP', $originalTarget)
        }
    }
}
