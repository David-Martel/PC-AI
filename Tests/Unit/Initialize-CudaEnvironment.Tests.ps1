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
    # Needs: inert toolkit directories and stubbed nvcc target lists. Breadcrumb: Select-CudaToolkit.
    It 'chooses a complete toolkit supporting every requested code target' {
        $older = New-ToolkitFixture 'preferred-toolkit'
        $newer = New-ToolkitFixture 'compatible-toolkit'
        Mock Get-NvccGpuCodes { 'sm_89' } -ParameterFilter { $CompilerPath -like '*preferred-toolkit*' }
        Mock Get-NvccGpuCodes { 'sm_89'; 'sm_120'; 'sm_120a' } -ParameterFilter { $CompilerPath -like '*compatible-toolkit*' }

        $choice = Select-CudaToolkit -Candidates $older, $newer -TargetComputeCaps '8.9; 120,sm_120a'

        $choice.Path | Should -BeExactly $newer
        ($choice.TargetCodes -join ',') | Should -BeExactly 'sm_89,sm_120,sm_120a'
        ($choice.Notes -join ' ') | Should -Match 'Unsupported CUDA targets sm_120, sm_120a'
    }

    # Protects: unsupported selected targets fail without CPU or generic architecture substitution.
    # Detects: considering sm_120 code sufficient for an explicit sm_120a request.
    # Needs: inert toolkit with only generic nvcc code target. Breadcrumb: Select-CudaToolkit.
    It 'rejects a generic Blackwell toolkit for an architecture-specific target' {
        $generic = New-ToolkitFixture 'generic-toolkit'
        Mock Get-NvccGpuCodes { 'sm_120' }

        $choice = Select-CudaToolkit -Candidates $generic -TargetComputeCaps '120a'

        $choice.Path | Should -BeNullOrEmpty
        ($choice.Notes -join ' ') | Should -Match 'Unsupported CUDA targets sm_120a'
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
}
