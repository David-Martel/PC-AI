#Requires -Version 7.0
#Requires -Modules @{ ModuleName = 'Pester'; ModuleVersion = '5.0.0' }

BeforeAll {
    $repoRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
    . (Join-Path $repoRoot 'Tools/PcaiNativeBuildFlags.ps1')
    $tokens = $null
    $errors = $null
    $ast = [Management.Automation.Language.Parser]::ParseFile((Join-Path $repoRoot 'Build.ps1'), [ref]$tokens, [ref]$errors)
    if ($errors.Count) { throw 'Build script must parse before flag verification.' }
    $function = $ast.Find({param($node) $node -is [Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq 'Get-DotnetPublishDefaults'}, $true)
    . ([scriptblock]::Create($function.Extent.Text))
}

Describe 'Explicit native CPU build flag selection' {
    BeforeEach {
        $script:OriginalRustFlags = $env:RUSTFLAGS
        $script:OriginalEncodedFlags = $env:CARGO_ENCODED_RUSTFLAGS
        $script:OriginalUnits = $env:CARGO_PROFILE_RELEASE_CODEGEN_UNITS
        $NuGetConfigPath = $null
        $script:OriginalCacheEnvironment = @{}
        foreach ($name in @('SCCACHE_DISABLE','CCACHE_DISABLE','RUSTC_WRAPPER','CMAKE_C_COMPILER_LAUNCHER','CMAKE_CXX_COMPILER_LAUNCHER','CMAKE_GENERATOR')) {
            $script:OriginalCacheEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
        }
        $env:RUSTFLAGS = $null
        $env:CARGO_ENCODED_RUSTFLAGS = $null
        $env:CARGO_PROFILE_RELEASE_CODEGEN_UNITS = $null
    }
    AfterEach {
        $env:RUSTFLAGS = $script:OriginalRustFlags
        $env:CARGO_ENCODED_RUSTFLAGS = $script:OriginalEncodedFlags
        $env:CARGO_PROFILE_RELEASE_CODEGEN_UNITS = $script:OriginalUnits
        foreach ($entry in $script:OriginalCacheEnvironment.GetEnumerator()) {
            [Environment]::SetEnvironmentVariable($entry.Key, $entry.Value, 'Process')
        }
    }

    It 'adds native tuning only when explicitly invoked and records the host' {
        $env:RUSTFLAGS | Should -BeNullOrEmpty
        $result = Enable-PcaiNativeBuildOptimization
        $env:RUSTFLAGS | Should -BeExactly '-C target-cpu=native'
        $env:CARGO_PROFILE_RELEASE_CODEGEN_UNITS | Should -BeExactly '1'
        $result.CpuTarget | Should -BeExactly 'native'
        $result.Host | Should -BeExactly ([Environment]::MachineName)
        $result.Architecture | Should -BeExactly ([Runtime.InteropServices.RuntimeInformation]::ProcessArchitecture.ToString())
    }

    It 'retains an explicitly selected portable CPU and unrelated flags' {
        $env:RUSTFLAGS = '-D warnings -C target-cpu=x86-64-v2 -C opt-level=2'
        $result = Enable-PcaiNativeBuildOptimization
        $env:RUSTFLAGS | Should -BeExactly '-D warnings -C target-cpu=x86-64-v2 -C opt-level=2'
        $result.CpuTarget | Should -BeExactly 'x86-64-v2'
    }

    It 'is idempotent rather than appending repeated CPU switches' {
        $env:RUSTFLAGS = '-D warnings'
        $null = Enable-PcaiNativeBuildOptimization
        $first = $env:RUSTFLAGS
        $null = Enable-PcaiNativeBuildOptimization
        $env:RUSTFLAGS | Should -BeExactly $first
        $env:RUSTFLAGS | Should -BeExactly '-D warnings -C target-cpu=native'
    }

    It 'uses encoded flag precedence without corrupting argument boundaries' {
        $env:RUSTFLAGS = '-C target-cpu=ignored-by-cargo'
        $original = @('-C', 'link-arg=C:/directory with spaces/target-cpu=unrelated/object.lib')
        $env:CARGO_ENCODED_RUSTFLAGS = $original -join [char]31
        $result = Enable-PcaiNativeBuildOptimization
        $env:CARGO_ENCODED_RUSTFLAGS.Split([char]31) | Should -Be (@($original) + @('-C', 'target-cpu=native'))
        $env:RUSTFLAGS | Should -BeExactly '-C target-cpu=ignored-by-cargo'
        $result.CpuTarget | Should -BeExactly 'native'
    }

    It 'preserves an encoded explicit CPU target across repeated requests' {
        $original = @('-C', 'target-cpu=x86-64', '-D', 'warnings') -join [char]31
        $env:CARGO_ENCODED_RUSTFLAGS = $original
        $null = Enable-PcaiNativeBuildOptimization
        $result = Enable-PcaiNativeBuildOptimization
        $env:CARGO_ENCODED_RUSTFLAGS | Should -BeExactly $original
        $result.CpuTarget | Should -BeExactly 'x86-64'
    }

    It 'disables inherited compiler wrappers without changing the build generator' {
        $env:SCCACHE_DISABLE = '0'
        $env:CCACHE_DISABLE = '0'
        $env:RUSTC_WRAPPER = 'sccache'
        $env:CMAKE_C_COMPILER_LAUNCHER = 'ccache'
        $env:CMAKE_CXX_COMPILER_LAUNCHER = 'ccache'
        $env:CMAKE_GENERATOR = 'Ninja'
        Disable-PcaiBuildCompilerCaches
        $env:SCCACHE_DISABLE | Should -BeExactly '1'
        $env:CCACHE_DISABLE | Should -BeExactly '1'
        $env:RUSTC_WRAPPER | Should -BeNullOrEmpty
        $env:CMAKE_C_COMPILER_LAUNCHER | Should -BeNullOrEmpty
        $env:CMAKE_CXX_COMPILER_LAUNCHER | Should -BeNullOrEmpty
        $env:CMAKE_GENERATOR | Should -BeExactly 'Ninja'
    }

    It 'does not infer a private NuGet configuration when none was selected' {
        $NuGetConfigPath = $null
        @(Get-DotnetPublishDefaults -Configuration Release) | Where-Object { $_ -match 'RestoreConfigFile|configfile' } | Should -BeNullOrEmpty
    }

    It 'passes a selected private restore file as one MSBuild property argument' {
        $NuGetConfigPath = Join-Path $TestDrive 'private config.xml'
        Set-Content -LiteralPath $NuGetConfigPath -Value '<configuration />'
        $arguments = @(Get-DotnetPublishDefaults -Configuration Release)
        $arguments | Should -Contain "-p:RestoreConfigFile=$NuGetConfigPath"
        $arguments | Should -Not -Contain '--configfile'
    }

    It 'rejects a missing or directory restore configuration before invoking dotnet' {
        $NuGetConfigPath = Join-Path $TestDrive 'missing-config.xml'
        { Get-DotnetPublishDefaults -Configuration Release } | Should -Throw '*not a file*'
        $NuGetConfigPath = $TestDrive
        { Get-DotnetPublishDefaults -Configuration Release } | Should -Throw '*not a file*'
    }
}
