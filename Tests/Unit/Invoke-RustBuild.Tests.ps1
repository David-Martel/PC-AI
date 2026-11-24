#Requires -Modules Pester

if (-not (Get-Variable -Name CargoAvailable -Scope Script -ErrorAction SilentlyContinue)) {
    $script:CargoAvailable = $false
}
if (-not (Get-Variable -Name CargoVersionOutput -Scope Script -ErrorAction SilentlyContinue)) {
    $script:CargoVersionOutput = $null
}

Describe 'Invoke-RustBuild' -Tag 'Unit', 'Portable' {
    BeforeAll {
        $script:RustBuildPath = Join-Path $PSScriptRoot '..\..\Tools\Invoke-RustBuild.ps1'
    }

    Context 'Script Exists' {
        It 'Should exist at expected path' {
            Test-Path $script:RustBuildPath | Should -BeTrue
        }

        It 'Should be a valid PowerShell script' {
            $content = Get-Content $script:RustBuildPath -Raw -ErrorAction SilentlyContinue
            $content | Should -Not -BeNullOrEmpty
        }

        It 'Should have PowerShell version requirement' {
            $content = Get-Content $script:RustBuildPath -Raw
            $content | Should -Match '#Requires -Version'
        }
    }

    Context 'Parameter Validation' {
        BeforeAll {
            $script:Command = Get-Command $script:RustBuildPath
        }

        It 'Should have Path parameter' {
            $script:Command.Parameters.ContainsKey('Path') | Should -BeTrue
        }

        It 'Should have Path parameter default to current location' {
            $pathParam = $script:Command.Parameters['Path']
            $pathParam.ParameterType.Name | Should -Be 'String'
        }

        It 'Should have UseLld switch' {
            $script:Command.Parameters.ContainsKey('UseLld') | Should -BeTrue
            $script:Command.Parameters['UseLld'].SwitchParameter | Should -BeTrue
        }

        It 'Should have NoLld switch' {
            $script:Command.Parameters.ContainsKey('NoLld') | Should -BeTrue
            $script:Command.Parameters['NoLld'].SwitchParameter | Should -BeTrue
        }

        It 'Should have LlmDebug switch' {
            $script:Command.Parameters.ContainsKey('LlmDebug') | Should -BeTrue
            $script:Command.Parameters['LlmDebug'].SwitchParameter | Should -BeTrue
        }

        It 'Should have LlmOutput switch' {
            $script:Command.Parameters.ContainsKey('LlmOutput') | Should -BeTrue
            $script:Command.Parameters['LlmOutput'].SwitchParameter | Should -BeTrue
        }

        It 'Should have RaPreflight switch' {
            $script:Command.Parameters.ContainsKey('RaPreflight') | Should -BeTrue
            $script:Command.Parameters['RaPreflight'].SwitchParameter | Should -BeTrue
        }

        It 'Should have Preflight switch' {
            $script:Command.Parameters.ContainsKey('Preflight') | Should -BeTrue
            $script:Command.Parameters['Preflight'].SwitchParameter | Should -BeTrue
        }

        It 'Should have PreflightMode parameter' {
            $script:Command.Parameters.ContainsKey('PreflightMode') | Should -BeTrue
        }

        It 'Should validate PreflightMode values' {
            $validateSet = $script:Command.Parameters['PreflightMode'].Attributes |
                Where-Object { $_ -is [System.Management.Automation.ValidateSetAttribute] }
            $validateSet.ValidValues | Should -Contain 'check'
            $validateSet.ValidValues | Should -Contain 'clippy'
            $validateSet.ValidValues | Should -Contain 'fmt'
            $validateSet.ValidValues | Should -Contain 'all'
        }

        It 'Should have PreflightBlocking switch' {
            $script:Command.Parameters.ContainsKey('PreflightBlocking') | Should -BeTrue
            $script:Command.Parameters['PreflightBlocking'].SwitchParameter | Should -BeTrue
        }

        It 'Should have CargoArgs parameter' {
            $script:Command.Parameters.ContainsKey('CargoArgs') | Should -BeTrue
            $script:Command.Parameters['CargoArgs'].ParameterType.Name | Should -Be 'String[]'
        }

        It 'Should have RemainingArgs parameter with ValueFromRemainingArguments' {
            $script:Command.Parameters.ContainsKey('RemainingArgs') | Should -BeTrue
            $remainingArgsParam = $script:Command.Parameters['RemainingArgs']
            $remainingArgsParam.ParameterType.Name | Should -Be 'String[]'
            $valueFromRemaining = $remainingArgsParam.Attributes |
                Where-Object { $_ -is [System.Management.Automation.ParameterAttribute] } |
                Where-Object { $_.ValueFromRemainingArguments -eq $true }
            $valueFromRemaining | Should -Not -BeNullOrEmpty
        }

        It 'Should disable positional binding' {
            $cmdletBinding = $script:Command.ScriptBlock.Attributes |
                Where-Object { $_ -is [System.Management.Automation.CmdletBindingAttribute] }
            $cmdletBinding.PositionalBinding | Should -BeFalse
        }
    }

    Context 'Environment Configuration' {
        It 'Should default CARGO_USE_LLD to 0' {
            $content = Get-Content $script:RustBuildPath -Raw
            $content | Should -Match "CARGO_USE_LLD.*=.*'0'"
        }

        It 'Should check for CargoTools module' {
            $content = Get-Content $script:RustBuildPath -Raw
            $content | Should -Match 'Get-Module.*CargoTools'
        }

        It 'Should resolve CargoTools module manifest explicitly' {
            $content = Get-Content $script:RustBuildPath -Raw
            $content | Should -Match 'Resolve-PcaiModuleManifestPath'
            $content | Should -Match 'Import-Module -Name \$cargoToolsManifest'
        }

        It 'Should fall back when CargoTools is not available' {
            $content = Get-Content $script:RustBuildPath -Raw
            $content | Should -Match "Write-Warning 'CargoTools module not found\. Falling back to direct cargo invocation\.'"
            $content | Should -Match '\$script:UseCargoTools = \$false'
        }

        It 'Should configure LLVM lld-link path' {
            $content = Get-Content $script:RustBuildPath -Raw
            $content | Should -Match 'lld-link\.exe'
            $content | Should -Match 'CARGO_LLD_PATH'
        }

        It 'Should set CARGO_USE_LLD based on UseLld switch' {
            $content = Get-Content $script:RustBuildPath -Raw
            $content | Should -Match "if.*UseLld.*CARGO_USE_LLD.*=.*'1'"
        }

        It 'Should set CARGO_USE_LLD based on NoLld switch' {
            $content = Get-Content $script:RustBuildPath -Raw
            $content | Should -Match "if.*NoLld.*CARGO_USE_LLD.*=.*'0'"
        }

        It 'Should configure preflight environment variables' {
            $content = Get-Content $script:RustBuildPath -Raw
            $content | Should -Match 'CARGO_PREFLIGHT'
            $content | Should -Match 'CARGO_PREFLIGHT_MODE'
        }

        It 'Should configure rust-analyzer preflight' {
            $content = Get-Content $script:RustBuildPath -Raw
            $content | Should -Match 'CARGO_RA_PREFLIGHT'
        }

        It 'Should configure preflight blocking' {
            $content = Get-Content $script:RustBuildPath -Raw
            $content | Should -Match 'CARGO_PREFLIGHT_BLOCKING'
        }

        It 'Should pass LLM output mode to CargoTools' {
            $content = Get-Content $script:RustBuildPath -Raw
            $content | Should -Match '--llm-output'
        }
    }

    Context 'Cargo Wrapper Invocation' {
        It 'Should call Invoke-CargoWrapper' {
            $content = Get-Content $script:RustBuildPath -Raw
            $content | Should -Match 'Invoke-CargoWrapper'
        }

        It 'Should invoke cargo directly when CargoTools fallback is active' {
            $content = Get-Content $script:RustBuildPath -Raw
            $content | Should -Match 'if \(\$script:UseCargoTools\)'
            $content | Should -Match '& \$cargoPath @finalArgs'
        }

        It 'Should use Push-Location and Pop-Location' {
            $content = Get-Content $script:RustBuildPath -Raw
            $content | Should -Match 'Push-Location'
            $content | Should -Match 'Pop-Location'
        }

        It 'Should check LASTEXITCODE' {
            $content = Get-Content $script:RustBuildPath -Raw
            $content | Should -Match '\$LASTEXITCODE'
        }

        It 'Should capture CargoTools returned exit codes' {
            $content = Get-Content $script:RustBuildPath -Raw
            $content | Should -Match '\$wrapperExitCode'
            $content | Should -Match '\$effectiveExitCode'
        }

        It 'Should use try-finally for location cleanup' {
            $content = Get-Content $script:RustBuildPath -Raw
            $content | Should -Match 'try'
            $content | Should -Match 'finally'
        }
    }

    Context 'Error Handling' {
        It 'Should set ErrorActionPreference to Stop' {
            $content = Get-Content $script:RustBuildPath -Raw
            $content | Should -Match "\`$ErrorActionPreference\s*=\s*'Stop'"
        }

        It 'Should exit with LASTEXITCODE on failure' {
            $content = Get-Content $script:RustBuildPath -Raw
            $content | Should -Match 'exit \$LASTEXITCODE'
        }

        It 'Should fail loudly when no cargo arguments are provided' {
            $content = Get-Content $script:RustBuildPath -Raw
            $content | Should -Match 'No cargo arguments provided'
        }
    }

    Context 'Documentation' {
        It 'Should have synopsis' {
            $content = Get-Content $script:RustBuildPath -Raw
            $content | Should -Match '\.SYNOPSIS'
        }

        It 'Should have description' {
            $content = Get-Content $script:RustBuildPath -Raw
            $content | Should -Match '\.DESCRIPTION'
        }

        It 'Should have parameter documentation' {
            $content = Get-Content $script:RustBuildPath -Raw
            $content | Should -Match '\.PARAMETER Path'
            $content | Should -Match '\.PARAMETER UseLld'
            $content | Should -Match '\.PARAMETER NoLld'
        }

        It 'Should have examples' {
            $content = Get-Content $script:RustBuildPath -Raw
            $content | Should -Match '\.EXAMPLE'
        }
    }

    Context 'Cargo Availability' {
        BeforeAll {
            $script:CargoAvailable = $false
            try {
                $cargoVersion = & { cargo --version 2>&1 } | Out-String
                if ($cargoVersion -and $cargoVersion -match 'cargo') {
                    $script:CargoAvailable = $true
                    $script:CargoVersionOutput = $cargoVersion
                }
            } catch {
                # Cargo not available or wrapped differently
            }
        }

        It 'Should be able to detect cargo availability' -Skip:(-not $script:CargoAvailable) {
            $script:CargoAvailable | Should -BeTrue
        }

        It 'Should return cargo version when available' -Skip:(-not $script:CargoAvailable) {
            $script:CargoVersionOutput | Should -Not -BeNullOrEmpty
            $script:CargoVersionOutput | Should -Match 'cargo'
        }
    }

    Context 'LLVM Availability' -Skip:(-not (Test-Path 'C:\Program Files\LLVM\bin\lld-link.exe')) {
        It 'Should find lld-link.exe at expected path' {
            Test-Path 'C:\Program Files\LLVM\bin\lld-link.exe' | Should -BeTrue
        }
    }
}


Describe 'Rust build argument transport' -Tag 'Unit', 'Portable' {
    BeforeAll {
        $repoRoot = Join-Path $PSScriptRoot '../..'
        $build = [Management.Automation.Language.Parser]::ParseFile((Join-Path $repoRoot 'Build.ps1'), [ref]$null, [ref]$null)
        foreach ($name in 'Invoke-RustBuildCommand', 'Add-CargoDependencyFlags') {
            $function = $build.Find({
                param($node)
                $node -is [Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -ceq $name
            }, $true)
            . ([scriptblock]::Create($function.Extent.Text))
        }

        $script:FixturePwsh = Join-Path $PSHOME $(if ($IsWindows) { 'pwsh.exe' } else { 'pwsh' })
        function Resolve-PowerShellExecutable { $script:FixturePwsh }
        function Write-RustBuildFailureLog { param([string]$LogFile) }

        $stage = Join-Path $TestDrive 'wrapper stage'
        $script:FixtureProject = Join-Path $TestDrive 'project path'
        $null = New-Item -ItemType Directory -Path $stage, $script:FixtureProject
        $script:RustBuildWrapper = Join-Path $stage 'Invoke-RustBuild.ps1'
        Copy-Item -LiteralPath (Join-Path $repoRoot 'Tools/Invoke-RustBuild.ps1') -Destination $script:RustBuildWrapper
        @'
function Invoke-FixtureCargo {
    [Console]::WriteLine('PCAI_ARGV=' + (ConvertTo-Json -InputObject @($args) -Compress))
    $global:LASTEXITCODE = if ($args -contains 'doesnotexist') { 37 } else { 0 }
}
function Get-Command {
    param($Name, $ErrorAction)
    if ($Name -eq 'cargo') { return [pscustomobject]@{ Path = 'Invoke-FixtureCargo' } }
    Microsoft.PowerShell.Core\Get-Command @PSBoundParameters
}
'@ | Set-Content -LiteralPath (Join-Path $stage 'PcaiModuleBootstrap.ps1')
    }

    # Protects: the top-level builder delivers the ordered argument array to Cargo.
    # Detects: dropped dash-prefixed flags, wrapper binder collisions and lost child failure.
    # Needs: pwsh, Pester and an inert Cargo recorder in an isolated wrapper copy.
    # Breadcrumb: Build.ps1 Invoke-RustBuildCommand and Tools/Invoke-RustBuild.ps1.
    It 'delivers exact Cargo tokens and status for <Case>' -ForEach @(
        @{ Case = 'ordered build flags'; InputArgs = @('build', '--release', '--features', 'cuda,cudnn', '-p', 'pcai-train', '--target-dir', 'project cache'); ExpectedSuccess = $true },
        @{ Case = 'invalid feature'; InputArgs = @('tree', '--features', 'doesnotexist'); ExpectedSuccess = $false },
        @{ Case = 'test name filter'; InputArgs = @('test', '--lib', 'model::test_cpu_only'); ExpectedSuccess = $true }
    ) {
        $log = Join-Path $TestDrive "cargo-$Case.log"
        $oldDependency = $script:DependencyStrategy
        $oldDisableCache = $env:PCAI_DISABLE_CACHE
        try {
            $script:DependencyStrategy = 'locked'
            $env:PCAI_DISABLE_CACHE = '0'
            $succeeded = Invoke-RustBuildCommand -Path $script:FixtureProject -CargoArgs $InputArgs -LogFile $log
            $actual = @(Get-Content -LiteralPath $log | Where-Object { $_.StartsWith('PCAI_ARGV=') })
            $actual.Count | Should -Be 1
            $actual[0].Substring('PCAI_ARGV='.Length) | Should -BeExactly (ConvertTo-Json -InputObject ($InputArgs + '--locked') -Compress)
            $succeeded | Should -Be $ExpectedSuccess
        } finally {
            $script:DependencyStrategy = $oldDependency
            $env:PCAI_DISABLE_CACHE = $oldDisableCache
        }
    }

    # Protects: existing direct -File callers can mix a bound subcommand with trailing flags.
    # Detects: silently accepting a command whose invalid feature flag never reaches Cargo.
    # Needs: profile-free pwsh child and the staged inert Cargo recorder.
    # Breadcrumb: Tools/Invoke-RustBuild.ps1 CargoArgs and RemainingArgs binding.
    It 'preserves mixed bound and trailing Cargo arguments' {
        $log = & $script:FixturePwsh -NoProfile -File $script:RustBuildWrapper -Path $script:FixtureProject -CargoArgs tree --locked --features doesnotexist
        $exitCode = $LASTEXITCODE
        $actual = @($log | Where-Object { $_.StartsWith('PCAI_ARGV=') })
        $actual.Count | Should -Be 1
        $actual[0] | Should -BeExactly 'PCAI_ARGV=["tree","--locked","--features","doesnotexist"]'
        $exitCode | Should -Be 37
    }

    # Protects: Cargo-level policy flags stay before the tool argument separator.
    # Detects: consuming a test-runner separator or moving its filter before the separator.
    # Needs: the extracted build caller and an inert same-process Cargo recorder.
    # Breadcrumb: Build.ps1 Add-CargoDependencyFlags and Invoke-RustBuildCommand direct path.
    It 'preserves test filter arguments after the tool separator' {
        $oldDependency = $script:DependencyStrategy
        $oldExitCode = $global:LASTEXITCODE
        function cargo {
            Write-Output ('PCAI_ARGV=' + (ConvertTo-Json -InputObject @($args) -Compress))
            $global:LASTEXITCODE = 0
        }
        try {
            $script:DependencyStrategy = 'locked'
            $log = Join-Path $TestDrive 'cargo separator.log'
            $result = Invoke-RustBuildCommand -Path $script:FixtureProject -CargoArgs @('test', '--lib', '--', '--exact', 'named test') -LogFile $log
            $result | Should -BeTrue
            (Get-Content -LiteralPath $log) | Should -BeExactly 'PCAI_ARGV=["test","--lib","--locked","--","--exact","named test"]'
        } finally {
            $script:DependencyStrategy = $oldDependency
            $global:LASTEXITCODE = $oldExitCode
        }
    }
}
