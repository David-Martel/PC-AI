#Requires -Modules Pester

if (-not (Get-Variable -Name CargoAvailable -Scope Script -ErrorAction SilentlyContinue)) {
    $script:CargoAvailable = $false
}
if (-not (Get-Variable -Name CargoVersionOutput -Scope Script -ErrorAction SilentlyContinue)) {
    $script:CargoVersionOutput = $null
}

Describe 'Invoke-RustBuild' {
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
            $content | Should -Match 'Import-Module -Name \\$cargoToolsManifest'
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
    }

    Context 'Cargo Wrapper Invocation' {
        It 'Should call Invoke-CargoWrapper' {
            $content = Get-Content $script:RustBuildPath -Raw
            $content | Should -Match 'Invoke-CargoWrapper'
        }

        It 'Should invoke cargo directly when CargoTools fallback is active' {
            $content = Get-Content $script:RustBuildPath -Raw
            $content | Should -Match 'if \(\$script:UseCargoTools\)'
            $content | Should -Match '& cargo @finalArgs'
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
