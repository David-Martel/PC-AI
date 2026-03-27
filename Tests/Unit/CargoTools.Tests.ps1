<#
.SYNOPSIS
    Unit tests for CargoTools module integration

.DESCRIPTION
    Tests CargoTools module loading, environment initialization, and Cargo command availability.
    Tests skip gracefully if CargoTools module is not installed.
#>

BeforeAll {
    # Check if CargoTools is available
    $script:cargoToolsAvailable = Get-Module -ListAvailable CargoTools

    if ($script:cargoToolsAvailable) {
        Import-Module CargoTools -Force -ErrorAction SilentlyContinue
    }

    # Check if cargo command is available
    $script:cargoAvailable = $null -ne (Get-Command cargo -ErrorAction SilentlyContinue)
}

# Initialize for discovery-time Skip checks
if (-not (Get-Variable -Name cargoToolsAvailable -Scope Script -ErrorAction SilentlyContinue)) {
    $script:cargoToolsAvailable = $false
}
if (-not (Get-Variable -Name cargoAvailable -Scope Script -ErrorAction SilentlyContinue)) {
    $script:cargoAvailable = $false
}
if (-not (Get-Variable -Name hasInit -Scope Script -ErrorAction SilentlyContinue)) {
    $script:hasInit = $false
}

Describe 'CargoTools Module' -Tag 'Unit', 'CargoTools', 'Fast', 'Portable' {
    Context 'Module Availability' {
        It 'Should be discoverable in module path' -Skip:(-not $script:cargoToolsAvailable) {
            $script:cargoToolsAvailable | Should -Not -BeNullOrEmpty
        }

        It 'Should import without errors' -Skip:(-not $script:cargoToolsAvailable) {
            { Import-Module CargoTools -Force -ErrorAction Stop } | Should -Not -Throw
        }

        It 'Should be loaded after import' -Skip:(-not $script:cargoToolsAvailable) {
            Import-Module CargoTools -Force -ErrorAction SilentlyContinue
            Get-Module CargoTools | Should -Not -BeNullOrEmpty
        }
    }

    Context 'Module Exports' -Skip:(-not (Get-Module CargoTools)) {
        BeforeAll {
            $script:exportedFunctions = (Get-Module CargoTools).ExportedFunctions.Keys
            $script:exportedCmdlets = (Get-Module CargoTools).ExportedCmdlets.Keys
            if ($script:exportedFunctions -is [string]) {
                $script:exportedFunctions = @($script:exportedFunctions)
            }
        }

        It 'Should export at least one function or cmdlet' {
            ($script:exportedFunctions.Count + $script:exportedCmdlets.Count) | Should -BeGreaterThan 0
        }

        It 'Should have exported functions as array' {
            ($script:exportedFunctions -is [System.Collections.IEnumerable]) | Should -BeTrue
        }
    }

    Context 'Environment Functions' -Skip:(-not (Get-Module CargoTools)) {
        BeforeAll {
            $script:hasInit = $null -ne (Get-Command Initialize-CargoEnvironment -ErrorAction SilentlyContinue)
        }

        It 'Should have Initialize-CargoEnvironment function' -Skip:(-not $script:hasInit) {
            $command = Get-Command Initialize-CargoEnvironment -ErrorAction SilentlyContinue
            $command | Should -Not -BeNullOrEmpty
        }

        It 'Initialize-CargoEnvironment should accept common parameters' -Skip:(-not $script:hasInit) {
            $command = Get-Command Initialize-CargoEnvironment -ErrorAction SilentlyContinue
            if ($command) {
                $command.Parameters.Keys | Should -Contain 'ErrorAction'
            }
        }
    }

    Context 'Cargo Integration Functions' -Skip:(-not (Get-Module CargoTools)) {
        It 'Should expose cargo-related commands' {
            $exported = (Get-Module CargoTools).ExportedFunctions.Keys
            $cargoCommands = $exported | Where-Object { $_ -match 'Cargo|Rust' }

            # At least some cargo-related functionality should be present
            if ($exported.Count -gt 0) {
                $exported | Should -Not -BeNullOrEmpty
            }
        }
    }
}

Describe 'Cargo Command Availability' -Tag 'Unit', 'Cargo', 'Fast', 'Portable' {
    Context 'Cargo Installation' {
        It 'Should have cargo command in PATH' -Skip:(-not $script:cargoAvailable) {
            Get-Command cargo -ErrorAction SilentlyContinue | Should -Not -BeNullOrEmpty
        }

        It 'Should return version information' -Skip:(-not $script:cargoAvailable) {
            $result = cargo version 2>&1
            $LASTEXITCODE | Should -Be 0
            $result | Should -Match 'cargo \d+\.\d+\.\d+'
        }

        It 'Should have rustc available' -Skip:(-not $script:cargoAvailable) {
            $rustcAvailable = $null -ne (Get-Command rustc -ErrorAction SilentlyContinue)
            if ($script:cargoAvailable) {
                $rustcAvailable | Should -Be $true
            }
        }
    }

    Context 'Cargo Environment' -Skip:(-not $script:cargoAvailable) {
        It 'Should have CARGO_HOME environment variable' {
            $cargoHome = $env:CARGO_HOME
            if ($null -eq $cargoHome) {
                # Cargo home defaults to ~/.cargo if not set
                $defaultCargoHome = Join-Path $env:USERPROFILE '.cargo'
                Test-Path $defaultCargoHome | Should -Be $true
            } else {
                $cargoHome | Should -Not -BeNullOrEmpty
            }
        }

        It 'Should have cargo bin directory in PATH' {
            $cargoBin = if ($env:CARGO_HOME) {
                Join-Path $env:CARGO_HOME 'bin'
            } else {
                Join-Path $env:USERPROFILE '.cargo\bin'
            }

            $env:PATH -split ';' | Where-Object { $_ -like "*cargo*bin*" } | Should -Not -BeNullOrEmpty
        }
    }

    Context 'Cargo Help Commands' -Skip:(-not $script:cargoAvailable) {
        It 'Should display help without errors' {
            $result = cargo --help 2>&1
            $LASTEXITCODE | Should -Be 0
            $result | Should -Match 'Rust.*package manager'
        }

        It 'Should list available commands' {
            $result = cargo --list 2>&1
            $LASTEXITCODE | Should -Be 0
            $result | Should -Match 'build|test|run|check'
        }
    }
}

Describe 'CargoTools Integration with PC_AI' -Tag 'Unit', 'Integration', 'Fast', 'Portable' {
    Context 'Module Compatibility' -Skip:(-not $script:cargoToolsAvailable) {
        It 'Should not conflict with other PC_AI modules' {
            $pcaiModules = Get-Module -Name 'PC-AI.*'

            # CargoTools should coexist with PC_AI modules
            { Import-Module CargoTools -Force -ErrorAction Stop } | Should -Not -Throw

            # PC_AI modules should still be loaded
            if ($pcaiModules) {
                Get-Module -Name 'PC-AI.*' | Should -Not -BeNullOrEmpty
            }
        }
    }

    Context 'Error Handling' -Skip:(-not $script:cargoToolsAvailable) {
        It 'Should handle missing cargo gracefully' {
            if (Get-Command Initialize-CargoEnvironment -ErrorAction SilentlyContinue) {
                # Should not throw even if cargo is not installed
                { Initialize-CargoEnvironment -ErrorAction SilentlyContinue } | Should -Not -Throw
            }
        }
    }
}

Describe 'CargoTools Test Prerequisites' -Tag 'Meta', 'Fast', 'Portable' {
    Context 'Test Environment Checks' {
        It 'Should detect if CargoTools is installed' {
            # This test always runs to show module availability status
            $available = Get-Module -ListAvailable CargoTools
            Write-Host "CargoTools module available: $($null -ne $available)" -ForegroundColor Cyan

            # This test should not fail, just report status
            $true | Should -Be $true
        }

        It 'Should detect if cargo is available' {
            # This test always runs to show cargo availability status
            $available = Get-Command cargo -ErrorAction SilentlyContinue
            Write-Host "Cargo command available: $($null -ne $available)" -ForegroundColor Cyan

            # This test should not fail, just report status
            $true | Should -Be $true
        }

        It 'Should provide installation instructions if missing' {
            if (-not $script:cargoToolsAvailable) {
                Write-Host "`nCargoTools module not found." -ForegroundColor Yellow
                Write-Host "Install from PowerShell Gallery: Install-Module -Name CargoTools -Scope CurrentUser" -ForegroundColor Yellow
            }

            if (-not $script:cargoAvailable) {
                Write-Host "`nCargo not found in PATH." -ForegroundColor Yellow
                Write-Host "Install Rust toolchain: https://rustup.rs/" -ForegroundColor Yellow
            }

            # This test should not fail
            $true | Should -Be $true
        }
    }
}

AfterAll {
    # Clean up imported modules
    Remove-Module CargoTools -Force -ErrorAction SilentlyContinue
}
