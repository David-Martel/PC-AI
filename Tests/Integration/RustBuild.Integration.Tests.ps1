#Requires -Modules Pester

<#
.SYNOPSIS
    Integration tests for Rust build toolchain and PC_AI Rust projects.

.DESCRIPTION
    Verifies that Rust toolchain (cargo, rustc) is available and that
    actual Rust projects (FunctionGemma runtime, train, pcai_core) can
    be built and tested successfully.

.NOTES
    Tests skip gracefully when cargo is not available.
    Run with: Invoke-Pester -Path Tests\Integration\RustBuild.Integration.Tests.ps1 -Tag Integration -Output Detailed
#>

Describe 'Rust Build Integration' -Tag 'Integration', 'Rust' {
    BeforeAll {
        $script:ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot '..\..')
        $script:RustBuildPath = Join-Path $script:ProjectRoot 'Tools\Invoke-RustBuild.ps1'
        $script:RuntimePath = Join-Path $script:ProjectRoot 'Deploy\rust-functiongemma-runtime'
        $script:TrainPath = Join-Path $script:ProjectRoot 'Deploy\rust-functiongemma-train'
        $script:FunctionGemmaWorkspace = Join-Path $script:ProjectRoot 'Deploy\rust-functiongemma'
        $script:CargoAvailable = $null -ne (Get-Command cargo -ErrorAction SilentlyContinue)
    }

    Context 'Build Environment' {
        It 'Should have cargo available' -Skip:(-not $script:CargoAvailable) {
            cargo --version | Should -Not -BeNullOrEmpty
        }

        It 'Should have rustc available' -Skip:(-not $script:CargoAvailable) {
            rustc --version | Should -Not -BeNullOrEmpty
        }

        It 'Should have Invoke-RustBuild.ps1 script' {
            Test-Path $script:RustBuildPath | Should -BeTrue
        }
    }

    Context 'FunctionGemma Runtime Build' -Skip:(-not ($script:CargoAvailable -and (Test-Path $script:RuntimePath) -and (Test-Path $script:FunctionGemmaWorkspace))) {
        It 'Should have Cargo.toml in runtime directory' {
            Test-Path (Join-Path $script:RuntimePath 'Cargo.toml') | Should -BeTrue
        }

        It 'Should pass cargo check' {
            Push-Location $script:FunctionGemmaWorkspace
            try {
                $result = cargo check -p rust-functiongemma-runtime 2>&1
                $LASTEXITCODE | Should -Be 0
            } finally {
                Pop-Location
            }
        }

        It 'Should run tests successfully' {
            Push-Location $script:FunctionGemmaWorkspace
            try {
                cargo test -p rust-functiongemma-runtime --no-fail-fast 2>&1
                $LASTEXITCODE | Should -Be 0
            } finally {
                Pop-Location
            }
        }

        It 'Should pass cargo clippy' {
            Push-Location $script:FunctionGemmaWorkspace
            try {
                cargo clippy -p rust-functiongemma-runtime --all-targets -- -D warnings 2>&1
                $LASTEXITCODE | Should -Be 0
            } finally {
                Pop-Location
            }
        }
    }

    Context 'FunctionGemma Train Build' -Skip:(-not ($script:CargoAvailable -and (Test-Path $script:TrainPath) -and (Test-Path $script:FunctionGemmaWorkspace))) {
        It 'Should have Cargo.toml in train directory' {
            Test-Path (Join-Path $script:TrainPath 'Cargo.toml') | Should -BeTrue
        }

        It 'Should pass cargo check' {
            Push-Location $script:FunctionGemmaWorkspace
            try {
                $result = cargo check -p rust-functiongemma-train 2>&1
                $LASTEXITCODE | Should -Be 0
            } finally {
                Pop-Location
            }
        }

        It 'Should run tests successfully' {
            Push-Location $script:FunctionGemmaWorkspace
            try {
                cargo test -p rust-functiongemma-train --no-fail-fast 2>&1
                $LASTEXITCODE | Should -Be 0
            } finally {
                Pop-Location
            }
        }

        It 'Should pass cargo clippy' {
            Push-Location $script:FunctionGemmaWorkspace
            try {
                cargo clippy -p rust-functiongemma-train --all-targets -- -D warnings 2>&1
                $LASTEXITCODE | Should -Be 0
            } finally {
                Pop-Location
            }
        }
    }

    Context 'pcai_core Build' {
        BeforeAll {
            $script:CorePath = Join-Path $script:ProjectRoot 'Native\pcai_core'
            $script:CorePathExists = Test-Path $script:CorePath
        }

        It 'Should have pcai_core workspace' -Skip:(-not ($script:CargoAvailable -and $script:CorePathExists)) {
            Test-Path (Join-Path $script:CorePath 'Cargo.toml') | Should -BeTrue
        }

        It 'Should pass cargo check for pcai_core_lib' -Skip:(-not ($script:CargoAvailable -and $script:CorePathExists)) {
            Push-Location $script:CorePath
            try {
                cargo check -p pcai_core_lib 2>&1
                $LASTEXITCODE | Should -Be 0
            } finally {
                Pop-Location
            }
        }

        It 'Should run tests for pcai_core_lib' -Skip:(-not ($script:CargoAvailable -and $script:CorePathExists)) {
            Push-Location $script:CorePath
            try {
                cargo test -p pcai_core_lib --no-fail-fast 2>&1
                $LASTEXITCODE | Should -Be 0
            } finally {
                Pop-Location
            }
        }

        It 'Should pass clippy for pcai_core workspace' -Skip:(-not ($script:CargoAvailable -and $script:CorePathExists)) {
            Push-Location $script:CorePath
            try {
                cargo clippy --workspace --all-targets -- -D warnings 2>&1
                $LASTEXITCODE | Should -Be 0
            } finally {
                Pop-Location
            }
        }
    }

    Context 'Build Script Integration' {
        It 'Should execute Invoke-RustBuild.ps1 without errors' -Skip:(-not ($script:CargoAvailable -and (Test-Path $script:RustBuildPath))) {
            # Test with -WhatIf to avoid actual build
            { & $script:RustBuildPath -WhatIf } | Should -Not -Throw
        }
    }
}
