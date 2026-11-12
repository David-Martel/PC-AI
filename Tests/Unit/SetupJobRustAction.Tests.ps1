#Requires -Version 7.0

BeforeAll {
    $script:ActionText = Get-Content -LiteralPath (Join-Path $PSScriptRoot '../../.github/actions/setup-job-rust/action.yml') -Raw
    function Get-ActionStep {
        param([string]$Name)
        $pattern = '(?ms)^    - name: ' + [regex]::Escape($Name) + '\r?\n(?<step>.*?)(?=^    - |\z)'
        $step = [regex]::Match($script:ActionText, $pattern).Groups['step'].Value
        $run = [regex]::Match($step, '(?ms)^      run: \|\r?\n(?<body>.*)$').Groups['body'].Value
        if (-not $run) { throw "Missing action run block: $Name" }
        [scriptblock]::Create(($run -replace '(?m)^        ', ''))
    }
}

Describe 'Job-local Rust keeps sccache keys stable across jobs' -Tag 'Unit', 'Portable' {
    # Protects: sccache can hit across jobs although every job builds from a fresh $RUNNER_TEMP/pcai-rust-<guid> home.
    # Detects: a missing SCCACHE_BASEDIRS. Without it, the dependency sources under the per-job CARGO_HOME give every
    # compile a new absolute path, and the rust-linux job on dtm-super-nuc showed 0 hits in 690 compiles.
    # Needs: PowerShell 7 and Pester 5; the step runs against TestDrive files, nothing is installed.
    # Breadcrumb: .github/actions/setup-job-rust/action.yml "Prepare job-local self-hosted Rust homes".
    It 'exports SCCACHE_BASEDIRS naming the job Rust root and the workspace' {
        $saved = @{}
        foreach ($name in 'RUNNER_TEMP', 'GITHUB_ENV', 'GITHUB_PATH', 'GITHUB_OUTPUT', 'GITHUB_WORKSPACE') {
            $saved[$name] = [Environment]::GetEnvironmentVariable($name)
        }
        try {
            $env:RUNNER_TEMP = Join-Path $TestDrive 'temp'
            New-Item -ItemType Directory -Path $env:RUNNER_TEMP | Out-Null
            $env:GITHUB_WORKSPACE = Join-Path $TestDrive 'work'
            foreach ($name in 'GITHUB_ENV', 'GITHUB_PATH', 'GITHUB_OUTPUT') {
                $file = Join-Path $TestDrive $name
                Set-Content -LiteralPath $file -Value ''
                [Environment]::SetEnvironmentVariable($name, $file)
            }
            & (Get-ActionStep 'Prepare job-local self-hosted Rust homes')
            $exports = Get-Content -LiteralPath $env:GITHUB_ENV
            $line = $exports | Where-Object { $_ -like 'SCCACHE_BASEDIRS=*' }
            $line | Should -HaveCount 1
            $dirs = $line.Substring('SCCACHE_BASEDIRS='.Length).Split([IO.Path]::PathSeparator)
            $rustRoot = Split-Path -Parent (($exports | Where-Object { $_ -like 'CARGO_HOME=*' }).Substring('CARGO_HOME='.Length))
            $dirs | Should -Contain $rustRoot
            $dirs | Should -Contain $env:GITHUB_WORKSPACE
            foreach ($dir in $dirs) { [IO.Path]::IsPathRooted($dir) | Should -BeTrue -Because 'sccache refuses relative base directories' }
        }
        finally {
            foreach ($name in $saved.Keys) { [Environment]::SetEnvironmentVariable($name, $saved[$name]) }
        }
    }

    # Protects: the new base directories reach the sccache server.
    # Detects: a server left running from an earlier job, which keeps the base directories it started with.
    # Needs: the action text only. Breadcrumb: .github/actions/setup-job-rust/action.yml.
    It 'restarts any running sccache server after exporting the base directories and before installing Rust' {
        $prepare = $script:ActionText.IndexOf('- name: Prepare job-local self-hosted Rust homes')
        $restart = $script:ActionText.IndexOf('--stop-server')
        $install = $script:ActionText.IndexOf('- name: Install job-local Rust')
        $restart | Should -BeGreaterThan $prepare
        $restart | Should -BeLessThan $install
    }
}
