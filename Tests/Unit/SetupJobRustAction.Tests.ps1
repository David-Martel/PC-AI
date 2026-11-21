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
    BeforeAll {
        # Runs the prepare step with RUNNER_TEMP and the GITHUB_* files in TestDrive and returns its GITHUB_ENV lines.
        function Invoke-PrepareStep {
            param([string]$RunnerTemp)
            $saved = @{}
            foreach ($name in 'RUNNER_TEMP', 'GITHUB_ENV', 'GITHUB_PATH', 'GITHUB_OUTPUT', 'GITHUB_WORKSPACE') {
                $saved[$name] = [Environment]::GetEnvironmentVariable($name)
            }
            try {
                $env:RUNNER_TEMP = $RunnerTemp
                if (-not (Test-Path -LiteralPath $RunnerTemp)) { New-Item -ItemType Directory -Path $RunnerTemp | Out-Null }
                $env:GITHUB_WORKSPACE = Join-Path $TestDrive 'work'
                foreach ($name in 'GITHUB_ENV', 'GITHUB_PATH', 'GITHUB_OUTPUT') {
                    $file = Join-Path $TestDrive $name
                    Set-Content -LiteralPath $file -Value ''
                    [Environment]::SetEnvironmentVariable($name, $file)
                }
                & (Get-ActionStep 'Prepare job-local self-hosted Rust homes')
                Get-Content -LiteralPath $env:GITHUB_ENV
            }
            finally {
                foreach ($name in $saved.Keys) { [Environment]::SetEnvironmentVariable($name, $saved[$name]) }
            }
        }
        function Get-Export {
            param([string[]]$Lines, [string]$Name)
            ($Lines | Where-Object { $_ -like "$Name=*" }).Substring($Name.Length + 1)
        }
    }

    # Protects: sccache hits across jobs on one runner. The runner empties RUNNER_TEMP between jobs, so two jobs see
    # the same RUNNER_TEMP path, and the Rust homes under it must get the same absolute path too.
    # Detects: a per-job name such as pcai-rust-<guid>. sccache 0.19.0 keys Rust compiles on the absolute CARGO_HOME
    # path even when SCCACHE_BASEDIRS names it: rust-linux on dtm-super-nuc had 0 Rust hits in 640 compiles (run
    # 44893813238), and minisforum-ms01 reproduced 0/7 with a random root against 7/7 with a fixed one.
    # Needs: PowerShell 7 and Pester 5; the step runs against TestDrive files, nothing is installed.
    # Breadcrumb: .github/actions/setup-job-rust/action.yml "Prepare job-local self-hosted Rust homes".
    It 'gives two jobs with the same RUNNER_TEMP the same CARGO_HOME and RUSTUP_HOME' {
        $runnerTemp = Join-Path $TestDrive 'job-temp'
        $first = Invoke-PrepareStep -RunnerTemp $runnerTemp
        Remove-Item -LiteralPath $runnerTemp -Recurse -Force   # what the runner does between jobs
        $second = Invoke-PrepareStep -RunnerTemp $runnerTemp
        (Get-Export $second 'CARGO_HOME') | Should -Be (Get-Export $first 'CARGO_HOME')
        (Get-Export $second 'RUSTUP_HOME') | Should -Be (Get-Export $first 'RUSTUP_HOME')
        (Get-Export $first 'CARGO_HOME') | Should -BeLike "$runnerTemp*" -Because 'the homes stay job-local'
    }

    # Protects: every use starts from empty Rust homes although the root name is fixed.
    # Detects: a stale toolchain or registry left by an earlier use in the same job leaking into the next one, or the
    # step failing because the fixed directory already exists.
    # Needs: PowerShell 7 and Pester 5. Breadcrumb: .github/actions/setup-job-rust/action.yml.
    It 'replaces Rust homes left by an earlier use in the same job' {
        $runnerTemp = Join-Path $TestDrive 'reuse-temp'
        $first = Invoke-PrepareStep -RunnerTemp $runnerTemp
        $stale = Join-Path (Get-Export $first 'CARGO_HOME') 'stale.txt'
        Set-Content -LiteralPath $stale -Value 'left by an earlier use'
        $second = Invoke-PrepareStep -RunnerTemp $runnerTemp
        (Get-Export $second 'CARGO_HOME') | Should -Be (Get-Export $first 'CARGO_HOME')
        Test-Path -LiteralPath $stale | Should -BeFalse
        Test-Path -LiteralPath (Join-Path (Get-Export $second 'CARGO_HOME') 'bin') | Should -BeTrue
    }

    # Protects: keys still repeat on a runner whose RUNNER_TEMP or workspace path differs from the one that filled the
    # cache, where sccache strips the base directories it supports.
    # Detects: SCCACHE_BASEDIRS missing the Rust root or the workspace, or naming a relative directory.
    # Needs: PowerShell 7 and Pester 5. Breadcrumb: .github/actions/setup-job-rust/action.yml.
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
