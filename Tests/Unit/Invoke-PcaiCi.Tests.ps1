#Requires -Version 7.0
#Requires -Modules @{ ModuleName = 'Pester'; ModuleVersion = '5.0.0' }

# Why this file exists: Tests/Invoke-PcaiCi.ps1 promises to run "the same checks as CI". That
# promise rots silently when ci.yml changes, so these tests compare the script to ci.yml and
# prove the accelerator switches behave. See Tests/Invoke-PcaiCi.ps1 and .github/workflows/ci.yml.

BeforeAll {
    $script:RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
    $script:Runner = Join-Path $script:RepoRoot 'Tests/Invoke-PcaiCi.ps1'
    $script:CiText = Get-Content -LiteralPath (Join-Path $script:RepoRoot '.github/workflows/ci.yml') -Raw
    # Probe name that cannot exist, to simulate a machine without the accelerator.
    $script:Absent = 'pcai-no-such-tool-zzz'
    $script:Present = 'pwsh'

    function Get-Plan {
        param([hashtable]$Parameters = @{})
        @(& $script:Runner -Plan -PassThru @Parameters)
    }
}

Describe 'Invoke-PcaiCi plan' -Tag 'Unit', 'Portable' {
    It 'lists the Quick tier by default and the Full tier as a superset' {
        $quick = Get-Plan
        $full = Get-Plan @{ Tier = 'Full' }
        $quick.Id | Should -Be @('CredentialScan', 'WorkflowSyntax', 'PowerShellLintChanged', 'RustFmt')
        $full.Id | Should -Be @('CredentialScan', 'WorkflowSyntax', 'PowerShellLint', 'RustFmt', 'RustCheck', 'RustClippy', 'RustTest', 'PesterUnitBoot')
    }

    It 'runs nothing in plan mode' {
        $resultsDir = Join-Path $TestDrive 'results'
        Get-Plan @{ ResultsDir = $resultsDir } | Out-Null
        Test-Path $resultsDir | Should -BeFalse
    }

    It 'filters with -Only and -Skip and rejects unknown ids' {
        (Get-Plan @{ Tier = 'Full'; Only = @('RustFmt', 'RustTest') }).Id | Should -Be @('RustFmt', 'RustTest')
        (Get-Plan @{ Skip = @('RustFmt') }).Id | Should -Not -Contain 'RustFmt'
        { Get-Plan @{ Only = @('NoSuchStep') } } | Should -Throw '*Unknown step id*'
        # A Full-tier step requested from the default tier must run, not vanish.
        (Get-Plan @{ Only = @('RustCheck') }).Id | Should -Be @('RustCheck')
        { Get-Plan @{ Skip = @('CredentialScan', 'WorkflowSyntax', 'PowerShellLintChanged', 'RustFmt') } } | Should -Throw '*No steps selected*'
    }
}

Describe 'Invoke-PcaiCi mirrors ci.yml' -Tag 'Unit', 'Portable' {
    # Why: a step whose command no longer appears in ci.yml means the local run checks something
    # CI does not, which is worse than having no local run.
    It 'every step command appears verbatim in ci.yml' -ForEach @(
        @{ Id = 'CredentialScan' }, @{ Id = 'WorkflowSyntax' }, @{ Id = 'PowerShellLint' }, @{ Id = 'RustFmt' },
        @{ Id = 'RustCheck' }, @{ Id = 'RustClippy' }, @{ Id = 'RustTest' }, @{ Id = 'PesterUnitBoot' }
    ) {
        param($Id)
        $step = (Get-Plan @{ Tier = 'Full'; NoNextest = $true }) | Where-Object Id -eq $Id
        $step.CiCommand | Should -Not -BeNullOrEmpty
        $script:CiText.Contains($step.CiCommand) | Should -BeTrue -Because "ci.yml must contain '$($step.CiCommand)'"
    }

    It 'every script a step launches exists' {
        foreach ($step in @(Get-Plan @{ Tier = 'Full' }) + @(Get-Plan)) {
            foreach ($arg in $step.Command) {
                if ($arg -like '*.ps1') { Test-Path -LiteralPath $arg | Should -BeTrue -Because "$($step.Id) launches $arg" }
            }
        }
    }

    It 'the Pester path list equals the list ci.yml passes to Invoke-Pester' {
        $match = [regex]::Match($script:CiText, '(?s)\$config\.Run\.Path = @\((?<body>.*?)\)')
        $match.Success | Should -BeTrue
        $ciPaths = [regex]::Matches($match.Groups['body'].Value, "'([^']+)'") | ForEach-Object { $_.Groups[1].Value }
        $step = (Get-Plan @{ Tier = 'Full' }) | Where-Object Id -eq 'PesterUnitBoot'
        $runnerPaths = [regex]::Matches($step.Command[-1], "'(Tests/[^']+)'") | ForEach-Object { $_.Groups[1].Value } |
            Where-Object { $_ -notmatch 'PesterConfiguration\.psd1$' }
        $runnerPaths | Should -Be $ciPaths
    }
}

Describe 'Invoke-PcaiCi nextest accelerator' -Tag 'Unit', 'Portable' {
    It 'uses nextest only when it is installed and not disabled' {
        $with = (Get-Plan @{ Tier = 'Full'; NextestCommand = $script:Present }) | Where-Object Id -eq 'RustTest'
        $with.Command[0..2] | Should -Be @('cargo', 'nextest', 'run')
        foreach ($parameters in @{ Tier = 'Full'; NextestCommand = $script:Absent }, @{ Tier = 'Full'; NextestCommand = $script:Present; NoNextest = $true }) {
            $without = (Get-Plan $parameters) | Where-Object Id -eq 'RustTest'
            $without.Command[0..1] | Should -Be @('cargo', 'test')
        }
        $with.Command | Should -Contain '--lib'
        $with.WorkingDirectory | Should -Be (Join-Path $script:RepoRoot 'Native/pcai_core/pcai_inference')
    }
}

Describe 'Invoke-PcaiCi execution' -Tag 'Unit', 'Portable' {
    # Positive control: a real step runs, logs and reports PASS. A runner that cannot fail would
    # pass every other test, so the second case proves a failing step exits non-zero.
    It 'runs a step, writes its log and result JSON' {
        $resultsDir = Join-Path $TestDrive 'run'
        $results = @(& $script:Runner -Only WorkflowSyntax -PassThru -ResultsDir $resultsDir 6>$null)
        $results.Count | Should -Be 1
        $results[0].Status | Should -Be 'PASS'
        Test-Path (Join-Path $resultsDir 'pcai-ci-WorkflowSyntax.log') | Should -BeTrue
        (Get-Content (Join-Path $resultsDir 'pcai-ci-last.json') -Raw | ConvertFrom-Json).Results[0].Id | Should -Be 'WorkflowSyntax'
    }

    It 'exits non-zero when a step fails' {
        $fixture = Join-Path $TestDrive 'fail-repo'
        [IO.Directory]::CreateDirectory((Join-Path $fixture 'Tests')) | Out-Null
        [IO.Directory]::CreateDirectory((Join-Path $fixture 'Tools')) | Out-Null
        Copy-Item $script:Runner (Join-Path $fixture 'Tests/Invoke-PcaiCi.ps1')
        [IO.File]::WriteAllText((Join-Path $fixture 'Tools/Test-WorkflowSyntax.ps1'), 'exit 3', [Text.UTF8Encoding]::new($false))
        $out = pwsh -NoProfile -File (Join-Path $fixture 'Tests/Invoke-PcaiCi.ps1') -Only WorkflowSyntax -ResultsDir (Join-Path $fixture 'r') 2>&1
        $LASTEXITCODE | Should -Be 1
        ($out | Out-String) | Should -Match 'FAIL'
    }

    # Protects: a relative -ResultsDir names one directory, relative to the caller. Detects: the
    # directory being created against one working directory while step logs are written after
    # Push-Location into the step's directory, which made Tee-Object throw, the catch repeat the bad
    # write and the runner stop without a summary (codex review of #337).
    It 'writes logs and the summary under a relative -ResultsDir resolved from the caller' {
        $fixture = Join-Path $TestDrive 'relative-repo'
        $caller = Join-Path $TestDrive 'caller'
        foreach ($dir in (Join-Path $fixture 'Tests'), (Join-Path $fixture 'Tools'), $caller) {
            [IO.Directory]::CreateDirectory($dir) | Out-Null
        }
        Copy-Item $script:Runner (Join-Path $fixture 'Tests/Invoke-PcaiCi.ps1')
        [IO.File]::WriteAllText((Join-Path $fixture 'Tools/Test-WorkflowSyntax.ps1'), "'workflow syntax ok'; exit 0", [Text.UTF8Encoding]::new($false))
        Push-Location $caller
        try {
            $out = pwsh -NoProfile -File (Join-Path $fixture 'Tests/Invoke-PcaiCi.ps1') -Only WorkflowSyntax -ResultsDir 'ci-results' 2>&1
            $exit = $LASTEXITCODE
        } finally {
            Pop-Location
        }
        $exit | Should -Be 0 -Because ($out | Out-String)
        Test-Path (Join-Path $caller 'ci-results/pcai-ci-WorkflowSyntax.log') | Should -BeTrue
        (Get-Content (Join-Path $caller 'ci-results/pcai-ci-last.json') -Raw | ConvertFrom-Json).Results[0].Status | Should -Be 'PASS'
        Test-Path (Join-Path $fixture 'ci-results') | Should -BeFalse
    }
}
