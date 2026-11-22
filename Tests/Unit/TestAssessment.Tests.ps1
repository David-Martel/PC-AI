#Requires -Version 7.0
#Requires -Modules @{ ModuleName = 'Pester'; ModuleVersion = '5.0.0' }

# Why this file exists: the test runners and the coverage setting used to be unable to fail. A syntax-error test
# file left FailedCount at 0, an empty tag filter reported Passed, and a coverage glob analysed 17 of 241 files
# while `CoveragePercentTarget = 85` only coloured the console. Every Describe below plants the failure it is
# named for and asserts the guard reports it, and each one also runs the matching clean case, because a check that
# cannot pass proves as little as one that cannot fail.
# See Tests/coverage-report.ps1, Tests/Helpers/Get-PesterRunVerdict.ps1, Config/test-baseline.json.
#
# Nested Invoke-Pester runs happen in a child `pwsh`, never in this session: Pester 5 does not support running a
# run inside a run. Static assessment is only ever called with -NoRun, so this suite cannot recurse into itself.

BeforeAll {
    $script:RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
    $script:Tool = Join-Path $script:RepoRoot 'Tests/coverage-report.ps1'
    $script:Helper = Join-Path $script:RepoRoot 'Tests/Helpers/Get-PesterRunVerdict.ps1'
    $script:QualityGate = Join-Path $script:RepoRoot 'Tests/Invoke-QualityGate.ps1'
    . $script:Tool
    . $script:QualityGate

    function script:Write-Utf8 {
        param([string]$Path, [string]$Text)
        [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($Path)) | Out-Null
        [IO.File]::WriteAllText($Path, $Text, [Text.UTF8Encoding]::new($false))
    }

    # Runs git against a fixture with a clean identity and no user or system config, so a global
    # commit.gpgsign or hook setting on the developer's machine cannot change what the fixture does.
    function script:Invoke-FixtureGit {
        param([string]$Root, [string[]]$Arguments)
        $saved = @{}
        $vars = @{
            GIT_CONFIG_GLOBAL = (Join-Path ([IO.Path]::GetTempPath()) 'pcai-empty-gitconfig'); GIT_CONFIG_NOSYSTEM = '1'
            GIT_AUTHOR_NAME = 'fixture'; GIT_AUTHOR_EMAIL = 'fixture@example.invalid'
            GIT_COMMITTER_NAME = 'fixture'; GIT_COMMITTER_EMAIL = 'fixture@example.invalid'
        }
        foreach ($k in $vars.Keys) { $saved[$k] = [Environment]::GetEnvironmentVariable($k); [Environment]::SetEnvironmentVariable($k, $vars[$k]) }
        try { & git -C $Root @Arguments 2>&1 } finally {
            foreach ($k in $vars.Keys) { [Environment]::SetEnvironmentVariable($k, $saved[$k]) }
        }
    }

    # One child pwsh runs every Pester case, so the suite pays Pester start-up once.
    $script:ChildRunner = Join-Path $TestDrive 'child-runner.ps1'
    Write-Utf8 $script:ChildRunner @'
param([string]$CasesJson, [string]$HelperPath, [string]$OutPath)
. $HelperPath
Import-Module Pester -MinimumVersion 5.0.0
$out = [ordered]@{}
foreach ($case in ($CasesJson | ConvertFrom-Json)) {
    $c = New-PesterConfiguration
    $c.Run.PassThru = $true
    $c.Output.Verbosity = 'None'
    $c.Run.Path = [string]$case.path
    if ($case.tag) { $c.Filter.Tag = @($case.tag) }
    try {
        $r = Invoke-Pester -Configuration $c
        $known = @{}
        if ($case.known) { foreach ($kp in $case.known.PSObject.Properties) { $known[$kp.Name] = [int]$kp.Value } }
        $v = Get-PesterRunVerdict -Result $r -KnownFailures $known
        $out[$case.name] = [ordered]@{
            verdict = $v
            raw = [ordered]@{ Result = [string]$r.Result; FailedCount = $r.FailedCount; FailedBlocks = $r.FailedBlocksCount; FailedContainers = $r.FailedContainersCount; PassedCount = $r.PassedCount; TotalCount = $r.TotalCount }
        }
    } catch {
        $out[$case.name] = [ordered]@{ threw = $_.Exception.Message }
    }
}
# Pester prints discovery errors to the host; the result goes to a file so stray output cannot corrupt it.
[IO.File]::WriteAllText($OutPath, ($out | ConvertTo-Json -Depth 6))
'@
}

Describe 'Get-PesterRunVerdict plants the failures the old runners missed' -Tag 'Unit', 'Portable' {
    # Protects: every runner judging a Pester run by FailedCount alone. Detects: a test file with a syntax error
    # (Result Failed, FailedCount 0), an empty tag filter (Result Passed, nothing executed) and a path with no test
    # files being reported as a pass. Positive controls: a clean file passes, a failing test fails.
    BeforeAll {
        $fixtures = Join-Path $TestDrive 'verdict'
        Write-Utf8 (Join-Path $fixtures 'broken/broken.Tests.ps1') "Describe 'x' { It 'a' { 1 | Should -Be 1 }`n"
        Write-Utf8 (Join-Path $fixtures 'good/good.Tests.ps1') "Describe 'g' -Tag 'Portable' { It 'a' { 1 | Should -Be 1 } }`n"
        Write-Utf8 (Join-Path $fixtures 'failing/failing.Tests.ps1') "Describe 'f' { It 'a' { 1 | Should -Be 2 } }`n"
        New-Item -ItemType Directory -Path (Join-Path $fixtures 'empty') -Force | Out-Null
        $cases = @(
            @{ name = 'syntax_error'; path = (Join-Path $fixtures 'broken') }
            @{ name = 'empty_tag_filter'; path = (Join-Path $fixtures 'good'); tag = @('NoSuchTag') }
            @{ name = 'no_test_files'; path = (Join-Path $fixtures 'empty') }
            @{ name = 'clean'; path = (Join-Path $fixtures 'good'); tag = @('Portable') }
            @{ name = 'failing_test'; path = (Join-Path $fixtures 'failing') }
        )
        $outFile = Join-Path $TestDrive 'verdict-cases.json'
        pwsh -NoProfile -File $script:ChildRunner -CasesJson ($cases | ConvertTo-Json -Depth 4 -Compress) -HelperPath $script:Helper -OutPath $outFile *> $null
        $script:Cases = Get-Content $outFile -Raw | ConvertFrom-Json
    }

    It 'fails a syntax-error container although FailedCount is 0' {
        $c = $script:Cases.syntax_error
        $c.raw.FailedCount | Should -Be 0 -Because 'this is the blind spot: a FailedCount-only runner passes this'
        $c.raw.FailedContainers | Should -BeGreaterThan 0
        $c.verdict.Passed | Should -BeFalse
        $c.verdict.Violations -join ' ' | Should -Match 'failed to run.*broken\.Tests\.ps1'
    }

    It 'fails an empty tag filter although Pester says Passed' {
        $c = $script:Cases.empty_tag_filter
        $c.raw.Result | Should -Be 'Passed'
        $c.raw.PassedCount | Should -Be 0
        $c.verdict.Passed | Should -BeFalse
        $c.verdict.Violations -join ' ' | Should -Match 'only 0 test'
    }

    It 'fails a path with no test files instead of letting Invoke-Pester throw past the runner' {
        $c = $script:Cases.no_test_files
        # Either Pester throws (the runner must catch it) or it returns an empty run; both must read as failure.
        if ($c.PSObject.Properties['threw']) { $c.threw | Should -Match 'No test files were found' }
        else { $c.verdict.Passed | Should -BeFalse }
    }

    It 'passes a clean run (positive control: the guard can pass)' {
        $c = $script:Cases.clean
        $c.verdict.Passed | Should -BeTrue
        $c.verdict.Executed | Should -Be 1
    }

    It 'fails a failing test' {
        $c = $script:Cases.failing_test
        $c.verdict.Passed | Should -BeFalse
        $c.verdict.Failed | Should -Be 1
    }
}

Describe 'Get-PesterRunVerdict known-failure budget' -Tag 'Unit', 'Portable' {
    # Protects: the ratcheted baseline recording failures tracked elsewhere without admitting new ones. Detects: a
    # file exceeding its budget, a new failing file, or a Result of Failed that no counter explains.
    BeforeAll {
        function script:Get-FakeRun {
            param([hashtable]$FailedByFile = @{}, [string]$Result = 'Failed', [int]$Passed = 10, [int]$Containers = 0, [int]$Blocks = 0)
            $failed = foreach ($f in $FailedByFile.Keys) { 1..$FailedByFile[$f] | ForEach-Object { [pscustomobject]@{ ScriptBlock = [pscustomobject]@{ File = "C:/x/$f" } } } }
            [pscustomobject]@{
                Result = $Result; PassedCount = $Passed; FailedCount = @($failed).Count; FailedContainersCount = $Containers
                FailedBlocksCount = $Blocks; TotalCount = $Passed + @($failed).Count; Containers = @(); Failed = @($failed)
            }
        }
    }

    It 'tolerates failures inside a file budget and fails beyond it' {
        $known = @{ 'a.Tests.ps1' = 2 }
        (Get-PesterRunVerdict -Result (Get-FakeRun @{ 'a.Tests.ps1' = 2 }) -KnownFailures $known).Passed | Should -BeTrue
        $over = Get-PesterRunVerdict -Result (Get-FakeRun @{ 'a.Tests.ps1' = 3 }) -KnownFailures $known
        $over.Passed | Should -BeFalse
        $over.Violations -join ' ' | Should -Match 'a.Tests.ps1 has 3 failed test\(s\), allowed 2'
    }

    It 'fails a new failing file that has no budget' {
        $v = Get-PesterRunVerdict -Result (Get-FakeRun @{ 'a.Tests.ps1' = 1; 'new.Tests.ps1' = 1 }) -KnownFailures @{ 'a.Tests.ps1' = 1 }
        $v.Passed | Should -BeFalse
        $v.Violations -join ' ' | Should -Match 'new.Tests.ps1'
    }

    It 'fails failed containers and blocks beyond their allowance' {
        (Get-PesterRunVerdict -Result (Get-FakeRun -Containers 1)).Passed | Should -BeFalse
        (Get-PesterRunVerdict -Result (Get-FakeRun -Containers 1) -MaxFailedContainers 1).Passed | Should -BeTrue
        (Get-PesterRunVerdict -Result (Get-FakeRun -Blocks 3) -MaxFailedBlocks 2).Passed | Should -BeFalse
    }

    It 'fails a Result of Failed that nothing else explains, and a null result' {
        (Get-PesterRunVerdict -Result (Get-FakeRun -Result 'Failed')).Passed | Should -BeFalse
        (Get-PesterRunVerdict -Result (Get-FakeRun -Result 'Passed')).Passed | Should -BeTrue
        (Get-PesterRunVerdict -Result $null).Passed | Should -BeFalse
    }
}

Describe 'Get-PesterFileAnalysis counts weak It blocks through the AST' -Tag 'Unit', 'Portable' {
    # Protects: the quality gate counting what is actually in the test, not what the text mentions. Detects: a
    # constant-input assertion, an It with no assertion and an It that only proves nothing throws being missed, and
    # the false positives that would make the baseline meaningless (a failure marker, a skip, an assertion helper).
    BeforeAll {
        $file = Join-Path $TestDrive 'quality/sample.Tests.ps1'
        Write-Utf8 $file @'
Describe 'sample' -Tag 'Portable' {
    BeforeAll {
        function Assert-Thing { param($x) $x | Should -Be 1 }
    }
    It 'tautology' { $true | Should -BeTrue }
    It 'tautology via Be' { $true | Should -Be $true }
    It 'constant number' { 1 | Should -Be 1 }
    It 'no assertion' { $x = 1; Write-Host $x }
    It 'comment only mentions Should' {
        # $x | Should -Be 1
        $y = 2
    }
    It 'not-throw only' { { Get-Date } | Should -Not -Throw }
    It 'real assertion' { (1 + 1) | Should -Be 2 }
    It 'deliberate failure marker is not a tautology' { $false | Should -BeTrue -Because 'unreachable' }
    It 'negated constant is not a tautology' { $true | Should -Not -BeTrue }
    It 'static skip is not no-assertion' -Skip { }
    It 'dynamic skip is not static' -Skip:($PSVersionTable.PSVersion.Major -lt 1) { 1 | Should -Be 1 }
    It 'uses an assertion helper' { Assert-Thing 1 }
    It 'throws as its assertion' { if ((1 + 1) -ne 2) { throw 'math' } }
}
'@
        $script:A = Get-PesterFileAnalysis -Path $file
        $script:By = @{}
        foreach ($i in $script:A.It) { $script:By[$i.Name] = $i }
    }

    It 'counts a constant piped into Should as a tautology' {
        $script:By['tautology'].Tautology | Should -BeTrue
        $script:By['tautology via Be'].Tautology | Should -BeTrue
        $script:By['constant number'].Tautology | Should -BeTrue
    }

    It 'does not count a deliberate failure marker, a negated constant or a real assertion' {
        $script:By['deliberate failure marker is not a tautology'].Tautology | Should -BeFalse
        $script:By['negated constant is not a tautology'].Tautology | Should -BeFalse
        $script:By['real assertion'].Tautology | Should -BeFalse
    }

    It 'counts an It with no assertion, including one whose only Should is in a comment' {
        $script:By['no assertion'].NoAssertion | Should -BeTrue
        $script:By['comment only mentions Should'].NoAssertion | Should -BeTrue
    }

    It 'does not count a static skip, a helper assertion, a throw or a real assertion as no-assertion' {
        $script:By['static skip is not no-assertion'].StaticSkip | Should -BeTrue
        $script:By['static skip is not no-assertion'].NoAssertion | Should -BeFalse
        $script:By['dynamic skip is not static'].StaticSkip | Should -BeFalse
        $script:By['uses an assertion helper'].NoAssertion | Should -BeFalse
        $script:By['throws as its assertion'].NoAssertion | Should -BeFalse
        $script:By['real assertion'].NoAssertion | Should -BeFalse
    }

    It 'counts an It whose only assertion is Should -Not -Throw' {
        $script:By['not-throw only'].NotThrowOnly | Should -BeTrue
        $script:By['real assertion'].NotThrowOnly | Should -BeFalse
    }

    It 'reads the Portable tag and reports no parse errors for a valid file' {
        $script:A.HasPortableTag | Should -BeTrue
        $script:A.ParseErrors | Should -Be 0
    }

    It 'reports a parse error for a syntax-error file' {
        $bad = Join-Path $TestDrive 'quality/bad.Tests.ps1'
        Write-Utf8 $bad "Describe 'x' { It 'a' {`n"
        (Get-PesterFileAnalysis -Path $bad).ParseErrors | Should -BeGreaterThan 0
    }
}

Describe 'The coverage gate rejects a glob that analyses too few files' -Tag 'Unit', 'Portable' {
    # Protects: coverage meaning something. Detects: the shape of the old setting, './Modules/**/*.ps1', which Pester
    # treats like './Modules/*/*.ps1' and so skips every file nested more deeply (17 of 241 module files). Positive
    # control: Path 'Modules' with RecursePaths analyses all of them and the ratio gate does not fire.
    BeforeAll {
        $root = Join-Path $TestDrive 'covrepo'
        Write-Utf8 (Join-Path $root 'Modules/Alpha/Top.ps1') "function Get-Top { 'top' }`n"
        Write-Utf8 (Join-Path $root 'Modules/Alpha/Public/Mid.ps1') "function Get-Mid { 'mid' }`n"
        Write-Utf8 (Join-Path $root 'Modules/Beta/Private/Deep/Low.ps1') "function Get-Low { 'low' }`n"
        Write-Utf8 (Join-Path $root 'Modules/Beta/Beta.ps1') "function Get-Beta { 'beta' }`n"
        Write-Utf8 (Join-Path $root 'Tests/Unit/Cov.Tests.ps1') @'
BeforeAll {
    Get-ChildItem (Join-Path $PSScriptRoot '../../Modules') -Recurse -Filter *.ps1 | ForEach-Object { . $_.FullName }
}
Describe 'cov' { It 'calls them' { Get-Top; Get-Mid; Get-Low; Get-Beta | Should -Be 'beta' } }
'@
        $driver = Join-Path $TestDrive 'cov-driver.ps1'
        Write-Utf8 $driver @'
param([string]$Root, [string]$Tool, [string]$OutPath)
. $Tool
Import-Module Pester -MinimumVersion 5.0.0
$files = Get-AssessmentFileList -Root $Root
$moduleFiles = Get-ModuleSourceFile -Files $files
$result = [ordered]@{}
foreach ($mode in 'glob', 'recursive') {
    $c = New-PesterConfiguration
    $c.Run.Path = Join-Path $Root 'Tests/Unit'
    $c.Run.PassThru = $true
    $c.Output.Verbosity = 'None'
    $c.CodeCoverage.Enabled = $true
    $c.CodeCoverage.OutputPath = Join-Path $Root "cov-$mode.xml"
    $c.CodeCoverage.ExcludeTests = $true
    if ($mode -eq 'glob') {
        $c.CodeCoverage.Path = @((Join-Path $Root 'Modules/**/*.ps1'))
    } else {
        $c.CodeCoverage.Path = @((Join-Path $Root 'Modules'))
        $c.CodeCoverage.RecursePaths = $true
    }
    $r = Invoke-Pester -Configuration $c
    $result[$mode] = Get-CoverageSummary -CodeCoverage $r.CodeCoverage -Root $Root -ModuleFiles $moduleFiles
}
[IO.File]::WriteAllText($OutPath, ($result | ConvertTo-Json -Depth 8))
'@
        pwsh -NoProfile -File $driver -Root $root -Tool $script:Tool -OutPath (Join-Path $TestDrive 'cov.json') *> $null
        $script:Cov = Get-Content (Join-Path $TestDrive 'cov.json') -Raw | ConvertFrom-Json
        $script:Baseline = [pscustomobject]@{
            target_coverage_pct = 85; files_analyzed_ratio_min = 0.95
            quality = [pscustomobject]@{}; orphans_allowed = @()
            tiers = [pscustomobject]@{ Portable = [pscustomobject]@{ coverage_min_pct = 0; per_module_min_pct = [pscustomobject]@{}; known_failures = [pscustomobject]@{ containers = @(); tests_by_file = [pscustomobject]@{}; blocks_by_file = [pscustomobject]@{} } } }
        }
        $script:Static = [ordered]@{ quality = [ordered]@{ no_assertion_it = 0; not_throw_only_it = 0; tautology_it = 0; parse_error_files = 0 }; reach = [ordered]@{ orphans = @() } }
        $script:PassedRun = [pscustomobject]@{ Result = 'Passed'; PassedCount = 4; FailedCount = 0; FailedContainersCount = 0; FailedBlocksCount = 0; TotalCount = 4; Containers = @(); Failed = @() }
    }

    It 'the ** glob analyses fewer files than exist' {
        $script:Cov.glob.module_files_total | Should -Be 4
        $script:Cov.glob.files_analyzed | Should -BeLessThan $script:Cov.glob.module_files_total
    }

    It 'the files-analysed ratio gate fires on the glob and not on the recursive path' {
        $a = [ordered]@{ quality = $script:Static.quality; reach = $script:Static.reach; coverage = [ordered]@{ pester = $script:Cov.glob } }
        $codes = @(Test-AssessmentGate -Assessment $a -Baseline $script:Baseline -Tier Portable -PesterResult $script:PassedRun -Ran).code
        $codes | Should -Contain 'coverage_files_ratio'

        $b = [ordered]@{ quality = $script:Static.quality; reach = $script:Static.reach; coverage = [ordered]@{ pester = $script:Cov.recursive } }
        $script:Cov.recursive.files_analyzed | Should -Be 4
        @(Test-AssessmentGate -Assessment $b -Baseline $script:Baseline -Tier Portable -PesterResult $script:PassedRun -Ran).code |
            Should -Not -Contain 'coverage_files_ratio'
    }

    It 'the real tool config no longer uses a ** glob' {
        $cfg = Import-PowerShellDataFile (Join-Path $script:RepoRoot 'Tests/PesterConfiguration.psd1')
        $cfg.CodeCoverage.Path | Should -Not -Match '\*\*'
        $cfg.CodeCoverage.RecursePaths | Should -BeTrue
    }
}

Describe 'Test-AssessmentGate and the baseline ratchet' -Tag 'Unit', 'Portable' {
    # Protects: the gate failing closed and the baseline only ever tightening. Detects: a missing baseline passing,
    # a regression slipping under a count, coverage below the minimum, and -UpdateBaseline loosening a threshold
    # or allowlisting a new orphan.
    BeforeAll {
        function script:Get-FakeAssessment {
            param([int]$Tautology = 0, [string[]]$Orphans = @(), [double]$Pct = 50, [double]$Ratio = 1.0, [object[]]$Modules = @())
            [ordered]@{
                quality  = [ordered]@{ no_assertion_it = 0; not_throw_only_it = 0; tautology_it = $Tautology; parse_error_files = 0 }
                reach    = [ordered]@{ orphans = $Orphans }
                coverage = [ordered]@{ pester = [ordered]@{ pct = $Pct; files_analyzed_ratio = $Ratio; files_analyzed = 10; module_files_total = 10; by_module = $Modules } }
            }
        }
        $script:PassRun = [pscustomobject]@{ Result = 'Passed'; PassedCount = 4; FailedCount = 0; FailedContainersCount = 0; FailedBlocksCount = 0; TotalCount = 4; Containers = @(); Failed = @() }
        $script:GateBaseline = '{"schema_version":1,"target_coverage_pct":85,"files_analyzed_ratio_min":0.95,"quality":{"no_assertion_it_max":0,"not_throw_only_it_max":0,"tautology_it_max":2,"parse_error_files_max":0},"orphans_allowed":["a.Tests.ps1"],"tiers":{"Portable":{"coverage_min_pct":48,"per_module_min_pct":{"Alpha":40},"known_failures":{"containers":[],"tests_by_file":{},"blocks_by_file":{}}}}}' | ConvertFrom-Json
    }

    It 'fails closed when there is no baseline' {
        $codes = @(Test-AssessmentGate -Assessment (Get-FakeAssessment) -Baseline $null -Tier Portable -PesterResult $script:PassRun -Ran).code
        $codes | Should -Be @('baseline_missing')
    }

    It 'passes at the baseline and fails one count above it' {
        @(Test-AssessmentGate -Assessment (Get-FakeAssessment -Tautology 2 -Orphans 'a.Tests.ps1') -Baseline $script:GateBaseline -Tier Portable -PesterResult $script:PassRun -Ran).Count | Should -Be 0
        @(Test-AssessmentGate -Assessment (Get-FakeAssessment -Tautology 3 -Orphans 'a.Tests.ps1') -Baseline $script:GateBaseline -Tier Portable -PesterResult $script:PassRun -Ran).code | Should -Contain 'quality_tautology_it'
    }

    It 'fails an orphan that is not allowlisted' {
        @(Test-AssessmentGate -Assessment (Get-FakeAssessment -Orphans 'a.Tests.ps1', 'b.Tests.ps1') -Baseline $script:GateBaseline -Tier Portable -PesterResult $script:PassRun -Ran).code | Should -Contain 'new_orphan'
    }

    It 'fails coverage below the minimum and a module more than the tolerance below its own minimum' {
        $mod = @([pscustomobject]@{ module = 'Alpha'; pct = 38.9 })
        $codes = @(Test-AssessmentGate -Assessment (Get-FakeAssessment -Pct 47.9 -Orphans 'a.Tests.ps1' -Modules $mod) -Baseline $script:GateBaseline -Tier Portable -PesterResult $script:PassRun -Ran).code
        $codes | Should -Contain 'coverage_below_min'
        $codes | Should -Contain 'module_coverage_below_min'
        @(Test-AssessmentGate -Assessment (Get-FakeAssessment -Pct 48 -Orphans 'a.Tests.ps1' -Modules @([pscustomobject]@{ module = 'Alpha'; pct = 39.5 })) -Baseline $script:GateBaseline -Tier Portable -PesterResult $script:PassRun -Ran).Count | Should -Be 0
    }

    It 'fails the Pester verdict inside the gate' {
        $bad = [pscustomobject]@{ Result = 'Failed'; PassedCount = 0; FailedCount = 0; FailedContainersCount = 1; FailedBlocksCount = 0; TotalCount = 0; Containers = @(); Failed = @() }
        @(Test-AssessmentGate -Assessment (Get-FakeAssessment -Orphans 'a.Tests.ps1') -Baseline $script:GateBaseline -Tier Portable -PesterResult $bad -Ran).code | Should -Contain 'pester_run'
    }

    It 'does not evaluate coverage when nothing ran' {
        @(Test-AssessmentGate -Assessment (Get-FakeAssessment -Pct 1 -Orphans 'a.Tests.ps1') -Baseline $script:GateBaseline -Tier Portable -PesterResult $null).Count | Should -Be 0
    }

    It 'keeps blocks and containers keyed by file when it seeds and ratchets' {
        $failedBlock = [pscustomobject]@{ Name = 'd'; BlockContainer = [pscustomobject]@{ Item = [IO.FileInfo]::new('C:/x/a.Tests.ps1') } }
        $run = [pscustomobject]@{
            Result = 'Failed'; PassedCount = 4; FailedCount = 0; FailedContainersCount = 0; FailedBlocksCount = 2
            TotalCount = 4; Containers = @(); Failed = @(); FailedBlocks = @($failedBlock, $failedBlock)
        }
        $seed = Update-AssessmentBaseline -Baseline $null -Assessment (Get-FakeAssessment) -Tier Portable -PesterResult $run -Ran
        $seed.tiers.Portable.known_failures.blocks_by_file['a.Tests.ps1'] | Should -Be 2
        $json = $seed | ConvertTo-Json -Depth 10 | ConvertFrom-Json
        # Within the budget it passes; one more block in the same file fails even though that file has no failed tests.
        $verdict = Get-PesterRunVerdict -Result $run -KnownFailedBlocks @{ 'a.Tests.ps1' = 2 }
        $verdict.Passed | Should -BeTrue
        $third = [pscustomobject]@{ Result = 'Failed'; PassedCount = 4; FailedCount = 0; FailedContainersCount = 0; FailedBlocksCount = 3; TotalCount = 4; Containers = @(); Failed = @(); FailedBlocks = @($failedBlock, $failedBlock, $failedBlock) }
        (Get-PesterRunVerdict -Result $third -KnownFailedBlocks @{ 'a.Tests.ps1' = 2 }).Passed | Should -BeFalse
        # A block in a different file is not covered by a.Tests.ps1's budget.
        $other = [pscustomobject]@{ Name = 'o'; BlockContainer = [pscustomobject]@{ Item = [IO.FileInfo]::new('C:/x/b.Tests.ps1') } }
        $moved = [pscustomobject]@{ Result = 'Failed'; PassedCount = 4; FailedCount = 0; FailedContainersCount = 0; FailedBlocksCount = 2; TotalCount = 4; Containers = @(); Failed = @(); FailedBlocks = @($failedBlock, $other) }
        (Get-PesterRunVerdict -Result $moved -KnownFailedBlocks @{ 'a.Tests.ps1' = 2 }).Passed | Should -BeFalse
    }

    It 'ratchets only in the stricter direction' {
        $improved = Get-FakeAssessment -Tautology 1 -Orphans @() -Pct 61.7 -Modules @([pscustomobject]@{ module = 'Alpha'; pct = 70.2 })
        $u = Update-AssessmentBaseline -Baseline $script:GateBaseline -Assessment $improved -Tier Portable -PesterResult $script:PassRun -Ran
        $u.tiers.Portable.coverage_min_pct | Should -Be 61
        $u.tiers.Portable.per_module_min_pct.Alpha | Should -Be 70
        $u.quality.tautology_it_max | Should -Be 1
        @($u.orphans_allowed).Count | Should -Be 0

        $worse = Get-FakeAssessment -Tautology 5 -Orphans 'a.Tests.ps1', 'brand-new.Tests.ps1' -Pct 10 -Modules @([pscustomobject]@{ module = 'Alpha'; pct = 5.0 })
        $w = Update-AssessmentBaseline -Baseline $script:GateBaseline -Assessment $worse -Tier Portable -PesterResult $script:PassRun -Ran
        $w.tiers.Portable.coverage_min_pct | Should -Be 48 -Because 'a worse run must not lower the minimum'
        $w.tiers.Portable.per_module_min_pct.Alpha | Should -Be 40
        $w.quality.tautology_it_max | Should -Be 2 -Because 'a worse run must not raise the allowed count'
        @($w.orphans_allowed) | Should -Be @('a.Tests.ps1') -Because 'a new orphan is never allowlisted by the ratchet'
    }
}

Describe 'Static assessment is idempotent and fail-capable' -Tag 'Unit', 'Portable' {
    # Protects: the assessment being a read-only measurement. Detects: it writing anywhere but -OutputPath (and the
    # baseline with -UpdateBaseline), two runs giving different JSON, a missing baseline passing, and a new orphan
    # or tautology passing. Runs the real script in a child pwsh against a throwaway git fixture, with -NoRun.
    BeforeAll {
        $script:Fx = Join-Path $TestDrive 'static-repo'
        Write-Utf8 (Join-Path $script:Fx 'Tests/Unit/A.Tests.ps1') "Describe 'a' -Tag 'Portable' { It 'ok' { (1 + 1) | Should -Be 2 } }`n"
        Write-Utf8 (Join-Path $script:Fx 'Tests/Boot/B.Tests.ps1') "Describe 'b' { It 'ok' { 'x' | Should -Not -BeNullOrEmpty } }`n"
        Write-Utf8 (Join-Path $script:Fx 'Tools/Odd/C.Tests.ps1') "Describe 'c' { It 'ok' { (2 + 2) | Should -Be 4 } }`n"
        Write-Utf8 (Join-Path $script:Fx 'Native/crate/Cargo.toml') "[package]`nname='crate'`n"
        Write-Utf8 (Join-Path $script:Fx 'Native/crate/src/lib.rs') "#[test]`nfn a() {}`n#[tokio::test]`nasync fn b() {}`n#[cfg(test)]`nmod t {}`n"
        Invoke-FixtureGit $script:Fx @('init', '-q') | Out-Null
        Invoke-FixtureGit $script:Fx @('add', '-A') | Out-Null
        $script:Out = Join-Path $TestDrive 'out'
        $script:Assess = {
            param([string[]]$Extra, [string]$Name = 'a')
            $o = Join-Path $script:Out "$Name.json"
            $text = pwsh -NoProfile -File $script:Tool -RepoRoot $script:Fx -NoRun -OutputPath $o @Extra 2>&1 | Out-String
            [pscustomobject]@{ Exit = $LASTEXITCODE; Text = $text; Json = $(if (Test-Path $o) { Get-Content $o -Raw | ConvertFrom-Json } else { $null }); Path = $o }
        }
    }

    It 'fails closed without a baseline' {
        $r = & $script:Assess @() 'nobaseline'
        $r.Exit | Should -Be 1
        $r.Json.verdict | Should -Be 'fail'
        $r.Json.violations.code | Should -Contain 'baseline_missing'
    }

    It 'counts the inventory, reach and orphans of the fixture' {
        $r = & $script:Assess @() 'inventory'
        $r.Json.inventory.pester.totals.files | Should -Be 3
        $r.Json.inventory.pester.totals.it_blocks | Should -Be 3
        ($r.Json.inventory.rust.crates | Where-Object crate -EQ 'Native/crate').tests | Should -Be 2
        $r.Json.reach.orphans | Should -Be @('Tools/Odd/C.Tests.ps1')
        ($r.Json.reach.files | Where-Object file -EQ 'Tests/Unit/A.Tests.ps1').runners | Should -Contain 'Invoke-PortableTests'
        ($r.Json.reach.files | Where-Object file -EQ 'Tests/Boot/B.Tests.ps1').runners | Should -Not -Contain 'Invoke-PortableTests'
    }

    It 'seeds a baseline, then passes, and two runs give identical JSON without touching tracked files' {
        $seed = & $script:Assess @('-UpdateBaseline') 'seed'
        $seed.Exit | Should -Be 0 -Because $seed.Text
        Test-Path (Join-Path $script:Fx 'Config/test-baseline.json') | Should -BeTrue
        Invoke-FixtureGit $script:Fx @('add', '-A') | Out-Null

        $before = (Invoke-FixtureGit $script:Fx @('status', '--porcelain=v1', '-uall')) -join "`n"
        $one = & $script:Assess @() 'run1'
        $two = & $script:Assess @() 'run2'
        $after = (Invoke-FixtureGit $script:Fx @('status', '--porcelain=v1', '-uall')) -join "`n"

        $one.Exit | Should -Be 0 -Because $one.Text
        $two.Exit | Should -Be 0
        $after | Should -Be $before -Because 'the assessment must not write inside the repository'
        $strip = { param($p) (Get-Content $p -Raw) -replace '"generated_at":\s*"[^"]*"', '"generated_at": ""' }
        (& $strip $two.Path) | Should -Be (& $strip $one.Path)
    }

    It 'fails a new orphan and a new tautology, and passes again when they are removed' {
        $orphan = Join-Path $script:Fx 'Tools/Odd/New.Tests.ps1'
        Write-Utf8 $orphan "Describe 'n' { It 'ok' { (3 + 3) | Should -Be 6 } }`n"
        $r = & $script:Assess @() 'neworphan'
        $r.Exit | Should -Be 1
        $r.Json.violations.code | Should -Contain 'new_orphan'
        Remove-Item $orphan

        $taut = Join-Path $script:Fx 'Tests/Unit/Taut.Tests.ps1'
        Write-Utf8 $taut "Describe 't' -Tag 'Portable' { It 'always' { `$true | Should -BeTrue } }`n"
        $t = & $script:Assess @() 'newtaut'
        $t.Exit | Should -Be 1
        $t.Json.violations.code | Should -Contain 'quality_tautology_it'
        Remove-Item $taut

        (& $script:Assess @() 'clean').Exit | Should -Be 0
    }

    It 'ratcheting never allowlists a new orphan' {
        $orphan = Join-Path $script:Fx 'Tools/Odd/Another.Tests.ps1'
        Write-Utf8 $orphan "Describe 'n' { It 'ok' { (3 + 3) | Should -Be 6 } }`n"
        $r = & $script:Assess @('-UpdateBaseline') 'ratchet'
        $r.Exit | Should -Be 1 -Because 'the new orphan is still a violation after the ratchet'
        (Get-Content (Join-Path $script:Fx 'Config/test-baseline.json') -Raw | ConvertFrom-Json).orphans_allowed | Should -Not -Contain 'Tools/Odd/Another.Tests.ps1'
        Remove-Item $orphan
    }
}

Describe 'Runners fail on the planted failures' -Tag 'Unit', 'Portable' {
    # Protects: Invoke-PortableTests, Invoke-AllTests, test.ps1 and .pester.ps1 exiting 0 on a broken suite. Detects:
    # a syntax-error test file, an empty tag filter and an empty suite directory each leaving exit code 0. Each runner
    # is copied into a throwaway repo with a one-file suite, so the runs are seconds long. Positive control: the same
    # runner exits 0 on the clean suite.
    BeforeAll {
        $script:Mk = {
            param([string]$Name, [string[]]$Files, [string[]]$Runners)
            $repo = Join-Path $TestDrive "runner-$Name"
            foreach ($r in $Runners) { New-Item -ItemType Directory -Path (Join-Path $repo 'Tests') -Force | Out-Null; Copy-Item (Join-Path $script:RepoRoot "Tests/$r") (Join-Path $repo "Tests/$r") }
            New-Item -ItemType Directory -Path (Join-Path $repo 'Tests/Helpers') -Force | Out-Null
            Copy-Item $script:Helper (Join-Path $repo 'Tests/Helpers/Get-PesterRunVerdict.ps1')
            Copy-Item (Join-Path $script:RepoRoot 'Tests/PesterConfiguration.psd1') (Join-Path $repo 'Tests/PesterConfiguration.psd1')
            New-Item -ItemType Directory -Path (Join-Path $repo 'Modules') -Force | Out-Null
            foreach ($f in $Files) {
                $parts = $f -split '::'
                Write-Utf8 (Join-Path $repo $parts[0]) $parts[1]
            }
            $repo
        }
        $script:Good = "Tests/Unit/Good.Tests.ps1::Describe 'g' -Tag 'Portable' { It 'a' { 1 | Should -Be 1 } }`n"
        $script:Broken = "Tests/Unit/Broken.Tests.ps1::Describe 'b' -Tag 'Portable' { It 'a' { 1 | Should -Be 1 }`n"
        $script:Exec = { param($repo, $runner, $more = @()) pwsh -NoProfile -File (Join-Path $repo "Tests/$runner") @more 2>&1 | Out-Null; $LASTEXITCODE }
    }

    It 'Invoke-PortableTests: clean passes, syntax error fails, empty tag filter fails' {
        $clean = & $script:Mk 'pt-clean' @($script:Good) @('Invoke-PortableTests.ps1')
        & $script:Exec $clean 'Invoke-PortableTests.ps1' @('-Section', 'Pester', '-ResultsDir', (Join-Path $clean 'r')) | Should -Be 0

        $broken = & $script:Mk 'pt-broken' @($script:Good, $script:Broken) @('Invoke-PortableTests.ps1')
        & $script:Exec $broken 'Invoke-PortableTests.ps1' @('-Section', 'Pester', '-ResultsDir', (Join-Path $broken 'r')) | Should -Be 1

        $untagged = & $script:Mk 'pt-untagged' @("Tests/Unit/Untagged.Tests.ps1::Describe 'u' { It 'a' { 1 | Should -Be 1 } }`n") @('Invoke-PortableTests.ps1')
        & $script:Exec $untagged 'Invoke-PortableTests.ps1' @('-Section', 'Pester', '-ResultsDir', (Join-Path $untagged 'r')) | Should -Be 1
    }

    It 'Invoke-PortableTests: a section that threw is FAIL, not an uncounted ErrorRecord' -Skip:(-not (Get-Module -ListAvailable -Name PSScriptAnalyzer)) {
        # The Lint section analyses <repo>/Modules; with that directory missing Invoke-ScriptAnalyzer throws. The old
        # Invoke-Timed stored the thrown ErrorRecord, whose .status is null, so the section was neither PASS nor FAIL.
        $repo = & $script:Mk 'pt-throw' @($script:Good) @('Invoke-PortableTests.ps1')
        Remove-Item (Join-Path $repo 'Modules') -Recurse -Force
        & $script:Exec $repo 'Invoke-PortableTests.ps1' @('-Section', 'Lint', '-ResultsDir', (Join-Path $repo 'r')) | Should -Be 1
        $summary = Get-ChildItem (Join-Path $repo 'r') -Filter 'portable-*.json' | Select-Object -First 1 | Get-Content -Raw | ConvertFrom-Json
        $summary.sections.Lint.status | Should -Be 'FAIL'
    }

    It 'Invoke-PortableTests: a skipped section fails the run unless -AllowSkip' {
        # Nothing in the fixture has a Cargo.toml, so `cargo` is on PATH or not; force the SKIP with an empty PATH entry
        # check instead: the CSharp section skips when there is no csproj, which a fixture never has.
        $repo = & $script:Mk 'pt-skip' @($script:Good) @('Invoke-PortableTests.ps1')
        & $script:Exec $repo 'Invoke-PortableTests.ps1' @('-Section', 'CSharp', '-ResultsDir', (Join-Path $repo 'r1')) | Should -Be 1
        & $script:Exec $repo 'Invoke-PortableTests.ps1' @('-Section', 'CSharp', '-ResultsDir', (Join-Path $repo 'r2'), '-AllowSkip') | Should -Be 0
    }

    It 'Invoke-AllTests: clean passes, syntax error fails, no files fails' {
        $clean = & $script:Mk 'at-clean' @($script:Good) @('Invoke-AllTests.ps1')
        & $script:Exec $clean 'Invoke-AllTests.ps1' @('-Suite', 'Unit') | Should -Be 0
        $broken = & $script:Mk 'at-broken' @($script:Good, $script:Broken) @('Invoke-AllTests.ps1')
        & $script:Exec $broken 'Invoke-AllTests.ps1' @('-Suite', 'Unit') | Should -Be 1
        $none = & $script:Mk 'at-none' @($script:Good) @('Invoke-AllTests.ps1')
        & $script:Exec $none 'Invoke-AllTests.ps1' @('-Suite', 'E2E') | Should -Be 1
    }

    It 'test.ps1: clean passes, syntax error fails' {
        $clean = & $script:Mk 'tp-clean' @($script:Good) @('test.ps1')
        & $script:Exec $clean 'test.ps1' @('-UnitOnly') | Should -Be 0
        $broken = & $script:Mk 'tp-broken' @($script:Good, $script:Broken) @('test.ps1')
        & $script:Exec $broken 'test.ps1' @('-UnitOnly') | Should -Be 1
    }

    It '.pester.ps1: clean passes, syntax error fails, in non-CI mode too' {
        $clean = & $script:Mk 'pp-clean' @($script:Good) @('.pester.ps1')
        & $script:Exec $clean '.pester.ps1' @('-Type', 'Unit') | Should -Be 0
        $broken = & $script:Mk 'pp-broken' @($script:Good, $script:Broken) @('.pester.ps1')
        & $script:Exec $broken '.pester.ps1' @('-Type', 'Unit') | Should -Be 1
    }
}

Describe 'Invoke-QualityGate measures or fails, and never moves the live checkout' -Tag 'Unit', 'Portable' {
    # Protects: the A/B quality gate reporting numbers it did not measure. Detects: Rust test counts read from the
    # last result line only, a compile failure read as 0/0 "unchanged", clippy counted at the wrong level, a dead
    # `return if`, and the base ref being checked out into the live tree. A function named `cargo` stands in for the
    # toolchain, because PowerShell resolves functions before applications.
    BeforeAll {
        $script:CargoRoot = Join-Path $TestDrive 'qg/Native/pcai_core'
        New-Item -ItemType Directory -Path $script:CargoRoot -Force | Out-Null
    }
    AfterEach { Remove-Item -Path Function:\cargo -ErrorAction SilentlyContinue }

    It 'sums every test result line, not only the last' {
        function global:cargo { $global:LASTEXITCODE = 0; 'test result: ok. 3 passed; 0 failed; 1 ignored'; 'test result: ok. 4 passed; 1 failed; 0 ignored' }
        $r = Measure-RustTests -Root (Join-Path $TestDrive 'qg')
        $r.Passed | Should -Be 7
        $r.Failed | Should -Be 1
    }

    It 'treats a compile failure as a measurement error, not 0/0' {
        function global:cargo { $global:LASTEXITCODE = 101; 'error[E0432]: unresolved import' }
        { Measure-RustTests -Root (Join-Path $TestDrive 'qg') } | Should -Throw '*build failed*'
    }

    It 'counts clippy lints (capped to warnings) once each, and treats a denied-lint-shaped error or a silent failure as a build failure' {
        function global:cargo {
            $global:LASTEXITCODE = 0
            '{"reason":"compiler-message","message":{"level":"warning","code":{"code":"clippy::needless_return"},"message":"x"}}'
            '{"reason":"compiler-message","message":{"level":"warning","code":{"code":"unused_variables"},"message":"y"}}'
            '{"reason":"compiler-message","message":{"level":"warning","code":{"code":"unused_variables"},"message":"y"}}'
            '{"reason":"build-finished","success":true}'
        }
        Measure-ClippyFindings -Root (Join-Path $TestDrive 'qg') | Should -Be 2
        function global:cargo {
            $global:LASTEXITCODE = 101
            '{"reason":"compiler-message","message":{"level":"error","code":{"code":"clippy::needless_return"},"message":"x"}}'
            '{"reason":"build-finished","success":false}'
        }
        { Measure-ClippyFindings -Root (Join-Path $TestDrive 'qg') } | Should -Throw '*build failed*'
        function global:cargo { $global:LASTEXITCODE = 101; 'error: could not find Cargo.toml' }
        { Measure-ClippyFindings -Root (Join-Path $TestDrive 'qg') } | Should -Throw '*build failed*'
    }

    It 'Get-DeltaStatus classifies both directions (the old `return if` threw)' {
        Get-DeltaStatus -Metric 'ClippyWarnings' -Delta -1 | Should -Be 'improved'
        Get-DeltaStatus -Metric 'ClippyWarnings' -Delta 2 | Should -Be 'regressed'
        Get-DeltaStatus -Metric 'RustTestsPassed' -Delta 3 | Should -Be 'improved'
        Get-DeltaStatus -Metric 'RustTestsPassed' -Delta -3 | Should -Be 'regressed'
        Get-DeltaStatus -Metric 'PesterFailed' -Delta 0 | Should -Be 'unchanged'
    }

    It 'checks the base ref out into a temporary worktree and leaves branch, HEAD, worktree list and dirty tree unchanged' {
        $repo = Join-Path $TestDrive 'qg-repo'
        Write-Utf8 (Join-Path $repo 'file.txt') "base`n"
        Invoke-FixtureGit $repo @('init', '-q', '-b', 'main') | Out-Null
        Invoke-FixtureGit $repo @('add', '-A') | Out-Null
        Invoke-FixtureGit $repo @('commit', '-q', '-m', 'base') | Out-Null
        $baseSha = (Invoke-FixtureGit $repo @('rev-parse', 'HEAD')) -join ''
        Write-Utf8 (Join-Path $repo 'file.txt') "head dirty`n"

        $snapshot = {
            [pscustomobject]@{
                Branch = (Invoke-FixtureGit $repo @('branch', '--show-current')) -join ''
                Head = (Invoke-FixtureGit $repo @('rev-parse', 'HEAD')) -join ''
                Worktrees = (Invoke-FixtureGit $repo @('worktree', 'list', '--porcelain')) -join "`n"
                Status = (Invoke-FixtureGit $repo @('status', '--porcelain')) -join "`n"
                Content = Get-Content (Join-Path $repo 'file.txt') -Raw
            }
        }
        $before = & $snapshot
        $seen = @{}
        $parent = Join-Path $TestDrive 'qg-parent'
        New-Item -ItemType Directory -Path $parent | Out-Null
        $result = Invoke-WithBaseRefWorktree -RepoRoot $repo -Ref $baseSha -ParentDir $parent -Script {
            param($tree)
            $seen.Tree = $tree
            $seen.Content = Get-Content (Join-Path $tree 'file.txt') -Raw
            'measured'
        }
        $result | Should -Be 'measured'
        $seen.Content.Trim() | Should -Be 'base' -Because 'the base side must come from the ref, not the dirty working tree'
        Test-Path $seen.Tree | Should -BeFalse
        $after = & $snapshot
        $after.Branch | Should -Be $before.Branch
        $after.Head | Should -Be $before.Head
        $after.Worktrees | Should -Be $before.Worktrees
        $after.Status | Should -Be $before.Status
        $after.Content.Trim() | Should -Be 'head dirty'
    }

    It 'removes the temporary worktree even when the measurement throws' {
        $repo = Join-Path $TestDrive 'qg-repo'
        $parent = Join-Path $TestDrive 'qg-parent2'
        New-Item -ItemType Directory -Path $parent | Out-Null
        $seen = @{}
        { Invoke-WithBaseRefWorktree -RepoRoot $repo -Ref 'HEAD' -ParentDir $parent -Script { param($tree) $seen.Tree = $tree; throw 'boom' } } | Should -Throw 'boom'
        Test-Path $seen.Tree | Should -BeFalse
        ((Invoke-FixtureGit $repo @('worktree', 'list', '--porcelain')) -join "`n") | Should -Not -Match 'pcai-qg-base'
        { Invoke-WithBaseRefWorktree -RepoRoot $repo -Ref 'no-such-ref' -ParentDir $parent -Script { 'x' } } | Should -Throw '*git worktree add failed*'
    }

    # Protects: the shared cargo cache. This repo's target/ directories can be junctions into T:\RustCache, and a
    # recursive delete that walks through one destroys the cache for every worktree. Detects: the temporary
    # worktree removal following a link out of the tree (the sentinel file in the link target must survive).
    It 'removes a junction or symlink inside the temporary worktree without touching its target' {
        $repo = Join-Path $TestDrive 'qg-repo'
        $parent = Join-Path $TestDrive 'qg-parent3'
        New-Item -ItemType Directory -Path $parent | Out-Null
        $shared = Join-Path $TestDrive 'shared-cache'
        New-Item -ItemType Directory -Path $shared | Out-Null
        Write-Utf8 (Join-Path $shared 'sentinel.txt') 'keep me'
        $type = if ($IsWindows) { 'Junction' } else { 'SymbolicLink' }
        Invoke-WithBaseRefWorktree -RepoRoot $repo -Ref 'HEAD' -ParentDir $parent -Script {
            param($tree)
            New-Item -ItemType $type -Path (Join-Path $tree 'target-link') -Target $shared | Out-Null
        }
        Test-Path (Join-Path $shared 'sentinel.txt') | Should -BeTrue -Because 'removing the worktree must not follow the link into the shared directory'
        Get-ChildItem $parent | Should -BeNullOrEmpty
    }
}

Describe 'Invoke-AssessmentPester measures under the default error preference' -Tag 'Unit', 'Portable' {
    # Protects: the assessment reporting what a developer's own Invoke-Pester sees. Detects: the tool's own
    # `$ErrorActionPreference = 'Stop'` leaking into the tests, where a tolerated non-terminating error becomes a
    # failure (measured: PcaiNativeInitializationContracts.Tests.ps1 gave 1 failed test bare and 3 plus a failed block
    # under Stop). The fixture test writes a non-terminating error and still asserts; it passes only under Continue.
    It 'does not let the caller Stop preference fail a test that tolerates a non-terminating error' {
        $root = Join-Path $TestDrive 'eap-repo'
        Write-Utf8 (Join-Path $root 'Modules/M/F.ps1') "function Get-F { 1 }`n"
        Write-Utf8 (Join-Path $root 'Tests/Unit/E.Tests.ps1') "Describe 'e' -Tag 'Portable' { It 'tolerates a non-terminating error' { Write-Error 'tolerated' -ErrorAction Continue; (1 + 1) | Should -Be 2 } }`n"
        $driver = Join-Path $TestDrive 'eap-driver.ps1'
        Write-Utf8 $driver @'
param([string]$Root, [string]$Tool, [string]$OutPath)
. $Tool
$ErrorActionPreference = 'Stop'
$scratch = Join-Path ([IO.Path]::GetTempPath()) ('eap-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $scratch | Out-Null
$r = Invoke-AssessmentPester -Root $Root -Tier Portable -ScratchDir $scratch
[IO.File]::WriteAllText($OutPath, ([ordered]@{ Passed = $r.PassedCount; Failed = $r.FailedCount } | ConvertTo-Json))
Remove-Item $scratch -Recurse -Force
'@
        $out = Join-Path $TestDrive 'eap.json'
        pwsh -NoProfile -File $driver -Root $root -Tool $script:Tool -OutPath $out *> $null
        $r = Get-Content $out -Raw | ConvertFrom-Json
        $r.Failed | Should -Be 0
        $r.Passed | Should -Be 1
    }
}

Describe 'Invoke-QualityGate runs end to end in a fixture repo' -Tag 'Unit', 'Portable' -Skip:(-not (Get-Module -ListAvailable -Name PSScriptAnalyzer)) {
    # Protects: the gate body, not just its helpers. Detects: head and base being measured from the same tree, a
    # measurement that cannot run reading as "unchanged", and the base worktree or a branch switch surviving the run.
    # A fake cargo on PATH stands in for the toolchain; Pester and PSScriptAnalyzer are the real ones. The head commit
    # adds one Portable test, so a gate that measured the same tree twice would report no change.
    BeforeAll {
        $script:Fx = Join-Path $TestDrive 'qg-e2e'
        Write-Utf8 (Join-Path $script:Fx 'Native/pcai_core/placeholder.txt') "x`n"
        Write-Utf8 (Join-Path $script:Fx 'Modules/Mod/F.ps1') "function Get-F { 1 }`n"
        Write-Utf8 (Join-Path $script:Fx 'Tests/Unit/A.Tests.ps1') "Describe 'a' -Tag 'Portable' { It 'a' { (1 + 1) | Should -Be 2 } }`n"
        New-Item -ItemType Directory -Path (Join-Path $script:Fx 'Tests/Helpers') -Force | Out-Null
        Copy-Item $script:QualityGate (Join-Path $script:Fx 'Tests/Invoke-QualityGate.ps1')
        Copy-Item $script:Helper (Join-Path $script:Fx 'Tests/Helpers/Get-PesterRunVerdict.ps1')
        Invoke-FixtureGit $script:Fx @('init', '-q', '-b', 'main') | Out-Null
        Invoke-FixtureGit $script:Fx @('add', '-A') | Out-Null
        Invoke-FixtureGit $script:Fx @('commit', '-q', '-m', 'base') | Out-Null
        $script:BaseSha = (Invoke-FixtureGit $script:Fx @('rev-parse', 'HEAD')) -join ''
        Write-Utf8 (Join-Path $script:Fx 'Tests/Unit/B.Tests.ps1') "Describe 'b' -Tag 'Portable' { It 'b' { (2 + 2) | Should -Be 4 } }`n"
        Invoke-FixtureGit $script:Fx @('add', '-A') | Out-Null
        Invoke-FixtureGit $script:Fx @('commit', '-q', '-m', 'head') | Out-Null

        function script:Get-FakeCargoDir {
            param([string]$Name, [int]$ExitCode)
            $dir = Join-Path $TestDrive $Name
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
            if ($IsWindows) {
                $body = if ($ExitCode -eq 0) { "@echo off`r`nif ""%1""==""clippy"" (echo {""reason"":""build-finished"",""success"":true} & exit /b 0)`r`necho test result: ok. 2 passed; 0 failed; 0 ignored`r`nexit /b 0`r`n" } else { "@echo off`r`necho error: fake toolchain failure`r`nexit /b $ExitCode`r`n" }
                [IO.File]::WriteAllText((Join-Path $dir 'cargo.cmd'), $body)
            } else {
                $body = if ($ExitCode -eq 0) { "#!/bin/sh`n[ ""`$1"" = clippy ] && { echo '{""reason"":""build-finished"",""success"":true}'; exit 0; }`necho 'test result: ok. 2 passed; 0 failed; 0 ignored'`nexit 0`n" } else { "#!/bin/sh`necho 'error: fake toolchain failure'`nexit $ExitCode`n" }
                [IO.File]::WriteAllText((Join-Path $dir 'cargo'), $body)
                chmod +x (Join-Path $dir 'cargo')
            }
            $dir
        }
        $script:RunGate = {
            param([string]$CargoDir, [string]$Name, [string]$Ref = $script:BaseSha)
            $results = Join-Path $TestDrive "qg-results-$Name"
            $savedPath = $env:PATH
            try {
                $env:PATH = "$CargoDir$([IO.Path]::PathSeparator)$env:PATH"
                pwsh -NoProfile -File (Join-Path $script:Fx 'Tests/Invoke-QualityGate.ps1') -BaseRef $Ref -ResultsDir $results *> $null
                $exit = $LASTEXITCODE
            } finally { $env:PATH = $savedPath }
            $file = Get-ChildItem $results -Filter 'quality-gate-*.json' -ErrorAction SilentlyContinue | Select-Object -First 1
            [pscustomobject]@{ Exit = $exit; Json = $(if ($file) { Get-Content $file.FullName -Raw | ConvertFrom-Json } else { $null }) }
        }
    }

    It 'measures head and base from different trees, exits 0, and leaves branch and worktree list unchanged' {
        $listBefore = (Invoke-FixtureGit $script:Fx @('worktree', 'list', '--porcelain')) -join "`n"
        $branchBefore = (Invoke-FixtureGit $script:Fx @('branch', '--show-current')) -join ''
        $r = & $script:RunGate (Get-FakeCargoDir 'cargo-ok' 0) 'ok'
        $r.Exit | Should -Be 0
        $r.Json.measurement_errors | Should -BeNullOrEmpty
        $r.Json.deltas.PesterPassed.base | Should -Be 1
        $r.Json.deltas.PesterPassed.head | Should -Be 2 -Because 'the head commit adds one test; equal numbers would mean one tree was measured twice'
        $r.Json.deltas.RustTestsPassed.head | Should -Be 2
        $r.Json.overall | Should -Be 'IMPROVED'
        ((Invoke-FixtureGit $script:Fx @('worktree', 'list', '--porcelain')) -join "`n") | Should -Be $listBefore
        ((Invoke-FixtureGit $script:Fx @('branch', '--show-current')) -join '') | Should -Be $branchBefore
    }

    It 'exits 1 with a measurement error when the toolchain cannot be measured, never reporting a pass' {
        $r = & $script:RunGate (Get-FakeCargoDir 'cargo-broken' 101) 'broken'
        $r.Exit | Should -Be 1
        $r.Json.overall | Should -Be 'ERROR'
        @($r.Json.measurement_errors).Count | Should -BeGreaterThan 0
        ($r.Json.measurement_errors -join ' ') | Should -Match 'could not be measured'
    }

    It 'exits 1 when the base ref does not exist' {
        $r = & $script:RunGate (Get-FakeCargoDir 'cargo-ok2' 0) 'badref' 'no-such-ref'
        $r.Exit | Should -Be 1
        $r.Json.overall | Should -Be 'ERROR'
    }
}

Describe 'A script that auto-loads Pester with #Requires cannot exit non-zero' -Tag 'Unit', 'Portable' {
    # Protects: every runner script being able to fail. Detects: `#Requires -Modules Pester` (in any spelling) together
    # with a top-level `exit <non-zero or variable>`. Measured on pwsh 7.6.7 with Pester 5.7.1, under `pwsh -File`:
    #   `#Requires -Modules Pester`                      then `exit 3`  ->  exit code 0
    #   `#Requires -Version 7.0`                         then `exit 3`  ->  3
    #   `#Requires -Modules Microsoft.PowerShell.Utility` then `exit 3`  ->  3
    #   `Import-Module Pester`                           then `exit 3`  ->  3
    # so it is specific to auto-loading Pester through #Requires. Two runners (Tests/Invoke-AllTests.ps1 and
    # Tools/SystemScripts/C-Scripts/Startup/Tests/Run-Tests.ps1) printed "TEST RUN FAILED" and exited 0 because of it.
    # Fix: an explicit `Import-Module Pester -MinimumVersion 5.0.0` with its own error and `exit 1` when missing.
    # `exit` inside a scriptblock value (a Describe/It body, an Invoke-Command block) or a here-string is another
    # process or another scope and is not flagged. The runtime control below pins the observation itself: if it ever
    # fails because the directive no longer swallows the code, pwsh fixed the behaviour and this guard can be relaxed.
    BeforeAll {
        function script:Get-RequiresPesterExitHazard {
            param([string]$Path)
            $errors = $null
            $ast = [System.Management.Automation.Language.Parser]::ParseFile($Path, [ref]$null, [ref]$errors)
            if ($null -eq $ast -or $null -eq $ast.ScriptRequirements) { return @() }
            $names = @($ast.ScriptRequirements.RequiredModules | ForEach-Object { $_.Name })
            if ('Pester' -notin $names) { return @() }
            $exits = $ast.FindAll({ param($n) $n -is [System.Management.Automation.Language.ExitStatementAst] }, $true)
            foreach ($e in $exits) {
                # Skip an exit that lives in a scriptblock value; keep top-level code and function bodies (an exit in a
                # function still ends the script with that code).
                $inBlock = $false
                for ($p = $e.Parent; $p; $p = $p.Parent) {
                    if ($p -is [System.Management.Automation.Language.ScriptBlockExpressionAst]) { $inBlock = $true; break }
                }
                if ($inBlock) { continue }
                $code = $e.Pipeline
                if ($null -eq $code) { continue }
                $isZero = $code -is [System.Management.Automation.Language.PipelineAst] -and
                $code.PipelineElements.Count -eq 1 -and
                $code.PipelineElements[0] -is [System.Management.Automation.Language.CommandExpressionAst] -and
                $code.PipelineElements[0].Expression -is [System.Management.Automation.Language.ConstantExpressionAst] -and
                [int]$code.PipelineElements[0].Expression.Value -eq 0
                if (-not $isZero) { [pscustomobject]@{ Path = $Path; Line = $e.Extent.StartLineNumber; Exit = $e.Extent.Text } }
            }
        }
        $reqFx = Join-Path $TestDrive 'requires'
        Write-Utf8 (Join-Path $reqFx 'hazard.ps1') "#Requires -Modules Pester`nexit 3`n"
        Write-Utf8 (Join-Path $reqFx 'hazard-spec.ps1') "#Requires -Version 7.0`n#Requires -Modules @{ ModuleName = 'Pester'; ModuleVersion = '5.0.0' }`nparam()`n`$code = 2`nexit `$code`n"
        Write-Utf8 (Join-Path $reqFx 'hazard-function.ps1') "#Requires -Modules Pester`nfunction Stop-Run { exit 1 }`nStop-Run`n"
        Write-Utf8 (Join-Path $reqFx 'safe-import.ps1') "Import-Module Pester -MinimumVersion 5.0.0`nexit 3`n"
        Write-Utf8 (Join-Path $reqFx 'safe-other-module.ps1') "#Requires -Modules Microsoft.PowerShell.Utility`nexit 3`n"
        Write-Utf8 (Join-Path $reqFx 'safe-zero.ps1') "#Requires -Modules Pester`nexit 0`n"
        Write-Utf8 (Join-Path $reqFx 'safe-nested.Tests.ps1') "#Requires -Modules Pester`nDescribe 'd' { It 'i' { `$r = pwsh -NoProfile -Command 'exit 3'; exit `$LASTEXITCODE } }`n`$text = @'`nexit 3`n'@`n"
    }

    It 'flags the directive plus a non-zero, variable or function-level exit' {
        @(Get-RequiresPesterExitHazard (Join-Path $reqFx 'hazard.ps1')).Count | Should -Be 1
        @(Get-RequiresPesterExitHazard (Join-Path $reqFx 'hazard-spec.ps1')).Count | Should -Be 1
        @(Get-RequiresPesterExitHazard (Join-Path $reqFx 'hazard-function.ps1')).Count | Should -Be 1
    }

    It 'does not flag an explicit import, another required module, exit 0, or an exit inside a scriptblock or here-string' {
        foreach ($name in 'safe-import.ps1', 'safe-other-module.ps1', 'safe-zero.ps1', 'safe-nested.Tests.ps1') {
            @(Get-RequiresPesterExitHazard (Join-Path $reqFx $name)).Count | Should -Be 0 -Because $name
        }
    }

    It 'no tracked script combines #Requires -Modules Pester with a top-level non-zero exit' {
        $files = @(Get-AssessmentFileList -Root $script:RepoRoot | Where-Object { $_ -like '*.ps1' })
        $hits = @($files | ForEach-Object { Get-RequiresPesterExitHazard (Join-Path $script:RepoRoot $_) })
        $hits | ForEach-Object { "$($_.Path):$($_.Line) $($_.Exit)" } | Should -BeNullOrEmpty -Because 'such a script reports exit code 0 under pwsh -File; use Import-Module Pester instead'
    }

    It 'runtime control: an explicit import keeps the exit code, the directive swallows it (pwsh -File)' {
        pwsh -NoProfile -File (Join-Path $reqFx 'safe-import.ps1') *> $null
        $LASTEXITCODE | Should -Be 3
        pwsh -NoProfile -File (Join-Path $reqFx 'safe-other-module.ps1') *> $null
        $LASTEXITCODE | Should -Be 3
        pwsh -NoProfile -File (Join-Path $reqFx 'hazard.ps1') *> $null
        $LASTEXITCODE | Should -Be 0 -Because 'observed on pwsh 7.6.7 / Pester 5.7.1; if this fails the swallow is fixed and the static guard can be relaxed'
    }
}

Describe 'Run-Tests.ps1 (startup tests runner) exits non-zero when the suite fails' -Tag 'Unit', 'Portable' {
    # Protects: Tools/SystemScripts/C-Scripts/Startup/Tests/Run-Tests.ps1 being able to fail. Detects: its `#Requires
    # -Modules Pester` swallowing `exit 1` (see the Describe above), and its function named New-PesterConfiguration
    # calling itself instead of Pester's cmdlet, which recursed until the call-depth limit on every run. The script
    # is copied into a temporary directory because it writes a TestResults folder next to itself.
    BeforeAll {
        $script:RunTestsSource = Join-Path $script:RepoRoot 'Tools/SystemScripts/C-Scripts/Startup/Tests/Run-Tests.ps1'
        function script:Invoke-StartupRunner {
            param([string]$Name, [string]$TestText)
            $dir = Join-Path $TestDrive "startup-$Name"
            Write-Utf8 (Join-Path $dir 'Case.Tests.ps1') $TestText
            Copy-Item $script:RunTestsSource (Join-Path $dir 'Run-Tests.ps1')
            pwsh -NoProfile -File (Join-Path $dir 'Run-Tests.ps1') -TestPath (Join-Path $dir 'Case.Tests.ps1') *> $null
            $LASTEXITCODE
        }
    }

    It 'exits 0 for a passing suite' {
        Invoke-StartupRunner 'pass' "Describe 'p' { It 'a' { (1 + 1) | Should -Be 2 } }`n" | Should -Be 0
    }

    It 'exits 1 for a failing test' {
        Invoke-StartupRunner 'fail' "Describe 'f' { It 'a' { (1 + 1) | Should -Be 3 } }`n" | Should -Be 1
    }

    It 'exits 1 for a test file that does not parse (FailedCount stays 0)' {
        Invoke-StartupRunner 'broken' "Describe 'b' { It 'a' { 1 | Should -Be 1 }`n" | Should -Be 1
    }

    It 'exits 1 when nothing executed' {
        Invoke-StartupRunner 'empty' "Describe 'e' { }`n" | Should -Be 1
    }
}

Describe 'Scheduled-task health tests keep their child exit inside a here-string' -Tag 'Unit', 'Portable' {
    # Protects: the two Tests/Boot files that combine `#Requires -Modules Pester` with an `exit` line. Detects: that
    # exit moving out of the here-string that is written to a fixture script and run in a child process. There it is
    # the child's own exit (no #Requires in the fixture), so the swallow does not apply; the static guard above passes
    # for the same reason.
    It 'has no flagged top-level exit in either file' {
        foreach ($name in 'ScheduledTaskHealth.Tests.ps1', 'ScheduledTaskHealthLauncher.Tests.ps1') {
            @(Get-RequiresPesterExitHazard (Join-Path $script:RepoRoot "Tests/Boot/$name")).Count | Should -Be 0 -Because $name
        }
    }
}

Describe 'A failed AfterAll or BeforeAll is a failure on every verdict path' -Tag 'Unit', 'Portable' {
    # Protects: no runner, and not the quality gate, passing a run whose only fault is a throwing AfterAll or BeforeAll.
    # Detects: a real Pester file with one passing It and `AfterAll { throw }`, which gives Result=Failed, FailedCount=0,
    # FailedContainers=0, FailedBlocks=1. Invoke-QualityGate dropped FailedBlocks (it reported Failed = FailedCount +
    # FailedContainers = 0), so its ratchet could pass that run. These are real Pester files run by a real Pester,
    # not mocked result objects. A file-level BeforeAll/AfterAll throw is a failed container instead; both are covered.
    # Positive control: the same file without the throw passes on every path.
    BeforeAll {
        $script:Blocks = Join-Path $TestDrive 'blocks'
        $script:Good = "Describe 'g' -Tag 'Portable' { It 'ok' { (1 + 1) | Should -Be 2 } }`n"
        $script:AfterAllThrow = "Describe 'a' -Tag 'Portable' { It 'ok' { (1 + 1) | Should -Be 2 }; AfterAll { throw 'x' } }`n"
        $script:BeforeAllThrow = "Describe 'b' -Tag 'Portable' { BeforeAll { throw 'x' }; It 'ok' { (1 + 1) | Should -Be 2 } }`n"
        $script:FileBeforeThrow = "BeforeAll { throw 'x' }`nDescribe 'c' -Tag 'Portable' { It 'ok' { (1 + 1) | Should -Be 2 } }`n"
        $script:FileAfterThrow = "AfterAll { throw 'x' }`nDescribe 'd' -Tag 'Portable' { It 'ok' { (1 + 1) | Should -Be 2 } }`n"
        $script:FailPlusAfterAll = "Describe 'm' -Tag 'Portable' { It 'known failure' { (1 + 1) | Should -Be 3 }; It 'ok' { (1 + 1) | Should -Be 2 }; AfterAll { throw 'new block' } }`n"
        $script:FailOnly = "Describe 'm' -Tag 'Portable' { It 'known failure' { (1 + 1) | Should -Be 3 }; It 'ok' { (1 + 1) | Should -Be 2 } }`n"
        $cases = [ordered]@{
            good = $script:Good; afterall = $script:AfterAllThrow; beforeall = $script:BeforeAllThrow
            filebefore = $script:FileBeforeThrow; fileafter = $script:FileAfterThrow
            mixed = $script:FailPlusAfterAll; failonly = $script:FailOnly
        }
        $list = foreach ($k in $cases.Keys) {
            Write-Utf8 (Join-Path $script:Blocks "$k/$k.Tests.ps1") $cases[$k]
            $entry = @{ name = $k; path = (Join-Path $script:Blocks "$k/$k.Tests.ps1") }
            if ($k -in 'mixed', 'failonly') { $entry.known = @{ 'mixed.Tests.ps1' = 1; 'failonly.Tests.ps1' = 1 } }
            $entry
        }
        $outFile = Join-Path $TestDrive 'blocks-cases.json'
        pwsh -NoProfile -File $script:ChildRunner -CasesJson ($list | ConvertTo-Json -Depth 5 -Compress) -HelperPath $script:Helper -OutPath $outFile *> $null
        $script:Real = Get-Content $outFile -Raw | ConvertFrom-Json
    }

    It 'the real AfterAll fixture has the shape the defect needs: Failed, no failed test, no container, one block' {
        $r = $script:Real.afterall.raw
        $r.Result | Should -Be 'Failed'
        $r.FailedCount | Should -Be 0
        $r.FailedContainers | Should -Be 0
        $r.FailedBlocks | Should -Be 1
        $r.PassedCount | Should -Be 1
    }

    It 'the verdict fails a throwing AfterAll, a throwing BeforeAll, and file-level throws; it passes the clean file' {
        $script:Real.good.verdict.Passed | Should -BeTrue
        foreach ($k in 'afterall', 'beforeall', 'filebefore', 'fileafter') {
            $script:Real.$k.verdict.Passed | Should -BeFalse -Because $k
        }
        $script:Real.afterall.verdict.Violations -join ' ' | Should -Match 'block\(s\) failed outside any test.*afterall\.Tests\.ps1'
    }

    It 'a new failed block in a file that already has known test failures still fails' {
        $script:Real.failonly.verdict.Passed | Should -BeTrue -Because 'the known test failure is inside its budget'
        $script:Real.mixed.verdict.Passed | Should -BeFalse
        $script:Real.mixed.verdict.Violations -join ' ' | Should -Match 'block\(s\).*mixed\.Tests\.ps1'
    }

    It 'every runner exits 1 for the AfterAll and BeforeAll fixtures and 0 for the clean one' {
        $runners = @(
            @{ Id = 1; Script = 'Invoke-PortableTests.ps1'; Args = { param($r) @('-Section', 'Pester', '-ResultsDir', (Join-Path $r 'r')) }; Dir = 'Tests/Unit' }
            @{ Id = 2; Script = 'Invoke-AllTests.ps1'; Args = { param($r) @('-Suite', 'Unit') }; Dir = 'Tests/Unit' }
            @{ Id = 3; Script = 'test.ps1'; Args = { param($r) @('-UnitOnly') }; Dir = 'Tests/Unit' }
            @{ Id = 4; Script = '.pester.ps1'; Args = { param($r) @('-Type', 'Unit') }; Dir = 'Tests/Unit' }
        )
        $run = {
            param($runner, $name, $text)
            $repo = Join-Path $TestDrive "blk-$($runner.Id)-$name"
            New-Item -ItemType Directory -Path (Join-Path $repo 'Tests/Helpers') -Force | Out-Null
            Copy-Item (Join-Path $script:RepoRoot "Tests/$($runner.Script)") (Join-Path $repo "Tests/$($runner.Script)")
            Copy-Item $script:Helper (Join-Path $repo 'Tests/Helpers/Get-PesterRunVerdict.ps1')
            Copy-Item (Join-Path $script:RepoRoot 'Tests/PesterConfiguration.psd1') (Join-Path $repo 'Tests/PesterConfiguration.psd1')
            New-Item -ItemType Directory -Path (Join-Path $repo 'Modules') -Force | Out-Null
            Write-Utf8 (Join-Path $repo "$($runner.Dir)/$name.Tests.ps1") $text
            $arguments = @(& $runner.Args $repo)
            pwsh -NoProfile -File (Join-Path $repo "Tests/$($runner.Script)") @arguments *> $null
            $LASTEXITCODE
        }
        foreach ($runner in $runners) {
            (& $run $runner 'good' $script:Good) | Should -Be 0 -Because "$($runner.Script) on the clean fixture"
            (& $run $runner 'afterall' $script:AfterAllThrow) | Should -Be 1 -Because "$($runner.Script) on AfterAll { throw }"
            (& $run $runner 'beforeall' $script:BeforeAllThrow) | Should -Be 1 -Because "$($runner.Script) on BeforeAll { throw }"
        }
    }

    It 'the assessment tool fails a run whose only fault is a throwing AfterAll, even with a seeded baseline' {
        $repo = Join-Path $TestDrive 'blk-assess'
        Write-Utf8 (Join-Path $repo 'Modules/M/F.ps1') "function Get-F { 1 }`n"
        Write-Utf8 (Join-Path $repo 'Tests/Unit/Case.Tests.ps1') $script:Good
        $out = Join-Path $TestDrive 'blk-assess.json'
        pwsh -NoProfile -File $script:Tool -RepoRoot $repo -Tier Portable -UpdateBaseline -OutputPath $out *> $null
        $LASTEXITCODE | Should -Be 0 -Because 'the clean fixture seeds and passes'
        pwsh -NoProfile -File $script:Tool -RepoRoot $repo -Tier Portable -OutputPath $out *> $null
        $LASTEXITCODE | Should -Be 0
        Write-Utf8 (Join-Path $repo 'Tests/Unit/Case.Tests.ps1') $script:AfterAllThrow
        pwsh -NoProfile -File $script:Tool -RepoRoot $repo -Tier Portable -OutputPath $out *> $null
        $LASTEXITCODE | Should -Be 1
        (Get-Content $out -Raw | ConvertFrom-Json).violations.code | Should -Contain 'pester_run'
    }
}

Describe 'Invoke-QualityGate counts failed blocks and rejects unusable child results' -Tag 'Unit', 'Portable' -Skip:(-not (Get-Module -ListAvailable -Name PSScriptAnalyzer)) {
    # Protects: the A/B ratchet. Detects: FailedBlocks dropped from the Pester failure metric (the codex review of #384:
    # a real AfterAll throw gave Failed = 0 and the head-vs-base comparison passed), and a child process whose exit code
    # or result file is unusable being read as a measurement. The first test runs the whole gate against real Pester
    # files in a fixture repo; the rest replace `pwsh` with a function that writes a chosen result file and exit code.
    BeforeAll {
        $script:QFx = Join-Path $TestDrive 'qg-blocks'
        Write-Utf8 (Join-Path $script:QFx 'Native/pcai_core/placeholder.txt') "x`n"
        Write-Utf8 (Join-Path $script:QFx 'Modules/Mod/F.ps1') "function Get-F { 1 }`n"
        Write-Utf8 (Join-Path $script:QFx 'Tests/Unit/A.Tests.ps1') "Describe 'a' -Tag 'Portable' { It 'a' { (1 + 1) | Should -Be 2 } }`n"
        New-Item -ItemType Directory -Path (Join-Path $script:QFx 'Tests/Helpers') -Force | Out-Null
        Copy-Item $script:QualityGate (Join-Path $script:QFx 'Tests/Invoke-QualityGate.ps1')
        Copy-Item $script:Helper (Join-Path $script:QFx 'Tests/Helpers/Get-PesterRunVerdict.ps1')
        Invoke-FixtureGit $script:QFx @('init', '-q', '-b', 'main') | Out-Null
        Invoke-FixtureGit $script:QFx @('add', '-A') | Out-Null
        Invoke-FixtureGit $script:QFx @('commit', '-q', '-m', 'base') | Out-Null
        $script:QBase = (Invoke-FixtureGit $script:QFx @('rev-parse', 'HEAD')) -join ''
        Write-Utf8 (Join-Path $script:QFx 'Tests/Unit/B.Tests.ps1') "Describe 'b' -Tag 'Portable' { It 'b' { (2 + 2) | Should -Be 4 }; AfterAll { throw 'x' } }`n"
        Invoke-FixtureGit $script:QFx @('add', '-A') | Out-Null
        Invoke-FixtureGit $script:QFx @('commit', '-q', '-m', 'head') | Out-Null
        $cargoDir = Join-Path $TestDrive 'qg-blocks-cargo'
        New-Item -ItemType Directory -Path $cargoDir -Force | Out-Null
        if ($IsWindows) {
            [IO.File]::WriteAllText((Join-Path $cargoDir 'cargo.cmd'), "@echo off`r`nif ""%1""==""clippy"" (echo {""reason"":""build-finished"",""success"":true} & exit /b 0)`r`necho test result: ok. 2 passed; 0 failed; 0 ignored`r`nexit /b 0`r`n")
        } else {
            [IO.File]::WriteAllText((Join-Path $cargoDir 'cargo'), "#!/bin/sh`n[ ""`$1"" = clippy ] && { echo '{""reason"":""build-finished"",""success"":true}'; exit 0; }`necho 'test result: ok. 2 passed; 0 failed; 0 ignored'`nexit 0`n")
            chmod +x (Join-Path $cargoDir 'cargo')
        }
        $script:QCargo = $cargoDir
    }
    AfterEach { Remove-Item -Path Function:\pwsh -ErrorAction SilentlyContinue }

    It 'reports the throwing AfterAll in the head tree as a Pester failure and a regression (exit 1 with -FailOnRegression)' {
        $results = Join-Path $TestDrive 'qg-blocks-results'
        $savedPath = $env:PATH
        try {
            $env:PATH = "$script:QCargo$([IO.Path]::PathSeparator)$env:PATH"
            pwsh -NoProfile -File (Join-Path $script:QFx 'Tests/Invoke-QualityGate.ps1') -BaseRef $script:QBase -ResultsDir $results -FailOnRegression *> $null
            $exit = $LASTEXITCODE
        } finally { $env:PATH = $savedPath }
        $json = Get-Content (Get-ChildItem $results -Filter 'quality-gate-*.json' | Select-Object -First 1).FullName -Raw | ConvertFrom-Json
        $json.deltas.PesterFailed.base | Should -Be 0
        $json.deltas.PesterFailed.head | Should -BeGreaterThan 0 -Because 'AfterAll { throw } is a failed block, and a gate that drops blocks reports 0'
        $json.overall | Should -Be 'REGRESSED'
        $exit | Should -Be 1
    }

    It 'rejects a child that wrote a result but exited non-zero, wrote nothing, wrote garbage, or omitted a field' {
        $root = Join-Path $TestDrive 'qg-child-root'
        New-Item -ItemType Directory -Path (Join-Path $root 'Tests/Unit') -Force | Out-Null
        $script:ChildMode = ''
        function global:pwsh {
            $a = @($args); $i = [array]::IndexOf($a, '-OutPath')
            $out = $a[$i + 1]
            $ok = '{"Passed":3,"Failed":0,"FailedContainers":0,"FailedBlocks":0,"Executed":3,"Result":"Passed","Violations":[]}'
            switch ($script:ChildMode) {
                'exit7' { [IO.File]::WriteAllText($out, $ok); $global:LASTEXITCODE = 7 }
                'nothing' { $global:LASTEXITCODE = 0 }
                'garbage' { [IO.File]::WriteAllText($out, 'not json {'); $global:LASTEXITCODE = 0 }
                'nofield' { [IO.File]::WriteAllText($out, '{"Passed":3,"Executed":3}'); $global:LASTEXITCODE = 0 }
                default { [IO.File]::WriteAllText($out, $ok); $global:LASTEXITCODE = 0 }
            }
        }
        $script:ChildMode = 'ok'
        (Measure-PesterTests -Root $root).Passed | Should -Be 3
        foreach ($mode in 'exit7', 'nothing', 'garbage', 'nofield') {
            $script:ChildMode = $mode
            { Measure-PesterTests -Root $root } | Should -Throw -Because "child mode $mode must not read as a measurement"
        }
    }

    It 'counts blocks and an unexplained Failed result in the failure metric' {
        $root = Join-Path $TestDrive 'qg-child-root'
        function global:pwsh {
            $a = @($args); $i = [array]::IndexOf($a, '-OutPath')
            [IO.File]::WriteAllText($a[$i + 1], $script:ChildJson); $global:LASTEXITCODE = 0
        }
        $script:ChildJson = '{"Passed":1,"Failed":0,"FailedContainers":0,"FailedBlocks":1,"Executed":1,"Result":"Failed","Violations":[]}'
        (Measure-PesterTests -Root $root).Failed | Should -Be 1
        $script:ChildJson = '{"Passed":1,"Failed":2,"FailedContainers":1,"FailedBlocks":3,"Executed":3,"Result":"Failed","Violations":[]}'
        (Measure-PesterTests -Root $root).Failed | Should -Be 6
        $script:ChildJson = '{"Passed":1,"Failed":0,"FailedContainers":0,"FailedBlocks":0,"Executed":1,"Result":"Failed","Violations":[]}'
        (Measure-PesterTests -Root $root).Failed | Should -Be 1 -Because 'Result Failed with every counter 0 must never read as zero failures'
    }
}

Describe 'Invoke-QualityGate treats a rustc build failure as a build failure, not a lint' -Tag 'Unit', 'Portable' {
    # Protects: the Rust half of the A/B gate. Detects: a `compiler-message` with a true rustc error (E0432, unresolved
    # import; cargo exits 101 and build-finished reports success:false) counted as one lint, so a tree that does not
    # compile matches a base that does not compile either and the comparison passes (codex review of #384). Real cargo
    # and real clippy run against tiny scratch crates with a private CARGO_TARGET_DIR. A crate with exactly one clippy
    # lint (denied by -D warnings, so cargo also exits 101) is the positive control: it is one lint, not a build failure.
    # Skipped with a reported reason when cargo or clippy is missing.
    BeforeAll {
        $script:HasCargo = [bool](Get-Command cargo -ErrorAction SilentlyContinue)
        if ($script:HasCargo) { cargo clippy --version *> $null; $script:HasCargo = ($LASTEXITCODE -eq 0) }
        function script:New-ScratchCrate {
            param([string]$Root, [string]$LibText)
            $crate = Join-Path $Root 'Native/pcai_core'
            Write-Utf8 (Join-Path $crate 'Cargo.toml') "[package]`nname = ""scratch""`nversion = ""0.0.0""`nedition = ""2021""`n[features]`nserver = []`nffi = []`n"
            Write-Utf8 (Join-Path $crate 'src/lib.rs') $LibText
            $Root
        }
        function script:Use-ScratchCargoEnv {
            param([scriptblock]$Body)
            $saved = @{}
            foreach ($k in 'CARGO_TARGET_DIR', 'RUSTC_WRAPPER', 'CARGO_BUILD_RUSTC_WRAPPER') { $saved[$k] = [Environment]::GetEnvironmentVariable($k) }
            try {
                $env:CARGO_TARGET_DIR = Join-Path $TestDrive 'scratch-target'
                $env:RUSTC_WRAPPER = ''
                $env:CARGO_BUILD_RUSTC_WRAPPER = ''
                & $Body
            } finally { foreach ($k in $saved.Keys) { [Environment]::SetEnvironmentVariable($k, $saved[$k]) } }
        }
        $script:CleanLib = "pub fn f() -> i32 { 1 }`n#[cfg(test)]`nmod t { #[test] fn ok() { assert_eq!(super::f(), 1); } }`n"
        $script:OneLintLib = "pub fn f() -> i32 {`n    return 1;`n}`n#[cfg(test)]`nmod t { #[test] fn ok() { assert_eq!(super::f(), 1); } }`n"
        $script:BrokenLib = "use missing_crate::thing;`npub fn f() -> i32 { 1 }`n"
    }

    It 'counts exactly one clippy lint in a crate that builds, and does not call it a build failure' {
        if (-not $script:HasCargo) { Set-ItResult -Skipped -Because 'cargo or clippy is not available on this runner'; return }
        $root = New-ScratchCrate (Join-Path $TestDrive 'crate-onelint') $script:OneLintLib
        Use-ScratchCargoEnv { Measure-ClippyFindings -Root $root } | Should -Be 1
        $clean = New-ScratchCrate (Join-Path $TestDrive 'crate-clean') $script:CleanLib
        Use-ScratchCargoEnv { Measure-ClippyFindings -Root $clean } | Should -Be 0
    }

    It 'fails the clippy measurement with a build-failure reason for an unresolved import (E0432)' {
        if (-not $script:HasCargo) { Set-ItResult -Skipped -Because 'cargo or clippy is not available on this runner'; return }
        $root = New-ScratchCrate (Join-Path $TestDrive 'crate-broken') $script:BrokenLib
        { Use-ScratchCargoEnv { Measure-ClippyFindings -Root $root } } | Should -Throw '*build failed*'
    }

    It 'fails the Rust test measurement when the crate does not compile, and counts tests when it does' {
        if (-not $script:HasCargo) { Set-ItResult -Skipped -Because 'cargo or clippy is not available on this runner'; return }
        $broken = New-ScratchCrate (Join-Path $TestDrive 'crate-broken-test') $script:BrokenLib
        { Use-ScratchCargoEnv { Measure-RustTests -Root $broken } } | Should -Throw '*build failed*'
        $clean = New-ScratchCrate (Join-Path $TestDrive 'crate-clean-test') $script:CleanLib
        $r = Use-ScratchCargoEnv { Measure-RustTests -Root $clean }
        $r.Passed | Should -Be 1
        $r.Failed | Should -Be 0
    }

    It 'the whole gate fails when base and head fail to build the same way (a matching baseline must not pass)' -Skip:(-not (Get-Module -ListAvailable -Name PSScriptAnalyzer)) {
        if (-not $script:HasCargo) { Set-ItResult -Skipped -Because 'cargo or clippy is not available on this runner'; return }
        $fx = New-ScratchCrate (Join-Path $TestDrive 'qg-rust-broken') $script:BrokenLib
        Write-Utf8 (Join-Path $fx 'Modules/Mod/F.ps1') "function Get-F { 1 }`n"
        Write-Utf8 (Join-Path $fx 'Tests/Unit/A.Tests.ps1') "Describe 'a' -Tag 'Portable' { It 'a' { (1 + 1) | Should -Be 2 } }`n"
        New-Item -ItemType Directory -Path (Join-Path $fx 'Tests/Helpers') -Force | Out-Null
        Copy-Item $script:QualityGate (Join-Path $fx 'Tests/Invoke-QualityGate.ps1')
        Copy-Item $script:Helper (Join-Path $fx 'Tests/Helpers/Get-PesterRunVerdict.ps1')
        Invoke-FixtureGit $fx @('init', '-q', '-b', 'main') | Out-Null
        Invoke-FixtureGit $fx @('add', '-A') | Out-Null
        Invoke-FixtureGit $fx @('commit', '-q', '-m', 'base') | Out-Null
        $base = (Invoke-FixtureGit $fx @('rev-parse', 'HEAD')) -join ''
        Write-Utf8 (Join-Path $fx 'Tests/Unit/B.Tests.ps1') "Describe 'b' -Tag 'Portable' { It 'b' { (2 + 2) | Should -Be 4 } }`n"
        Invoke-FixtureGit $fx @('add', '-A') | Out-Null
        Invoke-FixtureGit $fx @('commit', '-q', '-m', 'head') | Out-Null
        $results = Join-Path $TestDrive 'qg-rust-broken-results'
        Use-ScratchCargoEnv {
            pwsh -NoProfile -File (Join-Path $fx 'Tests/Invoke-QualityGate.ps1') -BaseRef $base -ResultsDir $results *> $null
            $script:BrokenExit = $LASTEXITCODE
        }
        $script:BrokenExit | Should -Be 1
        $json = Get-Content (Get-ChildItem $results -Filter 'quality-gate-*.json' | Select-Object -First 1).FullName -Raw | ConvertFrom-Json
        $json.overall | Should -Be 'ERROR'
        ($json.measurement_errors -join ' ') | Should -Match 'HEAD.*build failed'
        ($json.measurement_errors -join ' ') | Should -Match 'BASE.*build failed'
    }
}


Describe 'Invoke-QualityGate: any failed build is a build failure, and lints are capped to warnings' -Tag 'Unit', 'Portable' {
    # Protects: the build check in Measure-ClippyFindings. Detects: a lint masking a build that is broken in another way
    # (codex review of #384, third finding). A scratch workspace with one crate that has an unused variable and another
    # crate whose build.rs panics makes cargo exit 101; the panic goes to stderr as text ("failed to run custom build
    # command"), not into the JSON stream, and the earlier check accepted any failed build as long as some lint existed.
    # Design now: clippy runs with `--cap-lints warn`, so lints are warnings and can never fail the build; then ANY
    # non-zero exit, `build-finished` other than success:true, or error diagnostic is a build failure, with no stderr
    # classification or lint-denial reasoning. Stderr is still captured, for the failure message.
    BeforeAll {
        $script:HasCargo3 = [bool](Get-Command cargo -ErrorAction SilentlyContinue)
        if ($script:HasCargo3) { cargo clippy --version *> $null; $script:HasCargo3 = ($LASTEXITCODE -eq 0) }
        function script:Use-ScratchCargoEnv {
            param([scriptblock]$Body)
            $saved = @{}
            foreach ($k in 'CARGO_TARGET_DIR', 'RUSTC_WRAPPER', 'CARGO_BUILD_RUSTC_WRAPPER') { $saved[$k] = [Environment]::GetEnvironmentVariable($k) }
            try {
                $env:CARGO_TARGET_DIR = Join-Path $TestDrive 'scratch-target-mixed'
                $env:RUSTC_WRAPPER = ''
                $env:CARGO_BUILD_RUSTC_WRAPPER = ''
                & $Body
            } finally { foreach ($k in $saved.Keys) { [Environment]::SetEnvironmentVariable($k, $saved[$k]) } }
        }
        function script:Get-FakeCargoStream {
            param([string[]]$Lines, [int]$Exit)
            $script:FakeStream = $Lines; $script:FakeExit = $Exit; $script:CargoArgs = $null
            function global:cargo { $script:CargoArgs = @($args); $global:LASTEXITCODE = $script:FakeExit; $script:FakeStream }
        }
        $script:LintJson = '{"reason":"compiler-message","message":{"level":"warning","code":{"code":"unused_variables"},"message":"unused variable: `x`","spans":[{"is_primary":true,"file_name":"a/src/lib.rs","line_start":1,"column_start":5}]}}'
        $script:FinishedFalse = '{"reason":"build-finished","success":false}'
        $script:FinishedTrue = '{"reason":"build-finished","success":true}'
        $script:CouldNotCompile = 'error: could not compile `a` (lib) due to 1 previous error'
    }
    AfterEach { Remove-Item -Path Function:\cargo -ErrorAction SilentlyContinue }

    It 'counts capped lints once each and passes a successful build; the command caps lints instead of denying them' {
        Get-FakeCargoStream @($script:LintJson, $script:LintJson, $script:FinishedTrue) 0
        Measure-ClippyFindings -Root (Join-Path $TestDrive 'qg') | Should -Be 1
        ($script:CargoArgs -join ' ') | Should -Match '--cap-lints warn'
        ($script:CargoArgs -join ' ') | Should -Not -Match '-D warnings'
    }

    It 'rejects a lint plus a failed custom build command in stderr, and quotes the stderr in the message' {
        Get-FakeCargoStream @($script:LintJson, 'error: failed to run custom build command for `b v0.0.0 (C:\x\b)`', 'Caused by:', '  process didn''t exit successfully (exit code: 101)', $script:CouldNotCompile, $script:FinishedFalse) 101
        { Measure-ClippyFindings -Root (Join-Path $TestDrive 'qg') } | Should -Throw '*build failed*custom build command*'
    }

    It 'rejects every combination that is not exit 0 with build-finished success:true' {
        foreach ($case in @(
                @{ Lines = @($script:LintJson, $script:FinishedFalse); Exit = 101; Name = 'exit 101, success false, only lints' }
                @{ Lines = @($script:LintJson, $script:FinishedFalse); Exit = 0; Name = 'exit 0 but success false' }
                @{ Lines = @($script:LintJson, $script:FinishedTrue); Exit = 101; Name = 'success true but exit 101' }
                @{ Lines = @($script:LintJson); Exit = 0; Name = 'no build-finished at all' }
                @{ Lines = @($script:LintJson, 'error: linker `link.exe` not found', $script:FinishedFalse); Exit = 101; Name = 'a linker error' }
            )) {
            Get-FakeCargoStream $case.Lines $case.Exit
            { Measure-ClippyFindings -Root (Join-Path $TestDrive 'qg') } | Should -Throw '*build failed*' -Because $case.Name
        }
    }

    It 'rejects a codeless rustc error and an E0432 error even when the build-finished message says success' {
        $syntax = '{"reason":"compiler-message","message":{"level":"error","code":null,"message":"expected `;`, found `}`","spans":[]}}'
        $e0432 = '{"reason":"compiler-message","message":{"level":"error","code":{"code":"E0432"},"message":"unresolved import `x`","spans":[]}}'
        foreach ($bad in $syntax, $e0432) {
            Get-FakeCargoStream @($script:LintJson, $bad, $script:FinishedTrue) 0
            { Measure-ClippyFindings -Root (Join-Path $TestDrive 'qg') } | Should -Throw '*build failed*'
        }
    }

    It 'a real scratch workspace with a lint in one crate and a panicking build.rs in another is a build failure' {
        if (-not $script:HasCargo3) { Set-ItResult -Skipped -Because 'cargo or clippy is not available on this runner'; return }
        $root = Join-Path $TestDrive 'ws-mixed'
        $ws = Join-Path $root 'Native/pcai_core'
        Write-Utf8 (Join-Path $ws 'Cargo.toml') "[workspace]`nresolver = ""2""`nmembers = [""lintcrate"", ""buildcrate""]`n"
        Write-Utf8 (Join-Path $ws 'lintcrate/Cargo.toml') "[package]`nname = ""lintcrate""`nversion = ""0.0.0""`nedition = ""2021""`n"
        Write-Utf8 (Join-Path $ws 'lintcrate/src/lib.rs') "pub fn f() {`n    let unused = 1;`n}`n"
        Write-Utf8 (Join-Path $ws 'buildcrate/Cargo.toml') "[package]`nname = ""buildcrate""`nversion = ""0.0.0""`nedition = ""2021""`nbuild = ""build.rs""`n"
        Write-Utf8 (Join-Path $ws 'buildcrate/build.rs') "fn main() {`n    panic!(""scratch build script failure"");`n}`n"
        Write-Utf8 (Join-Path $ws 'buildcrate/src/lib.rs') "pub fn g() {}`n"
        { Use-ScratchCargoEnv { Measure-ClippyFindings -Root $root } } | Should -Throw '*build failed*'
        # Control: the same lint crate alone is one lint and a passing build, so the throw above is the build script.
        Remove-Item (Join-Path $ws 'buildcrate') -Recurse -Force
        Write-Utf8 (Join-Path $ws 'Cargo.toml') "[workspace]`nresolver = ""2""`nmembers = [""lintcrate""]`n"
        Use-ScratchCargoEnv { Measure-ClippyFindings -Root $root } | Should -Be 1
    }
}

Describe 'Invoke-QualityGate: a failing test binary does not hide a new failure in a later one' -Tag 'Unit', 'Portable' {
    # Protects: the Rust test comparison. Detects: `cargo test` without --no-fail-fast stopping at the first failing test
    # binary (a_fail.rs), so a later binary (z_hidden.rs) with a NEW failure never runs; base and head both read
    # Passed=0 Failed=1 and the gate sees no change (codex review of #384). Fix: --no-fail-fast, every `test result:`
    # line summed, and failures compared by name, not only by count. Real cargo, a two-binary scratch crate, a private
    # CARGO_TARGET_DIR; skipped with a reported reason when cargo or clippy is missing.
    BeforeAll {
        $script:HasCargo4 = [bool](Get-Command cargo -ErrorAction SilentlyContinue)
        if ($script:HasCargo4) { cargo clippy --version *> $null; $script:HasCargo4 = ($LASTEXITCODE -eq 0) }
        function script:Use-ScratchCargoEnv4 {
            param([scriptblock]$Body)
            $saved = @{}
            foreach ($k in 'CARGO_TARGET_DIR', 'RUSTC_WRAPPER', 'CARGO_BUILD_RUSTC_WRAPPER') { $saved[$k] = [Environment]::GetEnvironmentVariable($k) }
            try {
                $env:CARGO_TARGET_DIR = Join-Path $TestDrive 'scratch-target-two'
                $env:RUSTC_WRAPPER = ''
                $env:CARGO_BUILD_RUSTC_WRAPPER = ''
                & $Body
            } finally { foreach ($k in $saved.Keys) { [Environment]::SetEnvironmentVariable($k, $saved[$k]) } }
        }
        function script:Write-TwoBinaryCrate {
            param([string]$Root, [bool]$WithHidden)
            $crate = Join-Path $Root 'Native/pcai_core'
            Write-Utf8 (Join-Path $crate 'Cargo.toml') "[package]`nname = ""scratch""`nversion = ""0.0.0""`nedition = ""2021""`n[features]`nserver = []`nffi = []`n"
            Write-Utf8 (Join-Path $crate 'src/lib.rs') "pub fn f() -> i32 { 1 }`n"
            Write-Utf8 (Join-Path $crate 'tests/a_fail.rs') "#[test]`nfn known_failure() { assert_eq!(1, 2); }`n"
            if ($WithHidden) { Write-Utf8 (Join-Path $crate 'tests/z_hidden.rs') "#[test]`nfn new_failure() { assert_eq!(3, 4); }`n" }
        }
        $script:TwoPester = "Describe 'a' -Tag 'Portable' { It 'a' { (1 + 1) | Should -Be 2 } }`n"
    }
    AfterEach { Remove-Item -Path Function:\cargo -ErrorAction SilentlyContinue }

    It 'measures every test binary, passes --no-fail-fast, and names the failed tests' {
        if (-not $script:HasCargo4) { Set-ItResult -Skipped -Because 'cargo or clippy is not available on this runner'; return }
        $root = Join-Path $TestDrive 'two-binary-head'
        Write-TwoBinaryCrate $root $true
        $r = Use-ScratchCargoEnv4 { Measure-RustTests -Root $root }
        $r.Failed | Should -Be 2 -Because 'z_hidden must run even though a_fail failed first'
        @($r.FailedNames).Count | Should -Be 2
        ($r.FailedNames -join ' ') | Should -Match 'known_failure'
        ($r.FailedNames -join ' ') | Should -Match 'new_failure'
        Get-Content (Join-Path $PSScriptRoot '../Invoke-QualityGate.ps1') -Raw | Should -Match '--no-fail-fast'
    }

    It 'the whole gate reports the new failure in the later binary as a regression (exit 1 with -FailOnRegression)' -Skip:(-not (Get-Module -ListAvailable -Name PSScriptAnalyzer)) {
        if (-not $script:HasCargo4) { Set-ItResult -Skipped -Because 'cargo or clippy is not available on this runner'; return }
        $fx = Join-Path $TestDrive 'two-binary-repo'
        Write-TwoBinaryCrate $fx $false
        Write-Utf8 (Join-Path $fx 'Modules/Mod/F.ps1') "function Get-F { 1 }`n"
        Write-Utf8 (Join-Path $fx 'Tests/Unit/A.Tests.ps1') $script:TwoPester
        New-Item -ItemType Directory -Path (Join-Path $fx 'Tests/Helpers') -Force | Out-Null
        Copy-Item $script:QualityGate (Join-Path $fx 'Tests/Invoke-QualityGate.ps1')
        Copy-Item $script:Helper (Join-Path $fx 'Tests/Helpers/Get-PesterRunVerdict.ps1')
        Invoke-FixtureGit $fx @('init', '-q', '-b', 'main') | Out-Null
        Invoke-FixtureGit $fx @('add', '-A') | Out-Null
        Invoke-FixtureGit $fx @('commit', '-q', '-m', 'base') | Out-Null
        $base = (Invoke-FixtureGit $fx @('rev-parse', 'HEAD')) -join ''
        Write-TwoBinaryCrate $fx $true
        Invoke-FixtureGit $fx @('add', '-A') | Out-Null
        Invoke-FixtureGit $fx @('commit', '-q', '-m', 'head') | Out-Null
        $results = Join-Path $TestDrive 'two-binary-results'
        Use-ScratchCargoEnv4 {
            pwsh -NoProfile -File (Join-Path $fx 'Tests/Invoke-QualityGate.ps1') -BaseRef $base -ResultsDir $results -FailOnRegression *> $null
            $script:TwoExit = $LASTEXITCODE
        }
        $json = Get-Content (Get-ChildItem $results -Filter 'quality-gate-*.json' | Select-Object -First 1).FullName -Raw | ConvertFrom-Json
        $json.deltas.RustTestsFailed.base | Should -Be 1
        $json.deltas.RustTestsFailed.head | Should -Be 2 -Because 'without --no-fail-fast both sides read 1 and the new failure is hidden'
        $json.deltas.RustNewFailedTests.head | Should -Be 1
        $json.overall | Should -Be 'REGRESSED'
        $script:TwoExit | Should -Be 1
    }

    It 'pins CARGO_USE_NEXTEST to 0 while it runs cargo test and restores the caller value (nextest prints no test result lines)' {
        $fx = Join-Path $TestDrive 'nextest-pin'
        New-Item -ItemType Directory -Path (Join-Path $fx 'Native/pcai_core') -Force | Out-Null
        $script:SeenNextest = 'unset'
        function global:cargo {
            $script:SeenNextest = $env:CARGO_USE_NEXTEST
            $global:LASTEXITCODE = 0
            'test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out'
        }
        $saved = $env:CARGO_USE_NEXTEST
        try {
            $env:CARGO_USE_NEXTEST = '1'
            $r = Measure-RustTests -Root $fx
            $after = $env:CARGO_USE_NEXTEST
        } finally { $env:CARGO_USE_NEXTEST = $saved }
        $script:SeenNextest | Should -Be '0'
        $after | Should -Be '1'
        $r.Passed | Should -Be 2
    }
    It 'compares failures by name: a swapped failure with an unchanged count is still a regression' {
        $fx = Join-Path $TestDrive 'swap-names'
        function global:cargo {
            $global:LASTEXITCODE = 101
            '     Running tests\a_fail.rs (target\debug\deps\a_fail-1.exe)'
            "test $($script:SwapName) ... FAILED"
            'test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out'
        }
        New-Item -ItemType Directory -Path (Join-Path $fx 'Native/pcai_core') -Force | Out-Null
        $script:SwapName = 'old_failure'
        $a = Measure-RustTests -Root $fx
        $script:SwapName = 'different_failure'
        $b = Measure-RustTests -Root $fx
        $a.Failed | Should -Be $b.Failed
        ($a.FailedNames -join ',') | Should -Not -Be ($b.FailedNames -join ',')
        @(Compare-RustFailedNames -Base $a.FailedNames -Head $b.FailedNames).Count | Should -Be 1
        @(Compare-RustFailedNames -Base $a.FailedNames -Head $a.FailedNames).Count | Should -Be 0
    }
}

Describe 'coverage-report.ps1 treats a missing Pester result as a measurement error' -Tag 'Unit', 'Portable' {
    # Protects: the assessment failing closed. Detects: Invoke-Pester returning no result object (a crash, a swallowed
    # error, PassThru lost) leaving `$pester` null, so `$ran` was false, every run gate was skipped and a seeded baseline
    # passed on the static checks alone (codex review of #384, Tests/coverage-report.ps1 `$ran` and `if ($pester)`).
    # A fake Pester 5.99 module placed first on PSModulePath stands in for Pester in a real child run of the script, so
    # the real exit code is what is asserted. PCAI_FAKE_PESTER selects what its Invoke-Pester returns.
    BeforeAll {
        $script:NullFx = Join-Path $TestDrive 'null-pester-repo'
        Write-Utf8 (Join-Path $script:NullFx 'Modules/M/F.ps1') "function Get-F { 1 }`n"
        Write-Utf8 (Join-Path $script:NullFx 'Tests/Unit/Case.Tests.ps1') "Describe 'g' -Tag 'Portable' { It 'ok' { (1 + 1) | Should -Be 2 } }`n"
        pwsh -NoProfile -File $script:Tool -RepoRoot $script:NullFx -Tier Portable -UpdateBaseline -OutputPath (Join-Path $TestDrive 'null-seed.json') *> $null
        $script:NullSeedExit = $LASTEXITCODE
        $fake = Join-Path $TestDrive 'fakemods/Pester/5.99.0'
        New-Item -ItemType Directory -Path $fake -Force | Out-Null
        Write-Utf8 (Join-Path $fake 'Pester.psd1') "@{ RootModule = 'Pester.psm1'; ModuleVersion = '5.99.0'; GUID = '11111111-2222-3333-4444-555555555555'; FunctionsToExport = @('New-PesterConfiguration', 'Invoke-Pester') }`n"
        $fakeBody = @(
            'function New-PesterConfiguration {'
            '    [pscustomobject]@{'
            '        Run = [pscustomobject]@{ Path = $null; PassThru = $null; Exit = $null }'
            '        Filter = [pscustomobject]@{ Tag = $null }'
            '        Output = [pscustomobject]@{ Verbosity = $null }'
            '        TestResult = [pscustomobject]@{ Enabled = $null }'
            '        CodeCoverage = [pscustomobject]@{ Enabled = $null; Path = $null; RecursePaths = $null; ExcludeTests = $null; OutputFormat = $null; OutputPath = $null }'
            '    }'
            '}'
            'function Invoke-Pester {'
            '    param($Configuration)'
            '    switch ($env:PCAI_FAKE_PESTER) {'
            '        ''null'' { return }'
            '        ''empty'' { [pscustomobject]@{} }'
            '        ''emptyarray'' { , @() }'
            '        ''throws'' { throw ''pester crashed'' }'
            '        ''nocoverage'' { [pscustomobject]@{ Result = ''Passed''; PassedCount = 1; FailedCount = 0; FailedContainersCount = 0; FailedBlocksCount = 0; TotalCount = 1; NotRunCount = 0; SkippedCount = 0; Containers = @(); Failed = @(); FailedBlocks = @(); CodeCoverage = $null } }'
            '    }'
            '}'
        ) -join "`n"
        Write-Utf8 (Join-Path $fake 'Pester.psm1') ($fakeBody + "`n")
        $script:FakeModuleRoot = Join-Path $TestDrive 'fakemods'
        function script:Invoke-FakePesterRun {
            param([string]$Mode, [string]$Root, [string]$Name, [switch]$NoRun)
            $savedPath = $env:PSModulePath; $savedMode = $env:PCAI_FAKE_PESTER
            try {
                $env:PSModulePath = $script:FakeModuleRoot + [IO.Path]::PathSeparator + $env:PSModulePath
                $env:PCAI_FAKE_PESTER = $Mode
                $out = Join-Path $TestDrive "fake-$Name.json"
                $extra = @(if ($NoRun) { '-NoRun' })
                pwsh -NoProfile -File $script:Tool -RepoRoot $Root -Tier Portable -OutputPath $out @extra *> $null
                [pscustomobject]@{ Exit = $LASTEXITCODE; Doc = (Get-Content $out -Raw -ErrorAction SilentlyContinue | ConvertFrom-Json) }
            } finally { $env:PSModulePath = $savedPath; $env:PCAI_FAKE_PESTER = $savedMode }
        }
    }

    It 'seeds the fixture baseline from a real run (control)' {
        $script:NullSeedExit | Should -Be 0
        Test-Path (Join-Path $script:NullFx 'Config/test-baseline.json') | Should -BeTrue
    }

    It 'exits 1 with a pester_run violation when Invoke-Pester returns nothing, an empty object, an empty array or throws' {
        foreach ($mode in 'null', 'empty', 'emptyarray', 'throws') {
            $r = Invoke-FakePesterRun -Mode $mode -Root $script:NullFx -Name $mode
            $r.Exit | Should -Be 1 -Because "Invoke-Pester mode '$mode' must be a measurement error, not a pass"
            $r.Doc.verdict | Should -Be 'fail'
            @($r.Doc.violations | ForEach-Object { $_.code }) | Should -Contain 'pester_run' -Because $mode
        }
    }

    It 'exits 1 with coverage_missing when the result has no coverage data' {
        $r = Invoke-FakePesterRun -Mode 'nocoverage' -Root $script:NullFx -Name 'nocoverage'
        $r.Exit | Should -Be 1
        @($r.Doc.violations | ForEach-Object { $_.code }) | Should -Contain 'coverage_missing'
    }

    It 'with -NoRun the same fixture still passes its static checks (control: an explicit opt-out is not an error)' {
        $r = Invoke-FakePesterRun -Mode 'null' -Root $script:NullFx -Name 'norun' -NoRun
        $r.Exit | Should -Be 0
        $r.Doc.verdict | Should -Be 'pass'
        $r.Doc.PSObject.Properties.Name | Should -Not -Contain 'run_summary'
    }

    It 'positive control: the real Pester run of the same fixture still passes' {
        pwsh -NoProfile -File $script:Tool -RepoRoot $script:NullFx -Tier Portable -OutputPath (Join-Path $TestDrive 'null-real.json') *> $null
        $LASTEXITCODE | Should -Be 0
    }
}

Describe 'An explicit empty orphans_allowed means allow none and is never widened' -Tag 'Unit', 'Portable' {
    # Protects: the orphan allowlist ratchet. Detects: a truthiness check (`if ($old.orphans_allowed)`) treating a
    # present but empty list as absent, so -UpdateBaseline re-seeded it from the current orphans, widened it, and then
    # passed the gate it had just loosened (codex review of #384). Negative control: a baseline with
    # "orphans_allowed": [] plus a new orphan fails, and -UpdateBaseline leaves the list empty and still fails.
    BeforeAll {
        $script:OrphFx = Join-Path $TestDrive 'orphan-repo'
        Write-Utf8 (Join-Path $script:OrphFx 'Tests/Unit/A.Tests.ps1') "Describe 'a' -Tag 'Portable' { It 'ok' { (1 + 1) | Should -Be 2 } }`n"
        pwsh -NoProfile -File $script:Tool -RepoRoot $script:OrphFx -NoRun -UpdateBaseline -OutputPath (Join-Path $TestDrive 'orph-seed.json') *> $null
        $script:OrphSeedExit = $LASTEXITCODE
        $script:OrphBaselinePath = Join-Path $script:OrphFx 'Config/test-baseline.json'
    }

    It 'seeds an explicit empty allowlist for a fixture with no orphans (control)' {
        $script:OrphSeedExit | Should -Be 0
        $json = Get-Content $script:OrphBaselinePath -Raw
        $json | Should -Match '"orphans_allowed":\s*\[\s*\]'
    }

    It 'fails a new orphan, and -UpdateBaseline keeps the empty list empty and still fails' {
        Write-Utf8 (Join-Path $script:OrphFx 'Tools/Odd/New.Tests.ps1') "Describe 'n' { It 'ok' { (3 + 3) | Should -Be 6 } }`n"
        $before = Get-Content $script:OrphBaselinePath -Raw
        pwsh -NoProfile -File $script:Tool -RepoRoot $script:OrphFx -NoRun -OutputPath (Join-Path $TestDrive 'orph-plain.json') *> $null
        $LASTEXITCODE | Should -Be 1
        (Get-Content $script:OrphBaselinePath -Raw) | Should -Be $before -Because 'a plain run never writes the baseline'
        pwsh -NoProfile -File $script:Tool -RepoRoot $script:OrphFx -NoRun -UpdateBaseline -OutputPath (Join-Path $TestDrive 'orph-update.json') *> $null
        $LASTEXITCODE | Should -Be 1 -Because 'the new orphan must still be a violation after the ratchet'
        $after = Get-Content $script:OrphBaselinePath -Raw | ConvertFrom-Json
        @($after.orphans_allowed).Count | Should -Be 0
        Remove-Item (Join-Path $script:OrphFx 'Tools/Odd/New.Tests.ps1')
    }

    It 'keeps the other keys where 0 is legitimate (target, ratio, per-module floors, known-failure and quality counts)' {
        $seeded = [pscustomobject]@{
            schema_version = 1; target_coverage_pct = 0; files_analyzed_ratio_min = 0; per_module_tolerance_pct = 0
            quality = [pscustomobject]@{ no_assertion_it_max = 0; not_throw_only_it_max = 0; tautology_it_max = 0; parse_error_files_max = 0 }
            orphans_allowed = @()
            tiers = [pscustomobject]@{ Portable = [pscustomobject]@{ coverage_min_pct = 0; per_module_min_pct = [pscustomobject]@{ Alpha = 0 }
                    known_failures = [pscustomobject]@{ containers = @(); tests_by_file = [pscustomobject]@{}; blocks_by_file = [pscustomobject]@{} } } }
        }
        $assessment = [ordered]@{
            quality = [ordered]@{ no_assertion_it = 5; not_throw_only_it = 5; tautology_it = 5; parse_error_files = 5 }
            reach = [ordered]@{ orphans = @('x.Tests.ps1') }
            coverage = [ordered]@{ pester = [ordered]@{ pct = 0.0; files_analyzed = 1; module_files_total = 1; by_module = @([pscustomobject]@{ module = 'Alpha'; pct = 0.0 }) } }
        }
        $run = [pscustomobject]@{ Result = 'Passed'; PassedCount = 1; FailedCount = 0; FailedContainersCount = 0; FailedBlocksCount = 0; TotalCount = 1; Containers = @(); Failed = @(); FailedBlocks = @() }
        $u = Update-AssessmentBaseline -Baseline $seeded -Assessment $assessment -Tier Portable -PesterResult $run -Ran
        $u.target_coverage_pct | Should -Be 0
        $u.files_analyzed_ratio_min | Should -Be 0
        $u.per_module_tolerance_pct | Should -Be 0
        $u.quality.no_assertion_it_max | Should -Be 0 -Because 'a zero maximum must stay zero when the run is worse'
        $u.quality.tautology_it_max | Should -Be 0
        @($u.orphans_allowed).Count | Should -Be 0
        $u.tiers.Portable.coverage_min_pct | Should -Be 0
        $u.tiers.Portable.per_module_min_pct.Alpha | Should -Be 0
    }
}
