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
        $v = Get-PesterRunVerdict -Result $r
        $out[$case.name] = [ordered]@{
            verdict = $v
            raw = [ordered]@{ Result = [string]$r.Result; FailedCount = $r.FailedCount; FailedContainers = $r.FailedContainersCount; PassedCount = $r.PassedCount; TotalCount = $r.TotalCount }
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
            tiers = [pscustomobject]@{ Portable = [pscustomobject]@{ coverage_min_pct = 0; per_module_min_pct = [pscustomobject]@{}; known_failures = [pscustomobject]@{ max_failed_containers = 0; max_failed_blocks = 0; tests_by_file = [pscustomobject]@{} } } }
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
        $script:GateBaseline = '{"schema_version":1,"target_coverage_pct":85,"files_analyzed_ratio_min":0.95,"quality":{"no_assertion_it_max":0,"not_throw_only_it_max":0,"tautology_it_max":2,"parse_error_files_max":0},"orphans_allowed":["a.Tests.ps1"],"tiers":{"Portable":{"coverage_min_pct":48,"per_module_min_pct":{"Alpha":40},"known_failures":{"max_failed_containers":0,"max_failed_blocks":0,"tests_by_file":{}}}}}' | ConvertFrom-Json
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
    AfterEach { Remove-Item function:global:cargo -ErrorAction SilentlyContinue }

    It 'sums every test result line, not only the last' {
        function global:cargo { $global:LASTEXITCODE = 0; 'test result: ok. 3 passed; 0 failed; 1 ignored'; 'test result: ok. 4 passed; 1 failed; 0 ignored' }
        $r = Measure-RustTests -Root (Join-Path $TestDrive 'qg')
        $r.Passed | Should -Be 7
        $r.Failed | Should -Be 1
    }

    It 'treats a compile failure as a measurement error, not 0/0' {
        function global:cargo { $global:LASTEXITCODE = 101; 'error[E0432]: unresolved import' }
        { Measure-RustTests -Root (Join-Path $TestDrive 'qg') } | Should -Throw '*no test results*'
    }

    It 'counts clippy findings at level error (-D warnings) and treats a silent failure as an error' {
        function global:cargo {
            $global:LASTEXITCODE = 101
            '{"reason":"compiler-message","message":{"level":"error","code":{"code":"clippy::needless_return"},"message":"x"}}'
            '{"reason":"compiler-message","message":{"level":"warning","code":{"code":"unused_variables"},"message":"y"}}'
            '{"reason":"compiler-message","message":{"level":"error","code":null,"message":"aborting due to 1 previous error"}}'
        }
        Measure-ClippyFindings -Root (Join-Path $TestDrive 'qg') | Should -Be 2
        function global:cargo { $global:LASTEXITCODE = 101; 'error: could not find Cargo.toml' }
        { Measure-ClippyFindings -Root (Join-Path $TestDrive 'qg') } | Should -Throw '*no lint diagnostics*'
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
                $body = if ($ExitCode -eq 0) { "@echo off`r`nif ""%1""==""clippy"" exit /b 0`r`necho test result: ok. 2 passed; 0 failed; 0 ignored`r`nexit /b 0`r`n" } else { "@echo off`r`necho error: fake toolchain failure`r`nexit /b $ExitCode`r`n" }
                [IO.File]::WriteAllText((Join-Path $dir 'cargo.cmd'), $body)
            } else {
                $body = if ($ExitCode -eq 0) { "#!/bin/sh`n[ ""`$1"" = clippy ] && exit 0`necho 'test result: ok. 2 passed; 0 failed; 0 ignored'`nexit 0`n" } else { "#!/bin/sh`necho 'error: fake toolchain failure'`nexit $ExitCode`n" }
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
