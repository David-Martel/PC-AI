BeforeAll {
    . "$PSScriptRoot/../../Tools/Assert-PesterCoverageGate.ps1"
    function New-GateResult {
        [pscustomobject]@{FailedCount = 0; FailedContainersCount = 0; FailedBlocksCount = 0; TotalCount = 1; CodeCoverage = [pscustomobject]@{CoveragePercent = 85.0; CommandsAnalyzedCount = 100; CommandsExecutedCount = 85 } }
    }
}

Describe 'Shared Pester result validation without coverage' {
    BeforeAll { . "$PSScriptRoot/../../Tools/Assert-PesterCoverageGate.ps1" }
    It 'accepts a successful actual result shape without a coverage property' {
        $result = [pscustomobject]@{ TotalCount = 1; FailedCount = 0; FailedBlocksCount = 0; FailedContainersCount = 0 }
        { Assert-PesterResult -Result $result } | Should -Not -Throw
    }
    It 'rejects <Field>=<Value> without relying on coverage' -ForEach @(
        @{Field = 'FailedCount'; Value = 1 }, @{Field = 'FailedBlocksCount'; Value = 1 }, @{Field = 'FailedContainersCount'; Value = 1 },
        @{Field = 'TotalCount'; Value = 0 }, @{Field = 'FailedCount'; Value = -1 }, @{Field = 'TotalCount'; Value = 'unknown' }
    ) {
        $result = [pscustomobject]@{ TotalCount = 1; FailedCount = 0; FailedBlocksCount = 0; FailedContainersCount = 0 }
        $result.$Field = $Value
        { Assert-PesterResult -Result $result } | Should -Throw
    }
    It 'rejects a missing failure field without reporting success' {
        $result = [pscustomobject]@{ TotalCount = 1; FailedCount = 0; FailedContainersCount = 0 }
        { Assert-PesterResult -Result $result } | Should -Throw '*Missing*'
    }
    It 'rejects an actual all-NotRun selection without requiring coverage' {
        $result = [pscustomobject]@{ TotalCount = 12; FailedCount = 0; FailedBlocksCount = 0; FailedContainersCount = 0; NotRunCount = 12 }
        { Assert-PesterResult -Result $result } | Should -Throw '*all discovered tests are NotRun*'
    }
}
Describe 'Explicit Pester gate fail-closed contracts' {
    It 'accepts exactly 85 percent' { { Assert-PesterCoverageGate (New-GateResult) } | Should -Not -Throw }
    It 'rejects below target with zero test failures' { $r = New-GateResult; $r.CodeCoverage.CoveragePercent = 84.999; { Assert-PesterCoverageGate $r } | Should -Throw '*below required*' }
    It 'rejects invalid percentage <Value>' -ForEach @(@{Value = [double]::NaN }, @{Value = [double]::PositiveInfinity }, @{Value = -1 }, @{Value = 101 }, @{Value = 'invalid' }) { $r = New-GateResult; $r.CodeCoverage.CoveragePercent = $Value; { Assert-PesterCoverageGate $r } | Should -Throw }
    It 'rejects missing coverage' { $r = New-GateResult; $r.CodeCoverage = $null; { Assert-PesterCoverageGate $r } | Should -Throw '*Missing*' }
    It 'rejects missing percentage' { $r = New-GateResult; $r.CodeCoverage.PSObject.Properties.Remove('CoveragePercent'); { Assert-PesterCoverageGate $r } | Should -Throw '*Missing*' }
    It 'rejects zero analyzed commands' { $r = New-GateResult; $r.CodeCoverage.CommandsAnalyzedCount = 0; { Assert-PesterCoverageGate $r } | Should -Throw '*zero*' }
    It 'rejects impossible executed count' { $r = New-GateResult; $r.CodeCoverage.CommandsExecutedCount = 101; { Assert-PesterCoverageGate $r } | Should -Throw '*executed*' }
    It 'does not trust rounded or inconsistent percentage over actual commands' { $r = New-GateResult; $r.CodeCoverage.CommandsExecutedCount = 84; { Assert-PesterCoverageGate $r } | Should -Throw '*below required*' }
    It 'rejects actual test failure' { $r = New-GateResult; $r.FailedCount = 1; { Assert-PesterCoverageGate $r } | Should -Throw '*failures*' }
    It 'rejects container failure' { $r = New-GateResult; $r.FailedContainersCount = 1; { Assert-PesterCoverageGate $r } | Should -Throw '*failures*' }
    It 'rejects no test discovery' { $r = New-GateResult; $r.TotalCount = 0; { Assert-PesterCoverageGate $r } | Should -Throw '*No tests*' }
    It 'rejects missing failure field' { $r = New-GateResult; $r.PSObject.Properties.Remove('FailedCount'); { Assert-PesterCoverageGate $r } | Should -Throw '*Missing*' }
}

Describe 'Actual Pester coverage admission' {
    BeforeAll {
        $script:GatePath = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../../Tools/Assert-PesterCoverageGate.ps1'))
        $script:PesterManifest = Join-Path (Get-Module Pester).ModuleBase 'Pester.psd1'
    }

    It 'retains actual failed outcomes before the canonical covered runner rejects them' {
        $repo = Split-Path -Parent (Split-Path -Parent $script:GatePath)
        $owned = Join-Path $TestDrive 'canonical-covered-runner'
        foreach ($directory in @('Tests/Unit', 'Modules', 'Tools')) {
            [void][IO.Directory]::CreateDirectory((Join-Path $owned $directory))
        }
        foreach ($file in @('Tests/.pester.ps1', 'Tests/TestProfiles.psd1', 'Tests/PesterConfiguration.psd1', 'Tools/Assert-PesterCoverageGate.ps1', 'Tools/Merge-PesterChildCoverage.ps1')) {
            Copy-Item -LiteralPath (Join-Path $repo $file) -Destination (Join-Path $owned $file)
        }
        @'
function Get-CanonicalFixtureValue {
    param([bool]$Positive)
    if ($Positive) { return 'positive' }
    return 'other'
}
'@ | Set-Content -LiteralPath (Join-Path $owned 'Modules/Fixture.ps1')
        @'
BeforeAll { . "$PSScriptRoot/../../Modules/Fixture.ps1" }
Describe 'Actual canonical covered failure' {
    It 'executes both branches and retains its genuine failure' {
        $actual = Get-CanonicalFixtureValue $true
        Get-CanonicalFixtureValue $false | Should -Be 'other'
        $actual | Should -Be 'intentionally wrong expectation'
    }
}
'@ | Set-Content -LiteralPath (Join-Path $owned 'Tests/Unit/Fixture.Tests.ps1')
        $output = Join-Path $owned 'results'
        $start = [Diagnostics.ProcessStartInfo]::new()
        $start.FileName = Join-Path $PSHOME $(if ($IsWindows) { 'pwsh.exe' } else { 'pwsh' })
        $start.UseShellExecute = $false
        $start.RedirectStandardOutput = $true
        $start.RedirectStandardError = $true
        foreach ($argument in @('-NoLogo', '-NoProfile', '-File', (Join-Path $owned 'Tests/.pester.ps1'), '-Path', 'Tests/Unit/Fixture.Tests.ps1', '-Coverage', '-CI', '-OutputDirectory', $output)) {
            $start.ArgumentList.Add($argument)
        }
        $process = [Diagnostics.Process]::new()
        $process.StartInfo = $start
        try {
            [void]$process.Start()
            $stdout = $process.StandardOutput.ReadToEndAsync()
            $stderr = $process.StandardError.ReadToEndAsync()
            if (-not $process.WaitForExit(60000)) {
                $process.Kill($true)
                $process.WaitForExit()
                throw 'Owned canonical coverage child exceeded its 60-second bound.'
            }
            $process.ExitCode | Should -Be 1 -Because ($stdout.GetAwaiter().GetResult() + $stderr.GetAwaiter().GetResult())
            $summary = Get-Content -LiteralPath (Join-Path $output 'pester-run-summary.json') -Raw | ConvertFrom-Json
            $summary.Total | Should -Be 1
            $summary.Passed | Should -Be 0
            $summary.Failed | Should -Be 1
            $summary.FailedBlocks | Should -Be 0
            $summary.FailedContainers | Should -Be 0
            $summary.CoverageTarget | Should -Be 85
            $summary.ParentAnalyzed | Should -BeGreaterThan 0
            $summary.ParentExecuted | Should -Be $summary.ParentAnalyzed
            $packet = Get-Content -LiteralPath (Join-Path $output 'source-bound-coverage.json') -Raw | ConvertFrom-Json -Depth 20
            $packet.Parent.Executed | Should -Be $summary.ParentExecuted
            $packet.Parent.Analyzed | Should -Be $summary.ParentAnalyzed
        } finally { $process.Dispose() }
    }

    It 'returns a native failure below target and success for fully executed branches' {
        $fixture = Join-Path $TestDrive 'Fixture.ps1'
        $test = Join-Path $TestDrive 'Fixture.Tests.ps1'
        $probe = Join-Path $TestDrive 'probe.ps1'
        @'
function Invoke-CoverageFixture {
    param([bool]$Positive)
    if ($Positive) { return 'positive' }
    $first = 'first'
    $second = 'second'
    $third = 'third'
    $fourth = 'fourth'
    return "$first $second $third $fourth"
}
'@ | Set-Content -LiteralPath $fixture
        @'
BeforeAll { . "$PSScriptRoot/Fixture.ps1" }
Describe 'Actual measured fixture' {
    It 'executes the positive branch' { Invoke-CoverageFixture $true | Should -Be 'positive' }
    It 'executes the remaining branches when requested' -Skip:($env:PCAI_COVERAGE_FIXTURE_MODE -ne 'Full') {
        Invoke-CoverageFixture $false | Should -Be 'first second third fourth'
    }
}
'@ | Set-Content -LiteralPath $test
        @'
param($PesterManifest, $GatePath, $Mode)
$ErrorActionPreference = 'Stop'
Import-Module $PesterManifest -Force
. $GatePath
$env:PCAI_COVERAGE_FIXTURE_MODE = $Mode
$config = New-PesterConfiguration
$config.Run.Path = "$PSScriptRoot/Fixture.Tests.ps1"
$config.Run.PassThru = $true
$config.Run.Exit = $false
$config.CodeCoverage.Enabled = $true
$config.CodeCoverage.Path = "$PSScriptRoot/Fixture.ps1"
$config.CodeCoverage.OutputPath = "$PSScriptRoot/coverage-$Mode.xml"
$config.CodeCoverage.CoveragePercentTarget = 85
$result = Invoke-Pester -Configuration $config
[ordered]@{ Passed=$result.PassedCount; Failed=$result.FailedCount; Analyzed=$result.CodeCoverage.CommandsAnalyzedCount; Executed=$result.CodeCoverage.CommandsExecutedCount; Percent=$result.CodeCoverage.CoveragePercent } |
    ConvertTo-Json | Set-Content -LiteralPath "$PSScriptRoot/result-$Mode.json"
try { Assert-PesterCoverageGate -Result $result -Target 85 }
catch { Write-Error $_ -ErrorAction Continue; exit 1 }
exit 0
'@ | Set-Content -LiteralPath $probe
        $pwsh = (Get-Command pwsh -ErrorAction Stop).Source
        $output = & $pwsh -NoLogo -NoProfile -File $probe $script:PesterManifest $script:GatePath Partial 2>&1
        $LASTEXITCODE | Should -Be 1
        ($output -join "`n") | Should -Match 'below required 85'
        $partial = Get-Content -LiteralPath (Join-Path $TestDrive 'result-Partial.json') -Raw | ConvertFrom-Json
        $partial.Passed | Should -Be 1
        $partial.Failed | Should -Be 0
        $partial.Percent | Should -BeLessThan 85
        $output = & $pwsh -NoLogo -NoProfile -File $probe $script:PesterManifest $script:GatePath Full 2>&1
        if ($LASTEXITCODE) { $output | ForEach-Object { Write-Host $_ } }
        $LASTEXITCODE | Should -Be 0
        $full = Get-Content -LiteralPath (Join-Path $TestDrive 'result-Full.json') -Raw | ConvertFrom-Json
        $full.Passed | Should -Be 2
        $full.Failed | Should -Be 0
        $full.Percent | Should -Be 100
    }

    It 'rejects an actual teardown block error despite passing tests and full coverage' {
        $fixture = Join-Path $TestDrive 'BlockFixture.ps1'
        $test = Join-Path $TestDrive 'BlockFixture.Tests.ps1'
        $probe = Join-Path $TestDrive 'block-probe.ps1'
        [IO.File]::WriteAllText($fixture, "function Invoke-BlockFixture { return 'covered' }")
        @'
BeforeAll { . "$PSScriptRoot/BlockFixture.ps1" }
Describe 'Actual failed teardown' {
    It 'executes the covered command' { Invoke-BlockFixture | Should -BeExactly 'covered' }
    AfterAll { throw 'Synthetic teardown refusal' }
}
'@ | Set-Content -LiteralPath $test
        @'
param($PesterManifest, $GatePath)
$ErrorActionPreference = 'Stop'
Import-Module $PesterManifest -Force
. $GatePath
$configuration = New-PesterConfiguration
$configuration.Run.Path = "$PSScriptRoot/BlockFixture.Tests.ps1"
$configuration.Run.PassThru = $true
$configuration.Run.Exit = $false
$configuration.CodeCoverage.Enabled = $true
$configuration.CodeCoverage.Path = "$PSScriptRoot/BlockFixture.ps1"
$configuration.CodeCoverage.OutputPath = "$PSScriptRoot/block-coverage.xml"
$configuration.CodeCoverage.CoveragePercentTarget = 85
$result = Invoke-Pester -Configuration $configuration
[ordered]@{ Total=$result.TotalCount; Passed=$result.PassedCount; Failed=$result.FailedCount; FailedContainers=$result.FailedContainersCount; FailedBlocks=$result.FailedBlocksCount; Percent=$result.CodeCoverage.CoveragePercent } |
    ConvertTo-Json | Set-Content -LiteralPath "$PSScriptRoot/block-result.json"
try { Assert-PesterCoverageGate -Result $result -Target 85 }
catch { Write-Error $_ -ErrorAction Continue; exit 1 }
exit 0
'@ | Set-Content -LiteralPath $probe
        $pwsh = (Get-Command pwsh -CommandType Application -ErrorAction Stop | Select-Object -First 1).Source
        $output = & $pwsh -NoLogo -NoProfile -File $probe $script:PesterManifest $script:GatePath 2>&1
        $LASTEXITCODE | Should -Be 1
        ($output -join "`n") | Should -Match 'FailedBlocksCount=1'
        $actual = Get-Content -LiteralPath (Join-Path $TestDrive 'block-result.json') -Raw | ConvertFrom-Json
        $actual.Total | Should -Be 1
        $actual.Passed | Should -Be 1
        $actual.Failed | Should -Be 0
        $actual.FailedContainers | Should -Be 0
        $actual.FailedBlocks | Should -Be 1
        $actual.Percent | Should -Be 100
    }

    It 'selects every nested module script through actual Pester path resolution' {
        $repo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
        $configuration = Import-PowerShellDataFile -LiteralPath (Join-Path $repo 'Tests/PesterConfiguration.psd1')
        $paths = foreach ($relative in $configuration.CodeCoverage.Path) {
            (Resolve-Path -Path (Join-Path $repo $relative) -ErrorAction Stop).ProviderPath
        }
        $actual = @(& (Get-Module Pester) {
                param($Paths, $Recurse)
                Get-CodeCoverageFilePaths -Paths $Paths -IncludeTests $false -RecursePaths $Recurse
            } $paths $configuration.CodeCoverage.RecursePaths)
        $expected = @(Get-ChildItem -LiteralPath (Join-Path $repo 'Modules') -Recurse -File |
                Where-Object { $_.Extension -in @('.ps1', '.psm1') -and $_.Name -notlike '*.Tests.ps1' } |
                Select-Object -ExpandProperty FullName)
        $actual.Count | Should -Be $expected.Count
        @(Compare-Object -ReferenceObject $expected -DifferenceObject $actual).Count | Should -Be 0
        $actual | Should -Contain (Join-Path $repo 'Modules/PC-AI.Acceleration/Private/Initialize-PcaiNative.ps1')
        $actual | Should -Contain (Join-Path $repo 'Modules/PcaiInference.psm1')
    }
}
