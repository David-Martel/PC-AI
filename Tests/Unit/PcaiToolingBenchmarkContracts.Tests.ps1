#Requires -Version 7.0
param(
    [string]$BenchmarkSourcePath = (Join-Path $PSScriptRoot '../Benchmarks/Invoke-PcaiToolingBenchmarks.ps1'),
    [string]$RepoRoot = (Join-Path $PSScriptRoot '../..')
)

Describe 'Tooling benchmark provenance and useful output admission' -Tag 'Unit','Benchmarks','Acceleration','Portable' {
    BeforeAll {
        function Get-FixtureAst {
            param([string]$Path)
            $tokens=$null;$errors=$null
            $ast=[Management.Automation.Language.Parser]::ParseFile($Path,[ref]$tokens,[ref]$errors)
            if($errors.Count){throw "Source did not parse: $Path"}
            $ast
        }
        $script:BenchmarkAst=Get-FixtureAst $BenchmarkSourcePath
        foreach($name in @('Get-ConfigValue','Get-ReferenceTokenEstimate','Assert-TokenEstimateParity','Test-BenchmarkCaseEquivalence','Get-PowerShellDirectoryManifest','Test-PcaiBenchmarkNativeReady','Assert-PcaiManifestBenchmarkStats',
                'Get-ToolingCaseBenchmarks','Get-PcaiBenchmarkLoadedIdentity','Invoke-ValidatedBackendBenchmark','Add-BenchmarkMeasurement','Invoke-AccelerationModuleCommand','Invoke-ImportedModuleCommand')){
            $definition=$script:BenchmarkAst.Find({param($n)$n -is [Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq $name},$false)
            if(-not $definition){throw "Missing maintained benchmark definition: $name"}
            . ([scriptblock]::Create($definition.Extent.Text))
        }
        $helperAst=Get-FixtureAst (Join-Path $RepoRoot 'Modules/PC-AI.Acceleration/Private/Initialize-PcaiNative.ps1')
        $tokenBody=@(foreach($name in 'Invoke-PcaiNativeEstimateTokens','Get-PcaiTokenEstimate'){
            $definition=$helperAst.Find({param($n)$n -is [Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq $name},$false)
            if(-not $definition){throw "Missing maintained token helper: $name"}
            $definition.Extent.Text
        })
        # Preserve the real fallback body. Only its native transport is synthetic.
        $script:TokenBody=($tokenBody -join "`n").Replace('function Get-PcaiTokenEstimate {','function Invoke-FixtureActualTokenEstimate {')
        $reportAssignment=$script:BenchmarkAst.Find({param($n)$n -is [Management.Automation.Language.AssignmentStatementAst] -and $n.Left.Extent.Text -ceq '$benchmarkRows'},$false)
        if(-not $reportAssignment){throw 'Actual report-row producer is absent.'}
        $script:ReportRowsSource=$reportAssignment.Extent.Text
        function Invoke-BackendBenchmark {
            param($CaseId,$Backend,$Command,$Iterations,$Warmup)
            # Keep the maintained measurer's warmup catch boundary. The actual
            # validated producer must retain failures instead of losing them.
            for($index=0;$index -lt $Warmup;$index++){try{$null=& $Command}catch{}}
            for($index=0;$index -lt $Iterations;$index++){$null=& $Command}
            # These are algebraic input controls for the report gate, never measured
            # timings or a performance claim. No stopwatch/native process runs here.
            [pscustomobject]@{CaseId=$CaseId;Backend=$Backend;MeanMs=3;MedianMs=3;StdDevMs=0;MinMs=3;MaxMs=3
                Iterations=$Iterations;Tool='synthetic-clock-boundary';WorkingSetDeltaMeanBytes=$null;WorkingSetDeltaMaxBytes=$null
                PrivateMemoryDeltaMeanBytes=$null;PrivateMemoryDeltaMaxBytes=$null;ManagedMemoryDeltaMeanBytes=$null;ManagedMemoryDeltaMaxBytes=$null
                ManagedAllocatedMeanBytes=$null;ManagedAllocatedMaxBytes=$null}
        }
        function Invoke-FixtureCase {
            param([string]$Id,[string]$Sample='hello world',[uint32]$Depth=0,[uint64]$Limit=0,[int]$Iterations=1,[int]$Warmup=0)
            Get-ToolingCaseBenchmarks -Case ([pscustomobject]@{id=$Id;name='contract';category='contract'}) -Defaults ([pscustomobject]@{iterations=$Iterations;warmup=$Warmup;maxDepth=$Depth;maxResults=$Limit}) -RepoRoot $script:DataRoot -Capabilities ([pscustomobject]@{BackendCoverage=@([pscustomobject]@{Operation='TokenEstimate';CoverageState='PSOnly'},[pscustomobject]@{Operation='DirectoryManifest';CoverageState='PSOnly'})}) -TokenSample $Sample
        }
        function Get-FixtureReportRows {
            param([object]$Result)
            $caseResults=@($Result)
            . ([scriptblock]::Create($script:ReportRowsSource))
            $benchmarkRows
        }
        $script:CommonModule=$null;$script:CliModule=$null
        $script:FixtureRevision=0
    }
    BeforeEach {
        $script:FixtureRevision++
        $script:DataRoot=Join-Path $TestDrive ('owned-tree-r'+$script:FixtureRevision)
        if(Test-Path -LiteralPath $script:DataRoot){throw 'Retain prior owned fixture tree.'}
        $null=[IO.Directory]::CreateDirectory($script:DataRoot)
        [IO.File]::WriteAllBytes((Join-Path $script:DataRoot 'first.txt'),[byte[]](1,2,3))
        $script:ExpectedStats=Get-PowerShellDirectoryManifest -Path $script:DataRoot
        # Inert identity bytes live outside the enumerated corpus. These prove
        # orchestration checks only; no native DLL is loaded by this fixture.
        $identityRoot=Join-Path $TestDrive ('native-identity-r'+$script:FixtureRevision)
        $null=[IO.Directory]::CreateDirectory($identityRoot)
        $coreIdentity=Join-Path $identityRoot 'core.fixture'
        $bridgeIdentity=Join-Path $identityRoot 'bridge.fixture'
        [IO.File]::WriteAllText($coreIdentity,'inert core binding bytes')
        [IO.File]::WriteAllText($bridgeIdentity,'inert bridge binding bytes')
        $moduleBody=$script:TokenBody+@'
function Test-PcaiNativeAvailable {
    $script:ReadyProbeCount++
    if($script:LoseLoadedOnProbe -gt 0 -and $script:ReadyProbeCount -ge $script:LoseLoadedOnProbe){return $false}
    return $script:Loaded
}
function Get-PcaiNativeStatus { [pscustomobject]@{CoreAvailable=$script:Available} }
function Get-PcaiNativeLoadedCorePath {$script:FixtureCoreIdentity}
function Get-PcaiNativeLoadedBridgePath {$script:FixtureBridgeIdentity}
function Get-PcaiTokenEstimate {
    param([AllowEmptyString()][string]$Text)
    $script:TokenCalls++
    if($script:CorruptTokenBinding){[IO.File]::AppendAllText($script:FixtureCoreIdentity,'changed')}
    if($script:InvalidTokenCall -eq $script:TokenCalls){return [uint64]0}
    if($script:LoseCoreAfterToken){$script:Available=$script:CoreAfterValue}
    if($script:SyntheticToken){return $script:TokenOutcome}
    Invoke-FixtureActualTokenEstimate -Text $Text
}
function Invoke-PcaiNativeDirectoryManifest {
    param($Path,$MaxDepth,$MaxResults,[switch]$StatsOnly)
    $script:ManifestCalls++
    if($script:LoseCoreAfterManifest){$script:Available=$script:CoreAfterValue}
    $script:ManifestOutcome
}
Export-ModuleMember -Function Test-PcaiNativeAvailable,Get-PcaiNativeStatus,Get-PcaiTokenEstimate,Invoke-PcaiNativeDirectoryManifest
'@
        $script:AccelerationModule=New-Module -ScriptBlock ([scriptblock]::Create($moduleBody))
        & $script:AccelerationModule {param($stats,$coreIdentity,$bridgeIdentity)
            $script:FixtureCoreIdentity=$coreIdentity;$script:FixtureBridgeIdentity=$bridgeIdentity
            $script:Loaded=$false;$script:Available=$false;$script:SyntheticToken=$false;$script:TokenOutcome=$null
            $script:TokenCalls=0;$script:ManifestCalls=0;$script:ManifestOutcome=$stats
            $script:CorruptTokenBinding=$false;$script:InvalidTokenCall=0
            $script:ReadyProbeCount=0;$script:LoseLoadedOnProbe=0
            $script:LoseCoreAfterToken=$false;$script:LoseCoreAfterManifest=$false
            $script:CoreAfterValue=$false
        } ([pscustomobject]@{Status='Success';EntriesReturned=$script:ExpectedStats.EntriesReturned;FileCount=$script:ExpectedStats.FileCount;DirectoryCount=$script:ExpectedStats.DirectoryCount;TotalSize=$script:ExpectedStats.TotalSize}) $coreIdentity $bridgeIdentity
    }
    AfterEach {if($script:AccelerationModule){Remove-Module $script:AccelerationModule -ErrorAction Stop}}

    It 'publishes only the scalar-reference baseline when the native core is absent' {
        $result=Invoke-FixtureCase token-estimate
        @($result.Results.Backend)|Should -Be @('powershell')
        (& $script:AccelerationModule {$script:TokenCalls})|Should -Be 0
        $result.UnavailableReason|Should -Match 'unavailable'
        $result.Qualification.ComparisonQualified|Should -BeFalse
    }
    It 'preserves the real nonempty fallback algorithm while admitting no speedup comparison' {
        (& $script:AccelerationModule {Get-PcaiTokenEstimate -Text 'hello world'})|Should -Be 4
        ([regex]::Matches('hello world','\w+')).Count|Should -Be 2
        $result=Invoke-FixtureCase token-estimate
        $result.Qualification.UsefulOutputParity|Should -Be 'not-established'
        foreach($row in @(Get-FixtureReportRows $result)){
            $row.SpeedupVsBaseline|Should -BeNullOrEmpty
        }
    }
    It 'admits matching scalar work through a synthetic native transport without claiming measured speedup' {
        & $script:AccelerationModule {$script:Loaded=$true;$script:Available=$true;$script:SyntheticToken=$true;$script:TokenOutcome=[uint64]4}
        $result=Invoke-FixtureCase token-estimate
        @($result.Results.Backend)|Should -Be @('native','powershell')
        (& $script:AccelerationModule {$script:TokenCalls})|Should -Be 2
        $result.EquivalenceVerified | Should -BeTrue
        $rows=@(Get-FixtureReportRows $result)
        $rows.Count|Should -Be 2
        @($rows|Where-Object{$null -ne $_.SpeedupVsBaseline}).Count|Should -Be 0
        @($rows|Where-Object ComparisonQualified).Count|Should -Be 2
    }
    It 'rejects invalid token useful output <Mode>' -TestCases @(@{Mode='null'},@{Mode='negative'},@{Mode='zero-nonempty'},@{Mode='bool'},@{Mode='string'},@{Mode='positive-empty'}) {
        param($Mode)
        & $script:AccelerationModule {param($mode)
            $script:Loaded=$true;$script:Available=$true;$script:SyntheticToken=$true
            $script:TokenOutcome=switch($mode){'null'{$null};'negative'{-1};'zero-nonempty'{0};'bool'{$true};'string'{'4'};'positive-empty'{1}}
        } $Mode
        $sample=if($Mode -eq 'positive-empty'){''}else{'hello world'}
        {Invoke-FixtureCase token-estimate -Sample $sample}|Should -Throw '*Token native useful output*'
    }
    It 'rejects loaded-bridge absence before calling a claimed native token transport' {
        & $script:AccelerationModule {$script:Available=$true;$script:Loaded=$true;$script:SyntheticToken=$true;$script:TokenOutcome=4;$script:LoseLoadedOnProbe=2}
        {Invoke-FixtureCase token-estimate}|Should -Throw '*native backend became unavailable*'
        (& $script:AccelerationModule {$script:TokenCalls})|Should -Be 0
    }
    It 'rejects manifest transport failures <Mode>' -TestCases @(@{Mode='failed-status'},@{Mode='null'},@{Mode='wrong-bytes'},@{Mode='negative-count'},@{Mode='string-count'},@{Mode='inconsistent-count'}) {
        param($Mode)
        & $script:AccelerationModule {param($mode)
            $script:Available=$true;$script:Loaded=$true
            switch($mode){
                'failed-status'{$script:ManifestOutcome.Status='NotImplemented'}
                'null'{$script:ManifestOutcome=$null}
                'wrong-bytes'{$script:ManifestOutcome.TotalSize=999}
                'negative-count'{$script:ManifestOutcome.FileCount=-1}
                'string-count'{$script:ManifestOutcome.FileCount='1'}
                'inconsistent-count'{$script:ManifestOutcome.EntriesReturned=2}
            }
        } $Mode
        {Invoke-FixtureCase directory-manifest}|Should -Throw '*Manifest native*'
    }
    It 'admits uncapped matching aggregate counts and bytes without claiming entry identity parity' {
        & $script:AccelerationModule {$script:Available=$true;$script:Loaded=$true}
        $result=Invoke-FixtureCase directory-manifest
        $result.Qualification.ComparisonQualified|Should -BeTrue
        $result.Qualification.UsefulOutputParity|Should -Be 'matched-aggregate-stats'
        $result.Qualification.OutputContract|Should -Match 'entry identities are not measured'
        (& $script:AccelerationModule {$script:ManifestCalls})|Should -Be 1
        $script:ExpectedStats.TotalSize|Should -Be 3
    }
    It 'leaves capped traversal subsets explicitly incomparable despite plausible successful statistics' {
        & $script:AccelerationModule {$script:Available=$true;$script:Loaded=$true;$script:ManifestOutcome.TotalSize=999}
        $result=Invoke-FixtureCase directory-manifest -Limit 1
        $result.Qualification.ComparisonQualified|Should -BeFalse
        $result.Qualification.UsefulOutputParity|Should -Be 'not-established'
        foreach($row in @(Get-FixtureReportRows $result)){
            $row.SpeedupVsBaseline|Should -BeNullOrEmpty
        }
    }
    # Capped traversal can select a different subset, but an empty or incomplete
    # result must not become a fast timing for work the real fixture requires.
    It 'rejects a capped native sample that omits eligible entries (<Limit>/<Returned>)' -Tag 'UtilityRegression' -TestCases @(
        @{Limit=10;Returned=0},@{Limit=10;Returned=1},@{Limit=2;Returned=0},@{Limit=2;Returned=1}
    ) {
        param([uint64]$Limit,[uint64]$Returned)
        [IO.File]::WriteAllBytes((Join-Path $script:DataRoot 'second.txt'),[byte[]](4,5))
        [IO.File]::WriteAllBytes((Join-Path $script:DataRoot 'third.txt'),[byte[]](6))
        $reference=Get-PowerShellDirectoryManifest -Path $script:DataRoot -MaxResults $Limit
        $reference.EntriesReturned|Should -Be ([Math]::Min(3,$Limit))
        & $script:AccelerationModule {param($returned)
            $script:Available=$true;$script:Loaded=$true
            $script:ManifestOutcome.EntriesReturned=$returned
            $script:ManifestOutcome.FileCount=$returned
            $script:ManifestOutcome.DirectoryCount=[uint64]0
            $script:ManifestOutcome.TotalSize=[uint64]0
        } $Returned
        {Invoke-FixtureCase directory-manifest -Limit $Limit}|Should -Throw '*Manifest native useful output mismatch: EntriesReturned*'
    }
    It 'admits a complete capped count below the full corpus size without claiming subset parity' -Tag 'UtilityRegression' {
        [IO.File]::WriteAllBytes((Join-Path $script:DataRoot 'second.txt'),[byte[]](4,5))
        [IO.File]::WriteAllBytes((Join-Path $script:DataRoot 'third.txt'),[byte[]](6))
        (Get-PowerShellDirectoryManifest -Path $script:DataRoot -MaxResults 0).EntriesReturned|Should -Be 3
        (Get-PowerShellDirectoryManifest -Path $script:DataRoot -MaxResults 2).EntriesReturned|Should -Be 2
        & $script:AccelerationModule {
            $script:Available=$true;$script:Loaded=$true
            $script:ManifestOutcome.EntriesReturned=[uint64]2
            $script:ManifestOutcome.FileCount=[uint64]2
            $script:ManifestOutcome.DirectoryCount=[uint64]0
            $script:ManifestOutcome.TotalSize=[uint64]0
        }
        $result=Invoke-FixtureCase directory-manifest -Limit 2
        @($result.Results.Backend)|Should -Be @('native','powershell')
        $result.Qualification.ComparisonQualified|Should -BeFalse
        $result.Qualification.UsefulOutputParity|Should -Be 'not-established'
        foreach($row in @(Get-FixtureReportRows $result)){$row.SpeedupVsBaseline|Should -BeNullOrEmpty}
    }
    It 'accepts a zero capped count for an actual empty corpus without claiming subset parity' -Tag 'UtilityRegression' {
        $script:DataRoot=Join-Path $script:DataRoot 'empty'
        $null=[IO.Directory]::CreateDirectory($script:DataRoot)
        & $script:AccelerationModule {
            $script:Available=$true;$script:Loaded=$true
            foreach($name in 'EntriesReturned','FileCount','DirectoryCount','TotalSize'){$script:ManifestOutcome.$name=[uint64]0}
        }
        $result=Invoke-FixtureCase directory-manifest -Limit 10
        @($result.Results.Backend)|Should -Be @('native','powershell')
        $result.Qualification.ComparisonQualified|Should -BeFalse
    }
    It 'matches direct-child native depth semantics using actual private files' {
        $nested=Join-Path $script:DataRoot 'nested';$null=[IO.Directory]::CreateDirectory($nested)
        [IO.File]::WriteAllBytes((Join-Path $nested 'deep.txt'),[byte[]](4,5))
        $result=Get-PowerShellDirectoryManifest -Path $script:DataRoot -MaxDepth 1
        $result.EntriesReturned|Should -Be 2;$result.FileCount|Should -Be 1
        $result.DirectoryCount|Should -Be 1;$result.TotalSize|Should -Be 3
    }
    It 'excludes actual private dot files and descendants of dot directories' {
        [IO.File]::WriteAllBytes((Join-Path $script:DataRoot '.hidden.txt'),[byte[]](4,5))
        $hidden=Join-Path $script:DataRoot '.hidden';$null=[IO.Directory]::CreateDirectory($hidden)
        [IO.File]::WriteAllBytes((Join-Path $hidden 'hidden.txt'),[byte[]](4,5))
        $result=Get-PowerShellDirectoryManifest -Path $script:DataRoot
        $result.EntriesReturned|Should -Be 1;$result.TotalSize|Should -Be 3
    }
    It 'counts actual file and directory links separately at native depth <Depth>' -TestCases @(
        @{Depth=1;Entries=4;Files=1;Bytes=3}
        @{Depth=2;Entries=5;Files=2;Bytes=8}
    ) {
        param($Depth,$Entries,$Files,$Bytes)
        $nested=Join-Path $script:DataRoot 'nested'
        $null=[IO.Directory]::CreateDirectory($nested)
        [IO.File]::WriteAllBytes((Join-Path $nested 'deep.txt'),[byte[]](4,5,6,7,8))
        $null=[IO.File]::CreateSymbolicLink((Join-Path $script:DataRoot 'file-link'),(Join-Path $script:DataRoot 'first.txt'))
        $null=[IO.Directory]::CreateSymbolicLink((Join-Path $script:DataRoot 'directory-link'),$nested)
        $result=Get-PowerShellDirectoryManifest -Path $script:DataRoot -MaxDepth $Depth
        $result.EntriesReturned|Should -Be $Entries
        $result.FileCount|Should -Be $Files
        $result.DirectoryCount|Should -Be 1
        $result.OtherEntries|Should -Be 2
        $result.TotalSize|Should -Be $Bytes
        # Constants describe the real fixture, rather than copying the baseline.
        $stats=[pscustomobject]@{Status='Success';EntriesReturned=$Entries;FileCount=$Files;DirectoryCount=1;TotalSize=$Bytes}
        {Assert-PcaiManifestBenchmarkStats -Stats $stats -Expected $result}|Should -Not -Throw
    }
    It 'returns numeric zero statistics for an actual empty private directory' {
        Set-StrictMode -Version Latest
        $empty=Join-Path $TestDrive 'empty';$null=[IO.Directory]::CreateDirectory($empty)
        $result=Get-PowerShellDirectoryManifest -Path $empty
        $result.EntriesReturned|Should -Be 0;$result.FileCount|Should -Be 0
        $result.DirectoryCount|Should -Be 0;$result.TotalSize|Should -Be 0
        $result.TotalSize|Should -BeOfType ([uint64])
    }
    It 'does not report speedup for an existing case with no explicit parity admission' {
        $result=Invoke-FixtureCase token-estimate
        $result.PSObject.Properties.Remove('Qualification')
        $rows=@(Get-FixtureReportRows $result)
        $rows[0].ComparisonQualified|Should -BeFalse
        $rows[0].UsefulOutputParity|Should -Be 'not-established'
        $rows[0].SpeedupVsBaseline|Should -BeNullOrEmpty
        $rows[0].ActualBackend|Should -Match '^unverified:'
    }
    It 'publishes only actual PowerShell manifest statistics when native availability is absent' {
        $result=Invoke-FixtureCase directory-manifest
        @($result.Results.Backend)|Should -Be @('powershell')
        (& $script:AccelerationModule {$script:ManifestCalls})|Should -Be 0
        $result.UnavailableReason|Should -Match 'unavailable'
        $result.Qualification.ComparisonQualified|Should -BeFalse
        $rows=@(Get-FixtureReportRows $result)
        $rows[0].SpeedupVsBaseline|Should -BeNullOrEmpty
        $rows[0].ActualBackend|Should -Not -Be 'Rust+C#'
    }
    It 'rejects truthy non-Boolean <Flag>/<ValueKind> initial availability in <Case>' -TestCases @(
        foreach($case in 'token-estimate','directory-manifest'){
            foreach($flag in 'Loaded','Core'){
                foreach($kind in 'string','integer'){@{Case=$case;Flag=$flag;ValueKind=$kind}}
            }
        }
    ) {
        param($Case,$Flag,$ValueKind)
        & $script:AccelerationModule {param($flag,$kind)
            $script:Loaded=$true;$script:Available=$true
            $value=if($kind -eq 'string'){'true'}else{1}
            if($flag -eq 'Loaded'){$script:Loaded=$value}else{$script:Available=$value}
        } $Flag $ValueKind
        $result=Invoke-FixtureCase $Case
        @($result.Results.Backend)|Should -Be @('powershell')
        (& $script:AccelerationModule {$script:TokenCalls+$script:ManifestCalls})|Should -Be 0
        $result.Qualification.ComparisonQualified|Should -BeFalse
    }
    It 'refuses a manifest sample when loaded availability is lost immediately before transport' {
        & $script:AccelerationModule {$script:Loaded=$true;$script:Available=$true;$script:LoseLoadedOnProbe=2}
        {Invoke-FixtureCase directory-manifest}|Should -Throw '*native backend became unavailable*'
        (& $script:AccelerationModule {$script:ManifestCalls})|Should -Be 0
    }
    It 'refuses a matching manifest result when core availability becomes <LossKind> during its transport' -TestCases @(@{LossKind='false';LossValue=$false},@{LossKind='truthy-string';LossValue='true'},@{LossKind='truthy-integer';LossValue=1}) {
        param($LossKind,$LossValue)
        & $script:AccelerationModule {param($value)$script:Loaded=$true;$script:Available=$true;$script:LoseCoreAfterManifest=$true;$script:CoreAfterValue=$value} $LossValue
        {Invoke-FixtureCase directory-manifest}|Should -Throw '*native backend became unavailable*'
        (& $script:AccelerationModule {$script:ManifestCalls})|Should -Be 1
    }
    It 'refuses plausible token output when core availability becomes <LossKind> during its transport' -TestCases @(@{LossKind='false';LossValue=$false},@{LossKind='truthy-string';LossValue='true'},@{LossKind='truthy-integer';LossValue=1}) {
        param($LossKind,$LossValue)
        & $script:AccelerationModule {param($value)$script:Loaded=$true;$script:Available=$true;$script:SyntheticToken=$true;$script:TokenOutcome=[uint64]4;$script:LoseCoreAfterToken=$true;$script:CoreAfterValue=$value} $LossValue
        {Invoke-FixtureCase token-estimate}|Should -Throw '*native backend became unavailable*'
        (& $script:AccelerationModule {$script:TokenCalls})|Should -Be 1
    }

    It 'retains warmup and measured phases with exact body, selection and validation calls' {
        $timingControl=[pscustomobject]@{Bodies=0;Selections=0;Validations=0}
        $result=Invoke-ValidatedBackendBenchmark -CaseId control -Backend inert -Iterations 2 -Warmup 1 -Command {$timingControl.Bodies++;[byte]7} -BeforeSample {$timingControl.Selections++} -AfterSample {$timingControl.Selections++} -ValidateOutput {
            param($value)
            $timingControl.Validations++
            if($value -isnot [byte] -or $value -ne 7){throw 'Incorrect useful control output'}
        }
        $timingControl.Bodies|Should -Be 3
        $timingControl.Selections|Should -Be 6
        $timingControl.Validations|Should -Be 3
        @($result.PhaseSamples|Where-Object Kind -eq 'warmup').Count|Should -Be 1
        @($result.PhaseSamples|Where-Object Kind -eq 'measured').Count|Should -Be 2
        $result.GuardInclusiveMeanMs|Should -Be 3
        $result.Tool|Should -BeExactly 'validated-body-stopwatch'
        $result.MemoryScope|Should -Match 'Process-wide.*observer.*not exclusive'
        $result.CpuScope|Should -Match 'not exclusive'
        foreach($sample in $result.PhaseSamples){
            $sample.BodyMs|Should -BeGreaterOrEqual 0
            $sample.SelectionMs|Should -BeGreaterOrEqual 0
            $sample.ValidationMs|Should -BeGreaterOrEqual 0
        }
    }
    It 'rejects invalid configured sample counts iterations=<Iterations> warmup=<Warmup>' -ForEach @(
        @{Iterations=0;Warmup=0},@{Iterations=-1;Warmup=0},@{Iterations=1;Warmup=-1}
    ) {
        {Invoke-ValidatedBackendBenchmark -CaseId control -Backend inert -Iterations $Iterations -Warmup $Warmup -Command {throw 'Must not execute'} -ValidateOutput {throw 'Must not execute'}}|Should -Throw '*Invalid validated benchmark sample counts*'
    }
    It 'preserves actual useful body errors instead of returning timing success' {
        {Invoke-ValidatedBackendBenchmark -CaseId control -Backend inert -Iterations 1 -Warmup 0 -Command {throw 'owned-body-error'} -ValidateOutput {throw 'Must not execute'}}|Should -Throw '*owned-body-error*'
    }
    It 'refuses a native timing with no proven loaded core path' {
        & $script:AccelerationModule {$script:Loaded=$true;$script:Available=$true;$script:FixtureCoreIdentity=$null}
        {Invoke-FixtureCase token-estimate}|Should -Throw '*loaded identity is unproven*'
        (& $script:AccelerationModule {$script:TokenCalls})|Should -Be 0
    }
    It 'rejects changed loaded file bytes after an actual token body' {
        & $script:AccelerationModule {$script:Loaded=$true;$script:Available=$true;$script:SyntheticToken=$true;$script:TokenOutcome=[uint64]4;$script:CorruptTokenBinding=$true}
        {Invoke-FixtureCase token-estimate}|Should -Throw '*loaded artifact identity changed*'
        (& $script:AccelerationModule {$script:TokenCalls})|Should -Be 1
    }
    It 'retains invalid native warmup output even when the ordinary measurer catches it' {
        & $script:AccelerationModule {$script:Loaded=$true;$script:Available=$true;$script:SyntheticToken=$true;$script:TokenOutcome=[uint64]4;$script:InvalidTokenCall=2}
        {Invoke-FixtureCase token-estimate -Iterations 1 -Warmup 1}|Should -Throw '*Validated benchmark sample failed*'
        # Preflight succeeds, warmup fails its oracle, and the measured body
        # succeeds. Discarding only the warmup must still fail the whole case.
        (& $script:AccelerationModule {$script:TokenCalls})|Should -Be 3
    }
    It 'preserves memory and useful-clock scope in actual exported report rows' {
        $rows=@(Get-FixtureReportRows (Invoke-FixtureCase token-estimate))
        $rows.Count|Should -Be 1
        $rows[0].TimingScope|Should -Match 'Public useful command only'
        $rows[0].MemoryScope|Should -Match 'not exclusive useful-body allocation or peak memory'
        $rows[0].GuardInclusiveMeanMs|Should -Be 3
        @($rows[0].PhaseSamples).Count|Should -Be 1
    }
}
