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
        foreach($name in @('Get-ConfigValue','Get-PowerShellDirectoryManifest','Test-PcaiBenchmarkNativeReady','Assert-PcaiManifestBenchmarkStats',
                'Get-ToolingCaseBenchmarks','Add-BenchmarkMeasurement','Invoke-AccelerationModuleCommand','Invoke-ImportedModuleCommand')){
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
            & $Command | Out-Null
            # These are algebraic input controls for the report gate, never measured
            # timings or a performance claim. No stopwatch/native process runs here.
            [pscustomobject]@{CaseId=$CaseId;Backend=$Backend;MeanMs=3;MedianMs=3;StdDevMs=0;MinMs=3;MaxMs=3
                Iterations=$Iterations;Tool='synthetic-clock-boundary';WorkingSetDeltaMeanBytes=$null;WorkingSetDeltaMaxBytes=$null
                PrivateMemoryDeltaMeanBytes=$null;PrivateMemoryDeltaMaxBytes=$null;ManagedMemoryDeltaMeanBytes=$null;ManagedMemoryDeltaMaxBytes=$null
                ManagedAllocatedMeanBytes=$null;ManagedAllocatedMaxBytes=$null}
        }
        function Invoke-FixtureCase {
            param([string]$Id,[string]$Sample='hello world',[uint32]$Depth=0,[uint64]$Limit=0)
            Get-ToolingCaseBenchmarks -Case ([pscustomobject]@{id=$Id;name='contract';category='contract'}) -Defaults ([pscustomobject]@{iterations=1;warmup=0;maxDepth=$Depth;maxResults=$Limit}) -RepoRoot $script:DataRoot -Capabilities ([pscustomobject]@{BackendCoverage=@([pscustomobject]@{Operation='TokenEstimate';CoverageState='PSOnly'},[pscustomobject]@{Operation='DirectoryManifest';CoverageState='PSOnly'})}) -TokenSample $Sample
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
        $moduleBody=$script:TokenBody+@'
function Test-PcaiNativeAvailable {
    $script:ReadyProbeCount++
    if($script:LoseLoadedOnProbe -gt 0 -and $script:ReadyProbeCount -ge $script:LoseLoadedOnProbe){return $false}
    return $script:Loaded
}
function Get-PcaiNativeStatus { [pscustomobject]@{CoreAvailable=$script:Available} }
function Get-PcaiTokenEstimate {
    param([AllowEmptyString()][string]$Text)
    $script:TokenCalls++
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
        & $script:AccelerationModule {param($stats)
            $script:Loaded=$false;$script:Available=$false;$script:SyntheticToken=$false;$script:TokenOutcome=$null
            $script:TokenCalls=0;$script:ManifestCalls=0;$script:ManifestOutcome=$stats
            $script:ReadyProbeCount=0;$script:LoseLoadedOnProbe=0
            $script:LoseCoreAfterToken=$false;$script:LoseCoreAfterManifest=$false
            $script:CoreAfterValue=$false
        } ([pscustomobject]@{Status='Success';EntriesReturned=$script:ExpectedStats.EntriesReturned;FileCount=$script:ExpectedStats.FileCount;DirectoryCount=$script:ExpectedStats.DirectoryCount;TotalSize=$script:ExpectedStats.TotalSize})
    }
    AfterEach {if($script:AccelerationModule){Remove-Module $script:AccelerationModule -ErrorAction Stop}}

    It 'publishes only the word-count baseline when the native core is absent' {
        $result=Invoke-FixtureCase token-estimate
        @($result.Results.Backend)|Should -Be @('powershell')
        (& $script:AccelerationModule {$script:TokenCalls})|Should -Be 0
        $result.UnavailableReason|Should -Match 'unavailable'
        $result.Qualification.ComparisonQualified|Should -BeFalse
    }
    It 'preserves the real nonempty fallback algorithm while admitting no speedup comparison' {
        (& $script:AccelerationModule {Get-PcaiTokenEstimate -Text 'hello world'})|Should -Be 3
        ([regex]::Matches('hello world','\w+')).Count|Should -Be 2
        $result=Invoke-FixtureCase token-estimate
        $result.Qualification.UsefulOutputParity|Should -Be 'not-comparable'
        foreach($row in @(Get-FixtureReportRows $result)){
            $row.SpeedupVsBaseline|Should -BeNullOrEmpty
        }
    }
    It 'runs a synthetic native transport control but refuses a cross-algorithm speedup claim' {
        & $script:AccelerationModule {$script:Loaded=$true;$script:Available=$true;$script:SyntheticToken=$true;$script:TokenOutcome=[uint64]4}
        $result=Invoke-FixtureCase token-estimate
        @($result.Results.Backend)|Should -Be @('native','powershell')
        (& $script:AccelerationModule {$script:TokenCalls})|Should -Be 1
        $rows=@(Get-FixtureReportRows $result)
        $rows.Count|Should -Be 2
        @($rows|Where-Object{$null -ne $_.SpeedupVsBaseline}).Count|Should -Be 0
        @($rows|Where-Object ComparisonQualified).Count|Should -Be 0
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
    It 'returns numeric zero statistics for an actual empty private directory' {
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
}
