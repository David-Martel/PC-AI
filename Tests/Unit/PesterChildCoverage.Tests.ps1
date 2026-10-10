BeforeAll {
    $script:RepositoryRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
    $script:MergeHelper=Join-Path $script:RepositoryRoot 'Tools/Merge-PesterChildCoverage.ps1'
    . $script:MergeHelper
    $script:PesterManifest=Join-Path (Get-Module Pester).ModuleBase 'Pester.psd1'
}

Describe 'Source-bound child coverage command union' -Tag 'Unit','Coverage','Portable' {
    BeforeAll {
        $script:Source=Join-Path $TestDrive 'measured.ps1'
        [IO.File]::WriteAllText($script:Source,"function Get-MeasuredValue {`n  return 'actual'`n}`n")
        $script:Binding=[pscustomobject]@{Path=$script:Source;SHA=(Get-FileHash -LiteralPath $script:Source).Hash}
        function New-CoverageCommand([int]$Column,[bool]$Executed) {
            [pscustomobject]@{File=$script:Source;StartLine=2;StartColumn=$Column;EndLine=2;EndColumn=$Column+4;Command="command-$Column";Executed=$Executed}
        }
        function New-ChildSnapshot([bool]$First,[bool]$Second) {
            [pscustomobject]@{SchemaVersion=1;Sources=@([pscustomobject]@{Path=$script:Source;SHA256=$script:Binding.SHA});Commands=@((New-CoverageCommand 3 $First),(New-CoverageCommand 12 $Second));Analyzed=2;Executed=[int]$First+[int]$Second}
        }
        function New-ParentResult([bool]$First,[bool]$Second,$Child) {
            $commands=@((New-CoverageCommand 3 $First),(New-CoverageCommand 12 $Second))
            $test=[pscustomobject]@{Result='Passed';PcaiChildCoverage=[pscustomobject]@{ParentSources=@($script:Binding);Child=$Child}}
            [pscustomobject]@{Tests=@($test);CodeCoverage=[pscustomobject]@{CommandsAnalyzedCount=2;CommandsExecutedCount=[int]$First+[int]$Second;CommandsExecuted=@($commands|Where-Object Executed);CommandsMissed=@($commands|Where-Object{ -not $_.Executed });CoveragePercent=50.0*([int]$First+[int]$Second)}}
        }
    }
    It 'adds a child-only command while retaining the exact parent denominator' {
        $result=New-ParentResult $true $false (New-ChildSnapshot $false $true)
        $merged=Merge-PesterChildCoverage $result
        $merged.CommandsAnalyzedCount|Should -Be 2
        $merged.CommandsExecutedCount|Should -Be 2
        $merged.AddedChildCommands|Should -Be 1
        $result.CodeCoverage.CommandsExecutedCount|Should -Be 1
    }
    It 'does not inflate commands from overlapping hits or repeated child receipts' {
        $result=New-ParentResult $true $false (New-ChildSnapshot $true $false)
        $result.Tests+= $result.Tests[0]
        $merged=Merge-PesterChildCoverage $result
        $merged.CommandsExecutedCount|Should -Be 1
        $merged.CoveragePercent|Should -Be 50
        $merged.AddedChildCommands|Should -Be 0
    }
    It 'keeps zero parent and child executions at zero' {
        $merged=Merge-PesterChildCoverage (New-ParentResult $false $false (New-ChildSnapshot $false $false))
        $merged.CommandsExecutedCount|Should -Be 0
        $merged.CoveragePercent|Should -Be 0
    }
    It 'rejects a child source SHA different from its exact parent preimage' {
        $result=New-ParentResult $false $false (New-ChildSnapshot $false $true)
        $result.Tests[0].PcaiChildCoverage.ParentSources=@([pscustomobject]@{Path=$script:Source;SHA=('0'*64)})
        {Merge-PesterChildCoverage $result}|Should -Throw '*identity mismatch*'
    }
    It 'rejects source bytes changed after child measurement' {
        $result=New-ParentResult $false $false (New-ChildSnapshot $false $true)
        $original=[IO.File]::ReadAllBytes($script:Source)
        try {
            [IO.File]::AppendAllText($script:Source,'# changed')
            {Merge-PesterChildCoverage $result}|Should -Throw '*identity mismatch*'
        } finally { [IO.File]::WriteAllBytes($script:Source,$original) }
    }
    It 'rejects an unknown command extent even on a known source line' {
        $result=New-ParentResult $true $false (New-ChildSnapshot $false $true)
        $result.Tests[0].PcaiChildCoverage.Child.Commands[1].StartColumn=30
        $result.Tests[0].PcaiChildCoverage.Child.Commands[1].EndColumn=34
        {Merge-PesterChildCoverage $result}|Should -Throw '*absent from parent denominator*'
    }
    It 'rejects duplicate commands within one child inventory' {
        $child=New-ChildSnapshot $true $false
        $child.Commands=@($child.Commands[0],$child.Commands[0]);$child.Executed=2
        {Merge-PesterChildCoverage (New-ParentResult $true $false $child)}|Should -Throw '*Duplicate*'
    }
    It 'rejects a count inconsistent with measured command records' {
        $child=New-ChildSnapshot $true $false;$child.Executed=2
        {Merge-PesterChildCoverage (New-ParentResult $true $false $child)}|Should -Throw '*reconciliation*'
    }
    It 'rejects receipt data on a nonpassed parent test' {
        $result=New-ParentResult $true $false (New-ChildSnapshot $false $true)
        $result.Tests[0].Result='Failed'
        {Merge-PesterChildCoverage $result}|Should -Throw '*Unpassed*'
    }
    It 'cannot consume a receipt attached to another run' {
        $result=New-ParentResult $true $false (New-ChildSnapshot $false $true)
        $other=[pscustomobject]@{Tests=@();CodeCoverage=$result.CodeCoverage}
        (Merge-PesterChildCoverage $other).CommandsExecutedCount|Should -Be 1
    }
    It 'uses ordinal Linux paths and case-insensitive Windows path keys' {
        $lower=Get-PesterCoveragePathKey $script:Source -CaseSensitive $true
        $upper=Get-PesterCoveragePathKey ($script:Source.ToUpperInvariant()) -CaseSensitive $true
        $lower|Should -Not -BeExactly $upper
        (Get-PesterCoveragePathKey $script:Source -CaseSensitive $false)|Should -BeExactly (Get-PesterCoveragePathKey ($script:Source.ToUpperInvariant()) -CaseSensitive $false)
    }
    It 'deduplicates Windows case aliases without expanding the denominator' {
        $saved=$PSDefaultParameterValues
        try {
            $PSDefaultParameterValues=@{'Get-PesterCoveragePathKey:CaseSensitive'=$false}
            $result=New-ParentResult $true $false (New-ChildSnapshot $true $true)
            foreach($command in $result.Tests[0].PcaiChildCoverage.Child.Commands){$command.File=$command.File.ToUpperInvariant()}
            $merged=Merge-PesterChildCoverage $result
            $merged.CommandsAnalyzedCount|Should -Be 2
            $merged.CommandsExecutedCount|Should -Be 2
            $merged.AddedChildCommands|Should -Be 1
            (Merge-PesterChildCoverage $result).CommandsExecutedCount|Should -Be 2
        } finally { $PSDefaultParameterValues=$saved }
    }
    It 'preserves distinct case-only Linux sources and their per-file inventories' {
        # Emulate Linux metadata on Windows without altering filesystem policy:
        # each case-only alias is bound to a distinct real temporary source SHA.
        $other=Join-Path $TestDrive 'second-physical-source.ps1'
        [IO.File]::WriteAllText($other,'return 2')
        $script:CaseAlias=$script:Source.ToUpperInvariant()
        $script:CaseAliasHash=(Get-FileHash -LiteralPath $other).Hash
        Mock Get-FileHash {
            param($LiteralPath)
            if([string]::Equals($LiteralPath,$script:CaseAlias,[StringComparison]::Ordinal)){return [pscustomobject]@{Hash=$script:CaseAliasHash}}
            if([string]::Equals($LiteralPath,$script:Source,[StringComparison]::Ordinal)){return [pscustomobject]@{Hash=$script:Binding.SHA}}
            throw 'Unexpected metadata lookup in case-only source control.'
        }
        $saved=$PSDefaultParameterValues
        try {
            $PSDefaultParameterValues=@{'Get-PesterCoveragePathKey:CaseSensitive'=$true}
            $first=New-CoverageCommand 3 $true
            $second=New-CoverageCommand 3 $true;$second.File=$script:CaseAlias
            $child=[pscustomobject]@{SchemaVersion=1;Sources=@([pscustomobject]@{Path=$script:Source;SHA256=$script:Binding.SHA},[pscustomobject]@{Path=$script:CaseAlias;SHA256=$script:CaseAliasHash});Commands=@($first,$second);Analyzed=2;Executed=2}
            $binding=@($script:Binding,[pscustomobject]@{Path=$script:CaseAlias;SHA=$script:CaseAliasHash})
            $parent=[pscustomobject]@{Tests=@([pscustomobject]@{Result='Passed';PcaiChildCoverage=[pscustomobject]@{ParentSources=$binding;Child=$child}});CodeCoverage=[pscustomobject]@{CommandsAnalyzedCount=2;CommandsExecutedCount=0;CommandsExecuted=@();CommandsMissed=@($first,$second)}}
            $merged=Merge-PesterChildCoverage $parent
            $merged.CommandsAnalyzedCount|Should -Be 2
            $merged.CommandsExecutedCount|Should -Be 2
            $merged.AddedChildCommands|Should -Be 2
            $child.Commands=@($first,$first)
            {Merge-PesterChildCoverage $parent}|Should -Throw '*Duplicate coverage command*'
        } finally { $PSDefaultParameterValues=$saved }
    }
    It 'rejects redistributed per-file counts despite equal total inventory' {
        $result=New-ParentResult $false $false (New-ChildSnapshot $true $true)
        $other=Join-Path $TestDrive 'extra-source.ps1';[IO.File]::WriteAllText($other,'return 3')
        $binding=[pscustomobject]@{Path=$other;SHA=(Get-FileHash -LiteralPath $other).Hash}
        $result.Tests[0].PcaiChildCoverage.ParentSources+= $binding
        $result.Tests[0].PcaiChildCoverage.Child.Sources+= [pscustomobject]@{Path=$other;SHA256=$binding.SHA}
        $result.Tests[0].PcaiChildCoverage.Child.Commands[1].File=$other
        {Merge-PesterChildCoverage $result}|Should -Throw '*inventory mismatch*'
    }
    It 'keeps delimiter, newline, Unicode and same-line command identities distinct' {
        $first=New-CoverageCommand 3 $true
        $second=New-CoverageCommand 3 $true
        $first.Command="a|b`n雪"
        $second.Command="a`nb|雪"
        (Get-PesterCoverageCommandKey $first)|Should -Not -BeExactly (Get-PesterCoverageCommandKey $second)
        $second.Command=$first.Command;$second.StartColumn=4
        (Get-PesterCoverageCommandKey $first)|Should -Not -BeExactly (Get-PesterCoverageCommandKey $second)
    }
}

Describe 'Actual Pester measured child coverage transport' -Tag 'Unit','Coverage','Windows' {
    It 'unions actual child-only commands, preserves overlap and zero, and retains the test attachment after Pester cleanup' {
        # The child only runs immutable generated tests; its outer runner owns a
        # bounded kill-on-close job. No production native DLL or device is used.
        . (Join-Path $script:RepositoryRoot 'Tools/SystemScripts/Networking/Invoke-VigilBoundedProcess.ps1')
        $fixture=Join-Path $TestDrive 'actual.ps1'
        $test=Join-Path $TestDrive 'actual.Tests.ps1'
        $probe=Join-Path $TestDrive 'actual-probe.ps1'
        [IO.File]::WriteAllText($fixture,@'
function Get-ActualUnionValue {
    param([bool]$Second)
    $label='measured'
    if($Second){$value='second'}else{$value='first'}
    return "$label $value"
}
'@)
        [IO.File]::WriteAllText($test,@'
param($Source,$Mode,$Child,$Bindings)
BeforeAll { . $Source }
Describe 'Actual source executions' {
    It 'executes exactly the selected source branches' {
        if($Mode -ne 'Zero'){Get-ActualUnionValue $false|Should -BeExactly 'measured first'}
        if($Mode -eq 'Child'){Get-ActualUnionValue $true|Should -BeExactly 'measured second'}
        if($Mode -eq 'Zero'){1|Should -Be 1}
        if($null -ne $Child){
            $test=& (Get-Module Pester) {Get-CurrentTest}
            $test|Add-Member -NotePropertyName PcaiChildCoverage -NotePropertyValue ([pscustomobject]@{ParentSources=$Bindings;Child=$Child})
        }
    }
}
'@)
        [IO.File]::WriteAllText($probe,@'
param($Manifest,$Helper)
$ErrorActionPreference='Stop'
Import-Module $Manifest -Force
. $Helper
$source=Join-Path $PSScriptRoot 'actual.ps1'
$binding=@([pscustomobject]@{Path=$source;SHA=(Get-FileHash -LiteralPath $source).Hash})
function Measure-ActualCase($Mode,$Child) {
    $config=New-PesterConfiguration
    $config.Run.Container=New-PesterContainer -Path (Join-Path $PSScriptRoot 'actual.Tests.ps1') -Data @{Source=$source;Mode=$Mode;Child=$Child;Bindings=$binding}
    $config.Run.PassThru=$true;$config.Run.Exit=$false
    $config.CodeCoverage.Enabled=$true;$config.CodeCoverage.Path=$source
    $config.CodeCoverage.OutputPath=Join-Path $PSScriptRoot ("coverage-$Mode.xml")
    $config.Output.Verbosity='None'
    $result=Invoke-Pester -Configuration $config
    if($result.PassedCount-ne1-or$result.TotalCount-ne1-or$result.FailedCount-or$result.FailedBlocksCount-or$result.FailedContainersCount){throw 'Actual fixture failed.'}
    return $result
}
$child=Measure-ActualCase Child $null
$snapshot=New-PesterCoverageSnapshot $child.CodeCoverage $binding
$parent=Measure-ActualCase Parent $snapshot
$merged=Merge-PesterChildCoverage $parent
. (Join-Path (Split-Path $Helper -Parent) 'Assert-PesterCoverageGate.ps1')
Assert-PesterCoverageGate -Result $parent -Target 85
$zero=Measure-ActualCase Zero $null
$zeroSnapshot=New-PesterCoverageSnapshot $zero.CodeCoverage $binding
$zero.Tests[0]|Add-Member -NotePropertyName PcaiChildCoverage -NotePropertyValue ([pscustomobject]@{ParentSources=$binding;Child=$zeroSnapshot})
$zeroMerged=Merge-PesterChildCoverage $zero
$zeroRejected=$false
try { Assert-PesterCoverageGate -Result $zero -Target 85 } catch { $zeroRejected=$_.Exception.Message -like '*below required 85*' }
if(-not $zeroRejected){throw 'Zero coverage did not fail the unchanged gate.'}
[pscustomobject]@{ParentExecuted=$parent.CodeCoverage.CommandsExecutedCount;ChildExecuted=$child.CodeCoverage.CommandsExecutedCount;Analyzed=$parent.CodeCoverage.CommandsAnalyzedCount;Merged=$merged;ZeroMerged=$zeroMerged;ZeroGateRejected=$zeroRejected;AttachmentRetained=($null-ne$parent.Tests[0].PSObject.Properties['PcaiChildCoverage'])}|ConvertTo-Json -Depth 6|Set-Content -LiteralPath (Join-Path $PSScriptRoot 'measured-result.json')
'@)
        $pwsh=(Get-Command pwsh -CommandType Application -ErrorAction Stop|Select-Object -First 1).Source
        $run=Invoke-VigilBoundedProcess -FilePath $pwsh -Arguments @('-NoLogo','-NoProfile','-File',$probe,$script:PesterManifest,$script:MergeHelper) -TimeoutSeconds 45 -WorkingDirectory $TestDrive
        if($run.ExitCode){Write-Host $run.Stdout;Write-Host $run.Stderr}
        $run.ExitCode|Should -Be 0
        $actual=Get-Content -LiteralPath (Join-Path $TestDrive 'measured-result.json') -Raw|ConvertFrom-Json
        $actual.AttachmentRetained|Should -BeTrue
        $actual.Merged.CommandsAnalyzedCount|Should -Be $actual.Analyzed
        $actual.Merged.CommandsExecutedCount|Should -Be $actual.ChildExecuted
        $actual.Merged.AddedChildCommands|Should -BeGreaterThan 0
        $actual.ZeroMerged.CommandsExecutedCount|Should -Be 0
        $actual.ZeroMerged.CoveragePercent|Should -Be 0
        $actual.ZeroGateRejected|Should -BeTrue
    }
}
