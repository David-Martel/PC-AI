#Requires -Version 7.0

BeforeAll {
    $script:ImportRepositoryRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
    $script:ImportChildPath = Join-Path $TestDrive 'virtualization-import-child.ps1'
    @'
param([string]$RepositoryRoot,[string]$OwnedRoot,[switch]$PreloadCommon)
$ErrorActionPreference='Stop'
$env:PCAI_CACHE_PROVIDER='memory'
$env:PCAI_DISABLE_NATIVE='1'
$env:LOCALAPPDATA=Join-Path $OwnedRoot 'localappdata'
[void][IO.Directory]::CreateDirectory($env:LOCALAPPDATA)
Remove-Item Env:PCAI_ROOT,Env:PCAI_ARTIFACTS_ROOT -ErrorAction SilentlyContinue
$env:PSModulePath=(Join-Path $RepositoryRoot 'Modules')+[IO.Path]::PathSeparator+$env:PSModulePath
$global:OwnedImportBoundaryCalls=0
function global:bw {$global:OwnedImportBoundaryCalls++;throw 'Forbidden provider in import fixture.'}
function global:jules {$global:OwnedImportBoundaryCalls++;throw 'Forbidden Jules in import fixture.'}
function global:Invoke-RestMethod {$global:OwnedImportBoundaryCalls++;throw 'Forbidden HTTP in import fixture.'}
function global:Invoke-WebRequest {$global:OwnedImportBoundaryCalls++;throw 'Forbidden HTTP in import fixture.'}
function global:Get-CimInstance {$global:OwnedImportBoundaryCalls++;throw 'Forbidden CIM in import fixture.'}
function global:Get-VM {$global:OwnedImportBoundaryCalls++;throw 'Forbidden Hyper-V in import fixture.'}
function global:Get-PnpDevice {$global:OwnedImportBoundaryCalls++;throw 'Forbidden device query in import fixture.'}
$common=$null;$cache=$null;$provider=$null
if($PreloadCommon){
 $common=Import-Module (Join-Path $RepositoryRoot 'Modules/PC-AI.Common/PC-AI.Common.psm1') -PassThru -ErrorAction Stop
 & $common {Set-PcaiSharedCacheEntry -Namespace 'virtualization-import-fixture' -Key 'owned' -Value 'owned-cache-value' -TtlSeconds 300|Out-Null}
 $cache=$global:PcaiSharedCache;$provider=$global:PcaiSharedCacheProviderState
}
$manifest=Join-Path $RepositoryRoot 'Modules/PC-AI.Virtualization/PC-AI.Virtualization.psd1'
$returned=@(Import-Module $manifest -PassThru -ErrorAction Stop -WarningAction SilentlyContinue)
$selected=@($returned|Where-Object Name -CEQ 'PC-AI.Virtualization')
$listed=@(Get-Module PC-AI.Virtualization -All)
$modules=@(Get-Module -All|Where-Object Name -Like 'PC*'|ForEach-Object Name|Sort-Object -Unique)
$same=$selected.Count -eq 1 -and $listed.Count -eq 1 -and [object]::ReferenceEquals($selected[0],$listed[0])
$exportsBound=$selected.Count -eq 1
if($exportsBound){foreach($command in $selected[0].ExportedFunctions.Values){if(-not [object]::ReferenceEquals($command.ScriptBlock.Module,$selected[0])){$exportsBound=$false;break}}}
$commonSame=$true;$cacheSame=$true;$payloadSame=$true
if($PreloadCommon){
 $loadedCommon=@(Get-Module PC-AI.Common -All)
 $commonSame=$loadedCommon.Count -eq 1 -and [object]::ReferenceEquals($common,$loadedCommon[0])
 $cacheSame=[object]::ReferenceEquals($cache,$global:PcaiSharedCache) -and [object]::ReferenceEquals($provider,$global:PcaiSharedCacheProviderState)
 $payloadSame=(& $common {Get-PcaiSharedCacheEntry -Namespace 'virtualization-import-fixture' -Key 'owned' -TtlSeconds 300}) -ceq 'owned-cache-value'
}
[ordered]@{SelectedCount=$selected.Count;ListedCount=$listed.Count;SelectedSameListed=$same;ExportScriptBlocksBoundToSelected=$exportsBound;Modules=$modules;CommonIdentityPreserved=$commonSame;CacheStatePreserved=$cacheSame;CachePayloadPreserved=$payloadSame;ForbiddenCalls=$global:OwnedImportBoundaryCalls}|ConvertTo-Json -Depth 5
'@ | Set-Content -LiteralPath $script:ImportChildPath -Encoding utf8

    function Invoke-OwnedVirtualizationImport {
        param([switch]$PreloadCommon)
        $ownedRoot = Join-Path $TestDrive ([guid]::NewGuid().ToString('N'))
        $null = New-Item -ItemType Directory -Path $ownedRoot
        $start = [Diagnostics.ProcessStartInfo]::new()
        $start.FileName = (Get-Process -Id $PID).Path
        $start.UseShellExecute = $false
        $start.CreateNoWindow = $true
        $start.RedirectStandardOutput = $true
        $start.RedirectStandardError = $true
        foreach ($argument in @('-NoLogo','-NoProfile','-NonInteractive','-File',$script:ImportChildPath,
                '-RepositoryRoot',$script:ImportRepositoryRoot,'-OwnedRoot',$ownedRoot)) {
            $start.ArgumentList.Add($argument)
        }
        if ($PreloadCommon) { $start.ArgumentList.Add('-PreloadCommon') }
        $process = [Diagnostics.Process]::new()
        $process.StartInfo = $start
        try {
            [void]$process.Start()
            $stdout = $process.StandardOutput.ReadToEndAsync()
            $stderr = $process.StandardError.ReadToEndAsync()
            if (-not $process.WaitForExit(45000)) {
                $process.Kill($true)
                [void]$process.WaitForExit(5000)
                throw 'Owned import child exceeded its 45-second bound.'
            }
            $outputText = $stdout.GetAwaiter().GetResult()
            $errorText = $stderr.GetAwaiter().GetResult()
            [IO.File]::WriteAllText((Join-Path $ownedRoot 'stdout.log'), $outputText)
            [IO.File]::WriteAllText((Join-Path $ownedRoot 'stderr.log'), $errorText)
            $process.ExitCode | Should -Be 0 -Because 'the owned import child must complete without invoking an external boundary'
            $errorText | Should -BeNullOrEmpty
            return $outputText | ConvertFrom-Json
        }
        finally { $process.Dispose() }
    }
}

Describe 'Virtualization import producer identity' -Tag 'Unit','Portable' {
    It 'imports only the selected producer when the repository Modules directory is discoverable' {
        $result = Invoke-OwnedVirtualizationImport
        $result.SelectedCount | Should -Be 1
        $result.ListedCount | Should -Be 1
        $result.SelectedSameListed | Should -BeTrue
        $result.ExportScriptBlocksBoundToSelected | Should -BeTrue
        @($result.Modules) | Should -Be @('PC-AI.Virtualization')
        $result.ForbiddenCalls | Should -Be 0
    }

    It 'preserves an already imported Common producer and its owned memory-cache entry' {
        $result = Invoke-OwnedVirtualizationImport -PreloadCommon
        $result.SelectedSameListed | Should -BeTrue
        $result.ExportScriptBlocksBoundToSelected | Should -BeTrue
        @($result.Modules) | Should -Be @('PC-AI.Common','PC-AI.Virtualization')
        $result.CommonIdentityPreserved | Should -BeTrue
        $result.CacheStatePreserved | Should -BeTrue
        $result.CachePayloadPreserved | Should -BeTrue
        $result.ForbiddenCalls | Should -Be 0
    }
}
