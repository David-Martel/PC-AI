#Requires -Version 7.0
param([string]$RuntimeHelperPath, [string]$VirtualizationManifestPath)

BeforeAll {
    $script:RuntimeRepo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
    $script:RuntimeProducer = if ($RuntimeHelperPath) { $RuntimeHelperPath } else { Join-Path $RuntimeRepo 'Modules/PC-AI.Common/Public/Get-PcaiRuntimeConfig.ps1' }
    $script:RuntimeCacheProducer = Join-Path (Split-Path $RuntimeProducer) 'Get-PcaiSharedCache.ps1'
    $script:RuntimeVirtualizationManifest = if ($VirtualizationManifestPath) { $VirtualizationManifestPath } else { Join-Path $RuntimeRepo 'Modules/PC-AI.Virtualization/PC-AI.Virtualization.psd1' }
    $script:RuntimePwsh = [Diagnostics.Process]::GetCurrentProcess().MainModule.FileName
    $script:RuntimeConsumer = Join-Path $TestDrive 'runtime-consumer.ps1'
    [IO.File]::WriteAllText($RuntimeConsumer, @'
param([string]$Mode,[string]$Target,[string]$Helper,[string]$Manifest,[string]$Fixture,[string]$Report)
$ErrorActionPreference='Stop'
$global:OwnedRuntimeCalls=@{Reads=[Collections.Generic.List[string]]::new();Writes=[Collections.Generic.List[string]]::new();Stamps=0}
$failure=$null;$runtime=$null;$modules=@();$removedStillDefined=$null
try {
    if($Mode -in @('reuse','removed')) {
        function global:Get-PcaiSharedCacheEntry {
            param($Namespace,$Key,$TtlSeconds,$DependencyStamp)
            $global:OwnedRuntimeCalls.Reads.Add([string]$Key)
            return $null
        }
        function global:Set-PcaiSharedCacheEntry {
            param($Namespace,$Key,$Value,$TtlSeconds,$DependencyStamp)
            $global:OwnedRuntimeCalls.Writes.Add([string]$Key)
        }
        function global:Get-PcaiDependencyStamp {
            param($InputObject)
            $global:OwnedRuntimeCalls.Stamps++
            return 'owned-dependency-stamp'
        }
    }
    if($Mode -eq 'module') {
        Import-Module $Manifest -Force -ErrorAction Stop
        $modules=@(Microsoft.PowerShell.Core\Get-Module PC-AI.Virtualization -All | ForEach-Object Path)
        $selected=@(Microsoft.PowerShell.Core\Get-Module PC-AI.Virtualization -All | Where-Object Path -EQ ([IO.Path]::ChangeExtension($Manifest,'.psm1')))
        if($selected.Count -ne 1){throw 'Exact selected component is not unique'}
        $runtime=& $selected[0] {param($root) Get-PcaiRuntimeConfig -ProjectRoot $root} $Fixture
    } else {
        . $Helper
        if($Mode -eq 'removed') {
            Microsoft.PowerShell.Management\Remove-Item -LiteralPath ('Function:\'+$Target) -ErrorAction Stop
            $removedStillDefined=Microsoft.PowerShell.Management\Test-Path -LiteralPath ('Function:\'+$Target)
        }
        $runtime=Get-PcaiRuntimeConfig -ProjectRoot $Fixture
    }
} catch { $failure=$_.Exception.ToString() }
[ordered]@{
    Failure=$failure;Modules=$modules;Runtime=$runtime;RemovedStillDefined=$removedStillDefined
    Calls=@{Reads=@($global:OwnedRuntimeCalls.Reads);Writes=@($global:OwnedRuntimeCalls.Writes);Stamps=$global:OwnedRuntimeCalls.Stamps}
    Imports=@(Microsoft.PowerShell.Core\Get-Module -All | Where-Object { $_.Name -like 'PoisonRuntime*' -or $_.Path -like '*foreign-component*' } | ForEach-Object Path)
} | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $Report -Encoding utf8NoBOM
if($failure){exit 1}
'@)

    function Invoke-OwnedRuntimeConsumer {
        param([string]$Mode,[string]$Target='Get-PcaiSharedCacheEntry')
        $root=Join-Path $TestDrive ([guid]::NewGuid().ToString('N'))
        $fixture=Join-Path $root 'project'
        $helpers=Join-Path $root 'helpers'
        $moduleRoot=Join-Path $root 'modules'
        $poison=Join-Path $moduleRoot 'PoisonRuntime'
        $foreign=Join-Path $root 'foreign-component/PC-AI.Virtualization'
        foreach($dir in @($fixture,$helpers,$poison,$foreign,(Join-Path $fixture 'Config'))){[void][IO.Directory]::CreateDirectory($dir)}
        [IO.File]::WriteAllText((Join-Path $fixture 'AGENTS.md'),'Owned runtime-root marker.')
        [IO.File]::WriteAllText((Join-Path $fixture 'Config/llm-config.json'),'{"providers":{"pcai-inference":{"baseUrl":"http://localhost:39997"}}}')
        $helper=Join-Path $helpers 'Get-PcaiRuntimeConfig.ps1'
        [IO.File]::WriteAllBytes($helper,[IO.File]::ReadAllBytes($script:RuntimeProducer))
        $sibling=Join-Path $helpers 'Get-PcaiSharedCache.ps1'
        if($Mode -eq 'local') {
            [IO.File]::WriteAllBytes($sibling,[IO.File]::ReadAllBytes($script:RuntimeCacheProducer))
        } else {
            [IO.File]::WriteAllText($sibling,"[IO.File]::WriteAllText(`$env:PCAI_RUNTIME_SIBLING_MARKER,'unexpected sibling');throw 'Loaded helpers must be reused'")
        }
        $marker=Join-Path $root 'poison-imported.txt'
        $siblingMarker=Join-Path $root 'sibling-entered.txt'
        [IO.File]::WriteAllText((Join-Path $foreign 'PC-AI.Virtualization.psm1'),"function Get-OwnedForeignComponent { 'foreign' };Export-ModuleMember -Function Get-OwnedForeignComponent")
        [IO.File]::WriteAllText((Join-Path $poison 'PoisonRuntime.psd1'),"@{RootModule='PoisonRuntime.psm1';ModuleVersion='1.0.0';FunctionsToExport=@('$Target');CmdletsToExport=@();AliasesToExport=@();VariablesToExport=@()}")
        [IO.File]::WriteAllText((Join-Path $poison 'PoisonRuntime.psm1'),("[IO.File]::WriteAllText(`$env:PCAI_RUNTIME_POISON_MARKER,'imported');Import-Module `'$((Join-Path $foreign 'PC-AI.Virtualization.psm1').Replace("'","''"))`' -Global;function $Target { throw 'Foreign optional helper was executed' };Export-ModuleMember -Function $Target"))
        $report=Join-Path $root 'report.json'
        $start=[Diagnostics.ProcessStartInfo]::new($script:RuntimePwsh)
        $start.UseShellExecute=$false;$start.RedirectStandardOutput=$true;$start.RedirectStandardError=$true
        $start.Environment['PSModulePath']=$moduleRoot+[IO.Path]::PathSeparator+(Join-Path $PSHOME 'Modules')
        $start.Environment['PCAI_RUNTIME_POISON_MARKER']=$marker
        $start.Environment['PCAI_RUNTIME_SIBLING_MARKER']=$siblingMarker
        $start.Environment['PCAI_CACHE_PROVIDER']='memory'
        foreach($arg in @('-NoLogo','-NoProfile','-File',$script:RuntimeConsumer,'-Mode',$Mode,'-Target',$Target,'-Helper',$helper,'-Manifest',$script:RuntimeVirtualizationManifest,'-Fixture',$fixture,'-Report',$report)){$start.ArgumentList.Add($arg)}
        $child=[Diagnostics.Process]::Start($start)
        try {
            $stdout=$child.StandardOutput.ReadToEndAsync();$stderr=$child.StandardError.ReadToEndAsync()
            if(-not$child.WaitForExit(20000)){$child.Kill($true);$child.WaitForExit();throw 'Owned runtime consumer exceeded twenty seconds.'}
            [pscustomobject]@{ExitCode=$child.ExitCode;StdOut=$stdout.GetAwaiter().GetResult();StdErr=$stderr.GetAwaiter().GetResult();PoisonImported=[IO.File]::Exists($marker);SiblingEntered=[IO.File]::Exists($siblingMarker);Data=if([IO.File]::Exists($report)){[IO.File]::ReadAllText($report)|ConvertFrom-Json}else{$null};Fixture=$fixture}
        } finally { if(-not$child.HasExited){$child.Kill($true);$child.WaitForExit()};$child.Dispose() }
    }
}

Describe 'Runtime optional discovery remains inside the selected source' -Tag 'Unit','Portable','Common' {
    # Protects: an absent optional command loads the actual adjacent cache helper.
    # Detects: advertised foreign commands turn a loaded-only lookup into imports.
    # Needs: real helper/cache source, owned module path, fresh PowerShell; no network.
    # Breadcrumb: Get-PcaiRuntimeConfig.ps1 bootstrap and runtime cache guards.
    It 'loads its real sibling cache without importing an advertised foreign helper' {
        $r=Invoke-OwnedRuntimeConsumer local
        $r.PoisonImported|Should -BeFalse
        $r.ExitCode|Should -Be 0 -Because $r.StdErr
        $r.Data.Failure|Should -BeNullOrEmpty
        @($r.Data.Imports)|Should -HaveCount 0
        $r.Data.Runtime.ProjectRoot|Should -Be $r.Fixture
        $r.Data.Runtime.PcaiInferenceUrl|Should -BeExactly 'http://localhost:39997'
    }
    # Protects: caller-supplied loaded helpers remain the actual cache boundary.
    # Detects: unnecessary sibling dot-source or dropping cache/stamp calls.
    # Needs: real runtime producer and three owned loaded functions with call logs.
    # Breadcrumb: Resolve-PcaiRepoRoot and Get-PcaiRuntimeConfig optional reuse.
    It 'reuses defined cache and dependency helpers instead of sourcing its sibling' {
        $r=Invoke-OwnedRuntimeConsumer reuse
        $r.PoisonImported|Should -BeFalse
        $r.SiblingEntered|Should -BeFalse
        $r.ExitCode|Should -Be 0 -Because $r.StdErr
        @($r.Data.Calls.Reads)|Should -HaveCount 2
        @($r.Data.Calls.Writes)|Should -HaveCount 2
        $r.Data.Calls.Stamps|Should -Be 1
        $r.Data.Runtime.PcaiInferenceUrl|Should -BeExactly 'http://localhost:39997'
    }
    # Protects: optional Get/Set/stamp commands may disappear after bootstrap.
    # Detects: later guards autoload advertised replacements or require the helper.
    # Needs: actual producer, owned loaded helpers and real function removal.
    # Breadcrumb: all optional guards in Resolve-PcaiRepoRoot/Get-PcaiRuntimeConfig.
    It 'keeps a valid runtime when loaded <Target> disappears' -ForEach @(
        @{Target='Get-PcaiSharedCacheEntry'},@{Target='Set-PcaiSharedCacheEntry'},@{Target='Get-PcaiDependencyStamp'}
    ) {
        $r=Invoke-OwnedRuntimeConsumer -Mode removed -Target $Target
        $r.Data.RemovedStillDefined|Should -BeFalse
        $r.PoisonImported|Should -BeFalse
        $r.SiblingEntered|Should -BeFalse
        $r.ExitCode|Should -Be 0 -Because $r.StdErr
        $r.Data.Failure|Should -BeNullOrEmpty
        $r.Data.Runtime.ProjectRoot|Should -Be $r.Fixture
        $r.Data.Runtime.PcaiInferenceUrl|Should -BeExactly 'http://localhost:39997'
        if($Target -eq 'Get-PcaiSharedCacheEntry'){@($r.Data.Calls.Reads)|Should -HaveCount 0}
        if($Target -eq 'Set-PcaiSharedCacheEntry'){@($r.Data.Calls.Writes)|Should -HaveCount 0}
        if($Target -eq 'Get-PcaiDependencyStamp'){$r.Data.Calls.Stamps|Should -Be 0}
    }
    # Protects: explicit component imports retain exactly the requested module.
    # Detects: shared helper discovery imports a second same-named component.
    # Needs: actual selected virtualization manifest and an owned foreign component.
    # Breadcrumb: PC-AI.Virtualization.psm1 shared-runtime import boundary.
    It 'imports only the exact selected virtualization component in a fresh consumer' {
        $r=Invoke-OwnedRuntimeConsumer module
        $r.PoisonImported|Should -BeFalse
        $r.ExitCode|Should -Be 0 -Because $r.StdErr
        @($r.Data.Modules)|Should -HaveCount 1
        $r.Data.Modules[0]|Should -Be ([IO.Path]::ChangeExtension($script:RuntimeVirtualizationManifest,'.psm1'))
        @($r.Data.Imports)|Should -HaveCount 0
        $r.Data.Runtime.ProjectRoot|Should -Be $r.Fixture
        $r.Data.Runtime.PcaiInferenceUrl|Should -BeExactly 'http://localhost:39997'
    }
}
