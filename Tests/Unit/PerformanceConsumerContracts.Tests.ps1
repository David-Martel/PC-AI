#Requires -Version 7.0
param(
    [switch]$IsolatedChild,
    [string]$RepositoryRoot,
    [string]$EvidenceRoot,
    [switch]$RequirePreloadedParent,
    [ValidateSet('Consumer','ZeroDiscovery','AfterAllFailure','ContainerFailure','PositiveControl')][string]$ChildCaseKind='Consumer'
)
if(-not $RepositoryRoot){$RepositoryRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))}
if($IsolatedChild -and $ChildCaseKind -ceq 'Consumer') {
Describe 'Performance typed producer consumer contracts' {
BeforeAll {
    $script:sourceRoot=Join-Path $RepositoryRoot 'Modules/PC-AI.Performance/Public'
    $transport=@'
#nullable enable
using System;
using System.Runtime.InteropServices;
namespace PcaiNative {
    // Inert synthetic transport only. Actual OptimizerModule.cs is compiled
    // verbatim against this boundary; no native DLL or runtime collection.
    public enum PcaiStatus { Success=0, InternalError=1 }
    public struct SyntheticStringBuffer {
        public string? Value;
        public string? ToManagedString() => Value;
    }
    public static class SyntheticOptimizerProducer {
        public static string? MemoryJson,CategoriesJson,RecommendationsJson;
        public static string? ThrowMethod;
        public static int FreeCalls;
        public static MemoryPressureReport MemoryReport;
    }
    internal static class NativeCore {
        public static uint pcai_core_test() => 0x50434149;
        public static MemoryPressureReport pcai_analyze_memory_pressure() {
            if(SyntheticOptimizerProducer.ThrowMethod=="summary") throw new DllNotFoundException("synthetic native summary transport unavailable");
            return SyntheticOptimizerProducer.MemoryReport;
        }
        private static SyntheticStringBuffer Get(string method,string? value) {
            if(SyntheticOptimizerProducer.ThrowMethod==method) throw new DllNotFoundException("synthetic native JSON transport unavailable");
            return new SyntheticStringBuffer {Value=value};
        }
        public static SyntheticStringBuffer pcai_get_memory_pressure_json() => Get("memory",SyntheticOptimizerProducer.MemoryJson);
        public static SyntheticStringBuffer pcai_get_process_categories_json() => Get("categories",SyntheticOptimizerProducer.CategoriesJson);
        public static SyntheticStringBuffer pcai_get_optimization_recommendations_json() => Get("recommendations",SyntheticOptimizerProducer.RecommendationsJson);
        public static void pcai_free_string_buffer(ref SyntheticStringBuffer buffer) {SyntheticOptimizerProducer.FreeCalls++;buffer=default;}
    }
}

'@
    Add-Type -TypeDefinition ($transport+"`n"+([IO.File]::ReadAllText((Join-Path $RepositoryRoot 'Native/PcaiNative/OptimizerModule.cs')) -replace '^using System;\s*using System.Runtime.InteropServices;','')) -ErrorAction Stop
    # The actual compiled managed wrapper needs its explicit StructLayout using.
    foreach($name in @('Get-PcaiMemoryPressure','Get-PcaiProcessCategories','Get-PcaiOptimizationPlan')){. (Join-Path $script:sourceRoot ($name+'.ps1'))}
    function Initialize-PcaiNative {throw 'Unmocked initializer forbidden'}
    function Get-CimInstance {param($ClassName,$Filter)throw 'Unmocked CIM forbidden'}
    function Get-Counter {param($Counter)throw 'Unmocked counter forbidden'}
    function New-SyntheticProcess([int]$Id,[string]$Name,[long]$Private=10MB,[int]$Handles=10){[pscustomobject]@{Id=$Id;ProcessName=$Name;WorkingSet64=10MB;PrivateMemorySize64=$Private;HandleCount=$Handles}}
    function Set-NativeRecommendations([int]$Count){
        $items=@(for($i=0;$i-lt$Count;$i++){[ordered]@{priority=2;category='large_process';description='Synthetic snapshot observation';estimated_savings_mb=10;action='investigate_process';safe_to_auto=$false}})
        [PcaiNative.SyntheticOptimizerProducer]::RecommendationsJson=[ordered]@{status='Success';elapsed_ms=1;recommendation_count=$Count;recommendations=$items}|ConvertTo-Json -Depth 5 -Compress
    }
}
BeforeEach {
    $script:native=$false;$script:processes=@();$script:cimProcesses=@();$script:freeKB=8192*1024;$script:totalBytes=32768MB;$script:pages=0;$script:pool=0;$script:commit=55;$script:counterStatus=0
    [PcaiNative.SyntheticOptimizerProducer]::ThrowMethod=$null;[PcaiNative.SyntheticOptimizerProducer]::FreeCalls=0
    $report=[PcaiNative.MemoryPressureReport]::new();$report.Status=[PcaiNative.PcaiStatus]::Success;$report.PressureLevel=1;$report.AvailableMB=8192;$report.CommittedPct=0.5
    [PcaiNative.SyntheticOptimizerProducer]::MemoryReport=$report
    [PcaiNative.SyntheticOptimizerProducer]::MemoryJson='{"status":"Success","pressure_level":1,"pressure_label":"moderate","available_mb":8192,"committed_pct":0.5,"pool_nonpaged_mb":0,"pages_per_sec":0,"top_consumer_count":0,"handle_leak_count":0,"orphan_terminal_count":0,"elapsed_ms":1}'
    [PcaiNative.SyntheticOptimizerProducer]::CategoriesJson='{"status":"Success","elapsed_ms":1,"categories":{"llm_agents":{"count":0,"working_set_mb":0,"private_mb":0,"handle_count":0},"browsers":{"count":0,"working_set_mb":0,"private_mb":0,"handle_count":0},"terminals":{"count":0,"working_set_mb":0,"private_mb":0,"handle_count":0},"build_tools":{"count":0,"working_set_mb":0,"private_mb":0,"handle_count":0},"system_services":{"count":0,"working_set_mb":0,"private_mb":0,"handle_count":0}}}'
    Set-NativeRecommendations 0
    Mock Import-Module {}
    Mock Initialize-PcaiNative {return $script:native}
    Mock Get-CimInstance {
        switch($ClassName){
            'Win32_OperatingSystem' {[pscustomobject]@{FreePhysicalMemory=$script:freeKB}}
            'Win32_ComputerSystem' {[pscustomobject]@{TotalPhysicalMemory=$script:totalBytes}}
            'Win32_Process' {if($Filter){$null}else{$script:cimProcesses}}
            default {throw 'Unexpected synthetic CIM class'}
        }
    }
    Mock Get-Counter {param($Counter) [pscustomobject]@{CounterSamples=@([pscustomobject]@{Status=$script:counterStatus;CookedValue=if($Counter-like'*Pages/sec'){$script:pages}elseif($Counter-like'*Committed*'){$script:commit}else{$script:pool}})}}
    Mock Get-Process {param($Name,$Id)
        if($PSBoundParameters.ContainsKey('Id')){return @($script:processes|Where-Object Id -eq $Id)}
        if($PSBoundParameters.ContainsKey('Name')){return @($script:processes|Where-Object {$_.ProcessName-in$Name})}
        return $script:processes
    }
}
Describe 'Synthetic typed producer controls, actual managed wrapper' {
    It 'retains native summary numeric fields through the actual C# wrapper' {
        $script:native=$true;$result=Get-PcaiMemoryPressure
        $result.AvailableMB|Should -Be 8192
        if ($IsWindows) { $result.CommittedPct|Should -Be 50 } else { $result.CommittedPct|Should -BeNullOrEmpty }
        $result.Source|Should -BeExactly 'PcaiNative.OptimizerModule'
        Should -Invoke Get-CimInstance -Times 0 -Exactly
    }
    It 'uses the actual wrapper string-buffer release on the synthetic producer' {
        $script:native=$true;$json=Get-PcaiMemoryPressure -AsJson
        ($json|ConvertFrom-Json).status|Should -BeExactly 'Success'
        [PcaiNative.SyntheticOptimizerProducer]::FreeCalls|Should -Be 1
    }
    It 'keeps ordinary fallback summary available on synthetic valid metadata' {
        $result=Get-PcaiMemoryPressure;$result.AvailableMB|Should -Be 8192;$result.Source|Should -BeExactly 'PowerShell-Fallback'
    }
    It 'does not label large-process observation safe for automatic action' {
        $script:processes=@(New-SyntheticProcess 12001 'synthetic-work' 3GB)
        $item=@(Get-PcaiOptimizationPlan|Where-Object Category -eq large_process)
        $item.Count|Should -Be 1;$item[0].SafeToAuto|Should -BeFalse
    }
}
Describe 'Public schema and JSON cardinality detecting contracts' {
    It 'returns JSON text for fallback memory AsJson as it does for native' {Get-PcaiMemoryPressure -AsJson|Should -BeOfType ([string])}
    It 'preserves the public summary field contract in native Detailed mode' {
        $script:native=$true;$summary=Get-PcaiMemoryPressure;$detail=Get-PcaiMemoryPressure -Detailed
        $detail.PSObject.Properties.Name|Should -Contain 'AvailableMB';$detail.AvailableMB|Should -Be $summary.AvailableMB
    }
    It 'returns public category rows on native path as on fallback' {
        $fallback=@(Get-PcaiProcessCategories);$script:native=$true;$nativeRows=@(Get-PcaiProcessCategories)
        $nativeRows[0].PSObject.Properties.Name|Should -Contain 'Category'
        $nativeRows[0].PSObject.Properties.Name|Should -Contain 'ProcessCount'
        $fallback[0].PSObject.Properties.Name|Should -Contain 'Category'
    }
    It 'returns <Count> public recommendation rows from a native producer envelope' -TestCases @(@{Count=0},@{Count=1},@{Count=2}) {
        param($Count)
        $script:native=$true;Set-NativeRecommendations $Count;$rows=@(Get-PcaiOptimizationPlan)
        $rows.Count|Should -Be $Count
        if($Count){$rows[0].PSObject.Properties.Name|Should -Contain 'Priority';$rows[0].PSObject.Properties.Name|Should -Contain 'SafeToAuto'}
    }
    It 'returns a JSON array for native <Count> recommendations' -TestCases @(@{Count=0},@{Count=1},@{Count=2}) {
        param($Count)
        $script:native=$true;Set-NativeRecommendations $Count;$json=Get-PcaiOptimizationPlan -AsJson
        $json|Should -BeOfType ([string]);$json.TrimStart().StartsWith('[')|Should -BeTrue
    }
    It 'returns [] JSON for empty fallback recommendations' {
        $json=Get-PcaiOptimizationPlan -AsJson;$json|Should -BeOfType ([string]);$json|Should -BeExactly '[]'
    }
    It 'returns a JSON array for one fallback recommendation' {
        $script:processes=@(New-SyntheticProcess 12001 'synthetic-work' 3GB)
        $json=Get-PcaiOptimizationPlan -AsJson;$json.TrimStart().StartsWith('[')|Should -BeTrue
    }
    It 'keeps fallback two-recommendation JSON array shape' {
        $script:processes=@((New-SyntheticProcess 12001 'synthetic-work' 3GB),(New-SyntheticProcess 12002 'synthetic-work2' 3GB))
        $json=Get-PcaiOptimizationPlan -AsJson;$json.TrimStart().StartsWith('[')|Should -BeTrue;@($json|ConvertFrom-Json).Count|Should -Be 2
    }
}
Describe 'Transport, metadata and snapshot safety detecting contracts' {
    It 'falls back when the native <Command> transport throws' -TestCases @(@{Command='Get-PcaiMemoryPressure';Method='summary'},@{Command='Get-PcaiProcessCategories';Method='categories'},@{Command='Get-PcaiOptimizationPlan';Method='recommendations'}) {
        param($Command,$Method)
        $script:native=$true;[PcaiNative.SyntheticOptimizerProducer]::ThrowMethod=$Method
        {& $Command}|Should -Not -Throw
        Should -Invoke Get-CimInstance -Times $(if($Command-eq'Get-PcaiProcessCategories'){0}else{1}) -Scope It
    }
    It 'refuses negative physical memory metadata instead of publishing a known numeric assessment' {
        $script:freeKB=-1024
        {Get-PcaiMemoryPressure}|Should -Throw
    }
    It 'does not silently label unavailable counters measured zeros' {
        Mock Get-Counter {throw 'synthetic counter unsupported'}
        $report=Get-PcaiMemoryPressure
        $report.PSObject.Properties.Name|Should -Contain 'MeasurementStatus'
    }
    It 'applies consistent physical pressure severity to equivalent large-memory snapshots' {
        $script:freeKB=5000*1024;$script:totalBytes=65536MB
        $r=[PcaiNative.SyntheticOptimizerProducer]::MemoryReport;$r.PressureLevel=3;$r.AvailableMB=5000;[PcaiNative.SyntheticOptimizerProducer]::MemoryReport=$r
        $fallback=Get-PcaiMemoryPressure;$script:native=$true;$nativeReport=Get-PcaiMemoryPressure
        $fallback.PressureLevelCode|Should -Be $nativeReport.PressureLevelCode
    }
    It 'never labels absent-parent-PID terminals safe for automatic kill without ownership' {
        $script:processes=@(for($i=0;$i-lt6;$i++){New-SyntheticProcess (13000+$i) 'cmd'})
        $script:cimProcesses=@(for($i=0;$i-lt6;$i++){[pscustomobject]@{Name='cmd.exe';ProcessId=(13000+$i);ParentProcessId=99999}})
        $item=@(Get-PcaiOptimizationPlan|Where-Object Category -eq orphan_cleanup)
        $item.Count|Should -Be 1;$item[0].SafeToAuto|Should -BeFalse
    }
    It 'never relays native PID-only orphan cleanup as safe for automatic kill' {
        $script:native=$true
        [PcaiNative.SyntheticOptimizerProducer]::RecommendationsJson='{"status":"Success","elapsed_ms":1,"recommendation_count":1,"recommendations":[{"priority":2,"category":"orphan_cleanup","description":"Synthetic parent PID absent snapshot","estimated_savings_mb":44,"action":"kill_orphan_terminals","safe_to_auto":true}]}'
        $result=@(Get-PcaiOptimizationPlan);$result.Count|Should -Be 1;$result[0].SafeToAuto|Should -BeFalse
    }
    It 'does not infer a leak from one synthetic high-handle snapshot' {
        $script:processes=@(New-SyntheticProcess 12001 'synthetic-work' 10MB 100001)
        $items=@(Get-PcaiOptimizationPlan|Where-Object Category -eq handle_leak)
        $items.Count|Should -Be 1;$items[0].Description|Should -Not -Match '(?i)likely.*leak'
    }
    It 'does not assert thrashing from a single synthetic Pages/sec observation' {
        $script:pages=1500
        $items=@(Get-PcaiOptimizationPlan|Where-Object Category -eq excessive_paging)
        $items.Count|Should -Be 1;$items[0].Description|Should -Not -Match '(?i)system is thrashing'
    }
}
Describe 'Successor metric and producer-schema controls' {
    It 'requires qualified PDH sample status <Case> before using finite data' -TestCases @(
        @{Case='invalid-data';Value=[uint32]3221228474;Accepted=$false},
        @{Case='missing';Value=$null;Accepted=$false},
        @{Case='string';Value='0';Accepted=$false},
        @{Case='boolean';Value=$false;Accepted=$false},
        @{Case='unsupported';Value=42;Accepted=$false},
        @{Case='new-data';Value=1;Accepted=$true}
    ) {
        param($Case,$Value,$Accepted)
        $script:counterStatus=$Value;$script:commit=55;$script:pages=1500;$script:pool=5GB
        $memory=Get-PcaiMemoryPressure
        if($Accepted){
            $memory.CommittedPct|Should -Be 55
            $memory.PagesPerSec|Should -Be 1500
            $memory.PoolNonpagedMB|Should -Be 5120
            @(Get-PcaiOptimizationPlan|Where-Object Category -in @('pool_nonpaged','excessive_paging')).Count|Should -Be 2
        }else{
            $memory.CommittedPct|Should -BeNullOrEmpty
            $memory.PagesPerSec|Should -BeNullOrEmpty
            $memory.PoolNonpagedMB|Should -BeNullOrEmpty
            @(Get-PcaiOptimizationPlan|Where-Object Category -in @('pool_nonpaged','excessive_paging')).Count|Should -Be 0
        }
    }
    It 'distinguishes actual commit percentage from physical used percentage' {
        $script:commit=55
        $result=Get-PcaiMemoryPressure
        $result.CommittedPct|Should -Be 55
        $result.PhysicalUsedPct|Should -Be 75
        $result.MeasurementStatus.CommittedPct|Should -BeExactly 'MeasuredSnapshot'
    }
    It 'does not fabricate native physical usage, pool, paging or private bytes' {
        $script:native=$true
        $memory=Get-PcaiMemoryPressure
        $memory.PhysicalUsedPct|Should -BeNullOrEmpty
        $memory.PoolNonpagedMB|Should -BeNullOrEmpty
        $memory.PagesPerSec|Should -BeNullOrEmpty
        $category=@(Get-PcaiProcessCategories)[0]
        $category.PrivateMB|Should -BeNullOrEmpty
        $category.MeasurementStatus.PrivateMB|Should -BeExactly 'UnavailableNativeVirtualSpace'
    }
    It 'keeps unavailable fallback counters null with provenance' {
        Mock Get-Counter {throw 'Synthetic unavailable optional counter'}
        $result=Get-PcaiMemoryPressure
        $result.CommittedPct|Should -BeNullOrEmpty
        $result.PoolNonpagedMB|Should -BeNullOrEmpty
        $result.PagesPerSec|Should -BeNullOrEmpty
        $result.MeasurementStatus.CommittedPct|Should -BeExactly 'Unavailable'
    }
    It 'rejects invalid physical <Case> metadata before numeric reporting' -TestCases @(
        @{Case='zero-total';Free=8192*1024;Total=0},
        @{Case='free-over-total';Free=999999999;Total=1MB},
        @{Case='NaN';Free=[double]::NaN;Total=1MB},
        @{Case='infinity';Free=[double]::PositiveInfinity;Total=1MB},
        @{Case='missing';Free=$null;Total=1MB},
        @{Case='boolean';Free=$true;Total=1MB},
        @{Case='numeric-string';Free='100';Total=1MB}
    ) {
        param($Case,$Free,$Total)
        $script:freeKB=$Free;$script:totalBytes=$Total
        {Get-PcaiMemoryPressure}|Should -Throw
        {Get-PcaiOptimizationPlan}|Should -Throw
    }
    It 'marks invalid counter <Case> unavailable and never emits paging advice' -TestCases @(
        @{Case='negative';Value=-1},@{Case='NaN';Value=[double]::NaN},@{Case='infinite';Value=[double]::PositiveInfinity},@{Case='missing';Value=$null},@{Case='boolean';Value=$true},@{Case='string';Value='1500'}
    ) {
        param($Case,$Value)
        $script:commit=$Value;$script:pool=$Value;$script:pages=$Value
        $memory=Get-PcaiMemoryPressure
        $memory.CommittedPct|Should -BeNullOrEmpty
        $memory.PoolNonpagedMB|Should -BeNullOrEmpty
        $memory.PagesPerSec|Should -BeNullOrEmpty
        @(Get-PcaiOptimizationPlan|Where-Object Category -in @('pool_nonpaged','excessive_paging')).Count|Should -Be 0
    }
    It 'falls back for <Kind> malformed JSON' -TestCases @(@{Kind='memory'},@{Kind='categories'},@{Kind='recommendations'}) {
        param($Kind)
        $script:native=$true
        switch($Kind){
            memory {[PcaiNative.SyntheticOptimizerProducer]::MemoryJson='{';$result=Get-PcaiMemoryPressure -Detailed}
            categories {[PcaiNative.SyntheticOptimizerProducer]::CategoriesJson='{';$result=@(Get-PcaiProcessCategories)[0]}
            recommendations {[PcaiNative.SyntheticOptimizerProducer]::RecommendationsJson='{';$script:processes=@(New-SyntheticProcess 18000 'synthetic-work' 3GB);$result=@(Get-PcaiOptimizationPlan)[0]}
        }
        $result.Source|Should -BeExactly 'PowerShell-Fallback'
    }
    It 'falls back for <Kind> non-success producer status' -TestCases @(@{Kind='memory'},@{Kind='categories'},@{Kind='recommendations'}) {
        param($Kind)
        $script:native=$true
        switch($Kind){
            memory {[PcaiNative.SyntheticOptimizerProducer]::MemoryJson=[PcaiNative.SyntheticOptimizerProducer]::MemoryJson.Replace('Success','InternalError');$result=Get-PcaiMemoryPressure -Detailed}
            categories {[PcaiNative.SyntheticOptimizerProducer]::CategoriesJson=[PcaiNative.SyntheticOptimizerProducer]::CategoriesJson.Replace('Success','InternalError');$result=@(Get-PcaiProcessCategories)[0]}
            recommendations {[PcaiNative.SyntheticOptimizerProducer]::RecommendationsJson=[PcaiNative.SyntheticOptimizerProducer]::RecommendationsJson.Replace('Success','InternalError');$script:processes=@(New-SyntheticProcess 18000 'synthetic-work' 3GB);$result=@(Get-PcaiOptimizationPlan)[0]}
        }
        $result.Source|Should -BeExactly 'PowerShell-Fallback'
    }
    It 'falls back for native summary error status' {
        $script:native=$true
        $r=[PcaiNative.SyntheticOptimizerProducer]::MemoryReport;$r.Status=[PcaiNative.PcaiStatus]::InternalError;[PcaiNative.SyntheticOptimizerProducer]::MemoryReport=$r
        (Get-PcaiMemoryPressure).Source|Should -BeExactly 'PowerShell-Fallback'
    }
    It 'falls back for invalid native numeric JSON before public projection' {
        $script:native=$true
        [PcaiNative.SyntheticOptimizerProducer]::MemoryJson=[PcaiNative.SyntheticOptimizerProducer]::MemoryJson.Replace('"committed_pct":0.5','"committed_pct":"0.5"')
        (Get-PcaiMemoryPressure -Detailed).Source|Should -BeExactly 'PowerShell-Fallback'
    }
    It 'falls back for recommendation cardinality mismatch and null entries' -TestCases @(@{Value='{"status":"Success","recommendation_count":2,"recommendations":[]}'},@{Value='{"status":"Success","recommendation_count":1,"recommendations":[null]}'}) {
        param($Value)
        $script:native=$true;$script:processes=@(New-SyntheticProcess 18000 'synthetic-work' 3GB)
        [PcaiNative.SyntheticOptimizerProducer]::RecommendationsJson=$Value
        @((Get-PcaiOptimizationPlan))[0].Source|Should -BeExactly 'PowerShell-Fallback'
    }
    It 'preserves native category array JSON with <Count> rows' -TestCases @(@{Count=0},@{Count=1},@{Count=2}) {
        param($Count)
        $script:native=$true;$map=[ordered]@{}
        foreach($name in @('browsers','terminals')|Select-Object -First $Count){$map[$name]=@{count=1;working_set_mb=10;private_mb=20;handle_count=10}}
        [PcaiNative.SyntheticOptimizerProducer]::CategoriesJson=[ordered]@{status='Success';elapsed_ms=1;categories=$map}|ConvertTo-Json -Depth 5
        $json=Get-PcaiProcessCategories -AsJson
        $json.TrimStart().StartsWith('[')|Should -BeTrue
        @($json|ConvertFrom-Json).Count|Should -Be $Count
    }
    It 'uses exact ordered native name groups without double counting' {
        $script:processes=@((New-SyntheticProcess 19001 'CLAUDE'),(New-SyntheticProcess 19002 'chrome'),(New-SyntheticProcess 19003 'pwsh'),(New-SyntheticProcess 19004 'node'),(New-SyntheticProcess 19005 'ordinary-unknown'))
        $rows=@(Get-PcaiProcessCategories)
        @($rows|Where-Object Category -eq LLM_Agents)[0].ProcessCount|Should -Be 1
        @($rows|Where-Object Category -eq Build_Tools)[0].ProcessCount|Should -Be 1
        @($rows|Where-Object Category -eq System_Services)[0].ProcessCount|Should -Be 1
        ($rows|Measure-Object ProcessCount -Sum).Sum|Should -Be 5
        @($rows.Taxonomy|Select-Object -Unique)|Should -BeExactly @('NativeOrderedNameHeuristicV1')
        (Get-PcaiProcessCategories -AsJson).TrimStart().StartsWith('[')|Should -BeTrue
    }
    It 'never invents savings or causal diagnoses for native advice' {
        $script:native=$true
        [PcaiNative.SyntheticOptimizerProducer]::RecommendationsJson='{"status":"Success","elapsed_ms":1,"recommendation_count":1,"recommendations":[{"priority":1,"category":"handle_leak","description":"Strong indicator of a leak","estimated_savings_mb":4000,"action":"restart_handle_leak_processes","safe_to_auto":false}]}'
        $row=@(Get-PcaiOptimizationPlan)[0]
        $row.EstimatedSavingsMB|Should -BeNullOrEmpty
        $row.Description|Should -Not -Match '(?i)strong indicator|driver leak|system is thrashing'
        $row.SafeToAuto|Should -BeFalse
        $row.Action|Should -BeExactly 'restart_handle_leak_processes'
    }
}

}

} elseif($IsolatedChild) {
    switch($ChildCaseKind) {
        ZeroDiscovery { Describe 'Owned zero discovery control' {} }
        AfterAllFailure { Describe 'Owned failed AfterAll control' {It 'passes ordinary body' {1|Should -Be 1};AfterAll {throw 'Synthetic owned AfterAll failure'}} }
        ContainerFailure { BeforeAll {throw 'Synthetic owned container failure'};Describe 'Owned failed container control' {It 'cannot execute its body' {1|Should -Be 1}} }
        PositiveControl { Describe 'Owned positive child control' {It 'passes ordinary body' {1|Should -Be 1}} }
    }
} else {
Describe 'Performance consumer child isolation and result admission' -Tag 'Unit','Performance','Portable' {
    BeforeAll {
        . (Join-Path $PSScriptRoot '../Fixtures/PerformanceContractChild.ps1')
        $script:parentTypes=@(Get-PerformanceContractTypes)
        $script:contractEvidence=Join-Path $TestDrive 'performance-contracts'
        if($EvidenceRoot){$script:contractEvidence=$EvidenceRoot}
        $loadedPester=@(Get-Module Pester)
        if($loadedPester.Count-ne1){throw 'Exactly one loaded parent Pester package is required.'}
        $script:pesterPath=Join-Path $loadedPester[0].ModuleBase 'Pester.psd1'
        $package=Test-ModuleManifest -Path $script:pesterPath -ErrorAction Stop
        if($package.Guid-eq[Guid]::Empty-or$package.Guid-ne$loadedPester[0].Guid-or$package.Version-ne$loadedPester[0].Version){throw 'Parent Pester package manifest identity mismatch.'}
        $script:pesterVersion=$package.Version.ToString()
        $script:fixturePath=$PSCommandPath
    }
    It 'qualifies all63 consumer bodies in a fresh child without using the parent bridge' {
        if($RequirePreloadedParent){
            @($script:parentTypes|Where-Object Type -eq 'PcaiNative.OptimizerModule').Count|Should -Be 1
            @($script:parentTypes|Where-Object Type -eq 'PcaiNative.OptimizerModule')[0].Location|Should -Not -BeNullOrEmpty
        }
        $result=Invoke-PerformanceContractChild -FixturePath $script:fixturePath -RepositoryRoot $RepositoryRoot -EvidenceRoot (Join-Path $script:contractEvidence 'consumer') -PesterManifest $script:pesterPath
        $result.Passed|Should -Be 63;$result.Total|Should -Be 63
        $result.Pester|Should -BeExactly $script:pesterVersion
        $result.SourceStable|Should -BeTrue
        @($result.TypesBefore).Count|Should -Be 0
        $childType=@($result.TypesAfter|Where-Object Type -eq 'PcaiNative.OptimizerModule')
        $childType.Count|Should -Be 1
        $childType[0].Location|Should -BeNullOrEmpty
        foreach($parent in $script:parentTypes){$childType[0].Assembly|Should -Not -BeExactly $parent.Assembly}
    }
    It 'rejects actual zero discovery with a retained raw0 receipt' {
        $caught=$null
        try { Invoke-PerformanceContractChild -FixturePath $script:fixturePath -RepositoryRoot $RepositoryRoot -EvidenceRoot (Join-Path $script:contractEvidence 'zero') -PesterManifest $script:pesterPath -Kind ZeroDiscovery } catch {$caught=$_}
        $caught|Should -Not -BeNullOrEmpty
        $caught.Exception.Data['ChildResult'].Total|Should -Be 0
        $caught.Exception.Data['ChildResult'].Failed|Should -Be 0
    }
    It 'rejects an actual failed AfterAll despite a passed test body' {
        $caught=$null
        try { Invoke-PerformanceContractChild -FixturePath $script:fixturePath -RepositoryRoot $RepositoryRoot -EvidenceRoot (Join-Path $script:contractEvidence 'afterall') -PesterManifest $script:pesterPath -Kind AfterAllFailure } catch {$caught=$_}
        $caught|Should -Not -BeNullOrEmpty
        $raw=$caught.Exception.Data['ChildResult']
        $raw.Passed|Should -Be 1;$raw.Failed|Should -Be 0;$raw.Total|Should -Be 1;$raw.FailedBlocks|Should -Be 1
    }
    It 'rejects an actual failed container' {
        $caught=$null
        try { Invoke-PerformanceContractChild -FixturePath $script:fixturePath -RepositoryRoot $RepositoryRoot -EvidenceRoot (Join-Path $script:contractEvidence 'container') -PesterManifest $script:pesterPath -Kind ContainerFailure } catch {$caught=$_}
        $caught|Should -Not -BeNullOrEmpty
        $caught.Exception.Data['ChildResult'].FailedContainers|Should -Be 1
    }
    It 'admits an actual positive1 control only as a control' {
        $result=Invoke-PerformanceContractChild -FixturePath $script:fixturePath -RepositoryRoot $RepositoryRoot -EvidenceRoot (Join-Path $script:contractEvidence 'positive') -PesterManifest $script:pesterPath -Kind PositiveControl
        $result.Passed|Should -Be 1;$result.Kind|Should -BeExactly 'PositiveControl'
    }
    It 'rejects missing mandatory <Field> result data' -TestCases @(@{Field='Total'},@{Field='FailedBlocks'},@{Field='FailedContainers'}) {
        param($Field)
        $data=[ordered]@{Passed=1;Failed=0;Skipped=0;NotRun=0;Total=1;FailedContainers=0;FailedBlocks=0;SourceStable=$true;Cases=@([pscustomobject]@{Result='Passed'})}
        $data.Remove($Field)
        {Assert-PerformanceContractResult ([pscustomobject]$data) 1}|Should -Throw
    }
    It 'rejects skipped-body evidence' {
        $data=[pscustomobject]@{Passed=0;Failed=0;Skipped=1;NotRun=0;Total=1;FailedContainers=0;FailedBlocks=0;SourceStable=$true;Cases=@([pscustomobject]@{Result='Skipped'})}
        {Assert-PerformanceContractResult $data 1}|Should -Throw
    }
}

}
