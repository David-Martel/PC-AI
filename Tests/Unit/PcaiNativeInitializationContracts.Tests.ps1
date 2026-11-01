#Requires -Version 7.0
param(
    [switch]$IsolatedChild,
    [switch]$RunChild,
    [string]$RequestPath,
    [string]$RepositoryRoot,
    [string]$EvidenceRoot
)
if (-not $RepositoryRoot) { $RepositoryRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..')) }

if ($RunChild) {
    $ErrorActionPreference = 'Stop'
    $request = Get-Content -LiteralPath $RequestPath -Raw | ConvertFrom-Json -ErrorAction Stop
    Import-Module $request.PesterManifest -Force -ErrorAction Stop
    . (Join-Path $request.RepositoryRoot 'Tools/Merge-PesterChildCoverage.ps1')
    $bindings = @($request.Bindings)
    foreach ($binding in $bindings) {
        if ((Get-FileHash -LiteralPath $binding.Path).Hash -cne $binding.SHA) { throw 'Child input changed before admission.' }
    }
    if ('PcaiNative.PcaiCore' -as [type]) { throw 'Fresh child already has a native bridge type.' }
    $config = New-PesterConfiguration
    $config.Run.Container = New-PesterContainer -Path $request.FixturePath -Data @{ IsolatedChild=$true; RepositoryRoot=$request.RepositoryRoot }
    $config.Run.PassThru = $true
    $config.Output.Verbosity = 'Detailed'
    $config.TestResult.Enabled = $true
    $config.TestResult.OutputPath = Join-Path $request.EvidenceRoot 'tests.xml'
    $config.CodeCoverage.Enabled = $true
    $config.CodeCoverage.Path = @($request.ProducerPath)
    $config.CodeCoverage.OutputPath = Join-Path $request.EvidenceRoot 'coverage.xml'
    $result = Invoke-Pester -Configuration $config
    $stable = $true
    foreach ($binding in $bindings) {
        if ((Get-FileHash -LiteralPath $binding.Path).Hash -cne $binding.SHA) { $stable=$false }
    }
    $receipt = [ordered]@{
        ChildPid=$PID; ParentPid=$request.ParentPid; SourceStable=$stable
        Scope='Complete actual PowerShell producer with inert managed collaborators; no native DLL, hardware, provider or network qualification.'
        Pester=(Get-Module Pester).Version.ToString()
        Total=$result.TotalCount; Passed=$result.PassedCount; Failed=$result.FailedCount
        Skipped=$result.SkippedCount; NotRun=$result.NotRunCount
        FailedBlocks=$result.FailedBlocksCount; FailedContainers=$result.FailedContainersCount
        Cases=@($result.Tests | Select-Object ExpandedName,Result)
        Bindings=$bindings
        NativeAssemblyLoaded=[bool](@([AppDomain]::CurrentDomain.GetAssemblies() | Where-Object {$_.GetName().Name -eq 'PcaiNative'}).Count)
        ExternalCalls=[PcaiNative.InitializationFixture]::ExternalCalls
        Coverage=if ($stable) { New-PesterCoverageSnapshot -Coverage $result.CodeCoverage -SourceBindings @($bindings | Where-Object Path -eq $request.ProducerPath) } else { $null }
    }
    $receipt | ConvertTo-Json -Depth 9 | Set-Content -LiteralPath (Join-Path $request.EvidenceRoot 'result.json') -Encoding utf8
    if (-not $stable -or $result.TotalCount -le 0 -or $result.FailedCount -or $result.SkippedCount -or $result.NotRunCount -or $result.FailedBlocksCount -or $result.FailedContainersCount -or $receipt.NativeAssemblyLoaded -or $receipt.ExternalCalls) { exit 1 }
    exit 0
}

if ($IsolatedChild) {
Describe 'Actual native initialization with an inert bridge boundary' -Tag 'Unit','Acceleration','Portable' {
    BeforeAll {
        # This deliberately incomplete transport has no native imports, assembly
        # resolver or production algorithms. Only the PowerShell producer is under test.
        Add-Type -TypeDefinition @'
using System;
namespace PcaiNative {
    public static class InitializationFixture {
        public static bool Ready=true, ThrowAvailability, ThrowResolver, Activate=true, MutateCore;
        public static string LoadedPath, LastMethod;
        public static object[] Arguments;
        public static int AvailabilityCalls, ResolverCalls, ExternalCalls;
        public static void Record(string method, params object[] args) { LastMethod=method; Arguments=args; }
        public static void Reset() {
            Ready=true; ThrowAvailability=false; ThrowResolver=false; Activate=true; MutateCore=false;
            LoadedPath=null; LastMethod=null; Arguments=null; AvailabilityCalls=0; ResolverCalls=0;
        }
    }
    public static class NativeResolver {
        public static bool TryGetLoadedLibraryPath(string name, out string path) {
            InitializationFixture.ResolverCalls++;
            if (InitializationFixture.ThrowResolver) throw new InvalidOperationException("synthetic resolver refusal");
            path=InitializationFixture.LoadedPath; return path!=null;
        }
    }
    public static class PcaiCore {
        public static bool IsAvailable { get {
            InitializationFixture.AvailabilityCalls++;
            if(InitializationFixture.ThrowAvailability) throw new InvalidOperationException("synthetic unavailable bridge");
            if(InitializationFixture.Activate) InitializationFixture.LoadedPath=Environment.GetEnvironmentVariable("PCAI_CORE_LIB_PATH");
            if(InitializationFixture.MutateCore) {
                System.IO.File.WriteAllText(Environment.GetEnvironmentVariable("PCAI_CORE_LIB_PATH"), "changed owned marker");
                InitializationFixture.MutateCore=false;
            }
            return InitializationFixture.Ready;
        } }
        public static string Version { get { return "inert-contract-version"; } }
        public static uint CpuCount { get { return 7; } }
        public static bool CheckResourceSafety(float limit) { InitializationFixture.Record("safety",limit); return limit<0.9f; }
        public static ulong EstimateTokens(string text) { InitializationFixture.Record("tokens",text); return 42; }
    }
    public static class PcaiSearch {
        public static bool IsAvailable { get { return false; } }
        public static string FindDuplicates(string p,ulong n,string i,string e) { InitializationFixture.Record("duplicates",p,n,i,e); return "duplicates-result"; }
        public static string FindDuplicatesStats(string p,ulong n,string i,string e) { InitializationFixture.Record("duplicate-stats",p,n,i,e); return "duplicate-stats-result"; }
        public static string FindFiles(string p,string d,uint n) { InitializationFixture.Record("files",p,d,n); return "files-result"; }
        public static string FindFilesStats(string p,string d,ulong n) { InitializationFixture.Record("file-stats",p,d,n); return "file-stats-result"; }
        public static string CollectDirectoryManifest(string p,uint d,ulong n) { InitializationFixture.Record("manifest",p,d,n); return "manifest-result"; }
        public static string CollectDirectoryManifestStats(string p,uint d,ulong n) { InitializationFixture.Record("manifest-stats",p,d,n); return "manifest-stats-result"; }
        public static string SearchContent(string p,string d,string f,uint n,uint c) { InitializationFixture.Record("content",p,d,f,n,c); return "content-result"; }
        public static string SearchContentStats(string p,string d,string f,ulong n) { InitializationFixture.Record("content-stats",p,d,f,n); return "content-stats-result"; }
    }
    public static class SystemModule { public static bool IsAvailable { get {return true;} } public static string QuerySystemInfo() { InitializationFixture.Record("system"); return "system-result"; } }
    public static class FsModule { public static bool IsAvailable { get {return false;} } }
    public static class PerformanceModule { public static string QueryHardwareMetrics() { InitializationFixture.Record("metrics"); return "metrics-result"; } }
    public enum DiagnosticVerbosity { Basic,Normal,Full }
    public static class PcaiDiagnostics { public static string GetUnifiedHardwareReportJson(DiagnosticVerbosity v) { InitializationFixture.Record("report",v); return "report-result"; } }
    public static class ProcessLassoModule { public static string GetSnapshotJson(string c,string l,int m) { InitializationFixture.Record("lasso",c,l,m); return "lasso-result"; } }
}
'@ -ErrorAction Stop
        $script:producerPath=Join-Path $RepositoryRoot 'Modules/PC-AI.Acceleration/Private/Initialize-PcaiNative.ps1'
        . $script:producerPath
        function Resolve-PcaiRepoRoot { param($StartPath) $script:ownedRepo }
        function Get-CimInstance { [PcaiNative.InitializationFixture]::ExternalCalls++; throw 'CIM forbidden' }
        function Get-Counter { [PcaiNative.InitializationFixture]::ExternalCalls++; throw 'Counters forbidden' }
        function Invoke-WebRequest { [PcaiNative.InitializationFixture]::ExternalCalls++; throw 'Network forbidden' }
        function Invoke-RestMethod { [PcaiNative.InitializationFixture]::ExternalCalls++; throw 'Network forbidden' }
        function Start-Process { [PcaiNative.InitializationFixture]::ExternalCalls++; throw 'Processes forbidden' }
        function New-InertBundle([string]$Name) {
            $path=Join-Path $TestDrive $Name
            $null=[IO.Directory]::CreateDirectory($path)
            foreach($leaf in @((Get-PcaiNativeLibraryFileName),'PcaiNative.dll')) { [IO.File]::WriteAllText((Join-Path $path $leaf),'owned inert artifact') }
            $path
        }
        $script:environmentNames=@('PCAI_CORE_LIB_PATH','PCAI_TEST_CORE_DLL','PCAI_NATIVE_DIR','PCAI_NATIVE_BUNDLE_ROOT','PCAI_ROOT','PATH')
    }
    BeforeEach {
        $script:savedEnvironment=@{}
        foreach($name in $script:environmentNames) {
            $script:savedEnvironment[$name]=[Environment]::GetEnvironmentVariable($name)
            if($name -ne 'PATH') { [Environment]::SetEnvironmentVariable($name,$null) }
        }
        [PcaiNative.InitializationFixture]::Reset()
        $script:ownedRepo=Join-Path $TestDrive ('owned-repository-'+[guid]::NewGuid().ToString('N'))
        $null=[IO.Directory]::CreateDirectory($script:ownedRepo)
        $script:bundle=New-InertBundle ([guid]::NewGuid().ToString('N'))
        $script:core=Join-Path $script:bundle (Get-PcaiNativeLibraryFileName)
        $env:PCAI_NATIVE_BUNDLE_ROOT=$script:bundle
        foreach($name in @('PcaiNativeVersion','PcaiNativeDllPath','PcaiNativeCoreLibraryPath','PcaiNativeWrapperPath','PcaiNativeArtifactHash','PcaiNativeRid')) { Set-Variable -Name $name -Value $null -Scope Script }
        $script:PcaiNativeLoaded=$false
        Mock Get-Module { $null }
        # This is the managed load boundary, never a mock of initializer logic.
        Mock Add-Type { if(-not $Path -or -not $Path.StartsWith($TestDrive,[StringComparison]::OrdinalIgnoreCase)) { [PcaiNative.InitializationFixture]::ExternalCalls++; throw 'Unowned managed load forbidden' } }
    }
    AfterEach {
        foreach($name in $script:environmentNames) { [Environment]::SetEnvironmentVariable($name,$script:savedEnvironment[$name]) }
        [PcaiNative.InitializationFixture]::ExternalCalls | Should -Be 0
        @([AppDomain]::CurrentDomain.GetAssemblies() | Where-Object {$_.GetName().Name -eq 'PcaiNative'}).Count | Should -Be 0
    }
    It 'publishes exact selected paths, stable marker digest and synthetic version after activation' {
        Initialize-PcaiNative | Should -BeTrue
        $script:PcaiNativeLoaded | Should -BeTrue
        $script:PcaiNativeCoreLibraryPath | Should -BeExactly $script:core
        $script:PcaiNativeWrapperPath | Should -BeExactly (Join-Path $script:bundle 'PcaiNative.dll')
        $script:PcaiNativeArtifactHash | Should -BeExactly (Get-FileHash -LiteralPath $script:core).Hash
        $script:PcaiNativeVersion | Should -BeExactly 'inert-contract-version'
        $script:PcaiNativeRid | Should -BeExactly (Get-PcaiNativeRuntimeIdentifier)
        Should -Invoke Add-Type -Times 1 -Exactly -ParameterFilter {$Path -eq (Join-Path $script:bundle 'PcaiNative.dll')}
    }
    It 'reuses the validated same bundle without another availability or managed-load call' {
        Initialize-PcaiNative | Should -BeTrue
        $path=$env:PATH; $hash=$script:PcaiNativeArtifactHash
        Initialize-PcaiNative | Should -BeTrue
        [PcaiNative.InitializationFixture]::AvailabilityCalls | Should -Be 1
        $env:PATH | Should -BeExactly $path
        $script:PcaiNativeArtifactHash | Should -BeExactly $hash
        Should -Invoke Add-Type -Times 1 -Exactly
    }
    It 'forces revalidation of the same selected core while preserving its prior proven digest' {
        Initialize-PcaiNative | Should -BeTrue
        $hash=$script:PcaiNativeArtifactHash
        Initialize-PcaiNative -Force | Should -BeTrue
        [PcaiNative.InitializationFixture]::AvailabilityCalls | Should -Be 2
        $script:PcaiNativeArtifactHash | Should -BeExactly $hash
        @($env:PATH -split [regex]::Escape([string][IO.Path]::PathSeparator) | Where-Object {$_ -eq $script:bundle}).Count | Should -Be 1
    }
    It 'does not publish a digest if activation changes the selected marker bytes' {
        [PcaiNative.InitializationFixture]::MutateCore=$true
        Initialize-PcaiNative | Should -BeTrue
        $script:PcaiNativeArtifactHash | Should -BeNullOrEmpty
        [IO.File]::ReadAllText($script:core) | Should -BeExactly 'changed owned marker'
    }
    It 'does not publish a digest if the bridge cannot locate its activated core' {
        [PcaiNative.InitializationFixture]::Activate=$false
        Initialize-PcaiNative | Should -BeTrue
        $script:PcaiNativeArtifactHash | Should -BeNullOrEmpty
    }
    It 'returns false and retains selected paths when the inert core is unavailable or throws: <Mode>' -TestCases @(@{Mode='unavailable'},@{Mode='throw'}) {
        param($Mode)
        if($Mode -eq 'throw'){[PcaiNative.InitializationFixture]::ThrowAvailability=$true}else{[PcaiNative.InitializationFixture]::Ready=$false}
        Initialize-PcaiNative -WarningAction SilentlyContinue | Should -BeFalse
        $script:PcaiNativeLoaded | Should -BeFalse
        $script:PcaiNativeCoreLibraryPath | Should -BeExactly $script:core
        $script:PcaiNativeArtifactHash | Should -BeNullOrEmpty
    }
    It 'fails closed before mutation when loaded-core lookup throws' {
        [PcaiNative.InitializationFixture]::ThrowResolver=$true
        Initialize-PcaiNative -WarningAction SilentlyContinue | Should -BeFalse
        $env:PATH | Should -BeExactly $script:savedEnvironment.PATH
        Should -Invoke Add-Type -Times 0 -Exactly
    }
    It 'returns a failed initialized state when the managed-load boundary rejects owned marker bytes' {
        Mock Add-Type { throw 'Synthetic managed-load rejection, no actual assembly load.' }
        Initialize-PcaiNative -WarningAction SilentlyContinue | Should -BeFalse
        $script:PcaiNativeLoaded | Should -BeFalse
        $script:PcaiNativeCoreLibraryPath | Should -BeExactly $script:core
        $script:PcaiNativeArtifactHash | Should -BeNullOrEmpty
        [PcaiNative.InitializationFixture]::AvailabilityCalls | Should -Be 0
    }
    It 'does not prepend a second exact PATH segment when the selected bundle is already present' {
        $separator=[IO.Path]::PathSeparator
        $env:PATH="$script:bundle$separator$($script:savedEnvironment.PATH)"
        $before=$env:PATH
        Initialize-PcaiNative | Should -BeTrue
        $env:PATH | Should -BeExactly $before
    }
    It 'rejects an invalid authoritative <Override> even after successful cached initialization' -TestCases @(@{Override='PCAI_CORE_LIB_PATH'},@{Override='PCAI_TEST_CORE_DLL'},@{Override='PCAI_NATIVE_DIR'}) {
        param($Override)
        Initialize-PcaiNative | Should -BeTrue
        [Environment]::SetEnvironmentVariable('PCAI_CORE_LIB_PATH',$null)
        [Environment]::SetEnvironmentVariable($Override,(Join-Path $TestDrive 'missing-authoritative-selection'))
        $before=$env:PATH
        Test-PcaiNativeAvailable -WarningAction SilentlyContinue | Should -BeFalse
        foreach($name in @('PcaiNativeVersion','PcaiNativeDllPath','PcaiNativeCoreLibraryPath','PcaiNativeWrapperPath','PcaiNativeArtifactHash')) { Get-Variable -Name $name -Scope Script -ValueOnly | Should -BeNullOrEmpty }
        $script:PcaiNativeLoaded | Should -BeFalse
        $env:PATH | Should -BeExactly $before
        [PcaiNative.InitializationFixture]::AvailabilityCalls | Should -Be 1
    }
    It 'uses the highest precedence valid override without considering invalid lower precedence values' {
        $env:PCAI_CORE_LIB_PATH=$script:core
        $env:PCAI_TEST_CORE_DLL=Join-Path $TestDrive 'missing-lower-core'
        $env:PCAI_NATIVE_DIR=Join-Path $TestDrive 'missing-lower-directory'
        Initialize-PcaiNative | Should -BeTrue
        $script:PcaiNativeCoreLibraryPath | Should -BeExactly $script:core
    }
    It 'accepts an existing directory supplied through the core override' {
        $env:PCAI_CORE_LIB_PATH=$script:bundle
        Initialize-PcaiNative | Should -BeTrue
        $env:PCAI_CORE_LIB_PATH | Should -BeExactly $script:core
    }
    It 'rejects a valid explicit core that conflicts with the selected paired directory' {
        $foreign=New-InertBundle 'conflicting-owned-pair'
        $env:PCAI_CORE_LIB_PATH=Join-Path $foreign (Get-PcaiNativeLibraryFileName)
        Initialize-PcaiNative -WarningAction SilentlyContinue | Should -BeFalse
        $env:PATH | Should -BeExactly $script:savedEnvironment.PATH
        Should -Invoke Add-Type -Times 0 -Exactly
    }
    It 'rejects an incomplete explicit bundle despite a complete owned repository fallback' {
        $repoBin=Join-Path $script:ownedRepo 'bin';$null=[IO.Directory]::CreateDirectory($repoBin)
        foreach($leaf in @((Get-PcaiNativeLibraryFileName),'PcaiNative.dll')){[IO.File]::WriteAllText((Join-Path $repoBin $leaf),'owned fallback marker')}
        Remove-Item -LiteralPath (Join-Path $script:bundle 'PcaiNative.dll')
        Initialize-PcaiNative -WarningAction SilentlyContinue | Should -BeFalse
        $env:PATH | Should -BeExactly $script:savedEnvironment.PATH
        $script:PcaiNativeCoreLibraryPath | Should -BeNullOrEmpty
        Should -Invoke Add-Type -Times 0 -Exactly
    }
    It 'rejects a nonfilesystem explicit bundle before any lookup-state mutation' {
        $env:PCAI_NATIVE_BUNDLE_ROOT='Env:'
        Initialize-PcaiNative -WarningAction SilentlyContinue | Should -BeFalse
        $env:PATH | Should -BeExactly $script:savedEnvironment.PATH
        Should -Invoke Add-Type -Times 0 -Exactly
    }
    It 'rejects a foreign or unlocated loaded core: <Kind>' -TestCases @(@{Kind='foreign'},@{Kind='relative'}) {
        param($Kind)
        [PcaiNative.InitializationFixture]::LoadedPath=if($Kind -eq 'relative'){'relative-core.dll'}else{Join-Path $TestDrive 'other-core.dll'}
        Initialize-PcaiNative -WarningAction SilentlyContinue | Should -BeFalse
        $env:PATH | Should -BeExactly $script:savedEnvironment.PATH
        Should -Invoke Add-Type -Times 0 -Exactly
    }
    It 'rejects an authoritative valid core when a different core is already bound' {
        $env:PCAI_CORE_LIB_PATH=$script:core
        [PcaiNative.InitializationFixture]::LoadedPath=Join-Path $TestDrive 'already-bound.dll'
        Initialize-PcaiNative -WarningAction SilentlyContinue | Should -BeFalse
        $env:PATH | Should -BeExactly $script:savedEnvironment.PATH
        Should -Invoke Add-Type -Times 0 -Exactly
    }
    It 'normalizes a prefixed native library name for the actual host without retaining another platform extension' {
        Get-PcaiNativeLibraryFileName -LibraryName ' libpcai_core_lib.so ' | Should -BeExactly (Get-PcaiNativeLibraryFileName -LibraryName 'pcai_core_lib')
    }
    It 'keeps candidate search order while sorting owned version bundles newest first' {
        $root=Join-Path $TestDrive 'candidate-root';$null=[IO.Directory]::CreateDirectory($root)
        [IO.File]::WriteAllText((Join-Path $root 'PcaiNative.dll'),'owned partial wrapper')
        $older=Join-Path $root 'native-bundles/older';$newer=Join-Path $root 'native-bundles/newer'
        foreach($path in @($older,$newer)){ $null=[IO.Directory]::CreateDirectory($path);[IO.File]::WriteAllText((Join-Path $path (Get-PcaiNativeLibraryFileName)),'owned core') }
        [IO.Directory]::SetLastWriteTimeUtc($older,[datetime]'2020-01-01Z');[IO.Directory]::SetLastWriteTimeUtc($newer,[datetime]'2021-01-01Z')
        @(Get-PcaiNativeCandidatePaths -BasePaths @((Join-Path $TestDrive 'missing'),$root,$root,$script:bundle)) | Should -Be @($root,$newer,$older,$script:bundle)
    }
    It 'recognizes a runtime-layout marker without requiring a flat core' {
        $root=Join-Path $TestDrive 'runtime-only'
        $native=Join-Path $root ('runtimes/'+(Get-PcaiNativeRuntimeIdentifier)+'/native')
        $null=[IO.Directory]::CreateDirectory($native)
        [IO.File]::WriteAllText((Join-Path $native (Get-PcaiNativeLibraryFileName)),'owned runtime core')
        @(Get-PcaiNativeCandidatePaths -BasePaths @($root)) | Should -Be @($root)
    }
    It 'prefers an intact later pair over an earlier wrapper-only discovery candidate' {
        $env:PCAI_NATIVE_BUNDLE_ROOT=$null
        $bin=Join-Path $script:ownedRepo 'bin';$null=[IO.Directory]::CreateDirectory($bin)
        [IO.File]::WriteAllText((Join-Path $bin 'PcaiNative.dll'),'owned unpaired wrapper')
        $paired=Join-Path $script:ownedRepo ('Native/PcaiNative/bin/Release/net8.0/'+(Get-PcaiNativeRuntimeIdentifier))
        $null=[IO.Directory]::CreateDirectory($paired)
        foreach($leaf in @((Get-PcaiNativeLibraryFileName),'PcaiNative.dll')){[IO.File]::WriteAllText((Join-Path $paired $leaf),'owned paired marker')}
        Initialize-PcaiNative | Should -BeTrue
        $script:PcaiNativeDllPath | Should -BeExactly $paired
        $script:PcaiNativeWrapperPath | Should -BeExactly (Join-Path $paired 'PcaiNative.dll')
    }
    It 'reports synthetic per-module availability and exact artifact presence without fabricating all modules available' {
        [IO.File]::WriteAllText((Join-Path $script:bundle (Get-PcaiNativeLibraryFileName pcai_inference)),'owned inference marker')
        $status=Get-PcaiNativeStatus
        $status.Available | Should -BeTrue; $status.CpuCount | Should -Be 7
        $status.Modules.Core | Should -BeTrue; $status.Modules.Search | Should -BeFalse
        $status.Modules.System | Should -BeTrue; $status.Modules.Fs | Should -BeFalse
        $status.Modules.Performance | Should -BeTrue
        $status.Dlls.PcaiNative | Should -BeTrue; $status.Dlls.CoreLib | Should -BeTrue; $status.Dlls.Inference | Should -BeTrue
        $status.ArtifactHash | Should -BeExactly (Get-FileHash -LiteralPath $script:core).Hash
    }
    It 'reports unavailable status without synthetic module or DLL claims when explicit selection is absent' {
        $env:PCAI_NATIVE_BUNDLE_ROOT=Join-Path $TestDrive 'absent-status-bundle'
        $status=Get-PcaiNativeStatus -WarningAction SilentlyContinue
        $status.Available | Should -BeFalse; $status.Modules | Should -BeNullOrEmpty; $status.Dlls | Should -BeNullOrEmpty
        $status.CpuCount | Should -Be ([uint32][Environment]::ProcessorCount)
        [PcaiNative.InitializationFixture]::AvailabilityCalls | Should -Be 0
    }
    It 'forwards duplicate selection and limits to the inert <Mode> transport' -TestCases @(@{Mode='entries'},@{Mode='stats'}) {
        param($Mode)
        $result=Invoke-PcaiNativeDuplicates -Path $script:bundle -MinimumSize 17 -IncludePattern '*.log' -ExcludePattern 'skip*' -StatsOnly:($Mode -eq 'stats')
        $result | Should -BeExactly $(if($Mode -eq 'stats'){'duplicate-stats-result'}else{'duplicates-result'})
        [PcaiNative.InitializationFixture]::Arguments | Should -Be @($script:bundle,[uint64]17,'*.log','skip*')
    }
    It 'forwards optional file-search path and result limit to the inert <Mode> transport' -TestCases @(@{Mode='entries'},@{Mode='stats'}) {
        param($Mode)
        Invoke-PcaiNativeFileSearch -Pattern '*.ps1' -MaxResults 9 -StatsOnly:($Mode -eq 'stats') | Should -BeExactly $(if($Mode -eq 'stats'){'file-stats-result'}else{'files-result'})
        [PcaiNative.InitializationFixture]::Arguments[0] | Should -BeExactly '*.ps1'
        # PowerShell converts the optional null path to the C# string's empty
        # value; both are the transport's unspecified-path contract.
        [PcaiNative.InitializationFixture]::Arguments[1] | Should -BeNullOrEmpty
        [PcaiNative.InitializationFixture]::Arguments[2] | Should -Be 9
        Invoke-PcaiNativeFileSearch -Pattern 'exact' -Path $script:bundle -MaxResults 2 -StatsOnly:($Mode -eq 'stats') | Out-Null
        [PcaiNative.InitializationFixture]::Arguments | Should -Be @('exact',$script:bundle,2)
    }
    It 'forwards manifest depth and count to the inert <Mode> transport' -TestCases @(@{Mode='entries'},@{Mode='stats'}) {
        param($Mode)
        Invoke-PcaiNativeDirectoryManifest -Path $script:bundle -MaxDepth 3 -MaxResults 11 -StatsOnly:($Mode -eq 'stats') | Should -BeExactly $(if($Mode -eq 'stats'){'manifest-stats-result'}else{'manifest-result'})
        [PcaiNative.InitializationFixture]::Arguments | Should -Be @($script:bundle,3,11)
    }
    It 'forwards regex, file pattern and context through the inert <Mode> transport' -TestCases @(@{Mode='entries'},@{Mode='stats'}) {
        param($Mode)
        Invoke-PcaiNativeContentSearch -Pattern 'literal.+regex' -Path $script:bundle -FilePattern '*.txt' -MaxResults 8 -ContextLines 2 -StatsOnly:($Mode -eq 'stats') | Should -BeExactly $(if($Mode -eq 'stats'){'content-stats-result'}else{'content-result'})
        $expected=@('literal.+regex',$script:bundle,'*.txt',8);if($Mode -eq 'entries'){$expected+=2}
        [PcaiNative.InitializationFixture]::Arguments | Should -Be $expected
        Invoke-PcaiNativeContentSearch -Pattern 'optional-path' -StatsOnly:($Mode -eq 'stats') | Out-Null
        [PcaiNative.InitializationFixture]::Arguments[1] | Should -BeNullOrEmpty
    }
    It 'selects the inert metrics or system transport without querying the OS' {
        Invoke-PcaiNativeSystemInfo | Should -BeExactly 'system-result'
        [PcaiNative.InitializationFixture]::LastMethod | Should -BeExactly 'system'
        Invoke-PcaiNativeSystemInfo -MetricsOnly | Should -BeExactly 'metrics-result'
        [PcaiNative.InitializationFixture]::LastMethod | Should -BeExactly 'metrics'
    }
    It 'forwards token text, resource threshold, diagnostic verbosity and explicit snapshot paths verbatim' {
        Get-PcaiTokenEstimate -Text 'unmodified text' | Should -Be 42
        [PcaiNative.InitializationFixture]::Arguments | Should -Be @('unmodified text')
        Test-PcaiResourceSafety -GpuLimit 0.95 | Should -BeFalse
        [PcaiNative.InitializationFixture]::Arguments[0] | Should -Be ([float]0.95)
        Invoke-PcaiNativeUnifiedHardwareReport -Verbosity Full | Should -BeExactly 'report-result'
        [PcaiNative.InitializationFixture]::Arguments[0].ToString() | Should -BeExactly 'Full'
        Invoke-PcaiNativeProcessLassoSnapshot -ConfigPath 'owned-config' -LogPath 'owned-log' -LookbackMinutes 13 | Should -BeExactly 'lasso-result'
        [PcaiNative.InitializationFixture]::Arguments | Should -Be @('owned-config','owned-log',13)
    }
    It 'fails native-required operations before any inert invocation when explicit initialization fails: <Command>' -TestCases @(
        @{Command='duplicates'},@{Command='files'},@{Command='manifest'},@{Command='content'},@{Command='report'},@{Command='lasso'}
    ) {
        param($Command)
        $env:PCAI_NATIVE_BUNDLE_ROOT=Join-Path $TestDrive 'absent-required-bundle'
        { switch($Command) {
            duplicates {Invoke-PcaiNativeDuplicates -Path $script:bundle -WarningAction SilentlyContinue}
            files {Invoke-PcaiNativeFileSearch -Pattern '*' -WarningAction SilentlyContinue}
            manifest {Invoke-PcaiNativeDirectoryManifest -Path $script:bundle -WarningAction SilentlyContinue}
            content {Invoke-PcaiNativeContentSearch -Pattern '*' -WarningAction SilentlyContinue}
            report {Invoke-PcaiNativeUnifiedHardwareReport -WarningAction SilentlyContinue}
            lasso {Invoke-PcaiNativeProcessLassoSnapshot -WarningAction SilentlyContinue}
        } } | Should -Throw '*PCAI Native tools not available*'
        [PcaiNative.InitializationFixture]::LastMethod | Should -BeNullOrEmpty
    }
    It 'preserves inert system absence, permissive resource fallback and real Unicode token fallback without native readiness' {
        $env:PCAI_NATIVE_BUNDLE_ROOT=Join-Path $TestDrive 'absent-fallback-bundle'
        Invoke-PcaiNativeSystemInfo -WarningAction SilentlyContinue | Should -BeNullOrEmpty
        Test-PcaiResourceSafety -WarningAction SilentlyContinue | Should -BeTrue
        Invoke-PcaiNativeEstimateTokens -Text '' -WarningAction SilentlyContinue | Should -Be 0
        Get-PcaiTokenEstimate -Text 'hello world' -WarningAction SilentlyContinue | Should -Be 4
        Get-PcaiTokenEstimate -Text '12345678901234567890' -WarningAction SilentlyContinue | Should -Be 6
        Get-PcaiTokenEstimate -Text ([char]::ConvertFromUtf32(0x1F600)) -WarningAction SilentlyContinue | Should -Be 2
        Get-PcaiTokenEstimate -Text ('abc '+[char]::ConvertFromUtf32(0x1F600)+' abc') -WarningAction SilentlyContinue | Should -Be 6
        [PcaiNative.InitializationFixture]::LastMethod | Should -BeNullOrEmpty
    }
}
} else {
Describe 'Native initialization child isolation and source-bound admission' -Tag 'Unit','Acceleration','Portable' {
    BeforeAll {
        function Invoke-OwnedInitializationProcess {
            param(
                [Parameter(Mandatory)][Diagnostics.ProcessStartInfo]$StartInfo,
                [Parameter(Mandatory)][string]$OutputRoot,
                [ValidateRange(1,120000)][int]$TimeoutMilliseconds=120000,
                [scriptblock]$StopOwnedProcess={param($OwnedProcess,$OutputTask,$ErrorTask) $OwnedProcess.Kill()}
            )
            $process=[Diagnostics.Process]::Start($StartInfo)
            if($null -eq $process){throw 'Owned child was not started.'}
            $resolved=$false;$output=$null;$errors=$null
            try {
                if($process.Handle -eq [IntPtr]::Zero){throw 'Owned child handle admission failed.'}
                $output=$process.StandardOutput.ReadToEndAsync()
                $errors=$process.StandardError.ReadToEndAsync()
                if(-not $process.WaitForExit($TimeoutMilliseconds)){
                    & $StopOwnedProcess $process $output $errors
                    if(-not $process.WaitForExit(5000)){throw 'Owned child termination remains unconfirmed.'}
                    $resolved=$true
                    $timeout=[TimeoutException]::new('Owned initialization child exceeded its bounded deadline.')
                    $timeout.Data['EvidenceRoot']=$OutputRoot
                    $timeout.Data['ChildPid']=$process.Id
                    $timeout.Data['TimeoutMilliseconds']=$TimeoutMilliseconds
                    $timeout.Data['ExitCode']=$process.ExitCode
                    $timeout.Data['TerminationConfirmed']=$true
                    $timeout.Data['OutputTask']=$output
                    $timeout.Data['ErrorTask']=$errors
                    throw $timeout
                }
                $resolved=$true
                [IO.File]::WriteAllText((Join-Path $OutputRoot 'stdout.log'),$output.GetAwaiter().GetResult())
                [IO.File]::WriteAllText((Join-Path $OutputRoot 'stderr.log'),$errors.GetAwaiter().GetResult())
                [pscustomobject]@{ChildPid=$process.Id;ExitCode=$process.ExitCode}
            } catch {
                if(-not $resolved){
                    $_.Exception.Data['OwnedProcess']=$process
                    $_.Exception.Data['OutputTask']=$output
                    $_.Exception.Data['ErrorTask']=$errors
                    $_.Exception.Data['EvidenceRoot']=$OutputRoot
                }
                throw
            } finally {if($resolved){$process.Dispose()}}
        }
        function New-OwnedTimeoutStartInfo {
            $executable=Join-Path $PSHOME $(if($IsWindows){'pwsh.exe'}else{'pwsh'})
            $start=[Diagnostics.ProcessStartInfo]::new($executable)
            $start.UseShellExecute=$false;$start.CreateNoWindow=$true
            $start.RedirectStandardOutput=$true;$start.RedirectStandardError=$true
            foreach($argument in @('-NoLogo','-NoProfile','-Command',"[Console]::Out.WriteLine('owned-output');[Console]::Error.WriteLine('owned-error');[Threading.Thread]::Sleep(30000)")){$start.ArgumentList.Add($argument)}
            $start
        }
        function Complete-OwnedInitializationControl {
            param([hashtable]$State,[string]$OutputRoot,$Failure)
            if(-not $State.Process -and $Failure -and $Failure.Exception.Data['OwnedProcess']){
                $State.Process=$Failure.Exception.Data['OwnedProcess']
                $State.Handle=$State.Process.SafeHandle
                $State.Output=$Failure.Exception.Data['OutputTask'];$State.Error=$Failure.Exception.Data['ErrorTask']
            }
            if(-not $State.Process -or $State.Handle.IsClosed){return $true}
            $resolved=$false
            try {
                if(-not $State.Process.HasExited){$State.Process.Kill()}
                if(-not $State.Process.WaitForExit(5000)){throw 'Exact owned control child remains unresolved.'}
                $resolved=$true
                return $State.Process.HasExited
            } catch {
                $_.Exception.Data['OwnedProcess']=$State.Process
                $_.Exception.Data['OutputTask']=$State.Output
                $_.Exception.Data['ErrorTask']=$State.Error
                $_.Exception.Data['EvidenceRoot']=$OutputRoot
                throw
            } finally {if($resolved){$State.Process.Dispose()}}
        }
        $script:fixturePath=$PSCommandPath
        $script:producerPath=Join-Path $RepositoryRoot 'Modules/PC-AI.Acceleration/Private/Initialize-PcaiNative.ps1'
        $script:coverageHelper=Join-Path $RepositoryRoot 'Tools/Merge-PesterChildCoverage.ps1'
        $script:bindings=@(foreach($path in @($script:fixturePath,$script:producerPath,$script:coverageHelper)) { [pscustomobject]@{Path=[IO.Path]::GetFullPath($path);SHA=(Get-FileHash -LiteralPath $path).Hash} })
        $script:parentCoreType='PcaiNative.PcaiCore' -as [type]
        $script:pester=@(Get-Module Pester)
        if($script:pester.Count -ne 1 -or $script:pester[0].Version.ToString() -ne '5.7.1') { throw 'This fixture requires exactly the maintained Pester 5.7.1 package.' }
        $script:pesterManifest=Join-Path $script:pester[0].ModuleBase 'Pester.psd1'
        $manifest=Test-ModuleManifest -Path $script:pesterManifest -ErrorAction Stop
        if($manifest.Guid -ne $script:pester[0].Guid -or $manifest.Version -ne $script:pester[0].Version){throw 'Parent Pester manifest mismatch.'}
    }
    It 'admits complete actual producer contracts only from a successful isolated source-stable child' {
        $outputRoot=if($EvidenceRoot){$EvidenceRoot}else{Join-Path $TestDrive 'native-initialization-child'}
        if(Test-Path -LiteralPath $outputRoot){throw 'Retain existing child evidence; choose a fresh output directory.'}
        $null=[IO.Directory]::CreateDirectory($outputRoot)
        $request=[ordered]@{ParentPid=$PID;FixturePath=$script:fixturePath;ProducerPath=$script:producerPath;RepositoryRoot=$RepositoryRoot;EvidenceRoot=$outputRoot;PesterManifest=$script:pesterManifest;Bindings=$script:bindings}
        $requestFile=Join-Path $outputRoot 'request.json'
        $request | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $requestFile -Encoding utf8
        $executable=Join-Path $PSHOME $(if($IsWindows){'pwsh.exe'}else{'pwsh'})
        $start=[Diagnostics.ProcessStartInfo]::new($executable)
        $start.UseShellExecute=$false;$start.CreateNoWindow=$true;$start.RedirectStandardOutput=$true;$start.RedirectStandardError=$true
        foreach($argument in @('-NoLogo','-NoProfile','-File',$script:fixturePath,'-RunChild','-RequestPath',$requestFile)){$start.ArgumentList.Add($argument)}
        $scratch=Join-Path $outputRoot 'scratch';$null=[IO.Directory]::CreateDirectory($scratch)
        foreach($name in @('TEMP','TMP','TMPDIR')){$start.Environment[$name]=$scratch}
            $child=Invoke-OwnedInitializationProcess -StartInfo $start -OutputRoot $outputRoot
            $raw=Get-Content -LiteralPath (Join-Path $outputRoot 'result.json') -Raw | ConvertFrom-Json -ErrorAction Stop
            $raw.ChildPid | Should -Be $child.ChildPid;$raw.ParentPid | Should -Be $PID
            $child.ExitCode | Should -Be 0
            $raw.Pester | Should -BeExactly '5.7.1'
            $raw.Total | Should -Be 44;$raw.Passed | Should -Be 44
            foreach($count in @('Failed','Skipped','NotRun','FailedBlocks','FailedContainers')){$raw.$count | Should -Be 0}
            @($raw.Cases).Count | Should -Be 44
            @($raw.Cases | Where-Object Result -cne Passed).Count | Should -Be 0
            $raw.SourceStable | Should -BeTrue;$raw.NativeAssemblyLoaded | Should -BeFalse;$raw.ExternalCalls | Should -Be 0
            foreach($binding in $script:bindings){(Get-FileHash -LiteralPath $binding.Path).Hash | Should -BeExactly $binding.SHA}
            ('PcaiNative.PcaiCore' -as [type]) | Should -Be $script:parentCoreType
            . $script:coverageHelper
            $null=Get-PesterChildCoverageMap -Snapshot $raw.Coverage
            $raw.Coverage.Executed | Should -BeGreaterThan 0
            $currentTest=& (Get-Module Pester) {Get-CurrentTest}
            if($null -eq $currentTest){throw 'Current parent test identity is unavailable.'}
            $currentTest | Add-Member -NotePropertyName PcaiChildCoverage -NotePropertyValue ([pscustomobject]@{ParentSources=@($script:bindings | Where-Object Path -eq $script:producerPath);Child=$raw.Coverage})
    }
    It 'confirms termination before disposing an actual inert child at its timeout' {
        $state=@{Process=$null;Handle=$null;Output=$null;Error=$null;Pid=$null}
        $stop={param($OwnedProcess,$OutputTask,$ErrorTask)
            $state.Process=$OwnedProcess;$state.Handle=$OwnedProcess.SafeHandle
            $state.Pid=$OwnedProcess.Id;$state.Output=$OutputTask;$state.Error=$ErrorTask
            $OwnedProcess.Kill()
        }.GetNewClosure()
        $caught=$null
        try {Invoke-OwnedInitializationProcess -StartInfo (New-OwnedTimeoutStartInfo) -OutputRoot $TestDrive -TimeoutMilliseconds 100 -StopOwnedProcess $stop}
        catch {$caught=$_}
        try {
            $caught | Should -Not -BeNullOrEmpty
            $caught.Exception | Should -BeOfType ([TimeoutException])
            $caught.Exception.Data['ChildPid'] | Should -Be $state.Pid
            $caught.Exception.Data['TerminationConfirmed'] | Should -BeTrue
            $caught.Exception.Data.Contains('OwnedProcess') | Should -BeFalse
            [object]::ReferenceEquals($caught.Exception.Data['OutputTask'],$state.Output) | Should -BeTrue
            [object]::ReferenceEquals($caught.Exception.Data['ErrorTask'],$state.Error) | Should -BeTrue
            $state.Handle.IsClosed | Should -BeTrue
        } finally {
            # If the detecting assertions expose an unresolved-process regression,
            # recover this exact original handle rather than looking up a PID.
            $null=Complete-OwnedInitializationControl -State $state -OutputRoot $TestDrive -Failure $caught
        }
    }
    It 'retains the original owned process and both pipe tasks when injected termination is refused' {
        $state=@{Process=$null;Handle=$null;Output=$null;Error=$null}
        $refuse={param($OwnedProcess,$OutputTask,$ErrorTask)
            $state.Process=$OwnedProcess;$state.Handle=$OwnedProcess.SafeHandle
            $state.Output=$OutputTask;$state.Error=$ErrorTask
            throw 'Injected exact owned-child kill refusal.'
        }.GetNewClosure()
        $caught=$null;$closed=$false
        try {
            try {Invoke-OwnedInitializationProcess -StartInfo (New-OwnedTimeoutStartInfo) -OutputRoot $TestDrive -TimeoutMilliseconds 100 -StopOwnedProcess $refuse}
            catch {$caught=$_}
            $caught | Should -Not -BeNullOrEmpty
            $caught.Exception.Message | Should -BeExactly 'Injected exact owned-child kill refusal.'
            [object]::ReferenceEquals($caught.Exception.Data['OwnedProcess'],$state.Process) | Should -BeTrue
            [object]::ReferenceEquals($caught.Exception.Data['OutputTask'],$state.Output) | Should -BeTrue
            [object]::ReferenceEquals($caught.Exception.Data['ErrorTask'],$state.Error) | Should -BeTrue
            $caught.Exception.Data['EvidenceRoot'] | Should -BeExactly $TestDrive
            $state.Handle.IsClosed | Should -BeFalse
            $state.Process.HasExited | Should -BeFalse
        } finally {
            $closed=Complete-OwnedInitializationControl -State $state -OutputRoot $TestDrive -Failure $caught
        }
        $closed | Should -BeTrue
        $state.Handle.IsClosed | Should -BeTrue
        $state.Output.GetAwaiter().GetResult() | Should -BeOfType ([string])
        $state.Error.GetAwaiter().GetResult() | Should -BeOfType ([string])
    }
}
}
