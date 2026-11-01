#Requires -Version 7.0
param([string]$ImporterSourcePath)

BeforeAll {
    $script:ImportSource = if ($ImporterSourcePath) { $ImporterSourcePath }
        else { Join-Path $PSScriptRoot '../../Modules/PC-AI.Common/Public/Import-PcaiAccelerationStack.ps1' }
    $script:CapabilityMap = [ordered]@{
        'Find-CommandCached' = 'CommandLookupAvailable'
        'Find-CommandNative' = 'NativeCommandLookupAvailable'
        'Invoke-ProfileDispatch' = 'DispatchAvailable'
        'Find-FilesFast' = 'FileSearchAvailable'
        'Search-ContentFast' = 'ContentSearchAvailable'
        'Test-PcaiNativeBridge' = 'NativeBridgeAvailable'
    }
    $script:ColdResults = [Collections.Generic.List[object]]::new()
    function New-OwnedFakeModule {
        param([string]$Root, [string]$Name, [string[]]$Commands)
        $directory = Join-Path $Root $Name
        [void][IO.Directory]::CreateDirectory($directory)
        $definition = "[IO.File]::AppendAllText(`$env:PCAI_TEST_IMPORT_MARKER, '$Name'+[Environment]::NewLine)`n"
        foreach ($command in $Commands) {
            if ($command -notmatch '^[A-Za-z][A-Za-z0-9-]+$') { throw 'Literal fixture command required.' }
            $definition += if ($command -ceq 'Resolve-PcaiRepoRoot') {
                "function Resolve-PcaiRepoRoot { param(`$StartPath) return `$env:PCAI_TEST_REPO_ROOT }`n"
            } else { "function $command { return 'owned-fixture-only' }`n" }
        }
        $exports = @($Commands | ForEach-Object { "'$_'" }) -join ','
        $definition += if ($Commands.Count) { "Export-ModuleMember -Function @($exports)`n" }
            else { "Export-ModuleMember -Function @()`n" }
        [IO.File]::WriteAllText((Join-Path $directory ($Name+'.psm1')), $definition, [Text.UTF8Encoding]::new($false))
        $manifest = "@{RootModule='$Name.psm1';ModuleVersion='1.0.0';FunctionsToExport=@($exports);CmdletsToExport=@();AliasesToExport=@();VariablesToExport=@()}`n"
        [IO.File]::WriteAllText((Join-Path $directory ($Name+'.psd1')), $manifest, [Text.UTF8Encoding]::new($false))
    }
    function Invoke-OwnedColdImport {
        param([string]$Mode, [string]$OptionalCommand)
        $fixture = Join-Path $TestDrive ([guid]::NewGuid().ToString('N'))
        $moduleRoot = Join-Path $fixture 'modules'
        $repoRoot = Join-Path $fixture 'repo'
        $userRoot = Join-Path $fixture 'user'
        foreach ($path in @($fixture,$moduleRoot,$repoRoot,$userRoot,(Join-Path $fixture 'temp'))) {
            [void][IO.Directory]::CreateDirectory($path)
        }
        New-OwnedFakeModule -Root $moduleRoot -Name 'ProfileAccelerator' -Commands @()
        if ($Mode -ceq 'DefaultStack') {
            New-OwnedFakeModule -Root $moduleRoot -Name 'PcaiNativeBridge' -Commands @('Find-CommandNative','Test-PcaiNativeBridge')
            New-OwnedFakeModule -Root $moduleRoot -Name 'PC-AI.Acceleration' -Commands @('Find-FilesFast','Search-ContentFast')
            # This is the explicitly requested first module, with its genuine exports.
            $profileDirectory = Join-Path $moduleRoot 'ProfileAccelerator'
            [IO.File]::WriteAllText((Join-Path $profileDirectory 'ProfileAccelerator.psm1'), "[IO.File]::AppendAllText(`$env:PCAI_TEST_IMPORT_MARKER, 'ProfileAccelerator'+[Environment]::NewLine)`nfunction Find-CommandCached { 'owned-fixture-only' }`nfunction Invoke-ProfileDispatch { 'owned-fixture-only' }`nExport-ModuleMember -Function Find-CommandCached,Invoke-ProfileDispatch`n", [Text.UTF8Encoding]::new($false))
            [IO.File]::WriteAllText((Join-Path $profileDirectory 'ProfileAccelerator.psd1'), "@{RootModule='ProfileAccelerator.psm1';ModuleVersion='1.0.0';FunctionsToExport=@('Find-CommandCached','Invoke-ProfileDispatch');CmdletsToExport=@();AliasesToExport=@();VariablesToExport=@()}`n", [Text.UTF8Encoding]::new($false))
        } elseif ($Mode -ceq 'Resolver') {
            New-OwnedFakeModule -Root $moduleRoot -Name 'FixtureRepoResolver' -Commands @('Resolve-PcaiRepoRoot')
        } elseif ($Mode -ceq 'Preloaded' -or $Mode -ceq 'Partial') {
            New-OwnedFakeModule -Root $moduleRoot -Name 'FixtureCapabilityProvider' -Commands @($script:CapabilityMap.Keys)
        } elseif ($Mode -ceq 'Optional') {
            New-OwnedFakeModule -Root $moduleRoot -Name 'FixtureCapabilityProvider' -Commands @($OptionalCommand)
        }
        $driver = Join-Path $fixture 'Invoke-Child.ps1'
        $driverBody = @'
param([string]$SourcePath,[string]$Mode)
$ErrorActionPreference='Stop'
$PSModuleAutoLoadingPreference='All'
function Write-FixtureStage([string]$Name){[IO.File]::AppendAllText($env:PCAI_TEST_STAGE_MARKER,$Name+'|'+[DateTimeOffset]::UtcNow.ToString('o')+[Environment]::NewLine)}
Write-FixtureStage 'CHILD_ENTER'
$modulePathBefore=@($env:PSModulePath -split [IO.Path]::PathSeparator)
$env:PSModulePath=$env:PCAI_TEST_MODULE_PATH
if($env:PSModulePath -cne $env:PCAI_TEST_MODULE_PATH){throw 'Exact child-only module path required'}
Write-FixtureStage 'MODULE_PATH_RESET'
Write-FixtureStage 'SOURCE_LOAD_ENTER'
. $SourcePath
Write-FixtureStage 'SOURCE_LOAD_COMPLETE'
$known=@('ProfileAccelerator','PcaiNativeBridge','PC-AI.Acceleration','FixtureCapabilityProvider','FixtureRepoResolver')
if(@(Get-Module -Name $known).Count){throw 'Cold fixture must not preload known modules'}
if($Mode -ceq 'Preloaded'){Import-Module FixtureCapabilityProvider -ErrorAction Stop}
$parameters=@{Modules=@('ProfileAccelerator');RepoRoot=$env:PCAI_TEST_REPO_ROOT}
if($Mode -ceq 'DefaultStack'){$parameters.Remove('Modules')}
if($Mode -ceq 'Resolver'){$parameters.Remove('RepoRoot')}
if($Mode -cin @('Missing','RequireAll')){$parameters.Modules=@('DefinitelyMissingAccelerationFixtureModule')}
if($Mode -ceq 'RequireAll'){$parameters.RequireAll=$true}
$failure=$false;$result=$null
$clock=[Diagnostics.Stopwatch]::StartNew()
try{
 Write-FixtureStage 'IMPORTER_ENTER';$result=Import-PcaiAccelerationStack @parameters;Write-FixtureStage 'IMPORTER_RETURN'
 if($Mode -ceq 'Force'){$parameters.Force=$true;Write-FixtureStage 'IMPORTER_ENTER';$result=Import-PcaiAccelerationStack @parameters;Write-FixtureStage 'IMPORTER_RETURN'}
}catch{$failure=$true}
$clock.Stop()
Write-FixtureStage 'INSPECT_ENTER'
$loadedPaths=@(Get-Module -Name $known|ForEach-Object{$_.Path})
$ownedModulePrefix=[IO.Path]::GetFullPath($env:POWERSHELL_MODULES_PATH).TrimEnd([IO.Path]::DirectorySeparatorChar)+[IO.Path]::DirectorySeparatorChar
foreach($modulePath in $loadedPaths){if(-not [IO.Path]::GetFullPath($modulePath).StartsWith($ownedModulePrefix,[StringComparison]::OrdinalIgnoreCase)){throw 'Imported module outside exact owned fixtures'}}
Write-FixtureStage 'OWNED_MODULE_PATHS_VERIFIED'
$projection=$null
if($result){
 $moduleFields=[ordered]@{}
 foreach($property in $result.Modules.PSObject.Properties){$value=$property.Value;$moduleFields[$property.Name]=@{Available=$value.Available;Imported=$value.Imported;Source=$value.Source;Path=$value.Path}}
 $projection=@{RepoRoot=$result.RepoRoot;Modules=$moduleFields}
 foreach($field in @('CommandLookupAvailable','NativeCommandLookupAvailable','DispatchAvailable','FileSearchAvailable','ContentSearchAvailable','NativeBridgeAvailable')){$projection[$field]=$result.$field}
}
[string[]]$imports=@();if([IO.File]::Exists($env:PCAI_TEST_IMPORT_MARKER)){$imports=[IO.File]::ReadAllLines($env:PCAI_TEST_IMPORT_MARKER)}
[ordered]@{Mode=$Mode;Failed=$failure;Result=$projection;Imports=$imports;Loaded=@(Get-Module -Name $known|Select-Object -ExpandProperty Name);LoadedPaths=$loadedPaths;ModulePathsOwned=$true;ElapsedMilliseconds=$clock.Elapsed.TotalMilliseconds;SourceSHA256=(Get-FileHash -LiteralPath $SourcePath).Hash;ModulePathBefore=$modulePathBefore;ModulePathAfter=@($env:PSModulePath -split [IO.Path]::PathSeparator);Scope='Actual owned module-path-clamped diagnostic; no production performance acceptance'}|ConvertTo-Json -Depth 8 -Compress
'@
        [IO.File]::WriteAllText($driver,$driverBody,[Text.UTF8Encoding]::new($false))
        $psi=[Diagnostics.ProcessStartInfo]::new()
        $psi.FileName=Join-Path $PSHOME 'pwsh.exe'
        if(-not [IO.File]::Exists($psi.FileName)){$psi.FileName=Join-Path $PSHOME 'pwsh'}
        $psi.UseShellExecute=$false;$psi.CreateNoWindow=$true
        $psi.RedirectStandardOutput=$true;$psi.RedirectStandardError=$true
        foreach($argument in @('-NoLogo','-NoProfile','-NonInteractive','-File',$driver,'-SourcePath',$script:ImportSource,'-Mode',$Mode)){$psi.ArgumentList.Add($argument)}
        # Child-only paths: prohibit user/repo-installed modules from satisfying probes.
        $systemModules=@((Join-Path $PSHOME 'Modules'))
        # Core system modules satisfy all fixture operations; exclude unrelated legacy discovery.
        $psi.Environment['PSModulePath']=(@($moduleRoot)+$systemModules)-join[IO.Path]::PathSeparator
        $psi.Environment['PCAI_TEST_MODULE_PATH']=$psi.Environment['PSModulePath']
        $psi.Environment['PCAI_TEST_STAGE_MARKER']=Join-Path $fixture 'child-stages.txt'
        $psi.Environment['PSModuleAnalysisCachePath']=Join-Path $fixture 'analysis-cache'
        $psi.Environment['POWERSHELL_MODULES_PATH']=$moduleRoot
        $psi.Environment['PCAI_ROOT']=$repoRoot
        $psi.Environment['USERPROFILE']=$userRoot
        $psi.Environment['LOCALAPPDATA']=Join-Path $userRoot 'local'
        $psi.Environment['TEMP']=Join-Path $fixture 'temp';$psi.Environment['TMP']=$psi.Environment['TEMP']
        $psi.Environment['PCAI_TEST_IMPORT_MARKER']=Join-Path $fixture 'imports.txt'
        $psi.Environment['PCAI_TEST_REPO_ROOT']=$repoRoot
        $process=$null;$processHandle=$null;$stdoutHandle=$null;$stderrHandle=$null
        $caught=$null;$secondary=[Collections.Generic.List[object]]::new();$data=$null;$stdout='';$stderr='';$cpuMilliseconds=$null;$childPid=$null;$childBirth=$null;$stdoutTask=$null;$stderrTask=$null;$stdoutStream=$null;$stderrStream=$null
        $natural=$false;$stdoutDisposed=$false;$stderrDisposed=$false;$processDisposed=$false
        $startMilliseconds=$null;$waitMilliseconds=$null;$cleanupMilliseconds=$null
        $outer=[Diagnostics.Stopwatch]::StartNew()
        try{
            $startClock=[Diagnostics.Stopwatch]::StartNew()
            $process=[Diagnostics.Process]::Start($psi)
            $startClock.Stop();$startMilliseconds=$startClock.Elapsed.TotalMilliseconds
            $childPid=$process.Id;$childBirth=$process.StartTime.ToUniversalTime().ToString('o')
            $processHandle=$process.SafeHandle
            $stdoutStream=$process.StandardOutput.BaseStream;$stderrStream=$process.StandardError.BaseStream
            if($stdoutStream -is [IO.FileStream]){$stdoutHandle=$stdoutStream.SafeFileHandle}
            elseif($stdoutStream -is [IO.Pipes.PipeStream]){$stdoutHandle=$stdoutStream.SafePipeHandle}
            else{throw 'Unsupported original stdout stream type'}
            if($stderrStream -is [IO.FileStream]){$stderrHandle=$stderrStream.SafeFileHandle}
            elseif($stderrStream -is [IO.Pipes.PipeStream]){$stderrHandle=$stderrStream.SafePipeHandle}
            else{throw 'Unsupported original stderr stream type'}
            $stdoutTask=$process.StandardOutput.ReadToEndAsync();$stderrTask=$process.StandardError.ReadToEndAsync()
            $waitClock=[Diagnostics.Stopwatch]::StartNew()
            if(-not $process.WaitForExit(30000)){throw 'Owned cold fixture exceeded30s'}
            $natural=$true
            $stdout=$stdoutTask.GetAwaiter().GetResult();$stderr=$stderrTask.GetAwaiter().GetResult()
            $waitClock.Stop();$waitMilliseconds=$waitClock.Elapsed.TotalMilliseconds
            if($process.ExitCode -ne 0 -or -not [string]::IsNullOrWhiteSpace($stderr)){throw 'Owned cold fixture child failed; no stderr is published'}
            $data=$stdout|ConvertFrom-Json
            $outer.Stop();$data|Add-Member -NotePropertyName ChildMilliseconds -NotePropertyValue $outer.Elapsed.TotalMilliseconds
        }catch{$caught=$_}finally{
            try{if($process){$cpuMilliseconds=$process.TotalProcessorTime.TotalMilliseconds}}catch{}
            $cleanupClock=[Diagnostics.Stopwatch]::StartNew()
            if($process){
                try{if(-not $natural -and -not $process.HasExited){$process.Kill($true);if(-not $process.WaitForExit(5000)){throw 'Owned child remains pending after bounded stop'}}}catch{if($null-eq$caught){$caught=$_}else{$secondary.Add($_)}}
                try{$process.StandardOutput.Dispose();$stdoutDisposed=$true}catch{if($null-eq$caught){$caught=$_}else{$secondary.Add($_)}}
                try{$process.StandardError.Dispose();$stderrDisposed=$true}catch{if($null-eq$caught){$caught=$_}else{$secondary.Add($_)}}
                try{$process.Dispose();$processDisposed=$true}catch{if($null-eq$caught){$caught=$_}else{$secondary.Add($_)}}
            }
            foreach($handle in @($stdoutHandle,$stderrHandle,$processHandle)){
                try{if($null-ne$handle){$handle.Dispose()}}catch{if($null-eq$caught){$caught=$_}else{$secondary.Add($_)}}
            }
            $cleanupClock.Stop();$cleanupMilliseconds=$cleanupClock.Elapsed.TotalMilliseconds
        }
        if($caught){
            try{if($stdoutTask){$stdout=$stdoutTask.GetAwaiter().GetResult()}}catch{}
            try{if($stderrTask){$stderr=$stderrTask.GetAwaiter().GetResult()}}catch{}
        }
        if($env:PCAI_AUTOLOAD_EVIDENCE_PATH){
            $evidenceRoot=[IO.Path]::GetDirectoryName($env:PCAI_AUTOLOAD_EVIDENCE_PATH)
            $label=($Mode+'-'+$OptionalCommand).TrimEnd('-')
            $files=@(@{Leaf=$label+'-stdout.txt';Text=$stdout},@{Leaf=$label+'-stderr.txt';Text=$stderr},@{Leaf=$label+'-stages.txt';Text=if([IO.File]::Exists($psi.Environment['PCAI_TEST_STAGE_MARKER'])){[IO.File]::ReadAllText($psi.Environment['PCAI_TEST_STAGE_MARKER'])}else{''}})
            foreach($item in $files){try{$path=Join-Path $evidenceRoot $item.Leaf;if([IO.File]::Exists($path)){throw 'Preserve diagnostic stream evidence'};[IO.File]::WriteAllText($path,$item.Text,[Text.UTF8Encoding]::new($false))}catch{if($null-eq$caught){$caught=$_}else{$secondary.Add($_)}}}
        }
        if($null -eq $data){$data=[pscustomobject]@{Mode=$Mode;Failed=$true;Result=$null;Imports=@();Loaded=@();ElapsedMilliseconds=$null;SourceSHA256=(Get-FileHash -LiteralPath $script:ImportSource).Hash;ChildMilliseconds=$outer.Elapsed.TotalMilliseconds}}
        $data|Add-Member -NotePropertyName ChildIdentity -NotePropertyValue ([pscustomobject]@{PID=$childPid;BirthUtc=$childBirth;CPUMilliseconds=$cpuMilliseconds})
        $closure=[ordered]@{NaturalExit=$natural;StdoutDisposalReturned=$stdoutDisposed;StderrDisposalReturned=$stderrDisposed;ProcessDisposalReturned=$processDisposed;OriginalStdoutHandleClosed=$stdoutHandle.IsClosed;OriginalStderrHandleClosed=$stderrHandle.IsClosed;OriginalProcessHandleClosed=$processHandle.IsClosed}
        $data|Add-Member -NotePropertyName Closure -NotePropertyValue ([pscustomobject]$closure)
        $data|Add-Member -NotePropertyName StreamTypes -NotePropertyValue ([pscustomobject]@{Stdout=$stdoutStream.GetType().FullName;Stderr=$stderrStream.GetType().FullName})
        $data|Add-Member -NotePropertyName Timing -NotePropertyValue ([pscustomobject]@{ProcessStartCallMilliseconds=$startMilliseconds;NaturalChildAndDrainWaitMilliseconds=$waitMilliseconds;ObserverCleanupMilliseconds=$cleanupMilliseconds;ImporterCallMilliseconds=$data.ElapsedMilliseconds;Scope='One complete synthetic fake-module scenario; no attribution to system startup or fleet responsiveness'})
        $data|Add-Member -NotePropertyName FailureEvidence -NotePropertyValue ([pscustomobject]@{FirstErrorType=if($caught){$caught.Exception.GetType().FullName}else{$null};SecondaryErrorCount=$secondary.Count;AllClosureFlagsTrue=(@($closure.Values|Where-Object{$_ -ne $true}).Count-eq0)})
        $script:ColdResults.Add($data)
        if($caught){throw $caught}
        if(@($closure.Values|Where-Object{$_ -ne $true}).Count){throw 'Actual owned fixture handles must close'}
        return $data
    }
}

Describe 'Acceleration observational capability probes preserve import intent' -Tag 'Unit','Acceleration','Portable' {
    It 'does not autoload an unrelated provider for <Command>' -ForEach @(
        @{Command='Find-CommandCached';Capability='CommandLookupAvailable'},
        @{Command='Find-CommandNative';Capability='NativeCommandLookupAvailable'},
        @{Command='Invoke-ProfileDispatch';Capability='DispatchAvailable'},
        @{Command='Find-FilesFast';Capability='FileSearchAvailable'},
        @{Command='Search-ContentFast';Capability='ContentSearchAvailable'},
        @{Command='Test-PcaiNativeBridge';Capability='NativeBridgeAvailable'}
    ) {
        $actual=Invoke-OwnedColdImport -Mode Optional -OptionalCommand $Command
        $actual.Failed|Should -BeFalse
        $actual.Result.$Capability|Should -BeFalse
        @($actual.Loaded)|Should -Be @('ProfileAccelerator')
        @($actual.Imports)|Should -Be @('ProfileAccelerator')
    }
    It 'does not load an unrelated all-capability module for a partial requested stack' {
        $actual=Invoke-OwnedColdImport -Mode Partial
        $actual.Failed|Should -BeFalse
        @($actual.Loaded)|Should -Be @('ProfileAccelerator')
        @($actual.Imports)|Should -Be @('ProfileAccelerator')
        foreach($field in $script:CapabilityMap.Values){$actual.Result.$field|Should -BeFalse}
    }
    It 'does not autoload an optional repository resolver' {
        $actual=Invoke-OwnedColdImport -Mode Resolver
        $actual.Failed|Should -BeFalse
        @($actual.Loaded)|Should -Be @('ProfileAccelerator')
        @($actual.Imports)|Should -Be @('ProfileAccelerator')
        $actual.Result.RepoRoot|Should -Not -BeNullOrEmpty
    }
    It 'retains every capability from the explicitly requested default fake stack' {
        $actual=Invoke-OwnedColdImport -Mode DefaultStack
        $actual.Failed|Should -BeFalse
        @($actual.Imports)|Should -Be @('ProfileAccelerator','PcaiNativeBridge','PC-AI.Acceleration')
        foreach($field in $script:CapabilityMap.Values){$actual.Result.$field|Should -BeTrue}
        foreach($module in @('ProfileAccelerator','PcaiNativeBridge','PC-AI.Acceleration')){$actual.Result.Modules.$module.Available|Should -BeTrue}
    }
    It 'retains capabilities that were already explicitly imported' {
        $actual=Invoke-OwnedColdImport -Mode Preloaded
        $actual.Failed|Should -BeFalse
        @($actual.Imports)|Should -Be @('FixtureCapabilityProvider','ProfileAccelerator')
        foreach($field in $script:CapabilityMap.Values){$actual.Result.$field|Should -BeTrue}
    }
    It 'retains missing-module results without RequireAll' {
        $actual=Invoke-OwnedColdImport -Mode Missing
        $actual.Failed|Should -BeFalse
        $actual.Result.Modules.DefinitelyMissingAccelerationFixtureModule.Available|Should -BeFalse
        $actual.Result.Modules.DefinitelyMissingAccelerationFixtureModule.Source|Should -Be missing
        @($actual.Imports)|Should -HaveCount 0
    }
    It 'retains RequireAll refusal for a genuinely missing module' {
        $actual=Invoke-OwnedColdImport -Mode RequireAll
        $actual.Failed|Should -BeTrue
        $actual.Result|Should -BeNullOrEmpty
        @($actual.Imports)|Should -HaveCount 0
    }
    It 'retains explicit Force import behavior' {
        $actual=Invoke-OwnedColdImport -Mode Force
        $actual.Failed|Should -BeFalse
        @($actual.Imports)|Should -Be @('ProfileAccelerator','ProfileAccelerator')
        $actual.Result.Modules.ProfileAccelerator.Imported|Should -BeTrue
    }
}

AfterAll {
    if($env:PCAI_AUTOLOAD_EVIDENCE_PATH){
        [ordered]@{Scope='Actual owned cold fake-module scenarios only';SourcePath=$script:ImportSource;SourceSHA256=(Get-FileHash -LiteralPath $script:ImportSource).Hash;Cases=$script:ColdResults.ToArray();ArtificialDelay=0;BootstrapIsolation='Require per-case ModulePathsOwned and actual loaded paths'}|ConvertTo-Json -Depth 12|Set-Content -LiteralPath $env:PCAI_AUTOLOAD_EVIDENCE_PATH -Encoding utf8
    }
}
