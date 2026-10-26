#Requires -Version 7.2
<#
.SYNOPSIS
    Verifies native manifest path semantics using an explicit real DLL bundle.
.DESCRIPTION
    Run in a fresh profile-free PowerShell process. No native mocks, downloads,
    installs, or hardware collection occur. OutputRoot must be a new owned path.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$BundleRoot,
    [Parameter(Mandatory)][string]$OutputRoot,
    [string]$SourceRoot=(Join-Path $PSScriptRoot '../..')
)
$ErrorActionPreference='Stop'
$BundleRoot=(Resolve-Path -LiteralPath $BundleRoot).ProviderPath
$SourceRoot=(Resolve-Path -LiteralPath $SourceRoot).ProviderPath
if(Test-Path -LiteralPath $OutputRoot){throw 'OutputRoot must be a new owned directory.'}
if(@([AppDomain]::CurrentDomain.GetAssemblies()|Where-Object {$_.GetName().Name-eq'PcaiNative'}).Count){throw 'Run in a fresh PowerShell process.'}
[void][IO.Directory]::CreateDirectory($OutputRoot)
foreach($name in @('PCAI_CORE_LIB_PATH','PCAI_TEST_CORE_DLL','PCAI_NATIVE_DIR')){[Environment]::SetEnvironmentVariable($name,$null,'Process')}
$env:PCAI_NATIVE_BUNDLE_ROOT=$BundleRoot;$env:PCAI_ROOT=$SourceRoot;$env:PCAI_CACHE_PROVIDER='memory'
$module=Import-Module (Join-Path $SourceRoot 'Modules/PC-AI.Acceleration/PC-AI.Acceleration.psd1') -Force -PassThru
$status=Get-PcaiNativeStatus
$core=$null;[void][PcaiNative.NativeResolver]::TryGetLoadedLibraryPath('pcai_core_lib',[ref]$core)
$bridge=[PcaiNative.PcaiCore].Assembly.Location
if(-not$status.Available-or[IO.Path]::GetFullPath($core)-ine[IO.Path]::GetFullPath((Join-Path $BundleRoot (Get-Item -LiteralPath $core).Name))-or[IO.Path]::GetFullPath($bridge)-ine[IO.Path]::GetFullPath((Join-Path $BundleRoot 'PcaiNative.dll'))){throw 'Actual loaded DLL paths do not match the explicit bundle.'}
$ordinary=Join-Path $OutputRoot 'ordinary';$bracket=Join-Path $OutputRoot 'inputs[scope]';$decoy=Join-Path $OutputRoot 'inputss'
foreach($root in @($ordinary,$bracket)){
 [void][IO.Directory]::CreateDirectory((Join-Path $root '東京'))
 [IO.File]::WriteAllText((Join-Path $root 'alpha.txt'),'abc')
 [IO.File]::WriteAllText((Join-Path $root '東京/café😀.txt'),'hello',[Text.UTF8Encoding]::new($false))
}
$cases=[Collections.Generic.List[object]]::new()
foreach($case in @(@('ordinary',$ordinary),@('literal-bracket',$bracket),@('bracket-with-decoy',$bracket),@('relative-dot',$ordinary))){
 if($case[0]-eq'bracket-with-decoy'){[void][IO.Directory]::CreateDirectory($decoy);[IO.File]::WriteAllText((Join-Path $decoy 'wrong.txt'),'wrong')}
 $root=$case[1];$files=@([IO.Directory]::EnumerateFiles($root,'*',[IO.SearchOption]::AllDirectories));$directories=@([IO.Directory]::EnumerateDirectories($root,'*',[IO.SearchOption]::AllDirectories));$expected=@(($files+$directories)|ForEach-Object {[IO.Path]::GetRelativePath($root,$_).Replace('\','/')}|Sort-Object)
 $bytes=($files|ForEach-Object {[IO.FileInfo]::new($_).Length}|Measure-Object -Sum).Sum
 foreach($statsOnly in @($false,$true)){
  $failure=$null;$actual=$null
  try{
   Push-Location -LiteralPath $root
   try{$path=if($case[0]-eq'relative-dot'){'.'}else{$root};$actual=& $module {param($p,$stats) Invoke-PcaiNativeDirectoryManifest -Path $p -MaxDepth 0 -MaxResults 0 -StatsOnly:$stats} $path $statsOnly}finally{Pop-Location}
   if(-not$actual.IsSuccess-or$actual.EntriesReturned-ne$expected.Count-or$actual.FileCount-ne$files.Count-or$actual.DirectoryCount-ne$directories.Count-or$actual.TotalSize-ne$bytes){throw 'Manifest status/count/byte totals differ from the independent filesystem reference.'}
   if(-not$statsOnly){$identities=@($actual.Entries|ForEach-Object {$_.RelativePath.Replace('\','/')}|Sort-Object);if($actual.Truncated-or@(Compare-Object $expected $identities).Count){throw 'Uncapped manifest entry identities differ from the independent filesystem reference.'}}
  }catch{$failure=$_.Exception.Message}
  $cases.Add([pscustomobject]@{Case=$case[0];StatsOnly=$statsOnly;Pass=($null-eq$failure);Failure=$failure;ExpectedEntries=$expected.Count;ActualEntries=$actual.EntriesReturned})
 }
}
$invalidRejected=$false
try{& $module {param($p) Invoke-PcaiNativeDirectoryManifest -Path $p} (Join-Path $OutputRoot 'absent')|Out-Null}catch{$invalidRejected=$_.FullyQualifiedErrorId-like'*ParameterArgumentValidationError*'}
$cases.Add([pscustomobject]@{Case='absent-root-rejected';Pass=$invalidRejected})
$result=[ordered]@{NoMocks=$true;LoadedCore=$core;CoreSHA256=(Get-FileHash -LiteralPath $core).Hash;LoadedBridge=$bridge;BridgeSHA256=(Get-FileHash -LiteralPath $bridge).Hash;SourceRoot=$SourceRoot;InitializerSHA256=(Get-FileHash -LiteralPath (Join-Path $SourceRoot 'Modules/PC-AI.Acceleration/Private/Initialize-PcaiNative.ps1')).Hash;Cases=$cases.ToArray();Passed=@($cases|Where-Object Pass).Count;Failed=@($cases|Where-Object {-not$_.Pass}).Count}
$result|ConvertTo-Json -Depth 7|Set-Content -LiteralPath (Join-Path $OutputRoot 'result.json') -Encoding utf8NoBOM
Write-Output ("Native manifest paths: {0} passed, {1} failed; actual explicit DLL bundle, no mocks." -f $result.Passed,$result.Failed)
if($result.Failed){exit 1}