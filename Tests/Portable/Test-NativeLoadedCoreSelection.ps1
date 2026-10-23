#Requires -Version 7.2
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$WrapperPath,
    [string]$CompilerPath,
    [string]$Initializer=(Join-Path $PSScriptRoot '../../Modules/PC-AI.Acceleration/Private/Initialize-PcaiNative.ps1'),
    [switch]$Child,
    [string]$BundleA,
    [string]$BundleB
)
$ErrorActionPreference='Stop'
function Assert-Control([bool]$Value,[string]$Message){if(-not $Value){throw $Message}}
function Clear-Overrides {
    foreach($name in @('PCAI_CORE_LIB_PATH','PCAI_TEST_CORE_DLL','PCAI_NATIVE_DIR','PCAI_NATIVE_BUNDLE_ROOT')){[Environment]::SetEnvironmentVariable($name,$null)}
}
if($Child){
    Import-Module Microsoft.PowerShell.Management,Microsoft.PowerShell.Utility
    $PSModuleAutoLoadingPreference='None'
    Clear-Overrides
    . $Initializer
    $coreName=Get-PcaiNativeLibraryFileName
    $coreA=Join-Path $BundleA $coreName
    $coreB=Join-Path $BundleB $coreName
    $script:ProbeDiscoveryRoot=$BundleA
    function Get-PcaiNativeCandidatePaths { param($BasePaths) @($script:ProbeDiscoveryRoot) }
    $env:PCAI_CORE_LIB_PATH=$coreA
    Assert-Control (Initialize-PcaiNative) 'Actual inert core A failed required-export/magic/string admission'
    Assert-Control ([PcaiNative.PcaiCore]::Version -ceq 'inert-A') 'Core A identity probe failed'
    $loaded=$null
    Assert-Control ([PcaiNative.NativeResolver]::TryGetLoadedLibraryPath('pcai_core_lib',[ref]$loaded) -and $loaded -ceq $coreA) 'Actual resolver load record is not A'
    $script:UnexpectedAddTypeCalls=0
    function Add-Type { $script:UnexpectedAddTypeCalls++; throw 'Unexpected managed reload after actual bridge activation.' }
    function Snapshot-Provenance {
        [ordered]@{Loaded=$script:PcaiNativeLoaded;Version=$script:PcaiNativeVersion;Directory=$script:PcaiNativeDllPath;Core=$script:PcaiNativeCoreLibraryPath;Bridge=$script:PcaiNativeWrapperPath;Hash=$script:PcaiNativeArtifactHash;Rid=$script:PcaiNativeRid}|ConvertTo-Json -Compress
    }
    foreach($selection in @('PCAI_CORE_LIB_PATH','PCAI_TEST_CORE_DLL','PCAI_NATIVE_DIR','PCAI_NATIVE_BUNDLE_ROOT','default')){
        foreach($force in @($false,$true)){
            Clear-Overrides
            $script:ProbeDiscoveryRoot=if($selection -ceq 'default'){$BundleB}else{$BundleA}
            if($selection -cne 'default'){
                $value=if($selection -in @('PCAI_NATIVE_DIR','PCAI_NATIVE_BUNDLE_ROOT')){$BundleB}else{$coreB}
                [Environment]::SetEnvironmentVariable($selection,$value)
                Assert-Control (-not [PcaiNative.PcaiCore]::IsAvailable) "Managed availability relabeled A as B: $selection"
            }
            $before=Snapshot-Provenance
            $beforePath=$env:PATH
            $beforeEnvironment=@(foreach($name in @('PCAI_CORE_LIB_PATH','PCAI_TEST_CORE_DLL','PCAI_NATIVE_DIR','PCAI_NATIVE_BUNDLE_ROOT')){[Environment]::GetEnvironmentVariable($name)})|ConvertTo-Json -Compress
            Assert-Control (-not (Initialize-PcaiNative -Force:$force -WarningAction SilentlyContinue)) "Selected B admitted: $selection Force=$force"
            Assert-Control (-not (Test-PcaiNativeAvailable -WarningAction SilentlyContinue)) "Availability wrapper returned stale A success: $selection"
            Assert-Control ((Snapshot-Provenance) -ceq $before) "Refusal relabeled provenance: $selection"
            Assert-Control ($env:PATH -ceq $beforePath) "Refusal mutated PATH: $selection"
            $afterEnvironment=@(foreach($name in @('PCAI_CORE_LIB_PATH','PCAI_TEST_CORE_DLL','PCAI_NATIVE_DIR','PCAI_NATIVE_BUNDLE_ROOT')){[Environment]::GetEnvironmentVariable($name)})|ConvertTo-Json -Compress
            Assert-Control ($afterEnvironment -ceq $beforeEnvironment) "Refusal mutated selection environment: $selection"
            Assert-Control ($script:UnexpectedAddTypeCalls -eq 0) 'Refusal attempted another managed load'
            Assert-Control ([PcaiNative.NativeResolver]::TryGetLoadedLibraryPath('pcai_core_lib',[ref]$loaded) -and $loaded -ceq $coreA) 'Loaded native path changed after refusal'
            Clear-Overrides
            $env:PCAI_CORE_LIB_PATH=$coreA
            $script:ProbeDiscoveryRoot=$BundleA
            Assert-Control (Initialize-PcaiNative -Force:$force) "Same A was refused: Force=$force"
            Assert-Control ([PcaiNative.PcaiCore]::IsAvailable -and [PcaiNative.PcaiCore]::Version -ceq 'inert-A') 'Restored A availability failed'
        }
    }
    # Dot-sourcing resets every script flag/hash. Actual resolver load custody
    # must still reject B, and same-A reuse must not invent a new disk hash.
    foreach($force in @($false,$true)){
        . $Initializer
        $script:ProbeDiscoveryRoot=$BundleA
        function Get-PcaiNativeCandidatePaths { param($BasePaths) @($script:ProbeDiscoveryRoot) }
        Clear-Overrides
        $env:PCAI_CORE_LIB_PATH=$coreB
        $before=Snapshot-Provenance
        $beforePath=$env:PATH
        Assert-Control (-not (Initialize-PcaiNative -Force:$force -WarningAction SilentlyContinue)) 'Module reimport forgot actual native binding'
        Assert-Control ((Snapshot-Provenance) -ceq $before -and $env:PATH -ceq $beforePath) 'Module-reimport refusal mutated state'
        $env:PCAI_CORE_LIB_PATH=$coreA
        Assert-Control (Initialize-PcaiNative -Force:$force) 'Same A after module reimport was refused'
        Assert-Control ($null -eq $script:PcaiNativeArtifactHash) 'Module reimport invented a new loaded-artifact hash'
    }
    Write-Output 'PASS: 10 actual loaded-A/valid-B refusals, same-A reuse, managed availability, and two module-reimport cases; no PATH/environment/provenance/Add-Type mutation on refusal'
    Write-Output 'LIMIT: inert C ABI fixture only; no production Rust/native runtime, GPU or model qualification'
    exit 0
}

if(-not $CompilerPath){$CompilerPath=(Get-Command clang -CommandType Application -ErrorAction Stop).Source}
$WrapperPath=(Resolve-Path -LiteralPath $WrapperPath).ProviderPath
$Initializer=(Resolve-Path -LiteralPath $Initializer).ProviderPath
$owned=Join-Path ([IO.Path]::GetTempPath()) ('pcai-loaded-core-'+[guid]::NewGuid().ToString('N'))
[void][IO.Directory]::CreateDirectory($owned)
try{
    Add-Type -Path $WrapperPath
    $coreName=[PcaiNative.NativeResolver]::GetExpectedFileName('pcai_core_lib')
    foreach($identity in @('A','B')){
        $bundle=Join-Path $owned $identity
        [void][IO.Directory]::CreateDirectory($bundle)
        Copy-Item -LiteralPath $WrapperPath -Destination (Join-Path $bundle 'PcaiNative.dll')
        $export=if($IsWindows){'__declspec(dllexport)'}else{'__attribute__((visibility("default")))'}
        $code=[Collections.Generic.List[string]]::new()
        $code.Add("$export unsigned int pcai_core_test(void) { return 0x50434149u; }")
        $code.Add("$export const char* pcai_core_version(void) { return `"inert-$identity`"; }")
        $code.Add("$export const char* pcai_string_copy(const char* input) { return `"pcai`"; }")
        $code.Add("$export void pcai_free_string(void* input) { }")
        foreach($name in [PcaiNative.PcaiCore]::RequiredExports){
            if($name -notin @('pcai_core_test','pcai_core_version','pcai_string_copy','pcai_free_string')){$code.Add("$export void $name(void) { }")}
        }
        $source=Join-Path $bundle 'inert-core.c'
        [IO.File]::WriteAllText($source,($code -join "`n"),[Text.UTF8Encoding]::new($false))
        $start=[Diagnostics.ProcessStartInfo]::new($CompilerPath)
        $start.UseShellExecute=$false;$start.CreateNoWindow=$true
        $arguments=if($IsWindows){@('--target=x86_64-pc-windows-msvc','-shared','-nostdlib','-fuse-ld=lld','-Wl,/noentry',$source,'-o',(Join-Path $bundle $coreName))}else{@('-shared','-fPIC','-nostdlib',$source,'-o',(Join-Path $bundle $coreName))}
        foreach($argument in $arguments){[void]$start.ArgumentList.Add($argument)}
        $compiler=[Diagnostics.Process]::Start($start)
        try{if(-not $compiler.WaitForExit(30000)){$compiler.Kill($true);throw 'Inert C fixture compilation exceeded30s'};Assert-Control ($compiler.ExitCode -eq 0) 'Inert C fixture compilation failed'}finally{$compiler.Dispose()}
    }
    $start=[Diagnostics.ProcessStartInfo]::new((Get-Process -Id $PID).Path)
    $start.UseShellExecute=$false;$start.CreateNoWindow=$true
    $start.RedirectStandardOutput=$true;$start.RedirectStandardError=$true
    foreach($argument in @('-NoLogo','-NoProfile','-File',$PSCommandPath,'-WrapperPath',$WrapperPath,'-Initializer',$Initializer,'-Child','-BundleA',(Join-Path $owned 'A'),'-BundleB',(Join-Path $owned 'B'))){[void]$start.ArgumentList.Add($argument)}
    $probe=[Diagnostics.Process]::Start($start)
    $stdout=$probe.StandardOutput.ReadToEndAsync();$stderr=$probe.StandardError.ReadToEndAsync()
    try{
        if(-not $probe.WaitForExit(45000)){$probe.Kill($true);throw 'Inert loaded-core probe exceeded45s'}
        Assert-Control ($stdout.Wait(5000) -and $stderr.Wait(5000)) 'Inert probe readers exceeded their bound'
        Write-Output $stdout.GetAwaiter().GetResult()
        $errors=$stderr.GetAwaiter().GetResult()
        if($errors){Write-Output $errors}
        Assert-Control ($probe.ExitCode -eq 0) 'Inert loaded-core probe failed'
    }finally{$probe.Dispose()}
}finally{
    $resolved=[IO.Path]::GetFullPath($owned)
    Assert-Control ($resolved.StartsWith([IO.Path]::GetFullPath([IO.Path]::GetTempPath()),[StringComparison]::OrdinalIgnoreCase)) 'Fixture cleanup escaped owned temp root'
    Remove-Item -LiteralPath $resolved -Recurse -Force
}
