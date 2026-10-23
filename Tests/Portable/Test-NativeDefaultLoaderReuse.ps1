#Requires -Version 7.2
[CmdletBinding()]
param([Parameter(Mandatory)][string]$WrapperPath,[string]$CompilerPath,[switch]$Child,[string]$OwnedRoot)
$ErrorActionPreference='Stop'
function Assert-Control([bool]$Value,[string]$Message){if(-not $Value){throw $Message}}
if($Child){
    Import-Module Microsoft.PowerShell.Management,Microsoft.PowerShell.Utility
    $PSModuleAutoLoadingPreference='None'
    Set-Location -LiteralPath $OwnedRoot
    foreach($name in @('PCAI_ROOT','PCAI_CORE_LIB_PATH','PCAI_TEST_CORE_DLL','PCAI_NATIVE_DIR','PCAI_NATIVE_BUNDLE_ROOT')){[Environment]::SetEnvironmentVariable($name,$null)}
    Add-Type -Path $WrapperPath
    $callers=@(foreach($id in @('First','Second')){
        $source=@"
using System;
using System.Runtime.InteropServices;
public static class Default$id {
 [DllImport("pcai_media",EntryPoint="pcai_core_test",CallingConvention=CallingConvention.Cdecl)] public static extern uint Test();
 [DllImport("pcai_media",EntryPoint="pcai_string_copy",CallingConvention=CallingConvention.Cdecl)] public static extern IntPtr Copy();
 [DllImport("pcai_media",EntryPoint="pcai_core_version",CallingConvention=CallingConvention.Cdecl)] public static extern IntPtr Version();
}
"@
        $assemblyPath=Join-Path $OwnedRoot ("Caller$id.dll")
        Add-Type -TypeDefinition $source -OutputAssembly $assemblyPath
        $assembly=[Reflection.Assembly]::LoadFrom($assemblyPath)
        [PcaiNative.NativeResolver]::EnsureRegistered($assembly)
        $path=$null;$provenance=$null
        $discovered=[PcaiNative.NativeResolver]::TryResolveLibraryPath('pcai_media',[ref]$path,[ref]$provenance,$assembly)
        Assert-Control (-not $discovered) "Fixture did not isolate default-loader discovery: $provenance $path"
        Assert-Control ($null -eq $provenance) 'Fixture encountered authoritative selection'
        $assembly.GetType("Default$id")
    })
    $first=$callers[0];$second=$callers[1]
    Assert-Control ($first.GetMethod('Test').Invoke($null,@()) -eq 0x50434149) 'First default-loader import failed'
    $loaded=$null
    Assert-Control ([PcaiNative.NativeResolver]::TryGetLoadedLibraryPath('pcai_media',[ref]$loaded) -and $loaded -ceq 'pcai_media') 'Default load fabricated a filesystem identity'
    Write-Output 'PASS: first actual default-loader import; candidate discovery absent and logical provenance retained'
    Assert-Control ([Runtime.InteropServices.Marshal]::PtrToStringUTF8($first.GetMethod('Copy').Invoke($null,@())) -ceq 'default-copy') 'Second distinct default-loader import failed'
    Assert-Control ([Runtime.InteropServices.Marshal]::PtrToStringUTF8($first.GetMethod('Version').Invoke($null,@())) -ceq 'default-version') 'Third distinct default-loader import failed'
    Assert-Control ([Runtime.InteropServices.Marshal]::PtrToStringUTF8($second.GetMethod('Version').Invoke($null,@())) -ceq 'default-version') 'Second registered caller failed'
    $resolve=[PcaiNative.NativeResolver].GetMethod('Resolve',[Reflection.BindingFlags]'Static,NonPublic')
    $handle=$resolve.Invoke($null,@('pcai_media',$first.Assembly,$null))
    Assert-Control ($handle -ne [IntPtr]::Zero) 'Repeated callback lost loaded handle'
    Assert-Control ($resolve.Invoke($null,@('pcai_media',$second.Assembly,$null)) -eq $handle) 'Second assembly did not reuse exact loaded handle'
    Write-Output 'PASS: distinct exports, second registered assembly and repeated resolver callbacks reuse actual default handle'
    foreach($selection in @('PCAI_NATIVE_DIR','PCAI_NATIVE_BUNDLE_ROOT')){
        $value=if($selection -in @('PCAI_NATIVE_DIR','PCAI_NATIVE_BUNDLE_ROOT')){Join-Path $OwnedRoot 'foreign'}else{Join-Path (Join-Path $OwnedRoot 'foreign') ([PcaiNative.NativeResolver]::GetExpectedFileName('pcai_media'))}
        [Environment]::SetEnvironmentVariable($selection,$value)
        try{
            Assert-Control (-not [PcaiNative.NativeResolver]::IsCurrentSelectionCompatibleWithLoadedLibrary('pcai_media')) 'Explicit path relabeled unlocated default binding'
            $refused=$false
            try{[void]$resolve.Invoke($null,@('pcai_media',$second.Assembly,$null))}catch{if($_.Exception.InnerException -is [DllNotFoundException]){$refused=$true}else{throw}}
            Assert-Control $refused 'Changed explicit selection reused default handle'
        }finally{[Environment]::SetEnvironmentVariable($selection,$null)}
    }
    Assert-Control ($resolve.Invoke($null,@('pcai_media',$second.Assembly,$null)) -eq $handle) 'Restored default selection did not reuse original handle'
    Assert-Control ([PcaiNative.NativeResolver]::GetResolvedProvenance('pcai_media') -ceq 'default-loader') 'Default load provenance changed'
    Assert-Control ([PcaiNative.NativeResolver]::TryGetLoadedLibraryPath('pcai_media',[ref]$loaded) -and $loaded -ceq 'pcai_media') 'Refusal fabricated filesystem identity'
    Write-Output 'PASS: both changed media selectors refuse default-handle reuse; restored default reuses original binding'
    Write-Output 'LIMIT: actual inert C default loading only; no production Rust/native/GPU/model qualification'
    exit 0
}
$WrapperPath=(Resolve-Path -LiteralPath $WrapperPath).ProviderPath
if(-not $CompilerPath){$CompilerPath=(Get-Command clang -CommandType Application|Select-Object -First 1).Source}
$OwnedRoot=Join-Path ([IO.Path]::GetTempPath()) ('pcai-default-loader-'+[guid]::NewGuid().ToString('N'))
[void][IO.Directory]::CreateDirectory($OwnedRoot)
try{
    $loader=Join-Path $OwnedRoot 'loader';$foreign=Join-Path $OwnedRoot 'foreign'
    [void][IO.Directory]::CreateDirectory($loader);[void][IO.Directory]::CreateDirectory($foreign)
    $wrapperCopy=Join-Path $OwnedRoot 'PcaiNative.dll';Copy-Item -LiteralPath $WrapperPath -Destination $wrapperCopy
    $export=if($IsWindows){'__declspec(dllexport)'}else{'__attribute__((visibility("default")))'}
    $source=Join-Path $OwnedRoot 'inert-default.c'
    [IO.File]::WriteAllText($source,"$export unsigned int pcai_core_test(void){return 0x50434149u;}`n$export const char* pcai_string_copy(void){return `"default-copy`";}`n$export const char* pcai_core_version(void){return `"default-version`";}")
    $leaf=if($IsWindows){'pcai_media.dll'}elseif($IsMacOS){'libpcai_media.dylib'}else{'libpcai_media.so'}
    $output=Join-Path $loader $leaf
    $start=[Diagnostics.ProcessStartInfo]::new($CompilerPath);$start.UseShellExecute=$false;$start.CreateNoWindow=$true
    $arguments=if($IsWindows){@('--target=x86_64-pc-windows-msvc','-shared','-nostdlib','-fuse-ld=lld','-Wl,/noentry',$source,'-o',$output)}else{@('-shared','-fPIC','-nostdlib',$source,'-o',$output)}
    foreach($argument in $arguments){[void]$start.ArgumentList.Add($argument)}
    $compiler=[Diagnostics.Process]::Start($start)
    try{if(-not $compiler.WaitForExit(30000)){$compiler.Kill($true);throw 'Default-loader compilation exceeded30s'};Assert-Control ($compiler.ExitCode -eq 0) 'Default-loader compilation failed'}finally{$compiler.Dispose()}
    Copy-Item -LiteralPath $output -Destination (Join-Path $foreign $leaf)
    $start=[Diagnostics.ProcessStartInfo]::new((Get-Process -Id $PID).Path)
    $start.UseShellExecute=$false;$start.CreateNoWindow=$true;$start.WorkingDirectory=$OwnedRoot
    $start.RedirectStandardOutput=$true;$start.RedirectStandardError=$true
    $start.Environment['PATH']=$loader+[IO.Path]::PathSeparator+$env:PATH
    if(-not $IsWindows){$start.Environment[ $(if($IsMacOS){'DYLD_LIBRARY_PATH'}else{'LD_LIBRARY_PATH'}) ]=$loader}
    foreach($argument in @('-NoLogo','-NoProfile','-File',$PSCommandPath,'-WrapperPath',$wrapperCopy,'-Child','-OwnedRoot',$OwnedRoot)){[void]$start.ArgumentList.Add($argument)}
    $probe=[Diagnostics.Process]::Start($start);$stdout=$probe.StandardOutput.ReadToEndAsync();$stderr=$probe.StandardError.ReadToEndAsync()
    try{
        if(-not $probe.WaitForExit(45000)){$probe.Kill($true);throw 'Default-loader child exceeded45s'}
        Assert-Control ($stdout.Wait(5000) -and $stderr.Wait(5000)) 'Default-loader readers exceeded bound'
        Write-Output $stdout.GetAwaiter().GetResult();$errors=$stderr.GetAwaiter().GetResult();if($errors){Write-Output $errors}
        Assert-Control ($probe.ExitCode -eq 0) 'Default-loader owned child failed'
    }finally{$probe.Dispose()}
}finally{
    $resolved=[IO.Path]::GetFullPath($OwnedRoot)
    Assert-Control ($resolved.StartsWith([IO.Path]::GetFullPath([IO.Path]::GetTempPath()),[StringComparison]::OrdinalIgnoreCase)) 'Fixture cleanup escaped owned temp root'
    Remove-Item -LiteralPath $resolved -Recurse -Force
}
