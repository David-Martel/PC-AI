#Requires -Version 7.0
[CmdletBinding()]
param()
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot '../../Modules/PC-AI.Acceleration/Private/Initialize-PcaiNative.ps1')
# This fixture exercises fallback/status only, never native initialization or hardware.
function Test-PcaiNativeAvailable { return $false }
# Selector fallback-counts: protects the seven former native fallback assertions;
# detects changed heuristic semantics. Needs pwsh7 only, native availability mocked.
# Breadcrumb: Initialize-PcaiNative.ps1::Invoke-PcaiNativeEstimateTokens.
$fixtures=@(
    @{Text='';Expected=0},@{Text='Hello world';Expected=4},
    @{Text='This is a test.';Expected=9},@{Text='你好世界';Expected=4},
    @{Text='internationalization';Expected=6},@{Text='abcdefghijklm';Expected=5},
    @{Text='a,b;c!';Expected=7}
)
foreach($fixture in $fixtures){
    $actual=Invoke-PcaiNativeEstimateTokens -Text $fixture.Text
    if($actual -ne $fixture.Expected){throw "Fallback token mismatch: expected $($fixture.Expected), got $actual"}
}
Write-Output 'PASS: seven retained fallback token assertions'
# Selector status-shape: protects host RID/filename and provenance shape; detects
# Windows naming on Linux and missing fields. Needs no artifact or privileges.
# Breadcrumb: Get-PcaiNativeRuntimeIdentifier/Get-PcaiNativeLibraryFileName/Get-PcaiNativeStatus.
$os=if($IsWindows){'win'}elseif($IsMacOS){'osx'}else{'linux'}
$arch=[Runtime.InteropServices.RuntimeInformation]::ProcessArchitecture.ToString().ToLowerInvariant()
$rid=Get-PcaiNativeRuntimeIdentifier
if($rid -cne "$os-$arch"){throw 'Native RID differs from actual host'}
$expectedFile=if($IsWindows){'pcai_core_lib.dll'}elseif($IsMacOS){'libpcai_core_lib.dylib'}else{'libpcai_core_lib.so'}
if((Get-PcaiNativeLibraryFileName -LibraryName 'pcai_core_lib') -cne $expectedFile){throw 'Native filename differs from host platform'}
$status=Get-PcaiNativeStatus
if($status.RuntimeIdentifier -cne $rid){throw 'Status RID differs'}
foreach($field in @('CoreLibraryPath','WrapperPath','ArtifactHash')){
    if($status.PSObject.Properties.Name -notcontains $field){throw "Missing native status field: $field"}
}
Write-Output 'PASS: retained status shape and host-specific RID/filename assertions'
