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
$failures=[System.Collections.Generic.List[string]]::new()
foreach($fixture in $fixtures){
    $actual=Invoke-PcaiNativeEstimateTokens -Text $fixture.Text
    if($actual -ne $fixture.Expected){$failures.Add("Retained fallback mismatch: expected $($fixture.Expected), got $actual")}
}
Write-Output "Retained fallback assertions: $($fixtures.Count) executed, $($failures.Count) failed"
# Selector scalar-counts: protects L/N runs, whitespace runs and individual Unicode
# scalars sized by UTF8 bytes; detects surrogate splitting and category omissions.
# No native artifact needed. Expectations match the maintained benchmark fixtures.
# Breadcrumb: Invoke-PcaiToolingBenchmarks.ps1::Get-ReferenceTokenEstimate.
$emoji=[char]::ConvertFromUtf32(0x1F600)
$letter=[char]::ConvertFromUtf32(0x10400)
$number=[char]::ConvertFromUtf32(0x1D7CE)
$scalarFixtures=@(
    @{Name='symbol';Text=$emoji;Expected=2},
    @{Name='symbols';Text=($emoji*2);Expected=3},
    @{Name='astral-letter';Text=($letter*3);Expected=4},
    @{Name='astral-letter-short';Text=($letter*2);Expected=2},
    @{Name='astral-number';Text=($number*3);Expected=4},
    @{Name='astral-number-short';Text=($number*2);Expected=2},
    @{Name='mixed-astral-word-number';Text=($letter+$number);Expected=2},
    @{Name='number-category';Text=([string][char]0x216B)*3;Expected=4},
    @{Name='number-category-group';Text=([string][char]0x216B)*2;Expected=2},
    @{Name='ascii-number-group';Text='12';Expected=2},
    @{Name='mixed-word-number';Text='abc123';Expected=2},
    @{Name='combining';Text=('e'+[char]0x301);Expected=3},
    @{Name='long-whitespace';Text=(' '*9);Expected=4},
    @{Name='cjk';Text=([string][char]0x6F22)*3;Expected=4},
    @{Name='whitespace';Text=([string][char]0xA0+"`n");Expected=2},
    @{Name='zwj';Text=($emoji+[char]0x200D+$emoji);Expected=4},
    @{Name='bmp-word-number-group';Text=([string][char]0xE9+'12'+[char]0x216B);Expected=2},
    @{Name='unpaired-high-surrogate';Text=([string][char]0xD800);Expected=2},
    @{Name='unpaired-low-surrogate';Text=([string][char]0xDC00);Expected=2},
    @{Name='reversed-surrogates';Text=([string][char]0xDC00+[char]0xD800);Expected=3},
    @{Name='unpaired-surrogate-between-runs';Text=('a'+[char]0xD800+'1');Expected=4}
)
$retainedFailures=$failures.Count
foreach($fixture in $scalarFixtures){
    $actual=Invoke-PcaiNativeEstimateTokens -Text $fixture.Text
    if($actual -ne $fixture.Expected){$failures.Add("Scalar fallback mismatch $($fixture.Name): expected $($fixture.Expected), got $actual")}
}
Write-Output "Scalar fallback assertions: $($scalarFixtures.Count) executed, $($failures.Count-$retainedFailures) failed"
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
if($failures.Count){
    $failures | Write-Output
    throw "$($failures.Count) of $($fixtures.Count+$scalarFixtures.Count) fallback assertions failed"
}
Write-Output "PASS: all $($fixtures.Count+$scalarFixtures.Count) fallback token assertions"
