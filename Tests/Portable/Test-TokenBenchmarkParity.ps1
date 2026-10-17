#Requires -Version 7.0
[CmdletBinding()]
param([string]$SourcePath=(Join-Path $PSScriptRoot '../Benchmarks/Invoke-PcaiToolingBenchmarks.ps1'), [switch]$NestedChild)
$ErrorActionPreference='Stop'
$tokens=$null; $errors=$null
$ast=[Management.Automation.Language.Parser]::ParseFile((Resolve-Path $SourcePath).Path,[ref]$tokens,[ref]$errors)
if($errors.Count){throw 'Runner parse failure'}
foreach($name in @('Get-ReferenceTokenEstimate','Assert-TokenEstimateParity','Test-BenchmarkCaseEquivalence','Get-ConfigValue','Add-BenchmarkMeasurement','Get-ToolingCaseBenchmarks')) {
    $function=$ast.Find({param($n) $n -is [Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq $name},$true)
    if(-not $function){throw "Missing production function $name"}
    . ([scriptblock]::Create($function.Extent.Text))
}
# Selectors unicode-empty/ascii/symbol/astral-letter/astral-number/number-category/
# combining/long-whitespace/CJK/whitespace/ZWJ protect Rust-equivalent token work.
# Detects UTF-16 surrogate splitting, byte/character confusion, category omissions.
# Needs PowerShell 7; native comparison additionally needs PCAI_TOKEN_NATIVE_PATH.
# Breadcrumb: Tests/Benchmarks/Invoke-PcaiToolingBenchmarks.ps1::Get-ReferenceTokenEstimate.
$emoji=[char]::ConvertFromUtf32(0x1F600)
$letter=[char]::ConvertFromUtf32(0x10400)
$number=[char]::ConvertFromUtf32(0x1D7CE)
$fixtures=@(
    @{Name='empty';Text='';Expected=0},
    @{Name='ascii';Text='Hello world';Expected=4},
    @{Name='symbol';Text=$emoji;Expected=2},
    @{Name='symbols';Text=($emoji*2);Expected=3},
    @{Name='astral-letter';Text=($letter*3);Expected=4},
    @{Name='astral-number';Text=($number*3);Expected=4},
    @{Name='number-category';Text=([string][char]0x216B)*3;Expected=4},
    @{Name='number-category-group';Text=([string][char]0x216B)*2;Expected=2},
    @{Name='ascii-number-group';Text='12';Expected=2},
    @{Name='mixed-word-number';Text='abc123';Expected=2},
    @{Name='combining';Text=('e'+[char]0x301);Expected=3},
    @{Name='long-whitespace';Text=(' '*9);Expected=4},
    @{Name='cjk';Text=([string][char]0x6F22)*3;Expected=4},
    @{Name='whitespace';Text=([string][char]0xA0+"`n");Expected=2},
    @{Name='zwj';Text=($emoji+[char]0x200D+$emoji);Expected=4}
)
foreach($fixture in $fixtures){
    $actual=Get-ReferenceTokenEstimate -Text $fixture.Text
    if($actual -ne $fixture.Expected){throw "Reference mismatch $($fixture.Name): $actual"}
    Assert-TokenEstimateParity -Text $fixture.Text -NativeTokens $actual
}
# Selector mismatch-refusal protects pre-timing parity. Detects absent, bogus or
# different output. Needs no native artifact. Breadcrumb: Assert-TokenEstimateParity.
foreach($value in @($null,'bogus',0,3)){
    $rejected=$false
    try{Assert-TokenEstimateParity -Text $emoji -NativeTokens $value}catch{$rejected=$true}
    if(-not $rejected){throw "Invalid native output admitted: $value"}
}
# Selector unknown-proof protects reporting honesty; detects truthy strings and
# absent proof. Needs no hardware. Breadcrumb: Test-BenchmarkCaseEquivalence.
foreach($result in @([pscustomobject]@{},[pscustomobject]@{EquivalenceVerified='true'},[pscustomobject]@{EquivalenceVerified=$false})){
    if(Test-BenchmarkCaseEquivalence $result){throw 'Unknown/nonboolean proof admitted'}
}
if(-not(Test-BenchmarkCaseEquivalence ([pscustomobject]@{EquivalenceVerified=$true}))){throw 'Real proof rejected'}
if($env:PCAI_TOKEN_NATIVE_PATH){
    Add-Type @'
using System;
using System.Runtime.InteropServices;
public static class TokenParityInterop {
 [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
 public delegate UIntPtr Estimate([MarshalAs(UnmanagedType.LPUTF8Str)] string text);
}
'@
    $handle=[Runtime.InteropServices.NativeLibrary]::Load($env:PCAI_TOKEN_NATIVE_PATH)
    try{
        $export=[Runtime.InteropServices.NativeLibrary]::GetExport($handle,'pcai_estimate_tokens')
        $estimate=[Runtime.InteropServices.Marshal]::GetDelegateForFunctionPointer($export,[TokenParityInterop+Estimate])
        foreach($fixture in $fixtures){
            $actual=$estimate.Invoke($fixture.Text).ToUInt64()
            Assert-TokenEstimateParity -Text $fixture.Text -NativeTokens $actual
        }
        Write-Output "PASS: $($fixtures.Count) real native token parity cases"
    }finally{[Runtime.InteropServices.NativeLibrary]::Free($handle)}
}else{Write-Output 'SKIP: real native parity (PCAI_TOKEN_NATIVE_PATH not supplied)'}
Write-Output "PASS: $($fixtures.Count) reference cases, mismatch refusal and proof controls"

# Selector actual-case-preflight protects the wiring before timing and a callable
# scalar baseline closure; detects helper-only tests passing with callsite removed.
# Needs a fake module and timer, no hardware. Breadcrumb: Get-ToolingCaseBenchmarks.
$script:CommonModule=$null; $script:CliModule=$null
$script:TimingCalls=0
function Invoke-BackendBenchmark {
    param($CaseId,$Backend,$Command,$Iterations,$Warmup)
    $script:TimingCalls++
    $commandOutput=@(& $Command 2>&1)
    if(@($commandOutput | Where-Object { $_ -is [Management.Automation.ErrorRecord] }).Count -gt 0){
        throw 'Timed command produced an error record'
    }
    return [pscustomobject]@{Backend=$Backend;MeanMs=1;Iterations=$Iterations}
}
$case=[pscustomobject]@{Id='token-estimate'}
$defaults=[pscustomobject]@{}
$capabilities=[pscustomobject]@{}
$script:AccelerationModule=New-Module { function Get-PcaiTokenEstimate { return [uint64]3 } }
$rejected=$false
try { Get-ToolingCaseBenchmarks -Case $case -Defaults $defaults -RepoRoot $PSScriptRoot -Capabilities $capabilities -TokenSample $emoji | Out-Null }
catch { if($_ -notmatch 'Token output mismatch'){throw}; $rejected=$true }
if(-not $rejected -or $script:TimingCalls -ne 0){throw 'Mismatched work reached timer'}
$script:AccelerationModule=New-Module { function Get-PcaiTokenEstimate { return [uint64]2 } }
$result=Get-ToolingCaseBenchmarks -Case $case -Defaults $defaults -RepoRoot $PSScriptRoot -Capabilities $capabilities -TokenSample $emoji
if($script:TimingCalls -ne 2 -or -not $result.EquivalenceVerified){throw 'Equal work did not reach both timers'}
Write-Output 'PASS: actual token case refuses mismatches before timing and executes equal-work baseline'

# Selector report-proof protects unknown and empty report semantics and suppressed
# ratios; detects reporting bypass of the proof helper. Needs no artifacts.
# Breadcrumb: runner $benchmarkRows and OutputEquivalenceValidated assignments.
$rowAssignment=$ast.Find({param($n) $n -is [Management.Automation.Language.AssignmentStatementAst] -and $n.Left.Extent.Text -eq '$benchmarkRows'},$true)
$caseResults=@([pscustomobject]@{
    Case=[pscustomobject]@{id='unknown';name='Unknown';category='fixture'}
    Path=$PSScriptRoot;Coverage=$null
    Results=@([pscustomobject]@{Backend='powershell';MeanMs=2})
})
. ([scriptblock]::Create($rowAssignment.Extent.Text))
if(@($benchmarkRows).Count -ne 1 -or $benchmarkRows.EquivalenceVerified -or $null -ne $benchmarkRows.SpeedupVsBaseline){throw 'Unknown report claims proof or speedup'}
$summaryTable=$ast.Find({param($n) $n -is [Management.Automation.Language.HashtableAst] -and @($n.KeyValuePairs | Where-Object {$_.Item1.Extent.Text -eq 'OutputEquivalenceValidated'}).Count -gt 0},$true)
$proofExpression=@($summaryTable.KeyValuePairs | Where-Object {$_.Item1.Extent.Text -eq 'OutputEquivalenceValidated'})[0].Item2.Extent.Text
$benchmarkRows=@()
if(& ([scriptblock]::Create($proofExpression))){throw 'Empty report claims equivalence'}
$benchmarkRows=@([pscustomobject]@{EquivalenceVerified=$true})
if(-not (& ([scriptblock]::Create($proofExpression)))){throw 'Proven nonempty report lost proof'}
Write-Output 'PASS: actual report suppresses unknown proof/ratios and empty-report equivalence'

# Selector nested-invocation protects CI/module closure scope; detects a reference
# function that only resolves from direct -File/global scope. Needs child pwsh.
# Breadcrumb: token baseline GetNewClosure in Get-ToolingCaseBenchmarks.
if(-not $NestedChild){
    $testPath=$PSCommandPath.Replace("'","''")
    $sourceArgument=(Resolve-Path -LiteralPath $SourcePath).Path.Replace("'","''")
    $code="& '$testPath' -SourcePath '$sourceArgument' -NestedChild"
    $encoded=[Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($code))
    & (Get-Process -Id $PID).Path -NoLogo -NoProfile -EncodedCommand $encoded
    if($LASTEXITCODE -ne 0){throw 'Nested token parity invocation failed'}
    Write-Output 'PASS: nested script invocation preserves baseline closure scope'
}
