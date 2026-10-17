#Requires -Version 7.0
[CmdletBinding()]
param([string]$SourcePath=(Join-Path $PSScriptRoot '../Benchmarks/Invoke-PcaiToolingBenchmarks.ps1'))
$ErrorActionPreference='Stop'
$tokens=$null; $errors=$null
$ast=[Management.Automation.Language.Parser]::ParseFile((Resolve-Path $SourcePath).Path,[ref]$tokens,[ref]$errors)
if($errors.Count){throw 'Runner parse failure'}
$function=$ast.Find({param($n) $n -is [Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq 'Invoke-BackendBenchmark'},$true)
. ([scriptblock]::Create($function.Extent.Text))
$script:AccelerationModule=$null
function Get-ModuleFunctionCommand {
    param($Module,$CommandName)
    return {
        param($Command,$Iterations,$Warmup,$Name,$Shell)
        if($script:FixtureCount -eq 0){return $null}
        [pscustomobject]@{Iterations=$script:FixtureCount;Mean=1;Median=1;Min=1;Max=1;StdDev=0;Tool='fixture'}
    }
}
# Protects complete measurements; detects all/partial iteration failure accepted.
# Needs pwsh only; calls real Invoke-BackendBenchmark extracted from runner AST.
foreach($count in @(0,2,3)) {
    $script:FixtureCount=$count; $caught=$false
    try {$null=Invoke-BackendBenchmark -CaseId fixture -Backend fixture -Command {} -Iterations 3 -Warmup 0}
    catch {$caught=$true}
    if($caught -ne ($count -ne 3)){throw "Iteration completeness oracle failed for $count"}
}
Write-Output 'PASS: runner rejects absent/partial measurements and accepts complete measurements'
$dir=Join-Path ([IO.Path]::GetTempPath()) ([IO.Path]::GetRandomFileName())
New-Item -ItemType Directory -Path $dir|Out-Null
try {
    $modulePath=Join-Path $dir 'failure.psm1'; [IO.File]::WriteAllText($modulePath,"throw 'intentional fixture import failure'")
    $benchmarkScript=Join-Path $dir 'failure.ps1'; [IO.File]::WriteAllText($benchmarkScript,'exit 23')
    $interfaceAlias='fixture'
    # Protects subprocess outcomes; detects nonzero pwsh exits counted as timings.
    # Needs child pwsh only; real runner child blocks, no production modules/devices.
    foreach($pattern in @('Import-Module','-InterfaceAlias','-File \$benchmarkScript \|')) {
        $blocks=@($ast.FindAll({param($n) $n -is [Management.Automation.Language.ScriptBlockAst] -and $n.Extent.Text -match $pattern -and $n.Extent.Text -match 'pwsh'},$true)|Sort-Object {$_.Extent.Text.Length})
        if($blocks.Count -eq 0){throw "No actual runner child block for $pattern"}
        $text=$blocks[0].Extent.Text
        $block=[scriptblock]::Create($text.Substring(1,$text.Length-2))
        $caught=$false
        try {& $block 2>$null} catch {$caught=$true}
        if(-not $caught){throw "Failed child accepted for $pattern"}
    }
    Write-Output 'PASS: all three real runner child blocks reject failing child processes'
} finally {Remove-Item -LiteralPath $dir -Recurse}
