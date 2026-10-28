#Requires -Version 7.0
<#
.SYNOPSIS
    Compatibility entrypoint for the canonical Pester runner and tooling benchmarks.
.DESCRIPTION
    All retains the original flat Unit, Integration, E2E and Functional cohorts.
    CodeCoverage intentionally measures all Modules at the shared 85 percent target.
    Plan performs no Pester import/discovery, writes or environment changes.
.EXAMPLE
    ./Tests/Invoke-AllTests.ps1 -Suite All -Plan
#>
[CmdletBinding()]
param(
    [ValidateSet('All', 'Unit', 'Integration', 'E2E', 'Functional', 'Benchmarks')][string]$Suite = 'All',
    [string[]]$Tag,
    [string[]]$ExcludeTag,
    [switch]$SkipSlow,
    [switch]$CodeCoverage,
    [ValidateSet('None', 'NUnitXml', 'JUnitXml')][string]$OutputFormat = 'None',
    [string]$OutputFile = 'test-results.xml',
    [string[]]$Path,
    [string[]]$FullName,
    [switch]$Plan,
    [switch]$PassThru
)
if ($Suite -eq 'Benchmarks') {
    if ($Plan) { return [pscustomobject]@{ Profile = 'Benchmarks'; Script = Join-Path $PSScriptRoot 'Benchmarks/Invoke-PcaiToolingBenchmarks.ps1'; Discovery = 'NotRun' } }
    $benchmark = & (Join-Path $PSScriptRoot 'Benchmarks/Invoke-PcaiToolingBenchmarks.ps1') -PassThru
    if ($PassThru) { return $benchmark }
    Write-Host "Benchmark report: $($benchmark.MarkdownPath)"
    return
}
$forward = @{
    Profile      = 'Legacy' + $Suite
    Coverage     = $CodeCoverage
    OutputFormat = $OutputFormat
    OutputFile   = $(if ([IO.Path]::IsPathRooted($OutputFile)) { $OutputFile } else { Join-Path $PSScriptRoot $OutputFile })
    Plan         = $Plan
    PassThru     = $PassThru
}
if ($Path) { $forward.Path = $Path }
if ($FullName) { $forward.FullName = $FullName }
if ($Tag) { $forward.Tag = $Tag }
if ($ExcludeTag -or $SkipSlow) {
    $forward.ExcludeTag = @($ExcludeTag | Where-Object { $_ }) + @($(if ($SkipSlow) { 'Performance'; 'Model'; 'Slow' }))
}
& (Join-Path $PSScriptRoot '.pester.ps1') @forward
