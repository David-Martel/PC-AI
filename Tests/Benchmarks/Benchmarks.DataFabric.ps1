#Requires -Version 7.0
<#
.SYNOPSIS
    Compatibility wrapper for the consolidated tooling benchmark runner.
#>

$benchmarkScript = Join-Path $PSScriptRoot 'Invoke-PcaiToolingBenchmarks.ps1'
& $benchmarkScript -CaseId full-context,directory-manifest -PassThru | Select-Object -ExpandProperty Summary
