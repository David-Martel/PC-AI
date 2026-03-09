#Requires -Version 7.0
<#
.SYNOPSIS
    Compatibility wrapper for the consolidated tooling benchmark runner.
#>

$benchmarkScript = Join-Path $PSScriptRoot 'Invoke-PcaiToolingBenchmarks.ps1'
& $benchmarkScript -CaseId token-estimate,full-context -PassThru | Select-Object -ExpandProperty Summary
