#Requires -Version 7.0
<#
.SYNOPSIS
    Returns conservative Shift aggregates for an existing capture.
.DESCRIPTION
    Compatibility entrypoint for the former nearest-Shift timing heuristic.
    Temporal proximity cannot establish intended capitalization or application
    failure, so this command now uses the shared device-and-side state analyzer.
    It emits no typed text, per-letter timeline, failure rate, or causal verdict.
.PARAMETER Path
    Existing shift-source-live JSONL file. Defaults to the latest local capture.
.PARAMETER PassThru
    Return structured aggregates instead of JSON.
#>
[CmdletBinding()]
param(
    [string]$Path,
    [switch]$PassThru
)

$arguments = @{ PassThru = $true }
if ($Path) { $arguments.Path = $Path }
$summary = & (Join-Path $PSScriptRoot 'Analyze-ShiftTrace.ps1') @arguments
if ($PassThru) { $summary }
else { $summary | ConvertTo-Json -Depth 8 }
