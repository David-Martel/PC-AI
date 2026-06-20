#Requires -Version 7.0
[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot '..\..') | Select-Object -ExpandProperty Path
Push-Location $repoRoot
try {
    Invoke-Pester -Path '.\Tests\InputDiagnostics\InputDiagnostics.Tests.ps1' -PassThru
} finally {
    Pop-Location
}
