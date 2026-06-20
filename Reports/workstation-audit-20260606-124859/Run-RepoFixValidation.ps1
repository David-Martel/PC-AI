#Requires -Version 7.0
<#
.SYNOPSIS
    Runs focused validation for the 2026-06-06 repo-fix pass.

.DESCRIPTION
    Preserves logs for the FunctionGemma config, CargoTools wrapper, input
    diagnostics, and no-default-features FunctionGemma runtime check.
#>
[CmdletBinding()]
param(
    [string]$OutputDirectory = (Join-Path $PSScriptRoot 'repo-fix-validation-20260606'),
    [switch]$SkipRust
)

$ErrorActionPreference = 'Continue'
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot '..\..') | Select-Object -ExpandProperty Path
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null

$results = [System.Collections.Generic.List[pscustomobject]]::new()

function Invoke-ValidationStep {
    param(
        [Parameter(Mandatory)] [string] $Name,
        [Parameter(Mandatory)] [scriptblock] $ScriptBlock
    )

    $logPath = Join-Path $OutputDirectory ("$Name.log")
    Push-Location $repoRoot
    try {
        & $ScriptBlock *> $logPath
        $exitCode = if ($LASTEXITCODE -is [int]) { [int]$LASTEXITCODE } elseif ($?) { 0 } else { 1 }
    } catch {
        $_ | Out-String | Add-Content -LiteralPath $logPath -Encoding UTF8
        $exitCode = 1
    } finally {
        Pop-Location
    }

    $results.Add([pscustomobject]@{
        Name = $Name
        ExitCode = $exitCode
        Log = $logPath
        Passed = ($exitCode -eq 0)
    })
}

Invoke-ValidationStep -Name 'pester-invoke-rustbuild' -ScriptBlock {
    Invoke-Pester -Path '.\Tests\Unit\Invoke-RustBuild.Tests.ps1' -PassThru
}

Invoke-ValidationStep -Name 'pester-cargotools' -ScriptBlock {
    Invoke-Pester -Path '.\Tests\Unit\CargoTools.Tests.ps1' -PassThru
}

Invoke-ValidationStep -Name 'pester-functiongemma-config' -ScriptBlock {
    Invoke-Pester -Path '.\Tests\Unit\FunctionGemmaConfig.Tests.ps1' -PassThru
}

Invoke-ValidationStep -Name 'pester-cuda-initializer' -ScriptBlock {
    Invoke-Pester -Path '.\Tests\Unit\Initialize-CudaEnvironment.Tests.ps1' -PassThru
}

Invoke-ValidationStep -Name 'pester-inputdiagnostics' -ScriptBlock {
    Invoke-Pester -Path '.\Tests\InputDiagnostics\InputDiagnostics.Tests.ps1' -PassThru
}

Invoke-ValidationStep -Name 'nvidia-health-readonly' -ScriptBlock {
    & '.\Tools\InputDiagnostics\Test-NvidiaDualGpuDriverHealth.ps1' -AsJson
}

if (-not $SkipRust) {
    Invoke-ValidationStep -Name 'rust-functiongemma-runtime-check' -ScriptBlock {
        & '.\Tools\Invoke-RustBuild.ps1' `
            -Path '.\Deploy\rust-functiongemma-runtime' `
            -LlmOutput `
            -CargoArgs @('check', '--no-default-features')
    }
}

$results | ConvertTo-Json -Depth 4 |
    Set-Content -LiteralPath (Join-Path $OutputDirectory 'summary.json') -Encoding UTF8

$results | Format-Table -AutoSize
if ($results.Where({ -not $_.Passed }).Count -gt 0) {
    exit 1
}
