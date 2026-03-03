$PC_AIRoot = if ($env:PCAI_ROOT -and (Test-Path $env:PCAI_ROOT)) {
    (Resolve-Path $env:PCAI_ROOT).Path
} elseif ($PSCommandPath) {
    (Resolve-Path (Split-Path -Parent $PSCommandPath)).Path
} else {
    (Get-Location).Path
}

Write-Host 'Importing Common and Acceleration modules...'
Import-Module (Join-Path $PC_AIRoot 'Modules\PC-AI.Common') -Force
Import-Module (Join-Path $PC_AIRoot 'Modules\PC-AI.Acceleration') -Force

Write-Host 'Initializing PC-AI Native tools...'
Initialize-PcaiNative -Force -Verbose

$TestPath = Join-Path $PC_AIRoot 'Modules\PC-AI.Hardware\Tests\PC-AI.Hardware.Tests.ps1'
Write-Host "Running Pester tests: $TestPath"
Invoke-Pester -Path $TestPath
