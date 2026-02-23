<#
.SYNOPSIS
    PC-AI Performance Benchmark: Native (Rust) vs. Managed (PowerShell/CIM)
    Verifies the performance gains documented in Phase 3.
#>

. (Join-Path $PSScriptRoot '..\Helpers\Resolve-TestRepoRoot.ps1')

$PcaiRoot = Resolve-TestRepoRoot -StartPath $PSScriptRoot
$AccelerationModule = Join-Path $PcaiRoot "Modules\PC-AI.Acceleration\PC-AI.Acceleration.psm1"

# Load module
if (-not (Get-Module PC-AI.Acceleration)) {
    Import-Module $AccelerationModule -Force
}

function Invoke-LegacyTokenEstimate {
    param([string]$Text)
    # Simple regex word count approximation (Legacy logic)
    if ([string]::IsNullOrEmpty($Text)) { return 0 }
    return ([regex]::Matches($Text, "\w+")).Count
}

function Invoke-LegacySystemInfo {
    # Typical CIM-based system info retrieval
    $cpu = Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average | Select-Object -ExpandProperty Average
    $mem = Get-CimInstance Win32_OperatingSystem | Select-Object TotalVisibleMemorySize, FreePhysicalMemory
    return @{ cpu_usage = $cpu; memory_used_mb = ($mem.TotalVisibleMemorySize - $mem.FreePhysicalMemory) / 1024 }
}

$TestString = "The quick brown fox jumps over the lazy dog. " * 5000 # ~45,000 words / ~250KB
$Iterations = 50

Write-Host "`n=== PC-AI Phase 3 Performance Benchmarks ===" -ForegroundColor Cyan

# 1. Token Estimation Benchmark
Write-Host "`n[1] Token Estimation Benchmark ($Iterations iterations)..."
$nativeTokenTime = Measure-Command {
    for ($i=0; $i -lt $Iterations; $i++) { $null = Get-PcaiTokenEstimate -Text $TestString }
}
$legacyTokenTime = Measure-Command {
    for ($i=0; $i -lt $Iterations; $i++) { $null = Invoke-LegacyTokenEstimate -Text $TestString }
}

# 2. System Telemetry Benchmark
Write-Host "[2] System Telemetry Benchmark ($Iterations iterations)..."
$nativeSystemTime = Measure-Command {
    for ($i=0; $i -lt $Iterations; $i++) { $null = Invoke-PcaiNativeSystemInfo -HighFidelity }
}
$legacySystemTime = Measure-Command {
    for ($i=0; $i -lt $Iterations; $i++) { $null = Invoke-LegacySystemInfo }
}

# Results Table
$Results = @(
    [PSCustomObject]@{
        Metric      = "Token Estimation"
        Native_ms   = [math]::Round($nativeTokenTime.TotalMilliseconds / $Iterations, 4)
        Legacy_ms   = [math]::Round($legacyTokenTime.TotalMilliseconds / $Iterations, 4)
        Speedup     = "{0:N1}x" -f ($legacyTokenTime.Ticks / $nativeTokenTime.Ticks)
    }
    [PSCustomObject]@{
        Metric      = "System Telemetry"
        Native_ms   = [math]::Round($nativeSystemTime.TotalMilliseconds / $Iterations, 4)
        Legacy_ms   = [math]::Round($legacySystemTime.TotalMilliseconds / $Iterations, 4)
        Speedup     = "{0:N1}x" -f ($legacySystemTime.Ticks / $nativeSystemTime.Ticks)
    }
)

$Results | Format-Table -AutoSize
Write-Host "=== End of Benchmarks ===`n" -ForegroundColor Cyan
