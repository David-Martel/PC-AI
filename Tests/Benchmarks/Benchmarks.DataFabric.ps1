<#
.SYNOPSIS
    PC-AI Phase 4 Benchmark: Zero-Overhead Data Fabric vs. PowerShell Interrogation
    Compares the native consolidated context retrieval against legacy PS/CIM methods.
#>

. (Join-Path $PSScriptRoot '..\Helpers\Resolve-TestRepoRoot.ps1')

$PcaiRoot = Resolve-TestRepoRoot -StartPath $PSScriptRoot
$NativeDll = Join-Path $PcaiRoot "bin\PcaiNative.dll"

# Load native assembly
Add-Type -Path $NativeDll

function Invoke-LegacyFullInterrogation {
    # Simulates the old way of gathering context (Multi-call, Multi-JSON)
    $sys = Get-CimInstance Win32_OperatingSystem | Select-Object Caption, Version, CSName
    $cpu = Get-CimInstance Win32_Processor | Select-Object Name, NumberOfCores, LoadPercentage
    $net = Get-NetIPConfiguration
    $wsl = wsl --list --verbose | Out-String

    $payload = @{
        System = $sys
        CPU = $cpu
        Network = $net
        Vmm = $wsl
    }
    return $payload | ConvertTo-Json -Depth 3
}

$Iterations = 5

Write-Host "`n=== PC-AI Phase 4: Zero-Overhead Data Fabric Benchmark ===" -ForegroundColor Cyan
Write-Host "[*] Comparing native consolidated context vs. legacy PS interrogation ($Iterations iterations)...`n"

# 1. Legacy Interrogation (Multiple PS/CIM calls)
$legacyTime = Measure-Command {
    for ($i=0; $i -lt $Iterations; $i++) {
        $null = Invoke-LegacyFullInterrogation
    }
}

# 2. Native Data Fabric (Single FFI call with background Rust collection)
$nativeTime = Measure-Command {
    for ($i=0; $i -lt $Iterations; $i++) {
        $null = [PcaiNative.PcaiCore]::QueryFullContextJson()
    }
}

$legacyAvg = $legacyTime.TotalMilliseconds / $Iterations
$nativeAvg = $nativeTime.TotalMilliseconds / $Iterations
$speedup = $legacyAvg / $nativeAvg

$Results = [PSCustomObject]@{
    Method              = "Data Interrogation"
    Legacy_PS_ms        = [math]::Round($legacyAvg, 2)
    Native_Fabric_ms    = [math]::Round($nativeAvg, 2)
    LatencyReduction    = "{0:P0}" -f (1 - ($nativeAvg / $legacyAvg))
    ThroughputGain      = "{0:N1}x" -f $speedup
}

$Results | Format-Table -AutoSize

Write-Host "[OK] Phase 4 results: Data Fabric is $([math]::Round($speedup, 1))x faster than PowerShell interrogation." -ForegroundColor Green
Write-Host "[OK] Latency reduced by $([math]::Round((1 - ($nativeAvg / $legacyAvg)) * 100, 1))%." -ForegroundColor Green
Write-Host "=== End of Phase 4 Benchmarks ===`n"
