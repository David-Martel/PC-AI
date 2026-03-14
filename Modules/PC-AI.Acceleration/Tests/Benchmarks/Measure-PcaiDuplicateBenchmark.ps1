#Requires -Version 7.0
[CmdletBinding()]
param(
    [int]$DuplicateGroups = 40,
    [int]$DuplicatesPerGroup = 3,
    [int]$UniqueFiles = 120,
    [int]$PayloadSizeBytes = 8192,
    [int]$Iterations = 4,
    [int]$Warmup = 1
)

$moduleRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Import-Module (Join-Path $moduleRoot 'PC-AI.Acceleration.psd1') -Force

$benchmarkRoot = Join-Path $env:TEMP ("pcai-duplicates-bench-" + [guid]::NewGuid().ToString('N'))
$null = New-Item -ItemType Directory -Path $benchmarkRoot -Force

function New-RandomPayload {
    param([int]$Length)
    $bytes = New-Object byte[] $Length
    [System.Random]::new().NextBytes($bytes)
    return $bytes
}

try {
    for ($groupIndex = 0; $groupIndex -lt $DuplicateGroups; $groupIndex++) {
        $payload = New-RandomPayload -Length $PayloadSizeBytes
        $groupDir = Join-Path $benchmarkRoot ("dup-" + $groupIndex.ToString('000'))
        $null = New-Item -ItemType Directory -Path $groupDir -Force
        for ($copyIndex = 0; $copyIndex -lt $DuplicatesPerGroup; $copyIndex++) {
            [System.IO.File]::WriteAllBytes((Join-Path $groupDir ("copy-" + $copyIndex.ToString('000') + '.bin')), $payload)
        }
    }

    for ($uniqueIndex = 0; $uniqueIndex -lt $UniqueFiles; $uniqueIndex++) {
        [System.IO.File]::WriteAllBytes(
            (Join-Path $benchmarkRoot ("unique-" + $uniqueIndex.ToString('0000') + '.bin')),
            (New-RandomPayload -Length $PayloadSizeBytes)
        )
    }

    $native = Measure-CommandPerformance -Name 'Find-DuplicatesFast Native' -Iterations $Iterations -Warmup $Warmup -Command {
        Find-DuplicatesFast -Path $benchmarkRoot -Recurse -MinimumSize 1 -Algorithm SHA256 | Out-Null
    }

    $fallback = Measure-CommandPerformance -Name 'Find-DuplicatesFast PowerShell' -Iterations $Iterations -Warmup $Warmup -Command {
        Find-DuplicatesFast -Path $benchmarkRoot -Recurse -MinimumSize 1 -Algorithm SHA256 -DisableNative | Out-Null
    }

    $summary = [pscustomobject]@{
        CorpusRoot = $benchmarkRoot
        DuplicateGroups = $DuplicateGroups
        DuplicatesPerGroup = $DuplicatesPerGroup
        UniqueFiles = $UniqueFiles
        PayloadSizeBytes = $PayloadSizeBytes
        NativeMeanMs = $native.Mean
        PowerShellMeanMs = $fallback.Mean
        Speedup = if ($native.Mean -gt 0) { [Math]::Round(($fallback.Mean / $native.Mean), 2) } else { $null }
        NativeWorkingSetDeltaMeanBytes = $native.WorkingSetDeltaMeanBytes
        PowerShellWorkingSetDeltaMeanBytes = $fallback.WorkingSetDeltaMeanBytes
    }

    [pscustomobject]@{
        Native = $native
        PowerShell = $fallback
        Summary = $summary
    }
}
finally {
    Remove-Item -LiteralPath $benchmarkRoot -Recurse -Force -ErrorAction SilentlyContinue
}
