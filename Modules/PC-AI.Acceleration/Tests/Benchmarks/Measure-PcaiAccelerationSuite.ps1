#Requires -Version 7.0
[CmdletBinding()]
param(
    [Parameter()]
    [int]$Iterations = 3,

    [Parameter()]
    [int]$Warmup = 1,

    [Parameter()]
    [int]$FileCount = 240,

    [Parameter()]
    [int]$DuplicateGroups = 40,

    [Parameter()]
    [int]$PayloadSizeBytes = 8192
)

$moduleRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Import-Module (Join-Path $moduleRoot 'PC-AI.Acceleration.psd1') -Force

$benchmarkRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("pcai-accel-suite-" + [guid]::NewGuid().ToString('N'))
$null = New-Item -ItemType Directory -Path $benchmarkRoot -Force

try {
    $duplicateCopies = 3
    for ($group = 0; $group -lt $DuplicateGroups; $group++) {
        $payload = New-Object byte[] $PayloadSizeBytes
        [System.Random]::new($group + 17).NextBytes($payload)
        for ($copy = 0; $copy -lt $duplicateCopies; $copy++) {
            $path = Join-Path $benchmarkRoot ("dup-{0:D3}-{1:D2}.bin" -f $group, $copy)
            [System.IO.File]::WriteAllBytes($path, $payload)
        }
    }

    $uniqueFiles = [Math]::Max(0, $FileCount - ($DuplicateGroups * $duplicateCopies))
    for ($index = 0; $index -lt $uniqueFiles; $index++) {
        $payload = New-Object byte[] $PayloadSizeBytes
        [System.Random]::new($index + 1000).NextBytes($payload)
        $path = Join-Path $benchmarkRoot ("uniq-{0:D3}.bin" -f $index)
        [System.IO.File]::WriteAllBytes($path, $payload)
    }

    $allFiles = Get-ChildItem -Path $benchmarkRoot -File | Sort-Object FullName
    $allFilePaths = @($allFiles.FullName)

    $hashParallel = Measure-CommandPerformance -Name 'Get-FileHashParallel' -Iterations $Iterations -Warmup $Warmup -Command {
        Get-FileHashParallel -Path $allFilePaths -Algorithm SHA256 | Out-Null
    }

    $hashSequential = Measure-CommandPerformance -Name 'Get-FileHash Sequential' -Iterations $Iterations -Warmup $Warmup -Command {
        foreach ($filePath in $allFilePaths) {
            Get-FileHash -Path $filePath -Algorithm SHA256 | Out-Null
        }
    }

    $duplicatesNative = Measure-CommandPerformance -Name 'Find-DuplicatesFast Native' -Iterations $Iterations -Warmup $Warmup -Command {
        Find-DuplicatesFast -Path $benchmarkRoot -Recurse -MinimumSize 1 -Algorithm SHA256 | Out-Null
    }

    $duplicatesFallback = Measure-CommandPerformance -Name 'Find-DuplicatesFast PowerShell' -Iterations $Iterations -Warmup $Warmup -Command {
        Find-DuplicatesFast -Path $benchmarkRoot -Recurse -MinimumSize 1 -Algorithm SHA256 -DisableNative | Out-Null
    }

    [pscustomobject]@{
        Corpus = [pscustomobject]@{
            Root = $benchmarkRoot
            FileCount = $allFiles.Count
            DuplicateGroups = $DuplicateGroups
            DuplicateCopies = $duplicateCopies
            UniqueFiles = $uniqueFiles
            PayloadSizeBytes = $PayloadSizeBytes
        }
        Hashing = [pscustomobject]@{
            Parallel = $hashParallel
            Sequential = $hashSequential
            Speedup = if ($hashParallel.Mean -gt 0) { [Math]::Round($hashSequential.Mean / $hashParallel.Mean, 2) } else { $null }
        }
        Duplicates = [pscustomobject]@{
            Native = $duplicatesNative
            Fallback = $duplicatesFallback
            Speedup = if ($duplicatesNative.Mean -gt 0) { [Math]::Round($duplicatesFallback.Mean / $duplicatesNative.Mean, 2) } else { $null }
        }
    }
}
finally {
    Remove-Item -Path $benchmarkRoot -Recurse -Force -ErrorAction SilentlyContinue
}
