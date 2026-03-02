#Requires -Version 7.0
<#
.SYNOPSIS
    Compares performance between Rust tools and PowerShell equivalents

.DESCRIPTION
    Benchmarks Rust tools against PowerShell equivalents to measure
    performance gains. Useful for validating acceleration benefits.

.PARAMETER Test
    Test to run: FileSearch, ContentSearch, ProcessList, DiskUsage, Hash, All

.PARAMETER Path
    Path to use for file-based tests

.PARAMETER Iterations
    Number of benchmark iterations

.EXAMPLE
    Compare-ToolPerformance -Test FileSearch -Path "C:\Windows"
    Compares fd vs Get-ChildItem

.EXAMPLE
    Compare-ToolPerformance -Test All -Iterations 5
    Runs all comparison tests

.OUTPUTS
    PSCustomObject[] with comparison results
#>
function Compare-ToolPerformance {
    [CmdletBinding()]
    [OutputType([PSCustomObject[]])]
    param(
        [Parameter(Position = 0)]
        [ValidateSet('FileSearch', 'ContentSearch', 'ProcessList', 'Hash', 'All')]
        [string]$Test = 'All',

        [Parameter()]
        [string]$Path = $env:USERPROFILE,

        [Parameter()]
        [int]$Iterations = 5,

        [Parameter()]
        [switch]$Detailed
    )

    $results = @()

    $tests = if ($Test -eq 'All') {
        @('FileSearch', 'ContentSearch', 'ProcessList', 'Hash')
    }
    else {
        @($Test)
    }

    foreach ($t in $tests) {
        Write-Host "Running $t benchmark..." -ForegroundColor Cyan

        $comparison = switch ($t) {
            'FileSearch'    { Compare-FileSearchPerformance -Path $Path -Iterations $Iterations }
            'ContentSearch' { Compare-ContentSearchPerformance -Path $Path -Iterations $Iterations }
            'ProcessList'   { Compare-ProcessListPerformance -Iterations $Iterations }
            'Hash'          { Compare-HashPerformance -Path $Path -Iterations $Iterations }
        }

        if ($comparison) {
            $results += $comparison
        }
    }

    # Summary
    Write-Host ""
    Write-Host "=== Performance Comparison Summary ===" -ForegroundColor Green

    foreach ($result in $results) {
        $speedup = if ($result.PowerShellMs -gt 0) {
            [Math]::Round($result.PowerShellMs / $result.RustMs, 1)
        }
        else { 0 }

        $color = if ($speedup -ge 5) { 'Green' }
                 elseif ($speedup -ge 2) { 'Yellow' }
                 else { 'White' }

        Write-Host "$($result.Test): " -NoNewline
        Write-Host "${speedup}x faster" -ForegroundColor $color -NoNewline
        Write-Host " (Rust: $($result.RustMs)ms, PowerShell: $($result.PowerShellMs)ms)"
    }

    return $results
}

function Compare-FileSearchPerformance {
    [CmdletBinding()]
    param([string]$Path, [int]$Iterations)

    $fdPath = Get-RustToolPath -ToolName 'fd'
    if (-not $fdPath) {
        Write-Warning "fd not available, skipping FileSearch comparison"
        return $null
    }

    # Rust (fd)
    $fdResult = Measure-CommandPerformance -Command "& '$fdPath' -t f -e ps1 '$Path'" -Iterations $Iterations -Name 'fd'

    # PowerShell
    $psResult = Measure-CommandPerformance -Command { Get-ChildItem -Path $using:Path -Filter "*.ps1" -Recurse -File -ErrorAction SilentlyContinue } -Iterations $Iterations -Name 'Get-ChildItem'

    return [PSCustomObject]@{
        Test          = 'FileSearch'
        RustTool      = 'fd'
        RustMs        = $fdResult.Mean
        PowerShellMs  = $psResult.Mean
        Speedup       = [Math]::Round($psResult.Mean / $fdResult.Mean, 2)
        Iterations    = $Iterations
    }
}

function Compare-ContentSearchPerformance {
    [CmdletBinding()]
    param([string]$Path, [int]$Iterations)

    $rgPath = Get-RustToolPath -ToolName 'rg'
    if (-not $rgPath) {
        Write-Warning "ripgrep not available, skipping ContentSearch comparison"
        return $null
    }

    $pattern = 'function'
    $fileType = '*.ps1'

    # Rust (ripgrep)
    $rgResult = Measure-CommandPerformance -Command "& '$rgPath' -l '$pattern' -g '$fileType' '$Path'" -Iterations $Iterations -Name 'ripgrep'

    # PowerShell
    $psResult = Measure-CommandPerformance -Command {
        Get-ChildItem -Path $using:Path -Filter "*.ps1" -Recurse -File -ErrorAction SilentlyContinue |
        Select-String -Pattern 'function' -List |
        Select-Object -ExpandProperty Path
    } -Iterations $Iterations -Name 'Select-String'

    return [PSCustomObject]@{
        Test          = 'ContentSearch'
        RustTool      = 'ripgrep'
        RustMs        = $rgResult.Mean
        PowerShellMs  = $psResult.Mean
        Speedup       = [Math]::Round($psResult.Mean / $rgResult.Mean, 2)
        Iterations    = $Iterations
    }
}

function Compare-ProcessListPerformance {
    [CmdletBinding()]
    param([int]$Iterations)

    $procsPath = Get-RustToolPath -ToolName 'procs'
    if (-not $procsPath) {
        Write-Warning "procs not available, skipping ProcessList comparison"
        return $null
    }

    # Rust (procs)
    $procsResult = Measure-CommandPerformance -Command "& '$procsPath' --no-header" -Iterations $Iterations -Name 'procs'

    # PowerShell
    $psResult = Measure-CommandPerformance -Command { Get-Process | Select-Object Id, ProcessName, CPU, WorkingSet64 } -Iterations $Iterations -Name 'Get-Process'

    return [PSCustomObject]@{
        Test          = 'ProcessList'
        RustTool      = 'procs'
        RustMs        = $procsResult.Mean
        PowerShellMs  = $psResult.Mean
        Speedup       = [Math]::Round($psResult.Mean / $procsResult.Mean, 2)
        Iterations    = $Iterations
    }
}

function Compare-HashPerformance {
    [CmdletBinding()]
    param([string]$Path, [int]$Iterations)

    # Find test files
    $testFiles = Get-ChildItem -Path $Path -File -Recurse -ErrorAction SilentlyContinue |
        Where-Object { $_.Length -gt 1MB -and $_.Length -lt 100MB } |
        Select-Object -First 5

    if ($testFiles.Count -lt 3) {
        Write-Warning "Not enough test files for hash comparison"
        return $null
    }

    $filePaths = $testFiles.FullName

    # Sequential hashing
    $seqResult = Measure-CommandPerformance -Command {
        $using:filePaths | ForEach-Object { Get-FileHash -Path $_ -Algorithm SHA256 }
    } -Iterations $Iterations -Name 'Sequential'

    # Parallel hashing (PS7+)
    $parResult = Measure-CommandPerformance -Command {
        $using:filePaths | ForEach-Object -Parallel {
            Get-FileHash -Path $_ -Algorithm SHA256
        } -ThrottleLimit 4
    } -Iterations $Iterations -Name 'Parallel'

    return [PSCustomObject]@{
        Test          = 'Hash'
        RustTool      = 'PS7-Parallel'
        RustMs        = $parResult.Mean
        PowerShellMs  = $seqResult.Mean
        Speedup       = [Math]::Round($seqResult.Mean / $parResult.Mean, 2)
        Iterations    = $Iterations
        FileCount     = $testFiles.Count
    }
}
