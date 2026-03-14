#Requires -Version 7.0
[CmdletBinding()]
param(
    [Parameter()]
    [int]$Iterations = 3,

    [Parameter()]
    [int]$Warmup = 1,

    [Parameter()]
    [switch]$SkipProcesses,

    [Parameter()]
    [switch]$SkipDisk,

    [Parameter()]
    [switch]$SkipSearch
)

$moduleRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Import-Module (Join-Path $moduleRoot 'PC-AI.Acceleration.psd1') -Force
$module = Get-Module PC-AI.Acceleration

$benchmarkRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("pcai-command-matrix-" + [guid]::NewGuid().ToString('N'))
$null = New-Item -ItemType Directory -Path $benchmarkRoot -Force

try {
    $srcRoot = Join-Path $benchmarkRoot 'src'
    $logRoot = Join-Path $benchmarkRoot 'logs'
    $null = New-Item -ItemType Directory -Path $srcRoot -Force
    $null = New-Item -ItemType Directory -Path $logRoot -Force

    for ($i = 0; $i -lt 150; $i++) {
        $folder = Join-Path $srcRoot ("pkg{0:D3}" -f ($i % 10))
        $null = New-Item -ItemType Directory -Path $folder -Force
        $code = @(
            "function Invoke-Test$i {"
            "    Write-Output 'TODO item $i'"
            "    # marker: alpha-beta-$i"
            "}"
        ) -join "`n"
        [System.IO.File]::WriteAllText((Join-Path $folder ("test{0:D3}.ps1" -f $i)), $code)
    }

    for ($i = 0; $i -lt 40; $i++) {
        $lines = @(
            "INFO startup $i",
            "WARN threshold $i",
            "ERROR exception trace $i"
        ) -join "`n"
        [System.IO.File]::WriteAllText((Join-Path $logRoot ("app{0:D3}.log" -f $i)), $lines)
    }

    $results = [ordered]@{}

    $results.FindFiles = [pscustomobject]@{
        Native = Measure-CommandPerformance -Name 'Find-FilesFast Native' -Iterations $Iterations -Warmup $Warmup -Command {
            Find-FilesFast -Path $srcRoot -Extension ps1 -PreferNative | Out-Null
        }
        Fallback = Measure-CommandPerformance -Name 'Find-FilesFast GetChildItem' -Iterations $Iterations -Warmup $Warmup -Command {
            & $module { param($path) Find-WithGetChildItem -Path $path -Extension @('ps1') } $srcRoot | Out-Null
        }
    }
    $results.FindFiles | Add-Member -NotePropertyName Speedup -NotePropertyValue ([Math]::Round($results.FindFiles.Fallback.Mean / $results.FindFiles.Native.Mean, 2))

    if (-not $SkipSearch) {
        $results.SearchContent = [pscustomobject]@{
            Native = Measure-CommandPerformance -Name 'Search-ContentFast Native' -Iterations $Iterations -Warmup $Warmup -Command {
                Search-ContentFast -Path $srcRoot -LiteralPattern 'TODO item' -FilePattern '*.ps1' | Out-Null
            }
            Fallback = Measure-CommandPerformance -Name 'Search-ContentFast Select-String' -Iterations $Iterations -Warmup $Warmup -Command {
                & $module {
                    param($path)
                    Search-WithParallelSelectString -Path $path -LiteralPattern 'TODO item' -SearchPattern ([regex]::Escape('TODO item')) -FilePattern @('*.ps1') -Context 0 -CaseSensitive:$false -WholeWord:$false -Invert:$false -MaxResults 0 -FilesOnly:$false -ThrottleLimit ([Environment]::ProcessorCount)
                } $srcRoot | Out-Null
            }
        }
        $results.SearchContent | Add-Member -NotePropertyName Speedup -NotePropertyValue ([Math]::Round($results.SearchContent.Fallback.Mean / $results.SearchContent.Native.Mean, 2))

        $results.SearchLogs = [pscustomobject]@{
            Native = Measure-CommandPerformance -Name 'Search-LogsFast Native' -Iterations $Iterations -Warmup $Warmup -Command {
                Search-LogsFast -Path $logRoot -Pattern 'ERROR|WARN' -Include '*.log' | Out-Null
            }
            Fallback = Measure-CommandPerformance -Name 'Search-LogsFast Select-String' -Iterations $Iterations -Warmup $Warmup -Command {
                & $module {
                    param($path)
                    Search-WithSelectString -Path $path -Pattern 'ERROR|WARN' -Include @('*.log') -Context 0 -CaseSensitive:$false -MaxCount 0 -CountOnly:$false
                } $logRoot | Out-Null
            }
        }
        $results.SearchLogs | Add-Member -NotePropertyName Speedup -NotePropertyValue ([Math]::Round($results.SearchLogs.Fallback.Mean / $results.SearchLogs.Native.Mean, 2))
    }

    if (-not $SkipProcesses) {
        $results.Processes = [pscustomobject]@{
            Native = Measure-CommandPerformance -Name 'Get-ProcessesFast Native' -Iterations $Iterations -Warmup $Warmup -Command {
                Get-ProcessesFast -Top 20 -SortBy cpu | Out-Null
            }
            Fallback = Measure-CommandPerformance -Name 'Get-ProcessesFast PowerShell' -Iterations $Iterations -Warmup $Warmup -Command {
                & $module { Get-ProcessesParallel -Top 20 -SortBy cpu } | Out-Null
            }
        }
        $results.Processes | Add-Member -NotePropertyName Speedup -NotePropertyValue ([Math]::Round($results.Processes.Fallback.Mean / $results.Processes.Native.Mean, 2))
    }

    if (-not $SkipDisk) {
        $results.Disk = [pscustomobject]@{
            Native = Measure-CommandPerformance -Name 'Get-DiskUsageFast Native' -Iterations $Iterations -Warmup $Warmup -Command {
                Get-DiskUsageFast -Path $benchmarkRoot -Top 20 -Depth 2 | Out-Null
            }
            Fallback = Measure-CommandPerformance -Name 'Get-DiskUsageFast PowerShell' -Iterations $Iterations -Warmup $Warmup -Command {
                & $module { param($path) Get-DiskUsageParallel -Path $path -Depth 2 -ThrottleLimit ([Environment]::ProcessorCount) } $benchmarkRoot | Out-Null
            }
        }
        $results.Disk | Add-Member -NotePropertyName Speedup -NotePropertyValue ([Math]::Round($results.Disk.Fallback.Mean / $results.Disk.Native.Mean, 2))
    }

    [pscustomobject]$results
}
finally {
    Remove-Item -Path $benchmarkRoot -Recurse -Force -ErrorAction SilentlyContinue
}
