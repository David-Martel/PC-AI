#Requires -Version 7.0
<#
.SYNOPSIS
    Compares selected command paths with explicit PowerShell fallbacks.
.DESCRIPTION
    Candidate contains measured selected-backend timings, including observation
    collection. ActualBackend summarizes returned Tool metadata; Observations
    contains each invocation's counts and CPU units. Missing metadata is unreported.
    Native and Speedup remain null because this legacy matrix does not establish
    useful-output parity. CPU sorting can compare different units, and disk paths
    can return different selections. Use Tests/Benchmarks/Invoke-PcaiToolingBenchmarks.ps1
    from the repository for maintained, case-specific qualification.
.EXAMPLE
    ./Measure-PcaiCommandMatrix.ps1 -Iterations 1 -Warmup 0 -SkipProcesses -SkipDisk
#>
[CmdletBinding()]
param(
    [Parameter()]
    [ValidateRange(1,1000)][int]$Iterations = 3,

    [Parameter()]
    [ValidateRange(0,1000)][int]$Warmup = 1,

    [Parameter()]
    [switch]$SkipProcesses,

    [Parameter()]
    [switch]$SkipDisk,

    [Parameter()]
    [switch]$SkipSearch
)

$moduleRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$module = Import-Module (Join-Path $moduleRoot 'PC-AI.Acceleration.psd1') -Force -PassThru

function Measure-PcaiObservedCandidate {
    param([string]$Name,[scriptblock]$Command)
    $observations=[Collections.Generic.List[object]]::new()
    $observedCommand={
        $rows=@(& $Command)
        $backends=@(foreach($row in $rows){
            if($row.PSObject.Properties['Tool'] -and $row.Tool){[string]$row.Tool}else{'unreported'}
        })
        $units=@(foreach($row in $rows){if($row.PSObject.Properties['CPUUnit']){[string]$row.CPUUnit}})
        $observations.Add([pscustomobject]@{RowCount=$rows.Count;Backends=@($backends|Sort-Object -Unique);CPUUnits=@($units|Sort-Object -Unique)})
    }.GetNewClosure()
    $measurement=& $module {
        param($command,$name,$iterations,$warmup)
        Measure-CommandPerformance -Command $command -Name $name -Iterations $iterations -Warmup $warmup
    } $observedCommand "$Name selected backend" $Iterations $Warmup
    [pscustomobject]@{Measurement=$measurement;Invocations=@($observations.ToArray())}
}

function Complete-PcaiMatrixResult {
    param([pscustomobject]$Result,[string]$Reason)
    $observed=$Result.Candidate
    $Result.Candidate=$observed.Measurement
    $Result|Add-Member -NotePropertyName Native -NotePropertyValue $null
    $Result|Add-Member -NotePropertyName Speedup -NotePropertyValue $null
    $Result|Add-Member -NotePropertyName ActualBackend -NotePropertyValue @($observed.Invocations.Backends|Sort-Object -Unique)
    $Result|Add-Member -NotePropertyName Observations -NotePropertyValue $observed.Invocations
    $Result|Add-Member -NotePropertyName Qualification -NotePropertyValue ([pscustomobject]@{
        NativeTimingQualified=$false;UsefulOutputParity='not-established';Reason=$Reason
        SourceSha256=(Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash;ModulePath=$module.Path
    })
}

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
        Candidate = Measure-PcaiObservedCandidate -Name 'Find-FilesFast' -Command {
            Find-FilesFast -Path $srcRoot -Extension ps1 -PreferNative
        }
        Fallback = Measure-CommandPerformance -Name 'Find-FilesFast GetChildItem' -Iterations $Iterations -Warmup $Warmup -Command {
            & $module { param($path) Find-WithGetChildItem -Path $path -Extension @('ps1') } $srcRoot | Out-Null
        }
    }
    Complete-PcaiMatrixResult $results.FindFiles 'File results do not expose their backend; PreferNative may fall back.'

    if (-not $SkipSearch) {
        $results.SearchContent = [pscustomobject]@{
            Candidate = Measure-PcaiObservedCandidate -Name 'Search-ContentFast' -Command {
                Search-ContentFast -Path $srcRoot -LiteralPattern 'TODO item' -FilePattern '*.ps1'
            }
            Fallback = Measure-CommandPerformance -Name 'Search-ContentFast Select-String' -Iterations $Iterations -Warmup $Warmup -Command {
                & $module {
                    param($path)
                    Search-WithParallelSelectString -Path $path -LiteralPattern 'TODO item' -SearchPattern ([regex]::Escape('TODO item')) -FilePattern @('*.ps1') -Context 0 -CaseSensitive:$false -WholeWord:$false -Invert:$false -MaxResults 0 -FilesOnly:$false -ThrottleLimit ([Environment]::ProcessorCount)
                } $srcRoot | Out-Null
            }
        }
        Complete-PcaiMatrixResult $results.SearchContent 'Returned backend identity is observed; match-by-match parity is not established.'

        $results.SearchLogs = [pscustomobject]@{
            Candidate = Measure-PcaiObservedCandidate -Name 'Search-LogsFast' -Command {
                Search-LogsFast -Path $logRoot -Pattern 'ERROR|WARN' -Include '*.log'
            }
            Fallback = Measure-CommandPerformance -Name 'Search-LogsFast Select-String' -Iterations $Iterations -Warmup $Warmup -Command {
                & $module {
                    param($path)
                    Search-WithSelectString -Path $path -Pattern 'ERROR|WARN' -Include @('*.log') -Context 0 -CaseSensitive:$false -MaxCount 0 -CountOnly:$false
                } $logRoot | Out-Null
            }
        }
        Complete-PcaiMatrixResult $results.SearchLogs 'Returned backend identity is observed; match-by-match parity is not established.'
    }

    if (-not $SkipProcesses) {
        $results.Processes = [pscustomobject]@{
            Candidate = Measure-PcaiObservedCandidate -Name 'Get-ProcessesFast' -Command {
                Get-ProcessesFast -Top 20 -SortBy cpu
            }
            Fallback = Measure-CommandPerformance -Name 'Get-ProcessesFast PowerShell' -Iterations $Iterations -Warmup $Warmup -Command {
                & $module { Get-ProcessesParallel -Top 20 -SortBy cpu } | Out-Null
            }
        }
        Complete-PcaiMatrixResult $results.Processes 'Live process sets vary; native percent and fallback lifetime seconds have different CPU-sort semantics.'
    }

    if (-not $SkipDisk) {
        $results.Disk = [pscustomobject]@{
            Candidate = Measure-PcaiObservedCandidate -Name 'Get-DiskUsageFast' -Command {
                Get-DiskUsageFast -Path $benchmarkRoot -Top 20 -Depth 2
            }
            Fallback = Measure-CommandPerformance -Name 'Get-DiskUsageFast PowerShell' -Iterations $Iterations -Warmup $Warmup -Command {
                & $module { param($path) Get-DiskUsageParallel -Path $path -Depth 2 -ThrottleLimit ([Environment]::ProcessorCount) } $benchmarkRoot | Out-Null
            }
        }
        Complete-PcaiMatrixResult $results.Disk 'Selected top rows and fallback depth traversal are not qualified as equivalent output.'
    }

    [pscustomobject]$results
}
finally {
    Remove-Item -Path $benchmarkRoot -Recurse -Force -ErrorAction SilentlyContinue
}
