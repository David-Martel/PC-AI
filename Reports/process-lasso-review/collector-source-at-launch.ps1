#Requires -Version 7.0
<#
.SYNOPSIS
Collects bounded, read-only workload resource samples for scheduling decisions.
.DESCRIPTION
Streams rotating JSONL samples and a JSON summary. Process CPU and IO rates use
actual elapsed intervals and PID plus creation time. Process IO is all IO, not
disk or network attribution. Names, PIDs and parent identities are collected;
command lines, window titles and remote endpoints are never collected.
No priorities, services, tasks, mounts or Process Lasso settings are changed.
Use separate labeled captures for idle, normal development and live calls.
.PARAMETER DurationMinutes
Requested duration, 60 by default, bounded to 24 hours. Initialization is excluded.
.PARAMETER IntervalSeconds
Target sampling interval, 30 by default to limit observer CPU on large process
populations. Overruns are recorded rather than hidden.
.PARAMETER OutputPath
New or empty report directory. Existing reports are never overwritten.
.PARAMETER Label
Scenario label, for example teams-live-screen-share. Do not put secrets in labels.
.PARAMETER MaxOutputMB
Maximum sample output budget. Capture stops early at the budget; summary is extra.
.PARAMETER RotateMB
Maximum approximate JSONL part size. A single sample may exceed this size.
.PARAMETER IncludeGpu
Sample aggregate Windows GPU engine and adapter-memory counters every 60 seconds.
This provider can be expensive on hosts with many GPU engine instances; use only
for targeted captures. Counters are optional; missing data is null and errors are recorded.
.PARAMETER DryRun
Return the collection plan without queries, compilation, sleeping or file writes.
.PARAMETER Help
Print help and exit. -h and --help are supported.
.EXAMPLE
pwsh -NoProfile -File .\Tools\Collect-WorkloadResourceProfile.ps1 -DurationMinutes 60 -Label teams-live
.EXAMPLE
pwsh -NoProfile -File .\Tools\Collect-WorkloadResourceProfile.ps1 -DryRun --help
#>
[CmdletBinding(PositionalBinding = $false)]
param(
    [ValidateRange(0.01, 1440)] [double]$DurationMinutes = 60,
    [ValidateRange(1, 300)] [double]$IntervalSeconds = 30,
    [string]$OutputPath = '',
    [ValidateLength(1, 100)] [string]$Label = 'baseline',
    [ValidateRange(1, 4096)] [int]$MaxOutputMB = 512,
    [ValidateRange(1, 128)] [int]$RotateMB = 16,
    [switch]$IncludeGpu,
    [switch]$DryRun,
    [Alias('h', '?')] [switch]$Help,
    [Parameter(ValueFromRemainingArguments = $true)] [string[]]$CliArgs
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$CliArgs = @($CliArgs | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
if ($CliArgs -contains '--help') { $Help = $true; $CliArgs = @($CliArgs | Where-Object { $_ -ne '--help' }) }
if ($CliArgs -contains '--DryRun') { $DryRun = $true; $CliArgs = @($CliArgs | Where-Object { $_ -ne '--DryRun' }) }
if ($Help) {
    [regex]::Match([IO.File]::ReadAllText($PSCommandPath), '(?s)<#\s*(.*?)\s*#>').Groups[1].Value.Trim()
    return
}
if ($CliArgs.Count -gt 0) { throw "Unknown CLI argument(s): $($CliArgs -join ', ')" }

function Get-WorkloadPercentile {
    param([object[]]$Values, [ValidateRange(0, 100)] [double]$Percentile)
    $sorted = @($Values | Where-Object { $null -ne $_ } | Sort-Object)
    if ($sorted.Count -eq 0) { return $null }
    # Nearest-rank: each observed host sample has equal weight.
    $index = [math]::Max([double]0, [math]::Ceiling(($Percentile / 100) * $sorted.Count) - 1)
    return [double]$sorted[$index]
}

function Get-WorkloadDelta {
    param($Current, $Previous, [double]$ElapsedSeconds, [int]$LogicalProcessors)
    $result = [ordered]@{ CpuHostPercent = $null; CpuSeconds = $null; IoReadBytesPerSecond = $null; IoWriteBytesPerSecond = $null }
    if ($null -eq $Previous -or $ElapsedSeconds -le 0 -or $LogicalProcessors -lt 1 -or $Current.Pid -eq 0 -or
        $null -eq $Current.CreationUtc -or $Current.Identity -ne $Previous.Identity) { return [pscustomobject]$result }
    if ($null -ne $Current.CpuSeconds -and $null -ne $Previous.CpuSeconds -and $Current.CpuSeconds -ge $Previous.CpuSeconds) {
        $result.CpuSeconds = $Current.CpuSeconds - $Previous.CpuSeconds
        $result.CpuHostPercent = 100 * $result.CpuSeconds / $ElapsedSeconds / $LogicalProcessors
    }
    foreach ($pair in @(@('IoReadBytes', 'IoReadBytesPerSecond'), @('IoWriteBytes', 'IoWriteBytesPerSecond'))) {
        if ($null -ne $Current.($pair[0]) -and $null -ne $Previous.($pair[0]) -and $Current.($pair[0]) -ge $Previous.($pair[0])) {
            $result[$pair[1]] = ($Current.($pair[0]) - $Previous.($pair[0])) / $ElapsedSeconds
        }
    }
    return [pscustomobject]$result
}

function Get-WorkloadCohort {
    param([string]$Name)
    if ($Name -match '^(Zoom|ms-teams|Teams|audiodg|obs64|Webex|CptHost|Video.UI)(\.exe)?$') { return 'RealtimeMedia' }
    if ($Name -match '^(codex|claude|gemini|llama.*|mistral.*|pcai.*|ollama.*)(\.exe)?$') { return 'LLMTools' }
    if ($Name -match '^(gh|git|cargo|rustc|uv|dotnet|MSBuild|VBCSCompiler)(\.exe)?$') { return 'BuildGitEvaluation' }
    if ($Name -match '^(pwsh|powershell|node|python.*|bun|deno)(\.exe)?$') { return 'SharedRuntimesNeedsContext' }
    if ($Name -match 'OneDrive|Dropbox|GoogleDrive|SearchIndexer|MsMpEng') { return 'SyncSearchSecurity' }
    return 'OtherNeedsContext'
}

function Get-WorkloadHostSample {
    param([System.Collections.Generic.List[string]]$Issues)
    $paths = @(
        '\Processor(_Total)\% Processor Time', '\Processor(_Total)\% DPC Time', '\Processor(_Total)\% Interrupt Time', '\Memory\Available MBytes',
        '\Memory\Committed Bytes', '\Memory\Commit Limit', '\Memory\Pages Input/sec',
        '\PhysicalDisk(_Total)\Avg. Disk sec/Transfer', '\PhysicalDisk(_Total)\Current Disk Queue Length',
        '\Network Interface(*)\Bytes Total/sec'
    )
    $sample = [ordered]@{ CpuPercent = $null; DpcPercent = $null; InterruptPercent = $null; SampledProcessCpuPercent = $null; CpuAccountingGapPercent = $null;
        CollectorCpuHostPercent = $null;
        AvailableMB = $null; CommittedBytes = $null; CommitLimitBytes = $null;
        CommitPercent = $null; PagesInputPerSecond = $null; DiskTransferLatencyMs = $null; DiskQueue = $null; NetworkAdapters = @() }
    try {
        $cache = Get-Variable WorkloadCounterCache -Scope Script -ErrorAction SilentlyContinue
        if ($null -ne $cache -and $null -ne $cache.Value) {
            $samples = [System.Collections.Generic.List[object]]::new()
            foreach ($entry in $cache.Value) {
                try { $samples.Add([pscustomobject]@{ Path = $entry.Path; InstanceName = $entry.Instance; Status = 0; CookedValue = $entry.Counter.NextValue() }) }
                catch { $Issues.Add('counter-unavailable:' + $entry.Path + ':' + $_.Exception.Message) }
            }
            $data = [pscustomobject]@{ CounterSamples = @($samples) }
        } else { $data = Get-Counter -Counter $paths -ErrorAction Stop }
        foreach ($counter in $data.CounterSamples) {
            if ($counter.Status -ne 0) { $Issues.Add("counter-status:$($counter.Status):$($counter.Path)"); continue }
            $value = [double]$counter.CookedValue
            switch -Regex ($counter.Path) {
                '\\processor\(_total\)\\% processor time$' { $sample.CpuPercent = $value }
                '\\processor\(_total\)\\% dpc time$' { $sample.DpcPercent = $value }
                '\\processor\(_total\)\\% interrupt time$' { $sample.InterruptPercent = $value }
                '\\available mbytes$' { $sample.AvailableMB = $value }
                '\\committed bytes$' { $sample.CommittedBytes = $value }
                '\\commit limit$' { $sample.CommitLimitBytes = $value }
                '\\pages input/sec$' { $sample.PagesInputPerSecond = $value }
                '\\avg\. disk sec/transfer$' { $sample.DiskTransferLatencyMs = $value * 1000 }
                '\\current disk queue length$' { $sample.DiskQueue = $value }
                '\\network interface\(' { $sample.NetworkAdapters += [pscustomobject]@{ Adapter = $counter.InstanceName; BytesPerSecond = $value } }
            }
        }
        if ($null -ne $sample.CommittedBytes -and $sample.CommitLimitBytes -gt 0) { $sample.CommitPercent = 100 * $sample.CommittedBytes / $sample.CommitLimitBytes }
    } catch { $Issues.Add('host-counters-unavailable:' + $_.Exception.Message) }
    return [pscustomobject]$sample
}

function Get-WorkloadProcesses {
    param([System.Collections.Generic.List[string]]$Issues)
    $inaccessible = 0
    $parentCache = Get-Variable WorkloadParentCache -Scope Script -ErrorAction SilentlyContinue
    try {
        foreach ($process in Get-Process -ErrorAction Stop) {
            $creation = $null; $creationKey = $null; $cpu = $null; $ioRead = $null; $ioWrite = $null; $parentPid = $null
            try {
                $created = $process.StartTime.ToUniversalTime(); $creation = $created.ToString('o')
                $creationKey = Get-WorkloadCreationKey $created; $cpu = $process.TotalProcessorTime.TotalSeconds
            }
            catch { $inaccessible++ }
            try {
                $io = [PcaiWorkloadIo]::Read($process.Handle)
                if ($io.Success) { $ioRead = $io.ReadBytes; $ioWrite = $io.WriteBytes }
            } catch { $inaccessible++ }
            if ($null -ne $parentCache -and $null -ne $creation) {
                $parentEntry = $parentCache.Value[$process.Id]
                if ($null -ne $parentEntry -and $parentEntry.CreationKey -eq $creationKey) { $parentPid = $parentEntry.ParentPid }
            }
            [pscustomobject]@{
                Identity = "$($process.Id):$creation"; Pid = [int]$process.Id; Name = $process.ProcessName + '.exe'
                CreationUtc = $creation; ParentPid = $parentPid; ParentIdentity = $null
                CpuSeconds = $cpu; PrivateBytes = $process.PrivateMemorySize64; WorkingSetBytes = $process.WorkingSet64
                IoReadBytes = $ioRead; IoWriteBytes = $ioWrite
                Threads = $null; Handles = $process.HandleCount; BasePriority = $process.BasePriority
                IntervalCpuSeconds = $null; IntervalCpuHostPercent = $null; IntervalIoReadBytesPerSecond = $null; IntervalIoWriteBytesPerSecond = $null
            }
            $process.Dispose()
        }
    } catch { $Issues.Add('process-query-unavailable:' + $_.Exception.Message) }
    if ($inaccessible -gt 0) { $Issues.Add("process-properties-inaccessible:$inaccessible") }
}

function Update-WorkloadParentCache {
    param([System.Collections.Generic.List[string]]$Issues)
    try {
        $cache = @{}
        foreach ($process in Get-CimInstance Win32_Process -Property ProcessId, ParentProcessId, CreationDate -OperationTimeoutSec 5 -ErrorAction Stop) {
            if ($null -ne $process.CreationDate) {
                $cache[[int]$process.ProcessId] = [pscustomobject]@{ CreationKey = (Get-WorkloadCreationKey $process.CreationDate); ParentPid = [int]$process.ParentProcessId }
            }
        }
        $script:WorkloadParentCache = $cache
    } catch { $Issues.Add('parent-cache-unavailable:' + $_.Exception.Message) }
}

function Get-WorkloadCreationKey {
    param([datetime]$Date)
    # CIM creation dates have microsecond precision; native start times have 100ns.
    $ticks = $Date.ToUniversalTime().Ticks
    return $ticks - ($ticks % 10)
}

if ([string]::IsNullOrWhiteSpace($OutputPath)) {
    $OutputPath = Join-Path (Split-Path -Parent $PSScriptRoot) ('Reports\workload-profiles\' + [datetime]::UtcNow.ToString('yyyyMMddTHHmmssfffZ'))
}
$resolvedOutput = [IO.Path]::GetFullPath($OutputPath)
foreach ($component in $resolvedOutput -split '[\\/]') {
    if ($component -match '^(?i:\$null|AUX|CON|NUL|PRN|COM[1-9]|LPT[1-9])(?:\..*)?[ .]*$') { throw "Unsafe output path component: $component" }
}
$plan = [pscustomobject]@{ OutputPath = $resolvedOutput; Label = $Label; DurationMinutes = $DurationMinutes;
    IntervalSeconds = $IntervalSeconds; MaxOutputMB = $MaxOutputMB; RotateMB = $RotateMB; IncludeGpu = [bool]$IncludeGpu;
    Privacy = 'No command lines, window titles or remote endpoints'; DryRun = [bool]$DryRun }
if ($DryRun) { return $plan }
if (-not $IsWindows) { throw 'This collector requires Windows performance counters and Win32_Process.' }
if ([IO.Directory]::Exists($resolvedOutput) -and [IO.Directory]::EnumerateFileSystemEntries($resolvedOutput).GetEnumerator().MoveNext()) {
    throw 'OutputPath must be new or empty; existing evidence will not be overwritten.'
}
[IO.Directory]::CreateDirectory($resolvedOutput) | Out-Null
$utf8 = [Text.UTF8Encoding]::new($false)
$issues = [System.Collections.Generic.List[string]]::new()
if (-not ('PcaiWorkloadIo' -as [type])) {
    Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class PcaiWorkloadIo {
    [StructLayout(LayoutKind.Sequential)] struct Counters {
        public ulong ReadOps, WriteOps, OtherOps, ReadBytes, WriteBytes, OtherBytes;
    }
    [DllImport("kernel32.dll", SetLastError=true)] static extern bool GetProcessIoCounters(IntPtr handle, out Counters counters);
    public sealed class Sample { public bool Success; public ulong ReadBytes, WriteBytes; }
    public static Sample Read(IntPtr handle) {
        Counters c; bool ok = GetProcessIoCounters(handle, out c);
        return new Sample { Success = ok, ReadBytes = c.ReadBytes, WriteBytes = c.WriteBytes };
    }
}
'@
}
Update-WorkloadParentCache -Issues $issues
$script:WorkloadCounterCache = [System.Collections.Generic.List[object]]::new()
foreach ($definition in @(
    @('Processor', '% Processor Time', '_Total'), @('Processor', '% DPC Time', '_Total'), @('Processor', '% Interrupt Time', '_Total'),
    @('Memory', 'Available MBytes', ''), @('Memory', 'Committed Bytes', ''), @('Memory', 'Commit Limit', ''), @('Memory', 'Pages Input/sec', ''),
    @('PhysicalDisk', 'Avg. Disk sec/Transfer', '_Total'), @('PhysicalDisk', 'Current Disk Queue Length', '_Total')
)) {
    try {
        $counter = [Diagnostics.PerformanceCounter]::new($definition[0], $definition[1], $definition[2], $true)
        [void]$counter.NextValue()
        $instancePart = if ($definition[2]) { '(' + $definition[2] + ')' } else { '' }
        $script:WorkloadCounterCache.Add([pscustomobject]@{ Counter = $counter; Path = '\' + $definition[0] + $instancePart + '\' + $definition[1]; Instance = $definition[2] })
    } catch { $issues.Add('counter-initialization-unavailable:' + $definition[0] + ':' + $definition[1] + ':' + $_.Exception.Message) }
}
try {
    $networkCategory = [Diagnostics.PerformanceCounterCategory]::new('Network Interface')
    foreach ($adapter in $networkCategory.GetInstanceNames()) {
        $counter = [Diagnostics.PerformanceCounter]::new('Network Interface', 'Bytes Total/sec', $adapter, $true)
        [void]$counter.NextValue()
        $script:WorkloadCounterCache.Add([pscustomobject]@{ Counter = $counter; Path = '\Network Interface(' + $adapter + ')\Bytes Total/sec'; Instance = $adapter })
    }
} catch { $issues.Add('network-counter-initialization-unavailable:' + $_.Exception.Message) }
$cpuInventory = @(Get-CimInstance Win32_Processor -Property Name, NumberOfCores, NumberOfLogicalProcessors -OperationTimeoutSec 5 -ErrorAction Stop |
    Select-Object Name, NumberOfCores, NumberOfLogicalProcessors)
$logical = [int](($cpuInventory | Measure-Object NumberOfLogicalProcessors -Sum).Sum)
$totalMemory = (Get-CimInstance Win32_ComputerSystem -Property TotalPhysicalMemory -OperationTimeoutSec 5 -ErrorAction Stop).TotalPhysicalMemory
$metadata = [ordered]@{ SchemaVersion = 1; Plan = $plan; Host = $env:COMPUTERNAME; Processors = $cpuInventory;
    LogicalProcessors = $logical; TotalPhysicalMemoryBytes = $totalMemory; CollectorPid = $PID; StartUtc = [datetime]::UtcNow.ToString('o');
    ProcessIoMeaning = 'All process IO including cached/file/device/network IO; not disk/network attribution';
    CpuMeaning = 'Interval CPU delta divided by elapsed seconds and host logical processors; unknown/new identities are null';
    CounterMeaning = 'Cached native counter rates use elapsed sample intervals; CPU accounting gaps remain diagnostic because process samples are not atomic. Pages Input/sec includes hard faults, not proof of pagefile swapping. English counter names may be unavailable on localized hosts';
    ParentMeaning = 'PID and creation-date guarded parent cache refreshes every 60 seconds; new processes may have null parents. Threads omitted to avoid enumerating every thread.';
    DurationMeaning = 'Stops at requested elapsed duration between ticks; in-flight provider calls can overrun. CIM requests have a five-second operation timeout. Network adapters are enumerated at initialization.';
    GpuMeaning = 'Optional aggregate engine instances; engine sums are not a single GPU utilization percentage';
    SummaryMeaning = 'Nearest-rank sample quantiles; per-name memory totals sum instances and working sets may contain shared pages' }
[IO.File]::WriteAllText((Join-Path $resolvedOutput 'metadata.json'), ($metadata | ConvertTo-Json -Depth 8), $utf8)
foreach ($entry in $script:WorkloadCounterCache) {
    try { [void]$entry.Counter.NextValue() } catch { $issues.Add('counter-prime-unavailable:' + $entry.Path) }
}
$clock = [Diagnostics.Stopwatch]::StartNew()
$collector = [Diagnostics.Process]::GetCurrentProcess()
$collectorCpuStart = $collector.TotalProcessorTime.TotalSeconds
$previous = @{}; $previousTime = $null
$hostHistory = [System.Collections.Generic.List[object]]::new()
$cohorts = @{}; $identities = [System.Collections.Generic.HashSet[string]]::new()
$sampleCount = 0; $gapCount = 0; $maxGap = 0.0; $overheadMs = [System.Collections.Generic.List[object]]::new()
$serializationMs = [System.Collections.Generic.List[object]]::new()
$part = 0; $bytesWritten = 0L; $partBytes = 0L; $writer = $null; $status = 'completed'; $nextGpu = 0.0; $nextParents = 60.0; $gpu = $null
try {
    do {
        $tickStart = $clock.Elapsed.TotalSeconds
        $tickIssues = [System.Collections.Generic.List[string]]::new()
        if ($tickStart -ge $nextParents) { Update-WorkloadParentCache -Issues $tickIssues; $nextParents = $tickStart + 60 }
        $processQueryStart = $clock.Elapsed.TotalSeconds
        $processes = @(Get-WorkloadProcesses -Issues $tickIssues)
        $processTime = $clock.Elapsed.TotalSeconds
        $processQueryMs = ($processTime - $processQueryStart) * 1000
        $interval = if ($null -eq $previousTime) { $null } else { $processTime - $previousTime }
        if ($null -ne $interval) { $maxGap = [math]::Max($maxGap, $interval); if ($interval -gt $IntervalSeconds * 1.5) { $gapCount++ } }
        $hostSample = Get-WorkloadHostSample -Issues $tickIssues
        $hostQueryMs = ($clock.Elapsed.TotalSeconds - $processTime) * 1000
        $gpuStart = $clock.Elapsed.TotalSeconds
        if ($IncludeGpu -and $tickStart -ge $nextGpu) {
            $nextGpu = $tickStart + 60
            try {
                $gpuData = Get-Counter '\GPU Engine(*)\Utilization Percentage', '\GPU Adapter Memory(*)\Dedicated Usage' -ErrorAction Stop
                $gpu = [pscustomobject]@{ TimestampUtc = [datetime]::UtcNow.ToString('o'); Counters = @($gpuData.CounterSamples |
                    Where-Object Status -EQ 0 | Select-Object InstanceName, Path, CookedValue) }
            } catch { $gpu = $null; $tickIssues.Add('gpu-counters-unavailable:' + $_.Exception.Message) }
        }
        $gpuQueryMs = ($clock.Elapsed.TotalSeconds - $gpuStart) * 1000
        $byPid = @{}; $current = @{}
        $processingStart = $clock.Elapsed.TotalSeconds
        foreach ($process in $processes) { $byPid[$process.Pid] = $process }
        $nameTotals = @{}
        foreach ($process in $processes) {
            $current[$process.Identity] = $process
            $parent = if ($null -ne $process.ParentPid) { $byPid[$process.ParentPid] } else { $null }
            if ($null -ne $parent -and $null -ne $parent.CreationUtc -and $null -ne $process.CreationUtc -and
                $parent.CreationUtc -le $process.CreationUtc -and $parent.Pid -ne $process.Pid) { $process.ParentIdentity = $parent.Identity }
            $delta = Get-WorkloadDelta -Current $process -Previous $previous[$process.Identity] -ElapsedSeconds $interval -LogicalProcessors $logical
            $process.IntervalCpuSeconds = $delta.CpuSeconds; $process.IntervalCpuHostPercent = $delta.CpuHostPercent
            $process.IntervalIoReadBytesPerSecond = $delta.IoReadBytesPerSecond; $process.IntervalIoWriteBytesPerSecond = $delta.IoWriteBytesPerSecond
            if ($process.Pid -eq $PID) { $hostSample.CollectorCpuHostPercent = $delta.CpuHostPercent }
            if (-not $nameTotals.ContainsKey($process.Name)) { $nameTotals[$process.Name] = @{ Count = 0; CpuSeconds = 0.0; CpuHostPercent = 0.0; KnownCpu = 0;
                PrivateBytes = $null; WorkingSetBytes = $null; KnownPrivate = 0; KnownWorkingSet = 0 } }
            $total = $nameTotals[$process.Name]; $total.Count++
            if ($null -ne $process.PrivateBytes) { $total.PrivateBytes += [double]$process.PrivateBytes; $total.KnownPrivate++ }
            if ($null -ne $process.WorkingSetBytes) { $total.WorkingSetBytes += [double]$process.WorkingSetBytes; $total.KnownWorkingSet++ }
            if ($null -ne $delta.CpuSeconds) { $total.CpuSeconds += $delta.CpuSeconds; $total.CpuHostPercent += $delta.CpuHostPercent; $total.KnownCpu++ }
        }
        if ($null -ne $interval -and (($nameTotals.Values | Measure-Object KnownCpu -Sum).Sum -gt 0)) {
            $hostSample.SampledProcessCpuPercent = [double](($nameTotals.Values | Measure-Object CpuHostPercent -Sum).Sum)
            if ($null -ne $hostSample.CpuPercent) { $hostSample.CpuAccountingGapPercent = $hostSample.CpuPercent - $hostSample.SampledProcessCpuPercent }
        }
        $record = [ordered]@{ Sample = $sampleCount; TimestampUtc = [datetime]::UtcNow.ToString('o'); MonotonicSeconds = $processTime;
            ElapsedIntervalSeconds = $interval; ProcessCount = $processes.Count; Host = $hostSample; Processes = $processes; Gpu = $gpu; Issues = @($tickIssues);
            StageTimingsMs = @{ ProcessQuery = $processQueryMs; HostQuery = $hostQueryMs; GpuQuery = $gpuQueryMs; ProcessRates = ($clock.Elapsed.TotalSeconds - $processingStart) * 1000 } }
        $serializationStart = $clock.Elapsed.TotalSeconds
        $line = ($record | ConvertTo-Json -Depth 8 -Compress) + "`n"
        $serializationMs.Add(($clock.Elapsed.TotalSeconds - $serializationStart) * 1000)
        $lineBytes = $utf8.GetByteCount($line)
        if ($bytesWritten + $lineBytes -gt $MaxOutputMB * 1MB) { $status = 'output-budget'; break }
        $newIdentities = @($processes | Where-Object { -not $identities.Contains($_.Identity) })
        if ($identities.Count + $newIdentities.Count -gt 100000) { $status = 'identity-budget'; break }
        foreach ($process in $newIdentities) { [void]$identities.Add($process.Identity) }
        foreach ($name in $nameTotals.Keys) {
            $total = $nameTotals[$name]
            if (-not $cohorts.ContainsKey($name)) { $cohorts[$name] = @{ Name = $name; Cohort = (Get-WorkloadCohort $name); Samples = 0; CpuSeconds = 0.0; KnownCpuIntervals = 0;
                PrivateBytesSum = 0.0; KnownPrivateSamples = 0; PeakPrivateBytes = $null; PeakWorkingSetBytes = $null; PeakInstances = 0 } }
            $aggregate = $cohorts[$name]; $aggregate.Samples++; $aggregate.CpuSeconds += $total.CpuSeconds; $aggregate.KnownCpuIntervals += $total.KnownCpu
            if ($total.KnownPrivate -gt 0) {
                $aggregate.PrivateBytesSum += $total.PrivateBytes; $aggregate.KnownPrivateSamples++
                $aggregate.PeakPrivateBytes = [math]::Max([double]$aggregate.PeakPrivateBytes, [double]$total.PrivateBytes)
            }
            if ($total.KnownWorkingSet -gt 0) { $aggregate.PeakWorkingSetBytes = [math]::Max([double]$aggregate.PeakWorkingSetBytes, [double]$total.WorkingSetBytes) }
            $aggregate.PeakInstances = [math]::Max($aggregate.PeakInstances, $total.Count)
        }
        $hostHistory.Add($hostSample)
        if ($null -eq $writer -or $partBytes + $lineBytes -gt $RotateMB * 1MB) {
            if ($null -ne $writer) { $writer.Dispose() }
            $part++; $partBytes = 0
            $writer = [IO.StreamWriter]::new((Join-Path $resolvedOutput ('samples-{0:d4}.jsonl' -f $part)), $false, $utf8)
        }
        $writer.Write($line); $writer.Flush(); $partBytes += $lineBytes; $bytesWritten += $lineBytes; $sampleCount++
        foreach ($issue in $tickIssues) { if (-not $issues.Contains($issue) -and $issues.Count -lt 200) { $issues.Add($issue) } }
        $previous = $current; $previousTime = $processTime
        $tickCost = $clock.Elapsed.TotalSeconds - $tickStart; $overheadMs.Add($tickCost * 1000)
        $remaining = $DurationMinutes * 60 - $clock.Elapsed.TotalSeconds
        if ($remaining -le 0) { break }
        $sleep = [math]::Min($remaining, [math]::Max([double]0, [double]($IntervalSeconds - $tickCost)))
        if ($sleep -gt 0) { Start-Sleep -Milliseconds ([int]($sleep * 1000)) }
    } while ($clock.Elapsed.TotalSeconds -lt $DurationMinutes * 60)
} catch { $status = 'failed'; $issues.Add($_.Exception.Message); throw }
finally {
    if ($null -ne $writer) { $writer.Dispose() }
    foreach ($entry in $script:WorkloadCounterCache) { $entry.Counter.Dispose() }
    $collector.Refresh(); $cpuConsumed = $collector.TotalProcessorTime.TotalSeconds - $collectorCpuStart
    $hostStats = [ordered]@{}
    foreach ($metric in @('CpuPercent', 'DpcPercent', 'InterruptPercent', 'SampledProcessCpuPercent', 'CpuAccountingGapPercent', 'CollectorCpuHostPercent',
        'AvailableMB', 'CommitPercent', 'PagesInputPerSecond', 'DiskTransferLatencyMs', 'DiskQueue')) {
        $values = @($hostHistory | ForEach-Object { $_.$metric })
        $hostStats[$metric] = @{ KnownSamples = @($values | Where-Object { $null -ne $_ }).Count;
            P50 = (Get-WorkloadPercentile $values 50); P95 = (Get-WorkloadPercentile $values 95); P99 = (Get-WorkloadPercentile $values 99) }
    }
    $ranked = @($cohorts.Values | ForEach-Object {
        $meanPrivate = if ($_.KnownPrivateSamples -gt 0) { $_.PrivateBytesSum / $_.KnownPrivateSamples } else { $null }
        $cpuSeconds = if ($_.KnownCpuIntervals -gt 0) { $_.CpuSeconds } else { $null }
        [pscustomobject]@{ Name = $_.Name; Cohort = $_.Cohort; ObservedSamples = $_.Samples; CpuSeconds = $cpuSeconds;
            KnownCpuIntervals = $_.KnownCpuIntervals; KnownPrivateSamples = $_.KnownPrivateSamples; MeanPrivateBytesWhenPresent = $meanPrivate;
            PeakPrivateBytes = $_.PeakPrivateBytes; PeakWorkingSetBytes = $_.PeakWorkingSetBytes; PeakInstances = $_.PeakInstances }
    })
    $summary = [ordered]@{ Status = $status; StartUtc = $metadata.StartUtc; EndUtc = [datetime]::UtcNow.ToString('o');
        RequestedDurationSeconds = $DurationMinutes * 60; ActualDurationSeconds = $clock.Elapsed.TotalSeconds; Samples = $sampleCount;
        ObservedProcessIdentities = $identities.Count; Parts = $part; SampleBytes = $bytesWritten; GapsOver150Percent = $gapCount; MaximumIntervalSeconds = $maxGap;
        CollectorCpuSeconds = $cpuConsumed; CollectorMeanHostCpuPercent = 100 * $cpuConsumed / [math]::Max(0.001, $clock.Elapsed.TotalSeconds) / $logical;
        CollectionWallTimeMsP50 = (Get-WorkloadPercentile @($overheadMs) 50); CollectionWallTimeMsP95 = (Get-WorkloadPercentile @($overheadMs) 95);
        SerializationMsP50 = (Get-WorkloadPercentile @($serializationMs) 50); SerializationMsP95 = (Get-WorkloadPercentile @($serializationMs) 95);
        HostStats = $hostStats; TopByCpuSeconds = @($ranked | Sort-Object CpuSeconds -Descending | Select-Object -First 30);
        TopByPrivateBytes = @($ranked | Sort-Object PeakPrivateBytes -Descending | Select-Object -First 30); AllProcessNames = $ranked; Issues = @($issues);
        Interpretation = 'Resource usage does not establish waste. Shared runtimes require workload context; LLM, git and evaluation are intentional work. Match calls and glitches with labeled time windows.' }
    [IO.File]::WriteAllText((Join-Path $resolvedOutput 'summary.json'), ($summary | ConvertTo-Json -Depth 8), $utf8)
    [pscustomobject]@{ OutputPath = $resolvedOutput; Status = $status; Samples = $sampleCount; ActualDurationSeconds = $summary.ActualDurationSeconds }
}
