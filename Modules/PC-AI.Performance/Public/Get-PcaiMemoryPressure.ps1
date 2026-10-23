function Get-PcaiMemoryPressure {
    <#
    .SYNOPSIS
        Reports memory observations with explicit measurement provenance.
    .DESCRIPTION
        CommittedPct is a percentage of the actual commit limit. PhysicalUsedPct
        is a distinct physical-memory percentage. Unavailable metrics are null.
        Pressure is a snapshot classification, not a diagnosis of thrashing.
    .OUTPUTS
        A consistent memory object, or its JSON representation with AsJson.
    #>
    [CmdletBinding()]
    param([switch]$Detailed, [switch]$AsJson)

    function Test-PressureNumber($Value) {
        if ($null -eq $Value -or $Value -is [bool] -or $Value -is [string]) { return $false }
        if ($Value -isnot [ValueType]) { return $false }
        try { $number=[double]$Value } catch { return $false }
        return -not [double]::IsNaN($number) -and -not [double]::IsInfinity($number) -and $number -ge 0
    }
    Import-Module PC-AI.Common -ErrorAction SilentlyContinue
    $nativeAvailable=$false
    try { $nativeAvailable=Initialize-PcaiNative } catch { Write-Verbose 'Native initialization unavailable.' }
    if ($nativeAvailable) {
        try {
            if ($Detailed -or $AsJson) {
                $json=[PcaiNative.OptimizerModule]::GetMemoryPressureJson()
                if ([string]::IsNullOrWhiteSpace($json) -or -not $json.TrimStart().StartsWith('{')) { throw 'Native memory object required.' }
                $raw=$json | ConvertFrom-Json -ErrorAction Stop
                if ($raw.status -cne 'Success') { throw 'Native memory status is not successful.' }
                foreach ($name in @('pressure_level','available_mb','committed_pct','top_consumer_count','handle_leak_count','orphan_terminal_count','elapsed_ms')) {
                    if (-not (Test-PressureNumber $raw.$name)) { throw 'Native memory metric is invalid.' }
                }
                $level=$raw.pressure_level; $available=$raw.available_mb; $commit=$raw.committed_pct
                $top=$raw.top_consumer_count; $handles=$raw.handle_leak_count; $orphans=$raw.orphan_terminal_count; $elapsed=$raw.elapsed_ms
            } else {
                $raw=[PcaiNative.OptimizerModule]::AnalyzeMemoryPressure()
                if (-not $raw.IsSuccess) { throw 'Native memory status is not successful.' }
                foreach ($name in @('PressureLevel','AvailableMB','CommittedPct','TopConsumerCount','HandleLeakCount','OrphanTerminalCount','ElapsedMs')) {
                    if (-not (Test-PressureNumber $raw.$name)) { throw 'Native memory metric is invalid.' }
                }
                $level=$raw.PressureLevel; $available=$raw.AvailableMB; $commit=$raw.CommittedPct
                $top=$raw.TopConsumerCount; $handles=$raw.HandleLeakCount; $orphans=$raw.OrphanTerminalCount; $elapsed=$raw.ElapsedMs
            }
            if ($level -gt 3 -or $level -ne [math]::Truncate($level)) { throw 'Native pressure level is invalid.' }
            $windowsCommit=([Environment]::OSVersion.Platform -eq [PlatformID]::Win32NT)
            $output=[pscustomobject]@{
                Status='Success'; PressureLevel=@('Low','Moderate','High','Critical')[[int]$level]; PressureLevelCode=[int]$level
                AvailableMB=$available; CommittedPct=if($windowsCommit){[double]$commit*100}else{$null}; PhysicalUsedPct=$null
                PoolNonpagedMB=$null; PagesPerSec=$null; TopConsumerCount=$top; HandleLeakCount=$handles; OrphanTerminalCount=$orphans
                ElapsedMs=$elapsed; Source='PcaiNative.OptimizerModule'
                MeasurementStatus=[ordered]@{AvailableMB='Measured';CommittedPct=if($windowsCommit){'MeasuredCommitLimitPercent'}else{'Unavailable'};PhysicalUsedPct='Unavailable';PoolNonpagedMB='UnavailableNativeHeuristic';PagesPerSec='UnavailableNativePlaceholder';HandleLeakCount='HighHandleSnapshotOnly';OrphanTerminalCount='ParentPidSnapshotOnly'}
            }
            if ($AsJson) { return ConvertTo-Json -InputObject $output -Depth 5 }
            return $output
        } catch { Write-Verbose 'Native memory acquisition or schema unavailable; using fallback.' }
    }

    $watch=[Diagnostics.Stopwatch]::StartNew()
    $os=Get-CimInstance Win32_OperatingSystem -ErrorAction Stop
    $cs=Get-CimInstance Win32_ComputerSystem -ErrorAction Stop
    if (-not (Test-PressureNumber $os.FreePhysicalMemory) -or -not (Test-PressureNumber $cs.TotalPhysicalMemory) -or $cs.TotalPhysicalMemory -le 0) {
        throw [IO.InvalidDataException]::new('Physical-memory metadata must be finite and nonnegative with a positive total.')
    }
    $freeBytes=[double]$os.FreePhysicalMemory*1024
    $totalBytes=[double]$cs.TotalPhysicalMemory
    if ([double]::IsInfinity($freeBytes) -or $freeBytes -gt $totalBytes) { throw [IO.InvalidDataException]::new('Free physical memory exceeds the validated total.') }
    $usedPct=(1-$freeBytes/$totalBytes)*100
    $level=if($usedPct -lt 60){0}elseif($usedPct -lt 80){1}elseif($usedPct -lt 90){2}else{3}
    $processes=@(Get-Process)
    $allPids=@($processes.Id)
    $orphans=@($processes | Where-Object { $_.ProcessName -in @('cmd','conhost') } | Where-Object {
        try {
            $parent=(Get-CimInstance Win32_Process -Filter "ProcessId=$($_.Id)" -ErrorAction Stop).ParentProcessId
            $null -ne $parent -and $parent -notin $allPids
        } catch { $false }
    })
    $metrics=@{}; $measurement=[ordered]@{AvailableMB='Measured';PhysicalUsedPct='Measured';HandleLeakCount='HighHandleSnapshotOnly';OrphanTerminalCount='ParentPidSnapshotOnly'}
    foreach ($counter in @(
        @{Name='CommittedPct';Path='\Memory\% Committed Bytes In Use';Divisor=1},
        @{Name='PoolNonpagedMB';Path='\Memory\Pool Nonpaged Bytes';Divisor=1MB},
        @{Name='PagesPerSec';Path='\Memory\Pages/sec';Divisor=1}
    )) {
        $metrics[$counter.Name]=$null; $measurement[$counter.Name]='Unavailable'
        try {
            $sample=Get-Counter $counter.Path -ErrorAction Stop
            if (@($sample.CounterSamples).Count -ne 1 -or -not (Test-PressureNumber $sample.CounterSamples[0].CookedValue)) { throw 'Counter observation is invalid.' }
            # PDH valid/new data statuses are 0/1; successful collection alone
            # does not establish that the individual sample is usable.
            if (-not (Test-PressureNumber $sample.CounterSamples[0].Status) -or $sample.CounterSamples[0].Status -notin @(0,1)) { throw 'Counter sample status is unavailable or invalid.' }
            $metrics[$counter.Name]=[double]$sample.CounterSamples[0].CookedValue/$counter.Divisor
            $measurement[$counter.Name]='MeasuredSnapshot'
        } catch { Write-Verbose 'Optional memory counter unavailable or invalid.' }
    }
    $watch.Stop()
    $output=[pscustomobject]@{
        Status='Success'; PressureLevel=@('Low','Moderate','High','Critical')[$level]; PressureLevelCode=$level
        AvailableMB=[math]::Round($freeBytes/1MB,0);CommittedPct=$metrics.CommittedPct;PhysicalUsedPct=[math]::Round($usedPct,1)
        PoolNonpagedMB=$metrics.PoolNonpagedMB;PagesPerSec=$metrics.PagesPerSec
        TopConsumerCount=@($processes|Where-Object { $_.WorkingSet64 -gt 500MB }).Count
        HandleLeakCount=@($processes|Where-Object { $_.HandleCount -ge 100000 }).Count;OrphanTerminalCount=$orphans.Count
        ElapsedMs=$watch.ElapsedMilliseconds;Source='PowerShell-Fallback';MeasurementStatus=$measurement
    }
    if ($AsJson) { return ConvertTo-Json -InputObject $output -Depth 5 }
    return $output
}
