#Requires -Version 7.0

<#
.SYNOPSIS
Persists bounded Windows telemetry for freeze and restart investigations.
.DESCRIPTION
Runs without a PowerShell profile or PC-AI modules. Each UTF-8 JSONL record is
flushed to disk. Queries have bounded CIM operation timeouts, and unavailable
providers are recorded as typed errors. ACPI thermal zones are not CPU cores.
Schema 2 adds Windows Energy Meter channel averages, CPU frequency/performance
limit counters and separately sourced performance-provider ACPI thermal zones.
Energy Meter original Energy/Time/Power fields and their verified units are
retained. Positive interval power in milliwatts is divided by 1000 for watts.
Zero/unsupported readings have unknown watts. CPU package power includes
overlapping PP0/PP1 subdomains: channels are never summed and _Total is omitted.
These counters do not measure whole-PC input watts, voltage or USB-PD power.

Files rotate at SegmentBytes, with at most two files and MaxDailyBytes per UTC
day. Existing logs count toward these limits. Nothing is deleted. A global named
mutex prevents simultaneous collectors for the same normalized output path.
Abrupt power loss or a kernel freeze cannot guarantee a stop record; the last
complete sample is intended to help correlate the failure with Windows events.

This script does not register tasks or alter event logs, drivers or power policy.
.PARAMETER OutputDirectory
Absolute filesystem directory for capture. Required unless using Help or DryRun.
.PARAMETER DurationMinutes
Maximum capture duration, from one minute to seven days. Default: one day.
.PARAMETER IntervalSeconds
Minimum delay between samples. Default: 15 seconds.
.PARAMETER MaxDailyBytes
Aggregate UTC daily log cap. Existing logs count toward this limit.
.PARAMETER SegmentBytes
Maximum size of each of the two permitted daily segments.
.PARAMETER Once
Write start, one sample and stop records, then return.
.PARAMETER DryRun
Describe capture without querying CIM, acquiring a mutex or writing anything.
.PARAMETER Help
Show help. Also accepts -h and --help.
.EXAMPLE
pwsh -NoLogo -NoProfile -File .\Tools\Watch-CrashTelemetry.ps1 -DryRun
.EXAMPLE
pwsh -NoLogo -NoProfile -File .\Tools\Watch-CrashTelemetry.ps1 -OutputDirectory C:\ProgramData\PC-AI\CrashTelemetry -Once
#>
[CmdletBinding()]
param(
    [string]$OutputDirectory,
    [ValidateRange(1, 10080)] [int]$DurationMinutes = 1440,
    [ValidateRange(1, 300)] [int]$IntervalSeconds = 15,
    [ValidateRange(65536, 1073741824)] [long]$MaxDailyBytes = 67108864,
    [ValidateRange(4096, 1073741824)] [long]$SegmentBytes = 33554432,
    [switch]$Once,
    [switch]$DryRun,
    [Alias('h', '-help')] [switch]$Help
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$schemaVersion = 2

if ($Help) {
    Microsoft.PowerShell.Core\Get-Help -Name $PSCommandPath -Detailed
    return
}
if ($DryRun) {
    [pscustomobject]@{
        Status = 'DryRun'; OutputDirectory = $OutputDirectory
        DurationMinutes = $DurationMinutes; IntervalSeconds = $IntervalSeconds
        MaxDailyBytes = $MaxDailyBytes; SegmentBytes = $SegmentBytes
        MaxSegmentsPerDay = 2; Once = $Once.IsPresent
        WritesPerformed = $false; SamplingPerformed = $false
    }
    return
}
if (-not $IsWindows) { throw 'Crash telemetry capture requires Windows.' }
if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    throw 'OutputDirectory is required for capture. Use -DryRun to preview.'
}
if (-not [System.IO.Path]::IsPathFullyQualified($OutputDirectory)) {
    throw 'OutputDirectory must be an absolute filesystem path.'
}
$capturePath = [System.IO.Path]::GetFullPath($OutputDirectory).TrimEnd('\', '/')
foreach ($component in $capturePath.Split([char[]]'\/')) {
    $trimmed = $component.TrimEnd('.', ' ')
    if ($trimmed -eq '$null' -or $trimmed -match '^(?i:AUX|CON|NUL|PRN|COM[1-9]|LPT[1-9])(?:\.|$)') {
        throw 'OutputDirectory contains a reserved Windows path component.'
    }
}
# Refuse roots and alternate data stream syntax; capture should have its own folder.
if ($capturePath.Length -le 3 -or $capturePath.Substring(2).Contains(':')) {
    throw 'OutputDirectory must be a dedicated directory without alternate stream syntax.'
}

function Add-CrashQueryError {
    param(
        [System.Collections.Generic.List[object]]$Errors,
        [string]$Source,
        [System.Management.Automation.ErrorRecord]$Record
    )
    $Errors.Add([ordered]@{
        source = $Source; exceptionType = $Record.Exception.GetType().FullName
        category = $Record.CategoryInfo.Category.ToString()
        errorId = $Record.FullyQualifiedErrorId; message = $Record.Exception.Message
    })
}

function ConvertTo-CrashCounterValue {
    param([AllowNull()][object]$Value, [long]$Scale = 1)
    # Nullable provider properties must remain unknown, while observed zero is valid.
    if ($null -eq $Value) { return $null }
    return ([long]$Value * $Scale)
}

function Get-CrashUnsignedCounterValue {
    param([psobject]$InputObject, [string]$Name)
    $property = $InputObject.PSObject.Properties[$Name]
    # Provider properties can be omitted or null; preserve either as unknown, including under StrictMode.
    if ($null -eq $property -or $null -eq $property.Value) { return $null }
    return [ulong]$property.Value
}

function Add-CrashCaptureFailure {
    param(
        [hashtable]$State,
        [string]$Phase,
        [System.Management.Automation.ErrorRecord]$Record
    )
    # Retain the actual first record; cleanup must not replace it.
    if ($null -eq $State.FirstErrorRecord) {
        $State.FirstErrorRecord = $Record
        $State.FirstErrorPhase = $Phase
        return
    }
    if ([object]::ReferenceEquals($State.FirstErrorRecord, $Record)) { return }
    foreach ($secondary in $State.SecondaryErrors) {
        if ([object]::ReferenceEquals($secondary.Record, $Record)) { return }
    }
    if ($State.SecondaryErrors.Count -ge 8) {
        $State.SecondaryErrorsTruncated = $true
        return
    }
    $State.SecondaryErrors.Add([pscustomobject]@{ Phase = $Phase; Record = $Record })
}

function Complete-CrashCapture {
    param(
        [hashtable]$State,
        [AllowNull()][object]$Mutex,
        [bool]$OwnsMutex,
        [bool]$CaptureStarted,
        [string]$SessionId,
        [long]$Sequence,
        [string]$StopReason
    )
    if ($CaptureStarted -and $OwnsMutex) {
        $State.Cleanup.StopMarkerAttempted = $true
        try {
            $now = [datetimeoffset]::UtcNow
            $State.Cleanup.StopMarkerWritten = Write-CrashRecord $State ([ordered]@{
                schemaVersion = $schemaVersion; recordType = 'stop'; sessionId = $SessionId
                utc = $now.ToString('o'); local = $now.ToLocalTime().ToString('o')
                reason = $StopReason; samplesAttempted = $Sequence
            }) -Lifecycle
        }
        catch { Add-CrashCaptureFailure $State 'StopMarker' $_ }
    }
    if ($null -ne $State.Writer) {
        $State.Cleanup.WriterDisposeAttempted = $true
        try { $State.Writer.Dispose(); $State.Cleanup.WriterDisposeReturned = $true }
        catch { Add-CrashCaptureFailure $State 'WriterDispose' $_ }
    }
    if ($null -ne $State.Stream) {
        $State.Cleanup.StreamDisposeAttempted = $true
        try { $State.Stream.Dispose(); $State.Cleanup.StreamDisposeReturned = $true }
        catch { Add-CrashCaptureFailure $State 'StreamDispose' $_ }
    }
    if ($OwnsMutex) {
        $State.Cleanup.MutexReleaseAttempted = $true
        try { $Mutex.ReleaseMutex(); $State.Cleanup.MutexReleaseReturned = $true }
        catch { Add-CrashCaptureFailure $State 'MutexRelease' $_ }
    }
    if ($null -ne $Mutex) {
        $State.Cleanup.MutexDisposeAttempted = $true
        try { $Mutex.Dispose(); $State.Cleanup.MutexDisposeReturned = $true }
        catch { Add-CrashCaptureFailure $State 'MutexDispose' $_ }
    }
}

function Get-CrashSample {
    param([string]$SessionId, [long]$Sequence)
    $sampleUtc = [datetimeoffset]::UtcNow
    $queryErrors = [System.Collections.Generic.List[object]]::new()
    $sample = [ordered]@{
        schemaVersion = $schemaVersion; recordType = 'sample'; sessionId = $SessionId
        sequence = $Sequence; utc = $sampleUtc.ToString('o')
        local = $sampleUtc.ToLocalTime().ToString('o')
        uptimeSeconds = $null; osMemory = $null; memoryCounters = $null
        cpu = $null; processorInformation = $null
        energyMeter = [ordered]@{
            source = 'Win32_PerfFormattedData_PowerMeterCounter_EnergyMeter'
            status = 'Unavailable'; channels = @()
            units = [ordered]@{ Energy = 'picowatt-hours'; Time = 'milliseconds'; Power = 'milliwatts' }
            unitProvenance = [ordered]@{
                module = 'Microsoft Windows umpoext.dll'; verifiedDate = '2026-10-11'
                stringResourceIds = [ordered]@{ Energy = 23; Time = 19; Power = 27 }
            }
            aggregation = 'None: PKG includes overlapping PP0/PP1 subdomains; never sum channels or use _Total'
            scope = 'Channel averages only; no whole-PC input watts, voltage or USB-PD sensing'
        }
        thermalZones = @(); rejectedThermalReadings = 0
        performanceThermalZones = @(); rejectedPerformanceThermalReadings = 0
        thermalInterpretation = 'ACPI zone, not CPU core temperature'
        topProcesses = @(); errors = $null
    }
    try {
        $os = CimCmdlets\Get-CimInstance -ClassName Win32_OperatingSystem -OperationTimeoutSec 3 -ErrorAction Stop
        $lastBootUtc = $null
        if ($null -ne $os.LastBootUpTime) {
            $bootUtc = ([datetime]$os.LastBootUpTime).ToUniversalTime()
            $lastBootUtc = $bootUtc.ToString('o')
            $sample.uptimeSeconds = [math]::Round(($sampleUtc.UtcDateTime - $bootUtc).TotalSeconds, 3)
        }
        $sample.osMemory = [ordered]@{
            lastBootUtc = $lastBootUtc
            totalPhysicalBytes = ConvertTo-CrashCounterValue $os.TotalVisibleMemorySize 1024
            availablePhysicalBytes = ConvertTo-CrashCounterValue $os.FreePhysicalMemory 1024
            totalVirtualBytes = ConvertTo-CrashCounterValue $os.TotalVirtualMemorySize 1024
            availableVirtualBytes = ConvertTo-CrashCounterValue $os.FreeVirtualMemory 1024
        }
    }
    catch { Add-CrashQueryError $queryErrors 'Win32_OperatingSystem' $_ }
    try {
        $memory = CimCmdlets\Get-CimInstance -ClassName Win32_PerfFormattedData_PerfOS_Memory -OperationTimeoutSec 3 -ErrorAction Stop
        $sample.memoryCounters = [ordered]@{
            availableBytes = ConvertTo-CrashCounterValue $memory.AvailableBytes
            committedBytes = ConvertTo-CrashCounterValue $memory.CommittedBytes
            commitLimitBytes = ConvertTo-CrashCounterValue $memory.CommitLimit
            percentCommittedBytesInUse = ConvertTo-CrashCounterValue $memory.PercentCommittedBytesInUse
            pagesPerSecond = ConvertTo-CrashCounterValue $memory.PagesPersec
            pageReadsPerSecond = ConvertTo-CrashCounterValue $memory.PageReadsPersec
            pageWritesPerSecond = ConvertTo-CrashCounterValue $memory.PageWritesPersec
        }
    }
    catch { Add-CrashQueryError $queryErrors 'Win32_PerfFormattedData_PerfOS_Memory' $_ }
    try {
        $cpu = CimCmdlets\Get-CimInstance -ClassName Win32_PerfFormattedData_PerfOS_Processor -Filter "Name='_Total'" -OperationTimeoutSec 3 -ErrorAction Stop
        $sample.cpu = [ordered]@{
            totalProcessorTimePercent = ConvertTo-CrashCounterValue $cpu.PercentProcessorTime
            privilegedTimePercent = ConvertTo-CrashCounterValue $cpu.PercentPrivilegedTime
            interruptTimePercent = ConvertTo-CrashCounterValue $cpu.PercentInterruptTime
            dpcTimePercent = ConvertTo-CrashCounterValue $cpu.PercentDPCTime
        }
    }
    catch { Add-CrashQueryError $queryErrors 'Win32_PerfFormattedData_PerfOS_Processor' $_ }
    try {
        $processorInfo = CimCmdlets\Get-CimInstance -ClassName Win32_PerfFormattedData_Counters_ProcessorInformation -Filter "Name='_Total'" -OperationTimeoutSec 3 -ErrorAction Stop
        $sample.processorInformation = [ordered]@{
            source = 'Win32_PerfFormattedData_Counters_ProcessorInformation'; name = '_Total'
            actualFrequencyMHz = ConvertTo-CrashCounterValue $processorInfo.ActualFrequency
            processorFrequencyMHz = ConvertTo-CrashCounterValue $processorInfo.ProcessorFrequency
            percentOfMaximumFrequency = ConvertTo-CrashCounterValue $processorInfo.PercentofMaximumFrequency
            percentProcessorPerformance = ConvertTo-CrashCounterValue $processorInfo.PercentProcessorPerformance
            percentPerformanceLimit = ConvertTo-CrashCounterValue $processorInfo.PercentPerformanceLimit
            performanceLimitFlagsRaw = ConvertTo-CrashCounterValue $processorInfo.PerformanceLimitFlags
            processorStateFlagsRaw = ConvertTo-CrashCounterValue $processorInfo.ProcessorStateFlags
            parkingStatusRaw = ConvertTo-CrashCounterValue $processorInfo.ParkingStatus
            interpretation = 'Provider counters and raw flags; not a diagnosis of a thermal or input-power fault'
        }
    }
    catch { Add-CrashQueryError $queryErrors 'Win32_PerfFormattedData_Counters_ProcessorInformation' $_ }
    try {
        $meters = @(CimCmdlets\Get-CimInstance -ClassName Win32_PerfFormattedData_PowerMeterCounter_EnergyMeter -OperationTimeoutSec 3 -ErrorAction Stop)
        $channels = [System.Collections.Generic.List[object]]::new()
        foreach ($meter in $meters) {
            $name = [string]$meter.Name
            if ($name -eq '_Total') { continue }
            $rawPower = Get-CrashUnsignedCounterValue $meter 'Power'
            $rawTime = Get-CrashUnsignedCounterValue $meter 'Time'
            $rawEnergy = Get-CrashUnsignedCounterValue $meter 'Energy'
            $watts = $null
            $readingStatus = 'ZeroOrUnsupported'
            if ($null -eq $rawPower) { $readingStatus = 'UnknownPowerCounter' }
            elseif ($null -eq $rawTime) { $readingStatus = 'UnknownMeterInterval' }
            elseif ($rawPower -gt 0 -and $rawTime -gt 0) {
                $watts = [double]$rawPower / 1000.0
                $readingStatus = 'MeasuredChannelAverage'
            }
            elseif ($rawPower -gt 0) { $readingStatus = 'UnknownMeterInterval' }
            $domain = switch -Regex ($name) {
                '_PKG$' { 'CPU package (includes overlapping PP0 and PP1)'; break }
                '_PP0$' { 'CPU core subdomain (included in PKG)'; break }
                '_PP1$' { 'Integrated graphics subdomain (included in PKG)'; break }
                '_DRAM$' { 'DRAM domain (zero may be unsupported)'; break }
                default { 'Metered channel; physical scope unverified' }
            }
            $channels.Add([ordered]@{
                name = $name; domain = $domain; readingStatus = $readingStatus; watts = $watts
                counters = [ordered]@{ Energy = $rawEnergy; Time = $rawTime; Power = $rawPower }
            })
        }
        $sample.energyMeter.channels = $channels.ToArray()
        $sample.energyMeter.status = if ($channels.Count -gt 0) { 'Available' } else { 'NoChannelInstances' }
    }
    catch { Add-CrashQueryError $queryErrors 'Win32_PerfFormattedData_PowerMeterCounter_EnergyMeter' $_ }
    try {
        $perfZones = @(CimCmdlets\Get-CimInstance -ClassName Win32_PerfFormattedData_Counters_ThermalZoneInformation -OperationTimeoutSec 3 -ErrorAction Stop)
        $validPerfZones = [System.Collections.Generic.List[object]]::new()
        foreach ($zone in $perfZones) {
            $raw = [long]$zone.HighPrecisionTemperature
            if ($raw -lt 2000 -or $raw -gt 4500) {
                $sample.rejectedPerformanceThermalReadings++
                continue
            }
            $validPerfZones.Add([ordered]@{
                source = 'Win32_PerfFormattedData_Counters_ThermalZoneInformation'
                name = [string]$zone.Name; kind = 'ACPI thermal zone, not CPU core'
                rawHighPrecisionTemperatureTenthsKelvin = $raw
                rawTemperatureKelvin = ConvertTo-CrashCounterValue $zone.Temperature
                celsius = [math]::Round($raw / 10.0 - 273.15, 2)
                percentPassiveLimitRaw = ConvertTo-CrashCounterValue $zone.PercentPassiveLimit
                throttleReasonsRaw = ConvertTo-CrashCounterValue $zone.ThrottleReasons
            })
        }
        $sample.performanceThermalZones = $validPerfZones.ToArray()
    }
    catch { Add-CrashQueryError $queryErrors 'Win32_PerfFormattedData_Counters_ThermalZoneInformation' $_ }
    try {
        $zones = @(CimCmdlets\Get-CimInstance -Namespace root/wmi -ClassName MSAcpi_ThermalZoneTemperature -OperationTimeoutSec 3 -ErrorAction Stop)
        $validZones = [System.Collections.Generic.List[object]]::new()
        foreach ($zone in $zones) {
            $raw = [long]$zone.CurrentTemperature
            if ($raw -lt 2000 -or $raw -gt 4500) {
                $sample.rejectedThermalReadings++
                continue
            }
            $validZones.Add([ordered]@{
                source = 'MSAcpi_ThermalZoneTemperature'
                name = [string]$zone.InstanceName; kind = 'ACPI thermal zone, not CPU core'
                rawTenthsKelvin = $raw; celsius = [math]::Round($raw / 10.0 - 273.15, 2)
            })
        }
        $sample.thermalZones = $validZones.ToArray()
    }
    catch { Add-CrashQueryError $queryErrors 'MSAcpi_ThermalZoneTemperature' $_ }
    $processes = @()
    try {
        $processes = [System.Diagnostics.Process]::GetProcesses()
        $rows = [System.Collections.Generic.List[object]]::new()
        foreach ($process in $processes) {
            try {
                $rows.Add([pscustomobject]@{
                    id = $process.Id; name = $process.ProcessName
                    workingSetBytes = $process.WorkingSet64
                    privateBytes = $process.PrivateMemorySize64
                })
            }
            catch { Add-CrashQueryError $queryErrors 'ProcessSnapshot' $_ }
        }
        $sample.topProcesses = @($rows | Microsoft.PowerShell.Utility\Sort-Object -Property workingSetBytes -Descending -Top 5)
    }
    catch { Add-CrashQueryError $queryErrors 'ProcessEnumeration' $_ }
    finally { foreach ($process in $processes) { $process.Dispose() } }
    $sample.errors = $queryErrors.ToArray()
    $sample.collectionMilliseconds = [math]::Round(([datetimeoffset]::UtcNow - $sampleUtc).TotalMilliseconds, 1)
    return $sample
}

function Write-CrashRecord {
    param([hashtable]$State, [System.Collections.IDictionary]$Record, [switch]$Lifecycle)
    $recordDay = ([datetimeoffset]$Record.utc).UtcDateTime.ToString('yyyy-MM-dd')
    $json = Microsoft.PowerShell.Utility\ConvertTo-Json -InputObject $Record -Depth 8 -Compress
    $bytes = $State.Encoding.GetByteCount($json + [environment]::NewLine)
    if ($State.Day -ne $recordDay) {
        if ($null -ne $State.Writer) { $State.Writer.Dispose(); $State.Writer = $null }
        $State.Day = $recordDay; $State.Segment = 1; $State.DailyBytes = 0
        for ($number = 1; $number -le 2; $number++) {
            $filePath = [System.IO.Path]::Combine($capturePath, ('crash-{0}-{1:D2}.jsonl' -f $recordDay, $number))
            if ([System.IO.File]::Exists($filePath)) {
                $State.DailyBytes += [System.IO.FileInfo]::new($filePath).Length
                $State.Segment = $number
            }
        }
    }
    $reserve = if ($Lifecycle) { 0 } else { 4096 }
    if ($State.DailyBytes + $bytes + $reserve -gt $MaxDailyBytes) { return $false }
    $filePath = [System.IO.Path]::Combine($capturePath, ('crash-{0}-{1:D2}.jsonl' -f $recordDay, $State.Segment))
    $existingLength = if ([System.IO.File]::Exists($filePath)) { [System.IO.FileInfo]::new($filePath).Length } else { 0 }
    if ($existingLength + $bytes + $reserve -gt $SegmentBytes) {
        if ($State.Segment -ge 2) { return $false }
        if ($null -ne $State.Writer) { $State.Writer.Dispose(); $State.Writer = $null }
        $State.Segment++
        $filePath = [System.IO.Path]::Combine($capturePath, ('crash-{0}-{1:D2}.jsonl' -f $recordDay, $State.Segment))
        if ($bytes + $reserve -gt $SegmentBytes) { return $false }
    }
    if ($null -eq $State.Writer) {
        # WriteThrough and Flush(true) improve survival of a restart. A hard freeze
        # still leaves a possible storage-controller or hardware cache loss window.
        $State.Stream = [System.IO.FileStream]::new($filePath, [System.IO.FileMode]::Append,
            [System.IO.FileAccess]::Write, [System.IO.FileShare]::Read, 4096, [System.IO.FileOptions]::WriteThrough)
        $State.Writer = [System.IO.StreamWriter]::new($State.Stream, $State.Encoding)
        $State.Writer.AutoFlush = $true
        $State.LastFile = $filePath
    }
    $State.Writer.WriteLine($json)
    $State.Stream.Flush($true)
    $State.DailyBytes += $bytes
    return $true
}

$mutex = $null
$ownsMutex = $false
$state = @{
    Day = ''; Segment = 1; DailyBytes = [long]0; Writer = $null; Stream = $null
    Encoding = [System.Text.UTF8Encoding]::new($false); LastFile = $null
    FirstErrorRecord = $null; FirstErrorPhase = $null
    SecondaryErrors = [System.Collections.Generic.List[object]]::new()
    SecondaryErrorsTruncated = $false
    Cleanup = [ordered]@{
        StopMarkerAttempted = $false; StopMarkerWritten = $false
        WriterDisposeAttempted = $false; WriterDisposeReturned = $false
        StreamDisposeAttempted = $false; StreamDisposeReturned = $false
        MutexReleaseAttempted = $false; MutexReleaseReturned = $false
        MutexDisposeAttempted = $false; MutexDisposeReturned = $false
    }
}
$sessionId = [guid]::NewGuid().ToString()
$sequence = [long]0
$stopReason = 'Interrupted'
$captureStarted = $false
try {
    $hasher = [System.Security.Cryptography.SHA256]::Create()
    try {
        $pathHash = [System.BitConverter]::ToString($hasher.ComputeHash(
            [System.Text.Encoding]::UTF8.GetBytes($capturePath.ToUpperInvariant()))).Replace('-', '')
    }
    finally { $hasher.Dispose() }
    $mutex = [System.Threading.Mutex]::new($false, ('Global\PC-AI-CrashTelemetry-{0}' -f $pathHash))
    try { $ownsMutex = $mutex.WaitOne(0) }
    catch [System.Threading.AbandonedMutexException] { $ownsMutex = $true }
    if (-not $ownsMutex) {
        $stopReason = 'AlreadyRunning'
    }
    else {
        [System.IO.Directory]::CreateDirectory($capturePath) | Microsoft.PowerShell.Core\Out-Null
        $now = [datetimeoffset]::UtcNow
        $captureStarted = Write-CrashRecord $state ([ordered]@{
            schemaVersion = $schemaVersion; recordType = 'start'; sessionId = $sessionId
            utc = $now.ToString('o'); local = $now.ToLocalTime().ToString('o')
            machine = [environment]::MachineName; processId = $PID
            powerShellVersion = $PSVersionTable.PSVersion.ToString()
            durationMinutes = $DurationMinutes; intervalSeconds = $IntervalSeconds
            maxDailyBytes = $MaxDailyBytes; segmentBytes = $SegmentBytes; maxSegmentsPerDay = 2
        }) -Lifecycle
        if (-not $captureStarted) { $stopReason = 'DailyStorageLimit' }
        else {
            $timer = [System.Diagnostics.Stopwatch]::StartNew()
            while ($timer.Elapsed.TotalMinutes -lt $DurationMinutes) {
                $sequence++
                $sample = Get-CrashSample $sessionId $sequence
                if (-not (Write-CrashRecord $state $sample)) { $stopReason = 'DailyStorageLimit'; break }
                if ($Once) { $stopReason = 'OnceCompleted'; break }
                $remaining = [math]::Max(0, [math]::Min($IntervalSeconds * 1000,
                    ($DurationMinutes * 60000 - $timer.Elapsed.TotalMilliseconds)))
                if ($remaining -gt 0) { Microsoft.PowerShell.Utility\Start-Sleep -Milliseconds ([int]$remaining) }
            }
            if ($stopReason -eq 'Interrupted') { $stopReason = 'DurationCompleted' }
        }
    }
}
catch {
    $stopReason = 'CollectorError'
    Add-CrashCaptureFailure $state 'Capture' $_
}
finally {
    Complete-CrashCapture $state $mutex $ownsMutex $captureStarted $sessionId $sequence $stopReason
}
if ($null -ne $state.FirstErrorRecord) {
    # Publish primitive custody evidence without serializing retained error graphs.
    [pscustomobject]@{
        Status = 'CollectorError'; SessionId = $sessionId; SamplesAttempted = $sequence
        FirstErrorPhase = $state.FirstErrorPhase
        SecondaryErrorCount = $state.SecondaryErrors.Count
        SecondaryErrorsTruncated = $state.SecondaryErrorsTruncated
        Cleanup = [pscustomobject]$state.Cleanup
    }
    $PSCmdlet.ThrowTerminatingError($state.FirstErrorRecord)
}
if ($state.Cleanup.StopMarkerAttempted -and -not $state.Cleanup.StopMarkerWritten) {
    Microsoft.PowerShell.Utility\Write-Warning 'Storage limit reached; no space remains for a stop marker.'
}
[pscustomobject]@{
    Status = $stopReason; SessionId = $sessionId; SamplesAttempted = $sequence
    OutputDirectory = $capturePath; LastFile = $state.LastFile
    Cleanup = [pscustomobject]$state.Cleanup
}
