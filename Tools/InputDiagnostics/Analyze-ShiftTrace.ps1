#Requires -Version 7.0
<#
.SYNOPSIS
    Summarizes observed Shift transitions without exposing typed content.
.DESCRIPTION
    Reads existing Raw Input JSONL only. State is tracked independently for each
    device and Shift side. Repeated DOWN events do not create additional holds.
    An unmatched UP or a hold at capture end is ambiguous, not proof of lost input.
    Device identifiers are replaced by capture-local labels. No text, timestamps,
    device paths, typing intent, failure rate, or hardware/software verdict is emitted.
.PARAMETER Path
    Existing shift-source-live JSONL file. Defaults to the latest local capture.
.PARAMETER PassThru
    Return the structured aggregate instead of its JSON representation.
#>
[CmdletBinding()]
param(
    [string]$Path,
    [switch]$PassThru
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
if (-not $Path) {
    $captureDirectory = Join-Path $PSScriptRoot '../../Logs/input-diagnostics'
    $latest = Get-ChildItem -LiteralPath $captureDirectory -Filter 'shift-source-live-*.jsonl' -File |
        Sort-Object LastWriteTimeUtc -Descending | Select-Object -First 1
    if (-not $latest) { throw 'No existing Shift capture was found; specify -Path.' }
    $Path = $latest.FullName
}

$devices = [System.Collections.Generic.Dictionary[string, object]]::new([StringComparer]::OrdinalIgnoreCase)
$orderedDevices = [System.Collections.Generic.List[object]]::new()
$totalEvents = 0
$lineNumber = 0
$reader = [IO.File]::OpenText((Resolve-Path -LiteralPath $Path).Path)
try {
while ($null -ne ($line = $reader.ReadLine())) {
    $lineNumber++
    if ([string]::IsNullOrWhiteSpace($line)) { continue }
    try { $event = ConvertFrom-Json -InputObject $line -NoEnumerate -ErrorAction Stop }
    catch { throw [IO.InvalidDataException]::new("Invalid trace record at line $lineNumber.") }
    if ($event -isnot [pscustomobject]) {
        throw [IO.InvalidDataException]::new("Invalid trace record at line $lineNumber.")
    }
    foreach ($field in @('dev', 'name', 'dir')) {
        $property = $event.PSObject.Properties[$field]
        if ($null -eq $property -or $property.Value -isnot [string] -or [string]::IsNullOrWhiteSpace($property.Value)) {
            throw [IO.InvalidDataException]::new("Invalid trace record at line ${lineNumber}: required string field $field.")
        }
    }
    if ($event.dir -cnotin @('DOWN', 'UP')) {
        throw [IO.InvalidDataException]::new("Invalid trace record at line ${lineNumber}: direction must be DOWN or UP.")
    }
    $totalEvents++
    if (-not $devices.ContainsKey($event.dev)) {
        $sides = [ordered]@{}
        foreach ($side in @('LSHIFT', 'RSHIFT')) {
            $sides[$side] = [pscustomobject][ordered]@{
                Side = $side
                DownEvents = 0
                UpEvents = 0
                ObservedPresses = 0
                ObservedReleases = 0
                RepeatedDownEvents = 0
                UnmatchedUpEvents = 0
                HeldAtCaptureEnd = $false
            }
        }
        $classProperty = $event.PSObject.Properties['cls']
        $reportedClass = if ($classProperty -and $classProperty.Value -cin @('INTERNAL', 'USB/HID', 'OTHER', 'UNKNOWN')) {
            $classProperty.Value
        } else { 'UNKNOWN' }
        $device = [pscustomobject]@{
            DeviceLabel = "Device$($devices.Count + 1)"
            ReportedClass = $reportedClass
            EventCount = 0
            OtherKeyDownEvents = 0
            OtherKeyUpEvents = 0
            Sides = $sides
        }
        $devices.Add($event.dev, $device)
        $orderedDevices.Add($device)
    }
    $device = $devices[$event.dev]
    $device.EventCount++
    if ($event.name -cnotin @('LSHIFT', 'RSHIFT')) {
        if ($event.dir -eq 'DOWN') { $device.OtherKeyDownEvents++ }
        else { $device.OtherKeyUpEvents++ }
        continue
    }
    $state = $device.Sides[$event.name]
    if ($event.dir -eq 'DOWN') {
        $state.DownEvents++
        if ($state.HeldAtCaptureEnd) { $state.RepeatedDownEvents++ }
        else {
            $state.ObservedPresses++
            $state.HeldAtCaptureEnd = $true
        }
    } else {
        $state.UpEvents++
        if ($state.HeldAtCaptureEnd) { $state.ObservedReleases++ }
        else { $state.UnmatchedUpEvents++ }
        $state.HeldAtCaptureEnd = $false
    }
}
} finally { $reader.Dispose() }

$summary = [pscustomobject][ordered]@{
    SchemaVersion = 1
    Analysis = 'observed_shift_state'
    EventCount = $totalEvents
    DeviceCount = $orderedDevices.Count
    Devices = @($orderedDevices | ForEach-Object {
        [pscustomobject][ordered]@{
            DeviceLabel = $_.DeviceLabel
            ReportedClass = $_.ReportedClass
            EventCount = $_.EventCount
            OtherKeyDownEvents = $_.OtherKeyDownEvents
            OtherKeyUpEvents = $_.OtherKeyUpEvents
            ShiftSides = @($_.Sides.Values)
        }
    })
    ActualFailureRate = $null
    Diagnosis = 'undetermined'
    Limitations = @(
        'Counts describe captured events only; no typing intent or application outcome was recorded.'
        'Repeated DOWN events while held are counted once as a hold; their cause is not determined.'
        'Unmatched UP events and holds at capture end may reflect capture boundaries; they do not prove dropped input.'
        'An absent event is inconclusive without a labeled trial and verified capture coverage.'
        'Device labels are local to this capture and do not establish internal or USB hardware identity.'
        'No failure rate or causal hardware/software verdict is available from this capture alone.'
    )
}
if ($PassThru) { $summary }
else { $summary | ConvertTo-Json -Depth 8 }
