#Requires -Version 5.1
<#
.SYNOPSIS
    Gets system events related to disk and USB devices

.DESCRIPTION
    Queries Windows Event Log for disk, storage, and USB related errors
    and warnings from the last few days.

.PARAMETER Days
    Number of days to look back (default: 3)

.PARAMETER MaxEvents
    Maximum number of events to return (default: 50)

.PARAMETER IncludeInfo
    Include informational events (Level 4)

.EXAMPLE
    Get-SystemEvents
    Returns disk/USB errors from the last 3 days

.EXAMPLE
    Get-SystemEvents -Days 7 -MaxEvents 100
    Returns more events from a longer period

.OUTPUTS
    PSCustomObject[] with properties: TimeCreated, ProviderName, Id, Level, Message, Severity
#>
function Get-SystemEvents {
    [CmdletBinding()]
    [OutputType([PSCustomObject[]])]
    param(
        [Parameter()]
        [ValidateRange(1, 30)]
        [int]$Days = 3,

        [Parameter()]
        [ValidateRange(1, 500)]
        [int]$MaxEvents = 50,

        [Parameter()]
        [switch]$IncludeInfo
    )

    try {
        $results = @()
        $nativeAvailable = $false

        # This two-argument native ABI samples levels 1-3. IncludeInfo uses the
        # established PowerShell path until native supports the same contract.
        if (-not $IncludeInfo) {
            try {
                $json = Get-HardwareSystemEventsNative -Days $Days -MaxEvents $MaxEvents
                if ($null -ne $json -and -not [string]::IsNullOrWhiteSpace($json)) {
                    if (-not $json.TrimStart().StartsWith('[')) { throw 'Native event response is not an array.' }
                    $nativeEvents = @($json | ConvertFrom-Json -ErrorAction Stop)
                    if ($nativeEvents.Count -gt $MaxEvents) { throw 'Native event response exceeds the requested maximum.' }
                    foreach ($ev in $nativeEvents) {
                        foreach ($name in @('time_created', 'provider_name', 'id', 'level', 'level_display', 'severity', 'message', 'full_message')) {
                            if ($null -eq $ev -or $null -eq $ev.PSObject.Properties[$name]) { throw 'Native event response lacks required fields.' }
                        }
                        foreach ($name in @('provider_name', 'level_display', 'severity', 'message', 'full_message')) {
                            if ($ev.$name -isnot [string]) { throw 'Native event text field has an invalid type.' }
                        }
                        $eventId = 0L
                        $level = 0L
                        $created = [DateTimeOffset]::MinValue
                        # Newer ConvertFrom-Json versions deserialize ISO dates;
                        # preserve their ticks instead of stringifying and truncating.
                        $validDate = if ($ev.time_created -is [DateTime] -or $ev.time_created -is [DateTimeOffset]) {
                            $created = [DateTimeOffset]$ev.time_created
                            $true
                        } elseif ($ev.time_created -is [string]) {
                            [DateTimeOffset]::TryParse($ev.time_created, [ref]$created)
                        } else { $false }
                        if (-not [long]::TryParse([string]$ev.id, [ref]$eventId) -or $eventId -lt 0 -or $eventId -gt 65535 -or
                            -not [long]::TryParse([string]$ev.level, [ref]$level) -or $level -notin @(1, 2, 3) -or
                            -not $validDate -or
                            [string]::IsNullOrWhiteSpace($ev.level_display) -or
                            $ev.provider_name -notmatch 'disk|storahci|nvme|usbhub|USB|nvstor|iaStor|stornvme|partmgr|ntfs|volmgr') {
                            throw 'Native event values do not satisfy the public contract.'
                        }
                        $severity = switch ($level) { 1 { 'Critical' } 2 { 'Error' } 3 { 'Warning' } }
                        if ($ev.severity -cne $severity) { throw 'Native event severity does not match its level.' }
                        $results += [PSCustomObject]@{
                            TimeCreated  = $created.LocalDateTime
                            ProviderName = $ev.provider_name
                            Id           = [int]$eventId
                            Level        = $ev.level_display
                            Severity     = $ev.severity
                            Message      = $ev.message
                            FullMessage  = $ev.full_message
                        }
                    }
                    # [] is successful empty sampling. NULL, errors, or stale
                    # fabricated/partial shapes are unavailable, and fall back.
                    $nativeAvailable = $true
                }
            } catch {
                $results = @()
                Write-Verbose 'Native event sampling unavailable or incompatible; using Get-WinEvent.'
            }
        }
        if (-not $nativeAvailable) {
            $startTime = (Get-Date).AddDays(-$Days)
            $levels = if ($IncludeInfo) { @(1, 2, 3, 4) } else { @(1, 2, 3) }

            # Get-WinEvent raises a terminating error for the perfectly normal
            # case of a filter matching nothing, which is why this used to be
            # -ErrorAction SilentlyContinue. But that also swallowed real
            # failures -- a locked or inaccessible event log looked exactly
            # like a quiet machine, and the caller's -ErrorAction Stop was
            # ignored because nothing reached the catch below. Ask for
            # terminating errors, then treat only the benign "no events" case
            # as empty and let everything else surface.
            try {
                $events = Get-WinEvent -FilterHashtable @{
                    LogName   = 'System'
                    Level     = $levels
                    StartTime = $startTime
                } -ErrorAction Stop | Where-Object {
                    $_.ProviderName -match 'disk|storahci|nvme|usbhub|USB|nvstor|iaStor|stornvme|partmgr|ntfs|volmgr'
                } | Select-Object -First $MaxEvents
            } catch {
                if ($_.Exception.Message -match 'No events were found') {
                    Write-Verbose 'No matching system events in the requested window.'
                    $events = @()
                } else {
                    throw
                }
            }

            if ($events) {
                foreach ($ev in $events) {
                    $severity = switch ($ev.Level) {
                        1 { 'Critical' }
                        2 { 'Error' }
                        3 { 'Warning' }
                        4 { 'Info' }
                        default { 'Unknown' }
                    }

                    $results += [PSCustomObject]@{
                        TimeCreated  = $ev.TimeCreated
                        ProviderName = $ev.ProviderName
                        Id           = $ev.Id
                        Level        = $ev.LevelDisplayName
                        Severity     = $severity
                        Message      = ($ev.Message -split "`n")[0]
                        FullMessage  = $ev.Message
                    }
                }
            }
        }

        return $results | Sort-Object -Property TimeCreated -Descending

    } catch {
        Write-Error "Failed to query system events: $($_.Exception.Message)"
        return @()
    }
}

function Get-HardwareSystemEventsNative {
    param($Days, $MaxEvents)
    if ($null -ne (Get-Module -Name 'PC-AI.Common' -ErrorAction SilentlyContinue) -and [PcaiNative.HardwareModule]::IsAvailable) {
        return [PcaiNative.HardwareModule]::SampleHardwareEventsJson($Days, $MaxEvents)
    }
    return $null
}
