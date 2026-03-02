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

        # Attempt to use Native Core if available
        $json = Get-HardwareSystemEventsNative -Days $Days -MaxEvents $MaxEvents
        if ($json) {
            $nativeEvents = $json | ConvertFrom-Json
            foreach ($ev in $nativeEvents) {
                $results += [PSCustomObject]@{
                    TimeCreated  = [DateTime]::Parse($ev.time_created)
                    ProviderName = $ev.provider_name
                    Id           = $ev.id
                    Level        = $ev.level_display
                    Severity     = $ev.severity
                    Message      = $ev.message
                    FullMessage  = $ev.full_message
                }
            }
            $nativeAvailable = $true
        }

        if (-not $nativeAvailable) {
            $startTime = (Get-Date).AddDays(-$Days)
            $levels = if ($IncludeInfo) { @(1, 2, 3, 4) } else { @(1, 2, 3) }

            $events = Get-WinEvent -FilterHashtable @{
                LogName   = 'System'
                Level     = $levels
                StartTime = $startTime
            } -ErrorAction SilentlyContinue | Where-Object {
                $_.ProviderName -match 'disk|storahci|nvme|usbhub|USB|nvstor|iaStor|stornvme|partmgr|ntfs|volmgr'
            } | Select-Object -First $MaxEvents

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
