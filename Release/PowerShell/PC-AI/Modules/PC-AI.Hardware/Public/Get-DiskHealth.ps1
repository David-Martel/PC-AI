#Requires -Version 5.1
<#
.SYNOPSIS
    Gets disk health status including SMART information

.DESCRIPTION
    Queries disk drives for model, status, and health information.
    Uses both WMI/CIM and wmic for comprehensive status.

.PARAMETER IncludePartitions
    Include partition information for each disk

.EXAMPLE
    Get-DiskHealth
    Returns disk health status for all drives

.EXAMPLE
    Get-DiskHealth -IncludePartitions
    Returns disk health with partition details

.OUTPUTS
    PSCustomObject[] with properties: Model, Status, MediaType, Size, Partitions, Severity
#>
function Get-DiskHealth {
    [CmdletBinding()]
    [OutputType([PSCustomObject[]])]
    param(
        [Parameter()]
        [switch]$IncludePartitions
    )

    try {
        $results = @()
        $nativeAvailable = $false

        # Attempt to use Native Core if available
        $json = Get-HardwareDiskHealthNative
        if ($json) {
            $nativeDisks = $json | ConvertFrom-Json
            foreach ($disk in $nativeDisks) {
                $results += [PSCustomObject]@{
                    Model          = $disk.model
                    Status         = $disk.status
                    MediaType      = 'Fixed hard disk media' # Native focuses on physical drives
                    InterfaceType  = 'Native'
                    SizeGB         = 0 # To be supplemented by CIM if needed
                    Partitions     = 0
                    Severity       = $disk.severity
                    DeviceID       = $disk.device_id
                    SerialNumber   = $disk.serial_number
                    IsSmartOK      = $disk.smart_status_ok
                    IsSmartCapable = $disk.smart_capable
                }
            }
            $nativeAvailable = $true
        }

        # Supplement or fallback with CIM information
        $cimDisks = Get-CimInstance -ClassName Win32_DiskDrive -ErrorAction SilentlyContinue

        if ($nativeAvailable) {
            # Supplement native results with CIM metadata (Size, Partitions, etc.)
            foreach ($res in $results) {
                $cimDisk = $cimDisks | Where-Object { $_.DeviceID -eq $res.DeviceID -or $_.Model -eq $res.Model }
                if ($cimDisk) {
                    $res.SizeGB = if ($cimDisk.Size) { [math]::Round($cimDisk.Size / 1GB, 2) } else { 0 }
                    $res.Partitions = $cimDisk.Partitions
                    $res.MediaType = $cimDisk.MediaType
                    $res.InterfaceType = $cimDisk.InterfaceType
                }
            }
        } else {
            # Fallback to pure PowerShell implementation
            Write-Verbose 'Native DiskHealth unavailable, using CIM fallback.'

            # Get SMART status via wmic (legacy fallback)
            $smartStatus = @{}
            try {
                $wmicOutput = wmic diskdrive get model, status 2>&1
                if ($wmicOutput) {
                    $lines = $wmicOutput -split "`n" | Where-Object { $_.Trim() -and $_ -notmatch '^Model\s+Status' }
                    foreach ($line in $lines) {
                        if ($line -match '(.+?)\s+(OK|Pred Fail|FAILING|Unknown|CAUTION)\s*$') {
                            $smartStatus[$matches[1].Trim()] = $matches[2].Trim()
                        }
                    }
                }
            } catch {}

            foreach ($disk in $cimDisks) {
                $status = if ($smartStatus.ContainsKey($disk.Model)) { $smartStatus[$disk.Model] } elseif ($disk.Status) { $disk.Status } else { 'Unknown' }
                $severity = switch ($status) {
                    { $_ -in @('Pred Fail', 'FAILING', 'PREDICTED FAIL') } { 'Critical' }
                    { $_ -in @('CAUTION', 'Unknown') } { 'Warning' }
                    default { 'OK' }
                }

                $results += [PSCustomObject]@{
                    Model         = $disk.Model
                    Status        = $status
                    MediaType     = $disk.MediaType
                    InterfaceType = $disk.InterfaceType
                    SizeGB        = if ($disk.Size) { [math]::Round($disk.Size / 1GB, 2) } else { 0 }
                    Partitions    = $disk.Partitions
                    Severity      = $severity
                    DeviceID      = $disk.DeviceID
                    SerialNumber  = $disk.SerialNumber
                }
            }
        }

        if ($IncludePartitions) {
            foreach ($res in $results) {
                if ($res.DeviceID -match 'PhysicalDrive(\d+)') {
                    $index = $matches[1]
                    $partitions = Get-CimInstance -ClassName Win32_DiskPartition -Filter "DiskIndex=$index" -ErrorAction SilentlyContinue
                    $res | Add-Member -NotePropertyName 'PartitionDetails' -NotePropertyValue $partitions
                }
            }
        }

        return $results | Sort-Object -Property @{Expression = 'Severity'; Descending = $true }, Model

    } catch {
        Write-Error "Failed to query disk health: $($_.Exception.Message)"
        return @()
    }
}

function Get-HardwareDiskHealthNative {
    if ($null -ne (Get-Module -Name 'PC-AI.Common' -ErrorAction SilentlyContinue) -and [PcaiNative.HardwareModule]::IsAvailable) {
        return [PcaiNative.HardwareModule]::GetDiskHealthJson()
    }
    return $null
}
