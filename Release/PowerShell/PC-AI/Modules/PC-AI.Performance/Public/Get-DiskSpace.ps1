#Requires -Version 5.1
function Get-DiskSpace {
    <#
    .SYNOPSIS
        Analyzes disk space usage on all or specified drives.

    .DESCRIPTION
        Retrieves comprehensive disk space information including total size, used space,
        free space, and usage percentage. Flags drives that fall below a configurable
        free space threshold. Supports filtering by drive letter.

    .PARAMETER DriveLetter
        Optional. One or more drive letters to analyze (e.g., 'C', 'D').
        If not specified, all fixed drives are analyzed.

    .PARAMETER ThresholdPercent
        The minimum free space percentage threshold. Drives below this value
        are flagged as critical. Default is 10%.

    .PARAMETER IncludeRemovable
        Include removable drives (USB drives, etc.) in the analysis.

    .PARAMETER IncludeNetwork
        Include network/mapped drives in the analysis.

    .EXAMPLE
        Get-DiskSpace
        Analyzes all fixed drives with the default 10% threshold.

    .EXAMPLE
        Get-DiskSpace -DriveLetter C, D
        Analyzes only drives C: and D:.

    .EXAMPLE
        Get-DiskSpace -ThresholdPercent 20 -IncludeRemovable
        Analyzes all drives including removable, flagging those with less than 20% free.

    .EXAMPLE
        Get-DiskSpace | Where-Object { $_.BelowThreshold } | Format-Table -AutoSize
        Shows only drives that are below the free space threshold.

    .OUTPUTS
        PSCustomObject with properties:
        - DriveLetter: The drive letter (e.g., 'C')
        - Label: Volume label
        - FileSystem: File system type (NTFS, exFAT, etc.)
        - DriveType: Fixed, Removable, Network, etc.
        - TotalSize: Total capacity in bytes
        - TotalSizeFormatted: Human-readable total size
        - UsedSpace: Used space in bytes
        - UsedSpaceFormatted: Human-readable used space
        - FreeSpace: Free space in bytes
        - FreeSpaceFormatted: Human-readable free space
        - UsedPercent: Percentage of space used
        - FreePercent: Percentage of space free
        - BelowThreshold: Boolean indicating if free space is below threshold
        - Status: 'OK', 'Warning', or 'Critical'

    .NOTES
        Author: PC_AI Project
        Version: 1.0.0
    #>
    [CmdletBinding()]
    [OutputType([PSCustomObject[]])]
    param(
        [Parameter(Position = 0, ValueFromPipeline = $true, ValueFromPipelineByPropertyName = $true)]
        [ValidatePattern('^[A-Za-z]$')]
        [string[]]$DriveLetter,

        [Parameter()]
        [ValidateRange(1, 99)]
        [int]$ThresholdPercent = 10,

        [Parameter()]
        [switch]$IncludeRemovable,

        [Parameter()]
        [switch]$IncludeNetwork
    )

    begin {
        Write-Verbose "Starting disk space analysis with $ThresholdPercent% threshold"
        $results = [System.Collections.ArrayList]::new()

        # Build list of drive types to include
        $driveTypes = @(3)  # 3 = Fixed
        if ($IncludeRemovable) {
            $driveTypes += 2  # 2 = Removable
        }
        if ($IncludeNetwork) {
            $driveTypes += 4  # 4 = Network
        }
    }

    process {
        try {
            # Get all logical disks
            $drives = Get-CimInstance -ClassName Win32_LogicalDisk -ErrorAction Stop |
                Where-Object { $_.DriveType -in $driveTypes }

            # Filter by specified drive letters if provided
            if ($DriveLetter) {
                $driveLettersUpper = $DriveLetter | ForEach-Object { $_.ToUpper() }
                $drives = $drives | Where-Object {
                    $_.DeviceID -replace ':', '' -in $driveLettersUpper
                }
            }

            foreach ($drive in $drives) {
                $letter = $drive.DeviceID -replace ':', ''

                # Skip drives with no size info (e.g., empty card readers)
                if (-not $drive.Size -or $drive.Size -eq 0) {
                    Write-Verbose "Skipping drive $letter (no size information)"
                    continue
                }

                # Calculate space metrics
                $totalSize = [long]$drive.Size
                $freeSpace = [long]$drive.FreeSpace
                $usedSpace = $totalSize - $freeSpace
                $freePercent = [math]::Round(($freeSpace / $totalSize) * 100, 2)
                $usedPercent = [math]::Round(($usedSpace / $totalSize) * 100, 2)

                # Determine drive type name
                $driveTypeName = switch ($drive.DriveType) {
                    2 { 'Removable' }
                    3 { 'Fixed' }
                    4 { 'Network' }
                    5 { 'CD-ROM' }
                    default { 'Unknown' }
                }

                # Determine status based on threshold
                $belowThreshold = $freePercent -lt $ThresholdPercent
                $status = if ($freePercent -lt 5) {
                    'Critical'
                }
                elseif ($freePercent -lt $ThresholdPercent) {
                    'Warning'
                }
                else {
                    'OK'
                }

                # Get media type (SSD/HDD) for fixed drives
                $mediaType = 'N/A'
                if ($drive.DriveType -eq 3) {
                    $mediaType = Get-DriveMediaType -DriveLetter $letter
                }

                $diskInfo = [PSCustomObject]@{
                    DriveLetter         = $letter
                    Label               = if ($drive.VolumeName) { $drive.VolumeName } else { '(No Label)' }
                    FileSystem          = $drive.FileSystem
                    DriveType           = $driveTypeName
                    MediaType           = $mediaType
                    TotalSize           = $totalSize
                    TotalSizeFormatted  = Format-ByteSize -Bytes $totalSize
                    UsedSpace           = $usedSpace
                    UsedSpaceFormatted  = Format-ByteSize -Bytes $usedSpace
                    FreeSpace           = $freeSpace
                    FreeSpaceFormatted  = Format-ByteSize -Bytes $freeSpace
                    UsedPercent         = $usedPercent
                    FreePercent         = $freePercent
                    BelowThreshold      = $belowThreshold
                    Status              = $status
                    ThresholdPercent    = $ThresholdPercent
                }

                # Add custom type for formatting
                $diskInfo.PSObject.TypeNames.Insert(0, 'PC-AI.Performance.DiskSpace')

                [void]$results.Add($diskInfo)
            }
        }
        catch {
            Write-Error "Failed to retrieve disk information: $_"
        }
    }

    end {
        # Sort results by drive letter
        $sortedResults = $results | Sort-Object DriveLetter

        # Output summary to verbose stream
        $criticalCount = ($sortedResults | Where-Object { $_.Status -eq 'Critical' }).Count
        $warningCount = ($sortedResults | Where-Object { $_.Status -eq 'Warning' }).Count

        if ($criticalCount -gt 0) {
            Write-Warning "$criticalCount drive(s) have critically low free space!"
        }
        if ($warningCount -gt 0) {
            Write-Verbose "$warningCount drive(s) are below the $ThresholdPercent% threshold"
        }

        Write-Verbose "Disk space analysis complete. Analyzed $($sortedResults.Count) drive(s)."

        return $sortedResults
    }
}
