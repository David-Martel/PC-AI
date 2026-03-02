#Requires -Version 5.1
function Find-DuplicateFiles {
    <#
    .SYNOPSIS
        Finds duplicate files by content hash in specified directory.

    .DESCRIPTION
        Scans a directory for duplicate files by computing and comparing
        file hashes. Files with identical content will have matching hashes.

        Features:
        - Supports recursive search
        - Configurable minimum file size filter
        - Multiple hash algorithms (SHA256, MD5, SHA1)
        - Groups duplicates with total wasted space calculation
        - Handles locked files gracefully

    .PARAMETER Path
        Directory to search for duplicates. Default is current directory.

    .PARAMETER Recurse
        Search subdirectories recursively.

    .PARAMETER MinimumSize
        Minimum file size in bytes to consider. Default is 1024 (1 KB).
        Use 0 to include all files.

    .PARAMETER MaximumSize
        Maximum file size in bytes to consider. Default is unlimited.

    .PARAMETER Algorithm
        Hash algorithm to use: SHA256, MD5, or SHA1. Default is SHA256.

    .PARAMETER Include
        File patterns to include (e.g., "*.jpg", "*.pdf").

    .PARAMETER Exclude
        File patterns to exclude.

    .PARAMETER ShowProgress
        Display progress bar during scan.

    .EXAMPLE
        Find-DuplicateFiles -Path "C:\Photos" -Recurse

        Find all duplicate files in Photos folder and subfolders.

    .EXAMPLE
        Find-DuplicateFiles -Path "D:\Downloads" -MinimumSize 1MB -Include "*.zip","*.exe"

        Find duplicate archives and executables larger than 1 MB.

    .EXAMPLE
        Find-DuplicateFiles -Path "C:\Documents" -Recurse -ShowProgress |
            Select-Object -ExpandProperty DuplicateGroups

        Find duplicates with progress bar and show groupings.

    .OUTPUTS
        PSCustomObject with properties:
        - Path: Searched path
        - TotalFilesScanned: Number of files examined
        - TotalFilesHashed: Number of files successfully hashed
        - DuplicateGroups: Array of duplicate file groups
        - TotalDuplicates: Total number of duplicate files
        - WastedSpace: Total bytes wasted by duplicates
        - WastedSpaceFormatted: Human-readable wasted space
        - ScanDuration: Time taken for scan
    #>
    [CmdletBinding()]
    [OutputType([PSCustomObject])]
    param(
        [Parameter(Position = 0)]
        [ValidateScript({ Test-Path -Path $_ -PathType Container })]
        [string]$Path = (Get-Location).Path,

        [Parameter()]
        [switch]$Recurse,

        [Parameter()]
        [long]$MinimumSize = 1024,

        [Parameter()]
        [long]$MaximumSize = [long]::MaxValue,

        [Parameter()]
        [ValidateSet('SHA256', 'MD5', 'SHA1')]
        [string]$Algorithm = 'SHA256',

        [Parameter()]
        [string[]]$Include,

        [Parameter()]
        [string[]]$Exclude,

        [Parameter()]
        [switch]$ShowProgress
    )

    $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()

    Write-Verbose "Starting duplicate file scan in: $Path"
    Write-CleanupLog -Message "Starting duplicate scan: $Path (Recurse: $Recurse, MinSize: $MinimumSize)" -Level Info

    $result = [PSCustomObject]@{
        Path = $Path
        Recurse = $Recurse
        Algorithm = $Algorithm
        MinimumSize = $MinimumSize
        MaximumSize = $MaximumSize
        TotalFilesScanned = 0
        TotalFilesHashed = 0
        FilesSkipped = 0
        DuplicateGroups = @()
        TotalDuplicates = 0
        WastedSpace = 0
        WastedSpaceFormatted = ''
        ScanDuration = $null
        Errors = @()
    }

    # Build file search parameters
    $getChildParams = @{
        Path = $Path
        File = $true
        ErrorAction = 'SilentlyContinue'
    }

    if ($Recurse) {
        $getChildParams['Recurse'] = $true
    }

    if ($Include -and $Include.Count -gt 0) {
        $getChildParams['Include'] = $Include
    }

    if ($Exclude -and $Exclude.Count -gt 0) {
        $getChildParams['Exclude'] = $Exclude
    }

    # Get all files
    Write-Verbose "Enumerating files..."
    $allFiles = @(Get-ChildItem @getChildParams)
    $result.TotalFilesScanned = $allFiles.Count

    Write-Verbose "Found $($allFiles.Count) files to examine"

    # Filter by size
    $filteredFiles = $allFiles | Where-Object {
        $_.Length -ge $MinimumSize -and $_.Length -le $MaximumSize
    }

    $result.FilesSkipped = $allFiles.Count - $filteredFiles.Count
    Write-Verbose "After size filter: $($filteredFiles.Count) files (skipped $($result.FilesSkipped))"

    if ($filteredFiles.Count -eq 0) {
        Write-Warning "No files found matching criteria."
        $stopwatch.Stop()
        $result.ScanDuration = $stopwatch.Elapsed
        return $result
    }

    # First pass: Group by size (quick filter)
    Write-Verbose "Grouping files by size..."
    $sizeGroups = $filteredFiles | Group-Object -Property Length | Where-Object { $_.Count -gt 1 }

    if ($sizeGroups.Count -eq 0) {
        Write-Verbose "No files with matching sizes found - no duplicates possible."
        $stopwatch.Stop()
        $result.ScanDuration = $stopwatch.Elapsed
        return $result
    }

    # Get files that need hashing (only those with size matches)
    $filesToHash = $sizeGroups | ForEach-Object { $_.Group }
    Write-Verbose "Files with matching sizes: $($filesToHash.Count) (potential duplicates)"

    # Second pass: Hash files and find exact duplicates
    $hashTable = @{}
    $processedCount = 0
    $totalToHash = $filesToHash.Count

    foreach ($file in $filesToHash) {
        $processedCount++

        if ($ShowProgress) {
            $percentComplete = [math]::Round(($processedCount / $totalToHash) * 100)
            Write-Progress -Activity "Computing file hashes" -Status "$processedCount of $totalToHash" `
                -PercentComplete $percentComplete -CurrentOperation $file.Name
        }

        $hash = Get-FileHashSafe -Path $file.FullName -Algorithm $Algorithm

        if ($hash) {
            $result.TotalFilesHashed++

            if (-not $hashTable.ContainsKey($hash)) {
                $hashTable[$hash] = @()
            }
            $hashTable[$hash] += [PSCustomObject]@{
                FullName = $file.FullName
                Name = $file.Name
                Directory = $file.DirectoryName
                Size = $file.Length
                SizeFormatted = Format-FileSize -Bytes $file.Length
                LastWriteTime = $file.LastWriteTime
                CreationTime = $file.CreationTime
            }
        }
        else {
            $result.Errors += "Could not hash: $($file.FullName)"
        }
    }

    if ($ShowProgress) {
        Write-Progress -Activity "Computing file hashes" -Completed
    }

    Write-Verbose "Hashed $($result.TotalFilesHashed) files"

    # Find duplicates (hashes with more than one file)
    $duplicateHashes = $hashTable.GetEnumerator() | Where-Object { $_.Value.Count -gt 1 }

    foreach ($group in $duplicateHashes) {
        $files = $group.Value | Sort-Object -Property LastWriteTime
        $fileSize = $files[0].Size
        $duplicateCount = $files.Count - 1  # Exclude the "original"
        $wastedBytes = $fileSize * $duplicateCount

        $duplicateGroup = [PSCustomObject]@{
            Hash = $group.Key
            FileCount = $files.Count
            DuplicateCount = $duplicateCount
            FileSize = $fileSize
            FileSizeFormatted = Format-FileSize -Bytes $fileSize
            WastedSpace = $wastedBytes
            WastedSpaceFormatted = Format-FileSize -Bytes $wastedBytes
            Files = $files
            OldestFile = $files | Sort-Object -Property CreationTime | Select-Object -First 1
            NewestFile = $files | Sort-Object -Property CreationTime -Descending | Select-Object -First 1
        }

        $result.DuplicateGroups += $duplicateGroup
        $result.TotalDuplicates += $duplicateCount
        $result.WastedSpace += $wastedBytes
    }

    # Sort groups by wasted space (largest first)
    $result.DuplicateGroups = $result.DuplicateGroups | Sort-Object -Property WastedSpace -Descending
    $result.WastedSpaceFormatted = Format-FileSize -Bytes $result.WastedSpace

    $stopwatch.Stop()
    $result.ScanDuration = $stopwatch.Elapsed

    Write-Verbose "Scan complete in $($result.ScanDuration.TotalSeconds) seconds"
    Write-Verbose "Found $($result.DuplicateGroups.Count) groups of duplicates ($($result.TotalDuplicates) duplicate files)"
    Write-Verbose "Total wasted space: $($result.WastedSpaceFormatted)"

    Write-CleanupLog -Message "Duplicate scan complete: $($result.TotalDuplicates) duplicates, $($result.WastedSpaceFormatted) wasted" -Level Info

    # Display summary
    if ($result.DuplicateGroups.Count -gt 0) {
        Write-Host "`nDuplicate File Scan Results" -ForegroundColor Cyan
        Write-Host "===========================" -ForegroundColor Cyan
        Write-Host "  Path scanned: $Path" -ForegroundColor White
        Write-Host "  Files scanned: $($result.TotalFilesScanned)" -ForegroundColor White
        Write-Host "  Files hashed: $($result.TotalFilesHashed)" -ForegroundColor White
        Write-Host "  Duplicate groups: $($result.DuplicateGroups.Count)" -ForegroundColor Yellow
        Write-Host "  Total duplicates: $($result.TotalDuplicates)" -ForegroundColor Yellow
        Write-Host "  Wasted space: $($result.WastedSpaceFormatted)" -ForegroundColor Red
        Write-Host "  Scan duration: $([math]::Round($result.ScanDuration.TotalSeconds, 2)) seconds" -ForegroundColor Gray
        Write-Host ""
    }
    else {
        Write-Host "No duplicate files found." -ForegroundColor Green
    }

    return $result
}
