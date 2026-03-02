#Requires -Version 5.1
function Clear-TempFiles {
    <#
    .SYNOPSIS
        Safely cleans temporary files from common system locations.

    .DESCRIPTION
        Removes temporary files from:
        - Windows Temp folder
        - User Temp folder
        - Browser caches (Chrome, Edge, Firefox)
        - Windows Update download cache
        - Thumbnail cache

        Features:
        - Skips files currently in use
        - Supports -WhatIf for preview
        - Shows space reclaimed
        - Detailed logging
        - Optional age filter to keep recent files

    .PARAMETER Target
        Which temp locations to clean: 'All', 'System', 'User', or 'Browser'.
        Default is 'All'.

    .PARAMETER OlderThanDays
        Only remove files older than this many days. Default is 0 (all files).
        Use this to preserve recently used temp files.

    .PARAMETER IncludePrefetch
        Also clean Windows Prefetch folder. Requires admin and may impact
        application startup performance temporarily.

    .PARAMETER IncludeWindowsUpdate
        Also clean Windows Update download cache. Requires admin.

    .PARAMETER Force
        Skip confirmation prompts.

    .EXAMPLE
        Clear-TempFiles -WhatIf

        Preview what would be deleted without actually removing files.

    .EXAMPLE
        Clear-TempFiles -Target User -OlderThanDays 7

        Clean user temp files older than 7 days.

    .EXAMPLE
        Clear-TempFiles -Target All -IncludePrefetch -Force

        Full cleanup including prefetch (requires admin), no confirmations.

    .EXAMPLE
        Clear-TempFiles -Target Browser

        Clean only browser caches.

    .OUTPUTS
        PSCustomObject with properties:
        - TotalFilesDeleted: Count of files removed
        - TotalFilesSkipped: Count of files that couldn't be removed
        - TotalBytesReclaimed: Bytes freed
        - SpaceReclaimed: Human-readable space freed
        - LocationResults: Per-location breakdown
        - Errors: Any errors encountered
    #>
    [CmdletBinding(SupportsShouldProcess, ConfirmImpact = 'Medium')]
    [OutputType([PSCustomObject])]
    param(
        [Parameter()]
        [ValidateSet('All', 'System', 'User', 'Browser')]
        [string]$Target = 'All',

        [Parameter()]
        [ValidateRange(0, 365)]
        [int]$OlderThanDays = 0,

        [Parameter()]
        [switch]$IncludePrefetch,

        [Parameter()]
        [switch]$IncludeWindowsUpdate,

        [Parameter()]
        [switch]$Force
    )

    $isAdmin = Test-IsAdministrator

    Write-Verbose "Starting temp file cleanup (Target: $Target, OlderThanDays: $OlderThanDays)"
    Write-CleanupLog -Message "Starting temp cleanup (Target: $Target, Admin: $isAdmin)" -Level Info

    $result = [PSCustomObject]@{
        Timestamp = Get-Date
        Target = $Target
        OlderThanDays = $OlderThanDays
        RunAsAdmin = $isAdmin
        TotalFilesDeleted = 0
        TotalFilesSkipped = 0
        TotalBytesReclaimed = 0
        SpaceReclaimed = ''
        LocationResults = @()
        Warnings = @()
        Errors = @()
    }

    # Get temp paths based on target
    $allTempPaths = Get-TempPaths

    $pathsToClean = switch ($Target) {
        'System' {
            $allTempPaths | Where-Object { $_.Name -like 'Windows*' }
        }
        'User' {
            $allTempPaths | Where-Object { $_.Name -eq 'User Temp' }
        }
        'Browser' {
            $allTempPaths | Where-Object { $_.Name -like '*Cache*' -and $_.Name -notlike 'Windows*' }
        }
        default {
            $allTempPaths
        }
    }

    # Add optional locations
    if ($IncludePrefetch) {
        $prefetch = $allTempPaths | Where-Object { $_.Name -eq 'Windows Prefetch' }
        if ($prefetch -and $prefetch -notin $pathsToClean) {
            $pathsToClean += $prefetch
        }
    }

    if ($IncludeWindowsUpdate) {
        $wuCache = $allTempPaths | Where-Object { $_.Name -eq 'Windows Update Download Cache' }
        if ($wuCache -and $wuCache -notin $pathsToClean) {
            $pathsToClean += $wuCache
        }
    }

    # Filter out admin-required paths if not admin
    $skippedForAdmin = @()
    if (-not $isAdmin) {
        $adminPaths = $pathsToClean | Where-Object { $_.RequiresAdmin }
        if ($adminPaths) {
            $skippedForAdmin = $adminPaths
            $pathsToClean = $pathsToClean | Where-Object { -not $_.RequiresAdmin }
            $result.Warnings += "Skipping $($adminPaths.Count) locations requiring admin: $($adminPaths.Name -join ', ')"
            Write-Warning "Some locations require administrator privileges and will be skipped."
        }
    }

    if ($pathsToClean.Count -eq 0) {
        Write-Warning "No temp locations to clean. Run as administrator for system locations."
        return $result
    }

    # Calculate space before cleanup
    $spaceBefore = 0
    foreach ($location in $pathsToClean) {
        if (Test-Path -Path $location.Path) {
            $getParams = @{
                Path = $location.Path
                File = $true
                Recurse = $true
                ErrorAction = 'SilentlyContinue'
            }
            if ($location.Filter) {
                $getParams['Filter'] = $location.Filter
            }
            $files = Get-ChildItem @getParams
            $spaceBefore += ($files | Measure-Object -Property Length -Sum).Sum
        }
    }

    $formattedSpaceBefore = Format-FileSize -Bytes $spaceBefore
    Write-Host "`nTemp File Cleanup" -ForegroundColor Cyan
    Write-Host "=================" -ForegroundColor Cyan
    Write-Host "  Target: $Target" -ForegroundColor White
    Write-Host "  Locations to clean: $($pathsToClean.Count)" -ForegroundColor White
    Write-Host "  Space used by temp files: $formattedSpaceBefore" -ForegroundColor Yellow

    if ($OlderThanDays -gt 0) {
        Write-Host "  Only files older than: $OlderThanDays days" -ForegroundColor White
    }

    Write-Host ""

    # Confirm before proceeding
    if (-not $Force -and -not $WhatIfPreference) {
        $confirmMessage = "Clean temp files from $($pathsToClean.Count) location(s)?"
        if (-not $PSCmdlet.ShouldContinue($confirmMessage, "Temp File Cleanup")) {
            Write-Host "Operation cancelled." -ForegroundColor Yellow
            return $result
        }
    }

    $disableNative = $env:PCAI_DISABLE_NATIVE -eq '1'

    # Clean each location
    foreach ($location in $pathsToClean) {
        Write-Verbose "Cleaning: $($location.Name) - $($location.Path)"

        if (-not (Test-Path -Path $location.Path)) {
            Write-Verbose "  Path does not exist, skipping."
            continue
        }

        $locationResult = [PSCustomObject]@{
            Name = $location.Name
            Path = $location.Path
            FilesDeleted = 0
            FilesSkipped = 0
            BytesReclaimed = 0
            Errors = @()
        }

        # Get files to clean
        $getParams = @{
            Path = $location.Path
            File = $true
            Recurse = $true
            ErrorAction = 'SilentlyContinue'
        }
        if ($location.Filter) {
            $getParams['Filter'] = $location.Filter
        }

        $files = Get-ChildItem @getParams

        # Filter by age if specified
        if ($OlderThanDays -gt 0) {
            $cutoffDate = (Get-Date).AddDays(-$OlderThanDays)
            $files = $files | Where-Object { $_.LastWriteTime -lt $cutoffDate }
        }

        $fileCount = ($files | Measure-Object).Count
        Write-Verbose "  Found $fileCount files to process"

        foreach ($file in $files) {
            $actionDescription = "Delete temp file: $($file.FullName)"

            if ($PSCmdlet.ShouldProcess($file.FullName, 'Delete')) {
                try {
                    $fileSize = $file.Length

                    # Use native high-performance delete if available (unless disabled)
                    if (-not $disableNative -and [PcaiNative.PcaiCore]::IsAvailable) {
                        $status = [PcaiNative.PcaiCore]::DeleteFsItem($file.FullName, $false)
                        if ($status -ne 'Success') {
                            throw "Native delete failed with status $status"
                        }
                    } else {
                        Remove-Item -Path $file.FullName -Force -ErrorAction Stop
                    }

                    $locationResult.FilesDeleted++
                    $locationResult.BytesReclaimed += $fileSize
                }
                catch {
                    $locationResult.FilesSkipped++
                    $errorMsg = "Could not delete '$($file.Name)': $($_.Exception.Message)"
                    $locationResult.Errors += $errorMsg
                    Write-Verbose "  $errorMsg"
                }
            }
        }

        # Try to clean empty subdirectories
        if (-not $WhatIfPreference) {
            $emptyDirs = Get-ChildItem -Path $location.Path -Directory -Recurse -ErrorAction SilentlyContinue |
                Where-Object { (Get-ChildItem -Path $_.FullName -Force -ErrorAction SilentlyContinue | Measure-Object).Count -eq 0 } |
                Sort-Object -Property FullName -Descending

            foreach ($dir in $emptyDirs) {
                try {
                    Remove-Item -Path $dir.FullName -Force -ErrorAction Stop
                    Write-Verbose "  Removed empty directory: $($dir.Name)"
                }
                catch {
                    # Ignore errors on directory removal
                }
            }
        }

        # Report for this location
        $formattedReclaimed = Format-FileSize -Bytes $locationResult.BytesReclaimed
        Write-Host "  $($location.Name): " -NoNewline
        if ($locationResult.FilesDeleted -gt 0) {
            Write-Host "$($locationResult.FilesDeleted) files, $formattedReclaimed freed" -ForegroundColor Green
        }
        else {
            Write-Host "No files removed" -ForegroundColor Gray
        }
        if ($locationResult.FilesSkipped -gt 0) {
            Write-Host "    ($($locationResult.FilesSkipped) files in use, skipped)" -ForegroundColor DarkGray
        }

        # Accumulate totals
        $result.TotalFilesDeleted += $locationResult.FilesDeleted
        $result.TotalFilesSkipped += $locationResult.FilesSkipped
        $result.TotalBytesReclaimed += $locationResult.BytesReclaimed
        $result.Errors += $locationResult.Errors
        $result.LocationResults += $locationResult
    }

    $result.SpaceReclaimed = Format-FileSize -Bytes $result.TotalBytesReclaimed

    # Summary
    Write-Host ""
    Write-Host "Cleanup Summary" -ForegroundColor Cyan
    Write-Host "---------------" -ForegroundColor Cyan
    Write-Host "  Files deleted: $($result.TotalFilesDeleted)" -ForegroundColor $(if ($result.TotalFilesDeleted -gt 0) { 'Green' } else { 'Gray' })
    Write-Host "  Files skipped: $($result.TotalFilesSkipped)" -ForegroundColor $(if ($result.TotalFilesSkipped -gt 0) { 'Yellow' } else { 'Gray' })
    Write-Host "  Space reclaimed: $($result.SpaceReclaimed)" -ForegroundColor $(if ($result.TotalBytesReclaimed -gt 0) { 'Green' } else { 'Gray' })

    if ($skippedForAdmin.Count -gt 0) {
        Write-Host ""
        Write-Host "  Tip: Run as administrator to clean:" -ForegroundColor DarkYellow
        foreach ($skipped in $skippedForAdmin) {
            Write-Host "    - $($skipped.Name)" -ForegroundColor DarkGray
        }
    }

    Write-CleanupLog -Message "Cleanup complete: $($result.TotalFilesDeleted) files, $($result.SpaceReclaimed) reclaimed" -Level Info

    return $result
}
