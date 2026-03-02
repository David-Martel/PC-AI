#Requires -Version 5.1
function Repair-MachinePath {
    <#
    .SYNOPSIS
        Repairs PATH environment variable by removing duplicates and invalid entries.

    .DESCRIPTION
        Cleans up the PATH environment variable by:
        - Removing duplicate entries (case-insensitive)
        - Optionally removing non-existent paths
        - Removing empty entries
        - Normalizing trailing slashes

        Creates a backup before any modifications.
        Supports -WhatIf for preview without changes.

    .PARAMETER Target
        Which PATH to repair: 'User' or 'Machine'. Default is 'User'.
        Machine PATH requires administrator privileges.

    .PARAMETER RemoveNonExistent
        Also remove paths that no longer exist on the filesystem.

    .PARAMETER NormalizeSlashes
        Remove trailing slashes from paths for consistency.

    .PARAMETER Force
        Skip confirmation prompts and proceed with changes.

    .PARAMETER BackupPath
        Custom path for the backup file. Default is in PC-AI Logs folder.

    .EXAMPLE
        Repair-MachinePath -WhatIf

        Preview what changes would be made to User PATH without applying them.

    .EXAMPLE
        Repair-MachinePath -Target User -RemoveNonExistent

        Repair User PATH, removing duplicates and non-existent paths.

    .EXAMPLE
        Repair-MachinePath -Target Machine -Force

        Repair Machine PATH (requires admin), skipping confirmations.

    .OUTPUTS
        PSCustomObject with properties:
        - Success: Boolean indicating if repair succeeded
        - Target: Which PATH was repaired
        - BackupPath: Location of backup file
        - EntriesRemoved: Count of entries removed
        - OriginalCount: Original entry count
        - FinalCount: Final entry count
        - Changes: List of specific changes made
    #>
    [CmdletBinding(SupportsShouldProcess, ConfirmImpact = 'High')]
    [OutputType([PSCustomObject])]
    param(
        [Parameter()]
        [ValidateSet('User', 'Machine')]
        [string]$Target = 'User',

        [Parameter()]
        [switch]$RemoveNonExistent,

        [Parameter()]
        [switch]$NormalizeSlashes,

        [Parameter()]
        [switch]$Force,

        [Parameter()]
        [string]$BackupPath
    )

    $result = [PSCustomObject]@{
        Success = $false
        Target = $Target
        BackupPath = $null
        OriginalCount = 0
        FinalCount = 0
        EntriesRemoved = 0
        DuplicatesRemoved = 0
        NonExistentRemoved = 0
        EmptyRemoved = 0
        Changes = @()
        Warnings = @()
    }

    Write-Verbose "Starting PATH repair for $Target"
    Write-CleanupLog -Message "Starting PATH repair (Target: $Target, RemoveNonExistent: $RemoveNonExistent)" -Level Info

    # Check for admin rights if targeting Machine PATH
    if ($Target -eq 'Machine' -and -not (Test-IsAdministrator)) {
        $errorMsg = "Administrator privileges required to modify Machine PATH. Please run as Administrator."
        Write-CleanupLog -Message $errorMsg -Level Error
        $result.Warnings += $errorMsg
        Write-Error $errorMsg
        return $result
    }

    # Get current PATH value
    $currentPath = [Environment]::GetEnvironmentVariable('PATH', $Target)

    if ([string]::IsNullOrEmpty($currentPath)) {
        Write-Warning "$Target PATH is empty. Nothing to repair."
        $result.Success = $true
        return $result
    }

    # Create backup
    $result.BackupPath = Backup-EnvironmentVariable -Name 'PATH' -Target $Target -BackupPath $BackupPath

    if (-not $result.BackupPath -and -not $Force) {
        $errorMsg = "Failed to create backup. Use -Force to proceed without backup (not recommended)."
        Write-CleanupLog -Message $errorMsg -Level Error
        $result.Warnings += $errorMsg
        Write-Error $errorMsg
        return $result
    }

    Write-Verbose "Backup created at: $($result.BackupPath)"

    # Parse PATH entries
    $pathItems = $currentPath -split ';'
    $result.OriginalCount = $pathItems.Count

    # Track what we've seen for duplicate detection
    $seenPaths = @{}
    $cleanedPaths = @()

    foreach ($item in $pathItems) {
        # Skip empty entries
        if ([string]::IsNullOrWhiteSpace($item)) {
            $result.EmptyRemoved++
            $result.Changes += [PSCustomObject]@{
                Action = 'Removed'
                Reason = 'Empty entry'
                OriginalValue = '<empty>'
            }
            continue
        }

        $trimmedItem = $item.Trim()

        # Normalize for comparison
        $normalizedPath = $trimmedItem.TrimEnd('\', '/').ToLowerInvariant()

        # Check for duplicate
        if ($seenPaths.ContainsKey($normalizedPath)) {
            $result.DuplicatesRemoved++
            $result.Changes += [PSCustomObject]@{
                Action = 'Removed'
                Reason = 'Duplicate'
                OriginalValue = $trimmedItem
                DuplicateOf = $seenPaths[$normalizedPath]
            }
            continue
        }

        # Check for non-existent (if requested)
        if ($RemoveNonExistent) {
            $expandedPath = [Environment]::ExpandEnvironmentVariables($trimmedItem)
            if (-not (Test-Path -Path $expandedPath -ErrorAction SilentlyContinue)) {
                $result.NonExistentRemoved++
                $result.Changes += [PSCustomObject]@{
                    Action = 'Removed'
                    Reason = 'Non-existent path'
                    OriginalValue = $trimmedItem
                    ExpandedPath = $expandedPath
                }
                continue
            }
        }

        # Normalize trailing slashes if requested
        $finalPath = $trimmedItem
        if ($NormalizeSlashes -and ($trimmedItem.EndsWith('\') -or $trimmedItem.EndsWith('/'))) {
            $finalPath = $trimmedItem.TrimEnd('\', '/')
            if ($finalPath -ne $trimmedItem) {
                $result.Changes += [PSCustomObject]@{
                    Action = 'Normalized'
                    Reason = 'Removed trailing slash'
                    OriginalValue = $trimmedItem
                    NewValue = $finalPath
                }
            }
        }

        # Keep this entry
        $seenPaths[$normalizedPath] = $finalPath
        $cleanedPaths += $finalPath
    }

    $result.FinalCount = $cleanedPaths.Count
    $result.EntriesRemoved = $result.OriginalCount - $result.FinalCount

    # Build new PATH string
    $newPath = $cleanedPaths -join ';'

    # Summary for confirmation
    $changesSummary = @"
PATH Repair Summary for $Target`:
  Original entries: $($result.OriginalCount)
  Final entries: $($result.FinalCount)
  Duplicates removed: $($result.DuplicatesRemoved)
  Non-existent removed: $($result.NonExistentRemoved)
  Empty entries removed: $($result.EmptyRemoved)
  Backup location: $($result.BackupPath)
"@

    Write-Verbose $changesSummary

    # Apply changes if confirmed
    if ($result.EntriesRemoved -eq 0 -and $result.Changes.Count -eq 0) {
        Write-Host "No changes needed for $Target PATH." -ForegroundColor Green
        $result.Success = $true
        return $result
    }

    $confirmMessage = "Apply $($result.EntriesRemoved) changes to $Target PATH?"

    if ($Force -or $PSCmdlet.ShouldProcess($Target, "Remove $($result.EntriesRemoved) entries from PATH")) {
        try {
            [Environment]::SetEnvironmentVariable('PATH', $newPath, $Target)

            Write-CleanupLog -Message "PATH repaired successfully. Removed $($result.EntriesRemoved) entries." -Level Info
            Write-Host "PATH repaired successfully!" -ForegroundColor Green
            Write-Host "  Removed $($result.EntriesRemoved) entries" -ForegroundColor Yellow
            Write-Host "  Backup saved to: $($result.BackupPath)" -ForegroundColor Cyan

            # Update current process PATH if we modified it
            if ($Target -eq 'User') {
                $machinePath = [Environment]::GetEnvironmentVariable('PATH', 'Machine')
                $env:PATH = "$newPath;$machinePath"
            }
            elseif ($Target -eq 'Machine') {
                $userPath = [Environment]::GetEnvironmentVariable('PATH', 'User')
                $env:PATH = "$userPath;$newPath"
            }

            $result.Success = $true
        }
        catch {
            $errorMsg = "Failed to update PATH: $_"
            Write-CleanupLog -Message $errorMsg -Level Error
            $result.Warnings += $errorMsg
            Write-Error $errorMsg
        }
    }
    else {
        Write-Host "Operation cancelled. No changes made." -ForegroundColor Yellow
        Write-Host "To restore from backup if needed:" -ForegroundColor Cyan
        Write-Host "  Get-Content '$($result.BackupPath)' | Set-Content env:PATH" -ForegroundColor Gray
    }

    return $result
}
