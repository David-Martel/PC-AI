#Requires -Version 5.1
function Get-PathDuplicates {
    <#
    .SYNOPSIS
        Analyzes PATH environment variable for duplicates and non-existent entries.

    .DESCRIPTION
        Examines both User and Machine PATH environment variables to identify:
        - Duplicate entries (exact and case-insensitive matches)
        - Non-existent paths
        - Empty or whitespace entries
        - Paths with trailing backslashes (normalization issues)

        Returns a detailed report with recommendations for cleanup.

    .PARAMETER Target
        Which PATH to analyze: 'User', 'Machine', or 'Both'. Default is 'Both'.

    .PARAMETER IncludeProcess
        Also include the current process PATH (combined view).

    .EXAMPLE
        Get-PathDuplicates

        Analyzes both User and Machine PATH variables and returns a report.

    .EXAMPLE
        Get-PathDuplicates -Target User

        Analyzes only the User PATH variable.

    .EXAMPLE
        Get-PathDuplicates -Target Both -IncludeProcess | Format-List

        Full analysis including process PATH, with detailed output.

    .OUTPUTS
        PSCustomObject with properties:
        - UserPath: Analysis of User PATH
        - MachinePath: Analysis of Machine PATH
        - CrossDuplicates: Paths duplicated between User and Machine
        - Summary: Overall statistics and recommendations
    #>
    [CmdletBinding()]
    [OutputType([PSCustomObject])]
    param(
        [Parameter()]
        [ValidateSet('User', 'Machine', 'Both')]
        [string]$Target = 'Both',

        [Parameter()]
        [switch]$IncludeProcess
    )

    Write-Verbose "Analyzing PATH environment variable (Target: $Target)"
    Write-CleanupLog -Message "Starting PATH analysis (Target: $Target)" -Level Info

    # Helper function to analyze a single PATH
    function Analyze-PathVariable {
        param(
            [string]$PathValue,
            [string]$Source
        )

        $entries = @()
        $duplicates = @()
        $nonExistent = @()
        $emptyEntries = 0
        $seenPaths = @{}

        if ([string]::IsNullOrEmpty($PathValue)) {
            return [PSCustomObject]@{
                Source                   = $Source
                TotalEntries             = 0
                UniqueEntries            = 0
                Duplicates               = @()
                NonExistent              = @()
                EmptyEntries             = 0
                EntriesWithTrailingSlash = @()
                AllEntries               = @()
            }
        }

        $pathItems = $PathValue -split ';'
        $index = 0

        foreach ($item in $pathItems) {
            $index++

            # Check for empty entries
            if ([string]::IsNullOrWhiteSpace($item)) {
                $emptyEntries++
                continue
            }

            # Normalize path for comparison (trim, lowercase, remove trailing slash)
            $normalizedPath = $item.Trim().TrimEnd('\', '/').ToLowerInvariant()

            # Expand environment variables for existence check
            $expandedPath = [Environment]::ExpandEnvironmentVariables($item.Trim())

            $entry = [PSCustomObject]@{
                Index            = $index
                OriginalValue    = $item
                NormalizedValue  = $normalizedPath
                ExpandedValue    = $expandedPath
                Exists           = Test-Path -Path $expandedPath -ErrorAction SilentlyContinue
                HasTrailingSlash = $item.Trim().EndsWith('\') -or $item.Trim().EndsWith('/')
                IsDuplicate      = $false
                DuplicateOf      = $null
            }

            # Check for duplicates
            if ($seenPaths.ContainsKey($normalizedPath)) {
                $entry.IsDuplicate = $true
                $entry.DuplicateOf = $seenPaths[$normalizedPath]
                $duplicates += $entry
            } else {
                $seenPaths[$normalizedPath] = $index
            }

            # Check for non-existent
            if (-not $entry.Exists) {
                $nonExistent += $entry
            }

            $entries += $entry
        }

        $trailingSlashEntries = $entries | Where-Object { $_.HasTrailingSlash }

        return [PSCustomObject]@{
            Source                   = $Source
            TotalEntries             = $entries.Count
            UniqueEntries            = $entries.Count - $duplicates.Count
            Duplicates               = $duplicates
            NonExistent              = $nonExistent
            EmptyEntries             = $emptyEntries
            EntriesWithTrailingSlash = $trailingSlashEntries
            AllEntries               = $entries
        }
    }

    $result = [PSCustomObject]@{
        Timestamp       = Get-Date
        UserPath        = $null
        MachinePath     = $null
        ProcessPath     = $null
        CrossDuplicates = @()
        Summary         = $null
    }

    # Check for native acceleration
    $useNative = (-not $env:PCAI_DISABLE_NATIVE) -and [PcaiNative.PcaiCore]::IsAvailable

    if ($useNative) {
        Write-Verbose 'Using native PCAI core for PATH analysis'
        $nativeJson = [PcaiNative.PcaiCore]::AnalyzePathJson()
        if ($nativeJson) {
            $report = $nativeJson | ConvertFrom-Json

            # Helper to map native entries to PSCustomObject
            function Map-NativeEntry($nativeEntry, $issues) {
                $path = $nativeEntry.path
                $source = $nativeEntry.source
                $index = $nativeEntry.index

                $issue = $issues | Where-Object { $_.path -eq $path -and $_.source -eq $source }

                return [PSCustomObject]@{
                    Index            = $index + 1 # Convert to 1-based for PS consistency
                    OriginalValue    = $path
                    NormalizedValue  = $path.Trim().TrimEnd('\', '/').ToLowerInvariant()
                    ExpandedValue    = [Environment]::ExpandEnvironmentVariables($path)
                    Exists           = ($issue -and $issue.issue_type -eq 'NonExistent') ? $false : $true
                    HasTrailingSlash = ($issue -and $issue.issue_type -eq 'TrailingSlash')
                    IsDuplicate      = ($issue -and ($issue.issue_type -eq 'Duplicate' -or $issue.issue_type -eq 'CrossDuplicate'))
                    DuplicateOf      = ($issue -and $issue.duplicate_of) ? $issue.duplicate_of : $null
                    IssueType        = $issue ? $issue.issue_type : 'None'
                    Severity         = $issue ? $issue.severity : 'None'
                    Description      = $issue ? $issue.description : 'OK'
                }
            }

            # Map native report to PowerShell structure
            $result = [PSCustomObject]@{
                Timestamp       = Get-Date
                UserPath        = [PSCustomObject]@{
                    Source                   = 'User'
                    TotalEntries             = $report.user_total_entries
                    UniqueEntries            = $report.user_unique_entries
                    Duplicates               = @($report.issues | Where-Object { $_.source -eq 'User' -and ($_.issue_type -eq 'Duplicate' -or $_.issue_type -eq 'CrossDuplicate') })
                    NonExistent              = @($report.issues | Where-Object { $_.source -eq 'User' -and $_.issue_type -eq 'NonExistent' })
                    EmptyEntries             = $report.empty_count # Shared or filtered
                    EntriesWithTrailingSlash = @($report.issues | Where-Object { $_.source -eq 'User' -and $_.issue_type -eq 'TrailingSlash' })
                    AllEntries               = @($report.user_entries | ForEach-Object { Map-NativeEntry $_ $report.issues })
                }
                MachinePath     = [PSCustomObject]@{
                    Source                   = 'Machine'
                    TotalEntries             = $report.machine_total_entries
                    UniqueEntries            = $report.machine_unique_entries
                    Duplicates               = @($report.issues | Where-Object { $_.source -eq 'Machine' -and ($_.issue_type -eq 'Duplicate' -or $_.issue_type -eq 'CrossDuplicate') })
                    NonExistent              = @($report.issues | Where-Object { $_.source -eq 'Machine' -and $_.issue_type -eq 'NonExistent' })
                    EmptyEntries             = $report.empty_count # Shared or filtered
                    EntriesWithTrailingSlash = @($report.issues | Where-Object { $_.source -eq 'Machine' -and $_.issue_type -eq 'TrailingSlash' })
                    AllEntries               = @($report.machine_entries | ForEach-Object { Map-NativeEntry $_ $report.issues })
                }
                ProcessPath     = $null # Native analysis focused on persistent PATH
                CrossDuplicates = @()
                Summary         = [PSCustomObject]@{
                    HealthStatus         = $report.health_status
                    TotalDuplicates      = $report.duplicate_count
                    TotalNonExistent     = $report.non_existent_count
                    TotalCrossDuplicates = $report.cross_duplicate_count
                    TotalEmptyEntries    = $report.empty_count
                    Recommendations      = $report.recommendations
                }
                NativeReport    = $report
            }

            # Map cross-duplicates explicitly
            foreach ($duplicate in $report.duplicates) {
                $hasUser = $duplicate.entries | Where-Object { $_.source -eq 'User' }
                $hasMachine = $duplicate.entries | Where-Object { $_.source -eq 'Machine' }

                if ($hasUser -and $hasMachine) {
                    $result.CrossDuplicates += [PSCustomObject]@{
                        Path           = $duplicate.normalized_path
                        UserIndex      = ($hasUser | Select-Object -First 1).index + 1
                        MachineIndex   = ($hasMachine | Select-Object -First 1).index + 1
                        Recommendation = 'Remove from User PATH (Machine PATH takes precedence)'
                    }
                }
            }

            Write-CleanupLog -Message "Native PATH analysis complete: $($report.health_status) ($($report.elapsed_ms)ms)" -Level Info
            return $result
        }
    }

    # Fallback to PowerShell-based analysis
    Write-Verbose 'Using PowerShell-based PATH analysis'

    # Analyze User PATH
    if ($Target -eq 'User' -or $Target -eq 'Both') {
        $userPathValue = [Environment]::GetEnvironmentVariable('PATH', 'User')
        $result.UserPath = Analyze-PathVariable -PathValue $userPathValue -Source 'User'
        Write-Verbose "User PATH: $($result.UserPath.TotalEntries) entries, $($result.UserPath.Duplicates.Count) duplicates"
    }

    # Analyze Machine PATH
    if ($Target -eq 'Machine' -or $Target -eq 'Both') {
        $machinePathValue = [Environment]::GetEnvironmentVariable('PATH', 'Machine')
        $result.MachinePath = Analyze-PathVariable -PathValue $machinePathValue -Source 'Machine'
        Write-Verbose "Machine PATH: $($result.MachinePath.TotalEntries) entries, $($result.MachinePath.Duplicates.Count) duplicates"
    }

    # Analyze Process PATH if requested
    if ($IncludeProcess) {
        $processPathValue = $env:PATH
        $result.ProcessPath = Analyze-PathVariable -PathValue $processPathValue -Source 'Process'
        Write-Verbose "Process PATH: $($result.ProcessPath.TotalEntries) entries"
    }

    # Find cross-duplicates (paths in both User and Machine)
    if ($Target -eq 'Both' -and $result.UserPath -and $result.MachinePath) {
        $userNormalized = @{}
        foreach ($entry in $result.UserPath.AllEntries) {
            if (-not $entry.IsDuplicate) {
                $userNormalized[$entry.NormalizedValue] = $entry
            }
        }

        foreach ($entry in $result.MachinePath.AllEntries) {
            if (-not $entry.IsDuplicate -and $userNormalized.ContainsKey($entry.NormalizedValue)) {
                $result.CrossDuplicates += [PSCustomObject]@{
                    Path           = $entry.OriginalValue
                    NormalizedPath = $entry.NormalizedValue
                    UserIndex      = $userNormalized[$entry.NormalizedValue].Index
                    MachineIndex   = $entry.Index
                    Recommendation = 'Remove from User PATH (Machine PATH takes precedence)'
                }
            }
        }

        Write-Verbose "Cross-duplicates found: $($result.CrossDuplicates.Count)"
    }

    # Generate summary and recommendations
    $totalDuplicates = 0
    $totalNonExistent = 0
    $totalEmpty = 0
    $recommendations = @()

    if ($result.UserPath) {
        $totalDuplicates += $result.UserPath.Duplicates.Count
        $totalNonExistent += $result.UserPath.NonExistent.Count
        $totalEmpty += $result.UserPath.EmptyEntries

        if ($result.UserPath.Duplicates.Count -gt 0) {
            $recommendations += "User PATH has $($result.UserPath.Duplicates.Count) duplicate entries - run Repair-MachinePath -Target User"
        }
        if ($result.UserPath.NonExistent.Count -gt 0) {
            $recommendations += "User PATH has $($result.UserPath.NonExistent.Count) non-existent paths - consider removing"
        }
    }

    if ($result.MachinePath) {
        $totalDuplicates += $result.MachinePath.Duplicates.Count
        $totalNonExistent += $result.MachinePath.NonExistent.Count
        $totalEmpty += $result.MachinePath.EmptyEntries

        if ($result.MachinePath.Duplicates.Count -gt 0) {
            $recommendations += "Machine PATH has $($result.MachinePath.Duplicates.Count) duplicate entries - run Repair-MachinePath -Target Machine (requires admin)"
        }
        if ($result.MachinePath.NonExistent.Count -gt 0) {
            $recommendations += "Machine PATH has $($result.MachinePath.NonExistent.Count) non-existent paths - consider removing"
        }
    }

    if ($result.CrossDuplicates.Count -gt 0) {
        $recommendations += "$($result.CrossDuplicates.Count) paths appear in both User and Machine PATH - consolidate for efficiency"
    }

    if ($totalEmpty -gt 0) {
        $recommendations += "$totalEmpty empty PATH entries found - will be removed during repair"
    }

    $healthStatus = 'Healthy'
    if ($totalDuplicates -gt 0 -or $totalNonExistent -gt 0 -or $result.CrossDuplicates.Count -gt 0) {
        $healthStatus = 'Needs Attention'
    }
    if ($totalDuplicates -gt 5 -or $totalNonExistent -gt 5) {
        $healthStatus = 'Needs Cleanup'
    }

    $result.Summary = [PSCustomObject]@{
        HealthStatus         = $healthStatus
        TotalDuplicates      = $totalDuplicates
        TotalNonExistent     = $totalNonExistent
        TotalCrossDuplicates = $result.CrossDuplicates.Count
        TotalEmptyEntries    = $totalEmpty
        Recommendations      = $recommendations
    }

    Write-CleanupLog -Message "PATH analysis complete: $healthStatus - $totalDuplicates duplicates, $totalNonExistent non-existent" -Level Info

    return $result
}
