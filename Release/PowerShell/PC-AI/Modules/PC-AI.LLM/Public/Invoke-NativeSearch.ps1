#Requires -Version 5.1
<#
.SYNOPSIS
    Invokes native high-performance search operations for LLM analysis

.DESCRIPTION
    Provides a unified interface for pcai-inference/LLM to invoke native Rust-powered
    search operations. Returns LLM-optimized JSON output that can be easily
    parsed and analyzed by language models.

    Supported operations:
    - Duplicates: Find duplicate files using parallel SHA-256 hashing
    - Files: Fast file search with glob patterns
    - Content: Parallel regex content search

.PARAMETER Operation
    The type of search operation to perform

.PARAMETER Path
    Root path to search (defaults to current directory)

.PARAMETER Pattern
    Pattern for file or content search

.PARAMETER MinimumSize
    Minimum file size for duplicate detection (default: 1KB)

.PARAMETER MaxResults
    Maximum number of results (0 = unlimited)

.PARAMETER ContextLines
    Context lines for content search

.PARAMETER AsJson
    Return raw JSON string instead of PSObject

.EXAMPLE
    Invoke-NativeSearch -Operation Duplicates -Path "D:\Downloads"
    Finds duplicate files in Downloads directory

.EXAMPLE
    Invoke-NativeSearch -Operation Content -Path "C:\Logs" -Pattern "ERROR|WARN" -FilePattern "*.log"
    Searches log files for errors and warnings

.EXAMPLE
    Invoke-NativeSearch -Operation Files -Pattern "*.ps1" -AsJson | Send-OllamaRequest -Prompt "Analyze these PowerShell scripts"
    Finds PS1 files and sends to pcai-inference for analysis

.OUTPUTS
    PSCustomObject with search results and metadata optimized for LLM consumption
#>
function Invoke-NativeSearch {
    [CmdletBinding()]
    [OutputType([PSCustomObject])]
    param(
        [Parameter(Mandatory, Position = 0)]
        [ValidateSet('Duplicates', 'Files', 'Content')]
        [string]$Operation,

        [Parameter()]
        [string]$Path,

        [Parameter()]
        [string]$Pattern,

        [Parameter()]
        [string]$FilePattern,

        [Parameter()]
        [int64]$MinimumSize = 1KB,

        [Parameter()]
        [int64]$MaxResults = 100,

        [Parameter()]
        [int]$ContextLines = 2,

        [Parameter()]
        [switch]$AsJson
    )

    # Import acceleration module if not loaded
    $accelModule = Get-Module -Name 'PC-AI.Acceleration'
    if (-not $accelModule) {
        $modulePath = Join-Path (Split-Path -Parent $PSScriptRoot) '..\PC-AI.Acceleration\PC-AI.Acceleration.psd1'
        if (Test-Path $modulePath) {
            Import-Module $modulePath -ErrorAction SilentlyContinue
        }
    }

    # Check native availability
    $nativeAvailable = $false
    try {
        $nativeAvailable = Test-PcaiNativeAvailable
    }
    catch {
        Write-Verbose "Native tools not available: $_"
    }

    $startTime = Get-Date
    $result = $null
    $engine = if ($nativeAvailable) { 'Native/Rust' } else { 'PowerShell' }

    # Resolve path
    $searchPath = if ($Path) {
        Resolve-Path $Path -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Path
    } else {
        Get-Location | Select-Object -ExpandProperty Path
    }

    if (-not $searchPath -or -not (Test-Path $searchPath)) {
        throw "Invalid path: $Path"
    }

    try {
        switch ($Operation) {
            'Duplicates' {
                if ($nativeAvailable) {
                    $nativeResult = Invoke-PcaiNativeDuplicates -Path $searchPath -MinimumSize $MinimumSize
                    $result = Format-DuplicateResultForLLM -NativeResult $nativeResult
                }
                else {
                    $result = Invoke-PowerShellDuplicates -Path $searchPath -MinimumSize $MinimumSize -MaxResults $MaxResults
                }
            }

            'Files' {
                if (-not $Pattern) {
                    throw "Pattern is required for Files operation"
                }

                if ($nativeAvailable) {
                    $nativeResult = Invoke-PcaiNativeFileSearch -Pattern $Pattern -Path $searchPath -MaxResults $MaxResults
                    $result = Format-FileSearchResultForLLM -NativeResult $nativeResult
                }
                else {
                    $result = Invoke-PowerShellFileSearch -Path $searchPath -Pattern $Pattern -MaxResults $MaxResults
                }
            }

            'Content' {
                if (-not $Pattern) {
                    throw "Pattern is required for Content operation"
                }

                if ($nativeAvailable) {
                    $nativeResult = Invoke-PcaiNativeContentSearch -Pattern $Pattern -Path $searchPath -FilePattern $FilePattern -MaxResults $MaxResults -ContextLines $ContextLines
                    $result = Format-ContentSearchResultForLLM -NativeResult $nativeResult
                }
                else {
                    $result = Invoke-PowerShellContentSearch -Path $searchPath -Pattern $Pattern -FilePattern $FilePattern -MaxResults $MaxResults -ContextLines $ContextLines
                }
            }
        }
    }
    catch {
        $result = [PSCustomObject]@{
            Operation = $Operation
            Status    = 'Error'
            Error     = $_.Exception.Message
            Path      = $searchPath
            Engine    = $engine
        }
    }

    $endTime = Get-Date
    $duration = ($endTime - $startTime).TotalMilliseconds

    # Add metadata
    $result | Add-Member -MemberType NoteProperty -Name 'Operation' -Value $Operation -Force
    $result | Add-Member -MemberType NoteProperty -Name 'Engine' -Value $engine -Force
    $result | Add-Member -MemberType NoteProperty -Name 'TotalDurationMs' -Value ([math]::Round($duration, 2)) -Force
    $result | Add-Member -MemberType NoteProperty -Name 'SearchPath' -Value $searchPath -Force
    $result | Add-Member -MemberType NoteProperty -Name 'Timestamp' -Value $startTime.ToString('o') -Force

    if ($AsJson) {
        return ($result | ConvertTo-Json -Depth 10 -Compress)
    }

    return $result
}

# Helper functions for formatting results

function Format-DuplicateResultForLLM {
    [CmdletBinding()]
    param($NativeResult)

    if (-not $NativeResult) {
        return [PSCustomObject]@{
            Status = 'NoResults'
            Summary = 'No duplicates found or search failed'
        }
    }

    # Build LLM-friendly summary
    $groups = $NativeResult.Groups | Select-Object -First 20  # Limit for context window

    [PSCustomObject]@{
        Status          = $NativeResult.Status
        FilesScanned    = $NativeResult.FilesScanned
        DuplicateGroups = $NativeResult.DuplicateGroupCount
        DuplicateFiles  = $NativeResult.DuplicateFiles
        WastedBytes     = $NativeResult.WastedBytes
        WastedMB        = [math]::Round($NativeResult.WastedBytes / 1MB, 2)
        WastedGB        = [math]::Round($NativeResult.WastedBytes / 1GB, 3)
        ElapsedMs       = $NativeResult.ElapsedMs
        TopGroups       = $groups | ForEach-Object {
            [PSCustomObject]@{
                Hash       = $_.Hash.Substring(0, 16) + '...'
                Size       = $_.Size
                SizeMB     = [math]::Round($_.Size / 1MB, 2)
                Count      = $_.Paths.Count
                WastedMB   = [math]::Round($_.WastedBytes / 1MB, 2)
                Paths      = $_.Paths | Select-Object -First 5
            }
        }
        Summary         = "Found $($NativeResult.DuplicateGroupCount) duplicate groups with $($NativeResult.DuplicateFiles) duplicate files wasting $([math]::Round($NativeResult.WastedBytes / 1MB, 2)) MB"
    }
}

function Format-FileSearchResultForLLM {
    [CmdletBinding()]
    param($NativeResult)

    if (-not $NativeResult) {
        return [PSCustomObject]@{
            Status = 'NoResults'
            Summary = 'No files found or search failed'
        }
    }

    [PSCustomObject]@{
        Status        = $NativeResult.Status
        Pattern       = $NativeResult.Pattern
        FilesScanned  = $NativeResult.FilesScanned
        FilesMatched  = $NativeResult.FilesMatched
        TotalSize     = $NativeResult.TotalSize
        TotalSizeMB   = [math]::Round($NativeResult.TotalSize / 1MB, 2)
        ElapsedMs     = $NativeResult.ElapsedMs
        Truncated     = $NativeResult.Truncated
        Files         = $NativeResult.Files | Select-Object -First 50 | ForEach-Object {
            [PSCustomObject]@{
                Path     = $_.Path
                Size     = $_.Size
                SizeKB   = [math]::Round($_.Size / 1KB, 2)
                ReadOnly = $_.ReadOnly
            }
        }
        Summary       = "Found $($NativeResult.FilesMatched) files matching '$($NativeResult.Pattern)' totaling $([math]::Round($NativeResult.TotalSize / 1MB, 2)) MB"
    }
}

function Format-ContentSearchResultForLLM {
    [CmdletBinding()]
    param($NativeResult)

    if (-not $NativeResult) {
        return [PSCustomObject]@{
            Status = 'NoResults'
            Summary = 'No matches found or search failed'
        }
    }

    [PSCustomObject]@{
        Status        = $NativeResult.Status
        Pattern       = $NativeResult.Pattern
        FilePattern   = $NativeResult.FilePattern
        FilesScanned  = $NativeResult.FilesScanned
        FilesMatched  = $NativeResult.FilesMatched
        TotalMatches  = $NativeResult.TotalMatches
        ElapsedMs     = $NativeResult.ElapsedMs
        Truncated     = $NativeResult.Truncated
        Matches       = $NativeResult.Matches | Select-Object -First 30 | ForEach-Object {
            [PSCustomObject]@{
                Path       = $_.Path
                LineNumber = $_.LineNumber
                Line       = $_.Line.Trim()
                Context    = @{
                    Before = $_.Before
                    After  = $_.After
                }
            }
        }
        Summary       = "Found $($NativeResult.TotalMatches) matches in $($NativeResult.FilesMatched) files for pattern '$($NativeResult.Pattern)'"
    }
}

# PowerShell fallback implementations

function Invoke-PowerShellDuplicates {
    [CmdletBinding()]
    param(
        [string]$Path,
        [int64]$MinimumSize,
        [int64]$MaxResults
    )

    $files = Get-ChildItem -Path $Path -File -Recurse -ErrorAction SilentlyContinue |
        Where-Object { $_.Length -ge $MinimumSize }

    $sizeGroups = $files | Group-Object -Property Length | Where-Object { $_.Count -gt 1 }

    $candidateFiles = $sizeGroups | ForEach-Object { $_.Group.FullName }

    $hashResults = @{}
    $count = 0
    foreach ($file in $candidateFiles) {
        if ($MaxResults -gt 0 -and $count -ge $MaxResults) { break }
        try {
            $hash = (Get-FileHash -Path $file -Algorithm SHA256).Hash
            if (-not $hashResults.ContainsKey($hash)) {
                $hashResults[$hash] = @()
            }
            $hashResults[$hash] += $file
            $count++
        }
        catch { }
    }

    $duplicates = $hashResults.GetEnumerator() | Where-Object { $_.Value.Count -gt 1 }
    $wastedBytes = 0
    $groups = @()

    foreach ($dup in $duplicates) {
        $size = (Get-Item $dup.Value[0]).Length
        $waste = $size * ($dup.Value.Count - 1)
        $wastedBytes += $waste
        $groups += [PSCustomObject]@{
            Hash       = $dup.Key
            Size       = $size
            Paths      = $dup.Value
            WastedBytes = $waste
        }
    }

    [PSCustomObject]@{
        Status           = 'Success'
        FilesScanned     = $files.Count
        DuplicateGroupCount = $groups.Count
        DuplicateFiles   = ($groups | ForEach-Object { $_.Paths.Count - 1 } | Measure-Object -Sum).Sum
        WastedBytes      = $wastedBytes
        WastedMB         = [math]::Round($wastedBytes / 1MB, 2)
        Groups           = $groups
        Summary          = "PowerShell fallback: Found $($groups.Count) duplicate groups"
    }
}

function Invoke-PowerShellFileSearch {
    [CmdletBinding()]
    param(
        [string]$Path,
        [string]$Pattern,
        [int64]$MaxResults
    )

    $files = Get-ChildItem -Path $Path -Recurse -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -like $Pattern }

    if ($MaxResults -gt 0) {
        $files = $files | Select-Object -First $MaxResults
    }

    $totalSize = ($files | Measure-Object -Property Length -Sum).Sum

    [PSCustomObject]@{
        Status       = 'Success'
        Pattern      = $Pattern
        FilesScanned = $files.Count
        FilesMatched = $files.Count
        TotalSize    = $totalSize
        Files        = $files | ForEach-Object {
            [PSCustomObject]@{
                Path     = $_.FullName
                Size     = $_.Length
                ReadOnly = $_.IsReadOnly
            }
        }
        Summary      = "PowerShell fallback: Found $($files.Count) files"
    }
}

function Invoke-PowerShellContentSearch {
    [CmdletBinding()]
    param(
        [string]$Path,
        [string]$Pattern,
        [string]$FilePattern,
        [int64]$MaxResults,
        [int]$ContextLines
    )

    $fileFilter = if ($FilePattern) { $FilePattern } else { '*' }

    $files = Get-ChildItem -Path $Path -Recurse -File -Filter $fileFilter -ErrorAction SilentlyContinue

    $matches = @()
    $filesMatched = @{}

    foreach ($file in $files) {
        if ($MaxResults -gt 0 -and $matches.Count -ge $MaxResults) { break }

        try {
            $content = Get-Content -Path $file.FullName -ErrorAction Stop
            $lineNum = 0
            foreach ($line in $content) {
                $lineNum++
                if ($line -match $Pattern) {
                    $filesMatched[$file.FullName] = $true
                    $matches += [PSCustomObject]@{
                        Path       = $file.FullName
                        LineNumber = $lineNum
                        Line       = $line
                        Before     = @()
                        After      = @()
                    }
                    if ($MaxResults -gt 0 -and $matches.Count -ge $MaxResults) { break }
                }
            }
        }
        catch { }
    }

    [PSCustomObject]@{
        Status       = 'Success'
        Pattern      = $Pattern
        FilePattern  = $filePattern
        FilesScanned = $files.Count
        FilesMatched = $filesMatched.Count
        TotalMatches = $matches.Count
        Matches      = $matches
        Summary      = "PowerShell fallback: Found $($matches.Count) matches in $($filesMatched.Count) files"
    }
}
