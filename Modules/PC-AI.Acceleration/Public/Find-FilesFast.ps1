#Requires -Version 5.1
<#
.SYNOPSIS
    Fast file finding using fd

.DESCRIPTION
    Finds files using fd (fast find alternative) when available,
    with fallback to Get-ChildItem. fd is typically 5-10x faster
    than PowerShell's Get-ChildItem for large directory trees.

.PARAMETER Path
    Root path to search from

.PARAMETER Pattern
    Filename pattern (supports regex)

.PARAMETER Extension
    File extension(s) to filter by

.PARAMETER Type
    Filter by type: file, directory, symlink

.PARAMETER MaxDepth
    Maximum directory depth to search

.PARAMETER Hidden
    Include hidden files

.PARAMETER Exclude
    Patterns to exclude

.EXAMPLE
    Find-FilesFast -Path "C:\Projects" -Extension "ps1"
    Finds all PowerShell files

.EXAMPLE
    Find-FilesFast -Path "D:\Data" -Pattern "backup" -Type file
    Finds files matching "backup" pattern

.OUTPUTS
    FileInfo[] or PSCustomObject[] with file information
#>
function Find-FilesFast {
    [CmdletBinding()]
    [OutputType([System.IO.FileInfo[]])]
    param(
        [Parameter(Position = 0)]
        [string]$Path = '.',

        [Parameter(Position = 1)]
        [string]$Pattern,

        [Parameter()]
        [string[]]$Extension,

        [Parameter()]
        [ValidateSet('file', 'directory', 'symlink', 'f', 'd', 'l')]
        [string]$Type,

        [Parameter()]
        [int]$MaxDepth = 0,

        [Parameter()]
        [switch]$Hidden,

        [Parameter()]
        [string[]]$Exclude,

        [Parameter()]
        [switch]$FullPath,

        [Parameter()]
        [switch]$NoIgnore,

        [Parameter()]
        [switch]$PreferNative
    )

    $resolvedPath = Resolve-PcaiPath -Path $Path
    $cacheKey = Get-PcaiCacheKey -Category 'find-files' -Parameters @{
        path         = $resolvedPath
        pattern      = $Pattern
        extension    = (@($Extension) -join ',')
        type         = $Type
        depth        = $MaxDepth
        hidden       = [bool]$Hidden
        exclude      = (@($Exclude) -join ',')
        fullPath     = [bool]$FullPath
        noIgnore     = [bool]$NoIgnore
        preferNative = [bool]$PreferNative
    }

    $cached = Get-PcaiCachedValue -Key $cacheKey -TtlSeconds 15
    if ($null -ne $cached) {
        return @($cached)
    }

    $nativeAvailable = $false
    if ($PreferNative -and ($Pattern -or $Extension)) {
        $nativeProbe = Get-Command -Name 'Test-PcaiNativeAvailable' -ErrorAction SilentlyContinue
        if ($nativeProbe) {
            try {
                $nativeAvailable = Test-PcaiNativeAvailable
            }
            catch {
                $nativeAvailable = $false
            }
        }
    }

    if ($nativeAvailable) {
        $results = @(Find-WithPcaiNative @PSBoundParameters)
        return @(Set-PcaiCachedValue -Key $cacheKey -Value $results)
    }

    $fdPath = Get-RustToolPath -ToolName 'fd'
    $useFd = $null -ne $fdPath -and (Test-Path $fdPath)

    $forwardParams = @{}
    foreach ($key in $PSBoundParameters.Keys) {
        if ($key -ne 'PreferNative') {
            $forwardParams[$key] = $PSBoundParameters[$key]
        }
    }

    if ($useFd) {
        $results = @(Find-WithFd @forwardParams -FdPath $fdPath)
        return @(Set-PcaiCachedValue -Key $cacheKey -Value $results)
    }
    else {
        Write-Verbose "fd not available, using Get-ChildItem fallback"
        $results = @(Find-WithGetChildItem @forwardParams)
        return @(Set-PcaiCachedValue -Key $cacheKey -Value $results)
    }
}

function Find-WithFd {
    [CmdletBinding()]
    param(
        [string]$Path,
        [string]$Pattern,
        [string[]]$Extension,
        [string]$Type,
        [int]$MaxDepth,
        [switch]$Hidden,
        [string[]]$Exclude,
        [switch]$FullPath,
        [switch]$NoIgnore,
        [string]$FdPath
    )

    $args = @()

    # Pattern (first positional argument) - use '.' for match-all if no pattern
    if ($Pattern) {
        # Convert glob patterns to regex if needed
        if ($Pattern -match '^\*\.(.+)$') {
            # *.ext pattern - use extension flag instead
            $args += '.'  # Match all
            $args += '-e'
            $args += $Matches[1]
        }
        else {
            $args += $Pattern
        }
    }
    else {
        $args += '.'  # Match all files
    }

    # Extensions (after pattern)
    foreach ($ext in $Extension) {
        $args += '-e'
        $args += $ext.TrimStart('*').TrimStart('.')
    }

    # Type
    if ($Type) {
        $args += '-t'
        switch ($Type) {
            'file'      { $args += 'f' }
            'directory' { $args += 'd' }
            'symlink'   { $args += 'l' }
            default     { $args += $Type }
        }
    }

    # Max depth
    if ($MaxDepth -gt 0) {
        $args += '-d'
        $args += $MaxDepth.ToString()
    }

    # Hidden files
    if ($Hidden) {
        $args += '-H'
    }

    if ($NoIgnore) {
        $args += '--no-ignore'
        $args += '--no-ignore-vcs'
        $args += '--no-ignore-parent'
    }

    # Exclusions
    foreach ($exc in $Exclude) {
        $args += '-E'
        $args += $exc
    }

    # Absolute paths (always use for reliable path resolution)
    $args += '-a'

    # Path (last positional argument)
    $args += $Path

    try {
        $output = & $FdPath @args 2>$null

        $results = [System.Collections.Generic.List[object]]::new()
        foreach ($line in $output) {
            if ($line) {
                $item = New-PcaiPathItem -Path $line -Type $Type
                if ($item) {
                    $results.Add($item)
                }
            }
        }
        return @($results)
    }
    catch {
        Write-Warning "fd search failed: $_"
        return @()
    }
}

function Find-WithPcaiNative {
    [CmdletBinding()]
    param(
        [string]$Path,
        [string]$Pattern,
        [string[]]$Extension,
        [string]$Type,
        [int]$MaxDepth,
        [switch]$Hidden,
        [string[]]$Exclude,
        [switch]$FullPath,
        [switch]$NoIgnore,
        [switch]$PreferNative
    )

    $resolvedPath = Resolve-PcaiPath -Path $Path

    if (-not $resolvedPath) {
        return @()
    }

    $patterns = @()
    if ($Extension) {
        $exts = $Extension | ForEach-Object { $_.TrimStart('*').TrimStart('.') } | Where-Object { $_ }
        foreach ($ext in $exts) {
            $patterns += "*.$ext"
        }
    }
    elseif ($Pattern) {
        $patterns += $Pattern
    }

    if ($patterns.Count -eq 0) {
        return @()
    }

    try {
        $uniquePaths = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
        foreach ($pat in $patterns) {
            $nativeResult = Invoke-PcaiNativeFileSearch -Pattern $pat -Path $resolvedPath -MaxResults 0
            if ($nativeResult -and $nativeResult.Files) {
                foreach ($file in @($nativeResult.Files)) {
                    if ($file.Path) {
                        $null = $uniquePaths.Add([string]$file.Path)
                    }
                }
            }
        }
        if ($uniquePaths.Count -eq 0) {
            return @()
        }

        if ($Type -and $Type -notin @('file', 'f')) {
            return @()
        }

        $results = [System.Collections.Generic.List[object]]::new()
        foreach ($path in $uniquePaths) {
            $excluded = $false
            foreach ($exc in $Exclude) {
                if ($path -match [regex]::Escape($exc)) {
                    $excluded = $true
                    break
                }
            }
            if ($excluded) {
                continue
            }

            if ($MaxDepth -gt 0 -and (Get-PcaiRelativeDepth -BasePath $resolvedPath -CandidatePath $path) -gt $MaxDepth) {
                continue
            }

            $item = New-PcaiPathItem -Path $path -Type 'file'
            if ($item) {
                $results.Add($item)
            }
        }

        return @($results)
    }
    catch {
        Write-Warning "PCAI native file search failed: $_"
        return @()
    }
}

function Find-WithGetChildItem {
    [CmdletBinding()]
    param(
        [string]$Path,
        [string]$Pattern,
        [string[]]$Extension,
        [string]$Type,
        [int]$MaxDepth,
        [switch]$Hidden,
        [string[]]$Exclude,
        [switch]$FullPath,
        [switch]$NoIgnore
    )

    $params = @{
        Path    = $Path
        Recurse = $true
        ErrorAction = 'SilentlyContinue'
    }

    if ($MaxDepth -gt 0) {
        $params.Depth = $MaxDepth
    }

    if ($Hidden) {
        $params.Force = $true
    }

    # Get all items
    $items = Get-ChildItem @params

    # Filter by type
    if ($Type) {
        $items = switch ($Type) {
            { $_ -in 'file', 'f' }      { $items | Where-Object { -not $_.PSIsContainer } }
            { $_ -in 'directory', 'd' } { $items | Where-Object { $_.PSIsContainer } }
            default { $items }
        }
    }

    # Filter by pattern
    if ($Pattern) {
        $items = $items | Where-Object { $_.Name -match $Pattern }
    }

    # Filter by extension
    if ($Extension) {
        $extPatterns = $Extension | ForEach-Object { ".$($_.TrimStart('.'))" }
        $items = $items | Where-Object { $_.Extension -in $extPatterns }
    }

    # Exclude patterns
    foreach ($exc in $Exclude) {
        $items = $items | Where-Object { $_.FullName -notmatch [regex]::Escape($exc) }
    }

    return $items
}
