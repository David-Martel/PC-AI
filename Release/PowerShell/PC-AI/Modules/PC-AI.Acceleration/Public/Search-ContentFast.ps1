#Requires -Version 7.0
<#
.SYNOPSIS
    Fast content search using ripgrep with parallel fallback

.DESCRIPTION
    Searches file contents using native PCAI search when available,
    then ripgrep (rg), with fallback to parallel Select-String using
    PS7+ ForEach-Object -Parallel.

.PARAMETER Path
    Path to search

.PARAMETER Pattern
    Regex pattern to search for

.PARAMETER LiteralPattern
    Search for literal string (no regex)

.PARAMETER FilePattern
    Glob pattern for files to search (e.g., "*.ps1")

.PARAMETER Context
    Lines of context before/after match

.PARAMETER IgnoreCase
    Case-insensitive search (default)

.PARAMETER WholeWord
    Match whole words only

.PARAMETER Invert
    Show non-matching lines

.EXAMPLE
    Search-ContentFast -Path "C:\Scripts" -Pattern "Get-Process" -FilePattern "*.ps1"
    Searches PowerShell files for Get-Process

.EXAMPLE
    Search-ContentFast -Path "." -LiteralPattern "TODO:" -Context 2
    Finds TODO comments with context

.OUTPUTS
    PSCustomObject[] with search results
#>
function Search-ContentFast {
    [CmdletBinding()]
    [OutputType([PSCustomObject[]])]
    param(
        [Parameter(Mandatory, Position = 0)]
        [string]$Path,

        [Parameter(Position = 1, ParameterSetName = 'Regex')]
        [string]$Pattern,

        [Parameter(ParameterSetName = 'Literal')]
        [string]$LiteralPattern,

        [Parameter()]
        [string[]]$FilePattern,

        [Parameter()]
        [int]$Context = 0,

        [Parameter()]
        [switch]$CaseSensitive,

        [Parameter()]
        [switch]$WholeWord,

        [Parameter()]
        [switch]$Invert,

        [Parameter()]
        [int]$MaxResults = 0,

        [Parameter()]
        [switch]$FilesOnly,

        [Parameter()]
        [int]$ThrottleLimit = [Environment]::ProcessorCount
        ,
        [Parameter()]
        [switch]$NoIgnore
    )

    $searchPattern = if ($LiteralPattern) {
        [regex]::Escape($LiteralPattern)
    }
    else {
        $Pattern
    }

    $rgPath = Get-RustToolPath -ToolName 'rg'
    $useRipgrep = $null -ne $rgPath -and (Test-Path $rgPath)

    $nativeType = ([System.Management.Automation.PSTypeName]'PcaiNative.PcaiCore').Type
    if ($nativeType -and [PcaiNative.PcaiCore]::IsAvailable -and -not $FilesOnly -and -not $Invert) {
        $nativeResults = Search-WithPcaiNativeContent @PSBoundParameters -SearchPattern $searchPattern
        if ($null -ne $nativeResults) {
            return $nativeResults
        }
    }

    if ($useRipgrep) {
        return Search-WithRipgrepAdvanced @PSBoundParameters -SearchPattern $searchPattern -RgPath $rgPath
    }
    else {
        Write-Verbose "ripgrep not available, using parallel Select-String"
        return Search-WithParallelSelectString @PSBoundParameters -SearchPattern $searchPattern
    }
}

<#
.SYNOPSIS
    Search file contents using the native PCAI core engine.

.PARAMETER Path
    Path to search.

.PARAMETER Pattern
    Regex pattern to search for.

.PARAMETER LiteralPattern
    Literal pattern (not used by native path).

.PARAMETER FilePattern
    File glob patterns to include.

.PARAMETER Context
    Number of context lines to include.

.PARAMETER CaseSensitive
    Enable case-sensitive matching.

.PARAMETER WholeWord
    Match whole words only.

.PARAMETER Invert
    Invert matches (not supported in native path).

.PARAMETER MaxResults
    Maximum matches to return.

.PARAMETER FilesOnly
    Return files only (not supported in native path).

.PARAMETER ThrottleLimit
    Throttle limit (not used by native path).

.PARAMETER SearchPattern
    Prepared regex pattern to use.
#>
function Search-WithPcaiNativeContent {
    [CmdletBinding()]
    param(
        [string]$Path,
        [string]$Pattern,
        [string]$LiteralPattern,
        [string[]]$FilePattern,
        [int]$Context,
        [switch]$CaseSensitive,
        [switch]$WholeWord,
        [switch]$Invert,
        [int]$MaxResults,
        [switch]$FilesOnly,
        [int]$ThrottleLimit,
        [string]$SearchPattern
    )

    $nativePattern = $SearchPattern
    if ($WholeWord -and $nativePattern) {
        $nativePattern = "\b(?:$nativePattern)\b"
    }
    if (-not $CaseSensitive) {
        $nativePattern = "(?i)$nativePattern"
    }

    $patterns = if ($FilePattern -and $FilePattern.Count -gt 0) { $FilePattern } else { @($null) }
    $matches = @()

    foreach ($pattern in $patterns) {
        try {
            $remaining = if ($MaxResults -gt 0) { [uint64]($MaxResults - $matches.Count) } else { [uint64]0 }
            if ($MaxResults -gt 0 -and $remaining -le 0) { break }

            $json = [PcaiNative.PcaiCore]::SearchContent(
                $Path,
                $nativePattern,
                $pattern,
                $remaining,
                [uint32]$Context
            )
            if (-not $json) {
                continue
            }

            $result = $json | ConvertFrom-Json

            foreach ($match in ($result.matches | Where-Object { $_ })) {
                $matches += [PSCustomObject]@{
                    Path       = $match.path
                    LineNumber = $match.line_number
                    Line       = $match.line
                    Column     = 0
                    Tool       = 'pcai_native'
                    Before     = if ($match.before) { $match.before -join "`n" } else { '' }
                    After      = if ($match.after) { $match.after -join "`n" } else { '' }
                }

                if ($MaxResults -gt 0 -and $matches.Count -ge $MaxResults) {
                    break
                }
            }
        }
        catch {
            return $null
        }

        if ($MaxResults -gt 0 -and $matches.Count -ge $MaxResults) {
            break
        }
    }

    if ($MaxResults -gt 0) {
        return $matches | Select-Object -First $MaxResults
    }

    return $matches
}

function Search-WithRipgrepAdvanced {
    [CmdletBinding()]
    param(
        [string]$Path,
        [string]$Pattern,
        [string]$LiteralPattern,
        [string[]]$FilePattern,
        [int]$Context,
        [switch]$CaseSensitive,
        [switch]$WholeWord,
        [switch]$Invert,
        [int]$MaxResults,
        [switch]$FilesOnly,
        [int]$ThrottleLimit,
        [string]$SearchPattern,
        [string]$RgPath,
        [switch]$NoIgnore
    )

    $args = @()

    # File patterns
    foreach ($fp in $FilePattern) {
        $args += '-g'
        $args += $fp
    }

    # Case sensitivity
    if (-not $CaseSensitive) {
        $args += '-i'
    }

    # Whole word
    if ($WholeWord) {
        $args += '-w'
    }

    # Invert match
    if ($Invert) {
        $args += '-v'
    }

    # Context
    if ($Context -gt 0) {
        $args += '-C'
        $args += $Context.ToString()
    }

    # Max results
    if ($MaxResults -gt 0) {
        $args += '-m'
        $args += $MaxResults.ToString()
    }

    # Files only mode
    if ($FilesOnly) {
        $args += '-l'
    }
    else {
        $args += '--json'
    }

    if ($NoIgnore) {
        $args += '--no-ignore'
        $args += '--no-ignore-vcs'
        $args += '--no-ignore-parent'
        $args += '--no-ignore-global'
    }

    # Fixed string for literal
    if ($LiteralPattern) {
        $args += '-F'
    }

    $args += $SearchPattern
    $args += $Path

    try {
        $output = & $RgPath @args 2>&1

        if ($FilesOnly) {
            return $output | Where-Object { $_ -and (Test-Path $_ -ErrorAction SilentlyContinue) } |
                ForEach-Object {
                    [PSCustomObject]@{
                        Path = $_
                        Tool = 'ripgrep'
                    }
                }
        }

        $results = @()
        foreach ($line in $output) {
            if ($line -match '^\{') {
                try {
                    $json = $line | ConvertFrom-Json
                    if ($json.type -eq 'match') {
                        $results += [PSCustomObject]@{
                            Path       = $json.data.path.text
                            LineNumber = $json.data.line_number
                            Line       = $json.data.lines.text.TrimEnd()
                            Column     = if ($json.data.submatches) { $json.data.submatches[0].start } else { 0 }
                            Tool       = 'ripgrep'
                        }
                    }
                }
                catch {
                    # Skip malformed JSON
                }
            }
        }

        return $results
    }
    catch {
        Write-Warning "ripgrep failed: $_"
        return @()
    }
}

function Search-WithParallelSelectString {
    [CmdletBinding()]
    param(
        [string]$Path,
        [string]$Pattern,
        [string]$LiteralPattern,
        [string[]]$FilePattern,
        [int]$Context,
        [switch]$CaseSensitive,
        [switch]$WholeWord,
        [switch]$Invert,
        [int]$MaxResults,
        [switch]$FilesOnly,
        [int]$ThrottleLimit,
        [string]$SearchPattern
    )

    # Get files to search
    $params = @{
        Path        = $Path
        Recurse     = $true
        File        = $true
        ErrorAction = 'SilentlyContinue'
    }

    if ($FilePattern) {
        $params.Include = $FilePattern
    }

    $files = Get-ChildItem @params

    if (-not $files) {
        return @()
    }

    $searchParams = @{
        Pattern       = if ($WholeWord -and $SearchPattern) { "\b(?:$SearchPattern)\b" } else { $SearchPattern }
        CaseSensitive = $CaseSensitive.IsPresent
        NotMatch      = $Invert.IsPresent
    }

    if ($Context -gt 0) {
        $searchParams.Context = $Context
    }

    # Use PowerShell 7 parallel processing
    $results = $files | ForEach-Object -Parallel {
        $file = $_
        $params = $using:searchParams
        $maxRes = $using:MaxResults
        $filesOnlyMode = $using:FilesOnly

        try {
            $matches = Select-String @params -Path $file.FullName -ErrorAction SilentlyContinue

            if ($filesOnlyMode -and $matches) {
                return [PSCustomObject]@{
                    Path = $file.FullName
                    Tool = 'Select-String'
                }
            }

            $fileResults = @()
            foreach ($match in $matches) {
                $fileResults += [PSCustomObject]@{
                    Path       = $match.Path
                    LineNumber = $match.LineNumber
                    Line       = $match.Line
                    Column     = 0
                    Tool       = 'Select-String'
                }

                if ($maxRes -gt 0 -and $fileResults.Count -ge $maxRes) {
                    break
                }
            }

            return $fileResults
        }
        catch {
            return $null
        }
    } -ThrottleLimit $ThrottleLimit | Where-Object { $_ }

    # Flatten and limit results
    $flatResults = @($results | ForEach-Object { $_ })

    if ($MaxResults -gt 0) {
        return $flatResults | Select-Object -First $MaxResults
    }

    return $flatResults
}
