#Requires -Version 5.1
<#
.SYNOPSIS
    Fast log file searching using ripgrep

.DESCRIPTION
    Searches log files using the native PCAI log search when available,
    with fallback to ripgrep (rg) or Select-String. Native and ripgrep
    are significantly faster than PowerShell's Select-String for large files.

.PARAMETER Path
    Path to search (file or directory)

.PARAMETER Pattern
    Regular expression pattern to search for

.PARAMETER Include
    File patterns to include (e.g., "*.log", "*.txt")

.PARAMETER Context
    Number of context lines before and after match

.PARAMETER CaseSensitive
    Enable case-sensitive matching

.PARAMETER MaxCount
    Maximum matches per file

.EXAMPLE
    Search-LogsFast -Path "C:\Windows\Logs" -Pattern "error|warning" -Include "*.log"
    Searches all log files for errors or warnings

.EXAMPLE
    Search-LogsFast -Path "C:\Logs\app.log" -Pattern "Exception" -Context 3
    Shows 3 lines of context around each match

.OUTPUTS
    PSCustomObject[] with match information
#>
function Search-LogsFast {
    [CmdletBinding()]
    [OutputType([PSCustomObject[]])]
    param(
        [Parameter(Mandatory, Position = 0)]
        [string]$Path,

        [Parameter(Mandatory, Position = 1)]
        [string]$Pattern,

        [Parameter()]
        [string[]]$Include = @('*.log', '*.txt', '*.evtx'),

        [Parameter()]
        [int]$Context = 0,

        [Parameter()]
        [switch]$CaseSensitive,

        [Parameter()]
        [int]$MaxCount = 0,

        [Parameter()]
        [switch]$CountOnly
    )

    $rgPath = Get-RustToolPath -ToolName 'rg'
    $useRipgrep = $null -ne $rgPath -and (Test-Path $rgPath)

    $nativeType = ([System.Management.Automation.PSTypeName]'PcaiNative.SystemModule').Type
    if ($nativeType -and [PcaiNative.SystemModule]::IsAvailable) {
        $nativeResults = Search-WithPcaiNativeLogs -Path $Path -Pattern $Pattern -Include $Include `
            -Context $Context -CaseSensitive:$CaseSensitive -MaxCount $MaxCount -CountOnly:$CountOnly
        if ($null -ne $nativeResults) {
            return $nativeResults
        }
    }

    if ($useRipgrep) {
        return Search-WithRipgrep -Path $Path -Pattern $Pattern -Include $Include `
            -Context $Context -CaseSensitive:$CaseSensitive -MaxCount $MaxCount -CountOnly:$CountOnly -RgPath $rgPath
    }
    else {
        Write-Verbose "ripgrep not available, using Select-String fallback"
        return Search-WithSelectString -Path $Path -Pattern $Pattern -Include $Include `
            -Context $Context -CaseSensitive:$CaseSensitive -MaxCount $MaxCount -CountOnly:$CountOnly
    }
}

<#
.SYNOPSIS
    Search log files using the native PCAI system module.

.PARAMETER Path
    Path to search (file or directory).

.PARAMETER Pattern
    Regex pattern to search for.

.PARAMETER Include
    File patterns to include.

.PARAMETER Context
    Number of context lines to include.

.PARAMETER CaseSensitive
    Enable case-sensitive matching.

.PARAMETER MaxCount
    Maximum matches per file.

.PARAMETER CountOnly
    Return counts only.
#>
function Search-WithPcaiNativeLogs {
    [CmdletBinding()]
    param(
        [string]$Path,
        [string]$Pattern,
        [string[]]$Include,
        [int]$Context,
        [switch]$CaseSensitive,
        [int]$MaxCount,
        [switch]$CountOnly
    )

    $patterns = if ($Include -and $Include.Count -gt 0) { $Include } else { @($null) }
    $matches = @()
    $total = 0

    foreach ($pattern in $patterns) {
        try {
            $json = [PcaiNative.SystemModule]::SearchLogsJson(
                $Path,
                $Pattern,
                $pattern,
                $CaseSensitive.IsPresent,
                [uint32]$Context,
                [uint32]$MaxCount
            )
            if (-not $json) {
                continue
            }

            $result = $json | ConvertFrom-Json

            if ($CountOnly) {
                $total += [int]$result.total_matches
                continue
            }

            foreach ($fileResult in ($result.results | Where-Object { $_ })) {
                foreach ($match in ($fileResult.matches | Where-Object { $_ })) {
                    $matches += [PSCustomObject]@{
                        Path          = $match.file_path
                        LineNumber    = $match.line_number
                        Line          = $match.line_content
                        ContextBefore = if ($match.context_before) { $match.context_before -join "`n" } else { '' }
                        ContextAfter  = if ($match.context_after) { $match.context_after -join "`n" } else { '' }
                        Tool          = 'pcai_native'
                    }

                    if ($MaxCount -gt 0 -and $matches.Count -ge $MaxCount) {
                        break
                    }
                }

                if ($MaxCount -gt 0 -and $matches.Count -ge $MaxCount) {
                    break
                }
            }
        }
        catch {
            return $null
        }

        if ($MaxCount -gt 0 -and $matches.Count -ge $MaxCount) {
            break
        }
    }

    if ($CountOnly) {
        return [PSCustomObject]@{
            Path    = $Path
            Pattern = $Pattern
            Count   = $total
            Tool    = 'pcai_native'
        }
    }

    if ($MaxCount -gt 0) {
        return $matches | Select-Object -First $MaxCount
    }

    return $matches
}

function Search-WithRipgrep {
    [CmdletBinding()]
    param(
        [string]$Path,
        [string]$Pattern,
        [string[]]$Include,
        [int]$Context,
        [switch]$CaseSensitive,
        [int]$MaxCount,
        [switch]$CountOnly,
        [string]$RgPath
    )

    $args = @()

    # Build glob patterns
    foreach ($inc in $Include) {
        $args += '-g'
        $args += $inc
    }

    # Options
    if (-not $CaseSensitive) {
        $args += '-i'
    }

    if ($Context -gt 0) {
        $args += '-C'
        $args += $Context.ToString()
    }

    if ($MaxCount -gt 0) {
        $args += '-m'
        $args += $MaxCount.ToString()
    }

    if ($CountOnly) {
        $args += '-c'
    }

    # JSON output for structured results
    if (-not $CountOnly) {
        $args += '--json'
    }

    $args += $Pattern
    $args += $Path

    try {
        $output = & $RgPath @args 2>&1

        if ($CountOnly) {
            return [PSCustomObject]@{
                Path    = $Path
                Pattern = $Pattern
                Count   = [int]($output | Measure-Object -Sum | Select-Object -ExpandProperty Sum)
                Tool    = 'ripgrep'
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
        Write-Warning "ripgrep search failed: $_"
        return @()
    }
}

function Search-WithSelectString {
    [CmdletBinding()]
    param(
        [string]$Path,
        [string]$Pattern,
        [string[]]$Include,
        [int]$Context,
        [switch]$CaseSensitive,
        [int]$MaxCount,
        [switch]$CountOnly
    )

    $files = if (Test-Path $Path -PathType Container) {
        Get-ChildItem -Path $Path -Include $Include -Recurse -File -ErrorAction SilentlyContinue
    }
    else {
        Get-Item $Path -ErrorAction SilentlyContinue
    }

    if (-not $files) {
        return @()
    }

    $selectParams = @{
        Pattern       = $Pattern
        CaseSensitive = $CaseSensitive
    }

    if ($Context -gt 0) {
        $selectParams.Context = $Context
    }

    $results = @()
    $totalCount = 0

    foreach ($file in $files) {
        try {
            $matches = Select-String @selectParams -Path $file.FullName -ErrorAction SilentlyContinue

            if ($CountOnly) {
                $totalCount += ($matches | Measure-Object).Count
            }
            else {
                foreach ($match in $matches) {
                    if ($MaxCount -gt 0 -and $results.Count -ge $MaxCount) {
                        break
                    }

                    $results += [PSCustomObject]@{
                        Path       = $match.Path
                        LineNumber = $match.LineNumber
                        Line       = $match.Line
                        Tool       = 'Select-String'
                    }
                }
            }
        }
        catch {
            # Skip inaccessible files
        }

        if ($MaxCount -gt 0 -and $results.Count -ge $MaxCount) {
            break
        }
    }

    if ($CountOnly) {
        return [PSCustomObject]@{
            Path    = $Path
            Pattern = $Pattern
            Count   = $totalCount
            Tool    = 'Select-String'
        }
    }

    return $results
}
