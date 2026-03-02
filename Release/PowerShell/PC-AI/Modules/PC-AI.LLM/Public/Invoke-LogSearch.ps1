#Requires -Version 5.1

function Invoke-LogSearch {
    <#
    .SYNOPSIS
        Search log files for a regex pattern (native-first).

    .DESCRIPTION
        Uses the native pcai_system.dll log search when available. Falls back to
        PowerShell Select-String when native search is unavailable.

    .PARAMETER Pattern
        Regex pattern to search for.

    .PARAMETER RootPath
        Root directory to search (default: %SystemRoot%\System32\LogFiles).

    .PARAMETER FilePattern
        File glob pattern (default: *.log).

    .PARAMETER CaseSensitive
        Enable case-sensitive matching.

    .PARAMETER ContextLines
        Number of context lines to include before/after matches.

    .PARAMETER MaxMatches
        Maximum number of matches to return.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory, Position = 0)]
        [string]$Pattern,

        [Parameter()]
        [string]$RootPath = (Join-Path $env:SystemRoot 'System32\LogFiles'),

        [Parameter()]
        [string]$FilePattern = '*.log',

        [Parameter()]
        [switch]$CaseSensitive,

        [Parameter()]
        [ValidateRange(0, 50)]
        [int]$ContextLines = 2,

        [Parameter()]
        [ValidateRange(1, 10000)]
        [int]$MaxMatches = 1000
    )

    $nativeSystemType = ([System.Management.Automation.PSTypeName]'PcaiNative.SystemModule').Type
    if ($nativeSystemType -and [PcaiNative.SystemModule]::IsAvailable) {
        try {
            $json = [PcaiNative.SystemModule]::SearchLogsJson(
                $RootPath,
                $Pattern,
                $FilePattern,
                $CaseSensitive.IsPresent,
                [uint32]$ContextLines,
                [uint32]$MaxMatches
            )
            if ($json) {
                return $json
            }
        } catch {
            Write-Verbose "Native log search failed: $_"
        }
    }

    $matches = @()
    $filesSearched = 0
    $filesWithMatches = 0

    try {
        $files = Get-ChildItem -Path $RootPath -Filter $FilePattern -File -Recurse -ErrorAction SilentlyContinue
        $filesSearched = $files.Count

        foreach ($file in $files) {
            $hit = Select-String -Path $file.FullName -Pattern $Pattern -CaseSensitive:$CaseSensitive -Context $ContextLines -ErrorAction SilentlyContinue
            if ($hit) {
                $filesWithMatches++
                foreach ($entry in $hit) {
                    if ($matches.Count -ge $MaxMatches) { break }
                    $matches += [PSCustomObject]@{
                        File    = $file.FullName
                        Line    = $entry.LineNumber
                        Match   = $entry.Line.Trim()
                        Before  = ($entry.Context.PreContext -join "`n")
                        After   = ($entry.Context.PostContext -join "`n")
                    }
                }
            }
            if ($matches.Count -ge $MaxMatches) { break }
        }
    } catch {
        return (@{
            Status  = 'Error'
            Error   = $_.Exception.Message
            Pattern = $Pattern
            Root    = $RootPath
        } | ConvertTo-Json -Depth 6)
    }

    $status = if ($matches.Count -gt 0) { 'Success' } else { 'NoMatches' }

    return (@{
        Status           = $status
        Pattern          = $Pattern
        RootPath         = $RootPath
        FilePattern      = $FilePattern
        CaseSensitive    = $CaseSensitive.IsPresent
        ContextLines     = $ContextLines
        MaxMatches       = $MaxMatches
        FilesSearched    = $filesSearched
        FilesWithMatches = $filesWithMatches
        TotalMatches     = $matches.Count
        Matches          = $matches
        Source           = 'PowerShell'
    } | ConvertTo-Json -Depth 6)
}
