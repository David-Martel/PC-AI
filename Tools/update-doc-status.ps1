#Requires -Version 5.1

<#
.SYNOPSIS
  Generate documentation/status reports using ast-grep (sg) with rg fallback.

.DESCRIPTION
  Scans the repo for TODO/FIXME/INCOMPLETE/@status/DEPRECATED markers and writes:
  - Reports\DOC_STATUS.json (raw sg json when available)
  - Reports\DOC_STATUS.md (human summary + matches)
#>

[CmdletBinding()]
param(
    [Parameter()]
    [string]$RepoRoot
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Convert-ToRepoRelativePath {
    param(
        [Parameter(Mandatory)]
        [string]$Path,
        [Parameter(Mandatory)]
        [string]$RepoRoot
    )

    if ([string]::IsNullOrWhiteSpace($Path)) {
        return $Path
    }

    $repoFull = [System.IO.Path]::GetFullPath($RepoRoot).TrimEnd('\', '/')
    $candidate = $Path
    if (-not [System.IO.Path]::IsPathRooted($candidate)) {
        $candidate = Join-Path $RepoRoot $candidate
    }

    try {
        $full = [System.IO.Path]::GetFullPath($candidate)
        if ($full.StartsWith($repoFull, [System.StringComparison]::OrdinalIgnoreCase)) {
            return $full.Substring($repoFull.Length).TrimStart('\', '/')
        }
    }
    catch {
        # Fall back to original path if normalization fails.
    }

    return $Path.TrimStart('.', '\', '/')
}

$scriptRoot = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
if (-not $RepoRoot) {
    $RepoRoot = Split-Path -Parent $scriptRoot
}

$reportDir = Join-Path $RepoRoot 'Reports'
if (-not (Test-Path $reportDir)) {
    New-Item -ItemType Directory -Path $reportDir -Force | Out-Null
}

$timestamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
$docStatusJson = Join-Path $reportDir 'DOC_STATUS.json'
$docStatusMd = Join-Path $reportDir 'DOC_STATUS.md'

$markers = 'TODO|FIXME|INCOMPLETE|@status|DEPRECATED'
# NOTE: Path must come AFTER all glob patterns for ripgrep
$rgArgs = @(
    '-n', '-S', $markers,
    '-g', '!**/.git/**',
    '-g', '!**/node_modules/**',
    '-g', '!**/bin/**',
    '-g', '!**/obj/**',
    '-g', '!**/target/**',
    '-g', '!**/target-ffi/**',
    '-g', '!**/target-ffi-nosccache/**',
    '-g', '!**/dist/**',
    '-g', '!**/output/**',
    '-g', '!**/checkpoints/**',
    # CRITICAL: Prevent self-referential scanning
    '-g', '!**/Reports/**',
    '-g', '!**/*.jsonl',
    '-g', '!**/Models/**/tokenizer*.json',
    '-g', '!**/.claude/context/**',
    '.'
)

$entries = @()
$entryIndex = @{}
$sgJson = $null

$sgExe = Get-Command sg.exe -ErrorAction SilentlyContinue
if ($sgExe) {
    try {
        $sgArgs = @('scan', '-c', (Join-Path $RepoRoot 'sgconfig.yml'), '--json=compact')
        $sgOutput = & $sgExe.Path @sgArgs 2>$null
        if ($LASTEXITCODE -eq 0 -and $sgOutput) {
            $sgJson = $sgOutput | ConvertFrom-Json
            $sgOutput | Set-Content -Path $docStatusJson -Encoding UTF8
            $sgMatches = $null
            if ($sgJson -is [System.Collections.IEnumerable] -and -not ($sgJson -is [string]) -and -not ($sgJson.PSObject.Properties.Name -contains 'matches')) {
                $sgMatches = $sgJson
            } elseif ($sgJson.matches) {
                $sgMatches = $sgJson.matches
            }
            if ($sgMatches) {
                foreach ($m in $sgMatches) {
                    $key = "$($m.file)|$($m.line)|$($m.text)"
                    if (-not $entryIndex.ContainsKey($key)) {
                        $entryIndex[$key] = $true
                        $entries += [PSCustomObject]@{
                            Path = Convert-ToRepoRelativePath -Path $m.file -RepoRoot $RepoRoot
                            Line = $m.line
                            Match = $m.text
                        }
                    }
                }
            }
        }
    }
    catch {
        $sgJson = $null
    }
}

# Capture stdout only, suppress stderr (handles Windows nul device errors)
# Temporarily allow errors since rg may emit errors for inaccessible paths
$prevEAP = $ErrorActionPreference
$ErrorActionPreference = 'SilentlyContinue'
Push-Location $RepoRoot
try {
    $rgOut = & rg @rgArgs 2>$null
}
finally {
    Pop-Location
}
$ErrorActionPreference = $prevEAP
# Process any output regardless of exit code (rg may have partial results)
if ($rgOut) {
    foreach ($line in $rgOut) {
        $parts = $line -split ':', 3
        if ($parts.Count -ge 3) {
            $key = "$($parts[0])|$($parts[1])|$($parts[2].Trim())"
            if (-not $entryIndex.ContainsKey($key)) {
                $entryIndex[$key] = $true
                $entries += [PSCustomObject]@{
                    Path = Convert-ToRepoRelativePath -Path $parts[0] -RepoRoot $RepoRoot
                    Line = $parts[1]
                    Match = $parts[2].Trim()
                }
            }
        }
    }
}

$counts = $entries | Group-Object -Property {
    if ($_.Match -match 'TODO') { 'TODO' }
    elseif ($_.Match -match 'FIXME') { 'FIXME' }
    elseif ($_.Match -match 'INCOMPLETE') { 'INCOMPLETE' }
    elseif ($_.Match -match '@status') { '@status' }
    elseif ($_.Match -match 'DEPRECATED') { 'DEPRECATED' }
    else { 'Other' }
}

$md = New-Object System.Text.StringBuilder
$null = $md.AppendLine("# DOC_STATUS")
$null = $md.AppendLine("")
$null = $md.AppendLine("Generated: $timestamp")
$null = $md.AppendLine("")
$null = $md.AppendLine("## Counts")
foreach ($c in $counts) {
    $null = $md.AppendLine("- $($c.Name): $($c.Count)")
}
$null = $md.AppendLine("")
$null = $md.AppendLine("## Matches")
foreach ($e in $entries) {
    $null = $md.AppendLine("- $($e.Path):$($e.Line) $($e.Match)")
}

$md.ToString() | Set-Content -Path $docStatusMd -Encoding UTF8

Write-Host "Wrote: $docStatusMd" -ForegroundColor Green
if ($sgJson) {
    Write-Host "Wrote: $docStatusJson" -ForegroundColor Green
}
