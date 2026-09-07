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
    } catch {
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
    # Build scratch and extra checkouts. `.pcai/` and `worktrees/` are
    # gitignored, but ripgrep still descends into `.pcai/` because of the
    # `!.pcai/.gitkeep` re-include rule -- and `.pcai/janus-python` (a vendored
    # third-party Python tree) supplied 1662 of 2074 matches, i.e. ~80% of this
    # report was other people's TODOs.
    '-g', '!**/.pcai/**',
    '-g', '!**/worktrees/**',
    '-g', '!**/.venv/**',
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
        $sgArgs = @('scan', '-c', (Join-Path $RepoRoot 'sgconfig.yml'), '--json=stream')
        $sgOutput = & $sgExe.Path @sgArgs 2>$null
        if ($LASTEXITCODE -eq 0 -and $sgOutput) {
            $sgOutput | Set-Content -Path $docStatusJson -Encoding UTF8
            foreach ($line in $sgOutput) {
                if ([string]::IsNullOrWhiteSpace($line)) { continue }
                try {
                    $m = $line | ConvertFrom-Json
                    if ($null -ne $m -and $m.file) {
                        $key = "$($m.file)|$($m.line)|$($m.text)"
                        if (-not $entryIndex.ContainsKey($key)) {
                            $entryIndex[$key] = $true
                            $entries += [PSCustomObject]@{
                                Path  = Convert-ToRepoRelativePath -Path $m.file -RepoRoot $RepoRoot
                                Line  = $m.line
                                Match = $m.text
                            }
                        }
                    }
                } catch {
                    # Skip malformed NDJSON lines
                }
            }
        }
    } catch {
        $sgJson = $null
    }
}

# Capture stdout only, suppress stderr (handles Windows nul device errors)
# Temporarily allow errors since rg may emit errors for inaccessible paths.
#
# $rgOut MUST be initialised before the try: ripgrep is not installed on the
# GitHub-hosted runners, so `& rg` never assigned it, and Set-StrictMode
# -Version Latest then failed the whole doc pipeline with "The variable
# '$rgOut' cannot be retrieved because it has not been set". Like the sg.exe
# path above, ripgrep is an optional accelerator, not a hard dependency.
$rgOut = @()
$rgExe = Get-Command rg -ErrorAction SilentlyContinue
if ($rgExe) {
    $prevEAP = $ErrorActionPreference
    $ErrorActionPreference = 'SilentlyContinue'
    Push-Location $RepoRoot
    try {
        $rgOut = & $rgExe.Path @rgArgs 2>$null
    } finally {
        Pop-Location
        $ErrorActionPreference = $prevEAP
    }
} else {
    Write-Verbose 'ripgrep (rg) not found; relying on the ast-grep scan for marker discovery.'
}
# Process any output regardless of exit code (rg may have partial results)
if ($rgOut) {
    foreach ($line in $rgOut) {
        $parts = $line -split ':', 3
        if ($parts.Count -ge 3) {
            $key = "$($parts[0])|$($parts[1])|$($parts[2].Trim())"
            if (-not $entryIndex.ContainsKey($key)) {
                $entryIndex[$key] = $true
                $entries += [PSCustomObject]@{
                    Path  = Convert-ToRepoRelativePath -Path $parts[0] -RepoRoot $RepoRoot
                    Line  = $parts[1]
                    Match = $parts[2].Trim()
                }
            }
        }
    }
}

# Neither sg.exe nor rg is installed on GitHub-hosted runners. Without a
# fallback the report is generated empty and DOC_STATUS silently claims the
# repo has zero markers -- a check that cannot fail. Scan in-process instead.
if ($entries.Count -eq 0) {
    Write-Host 'No external scanner produced matches; using the built-in PowerShell scan.' -ForegroundColor Yellow

    # File set comes from `git ls-files`, which honours .gitignore exactly the
    # way ripgrep does. A plain Get-ChildItem walk pulled in .pcai/ build
    # artifacts and worktrees/ and inflated DEPRECATED from 176 to 3480.
    $excludeDirs = @('node_modules', 'bin', 'obj', 'target', 'target-ffi',
        'target-ffi-nosccache', 'dist', 'output', 'checkpoints', 'Reports', 'Models')
    $includeExt = @('.ps1', '.psm1', '.psd1', '.rs', '.cs', '.py', '.md', '.toml', '.yml', '.yaml', '.json')

    $candidates = @()
    try {
        $tracked = & git -C $RepoRoot ls-files 2>$null
        if ($LASTEXITCODE -eq 0 -and $tracked) {
            $candidates = $tracked | ForEach-Object { Join-Path $RepoRoot $_ }
        }
    } catch { }
    if (-not $candidates -or $candidates.Count -eq 0) {
        Write-Warning 'git ls-files unavailable; falling back to a filesystem walk (results may include ignored paths).'
        $candidates = (Get-ChildItem -LiteralPath $RepoRoot -Recurse -File -ErrorAction SilentlyContinue).FullName
    }

    foreach ($file in $candidates) {
        $ext = [System.IO.Path]::GetExtension($file)
        if (-not $ext -or ($includeExt -notcontains $ext.ToLowerInvariant())) { continue }
        $leaf = [System.IO.Path]::GetFileName($file)
        if ($leaf -like 'tokenizer*.json') { continue }
        $rel = $file.Substring($RepoRoot.Length).TrimStart('\', '/')
        $segments = $rel -split '[\\/]'
        if ($segments | Where-Object { $excludeDirs -contains $_ }) { continue }
        if (-not (Test-Path -LiteralPath $file)) { continue }

        Select-String -LiteralPath $file -Pattern $markers -ErrorAction SilentlyContinue |
            ForEach-Object {
                $text = $_.Line.Trim()
                $key = "$file|$($_.LineNumber)|$text"
                if (-not $entryIndex.ContainsKey($key)) {
                    $entryIndex[$key] = $true
                    $entries += [PSCustomObject]@{
                        Path  = Convert-ToRepoRelativePath -Path $file -RepoRoot $RepoRoot
                        Line  = $_.LineNumber
                        Match = $text
                    }
                }
            }
    }
}

if ($entries.Count -eq 0) {
    Write-Warning 'DOC_STATUS scan found zero markers across the whole repo. That is almost certainly a scanner failure, not a clean repo.'
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
$null = $md.AppendLine('# DOC_STATUS')
$null = $md.AppendLine('')
$null = $md.AppendLine("Generated: $timestamp")
$null = $md.AppendLine('')
$null = $md.AppendLine('## Counts')
foreach ($c in $counts) {
    $null = $md.AppendLine("- $($c.Name): $($c.Count)")
}
$null = $md.AppendLine('')
$null = $md.AppendLine('## Matches')
foreach ($e in $entries) {
    $null = $md.AppendLine("- $($e.Path):$($e.Line) $($e.Match)")
}

$md.ToString() | Set-Content -Path $docStatusMd -Encoding UTF8

Write-Host "Wrote: $docStatusMd" -ForegroundColor Green
if ($sgJson) {
    Write-Host "Wrote: $docStatusJson" -ForegroundColor Green
}
