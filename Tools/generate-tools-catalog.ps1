#Requires -Version 5.1
<#+
.SYNOPSIS
  Generate a catalog of PowerShell helper scripts under Tools/.

.DESCRIPTION
  Scans Tools/*.ps1 for comment-based help and emits a Markdown + JSON catalog
  with synopsis/description for LLM-friendly documentation.
#>

[CmdletBinding()]
param(
    [string]$ToolsPath,
    [string]$OutputPath,
    [string]$JsonPath
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
if (-not $ToolsPath) { $ToolsPath = Join-Path $repoRoot 'Tools' }
if (-not $OutputPath) { $OutputPath = Join-Path $repoRoot 'Reports\TOOLS_CATALOG.md' }
if (-not $JsonPath) { $JsonPath = Join-Path $repoRoot 'Reports\TOOLS_CATALOG.json' }

if (-not (Test-Path $ToolsPath)) {
    throw "Tools path not found: $ToolsPath"
}

function Get-HelpBlock {
    param([string]$Text)
    $m = [regex]::Match($Text, '<#([\s\S]*?)#>')
    if ($m.Success) { return $m.Groups[1].Value }
    return $null
}

function Parse-HelpSection {
    param(
        [string]$HelpText,
        [string]$Section
    )

    if (-not $HelpText) { return $null }
    $lines = $HelpText -split "`r?`n"
    $sectionMarker = ".$Section"
    $capture = $false
    $buffer = New-Object System.Collections.Generic.List[string]

    foreach ($line in $lines) {
        if ($line.Trim().StartsWith('.')) {
            if ($capture) { break }
        }
        if ($line.Trim().Equals($sectionMarker, [System.StringComparison]::OrdinalIgnoreCase)) {
            $capture = $true
            continue
        }
        if ($capture) {
            if ($line.Trim() -ne '') {
                $buffer.Add($line.Trim())
            }
        }
    }

    if ($buffer.Count -eq 0) { return $null }
    return ($buffer -join ' ')
}

$entries = @()
Get-ChildItem -Path $ToolsPath -Filter '*.ps1' -File | Sort-Object Name | ForEach-Object {
    $text = Get-Content -Path $_.FullName -Raw
    $help = Get-HelpBlock -Text $text
    $synopsis = Parse-HelpSection -HelpText $help -Section 'SYNOPSIS'
    $description = Parse-HelpSection -HelpText $help -Section 'DESCRIPTION'

    $entries += [PSCustomObject]@{
        Name = $_.Name
        Path = $_.FullName
        Synopsis = $synopsis
        Description = $description
    }
}

# Write JSON
$entries | ConvertTo-Json -Depth 5 | Set-Content -Path $JsonPath -Encoding UTF8

# Write Markdown
$md = @()
$md += '# Tools Catalog'
$md += ''
$md += "Generated: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
$md += ''
$md += '| Script | Synopsis |'
$md += '|--------|----------|'
foreach ($entry in $entries) {
    $syn = if ($entry.Synopsis) { $entry.Synopsis } else { '' }
    $md += "| $($entry.Name) | $syn |"
}
$md += ''
$md += '## Details'
$md += ''
foreach ($entry in $entries) {
    $md += "### $($entry.Name)"
    $md += "Path: ``$($entry.Path)``"
    if ($entry.Synopsis) { $md += "Synopsis: $($entry.Synopsis)" }
    if ($entry.Description) { $md += "Description: $($entry.Description)" }
    $md += ''
}

$md | Set-Content -Path $OutputPath -Encoding UTF8

Write-Host "Tools catalog written: $OutputPath" -ForegroundColor Green
