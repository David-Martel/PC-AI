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

. "$repoRoot\Modules\PC-AI.Common\Public\Get-ScriptMetadata.ps1"

$entries = @()
Get-ChildItem -Path $ToolsPath -Filter '*.ps1' -File | Sort-Object Name | ForEach-Object {
    $meta = Get-ScriptMetadata -Path $_.FullName

    $entries += [PSCustomObject]@{
        Name        = $_.Name
        Path        = $_.FullName
        Synopsis    = $meta.Synopsis
        Description = $meta.Description
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
