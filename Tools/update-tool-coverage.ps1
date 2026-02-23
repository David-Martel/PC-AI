#Requires -Version 5.1

<#+
.SYNOPSIS
  Analyze tool schema coverage against PC_AI tool implementations.

.DESCRIPTION
  Loads Config\pcai-tools.json and compares tool names with the tool-mapping
  in Invoke-FunctionGemmaReAct.ps1. Uses ast-grep (sg) if available, with rg fallback.
  Writes Reports\TOOL_SCHEMA_REPORT.json and Reports\TOOL_SCHEMA_REPORT.md.
#>

[CmdletBinding()]
param(
    [Parameter()]
    [string]$RepoRoot
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$scriptRoot = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
if (-not $RepoRoot) {
    $RepoRoot = Split-Path -Parent $scriptRoot
}

$reportDir = Join-Path $RepoRoot 'Reports'
if (-not (Test-Path $reportDir)) {
    New-Item -ItemType Directory -Path $reportDir -Force | Out-Null
}

$toolSchemaPath = Join-Path $RepoRoot 'Config\pcai-tools.json'
$toolMappingPath = Join-Path $RepoRoot 'Modules\PC-AI.LLM\Public\Invoke-FunctionGemmaReAct.ps1'

if (-not (Test-Path $toolSchemaPath)) { throw "Missing tool schema: $toolSchemaPath" }
if (-not (Test-Path $toolMappingPath)) { throw "Missing tool mapping: $toolMappingPath" }

$toolData = Get-Content -Path $toolSchemaPath -Raw -Encoding UTF8 | ConvertFrom-Json
$schemaTools = @($toolData.tools | ForEach-Object { $_.function.name })

$mappedTools = @()
$sgExe = Get-Command sg.exe -ErrorAction SilentlyContinue
if ($sgExe) {
    try {
        $sgArgs = @('scan', '-c', (Join-Path $RepoRoot 'sgconfig.yml'), '--json=stream')
        $sgOutput = & $sgExe.Path @sgArgs 2>$null
        if ($LASTEXITCODE -eq 0 -and $sgOutput) {
            foreach ($line in $sgOutput) {
                if ([string]::IsNullOrWhiteSpace($line)) { continue }
                try {
                    $m = $line | ConvertFrom-Json
                    if ($null -ne $m -and $m.file -like '*Invoke-FunctionGemmaReAct.ps1' -and $m.text -match 'pcai_') {
                        if ($m.text -match "'(?<name>pcai_[^']+)'" ) {
                            $mappedTools += $Matches['name']
                        }
                    }
                } catch {
                    # Skip malformed lines
                }
            }
        }
    } catch {
        $mappedTools = @()
    }
}

if ($mappedTools.Count -eq 0) {
    $rgOut = & rg -n "'pcai_[^']+'" $toolMappingPath
    if ($LASTEXITCODE -eq 0 -and $rgOut) {
        foreach ($line in $rgOut) {
            if ($line -match "'(?<name>pcai_[^']+)'" ) {
                $mappedTools += $Matches['name']
            }
        }
    }
}

$mappedTools = @($mappedTools | Sort-Object -Unique)
$schemaTools = @($schemaTools | Sort-Object -Unique)

$missingInMapping = @($schemaTools | Where-Object { $mappedTools -notcontains $_ })
$extraMapping = @($mappedTools | Where-Object { $schemaTools -notcontains $_ })

$report = [PSCustomObject]@{
    Generated        = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss')
    ToolSchemaCount  = $schemaTools.Count
    ToolMappingCount = $mappedTools.Count
    MissingInMapping = $missingInMapping
    ExtraMapping     = $extraMapping
}

$reportJson = Join-Path $reportDir 'TOOL_SCHEMA_REPORT.json'
$reportMd = Join-Path $reportDir 'TOOL_SCHEMA_REPORT.md'

$report | ConvertTo-Json -Depth 6 | Set-Content -Path $reportJson -Encoding UTF8

$md = New-Object System.Text.StringBuilder
$null = $md.AppendLine('# TOOL_SCHEMA_REPORT')
$null = $md.AppendLine('')
$null = $md.AppendLine("Generated: $($report.Generated)")
$null = $md.AppendLine('')
$null = $md.AppendLine("- Tool schema count: $($report.ToolSchemaCount)")
$null = $md.AppendLine("- Tool mapping count: $($report.ToolMappingCount)")
$null = $md.AppendLine('')
$null = $md.AppendLine('## Missing in tool mapping')
if ($missingInMapping.Count -eq 0) {
    $null = $md.AppendLine('- None')
} else {
    foreach ($name in $missingInMapping) { $null = $md.AppendLine("- $name") }
}
$null = $md.AppendLine('')
$null = $md.AppendLine('## Extra mapping entries')
if ($extraMapping.Count -eq 0) {
    $null = $md.AppendLine('- None')
} else {
    foreach ($name in $extraMapping) { $null = $md.AppendLine("- $name") }
}

$md.ToString() | Set-Content -Path $reportMd -Encoding UTF8

Write-Host "Wrote: $reportMd" -ForegroundColor Green
Write-Host "Wrote: $reportJson" -ForegroundColor Green
