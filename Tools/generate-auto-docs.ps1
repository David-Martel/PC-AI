#Requires -Version 5.1

<#+
.SYNOPSIS
  Unified auto-documentation generator for PC_AI (PowerShell, C#, Rust, ast-grep).

.DESCRIPTION
  - Runs ast-grep-based doc status + tool coverage reports
  - Optionally runs global ast-grep rules from ~/.config/ast-grep
  - Builds PowerShell module command index
  - Optionally generates C# XML docs and Rust docs
  - Links outputs to PCAI_BUILD_VERSION (from Native\build.ps1)
#>

[CmdletBinding()]
param(
    [Parameter()]
    [string]$RepoRoot,

    [Parameter()]
    [switch]$IncludeAstGrep,

    [Parameter()]
    [switch]$IncludeGlobalAstGrep,

    [Parameter()]
    [switch]$IncludePowerShell,

    [Parameter()]
    [switch]$IncludeCSharp,

    [Parameter()]
    [switch]$IncludeRust,

    [Parameter()]
    [switch]$BuildDocs
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$scriptRoot = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
if (-not $RepoRoot) {
    $RepoRoot = Split-Path -Parent $scriptRoot
}

if (-not ($IncludeAstGrep -or $IncludeGlobalAstGrep -or $IncludePowerShell -or $IncludeCSharp -or $IncludeRust)) {
    $IncludeAstGrep = $true
    $IncludePowerShell = $true
    $IncludeCSharp = $true
    $IncludeRust = $true
}

$reportDir = Join-Path $RepoRoot 'Reports'
if (-not (Test-Path $reportDir)) {
    New-Item -ItemType Directory -Path $reportDir -Force | Out-Null
}

function Get-BuildVersion {
    if ($env:PCAI_BUILD_VERSION) { return $env:PCAI_BUILD_VERSION }
    try {
        $ver = git -C $RepoRoot describe --tags --always --dirty 2>$null
        if ($ver) { return $ver }
    } catch { }
    return '0.0.0-dev'
}

$version = Get-BuildVersion
$timestamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'

$summary = New-Object System.Text.StringBuilder
$null = $summary.AppendLine('# AUTO_DOCS_SUMMARY')
$null = $summary.AppendLine('')
$null = $summary.AppendLine("Generated: $timestamp")
$null = $summary.AppendLine("BuildVersion: $version")
$null = $summary.AppendLine('')

# -----------------------------------------------------------------------------
# Ast-grep doc status + tool coverage
# -----------------------------------------------------------------------------
if ($IncludeAstGrep) {
    $null = $summary.AppendLine('## ast-grep (repo config)')
    $null = $summary.AppendLine('- update-doc-status.ps1')
    $null = $summary.AppendLine('- update-tool-coverage.ps1')
    $null = $summary.AppendLine('')

    & (Join-Path $RepoRoot 'Tools\update-doc-status.ps1') -RepoRoot $RepoRoot | Out-Null
    & (Join-Path $RepoRoot 'Tools\update-tool-coverage.ps1') -RepoRoot $RepoRoot | Out-Null
}

if ($IncludeGlobalAstGrep) {
    $globalConfig = Join-Path $env:USERPROFILE '.config\ast-grep\sgconfig.yml'
    $sgExe = Get-Command sg.exe -ErrorAction SilentlyContinue
    if ((Test-Path $globalConfig) -and $sgExe) {
        $globalOut = Join-Path $reportDir 'ASTGREP_GLOBAL.json'
        $globalMd = Join-Path $reportDir 'ASTGREP_GLOBAL.md'
        try {
            $sgArgs = @('scan', '-c', $globalConfig, '--json=compact', $RepoRoot)
            & $sgExe.Path @sgArgs 2>$null | Out-File -FilePath $globalOut -Encoding UTF8

            $counts = @{}
            $fileInfo = Get-Item -Path $globalOut -ErrorAction SilentlyContinue
            if ($fileInfo -and $fileInfo.Length -gt 0) {
                try {
                    $json = Get-Content -Path $globalOut -Raw -Encoding UTF8 | ConvertFrom-Json -ErrorAction Stop
                    $sgMatches = $null
                    if ($json -is [System.Collections.IEnumerable] -and -not ($json -is [string]) -and -not ($json.PSObject.Properties.Name -contains 'matches')) {
                        $sgMatches = $json
                    } elseif ($json.matches) {
                        $sgMatches = $json.matches
                    }
                    if ($sgMatches) {
                        foreach ($m in $sgMatches) {
                            $rid = if ($m.ruleId) { $m.ruleId } elseif ($m.rule -and $m.rule.id) { $m.rule.id } else { 'unknown' }
                            if (-not $counts.ContainsKey($rid)) { $counts[$rid] = 0 }
                            $counts[$rid]++
                        }
                    }
                }
                catch {
                    $counts = $null
                }
            }

            $md = New-Object System.Text.StringBuilder
            $null = $md.AppendLine('# ASTGREP_GLOBAL')
            $null = $md.AppendLine('')
            $null = $md.AppendLine("Config: $globalConfig")
            $null = $md.AppendLine('')
            if ($counts -eq $null) {
                $counts = @{}
                $rulePattern = '"ruleId"\\s*:\\s*"(?<id>[^"]+)"'
                $ruleMatches = Select-String -Path $globalOut -Pattern $rulePattern -AllMatches -ErrorAction SilentlyContinue
                foreach ($match in $ruleMatches) {
                    foreach ($m in $match.Matches) {
                        $rid = $m.Groups['id'].Value
                        if ($rid) {
                            if (-not $counts.ContainsKey($rid)) { $counts[$rid] = 0 }
                            $counts[$rid]++
                        }
                    }
                }
                if ($counts.Count -eq 0) {
                    $null = $md.AppendLine('Failed to parse JSON output. See ASTGREP_GLOBAL.json for raw results.')
                } else {
                    foreach ($k in ($counts.Keys | Sort-Object)) {
                        $null = $md.AppendLine("- ${k}: $($counts[$k])")
                    }
                }
            } elseif ($counts.Count -eq 0) {
                $null = $md.AppendLine('No matches found.')
            } else {
                foreach ($k in ($counts.Keys | Sort-Object)) {
                    $null = $md.AppendLine("- ${k}: $($counts[$k])")
                }
            }
            $md.ToString() | Set-Content -Path $globalMd -Encoding UTF8
        }
        catch {
            $md = New-Object System.Text.StringBuilder
            $null = $md.AppendLine('# ASTGREP_GLOBAL')
            $null = $md.AppendLine('')
            $null = $md.AppendLine("Config: $globalConfig")
            $null = $md.AppendLine('')
            $null = $md.AppendLine("Error: $($_.Exception.Message)")
            $md.ToString() | Set-Content -Path $globalMd -Encoding UTF8
        }

        $null = $summary.AppendLine('## ast-grep (global config)')
        $null = $summary.AppendLine("- $globalOut")
        $null = $summary.AppendLine("- $globalMd")
        $null = $summary.AppendLine('')
    }
}

# -----------------------------------------------------------------------------
# PowerShell module docs
# -----------------------------------------------------------------------------
if ($IncludePowerShell) {
    $psModulesDir = Join-Path $RepoRoot 'Modules'
    $psReportJson = Join-Path $reportDir 'PS_MODULE_INDEX.json'
    $psReportMd = Join-Path $reportDir 'PS_MODULE_INDEX.md'

    $entries = @()
    $moduleDirs = Get-ChildItem -Path $psModulesDir -Directory -ErrorAction SilentlyContinue
    foreach ($moduleDir in $moduleDirs) {
        $psd1 = Get-ChildItem -Path $moduleDir.FullName -Filter '*.psd1' -ErrorAction SilentlyContinue | Select-Object -First 1
        if (-not $psd1) { continue }

        $data = Import-PowerShellDataFile -Path $psd1.FullName
        $moduleName = $data.RootModule
        if (-not $moduleName) { $moduleName = $moduleDir.Name }

        $publicDir = Join-Path $moduleDir.FullName 'Public'
        if (-not (Test-Path $publicDir)) { continue }

        $publicFiles = Get-ChildItem -Path $publicDir -Filter '*.ps1' -ErrorAction SilentlyContinue
        foreach ($file in $publicFiles) {
            $content = Get-Content -Path $file.FullName -Raw -Encoding UTF8
            if ($content -match 'function\s+([A-Za-z0-9_-]+)') {
                $fname = $Matches[1]
                $syn = ''
                if ($content -match '(?ms)\.SYNOPSIS\s*(?<syn>.+?)\r?\n\s*\.[A-Z]') {
                    $syn = $Matches['syn'].Trim()
                }
                $entries += [PSCustomObject]@{
                    Module = $moduleName
                    Function = $fname
                    Synopsis = $syn
                    Path = $file.FullName
                }
            }
        }
    }

    $entries | ConvertTo-Json -Depth 6 | Set-Content -Path $psReportJson -Encoding UTF8

    $md = New-Object System.Text.StringBuilder
    $null = $md.AppendLine('# PS_MODULE_INDEX')
    $null = $md.AppendLine('')
    $null = $md.AppendLine("Generated: $timestamp")
    $null = $md.AppendLine('')
    foreach ($group in ($entries | Group-Object Module)) {
        $null = $md.AppendLine("## $($group.Name)")
        foreach ($item in $group.Group) {
            $line = if ($item.Synopsis) { "- $($item.Function): $($item.Synopsis)" } else { "- $($item.Function)" }
            $null = $md.AppendLine($line)
        }
        $null = $md.AppendLine('')
    }

    $md.ToString() | Set-Content -Path $psReportMd -Encoding UTF8

    $null = $summary.AppendLine('## PowerShell module docs')
    $null = $summary.AppendLine("- $psReportMd")
    $null = $summary.AppendLine("- $psReportJson")
    $null = $summary.AppendLine('')
}

# -----------------------------------------------------------------------------
# C# docs (XML)
# -----------------------------------------------------------------------------
if ($IncludeCSharp) {
    $csproj = rg --files -g '*.csproj' (Join-Path $RepoRoot 'Native')
    $csDocs = @()
    foreach ($proj in $csproj) {
        $projectName = [System.IO.Path]::GetFileNameWithoutExtension($proj)
        if ($BuildDocs -and (Get-Command dotnet -ErrorAction SilentlyContinue)) {
            & dotnet build $proj -c Release /p:GenerateDocumentationFile=true /p:NoWarn=1591 | Out-Null
        }
        $docFile = Get-ChildItem -Path (Split-Path $proj) -Recurse -Filter "$projectName.xml" -ErrorAction SilentlyContinue | Select-Object -First 1
        $csDocs += [PSCustomObject]@{
            Project = $projectName
            Path = $proj
            DocXml = if ($docFile) { $docFile.FullName } else { $null }
        }
    }

    $csReport = Join-Path $reportDir 'CSHARP_DOCS.json'
    $csDocs | ConvertTo-Json -Depth 6 | Set-Content -Path $csReport -Encoding UTF8

    $null = $summary.AppendLine('## C# docs')
    foreach ($entry in $csDocs) {
        $line = if ($entry.DocXml) { "- $($entry.Project): $($entry.DocXml)" } else { "- $($entry.Project): (no xml found)" }
        $null = $summary.AppendLine($line)
    }
    $null = $summary.AppendLine('')
}

# -----------------------------------------------------------------------------
# Rust docs
# -----------------------------------------------------------------------------
if ($IncludeRust) {
    $rustDocs = @()

    $workspaceRoots = @(
        (Join-Path $RepoRoot 'Native\pcai_core')
        'C:\codedev\nukenul\nuker_core'
    )

    foreach ($root in $workspaceRoots) {
        if (-not (Test-Path (Join-Path $root 'Cargo.toml'))) { continue }
        if ($BuildDocs -and (Get-Command cargo -ErrorAction SilentlyContinue)) {
            Push-Location $root
            $prevWrapper = $env:RUSTC_WRAPPER
            $prevSccache = $env:SCCACHE_DISABLE
            try {
                # Avoid sccache issues during doc builds
                $env:RUSTC_WRAPPER = ''
                $env:SCCACHE_DISABLE = '1'
                & cargo doc --workspace --no-deps | Out-Null
            } finally {
                $env:RUSTC_WRAPPER = $prevWrapper
                $env:SCCACHE_DISABLE = $prevSccache
                Pop-Location
            }
        }
        $docIndex = $null
        $docRoot = $null
        if ($env:CARGO_TARGET_DIR) {
            $docRoot = Join-Path $env:CARGO_TARGET_DIR 'doc'
        } else {
            $docRoot = Join-Path $root 'target\doc'
        }
        if ($docRoot -and (Test-Path $docRoot)) {
            $candidate = Join-Path $docRoot 'index.html'
            if (Test-Path $candidate) {
                $docIndex = $candidate
            } else {
                $anyIndex = Get-ChildItem -Path $docRoot -Filter 'index.html' -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
                if ($anyIndex) {
                    $docIndex = $docRoot
                }
            }
        }
        $rustDocs += [PSCustomObject]@{
            Workspace = $root
            DocIndex = $docIndex
        }
    }

    $rustReport = Join-Path $reportDir 'RUST_DOCS.json'
    $rustDocs | ConvertTo-Json -Depth 6 | Set-Content -Path $rustReport -Encoding UTF8

    $null = $summary.AppendLine('## Rust docs')
    foreach ($entry in $rustDocs) {
        $line = if ($entry.DocIndex) { "- $($entry.Workspace): $($entry.DocIndex)" } else { "- $($entry.Workspace): (no docs found)" }
        $null = $summary.AppendLine($line)
    }
    $null = $summary.AppendLine('')
}

$summaryPath = Join-Path $reportDir 'AUTO_DOCS_SUMMARY.md'
$summary.ToString() | Set-Content -Path $summaryPath -Encoding UTF8

Write-Host "Wrote: $summaryPath" -ForegroundColor Green
