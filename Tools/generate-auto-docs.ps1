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

.PARAMETER RustWorkspaceRoots
  Explicit Cargo roots relative to RepoRoot or absolute. Defaults stay within
  this repository; missing explicit roots fail, while absent defaults are optional.

.PARAMETER AllowExternalRustRoots
  Allow explicitly selected workspaces outside RepoRoot. No external root is implicit.

.PARAMETER BuildDocs
  Generate attributable artifacts in a fresh per-run directory. This deliberately
  incurs a separate Cargo documentation build. Existing artifacts inspected without
  this switch carry ExistingUnverified provenance and do not establish build success.

.PARAMETER LibraryOnly
  Import generator functions without writing reports or invoking native tools.
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
    [switch]$BuildDocs,
    [string[]]$RustWorkspaceRoots,
    [switch]$AllowExternalRustRoots,
    [switch]$LibraryOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$scriptRoot = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
if (-not $RepoRoot) {
    $RepoRoot = Split-Path -Parent $scriptRoot
}

function Invoke-DocumentationCommand {
    <# .SYNOPSIS
    Run a native documentation command and reject nonzero exit status.
    #>
    [CmdletBinding()]
    param([Parameter(Mandatory)][string]$Tool, [Parameter(Mandatory)][string[]]$ArgumentList)
    $command = Get-Command $Tool -CommandType Application -ErrorAction Stop | Select-Object -First 1
    $output = & $command.Source @ArgumentList 2>&1
    $code = $LASTEXITCODE
    if ($code -ne 0) { throw "$Tool failed (exit $code): $($output | Out-String)" }
    return ($output | Out-String).Trim()
}

function Get-DocumentationWorkspace {
    <# .SYNOPSIS
    Resolve in-repository Cargo roots or explicitly authorized external roots.
    #>
    [CmdletBinding()]
    param([Parameter(Mandatory)][string]$Repository, [string[]]$Roots, [switch]$AllowExternal, [string[]]$DefaultRoots = @('Native/pcai_core'))
    $repositoryPath = [IO.Path]::GetFullPath($Repository).TrimEnd([IO.Path]::DirectorySeparatorChar)
    $explicitRoots = $null -ne $Roots -and $Roots.Count -gt 0
    if (-not $explicitRoots) { $Roots = $DefaultRoots }
    foreach ($root in $Roots) {
        $path = if ([IO.Path]::IsPathRooted($root)) { [IO.Path]::GetFullPath($root) } else { [IO.Path]::GetFullPath((Join-Path $repositoryPath $root)) }
        $prefix = $repositoryPath + [IO.Path]::DirectorySeparatorChar
        if (-not $AllowExternal -and $path -ne $repositoryPath -and -not $path.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
            throw "External Rust workspace requires -AllowExternalRustRoots: $path"
        }
        if (-not (Test-Path -LiteralPath (Join-Path $path 'Cargo.toml') -PathType Leaf)) {
            if ($explicitRoots) { throw "Cargo.toml not found in requested workspace: $path" }
            continue
        }
        $path
    }
}

function Invoke-RustDocumentation {
    <# .SYNOPSIS
    Generate exact Cargo package indexes with isolated artifact provenance.
    #>
    [CmdletBinding()]
    param([Parameter(Mandatory)][string]$Workspace, [switch]$Build, [switch]$DocumentPrivateItems)
    Push-Location -LiteralPath $Workspace
    try {
        $manifest = Join-Path $Workspace 'Cargo.toml'
        $metadata = Invoke-DocumentationCommand -Tool cargo.exe -ArgumentList @('metadata', '--format-version', '1', '--no-deps', '--manifest-path', $manifest) | ConvertFrom-Json
        $targetDirectory = $metadata.target_directory
        $state = 'ExistingUnverified'
        $documentedTargets = @()
        if ($Build) {
            # New output cannot be satisfied by stale files in a shared target.
            # Metadata honors workspace .cargo/config and CARGO_TARGET_DIR.
            $targetDirectory = Join-Path $targetDirectory ('pcai-docs/' + [Guid]::NewGuid().ToString('N'))
            $arguments = @('doc', '--workspace', '--no-deps', '--message-format=json', '--manifest-path', $manifest, '--target-dir', $targetDirectory)
            if ($DocumentPrivateItems) { $arguments += '--document-private-items' }
            $output = Invoke-DocumentationCommand -Tool cargo.exe -ArgumentList $arguments
            $messages = @(
                foreach ($line in ($output -split '\r?\n')) {
                    if ($line.TrimStart().StartsWith('{')) { $line | ConvertFrom-Json -ErrorAction Stop }
                }
            )
            $finished = @($messages | Where-Object reason -EQ 'build-finished')
            if ($finished.Count -ne 1 -or -not $finished[0].success) { throw 'Cargo documentation did not report a successful build-finished event' }
            $prefix = [IO.Path]::GetFullPath($targetDirectory).TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
            $indexes = @(
                foreach ($message in $messages) {
                    if ($message.reason -ne 'compiler-artifact' -or $metadata.workspace_members -notcontains $message.package_id) { continue }
                    if (-not $message.target.doc) { continue }
                    foreach ($filename in $message.filenames) {
                        if ([IO.Path]::GetFileName($filename) -ne 'index.html') { continue }
                        $index = [IO.Path]::GetFullPath($filename)
                        if (-not $index.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) { throw "Rust documentation artifact is outside this run's target directory: $index" }
                        if (-not (Test-Path -LiteralPath $index -PathType Leaf)) { throw "Required Rust documentation artifact missing: $index" }
                        if ((Get-Item -LiteralPath $index).Length -eq 0) { throw "Empty Rust documentation artifact: $index" }
                        $documentedTargets += [pscustomobject]@{ PackageId = $message.package_id; Target = $message.target.name; Features = $message.features; Index = $index }
                        $index
                    }
                }
            )
            # Cargo chooses enabled feature-gated targets and emits the actual
            # configured target-triple paths; metadata alone cannot select these.
            if ($indexes.Count -eq 0) { throw "No documentable Rust targets in requested workspace: $Workspace" }
            $state = 'Generated'
        }
        else {
            # Inspection retains explicitly unverified, exact-name metadata paths.
            # It does not claim Cargo's current feature/target selection was built.
            $indexes = @(
                foreach ($package in $metadata.packages) {
                    if ($metadata.workspace_members -notcontains $package.id) { continue }
                    foreach ($target in $package.targets) {
                        if (-not $target.doc -or ($target.kind -contains 'custom-build') -or ($target.kind -contains 'example') -or ($target.kind -contains 'test') -or ($target.kind -contains 'bench')) { continue }
                        $index = Join-Path $targetDirectory ('doc/' + $target.name.Replace('-', '_') + '/index.html')
                        if (Test-Path -LiteralPath $index -PathType Leaf) {
                            if ((Get-Item -LiteralPath $index).Length -eq 0) { throw "Empty Rust documentation artifact: $index" }
                            $index
                        }
                    }
                }
            )
        }
        [pscustomobject]@{
            Workspace         = $Workspace
            TargetDirectory   = $targetDirectory
            DocIndex          = if ($indexes.Count) { $indexes[0] } else { $null }
            DocIndexes        = $indexes
            DocumentedTargets = $documentedTargets
            Provenance        = $state
        }
    }
    finally { Pop-Location }
}

function Invoke-CSharpDocumentation {
    <# .SYNOPSIS
    Generate C# XML at an explicit location instead of selecting unrelated output.
    #>
    [CmdletBinding()]
    param([Parameter(Mandatory)][string]$Project, [Parameter(Mandatory)][string]$ReportDirectory, [switch]$Build)
    $projectName = [IO.Path]::GetFileNameWithoutExtension($Project)
    $docFile = $null
    $state = 'ExistingUnverified'
    if ($Build) {
        $outputDirectory = Join-Path $ReportDirectory ('csharp-docs/' + [Guid]::NewGuid().ToString('N'))
        $null = New-Item -ItemType Directory -Path $outputDirectory
        $docFile = Join-Path $outputDirectory "$projectName.xml"
        $null = Invoke-DocumentationCommand -Tool dotnet.exe -ArgumentList @('build', $Project, '-c', 'Release', '-p:GenerateDocumentationFile=true', "-p:DocumentationFile=$docFile", '-p:NoWarn=1591')
        if (-not (Test-Path -LiteralPath $docFile -PathType Leaf)) { throw "Required C# documentation artifact missing: $docFile" }
        [xml]$document = [IO.File]::ReadAllText($docFile)
        if (-not $document.doc.assembly.name) { throw "Invalid C# documentation artifact: $docFile" }
        $state = 'Generated'
    }
    else {
        # MSBuild resolves the assembly name and configuration-specific paths.
        $candidate = Invoke-DocumentationCommand -Tool dotnet.exe -ArgumentList @('msbuild', $Project, '-nologo', '-p:Configuration=Release', '-p:GenerateDocumentationFile=true', '-getProperty:DocumentationFile')
        if ($candidate) {
            $candidate = if ([IO.Path]::IsPathRooted($candidate)) { $candidate } else { Join-Path (Split-Path -Parent $Project) $candidate }
            if (Test-Path -LiteralPath $candidate -PathType Leaf) { $docFile = $candidate }
        }
    }
    [pscustomobject]@{ Project = $projectName; Path = $Project; DocXml = $docFile; Provenance = $state }
}

if ($LibraryOnly) { return }

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
    }
    catch { Write-Verbose "Build version unavailable: $($_.Exception.Message)" }
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
                    }
                    elseif ($json.matches) {
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
            if ($null -eq $counts) {
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
                }
                else {
                    foreach ($k in ($counts.Keys | Sort-Object)) {
                        $null = $md.AppendLine("- ${k}: $($counts[$k])")
                    }
                }
            }
            elseif ($counts.Count -eq 0) {
                $null = $md.AppendLine('No matches found.')
            }
            else {
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
                    Module   = $moduleName
                    Function = $fname
                    Synopsis = $syn
                    Path     = $file.FullName
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
    # `rg --files` was used purely to enumerate .csproj files. ripgrep is not
    # installed on GitHub-hosted runners, so this raised CommandNotFoundException
    # and took down the Generate Auto Docs step. Get-ChildItem needs no external
    # binary and there is no performance argument for one directory.
    $csproj = @(
        Get-ChildItem -Path (Join-Path $RepoRoot 'Native') -Recurse -Filter '*.csproj' -File -ErrorAction SilentlyContinue |
            Select-Object -ExpandProperty FullName
    )
    if ($BuildDocs -and $csproj.Count -eq 0) { throw 'No C# projects found for requested documentation build' }
    $csDocs = @(
        foreach ($proj in $csproj) {
            Invoke-CSharpDocumentation -Project $proj -ReportDirectory $reportDir -Build:$BuildDocs
        }
    )
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
    $rustDocs = @(
        foreach ($root in (Get-DocumentationWorkspace -Repository $RepoRoot -Roots $RustWorkspaceRoots -AllowExternal:$AllowExternalRustRoots)) {
            Invoke-RustDocumentation -Workspace $root -Build:$BuildDocs
        }
    )
    if ($BuildDocs -and $rustDocs.Count -eq 0) { throw 'No Rust workspaces found for requested documentation build' }
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

Write-Information "Wrote: $summaryPath" -InformationAction Continue
