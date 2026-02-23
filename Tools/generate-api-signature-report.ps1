#Requires -Version 5.1

<#+
.SYNOPSIS
  Generates API signature alignment reports for PowerShell, C#, and Rust.

.DESCRIPTION
  - Parses PowerShell public functions and compares parameters to help blocks
  - Compares C# DllImport declarations to Rust exported functions
  - Compares PowerShell wrapper calls to available C# methods
  Writes Reports\API_SIGNATURE_REPORT.json and Reports\API_SIGNATURE_REPORT.md
#>

[CmdletBinding()]
param(
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

. "$RepoRoot\Modules\PC-AI.Common\Public\Get-ScriptMetadata.ps1"

function Get-PublicFunctionInfo {
    param([string]$Path)
    $meta = Get-ScriptMetadata -Path $Path
    return $meta.Functions
}

function Get-CSharpDllImports {
    param([string]$Root)

    if (-not (Test-Path $Root)) { return @() }
    $files = Get-ChildItem -Path $Root -Filter '*.cs' -Recurse -ErrorAction SilentlyContinue
    $imports = @()
    foreach ($file in $files) {
        $content = Get-Content -Path $file.FullName -Raw -Encoding UTF8
        if ([string]::IsNullOrEmpty($content)) { continue }
        $matches = [regex]::Matches($content, '\[DllImport\([^\)]*\)\]\s*internal\s+static\s+extern\s+[^\s]+\s+(pcai_[A-Za-z0-9_]+)\s*\(')
        foreach ($m in $matches) {
            $imports += $m.Groups[1].Value
        }
    }
    return @($imports | Sort-Object -Unique)
}

function Get-RustExports {
    param([string]$Root)

    if (-not (Test-Path $Root)) { return @() }
    $files = Get-ChildItem -Path $Root -Filter '*.rs' -Recurse -ErrorAction SilentlyContinue
    $exports = @()
    foreach ($file in $files) {
        $content = Get-Content -Path $file.FullName -Raw -Encoding UTF8
        if ([string]::IsNullOrEmpty($content)) { continue }
        $matches = [regex]::Matches($content, 'pub\s+(?:unsafe\s+)?extern\s+"C"\s+fn\s+(pcai_[A-Za-z0-9_]+)')
        foreach ($m in $matches) {
            $exports += $m.Groups[1].Value
        }
    }
    return @($exports | Sort-Object -Unique)
}

function Get-PowerShellPcaiCalls {
    param([string]$ModuleRoot)

    $files = Get-ChildItem -Path $ModuleRoot -Filter '*.ps1' -Recurse -ErrorAction SilentlyContinue
    $calls = @()
    foreach ($file in $files) {
        $content = Get-Content -Path $file.FullName -Raw -Encoding UTF8
        if ([string]::IsNullOrEmpty($content)) { continue }
        $matches = [regex]::Matches($content, '\[PcaiNative\.PcaiCore\]::([A-Za-z0-9_]+)')
        foreach ($m in $matches) {
            $calls += $m.Groups[1].Value
        }
    }
    return @($calls | Sort-Object -Unique)
}

function Get-CSharpPcaiCoreMethods {
    param([string]$Path)

    if (-not (Test-Path $Path)) { return @() }
    $content = Get-Content -Path $Path -Raw -Encoding UTF8
    if ([string]::IsNullOrEmpty($content)) { return @() }
    $methodMatches = [regex]::Matches($content, 'public\s+static\s+[^\s]+\s+([A-Za-z0-9_]+)\s*\(')
    $propertyMatches = [regex]::Matches($content, 'public\s+static\s+[^\s]+\s+([A-Za-z0-9_]+)\s*(?:=>|\{)')
    $names = @()
    foreach ($m in $methodMatches) { $names += $m.Groups[1].Value }
    foreach ($m in $propertyMatches) { $names += $m.Groups[1].Value }
    return @($names | Sort-Object -Unique)
}

$modulesRoot = Join-Path $RepoRoot 'Modules'
$publicFiles = Get-ChildItem -Path $modulesRoot -Filter '*.ps1' -Recurse -ErrorAction SilentlyContinue |
    Where-Object { $_.FullName -match '\\Public\\' }

$psFunctions = @()
foreach ($file in $publicFiles) {
    $psFunctions += Get-PublicFunctionInfo -Path $file.FullName
}

$missingHelp = @($psFunctions | Where-Object { -not $_.HelpPresent })
$missingHelpParams = @($psFunctions | Where-Object { @($_.MissingHelpParameters).Count -gt 0 })
$extraHelpParams = @($psFunctions | Where-Object { @($_.ExtraHelpParameters).Count -gt 0 })

$csharpRoot = Join-Path $RepoRoot 'Native\PcaiNative'
$pcaiCorePath = Join-Path $csharpRoot 'PcaiCore.cs'
$rustRoots = @(
    (Join-Path $RepoRoot 'Native\pcai_core\pcai_core_lib\src')
)
$csDllImports = Get-CSharpDllImports -Root $csharpRoot
$rustExports = @()
foreach ($root in $rustRoots) {
    $rustExports += Get-RustExports -Root $root
}
$rustExports = @($rustExports | Sort-Object -Unique)

$missingRustExports = @($csDllImports | Where-Object { $rustExports -notcontains $_ })

$psPcaiCalls = Get-PowerShellPcaiCalls -ModuleRoot $modulesRoot
$csCoreMethods = Get-CSharpPcaiCoreMethods -Path $pcaiCorePath
$missingCsharpMethods = @($psPcaiCalls | Where-Object { $csCoreMethods -notcontains $_ })

$report = [PSCustomObject]@{
    Generated          = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss')
    PowerShell         = [PSCustomObject]@{
        FunctionCount         = @($psFunctions).Count
        MissingHelpCount      = @($missingHelp).Count
        MissingHelpParameters = $missingHelpParams
        ExtraHelpParameters   = $extraHelpParams
    }
    CSharp             = [PSCustomObject]@{
        DllImportCount     = @($csDllImports).Count
        MissingRustExports = $missingRustExports
    }
    PowerShellToCSharp = [PSCustomObject]@{
        PcaiCalls            = $psPcaiCalls
        MissingCsharpMethods = $missingCsharpMethods
    }
}

$reportJson = Join-Path $reportDir 'API_SIGNATURE_REPORT.json'
$reportMd = Join-Path $reportDir 'API_SIGNATURE_REPORT.md'

$report | ConvertTo-Json -Depth 6 | Set-Content -Path $reportJson -Encoding UTF8

$md = New-Object System.Text.StringBuilder
$null = $md.AppendLine('# API_SIGNATURE_REPORT')
$null = $md.AppendLine('')
$null = $md.AppendLine("Generated: $($report.Generated)")
$null = $md.AppendLine('')
$null = $md.AppendLine("PowerShell functions: $($report.PowerShell.FunctionCount)")
$null = $md.AppendLine("Missing help blocks: $($report.PowerShell.MissingHelpCount)")
$null = $md.AppendLine("C# DllImports: $($report.CSharp.DllImportCount)")
$null = $md.AppendLine("Missing Rust exports: $(@($report.CSharp.MissingRustExports).Count)")
$null = $md.AppendLine('')

if (@($report.PowerShell.MissingHelpParameters).Count -gt 0) {
    $null = $md.AppendLine('## Missing help parameters')
    foreach ($item in $report.PowerShell.MissingHelpParameters) {
        $null = $md.AppendLine("- $($item.Name): missing $($item.MissingHelpParameters -join ', ')")
    }
    $null = $md.AppendLine('')
}

if (@($report.PowerShell.ExtraHelpParameters).Count -gt 0) {
    $null = $md.AppendLine('## Extra help parameters')
    foreach ($item in $report.PowerShell.ExtraHelpParameters) {
        $null = $md.AppendLine("- $($item.Name): extra $($item.ExtraHelpParameters -join ', ')")
    }
    $null = $md.AppendLine('')
}

if (@($report.CSharp.MissingRustExports).Count -gt 0) {
    $null = $md.AppendLine('## Missing Rust exports for C# DllImports')
    foreach ($name in $report.CSharp.MissingRustExports) {
        $null = $md.AppendLine("- $name")
    }
    $null = $md.AppendLine('')
}

if (@($report.PowerShellToCSharp.MissingCsharpMethods).Count -gt 0) {
    $null = $md.AppendLine('## Missing C# methods referenced by PowerShell')
    foreach ($name in $report.PowerShellToCSharp.MissingCsharpMethods) {
        $null = $md.AppendLine("- $name")
    }
    $null = $md.AppendLine('')
}

$md.ToString() | Set-Content -Path $reportMd -Encoding UTF8

Write-Host "Wrote: $reportMd" -ForegroundColor Green
Write-Host "Wrote: $reportJson" -ForegroundColor Green
