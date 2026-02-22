#Requires -Version 5.1
<#
.SYNOPSIS
  Build FunctionGemma router datasets using the Rust pipeline.

.DESCRIPTION
  Uses CargoTools via Invoke-RustBuild.ps1 to run the rust-functiongemma-train
  prepare-router command. This keeps dataset generation in Rust while matching
  the Python I/O contract (tool_calls or NO_TOOL).

  Optional: Use PcaiNative.dll to run the same dataset generation via native FFI.
#>

[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$ToolsPath,
    [string]$DiagnosePrompt,
    [string]$ChatPrompt,
    [string]$ScenariosPath,
    [string]$Output,
    [string]$TestVectors,
    [int]$MaxCases = 24,
    [switch]$NoToolCoverage,
    [switch]$Stream,
    [switch]$UseLld,
    [switch]$LlmDebug,
    [switch]$UseNative,
    [switch]$NativeOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
$trainRoot = Join-Path $repoRoot 'Deploy\rust-functiongemma-train'

if (-not $ToolsPath) { $ToolsPath = Join-Path $repoRoot 'Config\pcai-tools.json' }
if (-not $DiagnosePrompt) { $DiagnosePrompt = Join-Path $repoRoot 'Deploy\rust-functiongemma-train\examples\router_diagnose_prompt.md' }
if (-not $ChatPrompt) { $ChatPrompt = Join-Path $repoRoot 'Deploy\rust-functiongemma-train\examples\router_chat_prompt.md' }
if (-not $ScenariosPath) { $ScenariosPath = Join-Path $repoRoot 'Deploy\rust-functiongemma-train\examples\scenarios.json' }
if (-not $Output) { $Output = Join-Path $repoRoot 'Deploy\rust-functiongemma-train\data\rust_router_train.jsonl' }
if (-not $TestVectors) { $TestVectors = Join-Path $repoRoot 'Deploy\rust-functiongemma-train\data\test_vectors.json' }

function Resolve-FullPath {
    param([string]$Path)
    if (-not $Path) { return $null }
    if (Test-Path $Path) {
        return (Resolve-Path $Path).Path
    }
    return [System.IO.Path]::GetFullPath($Path)
}

function Invoke-NativeRouterDataset {
    param(
        [string]$ToolsPath,
        [string]$DiagnosePrompt,
        [string]$ChatPrompt,
        [string]$ScenariosPath,
        [string]$Output,
        [string]$TestVectors,
        [int]$MaxCases,
        [switch]$NoToolCoverage
    )

    if ($PSVersionTable.PSVersion.Major -lt 7) {
        Write-Warning "Native router dataset generation requires PowerShell 7+."
        return $false
    }

    $binPath = Join-Path $repoRoot 'bin'
    $wrapperPath = Join-Path $binPath 'PcaiNative.dll'
    if (-not (Test-Path $wrapperPath)) {
        Write-Warning "PcaiNative.dll not found at $wrapperPath."
        return $false
    }

    $env:PATH = "$binPath;$env:PATH"

    try {
        $loadedAssembly = [System.AppDomain]::CurrentDomain.GetAssemblies() |
            Where-Object { $_.GetName().Name -eq 'PcaiNative' }
        if (-not $loadedAssembly) {
            Add-Type -Path $wrapperPath -ErrorAction Stop
        }
    } catch {
        Write-Warning "Failed to load PcaiNative.dll: $_"
        return $false
    }

    if (-not [PcaiNative.PcaiCore]::IsAvailable) {
        Write-Warning "PcaiCore not available via PcaiNative."
        return $false
    }

    $report = [PcaiNative.FunctionGemmaModule]::BuildRouterDataset(
        (Resolve-FullPath $ToolsPath),
        (Resolve-FullPath $Output),
        (Resolve-FullPath $DiagnosePrompt),
        (Resolve-FullPath $ChatPrompt),
        (Resolve-FullPath $ScenariosPath),
        (Resolve-FullPath $TestVectors),
        [uint32]$MaxCases,
        (-not $NoToolCoverage)
    )

    if (-not $report -or -not $report.IsSuccess) {
        Write-Warning "Native router dataset generation returned no report or failure."
        return $false
    }

    Write-Host "Native router dataset generation complete." -ForegroundColor Green
    Write-Host "Items: $($report.Items) | Vectors: $($report.Vectors) | Output: $($report.OutputJsonl)"
    return $true
}

if ($UseNative) {
    $nativeOk = Invoke-NativeRouterDataset `
        -ToolsPath $ToolsPath `
        -DiagnosePrompt $DiagnosePrompt `
        -ChatPrompt $ChatPrompt `
        -ScenariosPath $ScenariosPath `
        -Output $Output `
        -TestVectors $TestVectors `
        -MaxCases $MaxCases `
        -NoToolCoverage:$NoToolCoverage

    if ($nativeOk) { return }
    if ($NativeOnly) { throw "Native router dataset generation failed." }

    Write-Warning "Falling back to Rust CLI router dataset generation."
}

Write-Host "Preparing FunctionGemma router dataset (Rust)..." -ForegroundColor Cyan
Write-Host "Tools: $ToolsPath"
Write-Host "Output: $Output"
Write-Host "TestVectors: $TestVectors"

$cargoArgs = @(
    'run','--','prepare-router',
    '--tools', $ToolsPath,
    '--output', $Output,
    '--diagnose-prompt', $DiagnosePrompt,
    '--chat-prompt', $ChatPrompt,
    '--max-cases', $MaxCases
)

if ($ScenariosPath) { $cargoArgs += @('--scenarios', $ScenariosPath) }
if ($NoToolCoverage) { $cargoArgs += '--no-tool-coverage' }
if ($TestVectors) { $cargoArgs += @('--test-vectors', $TestVectors) }
if ($Stream) { $cargoArgs += '--stream' }

& (Join-Path $repoRoot 'Tools\Invoke-RustBuild.ps1') `
    -Path $trainRoot `
    -CargoArgs $cargoArgs `
    -UseLld:$UseLld `
    -LlmDebug:$LlmDebug

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Router dataset generation complete." -ForegroundColor Green
