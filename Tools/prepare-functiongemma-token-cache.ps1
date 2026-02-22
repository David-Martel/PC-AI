#Requires -Version 5.1
<#
.SYNOPSIS
  Build token cache for FunctionGemma training (Rust).

.DESCRIPTION
  Routes rust-functiongemma-train prepare-cache through Build.ps1 to pre-tokenize
  JSONL datasets under the unified build workflow.
#>

[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$Input,
    [string]$TokenizerPath,
    [string]$OutputDir,
    [string]$ChatTemplate,
    [switch]$UseLld,
    [switch]$LlmDebug
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot

if (-not $Input) { $Input = Join-Path $repoRoot 'Deploy\rust-functiongemma-train\data\rust_router_train.jsonl' }
if (-not $TokenizerPath) { $TokenizerPath = Join-Path $repoRoot 'Models\functiongemma-270m-it\tokenizer.json' }
if (-not $OutputDir) { $OutputDir = Join-Path $repoRoot 'output\functiongemma-token-cache' }
if (-not $ChatTemplate) {
    $modelDir = Split-Path -Parent $TokenizerPath
    $candidate = Join-Path $modelDir 'chat_template.jinja'
    if (Test-Path $candidate) { $ChatTemplate = $candidate }
}

$functionGemmaArgs = @(
    '--input', $Input,
    '--tokenizer', $TokenizerPath,
    '--output-dir', $OutputDir
)
if ($ChatTemplate) { $functionGemmaArgs += @('--chat-template', $ChatTemplate) }

if ($LlmDebug) {
    Write-Warning '-LlmDebug is not currently consumed by Build.ps1; continuing without extra debug toggles.'
}

$buildScript = Join-Path $repoRoot 'Build.ps1'
$prevUseLld = $env:CARGO_USE_LLD
if ($UseLld) { $env:CARGO_USE_LLD = '1' }

try {
    & $buildScript -Component functiongemma-token-cache -Configuration Release -FunctionGemmaArgs $functionGemmaArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} finally {
    if ($UseLld) {
        if ([string]::IsNullOrEmpty($prevUseLld)) {
            Remove-Item Env:CARGO_USE_LLD -ErrorAction SilentlyContinue
        } else {
            $env:CARGO_USE_LLD = $prevUseLld
        }
    }
}

Write-Host "Token cache built at $OutputDir" -ForegroundColor Green
