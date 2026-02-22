#Requires -Version 5.1

<#+
.SYNOPSIS
  Runs FunctionGemma fine-tuning test suite and tool coverage reports.
#>

[CmdletBinding()]
param(
    [Parameter()]
    [ValidateSet('unit', 'integration', 'e2e', 'functional', 'rust', 'all')]
    [string]$Category = 'unit',
    [Parameter()]
    [ValidateSet('rust', 'python', 'both')]
    [string]$Runtime = 'rust',
    [Parameter()]
    [switch]$Fast,
    [Parameter()]
    [switch]$EvalReport,
    [Parameter()]
    [switch]$EvalFast,
    [Parameter()]
    [string]$EvalModelPath,
    [Parameter()]
    [string]$EvalTestData,
    [Parameter()]
    [string]$EvalOutput,
    [Parameter()]
    [string]$EvalAdapters,
    [Parameter()]
    [int]$EvalMaxNewTokens = 64,
    [Parameter()]
    [int]$EvalLoraR = 16,
    [Parameter()]
    [switch]$EvalNoSchemaValidate
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
$fgRoot = Join-Path $repoRoot 'Deploy\functiongemma-finetune'
$rustTrainRoot = Join-Path $repoRoot 'Deploy\rust-functiongemma-train'

if (-not $Fast) {
    & (Join-Path $repoRoot 'Tools\update-doc-status.ps1') -RepoRoot $repoRoot | Out-Null
    & (Join-Path $repoRoot 'Tools\update-tool-coverage.ps1') -RepoRoot $repoRoot | Out-Null
} else {
    Write-Host 'Fast mode: skipping doc status/tool coverage updates.' -ForegroundColor Yellow
}

$pytestArgs = @('-m', $Category)
if ($Category -eq 'all' -or $Category -eq 'rust') { $pytestArgs = @() }

$runRust = $Runtime -in @('rust', 'both')
$runPython = $Runtime -in @('python', 'both')
if ($Category -eq 'rust') { $runPython = $false }

function Assert-FileNotEmpty {
    param(
        [string]$Path,
        [string]$Label
    )
    if (-not (Test-Path $Path)) {
        throw "$Label missing: $Path"
    }
    $size = (Get-Item $Path).Length
    if ($size -le 0) {
        throw "$Label is empty: $Path"
    }
}

if ($runRust) {
    Write-Host 'Running Rust FunctionGemma tests...' -ForegroundColor Cyan
    & (Join-Path $repoRoot 'Build.ps1') -Component functiongemma -RunTests
    if ($LASTEXITCODE -ne 0) { throw "Rust tests failed (exit $LASTEXITCODE)" }

    if (-not $Fast) {
        Write-Host 'Generating Rust router dataset + test vectors...' -ForegroundColor Cyan
        & (Join-Path $repoRoot 'Tools\prepare-functiongemma-router-data.ps1')
        if ($LASTEXITCODE -ne 0) { throw "Rust dataset generation failed (exit $LASTEXITCODE)" }

        Write-Host 'Generating FunctionGemma tool documentation...' -ForegroundColor Cyan
        & (Join-Path $repoRoot 'Tools\generate-functiongemma-tool-docs.ps1')
        if ($LASTEXITCODE -ne 0) { throw "Tool documentation generation failed (exit $LASTEXITCODE)" }

        $datasetPath = Join-Path $repoRoot 'Deploy\rust-functiongemma-train\data\rust_router_train.jsonl'
        $vectorsPath = Join-Path $repoRoot 'Deploy\rust-functiongemma-train\data\test_vectors.json'
        $docsPath = Join-Path $repoRoot 'Deploy\rust-functiongemma\TOOLS.md'

        Assert-FileNotEmpty -Path $datasetPath -Label 'Router dataset'
        Assert-FileNotEmpty -Path $vectorsPath -Label 'Tool test vectors'
        Assert-FileNotEmpty -Path $docsPath -Label 'Tool documentation'

        $firstLine = Get-Content $datasetPath -TotalCount 1
        if (-not $firstLine) { throw "Router dataset has no lines: $datasetPath" }
        $firstObj = $firstLine | ConvertFrom-Json
        if (-not ($firstObj.PSObject.Properties.Name -contains 'messages')) { throw 'Router dataset missing messages key' }
        if (-not ($firstObj.PSObject.Properties.Name -contains 'tools')) { throw 'Router dataset missing tools key' }

        $vectors = Get-Content $vectorsPath | ConvertFrom-Json
        if (-not $vectors -or $vectors.Count -lt 1) { throw 'Tool test vectors empty' }
        if (-not ($vectors[0].PSObject.Properties.Name -contains 'tool')) { throw 'Tool test vector missing tool key' }
        if (-not ($vectors[0].PSObject.Properties.Name -contains 'arguments')) { throw 'Tool test vector missing arguments key' }
    } else {
        Write-Host 'Fast mode: skipping dataset + tool docs generation.' -ForegroundColor Yellow
    }

    if ($EvalReport) {
        & (Join-Path $repoRoot 'Tools\run-functiongemma-eval.ps1') `
            -ModelPath $EvalModelPath `
            -TestData $EvalTestData `
            -Adapters $EvalAdapters `
            -Output $EvalOutput `
            -MaxNewTokens $EvalMaxNewTokens `
            -LoraR $EvalLoraR `
            -FastEval:$EvalFast `
            -NoSchemaValidate:$EvalNoSchemaValidate

        if ($LASTEXITCODE -ne 0) { throw "Eval report generation failed (exit $LASTEXITCODE)" }
    }

    Write-Host 'Rust FunctionGemma tests: PASS' -ForegroundColor Green
}

if ($runPython) {
    if (-not (Test-Path $fgRoot)) {
        Write-Warning "Python FunctionGemma repo not found at $fgRoot. Skipping Python tests."
    } else {
        Push-Location $fgRoot
        try {
            $env:PYTHONUTF8 = '1'
            if (-not $env:VLLM_BASE_URL) {
                $cfgPath = Join-Path $repoRoot 'Config\llm-config.json'
                if (Test-Path $cfgPath) {
                    $cfg = Get-Content $cfgPath -Raw | ConvertFrom-Json
                    if ($cfg.providers.vllm.baseUrl) {
                        $env:VLLM_BASE_URL = $cfg.providers.vllm.baseUrl
                    }
                }
            }

            # Use uv if available, fallback to python.
            $uv = Get-Command uv -ErrorAction SilentlyContinue
            if ($uv) {
                & uv run python -m pytest @pytestArgs .\
            } else {
                & python -m pytest @pytestArgs .\
            }
            if ($LASTEXITCODE -ne 0) { throw "Python tests failed (exit $LASTEXITCODE)" }
        } finally {
            Pop-Location
        }
    }
}
