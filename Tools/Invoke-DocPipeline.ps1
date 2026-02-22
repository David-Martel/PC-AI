#Requires -Version 5.1
<#
.SYNOPSIS
    Unified documentation generation and FunctionGemma training data pipeline.

.DESCRIPTION
    Master orchestrator that:
    1. Generates documentation from code (Rust, PowerShell, C#)
    2. Exports structured training data for FunctionGemma
    3. Validates training data format
    4. Updates Reports/ with current status

.PARAMETER Mode
    Pipeline mode: Full, DocsOnly, TrainingOnly, Validate

.PARAMETER OutputFormat
    Output format: Json, Markdown, Both

.PARAMETER SkipRust
    Skip Rust documentation generation (cargo doc)

.PARAMETER SkipTraining
    Skip training data generation

.PARAMETER Force
    Overwrite existing files without prompting

.PARAMETER UseNativeRouter
    Use the PcaiNative DLL to generate the router dataset when available.

.PARAMETER RouterMaxCases
    Maximum number of argument combinations per tool for router dataset generation.

.PARAMETER NoToolCoverage
    Skip auto-generated tool coverage examples.

.EXAMPLE
    .\Invoke-DocPipeline.ps1 -Mode Full

.EXAMPLE
    .\Invoke-DocPipeline.ps1 -Mode TrainingOnly -OutputFormat Json
#>

[CmdletBinding()]
param(
    [ValidateSet('Full', 'DocsOnly', 'TrainingOnly', 'Validate')]
    [string]$Mode = 'Full',

    [ValidateSet('Json', 'Markdown', 'Both')]
    [string]$OutputFormat = 'Both',

    [switch]$SkipRust,
    [switch]$SkipTraining,
    [switch]$Force,
    [switch]$UseNativeRouter,
    [int]$RouterMaxCases = 24,
    [switch]$NoToolCoverage
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# Paths
$repoRoot = Split-Path -Parent $PSScriptRoot
$configDir = Join-Path $repoRoot 'Config'
$reportsDir = Join-Path $repoRoot 'Reports'
$deployDir = Join-Path $repoRoot 'Deploy'
$toolsDir = $PSScriptRoot

# Ensure output directories exist
@($reportsDir, (Join-Path $deployDir 'rust-functiongemma')) | ForEach-Object {
    if (-not (Test-Path $_)) { New-Item -ItemType Directory -Path $_ -Force | Out-Null }
}

# Pipeline state
$pipelineState = [PSCustomObject]@{
    StartTime = Get-Date
    EndTime = $null
    Duration = ''
    Mode = $Mode
    Steps = @()
    Errors = @()
    Outputs = @()
}

function Add-PipelineStep {
    param([string]$Name, [string]$Status, [string]$Output = '', [string]$Error = '')
    $step = [PSCustomObject]@{
        Name = $Name
        Status = $Status
        Output = $Output
        Error = $Error
        Timestamp = Get-Date -Format 'HH:mm:ss'
    }
    $pipelineState.Steps += $step
    if ($Output) { $pipelineState.Outputs += $Output }
    if ($Error) { $pipelineState.Errors += $Error }

    $color = switch ($Status) {
        'Success' { 'Green' }
        'Warning' { 'Yellow' }
        'Error' { 'Red' }
        'Skipped' { 'Gray' }
        default { 'White' }
    }
    Write-Host "[$($step.Timestamp)] $Name : $Status" -ForegroundColor $color
}

# Initialize CMake environment (fix stale CMAKE_ROOT/CMAKE_PREFIX_PATH)
$script:CmakeInfo = $null
$cmakeHelper = Join-Path $toolsDir 'Initialize-CmakeEnvironment.ps1'
if (Test-Path $cmakeHelper) {
    . $cmakeHelper
    $script:CmakeInfo = Initialize-CmakeEnvironment -Quiet
    if ($script:CmakeInfo.Found) {
        Add-PipelineStep -Name 'CmakeEnv' -Status 'Success' -Output "CMAKE_ROOT=$($script:CmakeInfo.CmakeRoot)"
    } else {
        Add-PipelineStep -Name 'CmakeEnv' -Status 'Warning' -Error 'cmake.exe not found; CMake-dependent docs may fail'
    }
} else {
    Add-PipelineStep -Name 'CmakeEnv' -Status 'Warning' -Error "CMake helper not found at $cmakeHelper"
}

# Initialize CUDA environment (best-effort) for GPU-dependent Rust crates
$script:CudaInfo = $null
$cudaHelper = Join-Path $toolsDir 'Initialize-CudaEnvironment.ps1'
if (Test-Path $cudaHelper) {
    . $cudaHelper
    $script:CudaInfo = Initialize-CudaEnvironment -Quiet
    if ($script:CudaInfo.Found) {
        Add-PipelineStep -Name 'CudaEnv' -Status 'Success' -Output "CUDA_PATH=$($script:CudaInfo.CudaPath)"
    } else {
        Add-PipelineStep -Name 'CudaEnv' -Status 'Warning' -Error 'CUDA not detected; GPU-only steps will be skipped'
        $env:LLAMA_CUDA = '0'
    }
} else {
    Add-PipelineStep -Name 'CudaEnv' -Status 'Warning' -Error "CUDA helper not found at $cudaHelper"
    $env:LLAMA_CUDA = '0'
}

# ============================================================================
# Step 1: Generate DOC_STATUS report (TODO/FIXME/DEPRECATED markers)
# ============================================================================
function Invoke-DocStatusGeneration {
    Write-Host "`n=== Generating DOC_STATUS report ===" -ForegroundColor Cyan

    $script = Join-Path $toolsDir 'update-doc-status.ps1'
    if (-not (Test-Path $script)) {
        Add-PipelineStep -Name 'DOC_STATUS' -Status 'Error' -Error 'Script not found'
        return
    }

    try {
        & $script -RepoRoot $repoRoot
        Add-PipelineStep -Name 'DOC_STATUS' -Status 'Success' -Output (Join-Path $reportsDir 'DOC_STATUS.md')
    }
    catch {
        Add-PipelineStep -Name 'DOC_STATUS' -Status 'Error' -Error $_.Exception.Message
    }
}

# ============================================================================
# Step 2: Generate Tool Schema documentation
# ============================================================================
function Invoke-ToolSchemaGeneration {
    Write-Host "`n=== Generating Tool Schema documentation ===" -ForegroundColor Cyan

    $script = Join-Path $toolsDir 'generate-functiongemma-tool-docs.ps1'
    if (-not (Test-Path $script)) {
        Add-PipelineStep -Name 'ToolSchema' -Status 'Error' -Error 'Script not found'
        return
    }

    try {
        & $script
        Add-PipelineStep -Name 'ToolSchema' -Status 'Success' -Output (Join-Path $deployDir 'rust-functiongemma\TOOLS.md')
    }
    catch {
        Add-PipelineStep -Name 'ToolSchema' -Status 'Error' -Error $_.Exception.Message
    }
}

# ============================================================================
# Step 3: Generate Rust documentation (cargo doc)
# ============================================================================
function Invoke-RustDocGeneration {
    if ($SkipRust) {
        Add-PipelineStep -Name 'RustDocs' -Status 'Skipped'
        return
    }

    Write-Host "`n=== Generating Rust documentation ===" -ForegroundColor Cyan

    $workspaces = @(
        (Join-Path $repoRoot 'Native\pcai_core'),
        (Join-Path $repoRoot 'Native\pcai_core\pcai_inference'),
        (Join-Path $repoRoot 'Deploy\rust-functiongemma-runtime'),
        (Join-Path $repoRoot 'Deploy\rust-functiongemma-train')
    )

    $cudaRequired = @(
        (Join-Path $repoRoot 'Deploy\rust-functiongemma-train')
    )
    $cudaAvailable = ($script:CudaInfo -and $script:CudaInfo.Found)

    $found = $false
    foreach ($rustWorkspace in $workspaces) {
        if (-not (Test-Path $rustWorkspace)) {
            continue
        }
        $found = $true
        if (($cudaRequired -contains $rustWorkspace) -and (-not $cudaAvailable)) {
            Add-PipelineStep -Name 'RustDocs' -Status 'Skipped' -Error "$rustWorkspace requires CUDA"
            continue
        }

        try {
            Push-Location $rustWorkspace
            $cargoDoc = cargo doc --no-deps --document-private-items 2>&1
            Pop-Location

            if ($LASTEXITCODE -eq 0) {
                $docDir = if ($env:CARGO_TARGET_DIR) { "$env:CARGO_TARGET_DIR\doc" } else { "$rustWorkspace\target\doc" }
                Add-PipelineStep -Name 'RustDocs' -Status 'Success' -Output "$rustWorkspace -> $docDir"
            } else {
                Add-PipelineStep -Name 'RustDocs' -Status 'Warning' -Error ("$($rustWorkspace): " + ($cargoDoc | Select-Object -Last 5 | Out-String))
            }
        }
        catch {
            Pop-Location -ErrorAction SilentlyContinue
            Add-PipelineStep -Name 'RustDocs' -Status 'Error' -Error ("$($rustWorkspace): " + $_.Exception.Message)
        }
    }

    if (-not $found) {
        Add-PipelineStep -Name 'RustDocs' -Status 'Warning' -Error 'No Rust workspaces found'
    }
}

# ============================================================================
# Step 4: Generate PowerShell module documentation
# ============================================================================
function Invoke-PowerShellDocGeneration {
    Write-Host "`n=== Generating PowerShell documentation ===" -ForegroundColor Cyan

    $modules = Get-ChildItem -Path (Join-Path $repoRoot 'Modules') -Directory -ErrorAction SilentlyContinue
    $apiSignatures = @()

    foreach ($module in $modules) {
        $psd1 = Join-Path $module.FullName "$($module.Name).psd1"
        if (Test-Path $psd1) {
            try {
                $manifest = Import-PowerShellDataFile $psd1
                $exports = @($manifest.FunctionsToExport) | Where-Object { $_ -and $_ -ne '*' }

                foreach ($fn in $exports) {
                    $apiSignatures += [PSCustomObject]@{
                        Module = $module.Name
                        Function = $fn
                        Type = 'Exported'
                    }
                }
            }
            catch {
                # Skip invalid manifests
            }
        }
    }

    $outputPath = Join-Path $reportsDir 'POWERSHELL_EXPORTS.json'
    $apiSignatures | ConvertTo-Json -Depth 5 | Set-Content -Path $outputPath -Encoding UTF8
    Add-PipelineStep -Name 'PowerShellDocs' -Status 'Success' -Output $outputPath
}

# ============================================================================
# Step 4b: Generate API signature alignment report
# ============================================================================
function Invoke-ApiSignatureReport {
    Write-Host "`n=== Generating API signature report ===" -ForegroundColor Cyan

    $script = Join-Path $toolsDir 'generate-api-signature-report.ps1'
    if (-not (Test-Path $script)) {
        Add-PipelineStep -Name 'ApiSignatures' -Status 'Warning' -Error 'Script not found'
        return
    }

    try {
        & $script -RepoRoot $repoRoot
        Add-PipelineStep -Name 'ApiSignatures' -Status 'Success' -Output (Join-Path $reportsDir 'API_SIGNATURE_REPORT.md')
    }
    catch {
        Add-PipelineStep -Name 'ApiSignatures' -Status 'Error' -Error $_.Exception.Message
    }
}

# ============================================================================
# Step 4c: Generate Tools catalog
# ============================================================================
function Invoke-ToolsCatalogGeneration {
    Write-Host "`n=== Generating Tools catalog ===" -ForegroundColor Cyan

    $script = Join-Path $toolsDir 'generate-tools-catalog.ps1'
    if (-not (Test-Path $script)) {
        Add-PipelineStep -Name 'ToolsCatalog' -Status 'Warning' -Error 'Script not found'
        return
    }

    try {
        & $script
        Add-PipelineStep -Name 'ToolsCatalog' -Status 'Success' -Output (Join-Path $reportsDir 'TOOLS_CATALOG.md')
    }
    catch {
        Add-PipelineStep -Name 'ToolsCatalog' -Status 'Error' -Error $_.Exception.Message
    }
}

# ============================================================================
# Step 5: Generate FunctionGemma training data
# ============================================================================
function Invoke-TrainingDataGeneration {
    if ($SkipTraining -or $Mode -eq 'DocsOnly') {
        Add-PipelineStep -Name 'TrainingData' -Status 'Skipped'
        return
    }

    Write-Host "`n=== Generating FunctionGemma router dataset ===" -ForegroundColor Cyan

    $script = Join-Path $toolsDir 'prepare-functiongemma-router-data.ps1'
    if (-not (Test-Path $script)) {
        Add-PipelineStep -Name 'TrainingData' -Status 'Error' -Error 'prepare-functiongemma-router-data.ps1 not found'
        return
    }

    try {
        if (-not $UseNativeRouter -and -not ($script:CudaInfo -and $script:CudaInfo.Found)) {
            Add-PipelineStep -Name 'TrainingData' -Status 'Warning' -Error 'CUDA not detected; skipping router dataset generation (rust-functiongemma-train requires CUDA)'
            return
        }

        $routerParams = @{
            MaxCases = $RouterMaxCases
        }
        if ($UseNativeRouter) { $routerParams.UseNative = $true }
        if ($NoToolCoverage) { $routerParams.NoToolCoverage = $true }

        & $script @routerParams
        if ($LASTEXITCODE -ne 0) {
            Add-PipelineStep -Name 'TrainingData' -Status 'Error' -Error "Router dataset generation failed (exit $LASTEXITCODE)"
            return
        }

        $datasetPath = Join-Path $deployDir 'rust-functiongemma-train\data\rust_router_train.jsonl'
        $vectorsPath = Join-Path $deployDir 'rust-functiongemma-train\data\test_vectors.json'
        $outputLabel = "dataset: $datasetPath | vectors: $vectorsPath"
        Add-PipelineStep -Name 'TrainingData' -Status 'Success' -Output $outputLabel
    }
    catch {
        Add-PipelineStep -Name 'TrainingData' -Status 'Error' -Error $_.Exception.Message
    }
}

# ============================================================================
# Step 6: Validate training data format
# ============================================================================
function Invoke-TrainingDataValidation {
    if ($Mode -eq 'DocsOnly') {
        Add-PipelineStep -Name 'Validation' -Status 'Skipped'
        return
    }

    Write-Host "`n=== Validating training data ===" -ForegroundColor Cyan

    $datasetPath = Join-Path $deployDir 'rust-functiongemma-train\data\rust_router_train.jsonl'
    $vectorsPath = Join-Path $deployDir 'rust-functiongemma-train\data\test_vectors.json'

    if (-not (Test-Path $datasetPath)) {
        Add-PipelineStep -Name 'Validation' -Status 'Warning' -Error 'Router dataset JSONL not found'
        return
    }

    $errors = @()
    $lineNum = 0

    $firstLine = Get-Content $datasetPath -TotalCount 1
    if (-not $firstLine) {
        $errors += "Router dataset is empty: $datasetPath"
    } else {
        try {
            $obj = $firstLine | ConvertFrom-Json
            if (-not ($obj.PSObject.Properties.Name -contains 'messages')) {
                $errors += "Router dataset missing 'messages' field"
            }
            if (-not ($obj.PSObject.Properties.Name -contains 'tools')) {
                $errors += "Router dataset missing 'tools' field"
            }
        }
        catch {
            $errors += "Router dataset invalid JSON: $($_.Exception.Message)"
        }
    }

    if (Test-Path $vectorsPath) {
        try {
            $vectors = Get-Content $vectorsPath | ConvertFrom-Json
            if (-not $vectors -or $vectors.Count -lt 1) { $errors += "Tool test vectors empty" }
            if ($vectors -and $vectors.Count -gt 0) {
                $props = $vectors[0].PSObject.Properties.Name
                if (-not ($props -contains 'tool')) { $errors += "Tool test vector missing 'tool' key" }
                if (-not ($props -contains 'arguments')) { $errors += "Tool test vector missing 'arguments' key" }
            }
        }
        catch {
            $errors += "Tool test vectors invalid JSON: $($_.Exception.Message)"
        }
    }

    if ($errors.Count -eq 0) {
        Add-PipelineStep -Name 'Validation' -Status 'Success' -Output "Router dataset + vectors validated"
    } else {
        Add-PipelineStep -Name 'Validation' -Status 'Error' -Error ($errors | Select-Object -First 5 | Out-String)
    }
}

# ============================================================================
# Step 7: Generate pipeline summary report
# ============================================================================
function Invoke-PipelineSummary {
    Write-Host "`n=== Generating pipeline summary ===" -ForegroundColor Cyan

    $pipelineState.EndTime = Get-Date
    $pipelineState.Duration = ($pipelineState.EndTime - $pipelineState.StartTime).ToString('mm\:ss')

    $summary = [PSCustomObject]@{
        generated = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
        mode = $Mode
        duration = $pipelineState.Duration
        steps = $pipelineState.Steps
        outputs = $pipelineState.Outputs
        errors = $pipelineState.Errors
        success = ($pipelineState.Errors.Count -eq 0)
    }

    # Write JSON report
    $jsonPath = Join-Path $reportsDir 'DOC_PIPELINE_REPORT.json'
    $summary | ConvertTo-Json -Depth 5 | Set-Content -Path $jsonPath -Encoding UTF8

    # Write Markdown summary
    if ($OutputFormat -in @('Markdown', 'Both')) {
        $md = @"
# Documentation Pipeline Report

Generated: $($summary.generated)
Mode: $($summary.mode)
Duration: $($summary.duration)
Status: $(if ($summary.success) { '✅ Success' } else { '❌ Errors' })

## Steps

| Step | Status | Output |
|------|--------|--------|
$($pipelineState.Steps | ForEach-Object { "| $($_.Name) | $($_.Status) | $($_.Output) |" } | Out-String)

## Outputs

$($pipelineState.Outputs | ForEach-Object { "- ``$_``" } | Out-String)

$(if ($pipelineState.Errors.Count -gt 0) {
"## Errors

$($pipelineState.Errors | ForEach-Object { "- $_" } | Out-String)"
})
"@
        $mdPath = Join-Path $reportsDir 'DOC_PIPELINE_REPORT.md'
        $md | Set-Content -Path $mdPath -Encoding UTF8
    }

    Add-PipelineStep -Name 'Summary' -Status 'Success' -Output $jsonPath
}

# ============================================================================
# Main execution
# ============================================================================
Write-Host @"

╔══════════════════════════════════════════════════════════════════╗
║  PC_AI Documentation Pipeline                                     ║
║  Mode: $Mode
╚══════════════════════════════════════════════════════════════════╝

"@ -ForegroundColor Cyan

switch ($Mode) {
    'Full' {
        Invoke-DocStatusGeneration
        Invoke-ToolSchemaGeneration
        Invoke-RustDocGeneration
        Invoke-PowerShellDocGeneration
        Invoke-ApiSignatureReport
        Invoke-ToolsCatalogGeneration
        Invoke-TrainingDataGeneration
        Invoke-TrainingDataValidation
    }
    'DocsOnly' {
        Invoke-DocStatusGeneration
        Invoke-ToolSchemaGeneration
        Invoke-RustDocGeneration
        Invoke-PowerShellDocGeneration
        Invoke-ApiSignatureReport
        Invoke-ToolsCatalogGeneration
    }
    'TrainingOnly' {
        Invoke-TrainingDataGeneration
        Invoke-TrainingDataValidation
    }
    'Validate' {
        Invoke-TrainingDataValidation
    }
}

Invoke-PipelineSummary

# Final status
Write-Host "`n" -NoNewline
if ($pipelineState.Errors.Count -eq 0) {
    Write-Host "✅ Pipeline completed successfully" -ForegroundColor Green
} else {
    Write-Host "❌ Pipeline completed with $($pipelineState.Errors.Count) error(s)" -ForegroundColor Red
}
Write-Host "Duration: $($pipelineState.Duration)" -ForegroundColor Gray
Write-Host "Reports: $reportsDir" -ForegroundColor Gray
