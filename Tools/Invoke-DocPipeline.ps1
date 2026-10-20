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

.PARAMETER RustWorkspaceRoots
    Select explicit Cargo roots, relative to this repository or absolute.
    Requested missing roots and failed generators cause a failed pipeline.

.PARAMETER AllowExternalRustRoots
    Opt in to explicitly selected Cargo workspaces outside this repository.

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
    [switch]$NoToolCoverage,
    [string[]]$RustWorkspaceRoots,
    [switch]$AllowExternalRustRoots
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# Paths
$repoRoot = Split-Path -Parent $PSScriptRoot
$reportsDir = Join-Path $repoRoot 'Reports'
$deployDir = Join-Path $repoRoot 'Deploy'
$toolsDir = $PSScriptRoot

# Ensure output directories exist
@($reportsDir, (Join-Path $deployDir 'rust-functiongemma')) | ForEach-Object {
    if (-not (Test-Path $_)) { New-Item -ItemType Directory -Path $_ -Force | Out-Null }
}

# Snapshot options used by nested generators and the dot-sourced library.
$pipelineOptions = @{
    OutputFormat = $OutputFormat; SkipRust = $SkipRust; SkipTraining = $SkipTraining
    UseNativeRouter = $UseNativeRouter; RouterMaxCases = $RouterMaxCases; NoToolCoverage = $NoToolCoverage
    RustWorkspaceRoots = $RustWorkspaceRoots; AllowExternalRustRoots = $AllowExternalRustRoots
}
if ($Force) { Write-Verbose 'Generators overwrite their own reports; -Force is retained for CLI compatibility.' }
# Pipeline state
$pipelineState = [PSCustomObject]@{
    StartTime = Get-Date
    EndTime   = $null
    Duration  = ''
    Mode      = $Mode
    Steps     = @()
    Errors    = @()
    Warnings  = @()
    Outputs   = @()
}

function Add-PipelineStep {
    param([string]$Name, [string]$Status, [string]$Output = '', [Alias('Error')][string]$ErrorMessage = '')
    $step = [PSCustomObject]@{
        Name      = $Name
        Status    = $Status
        Output    = $Output
        Error     = $ErrorMessage
        Timestamp = Get-Date -Format 'HH:mm:ss'
    }
    $pipelineState.Steps += $step
    if ($Output) { $pipelineState.Outputs += $Output }
    # -Error carries the detail text for BOTH Warning and Error steps, so
    # routing it all into .Errors made the summary claim "5 error(s)" when only
    # one step had actually failed. Bucket by Status instead.
    if ($ErrorMessage) {
        if ($Status -eq 'Error') { $pipelineState.Errors += "${Name}: $ErrorMessage" }
        else { $pipelineState.Warnings += "${Name}: $ErrorMessage" }
    }

    Write-Information "[$($step.Timestamp)] $Name : $Status" -InformationAction Continue
}

# Initialize CMake environment (fix stale CMAKE_ROOT/CMAKE_PREFIX_PATH)
$script:CmakeInfo = $null
$cmakeHelper = Join-Path $toolsDir 'Initialize-CmakeEnvironment.ps1'
if (Test-Path $cmakeHelper) {
    . $cmakeHelper
    $script:CmakeInfo = Initialize-CmakeEnvironment -Quiet
    if ($script:CmakeInfo.Found) {
        Add-PipelineStep -Name 'CmakeEnv' -Status 'Success' -Output "CMAKE_ROOT=$($script:CmakeInfo.CmakeRoot)"
    }
    else {
        Add-PipelineStep -Name 'CmakeEnv' -Status 'Warning' -Error 'cmake.exe not found; CMake-dependent docs may fail'
    }
}
else {
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
    }
    else {
        Add-PipelineStep -Name 'CudaEnv' -Status 'Warning' -Error 'CUDA not detected; GPU-only steps will be skipped'
        $env:LLAMA_CUDA = '0'
    }
}
else {
    Add-PipelineStep -Name 'CudaEnv' -Status 'Warning' -Error "CUDA helper not found at $cudaHelper"
    $env:LLAMA_CUDA = '0'
}

# ============================================================================
# Step 1: Generate DOC_STATUS report (TODO/FIXME/DEPRECATED markers)
# ============================================================================
function Invoke-DocStatusGeneration {
    Write-Information "`n=== Generating DOC_STATUS report ===" -InformationAction Continue

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
    Write-Information "`n=== Generating Tool Schema documentation ===" -InformationAction Continue

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
    if ($pipelineOptions.SkipRust) {
        Add-PipelineStep -Name 'RustDocs' -Status 'Skipped'
        return
    }

    Write-Information "`n=== Generating Rust documentation ===" -InformationAction Continue

    try {
        $requestedRoots = $pipelineOptions.RustWorkspaceRoots
        $allowExternal = $pipelineOptions.AllowExternalRustRoots
        . (Join-Path $toolsDir 'generate-auto-docs.ps1') -RepoRoot $repoRoot -LibraryOnly
        $workspaces = @(Get-DocumentationWorkspace -Repository $repoRoot -Roots $requestedRoots -AllowExternal:$allowExternal -DefaultRoots @('Native/pcai_core', 'Deploy/rust-functiongemma-runtime', 'Deploy/rust-functiongemma-train'))
        if ($workspaces.Count -eq 0) { throw 'No Rust workspaces found for requested documentation build' }
        foreach ($rustWorkspace in $workspaces) {
            if (-not $requestedRoots -and $rustWorkspace -eq (Join-Path $repoRoot 'Deploy/rust-functiongemma-train') -and -not ($script:CudaInfo -and $script:CudaInfo.Found)) {
                Add-PipelineStep -Name 'RustDocs' -Status 'Skipped' -Error "$rustWorkspace requires CUDA"
                continue
            }
            try {
                $result = Invoke-RustDocumentation -Workspace $rustWorkspace -Build -DocumentPrivateItems
                Add-PipelineStep -Name 'RustDocs' -Status 'Success' -Output ($result.DocIndexes -join '; ')
            }
            catch {
                Add-PipelineStep -Name 'RustDocs' -Status 'Error' -Error $_.Exception.Message
            }
        }
    }
    catch {
        Add-PipelineStep -Name 'RustDocs' -Status 'Error' -Error $_.Exception.Message
    }
}
# ============================================================================
# Step 4: Generate PowerShell module documentation
# ============================================================================
function Invoke-PowerShellDocGeneration {
    Write-Information "`n=== Generating PowerShell documentation ===" -InformationAction Continue

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
                        Module   = $module.Name
                        Function = $fn
                        Type     = 'Exported'
                    }
                }
            }
            catch {
                Add-PipelineStep -Name 'PowerShellDocs' -Status 'Error' -Error $_.Exception.Message
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
    Write-Information "`n=== Generating API signature report ===" -InformationAction Continue

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
    Write-Information "`n=== Generating Tools catalog ===" -InformationAction Continue

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
# Step 4d: Litho AST extraction (tree-sitter based)
# ============================================================================
function Invoke-LithoExtraction {
    Write-Information "`n=== Litho AST Extraction ===" -InformationAction Continue

    $lithoExe = Get-Command litho -ErrorAction SilentlyContinue
    if (-not $lithoExe) {
        $lithoName = if ([IO.Path]::DirectorySeparatorChar -eq '\') { 'litho.exe' } else { 'litho' }
        $lithoPath = Join-Path $HOME (Join-Path 'bin' $lithoName)
        if (Test-Path $lithoPath) { $lithoExe = $lithoPath }
    }

    if (-not $lithoExe) {
        Add-PipelineStep -Name 'LithoExtract' -Status 'Skipped' -Error 'litho binary not found'
        return
    }

    try {
        $lithoCmd = if ($lithoExe -is [System.Management.Automation.ApplicationInfo]) { $lithoExe.Source } else { $lithoExe }
        & $lithoCmd extract $repoRoot --format json | Out-File (Join-Path $reportsDir 'LITHO_EXTRACT.json') -Encoding utf8
        & $lithoCmd extract $repoRoot --format summary | Out-File (Join-Path $reportsDir 'LITHO_EXTRACT_SUMMARY.md') -Encoding utf8
        Add-PipelineStep -Name 'LithoExtract' -Status 'Success' -Output (Join-Path $reportsDir 'LITHO_EXTRACT.json')
    }
    catch {
        Add-PipelineStep -Name 'LithoExtract' -Status 'Error' -Error $_.Exception.Message
    }
}

# ============================================================================
# Step 4e: Litho documentation generation (optional, requires codex-cli)
# ============================================================================
function Invoke-LithoDocGeneration {
    if ($Mode -ne 'Full') { return }

    Write-Information "`n=== Litho Documentation Generation ===" -InformationAction Continue

    $lithoExe = Get-Command litho -ErrorAction SilentlyContinue
    if (-not $lithoExe) {
        $lithoName = if ([IO.Path]::DirectorySeparatorChar -eq '\') { 'litho.exe' } else { 'litho' }
        $lithoPath = Join-Path $HOME (Join-Path 'bin' $lithoName)
        if (Test-Path $lithoPath) { $lithoExe = $lithoPath }
    }

    if (-not $lithoExe) {
        Add-PipelineStep -Name 'LithoDocs' -Status 'Skipped' -Error 'litho binary not found'
        return
    }

    $lithoDocsDir = Join-Path $repoRoot 'docs\auto\litho'
    if (-not (Test-Path $lithoDocsDir)) { New-Item -ItemType Directory -Path $lithoDocsDir -Force | Out-Null }

    try {
        $lithoCmd = if ($lithoExe -is [System.Management.Automation.ApplicationInfo]) { $lithoExe.Source } else { $lithoExe }
        & $lithoCmd generate $repoRoot --provider codex --output $lithoDocsDir
        Add-PipelineStep -Name 'LithoDocs' -Status 'Success' -Output $lithoDocsDir
    }
    catch {
        Add-PipelineStep -Name 'LithoDocs' -Status 'Warning' -Error $_.Exception.Message
    }
}

# ============================================================================
# Step 5: Generate FunctionGemma training data
# ============================================================================
function Invoke-TrainingDataGeneration {
    if ($pipelineOptions.SkipTraining -or $Mode -eq 'DocsOnly') {
        Add-PipelineStep -Name 'TrainingData' -Status 'Skipped'
        return
    }

    Write-Information "`n=== Generating FunctionGemma router dataset ===" -InformationAction Continue

    $script = Join-Path $toolsDir 'prepare-functiongemma-router-data.ps1'
    if (-not (Test-Path $script)) {
        Add-PipelineStep -Name 'TrainingData' -Status 'Error' -Error 'prepare-functiongemma-router-data.ps1 not found'
        return
    }

    try {
        if (-not $pipelineOptions.UseNativeRouter -and -not ($script:CudaInfo -and $script:CudaInfo.Found)) {
            Add-PipelineStep -Name 'TrainingData' -Status 'Warning' -Error 'CUDA not detected; skipping router dataset generation (rust-functiongemma-train requires CUDA)'
            return
        }

        $routerParams = @{
            MaxCases = $pipelineOptions.RouterMaxCases
        }
        if ($pipelineOptions.UseNativeRouter) { $routerParams.UseNative = $true }
        if ($pipelineOptions.NoToolCoverage) { $routerParams.NoToolCoverage = $true }

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

    Write-Information "`n=== Validating training data ===" -InformationAction Continue

    $datasetPath = Join-Path $deployDir 'rust-functiongemma-train\data\rust_router_train.jsonl'
    $vectorsPath = Join-Path $deployDir 'rust-functiongemma-train\data\test_vectors.json'

    if (-not (Test-Path $datasetPath)) {
        Add-PipelineStep -Name 'Validation' -Status 'Warning' -Error 'Router dataset JSONL not found'
        return
    }

    $errors = @()


    $firstLine = Get-Content $datasetPath -TotalCount 1
    if (-not $firstLine) {
        $errors += "Router dataset is empty: $datasetPath"
    }
    else {
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
    }
    else {
        Add-PipelineStep -Name 'Validation' -Status 'Error' -Error ($errors | Select-Object -First 5 | Out-String)
    }
}

# ============================================================================
# Step 7: Generate pipeline summary report
# ============================================================================
function Invoke-PipelineSummary {
    Write-Information "`n=== Generating pipeline summary ===" -InformationAction Continue

    $pipelineState.EndTime = Get-Date
    $pipelineState.Duration = ($pipelineState.EndTime - $pipelineState.StartTime).ToString('mm\:ss')

    $summary = [PSCustomObject]@{
        generated = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
        mode      = $Mode
        duration  = $pipelineState.Duration
        steps     = $pipelineState.Steps
        outputs   = $pipelineState.Outputs
        errors    = $pipelineState.Errors
        success   = ($pipelineState.Errors.Count -eq 0)
    }

    # Write JSON report
    $jsonPath = Join-Path $reportsDir 'DOC_PIPELINE_REPORT.json'
    $summary | ConvertTo-Json -Depth 5 | Set-Content -Path $jsonPath -Encoding UTF8

    # Write Markdown summary
    if ($pipelineOptions.OutputFormat -in @('Markdown', 'Both')) {
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
Write-Information @"

╔══════════════════════════════════════════════════════════════════╗
║  PC_AI Documentation Pipeline                                     ║
║  Mode: $Mode
╚══════════════════════════════════════════════════════════════════╝

"@ -InformationAction Continue

switch ($Mode) {
    'Full' {
        Invoke-DocStatusGeneration
        Invoke-ToolSchemaGeneration
        Invoke-RustDocGeneration
        Invoke-PowerShellDocGeneration
        Invoke-ApiSignatureReport
        Invoke-ToolsCatalogGeneration
        Invoke-LithoExtraction
        Invoke-LithoDocGeneration
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
        Invoke-LithoExtraction
        Invoke-LithoDocGeneration
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
Write-Information "" -InformationAction Continue
if ($pipelineState.Warnings.Count -gt 0) {
    Write-Information "$($pipelineState.Warnings.Count) warning(s):" -InformationAction Continue
    $pipelineState.Warnings | ForEach-Object { Write-Information "  - $_" -InformationAction Continue }
}
Write-Information "Duration: $($pipelineState.Duration)" -InformationAction Continue
Write-Information "Reports: $reportsDir" -InformationAction Continue

# The pipeline used to print the failure banner and still exit 0, so CI stayed
# green while doc generation was broken. Errors now set a real exit code;
# warnings (skipped optional generators, absent CUDA/Litho) do not.
if ($pipelineState.Errors.Count -eq 0) {
    Write-Information "OK Pipeline completed successfully" -InformationAction Continue
    exit 0
}

Write-Information "FAIL Pipeline completed with $($pipelineState.Errors.Count) error(s):" -InformationAction Continue
$pipelineState.Errors | ForEach-Object { Write-Information "  - $_" -InformationAction Continue }
exit 1
