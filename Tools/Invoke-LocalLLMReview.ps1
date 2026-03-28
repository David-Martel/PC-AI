#Requires -Version 7.0
<#
.SYNOPSIS
    Uses local LLM to perform code review, documentation, or test gap analysis.

.DESCRIPTION
    Leverages Ollama-hosted models for fast, local code analysis tasks that
    complement remote LLM agents (Claude, Codex, Jules). The local model handles
    quick triage (seconds) while expensive analysis goes to cloud agents.

.PARAMETER Task
    Review task type: CodeReview, Documentation, TestGap, CIPipeline, Triage.

.PARAMETER Path
    File or directory to analyze.

.PARAMETER Model
    Ollama model to use. Default: qwen2.5-coder:7b for review, :3b for docs.

.PARAMETER Format
    Output format: Text, Json, Markdown.

.EXAMPLE
    .\Invoke-LocalLLMReview.ps1 -Task CodeReview -Path Native/pcai_core/pcai_inference/src/ffi/mod.rs
    .\Invoke-LocalLLMReview.ps1 -Task Documentation -Path Modules/PC-AI.LLM/
    .\Invoke-LocalLLMReview.ps1 -Task Triage -Path Native/pcai_core/ -Format Json
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [ValidateSet('CodeReview', 'Documentation', 'TestGap', 'CIPipeline', 'Triage')]
    [string]$Task,

    [Parameter(Mandatory)]
    [string]$Path,

    [string]$Model,

    [ValidateSet('Text', 'Json', 'Markdown')]
    [string]$Format = 'Text',

    [int]$MaxTokens = 2048,
    [string]$OllamaUrl = 'http://127.0.0.1:11434'
)

$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $PSScriptRoot

# Default model selection based on task
if (-not $Model) {
    $Model = switch ($Task) {
        'CodeReview'    { 'qwen2.5-coder:7b' }
        'Documentation' { 'qwen2.5-coder:3b' }
        'TestGap'       { 'qwen2.5-coder:7b' }
        'CIPipeline'    { 'qwen2.5-coder:7b' }
        'Triage'        { 'qwen2.5-coder:3b' }
    }
}

# Resolve files
$resolvedPath = Join-Path $RepoRoot $Path
$files = if (Test-Path $resolvedPath -PathType Container) {
    Get-ChildItem -Path $resolvedPath -Recurse -Include '*.rs', '*.ps1', '*.cs', '*.yml' -File |
        Where-Object { $_.FullName -notmatch 'target|node_modules|\.pcai|vendor' } |
        Select-Object -First 10
} else {
    @(Get-Item -LiteralPath $resolvedPath)
}

# Task-specific prompt templates
$prompts = @{
    'CodeReview' = "Review this {lang} code. Find the top 5 issues (bugs, security, performance, missing error handling). For each: severity (high/medium/low), location, issue description, and fix suggestion. Be specific."
    'Documentation' = "Add inline documentation to every public function that lacks it. Use {doc_style}. Include parameter descriptions and usage examples. Return only the documented functions."
    'TestGap' = "Identify functions in this {lang} code that are NOT covered by tests. Write complete unit test functions for each gap. Use {test_framework}."
    'CIPipeline' = "Analyze this CI pipeline configuration. Identify: missing steps, parallelization opportunities, unhandled failures, and optimization suggestions."
    'Triage' = 'Scan this {lang} code for issues. Return ONLY a JSON array: [{{"severity":"high|medium|low","line":N,"issue":"description","fix":"suggestion"}}]'
}

$results = [System.Collections.Generic.List[PSObject]]::new()
$sw = [System.Diagnostics.Stopwatch]::StartNew()

foreach ($file in $files) {
    $code = Get-Content -LiteralPath $file.FullName -Raw -ErrorAction SilentlyContinue
    if (-not $code -or $code.Length -gt 8000) {
        $code = $code.Substring(0, [Math]::Min(8000, $code.Length))
    }

    $ext = $file.Extension
    $lang = switch ($ext) {
        '.rs'  { 'Rust' }
        '.ps1' { 'PowerShell' }
        '.cs'  { 'C#' }
        '.yml' { 'YAML' }
        default { 'code' }
    }
    $docStyle = if ($lang -eq 'Rust') { '/// doc comments with # Examples sections' }
                elseif ($lang -eq 'PowerShell') { 'comment-based help (.SYNOPSIS, .DESCRIPTION, .PARAMETER, .EXAMPLE)' }
                else { 'XML doc comments (///)' }
    $testFramework = if ($lang -eq 'Rust') { '#[test] functions in a #[cfg(test)] mod tests block' }
                     elseif ($lang -eq 'PowerShell') { 'Pester Describe/It blocks' }
                     else { 'xUnit [Fact] test methods' }

    $promptTemplate = $prompts[$Task]
    $prompt = $promptTemplate -replace '\{lang\}', $lang -replace '\{doc_style\}', $docStyle -replace '\{test_framework\}', $testFramework
    $fullPrompt = "$prompt`n`n``````$lang`n$code`n``````"

    $formatParam = if ($Task -eq 'Triage') { ', "format": "json"' } else { '' }
    $body = @{
        model   = $Model
        prompt  = $fullPrompt
        stream  = $false
        options = @{ num_predict = $MaxTokens; num_ctx = 8192; temperature = 0.2 }
    } | ConvertTo-Json -Depth 5

    try {
        $response = Invoke-RestMethod -Uri "$OllamaUrl/api/generate" -Method Post -Body $body -ContentType 'application/json' -TimeoutSec 120
        $tokS = if ($response.eval_duration -gt 0) {
            [math]::Round($response.eval_count / ($response.eval_duration / 1e9), 1)
        } else { 0 }

        $results.Add([PSCustomObject]@{
            File     = $file.Name
            Tokens   = $response.eval_count
            TokS     = $tokS
            Lines    = ($response.response -split "`n").Count
            Response = $response.response
        })
    } catch {
        $results.Add([PSCustomObject]@{
            File     = $file.Name
            Tokens   = 0
            TokS     = 0
            Lines    = 0
            Response = "ERROR: $($_.Exception.Message)"
        })
    }
}

$sw.Stop()

# Output
switch ($Format) {
    'Json' {
        [ordered]@{
            task       = $Task
            model      = $Model
            files      = $results.Count
            total_tok  = ($results | Measure-Object -Property Tokens -Sum).Sum
            duration_s = [math]::Round($sw.Elapsed.TotalSeconds, 1)
            results    = $results | ForEach-Object {
                [ordered]@{ file = $_.File; tokens = $_.Tokens; tok_s = $_.TokS; response = $_.Response }
            }
        } | ConvertTo-Json -Depth 5
    }
    'Markdown' {
        "## Local LLM $Task ($Model)`n"
        "| File | Tokens | tok/s | Lines |"
        "|------|--------|-------|-------|"
        foreach ($r in $results) {
            "| $($r.File) | $($r.Tokens) | $($r.TokS) | $($r.Lines) |"
        }
        "`n### Findings`n"
        foreach ($r in $results) {
            "#### $($r.File)`n"
            $r.Response
            ""
        }
    }
    default {
        Write-Host "`n=== Local LLM $Task ($Model) ===" -ForegroundColor Cyan
        Write-Host "Files: $($results.Count), Total: $(($results | Measure-Object -Property Tokens -Sum).Sum) tokens in $([math]::Round($sw.Elapsed.TotalSeconds, 1))s"
        Write-Host ""
        foreach ($r in $results) {
            Write-Host "--- $($r.File) ($($r.Tokens) tok, $($r.TokS) tok/s) ---" -ForegroundColor Yellow
            Write-Host $r.Response.Substring(0, [Math]::Min(1500, $r.Response.Length))
            if ($r.Response.Length -gt 1500) { Write-Host "... [$($r.Response.Length - 1500) more chars]" }
            Write-Host ""
        }
    }
}
