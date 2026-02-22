#Requires -Version 5.1
# Accuracy Validation Script for Documentation Generation
# Checks: TOOLS.md, training_data.jsonl, DOC_STATUS.md

$ErrorActionPreference = 'Continue'
$repoRoot = if ($env:PCAI_ROOT -and (Test-Path $env:PCAI_ROOT)) {
    $env:PCAI_ROOT
} else {
    Split-Path -Parent $PSScriptRoot
}

$report = @{
    timestamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    tools_md_accuracy = @{}
    training_data_accuracy = @{}
    doc_status_accuracy = @{}
}

# 1. TOOLS.md vs pcai-tools.json accuracy
Write-Host '=== TOOLS.MD ACCURACY CHECK ===' -ForegroundColor Cyan
$toolsJson = Get-Content (Join-Path $repoRoot 'Config\pcai-tools.json') -Raw | ConvertFrom-Json
$toolsMd = Get-Content (Join-Path $repoRoot 'Deploy\rust-functiongemma\TOOLS.md') -Raw

$schemaToolNames = @($toolsJson.tools | ForEach-Object { $_.function.name })
$docToolNames = @([regex]::Matches($toolsMd, '### (\w+)') | ForEach-Object { $_.Groups[1].Value })

$missing = @($schemaToolNames | Where-Object { $_ -notin $docToolNames })
$extra = @($docToolNames | Where-Object { $_ -notin $schemaToolNames })

$report.tools_md_accuracy.schema_count = $schemaToolNames.Count
$report.tools_md_accuracy.doc_count = $docToolNames.Count
$report.tools_md_accuracy.missing = $missing
$report.tools_md_accuracy.extra = $extra
$report.tools_md_accuracy.match = ($missing.Count -eq 0 -and $extra.Count -eq 0)

Write-Host "  Schema tools: $($schemaToolNames.Count)"
Write-Host "  Documented: $($docToolNames.Count)"
Write-Host "  Match: $($report.tools_md_accuracy.match)" -ForegroundColor $(if ($report.tools_md_accuracy.match) { 'Green' } else { 'Red' })

# 2. Training data accuracy
Write-Host "`n=== TRAINING DATA ACCURACY CHECK ===" -ForegroundColor Cyan
$trainingFile = Join-Path $repoRoot 'Deploy\rust-functiongemma-train\data\training_data.jsonl'
$lines = Get-Content $trainingFile
$validToolCalls = 0
$invalidToolCalls = 0
$noToolCorrect = 0
$invalidTools = @()

foreach ($line in $lines) {
    $item = $line | ConvertFrom-Json
    $lastMsg = $item.messages[-1]

    if ($lastMsg.tool_calls) {
        $toolName = $lastMsg.tool_calls[0].function.name
        if ($toolName -in $schemaToolNames) {
            $validToolCalls++
        } else {
            $invalidToolCalls++
            $invalidTools += $toolName
        }
    } elseif ($lastMsg.content -eq 'NO_TOOL') {
        $noToolCorrect++
    }
}

$report.training_data_accuracy.total_examples = $lines.Count
$report.training_data_accuracy.valid_tool_calls = $validToolCalls
$report.training_data_accuracy.invalid_tool_calls = $invalidToolCalls
$report.training_data_accuracy.invalid_tools = $invalidTools
$report.training_data_accuracy.no_tool_examples = $noToolCorrect

Write-Host "  Total examples: $($lines.Count)"
Write-Host "  Valid tool calls: $validToolCalls" -ForegroundColor Green
Write-Host "  Invalid tool calls: $invalidToolCalls" -ForegroundColor $(if ($invalidToolCalls -eq 0) { 'Green' } else { 'Red' })
Write-Host "  NO_TOOL examples: $noToolCorrect"

# 3. DOC_STATUS self-reference check
Write-Host "`n=== DOC_STATUS SELF-REFERENCE CHECK ===" -ForegroundColor Cyan
$docStatus = Get-Content (Join-Path $repoRoot 'Reports\DOC_STATUS.md')
$matchLines = $docStatus | Where-Object { $_ -match '^- ' }
$totalEntries = $matchLines.Count
$selfRefs = ($docStatus | Where-Object { $_ -match 'Reports[\\/]DOC_STATUS' }).Count
$contextRefs = ($docStatus | Where-Object { $_ -match '\.claude[\\/]context' }).Count
$tokenizerRefs = ($docStatus | Where-Object { $_ -match 'tokenizer\.json' }).Count

$report.doc_status_accuracy.total_entries = $totalEntries
$report.doc_status_accuracy.self_references = $selfRefs
$report.doc_status_accuracy.context_file_refs = $contextRefs
$report.doc_status_accuracy.tokenizer_refs = $tokenizerRefs
$report.doc_status_accuracy.false_positive_estimate = $selfRefs + $contextRefs + $tokenizerRefs
$report.doc_status_accuracy.true_entry_estimate = $totalEntries - $report.doc_status_accuracy.false_positive_estimate
$report.doc_status_accuracy.pollution_percentage = [math]::Round(($selfRefs / $totalEntries) * 100, 1)

Write-Host "  Total entries: $totalEntries"
Write-Host "  Self-references: $selfRefs (POLLUTION!)" -ForegroundColor Red
Write-Host "  Context file refs: $contextRefs"
Write-Host "  Tokenizer refs: $tokenizerRefs"
Write-Host "  Pollution: $($report.doc_status_accuracy.pollution_percentage)%" -ForegroundColor Yellow
Write-Host "  Estimated true entries: $($report.doc_status_accuracy.true_entry_estimate)"

# Output JSON report
Write-Host "`n=== ACCURACY VALIDATION REPORT ===" -ForegroundColor Yellow
$report | ConvertTo-Json -Depth 5
