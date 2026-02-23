#Requires -Version 5.1

. (Join-Path $PSScriptRoot '..\Helpers\Resolve-TestRepoRoot.ps1')

$PcaiRoot = Resolve-TestRepoRoot -StartPath $PSScriptRoot
Import-Module (Join-Path $PcaiRoot 'Modules\PC-AI.LLM\PC-AI.LLM.psd1') -Force -ErrorAction Stop

Write-Host 'E2E VERIFICATION STARTING...' -ForegroundColor Cyan

# 1. Verify Configuration Loading
$status = Get-LLMStatus
Write-Host "Active Provider (from config): $($status.ActiveProvider)" -ForegroundColor Gray
Write-Host "Default Model: $($status.ActiveModel)" -ForegroundColor Gray

# 2. Connectivity Test
$llmConfig = Load-LLMConfig
$vllmUrl = if ($llmConfig.providers.functiongemma.baseUrl) { $llmConfig.providers.functiongemma.baseUrl } else { throw 'No functiongemma URL' }
if (Get-Command Get-LLMStatus -ErrorAction SilentlyContinue) {
    # Test vLLM (Router) directly
    try {
        $null = Invoke-RestMethod -Uri "$vllmUrl/v1/models" -Method Get -ErrorAction Stop
        Write-Host 'vLLM Check: PASSED' -ForegroundColor Green
    } catch {
        Write-Warning "vLLM not available at $vllmUrl. Ensure vLLM is running for E2E."
    }
}

# 3. Full Routed Chat Test (The Core Pathway)
# This test verifies:
# - Prompt enrichment (telemetry injection)
# - Tool selection (via Router)
# - Tool execution (via shared helper)
# - Narrative response (via primary LLM)

$testMessage = 'Scan the system for any USB devices with errors. Use the available tools.'
Write-Host "`nTesting Routed Chat: $testMessage" -ForegroundColor White

try {
    # Force absolute URL for local test to bypass resolution issues
    $routerUrl = $vllmUrl

    $result = Invoke-LLMChatRouted `
        -Message $testMessage `
        -Mode 'diagnose' `
        -Model $status.ActiveModel `
        -ExecuteTools `
        -RouterBaseUrl $routerUrl `
        -TimeoutSeconds 180

    Write-Host "`n--- E2E RESULT ---" -ForegroundColor Cyan
    Write-Host "Selected Provider: $($result.Provider)"
    if ($result.ToolCalls) {
        Write-Host "Tool Calls Made: $($result.ToolCalls -join ', ')" -ForegroundColor Yellow
    }
    Write-Host 'Response Summary: ' -NoNewline
    $summary = $result.Response.Substring(0, [math]::Min(200, $result.Response.Length)) + '...'
    Write-Host $summary -ForegroundColor Gray

    if ($result.JsonValid) {
        Write-Host 'JSON Output: VALID' -ForegroundColor Green
    } else {
        Write-Warning 'JSON Output: INVALID or missing in response'
    }
} catch {
    Write-Error "E2E Routed Chat FAILED: $_"
}

Write-Host "`nE2E VERIFICATION COMPLETE." -ForegroundColor Cyan
