#Requires -Version 7.0

. (Join-Path $PSScriptRoot '..\Helpers\Resolve-TestRepoRoot.ps1')

$script:PcaiRoot = Resolve-TestRepoRoot -StartPath $PSScriptRoot
$dllDir = Join-Path $script:PcaiRoot "Native\PcaiNative\bin\Release\net8.0\win-x64"
$toolsPath = Join-Path $script:PcaiRoot "Config\pcai-tools.json"
$llmConfigPath = Join-Path $script:PcaiRoot "Config\llm-config.json"

# Load PcaiNative
$wrapperPath = Join-Path $dllDir "PcaiNative.dll"
Add-Type -Path $wrapperPath

# Helper to verify connectivity
function Test-LlmConnectivity {
    param($url)
    try {
        $response = Invoke-RestMethod -Uri "$url/api/version" -Method Get -TimeoutSec 5
        return $true
    } catch {
        return $false
    }
}

$ollamaUrl = "http://localhost:11434"
$model = "llama3.1:latest"
if (Test-Path $llmConfigPath) {
    try {
        $llmConfig = Get-Content -Path $llmConfigPath -Raw | ConvertFrom-Json
        if ($llmConfig.providers.ollama.baseUrl) {
            $ollamaUrl = $llmConfig.providers.ollama.baseUrl
        }
        if ($llmConfig.providers.ollama.defaultModel) {
            $model = $llmConfig.providers.ollama.defaultModel
        }
    } catch {
        Write-Warning "Failed to parse llm-config.json: $_"
    }
}
if (-not (Test-LlmConnectivity $ollamaUrl)) {
    Write-Warning "Ollama not found at $ollamaUrl. Evaluation will be restricted."
    # Skip real E2E if no LLM
    return
}

Write-Host "REAL LLM EVALUATION STARTING..." -ForegroundColor Cyan

# Setup Orchestrator
$client = [PcaiNative.PcaiOpenAiClient]::new($ollamaUrl)
$psHost = [PcaiNative.PowerShellHost]::new()
$executor = [PcaiNative.ToolExecutor]::new($toolsPath, $psHost)
$orchestrator = [PcaiNative.ReActOrchestrator]::new($client, $executor, $model)

# Bind events for logging
$orchestrator.add_OnThought({ param($thought) Write-Host "Thought: $thought" -ForegroundColor Gray })
$orchestrator.add_OnToolCall({ param($name, $args) Write-Host "Tool Call: $name($args)" -ForegroundColor Yellow })
$orchestrator.add_OnToolResult({ param($name, $result) Write-Host "Tool Result: $result" -ForegroundColor Green })
$orchestrator.add_OnFinalAnswer({ param($answer) Write-Host "Final Answer: $answer" -ForegroundColor Cyan })
$orchestrator.add_OnError({ param($err) Write-Error "ReAct Error: $err" })

# Run a simple diagnostic question - forcing tool usage via system prompt or specific framing
$prompt = "Query the system for USB category information to identify devices with errors. Use the GetSystemInfo tool."
Write-Host "Question: $prompt" -ForegroundColor White

try {
    $orchestrator.RunAsync($prompt).GetAwaiter().GetResult()
} finally {
    $psHost.Dispose()
}

Write-Host "REAL LLM EVALUATION COMPLETE." -ForegroundColor Cyan
