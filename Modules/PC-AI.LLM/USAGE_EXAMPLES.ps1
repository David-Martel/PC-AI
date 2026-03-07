#Requires -PSEdition Core
<#
.SYNOPSIS
    Usage examples for PC-AI.LLM PowerShell module

.DESCRIPTION
    Demonstrates how to use the PC-AI.LLM module for pcai-inference integration
    with PC diagnostics and analysis
#>

# Import the module
Import-Module "$PSScriptRoot\PC-AI.LLM.psd1" -Force

Write-Host "=== PC-AI.LLM Module Usage Examples ===" -ForegroundColor Cyan
Write-Host ""

# Example 1: Check LLM Status
Write-Host "Example 1: Check LLM Status" -ForegroundColor Yellow
Write-Host "Command: Get-LLMStatus -TestConnection" -ForegroundColor Gray
$status = Get-LLMStatus -TestConnection
Write-Host "pcai-inference Available: $($status.PcaiInference.ApiConnected)" -ForegroundColor Green
Write-Host "Models Available: $($status.PcaiInference.Models.Count)" -ForegroundColor Green
Write-Host "Default Model: $($status.PcaiInference.DefaultModel)" -ForegroundColor Green
Write-Host ""

# Example 2: View Configuration
Write-Host "Example 2: View Current Configuration" -ForegroundColor Yellow
Write-Host "Command: Set-LLMConfig -ShowConfig" -ForegroundColor Gray
Set-LLMConfig -ShowConfig
Write-Host ""

# Example 3: Send a Simple Request
Write-Host "Example 3: Send a Simple Request" -ForegroundColor Yellow
Write-Host "Command: Send-OllamaRequest -Prompt 'What is PowerShell?' -Model 'pcai-inference'" -ForegroundColor Gray
Write-Host "Note: This will take 5-15 seconds depending on your hardware" -ForegroundColor Gray
try {
    $response = Send-OllamaRequest -Prompt "Explain what PowerShell is in one sentence." -Model "pcai-inference"
    Write-Host "Response: $($response.Response)" -ForegroundColor Green
    Write-Host "Duration: $($response.RequestDurationSeconds) seconds" -ForegroundColor Green
    Write-Host "Tokens/sec: $($response.TokensPerSecond)" -ForegroundColor Green
}
catch {
    Write-Host "Error: $_" -ForegroundColor Red
}
Write-Host ""

# Example 4: Interactive Chat (commented out for automated testing)
Write-Host "Example 4: Interactive Chat" -ForegroundColor Yellow
Write-Host "Command: Invoke-LLMChat -Interactive -Model 'pcai-inference'" -ForegroundColor Gray
Write-Host "(Uncomment to run interactive chat session)" -ForegroundColor Gray
# Invoke-LLMChat -Interactive -Model "pcai-inference"
Write-Host ""

# Example 5: Single-Shot Chat
Write-Host "Example 5: Single-Shot Chat with System Prompt" -ForegroundColor Yellow
Write-Host "Command: Invoke-LLMChat -Message 'Explain cmdlets' -System 'You are a PowerShell expert'" -ForegroundColor Gray
try {
    $chatResponse = Invoke-LLMChat -Message "What are cmdlets in PowerShell?" -System "You are a PowerShell expert. Be concise."
    Write-Host "Response: $($chatResponse.Response)" -ForegroundColor Green
}
catch {
    Write-Host "Error: $_" -ForegroundColor Red
}
Write-Host ""

# Example 6: Analyze Diagnostic Report
Write-Host "Example 6: Analyze PC Diagnostic Report" -ForegroundColor Yellow
Write-Host "Command: Invoke-PCDiagnosis -DiagnosticReportPath 'path\to\report.txt'" -ForegroundColor Gray
Write-Host "(Requires existing diagnostic report from PC-AI.ps1 diagnose all)" -ForegroundColor Gray
$reportPath = Join-Path -Path ([Environment]::GetFolderPath('Desktop')) -ChildPath 'Hardware-Diagnostics-Report.txt'
if (Test-Path $reportPath) {
    Write-Host "Found diagnostic report at: $reportPath" -ForegroundColor Green
    Write-Host "Run: Invoke-PCDiagnosis -DiagnosticReportPath '$reportPath' -SaveReport" -ForegroundColor Cyan
}
else {
    Write-Host "No diagnostic report found. Run PC-AI.ps1 diagnose all first." -ForegroundColor Yellow
}
Write-Host ""

# Example 7: Change Default Model
Write-Host "Example 7: Change Default Model" -ForegroundColor Yellow
Write-Host "Command: Set-LLMConfig -DefaultModel 'pcai-inference'" -ForegroundColor Gray
Write-Host "(Example only - not executing)" -ForegroundColor Gray
# Set-LLMConfig -DefaultModel "pcai-inference"
Write-Host ""

# Example 8: Get Help
Write-Host "Example 8: Get Help for Functions" -ForegroundColor Yellow
Write-Host "Commands:" -ForegroundColor Gray
Write-Host "  Get-Help Get-LLMStatus -Full" -ForegroundColor Gray
Write-Host "  Get-Help Send-OllamaRequest -Examples" -ForegroundColor Gray
Write-Host "  Get-Help Invoke-LLMChat -Detailed" -ForegroundColor Gray
Write-Host "  Get-Help Invoke-PCDiagnosis -Full" -ForegroundColor Gray
Write-Host "  Get-Help Set-LLMConfig -Examples" -ForegroundColor Gray
Write-Host ""

Write-Host "=== Examples Complete ===" -ForegroundColor Cyan
Write-Host "For full help on any function, use: Get-Help <FunctionName> -Full" -ForegroundColor Green
