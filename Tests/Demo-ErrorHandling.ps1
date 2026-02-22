#Requires -Version 5.1

<#
.SYNOPSIS
    Demonstrates the LLM error handling module capabilities
#>

Import-Module "$PSScriptRoot\..\Modules\PC-AI.LLM\PC-AI.LLM.psd1" -Force
. "$PSScriptRoot\..\Modules\PC-AI.LLM\Private\LLM-ErrorHandling.ps1"

Write-Host "`n=== Error Categorization Demo ===" -ForegroundColor Cyan

$examples = @(
    @{ Type = 'Rate Limit'; Exception = New-Object System.Exception('429: Too Many Requests') }
    @{ Type = 'Connection'; Exception = New-Object System.Net.WebException('Connection refused', [System.Net.WebExceptionStatus]::ConnectFailure) }
    @{ Type = 'Timeout'; Exception = New-Object System.Net.WebException('Timeout', [System.Net.WebExceptionStatus]::Timeout) }
    @{ Type = 'Server Error'; Exception = New-Object System.Exception('500: Internal Server Error') }
    @{ Type = 'Invalid Request'; Exception = New-Object System.Exception('400: Bad Request') }
)

foreach ($example in $examples) {
    $category = Get-LLMErrorCategory -Exception $example.Exception
    Write-Host "  $($example.Type): " -NoNewline
    Write-Host $category -ForegroundColor Yellow
}

Write-Host "`n=== Retry with Backoff Demo ===" -ForegroundColor Cyan

$script:attemptCount = 0
$result = Invoke-WithRetry -ScriptBlock {
    $script:attemptCount++
    Write-Host "  Attempt $script:attemptCount..." -ForegroundColor Gray
    if ($script:attemptCount -lt 3) {
        throw (New-Object System.Net.WebException('Connection refused', [System.Net.WebExceptionStatus]::ConnectFailure))
    }
    return 'Success!'
} -Verbose:$false

Write-Host '  Result: ' -NoNewline
Write-Host $result -ForegroundColor Green
Write-Host "  Total attempts: $script:attemptCount"

Write-Host "`n=== Error Report Demo ===" -ForegroundColor Cyan

$ex = New-Object System.Exception('Sample LLM API error')
$report = New-LLMErrorReport -Exception $ex -Operation 'Send-OllamaRequest' -Context @{
    Model   = 'llama3.2'
    ApiUrl  = 'http://mock-server:11434'
    Timeout = 30
}

Write-Host "  Category: $($report.Category)"
Write-Host "  Operation: $($report.Operation)"
Write-Host "  IsRetryable: $($report.IsRetryable)"
Write-Host "  Context: $($report.Context.Model) at $($report.Context.ApiUrl)"

Write-Host "`n=== All demonstrations completed ===" -ForegroundColor Green
