#Requires -Version 5.1

<#
.SYNOPSIS
    Error handling and retry logic for LLM API operations

.DESCRIPTION
    Provides error categorization, exponential backoff retry logic, and structured error reporting
    for LLM API operations with robust failure handling.
#>

enum LLMErrorCategory {
    Connectivity   # Cannot reach endpoint
    Timeout        # Request timed out
    RateLimited    # 429 or rate limit
    InvalidRequest # 400 bad request
    ServerError    # 5xx errors
    ParseError     # JSON/response parsing
    Unknown        # Unclassified
}

function Get-LLMErrorCategory {
    <#
    .SYNOPSIS
        Classifies an exception into an LLM error category
    .DESCRIPTION
        Analyzes exception type and message to determine the appropriate error category
        for retry logic and error handling decisions
    .PARAMETER Exception
        The exception to classify
    .OUTPUTS
        [LLMErrorCategory] The classified error category
    .EXAMPLE
        $category = Get-LLMErrorCategory -Exception $_.Exception
        if ($category -eq [LLMErrorCategory]::Connectivity) {
            # Handle connectivity error
        }
    #>
    [CmdletBinding()]
    [OutputType([LLMErrorCategory])]
    param(
        [Parameter(Mandatory)]
        [System.Exception]$Exception
    )

    $message = $Exception.Message.ToLower()
    $exceptionType = $Exception.GetType().FullName

    # Connectivity errors
    if ($exceptionType -eq 'System.Net.WebException') {
        $webEx = [System.Net.WebException]$Exception
        if ($webEx.Status -eq [System.Net.WebExceptionStatus]::ConnectFailure) {
            return [LLMErrorCategory]::Connectivity
        }
        if ($webEx.Status -eq [System.Net.WebExceptionStatus]::Timeout) {
            return [LLMErrorCategory]::Timeout
        }
        if ($webEx.Status -eq [System.Net.WebExceptionStatus]::ProtocolError) {
            return [LLMErrorCategory]::ServerError
        }
    }

    if ($exceptionType -eq 'System.Net.Sockets.SocketException') {
        return [LLMErrorCategory]::Connectivity
    }

    if ($message -match 'connection refused|unable to connect|no such host|connection reset|network is unreachable') {
        return [LLMErrorCategory]::Connectivity
    }

    # Timeout errors
    if ($message -match 'timeout|timed out|request timeout') {
        return [LLMErrorCategory]::Timeout
    }

    # Rate limit errors
    if ($message -match '429|too many requests|rate limit') {
        return [LLMErrorCategory]::RateLimited
    }

    # Invalid request errors
    if ($message -match '400|bad request|invalid model|invalid parameter|invalid input') {
        return [LLMErrorCategory]::InvalidRequest
    }

    # Server errors (5xx)
    if ($message -match '500|internal server error|502|bad gateway|503|service unavailable|504|gateway timeout') {
        return [LLMErrorCategory]::ServerError
    }

    # Parse errors
    if ($exceptionType -eq 'System.ArgumentException' -or $message -match 'invalid json|parse|conversion failed|failed to parse') {
        return [LLMErrorCategory]::ParseError
    }

    # Default to Unknown
    return [LLMErrorCategory]::Unknown
}

function Invoke-WithRetry {
    <#
    .SYNOPSIS
        Executes a scriptblock with exponential backoff retry logic
    .DESCRIPTION
        Wraps operation execution with automatic retry on transient failures.
        Uses exponential backoff and configurable retry categories.
    .PARAMETER ScriptBlock
        The scriptblock to execute
    .PARAMETER MaxRetries
        Maximum number of retry attempts (default: 3)
    .PARAMETER BackoffSeconds
        Array of backoff delays in seconds for each retry (default: @(1, 2, 4))
    .PARAMETER RetryableCategories
        Array of error categories that should trigger a retry
        (default: Connectivity, Timeout, RateLimited, ServerError)
    .OUTPUTS
        Object - The result of the scriptblock execution
    .EXAMPLE
        $result = Invoke-WithRetry -ScriptBlock {
            Invoke-RestMethod -Uri $apiUrl -Method Post
        }
    .EXAMPLE
        $result = Invoke-WithRetry -ScriptBlock {
            Send-OllamaRequest -Prompt $prompt
        } -MaxRetries 5 -BackoffSeconds @(1, 2, 4, 8, 16)
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [scriptblock]$ScriptBlock,

        [Parameter()]
        [ValidateRange(1, 10)]
        [int]$MaxRetries = 3,

        [Parameter()]
        [int[]]$BackoffSeconds = @(1, 2, 4),

        [Parameter()]
        [LLMErrorCategory[]]$RetryableCategories = @(
            [LLMErrorCategory]::Connectivity,
            [LLMErrorCategory]::Timeout,
            [LLMErrorCategory]::RateLimited,
            [LLMErrorCategory]::ServerError
        )
    )

    $attempt = 0
    $lastException = $null

    while ($attempt -le $MaxRetries) {
        try {
            Write-Verbose "Attempt $($attempt + 1) of $($MaxRetries + 1)"

            # Execute the scriptblock
            $result = & $ScriptBlock

            # Success - return result
            if ($attempt -gt 0) {
                Write-Verbose "Operation succeeded after $attempt retry/retries"
            }
            return $result
        }
        catch {
            $lastException = $_.Exception
            $category = Get-LLMErrorCategory -Exception $lastException

            Write-Verbose "Attempt $($attempt + 1) failed with category: $category"
            Write-Verbose "Error: $($lastException.Message)"

            # Check if we should retry
            $shouldRetry = $category -in $RetryableCategories
            $hasRetriesLeft = $attempt -lt $MaxRetries

            if (-not $shouldRetry) {
                Write-Verbose "Error category '$category' is not retryable"
                throw
            }

            if (-not $hasRetriesLeft) {
                Write-Verbose "No retries remaining"
                throw
            }

            # Calculate backoff delay
            $backoffIndex = [Math]::Min($attempt, $BackoffSeconds.Length - 1)
            $delay = $BackoffSeconds[$backoffIndex]

            Write-Verbose "Retrying in $delay second(s)..."
            Start-Sleep -Seconds $delay

            $attempt++
        }
    }

    # Should not reach here, but throw last exception if we do
    if ($lastException) {
        throw $lastException
    }
}

function New-LLMErrorReport {
    <#
    .SYNOPSIS
        Creates a structured error report from an exception
    .DESCRIPTION
        Generates a comprehensive error report including categorization,
        timestamp, operation context, and exception details for logging
        and debugging purposes.
    .PARAMETER Exception
        The exception to report
    .PARAMETER Operation
        Optional name of the operation that failed
    .PARAMETER Context
        Optional hashtable of contextual information (model, API URL, parameters, etc.)
    .OUTPUTS
        [PSCustomObject] Structured error report
    .EXAMPLE
        $report = New-LLMErrorReport -Exception $_.Exception -Operation "Send-OllamaRequest" -Context @{
            Model = "llama3.2"
            ApiUrl = "http://localhost:11434"
        }
        $report | ConvertTo-Json | Out-File error.log
    #>
    [CmdletBinding()]
    [OutputType([PSCustomObject])]
    param(
        [Parameter(Mandatory)]
        [System.Exception]$Exception,

        [Parameter()]
        [string]$Operation,

        [Parameter()]
        [hashtable]$Context
    )

    $category = Get-LLMErrorCategory -Exception $Exception

    $report = [PSCustomObject]@{
        Timestamp     = Get-Date -Format "yyyy-MM-ddTHH:mm:ss.fffZ"
        Message       = $Exception.Message
        Category      = $category
        ExceptionType = $Exception.GetType().FullName
        Operation     = $Operation
        Context       = $Context
    }

    # Include inner exception details if present
    if ($Exception.InnerException) {
        $report | Add-Member -MemberType NoteProperty -Name InnerException -Value $Exception.InnerException.Message
    }

    # Include stack trace for debugging (truncated for readability)
    if ($Exception.StackTrace) {
        $stackLines = $Exception.StackTrace -split "`n" | Select-Object -First 5
        $report | Add-Member -MemberType NoteProperty -Name StackTrace -Value ($stackLines -join "`n")
    }

    # Add retry recommendation based on category
    $isRetryable = $category -in @(
        [LLMErrorCategory]::Connectivity,
        [LLMErrorCategory]::Timeout,
        [LLMErrorCategory]::RateLimited,
        [LLMErrorCategory]::ServerError
    )
    $report | Add-Member -MemberType NoteProperty -Name IsRetryable -Value $isRetryable

    return $report
}
