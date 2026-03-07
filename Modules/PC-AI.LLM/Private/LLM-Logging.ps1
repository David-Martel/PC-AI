#Requires -PSEdition Core

<#
.SYNOPSIS
    LLM request/response logging module for PC-AI with structured JSON logging.

.DESCRIPTION
    Provides structured logging capabilities for LLM router decisions, tool calls,
    latencies, and other diagnostic information. Logs are written in JSON format
    to Reports/Logs/llm-router.log with configurable log levels.

.NOTES
    Author: PC-AI Development Team
    Log Levels: Debug < Info < Warning < Error
    Default Level: Info
#>

# Module-scoped log level (default: Info)
$script:LogLevel = 'Info'

# Log level numeric mapping for comparison
$script:LogLevelMap = @{
    'Debug'   = 0
    'Info'    = 1
    'Warning' = 2
    'Error'   = 3
}

<#
.SYNOPSIS
    Sets the current log level for LLM logging.

.PARAMETER Level
    The log level to set: Debug, Info, Warning, or Error

.EXAMPLE
    Set-LLMLogLevel -Level 'Debug'
    # Enables all log levels including debug messages

.EXAMPLE
    Set-LLMLogLevel -Level 'Warning'
    # Only logs Warning and Error messages
#>
function Set-LLMLogLevel {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [ValidateSet('Debug', 'Info', 'Warning', 'Error')]
        [string]$Level
    )

    $script:LogLevel = $Level
    Write-Verbose "LLM log level set to: $Level"
}

<#
.SYNOPSIS
    Gets the current log level for LLM logging.

.OUTPUTS
    String - Current log level (Debug, Info, Warning, or Error)

.EXAMPLE
    $currentLevel = Get-LLMLogLevel
    Write-Host "Current log level: $currentLevel"
#>
function Get-LLMLogLevel {
    [CmdletBinding()]
    [OutputType([string])]
    param()

    return $script:LogLevel
}

<#
.SYNOPSIS
    Writes a structured log entry to the LLM router log file.

.PARAMETER Level
    The log level: Debug, Info, Warning, or Error

.PARAMETER Message
    The log message

.PARAMETER Data
    Optional hashtable of additional structured data to include in the log entry

.EXAMPLE
    Write-LLMLog -Level 'Info' -Message 'Router decision completed'

.EXAMPLE
    $routerData = @{
        Router = 'FunctionGemma'
        Decision = 'tool_call'
        ToolName = 'Get-SystemInfo'
        Latency = 342
    }
    Write-LLMLog -Level 'Info' -Message 'Router decision' -Data $routerData

.EXAMPLE
    Write-LLMLog -Level 'Debug' -Message 'Tool execution started' -Data @{ ToolName = 'Get-CimInstance' }
#>
function Write-LLMLog {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [ValidateSet('Debug', 'Info', 'Warning', 'Error')]
        [string]$Level,

        [Parameter(Mandatory)]
        [string]$Message,

        [Parameter()]
        [hashtable]$Data
    )

    # Check if this message should be logged based on current log level
    $currentLevelValue = $script:LogLevelMap[$script:LogLevel]
    $messageLevelValue = $script:LogLevelMap[$Level]

    if ($messageLevelValue -lt $currentLevelValue) {
        # Message level is below current threshold, don't log
        return
    }

    try {
        # Get log file path using Resolve-PcaiPath if available, otherwise fallback
        $logDir = $null
        if (Get-Command Resolve-PcaiPath -ErrorAction SilentlyContinue) {
            $logDir = Resolve-PcaiPath -PathType 'Logs'
        } else {
            $logDir = Join-Path (Get-Location).ProviderPath 'Reports\Logs'
        }

        # Ensure log directory exists
        if (-not (Test-Path $logDir)) {
            New-Item -ItemType Directory -Path $logDir -Force | Out-Null
        }

        $logFile = Join-Path $logDir 'llm-router.log'

        # Create structured log entry
        $logEntry = [ordered]@{
            Timestamp = (Get-Date).ToString('o')  # ISO 8601 format
            Level     = $Level
            Message   = $Message
        }

        # Add Data if provided
        if ($Data -and $Data.Count -gt 0) {
            $logEntry['Data'] = $Data
        }

        # Convert to JSON (compressed for single-line logging)
        $jsonLine = $logEntry | ConvertTo-Json -Compress -Depth 10

        # Append to log file
        $jsonLine | Add-Content -Path $logFile -Encoding UTF8

        Write-Verbose "Logged [$Level]: $Message"

    } catch {
        # Fail silently to avoid disrupting application flow
        # Could optionally write to Windows Event Log or debug stream
        Write-Debug "Failed to write LLM log: $_"
    }
}
