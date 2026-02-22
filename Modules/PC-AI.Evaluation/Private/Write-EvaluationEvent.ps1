function Write-EvaluationEvent {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$Type,
        [Parameter(Mandatory)]
        [string]$Message,
        [hashtable]$Data,
        [ValidateSet('info', 'warn', 'error')]
        [string]$Level = 'info'
    )

    $payload = [ordered]@{
        ts = (Get-Date).ToUniversalTime().ToString('o')
        type = $Type
        level = $Level
        message = $Message
        data = $Data
    }

    $json = $payload | ConvertTo-Json -Depth 8 -Compress

    if ($script:EvaluationConfig.EventsLogPath) {
        $eventsDir = Split-Path -Parent $script:EvaluationConfig.EventsLogPath
        if ($eventsDir -and -not (Test-Path $eventsDir)) {
            New-Item -ItemType Directory -Path $eventsDir -Force | Out-Null
        }
        Add-Content -Path $script:EvaluationConfig.EventsLogPath -Value $json
    }

    if ($script:EvaluationConfig.ProgressMode -in @('auto', 'stream')) {
        $color = switch ($Level) {
            'info' { 'Gray' }
            'warn' { 'Yellow' }
            'error' { 'Red' }
        }
        Write-Host "[pcai.eval] $Message" -ForegroundColor $color
    }

    if ($script:EvaluationConfig.EmitStructuredMessages) {
        Write-Output $json
    }
}
