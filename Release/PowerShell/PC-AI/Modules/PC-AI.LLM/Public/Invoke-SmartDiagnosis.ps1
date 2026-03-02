#Requires -Version 5.1
<#
.SYNOPSIS
    Smart PC diagnosis combining native high-performance tools with pcai-inference LLM analysis
#>
function Invoke-SmartDiagnosis {
    [CmdletBinding()]
    [OutputType([PSCustomObject])]
    param(
        [Parameter(Position = 0)]
        [string]$Path,

        [Parameter()]
        [string]$AnalysisType = 'Quick',

        [Parameter()]
        [string]$Model = $script:ModuleConfig.DefaultModel,

        [Parameter()]
        [switch]$SaveReport,

        [Parameter()]
        [string]$OutputPath = (Join-Path ([Environment]::GetFolderPath('Desktop')) 'Smart-Diagnosis-Report.json'),

        [Parameter()]
        [switch]$SkipLLMAnalysis,

        [Parameter()]
        [string]$OllamaBaseUrl,

        [Parameter()]
        [ValidateRange(30, 1800)]
        [int]$TimeoutSeconds = ([math]::Max(240, ($script:ModuleConfig.DefaultTimeout * 2)))
    )

    begin {
        $validTypes = @('Quick', 'Full', 'Duplicates', 'Storage')
        if ($validTypes -notcontains $AnalysisType) {
            $match = $validTypes | Where-Object { $_ -like "$AnalysisType*" } | Select-Object -First 1
            if ($match) {
                Write-Host "Using best match: '$match' for '$AnalysisType'" -ForegroundColor Gray
                $AnalysisType = $match
            } else {
                Write-Warning "Unknown AnalysisType '$AnalysisType'. Defaulting to 'Quick'."
                $AnalysisType = 'Quick'
            }
        }

        if ($OllamaBaseUrl) {
            $script:ModuleConfig.PcaiInferenceApiUrl = $OllamaBaseUrl
            $script:ModuleConfig.OllamaApiUrl = $OllamaBaseUrl
        }
    }

    process {
        $startTime = Get-Date
        Write-Host "`n[Smart Diagnosis] Starting $AnalysisType analysis..." -ForegroundColor Cyan

        $searchPath = if ($Path) {
            Resolve-Path $Path -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Path
        } else {
            $env:USERPROFILE
        }

        if (-not $searchPath -or -not (Test-Path $searchPath)) {
            throw "Invalid path: $Path"
        }

        Write-Host '[Phase 1] Collecting diagnostic data...' -ForegroundColor Cyan
        $diagnosticData = @{
            Path         = $searchPath
            AnalysisType = $AnalysisType
            StartTime    = $startTime
            Results      = @{}
        }

        try {
            switch ($AnalysisType) {
                'Quick' {
                    $diagnosticData.Results.Duplicates = Invoke-NativeSearch -Operation Duplicates -Path $searchPath -MinimumSize 1MB
                    $diagnosticData.Results.LargeFiles = Invoke-NativeSearch -Operation Files -Path $searchPath -Pattern '*.{zip,rar,7z,iso,exe,msi}' -MaxResults 50
                }
                'Full' {
                    $diagnosticData.Results.Duplicates = Invoke-NativeSearch -Operation Duplicates -Path $searchPath -MinimumSize 100KB
                    $diagnosticData.Results.AllFiles = Invoke-NativeSearch -Operation Files -Path $searchPath -Pattern '*' -MaxResults 100
                    $diagnosticData.Results.Logs = Invoke-NativeSearch -Operation Content -Path $searchPath -Pattern 'error|exception|failed' -FilePattern '*.log' -MaxResults 50 -ContextLines 2
                }
                'Duplicates' {
                    $diagnosticData.Results.Duplicates = Invoke-NativeSearch -Operation Duplicates -Path $searchPath -MinimumSize 1KB
                    $diagnosticData.Results.Backups = Invoke-NativeSearch -Operation Files -Path $searchPath -Pattern '*.{bak,old,backup,~}' -MaxResults 100
                }
                'Storage' {
                    $diagnosticData.Results.LargeFiles = Invoke-NativeSearch -Operation Files -Path $searchPath -Pattern '*' -MaxResults 200
                    $diagnosticData.Results.TempFiles = Invoke-NativeSearch -Operation Files -Path $searchPath -Pattern '*.{tmp,temp,cache}' -MaxResults 100
                    $diagnosticData.Results.Duplicates = Invoke-NativeSearch -Operation Duplicates -Path $searchPath -MinimumSize 10MB
                }
            }
        } catch {
            Write-Warning "Data collection partially failed: $_"
        }

        $collectionTime = (Get-Date) - $startTime
        Write-Host "[Phase 1] Data collection complete in $([math]::Round($collectionTime.TotalSeconds, 2))s" -ForegroundColor Green

        $diagnosticSummary = Build-DiagnosticSummary -DiagnosticData $diagnosticData

        if ($SkipLLMAnalysis) {
            return [PSCustomObject]@{
                AnalysisType      = $AnalysisType
                Path              = $searchPath
                CollectionTimeMs  = [math]::Round($collectionTime.TotalMilliseconds, 2)
                DiagnosticSummary = $diagnosticSummary
                RawResults        = $diagnosticData.Results
                LLMAnalysis       = 'Skipped'
                Timestamp         = $startTime
            }
        }

        Write-Host "`n[Phase 2] Analyzing with LLM ($Model)..." -ForegroundColor Cyan
        try {
            $promptContext = Get-LLMPromptContext -AnalysisType $AnalysisType
            $userPrompt = "Analyze this PC diagnostic data: `n`n Path: $searchPath `n`n $diagnosticSummary"

            $messages = @(
                @{ role = 'system'; content = $promptContext.SystemPrompt }
                @{ role = 'user'; content = $userPrompt }
            )

            $llmResponse = Invoke-LLMChatWithFallback -Messages $messages -Model $Model -Temperature 0.3 -TimeoutSeconds $TimeoutSeconds
            $analysisRaw = $llmResponse.message.content
            $parsedAnalysis = ConvertFrom-LLMJson -Content $analysisRaw

            $endTime = Get-Date
            $totalDuration = ($endTime - $startTime).TotalSeconds

            $result = [PSCustomObject]@{
                AnalysisType        = $AnalysisType
                Path                = $searchPath
                NativeEngineUsed    = [PcaiNative.PcaiCore]::IsAvailable
                CollectionTimeMs    = [math]::Round($collectionTime.TotalMilliseconds, 2)
                TotalTimeSeconds    = [math]::Round($totalDuration, 2)
                DiagnosticSummary   = $diagnosticSummary
                LLMAnalysis         = $parsedAnalysis
                Model               = $Model
                RawResults          = $diagnosticData.Results
                Timestamp           = $startTime
            }

            if ($SaveReport) {
                $reportJson = $result | ConvertTo-Json -Depth 10
                Set-Content -Path $OutputPath -Value $reportJson -Encoding utf8
                Write-Host "`nReport saved to: $OutputPath" -ForegroundColor Cyan
            }

            return $result
        } catch {
            Write-Warning "LLM analysis failed: $_. Returning partial results."
            return [PSCustomObject]@{
                AnalysisType      = $AnalysisType
                Path              = $searchPath
                CollectionTimeMs  = [math]::Round($collectionTime.TotalMilliseconds, 2)
                DiagnosticSummary = $diagnosticSummary
                RawResults        = $diagnosticData.Results
                LLMAnalysis       = 'Skipped' # Consistent with fallback expectations
                Error             = $_.Exception.Message
                Timestamp         = $startTime
            }
        }
    }
}

function Build-DiagnosticSummary {
    [CmdletBinding()]
    param($DiagnosticData)

    $summary = [System.Text.StringBuilder]::new()
    foreach ($key in $DiagnosticData.Results.Keys) {
        $result = $DiagnosticData.Results[$key]
        $null = $summary.AppendLine("### $key")
        if ($result.Summary) { $null = $summary.AppendLine($result.Summary) }
        $null = $summary.AppendLine("- Elapsed: $($result.ElapsedMs) ms (Engine: $($result.Engine))")
        $null = $summary.AppendLine()
    }
    return $summary.ToString()
}
