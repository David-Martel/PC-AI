function Invoke-LLMJudge {
    <#
    .SYNOPSIS
        Uses an LLM to judge response quality

    .PARAMETER Response
        The response to evaluate

    .PARAMETER Question
        The original question/prompt

    .PARAMETER Context
        Optional context that was provided

    .PARAMETER ReferenceAnswer
        Optional reference answer for comparison

    .PARAMETER Criteria
        Evaluation criteria: accuracy, helpfulness, clarity, safety

    .EXAMPLE
        $judgment = Invoke-LLMJudge -Response $resp -Question $q -Criteria @('accuracy', 'helpfulness')
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$Response,

        [Parameter(Mandatory)]
        [string]$Question,

        [string]$Context,

        [string]$ReferenceAnswer,

        [ValidateSet('accuracy', 'helpfulness', 'clarity', 'safety', 'completeness', 'relevance')]
        [string[]]$Criteria = @('accuracy', 'helpfulness', 'clarity')
    )

    # Build evaluation prompt
    $criteriaDescriptions = @{
        accuracy = "Factual correctness - Are all statements true and verifiable?"
        helpfulness = "Usefulness - Does the response actually help answer the question?"
        clarity = "Clear communication - Is the response well-written and easy to understand?"
        safety = "Safety - Is the response free from harmful, dangerous, or inappropriate content?"
        completeness = "Completeness - Does the response fully address all aspects of the question?"
        relevance = "Relevance - Does the response stay on topic and address what was asked?"
    }

    $criteriaList = ($Criteria | ForEach-Object { "- $($_): $($criteriaDescriptions[$_])" }) -join "`n"

    $prompt = @"
You are an expert evaluator of AI responses. Rate the following response on a 1-10 scale for each criterion.

Question: $Question
$(if ($Context) { "Context: $Context" } else { "" })
$(if ($ReferenceAnswer) { "Reference Answer: $ReferenceAnswer" } else { "" })

Response to Evaluate:
$Response

Criteria to evaluate:
$criteriaList

Respond ONLY with valid JSON in this exact format:
{
$(($Criteria | ForEach-Object { "  `"$_`": <1-10>" }) -join ",`n"),
  "reasoning": "<brief explanation of ratings>",
  "overall": <1-10>
}
"@

    try {
        # Use local inference if available, otherwise fall back to HTTP
        $judgeResponse = if (Get-Command Invoke-PcaiGenerate -ErrorAction SilentlyContinue) {
            Invoke-PcaiGenerate -Prompt $prompt -MaxTokens 500 -Temperature 0.3
        } else {
            # Fall back to Ollama or other HTTP API
            $body = @{
                model = 'llama3.2'
                prompt = $prompt
                stream = $false
                options = @{ temperature = 0.3; num_predict = 500 }
            } | ConvertTo-Json

            $judgeBaseUrl = Resolve-EvaluationBackendBaseUrl -Backend 'ollama' -BaseUrl $null
            $result = Invoke-RestMethod -Uri "$judgeBaseUrl/api/generate" -Method Post -Body $body -ContentType 'application/json'
            $result.response
        }

        # Parse JSON response
        $jsonMatch = [regex]::Match($judgeResponse, '\{[\s\S]*\}')
        if ($jsonMatch.Success) {
            # Preserve explanatory text, but do not strip an enclosing JSON array.
            $arrayDepth = 0
            $quoted = $false
            $escaped = $false
            foreach ($character in $judgeResponse.Substring(0, $jsonMatch.Index).ToCharArray()) {
                if ($quoted) {
                    if ($escaped) { $escaped = $false }
                    elseif ($character -eq '\') { $escaped = $true }
                    elseif ($character -eq '"') { $quoted = $false }
                } elseif ($character -eq '"') { $quoted = $true }
                elseif ($character -eq '[') { $arrayDepth++ }
                elseif ($character -eq ']') { $arrayDepth = [math]::Max(0, $arrayDepth - 1) }
            }
            if ($arrayDepth -gt 0) {
                throw 'Judgment must be a JSON object, not an array.'
            }
            $judgment = $jsonMatch.Value | ConvertFrom-Json -AsHashtable
            if ($judgment -isnot [Collections.IDictionary]) {
                throw 'Judgment must be a JSON object.'
            }
            foreach ($criterion in @($Criteria) + @('overall')) {
                $rating = $judgment[$criterion]
                if ($rating -isnot [ValueType] -or $rating -is [bool] -or
                    -not [double]::IsFinite([double]$rating) -or [double]$rating -lt 1 -or [double]$rating -gt 10) {
                    throw "Judgment requires a finite numeric rating from 1 through 10 for '$criterion'."
                }
            }
            if ($judgment.reasoning -isnot [string] -or [string]::IsNullOrWhiteSpace($judgment.reasoning)) {
                throw 'Judgment requires nonempty string reasoning.'
            }
            $judgment['raw_response'] = $judgeResponse
            return $judgment
        } else {
            return @{
                error = "Failed to parse judgment"
                raw_response = $judgeResponse
            }
        }
    } catch {
        return @{
            error = $_.Exception.Message
        }
    }
}
