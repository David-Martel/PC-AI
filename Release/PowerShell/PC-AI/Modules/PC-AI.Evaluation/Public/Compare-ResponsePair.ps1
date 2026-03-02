function Compare-ResponsePair {
    <#
    .SYNOPSIS
        Compares two responses using LLM-as-judge pairwise comparison

    .DESCRIPTION
        Useful for A/B testing different models or prompts
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$Question,

        [Parameter(Mandatory)]
        [string]$ResponseA,

        [Parameter(Mandatory)]
        [string]$ResponseB,

        [string]$Context
    )

    $prompt = @"
Compare these two responses to the same question. Determine which is better.

Question: $Question
$(if ($Context) { "Context: $Context" } else { "" })

Response A:
$ResponseA

Response B:
$ResponseB

Consider: accuracy, helpfulness, clarity, and completeness.

Respond ONLY with valid JSON:
{
  "winner": "A" or "B" or "tie",
  "confidence": <1-10>,
  "reasoning": "<brief explanation>",
  "a_strengths": ["<strength1>", "<strength2>"],
  "b_strengths": ["<strength1>", "<strength2>"],
  "a_weaknesses": ["<weakness1>"],
  "b_weaknesses": ["<weakness1>"]
}
"@

    try {
        $judgeResponse = Invoke-PcaiGenerate -Prompt $prompt -MaxTokens 500 -Temperature 0.3

        $jsonMatch = [regex]::Match($judgeResponse, '\{[\s\S]*\}')
        if ($jsonMatch.Success) {
            return $jsonMatch.Value | ConvertFrom-Json -AsHashtable
        } else {
            return @{ error = "Failed to parse comparison"; raw = $judgeResponse }
        }
    } catch {
        return @{ error = $_.Exception.Message }
    }
}
