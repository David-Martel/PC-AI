function Measure-DiagnosticQuality {
    <#
    .SYNOPSIS
        Evaluates diagnostic output quality specific to PC-AI use case

    .DESCRIPTION
        Checks diagnostic responses for:
        - Proper JSON structure
        - Valid diagnosis categories
        - Actionable recommendations
        - Safety warnings where appropriate
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$DiagnosticOutput,

        [string]$DiagnosticInput,

        [switch]$Strict
    )

    $results = @{
        valid_json = $false
        has_findings = $false
        has_recommendations = $false
        has_priority_classification = $false
        safety_warnings_present = $false
        score = 0
        issues = @()
    }

    # Check JSON validity
    try {
        $parsed = $DiagnosticOutput | ConvertFrom-Json -ErrorAction Stop
        $results.valid_json = $true
        $results.score += 0.2
    } catch {
        $results.issues += "Invalid JSON structure"
        if ($Strict) { return $results }
    }

    # Check for expected sections
    if ($parsed) {
        if ($parsed.findings -or $parsed.summary) {
            $results.has_findings = $true
            $results.score += 0.2
        } else {
            $results.issues += "Missing findings/summary section"
        }

        if ($parsed.recommendations -or $parsed.next_steps) {
            $results.has_recommendations = $true
            $results.score += 0.2
        } else {
            $results.issues += "Missing recommendations section"
        }

        if ($parsed.priority -or $parsed.severity) {
            $results.has_priority_classification = $true
            $results.score += 0.2
        } else {
            $results.issues += "Missing priority/severity classification"
        }

        # Check for safety warnings when disk/hardware issues detected
        $dangerousKeywords = @('disk failure', 'smart error', 'bad sector', 'hardware fault')
        $needsWarning = $dangerousKeywords | Where-Object { $DiagnosticInput -match $_ }

        if ($needsWarning) {
            $warningPatterns = @('backup', 'warning', 'caution', 'risk', 'data loss')
            $hasWarning = $warningPatterns | Where-Object { $DiagnosticOutput -match $_ }
            if ($hasWarning) {
                $results.safety_warnings_present = $true
                $results.score += 0.2
            } else {
                $results.issues += "Missing safety warnings for critical issues"
            }
        } else {
            $results.score += 0.2  # No warning needed
        }
    }

    $results.score = [math]::Round($results.score, 2)

    return $results
}
