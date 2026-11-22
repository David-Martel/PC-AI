#Requires -Version 7.0

BeforeAll {
    $runner = Join-Path $PSScriptRoot '../Evaluation/Invoke-InferenceEvaluation.ps1'
    $parserErrors = $null
    $ast = [System.Management.Automation.Language.Parser]::ParseFile($runner, [ref]$null, [ref]$parserErrors)
    if ($parserErrors) { throw ($parserErrors | Out-String) }
    $function = $ast.Find({
            param($node)
            $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
            $node.Name -eq 'Get-EvaluationAcceptance'
        }, $false)
    . ([scriptblock]::Create($function.Extent.Text))
}

Describe 'Inference evaluation acceptance' -Tag 'Unit', 'Portable' {
    # Protects: completed, successful transport and the exact requested pass rate.
    # Detects: unavailable, empty, partial, cancelled and errored runs being accepted,
    # or using a rounded display percentage for the gate.
    # Needs: Pester 5 and in-memory result summaries.
    # Breadcrumb: Invoke-InferenceEvaluation.ps1 Get-EvaluationAcceptance.
    It 'returns <Reason> for <Case>' -ForEach @(
        @{ Case = 'unavailable backend'; Planned = 0; Attempted = 0; Passed = 0; Errors = 0; Cancelled = $false; Available = $false; Threshold = 0; Reason = 'BackendUnavailable' }
        @{ Case = 'empty run'; Planned = 0; Attempted = 0; Passed = 0; Errors = 0; Cancelled = $false; Available = $true; Threshold = 0; Reason = 'Incomplete' }
        @{ Case = 'partial run'; Planned = 2; Attempted = 1; Passed = 1; Errors = 0; Cancelled = $false; Available = $true; Threshold = 0; Reason = 'Incomplete' }
        @{ Case = 'cancelled run'; Planned = 2; Attempted = 1; Passed = 1; Errors = 0; Cancelled = $true; Available = $true; Threshold = 0; Reason = 'Cancelled' }
        @{ Case = 'case error'; Planned = 2; Attempted = 2; Passed = 1; Errors = 1; Cancelled = $false; Available = $true; Threshold = 0; Reason = 'CaseErrors' }
        @{ Case = 'transport-only test with failed content'; Planned = 1; Attempted = 1; Passed = 0; Errors = 0; Cancelled = $false; Available = $true; Threshold = 0; Reason = 'Accepted' }
        @{ Case = 'exact threshold match'; Planned = 2; Attempted = 2; Passed = 1; Errors = 0; Cancelled = $false; Available = $true; Threshold = 50; Reason = 'Accepted' }
        @{ Case = 'display rounds up to the minimum'; Planned = 3; Attempted = 3; Passed = 2; Errors = 0; Cancelled = $false; Available = $true; Threshold = 66.67; Reason = 'BelowMinimumPassRate' }
    ) {
        $result = if ($Available) {
            @{
                Suite = @{ TestCases = [object[]]::new($Planned) }
                Results = @{ TotalTests = $Attempted; Passed = $Passed; Errors = $Errors; Cancelled = $Cancelled }
            }
        }
        $decision = Get-EvaluationAcceptance -Evaluation $result -MinimumPassRate $Threshold
        $decision.Accepted | Should -Be ($Reason -eq 'Accepted')
        $decision.Reason | Should -Be $Reason
        $decision.ExpectedCases | Should -Be $Planned
        $decision.CompletedCases | Should -Be $Attempted
        $decision.PassedCases | Should -Be $Passed
        $decision.ErrorCases | Should -Be $Errors
    }
}
