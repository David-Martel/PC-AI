function Get-EvaluationDataset {
    <#
    .SYNOPSIS
        Gets a built-in or custom evaluation dataset

    .PARAMETER Name
        Dataset name: 'diagnostic', 'general', 'safety', or path to custom dataset
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$Name
    )

    switch ($Name) {
        'diagnostic' {
            return Get-DiagnosticTestCases
        }
        'general' {
            return Get-GeneralTestCases
        }
        'safety' {
            return Get-SafetyTestCases
        }
        default {
            # Try to load from file
            if (Test-Path $Name) {
                return Import-EvaluationDataset -Path $Name
            } else {
                Write-Error "Dataset not found: $Name"
                return $null
            }
        }
    }
}
