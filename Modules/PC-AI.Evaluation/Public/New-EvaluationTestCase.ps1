function New-EvaluationTestCase {
    <#
    .SYNOPSIS
        Creates a new evaluation test case
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [ValidateScript({ -not [string]::IsNullOrWhiteSpace($_) })]
        [string]$Id,

        [Parameter(Mandatory)]
        [ValidateScript({ -not [string]::IsNullOrWhiteSpace($_) })]
        [string]$Prompt,

        [string]$Category = "general",

        [string]$ExpectedOutput,

        [hashtable]$Context = @{},

        [string[]]$Tags = @()
    )

    return [EvaluationTestCase]@{
        Id = $Id
        Category = $Category
        Prompt = $Prompt
        ExpectedOutput = $ExpectedOutput
        Context = $Context
        Tags = $Tags
    }
}
