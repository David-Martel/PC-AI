function Convert-HelpBlockToEntry {
    param(
        [Parameter(Mandatory)]
        [string]$HelpBlock,
        [Parameter(Mandatory)]
        [string]$FunctionName,
        [Parameter(Mandatory)]
        [string]$SourcePath
    )

    $synopsis = ''
    $description = ''
    $parameterHelp = @{}
    $examples = @()

    if ($HelpBlock -match '(?ms)^\s*\.SYNOPSIS\s*(?<syn>.+?)(?=^\s*\.[A-Z]|\z)') {
        $synopsis = $Matches['syn'].Trim()
    }
    if ($HelpBlock -match '(?ms)^\s*\.DESCRIPTION\s*(?<desc>.+?)(?=^\s*\.[A-Z]|\z)') {
        $description = $Matches['desc'].Trim()
    }
    $paramMatches = [regex]::Matches($HelpBlock, '(?ms)^\s*\.PARAMETER\s+(?<name>\S+)\s*(?<desc>.+?)(?=^\s*\.[A-Z]|\z)')
    foreach ($paramMatch in $paramMatches) {
        $paramName = $paramMatch.Groups['name'].Value.Trim()
        $paramDesc = $paramMatch.Groups['desc'].Value.Trim()
        if ($paramName) {
            $parameterHelp[$paramName] = $paramDesc
        }
    }
    $exampleMatches = [regex]::Matches($HelpBlock, '(?ms)^\s*\.EXAMPLE\s*(?<example>.+?)(?=^\s*\.[A-Z]|\z)')
    foreach ($exampleMatch in $exampleMatches) {
        $exampleText = $exampleMatch.Groups['example'].Value.Trim()
        if ($exampleText) {
            $examples += $exampleText
        }
    }

    return [PSCustomObject]@{
        Name = $FunctionName
        Synopsis = $synopsis
        Description = $description
        SourcePath = $SourcePath
        ParameterHelp = $parameterHelp
        Examples = $examples
    }
}
