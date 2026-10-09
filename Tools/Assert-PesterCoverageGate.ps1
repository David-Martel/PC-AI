function Assert-PesterCoverageGate {
    [CmdletBinding()]
    param([Parameter(Mandatory)]$Result, [double]$Target = 85)
    if ([double]::IsNaN($Target) -or [double]::IsInfinity($Target) -or $Target -lt 0 -or $Target -gt 100) { throw 'Invalid coverage target.' }
    foreach ($name in @('FailedCount','FailedContainersCount','TotalCount')) {
        $property=$Result.PSObject.Properties[$name]
        if ($null -eq $property -or $null -eq $property.Value) { throw "Missing Pester result field: $name" }
        $number=0L
        if (-not [long]::TryParse([string]$property.Value,[ref]$number) -or $number -lt 0) { throw "Invalid Pester result field: $name" }
        if ($name -eq 'TotalCount' -and $number -eq 0) { throw 'No tests discovered.' }
        if ($name -ne 'TotalCount' -and $number -ne 0) { throw "Pester failures: $name=$number" }
    }
    if ($null -eq $Result.CodeCoverage) { throw 'Missing Pester coverage.' }
    $coverage=$Result.CodeCoverage
    foreach ($name in @('CoveragePercent','CommandsAnalyzedCount','CommandsExecutedCount')) {
        if ($null -eq $coverage.PSObject.Properties[$name] -or $null -eq $coverage.$name) { throw "Missing coverage field: $name" }
    }
    $analyzed=0L; $executed=0L; $percent=0.0
    if (-not [long]::TryParse([string]$coverage.CommandsAnalyzedCount,[ref]$analyzed) -or $analyzed -le 0) { throw 'Invalid or zero analyzed coverage commands.' }
    if (-not [long]::TryParse([string]$coverage.CommandsExecutedCount,[ref]$executed) -or $executed -lt 0 -or $executed -gt $analyzed) { throw 'Invalid executed coverage commands.' }
    if (-not [double]::TryParse([string]$coverage.CoveragePercent,[ref]$percent) -or [double]::IsNaN($percent) -or [double]::IsInfinity($percent) -or $percent -lt 0 -or $percent -gt 100) { throw 'Invalid coverage percentage.' }
    $derivedPercent=100.0*$executed/$analyzed
    if ($percent -lt $Target -or $derivedPercent -lt $Target) { throw "Coverage $percent% (derived $derivedPercent%) is below required $Target% ($executed/$analyzed commands)." }
}
