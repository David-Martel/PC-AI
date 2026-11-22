<#
.SYNOPSIS
    Judges a Pester 5 run object so every PC-AI test runner fails the same way.

.DESCRIPTION
    Why this exists: the runners used to decide pass/fail from `FailedCount` alone. Measured on Pester
    5.7.1, that is wrong in two ways that both end in a green exit code:

      * a test file with a syntax error gives  Result=Failed, FailedContainersCount=1, FailedCount=0
      * a tag filter that matches nothing gives Result=Passed, PassedCount=0 (TotalCount counts the
        filtered-out tests as NotRun, so even `TotalCount -eq 0` does not catch it)

    Get-PesterRunVerdict reads those counters as well, and refuses to pass a run that executed nothing.
    Tests/Unit/TestAssessment.Tests.ps1 plants both failure modes and proves they are reported.

    Dot-source this file; it defines functions only and has no side effects.

.NOTES
    Used by Tests/Invoke-PortableTests.ps1, Tests/Invoke-AllTests.ps1, Tests/Invoke-QualityGate.ps1,
    Tests/.pester.ps1, Tests/test.ps1 and Tests/coverage-report.ps1.
#>

function Get-PesterRunVerdict {
    <#
    .SYNOPSIS
        Returns Passed plus the reasons a Pester run must not be trusted.
    .PARAMETER Result
        The object returned by Invoke-Pester -PassThru (Run.PassThru = $true).
    .PARAMETER MaxFailedTests
        Failed tests tolerated beyond the per-file allowances in KnownFailures. Default 0.
    .PARAMETER MaxFailedContainers
        Containers (files) that may fail to load. Default 0.
    .PARAMETER MaxFailedBlocks
        Describe/Context blocks that may fail outside any test (a BeforeAll/AfterAll error). Default 0.
    .PARAMETER MinExecuted
        Minimum passed + failed tests. Default 1, so an empty tag filter or an empty directory fails.
    .PARAMETER KnownFailures
        Hashtable of test-file leaf name to tolerated failed-test count. A ratcheted baseline uses it to
        record failures that are tracked elsewhere without letting new ones through.
    .OUTPUTS
        PSCustomObject: Passed, Violations (string[]), Executed, Failed, FailedContainers, Total.
    #>
    [CmdletBinding()]
    [OutputType([pscustomobject])]
    param(
        [AllowNull()]
        $Result,

        [int]$MaxFailedTests = 0,

        [int]$MaxFailedContainers = 0,

        [int]$MaxFailedBlocks = 0,

        [int]$MinExecuted = 1,

        [hashtable]$KnownFailures = @{}
    )

    $violations = [System.Collections.Generic.List[string]]::new()

    if ($null -eq $Result) {
        $violations.Add('Invoke-Pester returned no result object (was Run.PassThru enabled?)')
        return [pscustomobject]@{
            Passed = $false; Violations = $violations.ToArray()
            Executed = 0; Failed = 0; FailedContainers = 0; Total = 0
        }
    }

    $passed = [int]$Result.PassedCount
    $failed = [int]$Result.FailedCount
    $containers = [int]$Result.FailedContainersCount
    $blocks = [int]$Result.FailedBlocksCount
    $total = [int]$Result.TotalCount
    $executed = $passed + $failed

    if ($executed -lt $MinExecuted) {
        $violations.Add("only $executed test(s) executed (minimum $MinExecuted); an empty path or tag filter is a failure, not a pass")
    }
    if ($containers -gt $MaxFailedContainers) {
        # A container's Result is Failed whenever one of its tests failed; only one with an error record of its own
        # failed to load, so list those.
        $names = @($Result.Containers | Where-Object { $_.Result -eq 'Failed' -and @($_.ErrorRecord).Count -gt 0 } | ForEach-Object {
                if ($_.Item -is [System.IO.FileInfo]) { $_.Item.Name } else { [string]$_.Name }
            }) -join ', '
        $violations.Add("$containers test container(s) failed to run (syntax or discovery error), allowed ${MaxFailedContainers}: $names")
    }
    if ($blocks -gt $MaxFailedBlocks) {
        $violations.Add("$blocks test block(s) failed outside any test (BeforeAll/AfterAll error), allowed $MaxFailedBlocks")
    }

    # Failures are judged per file: a file may fail up to its KnownFailures allowance (default 0) and the
    # excess over every allowance may not exceed MaxFailedTests. A bare result object with no per-test
    # detail falls back to the plain count.
    $byFile = @{}
    foreach ($test in @($Result.Failed)) {
        $path = $test.ScriptBlock.File
        if (-not $path) { $path = '<unknown>' }
        $leaf = Split-Path -Leaf $path
        $byFile[$leaf] = 1 + [int]$byFile[$leaf]
    }
    $excess = 0
    $fileLines = [System.Collections.Generic.List[string]]::new()
    if ($byFile.Count -eq 0) {
        $excess = $failed
        if ($excess -gt $MaxFailedTests) { $fileLines.Add("$failed failed test(s), allowed $MaxFailedTests") }
    } else {
        foreach ($leaf in ($byFile.Keys | Sort-Object)) {
            $allowed = if ($KnownFailures.ContainsKey($leaf)) { [int]$KnownFailures[$leaf] } else { 0 }
            if ($byFile[$leaf] -gt $allowed) {
                $excess += $byFile[$leaf] - $allowed
                $fileLines.Add("$leaf has $($byFile[$leaf]) failed test(s), allowed $allowed")
            }
        }
    }
    if ($excess -gt $MaxFailedTests) { foreach ($line in $fileLines) { $violations.Add($line) } }

    # Pester's own verdict, when no counter above explains it. A run whose only failures are tolerated
    # legitimately leaves Result at 'Failed'.
    if ($Result.Result -ne 'Passed' -and $violations.Count -eq 0 -and $failed -eq 0 -and $containers -eq 0 -and $blocks -eq 0) {
        $violations.Add("Pester Result is '$($Result.Result)' with no failed test, container or block to explain it")
    }

    [pscustomobject]@{
        Passed           = ($violations.Count -eq 0)
        Violations       = $violations.ToArray()
        Executed         = $executed
        Failed           = $failed
        FailedContainers = $containers
        Total            = $total
    }
}
