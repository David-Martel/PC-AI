<#
.SYNOPSIS
    Judges a Pester 5 run object so every PC-AI test runner fails the same way.

.DESCRIPTION
    Why this exists: the runners used to decide pass/fail from `FailedCount` alone. Measured on Pester
    5.7.1, that is wrong in three ways that all end in a green exit code:

      * a test file with a syntax error gives  Result=Failed, FailedContainersCount=1, FailedCount=0
      * a Describe whose AfterAll throws, after passing tests, gives
        Result=Failed, FailedCount=0, FailedContainersCount=0, FailedBlocksCount=1
      * a tag filter that matches nothing gives Result=Passed, PassedCount=0 (TotalCount counts the
        filtered-out tests as NotRun, so even `TotalCount -eq 0` does not catch it)

    Get-PesterRunVerdict reads all of those counters, refuses to pass a run that executed nothing, and treats a
    Result other than Passed as a failure when no counter explains it. Tests/Unit/TestAssessment.Tests.ps1 plants
    each failure in a real Pester file and proves it is reported.

    Dot-source this file; it defines functions only and has no side effects.

.NOTES
    Used by Tests/Invoke-PortableTests.ps1, Tests/Invoke-AllTests.ps1, Tests/Invoke-QualityGate.ps1,
    Tests/.pester.ps1, Tests/test.ps1, Tests/coverage-report.ps1 and the startup tests runner.
#>

function Get-PesterBlockFileName {
    # The test file a failed block or container belongs to, as a leaf name.
    param($Node)
    $item = $null
    if ($Node.PSObject.Properties['BlockContainer'] -and $Node.BlockContainer) { $item = $Node.BlockContainer.Item }
    elseif ($Node.PSObject.Properties['Item']) { $item = $Node.Item }
    if ($item -is [System.IO.FileInfo]) { return $item.Name }
    if ($item) { return [string]$item }
    if ($Node.PSObject.Properties['ScriptBlock'] -and $Node.ScriptBlock -and $Node.ScriptBlock.File) { return (Split-Path -Leaf $Node.ScriptBlock.File) }
    '<unknown>'
}

function Get-PesterRunVerdict {
    <#
    .SYNOPSIS
        Returns Passed plus the reasons a Pester run must not be trusted.
    .PARAMETER Result
        The object returned by Invoke-Pester -PassThru (Run.PassThru = $true).
    .PARAMETER MaxFailedTests
        Failed tests tolerated beyond the per-file allowances in KnownFailures. Default 0.
    .PARAMETER MaxFailedContainers
        Failed containers (files that did not load) tolerated beyond KnownFailedContainers. Default 0.
    .PARAMETER MaxFailedBlocks
        Failed blocks (a throwing BeforeAll/AfterAll in a Describe or Context) tolerated beyond KnownFailedBlocks. Default 0.
    .PARAMETER MinExecuted
        Minimum passed + failed tests. Default 1, so an empty tag filter or an empty directory fails.
    .PARAMETER KnownFailures
        Hashtable of test-file leaf name to tolerated failed-test count.
    .PARAMETER KnownFailedBlocks
        Hashtable of test-file leaf name to tolerated failed-block count. Kept apart from KnownFailures so a new
        failed block in a file that already has known failed tests still fails.
    .PARAMETER KnownFailedContainers
        Leaf names of test files tolerated as failed containers.
    .OUTPUTS
        PSCustomObject: Passed, Violations (string[]), Executed, Failed, FailedBlocks, FailedContainers, Total.
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

        [hashtable]$KnownFailures = @{},

        [hashtable]$KnownFailedBlocks = @{},

        [string[]]$KnownFailedContainers = @()
    )

    $violations = [System.Collections.Generic.List[string]]::new()

    if ($null -eq $Result) {
        $violations.Add('Invoke-Pester returned no result object (was Run.PassThru enabled?)')
        return [pscustomobject]@{
            Passed = $false; Violations = $violations.ToArray()
            Executed = 0; Failed = 0; FailedBlocks = 0; FailedContainers = 0; Total = 0
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

    # Containers: a container's Result is Failed whenever one of its tests failed, so only one with an error record of
    # its own failed to load.
    $failedContainerNames = @(@($Result.Containers) | Where-Object { $_ -and $_.Result -eq 'Failed' -and @($_.ErrorRecord).Count -gt 0 } |
            ForEach-Object { Get-PesterBlockFileName $_ })
    if ($containers -gt 0) {
        $excessContainers = if ($failedContainerNames.Count -gt 0) { @($failedContainerNames | Where-Object { $_ -notin $KnownFailedContainers }).Count } else { $containers }
        if ($excessContainers -gt $MaxFailedContainers) {
            $violations.Add("$containers test container(s) failed to run (syntax or discovery error), $excessContainers not tolerated: $($failedContainerNames -join ', ')")
        }
    }

    # Blocks, judged per file like tests: the allowance is for the file, so a new failed block in a file that already
    # has known failed tests is still a violation.
    $blocksByFile = @{}
    foreach ($b in @($Result.FailedBlocks)) {
        if ($null -eq $b) { continue }
        $leaf = Get-PesterBlockFileName $b
        $blocksByFile[$leaf] = 1 + [int]$blocksByFile[$leaf]
    }
    if ($blocks -gt 0) {
        $blockLines = [System.Collections.Generic.List[string]]::new()
        $excessBlocks = 0
        if ($blocksByFile.Count -eq 0) {
            $excessBlocks = $blocks
            $blockLines.Add("$blocks test block(s) failed outside any test (BeforeAll/AfterAll error)")
        } else {
            foreach ($leaf in ($blocksByFile.Keys | Sort-Object)) {
                $allowed = if ($KnownFailedBlocks.ContainsKey($leaf)) { [int]$KnownFailedBlocks[$leaf] } else { 0 }
                if ($blocksByFile[$leaf] -gt $allowed) {
                    $excessBlocks += $blocksByFile[$leaf] - $allowed
                    $blockLines.Add("$($blocksByFile[$leaf]) test block(s) failed outside any test (BeforeAll/AfterAll error) in $leaf, allowed $allowed")
                }
            }
        }
        if ($excessBlocks -gt $MaxFailedBlocks) { foreach ($line in $blockLines) { $violations.Add($line) } }
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

    # Pester's own verdict, when no counter above explains it. A run whose only failures are tolerated legitimately
    # leaves Result at 'Failed'; one with every counter at 0 does not, and is never a pass.
    if ($Result.Result -ne 'Passed' -and $violations.Count -eq 0 -and $failed -eq 0 -and $containers -eq 0 -and $blocks -eq 0) {
        $violations.Add("Pester Result is '$($Result.Result)' with no failed test, container or block to explain it")
    }

    [pscustomobject]@{
        Passed           = ($violations.Count -eq 0)
        Violations       = $violations.ToArray()
        Executed         = $executed
        Failed           = $failed
        FailedBlocks     = $blocks
        FailedContainers = $containers
        Total            = $total
    }
}
