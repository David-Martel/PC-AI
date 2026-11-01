#Requires -Version 7.0

BeforeAll {
    $workflow = Join-Path $PSScriptRoot '../../.github/workflows/ci.yml'
    $lines = [IO.File]::ReadAllLines($workflow)
    $matches = @(0..($lines.Count - 1) | Where-Object { $lines[$_] -eq '      - name: Require complete PowerShell coverage artifacts' })
    if ($matches.Count -ne 1) { throw 'Expected exactly one complete coverage artifact guard step.' }
    $start = $matches[0] + 1
    $end = $start
    while ($end -lt $lines.Count -and $lines[$end] -notmatch '^      - ') { $end++ }
    $step = @($lines[$start..($end - 1)])
    if ('        if: always()' -notin $step -or '        shell: pwsh' -notin $step) {
        throw 'The required coverage artifact guard must always run with PowerShell.'
    }
    $run = [Array]::IndexOf($step, '        run: |')
    if ($run -lt 0 -or $run -eq ($step.Count - 1)) { throw 'The artifact guard needs an executable PowerShell body.' }
    $body = @(foreach ($line in $step[($run + 1)..($step.Count - 1)]) {
            if ([string]::IsNullOrWhiteSpace($line)) { '' }
            elseif ($line.StartsWith('          ')) { $line.Substring(10) }
            else { throw 'The artifact guard has an unexpected YAML body shape.' }
        })
    $script:CoverageArtifactGuard = [scriptblock]::Create(($body -join "`n"))
    $script:CoverageArtifacts = @(
        'Tests/TestResults/coverage.xml'
        'Tests/TestResults/source-bound-coverage.json'
        'Tests/TestResults/pester-run-summary.json'
    )
    function Invoke-OwnedCoverageArtifactGuard {
        param([string]$Root)
        Push-Location -LiteralPath $Root
        try { & $script:CoverageArtifactGuard }
        finally { Pop-Location }
    }
}

Describe 'Required PowerShell coverage artifact admission' -Tag 'Unit', 'Portable' {
    BeforeEach {
        $script:ArtifactRoot = Join-Path $TestDrive ([guid]::NewGuid().ToString('N'))
        foreach ($relative in $CoverageArtifacts) {
            $path = Join-Path $ArtifactRoot $relative
            [void][IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($path))
            [IO.File]::WriteAllText($path, 'owned presence-control payload')
        }
    }

    # Protects: A complete coverage packet remains admissible to the blocking upload.
    # Detects: Wrong CI paths or an unconditional guard failure despite all three real files.
    # Needs: Pester5, the actual embedded workflow step and owned TestDrive files; no network or mocks.
    # Breadcrumb: .github/workflows/ci.yml Require complete PowerShell coverage artifacts.
    It 'accepts the actual step when all three required artifacts exist' {
        { Invoke-OwnedCoverageArtifactGuard -Root $ArtifactRoot } | Should -Not -Throw
    }

    # Protects: Each required packet member is mandatory, including XML when both JSON sidecars exist (three cases).
    # Detects: upload-artifact any-match semantics masking a missing XML or a missing source/result binding.
    # Needs: The actual embedded PowerShell step, Pester5 and three owned files with only the selected member removed.
    # Breadcrumb: .github/workflows/ci.yml required presence check before multi-path Upload JaCoCo Artifact.
    It 'rejects missing <Leaf> even when the other two artifacts exist' -TestCases @(
        @{ Leaf = 'coverage.xml' }
        @{ Leaf = 'source-bound-coverage.json' }
        @{ Leaf = 'pester-run-summary.json' }
    ) {
        param($Leaf)
        [IO.File]::Delete((Join-Path $ArtifactRoot "Tests/TestResults/$Leaf"))
        foreach ($other in $CoverageArtifacts | Where-Object { [IO.Path]::GetFileName($_) -ne $Leaf }) {
            [IO.File]::Exists((Join-Path $ArtifactRoot $other)) | Should -BeTrue
        }
        { Invoke-OwnedCoverageArtifactGuard -Root $ArtifactRoot } | Should -Throw "*Required PowerShell coverage artifacts are missing*$Leaf*"
    }

    # Protects: A directory named coverage.xml cannot substitute for the measured XML file.
    # Detects: An existence-only guard accepting the wrong filesystem object while sidecars match.
    # Needs: Pester5, actual workflow body and an owned directory substituted for the XML leaf.
    # Breadcrumb: .github/workflows/ci.yml Test-Path -PathType Leaf boundary.
    It 'rejects a directory at the required coverage XML path' {
        $xml = Join-Path $ArtifactRoot 'Tests/TestResults/coverage.xml'
        [IO.File]::Delete($xml)
        [void][IO.Directory]::CreateDirectory($xml)
        { Invoke-OwnedCoverageArtifactGuard -Root $ArtifactRoot } | Should -Throw '*Required PowerShell coverage artifacts are missing*coverage.xml*'
    }
}
