#Requires -Version 7.0
<#
.SYNOPSIS
Reports source-bound missed commands, or an explicitly unmeasured test inventory.
.PARAMETER CoverageSnapshotPath
Source-bound-coverage.json from a maintained run. Source bytes and actual child
receipts are validated by the same merger as the coverage gate. This report
never admits a full test run or changes its result.
.PARAMETER RunTests
Compatibility option: execute the canonical Unit/Integration runner and return
its actual validated result. A full Windows gate is a separate explicit profile.
#>
[CmdletBinding()]
param(
    [string]$OutputPath = (Join-Path $PSScriptRoot '../Reports/COVERAGE_REPORT.md'),
    [string]$CoverageSnapshotPath,
    [switch]$RunTests
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
if ($RunTests) {
    if ($CoverageSnapshotPath) { throw 'Choose saved coverage or test execution, not both.' }
    & (Join-Path $PSScriptRoot '.pester.ps1') -Type All -PassThru
    return
}
$leaf = [IO.Path]::GetFileName($OutputPath).TrimEnd('.', ' ')
if ($leaf -match '^(?:\$null|AUX|CON|NUL|PRN|COM[1-9]|LPT[1-9])(?:\..*)?$') {
    throw "Reserved output filename: $leaf"
}

if ($CoverageSnapshotPath) {
    . (Join-Path $projectRoot 'Tools/Merge-PesterChildCoverage.ps1')
    $packet = Get-Content -LiteralPath $CoverageSnapshotPath -Raw | ConvertFrom-Json -Depth 40
    if ($packet.SchemaVersion -ne 1 -or $packet.SourceCommit -cnotmatch '^[a-f0-9]{40}$') {
        throw 'Unsupported source-bound coverage packet or missing source commit.'
    }
    $parent = Get-PesterChildCoverageMap -Snapshot $packet.Parent
    $coverage = [pscustomobject]@{
        CommandsAnalyzedCount = $packet.Parent.Analyzed
        CommandsExecutedCount = $packet.Parent.Executed
        CoveragePercent = 100.0 * $packet.Parent.Executed / $packet.Parent.Analyzed
        CommandsExecuted = @($packet.Parent.Commands | Where-Object Executed)
        CommandsMissed = @($packet.Parent.Commands | Where-Object { -not $_.Executed })
    }
    # Reuse admission and exact command identity checks; never turn an unpassed
    # child attachment into Passed just to calculate a larger numerator.
    $tests = @(foreach ($attachment in @($packet.Attachments)) {
        [pscustomobject]@{ Result=$attachment.Result; PcaiChildCoverage=$attachment.Receipt }
    })
    $merged = Merge-PesterChildCoverage -Result ([pscustomobject]@{ CodeCoverage=$coverage; Tests=$tests })
    $hits = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
    foreach ($entry in $parent.GetEnumerator()) {
        if ($entry.Value.Executed) { [void]$hits.Add($entry.Key) }
    }
    foreach ($attachment in @($packet.Attachments)) {
        $child = Get-PesterChildCoverageMap -Snapshot $attachment.Receipt.Child
        foreach ($entry in $child.GetEnumerator()) {
            if ($entry.Value.Executed) { [void]$hits.Add($entry.Key) }
        }
    }
    if ($hits.Count -ne $merged.CommandsExecutedCount) { throw 'Coverage report union does not reconcile.' }
    # The admitted map already has exact command keys. Reuse them rather than
    # serializing every command identity again merely to group missed counts.
    $bySource = [Collections.Generic.Dictionary[string,object]]::new([StringComparer]::Ordinal)
    foreach ($entry in $parent.GetEnumerator()) {
        $file = $entry.Value.File
        if (-not $bySource.ContainsKey($file)) {
            $bySource.Add($file, [pscustomobject]@{
                Source=[IO.Path]::GetRelativePath($projectRoot,$file).Replace('\','/')
                Analyzed=0; Executed=0; Missed=0
            })
        }
        $row = $bySource[$file]
        $row.Analyzed++
        if ($hits.Contains($entry.Key)) { $row.Executed++ } else { $row.Missed++ }
    }
    $rows = @($bySource.Values)
    $rows = @($rows | Sort-Object @{Expression='Missed';Descending=$true}, Source)
    $summary = [pscustomobject]@{
        Scope='Source-bound command inventory; test acceptance remains separate'
        SourceCommit=$packet.SourceCommit
        SnapshotSHA256=(Get-FileHash -LiteralPath $CoverageSnapshotPath -Algorithm SHA256).Hash
        Analyzed=$merged.CommandsAnalyzedCount
        Executed=$merged.CommandsExecutedCount
        CoveragePercent=$merged.CoveragePercent
        AdditionalUniqueHitsFor85=[Math]::Max(0, [Math]::Ceiling(0.85*$merged.CommandsAnalyzedCount)-$merged.CommandsExecutedCount)
        Sources=$rows
        ReportPath=$OutputPath
    }
    $lines = @(
        '# Source-bound command coverage gaps'
        ''
        ('Source commit: `' + $packet.SourceCommit + '`. Snapshot SHA256: `' + $summary.SnapshotSHA256 + '`.')
        ''
        ('Measured commands: {0}/{1} ({2:F4}%). Additional unique hits needed for 85%: {3}.' -f $summary.Executed,$summary.Analyzed,$summary.CoveragePercent,$summary.AdditionalUniqueHitsFor85)
        ''
        'Diagnostic only. Original failed/skipped test outcomes and full-run gates remain unchanged.'
        'Rank actual missed commands, then run the smallest relevant test cohort. File counts are not coverage.'
        ''
        '| Source | Executed | Analyzed | Missed |'
        '|---|---:|---:|---:|'
    ) + @($rows | ForEach-Object { '| {0} | {1} | {2} | {3} |' -f $_.Source,$_.Executed,$_.Analyzed,$_.Missed })
} else {
    # Inventory is useful for navigation, but test filenames cannot establish
    # behavioral or command coverage. Avoid recursively scanning caches/worktrees.
    $files = @(Get-ChildItem -LiteralPath $PSScriptRoot -Filter '*.Tests.ps1' -Recurse -File)
    $modules = @(Get-ChildItem -LiteralPath (Join-Path $projectRoot 'Modules') -Filter '*.psd1' -Recurse -File)
    $public = @(Get-ChildItem -LiteralPath (Join-Path $projectRoot 'Modules') -Filter '*.ps1' -Recurse -File |
        Where-Object { $_.Directory.Name -eq 'Public' })
    $summary = [pscustomobject]@{
        Scope='Inventory only; command coverage NOT_MEASURED'
        Modules=$modules.Count
        PublicFunctions=$public.Count
        UnitTestFiles=@($files | Where-Object { $_.Directory.Name -eq 'Unit' }).Count
        IntegrationTestFiles=@($files | Where-Object { $_.Directory.Name -eq 'Integration' }).Count
        TotalTestFiles=$files.Count
        ReportPath=$OutputPath
    }
    $lines = @(
        '# Test inventory (command coverage not measured)'
        ''
        ('Modules: {0}; public scripts: {1}; test files: {2}.' -f $summary.Modules,$summary.PublicFunctions,$summary.TotalTestFiles)
        ''
        'Provide -CoverageSnapshotPath with a maintained source-bound-coverage.json for actual command gaps.'
        'Use .pester.ps1 -Plan to inspect a named cohort, or -Path to select exact tests.'
        'The unchanged repository coverage target is 85%; inventory counts grant no coverage credit.'
    )
}
$parentDirectory = Split-Path -Parent ([IO.Path]::GetFullPath($OutputPath))
[void][IO.Directory]::CreateDirectory($parentDirectory)
$lines | Set-Content -LiteralPath $OutputPath -Encoding utf8
$summary
