<#
.SYNOPSIS
    Plans or executes the selected checkout's Pester tests.
.DESCRIPTION
    Type All retains Unit plus Integration. WindowsCI retains the complete
    maintained 19-path cohort and the 85 percent all-Modules coverage gate.
    Plan performs no Pester import/discovery, writes or process-state changes.
    Path selects literal test files; combine it with FullName for focused runs.
    PassThru returns only the actual validated Pester result.
.EXAMPLE
    ./Tests/.pester.ps1 -Type Unit -Path Tests/Unit/PesterCoverageGate.Tests.ps1 -Plan
.EXAMPLE
    ./Tests/.pester.ps1 -Profile WindowsCI -CI
#>
[CmdletBinding()]
param(
    [ValidateSet('Unit', 'Integration', 'All')][string]$Type = 'All',
    [ValidateSet('Unit', 'Integration', 'All', 'LegacyUnit', 'LegacyIntegration', 'LegacyAll', 'LegacyE2E', 'LegacyFunctional', 'Portable', 'WindowsCI')]
    [string]$Profile,
    [string[]]$Path,
    [string[]]$FullName,
    [string[]]$Tag,
    [string[]]$ExcludeTag,
    [switch]$Coverage,
    [switch]$CI,
    [switch]$Plan,
    [switch]$PassThru,
    [ValidateSet('None', 'NUnitXml', 'JUnitXml', 'NUnit3')][string]$OutputFormat,
    [string]$OutputFile,
    [string]$OutputDirectory
)
$testRoot = $PSScriptRoot
$repoRoot = Split-Path -Parent $testRoot
$pathComparison = if ([Environment]::OSVersion.Platform -eq [PlatformID]::Win32NT) { [StringComparison]::OrdinalIgnoreCase } else { [StringComparison]::Ordinal }
if ($Profile -and $PSBoundParameters.ContainsKey('Type')) { throw 'Choose Profile or Type, not both.' }
if (-not $Profile) { $Profile = $Type }
$profiles = Import-PowerShellDataFile -LiteralPath (Join-Path $testRoot 'TestProfiles.psd1') -ErrorAction Stop
$settings = Import-PowerShellDataFile -LiteralPath (Join-Path $testRoot 'PesterConfiguration.psd1') -ErrorAction Stop
$selected = $profiles[$Profile]
if ($Profile -eq 'WindowsCI' -and ($Path -or $FullName -or $Tag -or $ExcludeTag)) { throw 'WindowsCI cannot be narrowed. Use a focused profile for diagnostics.' }
$declaredPaths = @($selected.Path)
$runPaths = @(if ($Path) {
        foreach ($target in $Path) {
            $absolute = if ([IO.Path]::IsPathRooted($target)) { [IO.Path]::GetFullPath($target) } else { [IO.Path]::GetFullPath((Join-Path $repoRoot $target)) }
            $prefix = $testRoot + [IO.Path]::DirectorySeparatorChar
            if (-not $absolute.StartsWith($prefix, $pathComparison) -or -not $absolute.EndsWith('.Tests.ps1', [StringComparison]::OrdinalIgnoreCase)) { throw "Path must name a literal test file inside Tests: $target" }
            if (-not (Test-Path -LiteralPath $absolute -PathType Leaf -ErrorAction Stop)) { throw "Test file does not exist: $target" }
            $absolute
        }
    }
    else {
        foreach ($relative in $declaredPaths) {
            $absolute = Join-Path $repoRoot $relative
            if ([Management.Automation.WildcardPattern]::ContainsWildcardCharacters($relative)) {
                if (-not (Test-Path -LiteralPath (Split-Path -Parent $absolute) -PathType Container -ErrorAction Stop)) { continue }
                Get-ChildItem -Path $absolute -File -ErrorAction Stop | ForEach-Object FullName
            }
            else {
                if (-not (Test-Path -LiteralPath $absolute -ErrorAction Stop)) { throw "Profile path does not exist: $relative" }
                $absolute
            }
        }
    })
if ($runPaths.Count -eq 0) { throw 'No test paths selected.' }
$settings.Run.Path = $runPaths
$settings.Run.PassThru = $true
$settings.Run.Exit = $false
$settings.CodeCoverage.Enabled = [bool]$Coverage -or $Profile -eq 'WindowsCI'
$settings.CodeCoverage.Path = @((Join-Path $repoRoot 'Modules'))
$settings.CodeCoverage.RecursePaths = $true
$settings.CodeCoverage.CoveragePercentTarget = 85
$settings.Filter.Tag = if ($Tag) { @($Tag) } elseif ($selected.ContainsKey('Tag')) { @($selected.Tag) } else { @() }
$settings.Filter.ExcludeTag = @($ExcludeTag | Where-Object { $_ })
$settings.Filter.FullName = @($FullName | Where-Object { $_ })
$directorySpecified = [bool]$OutputDirectory
if (-not $OutputDirectory) { $OutputDirectory = Join-Path $repoRoot $(if ($Profile -eq 'WindowsCI') { 'Tests/TestResults' } else { 'TestResults' }) }
elseif (-not [IO.Path]::IsPathRooted($OutputDirectory)) { $OutputDirectory = Join-Path $repoRoot $OutputDirectory }
if (-not $OutputFormat) { $OutputFormat = if ($CI) { 'NUnitXml' } else { 'None' } }
if (-not $OutputFile) { $OutputFile = if ($directorySpecified -or $Profile -eq 'WindowsCI') { Join-Path $OutputDirectory 'test-results.xml' } else { Join-Path $repoRoot $settings.TestResult.OutputPath } }
elseif (-not [IO.Path]::IsPathRooted($OutputFile)) { $OutputFile = Join-Path $repoRoot $OutputFile }
$settings.TestResult.Enabled = $OutputFormat -ne 'None'
if ($settings.TestResult.Enabled) { $settings.TestResult.OutputFormat = $OutputFormat }
$settings.TestResult.OutputPath = $OutputFile
$settings.CodeCoverage.OutputPath = Join-Path $OutputDirectory 'coverage.xml'
foreach ($outputPath in @($OutputFile, $settings.CodeCoverage.OutputPath)) {
    $leaf = [IO.Path]::GetFileName($outputPath).TrimEnd('.', ' ')
    if ($leaf -match '^(?:\$null|AUX|CON|NUL|PRN|COM[1-9]|LPT[1-9])(?:\..*)?$') { throw "Reserved output filename: $leaf" }
}
if ($CI) { $settings.Output.Verbosity = 'Normal' }
$testPlan = [pscustomobject]@{
    Profile = $Profile; Repository = $repoRoot; DeclaredPaths = $declaredPaths
    Targeted                = [bool]($Path -or $FullName -or $Tag -or $ExcludeTag)
    FullWindowsScope        = $Profile -eq 'WindowsCI'
    Configuration           = $settings
    SourceBoundCoveragePath = Join-Path $OutputDirectory 'source-bound-coverage.json'
}
if ($Plan) { return $testPlan }

$ErrorActionPreference = 'Stop'
$originalModulePath = $env:PSModulePath
$originalLocation = (Microsoft.PowerShell.Management\Get-Location -ErrorAction Stop).Path
$primaryFailure = $null
$cleanupFailure = $null
$result = $null
try {
    $env:PSModulePath = (Join-Path $repoRoot 'Modules') + [IO.Path]::PathSeparator + $originalModulePath
    Microsoft.PowerShell.Management\Set-Location -LiteralPath $repoRoot -ErrorAction Stop
    $sourceCommit = $null
    if ($settings.CodeCoverage.Enabled -and (Test-Path -LiteralPath (Join-Path $repoRoot '.git') -ErrorAction Stop)) {
        $gitRoot = & git -C $repoRoot rev-parse --show-toplevel 2>&1
        if ($LASTEXITCODE -ne 0 -or -not [IO.Path]::GetFullPath([string]$gitRoot).Equals([IO.Path]::GetFullPath($repoRoot), $pathComparison)) { throw 'Coverage repository identity does not match the selected checkout.' }
        $sourceCommit = & git -C $repoRoot rev-parse --verify HEAD 2>&1
        if ($LASTEXITCODE -ne 0 -or $sourceCommit -isnot [string] -or $sourceCommit -cnotmatch '^[a-f0-9]{40}$') { throw 'Cannot establish exact coverage source commit.' }
    }
    Import-Module Pester -MinimumVersion 5.0.0 -ErrorAction Stop
    $config = New-PesterConfiguration -Hashtable $settings
    foreach ($output in @($(if ($settings.TestResult.Enabled) { $OutputFile }), $(if ($settings.CodeCoverage.Enabled) { $settings.CodeCoverage.OutputPath }))) {
        if ($output) { [IO.Directory]::CreateDirectory((Split-Path -Parent $output)) | Out-Null }
    }
    $sourceBindings = @(if ($settings.CodeCoverage.Enabled) {
            Get-ChildItem -LiteralPath (Join-Path $repoRoot 'Modules') -Recurse -File |
                Where-Object Extension -In '.ps1', '.psm1' | ForEach-Object {
                    [pscustomobject]@{ Path = $_.FullName; SHA = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash }
                }
        })
    $result = Invoke-Pester -Configuration $config
    . (Join-Path $repoRoot 'Tools/Assert-PesterCoverageGate.ps1')
    if ($settings.CodeCoverage.Enabled) {
        if ($sourceCommit) {
            $headAfter = & git -C $repoRoot rev-parse --verify HEAD 2>&1
            if ($LASTEXITCODE -ne 0 -or $headAfter -cne $sourceCommit) { throw 'Coverage source HEAD changed during execution.' }
        }
        . (Join-Path $repoRoot 'Tools/Merge-PesterChildCoverage.ps1')
        $snapshot = New-PesterCoverageSnapshot -Coverage $result.CodeCoverage -SourceBindings $sourceBindings
        $attachments = @(foreach ($test in @($result.Tests)) {
                if ($null -ne $test -and $null -ne $test.PSObject.Properties['PcaiChildCoverage']) { [pscustomobject]@{ Result = $test.Result; Receipt = $test.PcaiChildCoverage } }
            })
        [ordered]@{ SchemaVersion = 1; SourceCommit = $sourceCommit; Profile = $Profile; RunId = $env:GITHUB_RUN_ID; RunAttempt = $env:GITHUB_RUN_ATTEMPT; Parent = $snapshot; Attachments = $attachments } |
            ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $testPlan.SourceBoundCoveragePath -Encoding utf8
        # Save actual outcomes before enforcing the gate, including failures.
        # A failed covered run must remain inspectable without another runner.
        [ordered]@{
            SchemaVersion=1; SourceCommit=$sourceCommit; Profile=$Profile
            Total=$result.TotalCount; Passed=$result.PassedCount; Failed=$result.FailedCount
            Skipped=$result.SkippedCount; NotRun=$result.NotRunCount
            FailedBlocks=$result.FailedBlocksCount; FailedContainers=$result.FailedContainersCount
            DurationSeconds=$result.Duration.TotalSeconds; CoverageTarget=85
            ParentAnalyzed=$snapshot.Analyzed; ParentExecuted=$snapshot.Executed
        } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $OutputDirectory 'pester-run-summary.json') -Encoding utf8
        Assert-PesterCoverageGate -Result $result -Target 85
    }
    else { Assert-PesterResult -Result $result }
    $result | Add-Member -NotePropertyName PcaiTestPlan -NotePropertyValue $testPlan -Force
}
catch { $primaryFailure = $_ }
finally {
    $env:PSModulePath = $originalModulePath
    try { Microsoft.PowerShell.Management\Set-Location -LiteralPath $originalLocation -ErrorAction Stop }
    catch { $cleanupFailure = $_ }
}
if ($primaryFailure) {
    if ($cleanupFailure) { $primaryFailure.Exception.Data['PcaiRunnerCleanupFailure'] = $cleanupFailure.Exception.Message }
    throw $primaryFailure
}
if ($cleanupFailure) { throw $cleanupFailure }
if ($PassThru) { return $result }
Write-Host "Tests: $($result.PassedCount) passed, $($result.FailedCount) failed, $($result.SkippedCount) skipped. Profile: $Profile"
