#Requires -Version 7.0
<#
.SYNOPSIS
    Cross-platform test runner for PC_AI — entry point for remote AI agents (Jules, Codex).

.DESCRIPTION
    Runs every test section that works on both Linux and Windows without GPU, Windows
    CIM/WMI, or native DLLs.  Sections: RustFmt, RustClippy, Rust, CSharp, Pester, Lint.

.PARAMETER Section
    One or more sections to run.  Default: All.

.PARAMETER Format
    Console output format: Table (default), Json, or JUnit.

.PARAMETER FailFast
    Stop after the first section that fails.

.PARAMETER ResultsDir
    Directory for output artefacts.  Default: Tests/Results relative to repo root.

.EXAMPLE
    pwsh Tests/Invoke-PortableTests.ps1
    Runs all portable sections and writes results to Tests/Results/.

.EXAMPLE
    pwsh Tests/Invoke-PortableTests.ps1 -Section Rust,Pester -Format Json
    Runs Rust tests and Pester portable suite, emits JSON summary.

.EXAMPLE
    pwsh Tests/Invoke-PortableTests.ps1 -Section RustFmt,RustClippy -FailFast
    Format + clippy gate; exits 1 on first failure.
#>

[CmdletBinding()]
param(
    [ValidateSet('All', 'Rust', 'RustClippy', 'RustFmt', 'CSharp', 'Pester', 'Lint')]
    [string[]]$Section = 'All',

    [ValidateSet('Table', 'Json', 'JUnit')]
    [string]$Format = 'Table',

    [switch]$FailFast,

    [string]$ResultsDir
)

$ErrorActionPreference = 'Stop'

# --- Paths ---
$RepoRoot    = Split-Path -Parent $PSScriptRoot
$RustRoot    = Join-Path $RepoRoot 'Native/pcai_core'
$ModulesRoot = Join-Path $RepoRoot 'Modules'
$PssaSettings = Join-Path $RepoRoot 'PSScriptAnalyzerSettings.psd1'

if (-not $ResultsDir) {
    $ResultsDir = Join-Path $RepoRoot 'Tests/Results'
}
if (-not (Test-Path $ResultsDir)) {
    New-Item -ItemType Directory -Path $ResultsDir -Force | Out-Null
}

$Timestamp   = (Get-Date -Format 'yyyyMMdd_HHmmss')
$ResultsJson = Join-Path $ResultsDir "portable-$Timestamp.json"

# --- Helpers ---
function New-SectionResult {
    param([string]$Name, [string]$Status, [string]$Detail = '', [hashtable]$Counts = @{})
    [pscustomobject]@{
        section  = $Name
        status   = $Status   # PASS | FAIL | SKIP
        detail   = $Detail
        counts   = $Counts
        duration_ms = 0
    }
}

function Invoke-Timed {
    param([string]$Name, [scriptblock]$Block)
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    try   { $result = & $Block }
    catch { $result = $_ }
    $sw.Stop()
    if ($result -is [pscustomobject]) {
        $result.duration_ms = $sw.ElapsedMilliseconds
    }
    return $result
}

function Test-CommandAvailable {
    param([string]$Command)
    $null -ne (Get-Command $Command -ErrorAction SilentlyContinue)
}

# Sections requested (expand 'All' to ordered list)
$AllSections = @('RustFmt', 'RustClippy', 'Rust', 'CSharp', 'Pester', 'Lint')
$RunSections = if ($Section -contains 'All') { $AllSections } else {
    $AllSections | Where-Object { $Section -contains $_ }
}

$Results     = [ordered]@{}
$GlobalStart = [System.Diagnostics.Stopwatch]::StartNew()
$AnyFail     = $false

# --- Section 1: RustFmt ---
if ($RunSections -contains 'RustFmt') {
    $r = Invoke-Timed 'RustFmt' {
        if (-not (Test-CommandAvailable 'cargo')) {
            return New-SectionResult 'RustFmt' 'SKIP' 'cargo not on PATH'
        }
        Push-Location $RustRoot
        try {
            $output = & cargo fmt --all --check 2>&1
            if ($LASTEXITCODE -eq 0) {
                New-SectionResult 'RustFmt' 'PASS'
            } else {
                New-SectionResult 'RustFmt' 'FAIL' ($output -join "`n")
            }
        } finally { Pop-Location }
    }
    $Results['RustFmt'] = $r
    Write-Verbose "RustFmt: $($r.status)"
    if ($r.status -eq 'FAIL' -and $FailFast) { $AnyFail = $true }
}

# --- Section 2: RustClippy ---
if ($RunSections -contains 'RustClippy' -and -not $AnyFail) {
    $r = Invoke-Timed 'RustClippy' {
        if (-not (Test-CommandAvailable 'cargo')) {
            return New-SectionResult 'RustClippy' 'SKIP' 'cargo not on PATH'
        }
        $env:RUSTC_WRAPPER = ''
        Push-Location $RustRoot
        try {
            $output = & cargo clippy --workspace --all-targets --no-deps `
                -- -D warnings -A clippy::type_complexity 2>&1
            $warnCount = ($output | Select-String '^warning:').Count
            if ($LASTEXITCODE -eq 0) {
                New-SectionResult 'RustClippy' 'PASS' '' @{ warnings = $warnCount }
            } else {
                New-SectionResult 'RustClippy' 'FAIL' ($output | Select-Object -Last 20 | Out-String).Trim() @{ warnings = $warnCount }
            }
        } finally { Pop-Location }
    }
    $Results['RustClippy'] = $r
    Write-Verbose "RustClippy: $($r.status)"
    if ($r.status -eq 'FAIL' -and $FailFast) { $AnyFail = $true }
}

# --- Section 3: Rust unit tests ---
if ($RunSections -contains 'Rust' -and -not $AnyFail) {
    $r = Invoke-Timed 'Rust' {
        if (-not (Test-CommandAvailable 'cargo')) {
            return New-SectionResult 'Rust' 'SKIP' 'cargo not on PATH'
        }
        $env:RUSTC_WRAPPER = ''
        Push-Location $RustRoot
        try {
            $output = & cargo test --workspace --no-default-features --features server,ffi 2>&1
            $passed  = 0; $failed = 0; $ignored = 0
            foreach ($line in $output) {
                if ($line -match 'test result: \w+\.\s+(\d+) passed; (\d+) failed; (\d+) ignored') {
                    $passed  += [int]$Matches[1]
                    $failed  += [int]$Matches[2]
                    $ignored += [int]$Matches[3]
                }
            }
            $status = if ($LASTEXITCODE -eq 0) { 'PASS' } else { 'FAIL' }
            $detail = if ($status -eq 'FAIL') {
                ($output | Select-String 'FAILED|error\[' | Select-Object -First 20 | Out-String).Trim()
            } else { '' }
            New-SectionResult 'Rust' $status $detail @{ passed = $passed; failed = $failed; ignored = $ignored }
        } finally { Pop-Location }
    }
    $Results['Rust'] = $r
    Write-Verbose "Rust: $($r.status) (passed=$($r.counts.passed) failed=$($r.counts.failed))"
    if ($r.status -eq 'FAIL' -and $FailFast) { $AnyFail = $true }
}

# --- Section 4: CSharp build (+ optional test project) ---
if ($RunSections -contains 'CSharp' -and -not $AnyFail) {
    $r = Invoke-Timed 'CSharp' {
        if (-not (Test-CommandAvailable 'dotnet')) {
            return New-SectionResult 'CSharp' 'SKIP' 'dotnet not on PATH'
        }
        $csproj = Join-Path $RepoRoot 'Native/PcaiNative/PcaiNative.csproj'
        if (-not (Test-Path $csproj)) {
            return New-SectionResult 'CSharp' 'SKIP' "Project not found: $csproj"
        }
        $buildOut  = & dotnet build $csproj --nologo -v q 2>&1
        $warnCount = ($buildOut | Select-String ': warning ').Count
        if ($LASTEXITCODE -ne 0) {
            return New-SectionResult 'CSharp' 'FAIL' ($buildOut | Select-Object -Last 10 | Out-String).Trim() @{ warnings = $warnCount }
        }
        $testProj = Join-Path $RepoRoot 'Native/PcaiNative.Tests/PcaiNative.Tests.csproj'
        if (Test-Path $testProj) {
            $testOut = & dotnet test $testProj --nologo 2>&1
            if ($LASTEXITCODE -ne 0) {
                return New-SectionResult 'CSharp' 'FAIL' ($testOut | Select-Object -Last 10 | Out-String).Trim() @{ warnings = $warnCount }
            }
        }
        New-SectionResult 'CSharp' 'PASS' '' @{ warnings = $warnCount }
    }
    $Results['CSharp'] = $r
    Write-Verbose "CSharp: $($r.status)"
    if ($r.status -eq 'FAIL' -and $FailFast) { $AnyFail = $true }
}

# --- Section 5: Pester (Portable-tagged unit tests) ---
if ($RunSections -contains 'Pester' -and -not $AnyFail) {
    $r = Invoke-Timed 'Pester' {
        if (-not (Get-Module -ListAvailable -Name Pester | Where-Object Version -ge '5.0.0')) {
            return New-SectionResult 'Pester' 'SKIP' 'Pester 5+ not installed'
        }
        Import-Module Pester -MinimumVersion 5.0.0 -Force -ErrorAction SilentlyContinue
        $xmlOut = Join-Path $ResultsDir 'pester-portable.xml'
        $cfg = New-PesterConfiguration
        $cfg.Run.Path            = Join-Path $RepoRoot 'Tests/Unit'
        $cfg.Filter.Tag          = @('Portable')
        $cfg.Output.Verbosity    = 'Normal'
        $cfg.TestResult.Enabled  = $true
        $cfg.TestResult.OutputFormat = 'NUnit3'
        $cfg.TestResult.OutputPath   = $xmlOut
        $cfg.Run.PassThru        = $true
        $pResult = Invoke-Pester -Configuration $cfg
        $status = if ($pResult.FailedCount -eq 0) { 'PASS' } else { 'FAIL' }
        New-SectionResult 'Pester' $status '' @{
            passed  = $pResult.PassedCount
            failed  = $pResult.FailedCount
            skipped = $pResult.SkippedCount
        }
    }
    $Results['Pester'] = $r
    Write-Verbose "Pester: $($r.status)"
    if ($r.status -eq 'FAIL' -and $FailFast) { $AnyFail = $true }
}

# --- Section 6: PSScriptAnalyzer lint ---
if ($RunSections -contains 'Lint' -and -not $AnyFail) {
    $r = Invoke-Timed 'Lint' {
        if (-not (Get-Module -ListAvailable -Name PSScriptAnalyzer)) {
            return New-SectionResult 'Lint' 'SKIP' 'PSScriptAnalyzer not installed'
        }
        Import-Module PSScriptAnalyzer -Force -ErrorAction SilentlyContinue
        $pssaArgs = @{
            Path      = $ModulesRoot
            Recurse   = $true
            Severity  = @('Warning', 'Error')
        }
        if (Test-Path $PssaSettings) { $pssaArgs['Settings'] = $PssaSettings }
        $findings  = Invoke-ScriptAnalyzer @pssaArgs
        $errors    = ($findings | Where-Object Severity -eq 'Error').Count
        $warnings  = ($findings | Where-Object Severity -eq 'Warning').Count
        $status    = if ($errors -gt 0) { 'FAIL' } else { 'PASS' }
        $detail    = if ($errors -gt 0) {
            ($findings | Where-Object Severity -eq 'Error' |
                Select-Object -First 10 |
                ForEach-Object { "$($_.ScriptName):$($_.Line) [$($_.RuleName)] $($_.Message)" }) -join "`n"
        } else { '' }
        New-SectionResult 'Lint' $status $detail @{ errors = $errors; warnings = $warnings }
    }
    $Results['Lint'] = $r
    Write-Verbose "Lint: $($r.status)"
    if ($r.status -eq 'FAIL' -and $FailFast) { $AnyFail = $true }
}

# --- Summarise ---
$GlobalStart.Stop()
$overallStatus = if (($Results.Values | Where-Object status -eq 'FAIL').Count -gt 0) { 'FAIL' } else { 'PASS' }

$sectionsDict = [ordered]@{}
foreach ($key in $Results.Keys) {
    $v = $Results[$key]
    $sectionsDict[$key] = [ordered]@{
        status      = $v.status
        duration_ms = $v.duration_ms
        counts      = $v.counts
        detail      = $v.detail
    }
}
$summary = [ordered]@{
    timestamp   = (Get-Date -Format 'o')
    platform    = ($PSVersionTable.Platform ?? 'Win32NT')
    sections    = $sectionsDict
    overall     = $overallStatus
    duration_ms = $GlobalStart.ElapsedMilliseconds
}
$summary | ConvertTo-Json -Depth 5 | Set-Content -Path $ResultsJson -Encoding UTF8

# --- Console output ---
switch ($Format) {
    'Table' {
        Write-Host ''
        Write-Host '=== Portable Test Results ===' -ForegroundColor Cyan
        foreach ($key in $Results.Keys) {
            $v      = $Results[$key]
            $color  = switch ($v.status) { 'PASS' { 'Green' } 'FAIL' { 'Red' } default { 'Yellow' } }
            $counts = if ($v.counts.Count) {
                ' (' + (($v.counts.GetEnumerator() | ForEach-Object { "$($_.Key)=$($_.Value)" }) -join ' ') + ')'
            } else { '' }
            Write-Host ("  {0,-12} {1}{2}  [{3}ms]" -f $key, $v.status, $counts, $v.duration_ms) -ForegroundColor $color
            if ($v.status -eq 'FAIL' -and $v.detail) {
                Write-Host $v.detail -ForegroundColor DarkRed
            }
        }
        Write-Host ''
        $overallColor = if ($overallStatus -eq 'PASS') { 'Green' } else { 'Red' }
        Write-Host "Overall: $overallStatus  [$($GlobalStart.ElapsedMilliseconds)ms]" -ForegroundColor $overallColor
        Write-Host "Results: $ResultsJson" -ForegroundColor Gray
    }
    'Json' {
        $summary | ConvertTo-Json -Depth 5
    }
    'JUnit' {
        # Minimal JUnit XML suitable for CI test reporting
        $xml = [System.Text.StringBuilder]::new()
        [void]$xml.AppendLine('<?xml version="1.0" encoding="UTF-8"?>')
        [void]$xml.AppendLine('<testsuites>')
        foreach ($key in $Results.Keys) {
            $v       = $Results[$key]
            $failure = if ($v.status -eq 'FAIL') {
                "  <failure message=`"section failed`"><![CDATA[$($v.detail)]]></failure>"
            } else { '' }
            [void]$xml.AppendLine("  <testsuite name=`"Portable.$key`" tests=`"1`" failures=`"$(if($v.status -eq 'FAIL'){1}else{0})`" skipped=`"$(if($v.status -eq 'SKIP'){1}else{0})`" time=`"$([math]::Round($v.duration_ms/1000,3))`">")
            [void]$xml.AppendLine("    <testcase name=`"$key`" classname=`"Portable`" time=`"$([math]::Round($v.duration_ms/1000,3))`">")
            if ($failure) { [void]$xml.AppendLine($failure) }
            [void]$xml.AppendLine('    </testcase>')
            [void]$xml.AppendLine('  </testsuite>')
        }
        [void]$xml.AppendLine('</testsuites>')
        $xml.ToString()
    }
}

# Exit non-zero if any section failed
exit $(if ($overallStatus -eq 'PASS') { 0 } else { 1 })
