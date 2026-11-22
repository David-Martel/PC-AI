#Requires -Version 7.0
<#
.SYNOPSIS
    A/B code quality comparison between two git refs for PC_AI.
.DESCRIPTION
    Measures Rust Clippy findings, Rust test counts, Pester unit test counts, and PSScriptAnalyzer violations
    on the current working tree (head), then on BaseRef in a temporary detached worktree, and reports deltas.
    Dirty working trees are allowed (warned): head metrics include them.

    The base ref is checked out into a temporary `git worktree add --detach` under the system temp directory and
    removed (and pruned) in a finally block, so the live checkout is never switched and no registered worktree
    outlives the run. It used to run `git checkout <ref>` in place, which left a detached HEAD behind and, on a
    dirty tree, failed silently so the "base" numbers were really HEAD's.

    The gate fails closed. A metric that cannot be measured (no cargo, no Pester 5, no PSScriptAnalyzer, a Rust
    build that does not compile, a test run that executed nothing) is a measurement error, reported and exit 1,
    never a silent 0. It used to count `"level":"warning"` while running `-D warnings` (always 0), read only the
    last `test result:` line, and report 0 for a missing tool.
.PARAMETER BaseRef
    Baseline git ref (default: origin/main).
.PARAMETER HeadRef
    Label for the head side. Head metrics always come from the working tree of this checkout.
.PARAMETER Format
    Output format: Table (default), Json, or Markdown.
.PARAMETER FailOnRegression
    Exit 1 if any metric regresses. Measurement errors exit 1 regardless.
.PARAMETER ResultsDir
    Directory for result JSON files (relative to repo root).
.EXAMPLE
    .\Tests\Invoke-QualityGate.ps1 -Format Markdown -FailOnRegression
#>
param(
    [string]$BaseRef = 'origin/main',
    [string]$HeadRef = 'HEAD',
    [ValidateSet('Table', 'Json', 'Markdown')][string]$Format = 'Table',
    [switch]$FailOnRegression,
    [string]$ResultsDir = 'Tests/Results'
)

. (Join-Path $PSScriptRoot 'Helpers/Get-PesterRunVerdict.ps1')

# --- metric helpers: every one takes the root it measures and throws when it cannot measure --------------------

function Measure-ClippyFindings {
    param([Parameter(Mandatory)][string]$Root)
    if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) { throw 'cargo is not on PATH' }
    Push-Location (Join-Path $Root 'Native/pcai_core')
    try {
        $out = & cargo clippy --workspace --all-targets --no-deps --message-format=json `
            -- -D warnings -A clippy::type_complexity 2>&1
        $exit = $LASTEXITCODE
    } finally { Pop-Location }
    # With -D warnings a lint is reported at level "error", so counting "warning" read 0 whatever the code did.
    # Count every diagnostic that carries a lint or error code, at either level; the trailing "aborting due to"
    # summary has no code and is not a finding.
    $findings = 0
    $parsed = 0
    foreach ($line in $out) {
        $text = [string]$line
        if (-not $text.StartsWith('{')) { continue }
        try { $msg = $text | ConvertFrom-Json -ErrorAction Stop } catch { continue }
        if ($msg.reason -ne 'compiler-message') { continue }
        $parsed++
        if ($msg.message.level -in 'warning', 'error' -and $null -ne $msg.message.code) { $findings++ }
    }
    if ($exit -ne 0 -and $findings -eq 0) {
        throw "cargo clippy exited $exit with no lint diagnostics (toolchain or build failure): $((@($out) | Select-Object -Last 3) -join ' | ')"
    }
    $findings
}

function Measure-RustTests {
    param([Parameter(Mandatory)][string]$Root)
    if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) { throw 'cargo is not on PATH' }
    Push-Location (Join-Path $Root 'Native/pcai_core')
    try {
        $out = & cargo test --workspace --no-default-features --features server,ffi 2>&1
        $exit = $LASTEXITCODE
    } finally { Pop-Location }
    # One `test result:` line per test binary: sum them all. Reading only the last line reported one crate's tests,
    # and a compile failure printed none, which the old code turned into 0/0 and called "unchanged".
    $passed = 0; $failed = 0; $lines = 0
    foreach ($line in $out) {
        if ([string]$line -match 'test result: \w+\.\s+(\d+) passed;\s+(\d+) failed') {
            $passed += [int]$Matches[1]; $failed += [int]$Matches[2]; $lines++
        }
    }
    if ($lines -eq 0 -or ($passed + $failed) -eq 0) {
        throw "cargo test exited $exit and reported no test results (compile failure or nothing ran): $((@($out) | Select-Object -Last 3) -join ' | ')"
    }
    if ($exit -ne 0 -and $failed -eq 0) {
        throw "cargo test exited $exit with no failed test counted (build error after some tests ran)"
    }
    [pscustomobject]@{ Passed = $passed; Failed = $failed }
}

# Child process for the Pester measurement. Head and base are measured in separate sessions: the modules under
# test are imported by name (PC-AI.USB, PC-AI.Virtualization, ...), so a second tree measured in the same session
# collides with the first ("Multiple script or manifest modules named ... are currently loaded") and the base side
# would be inflated with failures that are not the base ref's. The child also runs under the default
# $ErrorActionPreference, because Stop leaks into the tests and turns tolerated errors into failures.
$script:PesterChildScript = @'
param([string]$Root, [string]$Helper, [string]$OutPath)
. $Helper
Import-Module Pester -MinimumVersion 5.0.0
Set-Location $Root
$cfg = New-PesterConfiguration
$cfg.Run.Path = Join-Path $Root 'Tests/Unit'
$cfg.Filter.Tag = @('Portable')
$cfg.Output.Verbosity = 'None'
$cfg.Run.PassThru = $true
$r = Invoke-Pester -Configuration $cfg
$v = Get-PesterRunVerdict -Result $r -MaxFailedTests ([int]::MaxValue) -MaxFailedContainers ([int]::MaxValue) -MaxFailedBlocks ([int]::MaxValue)
[IO.File]::WriteAllText($OutPath, ([ordered]@{
    Passed = [int]$r.PassedCount; Failed = [int]$r.FailedCount; FailedContainers = [int]$r.FailedContainersCount
    Executed = $v.Executed; Violations = $v.Violations } | ConvertTo-Json))
'@

function Measure-PesterTests {
    param([Parameter(Mandatory)][string]$Root)
    if (-not (Get-Module -ListAvailable Pester | Where-Object Version -ge '5.0')) { throw 'Pester 5+ is not installed' }
    if (-not (Test-Path (Join-Path $Root 'Tests/Unit'))) { throw "no Pester path at $(Join-Path $Root 'Tests/Unit')" }
    $scratch = Join-Path ([System.IO.Path]::GetTempPath()) ('pcai-qg-pester-' + [guid]::NewGuid().ToString('N'))
    [System.IO.Directory]::CreateDirectory($scratch) | Out-Null
    try {
        $childScript = Join-Path $scratch 'measure.ps1'
        $outPath = Join-Path $scratch 'result.json'
        [System.IO.File]::WriteAllText($childScript, $script:PesterChildScript)
        $helper = Join-Path $PSScriptRoot 'Helpers/Get-PesterRunVerdict.ps1'
        & pwsh -NoProfile -File $childScript -Root $Root -Helper $helper -OutPath $outPath *> $null
        if (-not (Test-Path -LiteralPath $outPath)) { throw "the Pester child process wrote no result (exit $LASTEXITCODE)" }
        $r = Get-Content -LiteralPath $outPath -Raw | ConvertFrom-Json
    } finally {
        Remove-Item -LiteralPath $scratch -Recurse -Force -ErrorAction SilentlyContinue
    }
    # Failures are a metric here, not a stop: the gate compares them between refs. Only a run that is not a
    # measurement at all (nothing executed) is an error. A container that did not load counts as a failure so a
    # syntax error is a regression instead of invisible.
    if ($r.Executed -lt 1) { throw "Pester executed no tests: $($r.Violations -join '; ')" }
    [pscustomobject]@{ Passed = [int]$r.Passed; Failed = [int]$r.Failed + [int]$r.FailedContainers }
}

function Measure-ScriptAnalyzer {
    param([Parameter(Mandatory)][string]$Root)
    if (-not (Get-Module -ListAvailable PSScriptAnalyzer)) { throw 'PSScriptAnalyzer is not installed' }
    $modules = Join-Path $Root 'Modules'
    if (-not (Test-Path $modules)) { throw "no Modules directory at $modules" }
    @(Invoke-ScriptAnalyzer -Path $modules -Recurse -Severity Warning, Error).Count
}

function Invoke-Metrics {
    param([string]$Label, [string]$Root)
    Write-Host "  Collecting $Label ..." -ForegroundColor Gray
    $metrics = [ordered]@{}
    $errors = [System.Collections.Generic.List[string]]::new()
    $try = {
        param([string]$Name, [scriptblock]$Measure)
        try { & $Measure } catch { $errors.Add("${Label}: $Name could not be measured: $($_.Exception.Message)"); $null }
    }
    $metrics.ClippyWarnings = & $try 'clippy' { Measure-ClippyFindings -Root $Root }
    $r = & $try 'Rust tests' { Measure-RustTests -Root $Root }
    $metrics.RustTestsPassed = if ($r) { $r.Passed } else { $null }
    $metrics.RustTestsFailed = if ($r) { $r.Failed } else { $null }
    $p = & $try 'Pester tests' { Measure-PesterTests -Root $Root }
    $metrics.PesterPassed = if ($p) { $p.Passed } else { $null }
    $metrics.PesterFailed = if ($p) { $p.Failed } else { $null }
    $metrics.PSAnalyzerViolations = & $try 'PSScriptAnalyzer' { Measure-ScriptAnalyzer -Root $Root }
    [pscustomobject]@{ Metrics = [pscustomobject]$metrics; Errors = $errors.ToArray() }
}

function Get-DeltaStatus ([string]$Metric, [int]$Delta) {
    if ($Delta -eq 0) { return 'unchanged' }
    $lowerBetter = 'ClippyWarnings', 'RustTestsFailed', 'PesterFailed', 'PSAnalyzerViolations'
    if ($Metric -in $lowerBetter) {
        if ($Delta -lt 0) { return 'improved' } else { return 'regressed' }
    }
    if ($Delta -gt 0) { return 'improved' } else { return 'regressed' }
}

function Invoke-WithBaseRefWorktree {
    <#
    .SYNOPSIS
        Runs a script block against a temporary detached worktree of a ref and always removes the worktree.
    .DESCRIPTION
        `git worktree add --detach` under ParentDir, never `git checkout` in the live tree: the checkout used to leave a
        detached HEAD behind and failed silently on a dirty tree. The removal and prune are in a finally block, so
        the registered worktree set is the same afterwards even when the script block throws.
    .PARAMETER RepoRoot
        The repository whose ref is checked out.
    .PARAMETER Ref
        Commit-ish to check out.
    .PARAMETER Script
        Script block; receives the worktree path as its argument. Its output is returned.
    .PARAMETER ParentDir
        Where the temporary directory is created. Default: the system temp directory.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$RepoRoot,
        [Parameter(Mandatory)][string]$Ref,
        [Parameter(Mandatory)][scriptblock]$Script,
        [string]$ParentDir = [System.IO.Path]::GetTempPath()
    )
    $tree = Join-Path $ParentDir ('pcai-qg-base-' + [guid]::NewGuid().ToString('N'))
    $added = $false
    try {
        git -C $RepoRoot worktree add --quiet --detach $tree $Ref 2>&1 | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "git worktree add failed for $Ref" }
        $added = $true
        & $Script $tree
    } finally {
        # A build inside the temporary tree may leave junctions or symlinks (this repo's target/ directories can be
        # junctions into the shared T:\RustCache cache). Remove each link itself, non-recursively, first: neither
        # `git worktree remove` nor a recursive Remove-Item is allowed to walk through one into the shared cache.
        if (Test-Path -LiteralPath $tree) {
            foreach ($link in @(Get-ChildItem -LiteralPath $tree -Recurse -Force -Attributes ReparsePoint -ErrorAction SilentlyContinue)) {
                try { $link.Delete() } catch { Write-Warning "could not remove link $($link.FullName): $($_.Exception.Message)" }
            }
        }
        if ($added) { git -C $RepoRoot worktree remove --force $tree 2>&1 | Out-Null }
        if (Test-Path -LiteralPath $tree) { Remove-Item -LiteralPath $tree -Recurse -Force -ErrorAction SilentlyContinue }
        git -C $RepoRoot worktree prune 2>&1 | Out-Null
    }
}

# --- body: only when run, so the unit tests can dot-source the helpers above -----------------------------------

if ($MyInvocation.InvocationName -eq '.') { return }

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$RepoRoot = git -C $PSScriptRoot rev-parse --show-toplevel
$ResultsDirAbs = if ([System.IO.Path]::IsPathRooted($ResultsDir)) { $ResultsDir } else { Join-Path $RepoRoot $ResultsDir }

if ((git -C $RepoRoot status --porcelain 2>&1) -ne '') {
    Write-Warning 'Working tree has uncommitted changes: head metrics include them.'
}

Write-Host "=== Quality Gate: $HeadRef vs $BaseRef ===" -ForegroundColor Cyan
$measurementErrors = [System.Collections.Generic.List[string]]::new()
$head = Invoke-Metrics -Label "HEAD ($HeadRef)" -Root $RepoRoot
foreach ($e in $head.Errors) { $measurementErrors.Add($e) }

$base = $null
try {
    Write-Host "  Checking out $BaseRef into a temporary worktree ..." -ForegroundColor Gray
    $base = Invoke-WithBaseRefWorktree -RepoRoot $RepoRoot -Ref $BaseRef -Script {
        param($tree)
        Invoke-Metrics -Label "BASE ($BaseRef)" -Root $tree
    }
    foreach ($e in $base.Errors) { $measurementErrors.Add($e) }
} catch {
    $measurementErrors.Add("BASE ($BaseRef): $($_.Exception.Message)")
}

# --- compute deltas ----------------------------------------------------------------------------------------------

$metricDefs = @(
    @{ Key = 'ClippyWarnings'; Label = 'Clippy findings' }
    @{ Key = 'RustTestsPassed'; Label = 'Rust tests passing' }
    @{ Key = 'RustTestsFailed'; Label = 'Rust tests failing' }
    @{ Key = 'PesterPassed'; Label = 'Pester tests passing' }
    @{ Key = 'PesterFailed'; Label = 'Pester tests failing (incl. unloadable files)' }
    @{ Key = 'PSAnalyzerViolations'; Label = 'PSScriptAnalyzer violations' }
)

$deltas = [ordered]@{}
$regressions = [System.Collections.Generic.List[string]]::new()
foreach ($m in $metricDefs) {
    $b = if ($base) { $base.Metrics.($m.Key) } else { $null }
    $h = $head.Metrics.($m.Key)
    if ($null -eq $b -or $null -eq $h) {
        $deltas[$m.Key] = [pscustomobject]@{ Label = $m.Label; Base = $b; Head = $h; Delta = $null; Status = 'unmeasured' }
        continue
    }
    $delta = [int]$h - [int]$b
    $status = Get-DeltaStatus -Metric $m.Key -Delta $delta
    $deltas[$m.Key] = [pscustomobject]@{ Label = $m.Label; Base = [int]$b; Head = [int]$h; Delta = $delta; Status = $status }
    if ($status -eq 'regressed') { $regressions.Add($m.Key) }
}

$overall = if ($measurementErrors.Count -gt 0) { 'ERROR' }
elseif ($regressions.Count -gt 0) { 'REGRESSED' }
elseif (@($deltas.Values | Where-Object Status -eq 'improved').Count -gt 0) { 'IMPROVED' }
else { 'UNCHANGED' }

# --- persist results ---------------------------------------------------------------------------------------------

$null = New-Item -ItemType Directory -Path $ResultsDirAbs -Force
$outFile = Join-Path $ResultsDirAbs "quality-gate-$(Get-Date -Format 'yyyyMMdd_HHmmss').json"
$jsonDeltas = [ordered]@{}
foreach ($kv in $deltas.GetEnumerator()) {
    $jsonDeltas[$kv.Key] = [ordered]@{ base = $kv.Value.Base; head = $kv.Value.Head; delta = $kv.Value.Delta; status = $kv.Value.Status }
}
[ordered]@{
    base_ref           = $BaseRef
    head_ref           = $HeadRef
    timestamp          = (Get-Date -Format 'o')
    deltas             = $jsonDeltas
    overall            = $overall
    regressions        = $regressions.ToArray()
    measurement_errors = $measurementErrors.ToArray()
} | ConvertTo-Json -Depth 5 | Set-Content -Path $outFile -Encoding utf8
Write-Host "Results saved: $outFile" -ForegroundColor Gray

# --- render output -----------------------------------------------------------------------------------------------

$statusColor = @{ improved = 'Green'; unchanged = 'Gray'; regressed = 'Red'; unmeasured = 'Red' }
$fmtDelta = { param($d) if ($null -eq $d) { 'n/a' } elseif ($d -gt 0) { "+$d" } else { "$d" } }

switch ($Format) {
    'Json' { Get-Content $outFile }
    'Markdown' {
        "## Quality Gate: $HeadRef vs $BaseRef`n"
        '| Metric | Base | Head | Delta | Status |'
        '|--------|------|------|-------|--------|'
        foreach ($kv in $deltas.GetEnumerator()) {
            $d = $kv.Value
            "| $($d.Label) | $($d.Base) | $($d.Head) | $(& $fmtDelta $d.Delta) | $($d.Status) |"
        }
        ''
        foreach ($e in $measurementErrors) { "- measurement error: $e" }
        "**Overall: $overall** ($($regressions.Count) regression(s), $($measurementErrors.Count) measurement error(s))"
    }
    default {
        Write-Host ''
        Write-Host ('{0,-45} {1,6} {2,6} {3,7}  {4}' -f 'Metric', 'Base', 'Head', 'Delta', 'Status') -ForegroundColor Cyan
        Write-Host ('-' * 75) -ForegroundColor DarkGray
        foreach ($kv in $deltas.GetEnumerator()) {
            $d = $kv.Value
            Write-Host ('{0,-45} {1,6} {2,6} {3,7}  {4}' -f $d.Label, $d.Base, $d.Head, (& $fmtDelta $d.Delta), $d.Status) -ForegroundColor $statusColor[$d.Status]
        }
        Write-Host ('-' * 75) -ForegroundColor DarkGray
        foreach ($e in $measurementErrors) { Write-Host "  measurement error: $e" -ForegroundColor Red }
        $oc = switch ($overall) { 'ERROR' { 'Red' } 'REGRESSED' { 'Red' } 'IMPROVED' { 'Green' } default { 'Gray' } }
        Write-Host "Overall: $overall  ($($regressions.Count) regression(s), $($measurementErrors.Count) measurement error(s))" -ForegroundColor $oc
        Write-Host ''
    }
}

if ($measurementErrors.Count -gt 0) {
    Write-Error "Quality gate could not measure: $($measurementErrors.Count) error(s). A metric that cannot be measured is not a pass." -ErrorAction Continue
    exit 1
}
if ($FailOnRegression -and $regressions.Count -gt 0) {
    Write-Error "Quality gate FAILED: $($regressions.Count) regression(s): $($regressions -join ', ')" -ErrorAction Continue
    exit 1
}
