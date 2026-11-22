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
    <#
    .SYNOPSIS
        Counts clippy lints in the cargo JSON stream and fails when the tree does not build.
    .DESCRIPTION
        Lints are measured with `--cap-lints warn`, so a lint is only ever a warning and can never fail the build.
        That leaves one rule with no lint-denial reasoning and no stderr classification: the measurement is valid only
        when cargo exits 0 and its `build-finished` message says success:true, and a `compiler-message` at level error
        is never acceptable. Anything else (an unresolved import, a syntax error, a `build.rs` that panics, a linker
        or manifest error, a crash) throws "build failed", so a tree that does not build can neither count as a
        baseline nor hide behind a lint in another crate (codex review of #384). The same diagnostic is emitted once
        per target (lib and lib test), so a lint is counted once. Stderr is captured with the JSON and is quoted in
        the failure message, because a custom build command failure is only reported there as text.
    .PARAMETER Root
        Repository root (the workspace is Root/Native/pcai_core).
    #>
    param([Parameter(Mandatory)][string]$Root)
    if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) { throw 'cargo is not on PATH' }
    Push-Location (Join-Path $Root 'Native/pcai_core')
    try {
        $out = & cargo clippy --workspace --all-targets --no-deps --message-format=json `
            -- --cap-lints warn -A clippy::type_complexity 2>&1
        $exit = $LASTEXITCODE
    } finally { Pop-Location }
    $seen = [System.Collections.Generic.HashSet[string]]::new()
    $lints = 0
    $errors = [System.Collections.Generic.List[string]]::new()
    $stderrErrors = [System.Collections.Generic.List[string]]::new()
    $finished = $null
    foreach ($line in $out) {
        $text = [string]$line
        if (-not $text.StartsWith('{')) {
            # Non-JSON output is cargo's own text (stderr): keep the error lines for the message.
            if ($text -match '^\s*error') { $stderrErrors.Add($text.Trim()) }
            continue
        }
        try { $msg = $text | ConvertFrom-Json -ErrorAction Stop } catch { continue }
        if ($msg.reason -eq 'build-finished') { $finished = [bool]$msg.success; continue }
        if ($msg.reason -ne 'compiler-message') { continue }
        $level = [string]$msg.message.level
        if ($level -notin 'warning', 'error') { continue }
        $code = if ($msg.message.code) { [string]$msg.message.code.code } else { '' }
        $primary = @($msg.message.spans | Where-Object { $_.is_primary } | Select-Object -First 1)
        $where = if ($primary.Count -gt 0 -and $primary[0]) { "$($primary[0].file_name):$($primary[0].line_start):$($primary[0].column_start)" } else { '' }
        $key = "$level|$code|$($msg.message.message)|$where"
        if (-not $seen.Add($key)) { continue }
        if ($level -eq 'warning') {
            if ($code) { $lints++ }
        } elseif ([string]$msg.message.message -notmatch '^(aborting due to|could not compile)') {
            $errors.Add($(if ($code) { "$code $($msg.message.message)" } else { [string]$msg.message.message }))
        }
    }
    $detail = @($errors) + @($stderrErrors | Where-Object { $_ -notmatch '^error: could not compile' })
    $shown = (@($detail) | Select-Object -First 4) -join ' | '
    if ($errors.Count -gt 0) {
        throw "build failed: $($errors.Count) rustc error(s), not lints (exit $exit): $shown"
    }
    if ($exit -ne 0 -or $finished -ne $true) {
        $fin = if ($null -eq $finished) { 'no build-finished message' } else { "build-finished success=$finished" }
        throw "build failed: cargo clippy exited $exit ($fin), so this is not a measurement: $(if ($shown) { $shown } else { (@($out | ForEach-Object { [string]$_ }) | Select-Object -Last 3) -join ' | ' })"
    }
    $lints
}
function Measure-RustTests {
    <#
    .SYNOPSIS
        Sums the Rust test results and fails when the workspace does not compile.
    .PARAMETER Root
        Repository root (the workspace is Root/Native/pcai_core).
    #>
    param([Parameter(Mandatory)][string]$Root)
    if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) { throw 'cargo is not on PATH' }
    Push-Location (Join-Path $Root 'Native/pcai_core')
    # The CargoTools wrapper rewrites `cargo test` to `cargo nextest run` when CARGO_USE_NEXTEST is set, and it sets the
    # variable itself in the calling process after its first run. nextest prints no `test result:` lines, so this
    # measurement pins the variable to 0 and restores it.
    $savedNextest = $env:CARGO_USE_NEXTEST
    try {
        $env:CARGO_USE_NEXTEST = '0'
        $out = & cargo test --workspace --no-fail-fast --no-default-features --features server,ffi 2>&1
        $exit = $LASTEXITCODE
    } finally {
        Pop-Location
        $env:CARGO_USE_NEXTEST = $savedNextest
    }
    $lines = @($out | ForEach-Object { [string]$_ })
    # Compilation must have succeeded before any count means anything: a test binary that built can run while another
    # target failed to compile.
    $compileFailures = @($lines | Where-Object { $_ -match '^\s*error(\[E\d{4}\])?: could not compile' -or $_ -match '^\s*error\[E\d{4}\]' })
    if ($compileFailures.Count -gt 0) {
        throw "build failed: cargo test did not compile: $((@($compileFailures) | Select-Object -First 3) -join ' | ')"
    }
    # One `test result:` line per test binary: sum them all. Reading only the last line reported one crate's tests,
    # and a compile failure printed none, which the old code turned into 0/0 and called "unchanged".
    # --no-fail-fast keeps cargo running the later test binaries after one fails; without it a failure in a_fail.rs
    # hides a new failure in z_hidden.rs (both sides read Failed=1). Failures are also kept by name, qualified by the
    # binary that ran them, so a swapped failure with an unchanged count is still a change.
    $passed = 0; $failed = 0; $results = 0
    $binary = ''
    $failedNames = [System.Collections.Generic.List[string]]::new()
    foreach ($line in $lines) {
        if ($line -match '^\s*(?:Running|Doc-tests)\s+(.+?)(?:\s+\(|$)') { $binary = $Matches[1].Trim() }
        if ($line -match '^test (\S+) \.\.\. FAILED') { $failedNames.Add("${binary}::$($Matches[1])") }
        if ($line -match 'test result: \w+\.\s+(\d+) passed;\s+(\d+) failed') {
            $passed += [int]$Matches[1]; $failed += [int]$Matches[2]; $results++
        }
    }
    if ($results -eq 0 -or ($passed + $failed) -eq 0) {
        throw "build failed or nothing ran: cargo test exited $exit and reported no test results: $((@($lines) | Select-Object -Last 3) -join ' | ')"
    }
    if ($exit -ne 0 -and $failed -eq 0) {
        throw "cargo test exited $exit with no failed test counted (build error after some tests ran)"
    }
    [pscustomobject]@{ Passed = $passed; Failed = $failed; FailedNames = $failedNames.ToArray() }
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
    FailedBlocks = [int]$r.FailedBlocksCount; Result = [string]$r.Result
    Executed = $v.Executed; Violations = $v.Violations } | ConvertTo-Json))
'@

function Measure-PesterTests {
    <#
    .SYNOPSIS
        Counts passed and failed Portable Pester tests in a child process and fails closed on an unusable child.
    .DESCRIPTION
        Failed = failed tests + failed containers (files that did not load) + failed blocks (a throwing
        BeforeAll/AfterAll). A Result other than Passed with every counter at 0 counts as one failure, so an
        unexplained failure never reads as zero. A child that exits non-zero, writes no result file, writes
        something that is not JSON, or omits a field is an error, not a measurement.
    .PARAMETER Root
        Repository root whose Tests/Unit is measured.
    #>
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
        $childExit = $LASTEXITCODE
        if ($childExit -ne 0) { throw "the Pester child process exited $childExit" }
        if (-not (Test-Path -LiteralPath $outPath)) { throw 'the Pester child process wrote no result' }
        try { $r = Get-Content -LiteralPath $outPath -Raw | ConvertFrom-Json -ErrorAction Stop }
        catch { throw "the Pester child result is not valid JSON: $($_.Exception.Message)" }
    } finally {
        Remove-Item -LiteralPath $scratch -Recurse -Force -ErrorAction SilentlyContinue
    }
    foreach ($field in 'Passed', 'Failed', 'FailedContainers', 'FailedBlocks', 'Executed', 'Result') {
        if ($null -eq $r -or $null -eq $r.PSObject.Properties[$field] -or $null -eq $r.$field) { throw "the Pester child result is missing '$field'" }
    }
    # Failures are a metric here, not a stop: the gate compares them between refs. Only a run that is not a
    # measurement at all (nothing executed) is an error.
    if ([int]$r.Executed -lt 1) { throw "Pester executed no tests: $(@($r.Violations) -join '; ')" }
    $failed = [int]$r.Failed + [int]$r.FailedContainers + [int]$r.FailedBlocks
    if ([string]$r.Result -ne 'Passed' -and $failed -eq 0) { $failed = 1 }
    [pscustomobject]@{ Passed = [int]$r.Passed; Failed = $failed }
}

function Measure-ScriptAnalyzer {
    param([Parameter(Mandatory)][string]$Root)
    if (-not (Get-Module -ListAvailable PSScriptAnalyzer)) { throw 'PSScriptAnalyzer is not installed' }
    $modules = Join-Path $Root 'Modules'
    if (-not (Test-Path $modules)) { throw "no Modules directory at $modules" }
    @(Invoke-ScriptAnalyzer -Path $modules -Recurse -Severity Warning, Error).Count
}

function Compare-RustFailedNames {
    <#
    .SYNOPSIS
        Returns the failed Rust tests that fail in Head and did not fail in Base, compared by name.
    #>
    param([string[]]$Base = @(), [string[]]$Head = @())
    $known = [System.Collections.Generic.HashSet[string]]::new([string[]]@($Base))
    @($Head | Where-Object { -not $known.Contains($_) })
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
    $metrics.RustFailedNames = if ($r) { @($r.FailedNames) } else { $null }
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

# A failure that is new by name is a regression even when the failing count is unchanged (a swapped failure).
if ($base -and $null -ne $head.Metrics.RustFailedNames -and $null -ne $base.Metrics.RustFailedNames) {
    $newFailed = @(Compare-RustFailedNames -Base @($base.Metrics.RustFailedNames) -Head @($head.Metrics.RustFailedNames))
    $newStatus = if ($newFailed.Count -gt 0) { 'regressed' } else { 'unchanged' }
    $deltas['RustNewFailedTests'] = [pscustomobject]@{ Label = 'Rust tests newly failing (by name)'; Base = 0; Head = $newFailed.Count; Delta = $newFailed.Count; Status = $newStatus }
    if ($newStatus -eq 'regressed') { $regressions.Add('RustNewFailedTests') }
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
