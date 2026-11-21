#Requires -Version 7.0
<#
.SYNOPSIS
    Runs the checks that .github/workflows/ci.yml runs, on this machine, with the local accelerators.

.DESCRIPTION
    GitHub-hosted runners are disabled for this account and the Windows CI jobs have no
    self-hosted runner, so the quickest honest signal before a push is a local run of the same
    commands. This script is a thin orchestrator: every step calls the script or command that
    ci.yml calls, and Tests/Unit/Invoke-PcaiCi.Tests.ps1 fails when ci.yml and this list drift.

    Tiers:
      Quick  credential scan, workflow syntax, PSScriptAnalyzer on the files this branch changed
             (Tools/Test-ChangedPowerShellLint.ps1), rustfmt. Seconds to about a minute.
      Full   everything ci.yml runs that this script mirrors: credential scan, workflow syntax,
             the full PSScriptAnalyzer pass (about 12 minutes serial), rustfmt, cargo check,
             clippy, the pcai_inference unit tests and the Pester Unit and Boot suites (the
             17-path list ci.yml uses).

    Accelerators:
      changed-file lint  Quick tier analyzes only changed PowerShell files (seconds, not 12 minutes).
      nextest            `cargo nextest run` replaces `cargo test --lib` when cargo-nextest is installed.
                         Nextest skips doctests, which ci.yml also skips (`--lib`).

    sccache is deliberately not toggled here: the CargoTools wrapper applies it to every Rust build on
    this fleet, and a measured cargo check into a fresh target directory got 0 hits (see the PR).

    CI-only jobs that need hosted services or the downloaded native artifacts (llamacpp build,
    FFI integration, .NET matrix, deploy runtime, cargo audit, coverage upload) are not mirrored.

.PARAMETER Tier
    Quick (default) or Full.

.PARAMETER Only
    Run only these step ids. Combine with -Plan to list the ids.

.PARAMETER Skip
    Skip these step ids.

.PARAMETER Plan
    Print the steps and environment that would run, without running anything.

.PARAMETER PassThru
    Emit the step objects (plan) or result objects (run).

.PARAMETER FailFast
    Stop at the first failing step.

.PARAMETER NoNextest
    Use `cargo test` even when cargo-nextest is installed.

.PARAMETER NextestCommand
    Name or path probed for cargo-nextest. Exposed so tests can simulate its absence.

.EXAMPLE
    pwsh Tests/Invoke-PcaiCi.ps1 -Plan
    Lists the Quick steps.

.EXAMPLE
    pwsh Tests/Invoke-PcaiCi.ps1 -Tier Full -FailFast
    Runs everything this script mirrors and stops at the first failure.
#>
[CmdletBinding()]
param(
    [ValidateSet('Quick', 'Full')]
    [string]$Tier = 'Quick',

    [string[]]$Only,

    [string[]]$Skip,

    [switch]$Plan,

    [switch]$PassThru,

    [switch]$FailFast,

    [switch]$NoNextest,

    [string]$NextestCommand = 'cargo-nextest',

    [string]$ResultsDir
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
$rustWorkspace = Join-Path $repoRoot 'Native/pcai_core'
$inferenceCrate = Join-Path $rustWorkspace 'pcai_inference'

# The Pester path list ci.yml hands to Invoke-Pester. A unit test compares this list to ci.yml.
$pesterPaths = @(
    'Tests/Unit',
    'Tests/Boot/WorkloadResourceProfile.Tests.ps1',
    'Tests/Boot/CloudClientGate.Tests.ps1',
    'Tests/Boot/ArchiveSourceCustody.Tests.ps1',
    'Tests/Boot/CloudCacheRepairIdentity.Tests.ps1',
    'Tests/Boot/CloudMountGate.Tests.ps1',
    'Tests/Boot/CloudMountGateRunner.Tests.ps1',
    'Tests/Boot/MachineCapabilities.Tests.ps1',
    'Tests/Boot/MediaToolchain.Tests.ps1',
    'Tests/Boot/HardwareMeasurementContracts.Tests.ps1',
    'Tests/Boot/DevModuleDeployment.Tests.ps1',
    'Tests/Boot/FleetToolRouting.Tests.ps1',
    'Tests/Boot/TelemetryPromptContracts.Tests.ps1',
    'Tests/Boot/PackagedRuntimeRoot.Tests.ps1',
    'Tests/Boot/ProfileBootstrap.Tests.ps1',
    'Tests/Boot/CrashProfileSafety.Tests.ps1',
    'Tests/Boot/CrashTelemetry.Tests.ps1'
)

$nextestFound = (-not $NoNextest) -and [bool](Get-Command $NextestCommand -CommandType Application -ErrorAction SilentlyContinue)

function New-CiStep {
    param(
        [string]$Id,
        [string[]]$Tiers,
        [string]$Description,
        [string]$CiCommand,
        [string[]]$Command,
        [string]$WorkingDirectory = $repoRoot
    )
    [pscustomobject]@{
        Id               = $Id
        Tiers            = $Tiers
        Description      = $Description
        CiCommand        = $CiCommand
        Command          = $Command
        WorkingDirectory = $WorkingDirectory
    }
}

$pesterProbe = @'
New-Item -ItemType Directory -Path Tests/TestResults -Force | Out-Null
$config = New-PesterConfiguration
if (Test-Path 'Tests/PesterConfiguration.psd1') {
    $config = New-PesterConfiguration -Hashtable (Import-PowerShellDataFile 'Tests/PesterConfiguration.psd1')
}
$config.Run.Path = @(__PATHS__)
$config.Run.Exit = $true
$config.Output.Verbosity = 'Normal'
Invoke-Pester -Configuration $config
'@.Replace('__PATHS__', (($pesterPaths | ForEach-Object { "'$_'" }) -join ','))

$rustTestCommand = if ($nextestFound) {
    @('cargo', 'nextest', 'run', '--no-default-features', '--features', 'server,ffi', '--lib')
} else {
    @('cargo', 'test', '--no-default-features', '--features', 'server,ffi', '--lib')
}

$allSteps = @(
    New-CiStep -Id 'CredentialScan' -Tiers @('Quick','Full') -Description 'Credential scan' `
        -CiCommand '.\Tools\Test-CredentialScan.ps1' `
        -Command @('pwsh', '-NoProfile', '-File', (Join-Path $repoRoot 'Tools/Test-CredentialScan.ps1'))
    New-CiStep -Id 'WorkflowSyntax' -Tiers @('Quick','Full') -Description 'Workflow syntax' `
        -CiCommand '.\Tools\Test-WorkflowSyntax.ps1' `
        -Command @('pwsh', '-NoProfile', '-File', (Join-Path $repoRoot 'Tools/Test-WorkflowSyntax.ps1'))
    New-CiStep -Id 'PowerShellLintChanged' -Tiers @('Quick') -Description 'PSScriptAnalyzer on changed files (local accelerator, not in ci.yml)' `
        -CiCommand '' `
        -Command @('pwsh', '-NoProfile', '-File', (Join-Path $repoRoot 'Tools/Test-ChangedPowerShellLint.ps1'))
    New-CiStep -Id 'PowerShellLint' -Tiers @('Full') -Description 'PSScriptAnalyzer lint, all files' `
        -CiCommand '.\Build.ps1 -Component lint -LintProfile powershell' `
        -Command @('pwsh', '-NoProfile', '-File', (Join-Path $repoRoot 'Build.ps1'), '-Component', 'lint', '-LintProfile', 'powershell')
    New-CiStep -Id 'RustFmt' -Tiers @('Quick','Full') -Description 'rustfmt check' `
        -CiCommand '.\Build.ps1 -Component format -LintProfile rust-inference-fmt -DependencyStrategy locked' `
        -Command @('pwsh', '-NoProfile', '-File', (Join-Path $repoRoot 'Build.ps1'), '-Component', 'format', '-LintProfile', 'rust-inference-fmt', '-DependencyStrategy', 'locked')
    New-CiStep -Id 'RustCheck' -Tiers @('Full') -Description 'cargo check' `
        -CiCommand '.\Build.ps1 -Component lint -LintProfile rust-inference-check -DependencyStrategy locked' `
        -Command @('pwsh', '-NoProfile', '-File', (Join-Path $repoRoot 'Build.ps1'), '-Component', 'lint', '-LintProfile', 'rust-inference-check', '-DependencyStrategy', 'locked')
    New-CiStep -Id 'RustClippy' -Tiers @('Full') -Description 'clippy' `
        -CiCommand '.\Build.ps1 -Component lint -LintProfile rust-inference-clippy -DependencyStrategy locked' `
        -Command @('pwsh', '-NoProfile', '-File', (Join-Path $repoRoot 'Build.ps1'), '-Component', 'lint', '-LintProfile', 'rust-inference-clippy', '-DependencyStrategy', 'locked')
    New-CiStep -Id 'RustTest' -Tiers @('Full') -Description 'pcai_inference unit tests' `
        -CiCommand 'cargo test --no-default-features --features server,ffi --lib' `
        -Command $rustTestCommand -WorkingDirectory $inferenceCrate
    New-CiStep -Id 'PesterUnitBoot' -Tiers @('Full') -Description 'Pester Unit and Boot suites' `
        -CiCommand "`$config.Run.Path" `
        -Command @('pwsh', '-NoProfile', '-Command', $pesterProbe)
)

# -Only names steps explicitly, so it ignores the tier; otherwise asking for a Full step from the
# default Quick tier would silently run nothing and exit 0.
$steps = if ($Only) { @($allSteps) } else { @($allSteps | Where-Object { $Tier -in $_.Tiers }) }
$knownIds = @($allSteps.Id)
foreach ($requested in @($Only) + @($Skip)) {
    if ($requested -and $requested -notin $knownIds) {
        throw "Unknown step id '$requested'. Known ids: $($knownIds -join ', ')"
    }
}
if ($Only) { $steps = @($steps | Where-Object { $_.Id -in $Only }) }
if ($Skip) { $steps = @($steps | Where-Object { $_.Id -notin $Skip }) }
if ($steps.Count -eq 0) { throw 'No steps selected: nothing would run, so refusing to report success.' }

if ($Plan) {
    if ($PassThru) { return $steps }
    Write-Host "PC-AI local CI plan (tier $Tier, nextest=$nextestFound)"
    foreach ($step in $steps) {
        Write-Host ("  {0,-22} {1}" -f $step.Id, $step.Description)
    }
    return
}

if (-not $ResultsDir) { $ResultsDir = Join-Path $repoRoot '.pcai/build/logs/pcai-ci' }
[IO.Directory]::CreateDirectory($ResultsDir) | Out-Null

$results = [System.Collections.Generic.List[object]]::new()
foreach ($step in $steps) {
    $log = Join-Path $ResultsDir "pcai-ci-$($step.Id).log"
    Write-Host "==> $($step.Id): $($step.Description)"
    $watch = [Diagnostics.Stopwatch]::StartNew()
    Push-Location $step.WorkingDirectory
    try {
        $exe = $step.Command[0]
        $arguments = @($step.Command | Select-Object -Skip 1)
        & $exe @arguments 2>&1 | Tee-Object -FilePath $log | Out-Host
        $exit = $LASTEXITCODE
    } catch {
        $_ | Out-String | Tee-Object -FilePath $log -Append | Out-Host
        $exit = 1
    } finally {
        Pop-Location
        $watch.Stop()
    }
    $results.Add([pscustomobject]@{
        Id         = $step.Id
        Status     = if ($exit -eq 0) { 'PASS' } else { 'FAIL' }
        ExitCode   = $exit
        DurationMs = [int]$watch.ElapsedMilliseconds
        Log        = $log
    })
    if ($exit -ne 0 -and $FailFast) { break }
}

$results | Format-Table Id, Status, ExitCode, DurationMs -AutoSize | Out-Host
$summary = [pscustomobject]@{
    Tier    = $Tier
    Nextest = $nextestFound
    Results = $results
}
[IO.File]::WriteAllText((Join-Path $ResultsDir 'pcai-ci-last.json'), ($summary | ConvertTo-Json -Depth 5), [Text.UTF8Encoding]::new($false))

$failed = @($results | Where-Object Status -eq 'FAIL')
if ($PassThru) { $results }
if ($failed.Count -gt 0) {
    Write-Error "Local CI failed: $($failed.Id -join ', ')" -ErrorAction Continue
    exit 1
}
