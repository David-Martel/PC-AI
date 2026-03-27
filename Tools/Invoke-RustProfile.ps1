#Requires -Version 7.0
<#
.SYNOPSIS
    Unified CPU profiling, benchmarking, and memory profiling for PC_AI Rust crates.

.DESCRIPTION
    Wraps cargo flamegraph, cargo bench, and working-set memory measurement into a
    single interface targeting the workspace at Native/pcai_core/.

    Profile modes:
      Flamegraph  CPU sampling via cargo-flamegraph -> SVG output
      Bench       Criterion/cargo bench with structured JSON output
      Memory      Peak working-set measurement during test execution
      Summary     Tool availability matrix and benchmark target inventory

.PARAMETER Profile
    Profiling mode: Flamegraph, Bench, Memory, or Summary.

.PARAMETER Crate
    Target crate package name (e.g. 'pcai-inference', 'pcai-media').
    Omit to target the full workspace.

.PARAMETER OutputDir
    Directory for generated reports. Default: Reports/profiles

.PARAMETER Features
    Additional cargo features to pass via --features.

.PARAMETER Release
    Use --release profile. Defaults on for Flamegraph and Bench.

.PARAMETER Open
    Open the primary output (SVG or criterion HTML) after generation.

.EXAMPLE
    .\Tools\Invoke-RustProfile.ps1 -Profile Summary

.EXAMPLE
    .\Tools\Invoke-RustProfile.ps1 -Profile Flamegraph -Crate pcai-inference -Open

.EXAMPLE
    .\Tools\Invoke-RustProfile.ps1 -Profile Bench -Crate pcai-core-lib

.EXAMPLE
    .\Tools\Invoke-RustProfile.ps1 -Profile Memory -Crate pcai-media -Features cuda
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [ValidateSet('Flamegraph', 'Bench', 'Memory', 'Summary')]
    [string]$Profile,

    [string]$Crate,

    [string]$OutputDir = 'Reports/profiles',

    [string[]]$Features,

    [switch]$Release,

    [switch]$Open
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$RepoRoot  = Split-Path -Parent $PSScriptRoot
$Workspace = Join-Path $RepoRoot 'Native\pcai_core'
$AbsOutput = Join-Path $RepoRoot $OutputDir

# Workspace package names derived from Cargo.toml members (underscore -> hyphen)
$WorkspaceCrates = @(
    'pcai-core-lib',
    'pcai-perf-cli',
    'pcai-ollama-rs',
    'pcai-inference',
    'pcai-media-model',
    'pcai-media',
    'pcai-media-server'
)

function Get-Timestamp { [DateTime]::UtcNow.ToString('yyyyMMdd_HHmmss') }

function Get-GitHash {
    $hash = & git -C $RepoRoot rev-parse --short HEAD 2>$null
    if ($LASTEXITCODE -ne 0) { return 'unknown' }
    return $hash.Trim()
}

function New-OutputDir {
    if (-not (Test-Path $AbsOutput)) {
        New-Item -ItemType Directory -Path $AbsOutput -Force | Out-Null
        Write-Verbose "Created output directory: $AbsOutput"
    }
}

function Build-CargoArgs {
    param(
        [string[]]$BaseArgs,
        [string]$CrateName,
        [bool]$UseRelease,
        [string[]]$ExtraFeatures
    )

    $args = [System.Collections.Generic.List[string]]::new()
    $args.AddRange($BaseArgs)

    if ($CrateName) {
        $args.Add('-p')
        $args.Add($CrateName)
    }

    if ($UseRelease) {
        $args.Add('--release')
    }

    if ($ExtraFeatures -and $ExtraFeatures.Count -gt 0) {
        $args.Add('--features')
        $args.Add($ExtraFeatures -join ',')
    }

    return $args.ToArray()
}

function Invoke-Flamegraph {
    Write-Host "Profile: Flamegraph" -ForegroundColor Cyan

    if (-not (Get-Command cargo-flamegraph -ErrorAction SilentlyContinue)) {
        # cargo flamegraph is a subcommand, check via cargo
        $check = & cargo flamegraph --help 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "cargo-flamegraph not found."
            Write-Host "Install with: cargo install flamegraph" -ForegroundColor Yellow
            Write-Host "Note: requires administrator on Windows for hardware perf counters."
            return
        }
    }

    if (-not $Crate) {
        Write-Warning "Flamegraph requires a specific -Crate target (not workspace-wide)."
        Write-Host "Available crates: $($WorkspaceCrates -join ', ')" -ForegroundColor Yellow
        return
    }

    New-OutputDir

    $useRelease = $Release.IsPresent -or $true  # always release for flamegraph
    $ts         = Get-Timestamp
    $svgPath    = Join-Path $AbsOutput "flamegraph-$Crate-$ts.svg"
    $cargoArgs  = Build-CargoArgs -BaseArgs @('flamegraph', '--root') `
                                  -CrateName $Crate `
                                  -UseRelease $useRelease `
                                  -ExtraFeatures $Features

    $cargoArgs += '-o'
    $cargoArgs += $svgPath

    Write-Host "Running: cargo $($cargoArgs -join ' ')" -ForegroundColor DarkGray
    Write-Host "Output:  $svgPath"
    Write-Host "Note: run from an elevated terminal for full perf-counter access." -ForegroundColor Yellow

    # Disable sccache to prevent port conflicts during perf instrumentation
    $env:RUSTC_WRAPPER = ''

    Push-Location $Workspace
    try {
        & cargo @cargoArgs
        if ($LASTEXITCODE -ne 0) {
            Write-Error "cargo flamegraph exited with code $LASTEXITCODE"
            return
        }
    } finally {
        Pop-Location
        Remove-Item env:RUSTC_WRAPPER -ErrorAction SilentlyContinue
    }

    Write-Host "Flamegraph written: $svgPath" -ForegroundColor Green

    if ($Open -and (Test-Path $svgPath)) {
        Start-Process $svgPath
    }
}

function Invoke-Bench {
    Write-Host "Profile: Bench" -ForegroundColor Cyan

    $useRelease = $Release.IsPresent -or $true  # criterion always needs release
    $ts         = Get-Timestamp
    $gitHash    = Get-GitHash
    $crateName  = if ($Crate) { $Crate } else { '(workspace)' }

    # Detect whether criterion benches exist
    $benchPattern = if ($Crate) {
        $memberDir = $Crate -replace '-', '_'
        Join-Path $Workspace "$memberDir\benches\*.rs"
    } else {
        Join-Path $Workspace '*\benches\*.rs'
    }

    $benchFiles = Get-ChildItem $benchPattern -ErrorAction SilentlyContinue
    $hasCriterion = $benchFiles.Count -gt 0

    New-OutputDir
    $jsonPath = Join-Path $AbsOutput "bench-$($Crate ?? 'workspace')-$ts.json"

    $env:RUSTC_WRAPPER = ''

    if ($hasCriterion) {
        Write-Host "Found $($benchFiles.Count) benchmark file(s). Running cargo bench..." -ForegroundColor DarkGray

        $cargoArgs = Build-CargoArgs -BaseArgs @('bench') `
                                     -CrateName $Crate `
                                     -UseRelease $useRelease `
                                     -ExtraFeatures $Features

        Push-Location $Workspace
        try {
            $rawOutput = & cargo @cargoArgs 2>&1
            $exitCode  = $LASTEXITCODE
        } finally {
            Pop-Location
            Remove-Item env:RUSTC_WRAPPER -ErrorAction SilentlyContinue
        }

        if ($exitCode -ne 0) {
            Write-Error "cargo bench exited with code $exitCode"
            return
        }

        # Parse criterion output: "test name ... bench: N ns/iter (+/- M)"
        $benchmarks = [System.Collections.Generic.List[hashtable]]::new()
        foreach ($line in $rawOutput) {
            if ($line -match '^\s*(.+?)\s+\.\.\.\s+bench:\s+([\d,]+)\s+ns/iter\s+\(\+/-\s+([\d,]+)\)') {
                $benchmarks.Add(@{
                    name        = $Matches[1].Trim()
                    mean_ns     = [long]($Matches[2] -replace ',', '')
                    std_dev_ns  = [long]($Matches[3] -replace ',', '')
                })
            }
        }

        Write-Host "Parsed $($benchmarks.Count) benchmark result(s)." -ForegroundColor Green
    } else {
        Write-Host "No criterion bench files found — falling back to cargo test --release timing." -ForegroundColor Yellow

        $cargoArgs = Build-CargoArgs -BaseArgs @('test', '--lib') `
                                     -CrateName $Crate `
                                     -UseRelease $useRelease `
                                     -ExtraFeatures $Features

        Push-Location $Workspace
        try {
            $sw = [System.Diagnostics.Stopwatch]::StartNew()
            & cargo @cargoArgs 2>&1 | Out-Null
            $sw.Stop()
            $exitCode = $LASTEXITCODE
        } finally {
            Pop-Location
            Remove-Item env:RUSTC_WRAPPER -ErrorAction SilentlyContinue
        }

        if ($exitCode -ne 0) {
            Write-Warning "cargo test returned exit code $exitCode — results may be incomplete."
        }

        $benchmarks = @(@{
            name       = 'cargo-test-suite (wall time)'
            mean_ns    = $sw.Elapsed.TotalNanoseconds -as [long]
            std_dev_ns = 0
        })
    }

    $result = [ordered]@{
        crate      = $crateName
        timestamp  = [DateTime]::UtcNow.ToString('o')
        git_hash   = $gitHash
        benchmarks = $benchmarks
    }

    $result | ConvertTo-Json -Depth 4 | Set-Content -Path $jsonPath -Encoding utf8
    Write-Host "Bench results written: $jsonPath" -ForegroundColor Green

    # Print summary table
    $benchmarks | ForEach-Object {
        $meanMs = [Math]::Round($_['mean_ns'] / 1e6, 3)
        Write-Host ("  {0,-50} {1,10} ms" -f $_['name'], $meanMs)
    }

    if ($Open -and (Test-Path $jsonPath)) {
        Start-Process $jsonPath
    }
}

function Invoke-Memory {
    Write-Host "Profile: Memory (working-set measurement)" -ForegroundColor Cyan

    if (-not $Crate) {
        Write-Warning "-Crate is required for Memory profiling to identify the target process."
        return
    }

    New-OutputDir

    $ts       = Get-Timestamp
    $jsonPath = Join-Path $AbsOutput "memory-$Crate-$ts.json"
    $gitHash  = Get-GitHash

    $useRelease = $Release.IsPresent
    $cargoArgs  = Build-CargoArgs -BaseArgs @('test', '--lib', '--no-run') `
                                  -CrateName $Crate `
                                  -UseRelease $useRelease `
                                  -ExtraFeatures $Features

    $env:RUSTC_WRAPPER = ''

    Write-Host "Compiling test binary..." -ForegroundColor DarkGray
    Push-Location $Workspace
    try {
        # Capture the test binary path from cargo output
        $buildOutput = & cargo @cargoArgs 2>&1
        $exitCode    = $LASTEXITCODE
    } finally {
        Pop-Location
        Remove-Item env:RUSTC_WRAPPER -ErrorAction SilentlyContinue
    }

    if ($exitCode -ne 0) {
        Write-Error "Build failed (exit $exitCode). Cannot measure memory."
        return
    }

    # cargo outputs: "Executable unittests src/lib.rs (target/.../deps/crate_name-hash)"
    $binaryPath = $buildOutput |
        Where-Object { $_ -match 'Executable.*\((.*)\)' } |
        ForEach-Object { $Matches[1].Trim() } |
        Select-Object -Last 1

    if (-not $binaryPath -or -not (Test-Path $binaryPath)) {
        Write-Warning "Could not locate compiled test binary from build output. Falling back to process name heuristic."
        $binaryPath = $null
    }

    # Baseline: current process memory before launch
    $baselineWs = (Get-Process -Id $PID).WorkingSet64

    $peakWs     = 0L
    $exitedCode = -1

    if ($binaryPath) {
        Write-Host "Running: $binaryPath" -ForegroundColor DarkGray
        $proc = Start-Process -FilePath $binaryPath -PassThru -NoNewWindow
        try {
            while (-not $proc.HasExited) {
                $proc.Refresh()
                if ($proc.WorkingSet64 -gt $peakWs) { $peakWs = $proc.WorkingSet64 }
                Start-Sleep -Milliseconds 100
            }
            $exitedCode = $proc.ExitCode
        } finally {
            if (-not $proc.HasExited) { $proc.Kill() }
        }
    } else {
        # Fallback: time the cargo test run and record this process's working set delta
        $env:RUSTC_WRAPPER = ''
        $runArgs = Build-CargoArgs -BaseArgs @('test', '--lib') `
                                   -CrateName $Crate `
                                   -UseRelease $useRelease `
                                   -ExtraFeatures $Features

        Push-Location $Workspace
        try {
            $proc        = Start-Process cargo -ArgumentList $runArgs -PassThru -NoNewWindow
            while (-not $proc.HasExited) {
                $proc.Refresh()
                if ($proc.WorkingSet64 -gt $peakWs) { $peakWs = $proc.WorkingSet64 }
                Start-Sleep -Milliseconds 100
            }
            $exitedCode = $proc.ExitCode
        } finally {
            Pop-Location
            Remove-Item env:RUSTC_WRAPPER -ErrorAction SilentlyContinue
            if (-not $proc.HasExited) { $proc.Kill() }
        }
    }

    $deltaBytes = [Math]::Max(0L, $peakWs - $baselineWs)

    $result = [ordered]@{
        crate            = $Crate
        timestamp        = [DateTime]::UtcNow.ToString('o')
        git_hash         = $gitHash
        baseline_bytes   = $baselineWs
        peak_ws_bytes    = $peakWs
        delta_bytes      = $deltaBytes
        peak_ws_mb       = [Math]::Round($peakWs   / 1MB, 2)
        delta_mb         = [Math]::Round($deltaBytes / 1MB, 2)
        binary_exit_code = $exitedCode
        note             = 'Windows working-set measurement. Full heap profiling requires DHAT/heaptrack on Linux.'
    }

    $result | ConvertTo-Json -Depth 3 | Set-Content -Path $jsonPath -Encoding utf8
    Write-Host "Memory report written: $jsonPath" -ForegroundColor Green
    Write-Host ("  Peak working set : {0,8} MB" -f $result.peak_ws_mb)
    Write-Host ("  Delta vs baseline: {0,8} MB" -f $result.delta_mb)
}

function Invoke-Summary {
    Write-Host "Profile: Summary" -ForegroundColor Cyan
    Write-Host ""

    # Tool availability
    & cargo flamegraph --help 2>$null | Out-Null
    $flamegraphInstalled = ($LASTEXITCODE -eq 0)

    $dhatPath = if ($env:CARGO_HOME) { Join-Path $env:CARGO_HOME 'bin\cargo-dhat.exe' } else { '' }

    $tools = [ordered]@{
        'cargo-flamegraph' = $flamegraphInstalled
        'cargo-criterion'  = $null -ne (Get-Command cargo-criterion -ErrorAction SilentlyContinue)
        'cargo-dhat'       = ($dhatPath -ne '') -and (Test-Path $dhatPath -ErrorAction SilentlyContinue)
        'perf (WSL/Linux)' = $false  # not applicable on Windows host
    }

    Write-Host "Tool Availability:" -ForegroundColor Yellow
    foreach ($tool in $tools.GetEnumerator()) {
        $status = if ($tool.Value) { 'INSTALLED' } else { 'missing' }
        $color  = if ($tool.Value) { 'Green' } else { 'DarkGray' }
        Write-Host ("  {0,-25} {1}" -f $tool.Key, $status) -ForegroundColor $color
    }

    Write-Host ""
    Write-Host "Install missing tools:" -ForegroundColor Yellow
    if (-not $tools['cargo-flamegraph']) {
        Write-Host "  cargo install flamegraph   # CPU flamegraph (requires admin on Windows)"
    }

    Write-Host ""
    Write-Host "Workspace Crates:" -ForegroundColor Yellow
    foreach ($crate in $WorkspaceCrates) {
        $memberDir   = $crate -replace '-', '_'
        $benchDir    = Join-Path $Workspace "$memberDir\benches"
        $benchCount  = if (Test-Path $benchDir) {
            (Get-ChildItem "$benchDir\*.rs" -ErrorAction SilentlyContinue).Count
        } else { 0 }

        $benchLabel = if ($benchCount -gt 0) { "$benchCount bench file(s)" } else { 'no benches' }
        Write-Host ("  {0,-30} {1}" -f $crate, $benchLabel)
    }

    Write-Host ""
    Write-Host "Output directory: $AbsOutput"
    if (Test-Path $AbsOutput) {
        $existing = Get-ChildItem $AbsOutput -File -ErrorAction SilentlyContinue
        Write-Host "  $($existing.Count) existing report file(s)"
    }
}

# --- Dispatch ---

switch ($Profile) {
    'Flamegraph' { Invoke-Flamegraph }
    'Bench'      { Invoke-Bench }
    'Memory'     { Invoke-Memory }
    'Summary'    { Invoke-Summary }
}
