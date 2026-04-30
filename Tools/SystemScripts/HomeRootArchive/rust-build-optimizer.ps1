# Rust Build Optimization Script for Windows Development
# Optimized for frequent recompilation of rust-fs and rust-commander projects
# Author: Claude AI Systems
# Version: 1.0.0

[CmdletBinding()]
param(
    [Parameter(Mandatory=$false)]
    [ValidateSet("dev", "quick-dev", "release-dev", "release", "test", "bench")]
    [string]$Profile = "dev",

    [Parameter(Mandatory=$false)]
    [ValidateSet("check", "build", "test", "clippy", "fmt", "clean", "bench", "timings")]
    [string]$Action = "build",

    [Parameter(Mandatory=$false)]
    [ValidateSet("rust-fs", "rust-commander", "both")]
    [string]$Project = "both",

    [Parameter(Mandatory=$false)]
    [switch]$Verbose,

    [Parameter(Mandatory=$false)]
    [switch]$Parallel = $true,

    [Parameter(Mandatory=$false)]
    [switch]$UseSccache = $true,

    [Parameter(Mandatory=$false)]
    [switch]$ShowStats = $false,

    [Parameter(Mandatory=$false)]
    [switch]$Watch = $false
)

# Configuration
$script:ProjectPaths = @{
    "rust-fs" = "T:\projects\rust-fs"
    "rust-commander" = "T:\projects\rust-commander"
}

$script:StartTime = Get-Date

function Write-Banner {
    param([string]$Text)
    Write-Host ""
    Write-Host "=" * 80 -ForegroundColor Cyan
    Write-Host $Text -ForegroundColor Yellow
    Write-Host "=" * 80 -ForegroundColor Cyan
    Write-Host ""
}

function Write-Section {
    param([string]$Text)
    Write-Host ""
    Write-Host $Text -ForegroundColor Green
    Write-Host "-" * $Text.Length -ForegroundColor Green
}

function Test-Prerequisites {
    Write-Section "Checking Prerequisites"

    # Check Rust installation
    if (-not (Get-Command "cargo" -ErrorAction SilentlyContinue)) {
        Write-Error "Cargo not found. Please install Rust."
        return $false
    }

    $rustVersion = cargo --version
    Write-Host "✓ Rust: $rustVersion" -ForegroundColor Green

    # Check sccache if enabled
    if ($UseSccache) {
        if (-not (Get-Command "sccache" -ErrorAction SilentlyContinue)) {
            Write-Warning "sccache not found. Install with: cargo install sccache"
            $script:UseSccache = $false
        } else {
            $sccacheVersion = sccache --version
            Write-Host "✓ sccache: $sccacheVersion" -ForegroundColor Green

            # Start sccache server if not running
            $stats = sccache --show-stats 2>$null
            if ($LASTEXITCODE -ne 0) {
                Write-Host "Starting sccache server..." -ForegroundColor Yellow
                sccache --start-server
            }
        }
    }

    return $true
}

function Initialize-BuildEnvironment {
    Write-Section "Initializing Build Environment"

    # Set environment variables for optimal performance
    $env:RUST_BACKTRACE = "1"
    $env:CARGO_TERM_COLOR = "always"

    if ($UseSccache) {
        $env:RUSTC_WRAPPER = "sccache"
        $env:SCCACHE_CACHE_SIZE = "15G"
        $env:SCCACHE_DIR = "C:\Users\david\AppData\Local\sccache\cache"
        $env:SCCACHE_IDLE_TIMEOUT = "1800"
        $env:SCCACHE_CACHE_COMPRESSION = "zstd"
        $env:SCCACHE_DIRECT_MODE = "true"

        # For dev builds, enable incremental; sccache will disable when beneficial
        if ($Profile -eq "dev" -or $Profile -eq "quick-dev") {
            $env:CARGO_INCREMENTAL = "1"
        } else {
            $env:CARGO_INCREMENTAL = "0"
        }

        Write-Host "✓ sccache environment configured" -ForegroundColor Green
    }

    # Set parallel jobs (use all CPU cores)
    $cpuCores = (Get-CimInstance Win32_ComputerSystem).NumberOfLogicalProcessors
    if ($Parallel) {
        $env:CARGO_BUILD_JOBS = $cpuCores
        Write-Host "✓ Using $cpuCores parallel build jobs" -ForegroundColor Green
    }

    # Memory optimization for large projects
    $env:RUST_MIN_STACK = "8388608"  # 8MB stack

    Write-Host "✓ Build environment optimized" -ForegroundColor Green
}

function Get-CargoArgs {
    param([string]$Action, [string]$Profile, [bool]$Verbose)

    $args = @()

    # Base command
    switch ($Action) {
        "check" {
            $args += "check", "--workspace", "--all-features"
            if (-not $Verbose) { $args += "--message-format=short" }
        }
        "build" {
            $args += "build", "--workspace"
            if ($Profile -ne "dev") { $args += "--profile", $Profile }
        }
        "test" {
            $args += "test", "--workspace"
            if ($Profile -ne "dev") { $args += "--profile", $Profile }
        }
        "clippy" {
            $args += "clippy", "--workspace", "--all-features", "--all-targets"
            $args += "--", "-D", "warnings"
        }
        "fmt" {
            $args += "fmt", "--all", "--", "--check"
        }
        "clean" {
            $args += "clean"
        }
        "bench" {
            $args += "bench", "--workspace"
        }
        "timings" {
            $args += "build", "--workspace", "--timings=html"
            if ($Profile -ne "dev") { $args += "--profile", $Profile }
        }
    }

    # Add verbose flag if requested
    if ($Verbose -and $Action -ne "fmt") {
        $args += "--verbose"
    }

    return $args
}

function Invoke-CargoCommand {
    param([string]$ProjectPath, [string]$ProjectName, [array]$Args)

    Write-Section "Building $ProjectName"
    Write-Host "Path: $ProjectPath" -ForegroundColor Cyan
    Write-Host "Command: cargo $($Args -join ' ')" -ForegroundColor Cyan

    $projectStartTime = Get-Date

    try {
        Push-Location $ProjectPath

        # Reset sccache stats for this build if requested
        if ($ShowStats -and $UseSccache) {
            sccache --zero-stats >$null 2>&1
        }

        # Execute cargo command
        $process = Start-Process -FilePath "cargo" -ArgumentList $Args -Wait -PassThru -NoNewWindow

        $projectEndTime = Get-Date
        $projectDuration = $projectEndTime - $projectStartTime

        if ($process.ExitCode -eq 0) {
            Write-Host "✓ $ProjectName completed in $($projectDuration.TotalSeconds.ToString('F2'))s" -ForegroundColor Green

            # Show sccache stats if requested
            if ($ShowStats -and $UseSccache) {
                Write-Host ""
                Write-Host "sccache Statistics:" -ForegroundColor Yellow
                sccache --show-stats
            }

            return $true
        } else {
            Write-Host "✗ $ProjectName failed with exit code $($process.ExitCode)" -ForegroundColor Red
            return $false
        }
    }
    catch {
        Write-Host "✗ $ProjectName failed with error: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
    finally {
        Pop-Location
    }
}

function Start-BuildProcess {
    $args = Get-CargoArgs -Action $Action -Profile $Profile -Verbose $Verbose
    $projects = @()

    switch ($Project) {
        "both" { $projects = @("rust-fs", "rust-commander") }
        default { $projects = @($Project) }
    }

    $results = @()

    foreach ($proj in $projects) {
        if (-not $ProjectPaths.ContainsKey($proj)) {
            Write-Error "Unknown project: $proj"
            continue
        }

        $projectPath = $ProjectPaths[$proj]
        if (-not (Test-Path $projectPath)) {
            Write-Error "Project path not found: $projectPath"
            continue
        }

        if ($Parallel -and $projects.Count -gt 1 -and $Action -in @("check", "build")) {
            # Run projects in parallel for check/build operations
            $job = Start-Job -ScriptBlock {
                param($ProjectPath, $ProjectName, $Args, $UseSccache, $ShowStats)
                # Set environment in job
                if ($UseSccache) {
                    $env:RUSTC_WRAPPER = "sccache"
                    $env:SCCACHE_CACHE_SIZE = "15G"
                    $env:SCCACHE_DIR = "C:\Users\david\AppData\Local\sccache\cache"
                }

                Set-Location $ProjectPath
                $process = Start-Process -FilePath "cargo" -ArgumentList $Args -Wait -PassThru -NoNewWindow
                return @{
                    ProjectName = $ProjectName
                    ExitCode = $process.ExitCode
                    Path = $ProjectPath
                }
            } -ArgumentList $projectPath, $proj, $args, $UseSccache, $ShowStats

            $results += $job
        } else {
            # Run sequentially
            $success = Invoke-CargoCommand -ProjectPath $projectPath -ProjectName $proj -Args $args
            $results += @{
                ProjectName = $proj
                Success = $success
                Path = $projectPath
            }
        }
    }

    # Handle parallel job results
    if ($results[0] -is [System.Management.Automation.Job]) {
        Write-Section "Waiting for parallel builds to complete..."
        $jobResults = $results | Receive-Job -Wait
        $results | Remove-Job

        foreach ($result in $jobResults) {
            $success = $result.ExitCode -eq 0
            $status = if ($success) { "✓" } else { "✗" }
            $color = if ($success) { "Green" } else { "Red" }
            Write-Host "$status $($result.ProjectName): $($result.Path)" -ForegroundColor $color
        }
    }

    return $results
}

function Show-BuildSummary {
    param([array]$Results)

    $endTime = Get-Date
    $totalDuration = $endTime - $script:StartTime

    Write-Banner "Build Summary"
    Write-Host "Profile: $Profile" -ForegroundColor Cyan
    Write-Host "Action: $Action" -ForegroundColor Cyan
    Write-Host "Total Time: $($totalDuration.TotalSeconds.ToString('F2'))s" -ForegroundColor Cyan

    if ($UseSccache) {
        Write-Host ""
        Write-Host "Final sccache Statistics:" -ForegroundColor Yellow
        sccache --show-stats
    }

    Write-Host ""
    Write-Host "Build complete!" -ForegroundColor Green
}

function Start-WatchMode {
    Write-Banner "Entering Watch Mode"
    Write-Host "Watching for file changes... Press Ctrl+C to exit" -ForegroundColor Yellow

    # This is a simplified watch mode - in practice, you'd use a proper file watcher
    while ($true) {
        Start-Sleep -Seconds 2

        # Check for .rs file modifications in the last 2 seconds
        $projects = if ($Project -eq "both") { @("rust-fs", "rust-commander") } else { @($Project) }

        foreach ($proj in $projects) {
            $path = $ProjectPaths[$proj]
            $recentFiles = Get-ChildItem -Path $path -Include "*.rs" -Recurse |
                Where-Object { $_.LastWriteTime -gt (Get-Date).AddSeconds(-2) }

            if ($recentFiles.Count -gt 0) {
                Write-Host "File changes detected in $proj, rebuilding..." -ForegroundColor Yellow
                $args = Get-CargoArgs -Action $Action -Profile $Profile -Verbose $false
                Invoke-CargoCommand -ProjectPath $path -ProjectName $proj -Args $args
                break
            }
        }
    }
}

# Main execution
try {
    Write-Banner "Rust Build Optimization System"
    Write-Host "Profile: $Profile | Action: $Action | Project: $Project" -ForegroundColor Cyan

    if (-not (Test-Prerequisites)) {
        exit 1
    }

    Initialize-BuildEnvironment

    if ($Watch) {
        Start-WatchMode
    } else {
        $results = Start-BuildProcess
        Show-BuildSummary -Results $results
    }
}
catch {
    Write-Error "Build failed: $($_.Exception.Message)"
    exit 1
}
finally {
    # Cleanup
    if ($UseSccache -and -not $Watch) {
        # Keep sccache server running for next build
    }
}