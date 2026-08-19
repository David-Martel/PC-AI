<#
.SYNOPSIS
    Integration test script for NukeNul hybrid Rust/C# project.

.DESCRIPTION
    This script performs comprehensive integration testing:
    1. Creates a safe test directory structure
    2. Creates "nul" files using \\?\ prefix (reserved names)
    3. Runs NukeNul.exe against the test directory
    4. Verifies files were deleted correctly
    5. Cleans up test artifacts
    6. Performs performance benchmarking

.PARAMETER Configuration
    Build configuration to test (Debug or Release)

.PARAMETER TestCount
    Number of "nul" files to create (default: 10)

.PARAMETER DeepNesting
    Create nested directory structure for stress testing

.PARAMETER SkipBenchmark
    Skip performance comparison with PowerShell script

.PARAMETER KeepTestDir
    Don't clean up test directory after tests

.PARAMETER DollarNullOnly
    Exercise the narrow literal-$null cleanup mode instead of reserved device names.

.EXAMPLE
    .\test.ps1
    Standard integration test

.EXAMPLE
    .\test.ps1 -TestCount 100 -DeepNesting
    Stress test with nested directories

.EXAMPLE
    .\test.ps1 -KeepTestDir
    Run tests but keep test directory for inspection
#>

[CmdletBinding()]
param(
    [ValidateSet('Debug', 'Release')]
    [string]$Configuration = 'Release',

    [ValidateRange(1, 10000)]
    [int]$TestCount = 10,

    [switch]$DeepNesting,
    [switch]$SkipBenchmark,
    [switch]$KeepTestDir,
    [switch]$DollarNullOnly
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

# Colors for output
$Colors = @{
    Success = 'Green'
    Error   = 'Red'
    Warning = 'Yellow'
    Info    = 'Cyan'
    Step    = 'Magenta'
}

function Write-TestStep {
    param([string]$Message)
    Write-Host "`n==> $Message" -ForegroundColor $Colors.Step
}

function Write-TestSuccess {
    param([string]$Message)
    Write-Host "[OK] $Message" -ForegroundColor $Colors.Success
}

function Write-TestError {
    param([string]$Message)
    Write-Host "[FAIL] $Message" -ForegroundColor $Colors.Error
}

function Write-TestInfo {
    param([string]$Message)
    Write-Host "[i] $Message" -ForegroundColor $Colors.Info
}

# ============================================================================
# PHASE 1: PRE-TEST VALIDATION
# ============================================================================

Write-TestStep "Phase 1: Pre-Test Validation"

$RootDir = $PSScriptRoot
$CSharpDir = $RootDir  # C# files are in root directory
$ExeDir = Join-Path $CSharpDir "bin\$Configuration\net8.0\win-x64"
$ExePath = Join-Path $ExeDir "NukeNul.exe"
$DllPath = Join-Path $ExeDir "nuker_core.dll"

# Check if executable exists
if (-not (Test-Path $ExePath)) {
    Write-TestError "NukeNul.exe not found: $ExePath"
    Write-Host "`nRun build first: .\build.ps1" -ForegroundColor Yellow
    exit 1
}

Write-TestSuccess "Executable found: $ExePath"

# Check if DLL exists
if (-not (Test-Path $DllPath)) {
    Write-TestError "nuker_core.dll not found: $DllPath"
    exit 1
}

Write-TestSuccess "Rust DLL found: $DllPath"

# Verify executability
try {
    $TestRun = & $ExePath "--help" 2>&1
    Write-TestSuccess "Executable is valid and runnable"
}
catch {
    Write-TestError "Failed to run executable: $_"
    exit 1
}

# ============================================================================
# PHASE 2: CREATE TEST ENVIRONMENT
# ============================================================================

Write-TestStep "Phase 2: Create Test Environment"

# Create test directory in TEMP
$TestDir = Join-Path $env:TEMP "NukeNul_Test_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
New-Item -ItemType Directory -Path $TestDir -Force | Out-Null
Write-TestInfo "Test directory: $TestDir"

# Create directory structure
$Directories = @($TestDir)

if ($DeepNesting) {
    Write-TestInfo "Creating nested directory structure..."
    $Depths = @(
        "Level1",
        "Level1\Level2",
        "Level1\Level2\Level3",
        "Level1\Level2\Level3\Level4",
        "AnotherBranch",
        "AnotherBranch\SubDir1",
        "AnotherBranch\SubDir2"
    )

    foreach ($Depth in $Depths) {
        $Dir = Join-Path $TestDir $Depth
        New-Item -ItemType Directory -Path $Dir -Force | Out-Null
        $Directories += $Dir
    }

    Write-TestSuccess "Created $($Directories.Count) directories"
}
else {
    # Simple flat structure
    $FlatDirs = @("Dir1", "Dir2", "Dir3")
    foreach ($Dir in $FlatDirs) {
        $DirPath = Join-Path $TestDir $Dir
        New-Item -ItemType Directory -Path $DirPath -Force | Out-Null
        $Directories += $DirPath
    }

    Write-TestSuccess "Created $($Directories.Count) directories (flat structure)"
}

# ============================================================================
# PHASE 3: CREATE "NUL" FILES
# ============================================================================

Write-TestStep "Phase 3: Create Test Files"

$TargetLeaf = if ($DollarNullOnly) { [char]36 + 'null' } else { 'nul' }
Write-TestInfo "Creating $TestCount '$TargetLeaf' files using extended path prefix..."

$CreatedFiles = @()
$NormalFiles = @()
$sw = [System.Diagnostics.Stopwatch]::StartNew()

# Distribute files across directories
$FilesPerDir = [math]::Ceiling($TestCount / $Directories.Count)

for ($i = 0; $i -lt $TestCount; $i++) {
    $DirIndex = [math]::Floor($i / $FilesPerDir)
    if ($DirIndex -ge $Directories.Count) {
        $DirIndex = $Directories.Count - 1
    }

    $TargetDir = $Directories[$DirIndex]

    # Create "nul" file using \\?\ prefix
    $NulPath = Join-Path $TargetDir $TargetLeaf
    $ExtendedPath = "\\?\$NulPath"

    try {
        # Use .NET to create the file (PowerShell New-Item doesn't work with \\?\ prefix)
        $FileStream = [System.IO.File]::Create($ExtendedPath)
        $FileStream.Close()
        $CreatedFiles += $NulPath
    }
    catch {
        Write-TestError "Failed to create $NulPath : $_"
    }

    # Also create some normal files for context
    if ($i % 3 -eq 0) {
        $NormalPath = Join-Path $TargetDir "file$i.txt"
        "Test content $i" | Out-File $NormalPath -Encoding UTF8
        $NormalFiles += $NormalPath
    }
}

$sw.Stop()

Write-TestSuccess "Created $($CreatedFiles.Count) nul files in $($sw.Elapsed.TotalSeconds)s"
Write-TestInfo "Created $($NormalFiles.Count) normal files for context"

# Verify files exist
Write-TestInfo "Verifying test files..."
$VerifiedCount = 0
foreach ($File in $CreatedFiles) {
    $ExtendedPath = "\\?\$File"
    if ([System.IO.File]::Exists($ExtendedPath)) {
        $VerifiedCount++
    }
}

Write-TestSuccess "Verified $VerifiedCount/$($CreatedFiles.Count) files exist"

# ============================================================================
# PHASE 4: RUN NUKENUL.EXE
# ============================================================================

Write-TestStep "Phase 4: Run NukeNul.exe"

$ExeArgs = @()
if ($DollarNullOnly) {
    $ExeArgs += '--dollar-null-only'
}
$ExeArgs += $TestDir
Write-TestInfo "Executing: $ExePath $($ExeArgs -join ' ')"
Write-Host ""

$sw = [System.Diagnostics.Stopwatch]::StartNew()

try {
    $Output = & $ExePath @ExeArgs 2>&1 | Out-String
    $ExitCode = $LASTEXITCODE
    $sw.Stop()

    Write-Host $Output

    if ($ExitCode -ne 0) {
        Write-TestError "NukeNul.exe failed with exit code: $ExitCode"
        exit 1
    }

    Write-TestSuccess "NukeNul.exe completed in $($sw.Elapsed.TotalSeconds)s"
}
catch {
    Write-TestError "Failed to execute NukeNul.exe: $_"
    exit 1
}

# Parse JSON output
$JsonOutput = $null
try {
    $JsonOutput = $Output | ConvertFrom-Json
    Write-Host ""
    Write-Host "Results:" -ForegroundColor Cyan
    Write-Host "  Files Scanned:  $($JsonOutput.Results.Scanned)" -ForegroundColor White
    Write-Host "  Files Deleted:  $($JsonOutput.Results.Deleted)" -ForegroundColor Yellow
    $errColor = if ($JsonOutput.Results.Errors -gt 0) { 'Red' } else { 'Green' }
    Write-Host "  Errors:         $($JsonOutput.Results.Errors)" -ForegroundColor $errColor
    Write-Host "  Elapsed:        $($JsonOutput.Performance.ElapsedMs) ms" -ForegroundColor White
}
catch {
    Write-TestError "Failed to parse JSON output: $_"
    Write-Host "Raw output:" -ForegroundColor Yellow
    Write-Host $Output
}

# ============================================================================
# PHASE 5: VERIFY DELETION
# ============================================================================

Write-TestStep "Phase 5: Verify Deletion"

Write-TestInfo "Checking if nul files were deleted..."

$RemainingFiles = 0
foreach ($File in $CreatedFiles) {
    $ExtendedPath = "\\?\$File"
    if ([System.IO.File]::Exists($ExtendedPath)) {
        $RemainingFiles++
        Write-Host "  [!] Still exists: $File" -ForegroundColor Red
    }
}

if ($RemainingFiles -eq 0) {
    Write-TestSuccess "All $($CreatedFiles.Count) nul files were deleted"
}
else {
    Write-TestError "$RemainingFiles/$($CreatedFiles.Count) files were NOT deleted"
}

# Verify normal files were NOT deleted
Write-TestInfo "Checking that normal files were preserved..."
$MissingNormalFiles = 0
foreach ($File in $NormalFiles) {
    if (-not (Test-Path $File)) {
        $MissingNormalFiles++
        Write-Host "  [!] Normal file deleted: $File" -ForegroundColor Red
    }
}

if ($MissingNormalFiles -eq 0) {
    Write-TestSuccess "All $($NormalFiles.Count) normal files were preserved"
}
else {
    Write-TestError "$MissingNormalFiles/$($NormalFiles.Count) normal files were incorrectly deleted"
}

# ============================================================================
# PHASE 6: PERFORMANCE BENCHMARK (Optional)
# ============================================================================

if (-not $SkipBenchmark) {
    Write-TestStep "Phase 6: Performance Benchmark"

    $OriginalScript = Join-Path $RootDir "delete-nul-files.ps1"

    if (Test-Path $OriginalScript) {
        Write-TestInfo "Comparing with original PowerShell script..."

        # Recreate test files for fair comparison
        Write-TestInfo "Recreating test files for benchmark..."
        $BenchFiles = @()
        for ($i = 0; $i -lt $TestCount; $i++) {
            $DirIndex = [math]::Floor($i / $FilesPerDir)
            if ($DirIndex -ge $Directories.Count) { $DirIndex = $Directories.Count - 1 }

            $TargetDir = $Directories[$DirIndex]
            $NulPath = Join-Path $TargetDir $TargetLeaf
            $ExtendedPath = "\\?\$NulPath"

            try {
                $FileStream = [System.IO.File]::Create($ExtendedPath)
                $FileStream.Close()
                $BenchFiles += $NulPath
            }
            catch {
                # Ignore errors for benchmark
            }
        }

        Write-TestInfo "Running PowerShell script..."
        $sw = [System.Diagnostics.Stopwatch]::StartNew()

        Push-Location $TestDir
        try {
            & $OriginalScript -SearchPath $TestDir -UsePowerShell -DollarNullOnly:$DollarNullOnly 2>&1 | Out-Null
            $sw.Stop()
            $PowerShellTime = $sw.Elapsed.TotalMilliseconds

            Write-Host ""
            Write-Host "Performance Comparison:" -ForegroundColor Cyan
            Write-Host "  PowerShell:     $([math]::Round($PowerShellTime, 2)) ms" -ForegroundColor White
            if ($null -ne $JsonOutput) {
                Write-Host "  NukeNul (Rust): $($JsonOutput.Performance.ElapsedMs) ms" -ForegroundColor Yellow

                if ($JsonOutput.Performance.ElapsedMs -lt $PowerShellTime) {
                    $Speedup = $PowerShellTime / $JsonOutput.Performance.ElapsedMs
                    Write-Host "  Speedup:        $([math]::Round($Speedup, 2))x faster" -ForegroundColor Green
                }
                else {
                    Write-Host "  Note: PowerShell was faster (likely due to small test size)" -ForegroundColor Yellow
                }
            }
        }
        finally {
            Pop-Location
        }
    }
    else {
        Write-TestInfo "Original PowerShell script not found, skipping benchmark"
    }
}

# ============================================================================
# PHASE 7: CLEANUP
# ============================================================================

Write-TestStep "Phase 7: Cleanup"

if ($KeepTestDir) {
    Write-TestInfo "Keeping test directory: $TestDir"
}
else {
    Write-TestInfo "Removing test directory..."
    try {
        Remove-Item $TestDir -Recurse -Force -ErrorAction Stop
        Write-TestSuccess "Test directory cleaned up"
    }
    catch {
        Write-Host "  [!] Warning: Failed to clean up test directory: $_" -ForegroundColor Yellow
        Write-Host "  Manual cleanup required: $TestDir" -ForegroundColor Yellow
    }
}

# ============================================================================
# PHASE 8: TEST SUMMARY
# ============================================================================

Write-TestStep "Phase 8: Test Summary"

Write-Host ""
$AllTestsPassed = ($RemainingFiles -eq 0) -and ($MissingNormalFiles -eq 0)

if ($AllTestsPassed) {
    Write-Host "ALL TESTS PASSED" -ForegroundColor Green
}
else {
    Write-Host "TESTS FAILED" -ForegroundColor Red
}

Write-Host ""
Write-Host "Test Statistics:" -ForegroundColor Cyan
Write-Host "  Test Files Created:       $($CreatedFiles.Count)"
Write-Host "  Normal Files Created:     $($NormalFiles.Count)"
Write-Host "  Files Successfully Deleted: $($CreatedFiles.Count - $RemainingFiles)"
$successRate = [math]::Round((($CreatedFiles.Count - $RemainingFiles) / $CreatedFiles.Count) * 100, 2)
Write-Host "  Deletion Success Rate:    $successRate%"
if ($null -ne $JsonOutput) {
    Write-Host "  Execution Time:           $($JsonOutput.Performance.ElapsedMs) ms"
}
Write-Host ""

if (-not $AllTestsPassed) {
    exit 1
}
