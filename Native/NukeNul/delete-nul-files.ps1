<#
.SYNOPSIS
    High-performance Windows reserved filename deletion tool.

.DESCRIPTION
    Deletes Windows reserved filenames (nul, con, prn, aux, com1-9, lpt1-9) that
    cannot be removed through normal means. Uses hybrid Rust/C# NukeNul tool for
    parallel processing with automatic PowerShell fallback.

.PARAMETER SearchPath
    Root directory to scan (default: current directory or script location)

.PARAMETER UsePowerShell
    Force use of pure PowerShell implementation (slower, single-threaded)

.PARAMETER Verbose
    Show detailed output including raw JSON

.EXAMPLE
    .\delete-nul-files.ps1
    Scan current directory using NukeNul.exe

.EXAMPLE
    .\delete-nul-files.ps1 -SearchPath "C:\Projects"
    Scan specific directory

.EXAMPLE
    .\delete-nul-files.ps1 -UsePowerShell
    Force PowerShell fallback mode
#>

[CmdletBinding()]
param(
    [string]$SearchPath = $PSScriptRoot,
    [switch]$UsePowerShell
)

$ErrorActionPreference = 'Continue'

# ============================================================================
# CONFIGURATION
# ============================================================================

# NukeNul.exe locations (in priority order)
$NukeNulLocations = @(
    "$env:USERPROFILE\bin\NukeNul.exe",
    "$PSScriptRoot\bin\Release\net8.0\win-x64\NukeNul.exe",
    "$PSScriptRoot\NukeNul.exe"
)

# Find NukeNul.exe
$NukeNulExe = $null
foreach ($loc in $NukeNulLocations) {
    if (Test-Path $loc) {
        $NukeNulExe = $loc
        break
    }
}

Write-Host "=== NukeNul - Reserved Filename Deletion Tool ===" -ForegroundColor Cyan
Write-Host "Target: $SearchPath" -ForegroundColor White
Write-Host ""

# ============================================================================
# RUST/C# HIGH-PERFORMANCE ENGINE
# ============================================================================

if (-not $UsePowerShell -and $null -ne $NukeNulExe) {
    Write-Host "[Mode] Rust/C# High-Performance Engine" -ForegroundColor Green
    Write-Host "[Path] $NukeNulExe" -ForegroundColor DarkGray
    Write-Host ""

    try {
        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        $Output = & $NukeNulExe $SearchPath 2>&1 | Out-String
        $ExitCode = $LASTEXITCODE
        $sw.Stop()

        # Parse JSON output
        try {
            $JsonOutput = $Output | ConvertFrom-Json

            Write-Host "Status: $($JsonOutput.Status)" -ForegroundColor $(if ($JsonOutput.Status -eq 'Success') { 'Green' } else { 'Red' })
            Write-Host ""
            Write-Host "Results:" -ForegroundColor Cyan
            Write-Host "  Files Scanned:  $($JsonOutput.Results.Scanned)" -ForegroundColor White
            Write-Host "  Files Deleted:  $($JsonOutput.Results.Deleted)" -ForegroundColor Yellow
            $errColor = if ($JsonOutput.Results.Errors -gt 0) { 'Red' } else { 'Green' }
            Write-Host "  Errors:         $($JsonOutput.Results.Errors)" -ForegroundColor $errColor
            Write-Host ""
            Write-Host "Performance:" -ForegroundColor Cyan
            Write-Host "  Mode:           $($JsonOutput.Performance.Mode)"
            Write-Host "  Threads:        $($JsonOutput.Performance.Threads)"
            Write-Host "  Elapsed:        $($JsonOutput.Performance.ElapsedMs) ms" -ForegroundColor Green
            Write-Host ""

            if ($JsonOutput.Results.Deleted -gt 0) {
                Write-Host "[SUCCESS] $($JsonOutput.Results.Deleted) reserved files deleted" -ForegroundColor Green
            }
            elseif ($JsonOutput.Results.Errors -gt 0) {
                Write-Host "[WARNING] $($JsonOutput.Results.Errors) errors occurred" -ForegroundColor Yellow
            }
            else {
                Write-Host "[INFO] No reserved files found" -ForegroundColor Cyan
            }

            exit $ExitCode
        }
        catch {
            Write-Host "[Warning] Failed to parse JSON output" -ForegroundColor Yellow
            Write-Host "Raw output:" -ForegroundColor DarkGray
            Write-Host $Output

            if ($ExitCode -eq 0) {
                exit 0
            }
            Write-Host ""
            Write-Host "Falling back to PowerShell implementation..." -ForegroundColor Yellow
        }
    }
    catch {
        Write-Host "[ERROR] Failed to execute NukeNul.exe: $_" -ForegroundColor Red
        Write-Host "Falling back to PowerShell implementation..." -ForegroundColor Yellow
        Write-Host ""
    }
}
elseif ($UsePowerShell) {
    Write-Host "[Mode] PowerShell Implementation (forced)" -ForegroundColor Yellow
    Write-Host ""
}
else {
    Write-Host "[Warning] NukeNul.exe not found in:" -ForegroundColor Yellow
    foreach ($loc in $NukeNulLocations) {
        Write-Host "  - $loc" -ForegroundColor DarkGray
    }
    Write-Host ""
    Write-Host "Build NukeNul or install to ~/bin/:" -ForegroundColor Cyan
    Write-Host "  cd $PSScriptRoot && .\build.ps1" -ForegroundColor White
    Write-Host ""
    Write-Host "Falling back to PowerShell implementation..." -ForegroundColor Yellow
    Write-Host ""
}

# ============================================================================
# POWERSHELL FALLBACK IMPLEMENTATION
# ============================================================================

Write-Host "[Mode] PowerShell Fallback" -ForegroundColor Yellow
Write-Host ""

$ExcludeDirs = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
$null = $ExcludeDirs.Add(".git")

$sw = [System.Diagnostics.Stopwatch]::StartNew()
$DeletedCount = 0
$ErrorCount = 0
$ScannedCount = 0

# Reserved filenames to search for
$ReservedNames = @("nul", "con", "prn", "aux") + (1..9 | ForEach-Object { "com$_", "lpt$_" })

function Get-ReservedFiles {
    param (
        [string]$RootPath,
        [System.Collections.Generic.HashSet[string]]$Exclusions,
        [string[]]$FileNames
    )

    $stack = [System.Collections.Generic.Stack[string]]::new()
    $stack.Push($RootPath)

    while ($stack.Count -gt 0) {
        $currentDir = $stack.Pop()

        try {
            # Find reserved files in the current directory
            foreach ($name in $FileNames) {
                [System.IO.Directory]::EnumerateFiles($currentDir, $name, [System.IO.SearchOption]::TopDirectoryOnly) | ForEach-Object {
                    $_
                }
            }

            # Find subdirectories to traverse
            [System.IO.Directory]::EnumerateDirectories($currentDir) | ForEach-Object {
                $dirName = [System.IO.Path]::GetFileName($_)
                if (-not $Exclusions.Contains($dirName)) {
                    $stack.Push($_)
                }
            }
        }
        catch {
            Write-Verbose "Skipping $currentDir : $_"
        }
    }
}

try {
    if ($PSVersionTable.PSVersion.Major -ge 7) {
        # PowerShell 7+ parallel mode
        Get-ReservedFiles -RootPath $SearchPath -Exclusions $ExcludeDirs -FileNames $ReservedNames | ForEach-Object -Parallel {
            $path = $_
            $extended = "\\?\$path"

            Write-Host "Found: $path" -ForegroundColor Red

            try {
                Remove-Item -LiteralPath $extended -Force -ErrorAction Stop
                Write-Host "  [DELETED] $path" -ForegroundColor Yellow
            }
            catch {
                Write-Host "  [ERROR] $path : $_" -ForegroundColor DarkRed
            }
        } -ThrottleLimit 32
    }
    else {
        # PowerShell 5.1 single-threaded mode
        Write-Warning "PowerShell 7+ not detected. Using single-threaded mode."
        Get-ReservedFiles -RootPath $SearchPath -Exclusions $ExcludeDirs -FileNames $ReservedNames | ForEach-Object {
            $path = $_
            $extended = "\\?\$path"
            $ScannedCount++

            Write-Host "Found: $path" -ForegroundColor Red

            try {
                Remove-Item -LiteralPath $extended -Force -ErrorAction Stop
                Write-Host "  [DELETED] $path" -ForegroundColor Yellow
                $DeletedCount++
            }
            catch {
                Write-Host "  [ERROR] $path : $_" -ForegroundColor DarkRed
                $ErrorCount++
            }
        }
    }
}
catch {
    Write-Host "Fatal error during execution: $_" -ForegroundColor Red
}

$sw.Stop()
Write-Host ""
Write-Host "Scan complete in $($sw.Elapsed.TotalSeconds.ToString('F2')) seconds." -ForegroundColor Cyan

