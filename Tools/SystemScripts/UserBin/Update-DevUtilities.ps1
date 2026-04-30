#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Manage and update utility symlinks for David's development environment.

.DESCRIPTION
    This script manages symlinks for various utilities including:
    - Core utilities (uu- prefixed) to their canonical forms
    - Wu- prefixed utilities to their canonical forms
    - Node.js/npm wrapper scripts
    - Developer tools from various locations

.PARAMETER Action
    The action to perform: Update, List, Verify, or Clean

.EXAMPLE
    .\Update-DevUtilities.ps1 -Action Update
    .\Update-DevUtilities.ps1 -Action List
    .\Update-DevUtilities.ps1 -Action Verify
#>

param(
    [Parameter(Mandatory=$false)]
    [ValidateSet('Update', 'List', 'Verify', 'Clean', 'FixNodejs')]
    [string]$Action = 'Update'
)

# Configuration
$BinDirectories = @(
    "C:\users\david\bin",
    "C:\users\david\.local\bin"
)

$NodejsDirectories = @(
    "C:\nvm4w\nodejs",
    "C:\Users\david\AppData\Local\nvm\v24.7.0"
)

$PythonExecutable = "C:\Users\david\AppData\Local\Programs\Python\Python312\python.exe"

function Write-ColoredOutput {
    param(
        [string]$Message,
        [string]$Color = 'White'
    )
    Write-Host $Message -ForegroundColor $Color
}

function Test-AdminRights {
    $currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($currentUser)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Get-CoreUtilities {
    <#
    .SYNOPSIS
    Get list of uu- prefixed utilities and their canonical names
    #>
    $utilities = @()

    foreach ($binDir in $BinDirectories) {
        if (Test-Path $binDir) {
            $uuUtilities = Get-ChildItem -Path $binDir -Filter "uu-*.exe" -ErrorAction SilentlyContinue
            foreach ($util in $uuUtilities) {
                $canonicalName = $util.Name -replace "^uu-", ""
                $utilities += [PSCustomObject]@{
                    SourcePath = $util.FullName
                    CanonicalName = $canonicalName
                    TargetPath = Join-Path $binDir $canonicalName
                    Type = "CoreUtil"
                }
            }
        }
    }

    return $utilities
}

function Get-WuUtilities {
    <#
    .SYNOPSIS
    Get list of wu- prefixed utilities (if any exist)
    #>
    $utilities = @()

    # Check various locations for wu- utilities
    $searchPaths = @(
        "T:\projects\coreutils\winutils\target\release",
        "T:\projects\coreutils\winutils\target\debug",
        "T:\projects\coreutils\winutils"
    )

    foreach ($searchPath in $searchPaths) {
        if (Test-Path $searchPath) {
            $wuUtilities = Get-ChildItem -Path $searchPath -Filter "wu-*.exe" -ErrorAction SilentlyContinue
            foreach ($util in $wuUtilities) {
                $canonicalName = $util.Name -replace "^wu-", ""
                $targetDir = $BinDirectories[1]  # Prefer .local\bin
                $utilities += [PSCustomObject]@{
                    SourcePath = $util.FullName
                    CanonicalName = $canonicalName
                    TargetPath = Join-Path $targetDir $canonicalName
                    Type = "WuUtil"
                }
            }
        }
    }

    return $utilities
}

function Get-WrapperUtilities {
    <#
    .SYNOPSIS
    Get list of -wrapper.exe utilities
    #>
    $utilities = @()

    $searchPaths = @(
        "T:\projects\coreutils\winutils\target\release",
        "T:\projects\coreutils\winutils\target\debug",
        "T:\projects\coreutils\winutils"
    )

    foreach ($searchPath in $searchPaths) {
        if (Test-Path $searchPath) {
            $wrapperUtilities = Get-ChildItem -Path $searchPath -Filter "*-wrapper.exe" -ErrorAction SilentlyContinue
            foreach ($util in $wrapperUtilities) {
                $canonicalName = $util.Name -replace "-wrapper", ""
                $targetDir = $BinDirectories[1]  # Prefer .local\bin
                $utilities += [PSCustomObject]@{
                    SourcePath = $util.FullName
                    CanonicalName = $canonicalName
                    TargetPath = Join-Path $targetDir $canonicalName
                    Type = "WrapperUtil"
                }
            }
        }
    }

    return $utilities
}

function Update-NodejsWrappers {
    <#
    .SYNOPSIS
    Update npm and npx wrapper scripts to use NVM's Node.js
    #>
    Write-ColoredOutput "Updating Node.js wrapper scripts..." "Yellow"

    $npmContent = @"
@echo off
setlocal
"C:\nvm4w\nodejs\node.exe" "C:\Program Files\nodejs\node_modules\npm\bin\npm-cli.js" %*
endlocal
"@

    $npxContent = @"
@echo off
setlocal
"C:\nvm4w\nodejs\node.exe" "C:\Program Files\nodejs\node_modules\npm\bin\npx-cli.js" %*
endlocal
"@

    # Update npm.cmd files
    foreach ($dir in $NodejsDirectories + $BinDirectories) {
        if (Test-Path $dir) {
            $npmPath = Join-Path $dir "npm.cmd"
            $npxPath = Join-Path $dir "npx.cmd"

            try {
                $npmContent | Out-File -FilePath $npmPath -Encoding ASCII -Force
                Write-ColoredOutput "  Updated: $npmPath" "Green"

                $npxContent | Out-File -FilePath $npxPath -Encoding ASCII -Force
                Write-ColoredOutput "  Updated: $npxPath" "Green"
            }
            catch {
                Write-ColoredOutput "  Failed to update: $dir - $($_.Exception.Message)" "Red"
            }
        }
    }

    # Set environment variable for Python
    if (Test-Path $PythonExecutable) {
        [System.Environment]::SetEnvironmentVariable("PYTHON", $PythonExecutable, "User")
        Write-ColoredOutput "  Set PYTHON environment variable to: $PythonExecutable" "Green"
    }
}

function Update-Symlinks {
    param(
        [Parameter(Mandatory=$true)]
        [array]$Utilities
    )

    if (-not (Test-AdminRights)) {
        Write-ColoredOutput "Administrative rights required for creating symlinks. Please run as Administrator." "Red"
        return
    }

    $created = 0
    $updated = 0
    $skipped = 0

    foreach ($util in $Utilities) {
        if (Test-Path $util.SourcePath) {
            # Ensure target directory exists
            $targetDir = Split-Path $util.TargetPath -Parent
            if (-not (Test-Path $targetDir)) {
                New-Item -ItemType Directory -Path $targetDir -Force | Out-Null
            }

            $targetExists = Test-Path $util.TargetPath

            if ($targetExists) {
                # Check if it's already the correct symlink
                try {
                    $linkTarget = (Get-Item $util.TargetPath).Target
                    if ($linkTarget -eq $util.SourcePath) {
                        Write-ColoredOutput "  Skipped: $($util.CanonicalName) (already correct)" "Gray"
                        $skipped++
                        continue
                    }
                    else {
                        # Remove existing file/symlink
                        Remove-Item $util.TargetPath -Force
                    }
                }
                catch {
                    # Not a symlink, remove it
                    Remove-Item $util.TargetPath -Force
                }
                $updated++
            }
            else {
                $created++
            }

            # Create symlink
            try {
                New-Item -ItemType SymbolicLink -Path $util.TargetPath -Target $util.SourcePath -Force | Out-Null
                Write-ColoredOutput "  Created: $($util.CanonicalName) -> $($util.SourcePath)" "Green"
            }
            catch {
                Write-ColoredOutput "  Failed: $($util.CanonicalName) - $($_.Exception.Message)" "Red"
            }
        }
        else {
            Write-ColoredOutput "  Source not found: $($util.SourcePath)" "Yellow"
        }
    }

    Write-ColoredOutput "Summary: $created created, $updated updated, $skipped skipped" "Cyan"
}

function Show-UtilityList {
    param(
        [Parameter(Mandatory=$true)]
        [array]$Utilities
    )

    if ($Utilities.Count -eq 0) {
        Write-ColoredOutput "No utilities found." "Yellow"
        return
    }

    Write-ColoredOutput "`nUtilities found:" "Cyan"
    foreach ($util in $Utilities) {
        $sourceExists = Test-Path $util.SourcePath
        $targetExists = Test-Path $util.TargetPath
        $isSymlink = $false

        if ($targetExists) {
            try {
                $linkTarget = (Get-Item $util.TargetPath).Target
                $isSymlink = $linkTarget -eq $util.SourcePath
            }
            catch { }
        }

        $status = if ($isSymlink) { "[LINKED]" } elseif ($targetExists) { "[EXISTS]" } elseif ($sourceExists) { "[AVAILABLE]" } else { "[MISSING]" }
        $color = switch ($status) {
            "[LINKED]" { "Green" }
            "[EXISTS]" { "Yellow" }
            "[AVAILABLE]" { "White" }
            "[MISSING]" { "Red" }
        }

        Write-ColoredOutput "  $status $($util.CanonicalName) ($($util.Type))" $color
    }
}

function Verify-Environment {
    Write-ColoredOutput "`nVerifying environment..." "Cyan"

    # Check Node.js and npm
    try {
        $nodeVersion = & node --version 2>$null
        $npmVersion = & npm --version 2>$null
        Write-ColoredOutput "  Node.js: $nodeVersion" "Green"
        Write-ColoredOutput "  npm: $npmVersion" "Green"
    }
    catch {
        Write-ColoredOutput "  Node.js/npm: Not working properly" "Red"
    }

    # Check Python
    if (Test-Path $PythonExecutable) {
        try {
            $pythonVersion = & $PythonExecutable --version 2>$null
            Write-ColoredOutput "  Python: $pythonVersion" "Green"
        }
        catch {
            Write-ColoredOutput "  Python: Configured but not working" "Yellow"
        }
    }
    else {
        Write-ColoredOutput "  Python: Not found at expected location" "Red"
    }

    # Check PATH directories
    Write-ColoredOutput "  PATH directories:" "White"
    foreach ($dir in $BinDirectories) {
        $exists = Test-Path $dir
        $inPath = $env:PATH -split ';' -contains $dir
        $status = if ($exists -and $inPath) { "[OK]" } elseif ($exists) { "[NOT IN PATH]" } else { "[MISSING]" }
        $color = if ($exists -and $inPath) { "Green" } elseif ($exists) { "Yellow" } else { "Red" }
        Write-ColoredOutput "    $status $dir" $color
    }
}

# Main execution
Write-ColoredOutput "=== Development Utilities Management Script ===" "Cyan"
Write-ColoredOutput "Action: $Action" "White"

switch ($Action) {
    'Update' {
        Write-ColoredOutput "`nGathering utilities..." "Yellow"
        $allUtils = @()
        $allUtils += Get-CoreUtilities
        $allUtils += Get-WuUtilities
        $allUtils += Get-WrapperUtilities

        Write-ColoredOutput "Found $($allUtils.Count) utilities to process." "White"

        if ($allUtils.Count -gt 0) {
            Update-Symlinks -Utilities $allUtils
        }

        Update-NodejsWrappers
    }

    'List' {
        $allUtils = @()
        $allUtils += Get-CoreUtilities
        $allUtils += Get-WuUtilities
        $allUtils += Get-WrapperUtilities
        Show-UtilityList -Utilities $allUtils
    }

    'Verify' {
        Verify-Environment
    }

    'FixNodejs' {
        Update-NodejsWrappers
    }

    'Clean' {
        Write-ColoredOutput "Clean action not implemented yet." "Yellow"
    }
}

Write-ColoredOutput "`n=== Script completed ===" "Cyan"
