#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Comprehensive Developer Tools Discovery and Symlink Management

.DESCRIPTION
    Discovers all development tools from various toolchains (MSVC, MinGW, Rust, Python, Node.js)
    and creates symlinks in user bin directories for easy access.

.PARAMETER Action
    The action to perform: Discover, Symlink, Verify, or Report

.EXAMPLE
    .\Manage-DevTools.ps1 -Action Discover
    .\Manage-DevTools.ps1 -Action Symlink -Force
    .\Manage-DevTools.ps1 -Action Report
#>

param(
    [Parameter(Mandatory=$false)]
    [ValidateSet('Discover', 'Symlink', 'Verify', 'Report', 'SccacheFix')]
    [string]$Action = 'Discover',

    [Parameter(Mandatory=$false)]
    [switch]$Force
)

$ErrorActionPreference = 'Continue'

# Configuration
$UserBinDir = "C:\users\david\bin"
$UserLocalBinDir = "C:\users\david\.local\bin"

$ToolCategories = @{
    'MSVC' = @(
        'cl.exe', 'link.exe', 'lib.exe', 'dumpbin.exe', 'editbin.exe',
        'nmake.exe', 'msbuild.exe', 'vcvarsall.bat'
    )
    'MinGW' = @(
        'gcc.exe', 'g++.exe', 'gfortran.exe', 'mingw32-make.exe',
        'x86_64-w64-mingw32-gcc.exe', 'x86_64-w64-mingw32-g++.exe'
    )
    'MSYS' = @(
        'bash.exe', 'sh.exe', 'awk.exe', 'sed.exe', 'grep.exe',
        'find.exe', 'xargs.exe'
    )
    'BuildTools' = @(
        'make.exe', 'cmake.exe', 'ninja.exe', 'meson.exe', 'pkg-config.exe'
    )
    'Debuggers' = @(
        'gdb.exe', 'lldb.exe', 'windbg.exe', 'cdb.exe'
    )
    'BinUtils' = @(
        'ar.exe', 'as.exe', 'ld.exe', 'nm.exe', 'objdump.exe', 'objcopy.exe',
        'ranlib.exe', 'strip.exe', 'readelf.exe', 'size.exe'
    )
    'Rust' = @(
        'cargo.exe', 'rustc.exe', 'rustup.exe', 'rustfmt.exe',
        'cargo-clippy.exe', 'rust-analyzer.exe', 'cargo-fmt.exe',
        'cargo-miri.exe', 'cargo-nextest.exe'
    )
    'Python' = @(
        'python.exe', 'python3.exe', 'pip.exe', 'pip3.exe',
        'poetry.exe', 'black.exe', 'ruff.exe', 'mypy.exe',
        'pytest.exe', 'uvx.exe', 'uv.exe'
    )
    'NodeJS' = @(
        'node.exe', 'npm.cmd', 'npx.cmd', 'tsc.cmd', 'tsx.exe',
        'eslint.exe', 'prettier.exe', 'webpack.exe', 'vite.exe'
    )
    'Performance' = @(
        'sccache.exe', 'ccache.exe', 'mold.exe', 'lld.exe'
    )
    'VersionControl' = @(
        'git.exe', 'gh.exe', 'hg.exe', 'svn.exe'
    )
    'Containers' = @(
        'docker.exe', 'kubectl.exe', 'helm.exe', 'podman.exe'
    )
}

function Write-ColoredOutput {
    param(
        [string]$Message,
        [string]$Color = 'White'
    )
    Write-Host $Message -ForegroundColor $Color
}

function Find-ToolsInPath {
    param(
        [string[]]$ToolNames
    )

    $found = @{}
    foreach ($tool in $ToolNames) {
        $cmd = Get-Command $tool -ErrorAction SilentlyContinue
        if ($cmd) {
            $found[$tool] = $cmd.Source
        }
    }
    return $found
}

function Find-ToolsInDirectory {
    param(
        [string]$Directory,
        [string[]]$ToolNames
    )

    if (-not (Test-Path $Directory)) {
        return @{}
    }

    $found = @{}
    foreach ($tool in $ToolNames) {
        $path = Join-Path $Directory $tool
        if (Test-Path $path) {
            $found[$tool] = $path
        }
    }
    return $found
}

function Search-CommonLocations {
    param(
        [string[]]$ToolNames
    )

    $locations = @(
        "C:\Program Files\Microsoft Visual Studio",
        "C:\Program Files (x86)\Microsoft Visual Studio",
        "C:\codedev\msys64\mingw64\bin",
        "C:\codedev\msys64\usr\bin",
        "C:\msys64\mingw64\bin",
        "C:\msys64\usr\bin",
        "C:\MinGW\bin",
        "$env:USERPROFILE\.cargo\bin",
        "$env:USERPROFILE\.rustup\toolchains\stable-x86_64-pc-windows-msvc\bin",
        "$env:LOCALAPPDATA\Programs\Python",
        "C:\Python3*",
        "$env:APPDATA\npm"
    )

    $found = @{}
    foreach ($location in $locations) {
        if ($location -match "\*") {
            $dirs = Get-Item $location -ErrorAction SilentlyContinue
        } else {
            $dirs = @($location)
        }

        foreach ($dir in $dirs) {
            if (Test-Path $dir) {
                $tools = Find-ToolsInDirectory -Directory $dir -ToolNames $ToolNames
                foreach ($tool in $tools.Keys) {
                    if (-not $found.ContainsKey($tool)) {
                        $found[$tool] = $tools[$tool]
                    }
                }
            }
        }
    }

    return $found
}

function Get-AllDiscoveredTools {
    Write-ColoredOutput "`nDiscovering developer tools..." "Cyan"

    $allTools = @{}

    foreach ($category in $ToolCategories.Keys) {
        Write-ColoredOutput "  Searching for $category tools..." "Yellow"

        $tools = $ToolCategories[$category]

        # First check PATH
        $foundInPath = Find-ToolsInPath -ToolNames $tools

        # Then check common locations
        $foundInDirs = Search-CommonLocations -ToolNames $tools

        # Merge results
        foreach ($tool in $tools) {
            $location = $null
            if ($foundInPath.ContainsKey($tool)) {
                $location = $foundInPath[$tool]
            } elseif ($foundInDirs.ContainsKey($tool)) {
                $location = $foundInDirs[$tool]
            }

            if ($location) {
                $allTools[$tool] = @{
                    Category = $category
                    Path = $location
                    InUserBin = ($location -like "$UserBinDir*" -or $location -like "$UserLocalBinDir*")
                }
            }
        }
    }

    return $allTools
}

function Show-ToolReport {
    param(
        [hashtable]$Tools
    )

    Write-ColoredOutput "`n=== Developer Tools Discovery Report ===" "Cyan"

    $categories = $Tools.Values.Category | Select-Object -Unique | Sort-Object

    foreach ($category in $categories) {
        Write-ColoredOutput "`n[$category]" "Green"

        $categoryTools = $Tools.GetEnumerator() | Where-Object { $_.Value.Category -eq $category } | Sort-Object Name

        foreach ($tool in $categoryTools) {
            $name = $tool.Key
            $path = $tool.Value.Path
            $inUserBin = $tool.Value.InUserBin

            $status = if ($inUserBin) { "[IN BIN]" } else { "[EXTERNAL]" }
            $color = if ($inUserBin) { "Gray" } else { "White" }

            Write-ColoredOutput "  $status $name" $color
            Write-ColoredOutput "         $path" "DarkGray"
        }
    }

    # Summary
    $total = $Tools.Count
    $inUserBin = ($Tools.Values | Where-Object { $_.InUserBin }).Count
    $external = $total - $inUserBin

    Write-ColoredOutput "`n=== Summary ===" "Cyan"
    Write-ColoredOutput "  Total tools found: $total" "White"
    Write-ColoredOutput "  Already in user bin: $inUserBin" "Green"
    Write-ColoredOutput "  External (can be symlinked): $external" "Yellow"
}

function New-ToolSymlinks {
    param(
        [hashtable]$Tools,
        [switch]$Force
    )

    if (-not ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        Write-ColoredOutput "ERROR: Administrative privileges required for creating symlinks." "Red"
        Write-ColoredOutput "Please run this script as Administrator." "Yellow"
        return
    }

    Write-ColoredOutput "`nCreating symlinks for external tools..." "Cyan"

    # Ensure user bin directories exist
    @($UserBinDir, $UserLocalBinDir) | ForEach-Object {
        if (-not (Test-Path $_)) {
            New-Item -ItemType Directory -Path $_ -Force | Out-Null
            Write-ColoredOutput "  Created directory: $_" "Green"
        }
    }

    $created = 0
    $skipped = 0
    $failed = 0

    $externalTools = $Tools.GetEnumerator() | Where-Object { -not $_.Value.InUserBin }

    foreach ($tool in $externalTools) {
        $name = $tool.Key
        $sourcePath = $tool.Value.Path

        # Prefer .local\bin for new symlinks
        $targetPath = Join-Path $UserLocalBinDir $name

        # Check if already exists
        if (Test-Path $targetPath) {
            if (-not $Force) {
                Write-ColoredOutput "  Skipped: $name (already exists, use -Force to overwrite)" "Gray"
                $skipped++
                continue
            } else {
                Remove-Item $targetPath -Force -ErrorAction SilentlyContinue
            }
        }

        try {
            New-Item -ItemType SymbolicLink -Path $targetPath -Target $sourcePath -Force | Out-Null
            Write-ColoredOutput "  Created: $name -> $sourcePath" "Green"
            $created++
        } catch {
            Write-ColoredOutput "  Failed: $name - $($_.Exception.Message)" "Red"
            $failed++
        }
    }

    Write-ColoredOutput "`nSymlink Summary:" "Cyan"
    Write-ColoredOutput "  Created: $created" "Green"
    Write-ColoredOutput "  Skipped: $skipped" "Yellow"
    Write-ColoredOutput "  Failed: $failed" "Red"
}

function Test-ToolchainIntegrity {
    Write-ColoredOutput "`n=== Verifying Toolchain Integrity ===" "Cyan"

    # Test C/C++
    Write-ColoredOutput "`n[C/C++ Toolchain]" "Green"
    try {
        $clVersion = & cl /? 2>&1 | Select-String "Version" | Select-Object -First 1
        if ($clVersion) {
            Write-ColoredOutput "  MSVC: OK - $clVersion" "Green"
        }
    } catch {
        Write-ColoredOutput "  MSVC: Not available" "Yellow"
    }

    try {
        $gccVersion = & gcc --version 2>&1 | Select-Object -First 1
        Write-ColoredOutput "  GCC: OK - $gccVersion" "Green"
    } catch {
        Write-ColoredOutput "  GCC: Not available" "Yellow"
    }

    # Test Rust
    Write-ColoredOutput "`n[Rust Toolchain]" "Green"
    try {
        $rustcVersion = & rustc --version
        $cargoVersion = & cargo --version
        Write-ColoredOutput "  rustc: $rustcVersion" "Green"
        Write-ColoredOutput "  cargo: $cargoVersion" "Green"
    } catch {
        Write-ColoredOutput "  Rust: Not available" "Red"
    }

    # Test Python
    Write-ColoredOutput "`n[Python Toolchain]" "Green"
    try {
        $pythonVersion = & python --version 2>&1
        Write-ColoredOutput "  Python: $pythonVersion" "Green"
    } catch {
        Write-ColoredOutput "  Python: Not available" "Red"
    }

    # Test Node.js
    Write-ColoredOutput "`n[Node.js Toolchain]" "Green"
    try {
        $nodeVersion = & node --version
        $npmVersion = & npm --version
        Write-ColoredOutput "  Node.js: $nodeVersion" "Green"
        Write-ColoredOutput "  npm: $npmVersion" "Green"
    } catch {
        Write-ColoredOutput "  Node.js: Not available" "Red"
    }

    # Test Build Tools
    Write-ColoredOutput "`n[Build Tools]" "Green"
    @('cmake', 'make', 'ninja') | ForEach-Object {
        $toolName = $_
        try {
            $version = & $toolName --version 2>&1 | Select-Object -First 1
            Write-ColoredOutput "  ${toolName}: OK - $version" "Green"
        } catch {
            Write-ColoredOutput "  ${toolName}: Not available" "Yellow"
        }
    }
}

function Repair-SccacheConfiguration {
    Write-ColoredOutput "`n=== Fixing sccache Configuration ===" "Cyan"

    # Check sccache status
    try {
        Write-ColoredOutput "Current sccache statistics:" "Yellow"
        & sccache --show-stats

        Write-ColoredOutput "`nRestarting sccache server..." "Yellow"
        & sccache --stop-server 2>&1 | Out-Null
        Start-Sleep -Seconds 2
        & sccache --start-server

        Write-ColoredOutput "sccache server restarted successfully" "Green"

        # Display configuration recommendations
        Write-ColoredOutput "`nRecommended sccache environment variables (add to your profile):" "Cyan"
        Write-ColoredOutput '  $env:RUSTC_WRAPPER = "sccache"' "White"
        Write-ColoredOutput '  $env:SCCACHE_DIR = "T:\RustCache\sccache"' "White"
        Write-ColoredOutput '  $env:SCCACHE_CACHE_SIZE = "15G"' "White"
        Write-ColoredOutput '  $env:CARGO_INCREMENTAL = "0"  # REQUIRED for sccache' "White"

    } catch {
        Write-ColoredOutput "Failed to manage sccache: $($_.Exception.Message)" "Red"
    }
}

# Main execution
Write-ColoredOutput "=== Developer Tools Management ===" "Cyan"
Write-ColoredOutput "Action: $Action" "White"

switch ($Action) {
    'Discover' {
        $tools = Get-AllDiscoveredTools
        Show-ToolReport -Tools $tools
    }

    'Symlink' {
        $tools = Get-AllDiscoveredTools
        New-ToolSymlinks -Tools $tools -Force:$Force
    }

    'Verify' {
        Test-ToolchainIntegrity
    }

    'Report' {
        $tools = Get-AllDiscoveredTools
        Show-ToolReport -Tools $tools
        Test-ToolchainIntegrity
    }

    'SccacheFix' {
        Repair-SccacheConfiguration
    }
}

Write-ColoredOutput "`n=== Script completed ===" "Cyan"
