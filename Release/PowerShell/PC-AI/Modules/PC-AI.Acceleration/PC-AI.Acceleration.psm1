#Requires -Version 5.1
<#
.SYNOPSIS
    PC-AI Acceleration Module - Rust and .NET performance optimizations

.DESCRIPTION
    Provides high-performance alternatives to standard PowerShell operations
    using Rust CLI tools (ripgrep, fd, procs) and .NET parallel processing.
#>

# Module-level tool cache
$script:RustToolCache = @{}
$script:ToolPaths = @{
    rg       = $null
    fd       = $null
    bat      = $null
    procs    = $null
    tokei    = $null
    sd       = $null
    eza      = $null
    hyperfine = $null
}

# Initialize tool detection on module load
$script:SearchPaths = @(
    "$env:USERPROFILE\.cargo\bin"
    "$env:USERPROFILE\bin"
    "$env:USERPROFILE\.local\bin"
    "$env:LOCALAPPDATA\Microsoft\WinGet\Links"
    "C:\Program Files\ripgrep"
)

# Dot-source all function files
$PublicPath = Join-Path $PSScriptRoot 'Public'
$PrivatePath = Join-Path $PSScriptRoot 'Private'

if (Test-Path $PrivatePath) {
    Get-ChildItem -Path $PrivatePath -Filter '*.ps1' -Recurse | ForEach-Object {
        . $_.FullName
    }
}

if (Test-Path $PublicPath) {
    Get-ChildItem -Path $PublicPath -Filter '*.ps1' -Recurse | ForEach-Object {
        . $_.FullName
    }
}

# Initialize tool paths on module load
Initialize-RustTool

# Initialize PCAI Native DLLs (silent on failure - falls back to PowerShell)
$null = Initialize-PcaiNative -ErrorAction SilentlyContinue
