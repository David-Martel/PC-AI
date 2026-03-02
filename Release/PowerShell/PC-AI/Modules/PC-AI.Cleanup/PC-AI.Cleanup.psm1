#Requires -Version 5.1
<#
.SYNOPSIS
    PC-AI.Cleanup module loader.

.DESCRIPTION
    This module provides cleanup and maintenance functions for Windows systems:
    - PATH environment variable analysis and repair
    - Duplicate file detection
    - Temporary file cleanup

.NOTES
    Module: PC-AI.Cleanup
    Author: PC_AI Project
    Version: 1.0.0
#>

# Module-level variables
$script:ModuleName = 'PC-AI.Cleanup'
$script:ModulePath = $PSScriptRoot
$script:LogPath = Join-Path -Path $env:LOCALAPPDATA -ChildPath 'PC-AI\Logs'

# Ensure log directory exists
if (-not (Test-Path -Path $script:LogPath)) {
    $null = New-Item -Path $script:LogPath -ItemType Directory -Force
}

# Get public and private function definition files
$Public = @(Get-ChildItem -Path "$PSScriptRoot\Public\*.ps1" -ErrorAction SilentlyContinue)
$Private = @(Get-ChildItem -Path "$PSScriptRoot\Private\*.ps1" -ErrorAction SilentlyContinue)

# Dot source the files
foreach ($import in @($Public + $Private)) {
    try {
        Write-Verbose "Importing $($import.FullName)"
        . $import.FullName
    }
    catch {
        Write-Error "Failed to import function $($import.FullName): $_"
    }
}

# Export public functions
Export-ModuleMember -Function $Public.BaseName

# Module initialization message
Write-Verbose "PC-AI.Cleanup module loaded. $($Public.Count) public functions available."
