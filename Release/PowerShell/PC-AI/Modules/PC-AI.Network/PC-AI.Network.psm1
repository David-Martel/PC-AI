#Requires -Version 5.1
<#
.SYNOPSIS
    PC-AI Network Diagnostics and Optimization Module

.DESCRIPTION
    Provides network diagnostic and optimization functions for PC-AI framework including:
    - Comprehensive network stack analysis
    - VSock and TCP optimization for WSL2
    - Real-time VSock performance monitoring
    - WSL connectivity testing

.NOTES
    Author: PC_AI Framework
    Version: 1.0.0
#>

# Module variables
$script:ModuleRoot = $PSScriptRoot
$script:ConfigPath = Join-Path (Split-Path $ModuleRoot -Parent | Split-Path -Parent) 'Config'

# Import private functions
$privatePath = Join-Path $PSScriptRoot 'Private'
if (Test-Path $privatePath) {
    Get-ChildItem -Path $privatePath -Filter '*.ps1' | ForEach-Object {
        . $_.FullName
    }
}

# Import public functions
$publicPath = Join-Path $PSScriptRoot 'Public'
if (Test-Path $publicPath) {
    Get-ChildItem -Path $publicPath -Filter '*.ps1' | ForEach-Object {
        . $_.FullName
    }
}

# Export public functions
Export-ModuleMember -Function @(
    'Get-NetworkDiagnostics',
    'Optimize-VSock',
    'Watch-VSockPerformance',
    'Test-WSLConnectivity'
)
