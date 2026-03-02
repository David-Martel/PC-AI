#Requires -Version 5.1
<#
.SYNOPSIS
    Adds Windows Defender exclusions for WSL

.DESCRIPTION
    Adds path and process exclusions to Windows Defender to improve WSL performance.

.PARAMETER WhatIf
    Show what would be added without making changes

.PARAMETER Confirm
    Prompt for confirmation before making changes

.EXAMPLE
    Set-WSLDefenderExclusion
    Adds all recommended exclusions

.OUTPUTS
    PSCustomObject with applied exclusions
#>
function Set-WSLDefenderExclusion {
    [CmdletBinding(SupportsShouldProcess)]
    [OutputType([PSCustomObject])]
    param()

    # Check for Administrator privileges
    $isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
    if (-not $isAdmin) {
        Write-Error 'This function requires Administrator privileges. Please run PowerShell as Administrator.'
        return
    }

    $result = [PSCustomObject]@{
        PathExclusions    = @()
        ProcessExclusions = @()
        Applied           = $false
        Errors            = @()
    }

    $pathExclusions = @(
        "$env:LOCALAPPDATA\Packages\CanonicalGroupLimited*",
        "$env:USERPROFILE\AppData\Local\Docker",
        '\\wsl$',
        '\\wsl.localhost'
    )

    $processExclusions = @(
        'wsl.exe',
        'wslhost.exe',
        'docker.exe',
        'dockerd.exe',
        'com.docker.backend.exe',
        'vpnkit.exe'
    )

    Write-Host '[*] Adding Windows Defender exclusions for WSL...' -ForegroundColor Yellow

    # Add path exclusions
    foreach ($path in $pathExclusions) {
        if ($PSCmdlet.ShouldProcess($path, 'Add Defender path exclusion')) {
            try {
                Add-MpPreference -ExclusionPath $path -ErrorAction Stop
                $result.PathExclusions += $path
                Write-Host "  [+] Added path: $path" -ForegroundColor Green
            } catch {
                $result.Errors += "Path $path : $_"
                Write-Host "  [!] Failed to add path: $path" -ForegroundColor Yellow
            }
        }
    }

    # Add process exclusions
    foreach ($process in $processExclusions) {
        if ($PSCmdlet.ShouldProcess($process, 'Add Defender process exclusion')) {
            try {
                Add-MpPreference -ExclusionProcess $process -ErrorAction Stop
                $result.ProcessExclusions += $process
                Write-Host "  [+] Added process: $process" -ForegroundColor Green
            } catch {
                $result.Errors += "Process $process : $_"
                Write-Host "  [!] Failed to add process: $process" -ForegroundColor Yellow
            }
        }
    }

    $result.Applied = ($result.PathExclusions.Count -gt 0 -or $result.ProcessExclusions.Count -gt 0)

    if ($result.Applied) {
        Write-Host '[*] Windows Defender exclusions configured' -ForegroundColor Green
    }

    return $result
}
