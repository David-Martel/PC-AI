#Requires -Version 5.1
<#
.SYNOPSIS
    Gets comprehensive WSL status information

.DESCRIPTION
    Queries WSL for distribution list, version information, and configuration.

.PARAMETER Detailed
    Include detailed configuration from .wslconfig

.EXAMPLE
    Get-WSLStatus
    Returns basic WSL status

.EXAMPLE
    Get-WSLStatus -Detailed
    Returns detailed WSL configuration

.OUTPUTS
    PSCustomObject with WSL status information
#>
function Get-WSLStatus {
    [CmdletBinding()]
    [OutputType([PSCustomObject])]
    param(
        [Parameter()]
        [switch]$Detailed
    )

    $result = [PSCustomObject]@{
        Installed      = $false
        Version        = $null
        Distributions  = @()
        DefaultDistro  = $null
        WSLConfigPath  = "$env:USERPROFILE\.wslconfig"
        WSLConfig      = $null
        KernelVersion  = $null
        Severity       = 'OK'
    }

    try {
        # Check if WSL is available
        $wslPath = Get-Command wsl.exe -ErrorAction SilentlyContinue
        if (-not $wslPath) {
            $result.Severity = 'Critical'
            return $result
        }

        $result.Installed = $true

        # Get WSL version
        try {
            $versionOutput = wsl --version 2>&1
            if ($versionOutput -match 'WSL version:\s*(.+)') {
                $result.Version = $matches[1].Trim()
            }
            if ($versionOutput -match 'Kernel version:\s*(.+)') {
                $result.KernelVersion = $matches[1].Trim()
            }
        }
        catch {
            Write-Verbose "Could not get WSL version: $_"
        }

        # Get distribution list
        try {
            $listOutput = wsl -l -v 2>&1
            $lines = $listOutput -split "`n" | Where-Object { $_.Trim() -and $_ -notmatch '^\s*NAME\s+STATE\s+VERSION' }

            foreach ($line in $lines) {
                if ($line -match '^\s*(\*?)\s*(\S+)\s+(\S+)\s+(\d+)') {
                    $isDefault = $matches[1] -eq '*'
                    $distro = [PSCustomObject]@{
                        Name      = $matches[2]
                        State     = $matches[3]
                        Version   = [int]$matches[4]
                        IsDefault = $isDefault
                    }
                    $result.Distributions += $distro

                    if ($isDefault) {
                        $result.DefaultDistro = $matches[2]
                    }
                }
            }
        }
        catch {
            Write-Verbose "Could not get distribution list: $_"
        }

        # Read .wslconfig if detailed requested
        if ($Detailed -and (Test-Path $result.WSLConfigPath)) {
            $result.WSLConfig = Get-Content $result.WSLConfigPath -Raw
        }

        # Determine severity
        if ($result.Distributions.Count -eq 0) {
            $result.Severity = 'Warning'
        }
        elseif ($result.Distributions | Where-Object { $_.State -eq 'Stopped' -and $_.IsDefault }) {
            $result.Severity = 'Info'
        }

        return $result

    }
    catch {
        Write-Error "Failed to get WSL status: $($_.Exception.Message)"
        $result.Severity = 'Error'
        return $result
    }
}
