#Requires -Version 5.1
<#
.SYNOPSIS
    Gets Docker Desktop status and configuration

.DESCRIPTION
    Checks Docker Desktop installation, service status, and configuration.

.PARAMETER IncludeContainers
    Include running container list

.EXAMPLE
    Get-DockerStatus
    Returns Docker status

.OUTPUTS
    PSCustomObject with Docker status information
#>
function Get-DockerStatus {
    [CmdletBinding()]
    [OutputType([PSCustomObject])]
    param(
        [Parameter()]
        [switch]$IncludeContainers
    )

    $result = [PSCustomObject]@{
        Installed     = $false
        Running       = $false
        Version       = $null
        Backend       = $null
        Containers    = @()
        ConfigPath    = "$env:APPDATA\Docker\settings.json"
        Severity      = 'Unknown'
    }

    try {
        # Check if Docker is installed
        $dockerPath = Get-Command docker.exe -ErrorAction SilentlyContinue
        if (-not $dockerPath) {
            $result.Severity = 'Info'
            return $result
        }

        $result.Installed = $true

        # Check Docker version and status
        try {
            $versionOutput = docker version --format '{{.Server.Version}}' 2>&1
            if ($LASTEXITCODE -eq 0) {
                $result.Version = $versionOutput
                $result.Running = $true
                $result.Severity = 'OK'
            }
            else {
                $result.Severity = 'Warning'
            }
        }
        catch {
            $result.Severity = 'Warning'
        }

        # Determine backend
        $infoOutput = docker info --format '{{.OSType}}' 2>&1
        if ($LASTEXITCODE -eq 0) {
            $result.Backend = if ($infoOutput -eq 'linux') { 'WSL2' } else { 'Windows' }
        }

        # Get containers if requested and Docker is running
        if ($IncludeContainers -and $result.Running) {
            try {
                $containers = docker ps --format '{{.ID}}\t{{.Names}}\t{{.Status}}\t{{.Image}}' 2>&1
                if ($LASTEXITCODE -eq 0 -and $containers) {
                    $containerLines = $containers -split "`n" | Where-Object { $_.Trim() }
                    foreach ($line in $containerLines) {
                        $parts = $line -split "`t"
                        if ($parts.Count -ge 4) {
                            $result.Containers += [PSCustomObject]@{
                                ID     = $parts[0]
                                Name   = $parts[1]
                                Status = $parts[2]
                                Image  = $parts[3]
                            }
                        }
                    }
                }
            }
            catch {
                Write-Verbose "Could not get container list: $_"
            }
        }

        return $result

    }
    catch {
        Write-Error "Failed to get Docker status: $($_.Exception.Message)"
        $result.Severity = 'Error'
        return $result
    }
}
