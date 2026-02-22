#Requires -Version 5.1
<#
.SYNOPSIS
    Installs and enables the PC_AI VSock bridge in WSL.
#>
function Install-WSLVsockBridge {
    [CmdletBinding()]
    [OutputType([PSCustomObject])]
    param(
        [Parameter()]
        [string]$Distribution = 'Ubuntu',

        [Parameter()]
        [string]$BridgeScriptPath,

        [Parameter()]
        [string]$ServiceFilePath,

        [Parameter()]
        [string]$ConfigPath,

        [Parameter()]
        [switch]$EnableService = $true,

        [Parameter()]
        [switch]$StartService = $true
    )

    $result = [PSCustomObject]@{
        Distribution = $Distribution
        ScriptInstalled = $false
        ServiceInstalled = $false
        ConfigInstalled = $false
        SocatInstalled = $false
        ServiceEnabled = $false
        ServiceStarted = $false
        Errors = @()
    }

    function Convert-WindowsToWslPath {
        param([Parameter(Mandatory)][string]$WindowsPath)

        $fullPath = [System.IO.Path]::GetFullPath($WindowsPath)
        if ($fullPath -match '^(?<drive>[A-Za-z]):\\(?<rest>.*)$') {
            $drive = $Matches['drive'].ToLowerInvariant()
            $rest = ($Matches['rest'] -replace '\\', '/')
            if ($rest) {
                return "/mnt/$drive/$rest"
            }
            return "/mnt/$drive"
        }

        throw "Unable to convert Windows path to WSL path: $WindowsPath"
    }

    try {
        if (-not $BridgeScriptPath -or -not $ServiceFilePath -or -not $ConfigPath) {
            $repoRoot = if ($env:PCAI_ROOT -and (Test-Path $env:PCAI_ROOT)) {
                $env:PCAI_ROOT
            } else {
                Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
            }

            if (-not $BridgeScriptPath) { $BridgeScriptPath = Join-Path $repoRoot 'Tools\pcai-vsock-bridge.sh' }
            if (-not $ServiceFilePath) { $ServiceFilePath = Join-Path $repoRoot 'Tools\pcai-vsock-bridge.service' }
            if (-not $ConfigPath) { $ConfigPath = Join-Path $repoRoot 'Config\vsock-bridges.conf' }
        }

        if (-not (Test-Path $BridgeScriptPath)) { throw "Bridge script not found: $BridgeScriptPath" }
        if (-not (Test-Path $ServiceFilePath)) { throw "Service file not found: $ServiceFilePath" }
        if (-not (Test-Path $ConfigPath)) { throw "Config file not found: $ConfigPath" }

        $bridgeWslSource = Convert-WindowsToWslPath -WindowsPath $BridgeScriptPath
        $serviceWslSource = Convert-WindowsToWslPath -WindowsPath $ServiceFilePath
        $configWslSource = Convert-WindowsToWslPath -WindowsPath $ConfigPath

        # Ensure systemd enabled
        Enable-WSLSystemd -Distribution $Distribution -RestartWSL | Out-Null

        # Ensure distro is up
        & wsl -d $Distribution -- echo "WSL up" 2>$null | Out-Null

        # Install socat if missing
        $socatCheck = wsl -d $Distribution -- bash -lc "command -v socat >/dev/null 2>&1" 2>$null
        if ($LASTEXITCODE -ne 0) {
            wsl -d $Distribution -- bash -lc "sudo apt-get update && sudo apt-get install -y socat" | Out-Null
        }
        $socatCheck = wsl -d $Distribution -- bash -lc "command -v socat >/dev/null 2>&1" 2>$null
        if ($LASTEXITCODE -eq 0) { $result.SocatInstalled = $true }

        # Copy bridge script
        $bridgeWslPath = '/usr/local/bin/pcai-vsock-bridge'
        wsl -d $Distribution -- bash -lc "sudo cp '$bridgeWslSource' $bridgeWslPath && sudo chmod 755 $bridgeWslPath" | Out-Null
        $result.ScriptInstalled = $true

        # Copy config
        wsl -d $Distribution -- bash -lc "sudo mkdir -p /etc/pcai && sudo cp '$configWslSource' /etc/pcai/vsock-bridges.conf" | Out-Null
        $result.ConfigInstalled = $true

        # Copy systemd service
        wsl -d $Distribution -- bash -lc "sudo cp '$serviceWslSource' /etc/systemd/system/pcai-vsock-bridge.service" | Out-Null
        wsl -d $Distribution -- bash -lc "sudo systemctl daemon-reload" | Out-Null
        $result.ServiceInstalled = $true

        if ($EnableService) {
            wsl -d $Distribution -- bash -lc "sudo systemctl enable pcai-vsock-bridge" | Out-Null
            $result.ServiceEnabled = $true
        }

        if ($StartService) {
            wsl -d $Distribution -- bash -lc "sudo systemctl restart pcai-vsock-bridge" | Out-Null
            $result.ServiceStarted = $true
        }
    }
    catch {
        $result.Errors += $_.Exception.Message
    }

    return $result
}
