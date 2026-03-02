#Requires -Version 5.1
<#
.SYNOPSIS
    Installs WinSocat for HVSOCK<->TCP proxy support.
#>
function Install-HVSockProxy {
    [CmdletBinding()]
    [OutputType([PSCustomObject])]
    param(
        [Parameter()]
        [ValidateSet('Auto','Winget','Chocolatey')]
        [string]$Installer = 'Auto',

        [Parameter()]
        [switch]$Force
    )

    $result = [PSCustomObject]@{
        WinSocatPath = $null
        Installed    = $false
        Method       = $null
        Errors       = @()
    }

    $cmd = Get-Command winsocat.exe -ErrorAction SilentlyContinue
    if ($cmd -and -not $Force) {
        $result.WinSocatPath = $cmd.Path
        $result.Installed = $true
        return $result
    }

    $useWinget = $false
    $useChoco = $false

    if ($Installer -eq 'Auto') {
        $useWinget = [bool](Get-Command winget.exe -ErrorAction SilentlyContinue)
        if (-not $useWinget) {
            $useChoco = [bool](Get-Command choco.exe -ErrorAction SilentlyContinue)
        }
    }
    elseif ($Installer -eq 'Winget') {
        $useWinget = $true
    }
    elseif ($Installer -eq 'Chocolatey') {
        $useChoco = $true
    }

    try {
        if ($useWinget) {
            $result.Method = 'Winget'
            winget install -e --id Firejox.WinSocat --accept-package-agreements --accept-source-agreements | Out-Null
        }
        elseif ($useChoco) {
            $result.Method = 'Chocolatey'
            choco install winsocat -y | Out-Null
        }
        else {
            throw 'No installer available. Install WinSocat manually.'
        }
    }
    catch {
        $result.Errors += $_.Exception.Message
    }

    $cmd = Get-Command winsocat.exe -ErrorAction SilentlyContinue
    if ($cmd) {
        $result.WinSocatPath = $cmd.Path
        $result.Installed = $true
    }

    return $result
}
