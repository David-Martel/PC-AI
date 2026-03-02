#Requires -Version 5.1
<#
.SYNOPSIS
    Retrieves status of the PC_AI VSock bridge service in WSL.
#>
function Get-WSLVsockBridgeStatus {
    [CmdletBinding()]
    [OutputType([PSCustomObject])]
    param(
        [Parameter()]
        [string]$Distribution = 'Ubuntu'
    )

    $result = [PSCustomObject]@{
        Distribution = $Distribution
        ServiceState = $null
        BridgeStatus = $null
        Errors = @()
    }

    try {
        $svc = wsl -d $Distribution -- systemctl is-active pcai-vsock-bridge 2>&1
        $result.ServiceState = $svc.Trim()

        $status = wsl -d $Distribution -- /usr/local/bin/pcai-vsock-bridge status 2>&1
        $result.BridgeStatus = ($status | Out-String).Trim()
    }
    catch {
        $result.Errors += $_.Exception.Message
    }

    return $result
}
