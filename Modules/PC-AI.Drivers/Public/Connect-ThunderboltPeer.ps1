#Requires -Version 7.0
<#
.SYNOPSIS
    Connects to a Windows peer over a Thunderbolt / USB4 networking link.

.DESCRIPTION
    Uses Get-ThunderboltNetworkStatus to discover a peer address when no explicit
    ComputerName or Address is provided. The connection path prefers standard WinRM
    Negotiate authentication and supports Microsoft account credentials when a
    SecureString password is supplied.

.PARAMETER ComputerName
    Explicit WinRM computer name to target.

.PARAMETER Address
    Explicit IPv4 or IPv6 address to target. Link-local IPv6 addresses may omit the
    scope suffix; the selected interface index is appended automatically.

.PARAMETER InterfaceAlias
    Thunderbolt / USB4 interface alias used for discovery and link-local IPv6 scope.

.PARAMETER Credential
    Optional explicit PSCredential.

.PARAMETER Password
    Optional SecureString password used to build a Microsoft account credential.

.PARAMETER MicrosoftAccountEmail
    Email used when building a Microsoft account credential.

.PARAMETER ScriptBlock
    Command to run after the connection succeeds. Defaults to a small identity probe.

.PARAMETER ReturnSession
    Return a PSSession instead of invoking a script block.
#>
function Connect-ThunderboltPeer {
    [CmdletBinding()]
    param(
        [Parameter()]
        [string]$ComputerName,

        [Parameter()]
        [string]$Address,

        [Parameter()]
        [string]$InterfaceAlias,

        [Parameter()]
        [System.Management.Automation.PSCredential]$Credential,

        [Parameter()]
        [Security.SecureString]$Password,

        [Parameter()]
        [string]$MicrosoftAccountEmail = 'davidmartel07@gmail.com',

        [Parameter()]
        [ValidateSet('Auto', 'WinRM', 'SSH')]
        [string]$Protocol = 'Auto',

        [Parameter()]
        [string]$SshUser = 'david',

        [Parameter()]
        [string]$SshCommand = 'hostname && whoami',

        [Parameter()]
        [scriptblock]$ScriptBlock = {
            [PSCustomObject]@{
                ComputerName = $env:COMPUTERNAME
                UserName     = $env:USERNAME
                TimeUtc      = (Get-Date).ToUniversalTime().ToString('o')
            }
        },

        [Parameter()]
        [switch]$ReturnSession
    )

    Set-StrictMode -Version Latest
    $ErrorActionPreference = 'Stop'

    if (-not $Credential -and $Password) {
        $Credential = [System.Management.Automation.PSCredential]::new(
            "MicrosoftAccount\$MicrosoftAccountEmail",
            $Password
        )
    }

    $target = $null
    $peerHostname = $null
    $localBindIp = $null
    if ($ComputerName) {
        $target = $ComputerName
        $peerHostname = $ComputerName
        # If ComputerName does not resolve via DNS/hosts, fall back to deterministic USB4 ULA / Link-Local
        $dnsResolved = $false
        try {
            $addrs = @([System.Net.Dns]::GetHostAddresses($ComputerName))
            if ($addrs.Count -gt 0) { $dnsResolved = $true }
        } catch { }
        if (-not $dnsResolved -and (Get-Command Get-ThunderboltPeerIdentity -ErrorAction SilentlyContinue)) {
            $peerId = Get-ThunderboltPeerIdentity -ComputerName $ComputerName
            $peerHostname = $peerId.ComputerName
            $target = $peerId.PrimaryUla
        }
    } elseif ($Address) {
        $target = $Address
    } else {
        $status = if ([string]::IsNullOrWhiteSpace($InterfaceAlias)) {
            @(Get-ThunderboltNetworkStatus -ProbeWinRM)
        } else {
            @(Get-ThunderboltNetworkStatus -InterfaceAlias $InterfaceAlias -ProbeWinRM)
        }
        if ($status.Count -eq 0 -and -not [string]::IsNullOrWhiteSpace($InterfaceAlias)) {
            $status = @(Get-ThunderboltNetworkStatus -ProbeWinRM)
        }
        $annotatedCandidates = @(foreach ($adapterStatus in $status) {
            foreach ($cand in @($adapterStatus.NeighborCandidates)) {
                [PSCustomObject]@{
                    Candidate      = $cand
                    AdapterStatus  = $adapterStatus
                }
            }
        })
        $selectedEntry = $annotatedCandidates |
            Where-Object {
                ($Protocol -eq 'SSH' -and $_.Candidate.SshReachable -eq $true) -or
                ($Protocol -ne 'SSH' -and ($_.Candidate.WsManReachable -eq $true -or ($Protocol -eq 'Auto' -and $_.Candidate.SshReachable -eq $true)))
            } |
            Select-Object -First 1

        if (-not $selectedEntry) {
            $allCandidates = @($annotatedCandidates | ForEach-Object { $_.Candidate.IPAddress } | Select-Object -Unique)
            if ($allCandidates.Count -gt 0) {
                throw "No reachable Thunderbolt peer was discovered on '$InterfaceAlias'. Discovered peer addresses: $($allCandidates -join ', ')"
            }
            $resolvedAlias = if ($status.Count -gt 0) { [string]$status[0].InterfaceAlias } else { [string]$InterfaceAlias }
            throw "No Thunderbolt peer address was discovered on '$resolvedAlias'. Verify the remote machine exposes a live USB4 P2P adapter."
        }

        $candidate = $selectedEntry.Candidate
        $owningAdapter = $selectedEntry.AdapterStatus
        $target = $candidate.IPAddress
        if ($candidate.PSObject.Properties['PeerHostname']) {
            $peerHostname = $candidate.PeerHostname
        }
        if ([string]::IsNullOrWhiteSpace($InterfaceAlias) -and $owningAdapter) {
            $InterfaceAlias = [string]$owningAdapter.InterfaceAlias
        }
        if ($owningAdapter) {
            $localIps = @($owningAdapter.IPv4Addresses | Where-Object { $_ -like '169.254.*' })
            if ($localIps.Count -gt 0) { $localBindIp = [string]$localIps[0] }
        }
    }

    if ($target -match '^fe80:' -and $target -notmatch '%') {
        $scope = Get-CimInstance -Namespace 'root/StandardCimv2' -ClassName 'MSFT_NetIPInterface' -ErrorAction SilentlyContinue |
            Where-Object { $_.InterfaceAlias -eq $InterfaceAlias -and $_.AddressFamily -eq 23 } |
            Select-Object -First 1 -ExpandProperty InterfaceIndex
        if ($scope) {
            $target = "$target%$scope"
        }
    }

    if ($Protocol -eq 'SSH') {
        $sshArgs = @('-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5', '-o', 'StrictHostKeyChecking=accept-new')
        if ($localBindIp -and $target -like '169.254.*') {
            $sshArgs += @('-o', "BindAddress=$localBindIp")
        }
        $sshArgs += @('-l', $SshUser, $target, $SshCommand)
        $sshOutput = & ssh.exe @sshArgs 2>&1
        if ($LASTEXITCODE -ne 0) {
            throw "Thunderbolt peer SSH connection failed for '$SshUser@$target': $($sshOutput -join ' ')"
        }
        return [PSCustomObject]@{
            Target         = $target
            PeerHostname   = $peerHostname
            Protocol       = 'SSH'
            Authentication = "SSHKey($SshUser)"
            Result         = ($sshOutput -join [Environment]::NewLine).Trim()
        }
    }

    $probeParams = @{
        ComputerName   = $target
        Authentication = 'Negotiate'
        ErrorAction    = 'Stop'
    }
    if ($Credential) {
        $probeParams['Credential'] = $Credential
    }

    try {
        Test-WSMan @probeParams | Out-Null
    } catch {
        throw "Thunderbolt peer WSMan probe failed for '$target': $($_.Exception.Message)"
    }

    $sessionParams = @{
        ComputerName   = $target
        Authentication = 'Negotiate'
        ErrorAction    = 'Stop'
    }
    if ($Credential) {
        $sessionParams['Credential'] = $Credential
    }

    if ($ReturnSession) {
        return New-PSSession @sessionParams
    }

    $result = Invoke-Command @sessionParams -ScriptBlock $ScriptBlock
    return [PSCustomObject]@{
        Target         = $target
        PeerHostname   = $peerHostname
        Protocol       = 'WinRM'
        Authentication = if ($Credential) { 'Negotiate+ExplicitCredential' } else { 'Negotiate+CurrentUser' }
        Result         = $result
    }
}
