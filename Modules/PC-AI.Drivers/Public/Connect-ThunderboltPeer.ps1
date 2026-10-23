#Requires -Version 7.0
<#
.SYNOPSIS
    Connects to a Windows or Linux peer over a Thunderbolt / USB4 networking link.

.DESCRIPTION
    Uses Get-ThunderboltNetworkStatus to discover a peer address when no explicit
    ComputerName or Address is provided. Supports WinRM, SSH, and Auto protocol
    negotiation (falling back cleanly to SSH when a peer is SSH-reachable or when
    WinRM is unavailable).
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

    function Invoke-SshPeerConnection {
        param(
            [string]$TargetHost,
            [string]$PeerName,
            [string]$User,
            [string]$LocalBind,
            [string]$CommandText
        )
        $sshArgs = @('-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5', '-o', 'StrictHostKeyChecking=accept-new')
        if ($LocalBind -and $TargetHost -like '169.254.*') {
            $sshArgs += @('-o', "BindAddress=$LocalBind")
        }
        $sshArgs += @('-l', $User, $TargetHost, $CommandText)
        $sshOutput = & ssh.exe @sshArgs 2>&1
        if ($LASTEXITCODE -ne 0) {
            throw "Thunderbolt peer SSH connection failed for '$User@$TargetHost': $($sshOutput -join ' ')"
        }
        return [PSCustomObject]@{
            Target         = $TargetHost
            PeerHostname   = $PeerName
            Protocol       = 'SSH'
            Authentication = "SSHKey($User)"
            Result         = ($sshOutput -join [Environment]::NewLine).Trim()
        }
    }

    if (-not $Credential -and $Password) {
        $Credential = [System.Management.Automation.PSCredential]::new(
            "MicrosoftAccount\$MicrosoftAccountEmail",
            $Password
        )
    }

    $target = $null
    $peerHostname = $null
    $localBindIp = $null
    $preferSshFromDiscovery = $false

    if ($ComputerName) {
        $target = $ComputerName
        $peerHostname = $ComputerName
        $peerId = $null
        if (Get-Command Get-ThunderboltPeerIdentity -ErrorAction SilentlyContinue) {
            $peerId = Get-ThunderboltPeerIdentity -ComputerName $ComputerName
            $peerHostname = $peerId.ComputerName
            if (-not $PSBoundParameters.ContainsKey('SshUser') -and $peerId.SshUser) {
                $SshUser = $peerId.SshUser
            }
            if ($Protocol -eq 'Auto' -and $peerId.OsFamily -eq 'linux') {
                $preferSshFromDiscovery = $true
            }
        }
        $dnsResolved = $false
        try {
            $addrs = @([System.Net.Dns]::GetHostAddresses($ComputerName))
            if ($addrs.Count -gt 0) { $dnsResolved = $true }
        } catch { }
        if (-not $dnsResolved -and $peerId) {
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
                ($Protocol -eq 'WinRM' -and $_.Candidate.WsManReachable -eq $true) -or
                ($Protocol -eq 'Auto' -and ($_.Candidate.WsManReachable -eq $true -or $_.Candidate.SshReachable -eq $true))
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
        if ($candidate.PSObject.Properties['PeerHostname'] -and $candidate.PeerHostname) {
            $peerHostname = $candidate.PeerHostname
            if (-not $PSBoundParameters.ContainsKey('SshUser') -and (Get-Command Get-ThunderboltPeerIdentity -ErrorAction SilentlyContinue)) {
                try {
                    $resolvedPeerId = Get-ThunderboltPeerIdentity -ComputerName $peerHostname
                    if ($resolvedPeerId.SshUser) { $SshUser = $resolvedPeerId.SshUser }
                } catch { }
            }
        }
        if ($Protocol -eq 'Auto' -and $candidate.WsManReachable -ne $true -and $candidate.SshReachable -eq $true) {
            $preferSshFromDiscovery = $true
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

    if ($Protocol -eq 'SSH' -or ($Protocol -eq 'Auto' -and $preferSshFromDiscovery)) {
        return Invoke-SshPeerConnection -TargetHost $target -PeerName $peerHostname -User $SshUser -LocalBind $localBindIp -CommandText $SshCommand
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
        if ($Protocol -eq 'Auto' -and -not $ReturnSession) {
            return Invoke-SshPeerConnection -TargetHost $target -PeerName $peerHostname -User $SshUser -LocalBind $localBindIp -CommandText $SshCommand
        }
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
