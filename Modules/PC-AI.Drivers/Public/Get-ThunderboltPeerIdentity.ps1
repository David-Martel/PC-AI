#Requires -Version 7.0
<#
.SYNOPSIS
    Computes deterministic USB4 / Thunderbolt peer identity, IPv6 Link-Local, IPv6 ULA,
    and commutative pairwise IPv4 /30 addresses for any dtm-* or arbitrary computer.

.DESCRIPTION
    Eliminates hardcoded machine assumptions by deriving collision-resistant IPv6
    Link-Local (fe80::d700:<h0>:<h1>:<h2>) and IPv6 Unique Local Addresses
    (fd00:7462:7032:d700:<h0>:<h1>:<h2>) algorithmically from the canonical hostname
    via SHA-256, while optionally overlaying human-readable short aliases (e.g.,
    fe80::d700:7 / fd00:7462:7032::7) and SSH/WinRM port metadata from
    Config/thunderbolt-peers.json or ~/.machine/thunderbolt-peers.json.

    When -PeerComputerName is supplied, also computes a commutative, deterministic
    pairwise IPv4 /30 point-to-point subnet from 172.31.240.0/20 so any two machines
    (e.g., dtm-carbon-two <-> dtm-super-nuc, or dtm-p1gen7 connected simultaneously
    to both dtm-p1gen9 and dtm-carbon-two) derive disjoint /30 subnets and non-colliding
    .1 / .2 host addresses independently without prior coordination.

.PARAMETER ComputerName
    Target or local computer name (defaults to $env:COMPUTERNAME). Works for any
    known dtm-* machine as well as arbitrary unknown computers.

.PARAMETER PeerComputerName
    Optional remote peer computer name used to derive a commutative pairwise /30 IPv4
    link subnet and deterministic .1 / .2 endpoint assignment.

.PARAMETER RegistryPath
    Optional path to thunderbolt-peers.json.

.OUTPUTS
    PSCustomObject representing the deterministic identity and addressing contract.
#>
function Get-ThunderboltPeerIdentity {
    [CmdletBinding()]
    [OutputType([PSCustomObject])]
    param(
        [Parameter(Position = 0, ValueFromPipeline, ValueFromPipelineByPropertyName)]
        [string]$ComputerName = $env:COMPUTERNAME,

        [Parameter()]
        [string]$PeerComputerName,

        [Parameter()]
        [string]$RegistryPath
    )

    begin {
        Set-StrictMode -Version Latest
        $ErrorActionPreference = 'Stop'

        function ConvertTo-CanonicalPeerName {
            param(
                [string]$RawName,
                [object[]]$KnownPeers
            )
            if ([string]::IsNullOrWhiteSpace($RawName)) {
                $RawName = $env:COMPUTERNAME
            }
            $cleaned = $RawName.Trim().ToLowerInvariant()
            # Strip common domain suffixes (.local, .radius.dtmventures.com, .dtmventures.com, -tb)
            $cleaned = $cleaned -replace '\.(local|lan|radius\.dtmventures\.com|dtmventures\.com)$', ''
            $cleaned = $cleaned -replace '-tb$', ''

            foreach ($peer in @($KnownPeers)) {
                if ($null -eq $peer -or -not $peer.PSObject.Properties['name']) { continue }
                if ([string]$peer.name -eq $cleaned) {
                    return [string]$peer.name
                }
                if ($peer.PSObject.Properties['aliases']) {
                    foreach ($alias in @($peer.aliases)) {
                        $normAlias = ([string]$alias).Trim().ToLowerInvariant() -replace '\.(local|lan|radius\.dtmventures\.com|dtmventures\.com)$', ''
                        if ($normAlias -eq $cleaned) {
                            return [string]$peer.name
                        }
                    }
                }
            }
            return $cleaned
        }

        function Get-DeterministicNodeHextets {
            param([Parameter(Mandatory)][string]$CanonicalName)

            $bytes = [System.Text.Encoding]::UTF8.GetBytes("pc-ai:usb4-peer:v1:$CanonicalName")
            $hash = [System.Security.Cryptography.SHA256]::HashData($bytes)
            $h0 = ([uint16]$hash[0] -shl 8) -bor [uint16]$hash[1]
            $h1 = ([uint16]$hash[2] -shl 8) -bor [uint16]$hash[3]
            $h2 = ([uint16]$hash[4] -shl 8) -bor [uint16]$hash[5]
            if ($h0 -eq 0) { $h0 = 1 }
            if ($h1 -eq 0) { $h1 = 1 }
            if ($h2 -eq 0) { $h2 = 1 }
            return ('{0:x4}:{1:x4}:{2:x4}' -f $h0, $h1, $h2)
        }

        function Get-CommutativePairwiseIPv4 {
            param(
                [Parameter(Mandatory)][string]$LocalCanonical,
                [Parameter(Mandatory)][string]$PeerCanonical
            )
            if ($LocalCanonical -eq $PeerCanonical) {
                throw "LocalComputer ('$LocalCanonical') and PeerComputer ('$PeerCanonical') must be distinct hosts to derive a pairwise /30 link."
            }

            $sorted = @($LocalCanonical, $PeerCanonical) | Sort-Object
            $low = $sorted[0]
            $high = $sorted[1]
            $pairKey = "pc-ai:usb4-ipv4-pair:v1:$low|$high"
            $hash = [System.Security.Cryptography.SHA256]::HashData([System.Text.Encoding]::UTF8.GetBytes($pairKey))

            # Map into 172.31.240.0/20 (16 /24 blocks: 172.31.240.0 .. 172.31.255.252 -> 1024 /30 subnets)
            $rawSlot = (([uint32]$hash[0] -shl 8) -bor [uint32]$hash[1]) % 1022 + 1
            $baseIpInt = ([uint32]172 -shl 24) -bor ([uint32]31 -shl 16) -bor ([uint32]240 -shl 8)
            $subnetInt = $baseIpInt + ([uint32]$rawSlot * 4)

            $lowIpInt  = $subnetInt + 1
            $highIpInt = $subnetInt + 2

            function ConvertFrom-UInt32Ip([uint32]$Val) {
                return '{0}.{1}.{2}.{3}' -f (($Val -shr 24) -band 0xFF), (($Val -shr 16) -band 0xFF), (($Val -shr 8) -band 0xFF), ($Val -band 0xFF)
            }

            $subnetStr = ConvertFrom-UInt32Ip $subnetInt
            $lowIpStr  = ConvertFrom-UInt32Ip $lowIpInt
            $highIpStr = ConvertFrom-UInt32Ip $highIpInt

            $isLocalLow = ($LocalCanonical -ceq $low)
            return [PSCustomObject]@{
                PairKey          = "$low<->$high"
                SubnetPrefix     = "$subnetStr/30"
                PrefixLength     = 30
                LocalIPv4Address = if ($isLocalLow) { $lowIpStr } else { $highIpStr }
                PeerIPv4Address  = if ($isLocalLow) { $highIpStr } else { $lowIpStr }
            }
        }

        $candidatePaths = [System.Collections.Generic.List[string]]::new()
        if (-not [string]::IsNullOrWhiteSpace($RegistryPath)) {
            $candidatePaths.Add($RegistryPath)
        } else {
            $moduleRoot = Split-Path $PSScriptRoot -Parent
            $repoRoot   = Split-Path (Split-Path $moduleRoot -Parent) -Parent
            $candidatePaths.Add((Join-Path $repoRoot 'Config\thunderbolt-peers.json'))
            $candidatePaths.Add((Join-Path $moduleRoot 'Config\thunderbolt-peers.json'))
            $candidatePaths.Add((Join-Path $HOME '.machine\thunderbolt-peers.json'))
        }

        $registryData = $null
        $loadedRegistryPath = $null
        foreach ($p in $candidatePaths) {
            if ($p -and (Test-Path -LiteralPath $p)) {
                try {
                    $registryData = Get-Content -LiteralPath $p -Raw -Encoding UTF8 | ConvertFrom-Json
                    $loadedRegistryPath = $p
                    break
                } catch {
                    Write-Verbose "Failed to parse thunderbolt-peers.json at '$p': $($_.Exception.Message)"
                }
            }
        }

        $knownPeers = if ($registryData -and $registryData.PSObject.Properties['fleetPeers']) {
            @($registryData.fleetPeers)
        } elseif ($registryData -and $registryData.PSObject.Properties['peers']) {
            if ($registryData.peers -is [System.Collections.IEnumerable] -and $registryData.peers -isnot [string]) {
                @($registryData.peers)
            } else {
                @(
                    foreach ($prop in $registryData.peers.PSObject.Properties) {
                        $val = $prop.Value
                        if ($null -ne $val -and -not $val.PSObject.Properties['name']) {
                            $val | Add-Member -NotePropertyName 'name' -NotePropertyValue $prop.Name -Force
                        }
                        $val
                    }
                )
            }
        } else {
            @()
        }
        $ulaPrefix = if ($registryData -and $registryData.PSObject.Properties['ulaPrefix'] -and $registryData.ulaPrefix) {
            [string]$registryData.ulaPrefix
        } elseif ($registryData -and $registryData.PSObject.Properties['ipam'] -and $registryData.ipam.PSObject.Properties['ulaPrefix'] -and $registryData.ipam.ulaPrefix) {
            [string]$registryData.ipam.ulaPrefix
        } else {
            'fd00:7462:7032'
        }
    }

    process {
        $canonicalName = ConvertTo-CanonicalPeerName -RawName $ComputerName -KnownPeers $knownPeers
        $matchedPeer = $knownPeers | Where-Object { $_ -and $_.PSObject.Properties['name'] -and [string]$_.name -eq $canonicalName } | Select-Object -First 1

        $nodeHextets = Get-DeterministicNodeHextets -CanonicalName $canonicalName
        $deterministicLinkLocal = "fe80::d700:$nodeHextets"
        $deterministicUla = "${ulaPrefix}:d700::$nodeHextets"

        $shortLinkLocal = if ($matchedPeer -and $matchedPeer.PSObject.Properties['shortLinkLocal'] -and $matchedPeer.shortLinkLocal) {
            [string]$matchedPeer.shortLinkLocal
        } else {
            $null
        }
        $shortUla = if ($matchedPeer -and $matchedPeer.PSObject.Properties['shortUla'] -and $matchedPeer.shortUla) {
            [string]$matchedPeer.shortUla
        } else {
            $null
        }

        $primaryLinkLocal = if ($shortLinkLocal) { $shortLinkLocal } else { $deterministicLinkLocal }
        $primaryUla = if ($shortUla) { $shortUla } else { $deterministicUla }

        $allLinkLocal = @($deterministicLinkLocal)
        if ($shortLinkLocal -and $shortLinkLocal -notin $allLinkLocal) {
            $allLinkLocal = @($shortLinkLocal) + $allLinkLocal
        }

        $allUla = @($deterministicUla)
        if ($shortUla -and $shortUla -notin $allUla) {
            $allUla = @($shortUla) + $allUla
        }

        $pairwiseIpv4 = $null
        $canonicalPeer = $null
        if (-not [string]::IsNullOrWhiteSpace($PeerComputerName)) {
            $canonicalPeer = ConvertTo-CanonicalPeerName -RawName $PeerComputerName -KnownPeers $knownPeers
            $pairwiseIpv4 = Get-CommutativePairwiseIPv4 -LocalCanonical $canonicalName -PeerCanonical $canonicalPeer
        }

        [PSCustomObject]@{
            ComputerName           = $canonicalName
            InputName              = $ComputerName
            IsKnownFleetPeer       = ($null -ne $matchedPeer)
            NodeTokenHextets       = $nodeHextets
            DeterministicLinkLocal = $deterministicLinkLocal
            DeterministicUla       = $deterministicUla
            ShortLinkLocal         = $shortLinkLocal
            ShortUla               = $shortUla
            PrimaryLinkLocal       = $primaryLinkLocal
            PrimaryUla             = $primaryUla
            AllLinkLocalAddresses  = @($allLinkLocal)
            AllUlaAddresses        = @($allUla)
            TbHostAlias            = if ($matchedPeer -and $matchedPeer.PSObject.Properties['tbHostAlias'] -and $matchedPeer.tbHostAlias) { [string]$matchedPeer.tbHostAlias } else { "$canonicalName-tb" }
            SshPort                = if ($matchedPeer -and $matchedPeer.PSObject.Properties['sshPort'] -and $matchedPeer.sshPort) { [int]$matchedPeer.sshPort } else { 22 }
            SshUser                = if ($matchedPeer -and $matchedPeer.PSObject.Properties['sshUser'] -and $matchedPeer.sshUser) { [string]$matchedPeer.sshUser } else { 'david' }
            WinRmPort              = if ($matchedPeer -and $matchedPeer.PSObject.Properties['winRmPort'] -and $matchedPeer.winRmPort) { [int]$matchedPeer.winRmPort } else { 5985 }
            LanAddresses           = if ($matchedPeer -and $matchedPeer.PSObject.Properties['lanAddresses']) { @($matchedPeer.lanAddresses) } else { @() }
            DefaultShares          = if ($matchedPeer -and $matchedPeer.PSObject.Properties['defaultShares']) { @($matchedPeer.defaultShares) } else { @('C$') }
            PeerComputerName       = $canonicalPeer
            PairwiseIPv4           = $pairwiseIpv4
            RegistryPath           = $loadedRegistryPath
        }
    }
}
