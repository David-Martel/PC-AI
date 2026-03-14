#Requires -Version 7.0
function Find-ThunderboltPeer {
    [CmdletBinding()]
    [OutputType([PSCustomObject])]
    param(
        [Parameter()]
        [string[]]$ComputerNameCandidates = @(
            'DESKTOP-86FC90B',
            'dtm-supernuc',
            'dtm-work',
            'dtm-work-mobo'
        ),

        [Parameter()]
        [int]$WinRmPort = 5985,

        [Parameter()]
        [ValidateRange(250, 10000)]
        [int]$TcpTimeoutMs = 1500,

        [Parameter()]
        [switch]$IncludeRaw
    )

    Set-StrictMode -Version Latest
    $ErrorActionPreference = 'Stop'

    function Invoke-WindowsPowerShellJson {
        param(
            [Parameter(Mandatory)]
            [string]$Script
        )

        $encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($Script))
        $output = & powershell.exe -NoLogo -NoProfile -EncodedCommand $encoded 2>$null
        if ($LASTEXITCODE -ne 0) {
            throw 'Windows PowerShell Thunderbolt discovery probe failed.'
        }

        $text = ($output -join [Environment]::NewLine).Trim()
        if ([string]::IsNullOrWhiteSpace($text)) {
            return $null
        }

        return $text | ConvertFrom-Json -Depth 12
    }

    function ConvertTo-PowerShellArrayLiteral {
        param(
            [Parameter(Mandatory)]
            [string[]]$Value
        )

        if (-not $Value -or $Value.Count -eq 0) {
            return '@()'
        }

        $quoted = foreach ($entry in $Value) {
            "'{0}'" -f $entry.Replace("'", "''")
        }

        return '@({0})' -f ($quoted -join ', ')
    }

    $candidateLiteral = ConvertTo-PowerShellArrayLiteral -Value $ComputerNameCandidates
    $portLiteral = [string]$WinRmPort
    $timeoutLiteral = [string]$TcpTimeoutMs
    $includeRawLiteral = if ($IncludeRaw) { '$true' } else { '$false' }

    return Invoke-WindowsPowerShellJson -Script @"
`$computerNameCandidates = $candidateLiteral
`$winRmPort = $portLiteral
`$tcpTimeoutMs = $timeoutLiteral
`$includeRaw = $includeRawLiteral

`$adapters = @(Get-NetAdapter -IncludeHidden -ErrorAction SilentlyContinue |
    Where-Object {
        `$_.Status -ne 'Disabled' -and (
            `$_.InterfaceDescription -match 'USB4' -or
            `$_.InterfaceDescription -match 'Thunderbolt' -or
            `$_.InterfaceDescription -match 'P2P'
        )
    } |
    Sort-Object InterfaceMetric, ifIndex |
    Select-Object Name, InterfaceDescription, Status, MacAddress, LinkSpeed, InterfaceMetric, ifIndex)

`$interfaceIndexes = @(`$adapters | ForEach-Object ifIndex)
`$localAddresses = @()
`$neighbors = @()
if (`$interfaceIndexes.Count -gt 0) {
    `$localAddresses = @(Get-NetIPAddress -InterfaceIndex `$interfaceIndexes -ErrorAction SilentlyContinue |
        Where-Object { `$_.AddressFamily -in @('IPv4', 'IPv6') } |
        Select-Object InterfaceIndex, InterfaceAlias, AddressFamily, IPAddress, PrefixLength)

    `$neighbors = @(Get-NetNeighbor -InterfaceIndex `$interfaceIndexes -ErrorAction SilentlyContinue |
        Where-Object {
            `$_.IPAddress -and
            `$_.State -notin @(0, '0', 'Unreachable', 'Invalid') -and
            `$_.IPAddress -notin @('255.255.255.255', '169.254.255.255') -and
            `$_.IPAddress -notmatch '^(22[4-9]|23\d)\.' -and
            `$_.IPAddress -notlike 'ff*' -and
            `$_.LinkLayerAddress -and
            `$_.LinkLayerAddress -notin @('00-00-00-00-00-00', '00:00:00:00:00:00', 'FF-FF-FF-FF-FF-FF', 'ff-ff-ff-ff-ff-ff', 'ff:ff:ff:ff:ff:ff') -and
            `$_.IPAddress -ne 'ff02::1'
        } |
        Select-Object InterfaceIndex, IPAddress, LinkLayerAddress, State)
}

`$candidateTargets = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
foreach (`$name in `$computerNameCandidates) {
    if (-not [string]::IsNullOrWhiteSpace(`$name)) {
        [void]`$candidateTargets.Add(`$name.Trim())
    }
}
foreach (`$neighbor in `$neighbors) {
    if (-not [string]::IsNullOrWhiteSpace(`$neighbor.IPAddress)) {
        [void]`$candidateTargets.Add(`$neighbor.IPAddress)
    }
}

`$nameResolution = foreach (`$target in `$candidateTargets) {
    `$resolved = @(Resolve-DnsName -Name `$target -ErrorAction SilentlyContinue)
    [pscustomobject]@{
        Target          = `$target
        Resolved        = `$resolved.Count -gt 0
        ResolvedAddress = @(`$resolved | Where-Object IPAddress | Select-Object -ExpandProperty IPAddress)
    }
}

`$winRmCandidates = foreach (`$target in `$candidateTargets) {
    `$tcpReachable = `$false
    `$client = New-Object System.Net.Sockets.TcpClient
    try {
        `$async = `$client.BeginConnect(`$target, `$winRmPort, `$null, `$null)
        if (`$async.AsyncWaitHandle.WaitOne(`$tcpTimeoutMs, `$false)) {
            try {
                `$client.EndConnect(`$async)
                `$tcpReachable = `$client.Connected
            } catch {
                `$tcpReachable = `$false
            }
        }
    } catch {
        `$tcpReachable = `$false
    } finally {
        `$client.Close()
    }
    [pscustomobject]@{
        Target       = `$target
        WinRmPort    = `$winRmPort
        TcpTimeoutMs = `$tcpTimeoutMs
        TcpReachable = [bool]`$tcpReachable
    }
}

`$raw = `$null
if (`$includeRaw) {
    `$raw = [pscustomobject]@{
        IpConfig = (ipconfig /all | Out-String).Trim()
        Arp      = (arp -a | Out-String).Trim()
        Route    = (route print | Out-String).Trim()
    }
}

[pscustomobject]@{
    Timestamp       = Get-Date
    LocalComputer   = `$env:COMPUTERNAME
    AdapterCount    = `$adapters.Count
    Adapters        = @(`$adapters)
    LocalAddresses  = @(`$localAddresses)
    Neighbors       = @(`$neighbors)
    NameResolution  = @(`$nameResolution)
    WinRmCandidates = @(`$winRmCandidates)
    Raw             = `$raw
} | ConvertTo-Json -Depth 12
"@
}
