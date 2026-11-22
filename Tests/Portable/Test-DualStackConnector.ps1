#Requires -Version 5.1
<#
.SYNOPSIS
Real-socket and cross-version checks for the dual-stack endpoint commands in PC-AI.Network.

.DESCRIPTION
No Pester: this script runs unchanged in Windows PowerShell 5.1 and PowerShell 7, so it is also the proof that the
module's 5.1-compatible code runs on both. It binds real TcpListener sockets on the loopback addresses, so it needs no
network, hardware or privilege. Providers are injected with synthetic data (2001:db8::/32, 192.0.2.0/24, fake MACs).

A check that cannot run says SKIP with its reason and is counted separately; a skip is never a pass. The exit code is
1 when any check fails.

Unit coverage with injected timing is in Tests/Unit/PC-AI.Network.Tests.ps1; design in docs/networking/dual-stack.md.

.EXAMPLE
pwsh -NoProfile -File .\Tests\Portable\Test-DualStackConnector.ps1
powershell.exe -NoProfile -File .\Tests\Portable\Test-DualStackConnector.ps1
#>
[CmdletBinding()]
param()
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$modulePath = Join-Path (Join-Path (Join-Path $PSScriptRoot '..') '..') 'Modules\PC-AI.Network\PC-AI.Network.psd1'
Import-Module -Name (Resolve-Path -LiteralPath $modulePath).Path -Force
$script:Passed = 0
$script:Failed = 0
$script:Skipped = 0

function Assert-Dual {
    param([string]$Name, [bool]$Condition, [string]$Detail = '')
    if ($Condition) { $script:Passed++; Write-Output "PASS: $Name" }
    else { $script:Failed++; Write-Output "FAIL: $Name $Detail" }
}
function Write-DualSkip {
    param([string]$Name, [string]$Reason)
    $script:Skipped++
    Write-Output "SKIP: $Name ($Reason)"
}
function New-Usb4Adapter {
    param([int]$Index, [string]$Alias)
    [pscustomobject]@{ Name = $Alias; InterfaceAlias = $Alias; InterfaceDescription = 'USB4 P2P Network Adapter'; ifIndex = $Index; Status = 'Up'; MacAddress = '02-00-5E-00-53-AA' }
}
$noRegistry = { param($Path) $null }
$noNeighbors = { param($InterfaceIndex) @() }

Write-Output ("Running on PowerShell {0} ({1})" -f $PSVersionTable.PSVersion, $PSVersionTable.PSEdition)

# Selector: Eui64KnownVector
# Protects: the modified EUI-64 link-local derivation, with the zone attached from the live ifIndex.
# Detects: a missing U/L bit flip (the address would read fe80::11:22ff:fe33:4455), a missing ff:fe, a dropped zone.
# Needs: nothing live; one injected USB4 adapter.
# Breadcrumb: ConvertTo-PcaiEui64LinkLocal in Modules/PC-AI.Network/Private/Endpoint-Helpers.ps1.
$adapter = New-Usb4Adapter -Index 25 -Alias 'Ethernet 4'
$eui = @(Get-PcaiEndpointCandidate -MacAddress '00-11-22-33-44-55' -Provider Eui64 -InterfaceSource { @($adapter) }.GetNewClosure() -RegistrySource $noRegistry -NeighborSource $noNeighbors)
Assert-Dual 'EUI-64 of 00-11-22-33-44-55 is fe80::211:22ff:fe33:4455%25' ($eui.Count -eq 1 -and $eui[0].Address -ceq 'fe80::211:22ff:fe33:4455%25') ("got " + (($eui | ForEach-Object { $_.Address }) -join ','))

# Selector: ReplugFollowsInterface
# Protects: topology change. A peer that moved from ifIndex 19 to 25 is found on 25 with no trace of 19.
# Detects: an ifIndex remembered between calls.
# Needs: an injected adapter table that changes between calls.
# Breadcrumb: Select-PcaiScopeInterface; no script-scope state in Get-PcaiEndpointCandidate.
$state = @{ Adapters = @((New-Usb4Adapter -Index 19 -Alias 'Ethernet 18')) }
$source = { $state.Adapters }.GetNewClosure()
$before = @(Get-PcaiEndpointCandidate -MacAddress '00-11-22-33-44-55' -Provider Eui64 -InterfaceSource $source -RegistrySource $noRegistry -NeighborSource $noNeighbors)
$state.Adapters = @((New-Usb4Adapter -Index 25 -Alias 'Ethernet 4'))
$after = @(Get-PcaiEndpointCandidate -MacAddress '00-11-22-33-44-55' -Provider Eui64 -InterfaceSource $source -RegistrySource $noRegistry -NeighborSource $noNeighbors)
Assert-Dual 'replug 19 -> 25 changes the zone' ($before[0].Address -ceq 'fe80::211:22ff:fe33:4455%19' -and $after[0].Address -ceq 'fe80::211:22ff:fe33:4455%25')

# Loopback sockets. TcpListener has no Dispose in .NET Framework, so Stop() in finally.
function Test-Ipv6Loopback {
    if (-not [System.Net.Sockets.Socket]::OSSupportsIPv6) { return 'the OS reports no IPv6 support' }
    try {
        $probe = New-Object System.Net.Sockets.TcpListener([System.Net.IPAddress]::IPv6Loopback, 0)
        $probe.Start(); $probe.Stop()
        return $null
    }
    catch { return "cannot bind [::1]: $($_.Exception.Message)" }
}
function Start-Loopback {
    # A port free on BOTH families is needed so the two listeners share one number.
    for ($i = 0; $i -lt 20; $i++) {
        $v4 = New-Object System.Net.Sockets.TcpListener([System.Net.IPAddress]::Loopback, 0)
        $v4.Start()
        $port = ([System.Net.IPEndPoint]$v4.LocalEndpoint).Port
        $v6 = $null
        try {
            $v6 = New-Object System.Net.Sockets.TcpListener([System.Net.IPAddress]::IPv6Loopback, $port)
            $v6.Start()
            return [pscustomobject]@{ Port = $port; V4 = $v4; V6 = $v6 }
        }
        catch { $v4.Stop() }
    }
    return $null
}
function New-LoopbackCandidate { param([string]$Address) [pscustomobject]@{ Address = $Address; Source = 'Dns' } }

$v6Problem = Test-Ipv6Loopback

# Selector: LoopbackPrefersIPv6
# Protects: with both families listening, the IPv6 candidate wins even when IPv4 is listed first.
# Detects: IPv4-first ordering; a connector that cannot dial [::1]; a leaked socket (the winner is closed again).
# Needs: real TcpListeners on ::1 and 127.0.0.1; skipped, with the reason, when IPv6 loopback is unavailable.
# Breadcrumb: Get-PcaiConnectionOrder; New-PcaiTcpConnection.
if ($v6Problem) { Write-DualSkip 'both families listening: IPv6 wins' $v6Problem }
else {
    $both = Start-Loopback
    if (-not $both) { Write-DualSkip 'both families listening: IPv6 wins' 'no port free on both loopback families after 20 tries' }
    else {
        try {
            $r = Connect-PcaiEndpoint -Candidate @((New-LoopbackCandidate '127.0.0.1'), (New-LoopbackCandidate '::1')) -Port $both.Port -TimeoutMs 4000
            Assert-Dual 'both families listening: connected over IPv6 loopback' ($r.Connected -and $r.Winner.Family -eq 'InterNetworkV6' -and $r.Winner.Class -eq 'Loopback') ("outcomes " + (($r.Attempts | ForEach-Object { "$($_.Address):$($_.Outcome)" }) -join ','))
            Assert-Dual 'winner was closed again (no -PassThru)' ($null -eq $r.Client)
        }
        finally { $both.V4.Stop(); $both.V6.Stop() }
    }
}

# Selector: LoopbackFallsBackToIPv4
# Protects: a dead IPv6 address does not stall the connect: with only IPv4 listening, IPv4 wins within the attempt delay
# plus a margin, and the dead attempt is recorded with its outcome.
# Detects: serial attempts, an ignored deadline, a missing PassThru client.
# Needs: a real IPv4 listener and an unused [::1] port; skipped with a reason when IPv6 loopback is unavailable.
# Breadcrumb: Connect-PcaiEndpoint race loop.
if ($v6Problem) { Write-DualSkip 'IPv6 dead, IPv4 listening: IPv4 wins' $v6Problem }
else {
    $only4 = New-Object System.Net.Sockets.TcpListener([System.Net.IPAddress]::Loopback, 0)
    $only4.Start()
    try {
        $port = ([System.Net.IPEndPoint]$only4.LocalEndpoint).Port
        $unused6 = $null
        try { $unused6 = New-Object System.Net.Sockets.TcpListener([System.Net.IPAddress]::IPv6Loopback, $port); $unused6.Start(); $unused6.Stop(); $unused6 = 'free' }
        catch { $unused6 = $null }
        if (-not $unused6) { Write-DualSkip 'IPv6 dead, IPv4 listening: IPv4 wins' "port $port is taken on [::1]" }
        else {
            $r = Connect-PcaiEndpoint -Candidate @((New-LoopbackCandidate '::1'), (New-LoopbackCandidate '127.0.0.1')) -Port $port -AttemptDelayMs 250 -TimeoutMs 5000 -PassThru
            try {
                Assert-Dual 'IPv6 dead, IPv4 listening: IPv4 wins' ($r.Connected -and $r.Winner.Family -eq 'InterNetwork') ("outcomes " + (($r.Attempts | ForEach-Object { "$($_.Address):$($_.Outcome)" }) -join ','))
                Assert-Dual 'dead IPv6 attempt is recorded and not Connected' ($r.Attempts.Count -eq 2 -and $r.Attempts[0].Outcome -ne 'Connected')
                Assert-Dual 'fallback finished well inside the 5 s deadline' ($r.ElapsedMs -lt 3500) "took $($r.ElapsedMs) ms"
                Assert-Dual '-PassThru returns an open connected client' ($null -ne $r.Client -and $r.Client.Connected)
            }
            finally { if ($r.Client) { $r.Client.Dispose() } }
        }
    }
    finally { $only4.Stop() }
}

# Selector: AllFailIsData
# Protects: with nothing listening anywhere, the result says Connected=false with every attempt and does not throw.
# Detects: a throw on plain failure, a missing attempt, a success claimed without a winner.
# Needs: two loopback ports nothing listens on (found by binding and releasing); IPv4 only so no IPv6 requirement.
# Breadcrumb: Connect-PcaiEndpoint result assembly.
$free = New-Object System.Net.Sockets.TcpListener([System.Net.IPAddress]::Loopback, 0)
$free.Start(); $freePort = ([System.Net.IPEndPoint]$free.LocalEndpoint).Port; $free.Stop()
$none = Connect-PcaiEndpoint -Candidate @((New-LoopbackCandidate '127.0.0.1')) -Port $freePort -TimeoutMs 4000
Assert-Dual 'nothing listening: Connected=false, one attempt, no winner' (-not $none.Connected -and $none.Attempts.Count -eq 1 -and $null -eq $none.Winner)

Write-Output ("Summary: {0} passed, {1} failed, {2} skipped" -f $script:Passed, $script:Failed, $script:Skipped)
if ($script:Failed -gt 0) { exit 1 }
exit 0
