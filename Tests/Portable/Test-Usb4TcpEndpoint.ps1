#Requires -Version 7.0
$ErrorActionPreference = 'Stop'
$source = Get-Content -LiteralPath (Join-Path $PSScriptRoot '../Hardware/Invoke-PcaiUsb4TcpEndpoint.ps1') -Raw
# On Linux, replace only the platform predicate to exercise the Windows provider fixtures.
if (-not $IsWindows) { $source = $source.Replace('$IsWindows', '$fixtureWindowsProvider') }
$fixtureWindowsProvider = $true
# Use a unique fixture mutex so tests cannot take the production listener lock.
$ownedMutexLiteral = "'Global\PC-AI USB4 TCP HIL Firewall'"
if (-not $source.Contains($ownedMutexLiteral)) { throw 'The endpoint listener mutex binding changed.' }
$fixtureMutexName = 'PC-AI USB4 TCP HIL Fixture ' + [guid]::NewGuid().ToString('N')
$source = $source.Replace($ownedMutexLiteral, '$fixtureMutexName')
$endpoint = [scriptblock]::Create($source)
$fixturePath = 'C:/fixture/iperf3.exe'
$fixtureGuid = [guid]'0413a23a-e1a7-4cbe-947c-860a65a936bd'
$fixtureBinary = New-TemporaryFile
$fixtureRunner = @'
function Invoke-VigilBoundedProcess {
    param($FilePath, $Arguments, $TimeoutSeconds)
    if ($FilePath -ne $fixtureBinary.FullName) { throw 'Unexpected executable.' }
    if ($Arguments -contains '--version') {
        return @{ ExitCode = 0; Stdout = 'iperf 3.22'; Stderr = '' }
    }
    $expectedTimeout = if ($Request.role -eq 'listen') { 48 } else { 28 }
    if ($TimeoutSeconds -ne $expectedTimeout -or
        ($Arguments[0..5] -join ' ') -ne '-6 -B fe80::1%25 -p 42000 --json') {
        throw 'The process was not bounded and bound to the selected USB4 source.'
    }
    $expectedArguments = if ($Request.role -eq 'listen') {
        '-s -1'
    } else {
        '-c fe80::2%25 --connect-timeout 4000 --bidir -P 2 -t 3'
    }
    if (($Arguments[6..($Arguments.Count - 1)] -join ' ') -ne $expectedArguments) {
        throw 'The process did not request the expected one-shot server or explicit duplex client.'
    }
    $fixtureState.Transfers++
    if ($fixtureState.Timeout) { throw 'Fixture owned-job timeout.' }
    @{ ExitCode = 0; Stdout = '{"end":{}}'; Stderr = '' }
}
'@

function Get-Item {
    param([string]$LiteralPath)
    if ($LiteralPath -ne $fixturePath) { throw 'Unexpected executable lookup.' }
    $fixtureBinary
}
function Get-NetAdapter {
    param([switch]$IncludeHidden)
    [pscustomobject]@{
        InterfaceGuid = $fixtureGuid; InterfaceIndex = 25; Status = 'Up'
        PnPDeviceID = 'USB4\D&PROT_USB4NET\P'; MacAddress = '02-00-00-00-00-01'; Name = 'USB4 Fixture'
    }
}
function Get-NetIPAddress {
    param($InterfaceIndex, $AddressFamily)
    if ($InterfaceIndex -ne 25 -or $AddressFamily -ne 'IPv6') { throw 'Wrong source inventory scope.' }
    [pscustomobject]@{ AddressState = 'Preferred'; IPAddress = 'fe80::1%25' }
}
function Get-NetNeighbor {
    param($InterfaceIndex, $AddressFamily)
    if ($InterfaceIndex -ne 25 -or $AddressFamily -ne 'IPv6') { throw 'Wrong neighbor inventory scope.' }
    [pscustomobject]@{ State = 'Reachable'; IPAddress = $fixtureState.NeighborIp; LinkLayerAddress = $fixtureState.NeighborMac }
}
function Find-NetRoute {
    param($LocalIPAddress, $RemoteIPAddress, $InterfaceIndex)
    if ($LocalIPAddress -ne 'fe80::1%25' -or $RemoteIPAddress -ne 'fe80::2%25' -or $InterfaceIndex -ne 25) {
        throw 'Wrong route lookup scope.'
    }
    [pscustomobject]@{ InterfaceIndex = 25; IPAddress = $LocalIPAddress }
    [pscustomobject]@{ InterfaceIndex = 25; DestinationPrefix = $fixtureState.DestinationPrefix; NextHop = $fixtureState.NextHop }
}
function Get-NetIPInterface {
    param($InterfaceIndex, $AddressFamily)
    [pscustomobject]@{ NlMtu = 62000 }
}
function Get-NetAdapterStatistics {
    param($Name)
    if ($Name -ne 'USB4 Fixture') { throw 'Wrong counter interface.' }
    $fixtureState.Counter += 100
    [pscustomobject]@{
        Name = $Name
        ReceivedBytes = if ($fixtureState.FlatRx) { [uint64]100 } else { [uint64]($fixtureState.Counter * 2) }
        SentBytes = if ($fixtureState.FlatTx) { [uint64]100 } else { [uint64]$fixtureState.Counter }
    }
}
function New-NetFirewallRule {
    param($PolicyStore, $Group, $Name, $DisplayName, $Direction, $Action, $Protocol, $LocalPort, $LocalAddress,
        $RemoteAddress, $InterfaceAlias, $Program, $Profile)
    if ($PolicyStore -ne 'PersistentStore' -or $Group -ne 'PC-AI USB4 TCP HIL' -or
        $Name -notmatch '^PC-AI USB4 TCP HIL [a-f0-9]{32}$' -or
        $Direction -ne 'Inbound' -or $Action -ne 'Allow' -or $Protocol -ne 'TCP' -or $LocalPort -ne 42000 -or
        $LocalAddress -ne 'fe80::1' -or $RemoteAddress -ne 'fe80::2' -or $InterfaceAlias -ne 'USB4 Fixture' -or
        $Program -ne $fixtureBinary.FullName) { throw 'Unscoped fixture firewall rule.' }
    $fixtureState.Rules[$Name] = $Group
    $fixtureState.Added++
    $fixtureState.LastRule = $Name
}
function Get-NetFirewallRule {
    param($PolicyStore, $Group, $Name)
    if ($PolicyStore -ne 'PersistentStore') { throw 'Wrong firewall policy store.' }
    foreach ($entry in $fixtureState.Rules.GetEnumerator()) {
        if (($Group -and $entry.Value -eq $Group) -or ($Name -and $entry.Key -like $Name)) {
            [pscustomobject]@{ Name = $entry.Key }
        }
    }
}
function Remove-NetFirewallRule {
    param($PolicyStore, $Name)
    if ($PolicyStore -ne 'PersistentStore' -or $Name -eq 'unrelated' -or -not $fixtureState.Rules.ContainsKey($Name)) {
        throw 'Removal attempted outside the exact fixture rule.'
    }
    if ($fixtureState.CleanupFails) { throw 'Fixture firewall provider denied removal.' }
    $fixtureState.Rules.Remove($Name)
    $fixtureState.Removed++
}

function Invoke-FixtureEndpoint([string]$Role, [bool]$Firewall = $false) {
    $request = @{
        role = $Role; tcp_port = 42000; seconds = 3; streams = 2; allow_temporary_firewall = $Firewall
        local = @{ host = [Environment]::MachineName; adapter_guid = $fixtureGuid; mac = '02-00-00-00-00-01'; ip = 'fe80::1'; iperf_path = $fixturePath }
        peer = @{ host = 'fixture-peer'; mac = '02-00-00-00-00-02'; ip = 'fe80::2' }
    }
    & $endpoint -Request $request -ProcessRunnerSource $fixtureRunner | ConvertFrom-Json
}

# Protects: USB4 source binding and underlay proof for the benchmark client.
# Detects: wrong routes, missing duplex options, incorrect client deadline, or unattributed counters.
# Needs: PowerShell 7, synthetic provider data and a temporary file for binary hashing.
# Breadcrumb: Tests/Hardware/Invoke-PcaiUsb4TcpEndpoint.ps1::Assert-DirectRoute, client branch, Get-Usb4Counters.
function Test-ClientUsesScopedOnLinkRouteAndCounters {
    $result = Invoke-FixtureEndpoint -Role client -Firewall $true
    if ($result.index -ne 25 -or $result.source -ne 'fe80::1%25' -or $result.peer -ne 'fe80::2%25' -or
        $result.rx_delta -ne 200 -or $result.tx_delta -ne 100 -or $result.binary_sha256 -notmatch '^[0-9a-f]{64}$' -or
        $fixtureState.Added -ne 0) { throw 'Client did not provide scoped evidence or attempted a firewall change.' }
    $fixtureState.DestinationPrefix = 'fe80::2/128'
    Invoke-FixtureEndpoint -Role client | Out-Null
    foreach ($route in @(@('fe80::/64', 'fe80::3'), @('fe80::/10', '::'), @('fe80::3/128', '::'))) {
        $fixtureState.DestinationPrefix, $fixtureState.NextHop = $route
        try {
            Invoke-FixtureEndpoint -Role client | Out-Null
            throw 'The endpoint accepted an invalid direct route.'
        } catch {
            if ($_.Exception.Message -notlike '*on-link peer*' -or $fixtureState.Transfers -ne 2) { throw }
        }
    }
}

# Protects: peer identity on the enrolled USB4 interface.
# Detects: mixing a valid IP with a wrong MAC, or a valid MAC with a wrong IP.
# Needs: PowerShell 7 and synthetic neighbor rows.
# Breadcrumb: Tests/Hardware/Invoke-PcaiUsb4TcpEndpoint.ps1::peer IP/MAC neighbor admission.
function Test-ClientRejectsWrongNeighborIpOrMac {
    foreach ($neighbor in @(@('fe80::2%25', '02-00-00-00-00-03'), @('fe80::3%25', '02-00-00-00-00-02'))) {
        $fixtureState.NeighborIp, $fixtureState.NeighborMac = $neighbor
        try {
            Invoke-FixtureEndpoint -Role client | Out-Null
            throw 'The endpoint accepted an unexpected neighbor.'
        } catch {
            if ($_.Exception.Message -notlike '*expected USB4 peer IP/MAC neighbor*' -or $fixtureState.Transfers -ne 0) { throw }
        }
    }
}

# Protects: proof of transfer in both directions on the chosen USB4 adapter.
# Detects: receipt emission while either interface byte counter remains flat.
# Needs: PowerShell 7, synthetic byte counters and a stub runner.
# Breadcrumb: Tests/Hardware/Invoke-PcaiUsb4TcpEndpoint.ps1::post-transfer counter comparison.
function Test-ClientRejectsFlatReceiveOrTransmitCounter {
    foreach ($flatCounter in @('FlatRx', 'FlatTx')) {
        $fixtureState[$flatCounter] = $true
        try {
            Invoke-FixtureEndpoint -Role client | Out-Null
            throw 'The endpoint accepted a flat USB4 counter.'
        } catch {
            if ($_.Exception.Message -notlike '*bidirectional counter growth*') { throw }
        }
        $fixtureState[$flatCounter] = $false
    }
    if ($fixtureState.Transfers -ne 2) { throw 'Both transfer attempts must reach the counter proof.' }
}

# Protects: the successful one-shot listener and the lifetime of its exact temporary allow rule.
# Detects: an incorrect listener deadline, a broadened rule, or cleanup outside the owned rule.
# Needs: PowerShell 7, synthetic Windows route/firewall providers and a stub runner.
# Breadcrumb: Tests/Hardware/Invoke-PcaiUsb4TcpEndpoint.ps1::listen branch and firewall finally block.
function Test-ListenerRemovesScopedFirewallRuleAfterSuccess {
    $result = Invoke-FixtureEndpoint -Role listen -Firewall $true
    if ($result.role -ne 'listen' -or $fixtureState.Transfers -ne 1 -or $fixtureState.Added -ne 1 -or
        $fixtureState.Removed -ne 1 -or $fixtureState.Rules.Count -ne 1 -or -not $fixtureState.Rules.ContainsKey('unrelated')) {
        throw 'Successful listener did not remove only its own rule.'
    }
}

# Protects: firewall custody when the bounded job fails to finish.
# Detects: a leaked rule, hidden timeout, or deletion of unrelated rules on failure.
# Needs: PowerShell 7, synthetic firewall provider and a simulated owned-job timeout.
# Breadcrumb: Tests/Hardware/Invoke-PcaiUsb4TcpEndpoint.ps1::Invoke-VigilBoundedProcess call and firewall finally block.
function Test-ListenerPreservesTimeoutAndRemovesScopedFirewallRule {
    $fixtureState.Timeout = $true
    try {
        Invoke-FixtureEndpoint -Role listen -Firewall $true | Out-Null
        throw 'The listener hid the owned-job timeout.'
    } catch {
        if ($_.Exception.Message -ne 'Fixture owned-job timeout.' -or $fixtureState.Added -ne 1 -or
            $fixtureState.Removed -ne 1 -or $fixtureState.Rules.Count -ne 1 -or -not $fixtureState.Rules.ContainsKey('unrelated')) { throw }
    }
}

# Protects: custody of interrupted firewall sessions before any new listener starts.
# Detects: a stale grouped or legacy rule overlooked or silently removed.
# Needs: PowerShell 7, a fixture mutex and synthetic PersistentStore rules.
# Breadcrumb: Tests/Hardware/Invoke-PcaiUsb4TcpEndpoint.ps1::Assert-NoAbandonedFirewallRules.
function Test-ListenerNamesAndPreservesAbandonedFirewallRules {
    $priorRules = @{ 'fixture-old-group-member' = 'PC-AI USB4 TCP HIL'; ('PC-AI USB4 TCP HIL ' + ('1' * 32)) = '' }
    foreach ($prior in $priorRules.GetEnumerator()) {
        $fixtureState.Rules[$prior.Key] = $prior.Value
        try {
            Invoke-FixtureEndpoint -Role listen -Firewall ([bool]$prior.Value) | Out-Null
            throw 'The endpoint started with an abandoned firewall rule.'
        } catch {
            if ($_.Exception.Message -notlike "*operator cleanup: $($prior.Key)*" -or
                $fixtureState.Transfers -ne 0 -or $fixtureState.Added -ne 0 -or $fixtureState.Removed -ne 0 -or
                -not $fixtureState.Rules.ContainsKey($prior.Key)) { throw }
        }
        $fixtureState.Rules.Remove($prior.Key)
    }
}

# Protects: a failed removal surfaces the owned rule identifier for recovery.
# Detects: cleanup errors discarded or a rule reported without its exact name.
# Needs: PowerShell 7 and a stub firewall provider refusing removal.
# Breadcrumb: Tests/Hardware/Invoke-PcaiUsb4TcpEndpoint.ps1::firewall cleanup catch.
function Test-ListenerReportsExactRuleWhenFirewallCleanupFails {
    $fixtureState.CleanupFails = $true
    try {
        Invoke-FixtureEndpoint -Role listen -Firewall $true | Out-Null
        throw 'The endpoint reported success after a firewall cleanup failure.'
    } catch {
        if (-not $fixtureState.LastRule -or $_.Exception.Message -notlike "*'$($fixtureState.LastRule)'*" -or
            -not $fixtureState.Rules.ContainsKey($fixtureState.LastRule)) { throw }
    }
}

try {
    foreach ($selector in @('Test-ClientUsesScopedOnLinkRouteAndCounters', 'Test-ClientRejectsWrongNeighborIpOrMac',
            'Test-ClientRejectsFlatReceiveOrTransmitCounter', 'Test-ListenerRemovesScopedFirewallRuleAfterSuccess',
            'Test-ListenerPreservesTimeoutAndRemovesScopedFirewallRule', 'Test-ListenerNamesAndPreservesAbandonedFirewallRules',
            'Test-ListenerReportsExactRuleWhenFirewallCleanupFails')) {
        $fixtureState = @{
            NextHop = '::'; DestinationPrefix = 'fe80::/64'; NeighborIp = 'fe80::2%25'; NeighborMac = '02-00-00-00-00-02'
            Counter = 0; FlatRx = $false; FlatTx = $false; Timeout = $false; Transfers = 0
            Added = 0; Removed = 0; Rules = @{ unrelated = '' }; CleanupFails = $false; LastRule = ''
        }
        & $selector
        Write-Output "PASS: $selector"
    }
} finally {
    Remove-Item -LiteralPath $fixtureBinary.FullName
}
