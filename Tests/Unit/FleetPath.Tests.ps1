#Requires -Version 7.0
# Hermetic controls for Tools/SystemScripts/Networking/Test-FleetPath.ps1. No test opens a
# network connection, reads the ARP or route table, or runs tailscale: the TCP connect is an
# injected connector, and the neighbour, route and tailscale reads are mocked wrapper functions.
# The one real process is a child pwsh used to prove the native-call deadline.
#
# Why this file exists: a roaming laptop picks LAN or tailnet addresses from this tool's
# output. A wrong "home" verdict sends traffic to a stranger's machine on a hotel
# 192.168.1.0/24, and a probe that ignores its budget turns a diagnostic into a hang.
# Each selector below names the failure it detects. Breadcrumbs point at the code under test;
# the user-facing description is in Tools/SystemScripts/README.md ("Fleet path diagnostic").
Set-StrictMode -Version Latest

BeforeAll {
    $script:RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
    $script:ScriptPath = Join-Path $script:RepoRoot 'Tools/SystemScripts/Networking/Test-FleetPath.ps1'
    $script:InventoryPath = Join-Path $script:RepoRoot 'Tools/SystemScripts/Networking/fleet-paths.example.json'
    $script:FixturePath = Join-Path $PSScriptRoot '../Fixtures/FleetPath/tailscale-status.redacted.json'
    . $script:ScriptPath

    $script:HomeMac = '00-00-5E-00-53-01'
    $script:TailscaleFixture = Get-Content -LiteralPath $script:FixturePath -Raw | ConvertFrom-Json

    # Connectors replace the TCP connect (contract: Address, Port -> object with Task and Client).
    $script:ConnectAll = { param($Address, $Port) [pscustomobject]@{ Task = [Threading.Tasks.Task]::CompletedTask; Client = $null } }
    $script:ConnectRefused = {
        param($Address, $Port)
        [pscustomobject]@{ Task = [Threading.Tasks.Task]::FromException([Net.Sockets.SocketException]::new(10061)); Client = $null }
    }
    $script:ConnectHang = {
        param($Address, $Port)
        [pscustomobject]@{ Task = [Threading.Tasks.TaskCompletionSource[bool]]::new().Task; Client = $null }
    }
    # Everything connects except the minipc LAN address, which the inventory marks as blocked from the local subnet.
    $script:ConnectAllButMinipcLan = {
        param($Address, $Port)
        $task = if ($Address -eq '203.0.113.75') { [Threading.Tasks.Task]::FromException([Net.Sockets.SocketException]::new(10061)) }
        else { [Threading.Tasks.Task]::CompletedTask }
        [pscustomobject]@{ Task = $task; Client = $null }
    }

    function New-TestGateway {
        param([string]$Ip = '192.0.2.1', [string]$Mac = '00-00-5E-00-53-01', [int[]]$Ports = @(443), [string]$Name = 'gw')
        [pscustomobject]@{ Name = $Name; Ip = $Ip; Mac = $Mac; ProbePorts = $Ports }
    }
    function New-TestProbeResult {
        param([string]$Address = '192.0.2.1', [int]$Port = 443, [bool]$Reachable = $true)
        [pscustomobject]@{ Address = $Address; Port = $Port; Reachable = $Reachable; LatencyMs = if ($Reachable) { 3.0 } else { $null }; Error = $null; RemoteAddress = $null }
    }
    function New-TestNeighbor {
        param([string]$Ip = '192.0.2.1', [string]$Mac = '00-00-5E-00-53-01', [string]$State = 'Reachable', [string]$Alias = 'Ethernet 12', [int]$IfIndex = 12)
        [pscustomobject]@{ IPAddress = $Ip; LinkLayerAddress = $Mac; State = $State; InterfaceAlias = $Alias; InterfaceIndex = $IfIndex }
    }
    function New-TestCandidate {
        param([string]$Kind, [string]$Address = '10.0.0.1', [int]$Order = 0, [bool]$Reachable = $true, $Skipped = $null, [string]$Path = $null)
        [pscustomobject]@{
            Kind = $Kind; Address = $Address; Port = 22; Order = $Order; Reachable = $Reachable; Skipped = $Skipped
            TailscalePath = if ($Path) { [pscustomobject]@{ State = $Path; Display = $Path } } else { $null }
        }
    }
    # Stands in for the DNS resolver: each known MagicDNS name resolves to its inventory tailnet address.
    function Get-TestMagicDns {
        param([string[]]$Names)
        $table = @{ 'nas.tailnet.example.invalid' = '100.64.0.101'; 'workstation.tailnet.example.invalid' = '100.64.0.102' }
        $answers = @{}
        foreach ($name in $Names) { $answers[$name] = if ($table.ContainsKey($name)) { @($table[$name]) } else { @() } }
        $answers
    }
    function Write-TestInventory {
        param([string]$Json)
        $file = Join-Path $TestDrive ([guid]::NewGuid().ToString('N') + '.json')
        Set-Content -LiteralPath $file -Value $Json -Encoding utf8
        $file
    }
}

Describe 'Fleet inventory' -Tag 'Unit' {
    # Protects: the checked-in example is the schema contract copied into the private inventory; a typo must not ship.
    # Detects: a missing host, a lost agent-bus tailnet-only restriction, or a gateway without a verifiable MAC.
    # Needs: the repository checkout only.
    # Breadcrumb: Tools/SystemScripts/Networking/fleet-paths.example.json; Import-FleetInventory
    It 'loads the example inventory with all six hosts and a MAC on every gateway' {
        $inventory = Import-FleetInventory -Path $script:InventoryPath
        $inventory.Hosts.Name | Should -Be @('nas', 'workstation', 'nuc', 'minipc', 'laptop2', 'router')
        $inventory.Gateways.Count | Should -BeGreaterOrEqual 1
        foreach ($gateway in $inventory.Gateways) { $gateway.Mac | Should -Match '^([0-9A-F]{2}-){5}[0-9A-F]{2}$' }
        $nas = $inventory.Hosts | Where-Object Name -EQ 'nas'
        ($nas.Services | Where-Object Service -EQ 'https').Ports | Should -Be @(443, 8080)
        $bus = (($inventory.Hosts | Where-Object Name -EQ 'workstation').Services | Where-Object Service -EQ 'agentbus')
        $bus.Via | Should -Be @('tailnet')
        # Every example host with a tailnet address also records its tailnet IPv6 address, which MagicDNS validation needs.
        foreach ($tailnetHost in ($inventory.Hosts | Where-Object { $_.Tailnet })) { $tailnetHost.TailnetV6 | Should -Match '^fd7a:115c:a1e0:' }
    }

    # Protects: a malformed inventory fails loudly instead of silently probing the wrong thing.
    # Detects: a bad MAC, port, service name or via kind being accepted.
    # Needs: TestDrive.
    # Breadcrumb: Import-FleetInventory
    It 'rejects <Case>' -ForEach @(
        @{ Case = 'a malformed gateway MAC'; Pattern = 'unusable MAC'; Json = '{"schema":"pcai.fleet-paths/1","gateways":[{"name":"g","ip":"192.0.2.1","mac":"not-a-mac"}],"hosts":{}}' }
        @{ Case = 'an out-of-range port'; Pattern = 'invalid port'; Json = '{"schema":"pcai.fleet-paths/1","hosts":{"h":{"lan":["10.0.0.1"],"services":{"ssh":{"ports":[70000]}}}}}' }
        @{ Case = 'an unknown service'; Pattern = 'unknown service'; Json = '{"schema":"pcai.fleet-paths/1","hosts":{"h":{"lan":["10.0.0.1"],"services":{"telnet":{"ports":[23]}}}}}' }
        @{ Case = 'an unknown via kind'; Pattern = 'unknown via'; Json = '{"schema":"pcai.fleet-paths/1","hosts":{"h":{"lan":["10.0.0.1"],"services":{"ssh":{"ports":[22],"via":["wifi"]}}}}}' }
        @{ Case = 'a malformed tailnetV6 address'; Pattern = 'invalid tailnetV6'; Json = '{"schema":"pcai.fleet-paths/1","hosts":{"h":{"tailnetV6":"not-an-ip","lan":["192.0.2.9"],"services":{"ssh":{"ports":[22]}}}}}' }
        @{ Case = 'a wrong schema'; Pattern = 'schema must be'; Json = '{"schema":"other/9","hosts":{}}' }
        @{ Case = 'invalid JSON'; Pattern = 'not valid JSON'; Json = '{ nope' }
    ) {
        { Import-FleetInventory -Path (Write-TestInventory $Json) } | Should -Throw -ExpectedMessage "*$Pattern*"
    }

    # Protects: a wrong -InventoryPath is an error, not an empty fleet that reports nothing wrong.
    # Detects: a missing file returning an empty inventory.
    # Needs: nothing.
    # Breadcrumb: Import-FleetInventory
    It 'throws for a missing inventory file and names the example to copy' {
        { Import-FleetInventory -Path (Join-Path $TestDrive 'absent.json') } | Should -Throw -ExpectedMessage '*not found*fleet-paths.example.json*'
    }

    # Protects: the live inventory lives outside the repository, so the default path must come from the environment or the home folder.
    # Detects: a default that points back into the repository, which would invite committing real addresses.
    # Needs: the script's AST only; nothing runs.
    # Breadcrumb: Test-FleetPath.ps1 param block default for -InventoryPath
    It 'defaults -InventoryPath to PCAI_FLEET_PATHS, else ~/.machine/fleet-paths.json' {
        $parameter = (Get-Command -Name $script:ScriptPath).ScriptBlock.Ast.ParamBlock.Parameters | Where-Object { $_.Name.VariablePath.UserPath -eq 'InventoryPath' }
        $default = $parameter.DefaultValue.Extent.Text
        $default | Should -Match 'PCAI_FLEET_PATHS'
        $default | Should -Match '\.machine.fleet-paths\.json'
        $default | Should -Not -Match 'PSScriptRoot'
    }
}

Describe 'Candidate ordering' -Tag 'Unit' {
    BeforeAll {
        $script:Inventory = Import-FleetInventory -Path $script:InventoryPath
        $script:DtmWork = $script:Inventory.Hosts | Where-Object Name -EQ 'workstation'
    }

    # Protects: "first reachable wins" depends on LAN addresses being tried before the tailnet.
    # Detects: a reordering that would recommend a relayed tailnet path while a LAN path works.
    # Needs: the example inventory.
    # Breadcrumb: New-FleetCandidatePlan
    It 'plans LAN addresses in listed order, then the tailnet address, then the MagicDNS name' {
        $plan = @(New-FleetCandidatePlan -FleetHost $script:DtmWork -Via @('lan', 'tailnet', 'magicdns') -Port 22)
        $plan.Kind | Should -Be @('lan', 'lan', 'tailnet', 'magicdns')
        @($plan | ForEach-Object { $_.Address }) | Should -Be @('192.0.2.33', '198.51.100.34', '100.64.0.102', 'workstation.tailnet.example.invalid')
        $plan.Order | Should -Be @(0, 1, 2, 3)
    }

    # Protects: the agent-bus hub listens on the tailnet only; a LAN probe would test the wrong thing.
    # Detects: the via restriction being ignored.
    # Needs: the example inventory.
    # Breadcrumb: New-FleetCandidatePlan; fleet-paths.example.json workstation agentbus via
    It 'honors a tailnet-only via list' {
        $plan = @(New-FleetCandidatePlan -FleetHost $script:DtmWork -Via @('tailnet') -Port 8400)
        $plan.Kind | Should -Be @('tailnet')
    }

    # Protects: the router has no tailnet address; inventing a candidate would probe a null.
    # Detects: tailnet or MagicDNS candidates for a LAN-only host.
    # Needs: the example inventory.
    # Breadcrumb: New-FleetCandidatePlan
    It 'plans only LAN candidates for a LAN-only host' {
        $router = $script:Inventory.Hosts | Where-Object Name -EQ 'router'
        @(New-FleetCandidatePlan -FleetHost $router -Via @('lan', 'tailnet', 'magicdns') -Port 443).Kind | Should -Be @('lan')
    }
}

Describe 'Probe budget' -Tag 'Unit' {
    # Protects: a diagnostic that hangs is worse than none; one unreachable candidate must not stall the run.
    # Detects: the deadline living inside the connector, or a wait without a timeout.
    # Needs: nothing; the hung connector never completes, so the real Task.Wait(budget) is what ends it.
    # Breadcrumb: $script:FleetTcpProbe; Test-FleetTcpPort
    It 'ends a hung probe within 1.5 times its budget and reports a timeout' {
        $budget = 300
        $clock = [Diagnostics.Stopwatch]::StartNew()
        $result = Test-FleetTcpPort -Address '192.0.2.1' -Port 22 -BudgetMs $budget -Connector $script:ConnectHang
        $clock.Stop()
        $result.Reachable | Should -BeFalse
        $result.Error | Should -Match 'timeout after 300ms'
        $clock.ElapsedMilliseconds | Should -BeLessThan ($budget * 1.5)
        # It waited for the budget instead of failing early, so the bound above is not vacuous.
        $clock.ElapsedMilliseconds | Should -BeGreaterOrEqual ($budget - 50)
    }

    # Protects: the timeout test above must be able to fail; this is its positive control.
    # Detects: a harness in which every probe reports unreachable, or in which success also waits the budget.
    # Needs: nothing.
    # Breadcrumb: Test-FleetTcpPort
    It 'positive control: a completed connect is reachable and does not wait out the budget' {
        $clock = [Diagnostics.Stopwatch]::StartNew()
        $result = Test-FleetTcpPort -Address '192.0.2.1' -Port 22 -BudgetMs 2000 -Connector $script:ConnectAll
        $clock.Stop()
        $result.Reachable | Should -BeTrue
        $result.Error | Should -BeNullOrEmpty
        $clock.ElapsedMilliseconds | Should -BeLessThan 1000
    }

    # Protects: a refused port must read as the socket error, not as a timeout, so the operator sees why.
    # Detects: AggregateException or MethodInvocationException text leaking into Error.
    # Needs: nothing.
    # Breadcrumb: $script:FleetTcpProbe catch block (copied from Mount-VigilFleetShare's Test-VigilNfsPort handling)
    It 'reports a refused connect as the socket error, quickly' {
        $clock = [Diagnostics.Stopwatch]::StartNew()
        $result = Test-FleetTcpPort -Address '192.0.2.1' -Port 22 -BudgetMs 2000 -Connector $script:ConnectRefused
        $clock.Stop()
        $result.Reachable | Should -BeFalse
        $result.Error | Should -Be 'ConnectionRefused'
        $clock.ElapsedMilliseconds | Should -BeLessThan 1000
    }

    # Protects: the 5 s runtime target; probes in a wave must overlap, not queue.
    # Detects: a throttle below the probe count, which turns N hung probes into N/throttle budgets.
    # Needs: PowerShell 7 ForEach-Object -Parallel; no network. Serial execution would take 8 x 500 ms.
    # Breadcrumb: Invoke-FleetProbeBatch
    It 'runs eight hung probes concurrently, in far less than eight budgets' {
        $probes = 1..8 | ForEach-Object { [pscustomobject]@{ Address = "192.0.2.$_"; Port = 22 } }
        $clock = [Diagnostics.Stopwatch]::StartNew()
        $results = @(Invoke-FleetProbeBatch -Probes $probes -BudgetMs 500 -Connector $script:ConnectHang)
        $clock.Stop()
        $results.Count | Should -Be 8
        @($results | Where-Object Reachable).Count | Should -Be 0
        $clock.ElapsedMilliseconds | Should -BeLessThan 2500
    }

    # Protects: an empty wave (everything skipped off-LAN) still runs the caller's wait work.
    # Detects: an early return that drops the WhileWaiting block, which would skip the route lookups.
    # Needs: nothing.
    # Breadcrumb: Invoke-FleetProbeBatch
    It 'returns nothing for an empty batch but still runs the wait block' {
        $script:ranWait = $false
        @(Invoke-FleetProbeBatch -Probes @() -BudgetMs 500 -WhileWaiting { $script:ranWait = $true }).Count | Should -Be 0
        $script:ranWait | Should -BeTrue
    }

    # Protects: MagicDNS candidates are routed by the address they connected to; a dual-mode socket reports ::ffff:a.b.c.d.
    # Detects: Find-NetRoute being handed an IPv4-mapped IPv6 address, which finds no route.
    # Needs: nothing.
    # Breadcrumb: $script:FleetTcpProbe RemoteAddress normalization
    It 'maps an IPv4-mapped IPv6 remote address back to IPv4' {
        $connector = {
            param($Address, $Port)
            $client = [pscustomobject]@{ Connected = $true; Client = [pscustomobject]@{ RemoteEndPoint = [Net.IPEndPoint]::new([ipaddress]::Parse('::ffff:100.64.0.101'), 22) } }
            $client | Add-Member -MemberType ScriptMethod -Name Dispose -Value { }
            [pscustomobject]@{ Task = [Threading.Tasks.Task]::CompletedTask; Client = $client }
        }
        (Test-FleetTcpPort -Address 'nas.tailnet.example.invalid' -Port 22 -BudgetMs 1000 -Connector $connector).RemoteAddress | Should -Be '100.64.0.101'
    }
}

Describe 'Location: home LAN versus a colliding subnet' -Tag 'Unit' {
    BeforeEach {
        # By default the gateway probe egresses interface 12, the interface the neighbour rows below sit on.
        Mock Find-FleetRoute { [pscustomobject]@{ Interface = 'Ethernet 12'; InterfaceIndex = 12; SourceAddress = '192.0.2.200'; NextHop = 'on-link' } }
    }
    # Protects: the home verdict gates every LAN recommendation, so it needs two independent facts.
    # Detects: a verdict that trusts the subnet or the TCP connect alone.
    # Needs: mocked neighbour table.
    # Breadcrumb: Resolve-FleetLocation
    It 'is home when the gateway answers and its on-link MAC matches the inventory' {
        Mock Get-FleetNeighbor { New-TestNeighbor }
        $location = Resolve-FleetLocation -Gateways @(New-TestGateway) -ProbeResults @(New-TestProbeResult)
        $location.State | Should -Be 'home-lan'
        $location.Collision | Should -BeFalse
        $location.Gateways[0].MacMatch | Should -BeTrue
    }

    # Protects: Windows prints MACs as AA-BB-..., other tools as aa:bb:...; both must match.
    # Detects: a case- or separator-sensitive comparison that reports home as off-LAN.
    # Needs: mocked neighbour table.
    # Breadcrumb: ConvertTo-FleetMac
    It 'matches a MAC written as <Format>' -ForEach @(
        @{ Format = 'lowercase with colons'; Mac = '00:00:5e:00:53:01' }
        @{ Format = 'uppercase without separators'; Mac = '00005E005301' }
        @{ Format = 'dotted groups'; Mac = '0000.5e00.5301' }
    ) {
        $row = New-TestNeighbor -Mac $Mac
        Mock Get-FleetNeighbor -MockWith ({ $row }.GetNewClosure())
        (Resolve-FleetLocation -Gateways @(New-TestGateway) -ProbeResults @(New-TestProbeResult)).State | Should -Be 'home-lan'
    }

    # Protects: the Collision case. A foreign network can reuse the home subnet and its router then answers at the same gateway address (the real-world case is a hotel on 192.168.1.0/24).
    # Detects: the subnet or an open port deciding "home"; that would send SSH to a stranger's 192.0.2.33.
    # Needs: mocked neighbour table returning a different gateway MAC.
    # Breadcrumb: Resolve-FleetLocation Collision
    It 'rejects a network that reuses the home subnet but has a different gateway MAC' {
        Mock Get-FleetNeighbor { New-TestNeighbor -Mac '00-11-22-33-44-55' }
        $location = Resolve-FleetLocation -Gateways @(New-TestGateway) -ProbeResults @(New-TestProbeResult)
        $location.State | Should -Be 'off-lan'
        $location.Collision | Should -BeTrue
        $location.Reason | Should -Match 'another network is using the home subnet'
        $location.Gateways[0].ObservedMac | Should -Be '00-11-22-33-44-55'
    }

    # Protects: fail closed. When the MAC cannot be established the answer is off-LAN, never home.
    # Detects: a missing, unresolved or zero MAC being treated as a match.
    # Needs: mocked neighbour table.
    # Breadcrumb: Resolve-FleetLocation; ConvertTo-FleetMac
    It 'fails closed to off-LAN when the neighbour MAC is <Case>' -ForEach @(
        @{ Case = 'missing'; Rows = @() }
        @{ Case = 'all zeros'; Rows = @(@{ Mac = '00-00-00-00-00-00'; State = 'Reachable' }) }
        @{ Case = 'empty'; Rows = @(@{ Mac = ''; State = 'Stale' }) }
        @{ Case = 'an Incomplete entry'; Rows = @(@{ Mac = '00-00-5E-00-53-01'; State = 'Incomplete' }) }
        @{ Case = 'an Unreachable entry'; Rows = @(@{ Mac = '00-00-5E-00-53-01'; State = 'Unreachable' }) }
    ) {
        $rows = @($Rows | ForEach-Object { New-TestNeighbor -Mac $_.Mac -State $_.State })
        Mock Get-FleetNeighbor -MockWith ({ $rows }.GetNewClosure())
        $location = Resolve-FleetLocation -Gateways @(New-TestGateway) -ProbeResults @(New-TestProbeResult)
        $location.State | Should -Be 'off-lan'
        $location.Collision | Should -BeFalse
        $location.Reason | Should -Match 'cannot be verified|home cannot be verified'
    }

    # Protects: a leftover ARP row from the home network must not make a hotel look like home.
    # Detects: the verdict using the neighbour table without a live TCP connect.
    # Needs: mocked neighbour table with a matching MAC and a probe that failed.
    # Breadcrumb: Resolve-FleetLocation
    It 'is off-LAN when the MAC matches but the gateway does not answer' {
        Mock Get-FleetNeighbor { New-TestNeighbor }
        $location = Resolve-FleetLocation -Gateways @(New-TestGateway) -ProbeResults @(New-TestProbeResult -Reachable $false)
        $location.State | Should -Be 'off-lan'
        $location.Reason | Should -Match 'No inventory gateway answered'
    }

    # Protects: two interfaces on different networks can both hold a row for the gateway address; only the
    # interface whose row matches the inventory MAC is verified.
    # Detects: "any row matches" logic that treats the whole host as home, or lets the home row hide a foreign row.
    # Needs: mocked neighbour table with two rows on different interface indexes.
    # Breadcrumb: Resolve-FleetLocation VerifiedInterfaces / CollidingInterfaces
    It 'records the verified interface and the colliding interface separately' {
        Mock Get-FleetNeighbor { New-TestNeighbor -IfIndex 12; New-TestNeighbor -Mac 'AA-BB-CC-DD-EE-FF' -Alias 'Wi-Fi' -IfIndex 7 }
        $location = Resolve-FleetLocation -Gateways @(New-TestGateway) -ProbeResults @(New-TestProbeResult)
        $location.State | Should -Be 'home-lan'
        $location.Collision | Should -BeTrue
        @($location.VerifiedInterfaces) | Should -Be @(12)
        @($location.CollidingInterfaces) | Should -Be @(7)
        $location.Reason | Should -Match 'interface 7'
    }

    # Protects: interface-specific evidence. The gateway answered over interface 12, but the only matching MAC row is on
    # interface 7, which the probe never used; that row is not proof the probe's network is home.
    # Detects: a reachable gateway plus a matching row anywhere authorizing that row's interface.
    # Needs: mocked route (egress ifIndex 12) and a neighbour table holding the matching MAC only on ifIndex 7.
    # Breadcrumb: Resolve-FleetLocation (verified interfaces require the probe's egress ifIndex)
    It 'does not verify an interface the gateway probe did not egress' {
        Mock Get-FleetNeighbor { New-TestNeighbor -Alias 'Wi-Fi' -IfIndex 7 }
        $location = Resolve-FleetLocation -Gateways @(New-TestGateway) -ProbeResults @(New-TestProbeResult)
        $location.State | Should -Be 'off-lan'
        @($location.VerifiedInterfaces).Count | Should -Be 0
        $location.Reason | Should -Match 'cannot be verified'
    }

    # Protects: the control for the test above; the same table with the row on the probe's egress interface verifies it.
    # Detects: a harness that can never verify anything.
    # Needs: the same mocks with the row on ifIndex 12.
    # Breadcrumb: Resolve-FleetLocation
    It 'positive control: verifies the interface the gateway probe egressed when its row matches' {
        Mock Get-FleetNeighbor { New-TestNeighbor -IfIndex 12 }
        $location = Resolve-FleetLocation -Gateways @(New-TestGateway) -ProbeResults @(New-TestProbeResult)
        $location.State | Should -Be 'home-lan'
        @($location.VerifiedInterfaces) | Should -Be @(12)
    }

    # Protects: fail closed when the probe's egress cannot be determined.
    # Detects: an unresolved gateway route being treated as matching any row.
    # Needs: Find-FleetRoute returning nothing.
    # Breadcrumb: Resolve-FleetLocation
    It 'verifies nothing when the gateway route cannot be resolved' {
        Mock Get-FleetNeighbor { New-TestNeighbor -IfIndex 12 }
        Mock Find-FleetRoute { $null }
        $location = Resolve-FleetLocation -Gateways @(New-TestGateway) -ProbeResults @(New-TestProbeResult)
        $location.State | Should -Be 'off-lan'
        @($location.VerifiedInterfaces).Count | Should -Be 0
    }

    # Protects: when every interface that holds a row disagrees with the inventory there is no verified interface.
    # Detects: a collision still leaving a non-empty verified set.
    # Needs: mocked neighbour table with a foreign MAC only.
    # Breadcrumb: Resolve-FleetLocation
    It 'has no verified interface when the only row is a foreign MAC' {
        Mock Get-FleetNeighbor { New-TestNeighbor -Mac 'AA-BB-CC-DD-EE-FF' -Alias 'Wi-Fi' -IfIndex 7 }
        $location = Resolve-FleetLocation -Gateways @(New-TestGateway) -ProbeResults @(New-TestProbeResult)
        $location.State | Should -Be 'off-lan'
        @($location.VerifiedInterfaces).Count | Should -Be 0
        @($location.CollidingInterfaces) | Should -Be @(7)
    }
    # Protects: an inventory entry without a MAC cannot verify anything.
    # Detects: a null expected MAC comparing equal to an observed one.
    # Needs: mocked neighbour table.
    # Breadcrumb: Resolve-FleetLocation
    It 'fails closed when the inventory records no MAC for the gateway' {
        Mock Get-FleetNeighbor { New-TestNeighbor }
        $location = Resolve-FleetLocation -Gateways @(New-TestGateway -Mac $null) -ProbeResults @(New-TestProbeResult)
        $location.State | Should -Be 'off-lan'
        $location.Collision | Should -BeFalse
    }

    # Protects: this-host reaches 198.51.100.1 by routing, so it has no neighbour row for it; that must not veto 192.0.2.1.
    # Detects: requiring every inventory gateway to verify.
    # Needs: mocked neighbour table holding only the on-link gateway.
    # Breadcrumb: Resolve-FleetLocation (any verified gateway makes home)
    It 'is home when one gateway verifies and a routed one has no neighbour row' {
        Mock Get-FleetNeighbor { New-TestNeighbor }
        $gateways = @((New-TestGateway -Ip '198.51.100.1' -Name 'routed'), (New-TestGateway -Ip '192.0.2.1' -Name 'on-link'))
        $probes = @((New-TestProbeResult -Address '198.51.100.1'), (New-TestProbeResult -Address '192.0.2.1'))
        $location = Resolve-FleetLocation -Gateways $gateways -ProbeResults $probes
        $location.State | Should -Be 'home-lan'
        ($location.Gateways | Where-Object Name -EQ 'routed').MacMatch | Should -BeFalse
    }
}

Describe 'Location retry after a transient gateway stall' -Tag 'Unit' {
    BeforeAll {
        $script:MiniInventory = Write-TestInventory '{"schema":"pcai.fleet-paths/1","gateways":[{"name":"gw","ip":"192.0.2.1","mac":"00-00-5E-00-53-01","probePorts":[443]}],"hosts":{"h":{"tailnet":"100.64.0.102","lan":["192.0.2.33"],"services":{"ssh":{"ports":[22]}}}}}'
        Mock Find-FleetRoute { [pscustomobject]@{ Interface = 'Ethernet 12'; InterfaceIndex = 12; SourceAddress = '192.0.2.200'; NextHop = 'on-link' } }
        Mock Get-FleetTailscaleStatus { [pscustomobject]@{ Available = $false; Error = 'test'; Status = $null } }
        # Builds a connector that logs each gateway attempt to $Counter and fails the first $FailFirst attempts.
        function New-GatewayConnector {
            param([string]$Counter, [int]$FailFirst)
            [scriptblock]::Create(@"
param(`$Address, `$Port)
`$fail = `$false
if (`$Address -eq '192.0.2.1') {
    Add-Content -LiteralPath '$Counter' -Value 'attempt'
    `$fail = @(Get-Content -LiteralPath '$Counter').Count -le $FailFirst
}
`$task = if (`$fail) { [Threading.Tasks.Task]::FromException([Net.Sockets.SocketException]::new(10060)) } else { [Threading.Tasks.Task]::CompletedTask }
[pscustomobject]@{ Task = `$task; Client = `$null }
"@)
        }
    }

    # Protects: one transient stall on the gateway (seen about once in 40 live runs) must not read as "away from home".
    # Detects: a single failed gateway wave producing off-LAN, which would skip every LAN candidate.
    # Needs: a connector that fails the first gateway attempt, then connects; the attempt counter proves the retry ran.
    # Breadcrumb: Invoke-FleetPathTest gateway retry
    It 'retries the gateway probes once and reaches home when the second attempt answers' {
        Mock Get-FleetNeighbor { New-TestNeighbor }
        $counter = Join-Path $TestDrive 'stall-then-answer.log'
        $report = Invoke-FleetPathTest -InventoryPath $script:MiniInventory -BudgetMs 500 -Connector (New-GatewayConnector -Counter $counter -FailFirst 1)
        @(Get-Content -LiteralPath $counter).Count | Should -Be 2
        $report.Location.State | Should -Be 'home-lan'
        ($report.Targets[0].Endpoints[0].Candidates | Where-Object Kind -EQ 'lan').Reachable | Should -BeTrue
    }

    # Protects: the retry is bounded, so a real away-from-home run costs two gateway waves, not an endless loop.
    # Detects: an unbounded retry, or no retry at all; the control above proves the retry can succeed.
    # Needs: a connector that never lets the gateway answer.
    # Breadcrumb: Invoke-FleetPathTest gateway retry
    It 'gives up after exactly one retry when the gateway never answers' {
        Mock Get-FleetNeighbor { New-TestNeighbor }
        $counter = Join-Path $TestDrive 'never-answers.log'
        $report = Invoke-FleetPathTest -InventoryPath $script:MiniInventory -BudgetMs 500 -Connector (New-GatewayConnector -Counter $counter -FailFirst 99)
        @(Get-Content -LiteralPath $counter).Count | Should -Be 2
        $report.Location.State | Should -Be 'off-lan'
        $report.Location.Reason | Should -Match 'No inventory gateway answered'
    }

    # Protects: a colliding MAC is a definite answer; retrying it only delays the hotel-LAN verdict.
    # Detects: the retry firing on a collision.
    # Needs: a different gateway MAC and a connector that answers.
    # Breadcrumb: Invoke-FleetPathTest gateway retry
    It 'does not retry when the gateway answers with the wrong MAC' {
        Mock Get-FleetNeighbor { New-TestNeighbor -Mac '00-11-22-33-44-55' }
        $counter = Join-Path $TestDrive 'collision.log'
        $report = Invoke-FleetPathTest -InventoryPath $script:MiniInventory -BudgetMs 500 -Connector (New-GatewayConnector -Counter $counter -FailFirst 0)
        @(Get-Content -LiteralPath $counter).Count | Should -Be 1
        $report.Location.Collision | Should -BeTrue
    }
}

Describe 'LAN egress gating on a multihomed host' -Tag 'Unit' {
    BeforeAll {
        # Ethernet (ifIndex 12) verifies the home gateway; Wi-Fi (ifIndex 7) sits on an unrelated network that reuses
        # the subnet and answers at the same gateway address with a different MAC.
        $script:MultihomedInventory = Write-TestInventory '{"schema":"pcai.fleet-paths/1","gateways":[{"name":"gw","ip":"192.0.2.1","mac":"00-00-5E-00-53-01","probePorts":[443]}],"hosts":{"viaWired":{"tailnet":"100.64.0.102","lan":["198.51.100.20"],"services":{"ssh":{"ports":[22]}}},"viaWifi":{"tailnet":"100.64.0.103","lan":["198.51.100.30"],"services":{"ssh":{"ports":[22]}}},"noRoute":{"tailnet":"100.64.0.104","lan":["198.51.100.40"],"services":{"ssh":{"ports":[22]}}}}}'
        Mock Get-FleetNeighbor { New-TestNeighbor -IfIndex 12; New-TestNeighbor -Mac 'AA-BB-CC-DD-EE-FF' -Alias 'Wi-Fi' -IfIndex 7 }
        Mock Get-FleetTailscaleStatus { [pscustomobject]@{ Available = $false; Error = 'test'; Status = $null } }
        Mock Find-FleetRoute {
            switch ($Address) {
                '198.51.100.20' { [pscustomobject]@{ Interface = 'Ethernet 12'; InterfaceIndex = 12; SourceAddress = '192.0.2.200'; NextHop = 'on-link' } }
                '198.51.100.30' { [pscustomobject]@{ Interface = 'Wi-Fi'; InterfaceIndex = 7; SourceAddress = '192.0.2.77'; NextHop = '192.0.2.254' } }
                '198.51.100.40' { $null }
                '192.0.2.1' { [pscustomobject]@{ Interface = 'Ethernet 12'; InterfaceIndex = 12; SourceAddress = '192.0.2.200'; NextHop = 'on-link' } }
                default { [pscustomobject]@{ Interface = 'Tailscale'; InterfaceIndex = 3; SourceAddress = '100.64.0.100'; NextHop = 'on-link' } }
            }
        }
        # Records every address:port a connect was attempted for, one file each, so "never probed" is observable
        # across the parallel runspaces. Every connect succeeds, so a probed candidate would show Reachable.
        $script:ProbeLog = Join-Path $TestDrive 'multihomed-probes'
        $null = New-Item -ItemType Directory -Path $script:ProbeLog
        $connector = [scriptblock]::Create(@"
param(`$Address, `$Port)
`$null = New-Item -ItemType File -Force -Path (Join-Path '$($script:ProbeLog)' "`$Address-`$Port")
[pscustomobject]@{ Task = [Threading.Tasks.Task]::CompletedTask; Client = `$null }
"@)
        $script:Multihomed = Invoke-FleetPathTest -InventoryPath $script:MultihomedInventory -BudgetMs 500 -Connector $connector
        $script:Cand = @{}
        foreach ($target in $script:Multihomed.Targets) {
            $script:Cand[$target.Name] = @($target.Endpoints[0].Candidates | Where-Object Kind -EQ 'lan')[0]
        }
    }

    # Protects: one verified home gateway must not enable every LAN address. A LAN candidate that routes out the
    # interface on the colliding network would connect to a stranger's machine.
    # Detects: gating on the host-wide home verdict instead of the candidate's own egress interface.
    # Needs: mocked neighbour rows (ifIndex 12 verified, ifIndex 7 colliding) and a route that sends the candidate via ifIndex 7.
    # Breadcrumb: Invoke-FleetPathTest wave B; Resolve-FleetLocation VerifiedInterfaces
    It 'skips and never probes a LAN candidate routed via the colliding interface' {
        $script:Multihomed.Location.State | Should -Be 'home-lan'
        $script:Cand['viaWifi'].Skipped | Should -Be 'egress-not-verified'
        $script:Cand['viaWifi'].Reachable | Should -BeFalse
        $script:Cand['viaWifi'].Interface | Should -Be 'Wi-Fi'
        Test-Path -LiteralPath (Join-Path $script:ProbeLog '198.51.100.30-22') | Should -BeFalse
        $wifiHost = $script:Multihomed.Targets | Where-Object Name -EQ 'viaWifi'
        $wifiHost.Endpoints[0].Recommended.Kind | Should -Be 'tailnet'
    }

    # Protects: the control for the test above. Without it a harness that probes nothing would pass.
    # Detects: gating that skips every LAN candidate, or a connector that never records.
    # Needs: the same run; this candidate routes via the verified interface.
    # Breadcrumb: Invoke-FleetPathTest wave B
    It 'positive control: probes and recommends a LAN candidate routed via the verified interface' {
        $script:Cand['viaWired'].Skipped | Should -BeNullOrEmpty
        $script:Cand['viaWired'].Reachable | Should -BeTrue
        Test-Path -LiteralPath (Join-Path $script:ProbeLog '198.51.100.20-22') | Should -BeTrue
        ($script:Multihomed.Targets | Where-Object Name -EQ 'viaWired').Endpoints[0].Recommended.Kind | Should -Be 'lan'
    }

    # Protects: fail closed when the egress cannot be determined.
    # Detects: an unresolved route being treated as eligible.
    # Needs: Find-FleetRoute returning nothing for this address.
    # Breadcrumb: Test-FleetLanEgress
    It 'skips a LAN candidate whose egress cannot be resolved' {
        $script:Cand['noRoute'].Skipped | Should -Be 'egress-not-verified'
        Test-Path -LiteralPath (Join-Path $script:ProbeLog '198.51.100.40-22') | Should -BeFalse
    }

    # Protects: the eligibility rule itself, independent of the report plumbing.
    # Detects: a route without an ifIndex, or an empty verified set, passing.
    # Needs: nothing.
    # Breadcrumb: Test-FleetLanEgress
    It 'treats a route as eligible only when its ifIndex is a verified interface' -ForEach @(
        @{ Route = [pscustomobject]@{ InterfaceIndex = 12 }; Verified = @(12); Expected = $true }
        @{ Route = [pscustomobject]@{ InterfaceIndex = 7 }; Verified = @(12); Expected = $false }
        @{ Route = [pscustomobject]@{ InterfaceIndex = 12 }; Verified = @(); Expected = $false }
        @{ Route = [pscustomobject]@{ InterfaceIndex = $null }; Verified = @(12); Expected = $false }
        @{ Route = $null; Verified = @(12); Expected = $false }
    ) {
        Test-FleetLanEgress -Route $Route -VerifiedInterfaces $Verified | Should -Be $Expected
    }
}

Describe 'MagicDNS resolution guard' -Tag 'Unit' {
    BeforeAll {
        # Roaming: the system resolver, not Tailscale, may answer for the MagicDNS name, and a foreign network can
        # point it at one of its own hosts. The name must resolve to this host's tailnet address before any connect,
        # and the connect must then target that validated address, never the name (a second lookup could differ).
        # ssh is offered over MagicDNS only, so a probe of the tailnet address cannot stand in for a probe of the name.
        $script:DnsInventory = Write-TestInventory '{"schema":"pcai.fleet-paths/1","gateways":[{"name":"gw","ip":"192.0.2.1","mac":"00-00-5E-00-53-01","probePorts":[443]}],"hosts":{"box":{"tailnet":"100.64.0.102","tailnetV6":"fd7a:115c:a1e0::66","magicDns":"box.tailnet.example.invalid","services":{"ssh":{"ports":[22],"via":["magicdns"]}}}}}'
        Mock Get-FleetNeighbor { New-TestNeighbor }
        Mock Find-FleetRoute {
            $index = if ($Address -eq '192.0.2.1') { 12 } else { 3 }
            [pscustomobject]@{ Interface = 'Ethernet 12'; InterfaceIndex = $index; SourceAddress = '100.64.0.100'; NextHop = 'on-link' }
        }
        Mock Get-FleetTailscaleStatus { [pscustomobject]@{ Available = $false; Error = 'tailscale unavailable'; Status = $null } }
        # Every connect succeeds and leaves one file per address-port, so a skipped candidate is provably not connected
        # to. A connect handed a NAME models the connector's own second lookup, which answers with a foreign address.
        function Invoke-DnsScenario {
            param([string]$Name, [scriptblock]$Resolver)
            $log = Join-Path $TestDrive "dns-$Name"
            $null = New-Item -ItemType Directory -Path $log
            $connector = [scriptblock]::Create(@"
param(`$Address, `$Port)
if (`$Address -notmatch '^[0-9.]+`$') { `$null = New-Item -ItemType File -Force -Path (Join-Path '$log' 'second-lookup-203.0.113.9') }
`$null = New-Item -ItemType File -Force -Path (Join-Path '$log' "`$Address-`$Port")
[pscustomobject]@{ Task = [Threading.Tasks.Task]::CompletedTask; Client = `$null }
"@)
            Mock Resolve-FleetDnsNames -MockWith $Resolver
            $report = Invoke-FleetPathTest -InventoryPath $script:DnsInventory -BudgetMs 500 -Connector $connector -Target 'box'
            [pscustomobject]@{
                Log = $log
                Dns = @($report.Targets[0].Endpoints[0].Candidates | Where-Object Kind -EQ 'magicdns')[0]
                Recommended = $report.Targets[0].Endpoints[0].Recommended
            }
        }
        $script:Foreign = Invoke-DnsScenario -Name 'foreign' -Resolver { @{ 'box.tailnet.example.invalid' = @('203.0.113.9') } }
        $script:Matching = Invoke-DnsScenario -Name 'matching' -Resolver { @{ 'box.tailnet.example.invalid' = @('100.64.0.102') } }
        $script:Failed = Invoke-DnsScenario -Name 'failed' -Resolver { @{} }
        $script:Mixed = Invoke-DnsScenario -Name 'mixed' -Resolver { @{ 'box.tailnet.example.invalid' = @('100.64.0.102', '203.0.113.9') } }
        $script:OtherPeerV6 = Invoke-DnsScenario -Name 'otherv6' -Resolver { @{ 'box.tailnet.example.invalid' = @('100.64.0.102', 'fd7a:115c:a1e0::67') } }
    }

    # Protects: the off-LAN guard. If tailscale is unavailable and a roaming network's resolver sends the MagicDNS
    # name to a foreign host, a successful connect must not be probed, reported reachable or recommended.
    # Detects: MagicDNS candidates being connected to before checking where the name resolved.
    # Needs: mocked resolver answering an RFC 5737 address; the connector's probe log (apart from the gateway probe) proves no connect happened.
    # Breadcrumb: Invoke-FleetPathTest MagicDNS resolution; Resolve-FleetDnsNames
    It 'skips a MagicDNS name that resolves to a foreign address and never connects to it' {
        $script:Foreign.Dns.Skipped | Should -Be 'magicdns-mismatch'
        $script:Foreign.Dns.Reachable | Should -BeFalse
        @($script:Foreign.Dns.Resolved) | Should -Be @('203.0.113.9')
        @(Get-ChildItem -LiteralPath $script:Foreign.Log | Where-Object Name -NE '192.0.2.1-443') | Should -BeNullOrEmpty
        $script:Foreign.Recommended | Should -BeNullOrEmpty
    }

    # Protects: validation and connect must use the same address. Validating the name and then connecting by name lets a
    # second lookup (a TTL expiry, a different resolver) send the connect to a foreign host after validation passed.
    # Detects: the probe being handed the hostname instead of the validated IP literal.
    # Needs: mocked resolver answering the inventory address at validation; the connector treats any name it receives
    # as a second lookup that would answer a foreign address, and records what it was asked to connect to.
    # Breadcrumb: Invoke-FleetPathTest ProbeAddress
    It 'connects to the validated IP literal, never to the name' {
        $script:Matching.Dns.ProbeAddress | Should -Be '100.64.0.102'
        Test-Path -LiteralPath (Join-Path $script:Matching.Log '100.64.0.102-22') | Should -BeTrue
        Test-Path -LiteralPath (Join-Path $script:Matching.Log 'box.tailnet.example.invalid-22') | Should -BeFalse
        Test-Path -LiteralPath (Join-Path $script:Matching.Log 'second-lookup-203.0.113.9') | Should -BeFalse
    }

    # Protects: the control for the tests above; a harness that probes nothing would pass them.
    # Detects: the guard skipping a correct resolution, or a connector that never records.
    # Needs: mocked resolver answering the inventory tailnet address.
    # Breadcrumb: Invoke-FleetPathTest MagicDNS resolution
    It 'positive control: probes a MagicDNS name that resolves to the inventory tailnet address' {
        $script:Matching.Dns.Skipped | Should -BeNullOrEmpty
        $script:Matching.Dns.Reachable | Should -BeTrue
        @($script:Matching.Dns.Resolved) | Should -Be @('100.64.0.102')
        $script:Matching.Recommended.Kind | Should -Be 'magicdns'
    }

    # Protects: fail closed when the name does not resolve.
    # Detects: a resolver failure leaving the candidate eligible.
    # Needs: mocked resolver returning no addresses.
    # Breadcrumb: Resolve-FleetDnsNames
    It 'skips a MagicDNS name that does not resolve and never connects to it' {
        $script:Failed.Dns.Skipped | Should -Be 'magicdns-unresolved'
        $script:Failed.Dns.Reachable | Should -BeFalse
        @(Get-ChildItem -LiteralPath $script:Failed.Log | Where-Object Name -NE '192.0.2.1-443') | Should -BeNullOrEmpty
    }

    # Protects: DNS answers are a set; one foreign address among correct ones can still be the one the connect uses.
    # Detects: a guard that accepts a name when any address matches.
    # Needs: mocked resolver answering the right address plus a foreign one.
    # Breadcrumb: Test-FleetMagicDnsResolution
    It 'skips a MagicDNS name when any resolved address is foreign' {
        $script:Mixed.Dns.Skipped | Should -Be 'magicdns-mismatch'
        @(Get-ChildItem -LiteralPath $script:Mixed.Log | Where-Object Name -NE '192.0.2.1-443') | Should -BeNullOrEmpty
    }

    # Protects: an IPv6 address inside the tailnet range can belong to ANOTHER tailnet peer, which a foreign resolver
    # could answer with; only this host's own tailnet IPv6 address is acceptable.
    # Detects: acceptance of any address in fd7a:115c:a1e0::/48.
    # Needs: mocked resolver answering the right IPv4 plus another peer's tailnet IPv6.
    # Breadcrumb: Test-FleetMagicDnsResolution
    It 'skips a MagicDNS name that also resolves to another peer''s tailnet IPv6 address' {
        $script:OtherPeerV6.Dns.Skipped | Should -Be 'magicdns-mismatch'
        @(Get-ChildItem -LiteralPath $script:OtherPeerV6.Log | Where-Object Name -NE '192.0.2.1-443') | Should -BeNullOrEmpty
    }

    # Protects: the resolution rule itself, independent of the report plumbing.
    # Detects: acceptance of an empty answer, a host with no recorded tailnet address, a foreign address, another peer's
    # tailnet IPv6 address, or any IPv6 answer when the inventory records no IPv6 address.
    # Needs: nothing.
    # Breadcrumb: Test-FleetMagicDnsResolution
    It 'accepts <Case>: <Expected>' -ForEach @(
        @{ Case = 'the inventory address'; Addresses = @('100.64.0.102'); Tailnet = '100.64.0.102'; V6 = $null; Expected = $true }
        @{ Case = 'the inventory IPv4 and IPv6 addresses'; Addresses = @('100.64.0.102', 'fd7a:115c:a1e0::66'); Tailnet = '100.64.0.102'; V6 = 'fd7a:115c:a1e0::66'; Expected = $true }
        @{ Case = 'the IPv6 address written in another form'; Addresses = @('100.64.0.102', 'FD7A:115C:A1E0:0:0:0:0:66'); Tailnet = '100.64.0.102'; V6 = 'fd7a:115c:a1e0::66'; Expected = $true }
        @{ Case = 'another peer''s tailnet IPv6 address'; Addresses = @('100.64.0.102', 'fd7a:115c:a1e0::67'); Tailnet = '100.64.0.102'; V6 = 'fd7a:115c:a1e0::66'; Expected = $false }
        @{ Case = 'a tailnet IPv6 address when the inventory records none'; Addresses = @('100.64.0.102', 'fd7a:115c:a1e0::66'); Tailnet = '100.64.0.102'; V6 = $null; Expected = $false }
        @{ Case = 'a tailnet IPv6 address alone'; Addresses = @('fd7a:115c:a1e0::66'); Tailnet = '100.64.0.102'; V6 = 'fd7a:115c:a1e0::66'; Expected = $false }
        @{ Case = 'a different tailnet address'; Addresses = @('100.64.0.103'); Tailnet = '100.64.0.102'; V6 = $null; Expected = $false }
        @{ Case = 'a documentation address'; Addresses = @('203.0.113.9'); Tailnet = '100.64.0.102'; V6 = $null; Expected = $false }
        @{ Case = 'a foreign IPv6 address beside the right one'; Addresses = @('100.64.0.102', '2001:db8::1'); Tailnet = '100.64.0.102'; V6 = 'fd7a:115c:a1e0::66'; Expected = $false }
        @{ Case = 'an empty answer'; Addresses = @(); Tailnet = '100.64.0.102'; V6 = $null; Expected = $false }
        @{ Case = 'an answer when the inventory has no tailnet address'; Addresses = @('100.64.0.102'); Tailnet = $null; V6 = $null; Expected = $false }
    ) {
        Test-FleetMagicDnsResolution -Addresses $Addresses -TailnetAddress $Tailnet -TailnetV6Address $V6 | Should -Be $Expected
    }
}
Describe 'MagicDNS resolver wrapper' -Tag 'Unit' {
    # Protects: the real resolver wrapper is bounded and never throws, so a hung or failing lookup becomes "unresolved".
    # Detects: an unbounded wait or an exception escaping for an unresolvable name.
    # Needs: a reserved example.invalid name (RFC 6761), which must not resolve; needs no network answer.
    # Breadcrumb: Resolve-FleetDnsNames
    It 'returns an empty answer for an unresolvable name within the budget' {
        $clock = [Diagnostics.Stopwatch]::StartNew()
        $answers = Resolve-FleetDnsNames -Names @('nothing.example.invalid') -BudgetMs 3000
        $clock.Stop()
        @($answers['nothing.example.invalid']).Count | Should -Be 0
        $clock.ElapsedMilliseconds | Should -BeLessThan 5000
    }
}

Describe 'Tailscale path parsing' -Tag 'Unit' {
    # Protects: the recommendation prefers a direct tailnet path over a DERP relay.
    # Detects: `Relay` (the home DERP region, set on every peer) being read as "relayed".
    # Needs: the redacted capture in Tests/Fixtures/FleetPath.
    # Breadcrumb: Resolve-FleetTailscalePath
    It 'reports direct when the peer has a current endpoint, although Relay is also set' {
        $peer = ($script:TailscaleFixture.Peer.PSObject.Properties.Value | Where-Object HostName -EQ 'workstation')
        $peer.Relay | Should -Be 'home'
        $path = Resolve-FleetTailscalePath -Status $script:TailscaleFixture -TailnetAddress '100.64.0.102'
        $path.State | Should -Be 'direct'
        $path.Display | Should -Be 'direct'
        $path.Endpoint | Should -Not -BeNullOrEmpty
    }

    # Protects: the expectation that the NAS tailnet path shows relay home; its peer name is nas-3f9a1c, not nas.
    # Detects: matching peers by hostname instead of tailnet IP, or losing the region.
    # Needs: the redacted capture.
    # Breadcrumb: Resolve-FleetTailscalePath
    It 'reports relay "home" for the NAS, matched by tailnet IP' {
        $path = Resolve-FleetTailscalePath -Status $script:TailscaleFixture -TailnetAddress '100.64.0.101'
        $path.State | Should -Be 'relay'
        $path.Region | Should -Be 'home'
        $path.Display | Should -Be 'relay "home"'
    }

    # Protects: a peer relay is better than DERP and must not be reported as direct or as DERP.
    # Detects: PeerRelay being ignored.
    # Needs: the synthetic peer in the fixture (the live fleet showed none).
    # Breadcrumb: Resolve-FleetTailscalePath
    It 'reports peer-relay when PeerRelay is set' {
        $path = Resolve-FleetTailscalePath -Status $script:TailscaleFixture -TailnetAddress '100.64.0.201'
        $path.State | Should -Be 'peer-relay'
        $path.Endpoint | Should -Be '198.51.100.77:40000:7'
    }

    # Protects: an offline peer must never look like a relayed one.
    # Detects: Online being ignored in favour of the Relay field.
    # Needs: the redacted capture (offline-peer was offline).
    # Breadcrumb: Resolve-FleetTailscalePath
    It 'reports offline for a peer that is not online' {
        (Resolve-FleetTailscalePath -Status $script:TailscaleFixture -TailnetAddress '100.64.0.107').State | Should -Be 'offline'
    }

    # Protects: probing the local machine's own tailnet address.
    # Detects: the local node being reported as "not in netmap".
    # Needs: the redacted capture.
    # Breadcrumb: Resolve-FleetTailscalePath
    It 'reports self for the local tailnet address and unknown for an address outside the netmap' {
        (Resolve-FleetTailscalePath -Status $script:TailscaleFixture -TailnetAddress '100.64.0.100').State | Should -Be 'self'
        (Resolve-FleetTailscalePath -Status $script:TailscaleFixture -TailnetAddress '100.64.0.1009').State | Should -Be 'unknown'
    }

    # Protects: tailscale omits fields (PeerRelay, CurAddr, Relay, Active) on some peers and versions.
    # Detects: StrictMode throwing on a missing member, which would abort the whole report.
    # Needs: the synthetic sparse peer, which has only HostName, TailscaleIPs and Online.
    # Breadcrumb: Get-FleetProperty; Resolve-FleetTailscalePath
    It 'tolerates a peer with omitted path fields under StrictMode' {
        { Resolve-FleetTailscalePath -Status $script:TailscaleFixture -TailnetAddress '100.64.0.202' } | Should -Not -Throw
        (Resolve-FleetTailscalePath -Status $script:TailscaleFixture -TailnetAddress '100.64.0.202').State | Should -Be 'unknown'
    }

    # Protects: with the tailscale daemon stopped, every peer row is stale and must not be trusted.
    # Detects: stale peer rows being reported as live paths.
    # Needs: the redacted capture with BackendState changed.
    # Breadcrumb: Resolve-FleetTailscalePath
    It 'reports unavailable when the backend is not Running, and when there is no status at all' {
        $stopped = $script:TailscaleFixture | ConvertTo-Json -Depth 10 | ConvertFrom-Json
        $stopped.BackendState = 'Stopped'
        (Resolve-FleetTailscalePath -Status $stopped -TailnetAddress '100.64.0.102').State | Should -Be 'unavailable'
        (Resolve-FleetTailscalePath -Status $null -TailnetAddress '100.64.0.102').State | Should -Be 'unavailable'
    }

    # Protects: the fixture itself stays free of key material and logins.
    # Detects: someone refreshing the capture without redacting it.
    # Needs: the fixture text.
    # Breadcrumb: Tests/Fixtures/FleetPath/tailscale-status.redacted.json
    It 'keeps the fixture free of node keys, logins and PeerAPI URLs' {
        $text = Get-Content -LiteralPath $script:FixturePath -Raw
        $text | Should -Not -Match 'nodekey:[0-9a-f]{16}'
        $text | Should -Not -Match '[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}'
        $text | Should -Not -Match 'PeerAPIURL|PublicKey|"User"'
    }
}

Describe 'Tailscale CLI unavailable' -Tag 'Unit' {
    # Protects: a laptop without tailscale, or with a wedged daemon, still gets LAN and TCP results.
    # Detects: a throw from a missing binary, an empty body, bad JSON, a bad exit or a timeout.
    # Needs: mocked Get-Command and Invoke-FleetNative.
    # Breadcrumb: Get-FleetTailscaleStatus
    It 'returns Available=false instead of throwing when the binary is missing' {
        Mock Get-Command { $null } -ParameterFilter { $Name -eq 'tailscale' }
        $status = Get-FleetTailscaleStatus
        $status.Available | Should -BeFalse
        $status.Error | Should -Match 'not found'
        $status.Status | Should -BeNullOrEmpty
    }

    It 'returns Available=false for <Case>' -ForEach @(
        @{ Case = 'empty output'; Run = @{ ExitCode = 0; Stdout = ''; Stderr = ''; TimedOut = $false }; Pattern = 'empty output' }
        @{ Case = 'whitespace-only output'; Run = @{ ExitCode = 0; Stdout = "  `r`n"; Stderr = ''; TimedOut = $false }; Pattern = 'empty output' }
        @{ Case = 'invalid JSON'; Run = @{ ExitCode = 0; Stdout = '{ not json'; Stderr = ''; TimedOut = $false }; Pattern = 'invalid JSON' }
        @{ Case = 'a JSON scalar'; Run = @{ ExitCode = 0; Stdout = '42'; Stderr = ''; TimedOut = $false }; Pattern = 'unexpected document' }
        @{ Case = 'a nonzero exit'; Run = @{ ExitCode = 1; Stdout = '{}'; Stderr = 'x'; TimedOut = $false }; Pattern = 'exited 1' }
        @{ Case = 'a timeout'; Run = @{ ExitCode = -1; Stdout = ''; Stderr = ''; TimedOut = $true }; Pattern = 'timed out' }
    ) {
        Mock Get-Command { [pscustomobject]@{ Source = 'tailscale.exe' } } -ParameterFilter { $Name -eq 'tailscale' }
        $run = [pscustomobject]$Run
        Mock Invoke-FleetNative -MockWith ({ $run }.GetNewClosure())
        $status = Get-FleetTailscaleStatus
        $status.Available | Should -BeFalse
        $status.Error | Should -Match $Pattern
    }

    # Protects: the native-call deadline that keeps `tailscale status` from hanging the run.
    # Detects: a wait without a timeout; positive control: a quick child must return its output.
    # Needs: a child pwsh (Environment.ProcessPath); no network.
    # Breadcrumb: Invoke-FleetNative
    It 'kills a native process that outlives its deadline, and returns output from one that does not' {
        $pwsh = [Environment]::ProcessPath
        $ok = Invoke-FleetNative -FilePath $pwsh -Arguments @('-NoProfile', '-Command', 'Write-Output control-ok') -TimeoutMs 20000
        $ok.TimedOut | Should -BeFalse
        $ok.Stdout.Trim() | Should -Be 'control-ok'
        $clock = [Diagnostics.Stopwatch]::StartNew()
        $slow = Invoke-FleetNative -FilePath $pwsh -Arguments @('-NoProfile', '-Command', 'Start-Sleep -Seconds 60') -TimeoutMs 1500
        $clock.Stop()
        $slow.TimedOut | Should -BeTrue
        $clock.ElapsedMilliseconds | Should -BeLessThan 6000
    }
}

Describe 'Recommendation' -Tag 'Unit' {
    # Protects: on the home LAN the first reachable LAN candidate wins, in listed order.
    # Detects: tailnet candidates outranking a working LAN address.
    # Needs: nothing.
    # Breadcrumb: Select-FleetRecommendation; Get-FleetCandidateRank
    It 'picks the first reachable LAN candidate in listed order on the home LAN' {
        $candidates = @(
            (New-TestCandidate -Kind 'lan' -Address '192.0.2.33' -Order 0 -Reachable $false)
            (New-TestCandidate -Kind 'lan' -Address '198.51.100.34' -Order 1)
            (New-TestCandidate -Kind 'tailnet' -Address '100.64.0.102' -Order 2 -Path 'direct')
        )
        (Select-FleetRecommendation -Candidates $candidates -OnHomeLan $true).Address | Should -Be '198.51.100.34'
    }

    # Protects: a hotel's 192.0.2.33 is someone else's machine, even if it accepts the connection.
    # Detects: a reachable LAN candidate being recommended off the home LAN.
    # Needs: nothing.
    # Breadcrumb: Select-FleetRecommendation
    It 'never recommends a LAN candidate off the home LAN, even when it connected' {
        $candidates = @(
            (New-TestCandidate -Kind 'lan' -Address '192.0.2.33' -Order 0)
            (New-TestCandidate -Kind 'tailnet' -Address '100.64.0.102' -Order 1 -Path 'relay')
        )
        (Select-FleetRecommendation -Candidates $candidates -OnHomeLan $false).Kind | Should -Be 'tailnet'
        Select-FleetRecommendation -Candidates @($candidates[0]) -OnHomeLan $false | Should -BeNullOrEmpty
    }

    # Protects: the spec's "prefer a direct tailnet path over DERP relay".
    # Detects: order alone deciding between tailnet paths of different quality.
    # Needs: synthetic candidates, because one host has a single tailnet address in practice.
    # Breadcrumb: Get-FleetCandidateRank
    It 'prefers direct over peer-relay over DERP relay, and the tailnet address over MagicDNS at equal quality' {
        $relay = New-TestCandidate -Kind 'tailnet' -Address 'a' -Order 0 -Path 'relay'
        $peerRelay = New-TestCandidate -Kind 'tailnet' -Address 'b' -Order 1 -Path 'peer-relay'
        $direct = New-TestCandidate -Kind 'tailnet' -Address 'c' -Order 2 -Path 'direct'
        (Select-FleetRecommendation -Candidates @($relay, $peerRelay, $direct) -OnHomeLan $false).Address | Should -Be 'c'
        (Select-FleetRecommendation -Candidates @($relay, $peerRelay) -OnHomeLan $false).Address | Should -Be 'b'
        $dns = New-TestCandidate -Kind 'magicdns' -Address 'name' -Order 3 -Path 'direct'
        $ip = New-TestCandidate -Kind 'tailnet' -Address 'ip' -Order 2 -Path 'direct'
        (Select-FleetRecommendation -Candidates @($dns, $ip) -OnHomeLan $false).Address | Should -Be 'ip'
    }

    # Protects: "no recommendation" must be reported as such, not as a skipped or failed candidate.
    # Detects: skipped or unreachable candidates being chosen.
    # Needs: nothing.
    # Breadcrumb: Select-FleetRecommendation
    It 'returns nothing when no candidate is reachable and eligible' {
        $candidates = @(
            (New-TestCandidate -Kind 'lan' -Reachable $true -Skipped 'not on the home LAN')
            (New-TestCandidate -Kind 'tailnet' -Reachable $false -Path 'direct')
        )
        Select-FleetRecommendation -Candidates $candidates -OnHomeLan $true | Should -BeNullOrEmpty
        Select-FleetRecommendation -Candidates @() -OnHomeLan $true | Should -BeNullOrEmpty
    }
}

Describe 'Whole run on the home LAN' -Tag 'Unit' {
    BeforeAll {
        Mock Get-FleetNeighbor { New-TestNeighbor }
        Mock Find-FleetRoute {
            [pscustomobject]@{ Interface = $(if ($Address -like '100.64.*') { 'Tailscale' } else { 'Ethernet 12' }); InterfaceIndex = $(if ($Address -like '100.64.*') { 3 } else { 12 }); SourceAddress = '192.0.2.200'; NextHop = 'on-link' }
        }
        Mock Resolve-FleetDnsNames { Get-TestMagicDns -Names $Names }
        Mock Get-FleetTailscaleStatus { [pscustomobject]@{ Available = $true; Error = $null; Status = $script:TailscaleFixture } }
        $script:Json = Invoke-FleetPathTest -InventoryPath $script:InventoryPath -BudgetMs 500 -Connector $script:ConnectAllButMinipcLan -AsJson
        $script:Report = $script:Json | ConvertFrom-Json
    }

    # Protects: consumers of ~/.machine/logs parse this document; its shape is the contract.
    # Detects: a renamed, dropped or too-deeply-nested field (ConvertTo-Json defaults to depth 2).
    # Needs: mocked wrappers and an injected connector; no network.
    # Breadcrumb: Invoke-FleetPathTest report assembly
    It 'emits JSON that round-trips with the documented top-level schema' {
        $script:Json | Should -BeOfType [string]
        $script:Report.Schema | Should -Be 'pcai.fleet-path/1'
        $script:Report.PSObject.Properties.Name | Should -Be @(
            'Schema', 'GeneratedAt', 'ComputerName', 'BudgetMs', 'ElapsedMs', 'InventoryPath', 'InventoryMeasured',
            'Location', 'Tailscale', 'Targets', 'Summary')
        # ConvertFrom-Json turns ISO dates into local DateTime, so check the UTC form in the raw text.
        $script:Json | Should -Match '"GeneratedAt": "\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z"'
        $script:Report.BudgetMs | Should -Be 500
        $script:Report.Location.State | Should -Be 'home-lan'
        $script:Report.Tailscale.Available | Should -BeTrue
        $script:Report.Tailscale.BackendState | Should -Be 'Running'
    }

    # Protects: nested detail survives serialization (candidate rows sit four levels down).
    # Detects: a depth truncation that would flatten candidates to type names.
    # Needs: the report from BeforeAll.
    # Breadcrumb: Invoke-FleetPathTest -AsJson
    It 'keeps every candidate field at full depth' {
        $nas = $script:Report.Targets | Where-Object Name -EQ 'nas'
        $nas.Endpoints.Count | Should -Be 4
        $candidate = $nas.Endpoints[0].Candidates[0]
        $candidate.PSObject.Properties.Name | Should -Be @(
            'Kind', 'Address', 'Port', 'Order', 'Reachable', 'LatencyMs', 'Skipped', 'Error', 'RemoteAddress', 'Resolved', 'ProbeAddress',
            'Interface', 'SourceAddress', 'NextHop', 'TailscalePath')
        $candidate.Kind | Should -Be 'lan'
        $candidate.Reachable | Should -BeTrue
        $candidate.Interface | Should -Be 'Ethernet 12'
        $tailnet = $nas.Endpoints[0].Candidates | Where-Object Kind -EQ 'tailnet'
        $tailnet.Interface | Should -Be 'Tailscale'
        $tailnet.TailscalePath.State | Should -Be 'relay'
        $tailnet.TailscalePath.Region | Should -Be 'home'
    }

    # Protects: the NAS tailnet path state, and the per-target recommendation.
    # Detects: the NAS path state or the LAN preference regressing.
    # Needs: the report from BeforeAll.
    # Breadcrumb: Invoke-FleetPathTest target assembly
    It 'shows the NAS tailnet path as relay home and recommends its LAN address' {
        $nas = $script:Report.Targets | Where-Object Name -EQ 'nas'
        $nas.Tailscale.Display | Should -Be 'relay "home"'
        $nas.Recommended.Kind | Should -Be 'lan'
        $nas.Recommended.Address | Should -Be '198.51.100.20'
    }

    # Protects: minipc's LAN address is blocked from the local subnet; the report must show why and fall back to the tailnet.
    # Detects: a failed candidate hiding the working one, or its error text being lost.
    # Needs: the connector refusing 203.0.113.75.
    # Breadcrumb: Select-FleetRecommendation
    It 'reports a refused LAN candidate with its error and recommends the direct tailnet path' {
        $minipc = $script:Report.Targets | Where-Object Name -EQ 'minipc'
        $lan = $minipc.Endpoints[0].Candidates | Where-Object Kind -EQ 'lan'
        $lan.Reachable | Should -BeFalse
        $lan.Error | Should -Be 'ConnectionRefused'
        $minipc.Endpoints[0].Recommended.Kind | Should -Be 'tailnet'
        $minipc.Endpoints[0].Recommended.Path | Should -Be 'direct'
    }

    # Protects: the agent-bus hub is reachable on the tailnet only.
    # Detects: LAN candidates appearing for a tailnet-only service.
    # Needs: the report from BeforeAll.
    # Breadcrumb: New-FleetCandidatePlan
    It 'gives a tailnet-only service no LAN candidate' {
        $work = $script:Report.Targets | Where-Object Name -EQ 'workstation'
        $bus = $work.Endpoints | Where-Object Service -EQ 'agentbus'
        @($bus.Candidates.Kind) | Should -Be @('tailnet')
        $bus.Recommended.Address | Should -Be '100.64.0.102'
    }

    # Protects: the flat summary that operators read in a table, one row per target, service and port.
    # Detects: a row count that drifts from the inventory, or a missing recommendation cell.
    # Needs: the report from BeforeAll.
    # Breadcrumb: Invoke-FleetPathTest Summary
    It 'summarizes one row per target service and port' {
        $script:Report.Summary.Count | Should -Be 18
        $script:Report.Summary[0].PSObject.Properties.Name | Should -Be @('Target', 'Service', 'Port', 'Recommended', 'LatencyMs', 'Path', 'Interface')
        @($script:Report.Summary | Where-Object { $_.Recommended -eq 'none' }).Count | Should -Be 0
        ($script:Report.Summary | Where-Object { $_.Target -eq 'router' -and $_.Service -eq 'https' }).Recommended | Should -Be 'lan 198.51.100.1'
    }

    # Protects: -Target and -Service narrow the run; a typo is an error, not an empty success.
    # Detects: filters being ignored or an unknown target passing silently.
    # Needs: mocked wrappers and the connector.
    # Breadcrumb: Invoke-FleetPathTest
    It 'filters by -Target and -Service, and rejects an unknown target' {
        $one = Invoke-FleetPathTest -InventoryPath $script:InventoryPath -BudgetMs 500 -Connector $script:ConnectAll -Target 'NAS' -Service 'smb'
        @($one.Targets).Count | Should -Be 1
        @($one.Summary).Count | Should -Be 1
        $one.Summary[0].Service | Should -Be 'smb'
        { Invoke-FleetPathTest -InventoryPath $script:InventoryPath -Connector $script:ConnectAll -Target 'nope' } | Should -Throw -ExpectedMessage "*Unknown target 'nope'*"
    }
}

Describe 'Whole run on a colliding network' -Tag 'Unit' {
    BeforeAll {
        # A hotel network: 192.0.2.1 answers and every address "connects", but the gateway MAC is not ours.
        Mock Get-FleetNeighbor { New-TestNeighbor -Mac '00-11-22-33-44-55' }
        Mock Find-FleetRoute { [pscustomobject]@{ Interface = 'Wi-Fi'; InterfaceIndex = 7; SourceAddress = '192.0.2.77'; NextHop = '192.0.2.1' } }
        Mock Resolve-FleetDnsNames { Get-TestMagicDns -Names $Names }
        Mock Get-FleetTailscaleStatus { [pscustomobject]@{ Available = $true; Error = $null; Status = $script:TailscaleFixture } }
        $script:Hotel = Invoke-FleetPathTest -InventoryPath $script:InventoryPath -BudgetMs 500 -Connector $script:ConnectAll
    }

    # Protects: the whole point of the MAC check. Every probe succeeds here, as it would on a hotel LAN that
    # reuses the home subnet (documentation addresses stand in for it; only the MAC differs, which is what the collision case is about), and still no LAN address may be probed or recommended.
    # Detects: LAN candidates being connected to or recommended off the home LAN.
    # Needs: a connector that would connect to everything, so a probed LAN candidate would show Reachable.
    # Breadcrumb: Invoke-FleetPathTest wave B; Resolve-FleetLocation
    It 'skips every LAN candidate and recommends tailnet paths only' {
        $script:Hotel.Location.State | Should -Be 'off-lan'
        $script:Hotel.Location.Collision | Should -BeTrue
        foreach ($target in $script:Hotel.Targets) {
            foreach ($endpoint in $target.Endpoints) {
                foreach ($candidate in ($endpoint.Candidates | Where-Object Kind -EQ 'lan')) {
                    $candidate.Skipped | Should -Be 'not on the home LAN'
                    $candidate.Reachable | Should -BeFalse
                }
                if ($endpoint.Recommended) { $endpoint.Recommended.Kind | Should -Not -Be 'lan' }
            }
        }
        ($script:Hotel.Summary | Where-Object Target -EQ 'workstation' | Where-Object Service -EQ 'ssh' | Select-Object -First 1).Recommended | Should -Be 'tailnet 100.64.0.102'
    }

    # Protects: the router is LAN-only; off the home LAN it has no path and must say so.
    # Detects: a "none" being replaced by a guess.
    # Needs: the report from BeforeAll.
    # Breadcrumb: Select-FleetRecommendation
    It 'reports no recommendation for the LAN-only router' {
        ($script:Hotel.Summary | Where-Object Target -EQ 'router' | Select-Object -First 1).Recommended | Should -Be 'none'
        ($script:Hotel.Targets | Where-Object Name -EQ 'router').Tailscale.State | Should -Be 'none'
    }
}

Describe 'Whole run without tailscale' -Tag 'Unit' {
    BeforeAll {
        Mock Get-FleetNeighbor { New-TestNeighbor }
        Mock Find-FleetRoute { [pscustomobject]@{ Interface = 'Ethernet 12'; InterfaceIndex = 12; SourceAddress = '192.0.2.200'; NextHop = 'on-link' } }
        Mock Resolve-FleetDnsNames { Get-TestMagicDns -Names $Names }
        Mock Get-FleetTailscaleStatus { [pscustomobject]@{ Available = $false; Error = 'tailscale binary not found'; Status = $null } }
        $script:NoTs = Invoke-FleetPathTest -InventoryPath $script:InventoryPath -BudgetMs 500 -Connector $script:ConnectAll -Target 'workstation'
    }

    # Protects: without tailscale the tool still reports reachability and the home recommendation.
    # Detects: a null Status crashing path parsing, or tailnet candidates vanishing.
    # Needs: mocked Get-FleetTailscaleStatus returning Available=false.
    # Breadcrumb: Get-FleetTailscaleStatus; Resolve-FleetTailscalePath
    It 'still probes tailnet candidates and marks their path unavailable' {
        $script:NoTs.Tailscale.Available | Should -BeFalse
        $script:NoTs.Tailscale.Error | Should -Match 'not found'
        $script:NoTs.Tailscale.BackendState | Should -BeNullOrEmpty
        $ssh = $script:NoTs.Targets[0].Endpoints | Where-Object { $_.Service -eq 'ssh' -and $_.Port -eq 22 }
        $tailnet = $ssh.Candidates | Where-Object Kind -EQ 'tailnet'
        $tailnet.Reachable | Should -BeTrue
        $tailnet.TailscalePath.State | Should -Be 'unavailable'
        $ssh.Recommended.Kind | Should -Be 'lan'
    }
}

Describe 'Script entry point' -Tag 'Unit' {
    # Protects: the documented parameters and their validation.
    # Detects: a renamed parameter or a Service value outside the five documented services.
    # Needs: the script's parameter metadata only; nothing runs.
    # Breadcrumb: Test-FleetPath.ps1 param block
    It 'exposes the documented parameters with a validated -Service' {
        $parameters = (Get-Command -Name $script:ScriptPath).Parameters
        foreach ($name in 'Target', 'Service', 'BudgetMs', 'InventoryPath', 'AsJson') { $parameters.Keys | Should -Contain $name }
        $set = $parameters['Service'].Attributes | Where-Object { $_ -is [System.Management.Automation.ValidateSetAttribute] }
        $set.ValidValues | Should -Be @('ssh', 'smb', 'rdp', 'https', 'agentbus')
        $range = $parameters['BudgetMs'].Attributes | Where-Object { $_ -is [System.Management.Automation.ValidateRangeAttribute] }
        $range.MinRange | Should -BeGreaterThan 0
    }

    # Protects: running the script directly fails loudly on bad input, before any network I/O.
    # Detects: a wrong -InventoryPath or unknown -Target exiting 0.
    # Needs: a child pwsh; both cases throw before any probe, so no network is used. The unknown-target case passes the
    # committed example inventory explicitly: the live inventory is private and absent on a clean CI runner.
    # Breadcrumb: Test-FleetPath.ps1 main block
    It 'exits nonzero for <Case>' -ForEach @(
        @{ Case = 'a missing inventory'; Extra = @('-InventoryPath', 'does-not-exist.json'); Pattern = 'not found' }
        @{ Case = 'an unknown target'; Extra = @('-InventoryPath', (Join-Path $PSScriptRoot '../../Tools/SystemScripts/Networking/fleet-paths.example.json'), '-Target', 'no-such-host'); Pattern = 'Unknown target' }
    ) {
        $output = & ([Environment]::ProcessPath) -NoProfile -File $script:ScriptPath @Extra 2>&1 | Out-String
        $LASTEXITCODE | Should -Not -Be 0
        $output | Should -Match $Pattern
    }
}
