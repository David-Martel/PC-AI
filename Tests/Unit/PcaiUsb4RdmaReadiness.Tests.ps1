# Tests for the read-only Microsoft USB4 NetworkDirect preflight.
BeforeAll {
    . (Join-Path $PSScriptRoot '../../Tools/SystemScripts/Networking/Get-PcaiUsb4RdmaReadiness.ps1')
    $script:now = [DateTimeOffset]'2026-11-22T14:00:00Z'
    $script:doc = Get-Content -Raw (Join-Path $PSScriptRoot '../Fixtures/LinkHealth/usb4-route-plan-hub-chain.json') | ConvertFrom-Json -AsHashtable
    $script:doc.evaluated_at_utc = $now.AddSeconds(-10).ToString('o')
    $script:doc.expires_at_utc = $now.AddSeconds(80).ToString('o')
    $script:pair = Get-PcaiUsb4RdmaDirectPair -Plan $doc -LocalHost dtm-p1gen9 -PeerHost dtm-super-nuc -Now $now
    $script:adapter = [pscustomobject]@{
        InterfaceGuid = $pair.InterfaceGuid
        InterfaceIndex = 28
        MacAddress = '02-00-00-09-00-28'
        Status = 'Up'
        PnPDeviceID = 'USB4\VEN_1D6B&DEV_00000004&PROT_USB4NET\1'
    }
    $script:address = [pscustomobject]@{
        IPAddress = '172.31.255.21'
        InterfaceIndex = 28
        PrefixLength = 30
        PrefixOrigin = 'Manual'
        AddressState = 'Preferred'
    }
    $script:route = [pscustomobject]@{
        InterfaceIndex = 28
        NextHop = '0.0.0.0'
        DestinationPrefix = '172.31.255.20/30'
    }
}

Describe 'USB4 NetworkDirect preview preflight' -Tag 'Unit', 'Portable' {
    # Protects: current, directly paired planner output is usable.
    # Detects: choosing the wrong edge address on the second link of a chain.
    # Needs: Windows or Linux pwsh and the planner fixture. Breadcrumb: Get-PcaiUsb4RdmaReadiness.ps1::Get-PcaiUsb4RdmaDirectPair.
    It 'extracts the distinct source and destination of the direct /30 edge' {
        $pair.SourceAddress | Should -Be '172.31.255.21'
        $pair.PeerAddress | Should -Be '172.31.255.22'
        $pair.InterfaceIndex | Should -Be 28
        $pair.Subnet | Should -Be '172.31.255.20/30'
    }

    # Protects: UTC custody survives JSON deserialization on non-UTC Windows hosts.
    # Detects: string conversion of a parsed date losing its offset and rejecting a fresh plan.
    # Needs: actual planner serialization fixture and PowerShell 7.2+. Breadcrumb: Get-PcaiUsb4RdmaReadiness.ps1::Get-PcaiUsb4RdmaDirectPair.
    It 'accepts the unmodified planner JSON times during their observation window' {
        $original = Get-Content -Raw (Join-Path $PSScriptRoot '../Fixtures/LinkHealth/usb4-route-plan-hub-chain.json') | ConvertFrom-Json -AsHashtable
        $fresh = Get-PcaiUsb4RdmaDirectPair -Plan $original -LocalHost dtm-p1gen9 -PeerHost dtm-super-nuc -Now ([DateTimeOffset]'2026-11-21T18:27:00Z')
        $fresh.ExpiresAtUtc | Should -Be '2026-11-21T18:31:37.0000000Z'
    }

    # Protects: an ND link is a fresh direct edge.
    # Detects: a routable TCP hop or expired physical observation authorizing a preview test.
    # Needs: Windows or Linux pwsh and the planner fixture. Breadcrumb: Get-PcaiUsb4RdmaReadiness.ps1::Get-PcaiUsb4RdmaDirectPair.
    It 'refuses both routed paths and expired topology' {
        { Get-PcaiUsb4RdmaDirectPair -Plan $doc -LocalHost dtm-p1gen9 -PeerHost dtm-p1gen7 -Now $now } | Should -Throw '*direct*'
        { Get-PcaiUsb4RdmaDirectPair -Plan $doc -LocalHost dtm-p1gen9 -PeerHost dtm-super-nuc -Now ($now.AddMinutes(2)) } | Should -Throw '*current*'
    }

    # Protects: local routing is bound to the observed USB4NET interface and its manual /30.
    # Detects: a successful LAN route or a replacement adapter being accepted as USB4 evidence.
    # Needs: Windows or Linux pwsh and simulated Windows route/address rows. Breadcrumb: Get-PcaiUsb4RdmaReadiness.ps1::Test-PcaiUsb4RdmaDirectRoute.
    It 'accepts only the identity-matched adapter and on-link source route' {
        Test-PcaiUsb4RdmaDirectRoute -Pair $pair -Adapter @($adapter) -RouteSelection @($address, $route) | Should -BeTrue
        $foreignRoute = [pscustomobject]@{ InterfaceIndex = 99; NextHop = '0.0.0.0'; DestinationPrefix = $route.DestinationPrefix }
        Test-PcaiUsb4RdmaDirectRoute -Pair $pair -Adapter @($adapter) -RouteSelection @($address, $foreignRoute) | Should -BeFalse
        $replacement = [pscustomobject]@{
            InterfaceGuid = [guid]'ffffffff-ffff-ffff-ffff-ffffffffffff'
            InterfaceIndex = 28; MacAddress = $adapter.MacAddress; Status = 'Up'; PnPDeviceID = $adapter.PnPDeviceID
        }
        Test-PcaiUsb4RdmaDirectRoute -Pair $pair -Adapter @($replacement) -RouteSelection @($address, $route) | Should -BeFalse
    }

    # Protects: a capability flag or ordinary Windows build never implies preview availability.
    # Detects: promoting unsigned or undeclared fleet prerequisites to a live RDMA result.
    # Needs: Windows or Linux pwsh and typed mock evidence. Breadcrumb: Get-PcaiUsb4RdmaReadiness.ps1::Get-PcaiUsb4RdmaBlocker.
    It 'keeps undeclared builds and Secure Boot visible as blockers' {
        $blocked = Get-PcaiUsb4RdmaBlocker -Build '26300.10075' -PublishedBuild '' -SecureBoot Enabled -FeatureEnabled $false -RouteStatus Matched
        $blocked | Should -Contain 'PublishedBuildRequired'
        $blocked | Should -Contain 'SecureBootEnabled'
        $blocked | Should -Contain 'InterDomainRdmaDisabledOrUnknown'
        $eligible = Get-PcaiUsb4RdmaBlocker -Build '30000.1' -PublishedBuild '30000.1' -SecureBoot Disabled -FeatureEnabled $true -RouteStatus Matched
        @($eligible).Count | Should -Be 0
    }
}
