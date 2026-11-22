#Requires -Version 7.0
<# .SYNOPSIS Verify authoritative HNS endpoint inventory response shapes. #>
[CmdletBinding()]
param([string]$SourcePath = (Join-Path $PSScriptRoot '../../Tools/SystemScripts/Networking/Invoke-Usb4PeerNetwork.ps1'))

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
. $SourcePath

function Get-NetNat {
    [CmdletBinding()]
    param()
    [pscustomobject]@{
        Name = 'fixture'; InternalIPInterfaceAddressPrefix = '198.51.100.0/24'; ExternalIPInterfaceAddressPrefix = $null
        Active = $true; IcmpQueryTimeout = 30; TcpEstablishedConnectionTimeout = 1800; TcpTransientConnectionTimeout = 120
        TcpFilteringBehavior = 'AddressDependentFiltering'; UdpFilteringBehavior = 'AddressDependentFiltering'
        UdpIdleSessionTimeout = 120; UdpInboundRefresh = $false; Store = 'Local'
    }
}

function Get-HnsNetwork {
    [CmdletBinding()]
    param()
    [pscustomobject]@{ Id = '10000000-0000-0000-0000-000000000001'; Type = 'Transparent' }
}

function Get-HnsEndpoint {
    [CmdletBinding()]
    param()
    # The inbox wrapper projects missing Output as one null object.
    $null
}

function Get-NetNatStaticMapping {
    [CmdletBinding()]
    param()
}

function Get-NetNatSession {
    [CmdletBinding()]
    param()
}

function New-Object {
    [CmdletBinding()]
    param([string]$ComObject)
    if ($ComObject -cne 'hns.1') { throw 'Unexpected fixture COM class.' }
    $provider = [pscustomobject]@{}
    $provider | Add-Member -MemberType ScriptMethod -Name Request2 -Value {
        param($Method, $Path, $Body)
        if ($Method -cne 'GET' -or $Path -cne '/endpoints' -or $Body -cne '') { throw 'Unexpected fixture HNS operation.' }
        $script:RequestCount++
        if ($script:TransportFails) { throw [System.Runtime.InteropServices.COMException]::new('HNS fixture transport failure.') }
        $script:Response
    }
    $provider
}

$script:RequestCount = 0
$script:TransportFails = $false

# Protects successful empty enumeration without losing the surrounding NAT inventory.
# Detects wrapper-null rejection, treating an empty collection as a row, and missing GET.
# Needs only inert provider substitutes. Breadcrumb: Get-Usb4ExistingNatInventory.
function Test-Usb4HnsAuthoritativeEmpty {
    foreach ($response in @('{"Success":true}', '{"Success":true,"Output":[]}')) {
        $script:Response = $response
        $before = $script:RequestCount
        $inventory = Get-Usb4ExistingNatInventory
        if (-not $inventory.complete -or $inventory.endpoints.Count -ne 0 -or $inventory.nat.Count -ne 1 -or
            $inventory.networks.Count -ne 1 -or $script:RequestCount -ne $before + 1) {
            throw 'The successful empty HNS response did not reach complete NAT inventory.'
        }
    }
}

# Protects endpoint identity and both native address layouts used for reservation checks.
# Detects lost endpoints, addresses or address deduplication. Needs synthetic JSON only.
# Breadcrumb: Get-Usb4HnsEndpointInventory endpoint normalization.
function Test-Usb4HnsRetainsEndpointReservations {
    $script:Response = @'
{"Success":true,"Output":[{"Id":"10000000-0000-0000-0000-000000000002","IPAddress":"198.51.100.14","IpConfigurations":[{"IpAddress":"198.51.100.14"},{"IPAddress":"2001:db8::1"}]},{"ID":"10000000-0000-0000-0000-000000000003","IPAddress":"","IpConfigurations":[{"IpAddress":"198.51.100.16"}]}]}
'@
    $inventory = Get-Usb4ExistingNatInventory
    if ($inventory.endpoints.Count -ne 2 -or $inventory.endpoints[0].id -cne '10000000-0000-0000-0000-000000000002' -or
        $inventory.endpoints[0].addresses.Count -ne 2 -or '198.51.100.14' -notin $inventory.endpoints[0].addresses -or
        '2001:db8::1' -notin $inventory.endpoints[0].addresses -or $inventory.endpoints[1].id -cne '10000000-0000-0000-0000-000000000003' -or
        ($inventory.endpoints[1].addresses -join ',') -cne '198.51.100.16') {
        throw 'HNS endpoint identity or known address layouts were lost.'
    }
}

# Protects positive inventory authority. Detects treating transport, malformed JSON,
# unknown envelope keys or invalid endpoint rows as empty. Needs synthetic providers only.
# Breadcrumb: Get-Usb4HnsEndpointInventory response checks.
function Test-Usb4HnsRefusesUntrustedResponses {
    $invalid = [ordered]@{
        MissingResponse = $null; BlankResponse = ' '; BrokenJson = '{'; Scalar = 'true'; JsonNull = 'null'
        TopLevelArray = '[{"Success":true}]'; MissingSuccess = '{"Output":[]}'; FalseSuccess = '{"Success":false}'
        StringSuccess = '{"Success":"true"}'; NullSuccess = '{"Success":null}'
        ErrorPresent = '{"Success":true,"Error":null}'; UnknownKey = '{"Success":true,"Other":[]}'
        UnknownKeyWithOutput = '{"Success":true,"Output":[],"Other":[]}'
        NullOutput = '{"Success":true,"Output":null}'; ObjectOutput = '{"Success":true,"Output":{}}'
        StringOutput = '{"Success":true,"Output":"[]"}'; NullEndpoint = '{"Success":true,"Output":[null]}'
        ScalarEndpoint = '{"Success":true,"Output":[42]}'
        MissingIdentity = '{"Success":true,"Output":[{"IPAddress":"198.51.100.14"}]}'
        MissingAddresses = '{"Success":true,"Output":[{"Id":"10000000-0000-0000-0000-000000000002"}]}'
        InvalidAddress = '{"Success":true,"Output":[{"Id":"10000000-0000-0000-0000-000000000002","IPAddress":"unknown"}]}'
        NumericAddress = '{"Success":true,"Output":[{"Id":"10000000-0000-0000-0000-000000000002","IPAddress":2130706433}]}'
        NumericConfigurationAddress = '{"Success":true,"Output":[{"Id":"10000000-0000-0000-0000-000000000002","IpConfigurations":[{"IpAddress":2130706433}]}]}'
        MissingConfigurationAddress = '{"Success":true,"Output":[{"Id":"10000000-0000-0000-0000-000000000002","IpConfigurations":[{}]}]}'
    }
    foreach ($case in $invalid.Keys) {
        $script:Response = $invalid[$case]
        $before = $script:RequestCount
        try { Get-Usb4ExistingNatInventory | Out-Null }
        catch {
            if ($script:RequestCount -eq $before + 1 -and $_.Exception.Message -match 'HNS endpoint') { continue }
            throw
        }
        throw "HNS inventory admitted invalid fixture $case."
    }
}

# Protects native transport errors. Detects reporting a failed COM request as empty.
# Needs only the scripted COM substitute. Breadcrumb: Get-Usb4HnsEndpointInventory Request2.
function Test-Usb4HnsPreservesTransportFailure {
    $script:TransportFails = $true
    try { Get-Usb4ExistingNatInventory | Out-Null }
    catch {
        if ($_.Exception.Message -like '*HNS fixture transport failure.*') { return }
        throw
    }
    finally { $script:TransportFails = $false }
    throw 'A failed HNS request did not fail the inventory.'
}

Test-Usb4HnsAuthoritativeEmpty
Write-Output 'PASS: authoritative empty HNS inventory'
Test-Usb4HnsRetainsEndpointReservations
Write-Output 'PASS: populated HNS endpoint reservations'
Test-Usb4HnsRefusesUntrustedResponses
Write-Output 'PASS: malformed HNS endpoint responses'
Test-Usb4HnsPreservesTransportFailure
Write-Output 'PASS: HNS transport failure propagation'
