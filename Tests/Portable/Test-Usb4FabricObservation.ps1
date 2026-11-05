#Requires -Version 7.0
[CmdletBinding()]
param(
    [string]$SourcePath = (Join-Path $PSScriptRoot '../../Tools/SystemScripts/Networking/Get-Usb4FabricObservation.ps1'),
    [string[]]$Selector = @()
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# Import the exact provider function AST without running its platform wrapper.
# Host APIs below are fixtures on every OS. No readonly automatic variable, live
# networking setting, HOME or environment value is overwritten by these controls.
$tokens = $null; $parseErrors = $null
$sourceAst = [Management.Automation.Language.Parser]::ParseFile(
    (Resolve-Path -LiteralPath $SourcePath).Path, [ref]$tokens, [ref]$parseErrors)
if ($parseErrors.Count) { throw 'Collector source does not parse.' }
$provider = @($sourceAst.FindAll({ param($node)
    $node -is [Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -ceq 'Get-Usb4FabricHost'
}, $true))
if ($provider.Count -ne 1) { throw 'Expected exactly one real USB4 provider function.' }
. ([scriptblock]::Create($provider[0].Extent.Text))

function Reset-Fixture {
    $script:fixture = @{
        calls = 0; writes = 0; starts = 0; pnpError = ''; adapterError = ''; addressError = ''; neighborError = ''
        adapters = @([pscustomobject]@{
            PnPDeviceID = 'USB4\VEN_045E&PROT_USB4NET\edge-a'; Name = 'Ethernet 14'; ifIndex = 77
            InterfaceGuid = '19a70777-e51c-4b8f-97e7-1dbf1d9f3bd4'; MacAddress = '88-b9-b3-c9-54-dc'
            InterfaceOperationalStatus = 1; AdminStatus = 1
        })
        present = @('USB4\VEN_045E&PROT_USB4NET\edge-a', 'USB4\ROOT_DEVICE_ROUTER\root-a')
        parent = @{ 'USB4\VEN_045E&PROT_USB4NET\edge-a' = 'USB4\ROOT_DEVICE_ROUTER\root-a' }
        addresses = @([pscustomobject]@{ IPAddress = 'fe80::1'; AddressState = 'Preferred' })
        neighbors = @([pscustomobject]@{ IPAddress = 'fe80::2'; State = 'Reachable'; LinkLayerAddress = '10-C5-05-C4-3A-54' })
    }
}
function Get-NetAdapter { [CmdletBinding()]param([switch]$IncludeHidden)
    $script:fixture.calls++; if ($script:fixture.adapterError) { throw $script:fixture.adapterError }
    if (-not $IncludeHidden) { throw 'Inventory omitted hidden/phantom candidates before presence validation.' }
    $script:fixture.adapters
}
function Get-PnpDevice { [CmdletBinding()]param([switch]$PresentOnly)
    $script:fixture.calls++; if ($script:fixture.pnpError) { throw $script:fixture.pnpError }
    if (-not $PresentOnly) { throw 'PnP inventory did not require actual presence.' }
    $script:fixture.present | ForEach-Object { [pscustomobject]@{ InstanceId = $_ } }
}
function Get-PnpDeviceProperty { [CmdletBinding()]param($InstanceId, $KeyName)
    $script:fixture.calls++
    if ($KeyName -cne 'DEVPKEY_Device_Parent') { throw 'Wrong ancestry property selected.' }
    [pscustomobject]@{ Data = $script:fixture.parent[$InstanceId] }
}
function Get-NetIPAddress { [CmdletBinding()]param($InterfaceIndex, $PolicyStore)
    $script:fixture.calls++; if ($script:fixture.addressError) { throw $script:fixture.addressError }
    if ($InterfaceIndex -ne 77 -or $PolicyStore -cne 'ActiveStore') { throw 'Address query lost actual interface scope.' }
    $script:fixture.addresses
}
function Get-NetNeighbor { [CmdletBinding()]param($InterfaceIndex)
    $script:fixture.calls++
    if ($InterfaceIndex -ne 77) { throw 'Neighbor query lost actual interface scope.' }
    if ($script:fixture.neighborError) {
        $category = if ($script:fixture.neighborError -eq 'Denied') {
            [Management.Automation.ErrorCategory]::PermissionDenied
        } else { [Management.Automation.ErrorCategory]::ObjectNotFound }
        $id = switch ($script:fixture.neighborError) {
            'Empty' { 'CmdletizationQuery_NotFound' }
            'SimilarId' { 'CmdletizationQuery_NotFoundOther' }
            'Denied' { 'CimAccessDenied' }
            default { 'CimProviderMissing' }
        }
        $PSCmdlet.ThrowTerminatingError([Management.Automation.ErrorRecord]::new(
            [InvalidOperationException]::new('Fixture neighbor outcome'), $id, $category, $InterfaceIndex))
    }
    $script:fixture.neighbors
}
function Assert-Refused([scriptblock]$Operation, [string]$MessagePattern) {
    try { & $Operation | Out-Null } catch {
        if ($_.Exception.Message -notlike $MessagePattern) { throw }
        return
    }
    throw "Expected specific refusal matching: $MessagePattern"
}

# Protects: provider/native/write-free plan and help entrypoints on all platforms.
# Detects: preflight falling through into collection or ignoring the help aliases.
# Needs: actual script with Collect asserted to keep any regression in mocked host
# API scope; no native child is launched. Breadcrumb: collector's preflight branches.
function Test-DryRunAndHelpAvoidProviders {
    Reset-Fixture
    $script:fixture.adapterError = 'Preflight must not call host providers.'
    function Get-Help { param($Name, [switch]$Detailed) 'fixture-help-result' }
    function Set-Content { $script:fixture.writes++; throw 'Unexpected file write.' }
    function Out-File { $script:fixture.writes++; throw 'Unexpected file write.' }
    function Start-Process { $script:fixture.starts++; throw 'Unexpected native launch.' }
    function Invoke-VigilBoundedProcess { $script:fixture.starts++; throw 'Unexpected bounded child.' }
    $plan = & $SourcePath -DryRun -Collect -Source management-ssh -TtlSeconds 120 -TimeoutSeconds 7
    if ($plan.Schema -cne 'pcai.usb4-fabric-collection-plan/v1' -or -not $plan.ReadOnly -or
        $plan.ChangesNetworking -or $plan.RequiresThunderboltShare -or $plan.TtlSeconds -ne 120 -or
        $plan.Source -cne 'management-ssh' -or $plan.TimeoutSeconds -ne 7) { throw 'DryRun plan is invalid.' }
    foreach ($helpArgument in @('ShowHelp', 'h', 'help')) {
        $helpOptions = @{ $helpArgument = $true; Collect = $true }
        if ((& $SourcePath @helpOptions) -cne 'fixture-help-result') { throw 'Help did not return before host access.' }
    }
    if ($script:fixture.calls -or $script:fixture.writes -or $script:fixture.starts) {
        throw 'Plan/help performed provider, file or native side effects.'
    }
}

# Protects: actual GUID/PnP identity rather than aliases, and present-only inventory.
# Detects: phantom resurrection, alias-only selection and preferred-address leakage.
# Needs: exact provider AST with mocked Windows APIs. Breadcrumb: Get-Usb4FabricHost.
function Test-PresentIdentityExcludesPhantomsAndAliasDecoys {
    Reset-Fixture
    $phantom = $script:fixture.adapters[0].PSObject.Copy(); $phantom.PnPDeviceID = 'USB4\VEN_045E&PROT_USB4NET\phantom'
    $decoy = $script:fixture.adapters[0].PSObject.Copy(); $decoy.PnPDeviceID = 'PCI\ETHERNET\alias-decoy'
    $script:fixture.adapters += @($phantom, $decoy)
    $script:fixture.addresses += @([pscustomobject]@{ IPAddress = 'fe80::deprecated'; AddressState = 'Deprecated' })
    $hostObservation = Get-Usb4FabricHost -ObservationSource management-ssh
    if ($hostObservation.interfaces.Count -ne 1 -or $hostObservation.source -cne 'management-ssh') { throw 'Non-present or alias-only candidate admitted.' }
    $edge = $hostObservation.interfaces[0]
    if ($edge.interface_guid -cne '19a70777-e51c-4b8f-97e7-1dbf1d9f3bd4' -or $edge.interface_index -ne 77 -or
        $edge.mac -cne '88-B9-B3-C9-54-DC' -or $edge.root_router_id -cne 'USB4\ROOT_DEVICE_ROUTER\root-a' -or
        $edge.addresses.Count -ne 1 -or $edge.addresses[0] -cne 'fe80::1' -or -not $edge.up) { throw 'Present identity or address scope changed.' }
}

# Protects: unknown root evidence remains unknown instead of fabricated readiness.
# Detects: trusting a stale parent root string absent from actual PnP presence.
# Needs: root-not-present and no-root fixtures. Breadcrumb: ancestry/presentIds gate.
function Test-UnknownRootRetainsUnqualifiedInterface {
    foreach ($missingKind in @('AbsentRoot', 'NoRoot')) {
        Reset-Fixture
        $script:fixture.present = @($script:fixture.present[0])
        if ($missingKind -eq 'NoRoot') { $script:fixture.parent = @{} }
        $actual = Get-Usb4FabricHost
        if ($actual.interfaces.Count -ne 1 -or $null -ne $actual.interfaces[0].root_router_id) {
            throw 'Missing root was fabricated or its interface silently discarded.'
        }
    }
}

# Protects: provider failure is distinguished from a genuinely undocked machine.
# Detects: swallowing adapter, PnP or address-provider failures as empty success.
# Needs: specific failing host API mocks. Breadcrumb: Stop boundaries in provider.
function Test-ProviderErrorsRefuseEmptySuccess {
    foreach ($field in @('adapterError', 'pnpError', 'addressError')) {
        Reset-Fixture; $script:fixture[$field] = "fixture-$field-failure"
        Assert-Refused { Get-Usb4FabricHost } "*fixture-$field-failure*"
    }
    Reset-Fixture; $script:fixture.adapters = @()
    $actual = Get-Usb4FabricHost
    if ($actual.interfaces.Count -ne 0 -or $script:fixture.calls -ne 1) { throw 'True empty adapter result was not preserved.' }
}

# Protects: only the exact native empty-neighbor outcome is converted to an empty table.
# Detects: similar ID, access-denied or unrelated ObjectNotFound error suppression.
# Needs: real ErrorRecord categories/FQIDs from mocked Get-NetNeighbor. Breadcrumb: catch.
function Test-NeighborNotFoundBoundary {
    Reset-Fixture; $script:fixture.neighborError = 'Empty'
    if ((Get-Usb4FabricHost).interfaces[0].neighbors.Count -ne 0) { throw 'Exact empty table was not retained.' }
    foreach ($outcome in @('Denied', 'SimilarId', 'Other')) {
        Reset-Fixture; $script:fixture.neighborError = $outcome
        Assert-Refused { Get-Usb4FabricHost } '*Fixture neighbor outcome*'
    }
}

# Protects: usable unicast evidence without zero, multicast or malformed cache rows.
# Detects: accepting failed neighbor states or generating invalid native-plan input.
# Needs: one accepted row and explicit zero/multicast/empty/failed controls.
# Breadcrumb: Get-Usb4FabricHost neighbor filtering expression.
function Test-NeighborsFilterUnusableEntries {
    Reset-Fixture
    $script:fixture.neighbors += @(
        [pscustomobject]@{ IPAddress='fe80::3'; State='Reachable'; LinkLayerAddress='00-00-00-00-00-00' },
        [pscustomobject]@{ IPAddress='ff02::1'; State='Permanent'; LinkLayerAddress='33-33-00-00-00-01' },
        [pscustomobject]@{ IPAddress='fe80::4'; State='Reachable'; LinkLayerAddress='' },
        [pscustomobject]@{ IPAddress='fe80::5'; State='Unreachable'; LinkLayerAddress='10-C5-05-C4-3A-55' },
        [pscustomobject]@{ IPAddress='fe80::6'; State='Stale'; LinkLayerAddress='10:c5:05:c4:3a:56' }
    )
    $actual = (Get-Usb4FabricHost).interfaces[0].neighbors
    if ($actual.Count -ne 2 -or $actual[0].address -cne 'fe80::2' -or $actual[1].address -cne 'fe80::6') {
        throw 'Unusable neighbor was admitted or valid unicast evidence discarded.'
    }
}

# Protects: cardinality and ancestry termination bounds under pathological providers.
# Detects: unbounded candidate/address/neighbor results, present-device overload,
# cycles or false root claims after ancestry depth exhaustion. Needs fixture arrays.
# Breadcrumb: Get-Usb4FabricHost bounds and ancestry set/depth loop.
function Test-ProviderCardinalityAndAncestryBounds {
    Reset-Fixture; $script:fixture.adapters = @(1..65 | ForEach-Object { $script:fixture.adapters[0] })
    Assert-Refused { Get-Usb4FabricHost } '*64-interface bound*'
    Reset-Fixture; $script:fixture.present = @(1..16385 | ForEach-Object { 'USB4\fixture\' + $_ })
    Assert-Refused { Get-Usb4FabricHost } '*Present-device inventory exceeds its bound*'
    Reset-Fixture; $script:fixture.addresses = @(1..65 | ForEach-Object { [pscustomobject]@{ IPAddress='fe80::1'; AddressState='Preferred' } })
    Assert-Refused { Get-Usb4FabricHost } '*address/neighbor inventory exceeds its bound*'
    Reset-Fixture; $script:fixture.neighbors = @(1..65 | ForEach-Object { [pscustomobject]@{ IPAddress='fe80::2'; State='Reachable'; LinkLayerAddress='10-C5-05-C4-3A-54' } })
    Assert-Refused { Get-Usb4FabricHost } '*address/neighbor inventory exceeds its bound*'
    Reset-Fixture; $leaf = $script:fixture.adapters[0].PnPDeviceID; $script:fixture.parent[$leaf] = $leaf
    Assert-Refused { Get-Usb4FabricHost } '*PnP ancestry contains a cycle*'
    Reset-Fixture; $previous = $script:fixture.adapters[0].PnPDeviceID
    foreach ($n in 1..17) { $next = 'USB4\NODE\' + $n; $script:fixture.parent[$previous] = $next; $previous = $next }
    $actual = Get-Usb4FabricHost
    if ($actual.interfaces.Count -ne 1 -or $null -ne $actual.interfaces[0].root_router_id -or $script:fixture.calls -gt 21) {
        throw 'Depth exhaustion fabricated a root or exceeded its work budget.'
    }
}

$selectors = @('Test-DryRunAndHelpAvoidProviders', 'Test-PresentIdentityExcludesPhantomsAndAliasDecoys',
    'Test-UnknownRootRetainsUnqualifiedInterface', 'Test-ProviderErrorsRefuseEmptySuccess', 'Test-NeighborNotFoundBoundary',
    'Test-NeighborsFilterUnusableEntries', 'Test-ProviderCardinalityAndAncestryBounds')
if (@($Selector | Where-Object { $_ -cnotin $selectors }).Count) { throw 'Unknown USB4 observation test selector.' }
$passed = 0
foreach ($name in $selectors) {
    if ($Selector.Count -and $name -cnotin $Selector) { continue }
    & $name
    $passed++; Write-Output "PASS $name"
}
Write-Output "USB4 observation controls: $passed passed. Provider APIs were fixtures; no live collection or network changes."
