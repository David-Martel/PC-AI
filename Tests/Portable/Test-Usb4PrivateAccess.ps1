#Requires -Version 7.0
<# .SYNOPSIS Exercise private USB4 access with inert Windows providers. #>
[CmdletBinding()]
param([string]$SourcePath = (Join-Path $PSScriptRoot '../../Tools/SystemScripts/Networking/Install-PcaiUsb4PrivateAccess.ps1'))

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$source = Get-Content -LiteralPath $SourcePath -Raw
$platformGuard = 'if (-not $IsWindows) { throw ''USB4 private access installation requires Windows.'' }'
if (-not $source.Contains($platformGuard)) { throw 'The fixture cannot isolate the platform admission check.' }
$installer = [scriptblock]::Create($source.Replace($platformGuard, ''))
$guid = [guid]::NewGuid()
$ruleName = 'PC-AI-USB4-Private-Echo-' + $guid.ToString('N')
$testDir = [IO.Directory]::CreateTempSubdirectory('pcai-usb4-fixture-')

function Reset-Fixture {
    $script:fixture = @{
        Adapter = [pscustomobject]@{
            InterfaceGuid = $guid.ToString('B'); InterfaceIndex = 42; Name = 'Fixture USB4';
            Status = 'Up'; MacAddress = '02-00-00-00-00-41'; PnPDeviceID = 'USB4\FIXTURE&PROT_USB4NET\SYNTHETIC'
        }
        Address = [pscustomobject]@{ InterfaceIndex = 42; IPAddress = 'fe80::41'; AddressState = 'Preferred' }
        Route = [pscustomobject]@{ InterfaceIndex = 42; DestinationPrefix = 'fe80::/64'; NextHop = '::' }
        Neighbor = [pscustomobject]@{ InterfaceIndex = 42; IPAddress = 'fe80::42'; State = 'Reachable'; LinkLayerAddress = '02-00-00-00-00-42' }
        Profile = [pscustomobject]@{
            InstanceID = $guid.ToString('B'); InterfaceIndex = 42; InterfaceAlias = 'Fixture USB4'
            Name = 'Fixture profile'; NetworkCategory = 'Public'
        }
        UnrelatedProfile = [pscustomobject]@{
            InstanceID = [guid]::NewGuid().ToString('B'); InterfaceIndex = 44; InterfaceAlias = 'Fixture LAN'
            Name = 'Fixture LAN'; NetworkCategory = 'Public'
        }
        SharedAdapter = $null
        SharedProfile = $null
        UnexpectedProfileChange = $false
        Rule = $null
        SuppressEffectiveRule = $false
        ThrowAfterProfileChange = $false
        ChangeProfileBeforeOwnWrite = $false
        Mutations = [Collections.Generic.List[string]]::new()
    }
    $script:options = @{
        AdapterGuid = $guid; LocalMac = '02-00-00-00-00-41'; LocalAddress = 'fe80::41'
        PeerMac = '02-00-00-00-00-42'; PeerAddress = 'fe80::42'
        ReceiptPath = Join-Path $testDir.FullName ($([guid]::NewGuid().ToString('N')) + '.json')
    }
}

function Get-NetAdapter {
    [CmdletBinding()]param([switch]$IncludeHidden)
    $fixture.Adapter
    if ($fixture.SharedAdapter) { $fixture.SharedAdapter }
}
function Get-NetIPAddress {
    [CmdletBinding()]param([int]$InterfaceIndex, [string]$AddressFamily)
    if ($InterfaceIndex -ne 42 -or $AddressFamily -ne 'IPv6') { throw 'Unexpected address lookup in fixture.' }
    $fixture.Address
}
function Find-NetRoute {
    [CmdletBinding()]param([int]$InterfaceIndex, [string]$LocalIPAddress, [string]$RemoteIPAddress)
    if ($InterfaceIndex -ne 42 -or $LocalIPAddress -ne 'fe80::41%42' -or $RemoteIPAddress -ne 'fe80::42%42') {
        throw 'Unscoped route lookup in fixture.'
    }
    [pscustomobject]@{ InterfaceIndex = 42; IPAddress = 'fe80::41' }
    $fixture.Route
}
function Get-NetNeighbor {
    [CmdletBinding()]param([int]$InterfaceIndex, [string]$AddressFamily)
    if ($InterfaceIndex -ne 42 -or $AddressFamily -ne 'IPv6') { throw 'Unexpected neighbor lookup in fixture.' }
    $fixture.Neighbor
}
function Get-NetConnectionProfile {
    [CmdletBinding()]param([int]$InterfaceIndex)
    if ($PSBoundParameters.ContainsKey('InterfaceIndex') -and $InterfaceIndex -ne 42) { throw 'Unexpected profile lookup in fixture.' }
    $fixture.Profile.PSObject.Copy()
    if (-not $PSBoundParameters.ContainsKey('InterfaceIndex')) {
        $fixture.UnrelatedProfile.PSObject.Copy()
        if ($fixture.SharedProfile) { $fixture.SharedProfile.PSObject.Copy() }
    }
}
function Set-NetConnectionProfile {
    [CmdletBinding()]param($InputObject, [string]$NetworkCategory)
    if ($InputObject.InterfaceIndex -ne 42 -or $InputObject.InterfaceAlias -cne 'Fixture USB4') { throw 'Write outside fixture profile.' }
    $fixture.Profile.NetworkCategory = $NetworkCategory
    if ($fixture.SharedProfile) { $fixture.SharedProfile.NetworkCategory = $NetworkCategory }
    if ($fixture.UnexpectedProfileChange) { $fixture.UnrelatedProfile.NetworkCategory = $NetworkCategory }
    $fixture.Mutations.Add("Profile:$NetworkCategory")
    if ($fixture.ThrowAfterProfileChange -and $NetworkCategory -eq 'Private') { throw 'Provider failed after setting profile.' }
}
function Get-NetFirewallRule {
    [CmdletBinding()]param([string]$PolicyStore, [string]$Name)
    if ($PolicyStore -notin @('PersistentStore','ActiveStore') -or $Name -ne "$ruleName*") { throw 'Wrong store or rule lookup in fixture.' }
    if ($fixture.Rule -and -not ($PolicyStore -eq 'ActiveStore' -and $fixture.SuppressEffectiveRule)) { $fixture.Rule }
}
function Get-NetFirewallPortFilter {
    [CmdletBinding()]param([Parameter(ValueFromPipeline)]$Rule)
    process { $Rule.PortFilter }
}
function Get-NetFirewallAddressFilter {
    [CmdletBinding()]param([Parameter(ValueFromPipeline)]$Rule)
    process { $Rule.AddressFilter }
}
function Get-NetFirewallInterfaceFilter {
    [CmdletBinding()]param([Parameter(ValueFromPipeline)]$Rule)
    process { $Rule.InterfaceFilter }
}
function Get-NetFirewallApplicationFilter {
    [CmdletBinding()]param([Parameter(ValueFromPipeline)]$Rule)
    process { $Rule.ApplicationFilter }
}
function Get-NetFirewallServiceFilter {
    [CmdletBinding()]param([Parameter(ValueFromPipeline)]$Rule)
    process { $Rule.ServiceFilter }
}
function New-NetFirewallRule {
    [CmdletBinding()]param(
        [string]$PolicyStore, [string]$Name, [string]$DisplayName, [string]$Group,
        [string]$Description, [string]$Direction, [string]$Action, [string]$Enabled, [string]$Profile,
        [string]$EdgeTraversalPolicy, [string]$Protocol, [string]$IcmpType, [string]$InterfaceAlias,
        [string]$LocalAddress, [string]$RemoteAddress
    )
    if ($PolicyStore -cne 'PersistentStore' -or $Name -cne $ruleName -or $DisplayName -cne $ruleName -or
        $Group -cne 'PC-AI USB4 Private Access' -or $Direction -cne 'Inbound' -or $Action -cne 'Allow' -or
        $Enabled -cne 'True' -or $Profile -cne 'Private' -or $EdgeTraversalPolicy -cne 'Block' -or
        $Protocol -cne 'ICMPv6' -or $IcmpType -cne '128' -or $InterfaceAlias -cne 'Fixture USB4' -or
        $LocalAddress -cne 'fe80::41' -or $RemoteAddress -cne 'fe80::42') {
        throw 'The installer tried to broaden the firewall rule.'
    }
    if (-not [IO.File]::Exists($options.ReceiptPath)) { throw 'The rule was changed before the receipt was recorded.' }
    $fixture.Rule = [pscustomobject]@{
        Name = $Name; Description = $Description; Group = $Group; Direction = $Direction; Action = $Action
        Enabled = $Enabled; Profile = $Profile; EdgeTraversalPolicy = $EdgeTraversalPolicy
        PortFilter = [pscustomobject]@{ Protocol = 58; IcmpType = @('128:Any') }
        AddressFilter = [pscustomobject]@{ LocalAddress = @('fe80::41/128'); RemoteAddress = @('fe80::42') }
        InterfaceFilter = [pscustomobject]@{ InterfaceAlias = @($InterfaceAlias) }
        ApplicationFilter = [pscustomobject]@{ Program = 'Any' }
        ServiceFilter = [pscustomobject]@{ Service = 'Any' }
    }
    $fixture.Mutations.Add('NewRule')
    if ($fixture.ChangeProfileBeforeOwnWrite) { $fixture.Profile.NetworkCategory = 'Private' }
    $fixture.Rule
}
function Remove-NetFirewallRule {
    [CmdletBinding()]param([string]$PolicyStore, [string]$Name)
    if ($PolicyStore -cne 'PersistentStore' -or $Name -cne $ruleName -or $fixture.Rule.Name -cne $Name) {
        throw 'Deletion outside the installed fixture rule.'
    }
    $fixture.Rule = $null
    $fixture.Mutations.Add('RemoveRule')
}

function Assert-ReceiptState([string]$Expected) {
    $record = Get-Content -LiteralPath $options.ReceiptPath -Raw | ConvertFrom-Json
    if ($record.state -cne $Expected) { throw "Expected receipt state $Expected, got $($record.state)." }
}

function Assert-FixtureFailure([scriptblock]$Action, [string]$Expected) {
    try { & $Action | Out-Null }
    catch {
        if ($_.Exception.Message -like "*$Expected*") { return }
        throw
    }
    throw "The expected failure did not occur: $Expected."
}

# Protects default, explicit DryRun and WhatIf paths from file and network mutation.
# Detects a receipt or profile/firewall write from a preview. Needs inert providers.
# Breadcrumb: Install-PcaiUsb4PrivateAccess.ps1 top-level plan and ShouldProcess.
function Test-PreviewHasNoSideEffects {
    Reset-Fixture
    foreach ($preview in @((& $installer @options), (& $installer @options -DryRun), (& $installer @options -Apply -WhatIf))) {
        if ($preview.State -ne 'Planned' -or $preview.InterfaceIndex -ne 42 -or -not $preview.WillCreateRule) {
            throw 'Preview did not describe the expected installation.'
        }
    }
    if ($fixture.Mutations.Count -or [IO.File]::Exists($options.ReceiptPath)) { throw 'Preview wrote a receipt or changed host state.' }
}

# Protects exact USB4 identity, on-link routing and the enrolled source and peer.
# Detects non-USB4 hardware, a non-preferred address, off-link routing and an
# unexpected peer or profile. Needs inert providers. Breadcrumb: Get-PcaiUsb4AccessLink.
function Test-InvalidLinkCannotApply {
    foreach ($case in @(
        @{ Row = 'Adapter'; Key = 'PnPDeviceID'; Value = 'PCI\FIXTURE'; Error = 'USB4NET adapter' }
        @{ Row = 'Address'; Key = 'AddressState'; Value = 'Deprecated'; Error = 'must be preferred' }
        @{ Row = 'Route'; Key = 'NextHop'; Value = 'fe80::ff'; Error = 'on-link route' }
        @{ Row = 'Neighbor'; Key = 'LinkLayerAddress'; Value = '02-00-00-00-00-aa'; Error = 'enrolled peer IP and MAC' }
        @{ Row = 'Profile'; Key = 'NetworkCategory'; Value = 'DomainAuthenticated'; Error = 'Public or Private' }
    )) {
        Reset-Fixture
        $fixture[$case.Row].($case.Key) = $case.Value
        Assert-FixtureFailure { & $installer @options -Apply -Confirm:$false } $case.Error
        if ($fixture.Mutations.Count -or [IO.File]::Exists($options.ReceiptPath)) { throw 'Invalid link caused a write.' }
    }
}

# Protects one scoped persistent rule, repeat apply and recovery to the original state.
# Detects broad or duplicate rules, missing custody and an unrestored profile. Needs
# inert providers and temporary receipts. Breadcrumb: installer apply and rollback paths.
function Test-ApplyIsIdempotentAndRollbackRestoresProfile {
    Reset-Fixture
    $result = & $installer @options -Apply -Confirm:$false
    Assert-ReceiptState Applied
    if ($result.State -ne 'Applied' -or $fixture.Profile.NetworkCategory -ne 'Private' -or
        ($fixture.Mutations -join ',') -ne 'NewRule,Profile:Private') { throw 'Applied state did not verify.' }
    if ((& $installer @options -Apply -Confirm:$false).State -ne 'Unchanged' -or $fixture.Mutations.Count -ne 2) {
        throw 'Repeat apply rewrote the rule or the profile.'
    }
    if ((& $installer @options -Rollback -Confirm:$false).State -ne 'Reverted' -or
        $fixture.Rule -or $fixture.Profile.NetworkCategory -ne 'Public' -or
        ($fixture.Mutations -join ',') -ne 'NewRule,Profile:Private,RemoveRule,Profile:Public') {
        throw 'Rollback did not restore the exact preimage.'
    }
    Assert-ReceiptState Reverted
}

# Protects externally changed rules and receipts from overwrite or removal.
# Detects widened rule acceptance and destructive rollback of a drifted filter.
# Needs inert providers and a temporary receipt. Breadcrumb: Get-PcaiUsb4AccessRule.
function Test-DriftBlocksWrites {
    Reset-Fixture
    & $installer @options -Apply -Confirm:$false | Out-Null
    $fixture.Rule.AddressFilter.RemoteAddress = @('fe80::/64')
    Assert-FixtureFailure { & $installer @options -Apply -Confirm:$false } 'unexpected USB4 filters'
    Assert-FixtureFailure { & $installer @options -Rollback -Confirm:$false } 'unexpected USB4 filters'
    if ($fixture.Mutations.Count -ne 2 -or $fixture.Profile.NetworkCategory -ne 'Private') {
        throw 'Rule drift did not preserve current state.'
    }
    Assert-ReceiptState Applied
}

# Protects the prior profile and owned rule after a partial or ineffective install.
# Detects leaked changes after a setter fails or effective policy rejects a rule.
# Needs inert providers and temporary receipts. Breadcrumb: installer recovery catch.
function Test-PartialApplyRestoresOwnedChanges {
    foreach ($fault in 'ThrowAfterProfileChange', 'SuppressEffectiveRule') {
        Reset-Fixture
        $fixture[$fault] = $true
        $expected = if ($fault -eq 'ThrowAfterProfileChange') { 'Provider failed' } else { 'did not both verify' }
        Assert-FixtureFailure { & $installer @options -Apply -Confirm:$false } $expected
        if ($fixture.Rule -or $fixture.Profile.NetworkCategory -ne 'Public') { throw 'Partial apply left configuration behind.' }
        Assert-ReceiptState Reverted
    }
}

# Protects someone else's profile update during this transaction.
# Detects restoring Public when the installer never attempted the profile write.
# Needs inert providers and a temporary receipt. Breadcrumb: Undo-PcaiUsb4PrivateAccess.
function Test-RecoveryPreservesUnownedProfileChange {
    Reset-Fixture
    $fixture.ChangeProfileBeforeOwnWrite = $true
    Assert-FixtureFailure { & $installer @options -Apply -Confirm:$false } 'changed outside this transaction'
    if ($fixture.Rule -or $fixture.Profile.NetworkCategory -ne 'Private' -or $fixture.Mutations -contains 'Profile:Public') {
        throw 'Recovery changed a profile that it did not own.'
    }
    Assert-ReceiptState RollingBack
}

# Protects every adapter attached to a potentially shared Windows network profile.
# Detects unacknowledged sibling reclassification or unauthorized non-USB4
# adapters. Needs inert providers and temporary receipts. Breadcrumb: Get-PcaiUsb4ProfileScope.
function Test-SharedProfileRequiresExplicitUsb4Scope {
    Reset-Fixture
    $otherGuid = [guid]::NewGuid()
    $fixture.SharedAdapter = [pscustomobject]@{
        InterfaceGuid = $otherGuid.ToString('B'); InterfaceIndex = 43; Name = 'Fixture USB4 sibling'
        Status = 'Up'; PnPDeviceID = 'PCI\FIXTURE'; MacAddress = '02-00-00-00-00-43'
    }
    $fixture.SharedProfile = [pscustomobject]@{
        InstanceID = $otherGuid.ToString('B'); InterfaceIndex = 43; InterfaceAlias = 'Fixture USB4 sibling'
        Name = 'Fixture profile'; NetworkCategory = 'Public'
    }
    Assert-FixtureFailure { & $installer @options -Apply -Confirm:$false } 'SharedAdapterGuid'
    $options.SharedAdapterGuid = @($otherGuid)
    Assert-FixtureFailure { & $installer @options -Apply -Confirm:$false } 'unique Up USB4NET'
    if ($fixture.Mutations.Count -or [IO.File]::Exists($options.ReceiptPath)) { throw 'Missing or invalid scope caused mutation.' }
    $fixture.SharedAdapter.PnPDeviceID = 'USB4\FIXTURE&PROT_USB4NET\SIBLING'
    & $installer @options -Apply -Confirm:$false | Out-Null
    if ($fixture.SharedProfile.NetworkCategory -ne 'Private' -or $fixture.UnrelatedProfile.NetworkCategory -ne 'Public') {
        throw 'Shared category did not stay within the enrolled adapters.'
    }
    & $installer @options -Rollback -Confirm:$false | Out-Null
    if ($fixture.SharedProfile.NetworkCategory -ne 'Public' -or $fixture.Profile.NetworkCategory -ne 'Public' -or $fixture.Rule) {
        throw 'Shared category rollback did not restore both preimages.'
    }
    Assert-ReceiptState Reverted
}

# Protects unrelated profiles after Windows unexpectedly expands a category mutation.
# Detects admitting collateral category changes or leaving an owned scoped rule behind.
# Needs inert providers and temporary receipts. Breadcrumb: Assert-PcaiUsb4ProfileReadback.
function Test-UnexpectedCategoryChangeRollsBack {
    Reset-Fixture
    $fixture.UnexpectedProfileChange = $true
    Assert-FixtureFailure { & $installer @options -Apply -Confirm:$false } 'outside the authorized profile scope'
    if ($fixture.Profile.NetworkCategory -ne 'Public' -or $fixture.UnrelatedProfile.NetworkCategory -ne 'Public' -or $fixture.Rule) {
        throw 'Rollback did not restore category readback after out-of-scope mutation.'
    }
    Assert-ReceiptState Reverted
}

try {
    foreach ($test in 'Test-PreviewHasNoSideEffects', 'Test-InvalidLinkCannotApply', 'Test-ApplyIsIdempotentAndRollbackRestoresProfile',
        'Test-DriftBlocksWrites', 'Test-PartialApplyRestoresOwnedChanges', 'Test-RecoveryPreservesUnownedProfileChange',
        'Test-SharedProfileRequiresExplicitUsb4Scope', 'Test-UnexpectedCategoryChangeRollsBack') {
        & $test
        "PASS: $test"
    }
} finally {
    foreach ($file in $testDir.EnumerateFiles()) { [IO.File]::Delete($file.FullName) }
    [IO.Directory]::Delete($testDir.FullName)
}
