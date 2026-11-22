#Requires -Version 7.0
<#
.SYNOPSIS
Configure Private access and scoped IPv6 echo on one enrolled USB4NET link.

.DESCRIPTION
Inspects one explicit link by adapter GUID, local address and both MAC addresses.
The default and -DryRun print a plan. -Apply creates a custody receipt before
changing the connection profile or the local Windows Firewall policy store.

.PARAMETER ReceiptPath
Existing local directory and a JSON filename. Apply reserves a new receipt file.
Rollback uses the original receipt and the same endpoint identity arguments.

.PARAMETER SharedAdapterGuid
Other enrolled USB4NET adapters attached to the potentially shared Windows
profile. Apply requires every same-name connection before changing its category.

.EXAMPLE
./Install-PcaiUsb4PrivateAccess.ps1 @EnrolledLink -Apply -ReceiptPath $receipt

.LINK
../../../docs/usb4/private-access.md
#>
[CmdletBinding(SupportsShouldProcess)]
param(
    [guid]$AdapterGuid,
    [string]$LocalMac,
    [string]$LocalAddress,
    [string]$PeerMac,
    [string]$PeerAddress,
    [guid[]]$SharedAdapterGuid = @(),
    [string]$ReceiptPath,
    [switch]$DryRun,
    [switch]$Apply,
    [switch]$Rollback,
    [Alias('h', '-help')][switch]$Help
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
if ($Help) {
    Get-Help -Name $PSCommandPath -Detailed
    return
}
if (-not $IsWindows) { throw 'USB4 private access installation requires Windows.' }
if ($Apply -and $Rollback -or $DryRun -and ($Apply -or $Rollback)) {
    throw 'Specify one of DryRun, Apply or Rollback.'
}
if ($AdapterGuid -eq [guid]::Empty) { throw 'A non-empty enrolled adapter GUID is required.' }

function ConvertTo-PcaiUsb4MacKey([string]$Value) {
    $key = $Value -replace '[:-]', ''
    if ($key -notmatch '^[0-9a-f]{12}$' -or $key -eq '000000000000') { throw 'A nonzero MAC address is required.' }
    $key.ToUpperInvariant()
}

function ConvertTo-PcaiUsb4LinkAddress([string]$Value) {
    $ip = $null
    if ($Value.Contains('%') -or -not [Net.IPAddress]::TryParse($Value, [ref]$ip) -or -not $ip.IsIPv6LinkLocal) {
        throw 'An unscoped IPv6 link-local address is required.'
    }
    $ip.ToString()
}

$LocalMac = ConvertTo-PcaiUsb4MacKey $LocalMac
$PeerMac = ConvertTo-PcaiUsb4MacKey $PeerMac
$LocalAddress = ConvertTo-PcaiUsb4LinkAddress $LocalAddress
$PeerAddress = ConvertTo-PcaiUsb4LinkAddress $PeerAddress
if ($LocalMac -eq $PeerMac -or $LocalAddress -eq $PeerAddress) { throw 'The local endpoint and peer must be distinct.' }
$ruleName = 'PC-AI-USB4-Private-Echo-' + $AdapterGuid.ToString('N')
$groupName = 'PC-AI USB4 Private Access'
$ownerPattern = '^PC-AI USB4 Private Access [0-9a-f]{32}$'
$schema = 'pcai.usb4.private-access/v2'
$allowedGuids = @($AdapterGuid) + $SharedAdapterGuid
if ($allowedGuids -contains [guid]::Empty -or @($allowedGuids | Sort-Object -Unique).Count -ne $allowedGuids.Count) {
    throw 'The adapter allowlist must contain distinct non-empty GUIDs.'
}
$sharedGuids = @($SharedAdapterGuid | Sort-Object | ForEach-Object { $_.ToString() })

function Get-PcaiUsb4AccessLink {
    $adapters = @(Get-NetAdapter -IncludeHidden | Where-Object { [guid]$_.InterfaceGuid -eq $AdapterGuid })
    if ($adapters.Count -ne 1 -or $adapters[0].Status -ne 'Up' -or
        $adapters[0].PnPDeviceID -notmatch '^USB4\\[^\\]+&PROT_USB4NET\\' -or
        (ConvertTo-PcaiUsb4MacKey $adapters[0].MacAddress) -ne $LocalMac) {
        throw 'Expected one Up USB4NET adapter matching the enrolled GUID and MAC.'
    }
    $index = [int]$adapters[0].InterfaceIndex
    $alias = [string]$adapters[0].Name
    if ($index -le 0 -or -not $alias -or $alias.IndexOfAny([char[]]'*?[]') -ge 0) {
        throw 'Windows must report an exact USB4 interface alias and index.'
    }
    $localRows = @(Get-NetIPAddress -InterfaceIndex $index -AddressFamily IPv6 | Where-Object {
        $_.InterfaceIndex -eq $index -and $_.AddressState -eq 'Preferred' -and
        ($_.IPAddress -split '%', 2)[0] -eq $LocalAddress
    })
    if ($localRows.Count -ne 1) { throw 'The enrolled local address must be preferred on the selected USB4 adapter.' }
    $routes = @(Find-NetRoute -LocalIPAddress "$LocalAddress%$index" -RemoteIPAddress "$PeerAddress%$index" -InterfaceIndex $index)
    $sourceRows = @($routes | Where-Object { $_.PSObject.Properties['IPAddress'] })
    $routeRows = @($routes | Where-Object { $_.PSObject.Properties['DestinationPrefix'] })
    if ($sourceRows.Count -ne 1 -or $routeRows.Count -ne 1 -or
        @($routes | Where-Object InterfaceIndex -NE $index).Count -or
        ($sourceRows[0].IPAddress -split '%', 2)[0] -ne $LocalAddress -or
        $routeRows[0].NextHop -ne '::' -or $routeRows[0].DestinationPrefix -notin @('fe80::/64', "$PeerAddress/128")) {
        throw 'Windows did not select the enrolled USB4 source and an on-link route.'
    }
    $peers = @(Get-NetNeighbor -InterfaceIndex $index -AddressFamily IPv6 | Where-Object {
        $_.InterfaceIndex -eq $index -and ($_.IPAddress -split '%', 2)[0] -eq $PeerAddress
    })
    if ($peers.Count -ne 1 -or $peers[0].State -notin @('Reachable', 'Stale', 'Delay', 'Probe', 'Permanent') -or
        (ConvertTo-PcaiUsb4MacKey $peers[0].LinkLayerAddress) -ne $PeerMac) {
        throw 'Expected the enrolled peer IP and MAC in the USB4 neighbor cache.'
    }
    $profiles = @(Get-NetConnectionProfile -InterfaceIndex $index)
    if ($profiles.Count -ne 1 -or $profiles[0].InterfaceIndex -ne $index -or $profiles[0].InterfaceAlias -cne $alias -or
        [string]$profiles[0].NetworkCategory -notin @('Public', 'Private')) {
        throw 'Expected one Public or Private connection profile on the enrolled USB4 adapter.'
    }
    return [pscustomobject]@{
        Index = $index
        Alias = $alias
        ProfileName = [string]$profiles[0].Name
        ProfileId = ([guid]$profiles[0].InstanceID).ToString()
        Profile = $profiles[0]
        Category = [string]$profiles[0].NetworkCategory
    }
}

function Get-PcaiUsb4ProfileSnapshot {
    $snapshot = @{}
    foreach ($profile in Get-NetConnectionProfile) {
        $id = ([guid]$profile.InstanceID).ToString()
        if ($id -eq [guid]::Empty.ToString() -or $snapshot.ContainsKey($id) -or
            $profile.InterfaceIndex -le 0 -or -not $profile.Name -or -not $profile.InterfaceAlias -or
            [string]$profile.NetworkCategory -notin @('Public', 'Private', 'DomainAuthenticated')) {
            throw 'Windows network profile inventory is ambiguous.'
        }
        $snapshot[$id] = @{
            index = [int]$profile.InterfaceIndex
            alias = [string]$profile.InterfaceAlias
            name = [string]$profile.Name
            category = [string]$profile.NetworkCategory
        }
    }
    return $snapshot
}

function Get-PcaiUsb4ProfileScope([hashtable]$Snapshot, $Link, [switch]$RequireCoverage) {
    $selected = @{}
    foreach ($adapter in Get-NetAdapter -IncludeHidden) {
        $id = [guid]$adapter.InterfaceGuid
        if ($id -notin $allowedGuids) { continue }
        $profiles = @($Snapshot.Keys | Where-Object {
            $Snapshot[$_].index -eq $adapter.InterfaceIndex -and
            $Snapshot[$_].alias -ceq $adapter.Name -and $Snapshot[$_].name -ceq $Link.ProfileName
        })
        if ($adapter.Status -ne 'Up' -or $adapter.PnPDeviceID -notmatch '^USB4\\[^\\]+&PROT_USB4NET\\' -or
            $profiles.Count -ne 1 -or $selected.ContainsKey($profiles[0])) {
            throw 'An authorized adapter has no unique Up USB4NET connection in the selected profile scope.'
        }
        $selected[$profiles[0]] = $true
    }
    if ($selected.Count -ne $allowedGuids.Count -or -not $selected.ContainsKey($Link.ProfileId)) {
        throw 'Windows did not enumerate all explicitly authorized USB4NET profile connections.'
    }
    if ($RequireCoverage) {
        foreach ($id in $Snapshot.Keys) {
            if ($Snapshot[$id].name -ceq $Link.ProfileName -and -not $selected.ContainsKey($id)) {
                throw "Profile '$($Link.ProfileName)' also appears on '$($Snapshot[$id].alias)'. Authorize only its verified USB4 adapter using SharedAdapterGuid."
            }
        }
    }
    return $selected.Keys | Sort-Object
}

function Assert-PcaiUsb4ProfileReadback([hashtable]$Before, [hashtable]$After, [string[]]$MayBecomePrivate = @()) {
    if (-not $Before.Count -or $Before.Count -ne $After.Count) { throw 'The connection profile inventory changed.' }
    foreach ($id in $Before.Keys) {
        $old = $Before[$id]
        $new = $After[$id]
        if (-not $new -or $old.index -ne $new.index -or $old.alias -cne $new.alias -or $old.name -cne $new.name) {
            throw 'A connection profile was replaced or rebound during USB4 configuration.'
        }
        if ($old.category -cne $new.category -and ($id -notin $MayBecomePrivate -or $new.category -ne 'Private')) {
            throw "Connection category changed outside the authorized profile scope on '$($new.alias)'."
        }
    }
}

function Test-PcaiUsb4ExactAddress($Addresses, [string]$Expected) {
    if (@($Addresses).Count -ne 1 -or [string]$Addresses -notmatch '^([^/%]+)(?:/128)?$') { return $false }
    $ip = $null
    return [Net.IPAddress]::TryParse($Matches[1], [ref]$ip) -and $ip.ToString() -eq $Expected
}

function Get-PcaiUsb4AccessRule([string]$Store, [string]$Alias, [string]$Owner = '') {
    $rules = @(Get-NetFirewallRule -PolicyStore $Store -Name "$ruleName*" | Where-Object Name -CEQ $ruleName)
    if (-not $rules.Count) { return }
    if ($rules.Count -ne 1 -or $rules[0].Group -cne $groupName -or $rules[0].Description -cnotmatch $ownerPattern -or
        ($Owner -and $rules[0].Description -cne $Owner) -or [string]$rules[0].Enabled -ne 'True' -or
        [string]$rules[0].Direction -ne 'Inbound' -or [string]$rules[0].Action -ne 'Allow' -or
        [string]$rules[0].Profile -ne 'Private' -or [string]$rules[0].EdgeTraversalPolicy -ne 'Block') {
        throw "Firewall rule '$ruleName' in $Store conflicts with the expected USB4 rule."
    }
    $ports = @($rules[0] | Get-NetFirewallPortFilter)
    $addresses = @($rules[0] | Get-NetFirewallAddressFilter)
    $interfaces = @($rules[0] | Get-NetFirewallInterfaceFilter)
    $application = @($rules[0] | Get-NetFirewallApplicationFilter)
    $services = @($rules[0] | Get-NetFirewallServiceFilter)
    if ($ports.Count -ne 1 -or [string]$ports[0].Protocol -notin @('ICMPv6', '58') -or
        @($ports[0].IcmpType).Count -ne 1 -or [string]$ports[0].IcmpType -notin @('128', '128:Any') -or
        $addresses.Count -ne 1 -or -not (Test-PcaiUsb4ExactAddress $addresses[0].LocalAddress $LocalAddress) -or
        -not (Test-PcaiUsb4ExactAddress $addresses[0].RemoteAddress $PeerAddress) -or
        $interfaces.Count -ne 1 -or @($interfaces[0].InterfaceAlias).Count -ne 1 -or
        [string]$interfaces[0].InterfaceAlias -cne $Alias -or
        $application.Count -ne 1 -or $application[0].Program -ne 'Any' -or
        $services.Count -ne 1 -or $services[0].Service -ne 'Any') {
        throw "Firewall rule '$ruleName' in $Store has unexpected USB4 filters."
    }
    return $rules[0]
}

function Get-PcaiUsb4LegacyRuleStatus {
    $legacyName = 'PC-AI-USB4-P2P-Allow'
    $status = [ordered]@{ Name = $legacyName; BlocksApply = $false }
    foreach ($store in @('PersistentStore', 'ActiveStore')) {
        $rules = @(try {
            Get-NetFirewallRule -PolicyStore $store -Name $legacyName
        } catch {
            if ($_.FullyQualifiedErrorId -cne 'CmdletizationQuery_NotFound_InstanceID,Get-NetFirewallRule') { throw }
        })
        if (@($rules | Where-Object Name -NE $legacyName).Count) {
            throw "Firewall policy returned an unexpected name for '$legacyName' in $store."
        }
        $enabledInboundAllow = @($rules | Where-Object {
            [string]$_.Enabled -eq 'True' -and [string]$_.Direction -eq 'Inbound' -and [string]$_.Action -eq 'Allow'
        }).Count -gt 0
        $status[$store] = [pscustomobject]@{
            State = if ($enabledInboundAllow) { 'EnabledInboundAllow' } elseif ($rules.Count) { 'Present' } else { 'Absent' }
            Count = $rules.Count
        }
        $status.BlocksApply = $status.BlocksApply -or $enabledInboundAllow
    }
    return [pscustomobject]$status
}

function Get-PcaiUsb4PrivateProfileExposure {
    $sampleLimit = 20
    $names = @(Get-NetFirewallRule -PolicyStore ActiveStore -Enabled True -Direction Inbound -Action Allow |
        Where-Object {
            $profiles = ([string]$_.Profile -split ',').Trim()
            'Private' -in $profiles -and 'Public' -notin $profiles
        } | Select-Object -ExpandProperty Name | Sort-Object)
    return [pscustomobject]@{
        Total = $names.Count
        RuleNames = @($names | Select-Object -First $sampleLimit)
        Truncated = $names.Count -gt $sampleLimit
    }
}

function Get-PcaiUsb4ReceiptFile {
    if (-not $ReceiptPath -or -not [IO.Path]::IsPathFullyQualified($ReceiptPath) -or
        ($IsWindows -and $ReceiptPath -notmatch '^[A-Za-z]:[/\\]') -or $ReceiptPath -match ':.*:') {
        throw 'Apply and Rollback require an absolute, local receipt path.'
    }
    $fullPath = [IO.Path]::GetFullPath($ReceiptPath)
    if ($IsWindows -and [IO.DriveInfo]::new([IO.Path]::GetPathRoot($fullPath)).DriveType -ne [IO.DriveType]::Fixed) {
        throw 'The receipt must be on a local fixed drive.'
    }
    $leaf = [IO.Path]::GetFileName($fullPath)
    if ($leaf -notmatch '^[a-z0-9][a-z0-9._-]*\.json$' -or $leaf -match '^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])\.') {
        throw 'A regular JSON receipt filename is required.'
    }
    $directory = [IO.DirectoryInfo]::new([IO.Path]::GetDirectoryName($fullPath))
    for ($ancestor = $directory; $null -ne $ancestor; $ancestor = $ancestor.Parent) {
        if (-not $ancestor.Exists -or $ancestor.Attributes -band [IO.FileAttributes]::ReparsePoint) {
            throw 'The receipt must be inside an existing local directory without reparse points.'
        }
    }
    $item = [IO.FileInfo]::new($fullPath)
    if ([IO.Directory]::Exists($fullPath) -or ($item.Exists -and $item.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
        throw 'The receipt path must refer to a regular file.'
    }
    return $fullPath
}

function Write-PcaiUsb4Receipt([string]$Path, [hashtable]$Receipt, [switch]$Replace) {
    $content = [Text.Encoding]::UTF8.GetBytes(($Receipt | ConvertTo-Json -Depth 7) + [Environment]::NewLine)
    if ($content.Length -gt 131072) { throw 'The USB4 receipt exceeds the supported size.' }
    $target = if ($Replace) { "$Path.$([guid]::NewGuid().ToString('N')).tmp" } else { $Path }
    try {
        $stream = [IO.File]::Open($target, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
        try {
            $stream.Write($content, 0, $content.Length)
            $stream.Flush($true)
        } finally { $stream.Dispose() }
        if ($Replace) { [IO.File]::Move($target, $Path, $true) }
    } finally {
        if ($Replace -and [IO.File]::Exists($target)) { [IO.File]::Delete($target) }
    }
}

function Assert-PcaiUsb4ReceiptLink($Record, $Link) {
    if ($Record.schema -cne $schema -or $Record.host -cne [Environment]::MachineName -or
        $Record.adapterGuid -cne $AdapterGuid.ToString() -or $Record.alias -cne $Link.Alias -or
        $Record.index -cne $Link.Index -or $Record.profileName -cne $Link.ProfileName -or
        $Record.profileId -cne $Link.ProfileId -or (@($Record.sharedGuids) -join ',') -cne ($sharedGuids -join ',') -or
        $Record.localMac -cne $LocalMac -or $Record.peerMac -cne $PeerMac -or
        $Record.localAddress -cne $LocalAddress -or $Record.peerAddress -cne $PeerAddress -or
        $Record.ruleAdded -isnot [bool] -or $Record.profileWriteStarted -isnot [bool] -or
        $Record.previousCategory -notin @('Public', 'Private') -or
        ($Record.profileWriteStarted -and $Record.previousCategory -ne 'Public') -or
        $Record.profileSnapshot -isnot [hashtable] -or -not $Record.profileSnapshot.ContainsKey($Link.ProfileId) -or
        $Link.ProfileId -notin $Record.profileScope -or
        $Record.owner -cnotmatch $ownerPattern -or $Record.state -notin @('Pending', 'Applied', 'RollingBack', 'Reverted')) {
        throw 'Receipt does not match this host, enrolled peer and live USB4 profile.'
    }
}

function Undo-PcaiUsb4PrivateAccess([string]$Path, [hashtable]$Record) {
    $link = Get-PcaiUsb4AccessLink
    Assert-PcaiUsb4ReceiptLink $Record $link
    if ($Record.state -eq 'Reverted') { return 'Reverted' }
    $snapshot = Get-PcaiUsb4ProfileSnapshot
    $scope = @(Get-PcaiUsb4ProfileScope $snapshot $link -RequireCoverage:$Record.profileWriteStarted)
    if (($scope -join ',') -cne ($Record.profileScope -join ',')) { throw 'The authorized USB4 profile scope drifted.' }
    $rule = Get-PcaiUsb4AccessRule 'PersistentStore' $link.Alias $Record.owner
    if ($Record.state -eq 'Applied' -and (-not $rule -or $link.Category -ne 'Private')) {
        throw 'The applied firewall rule or connection profile drifted. The receipt and configuration are preserved.'
    }
    if ($Record.state -eq 'Applied') {
        $expected = @{}
        $actual = @{}
        foreach ($id in $scope) {
            $expected[$id] = $Record.appliedSnapshot[$id]
            $actual[$id] = $snapshot[$id]
        }
        Assert-PcaiUsb4ProfileReadback $expected $actual
        # Preserve unrelated profile changes made after a successful install.
        foreach ($id in $scope) { $snapshot[$id] = $Record.profileSnapshot[$id] }
        $Record.rollbackSnapshot = $snapshot
    } elseif ($Record.state -eq 'Pending') {
        $Record.rollbackSnapshot = $Record.profileSnapshot
    }
    $Record.state = 'RollingBack'
    Write-PcaiUsb4Receipt $Path $Record -Replace
    if ($Record.ruleAdded -and $rule) {
        Remove-NetFirewallRule -PolicyStore PersistentStore -Name $ruleName
        if (Get-PcaiUsb4AccessRule 'PersistentStore' $link.Alias $Record.owner) { throw 'Owned firewall rule remains after rollback.' }
    }
    $link = Get-PcaiUsb4AccessLink
    Assert-PcaiUsb4ReceiptLink $Record $link
    $snapshot = Get-PcaiUsb4ProfileSnapshot
    $scope = @(Get-PcaiUsb4ProfileScope $snapshot $link -RequireCoverage:$Record.profileWriteStarted)
    if (($scope -join ',') -cne ($Record.profileScope -join ',')) { throw 'The authorized USB4 profile scope drifted.' }
    if (@($scope | Where-Object { $snapshot[$_].category -cne $Record.profileSnapshot[$_].category }).Count) {
        if (-not $Record.profileWriteStarted) { throw 'The USB4 connection profile changed outside this transaction.' }
        Set-NetConnectionProfile -InputObject $link.Profile -NetworkCategory Public
    }
    Assert-PcaiUsb4ProfileReadback $Record.rollbackSnapshot (Get-PcaiUsb4ProfileSnapshot)
    $Record.state = 'Reverted'
    Write-PcaiUsb4Receipt $Path $Record -Replace
    return $Record.state
}

$link = Get-PcaiUsb4AccessLink
$snapshot = Get-PcaiUsb4ProfileSnapshot
$scope = @(Get-PcaiUsb4ProfileScope $snapshot $link)
$rule = Get-PcaiUsb4AccessRule 'PersistentStore' $link.Alias
$active = Get-PcaiUsb4AccessRule 'ActiveStore' $link.Alias
if ($active -and (-not $rule -or $active.Description -cne $rule.Description)) {
    throw "The effective rule '$ruleName' has different ownership from the local rule."
}
if ($rule -and -not $active -and $link.Category -eq 'Private' -and -not $Rollback) {
    throw "The private USB4 rule '$ruleName' is absent in ActiveStore."
}
$otherProfiles = @($snapshot.Keys | Where-Object {
    $snapshot[$_].name -ceq $link.ProfileName -and $_ -notin $scope
} | ForEach-Object { $snapshot[$_].alias } | Sort-Object)
$result = [ordered]@{
    State = 'Planned'; AdapterGuid = $AdapterGuid; InterfaceAlias = $link.Alias; InterfaceIndex = $link.Index
    ProfileBefore = $link.Category; ProfileTarget = 'Private'; FirewallRule = $ruleName
    SharedAdapterGuid = $sharedGuids
    UnapprovedSameNameProfiles = $otherProfiles
    WillCreateRule = -not [bool]$rule; ReceiptPath = $ReceiptPath
    LegacyRule = if (-not $Rollback) { Get-PcaiUsb4LegacyRuleStatus }
    PrivateProfileExposure = if (-not $Rollback) { Get-PcaiUsb4PrivateProfileExposure }
}
if (-not ($Apply -or $Rollback)) { return [pscustomobject]$result }
$receiptFile = Get-PcaiUsb4ReceiptFile
$operation = if ($Rollback) { 'Roll back the private USB4 profile and its scoped IPv6 echo rule' }
             else { 'Install the private USB4 profile and its scoped IPv6 echo rule' }
if (-not $PSCmdlet.ShouldProcess("$($link.Alias) ($AdapterGuid)", $operation)) { return [pscustomobject]$result }

$mutex = [Threading.Mutex]::new($false, 'Global\PC-AI USB4 Private Access')
$locked = $false
try {
    try { $locked = $mutex.WaitOne(0) } catch [Threading.AbandonedMutexException] { $locked = $true }
    if (-not $locked) { throw 'Another installer owns this USB4 adapter.' }
    Assert-PcaiUsb4ProfileReadback $snapshot (Get-PcaiUsb4ProfileSnapshot)
    $current = Get-PcaiUsb4AccessLink
    if ($current.Index -ne $link.Index -or $current.Alias -cne $link.Alias -or
        $current.ProfileName -cne $link.ProfileName -or $current.Category -cne $link.Category) {
        throw 'The USB4 profile changed after planning.'
    }
    if ($Rollback) {
        if (-not [IO.File]::Exists($receiptFile) -or [IO.FileInfo]::new($receiptFile).Length -gt 131072) {
            throw 'A bounded, existing USB4 private access receipt is required.'
        }
        $receipt = [IO.File]::ReadAllText($receiptFile) | ConvertFrom-Json -AsHashtable
        if ($receipt -isnot [hashtable]) { throw 'The USB4 receipt must be a JSON object.' }
        $result.State = Undo-PcaiUsb4PrivateAccess $receiptFile $receipt
        return [pscustomobject]$result
    }
    if ((Get-PcaiUsb4LegacyRuleStatus).BlocksApply) {
        throw "Legacy inbound firewall rule '$($result.LegacyRule.Name)' must be resolved before Apply."
    }
    $currentRule = Get-PcaiUsb4AccessRule 'PersistentStore' $link.Alias
    if ([bool]$currentRule -ne [bool]$rule -or ($rule -and $currentRule.Description -cne $rule.Description)) {
        throw 'The USB4 rule changed after planning.'
    }
    if ($rule -and $link.Category -eq 'Private') {
        if (-not (Get-PcaiUsb4AccessRule 'ActiveStore' $link.Alias $rule.Description)) { throw 'The effective USB4 rule disappeared.' }
        $result.State = 'Unchanged'
        return [pscustomobject]$result
    }
    if ($link.Category -eq 'Public') {
        $scope = @(Get-PcaiUsb4ProfileScope $snapshot $link -RequireCoverage)
        if (@($scope | Where-Object { $snapshot[$_].category -ne 'Public' }).Count) {
            throw 'The potentially shared profile has inconsistent categories before installation.'
        }
    }
    $owner = if ($rule) { [string]$rule.Description } else { "$groupName $([guid]::NewGuid().ToString('N'))" }
    $receipt = @{
        schema = $schema; host = [Environment]::MachineName; adapterGuid = $AdapterGuid.ToString()
        alias = $link.Alias; index = $link.Index; profileName = $link.ProfileName; previousCategory = $link.Category
        profileId = $link.ProfileId; sharedGuids = $sharedGuids; profileScope = $scope; profileSnapshot = $snapshot
        localMac = $LocalMac; peerMac = $PeerMac; localAddress = $LocalAddress; peerAddress = $PeerAddress
        owner = $owner; ruleAdded = -not [bool]$rule; profileWriteStarted = $false; state = 'Pending'
        appliedSnapshot = $null; rollbackSnapshot = $null
    }
    Write-PcaiUsb4Receipt $receiptFile $receipt
    try {
        if ($receipt.ruleAdded) {
            New-NetFirewallRule -PolicyStore PersistentStore -Name $ruleName -DisplayName $ruleName -Group $groupName `
                -Description $owner -Direction Inbound -Action Allow -Enabled True -Profile Private -EdgeTraversalPolicy Block `
                -Protocol ICMPv6 -IcmpType 128 -InterfaceAlias $link.Alias -LocalAddress $LocalAddress -RemoteAddress $PeerAddress | Out-Null
        }
        $current = Get-PcaiUsb4AccessLink
        Assert-PcaiUsb4ReceiptLink $receipt $current
        Assert-PcaiUsb4ProfileReadback $snapshot (Get-PcaiUsb4ProfileSnapshot)
        if ($current.Category -ne $link.Category) { throw 'The USB4 profile changed during installation.' }
        if ($link.Category -eq 'Public') {
            $receipt.profileWriteStarted = $true
            Write-PcaiUsb4Receipt $receiptFile $receipt -Replace
            Set-NetConnectionProfile -InputObject $current.Profile -NetworkCategory Private
        }
        $after = Get-PcaiUsb4AccessLink
        Assert-PcaiUsb4ReceiptLink $receipt $after
        $receipt.appliedSnapshot = Get-PcaiUsb4ProfileSnapshot
        $mayChange = if ($receipt.profileWriteStarted) { $scope } else { @() }
        Assert-PcaiUsb4ProfileReadback $snapshot $receipt.appliedSnapshot $mayChange
        if ($after.Category -ne 'Private' -or
            -not (Get-PcaiUsb4AccessRule 'PersistentStore' $link.Alias $owner) -or
            -not (Get-PcaiUsb4AccessRule 'ActiveStore' $link.Alias $owner)) {
            throw 'The Private profile and its scoped firewall rule did not both verify.'
        }
        $receipt.state = 'Applied'
        Write-PcaiUsb4Receipt $receiptFile $receipt -Replace
    } catch {
        $original = $_
        $receipt.state = 'Pending'
        try { Undo-PcaiUsb4PrivateAccess $receiptFile $receipt | Out-Null }
        catch { throw "USB4 private access failed: $original. Rollback could not be verified: $_. Keep the receipt at '$receiptFile'." }
        throw $original
    }
    $result.State = 'Applied'
    return [pscustomobject]$result
} finally {
    if ($locked) { $mutex.ReleaseMutex() }
    $mutex.Dispose()
}
