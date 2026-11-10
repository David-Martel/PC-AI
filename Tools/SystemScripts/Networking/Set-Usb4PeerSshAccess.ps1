#Requires -Version 7.0
<#
.SYNOPSIS
Plan or apply one owned USB4 peer SSH firewall rule.
.DESCRIPTION
Default invocation, -DryRun and -WhatIf produce an offline plan without OS or
firewall calls. Explicit -Apply verifies the current local interface and running
SSH service before adding an exact-address/interface/application/service rule.
Service binding requires a known configured service SID type 1 or 3. Explicit
Executable binding omits the firewall service-name condition while retaining
all other scope and running-service checks. Neither mode changes service SID
configuration, and configured SID metadata does not qualify a running token.
Existing foreign or changed rules are never adopted. Remove verifies the same
owned scope without requiring a connected device. No files, events, tasks,
services, routing, DNS, NAT, MTU or network-category settings are written.

Run Apply under an external finite owned-process deadline. Cached native/.NET
queries and the CIM service query (three-second operation timeout) can still be
blocked by an unhealthy OS; this script starts no child, thread or runspace.
A timeout or mutation exception leaves outcome unknown until fresh readback.
The named mutex coordinates this helper; COM is not an atomic compare/exchange
against an unrelated administrator editing the same rule concurrently.

Source and fixtures: ../../../Tests/Portable/Test-Usb4PeerSshAccess.ps1.
Deployment boundary: ../../../docs/usb4/peer-networking.md.
.PARAMETER DryRun
Suppress every external provider call, including when combined with Apply.
.PARAMETER Apply
Explicitly permit the exact planned rule operation after ShouldProcess.
.PARAMETER Remove
Remove only an exactly matching owned rule; never edit a foreign rule.
.PARAMETER ServiceBinding
Service (default) requires a supported configured service SID. Executable must
be selected explicitly and leaves the optional COM ServiceName unassigned.
.PARAMETER EnrolledPrivatePolicy
Explicit controller Peer policy object for an exact RFC1918 IPv4 /30 pair.
Requires ExpectedPeerMac; default admission remains link-local only. Apply
also requires the assigned /30 and exact freshly reachable peer on the USB4 NIC.
.PARAMETER Help
Show usage. Aliases -h and --help are accepted without other arguments.
#>
[CmdletBinding(SupportsShouldProcess = $true, ConfirmImpact = 'Medium')]
param(
    [string]$ExpectedHostName,
    [string]$InterfaceGuid,
    [string]$ExpectedInterfaceAlias,
    [string]$ExpectedMac,
    [string]$ExpectedPeerMac,
    [Collections.IDictionary]$EnrolledPrivatePolicy,
    [object[]]$LocalAddress,
    [object[]]$RemoteAddress,
    [string]$ProgramPath,
    [ValidateSet('Service', 'Executable')][string]$ServiceBinding = 'Service',
    [switch]$Remove,
    [switch]$Apply,
    [switch]$DryRun,
    [Alias('h', '-help')][switch]$Help,
    [Parameter(DontShow)][hashtable]$Operations
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function ConvertTo-Usb4SshProgramPath {
    param([object]$Path)
    if ($Path -isnot [string] -or $Path -notmatch '^[A-Za-z]:[\\/]' -or $Path -match '[\x00-\x1f\x7f*?"<>|]' -or $Path.Substring(2).Contains(':')) { throw 'SSH executable must be an explicit local drive path.' }
    $normalized = $Path.Replace('/', '\')
    $parts = $normalized.Substring(3).Split('\')
    foreach ($part in $parts) {
        if (-not $part -or $part -in @('.', '..') -or $part -match '[. ]$' -or $part -match '^(?i:\$null|AUX|CON|NUL|PRN|COM[1-9]|LPT[1-9])(?:[. ]|$)') { throw 'SSH executable path has an unsafe component.' }
    }
    if ($parts[-1] -ine 'sshd.exe') { throw 'SSH executable must be sshd.exe.' }
    $normalized
}
function ConvertTo-Usb4SshPrivateEnrollment {
    param([Collections.IDictionary]$Policy, [string]$HostName, [string]$Mac, [string]$PeerMac)
    if ($null -eq $Policy) {
        if ($PeerMac) { throw 'ExpectedPeerMac requires explicit private enrollment.' }
        return $null
    }
    foreach ($field in @('schemaVersion','hostName','mode','adapterMac','peerMac','localIPv4','peerIPv4','prefixLength')) {
        if (-not $Policy.Contains($field)) { throw 'Private enrollment is incomplete.' }
    }
    if (($Policy.schemaVersion -isnot [int] -and $Policy.schemaVersion -isnot [long]) -or $Policy.schemaVersion -ne 1 -or
        ($Policy.prefixLength -isnot [int] -and $Policy.prefixLength -isnot [long]) -or $Policy.prefixLength -ne 30) { throw 'Private enrollment requires schema 1 and an integer /30 prefix.' }
    if ($Policy.mode -isnot [string] -or $Policy.mode -cne 'Peer' -or $Policy.hostName -isnot [string] -or $Policy.hostName -ine $HostName) { throw 'Private enrollment must name this exact Peer host.' }
    if ($PeerMac -notmatch '^(?:[0-9a-fA-F]{2}-){5}[0-9a-fA-F]{2}$' -or $PeerMac -in @('00-00-00-00-00-00','FF-FF-FF-FF-FF-FF') -or
        ([Convert]::ToByte($PeerMac.Substring(0,2),16) -band 1) -ne 0 -or $PeerMac -ieq $Mac) { throw 'Private enrollment requires an explicit distinct unicast peer MAC.' }
    if ($Policy.adapterMac -isnot [string] -or $Policy.peerMac -isnot [string] -or $Policy.adapterMac -ine $Mac -or $Policy.peerMac -ine $PeerMac) { throw 'Private enrollment differs from the exact declared local/peer MACs.' }
    $networks = @(); $hosts = @()
    foreach ($field in @('localIPv4','peerIPv4')) {
        $address = $null; $value = $Policy[$field]
        if ($value -isnot [string] -or -not [Net.IPAddress]::TryParse($value,[ref]$address) -or $address.AddressFamily -ne [Net.Sockets.AddressFamily]::InterNetwork -or $value -cne $address.ToString()) { throw 'Private enrollment requires canonical IPv4 hosts.' }
        $bytes = $address.GetAddressBytes()
        $private = $bytes[0] -eq 10 -or ($bytes[0] -eq 172 -and $bytes[1] -ge 16 -and $bytes[1] -le 31) -or ($bytes[0] -eq 192 -and $bytes[1] -eq 168)
        if (-not $private -or ($bytes[3] -band 3) -notin @(1,2)) { throw 'Private enrollment requires usable RFC1918 /30 hosts.' }
        $hosts += $value
        $networks += ('{0}.{1}.{2}.{3}' -f $bytes[0],$bytes[1],$bytes[2],($bytes[3] -band 252))
    }
    if ($hosts[0] -ceq $hosts[1] -or $networks[0] -cne $networks[1]) { throw 'Private enrollment hosts must be distinct members of the same /30.' }
    [ordered]@{ schemaVersion=1; hostName=$HostName.ToLowerInvariant(); mode='Peer'; adapterMac=$Mac.ToUpperInvariant(); peerMac=$PeerMac.ToUpperInvariant(); localIPv4=$hosts[0]; peerIPv4=$hosts[1]; prefixLength=30 }
}
function ConvertTo-Usb4SshAddresses {
    param([object[]]$Address, [Collections.IDictionary]$PrivateEnrollment)
    if ($null -eq $Address -or $Address.Count -lt 1 -or $Address.Count -gt 2) { throw 'Provide one or two exact link-local addresses.' }
    $result = [Collections.Generic.List[string]]::new()
    $families = [Collections.Generic.HashSet[int]]::new()
    foreach ($item in $Address) {
        if ($item -isnot [string] -or $item -match '[%/\s,*]' -or -not $item) { throw 'Address must be a scalar exact host address without scope, range or prefix.' }
        $parsed = $null
        if (-not [Net.IPAddress]::TryParse($item, [ref]$parsed)) { throw 'Invalid address.' }
        $family = [int]$parsed.AddressFamily
        if ($family -eq 2) {
            $bytes = $parsed.GetAddressBytes()
            if ($null -ne $PrivateEnrollment) {
                if ($item -cne $parsed.ToString() -or $item -cnotin @($PrivateEnrollment.localIPv4,$PrivateEnrollment.peerIPv4)) { throw 'IPv4 address differs from the exact enrolled private pair.' }
            }
            elseif ($item -cne $parsed.ToString() -or $bytes[0] -ne 169 -or $bytes[1] -ne 254 -or $bytes[2] -in @(0, 255)) { throw 'IPv4 address must be a canonical assignable 169.254 link-local host.' }
        }
        elseif ($family -eq 23) {
            if (-not $parsed.IsIPv6LinkLocal -or $parsed.ScopeId -ne 0) { throw 'IPv6 address must be an unscoped link-local host.' }
        }
        else { throw 'Unsupported address family.' }
        if (-not $families.Add($family)) { throw 'Only one address per family is admitted.' }
        $result.Add($parsed.ToString().ToLowerInvariant())
    }
    @($result | Sort-Object)
}
function New-Usb4PeerSshAccessPlan {
    [CmdletBinding()]
    param(
        [string]$ExpectedHostName, [string]$InterfaceGuid, [string]$ExpectedInterfaceAlias, [string]$ExpectedMac,
        [string]$ExpectedPeerMac, [Collections.IDictionary]$EnrolledPrivatePolicy,
        [object[]]$LocalAddress, [object[]]$RemoteAddress, [string]$ProgramPath,
        [ValidateSet('Service', 'Executable')][string]$ServiceBinding = 'Service',
        [ValidateSet('Add', 'Remove')][string]$Action = 'Add'
    )
    if ($ExpectedHostName -notmatch '^[A-Za-z0-9][A-Za-z0-9_-]{0,62}$') { throw 'An exact local host name is required.' }
    $guid = [guid]::Empty
    if (-not [guid]::TryParse($InterfaceGuid, [ref]$guid) -or $guid -eq [guid]::Empty) { throw 'A nonempty interface GUID is required.' }
    if (-not $ExpectedInterfaceAlias -or $ExpectedInterfaceAlias.Length -gt 128 -or $ExpectedInterfaceAlias -match '[\x00-\x1f\x7f]') { throw 'An exact current interface alias is required.' }
    if ($ExpectedMac -notmatch '^(?:[0-9a-fA-F]{2}-){5}[0-9a-fA-F]{2}$' -or $ExpectedMac -in @('00-00-00-00-00-00', 'FF-FF-FF-FF-FF-FF')) { throw 'An exact six-byte interface MAC is required.' }
    if (([Convert]::ToByte($ExpectedMac.Substring(0, 2), 16) -band 1) -ne 0) { throw 'The interface MAC must be unicast.' }
    $enrollment = ConvertTo-Usb4SshPrivateEnrollment -Policy $EnrolledPrivatePolicy -HostName $ExpectedHostName -Mac $ExpectedMac -PeerMac $ExpectedPeerMac
    $local = @(ConvertTo-Usb4SshAddresses $LocalAddress -PrivateEnrollment $enrollment)
    $remote = @(ConvertTo-Usb4SshAddresses $RemoteAddress -PrivateEnrollment $enrollment)
    if ($null -ne $enrollment -and ($local.Count -ne 1 -or $remote.Count -ne 1 -or $local[0] -cne $enrollment.localIPv4 -or $remote[0] -cne $enrollment.peerIPv4)) { throw 'Private enrollment must retain only its exact local and remote IPv4 hosts.' }
    $localFamilies = @($local | ForEach-Object { [int]([Net.IPAddress]::Parse($_)).AddressFamily })
    $remoteFamilies = @($remote | ForEach-Object { [int]([Net.IPAddress]::Parse($_)).AddressFamily })
    if (@(Compare-Object $localFamilies $remoteFamilies).Count -ne 0) { throw 'Local and remote address families must match.' }
    foreach ($item in $local) { if ($remote -contains $item) { throw 'Local and remote addresses must differ.' } }
    $program = ConvertTo-Usb4SshProgramPath $ProgramPath
    $binding = if ($ServiceBinding -ieq 'Executable') { 'Executable' } else { 'Service' }
    $scope = [ordered]@{ hostName = $ExpectedHostName.ToUpperInvariant(); interfaceGuid = $guid.ToString('D'); interfaceAlias = $ExpectedInterfaceAlias
        mac = $ExpectedMac.ToUpperInvariant(); localAddresses = $local; remoteAddresses = $remote; program = $program; service = 'sshd'; serviceBinding = $binding; protocol = 6; localPort = 22; profiles = 7 }
    $hashScope = [ordered]@{ hostName = $scope.hostName.ToLowerInvariant(); interfaceGuid = $scope.interfaceGuid; interfaceAlias = $scope.interfaceAlias.ToLowerInvariant(); mac = $scope.mac
        localAddresses = $local; remoteAddresses = $remote; program = $program.ToLowerInvariant(); service = 'sshd'; serviceBinding = $binding; protocol = 6; localPort = 22; profiles = 7 }
    if ($null -ne $enrollment) { $scope.privateEnrollment=$enrollment; $hashScope.privateEnrollment=$enrollment }
    $bytes = [Text.Encoding]::UTF8.GetBytes(($hashScope | ConvertTo-Json -Depth 8 -Compress))
    $hash = [Security.Cryptography.SHA256]::Create()
    try { $digest = [BitConverter]::ToString($hash.ComputeHash($bytes)).Replace('-', '').ToLowerInvariant() } finally { $hash.Dispose() }
    $rule = [ordered]@{ name = 'PC-AI-USB4-SSH-' + $digest; description = 'PC-AI USB4 peer SSH v1; scope-sha256=' + $digest + '; peer=declared'
        grouping = 'PC-AI USB4 peer SSH'; enabled = $true; direction = 1; action = 1; protocol = 6; localPorts = '22'; remotePorts = '*'
        localAddresses = $local -join ','; remoteAddresses = $remote -join ','; applicationName = $program; serviceBinding = $binding; serviceName = $(if ($binding -eq 'Service') { 'sshd' } else { '' })
        interfaces = @($ExpectedInterfaceAlias); interfaceTypes = 'All'; profiles = 7; edgeTraversal = $false }
    if ($null -ne $enrollment) { $rule.privateEnrollment=$enrollment }
    [ordered]@{ schema = 'pcai.usb4-peer-ssh-access/v1'; action = $Action; scope = $scope; rule = $rule; peerIdentity = 'DeclaredByCaller'
        qualification = 'OfflinePlanNotRuntimeQualification'; externalDeadlineRequired = $true; writesFiles = $false }
}
function Get-Usb4SshRuleField {
    param([object]$Rule, [string]$Name)
    if ($Rule -isnot [Collections.IDictionary] -or -not $Rule.Contains($Name)) { throw 'Firewall rule shape is unknown.' }
    # Preserve the COM SAFEARRAY/container boundary. A bare pipeline return
    # unwraps one-element arrays, including valid Interfaces and invalid scalars.
    return ,($Rule[$Name])
}
function ConvertFrom-Usb4SshObservedAddresses {
    param([string]$Value, [Collections.IDictionary]$PrivateEnrollment)
    $hosts = [Collections.Generic.List[string]]::new()
    foreach ($entry in $Value.Split(',')) {
        $token = $entry.Trim()
        if ($token.Contains('-')) {
            $ends = @($token.Split('-'))
            $first = $null; $last = $null
            if ($ends.Count -ne 2 -or -not [Net.IPAddress]::TryParse($ends[0], [ref]$first) -or -not [Net.IPAddress]::TryParse($ends[1], [ref]$last)) { throw 'Observed firewall range is not an exact host.' }
            if ($first.AddressFamily -ne $last.AddressFamily -or [Convert]::ToBase64String($first.GetAddressBytes()) -cne [Convert]::ToBase64String($last.GetAddressBytes())) { throw 'Observed firewall address range is broader than one host.' }
            if ($first.AddressFamily -eq [Net.Sockets.AddressFamily]::InterNetworkV6 -and ($first.ScopeId -ne 0 -or $last.ScopeId -ne 0)) { throw 'Observed firewall range has an unsupported scope identifier.' }
            $hosts.Add($first.ToString())
            continue
        }
        $parts = @($token.Split('/'))
        if ($parts.Count -gt 2) { throw 'Observed firewall address has an invalid mask.' }
        $address = $null
        if (-not [Net.IPAddress]::TryParse($parts[0], [ref]$address)) { throw 'Observed firewall address is not an exact host.' }
        if ($parts.Count -eq 2) {
            $allowed = if ($address.AddressFamily -eq [Net.Sockets.AddressFamily]::InterNetwork) { @('32', '255.255.255.255') } else { @('128') }
            if ($parts[1] -cnotin $allowed) { throw 'Observed firewall address mask is broader than one host.' }
        }
        $hosts.Add($parts[0])
    }
    (ConvertTo-Usb4SshAddresses @($hosts) -PrivateEnrollment $PrivateEnrollment) -join ','
}
function ConvertTo-Usb4SshRuleIdentity {
    param([object]$Rule, [switch]$Observed, [ValidateSet('Service', 'Executable')][string]$ServiceBinding = 'Service', [Collections.IDictionary]$PrivateEnrollment)
    if ($null -eq $PrivateEnrollment -and $Rule -is [Collections.IDictionary] -and $Rule.Contains('privateEnrollment')) { $PrivateEnrollment=$Rule.privateEnrollment }
    $result = [ordered]@{}
    foreach ($field in @('name', 'description', 'grouping', 'localPorts', 'remotePorts', 'localAddresses', 'remoteAddresses', 'applicationName', 'serviceName', 'interfaceTypes')) {
        $value = Get-Usb4SshRuleField $Rule $field
        if ($field -eq 'serviceName') {
            if ($Observed -and $ServiceBinding -eq 'Executable' -and $null -eq $value) { $value = '' }
            if ($ServiceBinding -eq 'Executable' -and $value -is [string] -and $value -cne '') { throw 'Executable binding must omit the firewall service condition.' }
            if ($ServiceBinding -eq 'Service' -and $value -is [string] -and $value -ine 'sshd') { throw 'Service binding must retain the exact sshd service condition.' }
        }
        if ($value -isnot [string]) { throw 'Firewall rule scalar field is invalid.' }
        $result[$field] = $value
    }
    foreach ($field in @('direction', 'action', 'protocol', 'profiles')) {
        $value = Get-Usb4SshRuleField $Rule $field
        if ($value -isnot [int] -and $value -isnot [long]) { throw 'Firewall rule integer field is invalid.' }
        $result[$field] = [long]$value
    }
    foreach ($field in @('enabled', 'edgeTraversal')) {
        $value = Get-Usb4SshRuleField $Rule $field
        if ($value -isnot [bool]) { throw 'Firewall rule boolean field is invalid.' }
        $result[$field] = $value
    }
    $interfaces = Get-Usb4SshRuleField $Rule 'interfaces'
    if ($interfaces -isnot [Array] -or $interfaces.Count -ne 1 -or $interfaces[0] -isnot [string]) { throw 'Firewall rule interface scope is invalid.' }
    $result.interfaces = @($interfaces[0].ToLowerInvariant())
    foreach ($field in @('name', 'applicationName', 'serviceName', 'interfaceTypes')) { $result[$field] = $result[$field].ToLowerInvariant() }
    if ($Observed) {
        $result.localAddresses = ConvertFrom-Usb4SshObservedAddresses $result.localAddresses -PrivateEnrollment $PrivateEnrollment
        $result.remoteAddresses = ConvertFrom-Usb4SshObservedAddresses $result.remoteAddresses -PrivateEnrollment $PrivateEnrollment
    }
    else {
        $result.localAddresses = (ConvertTo-Usb4SshAddresses @($result.localAddresses.Split(',')) -PrivateEnrollment $PrivateEnrollment) -join ','
        $result.remoteAddresses = (ConvertTo-Usb4SshAddresses @($result.remoteAddresses.Split(',')) -PrivateEnrollment $PrivateEnrollment) -join ','
    }
    $result | ConvertTo-Json -Depth 8 -Compress
}
function Assert-Usb4SshRule {
    param([object[]]$Rules, [Collections.IDictionary]$Expected)
    $binding = if ($Expected.Contains('serviceBinding')) { $Expected.serviceBinding } else { 'Service' }
    $enrollment = if ($Expected.Contains('privateEnrollment')) { $Expected.privateEnrollment } else { $null }
    if ($Rules.Count -gt 1) { throw 'Refusing duplicate firewall rule names.' }
    if ($Rules.Count -eq 1 -and (ConvertTo-Usb4SshRuleIdentity $Rules[0] -Observed -ServiceBinding $binding -PrivateEnrollment $enrollment) -cne (ConvertTo-Usb4SshRuleIdentity $Expected -ServiceBinding $binding -PrivateEnrollment $enrollment)) { throw 'Existing firewall rule is foreign or drifted; no overwrite or removal is permitted.' }
}
function Get-Usb4SshSidConfiguration {
    param([Collections.IDictionary]$Snapshot)
    if ($Snapshot.Contains('serviceSidType') -and $Snapshot.serviceSidType -is [int] -and $Snapshot.serviceSidType -in @(0, 1, 3)) {
        return @{ status = 'KnownConfigured'; type = $Snapshot.serviceSidType; runningTokenQualified = $false }
    }
    @{ status = 'Unknown'; type = $null; runningTokenQualified = $false }
}
function Assert-Usb4SshSnapshot {
    param([Collections.IDictionary]$Snapshot, [Collections.IDictionary]$Plan)
    foreach ($field in @('hostName', 'interfaceGuid', 'alias', 'mac', 'serviceImagePath', 'runningImagePath', 'processStartedUtc')) {
        if (-not $Snapshot.Contains($field) -or $Snapshot[$field] -isnot [string] -or -not $Snapshot[$field]) { throw 'Current interface/service snapshot is incomplete.' }
    }
    foreach ($field in @('up', 'usb4', 'serviceRunning')) { if ($Snapshot[$field] -isnot [bool] -or -not $Snapshot[$field]) { throw 'Current USB4 link and SSH service must be running.' } }
    if ($Snapshot.hostName -ine $Plan.scope.hostName -or $Snapshot.interfaceGuid -ine $Plan.scope.interfaceGuid -or $Snapshot.alias -ine $Plan.scope.interfaceAlias -or $Snapshot.mac -ine $Plan.scope.mac) { throw 'Current interface identity differs from the declared host/GUID/alias/MAC.' }
    if ((ConvertTo-Usb4SshProgramPath $Snapshot.serviceImagePath) -ine $Plan.scope.program -or (ConvertTo-Usb4SshProgramPath $Snapshot.runningImagePath) -ine $Plan.scope.program) { throw 'Running SSH executable differs from the service or declared path.' }
    if (($Snapshot.processId -isnot [int] -and $Snapshot.processId -isnot [long]) -or $Snapshot.processId -le 0) { throw 'Running SSH service process is unknown.' }
    $instant = [DateTimeOffset]::MinValue
    if (-not [DateTimeOffset]::TryParse($Snapshot.processStartedUtc, [ref]$instant)) { throw 'Running SSH process generation is unknown.' }
    foreach ($address in $Plan.scope.localAddresses) { if (@($Snapshot.localAddresses) -notcontains $address) { throw 'Declared local address is not currently assigned to the exact USB4 interface.' } }
    if ($Plan.scope.Contains('privateEnrollment')) {
        $enrollment=$Plan.scope.privateEnrollment
        if (-not $Snapshot.Contains('prefixLengths') -or $Snapshot.prefixLengths -isnot [Collections.IDictionary] -or
            -not $Snapshot.prefixLengths.Contains($enrollment.localIPv4) -or
            ($Snapshot.prefixLengths[$enrollment.localIPv4] -isnot [int] -and $Snapshot.prefixLengths[$enrollment.localIPv4] -isnot [long]) -or
            $Snapshot.prefixLengths[$enrollment.localIPv4] -ne 30) { throw 'Enrolled local IPv4 /30 is not currently assigned to this USB4 interface.' }
        if (-not $Snapshot.Contains('peerMac') -or $Snapshot.peerMac -isnot [string] -or $Snapshot.peerMac -ine $enrollment.peerMac -or
            -not $Snapshot.Contains('peerNeighborState') -or ($Snapshot.peerNeighborState -isnot [int] -and $Snapshot.peerNeighborState -isnot [long]) -or $Snapshot.peerNeighborState -ne 5) { throw 'Exact enrolled peer MAC must be freshly reachable on this USB4 interface.' }
    }
    if ($Plan.scope.serviceBinding -eq 'Service') {
        $sid = Get-Usb4SshSidConfiguration $Snapshot
        if ($sid.status -ne 'KnownConfigured') { throw 'Service SID configuration is unknown; Service binding requires a registry DWORD value 1 or 3. Review explicit Executable binding without changing the service.' }
        if ($sid.type -eq 0) { throw 'Service SID type NONE cannot support a service-scoped firewall rule. Review explicit -ServiceBinding Executable while retaining exact executable/interface/address scope.' }
    }
}
function ConvertFrom-Usb4SshComRule {
    param([object]$Rule)
    # Read only this unregistered or already selected rule object. Keep raw
    # address text in receipts; the observed identity matcher admits host masks.
    [ordered]@{ name = [string]$Rule.Name; description = [string]$Rule.Description; grouping = [string]$Rule.Grouping; enabled = [bool]$Rule.Enabled
        direction = [int]$Rule.Direction; action = [int]$Rule.Action; protocol = [int]$Rule.Protocol; localPorts = [string]$Rule.LocalPorts; remotePorts = [string]$Rule.RemotePorts
        localAddresses = [string]$Rule.LocalAddresses; remoteAddresses = [string]$Rule.RemoteAddresses; applicationName = [string]$Rule.ApplicationName
        serviceName = $Rule.ServiceName; interfaces = @($Rule.Interfaces); interfaceTypes = [string]$Rule.InterfaceTypes
        profiles = [int]$Rule.Profiles; edgeTraversal = [bool]$Rule.EdgeTraversal }
}
function New-Usb4SshComRule {
    param([Collections.IDictionary]$Expected)
    # Construct only an unregistered COM object. This function never opens
    # FwPolicy2 or calls Rules.Add/Remove; the caller owns the returned RCW.
    $binding = if ($Expected.Contains('serviceBinding')) { $Expected.serviceBinding } else { 'Service' }
    [void](ConvertTo-Usb4SshRuleIdentity $Expected -ServiceBinding $binding)
    $rule = New-Object -ComObject HNetCfg.FWRule
    try {
        $rule.Name = $Expected.name; $rule.Description = $Expected.description; $rule.Grouping = $Expected.grouping
        $rule.Protocol = $Expected.protocol; $rule.LocalPorts = $Expected.localPorts; $rule.RemotePorts = $Expected.remotePorts
        $rule.LocalAddresses = $Expected.localAddresses; $rule.RemoteAddresses = $Expected.remoteAddresses
        $rule.ApplicationName = $Expected.applicationName
        if ($binding -eq 'Service') { $rule.ServiceName = $Expected.serviceName }
        $rule.Interfaces = [object[]]$Expected.interfaces; $rule.InterfaceTypes = $Expected.interfaceTypes
        $rule.Profiles = $Expected.profiles; $rule.Direction = $Expected.direction; $rule.Action = $Expected.action; $rule.EdgeTraversal = $Expected.edgeTraversal
        $rule.Enabled = $Expected.enabled
        $rule
    }
    catch {
        [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($rule)
        throw
    }
}
function New-Usb4SshAccessOperations {
    @{
        Runtime = {
            $windows = [Runtime.InteropServices.RuntimeInformation]::IsOSPlatform([Runtime.InteropServices.OSPlatform]::Windows)
            $administrator = $false
            if ($windows) { $administrator = [Security.Principal.WindowsPrincipal]::new([Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator) }
            @{ windows = $windows; hostName = [Environment]::MachineName; administrator = $administrator }
        }
        Acquire = {
            $mutex = [Threading.Mutex]::new($false, 'Global\PC_AI_Usb4PeerSshAccess_v1')
            try {
                try { $held = $mutex.WaitOne(0) } catch [Threading.AbandonedMutexException] { $held = $true }
                if (-not $held) { throw 'Another USB4 SSH access mutation is active.' }
                $mutex
            }
            catch { $mutex.Dispose(); throw }
        }
        Release = { param($Lease) try { $Lease.ReleaseMutex() } finally { $Lease.Dispose() } }
        Snapshot = { param($Plan)
            $all = @([Net.NetworkInformation.NetworkInterface]::GetAllNetworkInterfaces())
            if ($all.Count -gt 512) { throw 'Network interface bound exceeded.' }
            $matches = @($all | Where-Object { $_.Id.Trim('{}') -ieq $Plan.scope.interfaceGuid })
            if ($matches.Count -ne 1) { throw 'Exact network interface is missing or ambiguous.' }
            $nic = $matches[0]
            $properties = $nic.GetIPProperties()
            if ($properties.UnicastAddresses.Count -gt 128) { throw 'Interface address bound exceeded.' }
            $addresses = @($properties.UnicastAddresses | ForEach-Object { [Net.IPAddress]::new($_.Address.GetAddressBytes()).ToString().ToLowerInvariant() })
            $mac = $nic.GetPhysicalAddress().GetAddressBytes()
            $connection = [Microsoft.Win32.Registry]::LocalMachine.OpenSubKey('SYSTEM\CurrentControlSet\Control\Network\{4D36E972-E325-11CE-BFC1-08002BE10318}\{' + $Plan.scope.interfaceGuid + '}\Connection')
            try { $pnp = if ($connection) { $connection.GetValue('PnPInstanceId') } else { $null } } finally { if ($connection) { $connection.Dispose() } }
            $service = Get-CimInstance -ClassName Win32_Service -Filter "Name='sshd'" -OperationTimeoutSec 3 -ErrorAction Stop
            if (-not $service -or $service.State -ne 'Running' -or $service.ProcessId -le 0) { throw 'The existing sshd service is not running.' }
            $sidType = $null
            $serviceKey = $null
            try {
                $serviceKey = [Microsoft.Win32.Registry]::LocalMachine.OpenSubKey('SYSTEM\CurrentControlSet\Services\sshd')
                if ($serviceKey -and $serviceKey.GetValueKind('ServiceSidType') -eq [Microsoft.Win32.RegistryValueKind]::DWord) {
                    $configured = $serviceKey.GetValue('ServiceSidType')
                    if ($configured -is [int] -and $configured -in @(0, 1, 3)) { $sidType = $configured }
                }
            }
            catch { $sidType = $null }
            finally { if ($null -ne $serviceKey) { $serviceKey.Dispose() } }
            $servicePath = [string]$service.PathName
            if ($servicePath.StartsWith('"') -and $servicePath.EndsWith('"')) { $servicePath = $servicePath.Substring(1, $servicePath.Length - 2) }
            $servicePath = ConvertTo-Usb4SshProgramPath ([Environment]::ExpandEnvironmentVariables($servicePath))
            $drive = [IO.DriveInfo]::new($servicePath.Substring(0, 3))
            if ($drive.DriveType -ne [IO.DriveType]::Fixed) { throw 'SSH executable is not on an admitted local fixed drive.' }
            $cursor = $servicePath.Substring(0, 3)
            foreach ($segment in $servicePath.Substring(3).Split('\')) {
                $cursor = [IO.Path]::Combine($cursor, $segment)
                $attributes = [IO.File]::GetAttributes($cursor)
                if (($attributes -band [IO.FileAttributes]::ReparsePoint) -or ($attributes -band [IO.FileAttributes]::Offline) -or ([int]$attributes -band 0x440000)) { throw 'SSH executable traverses a reparse/offline/recall path.' }
            }
            if ($attributes -band [IO.FileAttributes]::Directory) { throw 'SSH executable path is a directory.' }
            $process = [Diagnostics.Process]::GetProcessById([int]$service.ProcessId)
            try { $runningPath = $process.MainModule.FileName; $started = $process.StartTime.ToUniversalTime().ToString('o') } finally { $process.Dispose() }
            $prefixLengths=@{}; $peerMac=$null; $peerState=$null
            if ($Plan.scope.Contains('privateEnrollment')) {
                foreach ($unicast in $properties.UnicastAddresses) {
                    if ($unicast.Address.AddressFamily -eq [Net.Sockets.AddressFamily]::InterNetwork) { $prefixLengths[$unicast.Address.ToString()]=[int]$unicast.PrefixLength }
                }
                $index=$properties.GetIPv4Properties().Index
                $peerAddress=$Plan.scope.privateEnrollment.peerIPv4
                $neighbors=@(Get-CimInstance -Namespace root/StandardCimv2 -ClassName MSFT_NetNeighbor -Filter "InterfaceIndex=$index AND IPAddress='$peerAddress'" -OperationTimeoutSec 3 -ErrorAction Stop)
                if ($neighbors.Count -ne 1) { throw 'Exact private peer neighbor is missing or ambiguous on this USB4 interface.' }
                $peerMac=[string]$neighbors[0].LinkLayerAddress; $peerState=[int]$neighbors[0].State
            }
            @{ hostName = [Environment]::MachineName; interfaceGuid = $nic.Id.Trim('{}'); alias = $nic.Name; mac = [BitConverter]::ToString($mac)
                up = $nic.OperationalStatus -eq [Net.NetworkInformation.OperationalStatus]::Up; localAddresses = $addresses
                usb4 = $pnp -is [string] -and $pnp -match '^USB4\\' -and $pnp -match '&PROT_USB4NET\\'
                prefixLengths=$prefixLengths; peerMac=$peerMac; peerNeighborState=$peerState
                serviceRunning = $true; serviceSidType = $sidType; serviceImagePath = $servicePath; runningImagePath = $runningPath; processId = [int]$service.ProcessId; processStartedUtc = $started }
        }
        ReadRules = { param($Name)
            $policy = New-Object -ComObject HNetCfg.FwPolicy2
            try {
                $count = 0
                foreach ($rule in $policy.Rules) {
                    if (++$count -gt 8192) { throw 'Firewall rule enumeration bound exceeded.' }
                    if ($rule.Name -ieq $Name) {
                        ConvertFrom-Usb4SshComRule $rule
                    }
                }
            }
            finally { [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($policy) }
        }
        AddRule = { param($Expected)
            $rule = New-Usb4SshComRule $Expected
            $policy = $null
            try {
                $policy = New-Object -ComObject HNetCfg.FwPolicy2
                [void]$policy.Rules.Add($rule)
            }
            finally {
                [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($rule)
                if ($null -ne $policy) { [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($policy) }
            }
        }
        RemoveRule = { param($Name)
            $policy = New-Object -ComObject HNetCfg.FwPolicy2
            try { [void]$policy.Rules.Remove($Name) } finally { [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($policy) }
        }
    }
}
function Invoke-Usb4PeerSshAccess {
    [CmdletBinding(SupportsShouldProcess = $true, ConfirmImpact = 'Medium')]
    param([Collections.IDictionary]$Request, [switch]$Apply, [switch]$DryRun, [Parameter(DontShow)][hashtable]$Operations)
    # Rebuild from declarations; a caller cannot widen an injected prebuilt rule.
    $plan = New-Usb4PeerSshAccessPlan @Request
    $result = [ordered]@{ schema = $plan.schema; status = 'Planned'; action = $plan.action; plan = $plan; peerIdentity = 'DeclaredByCaller'
        mutationAttempted = $false; preimage = @(); observedAfter = @(); error = $null; qualification = 'NotRuntimeQualified'
        serviceSidConfiguration = @{ status = 'NotCollected'; type = $null; runningTokenQualified = $false } }
    if (-not $Apply -or $DryRun) { return $result }
    if (-not $PSCmdlet.ShouldProcess($plan.rule.name, $plan.action + ' exact USB4 peer SSH rule')) { $result.status = 'WhatIf'; return $result }
    $ops = New-Usb4SshAccessOperations
    if ($Operations) { foreach ($key in $Operations.Keys) { if (-not $ops.ContainsKey($key) -or $Operations[$key] -isnot [scriptblock]) { throw 'Unknown or invalid operation boundary.' }; $ops[$key] = $Operations[$key] } }
    $lease = $null
    $failure = $null
    try {
        $runtime = & $ops.Runtime
        if ($runtime.windows -isnot [bool] -or -not $runtime.windows) { throw 'Apply requires Windows.' }
        if ($runtime.hostName -isnot [string] -or $runtime.hostName -ine $plan.scope.hostName) { throw 'Current local host differs from the declared host.' }
        if ($runtime.administrator -isnot [bool] -or -not $runtime.administrator) { throw 'Apply requires an elevated local administrator.' }
        $lease = & $ops.Acquire
        if ($null -eq $lease) { throw 'Mutation lease was not acquired.' }
        $snapshot = $null
        if ($plan.action -eq 'Add') { $snapshot = & $ops.Snapshot $plan; $result.serviceSidConfiguration = Get-Usb4SshSidConfiguration $snapshot; Assert-Usb4SshSnapshot $snapshot $plan }
        $before = @(& $ops.ReadRules $plan.rule.name)
        $result.preimage = $before
        Assert-Usb4SshRule $before $plan.rule
        $change = ($plan.action -eq 'Add' -and $before.Count -eq 0) -or ($plan.action -eq 'Remove' -and $before.Count -eq 1)
        if (-not $change) { $result.status = 'NoOp'; $result.observedAfter = $before }
        else {
            if ($plan.action -eq 'Add') {
                $fresh = & $ops.Snapshot $plan; $result.serviceSidConfiguration = Get-Usb4SshSidConfiguration $fresh; Assert-Usb4SshSnapshot $fresh $plan
                $cohortFields = @('hostName', 'interfaceGuid', 'alias', 'mac', 'serviceImagePath', 'runningImagePath', 'processId', 'processStartedUtc')
                if ($plan.scope.serviceBinding -eq 'Service') { $cohortFields += 'serviceSidType' }
                foreach ($field in $cohortFields) { if ($snapshot[$field] -ine $fresh[$field]) { throw 'Interface/service generation changed during preflight.' } }
            }
            $immediate = @(& $ops.ReadRules $plan.rule.name)
            Assert-Usb4SshRule $immediate $plan.rule
            if ($immediate.Count -ne $before.Count) { throw 'Firewall rule changed during preflight.' }
            $result.mutationAttempted = $true
            if ($plan.action -eq 'Add') { & $ops.AddRule $plan.rule | Out-Null } else { & $ops.RemoveRule $plan.rule.name | Out-Null }
            $after = @(& $ops.ReadRules $plan.rule.name); $result.observedAfter = $after
            Assert-Usb4SshRule $after $plan.rule
            if (($plan.action -eq 'Add' -and $after.Count -ne 1) -or ($plan.action -eq 'Remove' -and $after.Count -ne 0)) { throw 'Firewall mutation readback did not match the planned outcome.' }
            $result.status = if ($plan.action -eq 'Add') { 'Added' } else { 'Removed' }
        }
    }
    catch {
        $failure = $_.Exception.Message
        if ($result.mutationAttempted) { try { $result.observedAfter = @(& $ops.ReadRules $plan.rule.name) } catch { $failure += '; readback failed: ' + $_.Exception.Message } }
    }
    finally { if ($null -ne $lease) { try { & $ops.Release $lease | Out-Null } catch { $failure = [string]$failure + '; lease release failed: ' + $_.Exception.Message } } }
    if ($failure) {
        $result.status = if ($result.mutationAttempted) { 'MutationOutcomeUnknown' } else { 'Refused' }
        $result.error = $failure
        $exception = [InvalidOperationException]::new('USB4 SSH access ' + $result.status + ': ' + $failure)
        $exception.Data['Receipt'] = $result
        throw $exception
    }
    $result
}

if ($MyInvocation.InvocationName -ne '.') {
    if ($Help) {
        @'
Usage: Set-Usb4PeerSshAccess.ps1 -ExpectedHostName HOST -InterfaceGuid GUID
  -ExpectedInterfaceAlias ALIAS -ExpectedMac XX-XX-XX-XX-XX-XX
  -LocalAddress ADDRESS[,ADDRESS] -RemoteAddress ADDRESS[,ADDRESS]
  -ProgramPath C:\Windows\System32\OpenSSH\sshd.exe [-Apply] [-DryRun] [-Remove]
  [-ServiceBinding Service|Executable]
  -h | --help
Default, DryRun and WhatIf are offline. Apply requires elevated Windows and a fresh
exact interface/service match. Scope is TCP22 only; peer identity is declared by
caller. An explicit EnrolledPrivatePolicy plus ExpectedPeerMac permits only its
exact RFC1918 Peer /30 pair; Apply requires assigned prefix and reachable peer MAC.
Default Service binding needs configured service SID 1 or 3; explicit
Executable omits the firewall service condition without changing service settings.
Remove touches only the identical owned rule. Use an external owned-process
deadline; ambiguous mutation outcomes require fresh readback, never blind cleanup.
'@
        return
    }
    $request = @{ ExpectedHostName = $ExpectedHostName; InterfaceGuid = $InterfaceGuid; ExpectedInterfaceAlias = $ExpectedInterfaceAlias; ExpectedMac = $ExpectedMac
        ExpectedPeerMac=$ExpectedPeerMac; EnrolledPrivatePolicy=$EnrolledPrivatePolicy
        LocalAddress = $LocalAddress; RemoteAddress = $RemoteAddress; ProgramPath = $ProgramPath; ServiceBinding = $ServiceBinding; Action = $(if ($Remove) { 'Remove' } else { 'Add' }) }
    try { Invoke-Usb4PeerSshAccess -Request $request -Apply:$Apply -DryRun:$DryRun -Operations $Operations -WhatIf:$WhatIfPreference | ConvertTo-Json -Depth 20 }
    catch {
        if ($_.Exception.Data.Contains('Receipt')) { $_.Exception.Data['Receipt'] | ConvertTo-Json -Depth 20 }
        Write-Error -ErrorRecord $_ -ErrorAction Continue
        exit 1
    }
}
