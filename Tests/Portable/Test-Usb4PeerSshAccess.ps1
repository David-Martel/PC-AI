#Requires -Version 7.0
<#
.SYNOPSIS
Detect widening, unintended writes and lost ownership in USB4 SSH access.
.DESCRIPTION
Exercises the production planner and mutation gate with nine inert provider
groups. One Windows-only group constructs and reads an unregistered real FWRule
using a current interface alias; it never opens firewall policy or adds a rule. See
../../docs/usb4/peer-networking.md for the deployment boundary.
#>
[CmdletBinding()]
param([string]$SourcePath = (Join-Path $PSScriptRoot '../../Tools/SystemScripts/Networking/Set-Usb4PeerSshAccess.ps1'))
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
. $SourcePath
$script:checks = 0
$script:groups = 0
function Assert-SshAccessCondition {
    param([bool]$Condition, [string]$Message)
    if (-not $Condition) { throw $Message }
    $script:checks++
}
function Assert-SshAccessRefusal {
    param([scriptblock]$Operation, [string]$Pattern)
    $caught = $null
    try { & $Operation | Out-Null } catch { $caught = $_.Exception }
    Assert-SshAccessCondition ($null -ne $caught -and $caught.Message -match $Pattern) "Expected refusal /$Pattern/."
    $caught
}
function Copy-SshFixture {
    param($Value)
    if ($Value -is [Collections.IDictionary]) {
        $copy = @{}
        foreach ($key in $Value.Keys) { $copy[$key] = Copy-SshFixture $Value[$key] }
        return $copy
    }
    if ($Value -is [Array]) { return ,@($Value | ForEach-Object { Copy-SshFixture $_ }) }
    $Value
}
function New-SshAccessFixture {
    $script:store = @{}
    $script:calls = [Collections.Generic.List[string]]::new()
    $script:snapshotNumber = 0
    $script:ruleReadNumber = 0
    $script:drift = $null
    $script:raceRule = $null
    $script:addThenThrow = $false
    $script:removeThenThrow = $false
    $script:ignoreAdd = $false
    $script:ignoreRemove = $false
    $script:snapshot = @{ hostName = 'fixture-host'; interfaceGuid = '11111111-2222-3333-4444-555555555555'; alias = 'Fixture USB4'; mac = '02-00-00-00-00-01'; up = $true
        localAddresses = @('169.254.12.34', 'fe80::1234'); usb4 = $true; serviceRunning = $true; serviceSidType = 1; serviceImagePath = 'C:\Windows\System32\OpenSSH\sshd.exe'; runningImagePath = 'C:\Windows\System32\OpenSSH\sshd.exe'; processId = 4242; processStartedUtc = '2030-01-01T00:00:00Z' }
    @{
        Runtime = { $script:calls.Add('runtime'); @{ windows = $true; hostName = 'fixture-host'; administrator = $true } }
        Snapshot = { param($Plan)
            $script:calls.Add('snapshot'); $script:snapshotNumber++
            $copy = Copy-SshFixture $script:snapshot
            if ($script:snapshotNumber -eq 2 -and $script:drift) { $copy[$script:drift] = 'changed' }
            $copy
        }
        Acquire = { $script:calls.Add('acquire'); [pscustomobject]@{ fixtureLease = $true } }
        Release = { param($Lease) $script:calls.Add('release') }
        ReadRules = { param($Name)
            $script:calls.Add('read'); $script:ruleReadNumber++
            if ($script:raceRule -and $script:ruleReadNumber -eq 2) { $script:store[$Name] = Copy-SshFixture $script:raceRule }
            if ($script:store.ContainsKey($Name)) { Copy-SshFixture $script:store[$Name] }
        }
        AddRule = { param($Rule)
            $script:calls.Add('add')
            if (-not $script:ignoreAdd) { $script:store[$Rule.name] = Copy-SshFixture $Rule }
            if ($script:addThenThrow) { throw 'fixture add failed after commit' }
        }
        RemoveRule = { param($Name)
            $script:calls.Add('remove')
            if (-not $script:ignoreRemove) { $script:store.Remove($Name) }
            if ($script:removeThenThrow) { throw 'fixture remove failed after commit' }
        }
    }
}
$request = @{ ExpectedHostName = 'fixture-host'; InterfaceGuid = '11111111-2222-3333-4444-555555555555'; ExpectedInterfaceAlias = 'Fixture USB4'; ExpectedMac = '02-00-00-00-00-01'
    LocalAddress = @('169.254.12.34', 'fe80::1234'); RemoteAddress = @('169.254.56.78', 'fe80::5678'); ProgramPath = 'C:\Windows\System32\OpenSSH\sshd.exe'; Action = 'Add' }

# Selector group: OfflinePlansNeverReachExternalProviders.
# Protects default, DryRun and WhatIf against OS, mutation and file writes.
# Detects any provider invocation before an explicit approved Apply. Needs pwsh7
# and inert provider traps; no host/device. Breadcrumb: Invoke-Usb4PeerSshAccess gate.
$ops = New-SshAccessFixture
foreach ($flags in @(@{}, @{ DryRun = $true }, @{ Apply = $true; DryRun = $true }, @{ Apply = $true; WhatIf = $true })) {
    $script:calls.Clear()
    $result = Invoke-Usb4PeerSshAccess -Request $request -Operations $ops @flags
    Assert-SshAccessCondition ($result.status -in @('Planned', 'WhatIf')) 'Offline plan was not a plan.'
    Assert-SshAccessCondition ($script:calls.Count -eq 0) 'Offline plan reached an external provider.'
    Assert-SshAccessCondition (-not $result.mutationAttempted -and $result.peerIdentity -ceq 'DeclaredByCaller') 'Plan invented mutation or peer authentication.'
}
$script:groups++; Write-Output 'PASS: OfflinePlansNeverReachExternalProviders'

# Selector group: HelpAliasesRemainOffline.
# Protects discoverability without required inputs or external side effects.
# Detects -h/--help binding regressions and validation/provider calls during help.
# Needs this script path and inert traps; breadcrumb: entrypoint Help branch.
$ops = New-SshAccessFixture
foreach ($helpArguments in @(@{ h = $true }, @{ '-help' = $true })) {
    $helpArguments.Operations = $ops
    $usage = (& $SourcePath @helpArguments) -join "`n"
    Assert-SshAccessCondition ($usage.Contains('Usage: Set-Usb4PeerSshAccess.ps1')) 'Help alias did not show usage.'
    Assert-SshAccessCondition ($script:calls.Count -eq 0) 'Help reached an external provider.'
}
$script:groups++; Write-Output 'PASS: HelpAliasesRemainOffline'

# Selector group: ScopeIsExactAndStable.
# Protects one interface, exact hosts and running SSH service on TCP22 only.
# Detects all-port, wildcard, interface-free, application-free or unstable naming.
# Needs pure planner; breadcrumb: New-Usb4PeerSshAccessPlan and rule canonicalization.
$plan = New-Usb4PeerSshAccessPlan @request
$other = Copy-SshFixture $request
$other.LocalAddress = @($other.LocalAddress[1], $other.LocalAddress[0])
$other.RemoteAddress = @($other.RemoteAddress[1], $other.RemoteAddress[0])
$reordered = New-Usb4PeerSshAccessPlan @other
Assert-SshAccessCondition ($plan.rule.name -ceq $reordered.rule.name) 'Address order changed stable rule identity.'
Assert-SshAccessCondition ($plan.rule.protocol -eq 6 -and $plan.rule.localPorts -ceq '22' -and $plan.rule.direction -eq 1 -and $plan.rule.action -eq 1) 'SSH rule widened protocol, port or direction.'
Assert-SshAccessCondition ($plan.rule.applicationName -ceq $request.ProgramPath -and $plan.rule.serviceName -ceq 'sshd') 'SSH executable/service scope disappeared.'
Assert-SshAccessCondition (@($plan.rule.interfaces).Count -eq 1 -and $plan.rule.interfaces[0] -ceq 'Fixture USB4') 'Interface restriction disappeared.'
Assert-SshAccessCondition ($plan.rule.profiles -eq 7 -and -not $plan.rule.edgeTraversal) 'Profile mask or edge traversal changed.'
Assert-SshAccessCondition ($plan.rule.localAddresses -notmatch '[/*]' -and $plan.rule.remoteAddresses -notmatch '[/*]') 'Exact addresses became ranges.'
$script:groups++; Write-Output 'PASS: ScopeIsExactAndStable'

# Selector group: UnsafeDeclarationsRefuseBeforeProviders.
# Protects input boundaries; detects broad, ambiguous, remote or coercive input.
# Needs fixture request and traps only. Breadcrumb: New-Usb4PeerSshAccessPlan.
$cases = @(
    @('ExpectedHostName', 'other host'), @('InterfaceGuid', '00000000-0000-0000-0000-000000000000'), @('ExpectedMac', '*'), @('ExpectedMac', '01-00-00-00-00-01'), @('ExpectedInterfaceAlias', "bad`nname"),
    @('LocalAddress', @('0.0.0.0')), @('LocalAddress', @('169.254.12.34/16')), @('RemoteAddress', @('169.254.0.0/16')), @('RemoteAddress', @('fe80::/10')),
    @('RemoteAddress', @('*')), @('RemoteAddress', @('127.0.0.1')), @('RemoteAddress', @('fe80::5678%5')), @('RemoteAddress', @('169.254.56.78')),
    @('LocalAddress', @('169.254.12.34', '169.254.12.35')), @('RemoteAddress', @('169.254.12.34', 'fe80::5678')),
    @('ProgramPath', '\\server\share\sshd.exe'), @('ProgramPath', 'relative\sshd.exe'), @('ProgramPath', 'C:\Windows\..\sshd.exe'), @('ProgramPath', 'C:\AUX\sshd.exe'), @('ProgramPath', 'C:\Windows\ssh.exe')
)
foreach ($case in $cases) {
    $candidate = Copy-SshFixture $request; $candidate[$case[0]] = $case[1]
    $ops = New-SshAccessFixture
    $null = Assert-SshAccessRefusal { Invoke-Usb4PeerSshAccess -Request $candidate -Operations $ops -Apply } '.'
    Assert-SshAccessCondition ($script:calls.Count -eq 0 -and $script:store.Count -eq 0) 'Unsafe declaration reached providers.'
}
$script:groups++; Write-Output 'PASS: UnsafeDeclarationsRefuseBeforeProviders'

# Selector group: FreshIdentityAndServiceDriftRefuseWithoutMutation.
# Protects declared GUID/MAC/alias/local-address and exact running service identity.
# Detects stale, renamed, down, foreign or changed-process snapshots. Needs provider
# doubles. Breadcrumb: Assert-Usb4SshSnapshot and immediate second preflight.
foreach ($field in @('hostName', 'interfaceGuid', 'alias', 'mac', 'up', 'usb4', 'serviceRunning', 'serviceImagePath', 'runningImagePath', 'localAddresses', 'processId')) {
    $ops = New-SshAccessFixture
    $script:snapshot[$field] = if ($field -in @('up', 'usb4', 'serviceRunning')) { $false } elseif ($field -eq 'localAddresses') { @('169.254.99.99') } elseif ($field -eq 'processId') { 0 } else { 'changed' }
    $null = Assert-SshAccessRefusal { Invoke-Usb4PeerSshAccess -Request $request -Operations $ops -Apply } '.'
    Assert-SshAccessCondition (-not $script:calls.Contains('add') -and -not $script:calls.Contains('remove') -and $script:calls.Contains('release')) "Invalid $field mutated or stranded the lease."
}
foreach ($field in @('alias', 'mac', 'processStartedUtc')) {
    $ops = New-SshAccessFixture; $script:drift = $field
    $null = Assert-SshAccessRefusal { Invoke-Usb4PeerSshAccess -Request $request -Operations $ops -Apply } '.'
    Assert-SshAccessCondition (-not $script:calls.Contains('add')) 'Second preflight drift was ignored.'
}
$ops = New-SshAccessFixture
$ops.Runtime = { $script:calls.Add('runtime'); @{ windows = $false; hostName = 'fixture-host'; administrator = $false } }
$null = Assert-SshAccessRefusal { Invoke-Usb4PeerSshAccess -Request $request -Operations $ops -Apply } 'Windows'
Assert-SshAccessCondition ($script:calls.Count -eq 1) 'Unsupported platform reached OS/rule operations.'
$script:groups++; Write-Output 'PASS: FreshIdentityAndServiceDriftRefuseWithoutMutation'

# Selector group: ForeignDuplicateAndConcurrentRulesRemainUntouched.
# Protects ownership; detects replacing/deleting a foreign, duplicated, widened
# or coercively shaped rule and racing a new rule after preflight. Needs inert map and read seam.
# Breadcrumb: Assert-Usb4SshRule and immediate second rule read.
foreach ($action in @('Add', 'Remove')) {
    foreach ($field in @('description', 'localPorts', 'remoteAddresses', 'interfaces', 'applicationName', 'serviceName', 'profiles', 'enabled', 'edgeTraversal', 'scalarInterfaces', 'arrayProtocol', 'arrayDescription')) {
        $ops = New-SshAccessFixture; $plan = New-Usb4PeerSshAccessPlan @request
        $foreign = Copy-SshFixture $plan.rule
        switch ($field) {
            'scalarInterfaces' { $foreign.interfaces = 'Fixture USB4' }
            'arrayProtocol' { $foreign.protocol = @(6) }
            'arrayDescription' { $foreign.description = @($plan.rule.description) }
            default { $foreign[$field] = switch ($field) { 'interfaces' { @('Other interface') } 'profiles' { 4 } 'enabled' { $false } 'edgeTraversal' { $true } default { 'foreign' } } }
        }
        $script:store[$plan.rule.name] = $foreign; $before = $foreign | ConvertTo-Json -Depth 10 -Compress
        $candidate = Copy-SshFixture $request; $candidate.Action = $action
        $null = Assert-SshAccessRefusal { Invoke-Usb4PeerSshAccess -Request $candidate -Operations $ops -Apply } '.'
        Assert-SshAccessCondition (-not $script:calls.Contains('add') -and -not $script:calls.Contains('remove') -and ($script:store[$plan.rule.name] | ConvertTo-Json -Depth 10 -Compress) -ceq $before) 'Foreign rule was changed.'
    }
}
$ops = New-SshAccessFixture
$ops.ReadRules = { param($Name) $script:calls.Add('read'); $plan.rule; $plan.rule }
$null = Assert-SshAccessRefusal { Invoke-Usb4PeerSshAccess -Request $request -Operations $ops -Apply } 'duplicate'
Assert-SshAccessCondition (-not $script:calls.Contains('add')) 'Duplicate names reached mutation.'
$ops = New-SshAccessFixture; $script:raceRule = Copy-SshFixture $plan.rule; $script:raceRule.description = 'foreign'
$null = Assert-SshAccessRefusal { Invoke-Usb4PeerSshAccess -Request $request -Operations $ops -Apply } '.'
Assert-SshAccessCondition (-not $script:calls.Contains('add')) 'Concurrent rule was overwritten.'
$script:groups++; Write-Output 'PASS: ForeignDuplicateAndConcurrentRulesRemainUntouched'

# Selector group: OwnedAddNoOpAndRemoveAreIdempotent.
# Protects the full lifecycle without foreign mutation. Detects repeated Add,
# absent Remove writes, lost cleanup when hardware disappears, and lease loss.
# Needs in-memory providers only; breadcrumb: Invoke-Usb4PeerSshAccess transitions.
$ops = New-SshAccessFixture
$added = Invoke-Usb4PeerSshAccess -Request $request -Operations $ops -Apply
Assert-SshAccessCondition ($added.status -ceq 'Added' -and $script:store.Count -eq 1) 'Owned Add failed.'
$addCalls = @($script:calls | Where-Object { $_ -ceq 'add' }).Count
$noOp = Invoke-Usb4PeerSshAccess -Request $request -Operations $ops -Apply
Assert-SshAccessCondition ($noOp.status -ceq 'NoOp' -and @($script:calls | Where-Object { $_ -ceq 'add' }).Count -eq $addCalls) 'Identical Add wrote again.'
$removeRequest = Copy-SshFixture $request; $removeRequest.Action = 'Remove'
$ops.Snapshot = { throw 'Cleanup must not require a connected device or running SSH service.' }
$removed = Invoke-Usb4PeerSshAccess -Request $removeRequest -Operations $ops -Apply
$absent = Invoke-Usb4PeerSshAccess -Request $removeRequest -Operations $ops -Apply
Assert-SshAccessCondition ($removed.status -ceq 'Removed' -and $absent.status -ceq 'NoOp' -and $script:store.Count -eq 0) 'Owned cleanup lost idempotence.'
Assert-SshAccessCondition (@($script:calls | Where-Object { $_ -ceq 'acquire' }).Count -eq @($script:calls | Where-Object { $_ -ceq 'release' }).Count) 'Lease release count differs.'
$script:groups++; Write-Output 'PASS: OwnedAddNoOpAndRemoveAreIdempotent'

# Selector group: MutationErrorsAndReadbackDriftRemainUnknown.
# Protects outcome honesty and retained custody after an ambiguous write. Detects
# reporting success or automatic destructive rollback after an exception/no-op.
# Needs injected writes that commit then throw or silently fail; no live firewall.
# Breadcrumb: mutationAttempted, post-write readback and receipt-bearing exceptions.
foreach ($mode in @('addThenThrow', 'ignoreAdd', 'removeThenThrow', 'ignoreRemove')) {
    $ops = New-SshAccessFixture; $candidate = Copy-SshFixture $request; $plan = New-Usb4PeerSshAccessPlan @request
    if ($mode -like '*Remove*') { $candidate.Action = 'Remove'; $script:store[$plan.rule.name] = Copy-SshFixture $plan.rule }
    Set-Variable -Scope Script -Name $mode -Value $true
    $caughtFailure = Assert-SshAccessRefusal { Invoke-Usb4PeerSshAccess -Request $candidate -Operations $ops -Apply } '.'
    $receipt = $caughtFailure.Data['Receipt']
    Assert-SshAccessCondition ($receipt.status -ceq 'MutationOutcomeUnknown' -and $receipt.mutationAttempted) 'Ambiguous mutation became success or no change.'
    Assert-SshAccessCondition ($script:calls.Contains('release')) 'Mutation error stranded lease.'
    if ($mode -notlike '*Remove*') { Assert-SshAccessCondition (-not $script:calls.Contains('remove')) 'Mutation error deleted evidence automatically.' }
}
$script:groups++; Write-Output 'PASS: MutationErrorsAndReadbackDriftRemainUnknown'

# Selector group: ServiceSidNoneNeedsExplicitExecutableBinding.
# Protects configured service-SID admission and an explicit exact-executable mode.
# Detects adding a service rule for SID NONE/unknown/coercive data, implicit mode
# switching, losing other scope conditions or treating optional service as '*'.
# Needs inert snapshots/rule store only; breadcrumb: planner, Snapshot SID gate,
# ConvertTo-Usb4SshRuleIdentity and owned Add/NoOp/Remove transitions.
$ops = New-SshAccessFixture; $script:snapshot.serviceSidType = 0
$noneFailure = Assert-SshAccessRefusal { Invoke-Usb4PeerSshAccess -Request $request -Operations $ops -Apply } 'SID.*NONE'
Assert-SshAccessCondition (-not $script:calls.Contains('add') -and -not $script:calls.Contains('remove') -and $script:store.Count -eq 0) 'SID NONE reached service-rule mutation.'
foreach ($invalidSid in @($null, $true, '1', @(1), 2, -1, [long]1)) {
    $ops = New-SshAccessFixture; $script:snapshot.serviceSidType = $invalidSid
    $null = Assert-SshAccessRefusal { Invoke-Usb4PeerSshAccess -Request $request -Operations $ops -Apply } 'SID'
    Assert-SshAccessCondition (-not $script:calls.Contains('add')) 'Unknown or coercive SID data reached service-rule mutation.'
}
$ops = New-SshAccessFixture; [void]$script:snapshot.Remove('serviceSidType')
$null = Assert-SshAccessRefusal { Invoke-Usb4PeerSshAccess -Request $request -Operations $ops -Apply } 'SID'
Assert-SshAccessCondition (-not $script:calls.Contains('add')) 'Missing SID data reached service-rule mutation.'
foreach ($knownSid in @(1, 3)) {
    $ops = New-SshAccessFixture; $script:snapshot.serviceSidType = $knownSid
    $serviceResult = Invoke-Usb4PeerSshAccess -Request $request -Operations $ops -Apply
    Assert-SshAccessCondition ($serviceResult.status -ceq 'Added' -and $serviceResult.plan.scope.serviceBinding -ceq 'Service' -and $serviceResult.plan.rule.serviceName -ceq 'sshd') 'Supported default service binding changed.'
    Assert-SshAccessCondition (-not $serviceResult.serviceSidConfiguration.runningTokenQualified) 'Configured SID became running-token qualification.'
}
$execRequest = Copy-SshFixture $request; $execRequest.ServiceBinding = 'Executable'
$servicePlan = New-Usb4PeerSshAccessPlan @request
$execPlan = New-Usb4PeerSshAccessPlan @execRequest
Assert-SshAccessCondition ($execPlan.scope.serviceBinding -ceq 'Executable' -and $execPlan.scope.service -ceq 'sshd' -and $execPlan.rule.serviceName -ceq '' -and $execPlan.rule.name -cne $servicePlan.rule.name) 'Explicit binding did not retain logical sshd identity or distinct rule ownership.'
foreach ($field in @('applicationName', 'interfaces', 'localAddresses', 'remoteAddresses', 'protocol', 'localPorts', 'remotePorts', 'profiles', 'enabled', 'edgeTraversal', 'direction', 'action')) {
    Assert-SshAccessCondition (($execPlan.rule[$field] | ConvertTo-Json -Compress) -ceq ($servicePlan.rule[$field] | ConvertTo-Json -Compress)) "Executable binding widened $field."
}
foreach ($sid in @(0, 1, 3, $null, '1')) {
    $ops = New-SshAccessFixture; $script:snapshot.serviceSidType = $sid
    $addedExecutable = Invoke-Usb4PeerSshAccess -Request $execRequest -Operations $ops -Apply
    Assert-SshAccessCondition ($addedExecutable.status -ceq 'Added' -and $addedExecutable.plan.rule.serviceName -ceq '') 'Explicit executable binding required a service filter or failed exact Add.'
    if ($null -eq $sid -or $sid -is [string]) {
        Assert-SshAccessCondition ($addedExecutable.serviceSidConfiguration.status -ceq 'Unknown' -and $null -eq $addedExecutable.serviceSidConfiguration.type) 'Unknown SID data was coerced into a known configuration.'
    }
    $script:store[$execPlan.rule.name].serviceName = $null
    $noOpExecutable = Invoke-Usb4PeerSshAccess -Request $execRequest -Operations $ops -Apply
    Assert-SshAccessCondition ($noOpExecutable.status -ceq 'NoOp') 'Optional COM null service was not equivalent only for explicit executable binding.'
    $execRemove = Copy-SshFixture $execRequest; $execRemove.Action = 'Remove'
    $ops.Snapshot = { throw 'Exact owned cleanup does not require service SID or live hardware.' }
    $removedExecutable = Invoke-Usb4PeerSshAccess -Request $execRemove -Operations $ops -Apply
    Assert-SshAccessCondition ($removedExecutable.status -ceq 'Removed' -and $script:store.Count -eq 0) 'Explicit executable rule was not removed through exact owned matching.'
}
foreach ($wrongService in @('*', 'sshd', @(''))) {
    $ops = New-SshAccessFixture
    $foreignExecutable = Copy-SshFixture $execPlan.rule; $foreignExecutable.serviceName = $wrongService
    $script:store[$execPlan.rule.name] = $foreignExecutable
    $null = Assert-SshAccessRefusal { Invoke-Usb4PeerSshAccess -Request $execRequest -Operations $ops -Apply } '.'
    Assert-SshAccessCondition (-not $script:calls.Contains('add') -and -not $script:calls.Contains('remove')) 'Executable binding adopted a different or malformed service condition.'
}
$ops = New-SshAccessFixture; $script:drift = 'serviceSidType'
$null = Assert-SshAccessRefusal { Invoke-Usb4PeerSshAccess -Request $request -Operations $ops -Apply } 'SID'
Assert-SshAccessCondition (-not $script:calls.Contains('add')) 'Configured service SID drift was ignored during preflight.'
$script:groups++; Write-Output 'PASS: ServiceSidNoneNeedsExplicitExecutableBinding'

$inertGroups = $script:groups
$comGroups = 0
# Selector group: UnregisteredWindowsComRuleConstructorRequiresVariantArray.
# Protects the actual Windows COM constructor used before persistent rule Add.
# Detects invalid Interfaces SAFEARRAY marshaling, loss of any scope field and
# widening through observed address masks/ranges; caller prefixes remain refused.
# Needs Windows Firewall COM and one cached interface alias; no administrator,
# service, peer or hardware traffic. Breadcrumb: New-Usb4SshComRule (no policy API).
if ([Runtime.InteropServices.RuntimeInformation]::IsOSPlatform([Runtime.InteropServices.OSPlatform]::Windows)) {
    $interfaces = @([Net.NetworkInformation.NetworkInterface]::GetAllNetworkInterfaces())
    Assert-SshAccessCondition ($interfaces.Count -le 512) 'Windows constructor fixture exceeded the interface bound.'
    $aliases = @($interfaces | Where-Object { $_.OperationalStatus -eq [Net.NetworkInformation.OperationalStatus]::Up -and $_.Name } | ForEach-Object { [string]$_.Name })
    Assert-SshAccessCondition ($aliases.Count -gt 0) 'Windows COM constructor control needs a current interface alias.'
    $constructionRequest = Copy-SshFixture $request
    $constructionRequest.ExpectedInterfaceAlias = $aliases[0]
    $constructionPlan = New-Usb4PeerSshAccessPlan @constructionRequest
    $comRule = $null
    try {
        $comRule = New-Usb4SshComRule $constructionPlan.rule
        $actual = ConvertFrom-Usb4SshComRule $comRule
        $actualIdentity = ConvertTo-Usb4SshRuleIdentity $actual -Observed
        $expectedIdentity = ConvertTo-Usb4SshRuleIdentity $constructionPlan.rule
        Assert-SshAccessCondition ($actualIdentity -ceq $expectedIdentity) 'Real COM full-scope readback differs from the exact planned rule.'
        foreach ($observed in @('169.254.12.34/32', '169.254.12.34/255.255.255.255')) {
            Assert-SshAccessCondition ((ConvertFrom-Usb4SshObservedAddresses $observed) -ceq '169.254.12.34') 'Observed IPv4 host mask did not preserve one exact host.'
        }
        Assert-SshAccessCondition ((ConvertFrom-Usb4SshObservedAddresses 'fe80::1234/128') -ceq 'fe80::1234') 'Observed IPv6 host mask did not preserve one exact host.'
        Assert-SshAccessCondition ((ConvertFrom-Usb4SshObservedAddresses '169.254.12.34-169.254.12.34') -ceq '169.254.12.34') 'Observed equal-endpoint IPv4 range did not preserve one exact host.'
        foreach ($broader in @('169.254.12.34/24', '169.254.12.34/255.255.255.0', 'fe80::1234/64', 'fe80::1234-fe80::1235', '169.254.12.34-169.254.12.35', '169.254.12.34-fe80::1234')) {
            $null = Assert-SshAccessRefusal { ConvertFrom-Usb4SshObservedAddresses $broader } 'broader'
        }
        foreach ($declaredRange in @('169.254.56.78-169.254.56.78', 'fe80::5678-fe80::5678')) {
            $rangeRequest = Copy-SshFixture $constructionRequest
            $rangeRequest.RemoteAddress = if ($declaredRange.StartsWith('169.')) { @($declaredRange, 'fe80::5678') } else { @('169.254.56.78', $declaredRange) }
            $null = Assert-SshAccessRefusal { New-Usb4PeerSshAccessPlan @rangeRequest } 'Invalid address'
        }
    }
    finally { if ($null -ne $comRule) { [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($comRule) } }
    $execConstructionRequest = Copy-SshFixture $constructionRequest
    $execConstructionRequest.ServiceBinding = 'Executable'
    $execConstructionPlan = New-Usb4PeerSshAccessPlan @execConstructionRequest
    $execComRule = $null
    try {
        $execComRule = New-Usb4SshComRule $execConstructionPlan.rule
        $execActual = ConvertFrom-Usb4SshComRule $execComRule
        Assert-SshAccessCondition ($null -eq $execActual.serviceName -or ($execActual.serviceName -is [string] -and $execActual.serviceName -ceq '')) 'Fresh executable COM constructor gained a service-only condition.'
        $execActualIdentity = ConvertTo-Usb4SshRuleIdentity $execActual -Observed -ServiceBinding Executable
        $execExpectedIdentity = ConvertTo-Usb4SshRuleIdentity $execConstructionPlan.rule -ServiceBinding Executable
        Assert-SshAccessCondition ($execActualIdentity -ceq $execExpectedIdentity) 'Actual unregistered executable COM readback differs from the complete planned scope.'
    }
    finally { if ($null -ne $execComRule) { [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($execComRule) } }
    $script:groups++; $comGroups++; Write-Output 'PASS: UnregisteredWindowsComRuleConstructorRequiresVariantArray'
}
else { Write-Output 'NOT_RUN: UnregisteredWindowsComRuleConstructorRequiresVariantArray requires Windows Firewall COM.' }
Write-Output "PASS: $script:checks assertions in $script:groups USB4 SSH access groups; $inertGroups inert provider groups and $comGroups unregistered Windows COM constructor groups."
