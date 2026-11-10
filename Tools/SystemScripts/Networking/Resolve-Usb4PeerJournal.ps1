#Requires -Version 7.0
<#
.SYNOPSIS
Retire interrupted USB4 custody only after its owned addresses are proven absent.
.DESCRIPTION
An explicit file-only recovery for a disabled controller with a damaged journal.
Requires independently declared hashes, host, adapter and interrupted attempt.
It archives the exact damaged bytes, trusted pending generation and disabled task
before replacing custody with an empty object. It never changes networking or
enables a task. Any non-address ownership requires a separate ownership review.
Run without Apply to inspect the plan; DryRun takes precedence over Apply.
The global installer and controller mutexes coordinate cooperating administrators;
they do not provide compare-and-swap against arbitrary privileged file writers.
#>
[CmdletBinding(SupportsShouldProcess)]
param(
    [string]$BackupPath, [string]$StatePath, [string]$ArchivePath, [string]$InstalledPath,
    [string]$ExpectedBackupSha256, [string]$ExpectedStateSha256,
    [string]$ExpectedHostName, [string]$ExpectedAdapterGuid, [string]$ExpectedMac,
    [string]$ExpectedAttemptId, [string]$ExpectedPolicySha256, [string]$ExpectedControllerSha256,
    [switch]$Apply, [switch]$DryRun
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'Invoke-Usb4PeerNetwork.ps1')
. (Join-Path $PSScriptRoot 'Install-Usb4PeerNetwork.ps1')

function Get-Usb4JournalRetirementPlan {
    <# .SYNOPSIS Require exact disabled identity and absence of all declared custody. #>
    param([hashtable]$State, [hashtable]$Snapshot,
        [string]$ExpectedHostName, [string]$ExpectedAdapterGuid, [string]$ExpectedMac,
        [string]$ExpectedAttemptId, [string]$ExpectedPolicySha256, [string]$ExpectedControllerSha256)
    $guid = [guid]::Parse($ExpectedAdapterGuid).ToString('D')
    $mac = ConvertTo-Usb4Mac $ExpectedMac
    $known = @('schemaVersion', 'hostName', 'interfaceGuid', 'adapterMac', 'mode', 'backend', 'ownedAddresses', 'ownedRoutes', 'ownedForwarding', 'dns', 'ics', 'natReservation', 'pendingTransaction')
    if (@($State.Keys | Where-Object { $_ -notin $known }).Count) { throw 'Unknown custody fields require an explicit ownership review.' }
    if (-not $ExpectedHostName -or -not $ExpectedAttemptId -or
        $ExpectedPolicySha256 -notmatch '^[A-Fa-f0-9]{64}$' -or $ExpectedControllerSha256 -notmatch '^[A-Fa-f0-9]{64}$') { throw 'Explicit retirement declarations are incomplete.' }
    if ($State.schemaVersion -ne 1 -or $State.hostName -ine $ExpectedHostName -or $Snapshot.hostName -ine $ExpectedHostName -or
        [guid]$State.interfaceGuid -ne [guid]$guid -or (ConvertTo-Usb4Mac $State.adapterMac) -cne $mac) { throw 'Trusted custody does not match the declared host and adapter.' }
    if ($State.pendingTransaction -isnot [Collections.IDictionary] -or $State.pendingTransaction.attemptId -cne $ExpectedAttemptId -or
        $State.pendingTransaction.phase -cne 'Rollback') { throw 'The exact interrupted rollback must be declared.' }
    $pending = $State.pendingTransaction
    $knownPending = @('attemptId', 'phase', 'observedAt', 'firstFailure', 'lastOperation', 'policy', 'preimage')
    if (@($pending.Keys | Where-Object { $_ -notin $knownPending }).Count -or
        $pending.lastOperation -isnot [Collections.IDictionary] -or $pending.lastOperation.name -cne 'Snapshot' -or
        $pending.lastOperation.phase -cne 'Rollback' -or $pending.lastOperation.completed -isnot [bool] -or
        $pending.lastOperation.arguments -isnot [Collections.IDictionary] -or $pending.lastOperation.arguments.Count -ne 0) { throw 'Only a checkpointed rollback observation can retire address-only custody; an in-flight mutation requires review.' }
    if ($State.mode -cne 'Consumer' -or $State.backend -cne 'ExistingWinNat' -or
        $pending.policy -isnot [hashtable] -or $pending.preimage -isnot [hashtable]) { throw 'This bounded recovery supports only the reviewed ExistingWinNat consumer shape.' }
    $null = Assert-Usb4NetworkPolicy -Policy $pending.policy -CurrentHost $ExpectedHostName
    if ($pending.policy.mode -cne $State.mode -or $pending.policy.sharing.backend -cne $State.backend -or
        (ConvertTo-Usb4Mac $pending.policy.adapterMac) -cne $mac) { throw 'Interrupted policy does not match custody identity.' }
    $preimage = $pending.preimage
    if (-not $preimage.ContainsKey('adapters') -or -not $preimage.ContainsKey('addresses') -or $null -eq $preimage.addresses) { throw 'Complete transaction preimage is required.' }
    $originalAdapters = @($preimage.adapters | Where-Object { [guid]$_.guid -eq [guid]$guid -and (ConvertTo-Usb4Mac $_.mac) -ceq $mac })
    if ($originalAdapters.Count -ne 1) { throw 'Transaction preimage adapter identity is ambiguous.' }
    foreach ($name in @('ownedRoutes', 'ownedForwarding')) {
        if (-not $State.ContainsKey($name) -or $null -eq $State[$name] -or @($State[$name]).Count -ne 0) { throw "Non-address or unknown custody requires review: $name" }
    }
    foreach ($name in @('dns', 'ics', 'natReservation')) {
        if (-not $State.ContainsKey($name) -or $null -ne $State[$name]) { throw "Non-address or unknown custody requires review: $name" }
    }
    if (-not $State.ContainsKey('ownedAddresses') -or $null -eq $State.ownedAddresses -or
        -not $Snapshot.ContainsKey('addresses') -or $null -eq $Snapshot.addresses -or
        -not $Snapshot.ContainsKey('taskDisabled') -or $Snapshot.taskDisabled -isnot [bool] -or -not $Snapshot.taskDisabled -or
        $Snapshot.policySha256 -ine $ExpectedPolicySha256 -or $Snapshot.controllerSha256 -ine $ExpectedControllerSha256) { throw 'Disabled task, installed hashes and complete address inventory are required.' }
    $adapters = @($Snapshot.adapters | Where-Object { [guid]$_.guid -eq [guid]$guid })
    if ($adapters.Count -ne 1 -or $adapters[0].usb4 -isnot [bool] -or -not $adapters[0].usb4 -or
        (ConvertTo-Usb4Mac $adapters[0].mac) -cne $mac) { throw 'The exact physical USB4 adapter is not positively identified.' }
    $mustBeAbsent = @($State.ownedAddresses)
    $candidates = @(@{ address = $pending.policy.localIPv4; prefix = 30 }, @{ address = $pending.policy.sharing.consumerAddress; prefix = 30 })
    if ($pending.policy.ContainsKey('localUla')) { $candidates += @{ address = $pending.policy.localUla; prefix = 128 } }
    foreach ($candidate in $candidates) {
        $candidateIp = [Net.IPAddress]::Parse($candidate.address)
        $foreignBefore = @($preimage.addresses | Where-Object { $_.index -eq $originalAdapters[0].index -and
            [Net.IPAddress]::Parse(([string]$_.address -split '%')[0]).Equals($candidateIp) })
        if (-not $foreignBefore.Count) { $mustBeAbsent += $candidate }
    }
    foreach ($address in $mustBeAbsent) {
        if ($address -isnot [Collections.IDictionary] -or -not $address.Contains('address') -or -not $address.Contains('prefix')) { throw 'Owned address identity is incomplete.' }
        $ip = [Net.IPAddress]::Parse([string]$address.address)
        if ([int]$address.prefix -lt 0 -or [int]$address.prefix -gt $(if ($ip.AddressFamily -eq 'InterNetwork') { 32 } else { 128 })) { throw 'Owned prefix is invalid.' }
        foreach ($observed in @($Snapshot.addresses)) {
            if ([Net.IPAddress]::Parse(([string]$observed.address -split '%')[0]).Equals($ip)) { throw 'Owned address still exists; no retirement is allowed on any interface.' }
        }
    }
    @{ status = 'Admissible'; hostName = $ExpectedHostName; interfaceGuid = $guid; adapterMac = $mac;
        ownedAddressCount = @($State.ownedAddresses).Count; changesNetworking = $false;
        originalAttemptId = $ExpectedAttemptId; originalFailure = [string]$State.pendingTransaction.firstFailure.message }
}

function Get-Usb4JournalRetirementSnapshot {
    <# .SYNOPSIS Read disabled installed task and active addresses without network mutation. #>
    param([string]$InstalledPath, [string]$StatePath)
    $installed = ConvertFrom-Usb4StateBytes (Read-Usb4NetworkBytes $InstalledPath)
    if ($installed.owner -cne $script:Usb4InstallOwner -or $installed.hostName -ine [Environment]::MachineName -or
        [IO.Path]::GetFullPath($installed.statePath) -ine [IO.Path]::GetFullPath($StatePath) -or
        $installed.taskName -cne 'Usb4PeerNetwork' -or $installed.taskPath -cne '\PC-AI\') { throw 'Installed task ownership or state binding differs.' }
    $task = Get-Usb4ScheduledTaskSnapshot -Name $installed.taskName -Path $installed.taskPath
    Assert-Usb4InstalledTask -Snapshot $task -Executable $installed.executable -Controller $installed.controllerPath -Policy $installed.policyPath -State $StatePath -Report $installed.reportPath -Enabled $false
    if ($task.Running) { throw 'The disabled controller is still executing.' }
    $adapters = @(Get-NetAdapter -IncludeHidden -ErrorAction Stop | ForEach-Object {
        @{ guid = ([guid]$_.InterfaceGuid).ToString('D'); mac = [string]$_.MacAddress; index = [int]$_.ifIndex;
            usb4 = ([string]$_.InterfaceDescription -match '^USB4\(TM\) P2P Network Adapter(?: #\d+)?$') }
    })
    $addresses = @(Get-NetIPAddress -PolicyStore ActiveStore -ErrorAction Stop | ForEach-Object { @{ address = [string]$_.IPAddress; index = [int]$_.InterfaceIndex; prefix = [int]$_.PrefixLength } })
    @{ hostName = [Environment]::MachineName; taskDisabled = (-not $task.Enabled); taskXml = $task.Xml;
        controllerSha256 = (Get-FileHash -LiteralPath $installed.controllerPath -Algorithm SHA256).Hash;
        policySha256 = (Get-FileHash -LiteralPath $installed.policyPath -Algorithm SHA256).Hash;
        adapters = $adapters; addresses = $addresses }
}

function New-Usb4JournalRetirementOperations {
    <# .SYNOPSIS Provide privileged Windows observation, ACL and shared mutex boundaries. #>
    @{
        Snapshot = { param($InstalledPath, $StatePath) Get-Usb4JournalRetirementSnapshot -InstalledPath $InstalledPath -StatePath $StatePath }
        Runtime = {
            $principal = [Security.Principal.WindowsPrincipal]::new([Security.Principal.WindowsIdentity]::GetCurrent())
            @{ hostName = [Environment]::MachineName; administrator = $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator); platform = $(if ($IsWindows) { 'Windows' } else { 'Unsupported' }) }
        }
        Acquire = {
            $leases = [Collections.Generic.List[object]]::new()
            try {
                foreach ($name in @('Global\PcaiUsb4PeerNetworkInstall', 'Global\PcaiUsb4PeerNetwork')) {
                    $mutex = [Threading.Mutex]::new($false, $name)
                    $held = $false
                    try { $held = $mutex.WaitOne(0) } catch [Threading.AbandonedMutexException] { $held = $true }
                    if (-not $held) { $mutex.Dispose(); throw 'A USB4 installer or controller still holds its mutex.' }
                    $leases.Add($mutex)
                }
                return ,$leases
            }
            catch { foreach ($mutex in $leases) { $mutex.ReleaseMutex(); $mutex.Dispose() }; throw }
        }
        Release = { param($Lease) foreach ($mutex in $Lease) { $mutex.ReleaseMutex(); $mutex.Dispose() } }
        Protect = { param($Path) Set-Usb4RuntimeProtection -Path $Path }
    }
}

function Write-Usb4RecoveryArchive {
    <# .SYNOPSIS Preserve exact binary evidence using create-only flushed storage. #>
    param([string]$Path, [byte[]]$Bytes, [hashtable]$Operations)
    $full = Assert-Usb4PublicationPath $Path
    $stream = [IO.FileStream]::new($full, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None, 8192, [IO.FileOptions]::WriteThrough)
    try { $stream.Write($Bytes, 0, $Bytes.Length); $stream.Flush($true) } finally { $stream.Dispose() }
    & $Operations.Protect $full | Out-Null
    if ((Get-Usb4BytesHash (Read-Usb4NetworkBytes $full)) -cne (Get-Usb4BytesHash $Bytes)) { throw 'Recovery archive readback differs; retaining artifacts for review.' }
}

function Invoke-Usb4JournalRetirement {
    <# .SYNOPSIS Archive exact custody before explicitly retiring already-absent addresses. #>
    [CmdletBinding(SupportsShouldProcess)]
    param([string]$BackupPath, [string]$StatePath, [string]$ArchivePath, [string]$InstalledPath,
        [string]$ExpectedBackupSha256, [string]$ExpectedStateSha256,
        [string]$ExpectedHostName, [string]$ExpectedAdapterGuid, [string]$ExpectedMac,
        [string]$ExpectedAttemptId, [string]$ExpectedPolicySha256, [string]$ExpectedControllerSha256,
        [switch]$Apply, [switch]$DryRun, [hashtable]$Operations = (New-Usb4JournalRetirementOperations))
    $declarations = @{ ExpectedHostName = $ExpectedHostName; ExpectedAdapterGuid = $ExpectedAdapterGuid; ExpectedMac = $ExpectedMac;
        ExpectedAttemptId = $ExpectedAttemptId; ExpectedPolicySha256 = $ExpectedPolicySha256; ExpectedControllerSha256 = $ExpectedControllerSha256 }
    foreach ($hash in @($ExpectedBackupSha256, $ExpectedStateSha256)) { if ($hash -notmatch '^[A-Fa-f0-9]{64}$') { throw 'Exact source and damaged-state SHA256 declarations are required.' } }
    $stateFile = Assert-Usb4PublicationPath $StatePath; $archive = Assert-Usb4PublicationPath $ArchivePath
    $backup = Read-Usb4NetworkBytes $BackupPath; $damaged = Read-Usb4NetworkBytes $stateFile
    if ((Get-Usb4BytesHash $backup) -ine $ExpectedBackupSha256 -or (Get-Usb4BytesHash $damaged) -ine $ExpectedStateSha256) { throw 'Backup or current-state identity differs.' }
    $state = ConvertFrom-Usb4StateBytes $backup
    $runtime = & $Operations.Runtime
    if ($runtime.hostName -ine $ExpectedHostName -or -not $runtime.administrator -or $runtime.platform -notin @('Windows', 'Fixture')) { throw 'Local administrator and exact host are required.' }
    $snapshot = & $Operations.Snapshot $InstalledPath $stateFile
    $plan = Get-Usb4JournalRetirementPlan -State $state -Snapshot $snapshot @declarations
    if (-not $Apply -or $DryRun -or -not $PSCmdlet.ShouldProcess($stateFile, 'Archive interrupted custody and retire already-absent addresses')) { $plan.status = 'Plan'; return $plan }
    $lease = & $Operations.Acquire
    try {
        $snapshot = & $Operations.Snapshot $InstalledPath $stateFile
        $plan = Get-Usb4JournalRetirementPlan -State $state -Snapshot $snapshot @declarations
        if ((Get-Usb4BytesHash (Read-Usb4NetworkBytes $stateFile)) -ine $ExpectedStateSha256 -or
            (Get-Usb4BytesHash (Read-Usb4NetworkBytes $BackupPath)) -ine $ExpectedBackupSha256) { throw 'Custody changed while acquiring recovery authority.' }
        if ([IO.Directory]::Exists($archive) -or [IO.File]::Exists($archive)) { throw 'Recovery archive already exists; retain it for explicit interrupted-recovery review.' }
        [void][IO.Directory]::CreateDirectory($archive)
        & $Operations.Protect $archive | Out-Null
        Write-Usb4RecoveryArchive -Path (Join-Path $archive 'original-state.bin') -Bytes $damaged -Operations $Operations
        Write-Usb4RecoveryArchive -Path (Join-Path $archive 'trusted-pending-state.json') -Bytes $backup -Operations $Operations
        Write-Usb4RecoveryArchive -Path (Join-Path $archive 'disabled-task.xml') -Bytes ([Text.UTF8Encoding]::new($false).GetBytes($snapshot.taskXml)) -Operations $Operations
        $receipt = Copy-Usb4NetworkValue $plan
        $receipt.status = 'Prepared'; $receipt.observedAtUtc = [DateTimeOffset]::UtcNow.ToString('o')
        $receipt.damagedStateSha256 = $ExpectedStateSha256; $receipt.trustedBackupSha256 = $ExpectedBackupSha256
        $receipt.runtime = $runtime.platform; $receipt.taskRemainsDisabled = $true
        $receipt.foreignNetworkState = 'Preserved; no global rollback attempted'
        Write-Usb4NetworkJson -Path (Join-Path $archive 'retirement.json') -Value $receipt
        $fresh = & $Operations.Snapshot $InstalledPath $stateFile
        $null = Get-Usb4JournalRetirementPlan -State $state -Snapshot $fresh @declarations
        Write-Usb4NetworkBytes -Path $stateFile -Bytes ([Text.UTF8Encoding]::new($false).GetBytes('{}')) -ExpectedPreviousHash $ExpectedStateSha256.ToUpperInvariant()
        $receipt.status = 'RetiredAbsentCustody'; $receipt.retiredStateSha256 = Get-Usb4BytesHash (Read-Usb4NetworkBytes $stateFile)
        Write-Usb4NetworkJson -Path (Join-Path $archive 'retirement.json') -Value $receipt
        return $receipt
    }
    finally { & $Operations.Release $lease | Out-Null }
}

if ($MyInvocation.InvocationName -eq '.') { return }
$arguments = @{}
foreach ($name in @('BackupPath', 'StatePath', 'ArchivePath', 'InstalledPath', 'ExpectedBackupSha256', 'ExpectedStateSha256',
    'ExpectedHostName', 'ExpectedAdapterGuid', 'ExpectedMac', 'ExpectedAttemptId', 'ExpectedPolicySha256', 'ExpectedControllerSha256', 'Apply', 'DryRun', 'WhatIf', 'Confirm')) {
    if ($PSBoundParameters.ContainsKey($name)) { $arguments[$name] = $PSBoundParameters[$name] }
}
Invoke-Usb4JournalRetirement @arguments | ConvertTo-Json -Depth 20
