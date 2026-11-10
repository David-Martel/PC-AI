#Requires -Version 7.0
<# .SYNOPSIS Verify explicit absent-resource retirement never becomes network repair. #>
[CmdletBinding()]
param([string]$SourcePath = (Join-Path $PSScriptRoot '../../Tools/SystemScripts/Networking/Resolve-Usb4PeerJournal.ps1'))
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $SourcePath)) { throw 'No explicit, bounded absent-custody retirement operation is available.' }
. $SourcePath
$script:checks = 0
function Assert-Retirement([bool]$Condition, [string]$Message) {
    if (-not $Condition) { throw $Message }
    $script:checks++
}
function New-RetirementFixture {
    $script:state = @{ schemaVersion = 1; hostName = 'fixture'; adapterMac = '020000000001'; interfaceGuid = '11111111-1111-1111-1111-111111111111';
        mode = 'Consumer'; backend = 'ExistingWinNat'; ownedAddresses = @(@{ address = '172.16.252.2'; prefix = 30 });
        ownedRoutes = @(); ownedForwarding = @(); dns = $null; ics = $null; natReservation = $null;
        pendingTransaction = @{ attemptId = 'original-attempt'; phase = 'Rollback'; firstFailure = @{ message = 'Original cause preserved' } }
    }
    $script:snapshot = @{ hostName = 'fixture'; taskDisabled = $true; policySha256 = ('A' * 64); controllerSha256 = ('C' * 64);
        adapters = @(@{ guid = $state.interfaceGuid; mac = $state.adapterMac; index = 23; usb4 = $true }); addresses = @(); taskXml = '<Task />'
    }
    $script:declarations = @{ ExpectedHostName = 'fixture'; ExpectedAdapterGuid = $state.interfaceGuid; ExpectedMac = $state.adapterMac;
        ExpectedAttemptId = 'original-attempt'; ExpectedPolicySha256 = ('A' * 64); ExpectedControllerSha256 = ('C' * 64) }
    $state.pendingTransaction.lastOperation = @{ name = 'Snapshot'; phase = 'Rollback'; completed = $false; arguments = @{} }
    $state.pendingTransaction.policy = @{ schemaVersion = 1; hostName = 'fixture'; mode = 'Consumer'; adapterMac = $state.adapterMac;
        peerMac = '020000000002'; peerLinkLocal = 'fe80::2'; localIPv4 = '172.31.252.134'; peerIPv4 = '172.31.252.133'; prefixLength = 30;
        sharing = @{ backend = 'ExistingWinNat'; natName = 'WSL'; natPrefix = '172.16.0.0/16'; providerAddress = '172.16.252.1'; consumerAddress = '172.16.252.2'; prefixLength = 30;
            dnsServers = @('1.1.1.1'); preferSharedInternet = $false; probeAddress = '1.1.1.1'; probePort = 443; routeMetric = 10; allowWirelessFallback = $false } }
    $state.pendingTransaction.preimage = @{ adapters = @(@{ guid = $state.interfaceGuid; mac = $state.adapterMac; index = 20 }); addresses = @() }
}
New-RetirementFixture
# Protects explicit ownership closure; detects refusal despite exact identity and
# no owned resource remaining. Needs inert observed data only. Breadcrumb:
# Resolve-Usb4PeerJournal.ps1::Get-Usb4JournalRetirementPlan.
$plan = Get-Usb4JournalRetirementPlan -State $state -Snapshot $snapshot @declarations
Assert-Retirement ($plan.status -eq 'Admissible' -and $plan.ownedAddressCount -eq 1 -and -not $plan.changesNetworking) 'Exact absent custody was not admitted as file-only retirement.'
Assert-Retirement ($plan.originalAttemptId -eq 'original-attempt' -and $plan.originalFailure -eq 'Original cause preserved') 'Original interrupted transaction identity was lost.'
Write-Output 'PASS: exact absent custody retains original transaction identity'

# Protects foreign/live ownership and failure evidence. Detects treating unknown,
# missing, changed or active resources as absent. Needs fixtures only; same source.
foreach ($case in @('PresentAddress', 'ForeignAddress', 'WrongMac', 'WrongGuid', 'WrongHost', 'TaskEnabled', 'UnknownTask',
    'WrongPolicy', 'WrongController', 'OwnedRoute', 'OwnedDns', 'OwnedIcs', 'OwnedNat', 'OwnedForwarding', 'NoPending', 'WrongAttempt', 'NotUsb4',
    'InFlightAddress', 'InFlightRoute', 'UnknownCustody', 'UnknownPending', 'UntrackedAddress', 'MissingPreimage', 'UntrackedUla')) {
    New-RetirementFixture
    switch ($case) {
        'PresentAddress' { $snapshot.addresses = @(@{ address = '172.16.252.2'; index = 23 }) }
        'ForeignAddress' { $snapshot.addresses = @(@{ address = '172.16.252.2'; index = 99 }) }
        'WrongMac' { $snapshot.adapters[0].mac = '020000000003' }
        'WrongGuid' { $snapshot.adapters[0].guid = '22222222-2222-2222-2222-222222222222' }
        'WrongHost' { $snapshot.hostName = 'other-host' }
        'TaskEnabled' { $snapshot.taskDisabled = $false }
        'UnknownTask' { $snapshot.Remove('taskDisabled') }
        'WrongPolicy' { $snapshot.policySha256 = ('B' * 64) }
        'WrongController' { $snapshot.controllerSha256 = ('D' * 64) }
        'OwnedRoute' { $state.ownedRoutes = @(@{ destination = '0.0.0.0/0' }) }
        'OwnedDns' { $state.dns = @{ mode = 'Static' } }
        'OwnedIcs' { $state.ics = @{ publicGuid = 'foreign' } }
        'OwnedNat' { $state.natReservation = @{ committed = $true } }
        'OwnedForwarding' { $state.ownedForwarding = @(@{ guid = $state.interfaceGuid }) }
        'NoPending' { $state.Remove('pendingTransaction') }
        'WrongAttempt' { $state.pendingTransaction.attemptId = 'other-attempt' }
        'NotUsb4' { $snapshot.adapters[0].usb4 = $false }
        'InFlightAddress' { $state.pendingTransaction.lastOperation.name = 'AddAddress' }
        'InFlightRoute' { $state.pendingTransaction.lastOperation.name = 'AddRoute' }
        'UnknownCustody' { $state.newOwnedResources = @() }
        'UnknownPending' { $state.pendingTransaction.newOwnedResources = @() }
        'UntrackedAddress' { $snapshot.addresses = @(@{ address = '172.31.252.134'; index = 23 }) }
        'MissingPreimage' { $state.pendingTransaction.Remove('preimage') }
        'UntrackedUla' { $state.pendingTransaction.policy.localUla = 'fd00:7462:7032::9'; $state.pendingTransaction.policy.peerUla = 'fd00:7462:7032::7';
            $snapshot.addresses = @(@{ address = 'fd00:7462:7032:0:0:0:0:9'; index = 23 }) }
    }
    $refused = $false
    try { $null = Get-Usb4JournalRetirementPlan -State $state -Snapshot $snapshot @declarations } catch { $refused = $true }
    Assert-Retirement $refused "Unsafe retirement was admitted: $case"
}
Write-Output 'PASS: active foreign unknown and changed custody is refused'

$workspace = Join-Path ([IO.Path]::GetTempPath()) ('pcai-journal-retirement-' + [guid]::NewGuid().ToString('N'))
[void][IO.Directory]::CreateDirectory($workspace)
try {
    New-RetirementFixture
    $backup = Join-Path $workspace 'backup.json'; $current = Join-Path $workspace 'state.json'; $archive = Join-Path $workspace 'retired'
    [IO.File]::WriteAllText($backup, ($state | ConvertTo-Json -Depth 12), [Text.UTF8Encoding]::new($false))
    [IO.File]::WriteAllBytes($current, [byte[]]::new(64))
    $backupHash = (Get-FileHash $backup -Algorithm SHA256).Hash; $currentHash = (Get-FileHash $current -Algorithm SHA256).Hash
    $script:providerReads = 0; $script:leaseCalls = 0
    $ops = @{ Snapshot = { param($InstalledPath) $script:providerReads++; Copy-Usb4NetworkValue $script:snapshot };
        Runtime = { @{ hostName = 'fixture'; administrator = $true; platform = 'Fixture' } };
        Acquire = { $script:leaseCalls++; return 'fixture-lease' }; Release = { param($Lease) };
        Protect = { param($Path) }
    }
    $argsForRun = @{ BackupPath = $backup; StatePath = $current; ArchivePath = $archive; InstalledPath = (Join-Path $workspace 'installed.json');
        ExpectedBackupSha256 = $backupHash; ExpectedStateSha256 = $currentHash; Operations = $ops }
    # Protects read-only planning and Apply/DryRun precedence. Detects artifact,
    # lease or state writes before explicit Apply. Needs owned real files and
    # inert snapshot providers; breadcrumb: Invoke-Usb4JournalRetirement.
    $result = Invoke-Usb4JournalRetirement @argsForRun @declarations -Apply -DryRun
    Assert-Retirement ($result.status -eq 'Plan' -and -not (Test-Path $archive) -and $leaseCalls -eq 0) 'DryRun acquired a mutation lease or wrote evidence.'
    Assert-Retirement ((Get-FileHash $current -Algorithm SHA256).Hash -ceq $currentHash) 'Plan changed damaged journal bytes.'
    Write-Output 'PASS: explicit DryRun suppresses all writes and leases'

    # Protects complete evidence before clearing already-absent custody. Detects
    # lossy backup/zero-file omission, re-enabling tasks, or missing commit receipt.
    # Needs temp filesystem + injected OS data; no real task/network operations.
    $result = Invoke-Usb4JournalRetirement @argsForRun @declarations -Apply
    Assert-Retirement ($result.status -eq 'RetiredAbsentCustody' -and $result.changesNetworking -eq $false) 'File-only retirement did not complete.'
    Assert-Retirement ((Get-FileHash (Join-Path $archive 'original-state.bin') -Algorithm SHA256).Hash -ceq $currentHash) 'Damaged evidence was not retained exactly.'
    Assert-Retirement ((Get-FileHash (Join-Path $archive 'trusted-pending-state.json') -Algorithm SHA256).Hash -ceq $backupHash) 'Pending backup was not retained exactly.'
    Assert-Retirement ((Get-Content $current -Raw | ConvertFrom-Json -AsHashtable).Count -eq 0) 'Retirement created a new enrollment instead of an explicit empty state.'
    $receipt = Get-Content (Join-Path $archive 'retirement.json') -Raw | ConvertFrom-Json
    Assert-Retirement ($receipt.status -eq 'RetiredAbsentCustody' -and $receipt.originalAttemptId -eq 'original-attempt') 'Final receipt lost original custody disposition.'
    Assert-Retirement ($snapshot.taskDisabled -and $leaseCalls -eq 1 -and $providerReads -ge 3) 'Retirement skipped fresh readback or changed scheduler authority.'
    Write-Output 'PASS: exact damaged and pending records precede explicit retirement'
    Write-Output "PASS: $script:checks journal-retirement assertions"
}
finally { [IO.Directory]::Delete($workspace, $true) }
