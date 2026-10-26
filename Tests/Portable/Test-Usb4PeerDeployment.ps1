#Requires -Version 7.0
<#
.SYNOPSIS
Exercise deployment provenance against real temporary files and the actual getter.
.DESCRIPTION
Windows-only read-only OS queries are exercised without task or network changes.
No internet probe is requested. Temporary fixture files are the only writes.
#>
[CmdletBinding()]
param([string]$SourcePath = (Join-Path $PSScriptRoot '../../Tools/SystemScripts/Networking/Get-Usb4PeerDeployment.ps1'))
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
if (-not $IsWindows) {
    Write-Output 'NOT_RUN: Usb4PeerDeployment requires Windows NetTCPIP and ScheduledTasks; Linux runtime is not qualified by this harness.'
    return
}
$script:checks = 0
$script:groups = 0
function Assert-DeploymentCondition {
    param([bool]$Condition, [string]$Message)
    if (-not $Condition) { throw $Message }
    $script:checks++
}
function Write-FixtureJson {
    param([string]$Path, [object]$Value)
    [IO.File]::WriteAllText($Path, ($Value | ConvertTo-Json -Depth 12), [Text.UTF8Encoding]::new($false))
}
function Invoke-DeploymentFixture {
    param([string]$Root, [string]$Manifest, [switch]$InvalidInstallation, [switch]$MissingInstallation, [switch]$ReleaseReadError)
    $before = @(Get-ChildItem -LiteralPath $workspace -Recurse -File | ForEach-Object {
            $_.FullName + ':' + (Get-FileHash -LiteralPath $_.FullName).Hash
        }) | Sort-Object
    $arguments = @{ InstallRoot = $Root }
    if ($Manifest) { $arguments.ReleaseManifestPath = $Manifest }
    $json = & $SourcePath @arguments
    $data = $json | ConvertFrom-Json -AsHashtable -ErrorAction Stop
    Assert-DeploymentCondition ($data['schema'] -eq 'pcai.usb4-deployment/v1') 'Getter did not return its JSON schema.'
    Assert-DeploymentCondition ($data['internetProbe']['status'] -eq 'NotRun' -and
        $data['internetProbe']['packetSent'] -eq $false) 'Default inspection attempted an internet probe.'
    $after = @(Get-ChildItem -LiteralPath $workspace -Recurse -File | ForEach-Object {
            $_.FullName + ':' + (Get-FileHash -LiteralPath $_.FullName).Hash
        }) | Sort-Object
    Assert-DeploymentCondition (@(Compare-Object $before $after).Count -eq 0) 'Getter changed or created fixture files.'
    $errors = @($data['readErrors'])
    if ($InvalidInstallation) {
        Assert-DeploymentCondition ($data['status'] -eq 'Unknown' -and @($errors | Where-Object section -EQ 'installation').Count -eq 1) 'Malformed installation was not reported explicitly.'
    }
    elseif ($MissingInstallation) {
        Assert-DeploymentCondition ($data['status'] -eq 'NotInstalled' -and -not (Test-Path -LiteralPath $Root)) 'Missing installation was created or misclassified.'
        Assert-DeploymentCondition ($errors.Count -eq 0) 'Missing-installation inspection had unexpected read errors.'
    }
    else {
        Assert-DeploymentCondition ($data['status'] -eq 'Installed') 'Fixture installation was not recognized.'
        Assert-DeploymentCondition (@($errors | Where-Object section -EQ 'scheduled task').Count -eq 1 -and $null -eq $data['task']) 'Missing fixture task did not remain an explicit read error.'
        $unexpected = @($errors | Where-Object { $_['section'] -ne 'scheduled task' -and -not ($ReleaseReadError -and $_['section'] -eq 'release manifest') })
        Assert-DeploymentCondition ($unexpected.Count -eq 0) 'Inspection had unexpected read errors beyond declared fixture refusals.'
        if ($ReleaseReadError) { Assert-DeploymentCondition (@($errors | Where-Object section -EQ 'release manifest').Count -eq 1) 'Invalid release pointer was not reported as a read error.' }
    }
    $data
}
$token = [guid]::NewGuid().ToString('N')
$workspace = Join-Path ([IO.Path]::GetTempPath()) ('pcai-usb4-doctor-' + $token)
try {
    [void][IO.Directory]::CreateDirectory($workspace)
    $controller = Join-Path $workspace 'controller.ps1'
    $policy = Join-Path $workspace 'policy.json'
    [IO.File]::WriteAllText($controller, '# inert controller payload; never executed')
    Write-FixtureJson $policy @{ schemaVersion = 1; adapterMac = '02-00-00-00-00-01' }
    $installed = @{ hostName = [Environment]::MachineName; revision = 'fixture'; installedAtUtc = [DateTimeOffset]::UtcNow.ToString('o')
        controllerPath = $controller; controllerSha256 = (Get-FileHash -LiteralPath $controller).Hash
        policyPath = $policy; policySha256 = (Get-FileHash -LiteralPath $policy).Hash
        statePath = (Join-Path $workspace 'owned-state.json'); reportPath = (Join-Path $workspace 'current.json')
        taskName = ('pcai-usb4-doctor-' + $token); taskPath = '\PC-AI-Doctor-Fixture\'
    }
    Write-FixtureJson (Join-Path $workspace 'installed.json') $installed
    Write-FixtureJson $installed.statePath @{ schemaVersion = 1 }
    $completeReport = @{ observedAt = [DateTimeOffset]::UtcNow.ToString('o'); inProgress = $false; phase = 'Complete'
        exitCode = 0; networkReady = $true; internetReady = $false
    }
    Write-FixtureJson $installed.reportPath $completeReport
    $architecture = switch ([Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString()) {
        'X64' { 'x86_64' } 'Arm64' { 'aarch64' } 'X86' { 'x86' } default { throw 'Unsupported Windows fixture architecture.' }
    }
    $controllerEntry = @{ path = 'Tools/SystemScripts/Networking/Invoke-Usb4PeerNetwork.ps1'; sha256 = $installed.controllerSha256 }
    $policyEntry = @{ path = 'Config/usb4-fleet/hosts/fixture.json'; sha256 = $installed.policySha256 }
    $baseManifest = @{ schemaVersion = 1; artifactType = 'pcai-usb4-fleet-release'; kind = 'deployable'
        source = @{ repository = 'https://example.invalid/declaration-only'; commit = ('1' * 40)
            registryPath = 'Config/usb4-fleet/fleet.json'; registrySha256 = ('0' * 64)
        }
        host   = @{ id = [Environment]::MachineName; platform = 'windows'; architecture = $architecture; backend = 'windows-peer' }
        policy = $policyEntry; files = @($controllerEntry, $policyEntry)
    } | ConvertTo-Json -Depth 12
    $manifestPath = Join-Path $workspace 'release.json'
    # Each mutation starts from fresh real JSON. No getter function or OS provider is mocked.
    foreach ($case in 'Matching', 'WrongHost', 'WrongArchitecture', 'ControllerHash', 'PolicyHash', 'DuplicateController') {
        $manifest = $baseManifest | ConvertFrom-Json -AsHashtable
        switch ($case) {
            'WrongHost' { $manifest['host']['id'] = 'different-fixture-host' }
            'WrongArchitecture' { $manifest['host']['architecture'] = if ($architecture -eq 'aarch64') { 'x86_64' } else { 'aarch64' } }
            'ControllerHash' { $manifest['files'][0]['sha256'] = '0' * 64 }
            'PolicyHash' { $manifest['files'][1]['sha256'] = '0' * 64; $manifest['policy']['sha256'] = '0' * 64 }
            'DuplicateController' { $manifest['files'] += $manifest['files'][0] }
        }
        Write-FixtureJson $manifestPath $manifest
        $data = Invoke-DeploymentFixture -Root $workspace -Manifest $manifestPath
        $identity = $data['sourceIdentity']
        Assert-DeploymentCondition ($identity['sourceCommit'] -eq ('1' * 40) -and
            $identity['sourceRepository'] -eq 'https://example.invalid/declaration-only' -and $identity['authenticity'] -eq 'NotVerified') 'Declared provenance was lost or promoted to authenticated identity.'
        $expected = $case -eq 'Matching'
        Assert-DeploymentCondition ($identity['byteMatched'] -eq $expected -and
            $identity['status'] -eq $(if ($expected) { 'ByteMatchedDeclaration' } else { 'DriftOrUnverified' })) "Wrong manifest verdict for $case."
        $failedCheck = switch ($case) {
            'WrongHost' { 'host' } 'WrongArchitecture' { 'architecture' }
            'ControllerHash' { 'controllerBytes' } 'PolicyHash' { 'policyBytes' } 'DuplicateController' { 'controllerBytes' }
        }
        foreach ($check in 'schema', 'sourceDeclaration', 'host', 'platform', 'architecture', 'backend', 'controllerBytes', 'policyBytes') {
            Assert-DeploymentCondition ($identity['checks'][$check] -eq ($check -ne $failedCheck)) "Unexpected $check check for $case."
        }
        Assert-DeploymentCondition ($data['ownedPending'] -eq $false) 'Empty transaction state became pending.'
        $script:groups++; Write-Output "PASS: $case"
    }
    Write-FixtureJson $manifestPath ($baseManifest | ConvertFrom-Json -AsHashtable)
    foreach ($inProgress in $true, $false) {
        $partial = @{ observedAt = [DateTimeOffset]::UtcNow.ToString('o'); phase = 'Rollback' }
        if ($inProgress) { $partial['inProgress'] = $true }
        Write-FixtureJson $installed.reportPath $partial
        $data = Invoke-DeploymentFixture -Root $workspace -Manifest $manifestPath
        Assert-DeploymentCondition ($data['controllerReport']['status'] -eq $(if ($inProgress) { 'InProgress' } else { 'Unknown' })) 'Partial report was misclassified as completed or failed networking.'
        Assert-DeploymentCondition ($null -eq $data['controllerReport']['networkReady'] -and $null -eq $data['controllerReport']['internetReady']) 'Partial report fabricated readiness.'
        $script:groups++; Write-Output "PASS: PartialReport inProgress=$inProgress"
    }
    # A stale root manifest must never conceal a missing or changed revision pointer.
    $revisionRoot = Join-Path $workspace 'revision'
    [void][IO.Directory]::CreateDirectory($revisionRoot)
    $revisionManifest = Join-Path $revisionRoot 'release.json'
    Write-FixtureJson $revisionManifest ($baseManifest | ConvertFrom-Json -AsHashtable)
    $staleManifest = $baseManifest | ConvertFrom-Json -AsHashtable
    $staleManifest['source']['commit'] = '2' * 40
    Write-FixtureJson $manifestPath $staleManifest
    foreach ($case in 'InstalledPointer', 'ManifestHashMismatch', 'MissingRecordedHash', 'MissingPointerTarget', 'RelativePointer', 'OutOfRootPointer', 'ExplicitOverride') {
        $installed['releaseManifestPath'] = $revisionManifest
        $installed['releaseManifestSha256'] = (Get-FileHash -LiteralPath $revisionManifest).Hash
        $expectError = $false
        switch ($case) {
            'ManifestHashMismatch' { $installed['releaseManifestSha256'] = '0' * 64 }
            'MissingRecordedHash' { $installed.Remove('releaseManifestSha256') }
            'MissingPointerTarget' { $installed['releaseManifestPath'] = Join-Path $revisionRoot 'absent.json'; $expectError = $true }
            'RelativePointer' { $installed['releaseManifestPath'] = 'revision/release.json'; $expectError = $true }
            'OutOfRootPointer' { $installed['releaseManifestPath'] = Join-Path ([IO.Path]::GetTempPath()) 'outside-owned-installation.json'; $expectError = $true }
            'ExplicitOverride' { $installed['releaseManifestPath'] = Join-Path $revisionRoot 'absent.json' }
        }
        Write-FixtureJson (Join-Path $workspace 'installed.json') $installed
        $options = @{ Root = $workspace; ReleaseReadError = $expectError }
        if ($case -eq 'ExplicitOverride') { $options.Manifest = $manifestPath }
        $data = Invoke-DeploymentFixture @options
        $identity = $data['sourceIdentity']
        $expectedMatch = $case -in 'InstalledPointer', 'ExplicitOverride'
        Assert-DeploymentCondition ($identity['byteMatched'] -eq $expectedMatch) "Wrong source binding for $case."
        Assert-DeploymentCondition ($identity['authenticity'] -eq 'NotVerified') 'Pointer hash was promoted to cryptographic authenticity.'
        Assert-DeploymentCondition ($data['releaseManifest']['selectedBy'] -eq $(if ($case -eq 'ExplicitOverride') { 'ExplicitOverride' } else { 'InstalledPointer' })) 'Manifest selection ignored the recorded pointer or explicit override.'
        if ($expectedMatch) {
            $expectedCommit = if ($case -eq 'ExplicitOverride') { '2' * 40 } else { '1' * 40 }
            Assert-DeploymentCondition ($identity['sourceCommit'] -eq $expectedCommit) 'Wrong manifest bytes were parsed.'
            Assert-DeploymentCondition ($data['releaseManifest']['actualSha256'] -eq (Get-FileHash -LiteralPath $data['releaseManifest']['path']).Hash) 'Parsed manifest hash differs from actual bytes.'
        }
        else {
            Assert-DeploymentCondition ($identity['status'] -eq 'DriftOrUnverified') 'Invalid pointer or hash silently fell back to the root manifest.'
        }
        switch ($case) {
            'InstalledPointer' { Assert-DeploymentCondition ($data['releaseManifest']['hashStatus'] -eq 'MatchedRecordedHash' -and $identity['checks']['recordedManifestBytes']) 'Recorded manifest bytes were not checked.' }
            'ManifestHashMismatch' { Assert-DeploymentCondition ($data['releaseManifest']['hashStatus'] -eq 'RecordedHashMismatch' -and -not $identity['checks']['recordedManifestBytes']) 'Changed manifest bytes were accepted.' }
            'MissingRecordedHash' { Assert-DeploymentCondition ($data['releaseManifest']['hashStatus'] -eq 'MissingOrInvalidRecordedHash') 'Missing recorded hash became a byte match.' }
        }
        $script:groups++; Write-Output "PASS: $case"
    }
    $malformed = Join-Path $workspace 'malformed'
    [void][IO.Directory]::CreateDirectory($malformed)
    Write-FixtureJson (Join-Path $malformed 'installed.json') @{}
    $null = Invoke-DeploymentFixture -Root $malformed -InvalidInstallation
    $script:groups++; Write-Output 'PASS: MalformedInstallation'
    $null = Invoke-DeploymentFixture -Root (Join-Path $workspace 'missing') -MissingInstallation
    $script:groups++; Write-Output 'PASS: MissingInstallationNoCreation'
    Write-Output "PASS: $script:checks assertions in $script:groups deployment-inspection groups; no internet probe requested."
}
finally {
    $absolute = [IO.Path]::GetFullPath($workspace)
    $temporary = [IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar
    if (-not $absolute.StartsWith($temporary, [StringComparison]::OrdinalIgnoreCase) -or
        [IO.Path]::GetFileName($absolute) -cne ('pcai-usb4-doctor-' + $token)) { throw 'Refusing cleanup outside the owned temporary fixture.' }
    if (Test-Path -LiteralPath $absolute) { Remove-Item -LiteralPath $absolute -Recurse -Force -ErrorAction Stop }
}
