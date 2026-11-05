#Requires -Version 7.0
<#
.SYNOPSIS
Detect unsafe deployment inspection using real files and bounded OS-provider seams.
.DESCRIPTION
Exercises the actual getter. Real temporary JSON and payload bytes drive binding,
path, freshness and refusal results. OS/provider doubles avoid hardware and packet
traffic; the real filesystem is used for the remaining boundary checks.
#>
[CmdletBinding()]
param([string]$SourcePath = (Join-Path $PSScriptRoot '../../Tools/SystemScripts/Networking/Get-Usb4PeerDeployment.ps1'))
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$script:checks = 0
$script:groups = 0
$script:osCalls = 0
# These real command-name boundaries fail closed if the old implicit OS queries
# return. They do not replace file, hash, path, clock or declaration logic.
function Get-NetAdapter { $script:osCalls++; throw 'Unexpected OS provider' }
function Get-NetIPInterface { $script:osCalls++; throw 'Unexpected OS provider' }
function Get-NetIPAddress { $script:osCalls++; throw 'Unexpected OS provider' }
function Get-NetRoute { $script:osCalls++; throw 'Unexpected OS provider' }
function Find-NetRoute { $script:osCalls++; throw 'Unexpected OS provider' }
function Get-NetConnectionProfile { $script:osCalls++; throw 'Unexpected OS provider' }
function Get-NetTCPConnection { $script:osCalls++; throw 'Unexpected OS provider' }
function Get-ScheduledTask { $script:osCalls++; throw 'Unexpected OS provider' }
function Get-ScheduledTaskInfo { $script:osCalls++; throw 'Unexpected OS provider' }
function Assert-DeploymentCondition {
    param([bool]$Condition, [string]$Message)
    if (-not $Condition) { throw $Message }
    $script:checks++
}
function Write-FixtureJson {
    param([string]$Path, [object]$Value)
    [IO.File]::WriteAllText($Path, ($Value | ConvertTo-Json -Depth 30), [Text.UTF8Encoding]::new($false))
}
function Get-FixtureHashes {
    @(Get-ChildItem -LiteralPath $workspace -Recurse -File | ForEach-Object {
            $_.FullName + ':' + (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash
        }) | Sort-Object
}
function Invoke-DeploymentFixture {
    param([string]$Root = $installation, [hashtable]$Providers, [string]$Manifest)
    $before = @(Get-FixtureHashes)
    $arguments = @{ InstallRoot = $Root }
    if ($Providers) { $arguments.Operations = $Providers }
    if ($Manifest) { $arguments.ReleaseManifestPath = $Manifest }
    $json = & $SourcePath @arguments
    $data = $json | ConvertFrom-Json -AsHashtable -ErrorAction Stop
    Assert-DeploymentCondition ($data.schema -ceq 'pcai.usb4-deployment/v1') 'Getter lost its JSON result contract.'
    Assert-DeploymentCondition ($data.internetProbe.status -ceq 'NotRun' -and $data.internetProbe.packetSent -eq $false) 'Default inspection attempted network traffic.'
    Assert-DeploymentCondition (@(Compare-Object $before @(Get-FixtureHashes)).Count -eq 0) 'Inspection wrote or changed fixture files.'
    $data
}
function New-FixtureProviders {
    $script:reads = [Collections.Generic.List[string]]::new()
    $script:metadata = [Collections.Generic.List[string]]::new()
    $script:osCalls = 0
    # The boundary spy never follows an unexpected path even when the production
    # guard is deliberately regressed. Assertions still detect the attempted call.
    $script:allowedReads = @(@((Join-Path $installation 'installed.json'), $controller,
        $policyPath, $installed.statePath, $installed.reportPath, $manifestPath,
        $installed['releaseManifestPath']) | Where-Object { $_ -is [string] })
    $script:allowedMetadata = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    foreach ($safePath in $script:allowedReads) {
        $cursor = [IO.Path]::GetFullPath($safePath)
        while ($cursor) {
            [void]$script:allowedMetadata.Add($cursor)
            $parent = [IO.Path]::GetDirectoryName($cursor)
            if ($parent -eq $cursor) { break }
            $cursor = $parent
        }
    }
    $boundaryState = @{ reads = $script:reads; metadata = $script:metadata
        allowedReads = $script:allowedReads; allowedMetadata = $script:allowedMetadata }
    $fixtureArchitecture = $architecture
    @{
        Runtime = { @{ platform = 'windows'; hostName = [Environment]::MachineName; architecture = $fixtureArchitecture } }.GetNewClosure()
        NowUtc = { [DateTimeOffset]'2030-01-01T00:00:00Z' }
        DriveType = { param($Root) 'Fixed' }
        ReadBytes = { param($Path, $Limit)
            $boundaryState.reads.Add($Path)
            if ($Path -notin $boundaryState.allowedReads) { throw 'Fixture refused an undeclared read boundary.' }
            $bytes = [IO.File]::ReadAllBytes($Path)
            if ($bytes.Length -gt $Limit) { throw 'fixture bounded read refused' }
            ,$bytes
        }.GetNewClosure()
        Attributes = { param($Path)
            $boundaryState.metadata.Add($Path)
            if (-not $boundaryState.allowedMetadata.Contains($Path)) { throw 'Fixture refused an undeclared metadata boundary.' }
            try { [IO.File]::GetAttributes($Path) }
            catch [IO.FileNotFoundException] { $null }
            catch [IO.DirectoryNotFoundException] { $null }
        }.GetNewClosure()
    }
}
$token = [guid]::NewGuid().ToString('N')
$workspace = Join-Path ([IO.Path]::GetTempPath()) ('pcai-usb4-doctor-' + $token)
$installation = Join-Path $workspace 'installation'
try {
    [void][IO.Directory]::CreateDirectory($installation)
    $controller = Join-Path $installation 'controller.ps1'
    $policyPath = Join-Path $installation 'policy.json'
    [IO.File]::WriteAllText($controller, 'throw "controller must never execute during inspection"')
    Write-FixtureJson $policyPath @{ schemaVersion = 1; adapterMac = '02-00-00-00-00-01' }
    $architecture = switch ([Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString()) {
        'X64' { 'x86_64' } 'Arm64' { 'aarch64' } 'X86' { 'x86' } default { 'Unknown' }
    }
    $installed = @{ hostName = [Environment]::MachineName; revision = 'fixture'; installedAtUtc = '2029-01-01T00:00:00Z'
        controllerPath = $controller; controllerSha256 = (Get-FileHash -LiteralPath $controller).Hash
        policyPath = $policyPath; policySha256 = (Get-FileHash -LiteralPath $policyPath).Hash
        statePath = (Join-Path $installation 'owned-state.json'); reportPath = (Join-Path $installation 'current.json')
        taskName = 'FixtureTask'; taskPath = '\PC-AI-Fixture\' }
    $installedJson = $installed | ConvertTo-Json -Depth 10
    $report = @{ observedAt = [DateTimeOffset]::UtcNow.ToString('o'); inProgress = $false; phase = 'Complete'; exitCode = 0; networkReady = $true; internetReady = $false }
    $manifest = @{ schemaVersion = 1; artifactType = 'pcai-usb4-fleet-release'; kind = 'deployable'
        source = @{ repository = 'https://github.com/example/declaration-only'; commit = ('1' * 40) }
        host = @{ id = [Environment]::MachineName; platform = 'windows'; architecture = $architecture; backend = 'windows-peer' }
        policy = @{ path = 'Config/fixture.json'; sha256 = $installed.policySha256; size = ([IO.FileInfo]$policyPath).Length }
        files = @(@{ path = 'Tools/SystemScripts/Networking/Invoke-Usb4PeerNetwork.ps1'; sha256 = $installed.controllerSha256; size = ([IO.FileInfo]$controller).Length },
            @{ path = 'Config/fixture.json'; sha256 = $installed.policySha256; size = ([IO.FileInfo]$policyPath).Length }) }
    $manifestJson = $manifest | ConvertTo-Json -Depth 15
    $manifestPath = Join-Path $installation 'release.json'
    Write-FixtureJson (Join-Path $installation 'installed.json') $installed
    Write-FixtureJson $installed.statePath @{ schemaVersion = 1 }
    Write-FixtureJson $installed.reportPath $report
    Write-FixtureJson $manifestPath $manifest

    # Selector group: AncestorsCheckedBeforeDescendantMetadata.
    # Protects local inspection against a stable ancestor junction to a remote
    # namespace; detects a leaf-first Attributes call before that junction is
    # refused. Needs real safe fixtures plus metadata/read boundary traps only,
    # no junction privilege or actual remote filesystem. Breadcrumb: path walk
    # in Assert-Usb4InspectionPath before Attributes and ReadBytes.
    foreach ($pointer in @('InstallRoot', 'controllerPath', 'policyPath', 'statePath', 'reportPath', 'releaseManifestPath')) {
        $unsafeAncestor = Join-Path $installation 'synthetic-remote-junction'
        $unsafePrefix = $unsafeAncestor + [IO.Path]::DirectorySeparatorChar
        $candidate = $installedJson | ConvertFrom-Json -AsHashtable
        $inspectionRoot = $installation
        if ($pointer -ceq 'InstallRoot') { $inspectionRoot = Join-Path $unsafeAncestor 'nested-installation' }
        else { $candidate[$pointer] = Join-Path $unsafeAncestor 'descendant.json' }
        Write-FixtureJson (Join-Path $installation 'installed.json') $candidate
        $ops = New-FixtureProviders
        $innerAttributes = $ops.Attributes; $innerRead = $ops.ReadBytes
        $pathComparison = if ($IsWindows) { [StringComparison]::OrdinalIgnoreCase } else { [StringComparison]::Ordinal }
        $ordering = @{ ancestorQueries = 0
            descendantMetadata = [Collections.Generic.List[string]]::new()
            descendantReads = [Collections.Generic.List[string]]::new() }
        $ops.Attributes = { param($Path)
            if ($Path.Equals($unsafeAncestor, $pathComparison)) {
                $ordering.ancestorQueries++
                return [IO.FileAttributes]::Directory -bor [IO.FileAttributes]::ReparsePoint
            }
            if ($Path.StartsWith($unsafePrefix, $pathComparison)) {
                $ordering.descendantMetadata.Add($Path)
                throw 'Fixture trapped descendant metadata without touching a remote filesystem.'
            }
            & $innerAttributes $Path
        }.GetNewClosure()
        $ops.ReadBytes = { param($Path, $Limit)
            if ($Path.Equals($unsafeAncestor, $pathComparison) -or $Path.StartsWith($unsafePrefix, $pathComparison)) {
                $ordering.descendantReads.Add($Path)
                throw 'Fixture trapped descendant content without touching a remote filesystem.'
            }
            & $innerRead $Path $Limit
        }.GetNewClosure()
        $data = Invoke-DeploymentFixture -Root $inspectionRoot -Providers $ops
        Assert-DeploymentCondition ($ordering.descendantMetadata.Count -eq 0 -and
            $ordering.descendantReads.Count -eq 0) 'Reparse ancestor was traversed before validation.'
        Assert-DeploymentCondition ($ordering.ancestorQueries -eq 1 -and
            $data.status -ceq 'Unknown' -and $data.sourceIdentity.byteMatched -eq $false -and
            @($data.readErrors).Count -gt 0) 'Unsafe ancestor was not explicitly refused.'
    }
    Write-FixtureJson (Join-Path $installation 'installed.json') ($installedJson | ConvertFrom-Json -AsHashtable)
    $script:groups++; Write-Output 'PASS: AncestorsCheckedBeforeDescendantMetadata'

    # Selector group: DeclarationDoesNotAuthenticateOrExecute.
    # Protects local evidence inspection; detects executing a hashed controller or
    # calling byte equality authentication. Needs pwsh7 and temp files, no hardware.
    # Breadcrumb: Get-Usb4PeerDeployment.ps1 manifest binding and result publication.
    $data = Invoke-DeploymentFixture
    Assert-DeploymentCondition ($data.status -in @('Installed', 'UnsupportedPlatform')) 'Valid local installation was lost.'
    Assert-DeploymentCondition ($data.sourceIdentity.authenticity -ceq 'NotVerified') 'Declaration became authenticated source.'
    Assert-DeploymentCondition ($data.ownedPending -eq $false) 'Empty owned state became pending.'
    $script:groups++; Write-Output 'PASS: DeclarationDoesNotAuthenticateOrExecute'

    # Selector group: FutureAndMalformedReportTimeStayUnknown.
    # Protects report freshness; detects treating a negative age as fresh.
    # Needs pwsh7 and real JSON files; no network/hardware. Breadcrumb: report age.
    $future = $report.Clone(); $future.observedAt = '2999-01-01T00:00:00Z'
    Write-FixtureJson $installed.reportPath $future
    $data = Invoke-DeploymentFixture
    Assert-DeploymentCondition ($null -eq $data.controllerReport.stale) 'Future report was incorrectly classified as fresh.'
    foreach ($badTime in @('not-a-time', @('2029-01-01T00:00:00Z'), $null)) {
        $future.observedAt = $badTime; Write-FixtureJson $installed.reportPath $future
        $data = Invoke-DeploymentFixture
        Assert-DeploymentCondition ($null -eq $data.controllerReport.stale) 'Malformed report time received a freshness verdict.'
    }
    $script:groups++; Write-Output 'PASS: FutureAndMalformedReportTimeStayUnknown'

    # Selector group: FileOnlyInspectionNeverQueriesWindowsOrInternet.
    # Protects bounded file-only inspection; detects restoring implicit task/CIM
    # providers or silently accepting an active probe. Needs temporary files and
    # fail-closed command boundaries, no OS modules or packet traffic.
    # Breadcrumb: getter file-only publication and ProbeInternet argument refusal.
    Assert-DeploymentCondition ($script:osCalls -eq 0) 'File-only inspection invoked an OS provider.'
    Assert-DeploymentCondition ($data.networkObservation.status -ceq 'NotCollected' -and
        $data.taskObservation.status -ceq 'NotCollected' -and $null -eq $data.task -and
        $null -eq $data.enrolledAdapters) 'Omitted OS state looked like an empty or qualified observation.'
    $caught = $null
    try { & $SourcePath -InstallRoot $installation -ProbeInternet | Out-Null }
    catch { $caught = $_.Exception.Message }
    Assert-DeploymentCondition ($null -ne $caught -and $caught -match 'not supported') 'File-only inspection accepted an active probe.'
    $script:groups++; Write-Output 'PASS: FileOnlyInspectionNeverQueriesWindowsOrInternet'

    # Selector group: RootKindSeparatesMissingDirectoryFromFile.
    # Protects installation-state classification; detects reporting NotInstalled
    # when the proposed root is an existing regular file. Needs real temporary
    # files and the default file providers, no OS queries or injected refusals.
    # Breadcrumb: root directory admission before selecting installed.json.
    $missingRoot = Join-Path $workspace 'missing-installation'
    $data = Invoke-DeploymentFixture -Root $missingRoot
    Assert-DeploymentCondition ($data.status -in @('NotInstalled', 'UnsupportedPlatform') -and
        @($data.readErrors).Count -eq 0 -and -not (Test-Path -LiteralPath $missingRoot)) 'Missing installation was created or misclassified.'
    $data = Invoke-DeploymentFixture -Root $controller
    Assert-DeploymentCondition ($data.status -ceq 'Unknown' -and
        $null -eq $data.storage.installedPath -and
        @($data.readErrors | Where-Object section -EQ 'installation root').Count -eq 1) 'Regular-file installation root was incorrectly classified as absent.'
    $script:groups++; Write-Output 'PASS: RootKindSeparatesMissingDirectoryFromFile'

    # Selector group: RejectedPointersNeverReachReadOrHash.
    # Protects local-only inspection; detects following UNC, outside-root, relative,
    # reserved or reparse pointers before validation. Needs temp files and read seam.
    # Breadcrumb: installation/root/file validation before ReadBytes provider.
    $report.observedAt = '2029-12-31T23:59:00Z'; Write-FixtureJson $installed.reportPath $report
    $data = Invoke-DeploymentFixture -Providers (New-FixtureProviders)
    Assert-DeploymentCondition ($data.sourceIdentity.byteMatched -eq $true) 'Positive file-boundary fixture did not inspect matching bytes.'
    foreach ($field in @('controllerPath', 'policyPath', 'statePath', 'reportPath', 'releaseManifestPath')) {
        foreach ($badPath in @('\\untrusted.invalid\share\data.json', (Join-Path $workspace 'outside.json'), 'relative.json', (Join-Path $installation '$null'), (Join-Path $installation 'CON.txt'))) {
            $candidate = $installedJson | ConvertFrom-Json -AsHashtable
            $candidate[$field] = $badPath
            Write-FixtureJson (Join-Path $installation 'installed.json') $candidate
            $ops = New-FixtureProviders
            $data = Invoke-DeploymentFixture -Providers $ops
            Assert-DeploymentCondition (-not $script:reads.Contains($badPath)) "Rejected $field reached the file provider."
            Assert-DeploymentCondition (-not $script:metadata.Contains($badPath)) "Rejected $field reached the metadata provider."
            Assert-DeploymentCondition (@($data.readErrors).Count -gt 0) "Invalid $field did not report refusal."
            Assert-DeploymentCondition (-not $data.sourceIdentity -or $data.sourceIdentity.byteMatched -ne $true) 'Invalid pointer received a matched declaration.'
        }
    }
    Write-FixtureJson (Join-Path $installation 'installed.json') $installed
    foreach ($refusedFlag in @([long][IO.FileAttributes]::ReparsePoint, [long][IO.FileAttributes]::Offline, 0x00040000, 0x00400000)) {
        $ops = New-FixtureProviders
        $realAttributes = $ops.Attributes; $flaggedController = $controller
        $ops.Attributes = { param($Path)
            if ($Path -ceq $flaggedController) { $refusedFlag }
            else { & $realAttributes $Path }
        }.GetNewClosure()
        $data = Invoke-DeploymentFixture -Providers $ops
        Assert-DeploymentCondition (-not $script:reads.Contains($controller)) 'Reparse/offline/recall controller reached the read provider.'
        Assert-DeploymentCondition (@($data.readErrors).Count -gt 0) 'Attribute refusal was hidden.'
    }
    $script:groups++; Write-Output 'PASS: RejectedPointersNeverReachReadOrHash'

    # Selector group: InvalidRootNeverCallsProviders.
    # Protects pre-I/O root refusal; detects probing a UNC/root/reserved input.
    # Needs provider spies only; breadcrumb: initial root validation.
    foreach ($badRoot in @('\\untrusted.invalid\share', 'relative-root', (Join-Path $workspace 'NUL.txt'), [IO.Path]::GetPathRoot($installation))) {
        $ops = New-FixtureProviders
        $data = Invoke-DeploymentFixture -Root $badRoot -Providers $ops
        Assert-DeploymentCondition ($script:reads.Count -eq 0 -and $script:metadata.Count -eq 0 -and $script:osCalls -eq 0) 'Invalid root reached an external provider.'
        Assert-DeploymentCondition ($data.status -ceq 'Unknown' -and @($data.readErrors).Count -gt 0) 'Invalid root was hidden or misclassified.'
    }
    $script:groups++; Write-Output 'PASS: InvalidRootNeverCallsProviders'

    # Selector group: RemoteOrUnknownDriveRefusedBeforeAttributes.
    # Protects obvious mapped-drive refusal; detects touching network-backed
    # files before classifying the root. Needs cached drive provider, no real
    # remote drive. Breadcrumb: root DriveType admission before Attributes.
    foreach ($driveType in @('Network', 'Unknown', 'NoRootDirectory')) {
        $ops = New-FixtureProviders
        $ops.DriveType = { param($Root) $driveType }.GetNewClosure()
        $data = Invoke-DeploymentFixture -Providers $ops
        Assert-DeploymentCondition ($script:reads.Count -eq 0 -and $script:metadata.Count -eq 0 -and
            $data.status -ceq 'Unknown') 'Remote or unknown root reached filesystem providers.'
    }
    $script:groups++; Write-Output 'PASS: RemoteOrUnknownDriveRefusedBeforeAttributes'

    # Selector group: TypedUniqueManifestRecords.
    # Protects declaration binding; detects PowerShell collection comparison
    # acceptance, duplicate records, invalid hashes, sizes and schemas. Needs real
    # manifest bytes and synthetic Windows identity; breadcrumb: manifest checks.
    foreach ($case in @('Matching', 'HostArray', 'PlatformArray', 'ArchitectureArray', 'CommitArray', 'SchemaArray', 'FileHashArray', 'FilePathArray', 'PolicyHashArray', 'DuplicateFile', 'NegativeSize', 'BooleanSize', 'WrongHost', 'WrongHash', 'DuplicateJsonKey')) {
        $candidate = $manifestJson | ConvertFrom-Json -AsHashtable
        switch ($case) {
            'HostArray' { $candidate.host.id = @([Environment]::MachineName) }
            'PlatformArray' { $candidate.host.platform = @('windows') }
            'ArchitectureArray' { $candidate.host.architecture = @($architecture) }
            'CommitArray' { $candidate.source.commit = @('1' * 40) }
            'SchemaArray' { $candidate.schemaVersion = @(1) }
            'FileHashArray' { $candidate.files[0].sha256 = @($installed.controllerSha256) }
            'FilePathArray' { $candidate.files[0].path = @($candidate.files[0].path) }
            'PolicyHashArray' { $candidate.policy.sha256 = @($installed.policySha256) }
            'DuplicateFile' { $candidate.files += $candidate.files[0] }
            'NegativeSize' { $candidate.files[0].size = -1 }
            'BooleanSize' { $candidate.files[0].size = $true }
            'WrongHost' { $candidate.host.id = 'unrelated-host' }
            'WrongHash' { $candidate.files[0].sha256 = '0' * 64 }
        }
        Write-FixtureJson $manifestPath $candidate
        if ($case -ceq 'DuplicateJsonKey') { [IO.File]::WriteAllText($manifestPath, '{"schemaVersion":1,"schemaVersion":1}') }
        $ops = New-FixtureProviders
        $data = Invoke-DeploymentFixture -Providers $ops
        $expected = $case -ceq 'Matching'
        Assert-DeploymentCondition ($data.sourceIdentity.byteMatched -eq $expected) "Wrong binding result for $case."
        Assert-DeploymentCondition ($data.sourceIdentity.authenticity -ceq 'NotVerified') 'Byte declaration became authenticated.'
        if ($expected) {
            Assert-DeploymentCondition ($data.controllerReport.stale -eq $false -and
                $data.controllerReport.ageSeconds -eq 60) 'Valid ISO report timestamp lost its scalar type or freshness.'
        }
    }
    $script:groups++; Write-Output 'PASS: TypedUniqueManifestRecords'

    # Selector group: RecordedManifestPointerNeverFallsBack.
    # Protects revision custody; detects stale root fallback after missing or
    # changed recorded manifest. Needs real files, no hardware. Breadcrumb: selection.
    Write-FixtureJson $manifestPath ($manifestJson | ConvertFrom-Json -AsHashtable)
    $installed.releaseManifestPath = Join-Path $installation 'missing-release.json'
    $installed.releaseManifestSha256 = '0' * 64
    Write-FixtureJson (Join-Path $installation 'installed.json') $installed
    $data = Invoke-DeploymentFixture -Providers (New-FixtureProviders)
    Assert-DeploymentCondition ($data.releaseManifest.selectedBy -ceq 'InstalledPointer' -and $data.sourceIdentity.byteMatched -eq $false) 'Missing pointer fell back to the legacy manifest.'
    $installed.releaseManifestPath = $manifestPath
    Write-FixtureJson (Join-Path $installation 'installed.json') $installed
    $data = Invoke-DeploymentFixture -Providers (New-FixtureProviders)
    Assert-DeploymentCondition ($data.releaseManifest.hashStatus -ceq 'RecordedHashMismatch' -and $data.sourceIdentity.byteMatched -eq $false) 'Recorded hash mismatch was accepted.'
    $installed.releaseManifestSha256 = (Get-FileHash -LiteralPath $manifestPath).Hash
    Write-FixtureJson (Join-Path $installation 'installed.json') $installed
    $data = Invoke-DeploymentFixture -Providers (New-FixtureProviders)
    Assert-DeploymentCondition ($data.sourceIdentity.byteMatched -eq $true) 'Matching recorded manifest failed.'
    $script:groups++; Write-Output 'PASS: RecordedManifestPointerNeverFallsBack'

    # Selector group: PendingRecoveryAndPartialReportAreNotReadiness.
    # Protects interrupted-transaction custody; detects claiming healthy/completed
    # from a partial/pending report. Needs JSON and fake read-only OS state.
    # Breadcrumb: ownedPending and historical report projection.
    Write-FixtureJson $installed.statePath @{ schemaVersion = 1; pendingTransaction = @{ phase = 'Rollback' } }
    Write-FixtureJson $installed.reportPath @{ observedAt = '2029-12-31T23:59:00Z'; phase = 'Rollback' }
    $data = Invoke-DeploymentFixture -Providers (New-FixtureProviders)
    Assert-DeploymentCondition ($data.ownedPending -eq $true -and $data.controllerReport.status -ceq 'Unknown') 'Pending partial report became completed or lost recovery state.'
    Assert-DeploymentCondition ($null -eq $data.controllerReport.networkReady -and $data.controllerReport.qualification -ceq 'HistoricalControllerObservation') 'Historical evidence became current network qualification.'
    $script:groups++; Write-Output 'PASS: PendingRecoveryAndPartialReportAreNotReadiness'

    # Selector group: UnsupportedPlatformAndBoundedReads.
    # Protects portable use and bounded file reads; detects Windows provider calls
    # on Linux or accepting over-budget bytes. Needs provider seam and real files.
    # Breadcrumb: runtime guard and bounded ReadBytes boundary.
    $ops = New-FixtureProviders
    $ops.Runtime = { @{ platform = 'linux'; hostName = [Environment]::MachineName; architecture = $architecture } }.GetNewClosure()
    $data = Invoke-DeploymentFixture -Providers $ops
    Assert-DeploymentCondition ($data.status -ceq 'UnsupportedPlatform' -and $script:osCalls -eq 0) 'Unsupported OS called Windows providers or claimed deployment qualification.'
    $ops = New-FixtureProviders
    $ops.ReadBytes = { param($Path, $Limit) ,([byte[]]::new($Limit + 1)) }
    $data = Invoke-DeploymentFixture -Providers $ops
    Assert-DeploymentCondition ($data.status -ceq 'Unknown' -and @($data.readErrors).Count -gt 0) 'Over-budget installation bytes were accepted.'
    $script:groups++; Write-Output 'PASS: UnsupportedPlatformAndBoundedReads'
    Write-Output "PASS: $script:checks assertions in $script:groups deployment-inspection groups; no packet probe or mutation requested."
}
finally {
    $absolute = [IO.Path]::GetFullPath($workspace)
    $temporary = [IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar
    if (-not $absolute.StartsWith($temporary, [StringComparison]::OrdinalIgnoreCase) -or [IO.Path]::GetFileName($absolute) -cne ('pcai-usb4-doctor-' + $token)) { throw 'Refusing cleanup outside owned fixture.' }
    if (Test-Path -LiteralPath $absolute) { Remove-Item -LiteralPath $absolute -Recurse -Force -ErrorAction Stop }
}
