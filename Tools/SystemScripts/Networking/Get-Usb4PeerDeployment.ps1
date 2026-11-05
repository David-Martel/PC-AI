#Requires -Version 7.0
<#
.SYNOPSIS
Inspect local USB4 deployment evidence without changing network or task state.
.DESCRIPTION
Writes file evidence as JSON to stdout without querying tasks, network providers
or peers. Task and network observations remain NotCollected. Filesystem inspection
is portable; non-Windows hosts remain UnsupportedPlatform for deployment identity.
ProbeInternet is refused; use the maintained observation collector separately.
A matching declaration or historical report never qualifies current hardware or
authenticates a release signature.
See docs/usb4/peer-networking.md for deployment and recovery boundaries.
#>
[CmdletBinding()]
param(
    [string]$InstallRoot = (Join-Path ([Environment]::GetFolderPath('CommonApplicationData')) 'pcai-usb4/peer-network'),
    [string]$ReleaseManifestPath,
    [switch]$ProbeInternet,
    [Parameter(DontShow)][hashtable]$Operations,
    [Alias('h', 'help')][switch]$ShowHelp
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
if ($ShowHelp) { Get-Help $PSCommandPath -Detailed; return }
if ($ProbeInternet) { throw 'ProbeInternet is not supported by file-only inspection.' }

function Read-Usb4BytesBounded {
    <# .SYNOPSIS Read one file within a fixed byte budget. #>
    param([string]$Path, [int]$Limit)
    $stream = [IO.File]::Open($Path, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::Read)
    $memory = [IO.MemoryStream]::new()
    try {
        if ($stream.Length -gt $Limit) { throw 'File exceeds inspection byte budget.' }
        $buffer = [byte[]]::new(8192)
        while ($true) {
            $count = $stream.Read($buffer, 0, [Math]::Min($buffer.Length, $Limit + 1 - [int]$memory.Length))
            if ($count -eq 0) { break }
            $memory.Write($buffer, 0, $count)
            if ($memory.Length -gt $Limit) { throw 'File grew beyond inspection byte budget.' }
        }
        ,$memory.ToArray()
    }
    finally { $memory.Dispose(); $stream.Dispose() }
}
$providers = @{
    Runtime = {
        $platform = if ($IsWindows) { 'windows' } elseif ($IsLinux) { 'linux' } elseif ($IsMacOS) { 'macos' } else { 'unknown' }
        $architecture = switch ([Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString()) {
            'X64' { 'x86_64' } 'Arm64' { 'aarch64' } 'X86' { 'x86' } default { 'Unknown' }
        }
        @{ platform = $platform; hostName = [Environment]::MachineName; architecture = $architecture }
    }
    NowUtc = { [DateTimeOffset]::UtcNow }
    DriveType = { param($Root)
        if ($IsWindows) { [IO.DriveInfo]::new($Root).DriveType.ToString() }
        else { 'NotClassified' }
    }
    ReadBytes = { param($Path, $Limit) Read-Usb4BytesBounded $Path $Limit }
    Attributes = { param($Path)
        try { [IO.File]::GetAttributes($Path) }
        catch [IO.FileNotFoundException] { $null }
        catch [IO.DirectoryNotFoundException] { $null }
    }
}
if ($Operations) {
    foreach ($key in $Operations.Keys) {
        if (-not $providers.ContainsKey($key) -or $Operations[$key] -isnot [scriptblock]) { throw 'Unknown or non-scriptblock inspection provider.' }
        $providers[$key] = $Operations[$key]
    }
}
$readErrors = [Collections.Generic.List[object]]::new()
$rootDriveType = 'NotCollected'
function Read-DeploymentSection {
    param([string]$Section, [scriptblock]$Read)
    try { & $Read } catch { $readErrors.Add(@{ section = $Section; error = $_.Exception.Message }) }
}
function Assert-Usb4InspectionPath {
    <# .SYNOPSIS Refuse nonlocal, reserved or escaping paths before filesystem queries. #>
    param([object]$Path, [string]$Root, [switch]$Directory)
    if ($Path -isnot [string] -or [string]::IsNullOrWhiteSpace($Path) -or $Path.Length -gt 4096 -or
        $Path -match '[\x00-\x1f]|^[\\/]{2}|::|^[A-Za-z][A-Za-z0-9+.-]*://' -or
        -not [IO.Path]::IsPathFullyQualified($Path)) { throw 'Inspection requires a local absolute filesystem path.' }
    if ($Path.Contains(':') -and $Path -notmatch '^[A-Za-z]:[\\/][^:]*$') { throw 'Alternate streams and provider paths are not inspection targets.' }
    foreach ($part in ($Path -split '[\\/]')) {
        if ($part -in @('.', '..') -or $part -match '^(?i:\$null|AUX|CON|NUL|PRN|COM[1-9]|LPT[1-9])(?:\..*)?[ .]*$' -or
            ($part -and $part -cne $part.TrimEnd(' ', '.'))) { throw 'Reserved or ambiguous inspection path component.' }
    }
    $full = [IO.Path]::GetFullPath($Path)
    $comparison = if ($IsWindows) { [StringComparison]::OrdinalIgnoreCase } else { [StringComparison]::Ordinal }
    if ($Directory -and $full.TrimEnd('\', '/') -eq [IO.Path]::GetPathRoot($full).TrimEnd('\', '/')) { throw 'A filesystem root is not an installation directory.' }
    if ($Root) {
        $prefix = $Root.TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar
        if (-not $full.StartsWith($prefix, $comparison)) { throw 'Inspection pointer is outside the installation root.' }
    }
    if ($Directory) {
        # Classify the drive root once, before file attributes. This refuses
        # ordinary mapped network drives without querying a target file. It does
        # not discover transparent remote mounts behind a fixed local namespace.
        $driveType = & $providers.DriveType ([IO.Path]::GetPathRoot($full))
        if ($driveType -isnot [string] -or $driveType -notin @('Fixed', 'Removable', 'Ram', 'NotClassified') -or
            ($IsWindows -and $driveType -ceq 'NotClassified')) { throw 'Inspection installation drive is remote, missing or unclassified.' }
        $script:rootDriveType = $driveType
    }
    # Containment is checked before Attributes can touch a pointer. This is an
    # inspection of a mutable filesystem, not an atomic no-follow file capability.
    # Collect components lexically, then inspect the filesystem root first.
    # A descendant Attributes call can already traverse an unchecked junction.
    $components = [Collections.Generic.List[string]]::new()
    $cursor = $full
    while ($cursor) {
        $components.Add($cursor)
        $parent = [IO.Path]::GetDirectoryName($cursor)
        if ($parent -eq $cursor) { break }
        $cursor = $parent
    }
    $components.Reverse()
    foreach ($cursor in $components) {
        $attributes = & $providers.Attributes $cursor
        if ($null -ne $attributes -and ($Directory -or $cursor -cne $full) -and
            -not ([long]$attributes -band [long][IO.FileAttributes]::Directory)) { throw 'Installation root and existing path ancestors must be directories.' }
        # Offline/RecallOnOpen/RecallOnDataAccess can hydrate file content.
        $refusedAttributes = [long][IO.FileAttributes]::ReparsePoint -bor [long][IO.FileAttributes]::Offline -bor 0x00040000 -bor 0x00400000
        if ($null -ne $attributes -and ([long]$attributes -band $refusedAttributes)) { throw 'Inspection path traverses a reparse, offline or recall-on-access item.' }
    }
    $full
}
function Read-Usb4InspectionBytes {
    <# .SYNOPSIS Recheck a contained path and bound bytes returned by its reader. #>
    param([object]$Path, [int]$Limit, [switch]$Optional)
    $safe = Assert-Usb4InspectionPath $Path $safeRoot
    $attributes = & $providers.Attributes $safe
    if ($null -eq $attributes) {
        if ($Optional) { return }
        throw 'Inspection file is missing.'
    }
    if ($attributes -band [IO.FileAttributes]::Directory) { throw 'Inspection target must be a file.' }
    [byte[]]$bytes = & $providers.ReadBytes $safe $Limit
    if ($bytes.Length -gt $Limit) { throw 'Reader exceeded inspection byte budget.' }
    ,$bytes
}
function Assert-Usb4JsonKeys {
    <# .SYNOPSIS Reject duplicate object keys before PowerShell JSON conversion. #>
    param([System.Text.Json.JsonElement]$Element)
    if ($Element.ValueKind -eq [System.Text.Json.JsonValueKind]::Object) {
        $keys = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
        foreach ($property in $Element.EnumerateObject()) {
            if (-not $keys.Add($property.Name)) { throw 'Duplicate JSON object key.' }
            Assert-Usb4JsonKeys $property.Value
        }
    }
    elseif ($Element.ValueKind -eq [System.Text.Json.JsonValueKind]::Array) {
        foreach ($child in $Element.EnumerateArray()) { Assert-Usb4JsonKeys $child }
    }
}
function ConvertFrom-Usb4InspectionJson {
    <# .SYNOPSIS Parse bounded strict UTF-8 JSON with unique object keys. #>
    param([byte[]]$Bytes)
    $text = [Text.UTF8Encoding]::new($false, $true).GetString($Bytes).TrimStart([char]0xfeff)
    $options = [System.Text.Json.JsonDocumentOptions]::new(); $options.MaxDepth = 32
    $document = [System.Text.Json.JsonDocument]::Parse($text, $options)
    try {
        Assert-Usb4JsonKeys $document.RootElement
        $value = ConvertFrom-Usb4JsonElement $document.RootElement
    }
    finally { $document.Dispose() }
    if ($value -isnot [Collections.IDictionary]) { throw 'Inspection JSON must be an object.' }
    $value
}
function ConvertFrom-Usb4JsonElement {
    <# .SYNOPSIS Preserve JSON scalar types instead of coercing timestamps. #>
    param([System.Text.Json.JsonElement]$Element)
    switch ($Element.ValueKind) {
        ([System.Text.Json.JsonValueKind]::Object) {
            $map = @{}
            foreach ($property in $Element.EnumerateObject()) { $map[$property.Name] = ConvertFrom-Usb4JsonElement $property.Value }
            return $map
        }
        ([System.Text.Json.JsonValueKind]::Array) {
            $items = [Collections.Generic.List[object]]::new()
            foreach ($child in $Element.EnumerateArray()) { $items.Add((ConvertFrom-Usb4JsonElement $child)) }
            return ,$items.ToArray()
        }
        ([System.Text.Json.JsonValueKind]::String) { return $Element.GetString() }
        ([System.Text.Json.JsonValueKind]::Number) {
            [long]$integer = 0
            if ($Element.TryGetInt64([ref]$integer)) { return $integer }
            return $Element.GetDouble()
        }
        ([System.Text.Json.JsonValueKind]::True) { return $true }
        ([System.Text.Json.JsonValueKind]::False) { return $false }
        ([System.Text.Json.JsonValueKind]::Null) { return $null }
        default { throw 'Unsupported JSON element.' }
    }
}
function Read-DeploymentJson {
    param([object]$Path, [int]$Limit = 1048576, [switch]$Optional)
    $bytes = Read-Usb4InspectionBytes $Path $Limit -Optional:$Optional
    if ($null -ne $bytes) { ConvertFrom-Usb4InspectionJson $bytes }
}
function Get-Usb4InspectionHash {
    param([byte[]]$Bytes)
    $hash = [Security.Cryptography.SHA256]::Create()
    try { [BitConverter]::ToString($hash.ComputeHash($Bytes)).Replace('-', '') } finally { $hash.Dispose() }
}
function Test-Usb4InspectionHash {
    param([object]$Value)
    $Value -is [string] -and $Value -match '^[a-fA-F0-9]{64}$'
}
function Assert-Usb4ManifestRecord {
    <# .SYNOPSIS Require one scalar relative payload path, hash and byte length. #>
    param([object]$Record)
    if ($Record -isnot [Collections.IDictionary] -or $Record['path'] -isnot [string] -or
        $Record['path'] -notmatch '^[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*$' -or
        @($Record['path'].Split('/') | Where-Object { $_ -in @('.', '..') -or $_ -match '^(?i:\$null|AUX|CON|NUL|PRN|COM[1-9]|LPT[1-9])(?:\..*)?$' }).Count -or
        -not (Test-Usb4InspectionHash $Record['sha256']) -or
        ($Record['size'] -isnot [int] -and $Record['size'] -isnot [long]) -or $Record['size'] -lt 0) { throw 'Invalid scalar manifest file record.' }
}
$runtime = & $providers.Runtime
$now = & $providers.NowUtc
if ($runtime -isnot [Collections.IDictionary] -or $runtime['platform'] -isnot [string] -or
    $runtime['hostName'] -isnot [string] -or $runtime['architecture'] -isnot [string] -or $now -isnot [DateTimeOffset]) { throw 'Inspection runtime provider returned invalid data.' }
$result = [ordered]@{
    schema = 'pcai.usb4-deployment/v1'; observedAtUtc = $now.ToUniversalTime().ToString('o')
    hostName = $runtime.hostName; status = 'NotInstalled'; platform = $runtime.platform; qualification = 'InspectionOnly'
    storage = @{ installRoot = $InstallRoot; installedPath = $null }
    deployment = $null
    sourceIdentity = @{ status = 'NotDeclared'; byteMatched = $false; authenticity = 'NotVerified'; checks = @{} }
    releaseManifest = $null; task = $null; ownedPending = $null; recoveryState = 'Unknown'
    controllerReport = $null; enrolledAdapters = $null; ipv4 = $null; ipv6 = $null
    ncsiProfiles = $null; establishedTcpByLocalIP = $null
    networkObservation = @{ status = 'NotCollected'; reason = 'FileOnlyInspection'; packetSent = $false }
    taskObservation = @{ status = 'NotCollected'; reason = 'FileOnlyInspection' }
    internetProbe = @{ status = 'NotRun'; packetSent = $false; reason = 'FileOnlyInspection' }
}
$safeRoot = Read-DeploymentSection 'installation root' { Assert-Usb4InspectionPath $InstallRoot -Directory }
$result.storage.driveType = $rootDriveType
$installed = $null; $policy = $null; $fileEvidence = @{}
if ($safeRoot) {
    $installedPath = Join-Path $safeRoot 'installed.json'; $result.storage.installedPath = $installedPath
    $installed = Read-DeploymentSection 'installation' {
        $record = Read-DeploymentJson $installedPath -Optional
        if ($null -eq $record) { return }
        foreach ($field in @('controllerPath', 'policyPath', 'statePath', 'reportPath', 'hostName', 'taskName', 'taskPath')) {
            if ($record[$field] -isnot [string] -or [string]::IsNullOrWhiteSpace($record[$field])) { throw 'Installation fields must be nonempty scalar strings.' }
        }
        if (-not (Test-Usb4InspectionHash $record['controllerSha256']) -or -not (Test-Usb4InspectionHash $record['policySha256'])) { throw 'Installation hashes must be scalar SHA256 values.' }
        if ($record['taskName'] -match '[\\/:*?"<>|\x00-\x1f]' -or $record['taskName'].Length -gt 128 -or
            $record['taskPath'] -notmatch '^\\(?:[A-Za-z0-9_. -]+\\)*$') { throw 'Installation task pointer is invalid.' }
        foreach ($field in @('controllerPath', 'policyPath', 'statePath', 'reportPath')) { $record[$field] = Assert-Usb4InspectionPath $record[$field] $safeRoot }
        if ($record.Contains('releaseManifestPath')) { $record['releaseManifestPath'] = Assert-Usb4InspectionPath $record['releaseManifestPath'] $safeRoot }
        $record
    }
}
if ($installed) {
    $result.status = 'Installed'
    foreach ($kind in @('controller', 'policy', 'state', 'report')) { $result.storage["${kind}Path"] = $installed["${kind}Path"] }
    $result.deployment = @{ revision = $installed['revision']; installedAtUtc = $installed['installedAtUtc']; hashes = @() }
    foreach ($kind in @('controller', 'policy')) {
        $limit = if ($kind -eq 'controller') { 16777216 } else { 1048576 }
        $bytes = Read-DeploymentSection "$kind bytes" { Read-Usb4InspectionBytes $installed["${kind}Path"] $limit }
        $actual = $null
        if ($null -ne $bytes) { $actual = Get-Usb4InspectionHash $bytes; $fileEvidence[$kind] = @{ sha256 = $actual; size = $bytes.Length } }
        $result.deployment.hashes += @{ kind = $kind; expectedSha256 = $installed["${kind}Sha256"]; actualSha256 = $actual
            matches = $(if ($actual) { $actual -ieq $installed["${kind}Sha256"] } else { $null }) }
        if ($kind -eq 'policy' -and $null -ne $bytes) { $policy = Read-DeploymentSection 'policy' { ConvertFrom-Usb4InspectionJson $bytes } }
    }
    $state = Read-DeploymentSection 'owned state' { Read-DeploymentJson $installed['statePath'] -Limit 4194304 -Optional }
    if ($state) {
        $result.ownedPending = $state.Contains('pendingTransaction')
        $result.recoveryState = if ($result.ownedPending) { 'PendingOwnershipReview' } else { 'NoPendingTransactionRecorded' }
    }
    $report = Read-DeploymentSection 'controller report' { Read-DeploymentJson $installed['reportPath'] -Limit 4194304 -Optional }
    if ($report) {
        $age = Read-DeploymentSection 'report timestamp' {
            if ($report['observedAt'] -isnot [string] -or $report['observedAt'] -notmatch '(?:Z|[+-][0-9]{2}:[0-9]{2})$') { throw 'Report timestamp must be a scalar time with an explicit offset.' }
            $time = [DateTimeOffset]::MinValue
            if (-not [DateTimeOffset]::TryParse($report['observedAt'], [Globalization.CultureInfo]::InvariantCulture, [Globalization.DateTimeStyles]::None, [ref]$time)) { throw 'Report timestamp is invalid.' }
            $seconds = ($now - $time).TotalSeconds
            if ($seconds -lt 0) { throw 'Report timestamp is in the future; freshness is unknown.' }
            $seconds
        }
        $inProgress = $report['inProgress'] -is [bool] -and $report['inProgress']
        $completed = ($report['inProgress'] -is [bool] -and -not $report['inProgress']) -or ($report['phase'] -is [string] -and $report['phase'] -ceq 'Complete')
        $result.controllerReport = @{ status = $(if ($inProgress) { 'InProgress' } elseif ($completed) { 'Completed' } else { 'Unknown' })
            ageSeconds = $age; stale = $(if ($null -ne $age) { $age -gt 180 } else { $null })
            phase = $report['phase']; exitCode = $report['exitCode']; qualification = 'HistoricalControllerObservation'
            networkReady = $(if ($report['networkReady'] -is [bool]) { $report['networkReady'] } else { $null })
            internetReady = $(if ($report['internetReady'] -is [bool]) { $report['internetReady'] } else { $null }) }
    }
    else { $result.controllerReport = @{ status = 'MissingOrUnreadable'; qualification = 'HistoricalControllerObservation' } }
}
elseif ($readErrors.Count) { $result.status = 'Unknown' }

# A rejected installation cannot select fallback metadata from a different cohort.
if ($safeRoot -and ($installed -or $readErrors.Count -eq 0)) {
    $pointerRecorded = $installed -and $installed.Contains('releaseManifestPath')
    $selection = if ($ReleaseManifestPath) { 'ExplicitOverride' } elseif ($pointerRecorded) { 'InstalledPointer' } else { 'LegacyRoot' }
    $manifestPath = if ($ReleaseManifestPath) { $ReleaseManifestPath } elseif ($pointerRecorded) { $installed['releaseManifestPath'] } else { Join-Path $safeRoot 'release.json' }
    $manifestBytes = Read-DeploymentSection 'release manifest' { Read-Usb4InspectionBytes $manifestPath 2097152 -Optional:($selection -eq 'LegacyRoot') }
    if ($null -ne $manifestBytes -or $selection -ne 'LegacyRoot') {
        $release = @{ selectedBy = $selection; path = $manifestPath; actualSha256 = $null
            recordedSha256 = $(if ($installed) { $installed['releaseManifestSha256'] } else { $null }); hashStatus = 'Unreadable' }
        $identity = @{ status = 'DriftOrUnverified'; byteMatched = $false; authenticity = 'NotVerified'; checks = @{}
            note = 'Byte equality binds a declaration; it does not authenticate source or qualify current hardware.' }
        $result.releaseManifest = $release; $result.sourceIdentity = $identity
        if ($null -ne $manifestBytes) {
            $release.actualSha256 = Get-Usb4InspectionHash $manifestBytes
            $release.hashStatus = if ($selection -ne 'InstalledPointer') { 'NotBoundToInstalledPointer' }
            elseif (-not (Test-Usb4InspectionHash $release.recordedSha256)) { 'MissingOrInvalidRecordedHash' }
            elseif ($release.recordedSha256 -ieq $release.actualSha256) { 'MatchedRecordedHash' } else { 'RecordedHashMismatch' }
            $manifest = Read-DeploymentSection 'release manifest JSON' { ConvertFrom-Usb4InspectionJson $manifestBytes }
            if ($manifest) {
                $valid = Read-DeploymentSection 'release declaration' {
                    if (($manifest['schemaVersion'] -isnot [int] -and $manifest['schemaVersion'] -isnot [long]) -or $manifest['schemaVersion'] -ne 1 -or
                        $manifest['artifactType'] -isnot [string] -or $manifest['artifactType'] -cne 'pcai-usb4-fleet-release' -or
                        $manifest['kind'] -isnot [string] -or $manifest['kind'] -cne 'deployable' -or
                        $manifest['source'] -isnot [Collections.IDictionary] -or $manifest['host'] -isnot [Collections.IDictionary] -or
                        $manifest['files'] -isnot [array] -or -not $manifest['files'].Count) { throw 'Invalid deployable declaration shape.' }
                    $source = $manifest['source']; $hostRecord = $manifest['host']
                    if ($source['repository'] -isnot [string] -or $source['repository'] -notmatch '^https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:\.git)?$' -or
                        $source['commit'] -isnot [string] -or $source['commit'] -notmatch '^(?:[a-fA-F0-9]{40}|[a-fA-F0-9]{64})$') { throw 'Invalid scalar source declaration.' }
                    foreach ($field in @('id', 'platform', 'architecture', 'backend')) { if ($hostRecord[$field] -isnot [string] -or -not $hostRecord[$field]) { throw 'Invalid scalar host declaration.' } }
                    Assert-Usb4ManifestRecord $manifest['policy']
                    $paths = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
                    foreach ($entry in $manifest['files']) { Assert-Usb4ManifestRecord $entry; if (-not $paths.Add($entry['path'])) { throw 'Duplicated manifest payload path.' } }
                    $true
                }
                if ($valid) {
                    $source = $manifest['source']; $hostRecord = $manifest['host']; $policyRecord = $manifest['policy']
                    $identity.sourceCommit = $source['commit']; $identity.sourceRepository = $source['repository']; $identity.declaredHost = $hostRecord
                    $identity.checks = @{ schema = $true; sourceDeclaration = $true
                        host = [bool]($installed -and $hostRecord['id'] -ieq $runtime.hostName -and $hostRecord['id'] -ieq $installed['hostName'])
                        platform = $runtime.platform -ceq 'windows' -and $hostRecord['platform'] -ceq 'windows'
                        architecture = $runtime.architecture -cne 'Unknown' -and $hostRecord['architecture'] -ceq $runtime.architecture
                        backend = $hostRecord['backend'] -ceq 'windows-peer' }
                    foreach ($kind in @('controller', 'policy')) {
                        $name = if ($kind -eq 'controller') { 'Tools/SystemScripts/Networking/Invoke-Usb4PeerNetwork.ps1' } else { $policyRecord['path'] }
                        $entries = @($manifest['files'] | Where-Object { $_['path'] -ceq $name })
                        $actual = $fileEvidence[$kind]
                        $match = $entries.Count -eq 1 -and $null -ne $actual -and $entries[0]['sha256'] -ieq $actual.sha256 -and
                            $entries[0]['size'] -eq $actual.size -and $actual.sha256 -ieq $installed["${kind}Sha256"]
                        if ($kind -eq 'policy') { $match = $match -and $policyRecord['sha256'] -ieq $actual.sha256 -and $policyRecord['size'] -eq $actual.size }
                        $identity.checks["${kind}Bytes"] = [bool]$match
                    }
                    if ($selection -eq 'InstalledPointer') { $identity.checks.recordedManifestBytes = $release.hashStatus -ceq 'MatchedRecordedHash' }
                    $identity.byteMatched = @($identity.checks.Values | Where-Object { $_ -ne $true }).Count -eq 0
                    $identity.status = if ($identity.byteMatched) { 'ByteMatchedDeclaration' } else { 'DriftOrUnverified' }
                }
            }
        }
    }
}

if ($runtime.platform -cne 'windows' -and $result.status -cne 'Unknown') { $result.status = 'UnsupportedPlatform' }
$result['readErrors'] = $readErrors.ToArray()
$result | ConvertTo-Json -Depth 20
