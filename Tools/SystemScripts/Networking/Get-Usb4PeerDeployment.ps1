#Requires -Version 7.0
<#
.SYNOPSIS
Read local USB4 deployment identity and ordinary Windows routing without changes.
.DESCRIPTION
Writes JSON to stdout only. Routing queries do not send packets. ProbeInternet
opts into one unbound, certificate-validated TLS HEAD request to 1.1.1.1:443.
An installed task, NCSI indicator or historical controller report is not a fresh
proof of the route ordinary applications use.
#>
[CmdletBinding()]
param(
    [string]$InstallRoot = (Join-Path ([Environment]::GetFolderPath('CommonApplicationData')) 'pcai-usb4/peer-network'),
    [string]$ReleaseManifestPath,
    [switch]$ProbeInternet,
    [Alias('h', 'help')][switch]$ShowHelp
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
if ($ShowHelp) {
    Get-Help $PSCommandPath -Detailed
    return
}
$readErrors = [Collections.Generic.List[object]]::new()
function Read-DeploymentSection {
    param([string]$Section, [scriptblock]$Read)
    try { & $Read } catch { $readErrors.Add(@{ section = $Section; error = $_.Exception.Message }) }
}
function Read-DeploymentJson {
    param([string]$Path)
    if (Test-Path -LiteralPath $Path -PathType Leaf) {
        $value = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json -AsHashtable
        if ($value -isnot [Collections.IDictionary]) { throw 'Expected a JSON object.' }
        $value
    }
}
function Get-NormalizedLocalIP {
    param([Net.IPAddress]$Address)
    if ($Address.IsIPv4MappedToIPv6) { return $Address.MapToIPv4().ToString() }
    $Address.ToString()
}
$installedPath = Join-Path $InstallRoot 'installed.json'
$installed = Read-DeploymentSection 'installation' { Read-DeploymentJson $installedPath }
if ($null -ne $installed) {
    $missing = @('controllerPath', 'policyPath', 'statePath', 'reportPath', 'controllerSha256', 'policySha256', 'hostName', 'taskName', 'taskPath' |
            Where-Object { -not $installed[$_] })
    if ($missing.Count) {
        $readErrors.Add(@{ section = 'installation'; error = 'Missing installation fields: ' + ($missing -join ', ') })
        $installed = $null
    }
}
$result = [ordered]@{
    schema = 'pcai.usb4-deployment/v1'; observedAtUtc = [DateTimeOffset]::UtcNow.ToString('o')
    hostName = [Environment]::MachineName; status = 'NotInstalled'
    storage = @{ installRoot = $InstallRoot; installedPath = $installedPath }
    deployment = $null; sourceIdentity = $null; releaseManifest = $null; task = $null; ownedPending = $null
    controllerReport = $null; enrolledAdapters = @(); ipv4 = $null; ipv6 = $null
    ncsiProfiles = @(); establishedTcpByLocalIP = @(); internetProbe = @{ status = 'NotRun'; packetSent = $false }
}
if ($installed) {
    $result.status = 'Installed'
    $result.storage.controllerPath = $installed['controllerPath']
    $result.storage.policyPath = $installed['policyPath']
    $result.storage.statePath = $installed['statePath']
    $result.storage.reportPath = $installed['reportPath']
    $result.deployment = @{ revision = $installed['revision']; installedAtUtc = $installed['installedAtUtc']; hashes = @() }
    foreach ($kind in 'controller', 'policy') {
        $path = $installed["${kind}Path"]
        $expected = $installed["${kind}Sha256"]
        $actual = Read-DeploymentSection "$kind hash" { (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash }
        $result.deployment.hashes += @{ kind = $kind; expectedSha256 = $expected; actualSha256 = $actual
            matches = $(if ($actual -and $expected) { $actual -eq $expected } else { $null })
        }
    }
    $policy = Read-DeploymentSection 'policy' { Read-DeploymentJson $installed['policyPath'] }
    $state = Read-DeploymentSection 'owned state' { Read-DeploymentJson $installed['statePath'] }
    if ($state) { $result.ownedPending = $state.ContainsKey('pendingTransaction') }
    $report = Read-DeploymentSection 'controller report' { Read-DeploymentJson $installed['reportPath'] }
    if ($report) {
        $age = Read-DeploymentSection 'report timestamp' { ([DateTimeOffset]::UtcNow - [DateTimeOffset]::Parse($report['observedAt'])).TotalSeconds }
        $result.controllerReport = @{ status = $(if ($report['inProgress']) { 'InProgress' } elseif ($report.ContainsKey('inProgress') -or $report['phase'] -eq 'Complete') { 'Completed' } else { 'Unknown' })
            ageSeconds = $age; stale = $(if ($null -ne $age) { $age -gt 180 } else { $null })
            phase = $report['phase']; exitCode = $report['exitCode']; networkReady = $report['networkReady']
            internetReady = $report['internetReady']; qualification = 'HistoricalControllerObservation'
        }
    }
    else { $result.controllerReport = @{ status = 'MissingOrUnreadable' } }
    $result.task = Read-DeploymentSection 'scheduled task' {
        if (-not $installed['taskName'] -or -not $installed['taskPath']) { throw 'Missing scheduled task pointer.' }
        $task = Get-ScheduledTask -TaskName $installed['taskName'] -TaskPath $installed['taskPath']
        $info = $task | Get-ScheduledTaskInfo
        @{ path = $installed['taskPath']; name = $installed['taskName']; state = [string]$task.State
            enabled = $task.Settings.Enabled; principal = $task.Principal.UserId; runLevel = [string]$task.Principal.RunLevel
            lastRunTime = $info.LastRunTime; lastTaskResult = $info.LastTaskResult; nextRunTime = $info.NextRunTime
        }
    }
}
elseif ($readErrors.Count) { $result.status = 'Unknown' }
$pointerRecorded = $installed -and $installed.ContainsKey('releaseManifestPath')
$selection = 'ExplicitOverride'
if (-not $ReleaseManifestPath -and $pointerRecorded) {
    $ReleaseManifestPath = $installed['releaseManifestPath']
    $selection = 'InstalledPointer'
}
elseif (-not $ReleaseManifestPath) {
    $selection = 'LegacyRoot'
    $candidate = Join-Path $InstallRoot 'release.json'
    if (Test-Path -LiteralPath $candidate -PathType Leaf) { $ReleaseManifestPath = $candidate }
}
if ($ReleaseManifestPath -or $pointerRecorded) {
    $result.storage.releaseManifestPath = $ReleaseManifestPath
    $release = @{ selectedBy = $selection; path = $ReleaseManifestPath; actualSha256 = $null
        recordedSha256 = $(if ($installed) { $installed['releaseManifestSha256'] } else { $null }); hashStatus = 'Unreadable'
    }
    $result.releaseManifest = $release
    $manifest = Read-DeploymentSection 'release manifest' {
        if ($selection -eq 'InstalledPointer') {
            if (-not $ReleaseManifestPath -or -not [IO.Path]::IsPathFullyQualified($ReleaseManifestPath)) { throw 'Installed release manifest pointer must be absolute.' }
            $absolute = [IO.Path]::GetFullPath($ReleaseManifestPath)
            $rootPrefix = [IO.Path]::GetFullPath($InstallRoot).TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar
            if (-not $absolute.StartsWith($rootPrefix, [StringComparison]::OrdinalIgnoreCase)) { throw 'Installed release manifest pointer is outside the installation root.' }
            $cursor = $absolute
            while ($cursor) {
                if ((Test-Path -LiteralPath $cursor) -and ((Get-Item -LiteralPath $cursor -Force).Attributes -band [IO.FileAttributes]::ReparsePoint)) { throw 'Installed release manifest pointer traverses a reparse point.' }
                $cursor = [IO.Path]::GetDirectoryName($cursor)
            }
        }
        if (-not (Test-Path -LiteralPath $ReleaseManifestPath -PathType Leaf)) { throw 'Specified release manifest is missing.' }
        $bytes = [IO.File]::ReadAllBytes($ReleaseManifestPath)
        $hasher = [Security.Cryptography.SHA256]::Create()
        try { $release.actualSha256 = [BitConverter]::ToString($hasher.ComputeHash($bytes)).Replace('-', '') } finally { $hasher.Dispose() }
        $release.hashStatus = if ($selection -eq 'InstalledPointer') {
            if ($release.recordedSha256 -notmatch '^[a-f0-9]{64}$') { 'MissingOrInvalidRecordedHash' }
            elseif ($release.recordedSha256 -eq $release.actualSha256) { 'MatchedRecordedHash' } else { 'RecordedHashMismatch' }
        }
        else { 'NotBoundToInstalledPointer' }
        $parsed = [Text.UTF8Encoding]::new($false, $true).GetString($bytes).TrimStart([char]0xfeff) | ConvertFrom-Json -AsHashtable
        if ($parsed -isnot [Collections.IDictionary]) { throw 'Expected a release manifest JSON object.' }
        $parsed
    }
    if ($manifest) {
        $identity = @{ status = 'DeclaredUnverified'; byteMatched = $false; authenticity = 'NotVerified'; checks = @{} }
        if ($manifest['artifactType'] -eq 'pcai-usb4-fleet-release') {
            $declaredSource = $manifest['source']
            $hostRecord = $manifest['host']
            $policyRecord = $manifest['policy']
            if ($declaredSource -isnot [Collections.IDictionary] -or $hostRecord -isnot [Collections.IDictionary] -or
                $policyRecord -isnot [Collections.IDictionary]) {
                $identity.status = 'DriftOrUnverified'
                $identity.checks['manifestShape'] = $false
            }
            else {
                $identity.sourceCommit = $declaredSource['commit']
                $identity.sourceRepository = $declaredSource['repository']
                $identity.declaredHost = $hostRecord
                $identity.registryPath = $declaredSource['registryPath']
                $identity.registrySha256 = $declaredSource['registrySha256']
                $architecture = switch ([Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString()) {
                    'X64' { 'x86_64' } 'Arm64' { 'aarch64' } 'X86' { 'x86' } default { 'Unknown' }
                }
                $identity.checks = @{
                    schema            = $manifest['schemaVersion'] -eq 1
                    sourceDeclaration = [bool]($identity.sourceCommit -match '^(?:[a-f0-9]{40}|[a-f0-9]{64})$' -and $identity.sourceRepository)
                    host              = [bool]($installed -and $hostRecord['id'] -eq [Environment]::MachineName -and
                        $hostRecord['id'] -eq $installed['hostName'])
                    platform          = $hostRecord['platform'] -eq 'windows' -and [Runtime.InteropServices.RuntimeInformation]::IsOSPlatform([Runtime.InteropServices.OSPlatform]::Windows)
                    architecture      = $architecture -ne 'Unknown' -and $hostRecord['architecture'] -eq $architecture
                    backend           = $hostRecord['backend'] -eq 'windows-peer' -and $manifest['kind'] -eq 'deployable'
                }
                if ($selection -eq 'InstalledPointer') { $identity.checks['recordedManifestBytes'] = $release.hashStatus -eq 'MatchedRecordedHash' }
                $files = @($manifest['files'])
                $controllerEntries = @($files | Where-Object { $_ -is [Collections.IDictionary] -and
                        $_['path'] -eq 'Tools/SystemScripts/Networking/Invoke-Usb4PeerNetwork.ps1' })
                $policyEntries = @($files | Where-Object { $_ -is [Collections.IDictionary] -and
                        $_['path'] -and $_['path'] -eq $policyRecord['path'] })
                foreach ($kind in 'controller', 'policy') {
                    $entries = @(if ($kind -eq 'controller') { $controllerEntries } else { $policyEntries })
                    $actualHashes = @(if ($result.deployment) { $result.deployment.hashes | Where-Object kind -EQ $kind })
                    $matches = $entries.Count -eq 1 -and $actualHashes.Count -eq 1
                    if ($matches) {
                        $matches = $entries[0]['sha256'] -match '^[a-f0-9]{64}$' -and $actualHashes[0].matches -eq $true -and
                        $entries[0]['sha256'] -eq $actualHashes[0].actualSha256
                        if ($kind -eq 'policy') { $matches = $matches -and $policyRecord['sha256'] -eq $entries[0]['sha256'] }
                    }
                    $identity.checks["${kind}Bytes"] = [bool]$matches
                }
                $identity.byteMatched = @($identity.checks.Values | Where-Object { $_ -ne $true }).Count -eq 0
                $identity.status = if ($identity.byteMatched) { 'ByteMatchedDeclaration' } else { 'DriftOrUnverified' }
            }
        }
        else {
            foreach ($field in 'sourceCommit', 'sourceRepository', 'gitCommit', 'gitBranch', 'sourcePath', 'releaseId') {
                if ($manifest.ContainsKey($field)) { $identity[$field] = $manifest[$field] }
            }
        }
        $identity.note = 'Source metadata is a declaration; byte matching does not authenticate a Git commit or manifest signature.'
        if ($selection -eq 'InstalledPointer' -and $release.hashStatus -ne 'MatchedRecordedHash') {
            $identity.status = 'DriftOrUnverified'; $identity.byteMatched = $false
        }
        $result.sourceIdentity = $identity
    }
    else {
        $result.sourceIdentity = @{ status = 'DriftOrUnverified'; byteMatched = $false; authenticity = 'NotVerified' }
    }
}
$adapters = @(Read-DeploymentSection 'adapters' { Get-NetAdapter -IncludeHidden })
if ($installed -and $policy -and $policy['adapterMac']) {
    $mac = $policy['adapterMac'] -replace '[:-]', ''
    $result.enrolledAdapters = @($adapters | Where-Object { ($_.MacAddress -replace '[:-]', '') -eq $mac } |
            Select-Object Name, InterfaceGuid, ifIndex, MacAddress, Status, LinkSpeed)
}
$ipInterfaces = @(Read-DeploymentSection 'IP interfaces' { Get-NetIPInterface })
$addresses = @(Read-DeploymentSection 'IP addresses' { Get-NetIPAddress })
$routes = @(Read-DeploymentSection 'routes' { Get-NetRoute })
$defaults = @($routes | Where-Object { $_.DestinationPrefix -in '0.0.0.0/0', '::/0' } | ForEach-Object {
        $route = $_
        $ipif = @($ipInterfaces | Where-Object { $_.InterfaceIndex -eq $route.InterfaceIndex -and $_.AddressFamily -eq $route.AddressFamily })
        $metric = if ($ipif.Count -eq 1) { [int]$ipif[0].InterfaceMetric } else { $null }
        @{ destination = $route.DestinationPrefix; interfaceIndex = $route.InterfaceIndex; interfaceAlias = $route.InterfaceAlias
            nextHop = $route.NextHop; routeMetric = $route.RouteMetric; interfaceMetric = $metric
            totalMetric     = $(if ($null -ne $metric) { [int]$route.RouteMetric + $metric } else { $null })
            connectionState = $(if ($ipif.Count -eq 1) { [string]$ipif[0].ConnectionState } else { 'Unknown' })
        }
    })
$choice = @(Read-DeploymentSection 'ordinary IPv4 route' { Find-NetRoute -RemoteIPAddress '8.8.8.8' })
$source = @($choice | Where-Object { $_.PSObject.Properties['IPAddress'] })
$selectedRoute = @($choice | Where-Object { $_.PSObject.Properties['DestinationPrefix'] })
$selectedAdapter = @($adapters | Where-Object { $source.Count -eq 1 -and $_.ifIndex -eq $source[0].InterfaceIndex })
$result.ipv4 = @{ defaultRoutes = @($defaults | Where-Object destination -EQ '0.0.0.0/0' | Sort-Object totalMetric)
    ordinaryRoute               = $(if ($source.Count -eq 1 -and $selectedRoute.Count -eq 1) {
            @{ queryDestination = '8.8.8.8'; sourceIP = $source[0].IPAddress; interfaceIndex = $source[0].InterfaceIndex
                interfaceAlias = $source[0].InterfaceAlias; nextHop = $selectedRoute[0].NextHop
                destinationPrefix = $selectedRoute[0].DestinationPrefix; proof = 'RoutingQueryOnlyNoPacket'
                usesWireless = $(if ($selectedAdapter.Count -eq 1) { [string]$selectedAdapter[0].PhysicalMediaType -match '802.11|Wireless' } else { $null })
            }
        }
        else { $null })
    indicatorNote               = 'NCSI and established sockets are separate observations; neither selects the ordinary route.'
}
$v6Defaults = @($defaults | Where-Object destination -EQ '::/0' | Sort-Object totalMetric)
$v6Addresses = @($addresses | Where-Object AddressFamily -EQ IPv6 | Select-Object InterfaceIndex, InterfaceAlias, IPAddress, AddressState)
$result.ipv6 = @{ defaultRoutes = $v6Defaults; addresses = $v6Addresses
    globalAddresses = @($v6Addresses | Where-Object IPAddress -Match '^[23][0-9a-f]{3}:')
    ulaAddresses    = @($v6Addresses | Where-Object IPAddress -Match '^f[cd][0-9a-f]{2}:')
    wan             = $(if (@($readErrors | Where-Object { $_.section -eq 'routes' }).Count) { 'Unknown' } elseif (-not $v6Defaults.Count) { 'NoIPv6WANDefaultRoute' } else { 'DefaultPresentUnverified' })
}
$result.ncsiProfiles = @(Read-DeploymentSection 'NCSI profiles' {
        Get-NetConnectionProfile | Select-Object InterfaceIndex, InterfaceAlias, NetworkCategory, IPv4Connectivity, IPv6Connectivity
    })
$result.establishedTcpByLocalIP = @(Read-DeploymentSection 'established TCP' {
        Get-NetTCPConnection | Where-Object State -EQ Established | Group-Object LocalAddress |
            ForEach-Object { @{ localIP = $_.Name; count = $_.Count } }
    })
if ($ProbeInternet) {
    $probe = @{ status = 'Failed'; networkAttempted = $true; packetSent = $null; target = '1.1.1.1:443'; bound = $false; tlsAuthenticated = $false; sourceIP = $null; httpStatus = $null }
    $tcp = [Net.Sockets.TcpClient]::new()
    $tls = $null
    try {
        $null = $tcp.ConnectAsync('1.1.1.1', 443).WaitAsync([TimeSpan]::FromSeconds(10)).GetAwaiter().GetResult()
        $probe.packetSent = $true
        $probe.sourceIP = Get-NormalizedLocalIP $tcp.Client.LocalEndPoint.Address
        $tls = [Net.Security.SslStream]::new($tcp.GetStream(), $false)
        $options = [Net.Security.SslClientAuthenticationOptions]::new()
        $options.TargetHost = '1.1.1.1'
        $null = $tls.AuthenticateAsClientAsync($options).WaitAsync([TimeSpan]::FromSeconds(10)).GetAwaiter().GetResult()
        $probe.tlsAuthenticated = $tls.IsAuthenticated -and $tls.IsEncrypted
        $tls.ReadTimeout = 5000; $tls.WriteTimeout = 5000
        $request = [Text.Encoding]::ASCII.GetBytes("HEAD / HTTP/1.1`r`nHost: 1.1.1.1`r`nConnection: close`r`n`r`n")
        $tls.Write($request, 0, $request.Length)
        $buffer = [byte[]]::new(4096)
        $response = [Text.Encoding]::ASCII.GetString($buffer, 0, $tls.Read($buffer, 0, $buffer.Length))
        if ($response -notmatch '^HTTP/1\.[01] ([1-5][0-9]{2})\b') { throw 'TLS peer returned no valid HTTP status line.' }
        $probe.httpStatus = [int]$Matches[1]
        $probe.status = 'AuthenticatedHttpResponse'
    }
    catch { $probe.error = $_.Exception.Message } finally {
        if ($tls) { $tls.Dispose() }
        $tcp.Dispose()
    }
    $result.internetProbe = $probe
}
$result['readErrors'] = $readErrors.ToArray()
$result | ConvertTo-Json -Depth 12
