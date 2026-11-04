#Requires -Version 7.0
<#
.SYNOPSIS
Mount an enrolled SFTP peer only through its owned USB4 source and pinned key.
.DESCRIPTION
Refuses a missing/down adapter, duplicate or foreign source address and default
route fallback. Rclone binds every connection to the verified local source and
checks an explicitly supplied known_hosts file. Startup connection/retries are
bounded; the foreground mount itself remains alive until stopped. Task Scheduler
can retry after unavailable hardware returns. See docs/usb4/windows-boot.md.
.PARAMETER DryRun
Inspect local path and emit the mount decision without mounting or writing files.
.EXAMPLE
./Invoke-ThunderboltRcloneMount.ps1 -PeerAddress 172.31.252.133 -SourceAddress 172.31.252.134 -KnownHostsPath ~/.machine/usb4-mount-known-hosts -DryRun
#>
[CmdletBinding()]
param(
    [string]$PeerAddress = '172.31.252.133',
    [string]$SourceAddress = '172.31.252.134',
    [ValidatePattern('^[A-Za-z0-9_-]+$')][string]$RemoteName = 'dtm-p1gen7-tb',
    [ValidatePattern('^/[A-Za-z]:/$')][string]$RemotePath = '/D:/',
    [ValidatePattern('^[A-Za-z]:$')][string]$MountPoint = 'D:',
    [string]$KnownHostsPath = (Join-Path $HOME '.machine/usb4-mount-known-hosts'),
    [string]$ConfigPath = (Join-Path $HOME 'AppData/Roaming/rclone/rclone.conf'),
    [string]$RcloneExecutable = (Join-Path $HOME 'bin/rclone.exe'),
    [string]$CachePath = (Join-Path $HOME '.cache/rclone/p1gen7-d'),
    [string]$LogPath = (Join-Path $HOME '.cache/rclone/logs/p1gen7-d-mount.log'),
    [switch]$DryRun,
    [Alias('h', 'help')][switch]$ShowHelp
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
function Test-ThunderboltMountPath {
    <# .SYNOPSIS
    Require an Up USB4 adapter, one owned preferred source and only on-link routes.
    #>
    [CmdletBinding()]
    param($Adapter, [object[]]$Addresses, [object[]]$Routes, [string]$SourceAddress)
    # Status is an optional formatting/script property and is null in some
    # profile-free CIM projections. Raw operational/admin enums carry the state.
    if (-not $Adapter -or $Adapter.InterfaceOperationalStatus -ne 1 -or $Adapter.AdminStatus -ne 1 -or
        $Adapter.InterfaceDescription -notmatch '(?i)USB4.*P2P|Thunderbolt.*Network') { return $false }
    $sources = @($Addresses | Where-Object IPAddress -EQ $SourceAddress)
    if ($sources.Count -ne 1 -or $sources[0].InterfaceIndex -ne $Adapter.ifIndex -or
        $sources[0].AddressState -ne 'Preferred') { return $false }
    if (@($Routes).Count -eq 0) { return $false }
    foreach ($route in $Routes) {
        if ($route.InterfaceIndex -ne $Adapter.ifIndex -or $route.NextHop -ne '0.0.0.0' -or
            $route.DestinationPrefix -eq '0.0.0.0/0') { return $false }
    }
    return $true
}
function Get-ThunderboltMountTransportArguments {
    <# .SYNOPSIS
    Force the pinned native SFTP transport regardless of backend/environment drift.
    #>
    [CmdletBinding()]
    param([string]$PeerAddress, [string]$SourceAddress, [string]$KnownHostsPath)
    # ProcessStartInfo.ArgumentList retains the explicit empty --sftp-ssh value.
    # External SSH would otherwise bypass rclone's native bind/key validation.
    @('--sftp-host', $PeerAddress, '--sftp-known-hosts-file', $KnownHostsPath,
        '--sftp-ssh', '', '--sftp-pin-host-key=false', '--bind', $SourceAddress,
        '--contimeout', '5s', '--timeout', '30s', '--retries', '1', '--low-level-retries', '1')
}
function Test-ThunderboltKnownHostContent {
    <# .SYNOPSIS
    Reject malformed managed public host-pin records before reporting readiness.
    .DESCRIPTION
    Checks record and plain public-key wire framing, not peer identity or trust.
    Rclone's parser and authenticated host-key callback remain authoritative.
    #>
    [CmdletBinding()]
    param([AllowNull()][AllowEmptyString()][string]$Content)
    if ([string]::IsNullOrWhiteSpace($Content)) { return $false }
    $usableRecords = 0
    foreach ($rawLine in ($Content -split '\r?\n')) {
        $line = $rawLine.Trim()
        if (-not $line -or $line.StartsWith('#')) { continue }
        $fields = $line -split '\s+'
        $offset = 0
        if ($fields[0].StartsWith('@')) {
            if ($fields[0] -notin @('@cert-authority', '@revoked')) { return $false }
            $offset = 1
        }
        if ($fields.Count -lt ($offset + 3)) { return $false }
        if (@($fields[$offset] -split ',' | Where-Object { [string]::IsNullOrWhiteSpace($_) }).Count -gt 0) {
            return $false
        }
        if ($fields[$offset].StartsWith('|')) {
            $hashedHost = $fields[$offset] -split '\|'
            if ($hashedHost.Count -ne 4 -or $hashedHost[1] -ne '1') { return $false }
            try {
                if ([Convert]::FromBase64String($hashedHost[2]).Length -ne 20 -or
                    [Convert]::FromBase64String($hashedHost[3]).Length -ne 20) { return $false }
            } catch { return $false }
        }
        $keyType = $fields[$offset + 1]
        $expectedParts = switch -Regex ($keyType) {
            '^ssh-ed25519$' { 2; break }
            '^ssh-rsa$' { 3; break }
            '^ssh-dss$' { 5; break }
            '^ecdsa-sha2-nistp(256|384|521)$' { 3; break }
            '^sk-ssh-ed25519@openssh.com$' { 3; break }
            '^sk-ecdsa-sha2-nistp256@openssh.com$' { 4; break }
            default { return $false }
        }
        try { $blob = [Convert]::FromBase64String($fields[$offset + 2]) } catch { return $false }
        $position = 0
        $partLengths = [Collections.Generic.List[int]]::new()
        while ($position -lt $blob.Length) {
            if (($blob.Length - $position) -lt 4) { return $false }
            $length = [long]$blob[$position] * 16777216 + [long]$blob[$position + 1] * 65536 +
                [long]$blob[$position + 2] * 256 + [long]$blob[$position + 3]
            $position += 4
            if ($length -le 0 -or $length -gt ($blob.Length - $position)) { return $false }
            if ($partLengths.Count -eq 0 -and
                [Text.Encoding]::ASCII.GetString($blob, $position, [int]$length) -cne $keyType) {
                return $false
            }
            $partLengths.Add([int]$length)
            $position += [int]$length
        }
        if ($partLengths.Count -ne $expectedParts) { return $false }
        if ($keyType -in @('ssh-ed25519', 'sk-ssh-ed25519@openssh.com') -and $partLengths[1] -ne 32) {
            return $false
        }
        if ($offset -eq 0 -or $fields[0] -ne '@revoked') { $usableRecords++ }
    }
    return $usableRecords -gt 0
}
if ($MyInvocation.InvocationName -eq '.') { return }
if ($ShowHelp) { Get-Help $PSCommandPath -Detailed; return }
try {
    if (-not $IsWindows) { throw 'USB4 mount gate supports Windows only.' }
    foreach ($address in @($PeerAddress, $SourceAddress)) {
        $parsed = [Net.IPAddress]::Parse($address)
        if ($parsed.AddressFamily -ne [Net.Sockets.AddressFamily]::InterNetwork) { throw 'Mount gate requires explicit IPv4 addresses.' }
    }
    Import-Module (Join-Path $env:SystemRoot 'System32/WindowsPowerShell/v1.0/Modules/NetAdapter/NetAdapter.psd1')
    Import-Module (Join-Path $env:SystemRoot 'System32/WindowsPowerShell/v1.0/Modules/NetTCPIP/NetTCPIP.psd1')
    $addresses = @(Get-NetIPAddress -AddressFamily IPv4)
    $source = @($addresses | Where-Object IPAddress -EQ $SourceAddress)
    $adapter = if ($source.Count -eq 1) { Get-NetAdapter -InterfaceIndex $source[0].InterfaceIndex -IncludeHidden } else { $null }
    $routes = if ($source.Count -eq 1) {
        @(Find-NetRoute -RemoteIPAddress $PeerAddress -LocalIPAddress $SourceAddress |
            Where-Object { $_.PSObject.Properties['DestinationPrefix'] -and $_.DestinationPrefix })
    } else { @() }
    if (-not (Test-ThunderboltMountPath $adapter $addresses $routes $SourceAddress)) {
        [pscustomobject]@{ State = 'Unavailable'; Reason = 'No owned operational USB4 source/on-link route; mount skipped.' } | ConvertTo-Json
        exit 2
    }
    if (-not [IO.File]::Exists($KnownHostsPath) -or
        -not (Test-ThunderboltKnownHostContent -Content ([IO.File]::ReadAllText($KnownHostsPath)))) {
        throw 'Pinned known_hosts file is missing or has malformed/unsupported public-key records; mount refused.'
    }
    if ($DryRun) {
        [pscustomobject]@{ State = 'PathEligible'; PeerAddress = $PeerAddress; SourceAddress = $SourceAddress; Mounted = $false } | ConvertTo-Json
        exit 0
    }
    $arguments = @('mount', "${RemoteName}:$RemotePath", $MountPoint, '--config', $ConfigPath) +
        @(Get-ThunderboltMountTransportArguments $PeerAddress $SourceAddress $KnownHostsPath) + @(
        '--vfs-cache-mode', 'writes', '--cache-dir', $CachePath, '--vfs-cache-max-size', '20G',
        '--vfs-cache-min-free-space', '50G', '--buffer-size', '16M', '--dir-cache-time', '1m',
        '--volname', 'p1gen7-d', '--log-level', 'INFO', '--log-file', $LogPath,
        '-o', 'FileSecurity=D:P(A;;FA;;;WD)', '--no-console')
    $start = [Diagnostics.ProcessStartInfo]::new($RcloneExecutable)
    $start.UseShellExecute = $false; $start.CreateNoWindow = $true
    foreach ($argument in $arguments) { $start.ArgumentList.Add($argument) }
    $process = [Diagnostics.Process]::new(); $process.StartInfo = $start
    try {
        $null = $process.Start()
        $process.WaitForExit()
        $code = $process.ExitCode
    } finally { $process.Dispose() }
    exit $code
} catch {
    [pscustomobject]@{ State = 'Failed'; Reason = $_.Exception.Message } | ConvertTo-Json
    exit 1
}
