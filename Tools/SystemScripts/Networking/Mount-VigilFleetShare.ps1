#Requires -Version 7.0
<#
.SYNOPSIS
Inspect or explicitly mount the public ASUS fleet NFS share on a reachable LAN.
.DESCRIPTION
Defaults to read-only status. Mount and Unmount require -Apply and honor WhatIf.
DryRun and help do not invoke native programs or write files. Only the public
/srv/vigil-share export and the two approved Windows source addresses are used.
Existing foreign drive mappings are preserved. Native calls have a deadline.
.EXAMPLE
./Mount-VigilFleetShare.ps1 -Action Mount -DryRun
.EXAMPLE
./Mount-VigilFleetShare.ps1 -Action Mount -Apply
.EXAMPLE
./Mount-VigilFleetShare.ps1 -Action Unmount -Apply
#>
[CmdletBinding(SupportsShouldProcess, PositionalBinding = $false)]
param(
    [ValidateSet('Status', 'Mount', 'Unmount')][string]$Action = 'Status',
    [ValidatePattern('^[D-Zd-z]:?$')][string]$Drive = 'N:',
    [switch]$Apply,
    [switch]$DryRun,
    [ValidateRange(100, 3000)][int]$ProbeTimeoutMilliseconds = 750,
    [ValidateRange(1, 30)][int]$NativeTimeoutSeconds = 8,
    [switch]$ShowTaskTemplate,
    [Alias('h', 'help')][switch]$ShowHelp,
    [Parameter(ValueFromRemainingArguments)][string[]]$RemainingArguments = @()
)

Set-StrictMode -Version Latest

function Invoke-VigilNfsNative {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$FilePath,
        [string[]]$Arguments = @(),
        [ValidateRange(1, 30)][int]$TimeoutSeconds = 8
    )
    Initialize-VigilNfsProcessRunner
    [Vigil.BoundedProcess]::Run($FilePath, $Arguments, $TimeoutSeconds, $null, $null)
}

function Initialize-VigilNfsProcessRunner {
    [CmdletBinding()]
    param()
    . (Join-Path $PSScriptRoot 'Invoke-VigilBoundedProcess.ps1')
    Initialize-VigilBoundedProcessRunner
}
function Test-VigilNfsPort {
    [CmdletBinding()]
    [OutputType([bool])]
    param(
        [Parameter(Mandatory)][string]$Server,
        [Parameter(Mandatory)][string]$SourceAddress,
        [ValidateRange(100, 3000)][int]$TimeoutMilliseconds = 750
    )
    $client = [Net.Sockets.TcpClient]::new([Net.Sockets.AddressFamily]::InterNetwork)
    try {
        $client.Client.Bind([Net.IPEndPoint]::new([Net.IPAddress]::Parse($SourceAddress), 0))
        $connection = $client.ConnectAsync([Net.IPAddress]::Parse($Server), 2049)
        if (-not $connection.Wait($TimeoutMilliseconds)) { return $false }
        return $client.Connected
    }
    catch [Net.Sockets.SocketException] { return $false }
    catch [AggregateException] {
        if ($_.Exception.GetBaseException() -is [Net.Sockets.SocketException]) { return $false }
        throw
    }
    finally { $client.Dispose() }
}

function Select-VigilNfsEndpoint {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][AllowEmptyCollection()][object[]]$Addresses,
        [Parameter(Mandatory)][AllowEmptyCollection()][object[]]$Adapters,
        [ValidateRange(100, 3000)][int]$TimeoutMilliseconds = 750,
        [ValidateSet('192.168.50.2', '10.60.4.1')][string]$ServerHint
    )
    foreach ($pair in @(
            @{ Server = '192.168.50.2'; Source = '192.168.50.42'; Prefix = 24 },
            @{ Server = '10.60.4.1'; Source = '10.60.4.4'; Prefix = 29 }
        )) {
        if ($ServerHint -and $pair.Server -ne $ServerHint) { continue }
        foreach ($address in $Addresses) {
            if ($address.IPAddress -ne $pair.Source -or $address.PrefixLength -ne $pair.Prefix -or
                $address.AddressState -ne 'Preferred') { continue }
            $up = @($Adapters | Where-Object { $_.ifIndex -eq $address.InterfaceIndex -and $_.Status -eq 'Up' })
            if ($up.Count -ne 1) { continue }
            if (Test-VigilNfsPort -Server $pair.Server -SourceAddress $pair.Source -TimeoutMilliseconds $TimeoutMilliseconds) {
                return [pscustomobject]@{
                    Server         = $pair.Server
                    SourceAddress  = $pair.Source
                    InterfaceIndex = $address.InterfaceIndex
                    Remote         = "$($pair.Server):/srv/vigil-share"
                }
            }
        }
    }
    return $null
}

function ConvertFrom-VigilNfsMount {
    [CmdletBinding()]
    param([AllowEmptyString()][string]$Text)
    foreach ($line in $Text -split '\r?\n') {
        if ($line -match '^\s*(?<drive>[A-Za-z]:)\s+(?<remote>\\\\\S+|[0-9.]+:/\S+)') {
            [pscustomobject]@{ Drive = $Matches.drive.ToUpperInvariant(); Remote = $Matches.remote }
        }
    }
}

function Test-VigilOwnedNfsMount {
    [CmdletBinding()]
    param([Parameter(Mandatory)][string]$Remote)
    return $Remote -cin @(
        '192.168.50.2:/srv/vigil-share', '10.60.4.1:/srv/vigil-share',
        '\\192.168.50.2\srv\vigil-share', '\\10.60.4.1\srv\vigil-share'
    )
}

function Get-VigilDriveLetter {
    [CmdletBinding()]
    param()
    @([Environment]::GetLogicalDrives() | ForEach-Object { $_.Substring(0, 2).ToUpperInvariant() }) +
    @(Get-PSDrive -PSProvider FileSystem | Where-Object { $_.Name -match '^[A-Za-z]$' } |
            ForEach-Object { "$($_.Name.ToUpperInvariant()):" })
}

function Get-VigilNfsTaskTemplate {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][ValidateScript({ Test-Path -LiteralPath $_ -PathType Leaf })][string]$ScriptPath,
        [ValidatePattern('^S-1-\d+(?:-\d+)+$')][string]$UserId = ([Security.Principal.WindowsIdentity]::GetCurrent().User.Value),
        [string]$PowerShellPath = (Get-Command pwsh -CommandType Application -ErrorAction Stop | Select-Object -First 1).Source
    )
    $resolved = (Resolve-Path -LiteralPath $ScriptPath).Path
    if ($resolved.Contains('"')) { throw 'Task script path cannot contain a double quote.' }
    $escape = { param([string]$Value) [Security.SecurityElement]::Escape($Value) }
    $commandXml = & $escape $PowerShellPath
    $argumentsXml = & $escape "-NoProfile -NonInteractive -WindowStyle Hidden -File `"$resolved`" -Action Mount -Apply"
    $subscriptionXml = & $escape '<QueryList><Query Id="0" Path="Microsoft-Windows-NetworkProfile/Operational"><Select Path="Microsoft-Windows-NetworkProfile/Operational">*[System[(EventID=10000)]]</Select></Query></QueryList>'
    $xml = @"
<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.4" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo><Description>Mount public ASUS NFS share only when its approved LAN is reachable.</Description></RegistrationInfo>
  <Triggers>
    <LogonTrigger><Enabled>true</Enabled><UserId>$UserId</UserId><Delay>PT15S</Delay></LogonTrigger>
    <EventTrigger><Enabled>true</Enabled><Subscription>$subscriptionXml</Subscription><Delay>PT10S</Delay></EventTrigger>
  </Triggers>
  <Principals><Principal id="User"><UserId>$UserId</UserId><LogonType>InteractiveToken</LogonType><RunLevel>LeastPrivilege</RunLevel></Principal></Principals>
  <Settings><MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy><DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries><StopIfGoingOnBatteries>false</StopIfGoingOnBatteries><StartWhenAvailable>true</StartWhenAvailable><ExecutionTimeLimit>PT1M</ExecutionTimeLimit><Enabled>true</Enabled></Settings>
  <Actions Context="User"><Exec><Command>$commandXml</Command><Arguments>$argumentsXml</Arguments></Exec></Actions>
</Task>
"@
    [pscustomobject]@{ TaskName = 'VIGIL-Public-ASUS-NFS'; UserId = $UserId; Xml = $xml; Registered = $false }
}

function Invoke-VigilFleetShare {
    [CmdletBinding(SupportsShouldProcess)]
    param(
        [ValidateSet('Status', 'Mount', 'Unmount')][string]$Action = 'Status',
        [ValidatePattern('^[D-Zd-z]:?$')][string]$Drive = 'N:',
        [switch]$Apply,
        [switch]$DryRun,
        [ValidateRange(100, 3000)][int]$ProbeTimeoutMilliseconds = 750,
        [ValidateRange(1, 30)][int]$NativeTimeoutSeconds = 8
    )
    $driveLetter = $Drive.TrimEnd(':').ToUpperInvariant() + ':'
    $result = [ordered]@{ Action = $Action; Drive = $driveLetter; State = 'Plan'; Endpoint = $null; Native = $null }
    if ($DryRun) { return [pscustomobject]$result }
    if (-not $IsWindows) { throw 'This helper requires Windows Client for NFS.' }
    $mountPath = Join-Path $env:WINDIR 'System32/mount.exe'
    $unmountPath = Join-Path $env:WINDIR 'System32/umount.exe'
    if (-not (Test-Path -LiteralPath $mountPath -PathType Leaf)) { throw 'Windows Client for NFS mount.exe is unavailable.' }
    $inventory = Invoke-VigilNfsNative -FilePath $mountPath -TimeoutSeconds $NativeTimeoutSeconds
    if ($inventory.ExitCode -ne 0) { throw "NFS inventory failed: $($inventory.Stderr) $($inventory.Stdout)" }
    $mappings = @(ConvertFrom-VigilNfsMount -Text $inventory.Stdout | Where-Object { $_.Drive -eq $driveLetter })
    if ($mappings.Count -gt 1) { throw "Ambiguous NFS mappings for $driveLetter; preserved." }
    if ($mappings.Count -eq 1) {
        if (-not (Test-VigilOwnedNfsMount -Remote $mappings[0].Remote)) {
            $result.State = 'ForeignDrivePreserved'
            return [pscustomobject]$result
        }
        $result.Endpoint = $mappings[0].Remote
        $result.State = 'AlreadyMounted'
        if ($Action -ne 'Unmount') {
            $existingServer = if ($mappings[0].Remote.Contains('192.168.50.2')) { '192.168.50.2' } else { '10.60.4.1' }
            $reachable = Select-VigilNfsEndpoint -Addresses @(Get-NetIPAddress -AddressFamily IPv4) -Adapters @(Get-NetAdapter -IncludeHidden) -TimeoutMilliseconds $ProbeTimeoutMilliseconds -ServerHint $existingServer
            if ($null -eq $reachable) { $result.State = 'ExistingEndpointUnavailable' }
            return [pscustomobject]$result
        }
        if ($Apply -and $PSCmdlet.ShouldProcess($driveLetter, 'Unmount public ASUS NFS share')) {
            $current = Invoke-VigilNfsNative -FilePath $mountPath -TimeoutSeconds $NativeTimeoutSeconds
            if ($current.ExitCode -ne 0) { throw 'Cannot verify NFS custody before unmount; preserved.' }
            $currentMappings = @(ConvertFrom-VigilNfsMount -Text $current.Stdout | Where-Object { $_.Drive -eq $driveLetter })
            if ($currentMappings.Count -ne 1 -or $currentMappings[0].Remote -cne $mappings[0].Remote) {
                throw 'NFS drive identity changed before unmount; preserved.'
            }
            $native = Invoke-VigilNfsNative -FilePath $unmountPath -Arguments @($driveLetter) -TimeoutSeconds $NativeTimeoutSeconds
            if ($native.ExitCode -ne 0) { throw "NFS unmount failed (exit $($native.ExitCode)): $($native.Stderr) $($native.Stdout)" }
            $verified = Invoke-VigilNfsNative -FilePath $mountPath -TimeoutSeconds $NativeTimeoutSeconds
            if ($verified.ExitCode -ne 0) { throw 'NFS unmount returned success but verification inventory failed.' }
            if (@(ConvertFrom-VigilNfsMount -Text $verified.Stdout | Where-Object { $_.Drive -eq $driveLetter }).Count -gt 0) {
                throw 'NFS unmount returned success but a mapping is still observed on the drive.'
            }
            $result.Native = $native
            $result.State = 'Unmounted'
        }
        else { $result.State = 'WouldUnmount' }
        return [pscustomobject]$result
    }
    if ($driveLetter -in @(Get-VigilDriveLetter)) {
        $result.State = 'ForeignDrivePreserved'
        return [pscustomobject]$result
    }
    if ($Action -eq 'Unmount') {
        $result.State = 'AlreadyUnmounted'
        return [pscustomobject]$result
    }
    $endpoint = Select-VigilNfsEndpoint -Addresses @(Get-NetIPAddress -AddressFamily IPv4) -Adapters @(Get-NetAdapter -IncludeHidden) -TimeoutMilliseconds $ProbeTimeoutMilliseconds
    $result.Endpoint = $endpoint
    if ($null -eq $endpoint) {
        $result.State = 'LanUnavailable'
        return [pscustomobject]$result
    }
    $result.State = 'Available'
    if ($Action -eq 'Mount') {
        if ($Apply -and $PSCmdlet.ShouldProcess($driveLetter, "Mount $($endpoint.Remote)")) {
            # Recheck the drive immediately before changing state; never rebind it.
            if ($driveLetter -in @(Get-VigilDriveLetter)) { throw "$driveLetter became occupied; preserved." }
            $native = Invoke-VigilNfsNative -FilePath $mountPath -Arguments @(
                '-o', 'anon', 'mtype=soft', 'timeout=1', 'retry=1', $endpoint.Remote, $driveLetter
            ) -TimeoutSeconds $NativeTimeoutSeconds
            if ($native.ExitCode -ne 0) { throw "NFS mount failed (exit $($native.ExitCode)): $($native.Stderr) $($native.Stdout)" }
            $verified = Invoke-VigilNfsNative -FilePath $mountPath -TimeoutSeconds $NativeTimeoutSeconds
            if ($verified.ExitCode -ne 0) { throw 'NFS mount returned success but verification inventory failed.' }
            $match = @(ConvertFrom-VigilNfsMount -Text $verified.Stdout | Where-Object {
                    $_.Drive -eq $driveLetter -and (Test-VigilOwnedNfsMount -Remote $_.Remote)
                })
            if ($match.Count -ne 1) { throw 'NFS mount returned success but the expected mapping was not observed.' }
            $result.Native = $native
            $result.State = 'Mounted'
        }
        else { $result.State = 'WouldMount' }
    }
    [pscustomobject]$result
}

if ($MyInvocation.InvocationName -ne '.') {
    if ($ShowHelp -or '--help' -in $RemainingArguments) { Get-Help $PSCommandPath -Detailed; return }
    if ($RemainingArguments.Count -gt 0) { throw 'Unrecognized positional argument; use -h for help.' }
    if ($ShowTaskTemplate) { Get-VigilNfsTaskTemplate -ScriptPath $PSCommandPath; return }
    $forward = @{
        Action = $Action; Drive = $Drive; Apply = $Apply; DryRun = $DryRun
        ProbeTimeoutMilliseconds = $ProbeTimeoutMilliseconds; NativeTimeoutSeconds = $NativeTimeoutSeconds
    }
    if ($PSBoundParameters.ContainsKey('WhatIf')) { $forward.WhatIf = $PSBoundParameters.WhatIf }
    if ($PSBoundParameters.ContainsKey('Confirm')) { $forward.Confirm = $PSBoundParameters.Confirm }
    Invoke-VigilFleetShare @forward
}
