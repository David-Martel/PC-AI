#Requires -Version 7.0
<#
.SYNOPSIS
Inspect, prepare, configure or benchmark an isolated Linux Thunderbolt peer.
.DESCRIPTION
Uses existing key-authenticated LAN SSH with strict host-key checking. Status is
read-only. Mutations require -Apply and ShouldProcess approval. No gateway, DNS,
WinRM, global firewall or Thunderbolt security changes are made.
Select an SSH application/config explicitly when PATH or the agent pipe is
unreliable. DisableSshAgent selects configured/default identity files only for
this invocation; user and host-key verification remain in effect. A dedicated
config preserves aliases; explicit configless mode uses SSH's default user and
identity files and requires the profile alias to resolve the expected host.
DryRun reads the peer
profile only and does not contact either host or inspect adapters.
.EXAMPLE
./Invoke-ThunderboltLinuxPeer.ps1 -Peer millylaptop1 -Action Configure -DryRun
.EXAMPLE
./Invoke-ThunderboltLinuxPeer.ps1 -Peer millylaptop1 -Action Prepare -Apply
#>
[CmdletBinding(SupportsShouldProcess)]
param(
    [ValidateSet('Status', 'Prepare', 'Configure', 'Benchmark')][string]$Action = 'Status',
    [string]$Peer = 'millylaptop1',
    [string]$ConfigPath = (Join-Path $PSScriptRoot '../../../Config/thunderbolt-peers.json'),
    [string]$InterfaceAlias,
    [string]$LinuxInterface,
    [switch]$Apply,
    [switch]$DryRun,
    [ValidateRange(1, 60)][int]$DurationSeconds = 10,
    [ValidateRange(1024, 65535)][int]$Port = 5201,
    [string]$IperfPath = 'iperf3',
    [string]$SshPath = 'ssh',
    [string]$SshConfigFile,
    [switch]$DisableSshAgent,
    [Alias('h', 'help')][switch]$ShowHelp
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'Invoke-VigilBoundedProcess.ps1')

function Resolve-TbSshTransport {
    [CmdletBinding()]
    param([string]$Path = 'ssh', [string]$ConfigFile, [switch]$DisableAgent)
    $application = Get-Command -Name $Path -CommandType Application -ErrorAction Stop | Select-Object -First 1
    if (-not $application) { throw 'SSH application was not found.' }
    $arguments = @()
    if ($ConfigFile) {
        if ($ConfigFile -eq 'none') {
            # Explicit opt-in only: SSH default identity/user and resolvable alias must be suitable.
            $resolvedConfig = 'none'
        }
        else {
            $file = Get-Item -LiteralPath $ConfigFile -ErrorAction Stop
            if ($file.PSIsContainer -or $file.PSProvider.Name -ne 'FileSystem') { throw 'SSH config must be a filesystem file.' }
            $resolvedConfig = $file.FullName
        }
        $arguments += @('-F', $resolvedConfig)
    }
    if ($DisableAgent) { $arguments += @('-o', 'IdentityAgent=none', '-o', 'IdentitiesOnly=yes') }
    return @{ FilePath = $application.Source; Arguments = $arguments }
}

function Get-TbSshArguments {
    [CmdletBinding()]
    param([hashtable]$SshTransport, [string[]]$Arguments = @())
    if ($SshTransport) { return @($SshTransport.Arguments) + $Arguments }
    return $Arguments
}

function New-TbSshStartInfo {
    [CmdletBinding()]
    param([hashtable]$SshTransport, [string[]]$Arguments)
    $path = if ($SshTransport) { $SshTransport.FilePath } else { 'ssh' }
    $start = [Diagnostics.ProcessStartInfo]::new($path)
    $start.UseShellExecute = $false
    $start.CreateNoWindow = $true
    $start.RedirectStandardInput = $true
    $start.RedirectStandardOutput = $true
    $start.RedirectStandardError = $true
    foreach ($argument in @(Get-TbSshArguments -SshTransport $SshTransport -Arguments $Arguments)) {
        $start.ArgumentList.Add($argument)
    }
    return $start
}

function Invoke-TbNative {
    [CmdletBinding()]
    param([string]$FilePath, [string[]]$Arguments = @(), [string]$InputText = '',
        [ValidateRange(1, 3600)][int]$TimeoutSeconds = 30)
    $result = Invoke-VigilBoundedProcess -FilePath $FilePath -Arguments $Arguments `
        -InputText $InputText.Replace("`r`n", "`n") -TimeoutSeconds $TimeoutSeconds
    if ($result.ExitCode -ne 0) {
        throw "$FilePath exited $($result.ExitCode): $($result.Stderr) $($result.Stdout)"
    }
    return $result.Stdout
}

function Invoke-TbSsh {
    [CmdletBinding()]
    param([string]$SshAlias, [string]$Script, [int]$TimeoutSeconds = 30,
        [string]$Address, [string]$KnownHostIdentity, [string]$SourceAddress, [hashtable]$SshTransport)
    $arguments = @('-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=10')
    if ($Address) {
        $arguments += @('-o', "HostName=$Address", '-o', "HostKeyAlias=$KnownHostIdentity",
            '-o', 'ProxyJump=none', '-o', 'ProxyCommand=none', '-b', $SourceAddress)
    }
    $arguments += @($SshAlias, 'bash', '-s')
    $path = if ($SshTransport) { $SshTransport.FilePath } else { 'ssh' }
    Invoke-TbNative -FilePath $path -Arguments (Get-TbSshArguments -SshTransport $SshTransport -Arguments $arguments) `
        -InputText $Script -TimeoutSeconds $TimeoutSeconds
}

function Get-TbProfile {
    [CmdletBinding()]
    param([string]$Path, [string]$Name)
    $config = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json -AsHashtable
    if ($config.schemaVersion -ne 1 -or -not $config.peers.ContainsKey($Name)) { throw "Unknown peer profile: $Name." }
    $profile = $config.peers[$Name]
    foreach ($field in @('sshAlias', 'expectedHostname')) {
        if ($profile[$field] -notmatch '^[A-Za-z0-9][A-Za-z0-9_.-]*$') { throw "Unsafe or missing $field." }
    }
    if ($Name -notmatch '^[A-Za-z0-9][A-Za-z0-9_.-]*$') { throw 'Unsafe peer name.' }
    foreach ($field in @('prefixLength', 'mtuBytes')) {
        if ($profile[$field] -isnot [int] -and $profile[$field] -isnot [long]) {
            throw "Profile $field must be an integer JSON value."
        }
    }
    if ($profile.prefixLength -ne 30 -or $profile.mtuBytes -lt 1280 -or $profile.mtuBytes -gt 9000) {
        throw 'Profile requires a /30 and an MTU from 1280 through 9000.'
    }
    $profile.prefixLength = [int]$profile.prefixLength
    $profile.mtuBytes = [int]$profile.mtuBytes
    $addresses = foreach ($field in @('localAddress', 'peerAddress')) {
        $ip = [Net.IPAddress]::Parse($profile[$field])
        if ($ip.AddressFamily -ne [Net.Sockets.AddressFamily]::InterNetwork) { throw 'IPv4 required.' }
        $bytes = $ip.GetAddressBytes()
        if ($bytes[0] -ne 172 -or $bytes[1] -ne 31 -or $bytes[2] -ne 240 -or ($bytes[3] % 4) -notin @(1, 2)) {
            throw 'Profiles must use host addresses from the reserved 172.31.240.0/24 pool.'
        }
        $bytes[3]
    }
    if ($addresses[0] -eq $addresses[1] -or [math]::Floor($addresses[0] / 4) -ne [math]::Floor($addresses[1] / 4)) {
        throw 'Peer addresses must be distinct hosts in the same /30.'
    }
    foreach ($otherName in $config.peers.Keys) {
        if ($otherName -ne $Name) {
            $other = [Net.IPAddress]::Parse($config.peers[$otherName].localAddress).GetAddressBytes()
            if ([math]::Floor($other[3] / 4) -eq [math]::Floor($addresses[0] / 4)) { throw 'Peer profiles overlap.' }
        }
    }
    return $profile
}

function Select-TbWindowsAdapter {
    [CmdletBinding()]
    param([object[]]$Adapters, [string]$Alias)
    $candidates = @($Adapters | Where-Object {
            $_.InterfaceDescription -match '(?i)Thunderbolt.*(Network|Ethernet)|USB4.*(P2P|Peer).*Network'
        })
    if ($Alias) { $candidates = @($candidates | Where-Object Name -EQ $Alias) }
    if ($candidates.Count -ne 1) { throw "Blocked: expected exactly one genuine Thunderbolt peer adapter; found $($candidates.Count)." }
    if ($candidates[0].Status -ne 'Up') { throw 'Blocked: Thunderbolt adapter is not Up.' }
    return $candidates[0]
}

function Get-TbLinuxInventory {
    [CmdletBinding()]
    param([string]$SshAlias, [hashtable]$SshTransport)
    $script = @'
set -euo pipefail
python3 - <<'PY'
import glob,json,os,socket,subprocess
interfaces=[]
for path in glob.glob('/sys/class/net/*'):
    driver=os.path.basename(os.path.realpath(path+'/device/driver')) if os.path.exists(path+'/device/driver') else ''
    if driver=='thunderbolt-net':
        def read(name):
            try:
                with open(path+'/'+name) as f: return f.read().strip()
            except OSError: return ''
        interfaces.append({'name':os.path.basename(path),'driver':driver,'carrier':read('carrier'),'mtu':read('mtu')})
devices=[]
for path in glob.glob('/sys/bus/thunderbolt/devices/*'):
    devices.append({'name':os.path.basename(path),'path':os.path.realpath(path)})
def command(args):
    result=subprocess.run(args,check=True,text=True,capture_output=True)
    return json.loads(result.stdout)
print(json.dumps({'hostname':socket.gethostname(),'interfaces':interfaces,'thunderboltDevices':devices,
 'addresses':command(['ip','-j','-4','addr','show']),'routes':command(['ip','-j','-4','route','show','table','all']),
 'moduleLoaded':os.path.isdir('/sys/module/thunderbolt_net'),
 'iperf3Available':subprocess.run(['sh','-c','command -v iperf3 >/dev/null'],check=False).returncode==0,
 'networkManagerAvailable':subprocess.run(['sh','-c','command -v nmcli >/dev/null'],check=False).returncode==0}))
PY
'@
    Invoke-TbSsh -SshAlias $SshAlias -Script $script -SshTransport $SshTransport | ConvertFrom-Json
}

function Select-TbLinuxInterface {
    [CmdletBinding()]
    param([object]$Inventory, [string]$ExpectedHostname, [string]$Name)
    if ($Inventory.hostname -cne $ExpectedHostname) { throw "Peer identity mismatch: expected $ExpectedHostname; received $($Inventory.hostname)." }
    $interfaces = @($Inventory.interfaces | Where-Object driver -EQ 'thunderbolt-net')
    if ($Name) { $interfaces = @($interfaces | Where-Object name -EQ $Name) }
    if ($interfaces.Count -ne 1 -or $interfaces[0].carrier -ne '1') { throw 'Blocked: Linux needs one thunderbolt-net interface with carrier.' }
    if ($interfaces[0].name -notmatch '^[A-Za-z0-9_.-]+$') { throw 'Unsafe Linux interface name.' }
    return $interfaces[0]
}

function Assert-TbAddressSafety {
    [CmdletBinding()]
    param([hashtable]$Profile, [object]$Adapter, [object]$Inventory, [string]$RemoteInterface)
    $localIps = @(Get-NetIPAddress -AddressFamily IPv4 -ErrorAction Stop)
    foreach ($ip in $localIps) {
        if ($ip.IPAddress -eq $Profile.localAddress -and $ip.InterfaceIndex -ne $Adapter.ifIndex) { throw 'Local address belongs to another interface.' }
        if ($ip.InterfaceIndex -eq $Adapter.ifIndex -and $ip.IPAddress -notmatch '^169\.254\.' -and
            ($ip.IPAddress -ne $Profile.localAddress -or $ip.PrefixLength -ne 30)) { throw 'Unrelated IPv4 address on peer adapter; preserve it and resolve explicitly.' }
    }
    foreach ($route in @(Get-NetRoute -AddressFamily IPv4 -ErrorAction Stop)) {
        if ($route.InterfaceIndex -eq $Adapter.ifIndex -and $route.DestinationPrefix -eq '0.0.0.0/0') {
            throw 'Peer adapter already has a default route; resolve it before isolated configuration.'
        }
        if ($route.InterfaceIndex -ne $Adapter.ifIndex -and
            ((Test-TbRouteContainsAddress -Prefix $route.DestinationPrefix -Address $Profile.localAddress) -or
            (Test-TbRouteContainsAddress -Prefix $route.DestinationPrefix -Address $Profile.peerAddress))) {
            throw 'Peer subnet route belongs to another interface.'
        }
    }
    foreach ($link in $Inventory.addresses) {
        foreach ($ip in $link.addr_info) {
            if ($ip.local -eq $Profile.peerAddress -and $link.ifname -ne $RemoteInterface) { throw 'Peer address belongs to another Linux interface.' }
            if ($link.ifname -eq $RemoteInterface -and $ip.scope -eq 'global' -and
                ($ip.local -ne $Profile.peerAddress -or $ip.prefixlen -ne 30)) { throw 'Unrelated Linux global IPv4 address; preserve it and resolve explicitly.' }
        }
    }
    foreach ($route in $Inventory.routes) {
        if ($route.PSObject.Properties.Name -contains 'dev' -and $route.dev -eq $RemoteInterface -and
            ($route.dst -eq 'default' -or $route.PSObject.Properties.Name -contains 'gateway')) {
            throw 'Linux peer interface already has a gateway route; resolve it before isolated configuration.'
        }
        if ($route.PSObject.Properties.Name -contains 'dst' -and $route.dst -ne 'default' -and
            ((Test-TbRouteContainsAddress -Prefix $route.dst -Address $Profile.localAddress) -or
            (Test-TbRouteContainsAddress -Prefix $route.dst -Address $Profile.peerAddress))) {
            if ($route.PSObject.Properties.Name -notcontains 'dev' -or $route.dev -ne $RemoteInterface) {
                throw 'Peer subnet route belongs to another Linux interface.'
            }
        }
    }
}

function Test-TbRouteContainsAddress {
    [CmdletBinding()]
    param([string]$Prefix, [string]$Address)
    if ($Prefix -eq 'default' -or $Prefix -eq '0.0.0.0/0') { return $false }
    if ($Prefix -notmatch '^([0-9.]+)(?:/(\d+))?$') { throw "Cannot validate IPv4 route prefix $Prefix." }
    $networkAddress = $Matches[1]
    $bits = $(if ($Matches[2]) { [int]$Matches[2] } else { 32 })
    if ($bits -lt 0 -or $bits -gt 32) { throw 'Invalid IPv4 route prefix length.' }
    $size = [Math]::Pow(2, 32 - $bits)
    $values = foreach ($textAddress in @($networkAddress, $Address)) {
        $bytes = [Net.IPAddress]::Parse($textAddress).GetAddressBytes()
        if ($bytes.Count -ne 4) { throw 'IPv4 route expected.' }
        [uint64]$bytes[0] * 16777216 + [uint64]$bytes[1] * 65536 + [uint64]$bytes[2] * 256 + [uint64]$bytes[3]
    }
    return [Math]::Floor($values[0] / $size) -eq [Math]::Floor($values[1] / $size)
}

function New-TbLinuxConfigureScript {
    [CmdletBinding()]
    param([hashtable]$Profile, [string]$Name, [string]$Interface)
    # Every interpolated value was validated before this function is called.
    @"
set -euo pipefail
test "`$(hostname)" = '$($Profile.expectedHostname)'
test "`$(basename "`$(readlink -f /sys/class/net/$Interface/device/driver)")" = thunderbolt-net
test "`$(cat /sys/class/net/$Interface/carrier)" = 1
connection='pcai-tb-$Name'
rows=`$(nmcli -t --escape no -f UUID,NAME connection show)
matches=`$(printf '%s\n' "`$rows" | awk -F: -v name="`$connection" '`$2 == name {print `$1}')
if [ -n "`$matches" ]; then
    test "`$(printf '%s\n' "`$matches" | wc -l)" -eq 1 || { echo 'Ambiguous existing Thunderbolt profile.' >&2; exit 1; }
    connection_uuid="`$matches"
    require_setting() {
        actual=`$(nmcli -g "`$1" connection show uuid "`$connection_uuid")
        if [ "`$actual" != "`$2" ]; then
            printf 'Existing Thunderbolt profile has unexpected %s; refusing activation.\n' "`$1" >&2
            exit 1
        fi
    }
    require_setting connection.type 802-3-ethernet
    require_setting connection.interface-name '$Interface'
    require_setting connection.autoconnect yes
    require_setting ipv4.method manual
    require_setting ipv4.addresses '$($Profile.peerAddress)/30'
    require_setting ipv4.gateway ''
    require_setting ipv4.dns ''
    require_setting ipv4.routes ''
    require_setting ipv4.never-default yes
    require_setting ipv4.ignore-auto-routes yes
    require_setting ipv4.ignore-auto-dns yes
    require_setting ipv6.method disabled
    require_setting ipv6.gateway ''
    require_setting ipv6.dns ''
    require_setting ipv6.routes ''
    require_setting ipv6.never-default yes
    selector=(uuid "`$connection_uuid")
else
    sudo -n nmcli connection add type ethernet ifname '$Interface' con-name "`$connection" ipv4.method manual ipv4.addresses '$($Profile.peerAddress)/30' ipv4.never-default yes ipv4.ignore-auto-routes yes ipv4.ignore-auto-dns yes ipv6.method disabled ipv6.never-default yes connection.autoconnect yes
    selector=(id "`$connection")
fi
sudo -n nmcli connection modify "`${selector[@]}" 802-3-ethernet.mtu '$($Profile.mtuBytes)' ipv4.gateway '' ipv4.dns '' ipv4.never-default yes ipv4.ignore-auto-routes yes ipv4.ignore-auto-dns yes
sudo -n nmcli connection up "`${selector[@]}" ifname '$Interface'
"@
}

function Assert-TbBenchmarkRoute {
    [CmdletBinding()]
    param([hashtable]$Profile, [object]$Adapter)
    $selected = @(Find-NetRoute -RemoteIPAddress $Profile.peerAddress -LocalIPAddress $Profile.localAddress)
    if (-not $selected.Count -or @($selected | Where-Object InterfaceIndex -NE $Adapter.ifIndex).Count) {
        throw 'Benchmark refused: source-bound route does not use the selected Thunderbolt interface.'
    }
    $source = @(Get-NetIPAddress -InterfaceIndex $Adapter.ifIndex -AddressFamily IPv4 | Where-Object IPAddress -EQ $Profile.localAddress)
    if ($source.Count -ne 1 -or $source[0].PrefixLength -ne 30) { throw 'Benchmark refused: local /30 address is missing.' }
}

function Test-TbSshEndpoint {
    [CmdletBinding()]
    param([hashtable]$Profile, [string]$RemoteInterface, [hashtable]$SshTransport)
    $path = if ($SshTransport) { $SshTransport.FilePath } else { 'ssh' }
    $effective = Invoke-TbNative -FilePath $path -Arguments (Get-TbSshArguments -SshTransport $SshTransport `
        -Arguments @('-G', $Profile.sshAlias)) -TimeoutSeconds 10
    $hostname = ($effective -split "`n" | Where-Object { $_ -match '^hostname ' }) -replace '^hostname ', ''
    $hostKey = ($effective -split "`n" | Where-Object { $_ -match '^hostkeyalias ' }) -replace '^hostkeyalias ', ''
    if ($hostKey) { $hostKey = $hostKey.Trim() }
    if (-not $hostKey -or $hostKey -eq 'none') { $hostKey = $hostname.Trim() }
    if ($hostKey -notmatch '^[A-Za-z0-9_.:-]+$') { throw 'Unsafe effective SSH host-key identity.' }
    $script = @"
set -euo pipefail
test "`$(hostname)" = '$($Profile.expectedHostname)'
ip -j -4 route get '$($Profile.localAddress)' | python3 -c 'import json,sys; r=json.load(sys.stdin)[0]; assert r["dev"]=="$RemoteInterface"; print(json.dumps(r))'
"@
    $reply = Invoke-TbSsh -SshAlias $Profile.sshAlias -Address $Profile.peerAddress -KnownHostIdentity $hostKey `
        -SourceAddress $Profile.localAddress -Script $script -SshTransport $SshTransport | ConvertFrom-Json
    $command = "ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o ProxyJump=none -o ProxyCommand=none -b $($Profile.localAddress) -o HostName=$($Profile.peerAddress) -o HostKeyAlias=$hostKey $($Profile.sshAlias)"
    if ($SshTransport) {
        $commandArguments = Get-TbSshArguments -SshTransport $SshTransport -Arguments @('-o', 'BatchMode=yes',
            '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=10', '-o', "HostName=$($Profile.peerAddress)",
            '-o', "HostKeyAlias=$hostKey", '-o', 'ProxyJump=none', '-o', 'ProxyCommand=none',
            '-b', $Profile.localAddress, $Profile.sshAlias, 'bash', '-s')
        $quotedArguments = @($commandArguments | ForEach-Object { "'" + $_.Replace("'", "''") + "'" })
        $command = "& '" + $SshTransport.FilePath.Replace("'", "''") + "' " + ($quotedArguments -join ' ')
    }
    [pscustomobject]@{ Verified = $true; PeerAddress = $Profile.peerAddress; SourceAddress = $Profile.localAddress;
        KnownHostIdentity = $hostKey; LinuxReturnRoute = $reply;
        Command = $command
    }
}

function Invoke-TbBenchmark {
    [CmdletBinding()]
    param([hashtable]$Profile, [object]$Adapter, [string]$RemoteInterface, [string]$Executable, [int]$Seconds, [int]$ServerPort,
        [hashtable]$SshTransport)
    Assert-TbBenchmarkRoute -Profile $Profile -Adapter $Adapter
    $results = @()
    foreach ($reverse in @($false, $true)) {
        $start = New-TbSshStartInfo -SshTransport $SshTransport -Arguments @('-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=10', $Profile.sshAlias, 'bash', '-s')
        $server = [Diagnostics.Process]::new()
        $server.StartInfo = $start
        $serverStarted = $false
        try {
            if (-not $server.Start()) { throw 'Could not start benchmark SSH server.' }
            $serverStarted = $true
            $serverErr = $server.StandardError.ReadToEndAsync()
            $serverScript = @"
set -euo pipefail
test "`$(hostname)" = '$($Profile.expectedHostname)'
ip -4 addr show dev '$RemoteInterface' | grep -Fq 'inet $($Profile.peerAddress)/30 '
timeout $($Seconds + 15)s iperf3 -s -1 -B '$($Profile.peerAddress)' -p $ServerPort &
child=`$!
trap 'kill "`$child" 2>/dev/null || true' EXIT HUP INT TERM
sleep 0.5
kill -0 "`$child"
echo PCAI_IPERF_READY
wait "`$child"
"@
            $deadline = [DateTime]::UtcNow.AddSeconds(10)
            $inputTask = $server.StandardInput.WriteAsync($serverScript.Replace("`r`n", "`n"))
            if (-not $inputTask.Wait(10000)) { throw 'Remote benchmark server input timed out.' }
            $inputTask.GetAwaiter().GetResult()
            $server.StandardInput.Close()
            $ready = $false
            while (-not $ready -and [DateTime]::UtcNow -lt $deadline) {
                $lineTask = $server.StandardOutput.ReadLineAsync()
                if (-not $lineTask.Wait([int][Math]::Max(1, ($deadline - [DateTime]::UtcNow).TotalMilliseconds))) {
                    throw 'Remote benchmark server readiness timed out.'
                }
                $line = $lineTask.GetAwaiter().GetResult()
                if ($null -eq $line) {
                    if (-not $serverErr.Wait(1000)) { throw 'Remote benchmark server failed and stderr capture timed out.' }
                    throw "Remote benchmark server failed: $($serverErr.GetAwaiter().GetResult())"
                }
                $ready = $line -eq 'PCAI_IPERF_READY'
            }
            if (-not $ready) { throw 'Remote benchmark server did not become ready.' }
            $serverOut = $server.StandardOutput.ReadToEndAsync()
            Assert-TbBenchmarkRoute -Profile $Profile -Adapter $Adapter
            $arguments = @('-c', $Profile.peerAddress, '-B', $Profile.localAddress, '-p', "$ServerPort", '-t', "$Seconds", '-J')
            if ($reverse) { $arguments += '-R' }
            $json = Invoke-TbNative -FilePath $Executable -Arguments $arguments -TimeoutSeconds ($Seconds + 15) | ConvertFrom-Json
            if ($json.PSObject.Properties.Name -contains 'error') { throw "iperf3 reported: $($json.error)" }
            if (-not $server.WaitForExit(10000)) { throw 'Remote one-shot benchmark server did not finish.' }
            if (-not [Threading.Tasks.Task]::WaitAll(@($serverErr, $serverOut), 1000)) { throw 'Remote benchmark server output capture timed out.' }
            if ($server.ExitCode -ne 0) { throw "Remote benchmark server exited $($server.ExitCode): $($serverErr.GetAwaiter().GetResult())" }
            $null = $serverOut.GetAwaiter().GetResult()
            $results += [pscustomobject]@{ Direction = $(if ($reverse) { 'LinuxToWindows' } else { 'WindowsToLinux' }); Measurement = $json }
        }
        finally {
            if ($serverStarted -and -not $server.HasExited) { $server.Kill($true) }
            $server.Dispose()
        }
    }
    return $results
}

function Invoke-ThunderboltLinuxPeerMain {
    [CmdletBinding(SupportsShouldProcess)]
    param([string]$SelectedAction, [string]$SelectedPeer, [string]$ProfilePath, [string]$WindowsAlias,
        [string]$RemoteInterface, [switch]$EnableApply, [switch]$IsDryRun,
        [int]$Seconds = 10, [int]$ServerPort = 5201, [string]$Executable = 'iperf3',
        [string]$SshPath = 'ssh', [string]$SshConfigFile, [switch]$DisableSshAgent)
    $profile = Get-TbProfile -Path $ProfilePath -Name $SelectedPeer
    if ($IsDryRun) {
        return [pscustomobject]@{ Action = $SelectedAction; Peer = $SelectedPeer; State = 'Planned'; Applied = $false;
            Profile = $profile; WindowsAdapters = @(); Linux = $null; Blockers = @(); Measurements = @();
            Plan = @('Dry-run profile only: live adapter, SSH identity and route checks are required before applying.');
            SshRecoveryAlias = $profile.sshAlias; ThunderboltSsh = $null }
    }
    $sshTransport = Resolve-TbSshTransport -Path $SshPath -ConfigFile $SshConfigFile -DisableAgent:$DisableSshAgent
    $linux = Get-TbLinuxInventory -SshAlias $profile.sshAlias -SshTransport $sshTransport
    if ($linux.hostname -cne $profile.expectedHostname) { throw 'SSH peer hostname does not match profile.' }
    $adapters = @(Get-NetAdapter -IncludeHidden)
    $result = [ordered]@{ Action = $SelectedAction; Peer = $SelectedPeer; State = 'Observed'; Applied = $false;
        Profile = $profile; WindowsAdapters = @($adapters | Select-Object Name, InterfaceDescription, Status, ifIndex, LinkSpeed, MacAddress);
        Linux = $linux; Blockers = @(); Plan = @(); Measurements = @();
        SshRecoveryAlias = $profile.sshAlias; ThunderboltSsh = $null
    }
    $adapter = $null
    $remote = $null
    try { $adapter = Select-TbWindowsAdapter -Adapters $adapters -Alias $WindowsAlias } catch { $result.Blockers += $_.Exception.Message }
    try { $remote = Select-TbLinuxInterface -Inventory $linux -ExpectedHostname $profile.expectedHostname -Name $RemoteInterface } catch { $result.Blockers += $_.Exception.Message }
    if ($result.Blockers.Count) { $result.State = 'Blocked' }
    if ($SelectedAction -eq 'Status') { return [pscustomobject]$result }
    if ($SelectedAction -eq 'Prepare') {
        $result.Plan = @('Load and persist thunderbolt_net; install iperf3 without enabling its daemon; retain LAN SSH.')
        if ($EnableApply -and -not $IsDryRun -and $PSCmdlet.ShouldProcess($profile.sshAlias, 'Prepare Linux Thunderbolt networking tools')) {
            $script = @"
set -euo pipefail
test "`$(hostname)" = '$($profile.expectedHostname)'
sudo -n modprobe thunderbolt_net
printf '%s\n' thunderbolt_net | sudo -n tee /etc/modules-load.d/pcai-thunderbolt-net.conf >/dev/null
if ! command -v iperf3 >/dev/null; then
    printf '%s\n' 'iperf3 iperf3/start_daemon boolean false' | sudo -n debconf-set-selections
    sudo -n env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends iperf3
fi
"@
            $null = Invoke-TbSsh -SshAlias $profile.sshAlias -Script $script -TimeoutSeconds 180 -SshTransport $sshTransport
            $result.Applied = $true
            $result.State = 'Prepared'
        }
        return [pscustomobject]$result
    }
    if ($result.Blockers.Count) { throw "Cannot $SelectedAction. $($result.Blockers -join ' ')" }
    if ($SelectedAction -eq 'Configure') {
        if (-not $linux.networkManagerAvailable) { throw 'NetworkManager is required for persistent Linux configuration.' }
        Assert-TbAddressSafety -Profile $profile -Adapter $adapter -Inventory $linux -RemoteInterface $remote.name
        $result.Plan = @("Add $($profile.localAddress)/30 on $($adapter.Name) and $($profile.peerAddress)/30 on $($remote.name); MTU $($profile.mtuBytes); no gateway or DNS.")
        if ($EnableApply -and -not $IsDryRun -and $PSCmdlet.ShouldProcess($SelectedPeer, 'Configure isolated Thunderbolt /30 on both hosts')) {
            $script = New-TbLinuxConfigureScript -Profile $profile -Name $SelectedPeer -Interface $remote.name
            $null = Invoke-TbSsh -SshAlias $profile.sshAlias -Script $script -TimeoutSeconds 60 -SshTransport $sshTransport
            Set-NetIPInterface -InterfaceIndex $adapter.ifIndex -AddressFamily IPv4 -NlMtuBytes $profile.mtuBytes
            $existing = @(Get-NetIPAddress -InterfaceIndex $adapter.ifIndex -AddressFamily IPv4 | Where-Object IPAddress -EQ $profile.localAddress)
            if (-not $existing.Count) { $null = New-NetIPAddress -InterfaceIndex $adapter.ifIndex -IPAddress $profile.localAddress -PrefixLength 30 }
            Assert-TbBenchmarkRoute -Profile $profile -Adapter $adapter
            $result.ThunderboltSsh = Test-TbSshEndpoint -Profile $profile -RemoteInterface $remote.name -SshTransport $sshTransport
            $result.Applied = $true
            $result.State = 'Configured'
        }
    }
    else {
        Assert-TbAddressSafety -Profile $profile -Adapter $adapter -Inventory $linux -RemoteInterface $remote.name
        Assert-TbBenchmarkRoute -Profile $profile -Adapter $adapter
        if (-not $linux.iperf3Available) { throw 'Linux iperf3 is unavailable; run Prepare first.' }
        $result.Plan = @('Measure source-bound iperf3 forward and reverse throughput with bounded one-shot servers.')
        if ($EnableApply -and -not $IsDryRun -and $PSCmdlet.ShouldProcess($SelectedPeer, 'Run two iperf3 measurements')) {
            $result.Measurements = @(Invoke-TbBenchmark -Profile $profile -Adapter $adapter -RemoteInterface $remote.name -Executable $Executable -Seconds $Seconds -ServerPort $ServerPort -SshTransport $sshTransport)
            $result.Applied = $true
            $result.State = 'Measured'
        }
    }
    return [pscustomobject]$result
}

if ($MyInvocation.InvocationName -ne '.') {
    if ($ShowHelp) { Get-Help $PSCommandPath -Detailed; return }
    Invoke-ThunderboltLinuxPeerMain -SelectedAction $Action -SelectedPeer $Peer -ProfilePath $ConfigPath `
        -WindowsAlias $InterfaceAlias -RemoteInterface $LinuxInterface -EnableApply:$Apply -IsDryRun:$DryRun `
        -Seconds $DurationSeconds -ServerPort $Port -Executable $IperfPath -WhatIf:$WhatIfPreference `
        -SshPath $SshPath -SshConfigFile $SshConfigFile -DisableSshAgent:$DisableSshAgent
}
