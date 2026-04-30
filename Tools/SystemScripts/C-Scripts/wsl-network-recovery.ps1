# WSL Toolkit - Network and Configuration
# Run as Administrator for full repairs

[CmdletBinding()]
param(
    [string]$Distro,
    [switch]$Check,
    [switch]$Diagnose,
    [switch]$Repair,
    [switch]$Full,
    [switch]$Optimize,
    [switch]$ApplyConfig,
    [switch]$TestNetworkingMode,
    [ValidateSet('nat','mirrored','virtioproxy')][string]$NetworkingMode,
    [string]$Memory,
    [int]$Processors,
    [string]$Swap,
    [string]$SwapFile,
    [switch]$FixDns,
    [string[]]$DnsServers = @('8.8.8.8','1.1.1.1','8.8.4.4'),
    [switch]$RestartWsl,
    [switch]$ResetAdapters,
    [switch]$ResetWinsock,
    [switch]$RestartHns,
    [switch]$RestartWslService,
    [switch]$DisableVmqOnWsl,
    [switch]$Force
)

$ErrorActionPreference = 'Continue'

if (-not ($Check -or $Diagnose -or $Repair -or $Full -or $Optimize -or $ApplyConfig)) {
    $Check = $true
}

$script:LogDir = Join-Path $env:USERPROFILE '.wsl-toolkit'
if (-not (Test-Path $script:LogDir)) {
    New-Item -ItemType Directory -Path $script:LogDir -Force | Out-Null
}
$script:LogPath = Join-Path $script:LogDir ("wsl-toolkit-" + (Get-Date -Format 'yyyyMMdd-HHmmss') + '.log')

function Write-Log {
    param(
        [string]$Message,
        [string]$Color = 'Gray'
    )
    $stamp = (Get-Date).ToString('HH:mm:ss')
    $line = "[$stamp] $Message"
    Write-Host $line -ForegroundColor $Color
    Add-Content -Path $script:LogPath -Value $line
}

function Test-IsAdmin {
    $id = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = [Security.Principal.WindowsPrincipal]$id
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Ensure-WslPresent {
    if (-not (Get-Command wsl.exe -ErrorAction SilentlyContinue)) {
        Write-Log 'wsl.exe not found. Install WSL before running this script.' 'Red'
        exit 1
    }
}

function Normalize-WSLPath {
    param([string]$Path)
    if (-not $Path) { return $Path }
    return ($Path -replace '\\', '/')
}

function Get-DefaultDistroName {
    $list = & wsl.exe -l -q 2>$null
    $names = @($list | Where-Object { $_ -and $_.Trim() -ne '' })
    if (-not $names -or $names.Count -eq 0) {
        return $null
    }
    if ($script:Distro) {
        return $script:Distro
    }
    if ($names -contains 'Ubuntu') {
        return 'Ubuntu'
    }
    return $names[0]
}

function Invoke-Wsl {
    param(
        [string]$Command,
        [string]$User = 'root'
    )
    if (-not $script:DistroName) { return $false }
    $svc = Get-Service -Name WslService -ErrorAction SilentlyContinue
    if ($svc -and $svc.Status -ne 'Running') {
        Write-Log "WslService is $($svc.Status). Skipping WSL command." 'Yellow'
        return 1
    }
    $escaped = $Command.Replace('"', '\"')
    & wsl.exe -d $script:DistroName --user $User -- bash -lc "$escaped"
    return $LASTEXITCODE
}

function Invoke-WslWithTimeout {
    param(
        [string]$Command,
        [int]$TimeoutSec = 10,
        [string]$User = 'root'
    )
    if (-not $script:DistroName) { return 1 }
    $svc = Get-Service -Name WslService -ErrorAction SilentlyContinue
    if ($svc -and $svc.Status -ne 'Running') {
        Write-Log "WslService is $($svc.Status). Skipping WSL command." 'Yellow'
        return 1
    }
    $escaped = $Command.Replace('"', '\"')
    $args = @('-d', $script:DistroName, '--user', $User, '--', 'bash', '-lc', $escaped)
    $outFile = [System.IO.Path]::GetTempFileName()
    $errFile = [System.IO.Path]::GetTempFileName()
    $proc = Start-Process -FilePath 'wsl.exe' -ArgumentList $args -NoNewWindow -PassThru `
        -RedirectStandardOutput $outFile -RedirectStandardError $errFile
    $timedOut = $false
    try {
        Wait-Process -Id $proc.Id -Timeout $TimeoutSec | Out-Null
    } catch {
        $timedOut = $true
    }
    if ($timedOut) {
        try { Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue } catch { }
        Remove-Item -Path $outFile, $errFile -ErrorAction SilentlyContinue
        return 124
    }
    $exitCode = $proc.ExitCode
    Remove-Item -Path $outFile, $errFile -ErrorAction SilentlyContinue
    return $exitCode
}

function Start-WslDistro {
    param([int]$TimeoutSec = 20)
    $args = @('-d', $script:DistroName, '--', 'echo', 'WSL started')
    $proc = Start-Process -FilePath 'wsl.exe' -ArgumentList $args -NoNewWindow -PassThru
    $timedOut = $false
    try {
        Wait-Process -Id $proc.Id -Timeout $TimeoutSec | Out-Null
    } catch {
        $timedOut = $true
    }
    if ($timedOut) {
        Write-Log "WSL start timed out after $TimeoutSec seconds." 'Yellow'
        try { Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue } catch { }
        return $false
    }
    return $true
}

function Test-WslNetwork {
    for ($i = 0; $i -lt 3; $i++) {
        $code = Invoke-WslWithTimeout "command -v timeout >/dev/null 2>&1 && timeout 4s ping -c 1 -W 2 8.8.8.8 >/dev/null 2>&1 || ping -c 1 -W 2 8.8.8.8 >/dev/null 2>&1" 15
        if ($code -ne 0) {
            $code = Invoke-Wsl "command -v timeout >/dev/null 2>&1 && timeout 4s ping -c 1 -W 2 8.8.8.8 >/dev/null 2>&1 || ping -c 1 -W 2 8.8.8.8 >/dev/null 2>&1"
        }
        if ($code -eq 0) { return $true }
        Start-Sleep -Seconds 1
    }
    return $false
}

function Test-WslDns {
    for ($i = 0; $i -lt 3; $i++) {
        $code = Invoke-WslWithTimeout "command -v timeout >/dev/null 2>&1 && timeout 4s sh -c 'getent hosts google.com >/dev/null 2>&1 || nslookup google.com >/dev/null 2>&1' || sh -c 'getent hosts google.com >/dev/null 2>&1 || nslookup google.com >/dev/null 2>&1'" 15
        if ($code -ne 0) {
            $code = Invoke-Wsl "command -v timeout >/dev/null 2>&1 && timeout 4s sh -c 'getent hosts google.com >/dev/null 2>&1 || nslookup google.com >/dev/null 2>&1' || sh -c 'getent hosts google.com >/dev/null 2>&1 || nslookup google.com >/dev/null 2>&1'"
        }
        if ($code -eq 0) { return $true }
        Start-Sleep -Seconds 1
    }
    return $false
}

function Get-WslNetworkInfo {
    Write-Log 'WSL status:' 'Cyan'
    & wsl.exe --status
    Write-Log 'WSL distros:' 'Cyan'
    & wsl.exe -l -v

    if ($script:DistroName) {
        Write-Log "WSL distro: $script:DistroName" 'Cyan'
        Write-Log 'WSL network interfaces:' 'Cyan'
        Invoke-Wsl 'ip addr show'
        Write-Log 'WSL routing table:' 'Cyan'
        Invoke-Wsl 'ip route show'
        Write-Log 'WSL resolv.conf:' 'Cyan'
        Invoke-Wsl 'cat /etc/resolv.conf 2>/dev/null || true'
    }

    Write-Log 'Windows WSL adapters:' 'Cyan'
    Get-NetAdapter | Where-Object { $_.Name -like '*WSL*' -or $_.Name -like 'vEthernet*' } | Format-Table Name, Status, LinkSpeed, MacAddress

    $wslAdapter = Get-NetAdapter | Where-Object { $_.Name -like '*WSL*' } | Select-Object -First 1
    if ($wslAdapter) {
        Write-Log "WSL adapter: $($wslAdapter.Name)" 'Cyan'
        try {
            Get-NetIPAddress -InterfaceAlias $wslAdapter.Name -ErrorAction SilentlyContinue | Format-Table IPAddress, PrefixLength, AddressFamily
        } catch {
        }
    }
}

function Analyze-WslConfig {
    $path = Join-Path $env:USERPROFILE '.wslconfig'
    $result = [ordered]@{
        Path = $path
        Exists = Test-Path $path
        Content = $null
        Recommendations = @()
    }

    if ($result.Exists) {
        $result.Content = Get-Content $path -Raw
        $networkingMode = $null
        if ($result.Content -match 'networkingMode\s*=\s*(\w+)') {
            $networkingMode = $matches[1].ToLowerInvariant()
        }
        if ($result.Content -notmatch 'networkingMode\s*=') {
            $result.Recommendations += 'Consider adding networkingMode=nat'
        }
        if ($result.Content -notmatch 'dnsTunneling\s*=') {
            $result.Recommendations += 'Consider adding dnsTunneling=true'
        }
        if ($result.Content -notmatch 'localhostForwarding\s*=' -and $networkingMode -ne 'mirrored') {
            $result.Recommendations += 'Consider adding localhostForwarding=true'
        }
        if ($result.Content -notmatch 'autoMemoryReclaim\s*=') {
            $result.Recommendations += 'Consider adding autoMemoryReclaim=gradual under [experimental]'
        }
        if ($result.Content -notmatch 'memory\s*=') {
            $result.Recommendations += 'Consider setting memory=8GB (or appropriate value)'
        }
        if ($result.Content -notmatch 'processors\s*=') {
            $result.Recommendations += 'Consider setting processors=4 (or appropriate value)'
        }
        if ($result.Content -notmatch 'swap\s*=') {
            $result.Recommendations += 'Consider setting swap=2GB (or appropriate value)'
        }
    } else {
        $result.Recommendations += 'Create a .wslconfig file for consistent networking and memory behavior'
    }

    return [PSCustomObject]$result
}

function Apply-WslConfig {
    $path = Join-Path $env:USERPROFILE '.wslconfig'
    $backup = "$path.backup.$(Get-Date -Format 'yyyyMMdd-HHmmss')"

    if (Test-Path $path) {
        Copy-Item $path $backup -Force
        Write-Log "Backed up .wslconfig to $backup" 'Yellow'
    }

    $desired = [ordered]@{}
    if ($NetworkingMode) { $desired['networkingMode'] = $NetworkingMode } else { $desired['networkingMode'] = 'nat' }
    $desired['dnsTunneling'] = 'true'
    $enableLocalhostForwarding = $desired['networkingMode'] -ne 'mirrored'
    if ($enableLocalhostForwarding) {
        $desired['localhostForwarding'] = 'true'
    }
    $removeKeys = @()
    if (-not $enableLocalhostForwarding) {
        $removeKeys += 'localhostForwarding'
    }
    if ($Memory) { $desired['memory'] = $Memory }
    if ($Processors -gt 0) { $desired['processors'] = $Processors }
    if ($Swap) { $desired['swap'] = $Swap }
    if ($SwapFile) { $desired['swapFile'] = (Normalize-WSLPath $SwapFile) }

    $linesIn = @()
    $existingText = ''
    if (Test-Path $path) {
        $linesIn = Get-Content $path
        $existingText = Get-Content $path -Raw
    }


    $linesOut = New-Object System.Collections.Generic.List[string]
    $inWsl2 = $false
    $wsl2Found = $false
    $found = @{}

    foreach ($line in $linesIn) {
        if ($line -match '^\s*\[wsl2\]\s*$') {
            $inWsl2 = $true
            $wsl2Found = $true
            $linesOut.Add($line) | Out-Null
            continue
        }

        if ($line -match '^\s*\[.+\]\s*$') {
            if ($inWsl2) {
                foreach ($key in $desired.Keys) {
                    if (-not $found.ContainsKey($key)) {
                        $linesOut.Add("$key=$($desired[$key])") | Out-Null
                        $found[$key] = $true
                    }
                }
            }
            $inWsl2 = $false
            $linesOut.Add($line) | Out-Null
            continue
        }

        if ($inWsl2) {
            foreach ($removeKey in $removeKeys) {
                if ($line -match "^\s*$removeKey\s*=") {
                    continue
                }
            }
            $replaced = $false
            foreach ($key in $desired.Keys) {
                if ($line -match "^\s*$key\s*=") {
                    $linesOut.Add("$key=$($desired[$key])") | Out-Null
                    $found[$key] = $true
                    $replaced = $true
                    break
                }
            }
            if (-not $replaced) {
                $linesOut.Add($line) | Out-Null
            }
        } else {
            $linesOut.Add($line) | Out-Null
        }
    }

    if ($wsl2Found) {
        if ($inWsl2) {
            foreach ($key in $desired.Keys) {
                if (-not $found.ContainsKey($key)) {
                    $linesOut.Add("$key=$($desired[$key])") | Out-Null
                    $found[$key] = $true
                }
            }
        }
    } else {
        if ($linesOut.Count -gt 0) { $linesOut.Add('') | Out-Null }
        $linesOut.Add('[wsl2]') | Out-Null
        foreach ($key in $desired.Keys) {
            $linesOut.Add("$key=$($desired[$key])") | Out-Null
        }
    }

    $content = ($linesOut -join "`r`n") + "`r`n"
    Set-Content -Path $path -Value $content -Encoding UTF8
    Write-Log "Applied .wslconfig to $path" 'Green'
}

function Test-NetworkingModeSequence {
    param(
        [string[]]$Modes = @('mirrored','nat'),
        [int]$TimeoutSec = 20
    )

    $originalMode = $null
    $configPath = Join-Path $env:USERPROFILE '.wslconfig'
    if (Test-Path $configPath) {
        $content = Get-Content $configPath -Raw
        if ($content -match 'networkingMode\s*=\s*(\w+)') {
            $originalMode = $matches[1]
        }
    }
    if (-not $originalMode) { $originalMode = 'mirrored' }

    Write-Log "Testing networking modes: $($Modes -join ', ')" 'Cyan'
    foreach ($mode in $Modes) {
        Write-Log "Setting networkingMode=$mode" 'Yellow'
        $script:NetworkingMode = $mode
        Apply-WslConfig

        Write-Log "Restarting WSL for $mode..." 'Yellow'
        & wsl.exe --shutdown
        Start-Sleep -Seconds 2

        $started = Start-WslDistro -TimeoutSec $TimeoutSec
        if (-not $started) {
            Write-Log "Mode ${mode}: WSL start timed out." 'Red'
            continue
        }
        Start-Sleep -Seconds 2

        $netOk = $false
        $dnsOk = $false
        for ($i = 0; $i -lt 6; $i++) {
            $netOk = Test-WslNetwork
            $dnsOk = Test-WslDns
            if ($netOk -and $dnsOk) { break }
            Start-Sleep -Seconds 2
        }
        Write-Log ("Mode {0}: Internet {1}, DNS {2}" -f $mode, $(if ($netOk) { 'OK' } else { 'FAIL' }), $(if ($dnsOk) { 'OK' } else { 'FAIL' })) $(if ($netOk -and $dnsOk) { 'Green' } else { 'Red' })
    }

    Write-Log "Restoring networkingMode=$originalMode" 'Yellow'
    $script:NetworkingMode = $originalMode
    Apply-WslConfig
    & wsl.exe --shutdown
    Start-Sleep -Seconds 2
    Start-WslDistro -TimeoutSec $TimeoutSec | Out-Null
}

function Restart-Hns {
    Write-Log 'Restarting HNS service...' 'Yellow'
    try {
        Restart-Service hns -Force -ErrorAction Stop
        Start-Sleep -Seconds 3
        Write-Log 'HNS restarted.' 'Green'
        return $true
    } catch {
        Write-Log "HNS restart failed: $_" 'Red'
        return $false
    }
}

function Restart-WslSvc {
    $candidates = @('WslService', 'LxssManager')
    foreach ($name in $candidates) {
        try {
            $svc = Get-Service -Name $name -ErrorAction Stop
            Write-Log "Restarting $name service..." 'Yellow'
            Restart-Service $name -Force -ErrorAction Stop
            Start-Sleep -Seconds 3
            Write-Log "$name restarted." 'Green'
            return $true
        } catch {
            continue
        }
    }
    Write-Log "WSL service not restarted (tried: $($candidates -join ', '))." 'Red'
    return $false
}

function Reset-WslAdapters {
    $adapters = Get-NetAdapter | Where-Object { $_.Name -like '*WSL*' -or $_.Name -like 'vEthernet*WSL*' }
    if (-not $adapters) {
        Write-Log 'No WSL adapters found to reset.' 'Yellow'
        return $false
    }

    foreach ($adapter in $adapters) {
        try {
            Write-Log "Resetting adapter: $($adapter.Name)" 'Yellow'
            Disable-NetAdapter -Name $adapter.Name -Confirm:$false -ErrorAction Stop
            Start-Sleep -Seconds 2
            Enable-NetAdapter -Name $adapter.Name -Confirm:$false -ErrorAction Stop
            Start-Sleep -Seconds 2
        } catch {
            Write-Log "Adapter reset failed for $($adapter.Name): $_" 'Red'
        }
    }
    return $true
}

function Reset-NetStack {
    Write-Log 'Resetting Windows network stack (winsock/ip) ...' 'Yellow'
    try {
        netsh winsock reset | Out-Null
        netsh int ip reset | Out-Null
        ipconfig /flushdns | Out-Null
        Write-Log 'Network stack reset completed.' 'Green'
        return $true
    } catch {
        Write-Log "Network stack reset failed: $_" 'Red'
        return $false
    }
}

function Reset-HnsNetworks {
    if (-not (Get-Command Get-HnsNetwork -ErrorAction SilentlyContinue)) {
        Write-Log 'HNS cmdlets not available. Skipping HNS network cleanup.' 'Yellow'
        return $false
    }
    $networks = Get-HnsNetwork | Where-Object { $_.Name -like '*WSL*' }
    if (-not $networks) {
        Write-Log 'No HNS WSL networks found.' 'Yellow'
        return $false
    }
    foreach ($net in $networks) {
        try {
            Write-Log "Removing HNS network: $($net.Name)" 'Yellow'
            $net | Remove-HnsNetwork -ErrorAction Stop
        } catch {
            Write-Log "Failed to remove HNS network $($net.Name): $_" 'Red'
        }
    }
    return $true
}

function Fix-WslDnsInternal {
    $dnsText = ($DnsServers | ForEach-Object { "nameserver $_" }) -join '\n'
    $cmd1 = "printf '[network]\\ngenerateResolvConf = false\\n' > /etc/wsl.conf"
    $cmd2 = "printf '$dnsText\\n' > /etc/resolv.conf"
    $cmd3 = "cp /etc/resolv.conf /etc/resolv.conf.bak >/dev/null 2>&1 || true"
    $cmd = "$cmd3; $cmd1; $cmd2"
    $code = Invoke-Wsl $cmd 'root'
    if ($code -eq 0) {
        Write-Log 'WSL DNS configuration updated.' 'Green'
        return $true
    }
    Write-Log 'WSL DNS configuration update failed.' 'Red'
    return $false
}

function Disable-Vmq {
    $adapter = Get-NetAdapter | Where-Object { $_.Name -like '*WSL*' } | Select-Object -First 1
    if (-not $adapter) {
        Write-Log 'No WSL adapter found for VMQ check.' 'Yellow'
        return $false
    }
    if (-not (Get-Command Disable-NetAdapterVmq -ErrorAction SilentlyContinue)) {
        Write-Log 'Disable-NetAdapterVmq not available.' 'Yellow'
        return $false
    }
    try {
        $vmq = Get-NetAdapterVmq -Name $adapter.Name -ErrorAction SilentlyContinue
        if ($vmq -and $vmq.Enabled) {
            Write-Log "Disabling VMQ on adapter: $($adapter.Name)" 'Yellow'
            Disable-NetAdapterVmq -Name $adapter.Name -ErrorAction Stop
            Write-Log 'VMQ disabled.' 'Green'
        } else {
            Write-Log 'VMQ already disabled or not supported.' 'Green'
        }
        return $true
    } catch {
        Write-Log "Disable VMQ failed: $_" 'Red'
        return $false
    }
}

function Show-FeatureStatus {
    $features = @(
        'Microsoft-Hyper-V-All',
        'VirtualMachinePlatform',
        'Microsoft-Windows-Subsystem-Linux',
        'HypervisorPlatform'
    )
    Write-Log 'Windows feature status:' 'Cyan'
    foreach ($feature in $features) {
        try {
            $status = Get-WindowsOptionalFeature -Online -FeatureName $feature -ErrorAction SilentlyContinue
            if ($status) {
                Write-Log ("{0}: {1}" -f $feature, $status.State) 'Gray'
            }
        } catch {
        }
    }
}

function Show-ServiceStatus {
    $services = @('WslService','LxssManager','hns','vmcompute','vmms')
    Write-Log 'Service status:' 'Cyan'
    foreach ($svcName in $services) {
        try {
            $svc = Get-Service -Name $svcName -ErrorAction SilentlyContinue
            if ($svc) {
                Write-Log ("{0}: {1}" -f $svcName, $svc.Status) 'Gray'
            }
        } catch {
        }
    }
}

Ensure-WslPresent
$script:Distro = $Distro
$script:DistroName = Get-DefaultDistroName
if (-not $script:DistroName) {
    Write-Log 'No WSL distributions found. Install a distro before running this script.' 'Red'
    exit 1
}

Write-Log "Using WSL distro: $script:DistroName" 'Cyan'

if ($Diagnose) {
    Get-WslNetworkInfo
    Show-FeatureStatus
    Show-ServiceStatus

    $config = Analyze-WslConfig
    Write-Log "WSL config exists: $($config.Exists)" 'Cyan'
    if ($config.Exists) {
        Write-Log "WSL config path: $($config.Path)" 'Cyan'
    }
    foreach ($rec in $config.Recommendations) {
        Write-Log "Recommendation: $rec" 'Yellow'
    }

    Write-Log 'Connectivity tests:' 'Cyan'
    Write-Log ("Internet: {0}" -f $(if (Test-WslNetwork) { 'OK' } else { 'FAILED' })) 'Gray'
    Write-Log ("DNS: {0}" -f $(if (Test-WslDns) { 'OK' } else { 'FAILED' })) 'Gray'
    exit 0
}

if ($ApplyConfig) {
    Apply-WslConfig
}

if ($TestNetworkingMode) {
    Test-NetworkingModeSequence
    exit 0
}

if ($Optimize -or $DisableVmqOnWsl) {
    if (-not (Test-IsAdmin)) {
        Write-Log 'Optimize requires Administrator. Re-run as admin.' 'Yellow'
    } else {
        Disable-Vmq | Out-Null
    }
}

if ($Full) {
    $Repair = $true
    $ResetWinsock = $true
    $RestartHns = $true
    $RestartWslService = $true
    $ResetAdapters = $true
    $RestartWsl = $true
}

if ($Repair) {
    Write-Log 'Starting repair sequence...' 'Yellow'

    if ($RestartWsl -or $Full) {
        Write-Log 'Shutting down WSL...' 'Yellow'
        & wsl.exe --shutdown
        Start-Sleep -Seconds 3
    }

    if ($RestartHns) { Restart-Hns | Out-Null }
    if ($RestartWslService) { Restart-WslSvc | Out-Null }
    if ($ResetAdapters) { Reset-WslAdapters | Out-Null }
    if ($ResetWinsock) { Reset-NetStack | Out-Null }

    if ($Full) { Reset-HnsNetworks | Out-Null }

    Write-Log 'Starting WSL...' 'Yellow'
    $started = Start-WslDistro -TimeoutSec 20
    if (-not $started) {
        Write-Log 'WSL did not start within the timeout.' 'Yellow'
    }
    Start-Sleep -Seconds 2

    $netOk = Test-WslNetwork
    $dnsOk = Test-WslDns

    if (-not $dnsOk -and ($FixDns -or $Full)) {
        Fix-WslDnsInternal | Out-Null
        $dnsOk = Test-WslDns
    }

    Write-Log ("Internet after repair: {0}" -f $(if ($netOk) { 'OK' } else { 'FAILED' })) $(if ($netOk) { 'Green' } else { 'Red' })
    Write-Log ("DNS after repair: {0}" -f $(if ($dnsOk) { 'OK' } else { 'FAILED' })) $(if ($dnsOk) { 'Green' } else { 'Red' })

    if (-not $netOk -or -not $dnsOk) {
        Write-Log 'Repair did not fully resolve connectivity. Run with -Diagnose for details.' 'Yellow'
    } else {
        Write-Log 'WSL networking looks healthy.' 'Green'
    }
    exit 0
}

if ($Check) {
    Write-Log 'Checking WSL networking status...' 'Cyan'
    $netOk = Test-WslNetwork
    $dnsOk = Test-WslDns

    if ($netOk -and $dnsOk -and -not $Force) {
        Write-Log 'WSL networking is healthy.' 'Green'
        exit 0
    }

    Write-Log ("Internet: {0}" -f $(if ($netOk) { 'OK' } else { 'FAILED' })) $(if ($netOk) { 'Green' } else { 'Red' })
    Write-Log ("DNS: {0}" -f $(if ($dnsOk) { 'OK' } else { 'FAILED' })) $(if ($dnsOk) { 'Green' } else { 'Red' })

    Write-Log 'Consider running: .\\wsl-network-recovery.ps1 -Repair -FixDns' 'Yellow'
}
