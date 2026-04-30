param(
    [ValidateSet('All','Quick','Windows','WSL','Services','WSLInside')]
    [string[]]$Phase = 'All',
    [switch]$NoColor,
    [switch]$JsonRpc,
    [string]$JsonRpcId = "1",
    [string]$JsonRpcMethod = "Test-WslLocalhost"
)

# Simple status writers and helpers
function Write-Status {
    param(
        [Parameter(Mandatory)][ValidateSet('INFO','OK','WARN','ERROR')] [string]$Level,
        [Parameter(Mandatory)][string]$Message
    )
    $prefix = "[$Level]"

    # Always log to status log when JSON-RPC mode is enabled
    if ($JsonRpc) {
        if (-not $script:StatusLog) { $script:StatusLog = @() }
        $script:StatusLog += [pscustomobject]@{
            level     = $Level
            message   = $Message
            timestamp = (Get-Date).ToString('o')
        }
    }

    # Always write to terminal so progress is visible, even in JSON-RPC mode
    if ($NoColor) {
        Write-Host "$prefix $Message"
        return
    }
    switch ($Level) {
        'OK'    { Write-Host "$prefix $Message" -ForegroundColor Green }
        'WARN'  { Write-Host "$prefix $Message" -ForegroundColor Yellow }
        'ERROR' { Write-Host "$prefix $Message" -ForegroundColor Red }
        default { Write-Host "$prefix $Message" -ForegroundColor Cyan }
    }
}

function Write-Text {
    param(
        [Parameter(Mandatory)][string]$Text,
        [string]$Color = 'Gray'
    )
    # Always print text, even in JSON-RPC mode, so the user sees progress
    if ($NoColor) {
        Write-Host $Text
    } else {
        Write-Host $Text -ForegroundColor $Color
    }
}

# Run a scriptblock with a timeout so that slow network tests cannot hang the whole script
function Invoke-WithTimeout {
    param(
        [Parameter(Mandatory)][ScriptBlock]$ScriptBlock,
        [int]$TimeoutSeconds = 10,
        [string]$Description = ''
    )
    try {
        $job = Start-Job -ScriptBlock $ScriptBlock
    } catch {
        $errMsg = $_.Exception.Message
        Write-Status WARN ("Failed to start job for {0}: {1}" -f $Description, $errMsg)
        return $null
    }

    $completed = $null
    try {
        $completed = Wait-Job -Job $job -Timeout $TimeoutSeconds
    } catch {
        $errMsg = $_.Exception.Message
        Write-Status WARN ("Wait-Job failed for {0}: {1}" -f $Description, $errMsg)
    }

    if (-not $completed) {
        Write-Status WARN "Timed out after ${TimeoutSeconds}s: $Description"
        try { Stop-Job -Job $job -Force -ErrorAction SilentlyContinue | Out-Null } catch {}
        try { Remove-Job -Job $job -Force -ErrorAction SilentlyContinue | Out-Null } catch {}
        return $null
    }

    $result = $null
    try {
        $result = Receive-Job -Job $job -ErrorAction SilentlyContinue
    } catch {
        $errMsg = $_.Exception.Message
        Write-Status WARN ("Receive-Job failed for {0}: {1}" -f $Description, $errMsg)
    }
    try { Remove-Job -Job $job -Force -ErrorAction SilentlyContinue | Out-Null } catch {}
    return $result
}

function Get-DefaultWslDistro {
    try {
        $out = & wsl -l -v 2>$null
        if (-not $out) { return $null }
        foreach ($line in $out) {
            if ($line.TrimStart().StartsWith('*')) {
                $clean = $line.TrimStart('*').Trim()
                $parts = $clean -split '\s+' | Where-Object { $_ -ne '' }
                if ($parts.Count -gt 0) { return $parts[0] }
            }
        }
    } catch {
        $errMsg = $_.Exception.Message
        Write-Status WARN "Failed to query default WSL distro: $errMsg"
    }
    return $null
}

function Test-BasicLocalhost {
    $result = [ordered]@{}
    Write-Text "`n=== Phase: Windows basic localhost tests ===" 'Cyan'
    try {
        $ping4 = Test-Connection -ComputerName 127.0.0.1 -Count 2 -ErrorAction Stop
        $avgMs = [math]::Round(($ping4 | Measure-Object -Property ResponseTime -Average).Average,2)
        Write-Status OK "Ping 127.0.0.1 succeeded (avg: $avgMs ms)"
        $result.Ping127 = [ordered]@{
            success = $true
            avgMs   = $avgMs
        }
    } catch {
        $errMsg = $_.Exception.Message
        Write-Status ERROR "Ping 127.0.0.1 failed: $errMsg"
        $result.Ping127 = [ordered]@{
            success = $false
            error   = $errMsg
        }
    }
    try {
        $tn = Invoke-WithTimeout -TimeoutSeconds 5 -Description 'Test-NetConnection localhost' -ScriptBlock {
            Test-NetConnection -ComputerName localhost -InformationLevel Detailed
        }
        if ($null -eq $tn) {
            Write-Status WARN "Test-NetConnection localhost did not complete within timeout or returned no result."
            $result.TestNetConnection = [ordered]@{
                success        = $false
                timeoutSeconds = 5
            }
        } else {
            if ($tn.PingSucceeded) {
                Write-Status OK "Test-NetConnection localhost ping succeeded (Address: $($tn.RemoteAddress))"
            } else {
                Write-Status WARN "Test-NetConnection localhost ping failed"
            }
            if (-not $JsonRpc) {
                $tn | Format-List * | Out-String | Write-Host
            }
            $result.TestNetConnection = [ordered]@{
                success        = [bool]$tn.PingSucceeded
                remoteAddress  = $tn.RemoteAddress
                interfaceAlias = $tn.InterfaceAlias
            }
        }
    } catch {
        $errMsg = $_.Exception.Message
        Write-Status WARN "Test-NetConnection localhost threw: $errMsg"
        $result.TestNetConnection = [ordered]@{
            success = $false
            error   = $errMsg
        }
    }
    if ($JsonRpc) {
        if (-not $script:DiagResults) { $script:DiagResults = @{} }
        $script:DiagResults["WindowsBasic"] = $result
    }
}

function Test-WindowsNetwork {
    $result = [ordered]@{}
    Write-Text "`n=== Phase: Windows network & adapters (including vEthernet WSL) ===" 'Cyan'
    try {
        $adapters = Get-NetAdapter | Sort-Object ifIndex
        if (-not $JsonRpc) {
            Write-Text "-- Net adapters --"
            $adapters | Format-Table -Auto Name, InterfaceDescription, Status, ifIndex
        }
        $result.NetAdapters = $adapters | Select-Object Name, InterfaceDescription, Status, ifIndex
    } catch {
        $errMsg = $_.Exception.Message
        Write-Status WARN "Get-NetAdapter failed: $errMsg"
        $result.NetAdaptersError = $errMsg
    }

    # vEthernet (WSL) details
    try {
        $wslAdapter = Get-NetAdapter -Name 'vEthernet (WSL)' -ErrorAction SilentlyContinue
        if (-not $wslAdapter) {
            Write-Status WARN "No 'vEthernet (WSL)' adapter found. WSL2 NAT networking may not be initialized."
            $result.VEthernetWSL = [ordered]@{
                found = $false
            }
        } else {
            Write-Status INFO "Found vEthernet (WSL) adapter, Status: $($wslAdapter.Status)"
            $wslInfo = [ordered]@{
                found  = $true
                status = $wslAdapter.Status
            }
            $wslIPv4 = Get-NetIPAddress -InterfaceAlias 'vEthernet (WSL)' -AddressFamily IPv4 -ErrorAction SilentlyContinue
            if (-not $wslIPv4) {
                Write-Status WARN "vEthernet (WSL) has no IPv4 address. WSL2 networking will be broken."
                $wslInfo.IPv4 = @()
            } else {
                $wslInfo.IPv4 = @()
                foreach ($ip in $wslIPv4) {
                    Write-Status INFO "vEthernet (WSL) IPv4: $($ip.IPAddress)/$($ip.PrefixLength)"
                    $entry = [ordered]@{
                        ipAddress   = $ip.IPAddress
                        prefixLength = $ip.PrefixLength
                        isApipa     = $false
                        is172       = $false
                    }
                    if ($ip.IPAddress -like '169.254.*') {
                        Write-Status WARN "vEthernet (WSL) is using APIPA (169.254.x.x). ICS/NAT DHCP likely failed."
                        $entry.isApipa = $true
                    } elseif ($ip.IPAddress -like '172.*') {
                        Write-Status OK "vEthernet (WSL) has a private 172.x address, which is typical for healthy WSL2 NAT."
                        $entry.is172 = $true
                    }
                    $wslInfo.IPv4 += [pscustomobject]$entry
                }
            }

            try {
                $wslRoute = Get-NetRoute -InterfaceAlias 'vEthernet (WSL)' -DestinationPrefix '0.0.0.0/0' -ErrorAction SilentlyContinue
                if ($wslRoute) {
                    Write-Status OK "Default route exists on vEthernet (WSL): NextHop $($wslRoute.NextHop)"
                    $wslInfo.DefaultRoute = [ordered]@{
                        present = $true
                        nextHop = $wslRoute.NextHop
                    }
                } else {
                    Write-Status WARN "No default IPv4 route on vEthernet (WSL). WSL2 traffic may not be NATed correctly."
                    $wslInfo.DefaultRoute = [ordered]@{
                        present = $false
                    }
                }
            } catch {
                $errMsg = $_.Exception.Message
                Write-Status WARN "Get-NetRoute for vEthernet (WSL) failed: $errMsg"
                $wslInfo.DefaultRouteError = $errMsg
            }
            $result.VEthernetWSL = $wslInfo
        }
    } catch {
        $errMsg = $_.Exception.Message
        Write-Status WARN "Failed while inspecting vEthernet (WSL): $errMsg"
        $result.VEthernetWSLError = $errMsg
    }
    if ($JsonRpc) {
        if (-not $script:DiagResults) { $script:DiagResults = @{} }
        $script:DiagResults["WindowsNetwork"] = $result
    }
}

function Test-Services {
    $result = [ordered]@{}
    Write-Text "`n=== Phase: Windows services (ICS & WSL) ===" 'Cyan'
    $serviceNames = 'SharedAccess','WslService','LxssManager'
    try {
        $svcs = Get-Service -Name $serviceNames -ErrorAction SilentlyContinue
        if (-not $svcs) {
            Write-Status WARN "Could not query one or more of services: $($serviceNames -join ', ')."
            $result.ServicesError = "Could not query services."
        } else {
            if (-not $JsonRpc) {
                $svcs | Format-Table -Auto Name, Status, StartType
            }
            $result.Services = @()
            foreach ($svc in $svcs) {
                $result.Services += [pscustomobject]@{
                    Name      = $svc.Name
                    Status    = $svc.Status
                    StartType = $svc.StartType
                }
                switch ($svc.Name) {
                    'SharedAccess' {
                        if ($svc.Status -ne 'Running') {
                            Write-Status WARN "Internet Connection Sharing (SharedAccess) is $($svc.Status). WSL2 NAT may not work."
                        } else {
                            Write-Status OK "SharedAccess (ICS) is running."
                        }
                    }
                    'WslService' {
                        if ($svc.Status -ne 'Running') {
                            Write-Status WARN "WslService is $($svc.Status). WSL features may be degraded."
                        } else {
                            Write-Status OK "WslService is running."
                        }
                    }
                    'LxssManager' {
                        if ($svc.Status -ne 'Running') {
                            Write-Status WARN "LxssManager is $($svc.Status). WSL instances may not start properly."
                        } else {
                            Write-Status OK "LxssManager is running."
                        }
                    }
                }
            }
        }
    } catch {
        $errMsg = $_.Exception.Message
        Write-Status WARN "Get-Service failed: $errMsg"
        $result.ServicesError = $errMsg
    }
    if ($JsonRpc) {
        if (-not $script:DiagResults) { $script:DiagResults = @{} }
        $script:DiagResults["Services"] = $result
    }
}

function Test-WSLStatus {
    $result = [ordered]@{}
    Write-Text "`n=== Phase: WSL global status ===" 'Cyan'
    try {
        if (-not $JsonRpc) {
            Write-Text "-- wsl --status --"
        }
        $statusOut = & wsl --status 2>&1
        if (-not $JsonRpc) {
            $statusOut | Write-Host
        }
        $result.Status = $statusOut -join "`n"
    } catch {
        $errMsg = $_.Exception.Message
        Write-Status WARN "wsl --status failed: $errMsg"
        $result.StatusError = $errMsg
    }
    try {
        if (-not $JsonRpc) {
            Write-Text "`n-- wsl -l -v --"
        }
        $listOut = & wsl -l -v 2>&1
        if (-not $JsonRpc) {
            $listOut | Write-Host
        }
        $result.List = $listOut -join "`n"
    } catch {
        $errMsg = $_.Exception.Message
        Write-Status WARN "wsl -l -v failed: $errMsg"
        $result.ListError = $errMsg
    }
    if ($JsonRpc) {
        if (-not $script:DiagResults) { $script:DiagResults = @{} }
        $script:DiagResults["WSLStatus"] = $result
    }
}

function Test-WSLInside {
    $result = [ordered]@{}
    Write-Text "`n=== Phase: Inside default WSL distro (network & localhost) ===" 'Cyan'
    $distro = Get-DefaultWslDistro
    if (-not $distro) {
        Write-Status WARN "No default WSL distro detected. Skipping inside-WSL tests."
        $result.Error = "No default WSL distro detected"
        if ($JsonRpc) { $script:DiagResults["WSLInside"] = $result }
        return
    }
    Write-Status INFO "Using default WSL distro: $distro"
    $result.Distro = $distro

    # ip addr
    try {
        if (-not $JsonRpc) {
            Write-Text "`n-- ip addr (inside $distro) --"
        }
        $ipAddrOut = & wsl -d $distro -- ip addr 2>&1
        if (-not $JsonRpc) {
            $ipAddrOut | Write-Host
        }
        $result.IpAddr = $ipAddrOut -join "`n"
        if ($ipAddrOut -notmatch 'inet .*eth0') {
            Write-Status WARN "No 'inet' address on eth0 detected inside $distro. External networking is likely broken."
            $result.Eth0HasInet = $false
        } else {
            Write-Status OK "Found inet address on eth0 inside $distro."
            $result.Eth0HasInet = $true
        }
    } catch {
        $errMsg = $_.Exception.Message
        Write-Status WARN "Failed to run 'ip addr' inside ${distro}: $errMsg"
        $result.IpAddrError = $errMsg
    }

    # ip route
    try {
        if (-not $JsonRpc) {
            Write-Text "`n-- ip route (inside $distro) --"
        }
        $routes = & wsl -d $distro -- ip route 2>&1
        if (-not $JsonRpc) {
            $routes | Write-Host
        }
        $result.Routes = $routes -join "`n"
        if ($routes -notmatch 'default ') {
            Write-Status WARN "No default route inside $distro. It will not reach external networks."
            $result.HasDefaultRoute = $false
        } else {
            Write-Status OK "Default route present inside $distro."
            $result.HasDefaultRoute = $true
        }
    } catch {
        $errMsg = $_.Exception.Message
        Write-Status WARN "Failed to run 'ip route' inside ${distro}: $errMsg"
        $result.RoutesError = $errMsg
    }

    # Loopback & external connectivity tests
    try {
        if (-not $JsonRpc) {
            Write-Text "`n-- ping tests (inside $distro) --"
        }
        $pingLocal = & wsl -d $distro -- sh -lc "ping -c 2 localhost || echo '___PING_LOCALHOST_FAILED___'" 2>&1
        if (-not $JsonRpc) {
            $pingLocal | Write-Host
        }
        $result.PingLocalhostRaw = $pingLocal -join "`n"
        if ($pingLocal -match '___PING_LOCALHOST_FAILED___') {
            Write-Status WARN "Ping localhost inside $distro failed. Loopback may be broken."
            $result.PingLocalhost = $false
        } else {
            Write-Status OK "Ping localhost inside $distro succeeded."
            $result.PingLocalhost = $true
        }

        $pingLoop = & wsl -d $distro -- sh -lc "ping -c 2 127.0.0.1 || echo '___PING_127_FAILED___'" 2>&1
        if (-not $JsonRpc) {
            $pingLoop | Write-Host
        }
        $result.Ping127Raw = $pingLoop -join "`n"
        if ($pingLoop -match '___PING_127_FAILED___') {
            Write-Status WARN "Ping 127.0.0.1 inside $distro failed. Loopback may be broken."
            $result.Ping127 = $false
        } else {
            Write-Status OK "Ping 127.0.0.1 inside $distro succeeded."
            $result.Ping127 = $true
        }

        $pingExternal = & wsl -d $distro -- sh -lc "ping -c 2 8.8.8.8 || echo '___PING_8_8_8_8_FAILED___'" 2>&1
        if (-not $JsonRpc) {
            $pingExternal | Write-Host
        }
        $result.Ping8888Raw = $pingExternal -join "`n"
        if ($pingExternal -match '___PING_8_8_8_8_FAILED___') {
            Write-Status WARN "Ping 8.8.8.8 from inside $distro failed. External connectivity is broken."
            $result.Ping8888 = $false
        } else {
            Write-Status OK "Ping 8.8.8.8 from inside $distro succeeded."
            $result.Ping8888 = $true
        }

        $pingDnsName = & wsl -d $distro -- sh -lc "ping -c 2 google.com || echo '___PING_GOOGLE_FAILED___'" 2>&1
        if (-not $JsonRpc) {
            $pingDnsName | Write-Host
        }
        $result.PingGoogleRaw = $pingDnsName -join "`n"
        if ($pingDnsName -match '___PING_GOOGLE_FAILED___') {
            Write-Status WARN "Ping google.com from inside $distro failed. DNS or external connectivity is broken."
            $result.PingGoogle = $false
        } else {
            Write-Status OK "Ping google.com from inside $distro succeeded."
            $result.PingGoogle = $true
        }

        $hostsLocal = & wsl -d $distro -- getent hosts localhost 2>&1
        if (-not $JsonRpc) {
            Write-Text "`n-- getent hosts localhost (inside $distro) --"
            $hostsLocal | Write-Host
        }
        $result.GetentLocalhost = $hostsLocal -join "`n"
        if ($hostsLocal -match '127.0.0.1') {
            Write-Status OK "getent hosts localhost resolves to 127.0.0.1 inside $distro."
            $result.GetentLocalhostHas127 = $true
        } else {
            Write-Status WARN "getent hosts localhost does not show 127.0.0.1 inside $distro."
            $result.GetentLocalhostHas127 = $false
        }
    } catch {
        $errMsg = $_.Exception.Message
        Write-Status WARN "Ping/getent tests inside ${distro} failed: $errMsg"
        $result.PingTestsError = $errMsg
    }

    # resolv.conf
    try {
        if (-not $JsonRpc) {
            Write-Text "`n-- /etc/resolv.conf (inside $distro) --"
        }
        $resolv = & wsl -d $distro -- cat /etc/resolv.conf 2>&1
        if (-not $JsonRpc) {
            $resolv | Write-Host
        }
        $result.ResolvConf = $resolv -join "`n"
        if ($resolv -match '127\.0\.0\.1|127\.0\.1\.1|127\.0\.2\.2|127\.0\.2\.3') {
            Write-Status WARN "resolv.conf inside $distro uses 127.x DNS, which may be problematic with VPN/WARP setups."
            $result.ResolvUsesLocal127 = $true
        } else {
            Write-Status INFO "resolv.conf inside $distro does not use local 127.x DNS; likely pointing to external DNS."
            $result.ResolvUsesLocal127 = $false
        }
    } catch {
        $errMsg = $_.Exception.Message
        Write-Status WARN "Failed to read /etc/resolv.conf inside ${distro}: $errMsg"
        $result.ResolvError = $errMsg
    }

    # /etc/wsl.conf
    try {
        if (-not $JsonRpc) {
            Write-Text "`n-- /etc/wsl.conf (inside $distro, if present) --"
        }
        $wslConf = & wsl -d $distro -- sh -lc "if [ -f /etc/wsl.conf ]; then cat /etc/wsl.conf; else echo 'No /etc/wsl.conf'; fi" 2>&1
        if (-not $JsonRpc) {
            $wslConf | Write-Host
        }
        $result.WslConf = $wslConf -join "`n"
    } catch {
        $errMsg = $_.Exception.Message
        Write-Status WARN "Failed to inspect /etc/wsl.conf inside ${distro}: $errMsg"
        $result.WslConfError = $errMsg
    }

    if ($JsonRpc) {
        if (-not $script:DiagResults) { $script:DiagResults = @{} }
        $script:DiagResults["WSLInside"] = $result
    }
}

# Phase selection
$runAll      = $Phase -contains 'All'
$runQuick    = $Phase -contains 'Quick'
$runWindows  = $runAll -or $runQuick -or ($Phase -contains 'Windows')
$runWSL      = $runAll -or $runQuick -or ($Phase -contains 'WSL')
$runServices = $runAll -or ($Phase -contains 'Services')
$runInside   = $runAll -or $runQuick -or ($Phase -contains 'WSLInside')

# Global containers for JSON-RPC diagnostics
$script:DiagResults = @{}
$script:StatusLog   = @()

if (-not $JsonRpc) {
    Write-Text "Test-WslLocalhost.ps1 - comprehensive localhost/WSL diagnostics" 'Cyan'
    Write-Text "Selected phases: $($Phase -join ', ')" 'Cyan'
}

if ($runWindows) { Test-BasicLocalhost; Test-WindowsNetwork }
if ($runServices) { Test-Services }
if ($runWSL) { Test-WSLStatus }
if ($runInside) { Test-WSLInside }

if ($JsonRpc) {
    $rpc = [ordered]@{
        jsonrpc = '2.0'
        id      = $JsonRpcId
        result  = [ordered]@{
            method      = $JsonRpcMethod
            phase       = $Phase
            diagnostics = $script:DiagResults
            messages    = $script:StatusLog
        }
    }
    $rpc | ConvertTo-Json -Depth 8
} else {
    Write-Text "`n=== Diagnostics complete. Review WARN/ERROR entries above for actionable issues. ===" 'Cyan'
}
