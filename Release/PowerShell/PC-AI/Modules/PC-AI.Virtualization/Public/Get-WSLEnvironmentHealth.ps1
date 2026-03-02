#Requires -Version 5.1
<#
.SYNOPSIS
    Comprehensive health check for WSL, Docker, and VSock bridges

.DESCRIPTION
    Performs comprehensive health verification of the WSL development environment:
    - WSL distro status and systemd health
    - Docker service and daemon connectivity
    - Hyper-V socket bridges (socat processes)
    - RAG Redis backend and Search module
    - Network connectivity from WSL
    - Startup task configuration

    Based on patterns from wsl-docker-health-check.ps1 and wsl-vsock-bridge-configured.sh

.PARAMETER AutoRecover
    Attempt automatic recovery of failed services

.PARAMETER Quick
    Skip network connectivity tests for faster results

.PARAMETER Distribution
    Specific WSL distribution to test (default: Ubuntu)

.EXAMPLE
    Get-WSLEnvironmentHealth
    Run comprehensive health check

.EXAMPLE
    Get-WSLEnvironmentHealth -AutoRecover
    Run health check and attempt to fix issues

.EXAMPLE
    Get-WSLEnvironmentHealth -Quick
    Run quick health check without network tests

.OUTPUTS
    PSCustomObject with comprehensive health status
#>
function Get-WSLEnvironmentHealth {
    [CmdletBinding()]
    [OutputType([PSCustomObject])]
    param(
        [Parameter()]
        [switch]$AutoRecover,

        [Parameter()]
        [switch]$Quick,

        [Parameter()]
        [string]$Distribution = "Ubuntu"
    )

    $result = [PSCustomObject]@{
        Timestamp       = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
        WSL             = $null
        Docker          = $null
        VSockBridges    = $null
        RAGRedis        = $null
        Network         = $null
        StartupTask     = $null
        Issues          = @()
        RecoveryActions = @()
        OverallHealth   = 'Unknown'
    }

    Write-Host "[*] Starting WSL Environment Health Check..." -ForegroundColor Cyan

    # Check WSL Status
    Write-Host "[*] Checking WSL status..." -ForegroundColor Yellow
    $result.WSL = Test-WSLHealth -Distribution $Distribution -AutoRecover:$AutoRecover

    if ($result.WSL.Status -ne 'OK') {
        $result.Issues += $result.WSL.Issues
        if ($result.WSL.RecoveryAction) {
            $result.RecoveryActions += $result.WSL.RecoveryAction
        }
    }

    # Check Docker Status
    Write-Host "[*] Checking Docker status..." -ForegroundColor Yellow
    $result.Docker = Test-DockerHealth -Distribution $Distribution -AutoRecover:$AutoRecover

    if ($result.Docker.Status -ne 'OK') {
        $result.Issues += $result.Docker.Issues
        if ($result.Docker.RecoveryAction) {
            $result.RecoveryActions += $result.Docker.RecoveryAction
        }
    }

    # Check VSock Bridges
    Write-Host "[*] Checking VSock bridges..." -ForegroundColor Yellow
    $result.VSockBridges = Test-VSockBridgeHealth -Distribution $Distribution -AutoRecover:$AutoRecover

    if ($result.VSockBridges.Status -ne 'OK') {
        $result.Issues += $result.VSockBridges.Issues
    }

    # Check RAG Redis Status
    Write-Host "[*] Checking RAG Redis status..." -ForegroundColor Yellow
    $result.RAGRedis = Test-RAGRedisHealth -AutoRecover:$AutoRecover

    if ($result.RAGRedis.Status -ne 'OK') {
        $result.Issues += $result.RAGRedis.Issues
        if ($result.RAGRedis.RecoveryAction) {
            $result.RecoveryActions += $result.RAGRedis.RecoveryAction
        }
    }

    # Check Network (unless Quick mode)
    if (-not $Quick) {
        Write-Host "[*] Checking network connectivity..." -ForegroundColor Yellow
        $result.Network = Test-WSLNetworkHealth -Distribution $Distribution
        if ($result.Network.Status -ne 'OK') {
            $result.Issues += $result.Network.Issues
        }
    }
    else {
        $result.Network = [PSCustomObject]@{
            Status = 'Skipped'
            Reason = 'Quick mode enabled'
        }
    }

    # Check Startup Task
    Write-Host "[*] Checking startup task..." -ForegroundColor Yellow
    $result.StartupTask = Test-WSLStartupTask

    # Determine overall health
    $criticalCount = ($result.Issues | Where-Object { $_.Severity -eq 'Critical' }).Count
    $warningCount = ($result.Issues | Where-Object { $_.Severity -eq 'Warning' }).Count

    if ($criticalCount -gt 0) {
        $result.OverallHealth = 'Critical'
    }
    elseif ($warningCount -gt 0) {
        $result.OverallHealth = 'Warning'
    }
    elseif ($result.Issues.Count -eq 0) {
        $result.OverallHealth = 'Healthy'
    }
    else {
        $result.OverallHealth = 'Degraded'
    }

    # Display summary
    Write-Host ""
    Write-Host "== WSL Environment Health Summary ==" -ForegroundColor Cyan
    Write-Host "  WSL: $($result.WSL.Status)" -ForegroundColor $(Get-StatusColor $result.WSL.Status)
    Write-Host "  Docker: $($result.Docker.Status)" -ForegroundColor $(Get-StatusColor $result.Docker.Status)
    Write-Host "  VSock Bridges: $($result.VSockBridges.Status)" -ForegroundColor $(Get-StatusColor $result.VSockBridges.Status)
    Write-Host "  RAG Redis: $($result.RAGRedis.Status)" -ForegroundColor $(Get-StatusColor $result.RAGRedis.Status)

    if (-not $Quick) {
        Write-Host "  Network: $($result.Network.Status)" -ForegroundColor $(Get-StatusColor $result.Network.Status)
    }

    Write-Host "  Startup Task: $($result.StartupTask.Status)" -ForegroundColor $(Get-StatusColor $result.StartupTask.Status)
    Write-Host ""
    Write-Host "  Overall: $($result.OverallHealth)" -ForegroundColor $(Get-StatusColor $result.OverallHealth)

    if ($result.Issues.Count -gt 0) {
        Write-Host ""
        Write-Host "== Issues Found ==" -ForegroundColor Yellow
        foreach ($issue in $result.Issues) {
            $issueColor = if ($issue.Severity -eq 'Critical') { 'Red' } else { 'Yellow' }
            Write-Host "  [$($issue.Severity)] $($issue.Component): $($issue.Message)" -ForegroundColor $issueColor
        }
    }

    if ($result.RecoveryActions.Count -gt 0 -and -not $AutoRecover) {
        Write-Host ""
        Write-Host "Run with -AutoRecover to attempt automatic fixes" -ForegroundColor Cyan
    }

    return $result
}

# Helper function to test WSL health
function Test-WSLHealth {
    param(
        [string]$Distribution,
        [switch]$AutoRecover
    )

    $health = [PSCustomObject]@{
        Status         = 'Unknown'
        DistroRunning  = $false
        SystemdStatus  = $null
        Issues         = @()
        RecoveryAction = $null
    }

    try {
        # Check WSL list
        $wslList = wsl -l -v 2>&1
        if ($LASTEXITCODE -ne 0) {
            $health.Status = 'Error'
            $health.Issues += [PSCustomObject]@{
                Component = 'WSL'
                Severity  = 'Critical'
                Message   = 'WSL command failed'
            }
            return $health
        }

        # Check if distribution is running
        $distroRunning = $wslList | Select-String "$Distribution.*Running"
        if (-not $distroRunning) {
            $health.Issues += [PSCustomObject]@{
                Component = 'WSL'
                Severity  = 'Warning'
                Message   = "$Distribution distro not running"
            }

            if ($AutoRecover) {
                Write-Host "    Attempting to start $Distribution..." -ForegroundColor Yellow
                wsl -d $Distribution -- echo "Started" 2>&1 | Out-Null
                Start-Sleep -Seconds 3

                $distroRunning = (wsl -l -v 2>&1) | Select-String "$Distribution.*Running"
                if ($distroRunning) {
                    Write-Host "    [+] $Distribution started successfully" -ForegroundColor Green
                    $health.DistroRunning = $true
                    $health.RecoveryAction = "Started $Distribution"
                }
            }
        }
        else {
            $health.DistroRunning = $true
        }

        # Check systemd
        if ($health.DistroRunning) {
            $systemdStatus = wsl -d $Distribution -- systemctl is-system-running 2>&1
            $health.SystemdStatus = $systemdStatus.Trim()

            if ($systemdStatus -notmatch 'running|degraded') {
                $health.Issues += [PSCustomObject]@{
                    Component = 'WSL'
                    Severity  = 'Warning'
                    Message   = "Systemd status: $systemdStatus"
                }
            }
        }

        $health.Status = if ($health.Issues.Count -eq 0) { 'OK' } elseif ($health.DistroRunning) { 'Warning' } else { 'Error' }
    }
    catch {
        $health.Status = 'Error'
        $health.Issues += [PSCustomObject]@{
            Component = 'WSL'
            Severity  = 'Critical'
            Message   = $_.Exception.Message
        }
    }

    return $health
}

# Helper function to test Docker health
function Test-DockerHealth {
    param(
        [string]$Distribution,
        [switch]$AutoRecover
    )

    $health = [PSCustomObject]@{
        Status           = 'Unknown'
        ServiceActive    = $false
        DaemonResponding = $false
        ContainerCount   = 0
        Issues           = @()
        RecoveryAction   = $null
    }

    try {
        # Prefer Windows Docker Desktop engine if available
        $dockerCmd = Get-Command docker -ErrorAction SilentlyContinue
        if (-not $dockerCmd) {
            $dockerBin = 'C:\Program Files\Docker\Docker\resources\bin'
            if (Test-Path (Join-Path $dockerBin 'docker.exe')) {
                $env:Path = "$dockerBin;$env:Path"
                $dockerCmd = Get-Command docker -ErrorAction SilentlyContinue
            }
        }

        if ($dockerCmd) {
            $dockerInfo = docker info 2>&1
            if ($LASTEXITCODE -eq 0) {
                $health.ServiceActive = $true
                $health.DaemonResponding = $true
                $containers = docker ps -q 2>&1
                if ($LASTEXITCODE -eq 0 -and $containers) {
                    $health.ContainerCount = ($containers | Measure-Object -Line).Lines
                }
                $health.Status = 'OK'
                return $health
            }
        }

        # Check Docker service in WSL
        $dockerActive = wsl -d $Distribution -- systemctl is-active docker 2>&1
        $dockerActive = $dockerActive.Trim()

        if ($dockerActive -ne 'active') {
            $health.Issues += [PSCustomObject]@{
                Component = 'Docker'
                Severity  = 'Warning'
                Message   = "Docker service not active ($dockerActive)"
            }

            if ($AutoRecover) {
                Write-Host "    Attempting to start Docker..." -ForegroundColor Yellow
                wsl -d $Distribution -- sudo systemctl start docker 2>&1 | Out-Null
                Start-Sleep -Seconds 5

                $dockerActive = (wsl -d $Distribution -- systemctl is-active docker 2>&1).Trim()
                if ($dockerActive -eq 'active') {
                    Write-Host "    [+] Docker started successfully" -ForegroundColor Green
                    $health.ServiceActive = $true
                    $health.RecoveryAction = "Started Docker service"
                }
            }
        }
        else {
            $health.ServiceActive = $true
        }

        # Check Docker daemon connectivity
        if ($health.ServiceActive) {
            $dockerInfo = wsl -d $Distribution -- docker info 2>&1
            if ($LASTEXITCODE -eq 0) {
                $health.DaemonResponding = $true

                # Get container count
                $containers = wsl -d $Distribution -- docker ps -q 2>&1
                if ($LASTEXITCODE -eq 0 -and $containers) {
                    $health.ContainerCount = ($containers | Measure-Object -Line).Lines
                }
            }
            else {
                $health.Issues += [PSCustomObject]@{
                    Component = 'Docker'
                    Severity  = 'Warning'
                    Message   = 'Docker daemon not responding'
                }
            }
        }

        $health.Status = if ($health.Issues.Count -eq 0) { 'OK' } elseif ($health.ServiceActive) { 'Warning' } else { 'Error' }
    }
    catch {
        $health.Status = 'Error'
        $health.Issues += [PSCustomObject]@{
            Component = 'Docker'
            Severity  = 'Critical'
            Message   = $_.Exception.Message
        }
    }

    return $health
}

# Helper function to test VSock bridge health
function Test-VSockBridgeHealth {
    param(
        [string]$Distribution,
        [switch]$AutoRecover
    )

    $health = [PSCustomObject]@{
        Status         = 'Unknown'
        BridgeService  = $null
        SocatProcesses = 0
        Bridges        = @()
        Issues         = @()
    }

    try {
        # Check for system-level bridge service (PC_AI)
        $bridgeActive = wsl -d $Distribution -- systemctl is-active pcai-vsock-bridge 2>&1
        $health.BridgeService = $bridgeActive.Trim()

        # Also check user-level service
        if ($health.BridgeService -notmatch 'active|exited') {
            $bridgeActive = wsl -d $Distribution -- systemctl is-active wsl-vsock-bridge 2>&1
            if ($bridgeActive -match 'active|exited') {
                $health.BridgeService = $bridgeActive.Trim()
            }
            else {
                $bridgeActive = wsl -d $Distribution -- systemctl --user is-active hyper-v-socket-bridges 2>&1
                $health.BridgeService = "user: $($bridgeActive.Trim())"
            }
        }

        # Count socat processes (the actual bridges)
        $socatCount = wsl -d $Distribution -- pgrep -c socat 2>&1
        if ($LASTEXITCODE -eq 0) {
            $health.SocatProcesses = [int]$socatCount.Trim()
        }

        # Check common bridge ports
        $commonPorts = @(8000, 8001, 8002, 11434, 1234, 3001, 3002, 18000, 18001, 18002)
        foreach ($port in $commonPorts) {
            $listening = wsl -d $Distribution -- ss -tln "sport = :$port" 2>&1 | Select-String "LISTEN"
            if ($listening) {
                $health.Bridges += [PSCustomObject]@{
                    Port   = $port
                    Status = 'Listening'
                }
            }
        }

        # Determine status
        if ($health.SocatProcesses -eq 0 -and $health.Bridges.Count -eq 0) {
            $health.Status = 'Info'
            $health.Issues += [PSCustomObject]@{
                Component = 'VSockBridges'
                Severity  = 'Info'
                Message   = 'No VSock bridges detected (may not be configured)'
            }
        }
        elseif ($health.SocatProcesses -gt 0) {
            $health.Status = 'OK'
        }
        else {
            $health.Status = 'Warning'
        }
    }
    catch {
        $health.Status = 'Unknown'
    }

    return $health
}

# Helper function to test RAG Redis health
function Test-RAGRedisHealth {
    param(
        [switch]$AutoRecover,
        [string]$ComposePath = "C:\codedev\llm\rag-redis\docker-compose.yml"
    )

    $health = [PSCustomObject]@{
        Status         = 'Unknown'
        ContainerID    = $null
        ContainerStatus = $null
        RedisPing      = $false
        SearchModule   = $false
        Issues         = @()
        RecoveryAction = $null
    }

    $containerName = "rag-redis-backend"

    try {
        # Check if container exists and is running
        $container = docker ps --filter "name=$containerName" --format "{{.ID}}|{{.Status}}" 2>&1
        if ($LASTEXITCODE -ne 0 -or -not $container) {
            $health.Issues += [PSCustomObject]@{
                Component = 'RAGRedis'
                Severity  = 'Warning'
                Message   = "RAG-Redis container not running"
            }

            if ($AutoRecover -and (Test-Path $ComposePath)) {
                Write-Host "    Attempting to start RAG-Redis..." -ForegroundColor Yellow
                docker-compose -f $ComposePath up -d redis 2>&1 | Out-Null
                Start-Sleep -Seconds 10
                $container = docker ps --filter "name=$containerName" --format "{{.ID}}|{{.Status}}" 2>&1
                if ($container) {
                    Write-Host "    [+] RAG-Redis started successfully" -ForegroundColor Green
                    $health.RecoveryAction = "Started RAG-Redis container"
                }
            }
        }

        if ($container) {
            $parts = ($container | Out-String).Trim().Split('|')
            $health.ContainerID = $parts[0]
            $health.ContainerStatus = $parts[1]

            # Check Redis connectivity
            $pingTest = docker exec $health.ContainerID redis-cli ping 2>&1
            if (($pingTest | Out-String).Trim() -eq "PONG") {
                $health.RedisPing = $true

                # Check RediSearch module
                $moduleList = docker exec $health.ContainerID redis-cli module list 2>&1
                if ($moduleList -match "search") {
                    $health.SearchModule = $true
                }
                else {
                    $health.Issues += [PSCustomObject]@{
                        Component = 'RAGRedis'
                        Severity  = 'Warning'
                        Message   = 'RediSearch module not loaded'
                    }
                }
            }
            else {
                $health.Issues += [PSCustomObject]@{
                    Component = 'RAGRedis'
                    Severity  = 'Warning'
                    Message   = 'Redis not responding to PING'
                }
            }
        }

        $health.Status = if ($health.Issues.Count -eq 0) { 'OK' } elseif ($health.RedisPing) { 'Warning' } else { 'Error' }
    }
    catch {
        $health.Status = 'Error'
        $health.Issues += [PSCustomObject]@{
            Component = 'RAGRedis'
            Severity  = 'Critical'
            Message   = $_.Exception.Message
        }
    }

    return $health
}

# Helper function to test WSL network health
function Test-WSLNetworkHealth {
    param(
        [string]$Distribution
    )

    $health = [PSCustomObject]@{
        Status         = 'Unknown'
        DNSWorking     = $false
        InternetAccess = $false
        Issues         = @()
    }

    try {
        # Test DNS
        $dnsTest = wsl -d $Distribution -- nslookup google.com 2>&1
        if ($LASTEXITCODE -eq 0) {
            $health.DNSWorking = $true
        }
        else {
            $health.Issues += [PSCustomObject]@{
                Component = 'Network'
                Severity  = 'Critical'
                Message   = 'DNS resolution failed'
            }
        }

        # Test connectivity
        $pingTest = wsl -d $Distribution -- ping -c 1 -W 2 8.8.8.8 2>&1
        if ($LASTEXITCODE -eq 0) {
            $health.InternetAccess = $true
        }
        else {
            $health.Issues += [PSCustomObject]@{
                Component = 'Network'
                Severity  = 'Warning'
                Message   = 'External connectivity issues'
            }
        }

        $health.Status = if ($health.DNSWorking -and $health.InternetAccess) { 'OK' }
        elseif ($health.DNSWorking) { 'Warning' }
        else { 'Error' }
    }
    catch {
        $health.Status = 'Error'
        $health.Issues += [PSCustomObject]@{
            Component = 'Network'
            Severity  = 'Critical'
            Message   = $_.Exception.Message
        }
    }

    return $health
}

# Helper function to check startup task
function Test-WSLStartupTask {
    $health = [PSCustomObject]@{
        Status   = 'Unknown'
        TaskName = 'WSL-Docker-Startup'
        Exists   = $false
        Enabled  = $false
        LastRun  = $null
    }

    try {
        $task = Get-ScheduledTask -TaskName $health.TaskName -ErrorAction SilentlyContinue
        if ($task) {
            $health.Exists = $true
            $health.Enabled = ($task.State -ne 'Disabled')

            $taskInfo = Get-ScheduledTaskInfo -TaskName $health.TaskName -ErrorAction SilentlyContinue
            if ($taskInfo) {
                $health.LastRun = $taskInfo.LastRunTime
            }

            $health.Status = if ($health.Enabled) { 'OK' } else { 'Warning' }
        }
        else {
            $health.Status = 'Info'
        }
    }
    catch {
        $health.Status = 'Unknown'
    }

    return $health
}

# Helper function to get status color
function Get-StatusColor {
    param([string]$Status)

    switch ($Status) {
        'OK' { 'Green' }
        'Healthy' { 'Green' }
        'Warning' { 'Yellow' }
        'Degraded' { 'Yellow' }
        'Error' { 'Red' }
        'Critical' { 'Red' }
        'Info' { 'Cyan' }
        'Skipped' { 'Gray' }
        default { 'White' }
    }
}
