#Requires -Version 5.1
<#
.SYNOPSIS
    Initializes WSL, Docker, Podman, and container services with credential management at system startup.

.DESCRIPTION
    This script orchestrates the startup sequence for WSL-based container development environment.
    It performs the following phases:
    - Phase 0: System startup delay
    - Phase 1: Start LxssManager service
    - Phase 2: Initialize WSL distribution
    - Phase 3: Initialize DNS in WSL
    - Phase 4: Start Hyper-V socket bridges
    - Phase 5: Bitwarden unlock (automated login and vault unlock)
    - Phase 6: Credential sync (sync registry credentials from Bitwarden to wincred)
    - Phase 7: Start Docker Desktop
    - Phase 8: Wait for Docker engine readiness
    - Phase 9: Start RAG-Redis container
    - Phase 10: Ensure Podman machine is running

    Designed for Task Scheduler execution with -NoProfile flag.

.PARAMETER WslDistro
    The WSL distribution to use. Default: Ubuntu

.PARAMETER MaxRetries
    Maximum number of retry attempts for WSL initialization. Default: 10

.PARAMETER RetryDelay
    Delay in seconds between retry attempts. Default: 10

.PARAMETER StartupDelay
    Initial delay in seconds to allow system startup to complete. Default: 20

.PARAMETER DockerReadyRetries
    Maximum retries waiting for Docker engine. Default: 30

.PARAMETER DockerReadyDelay
    Delay in seconds between Docker readiness checks. Default: 5

.PARAMETER LogFile
    Path to the log file. Default: C:\Scripts\Startup\docker-wsl-startup.log

.PARAMETER MaxLogLines
    Maximum number of log lines to retain during rotation. Default: 2000

.PARAMETER SkipRAGRedis
    Skip starting the RAG-Redis container.

.PARAMETER SkipBitwarden
    Skip Bitwarden unlock and credential sync phases.

.PARAMETER SkipPodman
    Skip starting Podman machine.

.EXAMPLE
    .\Start-WSLDockerServices.ps1
    Runs the full startup sequence with default parameters.

.EXAMPLE
    .\Start-WSLDockerServices.ps1 -Verbose
    Runs with verbose output to console.

.EXAMPLE
    .\Start-WSLDockerServices.ps1 -WhatIf
    Shows what actions would be taken without executing them.

.EXAMPLE
    .\Start-WSLDockerServices.ps1 -WslDistro "Debian" -StartupDelay 30
    Uses Debian distribution with 30 second startup delay.

.NOTES
    Author: Claude Code Framework
    Version: 4.0
    Updated: 2026-01-25

    Changes in v4.0:
    - Added Bitwarden unlock phase (Phase 5)
    - Added credential sync phase (Phase 6)
    - Added Podman machine phase (Phase 10)
    - Renumbered Docker phases (7-9)

    For Task Scheduler, use:
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File "C:\Scripts\Startup\Start-WSLDockerServices.ps1"

.LINK
    https://docs.microsoft.com/en-us/windows/wsl/
#>

[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [Parameter()]
    [ValidateNotNullOrEmpty()]
    [string]$WslDistro = 'Ubuntu',

    [Parameter()]
    [ValidateRange(1, 100)]
    [int]$MaxRetries = 10,

    [Parameter()]
    [ValidateRange(1, 300)]
    [int]$RetryDelay = 10,

    [Parameter()]
    [ValidateRange(0, 300)]
    [int]$StartupDelay = 20,

    [Parameter()]
    [ValidateRange(1, 120)]
    [int]$DockerReadyRetries = 30,

    [Parameter()]
    [ValidateRange(1, 60)]
    [int]$DockerReadyDelay = 5,

    [Parameter()]
    [ValidateNotNullOrEmpty()]
    [string]$LogFile = 'C:\Scripts\Startup\docker-wsl-startup.log',

    [Parameter()]
    [ValidateRange(100, 100000)]
    [int]$MaxLogLines = 2000,

    [Parameter()]
    [switch]$SkipRAGRedis,

    [Parameter()]
    [switch]$SkipBitwarden,

    [Parameter()]
    [switch]$SkipPodman = $true
)

#region Script Configuration
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# Script-level configuration
$script:Config = @{
    DockerDesktopExe           = 'C:\Program Files\Docker\Docker\Docker Desktop.exe'
    DockerCliPath              = 'C:\Program Files\Docker\Docker\resources\bin'
    RAGRedisCompose            = 'C:\codedev\llm\rag-redis\docker-compose.yml'
    WslDnsInit                 = '/usr/local/sbin/wsl-dns-init'
    WslVsockBridge             = '/usr/local/sbin/wsl-vsock-bridge'
    InitBitwardenScript        = 'C:\Scripts\Startup\Initialize-BitwardenSession.ps1'
    SyncCredentialsScript      = 'C:\Scripts\Startup\Sync-RegistryCredentials.ps1'
    PodmanMachineName          = 'podman-machine-default'
}
#endregion

#region Logging Functions
function Invoke-LogRotation {
    <#
    .SYNOPSIS
        Rotates the log file, keeping only the last N lines.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$Path,

        [Parameter(Mandatory)]
        [int]$MaxLines
    )

    if (-not (Test-Path -Path $Path -PathType Leaf)) {
        return
    }

    try {
        $content = Get-Content -Path $Path -Tail $MaxLines -ErrorAction SilentlyContinue
        if ($content) {
            $content | Set-Content -Path $Path -Force
            Write-Verbose "Log rotated: kept last $MaxLines lines"
        }
    }
    catch {
        Write-Warning "Log rotation failed: $_"
    }
}

function Write-Log {
    <#
    .SYNOPSIS
        Writes a timestamped message to the log file and optionally to verbose output.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory, Position = 0)]
        [string]$Message,

        [Parameter()]
        [ValidateSet('INFO', 'SUCCESS', 'WARNING', 'ERROR')]
        [string]$Level = 'INFO'
    )

    $timestamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    $logEntry = "[$timestamp] [$Level] $Message"

    # Ensure log directory exists
    $logDir = Split-Path -Path $LogFile -Parent
    if (-not (Test-Path -Path $logDir)) {
        New-Item -Path $logDir -ItemType Directory -Force | Out-Null
    }

    # Append to log file
    Add-Content -Path $LogFile -Value $logEntry -ErrorAction SilentlyContinue

    # Output based on level
    switch ($Level) {
        'SUCCESS' { Write-Verbose $logEntry }
        'WARNING' { Write-Warning $Message }
        'ERROR'   { Write-Error $Message -ErrorAction Continue }
        default   { Write-Verbose $logEntry }
    }
}

function Write-LogSection {
    <#
    .SYNOPSIS
        Writes a section header to the log.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$Title
    )

    Write-Log ('=' * 50)
    Write-Log $Title
    Write-Log ('=' * 50)
}
#endregion

#region Environment Setup
function Initialize-DockerMcpMode {
    <#
    .SYNOPSIS
        Prepares the environment for Docker Desktop MCP mode.
    #>
    [CmdletBinding()]
    param()

    Write-Log 'Preparing Docker-MCP environment...'

    # Clear DOCKER_HOST to avoid pipe conflicts
    if ($env:DOCKER_HOST) {
        Write-Log "Clearing DOCKER_HOST ($env:DOCKER_HOST)" -Level WARNING
        Remove-Item Env:DOCKER_HOST -ErrorAction SilentlyContinue
    }

    # Ensure Docker CLI is available in PATH
    if (Test-Path (Join-Path $script:Config.DockerCliPath 'docker.exe')) {
        if ($env:Path -notmatch [regex]::Escape($script:Config.DockerCliPath)) {
            $env:Path = "$($script:Config.DockerCliPath);$env:Path"
            Write-Log "Added Docker CLI path for this session: $($script:Config.DockerCliPath)" -Level SUCCESS
        }
    }

    # Ensure WSL default version is 2
    try {
        $wslCmd = Get-Command wsl.exe -ErrorAction SilentlyContinue
        if ($wslCmd) {
            & wsl.exe --set-default-version 2 2>$null | Out-Null
            Write-Log 'WSL default version set to 2' -Level SUCCESS
        }
    }
    catch {
        Write-Log "Failed to set WSL default version: $_" -Level WARNING
    }
}
#endregion

#region Service Functions
function Start-LxssManager {
    <#
    .SYNOPSIS
        Ensures the LxssManager service is running.
    .OUTPUTS
        [bool] True if service is running, False otherwise.
    #>
    [CmdletBinding(SupportsShouldProcess)]
    [OutputType([bool])]
    param()

    $serviceCandidates = @('WSLService', 'LxssManager')
    $serviceName = $null

    foreach ($candidate in $serviceCandidates) {
        try {
            $service = Get-Service -Name $candidate -ErrorAction Stop
            $serviceName = $candidate
            break
        }
        catch {
            continue
        }
    }

    if (-not $serviceName) {
        Write-Log "Phase 1: WSL service not found (tried: $($serviceCandidates -join ', '))" -Level ERROR
        return $false
    }

    Write-Log "Phase 1: Ensuring $serviceName service is running..."

    # In WhatIf mode, simulate success
    if ($WhatIfPreference) {
        if ($PSCmdlet.ShouldProcess($serviceName, 'Check and Start Service')) {
            Write-Log "WhatIf: Would check and start $serviceName if needed"
        }
        return $true
    }

    try {
        $service = Get-Service -Name $serviceName -ErrorAction Stop

        if ($service.Status -eq 'Running') {
            Write-Log "$serviceName already running" -Level SUCCESS
            return $true
        }

        if ($PSCmdlet.ShouldProcess($serviceName, 'Start Service')) {
            Write-Log "$serviceName not running, attempting to start..."
            Start-Service -Name $serviceName -ErrorAction Stop

            # Wait for service to fully start
            $service.WaitForStatus('Running', [TimeSpan]::FromSeconds(30))
            Write-Log "$serviceName started successfully" -Level SUCCESS
            return $true
        }

        return $false
    }
    catch {
        Write-Log "Failed to start $serviceName: $_" -Level ERROR
        return $false
    }
}

function Initialize-WSL {
    <#
    .SYNOPSIS
        Initializes the WSL distribution with retry logic.
    .OUTPUTS
        [bool] True if WSL initialized successfully, False otherwise.
    #>
    [CmdletBinding(SupportsShouldProcess)]
    [OutputType([bool])]
    param(
        [Parameter(Mandatory)]
        [string]$Distro,

        [Parameter()]
        [int]$Retries = 10,

        [Parameter()]
        [int]$Delay = 10
    )

    Write-Log "Phase 2: Initializing WSL ($Distro)..."

    # In WhatIf mode, simulate success
    if ($WhatIfPreference) {
        if ($PSCmdlet.ShouldProcess("WSL Distribution: $Distro", 'Initialize')) {
            Write-Log "WhatIf: Would initialize WSL distribution '$Distro' with $Retries retries"
        }
        return $true
    }

    if (-not $PSCmdlet.ShouldProcess("WSL Distribution: $Distro", 'Initialize')) {
        return $true
    }

    for ($attempt = 1; $attempt -le $Retries; $attempt++) {
        try {
            $result = & wsl.exe -d $Distro --exec /bin/true 2>&1
            $exitCode = $LASTEXITCODE

            if ($exitCode -eq 0) {
                Write-Log 'WSL started successfully' -Level SUCCESS
                return $true
            }

            Write-Log "WSL start attempt $attempt/$Retries failed (exit code: $exitCode), retrying in ${Delay}s..." -Level WARNING
        }
        catch {
            Write-Log "WSL start attempt $attempt/$Retries exception: $_" -Level WARNING
        }

        if ($attempt -lt $Retries) {
            Start-Sleep -Seconds $Delay
        }
    }

    Write-Log "WSL failed to start after $Retries attempts" -Level ERROR
    return $false
}

function Initialize-DNS {
    <#
    .SYNOPSIS
        Initializes DNS in the WSL distribution.
    .OUTPUTS
        [bool] True if DNS initialized (or warning on partial failure), False on critical failure.
    #>
    [CmdletBinding(SupportsShouldProcess)]
    [OutputType([bool])]
    param(
        [Parameter(Mandatory)]
        [string]$Distro
    )

    Write-Log 'Phase 3: Initializing DNS...'

    # In WhatIf mode, simulate success
    if ($WhatIfPreference) {
        if ($PSCmdlet.ShouldProcess('WSL DNS', 'Initialize')) {
            Write-Log "WhatIf: Would run DNS initialization script in WSL"
        }
        return $true
    }

    if (-not $PSCmdlet.ShouldProcess('WSL DNS', 'Initialize')) {
        return $true
    }

    try {
        $result = & wsl.exe -d $Distro --exec sudo -n $script:Config.WslDnsInit 2>&1
        $exitCode = $LASTEXITCODE

        if ($exitCode -eq 0) {
            Write-Log 'DNS initialized successfully' -Level SUCCESS
            return $true
        }

        Write-Log "DNS initialization had issues (exit code: $exitCode) - continuing" -Level WARNING
        return $true  # Non-critical, continue startup
    }
    catch {
        Write-Log "DNS initialization exception: $_ - continuing" -Level WARNING
        return $true  # Non-critical, continue startup
    }
}

function Start-HyperVBridges {
    <#
    .SYNOPSIS
        Starts Hyper-V socket bridges in WSL.
    .OUTPUTS
        [bool] True if bridges started successfully, False otherwise.
    #>
    [CmdletBinding(SupportsShouldProcess)]
    [OutputType([bool])]
    param(
        [Parameter(Mandatory)]
        [string]$Distro
    )

    Write-Log 'Phase 4: Starting Hyper-V socket bridges...'

    # In WhatIf mode, simulate success
    if ($WhatIfPreference) {
        if ($PSCmdlet.ShouldProcess('Hyper-V Socket Bridges', 'Start')) {
            Write-Log "WhatIf: Would start Hyper-V socket bridges in WSL"
        }
        return $true
    }

    if (-not $PSCmdlet.ShouldProcess('Hyper-V Socket Bridges', 'Start')) {
        return $true
    }

    try {
        $result = & wsl.exe -d $Distro --exec sudo -n $script:Config.WslVsockBridge start 2>&1
        $exitCode = $LASTEXITCODE

        if ($exitCode -eq 0) {
            Write-Log 'Hyper-V bridges started successfully' -Level SUCCESS
            return $true
        }

        Write-Log "Bridge startup had issues (exit code: $exitCode)" -Level WARNING
        return $true  # Non-critical, continue startup
    }
    catch {
        Write-Log "Bridge startup exception: $_" -Level WARNING
        return $true  # Non-critical, continue startup
    }
}

function Initialize-Bitwarden {
    <#
    .SYNOPSIS
        Initializes Bitwarden session by running the Initialize-BitwardenSession script.
    .OUTPUTS
        [bool] True if Bitwarden session initialized successfully, False otherwise.
    #>
    [CmdletBinding(SupportsShouldProcess)]
    [OutputType([bool])]
    param()

    Write-Log 'Phase 5: Initializing Bitwarden session...'

    # In WhatIf mode, simulate success
    if ($WhatIfPreference) {
        if ($PSCmdlet.ShouldProcess('Bitwarden Session', 'Initialize')) {
            Write-Log 'WhatIf: Would run Initialize-BitwardenSession.ps1'
        }
        return $true
    }

    # Check if script exists
    if (-not (Test-Path -Path $script:Config.InitBitwardenScript)) {
        Write-Log "Bitwarden initialization script not found: $($script:Config.InitBitwardenScript)" -Level WARNING
        return $false
    }

    if (-not $PSCmdlet.ShouldProcess('Bitwarden Session', 'Initialize')) {
        return $true
    }

    try {
        # Run the initialization script
        $result = & $script:Config.InitBitwardenScript 2>&1
        $exitCode = $LASTEXITCODE

        if ($exitCode -eq 0) {
            Write-Log 'Bitwarden session initialized successfully' -Level SUCCESS
            return $true
        }

        Write-Log "Bitwarden initialization failed (exit code: $exitCode): $result" -Level WARNING
        return $false
    }
    catch {
        Write-Log "Bitwarden initialization exception: $_" -Level WARNING
        return $false
    }
}

function Sync-Credentials {
    <#
    .SYNOPSIS
        Synchronizes registry credentials from Bitwarden to Windows Credential Manager.
    .OUTPUTS
        [bool] True if credentials synced successfully, False otherwise.
    #>
    [CmdletBinding(SupportsShouldProcess)]
    [OutputType([bool])]
    param()

    Write-Log 'Phase 6: Syncing registry credentials...'

    # In WhatIf mode, simulate success
    if ($WhatIfPreference) {
        if ($PSCmdlet.ShouldProcess('Registry Credentials', 'Sync')) {
            Write-Log 'WhatIf: Would run Sync-RegistryCredentials.ps1'
        }
        return $true
    }

    # Check if script exists
    if (-not (Test-Path -Path $script:Config.SyncCredentialsScript)) {
        Write-Log "Credential sync script not found: $($script:Config.SyncCredentialsScript)" -Level WARNING
        return $false
    }

    # Check if BW_SESSION is set
    if ([string]::IsNullOrEmpty($env:BW_SESSION)) {
        Write-Log 'BW_SESSION not set - credential sync requires Bitwarden session' -Level WARNING
        return $false
    }

    if (-not $PSCmdlet.ShouldProcess('Registry Credentials', 'Sync')) {
        return $true
    }

    try {
        # Run the sync script
        $result = & $script:Config.SyncCredentialsScript 2>&1
        $exitCode = $LASTEXITCODE

        if ($exitCode -eq 0) {
            Write-Log 'Registry credentials synced successfully' -Level SUCCESS
            return $true
        }
        elseif ($exitCode -eq 1) {
            Write-Log 'Credential sync completed with partial success' -Level WARNING
            return $true  # Partial success is acceptable
        }

        Write-Log "Credential sync failed (exit code: $exitCode): $result" -Level WARNING
        return $false
    }
    catch {
        Write-Log "Credential sync exception: $_" -Level WARNING
        return $false
    }
}

function Start-PodmanMachine {
    <#
    .SYNOPSIS
        Ensures the Podman machine is running.
    .OUTPUTS
        [bool] True if Podman machine is running or was started, False on failure.
    #>
    [CmdletBinding(SupportsShouldProcess)]
    [OutputType([bool])]
    param()

    Write-Log 'Phase 10: Ensuring Podman machine is running...'

    # In WhatIf mode, simulate success
    if ($WhatIfPreference) {
        if ($PSCmdlet.ShouldProcess('Podman Machine', 'Start')) {
            Write-Log 'WhatIf: Would check and start Podman machine if needed'
        }
        return $true
    }

    # Skip Podman if Docker Desktop is running (avoids pipe/API conflicts)
    $dockerBackend = Get-Process -Name 'com.docker.backend' -ErrorAction SilentlyContinue
    if ($dockerBackend) {
        Write-Log 'Docker Desktop is running - skipping Podman machine to avoid pipe conflicts' -Level WARNING
        return $true
    }

    # Check if podman command exists
    $podmanCmd = Get-Command -Name 'podman' -ErrorAction SilentlyContinue
    if (-not $podmanCmd) {
        Write-Log 'Podman not found in PATH - skipping Podman machine' -Level WARNING
        return $false
    }

    if (-not $PSCmdlet.ShouldProcess('Podman Machine', 'Start')) {
        return $true
    }

    try {
        # Check if machine exists and get its state
        $machineList = & podman machine list --format json 2>&1
        $exitCode = $LASTEXITCODE

        if ($exitCode -ne 0) {
            Write-Log "Failed to list Podman machines: $machineList" -Level WARNING
            return $false
        }

        $machines = $machineList | ConvertFrom-Json

        # Find the default machine
        $targetMachine = $machines | Where-Object { $_.Name -eq $script:Config.PodmanMachineName }

        if (-not $targetMachine) {
            Write-Log "Podman machine '$($script:Config.PodmanMachineName)' not found" -Level WARNING
            return $false
        }

        # Check if already running
        if ($targetMachine.Running -eq $true) {
            Write-Log "Podman machine '$($script:Config.PodmanMachineName)' already running" -Level SUCCESS
            return $true
        }

        # Start the machine
        Write-Log "Starting Podman machine '$($script:Config.PodmanMachineName)'..."
        $startResult = & podman machine start $script:Config.PodmanMachineName 2>&1
        $exitCode = $LASTEXITCODE

        if ($exitCode -eq 0) {
            Write-Log "Podman machine '$($script:Config.PodmanMachineName)' started successfully" -Level SUCCESS
            return $true
        }

        Write-Log "Failed to start Podman machine (exit code: $exitCode): $startResult" -Level WARNING
        return $false
    }
    catch {
        Write-Log "Podman machine startup exception: $_" -Level WARNING
        return $false
    }
}

function Start-DockerDesktop {
    <#
    .SYNOPSIS
        Starts Docker Desktop if not already running.
    .OUTPUTS
        [bool] True if Docker Desktop is running or was started, False on failure.
    #>
    [CmdletBinding(SupportsShouldProcess)]
    [OutputType([bool])]
    param()

    Write-Log 'Phase 7: Starting Docker Desktop (if not running)...'

    # In WhatIf mode, simulate success
    if ($WhatIfPreference) {
        if ($PSCmdlet.ShouldProcess('Docker Desktop', 'Start')) {
            Write-Log "WhatIf: Would check and start Docker Desktop if needed"
        }
        return $true
    }

    try {
        # If Podman machine is running, stop it before starting Docker Desktop
        $podmanCmd = Get-Command -Name 'podman' -ErrorAction SilentlyContinue
        if ($podmanCmd) {
            try {
                $machineList = & podman machine list --format json 2>$null
                if ($LASTEXITCODE -eq 0 -and $machineList) {
                    $machines = $machineList | ConvertFrom-Json
                    $runningMachines = $machines | Where-Object { $_.Running -eq $true }
                    foreach ($machine in $runningMachines) {
                        Write-Log "Stopping Podman machine '$($machine.Name)' to avoid Docker Desktop conflicts..." -Level WARNING
                        & podman machine stop $machine.Name 2>&1 | Out-Null
                    }
                }
            }
            catch {
                Write-Log "Failed to check/stop Podman machine: $_" -Level WARNING
            }
        }

        $dockerProcess = Get-Process -Name 'Docker Desktop' -ErrorAction SilentlyContinue

        if ($dockerProcess) {
            Write-Log 'Docker Desktop already running' -Level SUCCESS
            return $true
        }

        if (-not (Test-Path -Path $script:Config.DockerDesktopExe)) {
            Write-Log "Docker Desktop not found at: $($script:Config.DockerDesktopExe)" -Level ERROR
            return $false
        }

        if ($PSCmdlet.ShouldProcess('Docker Desktop', 'Start')) {
            Write-Log 'Launching Docker Desktop...'
            Start-Process -FilePath $script:Config.DockerDesktopExe -WindowStyle Hidden
            Write-Log 'Docker Desktop launch initiated' -Level SUCCESS
            return $true
        }

        return $false
    }
    catch {
        Write-Log "Failed to start Docker Desktop: $_" -Level ERROR
        return $false
    }
}

function Wait-DockerEngine {
    <#
    .SYNOPSIS
        Waits for Docker engine to become ready.
    .OUTPUTS
        [bool] True if Docker engine is ready, False if timeout reached.
    #>
    [CmdletBinding(SupportsShouldProcess)]
    [OutputType([bool])]
    param(
        [Parameter()]
        [int]$Retries = 30,

        [Parameter()]
        [int]$Delay = 5
    )

    Write-Log 'Phase 8: Waiting for Docker engine to become ready...'

    # In WhatIf mode, simulate success
    if ($WhatIfPreference) {
        if ($PSCmdlet.ShouldProcess('Docker Engine', 'Wait for readiness')) {
            Write-Log "WhatIf: Would wait up to $($Retries * $Delay) seconds for Docker engine"
        }
        return $true
    }

    if (-not $PSCmdlet.ShouldProcess('Docker Engine', 'Wait for readiness')) {
        return $true
    }

    # Check if docker command exists (try to add default install path if missing)
    $dockerCmd = Get-Command -Name 'docker' -ErrorAction SilentlyContinue
    if (-not $dockerCmd) {
        $candidatePaths = @(
            $env:DOCKER_PATH,
            'C:\Program Files\Docker\Docker\resources\bin'
        ) | Where-Object { $_ -and (Test-Path (Join-Path $_ 'docker.exe')) }

        foreach ($path in $candidatePaths) {
            if ($env:Path -notmatch [regex]::Escape($path)) {
                $env:Path = "$path;$env:Path"
                Write-Log "Added Docker CLI path for this session: $path" -Level SUCCESS
            }
        }

        $dockerCmd = Get-Command -Name 'docker' -ErrorAction SilentlyContinue
        if (-not $dockerCmd) {
            Write-Log 'Docker CLI not found in PATH' -Level ERROR
            return $false
        }
    }

    for ($attempt = 1; $attempt -le $Retries; $attempt++) {
        try {
            $null = & docker info 2>&1
            $exitCode = $LASTEXITCODE

            if ($exitCode -eq 0) {
                Write-Log 'Docker engine is ready' -Level SUCCESS
                return $true
            }

            Write-Log "Waiting for Docker engine... attempt $attempt/$Retries"
        }
        catch {
            Write-Log "Docker check attempt $attempt failed: $_" -Level WARNING
        }

        if ($attempt -lt $Retries) {
            Start-Sleep -Seconds $Delay
        }
    }

    Write-Log "Docker engine not ready after $Retries attempts" -Level WARNING
    return $false
}

function Start-RAGRedis {
    <#
    .SYNOPSIS
        Starts the RAG-Redis container using docker-compose.
    .OUTPUTS
        [bool] True if container started successfully, False otherwise.
    #>
    [CmdletBinding(SupportsShouldProcess)]
    [OutputType([bool])]
    param()

    Write-Log 'Phase 9: Starting RAG-Redis container...'

    # In WhatIf mode, simulate success
    if ($WhatIfPreference) {
        if ($PSCmdlet.ShouldProcess('RAG-Redis Container', 'Start')) {
            Write-Log "WhatIf: Would start RAG-Redis container from $($script:Config.RAGRedisCompose)"
        }
        return $true
    }

    if (-not (Test-Path -Path $script:Config.RAGRedisCompose)) {
        Write-Log "RAG-Redis compose file not found: $($script:Config.RAGRedisCompose)" -Level WARNING
        return $false
    }

    if (-not $PSCmdlet.ShouldProcess('RAG-Redis Container', 'Start')) {
        return $true
    }

    try {
        # Try docker compose (v2) first, fall back to docker-compose
        $composeCmd = Get-Command -Name 'docker' -ErrorAction SilentlyContinue
        if ($composeCmd) {
            $result = & docker compose -f $script:Config.RAGRedisCompose up -d redis 2>&1
            $exitCode = $LASTEXITCODE
        }
        else {
            $result = & docker-compose -f $script:Config.RAGRedisCompose up -d redis 2>&1
            $exitCode = $LASTEXITCODE
        }

        if ($exitCode -eq 0) {
            Write-Log 'RAG-Redis container started successfully' -Level SUCCESS
            return $true
        }

        Write-Log "RAG-Redis container failed to start (exit code: $exitCode): $result" -Level WARNING
        return $false
    }
    catch {
        Write-Log "RAG-Redis startup exception: $_" -Level WARNING
        return $false
    }
}
#endregion

#region Main Execution
function Invoke-StartupSequence {
    <#
    .SYNOPSIS
        Orchestrates the complete startup sequence.
    .OUTPUTS
        [int] Exit code: 0 = Success, 1 = Error, 2 = Warnings
    #>
    [CmdletBinding(SupportsShouldProcess)]
    [OutputType([int])]
    param()

    $hasErrors = $false
    $hasWarnings = $false

    # Phase 0: Startup delay
    if ($StartupDelay -gt 0) {
        Write-Log "Phase 0: Waiting ${StartupDelay}s for system startup..."
        if (-not $WhatIfPreference) {
            Start-Sleep -Seconds $StartupDelay
        }
    }

    # Phase 0.5: Environment prep for Docker MCP mode
    Initialize-DockerMcpMode

    # Phase 1: LxssManager
    if (-not (Start-LxssManager)) {
        $hasErrors = $true
        Write-Log 'Critical: LxssManager failed - cannot continue' -Level ERROR
        return 1
    }

    # Phase 2: WSL Initialization
    if (-not (Initialize-WSL -Distro $WslDistro -Retries $MaxRetries -Delay $RetryDelay)) {
        $hasErrors = $true
        Write-Log 'Critical: WSL initialization failed - cannot continue' -Level ERROR
        return 1
    }

    # Phase 3: DNS (non-critical)
    if (-not (Initialize-DNS -Distro $WslDistro)) {
        $hasWarnings = $true
    }

    # Phase 4: Hyper-V Bridges (non-critical)
    if (-not (Start-HyperVBridges -Distro $WslDistro)) {
        $hasWarnings = $true
    }

    # Phase 5 & 6: Bitwarden and Credential Sync (non-critical)
    if (-not $SkipBitwarden) {
        $bwSuccess = Initialize-Bitwarden
        if ($bwSuccess) {
            # Only sync credentials if Bitwarden initialized
            if (-not (Sync-Credentials)) {
                $hasWarnings = $true
            }
        }
        else {
            $hasWarnings = $true
            Write-Log 'Bitwarden failed - skipping credential sync' -Level WARNING
        }
    }
    else {
        Write-Log 'Phase 5-6: Bitwarden and credential sync skipped (SkipBitwarden flag set)'
    }

    # Phase 7: Docker Desktop
    if (-not (Start-DockerDesktop)) {
        $hasWarnings = $true
        Write-Log 'Docker Desktop failed to start - skipping container phases' -Level WARNING
        return 2
    }

    # Phase 8: Wait for Docker
    $dockerReady = Wait-DockerEngine -Retries $DockerReadyRetries -Delay $DockerReadyDelay
    if (-not $dockerReady) {
        $hasWarnings = $true
        Write-Log 'Docker engine not ready - skipping container phases' -Level WARNING
        return 2
    }

    # Phase 9: RAG-Redis
    if (-not $SkipRAGRedis) {
        if (-not (Start-RAGRedis)) {
            $hasWarnings = $true
        }
    }
    else {
        Write-Log 'Phase 9: RAG-Redis skipped (SkipRAGRedis flag set)'
    }

    # Phase 10: Podman Machine (non-critical)
    if (-not $SkipPodman) {
        if (-not (Start-PodmanMachine)) {
            $hasWarnings = $true
        }
    }
    else {
        Write-Log 'Phase 10: Podman machine skipped (SkipPodman flag set)'
    }

    # Return appropriate exit code
    if ($hasErrors) {
        return 1
    }
    elseif ($hasWarnings) {
        return 2
    }
    return 0
}

# Main entry point
try {
    # Log rotation
    Invoke-LogRotation -Path $LogFile -MaxLines $MaxLogLines

    Write-LogSection 'WSL Startup v4.0 - Starting...'

    $exitCode = Invoke-StartupSequence

    switch ($exitCode) {
        0 { Write-LogSection 'Startup completed successfully' }
        1 { Write-LogSection 'Startup completed with ERRORS' }
        2 { Write-LogSection 'Startup completed with WARNINGS' }
    }

    exit $exitCode
}
catch {
    Write-Log "Unhandled exception: $_" -Level ERROR
    Write-Log $_.ScriptStackTrace -Level ERROR
    Write-LogSection 'Startup FAILED with unhandled exception'
    exit 1
}
#endregion
