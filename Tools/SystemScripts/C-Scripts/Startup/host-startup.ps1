#requires -Version 5.1
[CmdletBinding(SupportsShouldProcess = $true)]
param(
  [switch]$EnsureOllama,
  [switch]$StartDocker,
  [switch]$StartWSL,
  [switch]$DryRun
)

$ErrorActionPreference = 'Stop'

function Write-Log {
  param([string]$Message, [string]$Level = 'INFO')
  $timestamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
  Write-Host "[$timestamp][$Level] $Message"
}

function Ensure-DockerService {
  param([switch]$DryRunMode)
  $svc = Get-Service -Name 'com.docker.service' -ErrorAction SilentlyContinue
  if (-not $svc) {
    Write-Log 'Docker service not found.' 'WARN'
    return
  }
  if ($svc.Status -ne 'Running') {
    if ($DryRunMode) {
      Write-Log '[DryRun] Would start Docker service' 'DRYRUN'
      return
    }
    Start-Service -Name 'com.docker.service'
    Write-Log 'Docker service started'
  } else {
    Write-Log 'Docker service already running'
  }

  # Ensure Docker Desktop app is launched for MCP mode
  $dockerDesktopExe = 'C:\Program Files\Docker\Docker\Docker Desktop.exe'
  if (Test-Path $dockerDesktopExe) {
    $running = Get-Process -Name 'Docker Desktop' -ErrorAction SilentlyContinue
    if (-not $running) {
      if ($DryRunMode) {
        Write-Log '[DryRun] Would start Docker Desktop app' 'DRYRUN'
      } else {
        Start-Process -FilePath $dockerDesktopExe -WindowStyle Hidden
        Write-Log 'Docker Desktop app launch initiated'
      }
    }
  }
}

function Ensure-WSL {
  param([switch]$DryRunMode)
  $wsl = Get-Command wsl.exe -ErrorAction SilentlyContinue
  if (-not $wsl) {
    Write-Log 'WSL not available.' 'WARN'
    return
  }
  if ($DryRunMode) {
    Write-Log '[DryRun] Would check WSL status' 'DRYRUN'
    return
  }
  wsl.exe --status | Out-Null
  Write-Log 'WSL status checked'
}

if ($EnsureOllama) {
  $ollamaScript = 'C:\scripts\startup\ollama-service.ps1'
  if (Test-Path $ollamaScript) {
    if ($DryRun) {
      Write-Log '[DryRun] Would run Ollama service ensure' 'DRYRUN'
    } else {
      & $ollamaScript -Ensure
    }
  } else {
    Write-Log "Ollama service script not found: $ollamaScript" 'ERROR'
  }
}

if ($StartDocker) {
  # Docker-MCP mode defaults
  if ($env:DOCKER_HOST) {
    Write-Log "Clearing DOCKER_HOST ($env:DOCKER_HOST)" 'WARN'
    Remove-Item Env:DOCKER_HOST -ErrorAction SilentlyContinue
  }
  $dockerBin = 'C:\Program Files\Docker\Docker\resources\bin'
  if (Test-Path (Join-Path $dockerBin 'docker.exe')) {
    if ($env:Path -notmatch [regex]::Escape($dockerBin)) {
      $env:Path = "$dockerBin;$env:Path"
      Write-Log "Added Docker CLI path for this session: $dockerBin"
    }
  }
  if (Get-Command podman -ErrorAction SilentlyContinue) {
    if (-not $DryRun) {
      podman machine stop podman-machine-default 2>$null | Out-Null
    }
  }
  Ensure-DockerService -DryRunMode:$DryRun
}

if ($StartWSL) {
  Ensure-WSL -DryRunMode:$DryRun
}
