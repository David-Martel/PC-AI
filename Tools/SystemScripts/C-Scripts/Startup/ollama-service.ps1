#requires -Version 5.1
[CmdletBinding(SupportsShouldProcess = $true)]
param(
  [string]$ServiceName = 'Ollama',
  [string]$OllamaExePath,
  [string]$OllamaHost = 'http://localhost:11434',
  [switch]$DisableVulkan,
  [string]$VulkanDevices,
  [string]$CudaDevices,
  [string]$PreferCudaGpuName,
  [string]$PreferCudaGpuUuid,
  [switch]$PreferCuda,
  [switch]$StopExisting,
  [switch]$Install,
  [switch]$Configure,
  [switch]$Ensure,
  [switch]$Start,
  [switch]$Stop,
  [switch]$Restart,
  [switch]$Status,
  [switch]$DryRun
)

$ErrorActionPreference = 'Stop'

function Write-Log {
  param([string]$Message, [string]$Level = 'INFO')
  $timestamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
  Write-Host "[$timestamp][$Level] $Message"
}

function Get-OllamaExePath {
  param([string]$ExplicitPath)
  if ($ExplicitPath -and (Test-Path $ExplicitPath)) { return $ExplicitPath }

  $pathsToCheck = @(
    (Join-Path $env:ProgramFiles 'Ollama\ollama.exe'),
    (Join-Path $env:LOCALAPPDATA 'Programs\Ollama\ollama.exe')
  )

  foreach ($path in $pathsToCheck) {
    if (Test-Path $path) { return $path }
  }

  $cmd = Get-Command ollama -ErrorAction SilentlyContinue
  if ($cmd) { return $cmd.Path }

  return $null
}

function Get-OllamaServeHost {
  param([string]$HostValue)
  if (-not $HostValue) { return '127.0.0.1:11434' }
  if ($HostValue -match '://') {
    try {
      $uri = [Uri]$HostValue
      return "{0}:{1}" -f $uri.Host, $uri.Port
    } catch {
      return $HostValue -replace '^https?://', ''
    }
  }
  return $HostValue
}

function Get-ServiceRegPath {
  param([string]$Name)
  return "HKLM:\SYSTEM\CurrentControlSet\Services\$Name"
}

function Set-ServiceEnvironment {
  param(
    [string]$Name,
    [string[]]$EnvPairs,
    [switch]$DryRunMode
  )
  $regPath = Get-ServiceRegPath -Name $Name
  if (-not (Test-Path $regPath)) {
    if ($DryRunMode) {
      Write-Log "[DryRun] Service registry path not found (would create on install): $regPath" 'DRYRUN'
      return
    }
    throw "Service registry path not found: $regPath"
  }

  if ($DryRunMode) {
    Write-Log "[DryRun] Would set service environment for ${Name}: $($EnvPairs -join '; ')" 'DRYRUN'
    return
  }

  New-ItemProperty -Path $regPath -Name 'Environment' -Value $EnvPairs -PropertyType MultiString -Force | Out-Null
  Write-Log "Set service environment for $Name"
}

function Stop-OllamaProcesses {
  param([switch]$DryRunMode)
  $procs = Get-Process -Name 'ollama' -ErrorAction SilentlyContinue
  if (-not $procs) {
    return
  }
  if ($DryRunMode) {
    Write-Log "[DryRun] Would stop $($procs.Count) Ollama process(es)" 'DRYRUN'
    return
  }
  $procs | Stop-Process -Force
  Write-Log "Stopped $($procs.Count) Ollama process(es)"
}

function Install-OllamaService {
  param(
    [string]$Name,
    [string]$ExePath,
    [string]$ServeHost,
    [switch]$DryRunMode
  )

  $existing = Get-Service -Name $Name -ErrorAction SilentlyContinue
  if ($existing) {
    Write-Log "Service $Name already exists. Skipping install." 'WARN'
    return
  }

  $binPath = "`"$ExePath`" serve --host $ServeHost"

  if ($DryRunMode) {
    Write-Log "[DryRun] Would create service $Name with binPath: $binPath" 'DRYRUN'
    return
  }

  Write-Log "Creating service $Name"
  New-Service -Name $Name -BinaryPathName $binPath -StartupType Automatic
}

function Configure-OllamaService {
  param(
    [string]$Name,
    [switch]$DisableVulkanMode,
    [string]$VulkanDeviceList,
    [string]$CudaDeviceList,
    [switch]$PreferCudaMode,
    [switch]$DryRunMode
  )
  $envPairs = @()

  $useVulkan = -not $DisableVulkanMode
  if ($PreferCudaMode -or $CudaDeviceList) {
    $useVulkan = $false
  }

  if ($CudaDeviceList) {
    $envPairs += "CUDA_VISIBLE_DEVICES=$CudaDeviceList"
  }

  if ($useVulkan) {
    $envPairs += 'OLLAMA_VULKAN=1'
    if ($VulkanDeviceList) {
      $envPairs += "GGML_VK_VISIBLE_DEVICES=$VulkanDeviceList"
    }
  } else {
    $envPairs += 'OLLAMA_VULKAN=0'
  }

  Set-ServiceEnvironment -Name $Name -EnvPairs $envPairs -DryRunMode:$DryRunMode
}

function Get-NvidiaGpuList {
  $cmd = Get-Command nvidia-smi -ErrorAction SilentlyContinue
  if (-not $cmd) { return @() }

  $output = & $cmd.Path -L 2>$null
  if (-not $output) { return @() }

  $gpus = @()
  foreach ($line in $output -split "`r?`n") {
    if ($line -match '^GPU\s+(?<index>\d+):\s+(?<name>.+?)\s+\(UUID:\s*(?<uuid>[^)]+)\)') {
      $gpus += [PSCustomObject]@{
        Index = $matches.index
        Name = $matches.name
        Uuid = $matches.uuid
      }
    }
  }
  return $gpus
}

function Resolve-CudaDevices {
  param(
    [string]$ExplicitDevices,
    [string]$PreferredName,
    [string]$PreferredUuid
  )

  if ($ExplicitDevices) { return $ExplicitDevices }

  $gpus = Get-NvidiaGpuList
  if (-not $gpus -or $gpus.Count -eq 0) { return $null }

  if ($PreferredUuid) {
    $match = $gpus | Where-Object { $_.Uuid -eq $PreferredUuid } | Select-Object -First 1
    if ($match) { return $match.Uuid }
    Write-Log "Preferred CUDA GPU UUID not found: $PreferredUuid" 'WARN'
  }

  if ($PreferredName) {
    $match = $gpus | Where-Object { $_.Name -like "*$PreferredName*" } | Select-Object -First 1
    if ($match) { return $match.Index }
    Write-Log "Preferred CUDA GPU name not found: $PreferredName" 'WARN'
  }

  return $null
}

function Ensure-OllamaService {
  param(
    [string]$Name,
    [string]$ExePath,
    [string]$ServeHost,
    [switch]$DisableVulkanMode,
    [string]$VulkanDeviceList,
    [string]$CudaDeviceList,
    [string]$PreferCudaGpuNameValue,
    [string]$PreferCudaGpuUuidValue,
    [switch]$PreferCudaMode,
    [switch]$DryRunMode,
    [switch]$StopExistingMode
  )

  Install-OllamaService -Name $Name -ExePath $ExePath -ServeHost $ServeHost -DryRunMode:$DryRunMode
  $resolvedCudaDevices = Resolve-CudaDevices -ExplicitDevices $CudaDeviceList -PreferredName $PreferCudaGpuNameValue -PreferredUuid $PreferCudaGpuUuidValue
  if ($resolvedCudaDevices) {
    Write-Log "Resolved CUDA devices: $resolvedCudaDevices"
  }

  Configure-OllamaService -Name $Name -DisableVulkanMode:$DisableVulkanMode -VulkanDeviceList $VulkanDeviceList -CudaDeviceList $resolvedCudaDevices -PreferCudaMode:$PreferCudaMode -DryRunMode:$DryRunMode

  if ($StopExistingMode) {
    Stop-OllamaProcesses -DryRunMode:$DryRunMode
  }

  if (-not $DryRunMode) {
    $svc = Get-Service -Name $Name -ErrorAction SilentlyContinue
    if ($svc -and $svc.Status -ne 'Running') {
      Start-Service -Name $Name
      Write-Log "Started service $Name"
    }
  } else {
    Write-Log "[DryRun] Would ensure service $Name is running" 'DRYRUN'
  }
}

function Show-ServiceStatus {
  param([string]$Name)
  $svc = Get-Service -Name $Name -ErrorAction SilentlyContinue
  if (-not $svc) {
    Write-Log "Service $Name not found" 'WARN'
    return
  }
  Write-Log "Service $Name status: $($svc.Status)"
}

$resolvedExe = Get-OllamaExePath -ExplicitPath $OllamaExePath
if (-not $resolvedExe) {
  Write-Log 'Ollama executable not found. Set -OllamaExePath or install Ollama.' 'ERROR'
  exit 1
}

$serveHost = Get-OllamaServeHost -HostValue $OllamaHost

if ($Status) {
  Show-ServiceStatus -Name $ServiceName
  exit 0
}

if ($Ensure -or (-not $Install -and -not $Configure -and -not $Start -and -not $Stop -and -not $Restart)) {
  Ensure-OllamaService -Name $ServiceName -ExePath $resolvedExe -ServeHost $serveHost -DisableVulkanMode:$DisableVulkan -VulkanDeviceList $VulkanDevices -CudaDeviceList $CudaDevices -PreferCudaGpuNameValue $PreferCudaGpuName -PreferCudaGpuUuidValue $PreferCudaGpuUuid -PreferCudaMode:$PreferCuda -DryRunMode:$DryRun -StopExistingMode:$StopExisting
  exit 0
}

if ($Install) {
  Install-OllamaService -Name $ServiceName -ExePath $resolvedExe -ServeHost $serveHost -DryRunMode:$DryRun
}

if ($Configure) {
  Configure-OllamaService -Name $ServiceName -DisableVulkanMode:$DisableVulkan -VulkanDeviceList $VulkanDevices -CudaDeviceList $CudaDevices -PreferCudaMode:$PreferCuda -DryRunMode:$DryRun
}

if ($Start) {
  if ($DryRun) {
    Write-Log "[DryRun] Would start service $ServiceName" 'DRYRUN'
  } else {
    Start-Service -Name $ServiceName
    Write-Log "Started service $ServiceName"
  }
}

if ($Stop) {
  if ($DryRun) {
    Write-Log "[DryRun] Would stop service $ServiceName" 'DRYRUN'
  } else {
    Stop-Service -Name $ServiceName -Force
    Write-Log "Stopped service $ServiceName"
  }
}

if ($Restart) {
  if ($DryRun) {
    Write-Log "[DryRun] Would restart service $ServiceName" 'DRYRUN'
  } else {
    Restart-Service -Name $ServiceName -Force
    Write-Log "Restarted service $ServiceName"
  }
}
