[CmdletBinding()]
param(
    [ValidateSet('status', 'driver-sync', 'init-env', 'install', 'registry')]
    [string]$Action = 'status',

    [string]$ComponentId = 'cuda-toolkit',
    [string]$RegistryPath,
    [string]$PreferredCudaVersion,
    [string]$InstallerPath,
    [switch]$DownloadOnly,
    [switch]$Force,
    [switch]$Quiet
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$bootstrapPath = Join-Path $PSScriptRoot 'PcaiModuleBootstrap.ps1'

if (Test-Path -LiteralPath $bootstrapPath) {
    . $bootstrapPath
}

if (Get-Command -Name Import-PcaiResolvedModule -ErrorAction SilentlyContinue) {
    $module = Import-PcaiResolvedModule -ModuleName 'PC-AI.Gpu' -RepoRoot $repoRoot -Force
} else {
    $manifestPath = Join-Path $repoRoot 'Modules\PC-AI.Gpu\PC-AI.Gpu.psd1'
    if (-not (Test-Path -LiteralPath $manifestPath)) {
        throw "PC-AI.Gpu module manifest not found at $manifestPath"
    }
    $module = Import-Module -Name $manifestPath -PassThru -Force -ErrorAction Stop
}

if (-not $module) {
    throw 'Failed to import PC-AI.Gpu.'
}

switch ($Action) {
    'status' {
        $statusParams = @{}
        if ($RegistryPath) {
            $statusParams.RegistryPath = $RegistryPath
        }

        [PSCustomObject]@{
            Gpus = @(Get-NvidiaGpuInventory)
            Software = @(Get-NvidiaSoftwareStatus @statusParams)
        }
    }
    'driver-sync' {
        $syncScript = Join-Path $PSScriptRoot 'Sync-NvidiaDriverVersion.ps1'
        if (-not (Test-Path -LiteralPath $syncScript)) {
            throw "Driver sync script not found at $syncScript"
        }

        $syncArgs = @{}
        if ($RegistryPath) {
            $syncArgs.RegistryPath = $RegistryPath
        }
        if ($Force) {
            $syncArgs.Force = $true
        }

        & $syncScript @syncArgs
    }
    'init-env' {
        $initArgs = @{
            Scope = 'Process'
        }
        if ($PreferredCudaVersion) {
            $initArgs.PreferredCudaVersion = $PreferredCudaVersion
        }
        if ($Quiet) {
            $initArgs.Quiet = $true
        }

        Initialize-NvidiaEnvironment @initArgs
    }
    'install' {
        $installArgs = @{
            ComponentId = $ComponentId
        }
        if ($RegistryPath) {
            $installArgs.RegistryPath = $RegistryPath
        }
        if ($InstallerPath) {
            $installArgs.InstallerPath = $InstallerPath
        }
        if ($DownloadOnly) {
            $installArgs.DownloadOnly = $true
        }
        if ($Force) {
            $installArgs.Force = $true
        }

        Install-NvidiaSoftware @installArgs
    }
    'registry' {
        $registryArgs = @{}
        if ($RegistryPath) {
            $registryArgs.RegistryPath = $RegistryPath
        }

        Get-NvidiaSoftwareRegistry @registryArgs
    }
}
