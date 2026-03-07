function Write-Success {
    param([string]$Message)
    Write-Host $Message -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "WARNING: $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "ERROR: $Message" -ForegroundColor Red
}

function Write-Info {
    param([string]$Message)
    Write-Host $Message -ForegroundColor Cyan
}

function Write-Header {
    param([string]$Message)
    Write-Host ""
    Write-Host "=== $Message ===" -ForegroundColor Magenta
    Write-Host ""
}

function Write-SubHeader {
    param([string]$Message)
    Write-Host "--- $Message ---" -ForegroundColor DarkCyan
}

function Write-Bullet {
    param([string]$Message, [string]$Color = 'White')
    Write-Host "  * $Message" -ForegroundColor $Color
}

# Singleton initialization state
$script:PcaiNativeInitialized = $false
$script:PcaiAccelerationManifestPath = $null

function Resolve-PcaiAccelerationManifestPath {
    if ($script:PcaiAccelerationManifestPath -and (Test-Path -LiteralPath $script:PcaiAccelerationManifestPath)) {
        return $script:PcaiAccelerationManifestPath
    }

    $moduleRoot = Split-Path -Parent $PSScriptRoot
    $candidates = @(
        (Join-Path $moduleRoot 'PC-AI.Acceleration\PC-AI.Acceleration.psd1'),
        (Join-Path (Join-Path $env:LOCALAPPDATA 'PowerShell\Modules') 'PC-AI.Acceleration\PC-AI.Acceleration.psd1')
    )

    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate) {
            $script:PcaiAccelerationManifestPath = $candidate
            return $candidate
        }
    }

    $availableModule = Get-Module -ListAvailable -Name 'PC-AI.Acceleration' | Sort-Object Version -Descending | Select-Object -First 1
    if ($availableModule -and $availableModule.Path) {
        $script:PcaiAccelerationManifestPath = $availableModule.Path
        return $availableModule.Path
    }

    return $null
}

function Initialize-PcaiNative {
    [CmdletBinding()]
    param([switch]$Force)

    if ($script:PcaiNativeInitialized -and -not $Force) {
        return $true
    }

    $accelModule = Get-Module PC-AI.Acceleration
    if (-not $accelModule) {
        $accelManifest = Resolve-PcaiAccelerationManifestPath
        if ($accelManifest) {
            $accelModule = Import-Module -Name $accelManifest -PassThru -ErrorAction SilentlyContinue
        }
    }

    if ($accelModule) {
        $result = & $accelModule { Initialize-PcaiNative -Force:$Force }
        $script:PcaiNativeInitialized = $result
        return $result
    }

    Write-Error "PC-AI.Acceleration module not found. Native initialization failed."
    return $false
}

Export-ModuleMember -Function Write-Success, Write-Warning, Write-Error, Write-Info, Write-Header, Write-SubHeader, Write-Bullet, Initialize-PcaiNative
