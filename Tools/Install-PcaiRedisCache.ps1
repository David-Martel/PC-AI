#Requires -PSEdition Core
[CmdletBinding()]
param(
    [string]$ServiceName = 'Redis',
    [string]$DisplayName = 'Redis for Windows Agent',
    [switch]$Uninstall,
    [switch]$SkipVerify
)

$ErrorActionPreference = 'Stop'

$redisWindowsRoot = 'T:\projects\redis-windows'
$installScript = Join-Path $redisWindowsRoot 'install-redis-service.ps1'

if (Test-Path -LiteralPath $installScript -PathType Leaf) {
    $arguments = @{
        ServiceName = $ServiceName
        DisplayName = $DisplayName
    }
    if ($Uninstall) {
        $arguments.Uninstall = $true
    }
    if ($SkipVerify) {
        $arguments.SkipVerify = $true
    }

    & $installScript @arguments
    exit $LASTEXITCODE
}

$service = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue
if ($service) {
    [pscustomobject]@{
        Installed = $true
        Source    = 'ExistingService'
        Name      = $service.Name
        Status    = [string]$service.Status
        Message   = 'Redis service already exists. The redis-windows installer root was not found.'
    }
    exit 0
}

throw "Redis installer not found at '$installScript'. Clone or mount T:\projects\redis-windows, or install the Redis service manually before enabling PCAI_CACHE_PROVIDER=redis."
