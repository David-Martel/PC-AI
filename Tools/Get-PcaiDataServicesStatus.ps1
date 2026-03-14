#Requires -PSEdition Core
[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$commonModulePath = Join-Path $repoRoot 'Modules\PC-AI.Common\PC-AI.Common.psm1'
Import-Module $commonModulePath -Force | Out-Null

function Resolve-ExecutablePath {
    param([string[]]$Candidates)

    foreach ($candidate in @($Candidates)) {
        if ([string]::IsNullOrWhiteSpace([string]$candidate)) {
            continue
        }

        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            try {
                return (Resolve-Path -LiteralPath $candidate -ErrorAction Stop).Path
            } catch {
                return $candidate
            }
        }

        $command = Get-Command $candidate -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($command -and $command.Path) {
            return $command.Path
        }
    }

    return $null
}

function Get-ServiceSnapshot {
    param([string[]]$Names)

    foreach ($name in @($Names)) {
        if ([string]::IsNullOrWhiteSpace([string]$name)) {
            continue
        }

        $service = Get-Service -Name $name -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($service) {
            return [pscustomobject]@{
                Name      = $service.Name
                DisplayName = $service.DisplayName
                Status    = [string]$service.Status
                StartType = [string]$service.StartType
            }
        }
    }

    return $null
}

$redisStatus = Get-PcaiExternalCacheStatus -Refresh
$postgresService = Get-ServiceSnapshot -Names @('postgresql-x64-18', 'postgresql-x64-17', 'postgresql')
$postgresPath = Resolve-ExecutablePath -Candidates @(
    'C:\Program Files\PostgreSQL\18\bin\psql.exe',
    'psql.exe',
    'psql'
)
$redisService = Get-ServiceSnapshot -Names @('Redis')
$redisCliPath = Resolve-ExecutablePath -Candidates @(
    $redisStatus.RedisCliPath,
    'C:\Program Files\Redis\redis-cli.exe',
    'T:\projects\redis-windows\redis-cli.exe',
    'redis-cli.exe',
    'redis-cli'
)

[pscustomobject]@{
    Timestamp = (Get-Date).ToString('o')
    Redis = [pscustomobject]@{
        ProviderStatus = $redisStatus
        Service        = $redisService
        CliPath        = $redisCliPath
        PreferredPort  = 6380
        WslDevPort     = 6379
    }
    Postgres = [pscustomobject]@{
        Service       = $postgresService
        CliPath       = $postgresPath
        PreferredPort = 5432
    }
}
