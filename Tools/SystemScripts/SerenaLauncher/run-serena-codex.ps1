[CmdletBinding()]
param(
    [switch]$AllowManagedPostgresStartup,
    [switch]$DryRun,
    [Alias('h', '-help')][switch]$Help,
    [Parameter(ValueFromRemainingArguments = $true, Position = 0)]
    [string[]]$SerenaArgs
)

if (@($SerenaArgs) -ccontains '--help') {
    $Help = $true
    $SerenaArgs = @($SerenaArgs | Where-Object { $_ -cne '--help' })
}
if ($Help) {
    [Console]::Error.WriteLine('Serena launcher: [-AllowManagedPostgresStartup] [Serena arguments]. Default never starts Docker for an unavailable local database. -DryRun returns before IO.')
    return
}
if ($DryRun) {
    [Console]::Error.WriteLine('DRY_RUN_NO_IO: no path, environment, database, Docker or server access; managed PostgreSQL startup requires explicit opt-in.')
    return
}
$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$SerenaWrapperJs = Join-Path $PSScriptRoot "run-serena-codex.js"
$ManagedContainerName = "serena-source-postgres"
$ManagedVolumeName = "serena-source-postgres-data"

function Write-Stderr {
    param([string]$Message)
    [Console]::Error.WriteLine($Message)
}

function Test-TcpPort {
    param(
        [string]$Address,
        [int]$Port
    )

    $client = [System.Net.Sockets.TcpClient]::new()
    try {
        $async = $client.BeginConnect($Address, $Port, $null, $null)
        if (-not $async.AsyncWaitHandle.WaitOne(1500, $false)) {
            return $false
        }
        $null = $client.EndConnect($async)
        return $true
    } catch {
        return $false
    } finally {
        $client.Dispose()
    }
}

function Get-PostgresConfig {
    param([string]$Url)

    if (-not $Url) {
        return $null
    }

    $pattern = '^postgres(?:ql)?://(?:(?<user>[^:/?#]+)(?::(?<pass>[^@/?#]*))?@)?(?<host>[^:/?#]+)(?::(?<port>\d+))?/(?<db>[^?]+)'
    $match = [regex]::Match($Url, $pattern)
    if (-not $match.Success) {
        return $null
    }

    $user = if ($match.Groups["user"].Success) { $match.Groups["user"].Value } else { "postgres" }
    $pass = if ($match.Groups["pass"].Success) { $match.Groups["pass"].Value } else { "password" }
    $dbHost = $match.Groups["host"].Value
    $port = if ($match.Groups["port"].Success) { [int]$match.Groups["port"].Value } else { 5432 }
    $db = $match.Groups["db"].Value

    [PSCustomObject]@{
        User = $user
        Password = $pass
        Host = $dbHost
        Port = $port
        Database = $db
    }
}

function Invoke-SerenaDockerCommand {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$Executable,
        [Parameter(Mandatory)][string[]]$ArgumentList
    )
    # One literal executable with native argument array; callers suppress normal
    # run/start output so startup cannot contaminate MCP stdout.
    & $Executable @ArgumentList
    if ($LASTEXITCODE -ne 0) {
        throw 'Managed PostgreSQL Docker command failed.'
    }
}
function Ensure-ManagedPostgres {
    param([string]$Url, [switch]$AllowManagedPostgresStartup)

    $config = Get-PostgresConfig -Url $Url
    if ($null -eq $config) {
        return
    }

    $address = $config.Host.ToLowerInvariant()
    $isLocalHost = $address -in @("127.0.0.1", "localhost")
    if (-not $isLocalHost) {
        return
    }

    if (Test-TcpPort -Address $config.Host -Port $config.Port) {
        return
    }

    if (-not $AllowManagedPostgresStartup) {
        throw 'Local PostgreSQL is unavailable. Automatic Docker startup is disabled; start the required database independently or explicitly opt in with -AllowManagedPostgresStartup.'
    }
    $docker = Get-Command docker.exe -CommandType Application -ErrorAction SilentlyContinue
    if (-not $docker) {
        throw "PostgreSQL at $($config.Host):$($config.Port) is unavailable and docker.exe was not found."
    }

    $containerIdOutput = Invoke-SerenaDockerCommand -Executable $docker.Source -ArgumentList @('ps', '-aq', '-f', "name=^${ManagedContainerName}$")
    $containerId = "$containerIdOutput".Trim()
    if (-not $containerId) {
        Invoke-SerenaDockerCommand -Executable $docker.Source -ArgumentList @(
            'run', '--detach', '--name', $ManagedContainerName,
            '--restart', 'unless-stopped',
            '--env', "POSTGRES_USER=$($config.User)",
            '--env', "POSTGRES_PASSWORD=$($config.Password)",
            '--env', "POSTGRES_DB=$($config.Database)",
            '--publish', "127.0.0.1:$($config.Port):5432",
            '--volume', "${ManagedVolumeName}:/var/lib/postgresql/data", 'postgres:17'
        ) | Out-Null
    } else {
        Invoke-SerenaDockerCommand -Executable $docker.Source -ArgumentList @('start', $ManagedContainerName) | Out-Null
    }

    for ($attempt = 0; $attempt -lt 45; $attempt++) {
        if (Test-TcpPort -Address $config.Host -Port $config.Port) {
            return
        }
        Start-Sleep -Seconds 1
    }

    throw "Managed PostgreSQL container '$ManagedContainerName' did not become reachable at $($config.Host):$($config.Port)."
}

try {
    if (-not (Test-Path -LiteralPath $SerenaWrapperJs -PathType Leaf)) {
        throw "Serena transport shim not found at $SerenaWrapperJs."
    }

    if (
        [string]::IsNullOrWhiteSpace($env:SERENA_WEB_LOG_POSTGRES_URL) -and
        -not [string]::IsNullOrWhiteSpace($env:SERENA_POSTGRES_URL)
    ) {
        $env:SERENA_WEB_LOG_POSTGRES_URL = $env:SERENA_POSTGRES_URL
    }

    Ensure-ManagedPostgres -Url $env:SERENA_POSTGRES_URL -AllowManagedPostgresStartup:$AllowManagedPostgresStartup

    $node = Get-Command node.exe -CommandType Application -ErrorAction SilentlyContinue
    if (-not $node) {
        throw "node.exe was not found on PATH."
    }

    & $node.Source $SerenaWrapperJs @SerenaArgs
    exit $LASTEXITCODE
} catch {
    Write-Stderr $_.Exception.Message
    exit 1
}
