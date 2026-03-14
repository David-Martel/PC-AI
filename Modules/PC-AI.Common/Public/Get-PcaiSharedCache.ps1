#Requires -PSEdition Core

if (-not (Get-Variable -Name 'PcaiSharedCache' -Scope Global -ErrorAction SilentlyContinue) -or
    -not $global:PcaiSharedCache -or
    -not ($global:PcaiSharedCache -is [System.Collections.IDictionary]) -or
    -not $global:PcaiSharedCache.Contains('MaxEntries') -or
    -not $global:PcaiSharedCache.Contains('Entries') -or
    -not ($global:PcaiSharedCache.Entries -is [System.Collections.Specialized.OrderedDictionary])) {
    $global:PcaiSharedCache = @{
        MaxEntries = 256
        Entries    = [System.Collections.Specialized.OrderedDictionary]::new()
    }
}

if (-not (Get-Variable -Name 'PcaiSharedCacheProviderState' -Scope Global -ErrorAction SilentlyContinue) -or
    -not $global:PcaiSharedCacheProviderState -or
    -not ($global:PcaiSharedCacheProviderState -is [System.Collections.IDictionary])) {
    $global:PcaiSharedCacheProviderState = @{
        Signature    = ''
        Status       = $null
        CheckedAtUtc = [datetime]::MinValue
    }
}

$script:PcaiSharedCache = $global:PcaiSharedCache
$script:PcaiSharedCacheProviderState = $global:PcaiSharedCacheProviderState
$script:PcaiExternalCacheHealthTtlSeconds = 5

function Copy-PcaiCacheValue {
    [CmdletBinding()]
    param(
        [Parameter()]
        [AllowNull()]
        [object]$Value
    )

    if ($null -eq $Value) {
        return $null
    }

    if ($Value -is [string] -or
        $Value -is [ValueType] -or
        $Value -is [datetime] -or
        $Value -is [timespan] -or
        $Value -is [guid]) {
        return $Value
    }

    if ($Value -is [System.Collections.IDictionary]) {
        $copy = [ordered]@{}
        foreach ($key in $Value.Keys) {
            $copy[$key] = Copy-PcaiCacheValue -Value $Value[$key]
        }
        return $copy
    }

    if ($Value -is [System.Management.Automation.PSCustomObject] -or
        ($Value -is [psobject] -and -not ($Value -is [System.Collections.IEnumerable]))) {
        $copy = [ordered]@{}
        foreach ($prop in $Value.PSObject.Properties) {
            if ($prop.MemberType -notin @(
                    [System.Management.Automation.PSMemberTypes]::NoteProperty,
                    [System.Management.Automation.PSMemberTypes]::Property,
                    [System.Management.Automation.PSMemberTypes]::AliasProperty
                )) {
                continue
            }
            $copy[$prop.Name] = Copy-PcaiCacheValue -Value $prop.Value
        }
        return [PSCustomObject]$copy
    }

    if ($Value -is [System.Collections.IEnumerable] -and -not ($Value -is [string])) {
        return @($Value | ForEach-Object { Copy-PcaiCacheValue -Value $_ })
    }

    return $Value
}

function Remove-StalePcaiCacheEntries {
    [CmdletBinding()]
    param()

    while ($script:PcaiSharedCache.Entries.Count -gt $script:PcaiSharedCache.MaxEntries) {
        $oldest = $null
        foreach ($candidate in $script:PcaiSharedCache.Entries.GetEnumerator()) {
            if (-not $oldest -or $candidate.Value.LastAccessedUtc -lt $oldest.Value.LastAccessedUtc) {
                $oldest = $candidate
            }
        }

        if (-not $oldest) {
            break
        }

        $script:PcaiSharedCache.Entries.Remove($oldest.Key)
    }
}

function Set-PcaiLocalSharedCacheEntry {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$FullKey,

        [Parameter(Mandatory)]
        [AllowNull()]
        [object]$Value,

        [Parameter()]
        [string]$DependencyStamp,

        [Parameter()]
        [datetime]$CreatedUtc = [datetime]::UtcNow,

        [Parameter()]
        [datetime]$LastAccessedUtc = [datetime]::UtcNow
    )

    if ($script:PcaiSharedCache.Entries.Contains($FullKey)) {
        $script:PcaiSharedCache.Entries.Remove($FullKey)
    }

    $script:PcaiSharedCache.Entries.Add($FullKey, [PSCustomObject]@{
            Value           = Copy-PcaiCacheValue -Value $Value
            DependencyStamp = $DependencyStamp
            CreatedUtc      = $CreatedUtc
            LastAccessedUtc = $LastAccessedUtc
        })

    Remove-StalePcaiCacheEntries
}

function Get-PcaiDependencyStamp {
    [CmdletBinding()]
    [OutputType([string])]
    param(
        [Parameter()]
        [AllowEmptyCollection()]
        [object[]]$InputObject
    )

    if (-not $InputObject -or $InputObject.Count -eq 0) {
        return '<none>'
    }

    $parts = New-Object System.Collections.Generic.List[string]
    foreach ($item in $InputObject) {
        if ($null -eq $item) {
            $parts.Add('<null>')
            continue
        }

        switch ($item) {
            { $_ -is [System.IO.FileSystemInfo] } {
                $parts.Add(('{0}|{1}|{2}|{3}' -f $_.FullName, $_.Exists, $_.Length, $_.LastWriteTimeUtc.Ticks))
                continue
            }
            { $_ -is [string] } {
                $literal = [string]$item
                if ([string]::IsNullOrWhiteSpace($literal)) {
                    $parts.Add('<empty>')
                } elseif (Test-Path -LiteralPath $literal) {
                    $resolvedItem = Get-Item -LiteralPath $literal -ErrorAction SilentlyContinue
                    if ($resolvedItem) {
                        $parts.Add(('{0}|{1}|{2}|{3}' -f $resolvedItem.FullName, $resolvedItem.Exists, $resolvedItem.Length, $resolvedItem.LastWriteTimeUtc.Ticks))
                    } else {
                        $parts.Add("missing::$literal")
                    }
                } else {
                    $parts.Add("missing::$literal")
                }
                continue
            }
            default {
                $parts.Add(($item | Out-String).Trim())
            }
        }
    }

    $payload = ($parts | Sort-Object) -join "`n"
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try {
        $bytes = [System.Text.Encoding]::UTF8.GetBytes($payload)
        return [Convert]::ToHexString($sha.ComputeHash($bytes))
    }
    finally {
        $sha.Dispose()
    }
}

function Get-PcaiExternalCacheConfig {
    [CmdletBinding()]
    [OutputType([pscustomobject])]
    param()

    $rawProvider = [string]$env:PCAI_CACHE_PROVIDER
    $provider = if ([string]::IsNullOrWhiteSpace($rawProvider)) { 'memory' } else { $rawProvider.Trim().ToLowerInvariant() }
    $redisEnabled = $provider -in @('redis', 'hybrid', 'auto')

    $redisHost = if ([string]::IsNullOrWhiteSpace([string]$env:PCAI_REDIS_HOST)) { '127.0.0.1' } else { [string]$env:PCAI_REDIS_HOST }
    $redisPort = 6380
    if ($env:PCAI_REDIS_PORT) {
        $parsedPort = 0
        if ([int]::TryParse([string]$env:PCAI_REDIS_PORT, [ref]$parsedPort) -and $parsedPort -gt 0) {
            $redisPort = $parsedPort
        }
    }

    $timeoutMs = 1200
    if ($env:PCAI_REDIS_TIMEOUT_MS) {
        $parsedTimeout = 0
        if ([int]::TryParse([string]$env:PCAI_REDIS_TIMEOUT_MS, [ref]$parsedTimeout) -and $parsedTimeout -gt 0) {
            $timeoutMs = $parsedTimeout
        }
    }

    $keyPrefix = if ([string]::IsNullOrWhiteSpace([string]$env:PCAI_REDIS_KEY_PREFIX)) {
        'pcai:cache:'
    } else {
        [string]$env:PCAI_REDIS_KEY_PREFIX
    }

    $redisCliPath = $null
    $cliCandidates = @(
        [string]$env:PCAI_REDIS_CLI_PATH,
        'C:\Program Files\Redis\redis-cli.exe',
        'T:\projects\redis-windows\redis-cli.exe',
        (Join-Path $env:USERPROFILE 'bin\redis-cli.exe'),
        (Join-Path $env:LOCALAPPDATA 'Programs\Redis\redis-cli.exe')
    ) | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) }

    foreach ($candidate in $cliCandidates) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            try {
                $redisCliPath = (Resolve-Path -LiteralPath $candidate -ErrorAction Stop).Path
            } catch {
                $redisCliPath = $candidate
            }
            break
        }
    }

    if (-not $redisCliPath) {
        foreach ($commandName in @('redis-cli.exe', 'redis-cli')) {
            $command = Get-Command $commandName -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
            if ($command -and $command.Path) {
                $redisCliPath = $command.Path
                break
            }
        }
    }

    [pscustomobject]@{
        Provider    = if ($redisEnabled) { 'redis' } else { 'memory' }
        Enabled     = $redisEnabled
        RedisCliPath = $redisCliPath
        RedisHost   = $redisHost
        RedisPort   = $redisPort
        KeyPrefix   = $keyPrefix
        TimeoutMs   = $timeoutMs
    }
}

function Get-PcaiExternalCacheSignature {
    [CmdletBinding()]
    [OutputType([string])]
    param(
        [Parameter(Mandatory)]
        [pscustomobject]$Config
    )

    return '{0}|{1}|{2}|{3}|{4}|{5}' -f
        $Config.Provider,
        $Config.Enabled,
        $Config.RedisCliPath,
        $Config.RedisHost,
        $Config.RedisPort,
        $Config.KeyPrefix
}

function Invoke-PcaiProcessCapture {
    [CmdletBinding()]
    [OutputType([pscustomobject])]
    param(
        [Parameter(Mandatory)]
        [string]$FilePath,

        [Parameter()]
        [string[]]$Arguments = @(),

        [Parameter()]
        [int]$TimeoutMs = 1200
    )

    if (-not (Test-Path -LiteralPath $FilePath -PathType Leaf)) {
        return [pscustomobject]@{
            Success  = $false
            ExitCode = -1
            TimedOut = $false
            StdOut   = ''
            StdErr   = "File not found: $FilePath"
        }
    }

    $process = $null
    try {
        $psi = [System.Diagnostics.ProcessStartInfo]::new()
        $psi.FileName = $FilePath
        $psi.UseShellExecute = $false
        $psi.RedirectStandardOutput = $true
        $psi.RedirectStandardError = $true
        $psi.CreateNoWindow = $true
        foreach ($arg in @($Arguments)) {
            [void]$psi.ArgumentList.Add([string]$arg)
        }

        $process = [System.Diagnostics.Process]::new()
        $process.StartInfo = $psi
        [void]$process.Start()

        if (-not $process.WaitForExit($TimeoutMs)) {
            try {
                $process.Kill($true)
            } catch {
            }

            return [pscustomobject]@{
                Success  = $false
                ExitCode = -2
                TimedOut = $true
                StdOut   = ''
                StdErr   = "Timed out after ${TimeoutMs}ms"
            }
        }

        $stdout = $process.StandardOutput.ReadToEnd().Trim()
        $stderr = $process.StandardError.ReadToEnd().Trim()
        return [pscustomobject]@{
            Success  = ($process.ExitCode -eq 0)
            ExitCode = $process.ExitCode
            TimedOut = $false
            StdOut   = $stdout
            StdErr   = $stderr
        }
    }
    catch {
        return [pscustomobject]@{
            Success  = $false
            ExitCode = -3
            TimedOut = $false
            StdOut   = ''
            StdErr   = $_.Exception.Message
        }
    }
    finally {
        if ($process) {
            $process.Dispose()
        }
    }
}

function Invoke-PcaiRedisCli {
    [CmdletBinding()]
    [OutputType([pscustomobject])]
    param(
        [Parameter(Mandatory)]
        [pscustomobject]$Config,

        [Parameter(Mandatory)]
        [string[]]$Arguments,

        [Parameter()]
        [int]$TimeoutMs
    )

    if (-not $Config.Enabled -or [string]::IsNullOrWhiteSpace([string]$Config.RedisCliPath)) {
        return [pscustomobject]@{
            Success  = $false
            ExitCode = -1
            TimedOut = $false
            StdOut   = ''
            StdErr   = 'Redis CLI unavailable.'
        }
    }

    $effectiveTimeoutMs = if ($PSBoundParameters.ContainsKey('TimeoutMs') -and $TimeoutMs -gt 0) { $TimeoutMs } else { [int]$Config.TimeoutMs }
    $redisArguments = @(
        '-h', $Config.RedisHost,
        '-p', ([string]$Config.RedisPort),
        '--raw'
    ) + @($Arguments)

    Invoke-PcaiProcessCapture -FilePath $Config.RedisCliPath -Arguments $redisArguments -TimeoutMs $effectiveTimeoutMs
}

function Get-PcaiExternalCacheKey {
    [CmdletBinding()]
    [OutputType([string])]
    param(
        [Parameter(Mandatory)]
        [pscustomobject]$Config,

        [Parameter(Mandatory)]
        [string]$Namespace,

        [Parameter(Mandatory)]
        [string]$Key
    )

    return '{0}{1}::{2}' -f $Config.KeyPrefix, $Namespace, $Key
}

function ConvertTo-PcaiExternalCachePayload {
    [CmdletBinding()]
    [OutputType([string])]
    param(
        [Parameter(Mandatory)]
        [AllowNull()]
        [object]$Value,

        [Parameter()]
        [string]$DependencyStamp,

        [Parameter()]
        [datetime]$CreatedUtc = [datetime]::UtcNow,

        [Parameter()]
        [datetime]$LastAccessedUtc = [datetime]::UtcNow
    )

    $payload = [ordered]@{
        Value           = Copy-PcaiCacheValue -Value $Value
        DependencyStamp = $DependencyStamp
        CreatedUtc      = $CreatedUtc.ToString('o')
        LastAccessedUtc = $LastAccessedUtc.ToString('o')
    }

    return ($payload | ConvertTo-Json -Depth 20 -Compress)
}

function ConvertFrom-PcaiExternalCachePayload {
    [CmdletBinding()]
    [OutputType([pscustomobject])]
    param(
        [Parameter(Mandatory)]
        [string]$PayloadJson
    )

    if ([string]::IsNullOrWhiteSpace($PayloadJson)) {
        return $null
    }

    try {
        $parsed = $PayloadJson | ConvertFrom-Json -AsHashtable -Depth 20 -ErrorAction Stop
        if (-not $parsed) {
            return $null
        }

        $createdUtc = [datetime]::UtcNow
        $lastAccessedUtc = [datetime]::UtcNow
        if ($parsed.ContainsKey('CreatedUtc') -and -not [string]::IsNullOrWhiteSpace([string]$parsed.CreatedUtc)) {
            $createdUtc = [datetime]::Parse([string]$parsed.CreatedUtc, [Globalization.CultureInfo]::InvariantCulture, [Globalization.DateTimeStyles]::RoundtripKind)
        }
        if ($parsed.ContainsKey('LastAccessedUtc') -and -not [string]::IsNullOrWhiteSpace([string]$parsed.LastAccessedUtc)) {
            $lastAccessedUtc = [datetime]::Parse([string]$parsed.LastAccessedUtc, [Globalization.CultureInfo]::InvariantCulture, [Globalization.DateTimeStyles]::RoundtripKind)
        }

        [pscustomobject]@{
            Value           = if ($parsed.ContainsKey('Value')) { Copy-PcaiCacheValue -Value $parsed.Value } else { $null }
            DependencyStamp = if ($parsed.ContainsKey('DependencyStamp')) { [string]$parsed.DependencyStamp } else { $null }
            CreatedUtc      = $createdUtc
            LastAccessedUtc = $lastAccessedUtc
        }
    }
    catch {
        return $null
    }
}

function Remove-PcaiExternalCacheEntry {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [pscustomobject]$Config,

        [Parameter(Mandatory)]
        [string]$Namespace,

        [Parameter(Mandatory)]
        [string]$Key
    )

    $redisKey = Get-PcaiExternalCacheKey -Config $Config -Namespace $Namespace -Key $Key
    Invoke-PcaiRedisCli -Config $Config -Arguments @('DEL', $redisKey) | Out-Null
}

function Get-PcaiExternalCacheStatus {
    [CmdletBinding()]
    [OutputType([pscustomobject])]
    param(
        [Parameter()]
        [switch]$Refresh
    )

    $config = Get-PcaiExternalCacheConfig
    $signature = Get-PcaiExternalCacheSignature -Config $config
    $now = [datetime]::UtcNow

    if (-not $Refresh -and
        $script:PcaiSharedCacheProviderState.Status -and
        $script:PcaiSharedCacheProviderState.Signature -eq $signature -and
        ($now - $script:PcaiSharedCacheProviderState.CheckedAtUtc).TotalSeconds -lt $script:PcaiExternalCacheHealthTtlSeconds) {
        return $script:PcaiSharedCacheProviderState.Status
    }

    $status = [ordered]@{
        Provider     = $config.Provider
        Enabled      = [bool]$config.Enabled
        Available    = $false
        Reachable    = $false
        RedisCliPath = $config.RedisCliPath
        Host         = $config.RedisHost
        Port         = $config.RedisPort
        KeyPrefix    = $config.KeyPrefix
        Error        = $null
    }

    if ($config.Enabled) {
        if ([string]::IsNullOrWhiteSpace([string]$config.RedisCliPath)) {
            $status.Error = 'redis-cli executable not found.'
        } else {
            $ping = Invoke-PcaiRedisCli -Config $config -Arguments @('PING')
            if ($ping.Success -and $ping.StdOut -eq 'PONG') {
                $status.Available = $true
                $status.Reachable = $true
            } else {
                $status.Error = if ($ping.StdErr) { $ping.StdErr } elseif ($ping.StdOut) { $ping.StdOut } else { 'Redis ping failed.' }
            }
        }
    }

    $cachedStatus = [pscustomobject]$status
    $script:PcaiSharedCacheProviderState.Signature = $signature
    $script:PcaiSharedCacheProviderState.Status = $cachedStatus
    $script:PcaiSharedCacheProviderState.CheckedAtUtc = $now
    return $cachedStatus
}

function Get-PcaiExternalCacheEntryRecord {
    [CmdletBinding()]
    [OutputType([pscustomobject])]
    param(
        [Parameter(Mandatory)]
        [string]$Namespace,

        [Parameter(Mandatory)]
        [string]$Key,

        [Parameter()]
        [int]$TtlSeconds = 0,

        [Parameter()]
        [string]$DependencyStamp
    )

    $status = Get-PcaiExternalCacheStatus
    if (-not $status.Enabled -or -not $status.Available) {
        return $null
    }

    $config = Get-PcaiExternalCacheConfig
    $redisKey = Get-PcaiExternalCacheKey -Config $config -Namespace $Namespace -Key $Key
    $result = Invoke-PcaiRedisCli -Config $config -Arguments @('GET', $redisKey)
    if (-not $result.Success -or [string]::IsNullOrWhiteSpace($result.StdOut)) {
        return $null
    }

    $entry = ConvertFrom-PcaiExternalCachePayload -PayloadJson $result.StdOut
    if (-not $entry) {
        Remove-PcaiExternalCacheEntry -Config $config -Namespace $Namespace -Key $Key
        return $null
    }

    $now = [datetime]::UtcNow
    if ($TtlSeconds -gt 0 -and ($now - $entry.CreatedUtc).TotalSeconds -gt $TtlSeconds) {
        Remove-PcaiExternalCacheEntry -Config $config -Namespace $Namespace -Key $Key
        return $null
    }

    if ($PSBoundParameters.ContainsKey('DependencyStamp') -and $DependencyStamp -ne $entry.DependencyStamp) {
        Remove-PcaiExternalCacheEntry -Config $config -Namespace $Namespace -Key $Key
        return $null
    }

    return $entry
}

function Set-PcaiExternalCacheEntry {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$Namespace,

        [Parameter(Mandatory)]
        [string]$Key,

        [Parameter(Mandatory)]
        [AllowNull()]
        [object]$Value,

        [Parameter()]
        [string]$DependencyStamp,

        [Parameter()]
        [int]$TtlSeconds = 0
    )

    $status = Get-PcaiExternalCacheStatus
    if (-not $status.Enabled -or -not $status.Available) {
        return $false
    }

    $config = Get-PcaiExternalCacheConfig
    $payloadJson = ConvertTo-PcaiExternalCachePayload -Value $Value -DependencyStamp $DependencyStamp
    $redisKey = Get-PcaiExternalCacheKey -Config $config -Namespace $Namespace -Key $Key
    $arguments = @('SET', $redisKey, $payloadJson)
    if ($TtlSeconds -gt 0) {
        $arguments += @('EX', ([string]$TtlSeconds))
    }

    $result = Invoke-PcaiRedisCli -Config $config -Arguments $arguments
    return [bool]($result.Success -and $result.StdOut -eq 'OK')
}

function Clear-PcaiExternalCache {
    [CmdletBinding()]
    param(
        [Parameter()]
        [string]$Namespace
    )

    $status = Get-PcaiExternalCacheStatus
    if (-not $status.Enabled -or -not $status.Available) {
        return
    }

    $config = Get-PcaiExternalCacheConfig
    $pattern = if ([string]::IsNullOrWhiteSpace($Namespace)) {
        "$($config.KeyPrefix)*"
    } else {
        "$($config.KeyPrefix)$Namespace::*"
    }

    $scan = Invoke-PcaiRedisCli -Config $config -Arguments @('--scan', '--pattern', $pattern) -TimeoutMs ([Math]::Max($config.TimeoutMs, 3000))
    if (-not $scan.Success -or [string]::IsNullOrWhiteSpace($scan.StdOut)) {
        return
    }

    $keys = @($scan.StdOut -split "`r?`n" | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
    foreach ($key in $keys) {
        Invoke-PcaiRedisCli -Config $config -Arguments @('DEL', [string]$key) | Out-Null
    }
}

function Get-PcaiSharedCacheEntry {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$Namespace,

        [Parameter(Mandatory)]
        [string]$Key,

        [Parameter()]
        [int]$TtlSeconds = 0,

        [Parameter()]
        [string]$DependencyStamp
    )

    $fullKey = '{0}::{1}' -f $Namespace, $Key
    if ($script:PcaiSharedCache.Entries.Contains($fullKey)) {
        $entry = $script:PcaiSharedCache.Entries[$fullKey]
        $now = [datetime]::UtcNow
        if ($TtlSeconds -gt 0 -and ($now - $entry.CreatedUtc).TotalSeconds -gt $TtlSeconds) {
            $script:PcaiSharedCache.Entries.Remove($fullKey)
        } elseif ($PSBoundParameters.ContainsKey('DependencyStamp') -and $DependencyStamp -ne $entry.DependencyStamp) {
            $script:PcaiSharedCache.Entries.Remove($fullKey)
        } else {
            $entry.LastAccessedUtc = $now
            return (Copy-PcaiCacheValue -Value $entry.Value)
        }
    }

    $externalEntry = Get-PcaiExternalCacheEntryRecord -Namespace $Namespace -Key $Key -TtlSeconds $TtlSeconds -DependencyStamp $DependencyStamp
    if (-not $externalEntry) {
        return $null
    }

    Set-PcaiLocalSharedCacheEntry -FullKey $fullKey -Value $externalEntry.Value -DependencyStamp $externalEntry.DependencyStamp -CreatedUtc $externalEntry.CreatedUtc -LastAccessedUtc ([datetime]::UtcNow)
    return (Copy-PcaiCacheValue -Value $externalEntry.Value)
}

function Set-PcaiSharedCacheEntry {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$Namespace,

        [Parameter(Mandatory)]
        [string]$Key,

        [Parameter(Mandatory)]
        [AllowNull()]
        [object]$Value,

        [Parameter()]
        [string]$DependencyStamp,

        [Parameter()]
        [int]$TtlSeconds = 0
    )

    $fullKey = '{0}::{1}' -f $Namespace, $Key
    $now = [datetime]::UtcNow

    Set-PcaiLocalSharedCacheEntry -FullKey $fullKey -Value $Value -DependencyStamp $DependencyStamp -CreatedUtc $now -LastAccessedUtc $now
    [void](Set-PcaiExternalCacheEntry -Namespace $Namespace -Key $Key -Value $Value -DependencyStamp $DependencyStamp -TtlSeconds $TtlSeconds)

    return (Copy-PcaiCacheValue -Value $Value)
}

function Clear-PcaiSharedCache {
    [CmdletBinding()]
    param(
        [Parameter()]
        [string]$Namespace
    )

    if ([string]::IsNullOrWhiteSpace($Namespace)) {
        $script:PcaiSharedCache.Entries.Clear()
        Clear-PcaiExternalCache
        return
    }

    foreach ($key in @($script:PcaiSharedCache.Entries.Keys)) {
        if ($key -like "$Namespace::*") {
            $script:PcaiSharedCache.Entries.Remove($key)
        }
    }

    Clear-PcaiExternalCache -Namespace $Namespace
}
