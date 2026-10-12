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
        foreach ($key in $Value.PSBase.Keys) {
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
        # Emit one array object, including empty/singleton arrays and null elements.
        $copy = [System.Collections.Generic.List[object]]::new()
        foreach ($item in $Value) {
            $copy.Add((Copy-PcaiCacheValue -Value $item))
        }
        return ,($copy.ToArray())
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
        [datetime]$LastAccessedUtc = [datetime]::UtcNow,

        [Parameter()]
        [AllowNull()]
        [Nullable[datetime]]$ExpiresUtc
    )

    if ($script:PcaiSharedCache.Entries.Contains($FullKey)) {
        $script:PcaiSharedCache.Entries.Remove($FullKey)
    }

    $script:PcaiSharedCache.Entries.Add($FullKey, [PSCustomObject]@{
            Value           = Copy-PcaiCacheValue -Value $Value
            DependencyStamp = $DependencyStamp
            CreatedUtc      = $CreatedUtc
            LastAccessedUtc = $LastAccessedUtc
            ExpiresUtc      = $ExpiresUtc
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
                $length = if ($_ -is [System.IO.FileInfo]) { $_.Length } else { 0 }
                $parts.Add(('{0}|{1}|{2}|{3}' -f $_.FullName, $_.Exists, $length, $_.LastWriteTimeUtc.Ticks))
                continue
            }
            { $_ -is [string] } {
                $literal = [string]$item
                if ([string]::IsNullOrWhiteSpace($literal)) {
                    $parts.Add('<empty>')
                } elseif (Test-Path -LiteralPath $literal) {
                    $resolvedItem = Get-Item -LiteralPath $literal -ErrorAction SilentlyContinue
                    if ($resolvedItem) {
                        $length = if ($resolvedItem -is [System.IO.FileInfo]) { $resolvedItem.Length } else { 0 }
                        $parts.Add(('{0}|{1}|{2}|{3}' -f $resolvedItem.FullName, $resolvedItem.Exists, $length, $resolvedItem.LastWriteTimeUtc.Ticks))
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

function Invoke-PcaiPortableProcessCapture {
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

function Invoke-PcaiProcessCapture {
    [CmdletBinding()]
    [OutputType([pscustomobject])]
    param(
        [Parameter(Mandatory)][string]$FilePath,
        [string[]]$Arguments = @(),
        [int]$TimeoutMs = 1200,
        [Threading.CancellationToken]$CancellationToken = [Threading.CancellationToken]::None
    )
    # Portable Redis preserves the exact predecessor contract; its deadline gap remains OPEN.
    if (-not $IsWindows) { return Invoke-PcaiPortableProcessCapture -FilePath $FilePath -Arguments $Arguments -TimeoutMs $TimeoutMs }
    # Preserve the existing five fields; all ordinary failures stay nonthrowing.
    if ($TimeoutMs -le 0) { return [pscustomobject]@{ Success = $false; ExitCode = -3; TimedOut = $false; StdOut = ''; StdErr = 'TimeoutMs must be positive.'; Closure = 'NOT_STARTED' } }
    if (-not (Test-Path -LiteralPath $FilePath -PathType Leaf)) { return [pscustomobject]@{ Success = $false; ExitCode = -1; TimedOut = $false; StdOut = ''; StdErr = "File not found: $FilePath"; Closure = 'NOT_STARTED' } }
    if (-not $IsWindows) { return [pscustomobject]@{ Success = $false; ExitCode = -3; TimedOut = $false; StdOut = ''; StdErr = 'External cache capture is unsupported on this host; local cache remains available.'; Closure = 'UNSUPPORTED' } }
    $captureOperation = $null
    $captureResult = $null
    $captureWrapperFailure = $null
    $captureSecondaryFailures = [Collections.Generic.List[string]]::new()
    try {
        if (-not (Get-Command Initialize-PcaiProcessCaptureRunner -CommandType Function -ErrorAction SilentlyContinue)) {
            . (Join-Path $PSScriptRoot '../Private/Initialize-PcaiProcessCapture.ps1')
        }
        Initialize-PcaiProcessCaptureRunner
        $captureExecutable = [IO.Path]::GetFullPath($FilePath)
        $captureOperation = [Pcai.Common.ProcessCaptureV1]::Begin($captureExecutable, $Arguments, $TimeoutMs, $CancellationToken)
        while (-not $captureOperation.Completion.IsCompleted) {
            if ($PSCmdlet.Stopping) { $captureOperation.Cancel() }
            if ($captureOperation.FirstCleanup -and $captureOperation.FirstCleanup.Closure -eq 'HOLD') { [Threading.Thread]::Sleep(1000) }
            else { [Threading.Thread]::Sleep(5) }
        }
        $captureResult = $captureOperation.Completion.GetAwaiter().GetResult()
    }
    catch { $captureWrapperFailure = $_.Exception.ToString() }
    finally {
        if ($captureOperation -and -not $captureOperation.Completion.IsCompleted) {
            try { $captureOperation.Cancel() } catch { $captureSecondaryFailures.Add($_.Exception.ToString()) }
            # Exceptional HOLD retains this original caller until exact closure.
            # The C# foreground controller retries only original custody at low duty.
            # No new runtime/cleanup budget or success promotion is granted here.
            try { $captureResult = $captureOperation.Completion.GetAwaiter().GetResult() }
            catch { $captureSecondaryFailures.Add($_.Exception.ToString()) }
        }
    }
    if ($captureResult) {
        if ($captureWrapperFailure) {
            # A wrapper/pipeline error cannot be promoted by later native closure.
            $captureDiagnostic = @($captureWrapperFailure, $captureResult.PrimaryFailure) + @($captureResult.SecondaryFailures) + @($captureSecondaryFailures)
            return [pscustomobject]@{ Success = $false; ExitCode = -3; TimedOut = $captureResult.TimedOut; StdOut = ''; StdErr = (($captureDiagnostic | Where-Object { $_ }) -join [Environment]::NewLine); Closure = $captureResult.Closure; CustodyId = $captureResult.CustodyId; PrimaryFailure = $captureResult.PrimaryFailure; WrapperFailure = $captureWrapperFailure; FirstCleanup = $captureResult.FirstCleanup }
        }
        return [pscustomobject]@{
            Success = $captureResult.Success; ExitCode = $captureResult.ExitCode; TimedOut = $captureResult.TimedOut
            StdOut = $captureResult.StdOut; StdErr = $captureResult.StdErr; Closure = $captureResult.Closure; CustodyId = $captureResult.CustodyId
            PrimaryFailure = $captureResult.PrimaryFailure; SecondaryFailures = $captureResult.SecondaryFailures
            RepeatedFailures = $captureResult.RepeatedFailures
            StdoutBytes = $captureResult.StdoutBytes; StderrBytes = $captureResult.StderrBytes
            ElapsedMilliseconds = $captureResult.ElapsedMilliseconds; AssignedBeforeResume = $captureResult.AssignedBeforeResume
            OriginalPid = $captureResult.OriginalPid
            OriginalExitCode = $captureResult.OriginalExitCode
            ParentExited = $captureResult.ParentExited; JobEmpty = $captureResult.JobEmpty; ReadersSettled = $captureResult.ReadersSettled; HandlesClosed = $captureResult.HandlesClosed
            FirstCleanup = $captureResult.FirstCleanup
            CleanupContinuationUsed = $captureResult.CleanupContinuationUsed
        }
    }
    $captureDiagnostic = @($captureWrapperFailure) + @($captureSecondaryFailures)
    [pscustomobject]@{ Success = $false; ExitCode = -3; TimedOut = $false; StdOut = ''; StdErr = ($captureDiagnostic -join [Environment]::NewLine); Closure = if ($captureOperation -and -not $captureOperation.Completion.IsCompleted) { 'HOLD' } else { 'NOT_STARTED' }; PrimaryFailure = $captureWrapperFailure; SecondaryFailures = @($captureSecondaryFailures) }
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
        [datetime]$LastAccessedUtc = [datetime]::UtcNow,

        [Parameter()]
        [AllowNull()]
        [Nullable[datetime]]$ExpiresUtc
    )

    $payload = [ordered]@{
        Value           = Copy-PcaiCacheValue -Value $Value
        DependencyStamp = $DependencyStamp
        CreatedUtc      = $CreatedUtc.ToString('o')
        LastAccessedUtc = $LastAccessedUtc.ToString('o')
    }

    if ($null -ne $ExpiresUtc) {
        $payload.ExpiresUtc = ([datetime]$ExpiresUtc).ToUniversalTime().ToString('o')
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
        if ($parsed -isnot [System.Collections.IDictionary] -or -not $parsed.Contains('Value')) {
            return $null
        }

        $dates = @{ CreatedUtc = [datetime]::UtcNow; LastAccessedUtc = [datetime]::UtcNow; ExpiresUtc = $null }
        foreach ($name in @('CreatedUtc', 'LastAccessedUtc', 'ExpiresUtc')) {
            if ($parsed.Contains($name) -and -not [string]::IsNullOrWhiteSpace([string]$parsed[$name])) {
                # Newer PowerShell JSON readers can already materialize dates.
                $dates[$name] = if ($parsed[$name] -is [datetime]) {
                    $parsed[$name].ToUniversalTime()
                } else {
                    [datetime]::Parse([string]$parsed[$name], [Globalization.CultureInfo]::InvariantCulture, [Globalization.DateTimeStyles]::RoundtripKind).ToUniversalTime()
                }
            }
        }

        [pscustomobject]@{
            Value           = Copy-PcaiCacheValue -Value $parsed['Value']
            DependencyStamp = if ($parsed.Contains('DependencyStamp')) { [string]$parsed['DependencyStamp'] } else { $null }
            CreatedUtc      = $dates.CreatedUtc
            LastAccessedUtc = $dates.LastAccessedUtc
            ExpiresUtc      = $dates.ExpiresUtc
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
        # A later writer may have replaced the observed value; reads never delete it.
        return $null
    }

    $now = [datetime]::UtcNow
    if (($null -ne $entry.ExpiresUtc -and $now -ge $entry.ExpiresUtc) -or
        ($TtlSeconds -gt 0 -and ($now - $entry.CreatedUtc).TotalSeconds -gt $TtlSeconds)) {
        return $null
    }

    if ($PSBoundParameters.ContainsKey('DependencyStamp') -and $DependencyStamp -ne $entry.DependencyStamp) {
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
        [int]$TtlSeconds = 0,

        [Parameter()]
        [datetime]$CreatedUtc = [datetime]::UtcNow,

        [Parameter()]
        [AllowNull()]
        [Nullable[datetime]]$ExpiresUtc
    )

    $status = Get-PcaiExternalCacheStatus
    if (-not $status.Enabled -or -not $status.Available) {
        return $false
    }

    $config = Get-PcaiExternalCacheConfig
    if ($null -eq $ExpiresUtc -and $TtlSeconds -gt 0) {
        $ExpiresUtc = $CreatedUtc.AddSeconds($TtlSeconds)
    }
    $payloadParameters = @{ Value = $Value; CreatedUtc = $CreatedUtc; ExpiresUtc = $ExpiresUtc }
    if ($PSBoundParameters.ContainsKey('DependencyStamp')) {
        $payloadParameters.DependencyStamp = $DependencyStamp
    }
    $payloadJson = ConvertTo-PcaiExternalCachePayload @payloadParameters
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
    $literalPrefix = if ([string]::IsNullOrWhiteSpace($Namespace)) {
        [string]$config.KeyPrefix
    } else {
        '{0}{1}::' -f $config.KeyPrefix, $Namespace
    }
    # Redis MATCH uses glob syntax; prefix data must remain literal.
    $pattern = $literalPrefix.Replace('\', '\\').Replace('*', '\*').Replace('?', '\?').Replace('[', '\[').Replace(']', '\]') + '*'

    $scan = Invoke-PcaiRedisCli -Config $config -Arguments @('--scan', '--pattern', $pattern) -TimeoutMs ([Math]::Max($config.TimeoutMs, 3000))
    if (-not $scan.Success -or [string]::IsNullOrWhiteSpace($scan.StdOut)) {
        return
    }

    $keys = @($scan.StdOut -split "`r?`n" | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
    foreach ($key in $keys) {
        if ($key.StartsWith($literalPrefix, [System.StringComparison]::Ordinal)) {
            Invoke-PcaiRedisCli -Config $config -Arguments @('DEL', [string]$key) | Out-Null
        }
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
        # Old local records have no expiry property; retain their legacy semantics.
        $expiryProperty = $entry.PSObject.Properties['ExpiresUtc']
        if (($null -ne $expiryProperty -and $null -ne $expiryProperty.Value -and $now -ge $expiryProperty.Value) -or
            ($TtlSeconds -gt 0 -and ($now - $entry.CreatedUtc).TotalSeconds -gt $TtlSeconds)) {
            $script:PcaiSharedCache.Entries.Remove($fullKey)
        } elseif ($PSBoundParameters.ContainsKey('DependencyStamp') -and $DependencyStamp -ne $entry.DependencyStamp) {
            $script:PcaiSharedCache.Entries.Remove($fullKey)
        } else {
            $entry.LastAccessedUtc = $now
            return ,(Copy-PcaiCacheValue -Value $entry.Value)
        }
    }

    $externalParameters = @{ Namespace = $Namespace; Key = $Key; TtlSeconds = $TtlSeconds }
    if ($PSBoundParameters.ContainsKey('DependencyStamp')) {
        $externalParameters.DependencyStamp = $DependencyStamp
    }
    $externalEntry = Get-PcaiExternalCacheEntryRecord @externalParameters
    if (-not $externalEntry) {
        return $null
    }

    $expiryProperty = $externalEntry.PSObject.Properties['ExpiresUtc']
    $expiresUtc = if ($null -ne $expiryProperty) { $expiryProperty.Value } else { $null }
    Set-PcaiLocalSharedCacheEntry -FullKey $fullKey -Value $externalEntry.Value -DependencyStamp $externalEntry.DependencyStamp -CreatedUtc $externalEntry.CreatedUtc -LastAccessedUtc ([datetime]::UtcNow) -ExpiresUtc $expiresUtc
    return ,(Copy-PcaiCacheValue -Value $externalEntry.Value)
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

    $expiresUtc = if ($TtlSeconds -gt 0) { $now.AddSeconds($TtlSeconds) } else { $null }
    $localParameters = @{ FullKey = $fullKey; Value = $Value; CreatedUtc = $now; LastAccessedUtc = $now; ExpiresUtc = $expiresUtc }
    $externalParameters = @{ Namespace = $Namespace; Key = $Key; Value = $Value; TtlSeconds = $TtlSeconds; CreatedUtc = $now; ExpiresUtc = $expiresUtc }
    if ($PSBoundParameters.ContainsKey('DependencyStamp')) {
        $localParameters.DependencyStamp = $DependencyStamp
        $externalParameters.DependencyStamp = $DependencyStamp
    }
    Set-PcaiLocalSharedCacheEntry @localParameters
    [void](Set-PcaiExternalCacheEntry @externalParameters)

    return ,(Copy-PcaiCacheValue -Value $Value)
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

    $literalPrefix = '{0}::' -f $Namespace
    foreach ($key in @($script:PcaiSharedCache.Entries.Keys)) {
        if ($key.StartsWith($literalPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
            $script:PcaiSharedCache.Entries.Remove($key)
        }
    }

    Clear-PcaiExternalCache -Namespace $Namespace
}
