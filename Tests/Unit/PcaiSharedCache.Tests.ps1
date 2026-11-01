#Requires -Version 7.0
#Requires -Modules @{ ModuleName = 'Pester'; ModuleVersion = '5.0.0' }

param(
    [string]$CacheSourcePath,
    [switch]$SkipRedisProbe,
    [switch]$EnableRedisIntegration
)

BeforeAll {
    $script:ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
    $script:CommonModulePath = Join-Path $script:ProjectRoot 'Modules\PC-AI.Common\PC-AI.Common.psm1'
    $script:OriginalCacheEnv = @{
        PCAI_CACHE_PROVIDER   = [Environment]::GetEnvironmentVariable('PCAI_CACHE_PROVIDER', 'Process')
        PCAI_REDIS_CLI_PATH   = [Environment]::GetEnvironmentVariable('PCAI_REDIS_CLI_PATH', 'Process')
        PCAI_REDIS_HOST       = [Environment]::GetEnvironmentVariable('PCAI_REDIS_HOST', 'Process')
        PCAI_REDIS_PORT       = [Environment]::GetEnvironmentVariable('PCAI_REDIS_PORT', 'Process')
        PCAI_REDIS_KEY_PREFIX = [Environment]::GetEnvironmentVariable('PCAI_REDIS_KEY_PREFIX', 'Process')
        PCAI_REDIS_TIMEOUT_MS = [Environment]::GetEnvironmentVariable('PCAI_REDIS_TIMEOUT_MS', 'Process')
    }
    $env:PCAI_CACHE_PROVIDER = 'memory'
    if (-not $CacheSourcePath) {
        Import-Module $script:CommonModulePath -Force | Out-Null
    }
    $helperPath = if ($CacheSourcePath) { $CacheSourcePath } else {
        Join-Path $script:ProjectRoot 'Modules/PC-AI.Common/Public/Get-PcaiSharedCache.ps1'
    }
    . $helperPath
    $script:OwnedRedisPrefix = 'pcai-unit-owned-' + [guid]::NewGuid().ToString('N') + ':'

    function Restore-CacheEnv {
        foreach ($entry in $script:OriginalCacheEnv.GetEnumerator()) {
            [Environment]::SetEnvironmentVariable($entry.Key, $entry.Value, 'Process')
        }
    }

    $script:DetectedRedisCliPath = $null
    if ($EnableRedisIntegration -and -not $SkipRedisProbe) {
        $fixtureHome = [Environment]::GetFolderPath([Environment+SpecialFolder]::UserProfile)
        foreach ($candidate in @(
            'C:\Program Files\Redis\redis-cli.exe',
            'T:\projects\redis-windows\redis-cli.exe',
            (Join-Path $fixtureHome 'bin\redis-cli.exe')
        )) {
            if (Test-Path -LiteralPath $candidate -PathType Leaf) {
                $script:DetectedRedisCliPath = $candidate
                break
            }
        }
        if (-not $script:DetectedRedisCliPath) {
            $redisCliCommand = Get-Command redis-cli.exe -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
            if ($redisCliCommand -and $redisCliCommand.Path) {
                $script:DetectedRedisCliPath = $redisCliCommand.Path
            }
        }
    }

    $script:RedisAvailable = $false
    if ($EnableRedisIntegration -and $script:DetectedRedisCliPath -and -not $SkipRedisProbe) {
        try {
            $env:PCAI_CACHE_PROVIDER = 'redis'
            $env:PCAI_REDIS_CLI_PATH = $script:DetectedRedisCliPath
            $env:PCAI_REDIS_HOST = 'localhost'
            $env:PCAI_REDIS_PORT = '6380'
            $env:PCAI_REDIS_KEY_PREFIX = $script:OwnedRedisPrefix
            $env:PCAI_REDIS_TIMEOUT_MS = '1200'
            $script:RedisAvailable = (Get-PcaiExternalCacheStatus -Refresh).Available
        } catch {
            $script:RedisAvailable = $false
        } finally {
            $env:PCAI_CACHE_PROVIDER = 'memory'
        }
    }
}

Describe 'PcaiSharedCache literal dependency and invalidation contracts' -Tag 'Unit', 'Cache', 'CacheCorrectness', 'Portable' {
    BeforeAll {
        $helperPath = if ($CacheSourcePath) { $CacheSourcePath } else {
            Join-Path $script:ProjectRoot 'Modules/PC-AI.Common/Public/Get-PcaiSharedCache.ps1'
        }
        . $helperPath
    }

    BeforeEach {
        Set-StrictMode -Version Latest
        $env:PCAI_CACHE_PROVIDER = 'memory'
        Clear-PcaiSharedCache
    }

    # Protects: DirectoryInfo dependencies remain usable under StrictMode.
    # Detects: Accessing the nonexistent directory Length property.
    # Needs: A real ordinary directory and the complete cache helper.
    # Breadcrumb: Get-PcaiDependencyStamp FileSystemInfo branch.
    It 'stamps a real directory object without a Length property' {
        $directory = New-Item -ItemType Directory -Path (Join-Path $TestDrive 'object-dependency')
        $stamp = Get-PcaiDependencyStamp -InputObject @($directory)
        $stamp | Should -Match '^[0-9A-F]{64}$'
        $stamp | Should -Be (Get-PcaiDependencyStamp -InputObject @($directory))
    }

    # Protects: Directory paths have the same dependency identity as objects.
    # Detects: The independent resolved-path branch reading directory Length.
    # Needs: A real directory supplied by both path and DirectoryInfo.
    # Breadcrumb: Get-PcaiDependencyStamp string branch.
    It 'stamps a real directory path consistently with its object' {
        $directory = New-Item -ItemType Directory -Path (Join-Path $TestDrive 'path-dependency')
        $pathStamp = Get-PcaiDependencyStamp -InputObject @($directory.FullName)
        $pathStamp | Should -Match '^[0-9A-F]{64}$'
        $pathStamp | Should -Be (Get-PcaiDependencyStamp -InputObject @($directory))
    }

    # Protects: Directory metadata changes invalidate cached consumers.
    # Detects: Replacing the directory stamp with a constant or ignored path.
    # Needs: Explicit real directory timestamps without sleeps or external IO.
    # Breadcrumb: Get-PcaiDependencyStamp LastWriteTimeUtc.
    It 'changes a directory dependency stamp when its timestamp changes' {
        $directory = New-Item -ItemType Directory -Path (Join-Path $TestDrive 'changed-directory')
        [IO.Directory]::SetLastWriteTimeUtc($directory.FullName, [datetime]'2020-01-01T00:00:00Z')
        $before = Get-PcaiDependencyStamp -InputObject @($directory.FullName)
        [IO.Directory]::SetLastWriteTimeUtc($directory.FullName, [datetime]'2020-01-02T00:00:00Z')
        $after = Get-PcaiDependencyStamp -InputObject @($directory.FullName)
        $after | Should -Not -Be $before
    }

    # Protects: File length remains part of the existing dependency stamp.
    # Detects: A directory fix accidentally discarding Length for all files.
    # Needs: Real file bytes with an explicitly fixed timestamp.
    # Breadcrumb: Get-PcaiDependencyStamp FileInfo handling.
    It 'retains file length sensitivity and object-path parity' {
        $path = Join-Path $TestDrive 'file-dependency.bin'
        [IO.File]::WriteAllBytes($path, [byte[]]@(1))
        [IO.File]::SetLastWriteTimeUtc($path, [datetime]'2020-01-01T00:00:00Z')
        $before = Get-PcaiDependencyStamp -InputObject @($path)
        $before | Should -Be (Get-PcaiDependencyStamp -InputObject @((Get-Item -LiteralPath $path)))
        [IO.File]::WriteAllBytes($path, [byte[]]@(1, 2))
        [IO.File]::SetLastWriteTimeUtc($path, [datetime]'2020-01-01T00:00:00Z')
        (Get-PcaiDependencyStamp -InputObject @($path)) | Should -Not -Be $before
    }

    # Protects: Literal Keys entries and all nested siblings survive caching.
    # Detects: PowerShell dictionary adapters shadowing the real key collection.
    # Needs: Hashtable and OrderedDictionary values with mutable nested data.
    # Breadcrumb: Copy-PcaiCacheValue IDictionary recursion.
    It 'round-trips and isolates a <Kind> with a literal Keys entry' -TestCases @(
        @{ Kind = 'hashtable'; Ordered = $false }
        @{ Kind = 'ordered dictionary'; Ordered = $true }
    ) {
        param($Kind, $Ordered)
        $value = if ($Ordered) { [ordered]@{} } else { @{} }
        $value['Keys'] = 'literal'
        $value['Sibling'] = 'retained'
        $value['Nested'] = @{ Keys = 'nested-literal'; Value = 'original' }
        Set-PcaiSharedCacheEntry -Namespace 'dictionary' -Key 'payload' -Value $value | Out-Null
        $value['Nested']['Value'] = 'input-mutated'
        $hit = Get-PcaiSharedCacheEntry -Namespace 'dictionary' -Key 'payload'
        $hit.PSBase.Count | Should -Be 3
        $hit['Keys'] | Should -Be 'literal'
        $hit['Sibling'] | Should -Be 'retained'
        $hit['Nested']['Keys'] | Should -Be 'nested-literal'
        $hit['Nested']['Value'] | Should -Be 'original'
        $hit['Nested']['Value'] = 'output-mutated'
        (Get-PcaiSharedCacheEntry -Namespace 'dictionary' -Key 'payload')['Nested']['Value'] | Should -Be 'original'
    }

    # Protects: Clearing a namespace treats wildcard characters as literal data.
    # Detects: Wildcard matching that deletes siblings or fails to clear itself.
    # Needs: Real local entries for star, question mark, brackets and siblings.
    # Breadcrumb: Clear-PcaiSharedCache namespace prefix comparison.
    It 'clears only literal local namespace prefixes containing wildcards' {
        foreach ($namespace in @('ns*', 'ns?', 'ns[ab]')) {
            Set-PcaiSharedCacheEntry -Namespace $namespace -Key 'entry' -Value 'target' | Out-Null
        }
        foreach ($namespace in @('ns-other', 'nsa', 'nsab', 'ns*-child')) {
            Set-PcaiSharedCacheEntry -Namespace $namespace -Key 'entry' -Value 'sibling' | Out-Null
        }
        foreach ($namespace in @('ns*', 'ns?', 'ns[ab]')) {
            Clear-PcaiSharedCache -Namespace $namespace
            (Get-PcaiSharedCacheEntry -Namespace $namespace -Key 'entry') | Should -BeNullOrEmpty
        }
        foreach ($namespace in @('ns-other', 'nsa', 'nsab', 'ns*-child')) {
            (Get-PcaiSharedCacheEntry -Namespace $namespace -Key 'entry') | Should -Be 'sibling'
        }
    }

    # Protects: Redis namespace deletion cannot include unrelated scanned keys.
    # Detects: Unescaped Redis glob prefixes or unfiltered scan results.
    # Needs: Only Redis transport/status collaborators mocked, no real server.
    # Breadcrumb: Clear-PcaiExternalCache pattern and deletion boundary.
    It 'escapes Redis glob data and deletes only exact literal prefix matches' {
        Mock Get-PcaiExternalCacheStatus { [pscustomobject]@{ Enabled = $true; Available = $true } }
        Mock Get-PcaiExternalCacheConfig { [pscustomobject]@{ KeyPrefix = 'p[*]:'; TimeoutMs = 100 } }
        Mock Invoke-PcaiRedisCli {
            param($Config, $Arguments, $TimeoutMs)
            if ($Arguments[0] -eq '--scan') {
                return [pscustomobject]@{ Success = $true; StdOut = "p[*]:ns[*]?::one`np*:ns*?::foreign`np[*]:ns[*]?-child::foreign" }
            }
            return [pscustomobject]@{ Success = $true; StdOut = '1' }
        }
        Clear-PcaiExternalCache -Namespace 'ns[*]?'
        Should -Invoke Invoke-PcaiRedisCli -Times 1 -Exactly -ParameterFilter {
            $Arguments[0] -eq '--scan' -and $Arguments[2] -ceq 'p\[\*\]:ns\[\*\]\?::*'
        }
        Should -Invoke Invoke-PcaiRedisCli -Times 1 -Exactly -ParameterFilter { $Arguments[0] -eq 'DEL' }
        Should -Invoke Invoke-PcaiRedisCli -Times 1 -Exactly -ParameterFilter {
            $Arguments[0] -eq 'DEL' -and $Arguments[1] -ceq 'p[*]:ns[*]?::one'
        }
    }
}

AfterAll {
    foreach ($entry in $script:OriginalCacheEnv.GetEnumerator()) {
        [Environment]::SetEnvironmentVariable($entry.Key, $entry.Value, 'Process')
    }
}

Describe 'PcaiSharedCache' -Tag 'Unit', 'Cache', 'Acceleration', 'Portable' {
    BeforeEach {
        $env:PCAI_CACHE_PROVIDER = 'memory'
        Clear-PcaiSharedCache
    }

    # Protects: Cached consumer values retain their shape and dependency identity.
    # Detects: Lost nested paths or a hit accepted under a different stamp.
    # Needs: In-memory cache only; no external provider.
    # Breadcrumb: Set-PcaiSharedCacheEntry and Get-PcaiSharedCacheEntry.
    It 'round-trips in-memory cache values and honors dependency stamps' {
        $value = [pscustomobject]@{
            Name = 'runtime'
            Paths = @('Config\llm-config.json', 'Config\pcai-tools.json')
        }

        Set-PcaiSharedCacheEntry -Namespace 'pcai-unit' -Key 'runtime' -Value $value -DependencyStamp 'stamp-a' -TtlSeconds 30 | Out-Null
        $hit = Get-PcaiSharedCacheEntry -Namespace 'pcai-unit' -Key 'runtime' -TtlSeconds 30 -DependencyStamp 'stamp-a'
        $miss = Get-PcaiSharedCacheEntry -Namespace 'pcai-unit' -Key 'runtime' -TtlSeconds 30 -DependencyStamp 'stamp-b'

        $hit | Should -Not -BeNullOrEmpty
        $hit.Name | Should -Be 'runtime'
        @($hit.Paths) | Should -Be @('Config\llm-config.json', 'Config\pcai-tools.json')
        $miss | Should -Be $null
    }

    # Protects: Expired local values cannot be served to consumers.
    # Detects: TTL checks accepting an explicitly aged entry.
    # Needs: In-memory entry with a controlled creation time; no sleep.
    # Breadcrumb: Get-PcaiSharedCacheEntry expiration handling.
    It 'expires stale local entries when TTL is exceeded' {
        Set-PcaiSharedCacheEntry -Namespace 'pcai-unit' -Key 'ttl' -Value 'expired-value' -TtlSeconds 1 | Out-Null
        $global:PcaiSharedCache.Entries['pcai-unit::ttl'].CreatedUtc = [datetime]::UtcNow.AddSeconds(-10)

        $expired = Get-PcaiSharedCacheEntry -Namespace 'pcai-unit' -Key 'ttl' -TtlSeconds 1

        $expired | Should -Be $null
    }

    # Protects: Cache consumers can operate without Redis configuration.
    # Detects: Incorrect external-provider enabled or available status.
    # Needs: Cleared process-local provider environment, restored by the fixture.
    # Breadcrumb: Get-PcaiExternalCacheStatus.
    It 'reports memory mode when no external provider is configured' {
        [Environment]::SetEnvironmentVariable('PCAI_CACHE_PROVIDER', $null, 'Process')
        [Environment]::SetEnvironmentVariable('PCAI_REDIS_CLI_PATH', $null, 'Process')

        $status = Get-PcaiExternalCacheStatus -Refresh

        $status.Provider | Should -Be 'memory'
        $status.Enabled | Should -BeFalse
        $status.Available | Should -BeFalse
    }

    # Protects: Redis hydration preserves value and dependency identity.
    # Detects: Missing external reads or stale dependency acceptance.
    # Needs: Available Redis on the fixture port; absent or disabled probes SKIP.
    # Breadcrumb: Get-PcaiSharedCacheEntry external hydration.
    It 'hydrates a local cache miss from Redis when the provider is enabled' {
        if (-not $script:RedisAvailable) {
            Set-ItResult -Skipped -Because 'Redis provider was not available or its live probe was disabled.'
            return
        }

        [Environment]::SetEnvironmentVariable('PCAI_CACHE_PROVIDER', 'redis', 'Process')
        [Environment]::SetEnvironmentVariable('PCAI_REDIS_CLI_PATH', $script:DetectedRedisCliPath, 'Process')
        [Environment]::SetEnvironmentVariable('PCAI_REDIS_HOST', 'localhost', 'Process')
        [Environment]::SetEnvironmentVariable('PCAI_REDIS_PORT', '6380', 'Process')
        [Environment]::SetEnvironmentVariable('PCAI_REDIS_KEY_PREFIX', $script:OwnedRedisPrefix, 'Process')
        [Environment]::SetEnvironmentVariable('PCAI_REDIS_TIMEOUT_MS', '1200', 'Process')

        $namespace = 'pcai-unit-redis'
        $key = 'roundtrip'
        $value = [pscustomobject]@{
            Name = 'redis-hit'
            Count = 42
        }

        try {
            Set-PcaiSharedCacheEntry -Namespace $namespace -Key $key -Value $value -DependencyStamp 'stamp-redis' -TtlSeconds 30 | Out-Null
            $global:PcaiSharedCache.Entries.Clear()
            $restored = Get-PcaiSharedCacheEntry -Namespace $namespace -Key $key -TtlSeconds 30 -DependencyStamp 'stamp-redis'
            $restored | Should -Not -BeNullOrEmpty
            $restored.Name | Should -Be 'redis-hit'
            $restored.Count | Should -Be 42
        } finally {
            Clear-PcaiSharedCache -Namespace $namespace
            $env:PCAI_CACHE_PROVIDER = 'memory'
        }
    }
}

Describe 'PcaiSharedCache expiry and lossless value contracts' -Tag 'Unit', 'Cache', 'CacheCorrectness', 'Portable' {
    BeforeEach {
        Set-StrictMode -Version Latest
        $env:PCAI_CACHE_PROVIDER = 'memory'
        Clear-PcaiSharedCache
    }

    # Protects: Empty and singleton collections retain the caller's data shape.
    # Detects: Pipeline enumeration collapsing empty arrays to null or one value.
    # Needs: Real memory cache; no provider, compiler, or native process.
    # Breadcrumb: Copy-PcaiCacheValue and public cache return boundaries.
    It 'retains an array with <Count> members on both set and get' -TestCases @(
        @{ Count = 0 }
        @{ Count = 1 }
        @{ Count = 2 }
    ) {
        param($Count)
        $value = [object[]]::new($Count)
        for ($i = 0; $i -lt $Count; $i++) { $value[$i] = 'item-' + $i }
        $returned = Set-PcaiSharedCacheEntry -Namespace 'shape' -Key 'array' -Value $value
        $returned -is [array] | Should -BeTrue
        $returned.Count | Should -Be $Count
        $hit = Get-PcaiSharedCacheEntry -Namespace 'shape' -Key 'array'
        $hit -is [array] | Should -BeTrue
        $hit.Count | Should -Be $Count
        for ($i = 0; $i -lt $Count; $i++) { $hit[$i] | Should -Be ('item-' + $i) }
    }

    # Protects: Nested arrays, null slots and mutable values round-trip losslessly.
    # Detects: Flattening nested arrays or sharing input/output references.
    # Needs: Actual clone and public memory cache paths with nested dictionaries.
    # Breadcrumb: Copy-PcaiCacheValue IEnumerable and dictionary recursion.
    It 'preserves nested empty arrays and null slots while detaching mutable values' {
        $value = [object[]]::new(4)
        $value[0] = [object[]]@()
        $value[1] = [object[]]@('one')
        $value[2] = $null
        $value[3] = @{ State = 'original'; Children = [object[]]@('child') }
        Set-PcaiSharedCacheEntry -Namespace 'shape' -Key 'nested' -Value $value | Out-Null
        $value[1][0] = 'input-mutated'
        $value[3].State = 'input-mutated'
        $hit = Get-PcaiSharedCacheEntry -Namespace 'shape' -Key 'nested'
        $hit.Count | Should -Be 4
        $hit[0] -is [array] | Should -BeTrue
        $hit[0].Count | Should -Be 0
        $hit[1] -is [array] | Should -BeTrue
        $hit[1].Count | Should -Be 1
        $hit[1][0] | Should -Be 'one'
        $hit[2] | Should -Be $null
        $hit[3].State | Should -Be 'original'
        $hit[3].Children[0] = 'output-mutated'
        (Get-PcaiSharedCacheEntry -Namespace 'shape' -Key 'nested')[3].Children[0] | Should -Be 'child'
    }

    # Protects: A setter's expiry cannot be extended by the reader.
    # Detects: TTL stored only in Redis or ignored by a default/longer read.
    # Needs: Actual one-second memory expiry; bounded sleep instead of record edits.
    # Breadcrumb: Set-PcaiSharedCacheEntry local TTL and Get expiry handling.
    It 'honors setter expiry for <Mode> reads' -TestCases @(
        @{ Mode = 'default'; GetterTtl = 0 }
        @{ Mode = 'longer'; GetterTtl = 60 }
    ) {
        param($Mode, $GetterTtl)
        Set-PcaiSharedCacheEntry -Namespace 'expiry' -Key 'setter' -Value 'stale' -TtlSeconds 1 | Out-Null
        Start-Sleep -Milliseconds 1150
        Get-PcaiSharedCacheEntry -Namespace 'expiry' -Key 'setter' -TtlSeconds $GetterTtl | Should -Be $null
    }

    # Protects: An omitted dependency stamp imposes no empty-stamp requirement.
    # Detects: Public forwarding turns omission into a bound empty stamp.
    # Needs: Real payload/parser/cache with only Redis transport/status mocked.
    # Breadcrumb: Get-PcaiSharedCacheEntry external parameter forwarding.
    It 'hydrates a stamped external value when the getter omits its stamp' {
        Mock Get-PcaiExternalCacheStatus { [pscustomobject]@{ Enabled = $true; Available = $true } }
        Mock Get-PcaiExternalCacheConfig { [pscustomobject]@{ KeyPrefix = 'owned:' } }
        Mock Invoke-PcaiRedisCli {
            param($Config, $Arguments)
            [pscustomobject]@{ Success = $true; StdOut = '{"Value":"hydrated","DependencyStamp":"stamp-a","CreatedUtc":"2020-01-01T00:00:00Z"}' }
        }
        Get-PcaiSharedCacheEntry -Namespace 'external' -Key 'omitted' | Should -Be 'hydrated'
        Should -Invoke Invoke-PcaiRedisCli -Times 0 -Exactly -ParameterFilter { $Arguments[0] -eq 'DEL' }
    }

    # Protects: A validation miss cannot delete a concurrent writer's value.
    # Detects: GET followed by unconditional DEL after a stamp mismatch.
    # Needs: Complete validator; deterministic transport collaborator only.
    # Breadcrumb: Get-PcaiExternalCacheEntryRecord mismatch handling.
    It 'rejects an explicit stamp mismatch without deleting external data' {
        Mock Get-PcaiExternalCacheStatus { [pscustomobject]@{ Enabled = $true; Available = $true } }
        Mock Get-PcaiExternalCacheConfig { [pscustomobject]@{ KeyPrefix = 'owned:' } }
        Mock Invoke-PcaiRedisCli {
            param($Config, $Arguments)
            [pscustomobject]@{ Success = $true; StdOut = '{"Value":"observed-old","DependencyStamp":"old","CreatedUtc":"2020-01-01T00:00:00Z"}' }
        }
        Get-PcaiSharedCacheEntry -Namespace 'external' -Key 'changed' -DependencyStamp '' | Should -Be $null
        Should -Invoke Invoke-PcaiRedisCli -Times 0 -Exactly -ParameterFilter { $Arguments[0] -eq 'DEL' }
    }

    # Protects: External records cannot repopulate expired local values.
    # Detects: Ignoring setter expiry encoded in an external payload.
    # Needs: Complete payload parser/validator; only Redis transport mocked.
    # Breadcrumb: ConvertFrom-PcaiExternalCachePayload and external expiry checks.
    It 'rejects an external record whose stored expiry has passed' {
        Mock Get-PcaiExternalCacheStatus { [pscustomobject]@{ Enabled = $true; Available = $true } }
        Mock Get-PcaiExternalCacheConfig { [pscustomobject]@{ KeyPrefix = 'owned:' } }
        Mock Invoke-PcaiRedisCli {
            param($Config, $Arguments)
            [pscustomobject]@{ Success = $true; StdOut = '{"Value":"stale","DependencyStamp":"","CreatedUtc":"2020-01-01T00:00:00Z","ExpiresUtc":"2020-01-01T00:00:01Z"}' }
        }
        Get-PcaiSharedCacheEntry -Namespace 'external' -Key 'expired' | Should -Be $null
        Should -Invoke Invoke-PcaiRedisCli -Times 0 -Exactly -ParameterFilter { $Arguments[0] -eq 'DEL' }
    }

    # Protects: Hydrated values retain external setter expiry if Redis disappears.
    # Detects: Local hydration turning a finite external record into a permanent hit.
    # Needs: Actual parser and local cache; mocked transport unavailable after hydration.
    # Breadcrumb: Get-PcaiSharedCacheEntry hydration and local expiry handling.
    It 'retains external expiry locally when the provider becomes unavailable' {
        $script:ExpiryPayload = @{
            Value = 'finite'
            DependencyStamp = 'stamp-a'
        }
        $script:ExternalAvailable = $true
        Mock Get-PcaiExternalCacheStatus { [pscustomobject]@{ Enabled = $true; Available = $script:ExternalAvailable } }
        Mock Get-PcaiExternalCacheConfig { [pscustomobject]@{ KeyPrefix = 'owned:' } }
        Mock Invoke-PcaiRedisCli {
            param($Config, $Arguments)
            $script:HydratedExpiresUtc = [datetime]::UtcNow.AddSeconds(1)
            $script:ExpiryPayload.CreatedUtc = [datetime]::UtcNow.AddSeconds(-1).ToString('o')
            $script:ExpiryPayload.ExpiresUtc = $script:HydratedExpiresUtc.ToString('o')
            [pscustomobject]@{ Success = $true; StdOut = ($script:ExpiryPayload | ConvertTo-Json -Compress) }
        }
        Get-PcaiSharedCacheEntry -Namespace 'external' -Key 'finite' -DependencyStamp 'stamp-a' | Should -Be 'finite'
        $script:ExternalAvailable = $false
        $remainingMs = [Math]::Max(0, [Math]::Ceiling(($script:HydratedExpiresUtc - [datetime]::UtcNow).TotalMilliseconds))
        Start-Sleep -Milliseconds ([int]$remainingMs + 50)
        Get-PcaiSharedCacheEntry -Namespace 'external' -Key 'finite' | Should -Be $null
        Should -Invoke Invoke-PcaiRedisCli -Times 1 -Exactly -ParameterFilter { $Arguments[0] -eq 'GET' }
    }
}
