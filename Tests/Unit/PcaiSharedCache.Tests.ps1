#Requires -Version 7.0
#Requires -Modules @{ ModuleName = 'Pester'; ModuleVersion = '5.0.0' }

BeforeAll {
    $script:ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
    $script:CommonModulePath = Join-Path $script:ProjectRoot 'Modules\PC-AI.Common\PC-AI.Common.psm1'
    Import-Module $script:CommonModulePath -Force | Out-Null

    $script:OriginalCacheEnv = @{
        PCAI_CACHE_PROVIDER   = [Environment]::GetEnvironmentVariable('PCAI_CACHE_PROVIDER', 'Process')
        PCAI_REDIS_CLI_PATH   = [Environment]::GetEnvironmentVariable('PCAI_REDIS_CLI_PATH', 'Process')
        PCAI_REDIS_HOST       = [Environment]::GetEnvironmentVariable('PCAI_REDIS_HOST', 'Process')
        PCAI_REDIS_PORT       = [Environment]::GetEnvironmentVariable('PCAI_REDIS_PORT', 'Process')
        PCAI_REDIS_KEY_PREFIX = [Environment]::GetEnvironmentVariable('PCAI_REDIS_KEY_PREFIX', 'Process')
    }

    function Restore-CacheEnv {
        foreach ($entry in $script:OriginalCacheEnv.GetEnumerator()) {
            [Environment]::SetEnvironmentVariable($entry.Key, $entry.Value, 'Process')
        }
    }

    $script:DetectedRedisCliPath = $null
    foreach ($candidate in @(
        'C:\Program Files\Redis\redis-cli.exe',
        'T:\projects\redis-windows\redis-cli.exe',
        (Join-Path $env:USERPROFILE 'bin\redis-cli.exe')
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

    $script:RedisAvailable = $false
    if ($script:DetectedRedisCliPath) {
        try {
            $probe = & $script:DetectedRedisCliPath -h 127.0.0.1 -p 6380 --raw PING 2>$null
            $script:RedisAvailable = ($LASTEXITCODE -eq 0 -and ([string]$probe).Trim() -eq 'PONG')
        } catch {
            $script:RedisAvailable = $false
        }
    }
}

AfterAll {
    foreach ($entry in $script:OriginalCacheEnv.GetEnumerator()) {
        [Environment]::SetEnvironmentVariable($entry.Key, $entry.Value, 'Process')
    }
}

Describe 'PcaiSharedCache' -Tag 'Unit', 'Cache', 'Acceleration', Portable, 'Portable' {
    BeforeEach {
        foreach ($entry in $script:OriginalCacheEnv.GetEnumerator()) {
            [Environment]::SetEnvironmentVariable($entry.Key, $entry.Value, 'Process')
        }
        Clear-PcaiSharedCache
    }

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

    It 'expires stale local entries when TTL is exceeded' {
        Set-PcaiSharedCacheEntry -Namespace 'pcai-unit' -Key 'ttl' -Value 'expired-value' -TtlSeconds 1 | Out-Null
        $global:PcaiSharedCache.Entries['pcai-unit::ttl'].CreatedUtc = [datetime]::UtcNow.AddSeconds(-10)

        $expired = Get-PcaiSharedCacheEntry -Namespace 'pcai-unit' -Key 'ttl' -TtlSeconds 1

        $expired | Should -Be $null
    }

    It 'reports memory mode when no external provider is configured' {
        [Environment]::SetEnvironmentVariable('PCAI_CACHE_PROVIDER', $null, 'Process')
        [Environment]::SetEnvironmentVariable('PCAI_REDIS_CLI_PATH', $null, 'Process')

        $status = Get-PcaiExternalCacheStatus -Refresh

        $status.Provider | Should -Be 'memory'
        $status.Enabled | Should -BeFalse
        $status.Available | Should -BeFalse
    }

    It 'hydrates a local cache miss from Redis when the provider is enabled' {
        if (-not $script:RedisAvailable) {
            return
        }

        [Environment]::SetEnvironmentVariable('PCAI_CACHE_PROVIDER', 'redis', 'Process')
        [Environment]::SetEnvironmentVariable('PCAI_REDIS_CLI_PATH', $script:DetectedRedisCliPath, 'Process')
        [Environment]::SetEnvironmentVariable('PCAI_REDIS_PORT', '6380', 'Process')

        $namespace = 'pcai-unit-redis'
        $key = 'roundtrip'
        $value = [pscustomobject]@{
            Name = 'redis-hit'
            Count = 42
        }

        Clear-PcaiSharedCache -Namespace $namespace
        Set-PcaiSharedCacheEntry -Namespace $namespace -Key $key -Value $value -DependencyStamp 'stamp-redis' -TtlSeconds 30 | Out-Null
        $global:PcaiSharedCache.Entries.Clear()

        $restored = Get-PcaiSharedCacheEntry -Namespace $namespace -Key $key -TtlSeconds 30 -DependencyStamp 'stamp-redis'

        $restored | Should -Not -BeNullOrEmpty
        $restored.Name | Should -Be 'redis-hit'
        $restored.Count | Should -Be 42

        Clear-PcaiSharedCache -Namespace $namespace
    }
}
