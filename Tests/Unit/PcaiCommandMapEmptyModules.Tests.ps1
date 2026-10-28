#Requires -Version 7.0
#Requires -Modules @{ ModuleName = 'Pester'; ModuleVersion = '5.0.0' }

param([string]$CommandMapCacheKeySourcePath)

BeforeAll {
    $script:CommandMapRepo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
    $script:SavedCommandMapProvider = [Environment]::GetEnvironmentVariable('PCAI_CACHE_PROVIDER', 'Process')
    $script:SavedCommandMapGlobals = @{}
    foreach ($name in @('PcaiSharedCache', 'PcaiSharedCacheProviderState')) {
        $prior = Get-Variable -Name $name -Scope Global -ErrorAction SilentlyContinue
        $script:SavedCommandMapGlobals[$name] = @{ Exists = ($null -ne $prior); Value = $null }
        if ($null -ne $prior) { $script:SavedCommandMapGlobals[$name].Value = $prior.Value }
        Set-Variable -Name $name -Scope Global -Value @{}
    }
    $env:PCAI_CACHE_PROVIDER = 'memory'
    # Load complete maintained producers, without aggregate/native module import.
    . (Join-Path $script:CommandMapRepo 'Modules/PC-AI.Common/Public/Get-PcaiSharedCache.ps1')
    $cacheKeySource = if ($CommandMapCacheKeySourcePath) { $CommandMapCacheKeySourcePath } else {
        Join-Path $script:CommandMapRepo 'Modules/PC-AI.CLI/Private/Get-PcCommandMapCacheKey.ps1'
    }
    . $cacheKeySource
    . (Join-Path $script:CommandMapRepo 'Modules/PC-AI.CLI/Public/Get-PCCommandMap.ps1')
    function New-CommandMapFixture {
        param([string]$Name)
        $root = Join-Path $TestDrive $Name
        [void][IO.Directory]::CreateDirectory((Join-Path $root 'Modules'))
        return $root
    }
    function Write-CommandMapManifest {
        param([string]$Root, [string]$ModuleName, [string]$Command, [datetime]$Time)
        $directory = Join-Path (Join-Path $Root 'Modules') $ModuleName
        [void][IO.Directory]::CreateDirectory($directory)
        $file = Join-Path $directory ($ModuleName + '.psd1')
        [IO.File]::WriteAllText($file, "@{ PrivateData = @{ PCAI = @{ Commands = @('$Command') } } }")
        [IO.File]::SetLastWriteTimeUtc($file, $Time)
        return $file
    }
}

AfterAll {
    [Environment]::SetEnvironmentVariable('PCAI_CACHE_PROVIDER', $script:SavedCommandMapProvider, 'Process')
    foreach ($name in @('PcaiSharedCache', 'PcaiSharedCacheProviderState')) {
        $saved = $script:SavedCommandMapGlobals[$name]
        if ($saved.Exists) { Set-Variable -Name $name -Scope Global -Value $saved.Value }
        else { Remove-Variable -Name $name -Scope Global -ErrorAction SilentlyContinue }
    }
}

Describe 'CLI command discovery with absent or changing manifests' -Tag 'Unit', 'Portable', 'CacheCorrectness' {
    # Protects: Null discovery output reaches the existing zero-manifest cache key.
    # Detects: Mandatory parameter binding rejects null before the fallback body runs.
    # Needs: PowerShell 7, Pester 5; no native provider, network, model or hardware.
    # Breadcrumb: Modules/PC-AI.CLI/Private/Get-PcCommandMapCacheKey.ps1
    It 'retains the existing null-manifest fallback through actual parameter binding' {
        Get-PcCommandMapCacheKey -ModulesRoot $TestDrive -Manifests $null | Should -BeExactly "$TestDrive|0|0"
    }

    # Protects: An explicit empty manifest collection preserves the zero-manifest cache key.
    # Detects: Mandatory parameter binding rejects an empty array before the fallback body runs.
    # Needs: PowerShell 7, Pester 5; no native provider, network, model or hardware.
    # Breadcrumb: Modules/PC-AI.CLI/Private/Get-PcCommandMapCacheKey.ps1
    It 'retains the existing empty-array fallback through actual parameter binding' {
        Get-PcCommandMapCacheKey -ModulesRoot $TestDrive -Manifests @() | Should -BeExactly "$TestDrive|0|0"
    }

    # Protects: Public command discovery remains usable before any module manifests exist.
    # Detects: Real Get-ChildItem null output aborts Get-PCCommandMap instead of returning base commands.
    # Needs: PowerShell 7, Pester 5 and disposable TestDrive filesystem; memory cache only.
    # Breadcrumb: Modules/PC-AI.CLI/Public/Get-PCCommandMap.ps1
    It 'returns help and version for an existing empty Modules directory' {
        $root = New-CommandMapFixture 'empty'
        $map = Get-PCCommandMap -ProjectRoot $root
        @($map.Keys | Sort-Object) | Should -Be @('help', 'version')
        @($map['help']).Count | Should -Be 0
        @($map['version']).Count | Should -Be 0
    }

    # Protects: Public discovery uses valid empty-directory cache entries and isolates callers from stored values.
    # Detects: Ignored matching cache entries or shallow cache returns that persist caller mutation.
    # Needs: PowerShell 7, Pester 5 and TestDrive; explicit memory cache with restored global/environment state.
    # Breadcrumb: Modules/PC-AI.CLI/Public/Get-PCCommandMap.ps1; Modules/PC-AI.Common/Public/Get-PcaiSharedCache.ps1
    It 'isolates the cached empty-module command map from caller mutation' {
        $root = New-CommandMapFixture 'empty-cache'
        $first = Get-PCCommandMap -ProjectRoot $root
        $stamp = Get-PcaiDependencyStamp -InputObject @($null)
        $stored = Get-PcaiSharedCacheEntry -Namespace 'pcai-cli' -Key "command-map::$root" -DependencyStamp $stamp
        $stored | Should -Not -BeNullOrEmpty
        @($stored.Keys | Sort-Object) | Should -Be @('help', 'version')
        # A cache-only marker proves the public producer serves a matching entry
        # instead of silently recomputing an identical empty-directory result.
        Set-PcaiSharedCacheEntry -Namespace 'pcai-cli' -Key "command-map::$root" -DependencyStamp $stamp `
            -Value @{ help = @('owned-cache-control'); version = @() } | Out-Null
        $second = Get-PCCommandMap -ProjectRoot $root
        @($second['help']) | Should -Be @('owned-cache-control')
        $second['help'] = @('caller-mutated')
        $third = Get-PCCommandMap -ProjectRoot $root
        @($second.Keys | Sort-Object) | Should -Be @('help', 'version')
        @($third['help']) | Should -Be @('owned-cache-control')
    }

    # Protects: Nonempty manifest discovery and dependency-stamp cache invalidation while admitting empty inputs.
    # Detects: An unconditional empty result, missing manifest commands or reuse after changed file metadata.
    # Needs: PowerShell 7, Pester 5 and TestDrive files with controlled UTC modification times; memory cache.
    # Breadcrumb: Modules/PC-AI.CLI/Public/Get-PCCommandMap.ps1; Modules/PC-AI.Common/Public/Get-PcaiSharedCache.ps1
    It 'still maps real manifest commands and invalidates after manifest changes' {
        $root = New-CommandMapFixture 'changing'
        $path = Write-CommandMapManifest -Root $root -ModuleName 'OwnedFixture' -Command 'diagnose' -Time ([datetime]'2021-01-01T00:00:00Z')
        $first = Get-PCCommandMap -ProjectRoot $root
        @($first['diagnose']) | Should -Be @('OwnedFixture')
        $oldStamp = Get-PcaiDependencyStamp -InputObject @((Get-Item -LiteralPath $path))
        $cachedFirst = Get-PcaiSharedCacheEntry -Namespace 'pcai-cli' -Key "command-map::$root" -DependencyStamp $oldStamp
        $cachedFirst | Should -Not -BeNullOrEmpty
        @($cachedFirst['diagnose']) | Should -Be @('OwnedFixture')
        # This is a coarse real-file invalidation control; it does not define
        # freshness semantics for retained FileSystemInfo snapshots.
        [void](Write-CommandMapManifest -Root $root -ModuleName 'OwnedFixture' -Command 'capture-revised' -Time ([datetime]'2022-01-01T00:00:00Z'))
        Test-Path -LiteralPath $path -PathType Leaf | Should -BeTrue
        $newStamp = Get-PcaiDependencyStamp -InputObject @((Get-Item -LiteralPath $path))
        $newStamp | Should -Not -BeExactly $oldStamp
        $stillCachedOld = Get-PcaiSharedCacheEntry -Namespace 'pcai-cli' -Key "command-map::$root" -DependencyStamp $oldStamp
        $stillCachedOld | Should -Not -BeNullOrEmpty
        @($stillCachedOld['diagnose']) | Should -Be @('OwnedFixture')
        $second = Get-PCCommandMap -ProjectRoot $root
        $second.ContainsKey('diagnose') | Should -BeFalse
        @($second['capture-revised']) | Should -Be @('OwnedFixture')
        $cachedRevised = Get-PcaiSharedCacheEntry -Namespace 'pcai-cli' -Key "command-map::$root" -DependencyStamp $newStamp
        $cachedRevised | Should -Not -BeNullOrEmpty
        @($cachedRevised['capture-revised']) | Should -Be @('OwnedFixture')
    }
}
