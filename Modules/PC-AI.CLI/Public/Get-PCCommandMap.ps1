function Get-PCCommandMap {
    [CmdletBinding()]
    param([string]$ProjectRoot)

    if (-not (Get-Command Get-PcaiSharedCacheEntry -ErrorAction SilentlyContinue)) {
        $commonPublicRoot = Join-Path (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)) 'PC-AI.Common\Public'
        $sharedCacheHelper = Join-Path $commonPublicRoot 'Get-PcaiSharedCache.ps1'
        if (Test-Path $sharedCacheHelper) {
            try {
                . $sharedCacheHelper
            } catch {
                Write-Verbose "Shared PC-AI cache helper unavailable: $($_.Exception.Message)"
            }
        }
    }

    $root = if ($ProjectRoot) { $ProjectRoot } else { $script:ProjectRoot }
    $modulesRoot = Join-Path $root 'Modules'
    if (-not (Test-Path $modulesRoot)) { return @{} }

    $manifests = Get-ChildItem -Path $modulesRoot -Filter '*.psd1' -Recurse -ErrorAction SilentlyContinue |
        Where-Object { $_.DirectoryName -notmatch '\\Archive(\\|$)' }

    $cacheKey = Get-PcCommandMapCacheKey -ModulesRoot $modulesRoot -Manifests $manifests
    $dependencyStamp = if (Get-Command Get-PcaiDependencyStamp -ErrorAction SilentlyContinue) {
        Get-PcaiDependencyStamp -InputObject @($manifests)
    } else {
        $cacheKey
    }

    if (Get-Command Get-PcaiSharedCacheEntry -ErrorAction SilentlyContinue) {
        $cachedMap = Get-PcaiSharedCacheEntry -Namespace 'pcai-cli' -Key "command-map::$root" -DependencyStamp $dependencyStamp
        if ($cachedMap) {
            return $cachedMap
        }
    } elseif ($script:CommandMapCache -and $script:CommandMapCacheKey -eq $cacheKey) {
        return $script:CommandMapCache
    }

    $map = @{}

    foreach ($manifest in $manifests) {
        try {
            $data = Import-PowerShellDataFile -Path $manifest.FullName
        } catch {
            continue
        }
        $moduleName = Split-Path -Leaf (Split-Path -Parent $manifest.FullName)
        $commands = $data.PrivateData.PCAI.Commands
        if (-not $commands) { continue }
        foreach ($command in $commands) {
            if (-not $map.ContainsKey($command)) {
                $map[$command] = @()
            }
            if (-not ($map[$command] -contains $moduleName)) {
                $map[$command] += $moduleName
            }
        }
    }

    foreach ($baseCommand in @('help', 'version')) {
        if (-not $map.ContainsKey($baseCommand)) {
            $map[$baseCommand] = @()
        }
    }

    if (Get-Command Set-PcaiSharedCacheEntry -ErrorAction SilentlyContinue) {
        Set-PcaiSharedCacheEntry -Namespace 'pcai-cli' -Key "command-map::$root" -Value $map -DependencyStamp $dependencyStamp | Out-Null
    } else {
        $script:CommandMapCache = $map
        $script:CommandMapCacheKey = $cacheKey
    }

    return $map
}
