function Get-PCCommandMap {
    [CmdletBinding()]
    param([string]$ProjectRoot)

    $root = if ($ProjectRoot) { $ProjectRoot } else { $script:ProjectRoot }
    $modulesRoot = Join-Path $root 'Modules'
    if (-not (Test-Path $modulesRoot)) { return @{} }

    $manifests = Get-ChildItem -Path $modulesRoot -Filter '*.psd1' -Recurse -ErrorAction SilentlyContinue |
        Where-Object { $_.DirectoryName -notmatch '\\Archive(\\|$)' }

    $cacheKey = Get-PcCommandMapCacheKey -ModulesRoot $modulesRoot -Manifests $manifests
    if ($script:CommandMapCache -and $script:CommandMapCacheKey -eq $cacheKey) {
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

    $script:CommandMapCache = $map
    $script:CommandMapCacheKey = $cacheKey

    return $map
}
