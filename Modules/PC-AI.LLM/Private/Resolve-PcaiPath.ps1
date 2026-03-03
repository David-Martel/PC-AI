#Requires -Version 5.1

<#
.SYNOPSIS
    Resolves PC_AI paths dynamically with environment variable support.

.PARAMETER PathType
    The type of path to resolve: Root, Config, HVSockConfig, Models, Logs

.OUTPUTS
    String path to the requested resource
#>
function Resolve-PcaiPath {
    [CmdletBinding()]
    [OutputType([string])]
    param(
        [Parameter(Mandatory)]
        [ValidateSet('Root', 'Config', 'HVSockConfig', 'Models', 'Logs', 'Tools')]
        [string]$PathType
    )

    function Find-PcaiRoot {
        [CmdletBinding()]
        [OutputType([string])]
        param([string]$StartPath)

        if (-not $StartPath -or -not (Test-Path $StartPath)) {
            return $null
        }

        try {
            $cursor = (Resolve-Path -Path $StartPath -ErrorAction Stop).Path
        } catch {
            return $null
        }

        while ($cursor) {
            $hasRepoMarkers =
                (Test-Path (Join-Path $cursor 'PC-AI.ps1')) -or
                (Test-Path (Join-Path $cursor 'AGENTS.md')) -or
                (Test-Path (Join-Path $cursor '.git'))
            if ($hasRepoMarkers) {
                return $cursor
            }

            $parent = Split-Path -Parent $cursor
            if ([string]::IsNullOrWhiteSpace($parent) -or $parent -eq $cursor) {
                break
            }
            $cursor = $parent
        }

        return $null
    }

    # Determine root path (shared helper -> env override -> module location -> working directory)
    $root = $null
    if (Get-Command Resolve-PcaiRepoRoot -ErrorAction SilentlyContinue) {
        $root = Resolve-PcaiRepoRoot -StartPath $PSScriptRoot
    }

    if (-not $root -and $env:PCAI_ROOT -and (Test-Path $env:PCAI_ROOT)) {
        $root = $env:PCAI_ROOT
    }

    if (-not $root -and $PSScriptRoot) {
        # Private -> PC-AI.LLM -> Modules -> repo root
        $moduleDerivedRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
        $root = Find-PcaiRoot -StartPath $moduleDerivedRoot
        if (-not $root) {
            $root = $moduleDerivedRoot
        }
    }

    if (-not $root) {
        $root = Find-PcaiRoot -StartPath (Get-Location).ProviderPath
    }

    if (-not $root) {
        $root = (Get-Location).ProviderPath
    }

    switch ($PathType) {
        'Root' { return $root }
        'Config' { return Join-Path $root 'Config' }
        'HVSockConfig' { return Join-Path $root 'Config\hvsock-proxy.conf' }
        'Models' { return Join-Path $root 'Models' }
        'Logs' { return Join-Path $root 'Reports\Logs' }
        'Tools' { return Join-Path $root 'Config\pcai-tools.json' }
        default { return $root }
    }
}
