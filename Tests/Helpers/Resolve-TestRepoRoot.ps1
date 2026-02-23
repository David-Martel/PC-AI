function Resolve-TestRepoRoot {
    [CmdletBinding()]
    [OutputType([string])]
    param(
        [string]$StartPath = $PSScriptRoot
    )

    if ($env:PCAI_ROOT -and (Test-Path $env:PCAI_ROOT)) {
        return (Resolve-Path -Path $env:PCAI_ROOT).Path
    }

    $repoCandidate = if ($StartPath) { $StartPath } else { (Get-Location).ProviderPath }
    if (Test-Path $repoCandidate -PathType Leaf) {
        $repoCandidate = Split-Path -Parent $repoCandidate
    }

    $commonRuntimeHelper = Join-Path (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)) 'Modules\PC-AI.Common\Public\Get-PcaiRuntimeConfig.ps1'
    if (Test-Path $commonRuntimeHelper) {
        . $commonRuntimeHelper
        if (Get-Command Resolve-PcaiRepoRoot -ErrorAction SilentlyContinue) {
            return Resolve-PcaiRepoRoot -StartPath $repoCandidate
        }
    }

    $cursor = $repoCandidate
    while ($cursor) {
        if ((Test-Path (Join-Path $cursor 'PC-AI.ps1')) -or
            (Test-Path (Join-Path $cursor 'AGENTS.md')) -or
            (Test-Path (Join-Path $cursor '.git'))) {
            return $cursor
        }

        $parent = Split-Path -Parent $cursor
        if ([string]::IsNullOrWhiteSpace($parent) -or $parent -eq $cursor) {
            break
        }

        $cursor = $parent
    }

    return $repoCandidate
}