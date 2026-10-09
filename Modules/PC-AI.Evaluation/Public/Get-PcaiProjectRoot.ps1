function Get-PcaiProjectRoot {
    [CmdletBinding()]
    param()
    $resolver = Get-Command Resolve-PcaiRepoRoot -CommandType Function -ErrorAction SilentlyContinue
    if ($resolver) {
        $candidate = & $resolver -StartPath $PSScriptRoot
    } elseif ($env:PCAI_ROOT) {
        $candidate = $env:PCAI_ROOT
    } else {
        # Source-only consumers do not require Common or a loaded profile.
        $ancestor = Get-Item -LiteralPath $PSScriptRoot -ErrorAction Stop
        $candidate = $null
        while ($ancestor) {
            if ((Test-Path -LiteralPath (Join-Path $ancestor.FullName 'PC-AI.ps1') -PathType Leaf) -and
                (Test-Path -LiteralPath (Join-Path $ancestor.FullName 'Config/llm-config.json') -PathType Leaf)) {
                $candidate = $ancestor.FullName
                break
            }
            $ancestor = $ancestor.Parent
        }
    }

    if ([string]::IsNullOrWhiteSpace($candidate)) {
        throw 'Unable to locate a PC-AI configuration root. Set PCAI_ROOT to a directory containing PC-AI.ps1 and Config/llm-config.json.'
    }
    $root = Get-Item -LiteralPath $candidate -Force -ErrorAction Stop
    if ($root.PSProvider.Name -ne 'FileSystem' -or -not $root.PSIsContainer) {
        throw 'The PC-AI configuration root must be an existing filesystem directory.'
    }
    foreach ($marker in @('PC-AI.ps1', 'Config/llm-config.json')) {
        if (-not (Test-Path -LiteralPath (Join-Path $root.FullName $marker) -PathType Leaf)) {
            throw 'Unable to locate a genuine PC-AI configuration root. Set PCAI_ROOT to a directory containing PC-AI.ps1 and Config/llm-config.json.'
        }
    }
    return [IO.Path]::GetFullPath($root.FullName)
}
