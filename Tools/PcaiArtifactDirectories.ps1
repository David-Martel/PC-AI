#Requires -Version 7.0

function New-PcaiArtifactDirectory {
    <#
    .SYNOPSIS
    Allocates a stable artifact revision without overwriting earlier evidence.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$Root,
        [Parameter(Mandatory)][ValidatePattern('^[A-Za-z][A-Za-z0-9_-]*$')][string]$Name
    )

    $rootPath = [IO.Path]::GetFullPath($Root)
    if ($rootPath -eq [IO.Path]::GetPathRoot($rootPath)) { throw 'Artifact root must be a dedicated directory.' }
    $cursor = $rootPath
    while ($cursor) {
        $component = [IO.Path]::GetFileName($cursor).TrimEnd(' ', '.')
        if ($component -match '^(?i:\$null|AUX|CON|NUL|PRN|COM[1-9]|LPT[1-9])(?:\.|$)') {
            throw 'Unsafe Windows path in artifact root.'
        }
        if ((Test-Path -LiteralPath $cursor) -and
            ((Get-Item -LiteralPath $cursor -Force).Attributes -band [IO.FileAttributes]::ReparsePoint)) {
            throw 'Linked artifact roots require custody review.'
        }
        $parent = [IO.Path]::GetDirectoryName($cursor)
        if ($parent -eq $cursor) { break }
        $cursor = $parent
    }
    [void][IO.Directory]::CreateDirectory($rootPath)
    $lockStream = [IO.File]::Open((Join-Path $rootPath '.artifact-revision.lock'),
        [IO.FileMode]::OpenOrCreate, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
    try {
        $revision = 1
        do {
            $path = Join-Path $rootPath "$Name-r$revision"
            $revision++
        } while (Test-Path -LiteralPath $path)
        [void](New-Item -ItemType Directory -Path $path -ErrorAction Stop)
        return $path
    } finally {
        $lockStream.Dispose()
    }
}
