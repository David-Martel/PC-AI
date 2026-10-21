#Requires -Version 5.1

function Get-PcaiPerfToolPath {
    [CmdletBinding()]
    param()

    $moduleRoot = Split-Path -Parent $PSScriptRoot
    $repoRelativeCandidates = @(
        (Join-Path $moduleRoot '..\bin\pcai-perf.exe'),
        (Join-Path $moduleRoot '..\..\bin\pcai-perf.exe'),
        (Join-Path $moduleRoot '..\..\Native\pcai_core\target\release\pcai-perf.exe'),
        (Join-Path $moduleRoot '..\..\.pcai\build\artifacts\pcai-perf\pcai-perf.exe')
    ) | ForEach-Object {
        try {
            [System.IO.Path]::GetFullPath($_)
        }
        catch {
            $_
        }
    }

    $candidates = @(
        (Join-Path $moduleRoot 'bin\pcai-perf.exe')
        $repoRelativeCandidates
        $(if ($env:PCAI_ROOT) { Join-Path $env:PCAI_ROOT 'bin\pcai-perf.exe' })
        'C:\codedev\PC_AI\bin\pcai-perf.exe'
        (Get-RustToolPath -ToolName 'pcai-perf')
    ) | Where-Object { $_ } | Select-Object -Unique

    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            return $candidate
        }
    }

    return $null
}
