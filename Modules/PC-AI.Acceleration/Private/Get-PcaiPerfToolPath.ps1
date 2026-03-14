#Requires -Version 5.1

function Get-PcaiPerfToolPath {
    [CmdletBinding()]
    param()

    $moduleRoot = Split-Path -Parent $PSScriptRoot
    $candidates = @(
        (Join-Path $moduleRoot 'bin\pcai-perf.exe')
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
