#Requires -Version 5.1

function Get-PcaiPerfToolPath {
    [CmdletBinding()]
    param()

    if ($env:PCAI_NATIVE_BUNDLE_ROOT) {
        $bundleRoot = [IO.Path]::GetFullPath((Get-Item -LiteralPath $env:PCAI_NATIVE_BUNDLE_ROOT -ErrorAction Stop).FullName)
        $selected = Join-Path $bundleRoot 'pcai-perf.exe'
        # An explicit paired bundle may contain only the DLLs. Its absence of a
        # CLI never authorizes selection of a different checkout's executable.
        if (Test-Path -LiteralPath $selected -PathType Leaf) { return [IO.Path]::GetFullPath((Get-Item -LiteralPath $selected).FullName) }
        return $null
    }

    $moduleRoot = Split-Path -Parent $PSScriptRoot
    $candidates = @(
        (Join-Path $moduleRoot 'bin\pcai-perf.exe')
        $(if ($env:PCAI_ROOT) { Join-Path $env:PCAI_ROOT 'bin\pcai-perf.exe' })
        (Join-Path $moduleRoot '..\..\bin\pcai-perf.exe')
        (Join-Path $moduleRoot '..\..\Native\pcai_core\target\release\pcai-perf.exe')
        (Join-Path $moduleRoot '..\..\.pcai\build\artifacts\pcai-perf\pcai-perf.exe')
        (Join-Path $env:USERPROFILE 'PC_AI\bin\pcai-perf.exe')
        (Get-RustToolPath -ToolName 'pcai-perf')
    ) | Where-Object { $_ } | ForEach-Object { [System.IO.Path]::GetFullPath($_) } | Select-Object -Unique

    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            return $candidate
        }
    }

    return $null
}
