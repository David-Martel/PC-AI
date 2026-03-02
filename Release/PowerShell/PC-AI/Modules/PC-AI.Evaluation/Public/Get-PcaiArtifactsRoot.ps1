function Get-PcaiArtifactsRoot {
    [CmdletBinding()]
    param()

    $projectRoot = Get-PcaiProjectRoot
    $root = if ($env:PCAI_ARTIFACTS_ROOT) {
        $env:PCAI_ARTIFACTS_ROOT
    } else {
        Join-Path $projectRoot '.pcai'
    }

    if (-not (Test-Path $root)) {
        New-Item -ItemType Directory -Path $root -Force | Out-Null
    }

    return $root
}
