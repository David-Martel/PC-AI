function Get-PcaiCompiledBinaryPath {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [ValidateSet('llamacpp', 'mistralrs')]
        [string]$Backend
    )

    $binaryName = if ($Backend -eq 'llamacpp') { 'pcai-llamacpp.exe' } else { 'pcai-mistralrs.exe' }
    $projectRoot = Get-PcaiProjectRoot

    $candidates = @()
    $configPath = Join-Path $projectRoot 'Config\llm-config.json'
    if (Test-Path $configPath) {
        try {
            $config = Get-Content $configPath -Raw | ConvertFrom-Json
            $paths = $config.evaluation.binSearchPaths
            foreach ($path in $paths) {
                if (-not $path) { continue }
                if ([System.IO.Path]::IsPathRooted($path)) {
                    $candidates += $path
                } else {
                    $candidates += (Join-Path $projectRoot $path)
                }
            }
        } catch {
            Write-Verbose "Failed to parse ${configPath}: $_"
        }
    }

    $userProfile = [Environment]::GetFolderPath('UserProfile')
    $candidates += @(
        (Join-Path $userProfile '.local\bin'),
        (Join-Path $projectRoot 'bin'),
        (Join-Path ([Environment]::GetFolderPath('UserProfile')) '.local\bin'),
        (Join-Path $projectRoot 'Native\pcai_core\pcai_inference\target\release'),
        (Join-Path $projectRoot '.pcai\build\artifacts\pcai-llamacpp'),
        (Join-Path $projectRoot '.pcai\build\artifacts\pcai-mistralrs')
    ) | Where-Object { $_ }

    foreach ($dir in $candidates) {
        $candidate = Join-Path $dir $binaryName
        if (Test-Path $candidate) {
            return (Resolve-Path $candidate).Path
        }
    }

    return $null
}
