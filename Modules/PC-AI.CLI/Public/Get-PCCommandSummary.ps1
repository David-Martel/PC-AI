function Get-PCCommandSummary {
    [CmdletBinding()]
    param([string]$ProjectRoot)

    $root = if ($ProjectRoot) { $ProjectRoot } else { $script:ProjectRoot }
    $modulesRoot = Join-Path $root 'Modules'
    if (-not (Test-Path $modulesRoot)) { return @() }

    $map = Get-PCCommandMap -ProjectRoot $root
    $summaries = @()

    foreach ($command in ($map.Keys | Sort-Object)) {
        $modules = $map[$command]
        $descriptions = @()

        foreach ($moduleName in $modules) {
            $manifest = Join-Path $modulesRoot $moduleName "$moduleName.psd1"
            if (-not (Test-Path $manifest)) { continue }
            try {
                $data = Import-PowerShellDataFile -Path $manifest
                if ($data.Description) {
                    $descriptions += $data.Description.Trim()
                }
            } catch {
                continue
            }
        }

        $summaries += [PSCustomObject]@{
            Command = $command
            Modules = $modules
            Description = ($descriptions -join ' ')
        }
    }

    return $summaries
}
