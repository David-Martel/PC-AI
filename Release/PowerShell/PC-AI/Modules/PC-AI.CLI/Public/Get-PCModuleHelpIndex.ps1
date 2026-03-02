function Get-PCModuleHelpIndex {
    [CmdletBinding()]
    param(
        [string[]]$Modules,
        [string]$ProjectRoot
    )

    $root = if ($ProjectRoot) { $ProjectRoot } else { $script:ProjectRoot }
    $modulesRoot = Join-Path $root 'Modules'
    if (-not (Test-Path $modulesRoot)) { return @() }

    if (-not $Modules -or $Modules.Count -eq 0) {
        $Modules = Get-ChildItem -Path $modulesRoot -Directory -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Name
    }

    $allEntries = @()
    foreach ($moduleName in $Modules) {
        $modulePath = Join-Path $modulesRoot $moduleName
        $publicPath = Join-Path $modulePath 'Public'
        if (-not (Test-Path $publicPath)) { continue }

        $files = Get-ChildItem -Path $publicPath -Filter '*.ps1' -Recurse -ErrorAction SilentlyContinue | Select-Object -ExpandProperty FullName
        if (-not $files) { continue }

        $entries = Get-HelpEntriesFromFiles -Paths $files
        foreach ($entry in $entries) {
            $allEntries += [PSCustomObject]@{
                Module = $moduleName
                Name = $entry.Name
                Synopsis = $entry.Synopsis
                Description = $entry.Description
                SourcePath = $entry.SourcePath
                Parameters = $entry.Parameters
                Examples = $entry.Examples
            }
        }
    }

    return $allEntries
}
