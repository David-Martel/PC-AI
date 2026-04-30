# List-InstalledUtils.ps1
# Lists all installed custom utilities in ~/bin and ~/.local/bin

param(
    [switch]$Detailed,
    [switch]$GroupByType
)

$binDirs = @(
    "$env:USERPROFILE\bin",
    "$env:USERPROFILE\.local\bin"
)

$utilities = @()

foreach ($dir in $binDirs) {
    if (Test-Path $dir) {
        $items = Get-ChildItem -Path $dir -File | Where-Object {
            $_.Extension -in @('.exe', '.cmd', '.bat', '.ps1') -or
            $_.LinkType -eq 'SymbolicLink'
        }

        foreach ($item in $items) {
            $type = if ($item.LinkType -eq 'SymbolicLink') {
                "Symlink"
            } elseif ($item.Name -match '^uu-') {
                "CoreUtils"
            } elseif ($item.Extension -eq '.exe') {
                "Executable"
            } else {
                "Script"
            }

            $target = if ($item.LinkType -eq 'SymbolicLink') {
                $item.Target
            } else {
                $null
            }

            $utilities += [PSCustomObject]@{
                Name = $item.Name
                Type = $type
                Location = $dir
                Target = $target
                Size = if ($item.LinkType -ne 'SymbolicLink') { $item.Length } else { $null }
            }
        }
    }
}

if ($GroupByType) {
    Write-Host "`n=== Installed Utilities by Type ===" -ForegroundColor Cyan

    $groups = $utilities | Group-Object Type
    foreach ($group in $groups | Sort-Object Name) {
        Write-Host "`n$($group.Name) ($($group.Count)):" -ForegroundColor Yellow
        foreach ($util in $group.Group | Sort-Object Name) {
            if ($Detailed -and $util.Target) {
                Write-Host "  $($util.Name) -> $($util.Target)" -ForegroundColor Green
            } else {
                Write-Host "  $($util.Name)" -ForegroundColor Green
            }
        }
    }
} else {
    Write-Host "`n=== Installed Utilities ===" -ForegroundColor Cyan

    if ($Detailed) {
        $utilities | Sort-Object Name | Format-Table -AutoSize
    } else {
        $utilities | Sort-Object Name | ForEach-Object {
            $prefix = switch ($_.Type) {
                "CoreUtils" { "[UU] " }
                "Symlink" { "[→] " }
                "Script" { "[S] " }
                default { "[X] " }
            }
            Write-Host "$prefix$($_.Name)" -NoNewline -ForegroundColor $(
                switch ($_.Type) {
                    "CoreUtils" { "Cyan" }
                    "Symlink" { "Yellow" }
                    "Script" { "Magenta" }
                    default { "White" }
                }
            )
            if ($_.Name -match '^(cmd|powershell|pwsh)\.exe$') {
                Write-Host " (Shell)" -ForegroundColor DarkGray -NoNewline
            }
            Write-Host ""
        }
    }
}

Write-Host "`n=== Summary ===" -ForegroundColor Cyan
$summary = $utilities | Group-Object Location
foreach ($loc in $summary) {
    Write-Host "$($loc.Name): $($loc.Count) utilities" -ForegroundColor White
}
Write-Host "Total: $($utilities.Count) utilities" -ForegroundColor Green

# Check for uutils manifest
$manifestPath = "$env:USERPROFILE\.local\bin\uu-coreutils-manifest.json"
if (Test-Path $manifestPath) {
    $manifest = Get-Content $manifestPath | ConvertFrom-Json
    Write-Host "`nCoreutils installed: $($manifest.InstallDate)" -ForegroundColor DarkGray
    Write-Host "CoreUtils count: $($manifest.InstalledUtilities.Count)" -ForegroundColor DarkGray
}