param(
    [string]$OutputDirectory = (Join-Path $PSScriptRoot 'direct-pass-20260606')
)

$ErrorActionPreference = 'Stop'
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null

$targets = @(
    'OneDrive Reporting Task-S-1-5-21-357761096-1057752491-2136275046-1003',
    'OneDrive Startup Task-S-1-5-21-357761096-1057752491-2136275046-1003'
)

$before = foreach ($name in $targets) {
    Get-ScheduledTask -TaskName $name -TaskPath '\' -ErrorAction Stop |
        Select-Object TaskPath, TaskName, State, Author,
            @{ Name = 'UserId'; Expression = { $_.Principal.UserId } },
            @{ Name = 'Actions'; Expression = { @($_.Actions | ForEach-Object { "$($_.Execute) $($_.Arguments)".Trim() }) } }
}

$before | ConvertTo-Json -Depth 8 |
    Set-Content -LiteralPath (Join-Path $OutputDirectory 'onedrive-wsiaccount-tasks-before-disable.json') -Encoding UTF8

$results = foreach ($name in $targets) {
    Disable-ScheduledTask -TaskName $name -TaskPath '\' -ErrorAction Stop |
        Select-Object TaskPath, TaskName, State,
            @{ Name = 'UserId'; Expression = { $_.Principal.UserId } }
}

$results | ConvertTo-Json -Depth 8 |
    Set-Content -LiteralPath (Join-Path $OutputDirectory 'onedrive-wsiaccount-tasks-disable-result.json') -Encoding UTF8

$after = foreach ($name in $targets) {
    Get-ScheduledTask -TaskName $name -TaskPath '\' -ErrorAction Stop |
        Select-Object TaskPath, TaskName, State,
            @{ Name = 'UserId'; Expression = { $_.Principal.UserId } }
}

$after | ConvertTo-Json -Depth 8 |
    Set-Content -LiteralPath (Join-Path $OutputDirectory 'onedrive-wsiaccount-tasks-after-disable.json') -Encoding UTF8
