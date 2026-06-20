param(
    [string]$OutputDirectory = (Join-Path $PSScriptRoot 'direct-pass-20260606')
)

$ErrorActionPreference = 'Stop'
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null

$targets = @(
    'OneDrive Reporting Task-S-1-5-21-357761096-1057752491-2136275046-1007',
    'OneDrive Startup Task-S-1-5-21-357761096-1057752491-2136275046-1007',
    'OneDrive Reporting Task-S-1-5-21-357761096-1057752491-2136275046-1009',
    'OneDrive Startup Task-S-1-5-21-357761096-1057752491-2136275046-1009'
)

$before = foreach ($name in $targets) {
    Get-ScheduledTask -TaskName $name -TaskPath '\' -ErrorAction Stop |
        Select-Object TaskPath, TaskName, State, Author,
            @{ Name = 'UserId'; Expression = { $_.Principal.UserId } },
            @{ Name = 'Actions'; Expression = { @($_.Actions | ForEach-Object { "$($_.Execute) $($_.Arguments)".Trim() }) } }
}

$before | ConvertTo-Json -Depth 8 |
    Set-Content -LiteralPath (Join-Path $OutputDirectory 'onedrive-nonprimary-tasks-before-disable.json') -Encoding UTF8

$results = foreach ($name in $targets) {
    Disable-ScheduledTask -TaskName $name -TaskPath '\' -ErrorAction Stop |
        Select-Object TaskPath, TaskName, State,
            @{ Name = 'UserId'; Expression = { $_.Principal.UserId } }
}

$results | ConvertTo-Json -Depth 8 |
    Set-Content -LiteralPath (Join-Path $OutputDirectory 'onedrive-nonprimary-tasks-disable-result.json') -Encoding UTF8

$after = Get-ScheduledTask -ErrorAction SilentlyContinue |
    Where-Object { $_.TaskName -like '*OneDrive*' } |
    ForEach-Object {
        $info = $_ | Get-ScheduledTaskInfo -ErrorAction SilentlyContinue
        [pscustomobject]@{
            TaskPath = $_.TaskPath
            TaskName = $_.TaskName
            State = $_.State
            UserId = $_.Principal.UserId
            LastTaskResult = $info.LastTaskResult
            LastRunTime = $info.LastRunTime
            NextRunTime = $info.NextRunTime
        }
    }

$after | ConvertTo-Json -Depth 8 |
    Set-Content -LiteralPath (Join-Path $OutputDirectory 'onedrive-tasks-after-nonprimary-disable.json') -Encoding UTF8
