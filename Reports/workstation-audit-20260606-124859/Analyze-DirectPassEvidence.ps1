param(
    [string]$InputDirectory = (Join-Path $PSScriptRoot 'direct-pass-20260606')
)

$ErrorActionPreference = 'Stop'

function Read-JsonFile {
    param([Parameter(Mandatory)] [string] $Name)
    $path = Join-Path $InputDirectory $Name
    if (-not (Test-Path -LiteralPath $path)) {
        return @()
    }
    $raw = Get-Content -LiteralPath $path -Raw
    if ([string]::IsNullOrWhiteSpace($raw)) {
        return @()
    }
    $value = $raw | ConvertFrom-Json
    if ($null -eq $value) {
        return @()
    }
    @($value)
}

$summary = [ordered]@{}

$events2h = Read-JsonFile 'events-critical-error-warning-2h.json'
$summary.Events2hByProvider = @(
    $events2h |
        Group-Object ProviderName, Id, LevelDisplayName |
        Sort-Object Count -Descending |
        Select-Object Count, Name
)
$summary.Events2hLatest = @(
    $events2h |
        Sort-Object TimeCreated -Descending |
        Select-Object -First 20 TimeCreated, ProviderName, Id, LevelDisplayName, Message
)

$driverEvents = Read-JsonFile 'events-kernel-driver-storage-24h.json'
$summary.DriverStorage24hByProvider = @(
    $driverEvents |
        Group-Object ProviderName, Id, LevelDisplayName |
        Sort-Object Count -Descending |
        Select-Object Count, Name
)
$summary.DriverStorage24hImportant = @(
    $driverEvents |
        Where-Object {
            $_.LevelDisplayName -in @('Critical', 'Error', 'Warning') -or
            $_.Id -in @(41, 6008, 158, 219, 10111, 10120)
        } |
        Sort-Object TimeCreated -Descending |
        Select-Object TimeCreated, ProviderName, Id, LevelDisplayName, Message
)

$oneDriveTasks = Read-JsonFile 'onedrive-scheduled-tasks.json'
$summary.OneDriveTaskSummary = @(
    $oneDriveTasks |
        Select-Object TaskName, State, UserId, LastTaskResult, LastRunTime, NextRunTime,
            @{ Name = 'Action'; Expression = {
                $actions = @($_.Actions)
                ($actions | ForEach-Object {
                    "$($_.Execute) $($_.Arguments)".Trim()
                }) -join '; '
            }}
)
$summary.OneDriveStaleCandidates = @(
    $oneDriveTasks |
        Where-Object {
            $_.UserId -notin @('david', 'SYSTEM') -and
            $_.LastTaskResult -eq 267011 -and
            ([datetime]$_.LastRunTime).Year -lt 2000
        } |
        Select-Object TaskName, UserId, LastTaskResult, LastRunTime, NextRunTime
)

$containers = Read-JsonFile 'docker-containers.json'
$images = Read-JsonFile 'docker-images.json'
$usedImageNames = @($containers | ForEach-Object { $_.Image } | Sort-Object -Unique)
$summary.DockerRunningImages = $usedImageNames
$summary.DockerImages = @(
    $images |
        Select-Object Repository, Tag, ID, CreatedSince, Size,
            @{ Name = 'UsedByContainerName'; Expression = {
                $imageName = if ($_.Tag -and $_.Tag -ne '<none>') { "$($_.Repository):$($_.Tag)" } else { $_.Repository }
                $usedImageNames -contains $imageName
            }}
)
$summary.DockerUnusedTaggedImages = @(
    $summary.DockerImages |
        Where-Object {
            -not $_.UsedByContainerName -and
            $_.Repository -ne '<none>' -and
            $_.Tag -ne '<none>'
        }
)
$summary.DockerDanglingImages = @(
    $summary.DockerImages |
        Where-Object { $_.Repository -eq '<none>' -or $_.Tag -eq '<none>' }
)

$profiles = Read-JsonFile 'powershell-profile-files.json'
$summary.PowerShellAutomationHotspots = @(
    $profiles |
        Where-Object {
            $_.FullName -match '\\(\.machine|\.local|bin|Documents\\PowerShell|Documents\\WindowsPowerShell)\\' -and
            $_.FullName -match '(HVSock|vtss|VLLM|ToolRouter|OneDrive|Docker|RAG|Redis|DNS|profile|startup|init|machine)'
        } |
        Sort-Object FullName |
        Select-Object FullName, Length, LastWriteTime
)

$summary | ConvertTo-Json -Depth 10 |
    Set-Content -LiteralPath (Join-Path $InputDirectory 'analysis-summary.json') -Encoding UTF8

$markdown = @()
$markdown += '# Direct Pass Analysis'
$markdown += ''
$markdown += '## Event Counts'
$markdown += ($summary.Events2hByProvider | Format-Table -AutoSize | Out-String)
$markdown += ''
$markdown += '## Driver and Storage Important Events'
$markdown += ($summary.DriverStorage24hImportant | Select-Object -First 40 | Format-List | Out-String)
$markdown += ''
$markdown += '## OneDrive Tasks'
$markdown += ($summary.OneDriveTaskSummary | Format-Table -AutoSize | Out-String)
$markdown += ''
$markdown += '## OneDrive Stale Candidates'
$markdown += ($summary.OneDriveStaleCandidates | Format-Table -AutoSize | Out-String)
$markdown += ''
$markdown += '## Docker Running Images'
$markdown += ($summary.DockerRunningImages | ForEach-Object { "- $_" })
$markdown += ''
$markdown += '## Docker Unused Tagged Images'
$markdown += ($summary.DockerUnusedTaggedImages | Format-Table -AutoSize | Out-String)
$markdown += ''
$markdown += '## Docker Dangling Images'
$markdown += ($summary.DockerDanglingImages | Format-Table -AutoSize | Out-String)
$markdown += ''
$markdown += '## PowerShell Automation Hotspots'
$markdown += ($summary.PowerShellAutomationHotspots | Select-Object -First 200 | Format-Table -AutoSize | Out-String)

$markdown | Set-Content -LiteralPath (Join-Path $InputDirectory 'analysis-summary.md') -Encoding UTF8
