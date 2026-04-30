<#
.SYNOPSIS
Moves workstation system scripts into this repo and repoints scheduled tasks.

.DESCRIPTION
Centralizes PowerShell and command scripts that are used by Task Scheduler or
that can modify workstation startup, network, sync, WSL/Docker, RAG Redis, or
developer-tool state. The script preserves source provenance under
Tools\SystemScripts and can run in dry-run mode before any move.

.PARAMETER Apply
Perform the move and scheduled-task updates. Without this switch the script
runs as a dry-run.

.PARAMETER DestinationRoot
Destination folder. Defaults to Tools\SystemScripts under this repository.

.PARAMETER ReportPath
JSON report path. Defaults to Reports\system-script-migration-<timestamp>.json.

.PARAMETER RemoveEmptySourceDirectories
Remove explicitly migrated source directories when they are empty after moves.

.PARAMETER Help
Print help and exit. The aliases -h and --help are also accepted.

.PARAMETER DryRun
Force dry-run mode. The long form --DryRun is also accepted.
#>
[CmdletBinding(SupportsShouldProcess)]
param(
    [switch]$Apply,
    [string]$DestinationRoot,
    [string]$ReportPath,
    [switch]$RemoveEmptySourceDirectories = $true,
    [switch]$DryRun,
    [Alias('h', '?')]
    [switch]$Help,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$CliArgs
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$CliArgs = @($CliArgs | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
if (@($CliArgs) -contains '--help') {
    $Help = $true
    $CliArgs = @($CliArgs | Where-Object { $_ -ne '--help' })
}
if (@($CliArgs) -contains '--DryRun') {
    $DryRun = $true
    $CliArgs = @($CliArgs | Where-Object { $_ -ne '--DryRun' })
}
if ($Help) {
    $helpMatch = [regex]::Match((Get-Content -LiteralPath $PSCommandPath -Raw), '(?s)<#\s*(.*?)\s*#>')
    if ($helpMatch.Success) { $helpMatch.Groups[1].Value.Trim() } else { Get-Help -Detailed $PSCommandPath }
    return
}
if (@($CliArgs).Count -gt 0) {
    throw "Unknown CLI argument(s): $($CliArgs -join ', ')"
}

$repoRoot = Split-Path -Parent $PSScriptRoot
if (-not $DestinationRoot) {
    $DestinationRoot = Join-Path $PSScriptRoot 'SystemScripts'
}
if (-not $ReportPath) {
    $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
    $ReportPath = Join-Path $repoRoot "Reports\system-script-migration-$stamp.json"
}

$resolvedRepoRoot = [System.IO.Path]::GetFullPath($repoRoot)
$resolvedDestinationRoot = [System.IO.Path]::GetFullPath($DestinationRoot)
if (-not $resolvedDestinationRoot.StartsWith($resolvedRepoRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "DestinationRoot must stay inside the repository. Repo=$resolvedRepoRoot Destination=$resolvedDestinationRoot"
}

function New-MigrationItem {
    param(
        [Parameter(Mandatory)] [string]$Source,
        [Parameter(Mandatory)] [string]$RelativeDestination,
        [Parameter(Mandatory)] [string]$Reason,
        [ValidateSet('File', 'Directory')] [string]$Kind = 'File'
    )

    [pscustomobject]@{
        Source              = $Source
        Destination         = Join-Path $resolvedDestinationRoot $RelativeDestination
        RelativeDestination = $RelativeDestination
        Reason              = $Reason
        Kind                = $Kind
    }
}

$items = @(
    New-MigrationItem -Kind Directory -Source 'C:\Scripts' -RelativeDestination 'C-Scripts' -Reason 'Legacy WSL/Docker/VHD/startup system-modification scripts'
    New-MigrationItem -Kind Directory -Source 'C:\Users\david\OneDrive\Documents\PowerShell\Scripts' -RelativeDestination 'TaskScheduler\PowerShellScripts' -Reason 'DevEnvironmentStartup task and companion startup scripts'
    New-MigrationItem -Kind Directory -Source 'C:\Users\david\unifi_api\scripts\windows' -RelativeDestination 'unifi_api\scripts\windows' -Reason 'UnifiUdmDriveStackStartup task and companion registrar'
    New-MigrationItem -Kind Directory -Source 'C:\Users\david\unifi_api\scripts\udm_boot' -RelativeDestination 'unifi_api\scripts\udm_boot' -Reason 'UDM drive stack helper scripts referenced by moved startup script'
    New-MigrationItem -Kind Directory -Source 'C:\Users\david\bin\scripts\home-root-archive' -RelativeDestination 'HomeRootArchive' -Reason 'Archived registry/network/cloud-sync/system modification scripts'

    New-MigrationItem -Source 'C:\Users\david\.machine\Unlock-BwVault.ps1' -RelativeDestination 'Machine\Unlock-BwVault.ps1' -Reason 'BW-Auto-Unlock scheduled task'
    New-MigrationItem -Source 'C:\Users\david\.machine\Register-BwAutoUnlockTask.ps1' -RelativeDestination 'Machine\Register-BwAutoUnlockTask.ps1' -Reason 'Companion registrar for BW-Auto-Unlock'
    New-MigrationItem -Source 'C:\Users\david\.machine\Initialize-SecretsAtBoot.ps1' -RelativeDestination 'Machine\Initialize-SecretsAtBoot.ps1' -Reason 'Bitwarden\Initialize-MachineSecrets scheduled task'
    New-MigrationItem -Source 'C:\Users\david\.machine\Register-SecretsBootTask.ps1' -RelativeDestination 'Machine\Register-SecretsBootTask.ps1' -Reason 'Companion registrar for Bitwarden secret initialization'
    New-MigrationItem -Source 'C:\Users\david\.machine\Monitor-UDPSockets.ps1' -RelativeDestination 'Machine\Monitor-UDPSockets.ps1' -Reason 'UDP Socket Monitor scheduled task'
    New-MigrationItem -Source 'C:\Users\david\.machine\Sync-ProfileLogs.ps1' -RelativeDestination 'Machine\Sync-ProfileLogs.ps1' -Reason 'PowerShell\ProfileLogSync scheduled task'
    New-MigrationItem -Source 'C:\Users\david\.machine\Install-ProfileLogSyncTask.ps1' -RelativeDestination 'Machine\Install-ProfileLogSyncTask.ps1' -Reason 'Companion registrar for PowerShell\ProfileLogSync'
    New-MigrationItem -Source 'C:\Users\david\.machine\Migrate-ProfileLogs.ps1' -RelativeDestination 'Machine\Migrate-ProfileLogs.ps1' -Reason 'Profile-log migration companion utility'

    New-MigrationItem -Source 'C:\Users\david\.local\bin\Start-LspmuxServer.ps1' -RelativeDestination 'LocalBin\Start-LspmuxServer.ps1' -Reason 'LspmuxServer scheduled task'
    New-MigrationItem -Source 'C:\Users\david\.local\bin\ensure-everything.ps1' -RelativeDestination 'LocalBin\ensure-everything.ps1' -Reason 'Local environment bootstrap script'
    New-MigrationItem -Source 'C:\Users\david\.local\bin\setup-rustflags.ps1' -RelativeDestination 'LocalBin\setup-rustflags.ps1' -Reason 'Rust build environment mutation script'
    New-MigrationItem -Source 'C:\Users\david\.local\bin\setup-sccache-env.ps1' -RelativeDestination 'LocalBin\setup-sccache-env.ps1' -Reason 'sccache environment mutation script'
    New-MigrationItem -Source 'C:\Users\david\.local\bin\wsl-usb.ps1' -RelativeDestination 'LocalBin\wsl-usb.ps1' -Reason 'USB/WSL system integration script'

    New-MigrationItem -Source 'C:\Users\david\bin\Setup-RAGRedisAutoStart.ps1' -RelativeDestination 'UserBin\Setup-RAGRedisAutoStart.ps1' -Reason 'RAG Redis startup task creator'
    New-MigrationItem -Source 'C:\Users\david\bin\Start-RAGRedisNative.ps1' -RelativeDestination 'UserBin\Start-RAGRedisNative.ps1' -Reason 'RAG Redis startup/system modification script'
    New-MigrationItem -Source 'C:\Users\david\bin\Test-RAGRedisHealth.ps1' -RelativeDestination 'UserBin\Test-RAGRedisHealth.ps1' -Reason 'RAG Redis health/fix script'
    New-MigrationItem -Source 'C:\Users\david\bin\tests\Test-RAGRedisHealth.Tests.ps1' -RelativeDestination 'UserBin\tests\Test-RAGRedisHealth.Tests.ps1' -Reason 'RAG Redis script tests'
    New-MigrationItem -Source 'C:\Users\david\bin\LocalDNSProxy.ps1' -RelativeDestination 'UserBin\LocalDNSProxy.ps1' -Reason 'DNS proxy service/network mutation script'
    New-MigrationItem -Source 'C:\Users\david\bin\Install-AcrylicDNS.ps1' -RelativeDestination 'UserBin\Install-AcrylicDNS.ps1' -Reason 'Acrylic DNS installer/network mutation script'
    New-MigrationItem -Source 'C:\Users\david\bin\dns-proxy.bat' -RelativeDestination 'UserBin\dns-proxy.bat' -Reason 'DNS proxy service wrapper'
    New-MigrationItem -Source 'C:\Users\david\bin\setup-dns-env.ps1' -RelativeDestination 'UserBin\setup-dns-env.ps1' -Reason 'DNS environment mutation script'
    New-MigrationItem -Source 'C:\Users\david\bin\test-dns-setup.ps1' -RelativeDestination 'UserBin\test-dns-setup.ps1' -Reason 'DNS setup validator'
    New-MigrationItem -Source 'C:\Users\david\bin\sccache-manager.ps1' -RelativeDestination 'UserBin\sccache-manager.ps1' -Reason 'sccache service/cache manager'
    New-MigrationItem -Source 'C:\Users\david\bin\Diagnose-Sccache.ps1' -RelativeDestination 'UserBin\Diagnose-Sccache.ps1' -Reason 'sccache diagnostic script'
    New-MigrationItem -Source 'C:\Users\david\bin\Setup-BuildEnvironment.ps1' -RelativeDestination 'UserBin\Setup-BuildEnvironment.ps1' -Reason 'developer environment mutation script'
    New-MigrationItem -Source 'C:\Users\david\bin\Manage-DevTools.ps1' -RelativeDestination 'UserBin\Manage-DevTools.ps1' -Reason 'developer tool install/update script'
    New-MigrationItem -Source 'C:\Users\david\bin\Update-DevUtilities.ps1' -RelativeDestination 'UserBin\Update-DevUtilities.ps1' -Reason 'developer utility update script'
    New-MigrationItem -Source 'C:\Users\david\bin\Fix-Winget.ps1' -RelativeDestination 'UserBin\Fix-Winget.ps1' -Reason 'winget/package repair script'
    New-MigrationItem -Source 'C:\Users\david\bin\Fix-NPM-Issues.ps1' -RelativeDestination 'UserBin\Fix-NPM-Issues.ps1' -Reason 'npm repair script'
    New-MigrationItem -Source 'C:\Users\david\bin\Install-CoreUtils.ps1' -RelativeDestination 'UserBin\Install-CoreUtils.ps1' -Reason 'coreutils install script'
    New-MigrationItem -Source 'C:\Users\david\bin\Install-CoreUtils-Direct.ps1' -RelativeDestination 'UserBin\Install-CoreUtils-Direct.ps1' -Reason 'coreutils direct install script'
    New-MigrationItem -Source 'C:\Users\david\bin\List-InstalledUtils.ps1' -RelativeDestination 'UserBin\List-InstalledUtils.ps1' -Reason 'coreutils inventory companion'
    New-MigrationItem -Source 'C:\Users\david\bin\docker-health.ps1' -RelativeDestination 'UserBin\docker-health.ps1' -Reason 'Docker health/startup diagnostic script'
    New-MigrationItem -Source 'C:\Users\david\bin\mcp-health.ps1' -RelativeDestination 'UserBin\mcp-health.ps1' -Reason 'MCP health/startup diagnostic script'
)

$taskRepoints = @(
    [pscustomobject]@{
        TaskName = 'BW-Auto-Unlock'
        TaskPath = '\'
        OldPath  = 'C:\Users\david\.machine\Unlock-BwVault.ps1'
        NewPath  = Join-Path $resolvedDestinationRoot 'Machine\Unlock-BwVault.ps1'
        WorkingDirectory = Join-Path $resolvedDestinationRoot 'Machine'
    }
    [pscustomobject]@{
        TaskName = 'DevEnvironmentStartup'
        TaskPath = '\'
        OldPath  = 'C:\Users\david\OneDrive\Documents\PowerShell\Scripts\Start-DevEnvironment.ps1'
        NewPath  = Join-Path $resolvedDestinationRoot 'TaskScheduler\PowerShellScripts\Start-DevEnvironment.ps1'
        WorkingDirectory = Join-Path $resolvedDestinationRoot 'TaskScheduler\PowerShellScripts'
    }
    [pscustomobject]@{
        TaskName = 'LspmuxServer'
        TaskPath = '\'
        OldPath  = 'C:/Users/david\.local\bin\Start-LspmuxServer.ps1'
        NewPath  = Join-Path $resolvedDestinationRoot 'LocalBin\Start-LspmuxServer.ps1'
        WorkingDirectory = Join-Path $resolvedDestinationRoot 'LocalBin'
    }
    [pscustomobject]@{
        TaskName = 'Gemini-CLI-Update-stable'
        TaskPath = '\'
        OldPath  = 'C:\Users\david\gemini-cli\update-scripts\check-releases.ps1'
        NewPath  = Join-Path $resolvedDestinationRoot 'TaskScheduler\Gemini\check-releases.ps1'
        WorkingDirectory = Join-Path $resolvedDestinationRoot 'TaskScheduler\Gemini'
    }
    [pscustomobject]@{
        TaskName = 'UDP Socket Monitor'
        TaskPath = '\'
        OldPath  = 'C:\Users\david\.machine\Monitor-UDPSockets.ps1'
        NewPath  = Join-Path $resolvedDestinationRoot 'Machine\Monitor-UDPSockets.ps1'
        WorkingDirectory = Join-Path $resolvedDestinationRoot 'Machine'
    }
    [pscustomobject]@{
        TaskName = 'UnifiUdmDriveStackStartup'
        TaskPath = '\'
        OldPath  = 'C:\Users\david\unifi_api\scripts\windows\Start-UDMDriveStack.ps1'
        NewPath  = Join-Path $resolvedDestinationRoot 'unifi_api\scripts\windows\Start-UDMDriveStack.ps1'
        WorkingDirectory = Join-Path $resolvedDestinationRoot 'unifi_api\scripts\windows'
    }
    [pscustomobject]@{
        TaskName = 'Initialize-MachineSecrets'
        TaskPath = '\Bitwarden\'
        OldPath  = 'C:\Users\david\.machine\Initialize-SecretsAtBoot.ps1'
        NewPath  = Join-Path $resolvedDestinationRoot 'Machine\Initialize-SecretsAtBoot.ps1'
        WorkingDirectory = Join-Path $resolvedDestinationRoot 'Machine'
    }
    [pscustomobject]@{
        TaskName = 'ProfileLogSync'
        TaskPath = '\PowerShell\'
        OldPath  = 'C:\Users\david\.machine\Sync-ProfileLogs.ps1'
        NewPath  = Join-Path $resolvedDestinationRoot 'Machine\Sync-ProfileLogs.ps1'
        WorkingDirectory = Join-Path $resolvedDestinationRoot 'Machine'
    }
)

function Move-MigrationItem {
    param([Parameter(Mandatory)] $Item)

    $exists = Test-Path -LiteralPath $Item.Source
    $record = [ordered]@{
        source      = $Item.Source
        destination = $Item.Destination
        kind        = $Item.Kind
        reason      = $Item.Reason
        exists      = $exists
        action      = 'missing'
    }

    if (-not $exists) {
        return [pscustomobject]$record
    }

    $resolvedSource = [System.IO.Path]::GetFullPath((Resolve-Path -LiteralPath $Item.Source).Path)
    $resolvedDestination = [System.IO.Path]::GetFullPath($Item.Destination)
    if (-not $resolvedDestination.StartsWith($resolvedRepoRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing destination outside repo: $resolvedDestination"
    }
    if (Test-Path -LiteralPath $resolvedDestination) {
        if ($exists) {
            $record.action = 'destination-exists-source-remains'
        }
        else {
            $record.action = 'already-present'
        }
        return [pscustomobject]$record
    }

    $parent = Split-Path -Parent $resolvedDestination
    if ($DryRun -or -not $Apply) {
        $record.action = 'would-move'
        return [pscustomobject]$record
    }

    if ($PSCmdlet.ShouldProcess($resolvedSource, "Move to $resolvedDestination")) {
        try {
            New-Item -ItemType Directory -Path $parent -Force | Out-Null
            Move-Item -LiteralPath $resolvedSource -Destination $resolvedDestination -Force
            $record.action = 'moved'
        }
        catch {
            $record.action = 'move-failed'
            $record.error = $_.Exception.Message
        }
    }

    [pscustomobject]$record
}

function Update-ScheduledTaskActionPath {
    param([Parameter(Mandatory)] $TaskSpec)

    $record = [ordered]@{
        task_name = $TaskSpec.TaskName
        task_path = $TaskSpec.TaskPath
        old_path  = $TaskSpec.OldPath
        new_path  = $TaskSpec.NewPath
        found     = $false
        action    = 'missing-task'
    }

    $task = Get-ScheduledTask -TaskName $TaskSpec.TaskName -TaskPath $TaskSpec.TaskPath -ErrorAction SilentlyContinue
    if (-not $task) {
        return [pscustomobject]$record
    }

    $record.found = $true
    $currentAction = @($task.Actions | Where-Object { $_.Execute -match 'powershell|pwsh' } | Select-Object -First 1)
    if (-not $currentAction) {
        $record.action = 'no-powershell-action'
        return [pscustomobject]$record
    }

    $oldVariants = @($TaskSpec.OldPath, ($TaskSpec.OldPath -replace '\\', '/'))
    $newArguments = $currentAction.Arguments
    $changed = $false
    foreach ($old in $oldVariants) {
        if ($newArguments -like "*$old*") {
            $newArguments = $newArguments.Replace($old, $TaskSpec.NewPath)
            $changed = $true
        }
    }

    if (-not $changed) {
        if ($newArguments -like "*$($TaskSpec.NewPath)*") {
            $record.action = 'already-repointed'
        }
        else {
            $record.action = 'old-path-not-found'
        }
        return [pscustomobject]$record
    }

    if ($DryRun -or -not $Apply) {
        $record.action = 'would-repoint'
        return [pscustomobject]$record
    }

    if ($PSCmdlet.ShouldProcess("$($TaskSpec.TaskPath)$($TaskSpec.TaskName)", "Set action path to $($TaskSpec.NewPath)")) {
        $newAction = New-ScheduledTaskAction -Execute $currentAction.Execute -Argument $newArguments -WorkingDirectory $TaskSpec.WorkingDirectory
        Set-ScheduledTask -TaskName $TaskSpec.TaskName -TaskPath $TaskSpec.TaskPath -Action $newAction | Out-Null
        $record.action = 'repointed'
    }

    [pscustomobject]$record
}

function Remove-EmptyDirectoryTree {
    param([Parameter(Mandatory)] [string]$Path)

    $record = [ordered]@{
        path   = $Path
        exists = Test-Path -LiteralPath $Path
        action = 'missing'
    }
    if (-not $record.exists) { return [pscustomobject]$record }

    $resolved = [System.IO.Path]::GetFullPath((Resolve-Path -LiteralPath $Path).Path)
    $allowedPrefixes = @(
        'C:\Scripts',
        'C:\Users\david\OneDrive\Documents\PowerShell\Scripts',
        'C:\Users\david\unifi_api\scripts\windows',
        'C:\Users\david\unifi_api\scripts\udm_boot',
        'C:\Users\david\bin\scripts\home-root-archive'
    )
    if (-not ($allowedPrefixes | Where-Object { $resolved.Equals($_, [System.StringComparison]::OrdinalIgnoreCase) })) {
        $record.action = 'not-explicitly-allowed'
        return [pscustomobject]$record
    }

    $remaining = @(Get-ChildItem -LiteralPath $resolved -Force -ErrorAction SilentlyContinue)
    if ($remaining.Count -gt 0) {
        $record.action = 'not-empty'
        return [pscustomobject]$record
    }
    if ($DryRun -or -not $Apply) {
        $record.action = 'would-remove'
        return [pscustomobject]$record
    }
    if ($PSCmdlet.ShouldProcess($resolved, 'Remove empty migrated source directory')) {
        try {
            Remove-Item -LiteralPath $resolved -Force
            $record.action = 'removed'
        }
        catch {
            $record.action = 'remove-failed'
            $record.error = $_.Exception.Message
        }
    }
    [pscustomobject]$record
}

$moveResults = @($items | ForEach-Object { Move-MigrationItem -Item $_ })
$taskResults = @($taskRepoints | ForEach-Object { Update-ScheduledTaskActionPath -TaskSpec $_ })
$emptyDirectoryResults = @()
if ($RemoveEmptySourceDirectories) {
    $emptyDirectoryResults = @(
        'C:\Scripts',
        'C:\Users\david\OneDrive\Documents\PowerShell\Scripts',
        'C:\Users\david\unifi_api\scripts\windows',
        'C:\Users\david\unifi_api\scripts\udm_boot',
        'C:\Users\david\bin\scripts\home-root-archive'
    ) | ForEach-Object { Remove-EmptyDirectoryTree -Path $_ }
}

$report = [ordered]@{
    generated_at               = (Get-Date).ToString('o')
    repo_root                  = $resolvedRepoRoot
    destination_root           = $resolvedDestinationRoot
    apply                      = [bool]$Apply
    dry_run                    = [bool]($DryRun -or -not $Apply)
    moved_items                = $moveResults
    scheduled_task_repoints    = $taskResults
    empty_directory_cleanup    = $emptyDirectoryResults
}

if ($DryRun -or -not $Apply) {
    $report | ConvertTo-Json -Depth 8
}
else {
    $reportParent = Split-Path -Parent $ReportPath
    New-Item -ItemType Directory -Path $reportParent -Force | Out-Null
    $report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $ReportPath -Encoding UTF8
    Write-Output "Migration report: $ReportPath"
}
