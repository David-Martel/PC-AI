#Requires -Version 7.0
#Requires -Modules @{ ModuleName = 'Pester'; ModuleVersion = '5.0.0' }

BeforeAll {
    $script:RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
    $script:ToolPaths = @(
        Join-Path $script:RepoRoot 'Tools\Apply-ProcessLassoUiSyncTuning.ps1'
        Join-Path $script:RepoRoot 'Tools\Collect-UiGlitchDiagnostics.ps1'
        Join-Path $script:RepoRoot 'Tools\Ensure-ProcessLassoGovernor.ps1'
        Join-Path $script:RepoRoot 'Tools\Test-ProcessLassoBootSafety.ps1'
        Join-Path $script:RepoRoot 'Tools\Collect-BootDiagnostics.ps1'
        Join-Path $script:RepoRoot 'Tools\Collect-DrivePerformanceSyncRisk.ps1'
        Join-Path $script:RepoRoot 'Tools\Register-ProcessLassoGovernorWatchdog.ps1'
        Join-Path $script:RepoRoot 'Tools\Repair-OneDriveSync.ps1'
        Join-Path $script:RepoRoot 'Tools\Test-BootMountHealth.ps1'
        Join-Path $script:RepoRoot 'Tools\Test-SyncProviderHealth.ps1'
    )
    $script:DiagnosticToolPaths = @(
        Join-Path $script:RepoRoot 'Tools\Collect-BootDiagnostics.ps1'
        Join-Path $script:RepoRoot 'Tools\Collect-DrivePerformanceSyncRisk.ps1'
        Join-Path $script:RepoRoot 'Tools\Test-BootMountHealth.ps1'
        Join-Path $script:RepoRoot 'Tools\Test-SyncProviderHealth.ps1'
    )
    $script:DryRunToolPaths = @(
        Join-Path $script:RepoRoot 'Tools\Apply-ProcessLassoUiSyncTuning.ps1'
        Join-Path $script:RepoRoot 'Tools\Collect-UiGlitchDiagnostics.ps1'
        Join-Path $script:RepoRoot 'Tools\Ensure-ProcessLassoGovernor.ps1'
        Join-Path $script:RepoRoot 'Tools\Test-ProcessLassoBootSafety.ps1'
        Join-Path $script:RepoRoot 'Tools\Collect-BootDiagnostics.ps1'
        Join-Path $script:RepoRoot 'Tools\Collect-DrivePerformanceSyncRisk.ps1'
        Join-Path $script:RepoRoot 'Tools\Register-ProcessLassoGovernorWatchdog.ps1'
        Join-Path $script:RepoRoot 'Tools\Repair-OneDriveSync.ps1'
        Join-Path $script:RepoRoot 'Tools\Test-BootMountHealth.ps1'
        Join-Path $script:RepoRoot 'Tools\Test-SyncProviderHealth.ps1'
    )
    $script:ExternalWindowsScripts = @(
        'C:\Users\david\unifi_api\scripts\windows\Start-UDMDriveStack.ps1'
        'C:\Users\david\unifi_api\scripts\windows\Register-UDMDriveStartupTask.ps1'
        'C:\Users\david\bin\scripts\home-root-archive\optimize-registry-derp.ps1'
        'C:\Users\david\unifi_api\submodules\qnap\scripts\QNAP_Performance_Quick_Setup.ps1'
    ) | Where-Object { Test-Path -LiteralPath $_ }

    function Get-ScriptAst {
        param([Parameter(Mandatory)] [string]$Path)

        $tokens = $null
        $errors = $null
        $ast = [System.Management.Automation.Language.Parser]::ParseFile($Path, [ref]$tokens, [ref]$errors)
        [pscustomobject]@{
            Ast = $ast
            Errors = $errors
        }
    }
}

Describe 'Boot validation tooling scripts' -Tag 'Unit', 'Boot', 'Portable' {
    It 'ships all worker-owned validation scripts' {
        foreach ($path in @($script:ToolPaths + $script:ExternalWindowsScripts)) {
            $path | Should -Exist
        }
    }

    It 'parses without PowerShell syntax errors' {
        foreach ($path in @($script:ToolPaths + $script:ExternalWindowsScripts)) {
            $parsed = Get-ScriptAst -Path $path
            @($parsed.Errors).Count | Should -Be 0
        }
    }

    It 'keeps diagnostics read-only by avoiding known mutation cmdlets' {
        $blockedCommands = @(
            'Mount-VHD',
            'Dismount-VHD',
            'Register-ScheduledTask',
            'Set-ScheduledTask',
            'Unregister-ScheduledTask',
            'Set-ItemProperty',
            'New-ItemProperty',
            'Remove-ItemProperty',
            'Stop-Process',
            'Start-Service',
            'Stop-Service',
            'Restart-Service'
        )

        foreach ($path in $script:DiagnosticToolPaths) {
            $parsed = Get-ScriptAst -Path $path
            $commands = $parsed.Ast.FindAll({
                param($node)
                $node -is [System.Management.Automation.Language.CommandAst]
            }, $true) | ForEach-Object { $_.GetCommandName() } | Where-Object { $_ }

            foreach ($blocked in $blockedCommands) {
                $commands | Should -Not -Contain $blocked
            }
        }
    }

    It 'collector exposes the post-reboot verifier and Process Lasso integration point' {
        $content = Get-Content -LiteralPath (Join-Path $script:RepoRoot 'Tools\Collect-BootDiagnostics.ps1') -Raw
        $content | Should -Match '\$PostRebootVerify'
        $content | Should -Match 'Test-BootMountHealth\.ps1'
        $content | Should -Match 'Test-SyncProviderHealth\.ps1'
        $content | Should -Match 'Test-ProcessLassoBootSafety\.ps1'
    }

    It 'documents and exposes Help CLI switches on session scripts' {
        foreach ($path in @($script:ToolPaths + $script:ExternalWindowsScripts)) {
            $content = Get-Content -LiteralPath $path -Raw
            $content | Should -Match '\.PARAMETER Help'
            $content | Should -Match '\[switch\]\$Help'
            $content | Should -Match "Alias\('h'"
            $content | Should -Match '--help'
        }
    }

    It 'documents and exposes DryRun CLI switches on write-capable scripts' {
        foreach ($path in @($script:DryRunToolPaths + $script:ExternalWindowsScripts)) {
            $content = Get-Content -LiteralPath $path -Raw
            $content | Should -Match '\.PARAMETER DryRun'
            $content | Should -Match '\[switch\]\$DryRun'
            $content | Should -Match '--DryRun'
        }
    }

    It 'health scripts can return machine-readable PassThru objects' {
        foreach ($path in @(
            (Join-Path $script:RepoRoot 'Tools\Test-BootMountHealth.ps1'),
            (Join-Path $script:RepoRoot 'Tools\Test-SyncProviderHealth.ps1')
        )) {
            $content = Get-Content -LiteralPath $path -Raw
            $content | Should -Match '\[switch\]\$PassThru'
            $content | Should -Match '\[switch\]\$FailOnIssue'
            $content | Should -Match '\[string\]\$OutputJson'
        }
    }

    It 'supports --help and -h without executing script work' {
        foreach ($path in @($script:ToolPaths + $script:ExternalWindowsScripts)) {
            foreach ($helpArg in @('--help', '-h')) {
                $output = & pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -File $path $helpArg 2>&1
                $LASTEXITCODE | Should -Be 0
                ($output | Out-String) | Should -Match 'SYNOPSIS'
            }
        }
    }

    It 'dry-runs collectors without creating output directories' {
        $tempRoot = Join-Path ([System.IO.Path]::GetTempPath()) ('pcai-boot-cli-' + [guid]::NewGuid().ToString('N'))
        $uiRoot = Join-Path $tempRoot 'ui'
        $bootRoot = Join-Path $tempRoot 'boot'

        $ui = & (Join-Path $script:RepoRoot 'Tools\Collect-UiGlitchDiagnostics.ps1') -OutputRoot $uiRoot -DryRun
        $boot = & (Join-Path $script:RepoRoot 'Tools\Collect-BootDiagnostics.ps1') -OutputRoot $bootRoot -DryRun
        $drive = & (Join-Path $script:RepoRoot 'Tools\Collect-DrivePerformanceSyncRisk.ps1') -OutputRoot (Join-Path $tempRoot 'drive') -DryRun

        $ui.DryRun | Should -BeTrue
        $boot.DryRun | Should -BeTrue
        $drive.DryRun | Should -BeTrue
        Test-Path -LiteralPath $uiRoot | Should -BeFalse
        Test-Path -LiteralPath $bootRoot | Should -BeFalse
        Test-Path -LiteralPath (Join-Path $tempRoot 'drive') | Should -BeFalse
    }

    It 'dry-runs Process Lasso config updates without modifying the target INI' {
        $tempRoot = Join-Path ([System.IO.Path]::GetTempPath()) ('pcai-pl-dryrun-' + [guid]::NewGuid().ToString('N'))
        New-Item -Path $tempRoot -ItemType Directory -Force | Out-Null
        $configPath = Join-Path $tempRoot 'prolasso.ini'
        $reportPath = Join-Path $tempRoot 'report.json'
        $ini = @"
[OutOfControlProcessRestraint]
OocExclusions=explorer.exe
[MemoryManagement]
SmartTrimExclusions=explorer.exe
[Logging]
IncludeCommandLines=false
"@
        Set-Content -LiteralPath $configPath -Value $ini -Encoding Unicode
        $before = Get-Content -LiteralPath $configPath -Raw -Encoding Unicode

        & (Join-Path $script:RepoRoot 'Tools\Apply-ProcessLassoUiSyncTuning.ps1') -ConfigPath $configPath -ReportPath $reportPath -DryRun | Out-Null

        $after = Get-Content -LiteralPath $configPath -Raw -Encoding Unicode
        $after | Should -Be $before
        $report = Get-Content -LiteralPath $reportPath -Raw | ConvertFrom-Json
        $report.DryRun | Should -BeTrue
        $report.Changed | Should -BeTrue

        Remove-Item -LiteralPath $tempRoot -Recurse -Force -ErrorAction SilentlyContinue
    }

    It 'dry-runs report-writing health checks without writing OutputJson' {
        $tempRoot = Join-Path ([System.IO.Path]::GetTempPath()) ('pcai-boot-health-' + [guid]::NewGuid().ToString('N'))
        New-Item -Path $tempRoot -ItemType Directory -Force | Out-Null
        $mountJson = Join-Path $tempRoot 'mount.json'
        $syncJson = Join-Path $tempRoot 'sync.json'

        & (Join-Path $script:RepoRoot 'Tools\Test-BootMountHealth.ps1') -SinceMinutes 1 -OutputJson $mountJson -DryRun | Out-Null
        & (Join-Path $script:RepoRoot 'Tools\Test-SyncProviderHealth.ps1') -SinceMinutes 1 -OutputJson $syncJson -DryRun | Out-Null

        Test-Path -LiteralPath $mountJson | Should -BeFalse
        Test-Path -LiteralPath $syncJson | Should -BeFalse

        Remove-Item -LiteralPath $tempRoot -Recurse -Force -ErrorAction SilentlyContinue
    }

    It 'dry-runs UDM task registration without mutating Task Scheduler' {
        $registerScript = 'C:\Users\david\unifi_api\scripts\windows\Register-UDMDriveStartupTask.ps1'
        if (-not (Test-Path -LiteralPath $registerScript)) {
            Set-ItResult -Skipped -Because 'UDM registration script is not present on this machine.'
            return
        }

        $output = & pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -File $registerScript --DryRun 2>&1
        $LASTEXITCODE | Should -Be 0
        ($output | Out-String) | Should -Match 'would register scheduled task'
    }

    It 'dry-runs Process Lasso governor watchdog registration without mutating Task Scheduler' {
        $registerScript = Join-Path $script:RepoRoot 'Tools\Register-ProcessLassoGovernorWatchdog.ps1'
        $output = & pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -File $registerScript -DryRun 2>&1
        $LASTEXITCODE | Should -Be 0
        ($output | Out-String) | Should -Match 'would register scheduled task'
    }

    It 'dry-runs OneDrive repair without downloading installer or mutating sync state' {
        $tempRoot = Join-Path ([System.IO.Path]::GetTempPath()) ('pcai-onedrive-repair-' + [guid]::NewGuid().ToString('N'))
        $repairScript = Join-Path $script:RepoRoot 'Tools\Repair-OneDriveSync.ps1'

        $result = & pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -File $repairScript -OutputDirectory $tempRoot -DownloadInstaller -InstallLatest -StopBeforeInstall -StartAfterRepair -DryRun 2>&1
        $LASTEXITCODE | Should -Be 0
        ($result | Out-String) | Should -Match 'SkippedDryRun'
        Test-Path -LiteralPath (Join-Path $tempRoot 'OneDriveSetup.exe') | Should -BeFalse
        Test-Path -LiteralPath (Join-Path $tempRoot 'summary.json') | Should -BeTrue

        Remove-Item -LiteralPath $tempRoot -Recurse -Force -ErrorAction SilentlyContinue
    }

    It 'dry-runs risky performance tuning scripts without writing snapshots or reports' {
        $tempRoot = Join-Path ([System.IO.Path]::GetTempPath()) ('pcai-risky-dryrun-' + [guid]::NewGuid().ToString('N'))
        New-Item -Path $tempRoot -ItemType Directory -Force | Out-Null
        $registryScript = 'C:\Users\david\bin\scripts\home-root-archive\optimize-registry-derp.ps1'
        $qnapScript = 'C:\Users\david\unifi_api\submodules\qnap\scripts\QNAP_Performance_Quick_Setup.ps1'

        if (Test-Path -LiteralPath $registryScript) {
            $snapshot = Join-Path $tempRoot 'registry-snapshot.json'
            $report = Join-Path $tempRoot 'registry-report.json'
            $result = & pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -File $registryScript -DryRun -SnapshotPath $snapshot -OutputJson $report 2>&1
            $LASTEXITCODE | Should -Be 0
            ($result | Out-String) | Should -Match 'DryRun'
            Test-Path -LiteralPath $snapshot | Should -BeFalse
            Test-Path -LiteralPath $report | Should -BeFalse
        }

        if (Test-Path -LiteralPath $qnapScript) {
            $snapshot = Join-Path $tempRoot 'qnap-snapshot.json'
            $report = Join-Path $tempRoot 'qnap-report.json'
            $result = & pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -File $qnapScript -DryRun -SnapshotPath $snapshot -OutputJson $report 2>&1
            $LASTEXITCODE | Should -Be 0
            ($result | Out-String) | Should -Match 'Dry run complete'
            Test-Path -LiteralPath $snapshot | Should -BeFalse
            Test-Path -LiteralPath $report | Should -BeFalse
        }

        Remove-Item -LiteralPath $tempRoot -Recurse -Force -ErrorAction SilentlyContinue
    }
}
