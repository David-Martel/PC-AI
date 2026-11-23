#Requires -Version 7.0
#Requires -Modules Pester

Describe 'Batch driver registry selection' -Tag 'Unit', 'Drivers', 'Portable' {
    BeforeAll {
        $toolsDir = Join-Path $TestDrive 'Tools'
        $moduleDir = Join-Path $TestDrive 'Modules/PC-AI.Drivers'
        [void][IO.Directory]::CreateDirectory($toolsDir)
        [void][IO.Directory]::CreateDirectory($moduleDir)
        $batchScript = Join-Path $toolsDir 'Update-Drivers.ps1'
        Copy-Item -LiteralPath (Join-Path $PSScriptRoot '../../Tools/Update-Drivers.ps1') -Destination $batchScript
        New-ModuleManifest -Path (Join-Path $moduleDir 'PC-AI.Drivers.psd1') -RootModule 'PC-AI.Drivers.psm1'
        $fakeModule = @'
function Write-FixtureLog {
    param([string] $Command, [string] $RegistryPath, [bool] $Install)
    $data = @{ Command = $Command; RegistryPath = $RegistryPath; Install = $Install }
    [IO.File]::AppendAllText((Join-Path $PSScriptRoot 'fixture-log.jsonl'), ((ConvertTo-Json -InputObject $data -Compress) + [Environment]::NewLine))
}
function Get-DriverRegistry {
    [CmdletBinding()]
    param([string] $RegistryPath, [switch] $ForInstall)
    Write-FixtureLog 'Registry' $RegistryPath $ForInstall.IsPresent
    if ($ForInstall) { $RegistryPath = 'machine-trust.json' }
    $group = if ($ForInstall) { 'one-package' } else { $null }
    [PSCustomObject]@{
        RegistryPath = $RegistryPath
        Devices = @(
            [PSCustomObject]@{ id = 'first'; sharedDriverGroup = $group; driver = @{ installerType = 'exe' } },
            [PSCustomObject]@{ id = 'second'; sharedDriverGroup = $group; driver = @{ installerType = 'exe' } }
        )
    }
}
function Get-DriverReport {
    [CmdletBinding()]
    param([string] $RegistryPath)
    Write-FixtureLog 'Scan' $RegistryPath $false
    foreach ($id in @('first', 'second')) {
        [PSCustomObject]@{
            Status = 'Outdated'; Category = 'network'; DeviceName = $id
            InstalledVersion = '1'; TargetVersion = '2'; RegistryId = $id
        }
    }
}
function Install-DriverUpdate {
    [CmdletBinding()]
    param([string] $DeviceId, [string] $RegistryPath, [string] $DownloadDir, [switch] $DownloadOnly, [switch] $Force)
    Write-FixtureLog 'Install' $RegistryPath (-not $DownloadOnly -and -not $WhatIfPreference)
    [PSCustomObject]@{ DeviceName = $DeviceId; Action = 'fixture'; Success = $true; Message = 'fixture' }
}
'@
        $fakeModule | Set-Content -LiteralPath (Join-Path $moduleDir 'PC-AI.Drivers.psm1') -Encoding utf8NoBOM
        $trace = Join-Path $moduleDir 'fixture-log.jsonl'
    }

    # Protects: the version scan, shared-group mapping and actual install use one effective registry.
    # Detects: scanning the bundled registry before switching to the machine trust registry.
    # Needs: disposable copy of the batch entrypoint and an inert PC-AI.Drivers module in a child pwsh.
    # Breadcrumb: Update-Drivers.ps1 resolves -ForInstall before Phase 1 and reuses the returned registry in Phase 2.
    It 'selects one registry for the batch <Mode> path' -ForEach @(
        @{ Mode = 'Install'; Flags = @(); Registry = 'machine-trust.json'; Commands = @('Registry', 'Scan', 'Install') },
        @{ Mode = 'ScanOnly'; Flags = @('-ScanOnly', '-RegistryPath', 'caller.json'); Registry = 'caller.json'; Commands = @('Scan') },
        @{ Mode = 'DownloadOnly'; Flags = @('-DownloadOnly', '-RegistryPath', 'caller.json'); Registry = 'caller.json'; Commands = @('Scan', 'Registry', 'Install', 'Install') },
        @{ Mode = 'WhatIf'; Flags = @('-WhatIf', '-RegistryPath', 'caller.json'); Registry = 'caller.json'; Commands = @('Scan', 'Registry', 'Install', 'Install') }
    ) {
        [IO.File]::WriteAllText($trace, '')
        $shell = (Get-Process -Id $PID).Path
        & $shell -NoLogo -NoProfile -NonInteractive -File $batchScript @Flags | Out-Null
        $LASTEXITCODE | Should -Be 0
        $log = @(Get-Content -LiteralPath $trace | ConvertFrom-Json -Depth 4)
        ($log.Command -join ',') | Should -BeExactly ($Commands -join ',')
        if ($Mode -eq 'Install') {
            $log[0].Install | Should -BeTrue
            $log[0].RegistryPath | Should -BeNullOrEmpty
            $log = $log | Select-Object -Skip 1
        }
        foreach ($line in $log) { $line.RegistryPath | Should -BeExactly $Registry }
        if ($Mode -ne 'Install') { @($log | Where-Object Install).Count | Should -Be 0 }
    }
}
