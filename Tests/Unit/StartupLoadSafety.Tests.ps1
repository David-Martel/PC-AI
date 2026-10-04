#Requires -Version 7.0
#Requires -Modules @{ ModuleName = 'Pester'; ModuleVersion = '5.0.0' }

[Diagnostics.CodeAnalysis.SuppressMessageAttribute('PSAvoidGlobalVars', '', Justification = 'Pester mocks called by the real external script need a dedicated cross-script fixture; AfterAll removes it.')]
param()

BeforeAll {
    $script:StartupTool = Join-Path (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent) 'Tools/InputDiagnostics/Optimize-StartupLoad.ps1'
    $script:RunKey = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Explorer\StartupApproved\Run'
    $script:FolderKey = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Explorer\StartupApproved\StartupFolder'
    $script:MachineKey = 'HKLM:\Software\Microsoft\Windows\CurrentVersion\Explorer\StartupApproved\Run'
}

AfterAll { Remove-Variable -Name PcaiStartupSafetyFixture -Scope Global -ErrorAction SilentlyContinue }

Describe 'StartupLoad full-script mutation safety' {
    BeforeEach {
        Set-StrictMode -Version Latest
        $global:PcaiStartupSafetyFixture = @{}
        $global:PcaiStartupSafetyFixture.RunKey = $script:RunKey
        $global:PcaiStartupSafetyFixture.FolderKey = $script:FolderKey
        $global:PcaiStartupSafetyFixture.MachineKey = $script:MachineKey
        $global:PcaiStartupSafetyFixture.RunValues = @{}
        $global:PcaiStartupSafetyFixture.FolderValues = @{}
        $global:PcaiStartupSafetyFixture.MachineValues = @{}
        $global:PcaiStartupSafetyFixture.Commands = @()
        $global:PcaiStartupSafetyFixture.Writes = [System.Collections.Generic.List[object]]::new()
        $global:PcaiStartupSafetyFixture.BackupRoot = Join-Path $TestDrive ([guid]::NewGuid().ToString())
        Mock Get-CimInstance { $global:PcaiStartupSafetyFixture.Commands }
        Mock Test-Path { $true } -ParameterFilter { $Path -like 'HKCU:*' -or $Path -like 'HKLM:*' }
        Mock Get-ItemProperty {
            if ($Path -eq $global:PcaiStartupSafetyFixture.RunKey) { return [pscustomobject]$global:PcaiStartupSafetyFixture.RunValues }
            if ($Path -eq $global:PcaiStartupSafetyFixture.FolderKey) { return [pscustomobject]$global:PcaiStartupSafetyFixture.FolderValues }
            if ($Path -eq $global:PcaiStartupSafetyFixture.MachineKey) { return [pscustomobject]$global:PcaiStartupSafetyFixture.MachineValues }
            throw "Unexpected registry read: $Path"
        }
        Mock Set-ItemProperty {
            param($Path, $Name, $Value, $Type)
            $global:PcaiStartupSafetyFixture.Writes.Add([pscustomobject]@{ Path = $Path; Name = $Name; Value = $Value.Clone(); Type = $Type })
        }
        Mock Write-Host {}
    }

    It 'writes exactly the legacy enabled 12-byte value and preserves its source and backup' {
        $original = [byte[]]@(2, 9, 8, 7, 6, 5, 4, 3, 2, 1, 255, 11)
        $global:PcaiStartupSafetyFixture.RunValues['Ollama'] = $original
        & $script:StartupTool -Apply -BackupDir $global:PcaiStartupSafetyFixture.BackupRoot -Confirm:$false

        $global:PcaiStartupSafetyFixture.Writes.Count | Should -Be 1
        $global:PcaiStartupSafetyFixture.Writes[0].Path | Should -BeExactly $script:RunKey
        $global:PcaiStartupSafetyFixture.Writes[0].Name | Should -BeExactly 'Ollama'
        $global:PcaiStartupSafetyFixture.Writes[0].Type | Should -BeExactly 'Binary'
        ($global:PcaiStartupSafetyFixture.Writes[0].Value -join ',') | Should -BeExactly '3,9,8,7,6,5,4,3,2,1,255,11'
        ($original -join ',') | Should -BeExactly '2,9,8,7,6,5,4,3,2,1,255,11'
        $backups = @(Get-ChildItem -LiteralPath $global:PcaiStartupSafetyFixture.BackupRoot -File)
        $backups.Count | Should -Be 1
        $backup = [IO.File]::ReadAllText($backups[0].FullName) | ConvertFrom-Json
        ($backup.HkcuRun.Ollama -join ',') | Should -BeExactly '2,9,8,7,6,5,4,3,2,1,255,11'
    }

    It 'preserves unknown or unsupported values: <Label>' -ForEach @(
        @{ Label = 'modern 01'; Bytes = [byte[]]@(1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) }
        @{ Label = 'unknown 00'; Bytes = [byte[]]@(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) }
        @{ Label = 'unknown 04'; Bytes = [byte[]]@(4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) }
        @{ Label = 'unknown FF'; Bytes = [byte[]]@(255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) }
        @{ Label = 'empty'; Bytes = [byte[]]@() }
        @{ Label = 'one byte'; Bytes = [byte[]]@(2) }
        @{ Label = 'short disabled-looking'; Bytes = [byte[]]@(3) }
        @{ Label = 'eight bytes'; Bytes = [byte[]]@(2, 0, 0, 0, 0, 0, 0, 0) }
        @{ Label = 'eleven bytes'; Bytes = [byte[]]@(2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) }
        @{ Label = 'thirteen bytes'; Bytes = [byte[]]@(2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) }
    ) {
        $global:PcaiStartupSafetyFixture.RunValues['Ollama'] = $Bytes.Clone()
        & $script:StartupTool -Apply -BackupDir $global:PcaiStartupSafetyFixture.BackupRoot -Confirm:$false
        Should -Invoke Set-ItemProperty -Times 0 -Exactly
        ($global:PcaiStartupSafetyFixture.RunValues['Ollama'] -join ',') | Should -BeExactly ($Bytes -join ',')
        Should -Invoke Write-Host -Times 1 -Exactly -ParameterFilter { $Object -like '*[[]SKIP-UNSUPPORTED[]]*' }
        Test-Path -LiteralPath $global:PcaiStartupSafetyFixture.BackupRoot | Should -BeFalse
    }

    It 'creates no directories, backup files or registry writes with Apply WhatIf' {
        Mock New-Item {}
        Mock Set-Content {}
        $global:PcaiStartupSafetyFixture.RunValues['Ollama'] = [byte[]]@(2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        $before = $global:PcaiStartupSafetyFixture.RunValues['Ollama'].Clone()
        & $script:StartupTool -Apply -WhatIf -BackupDir $global:PcaiStartupSafetyFixture.BackupRoot
        Should -Invoke Set-ItemProperty -Times 0 -Exactly
        Should -Invoke New-Item -Times 0 -Exactly
        Should -Invoke Set-Content -Times 0 -Exactly
        Test-Path -LiteralPath $global:PcaiStartupSafetyFixture.BackupRoot | Should -BeFalse
        ($global:PcaiStartupSafetyFixture.RunValues['Ollama'] -join ',') | Should -BeExactly ($before -join ',')
    }

    It 'keeps already disabled values and essential, first GoogleDriveFS and HKLM entries unchanged' {
        $enabled = [byte[]]@(2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        $global:PcaiStartupSafetyFixture.RunValues = @{ Ollama = [byte[]]@(3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0); OneDrive = $enabled.Clone(); GoogleDriveFS = $enabled.Clone() }
        $global:PcaiStartupSafetyFixture.MachineValues = @{ 'Docker Desktop' = $enabled.Clone() }
        $global:PcaiStartupSafetyFixture.Commands = @([pscustomobject]@{ Name = 'Docker Desktop'; Location = 'HKLM Run'; Command = 'docker.exe' })
        & $script:StartupTool -Apply -BackupDir $global:PcaiStartupSafetyFixture.BackupRoot -Confirm:$false
        Should -Invoke Set-ItemProperty -Times 0 -Exactly
        Test-Path -LiteralPath $global:PcaiStartupSafetyFixture.BackupRoot | Should -BeFalse
    }

    It 'backs up all original HKCU arrays before a StartupFolder write' {
        $global:PcaiStartupSafetyFixture.RunValues['Ollama'] = [byte[]]@(1, 9, 8, 7, 6, 5, 4, 3, 2, 1, 255, 11)
        $global:PcaiStartupSafetyFixture.FolderValues['MATLAB Connector'] = [byte[]]@(2, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
        & $script:StartupTool -Apply -BackupDir $global:PcaiStartupSafetyFixture.BackupRoot -Confirm:$false
        $global:PcaiStartupSafetyFixture.Writes.Count | Should -Be 1
        $global:PcaiStartupSafetyFixture.Writes[0].Path | Should -BeExactly $script:FolderKey
        $global:PcaiStartupSafetyFixture.Writes[0].Name | Should -BeExactly 'MATLAB Connector'
        ($global:PcaiStartupSafetyFixture.Writes[0].Value -join ',') | Should -BeExactly '3,1,2,3,4,5,6,7,8,9,10,11'
        $backup = Get-Content -LiteralPath (Get-ChildItem -LiteralPath $global:PcaiStartupSafetyFixture.BackupRoot -File).FullName -Raw | ConvertFrom-Json
        ($backup.HkcuRun.Ollama -join ',') | Should -BeExactly '1,9,8,7,6,5,4,3,2,1,255,11'
        ($backup.HkcuFolder.'MATLAB Connector' -join ',') | Should -BeExactly '2,1,2,3,4,5,6,7,8,9,10,11'
    }

    It 'creates one complete original snapshot for multiple approved writes' {
        $global:PcaiStartupSafetyFixture.RunValues['Ollama'] = [byte[]]@(2, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
        $global:PcaiStartupSafetyFixture.FolderValues['MATLAB Connector'] = [byte[]]@(2, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1)
        & $script:StartupTool -Apply -BackupDir $global:PcaiStartupSafetyFixture.BackupRoot -Confirm:$false
        $global:PcaiStartupSafetyFixture.Writes.Count | Should -Be 2
        $files = @(Get-ChildItem -LiteralPath $global:PcaiStartupSafetyFixture.BackupRoot -File)
        $files.Count | Should -Be 1
        $backup = [IO.File]::ReadAllText($files[0].FullName) | ConvertFrom-Json
        ($backup.HkcuRun.Ollama -join ',') | Should -BeExactly '2,1,2,3,4,5,6,7,8,9,10,11'
        ($backup.HkcuFolder.'MATLAB Connector' -join ',') | Should -BeExactly '2,11,10,9,8,7,6,5,4,3,2,1'
        ($global:PcaiStartupSafetyFixture.RunValues.Ollama -join ',') | Should -BeExactly '2,1,2,3,4,5,6,7,8,9,10,11'
        ($global:PcaiStartupSafetyFixture.FolderValues.'MATLAB Connector' -join ',') | Should -BeExactly '2,11,10,9,8,7,6,5,4,3,2,1'
    }

    It 'fails before any registry write when backup creation fails' {
        $global:PcaiStartupSafetyFixture.RunValues['Ollama'] = [byte[]]@(2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        Mock Set-Content { throw 'Backup write failed' }
        { & $script:StartupTool -Apply -BackupDir $global:PcaiStartupSafetyFixture.BackupRoot -Confirm:$false } | Should -Throw '*Backup write failed*'
        Should -Invoke Set-ItemProperty -Times 0 -Exactly
    }

    It 'uses an explicit backup file without creating it in WhatIf' {
        $global:PcaiStartupSafetyFixture.RunValues['Ollama'] = [byte[]]@(2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        $backupFile = Join-Path $TestDrive 'explicit-backup.json'
        & $script:StartupTool -Apply -BackupDir $global:PcaiStartupSafetyFixture.BackupRoot -BackupFile $backupFile -WhatIf
        Test-Path -LiteralPath $backupFile | Should -BeFalse
        Test-Path -LiteralPath $global:PcaiStartupSafetyFixture.BackupRoot | Should -BeFalse
        Should -Invoke Set-ItemProperty -Times 0 -Exactly
        & $script:StartupTool -Apply -BackupDir $global:PcaiStartupSafetyFixture.BackupRoot -BackupFile $backupFile -Confirm:$false
        $backup = Get-Content -LiteralPath $backupFile -Raw | ConvertFrom-Json
        ($backup.HkcuRun.Ollama -join ',') | Should -BeExactly '2,0,0,0,0,0,0,0,0,0,0,0'
        $global:PcaiStartupSafetyFixture.Writes.Count | Should -Be 1
    }

    It 'restores exact saved bytes and keeps Revert WhatIf non-mutating' {
        $backupFile = Join-Path $TestDrive 'restore.json'
        @{ HkcuRun = @{ Ollama = @(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12) }; HkcuFolder = @{} } | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $backupFile
        & $script:StartupTool -Revert -BackupFile $backupFile -WhatIf
        Should -Invoke Set-ItemProperty -Times 0 -Exactly
        & $script:StartupTool -Revert -BackupFile $backupFile -Confirm:$false
        $global:PcaiStartupSafetyFixture.Writes.Count | Should -Be 1
        ($global:PcaiStartupSafetyFixture.Writes[0].Value -join ',') | Should -BeExactly '1,2,3,4,5,6,7,8,9,10,11,12'
        $global:PcaiStartupSafetyFixture.Writes[0].Path | Should -BeExactly $script:RunKey
        $global:PcaiStartupSafetyFixture.Writes[0].Type | Should -BeExactly 'Binary'
    }
}
