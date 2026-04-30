#Requires -Modules Pester

BeforeAll {
    $script:RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
    $script:MountScript = Join-Path $script:RepoRoot 'Tools\Mount-PersistentVHDX.ps1'
    $script:RegisterScript = Join-Path $script:RepoRoot 'Tools\Register-PersistentVHDXTasks.ps1'
    . $script:MountScript
    . $script:RegisterScript
}

Describe 'Mount-PersistentVHDX wrapper' -Tag 'Unit', 'Boot', 'VHD' {
    BeforeEach {
        $script:TempRoot = Join-Path ([System.IO.Path]::GetTempPath()) ('pcai-vhd-test-' + [guid]::NewGuid().ToString('N'))
        New-Item -Path $script:TempRoot -ItemType Directory -Force | Out-Null
        $script:VhdPath = Join-Path $script:TempRoot 'disk.vhdx'
        New-Item -Path $script:VhdPath -ItemType File -Force | Out-Null

        Mock Register-PersistentVHDXEventSource {}
        Mock Write-PersistentVHDXEvent {}
        Mock Start-Transcript {}
        Mock Stop-Transcript {}
        Mock Get-WinEvent { @() }
        Mock fltmc { "\\?\Volume{abc}\ F:" }
        Mock Get-Disk { [pscustomobject]@{ Number = 7; UniqueId = 'disk-7'; FriendlyName = 'Microsoft Virtual Disk' } }
        Mock Get-Partition { [pscustomobject]@{ DiskNumber = 7; PartitionNumber = 1; DriveLetter = 'F' } }
        Mock Get-Volume { [pscustomobject]@{ DriveLetter = 'F'; FileSystemLabel = 'cloud-cache-disk'; FileSystem = 'NTFS'; Path = '\\?\Volume{abc}\' } }
        Mock Mount-VHD {}
    }

    It 'mounts and validates an initially detached VHDX' {
        $script:GetVhdCalls = 0
        Mock Get-VHD {
            $script:GetVhdCalls++
            [pscustomobject]@{ Path = $Path; Attached = ($script:GetVhdCalls -gt 1); DiskNumber = 7; DiskIdentifier = 'disk-7' }
        }

        $result = Invoke-PersistentVHDXMount `
            -VhdPath $script:VhdPath `
            -ExpectedVolumeLabel 'cloud-cache-disk' `
            -ExpectedDriveLetter F `
            -ExpectedFileSystem NTFS `
            -TaskName 'AutoMount_VHDX_cloud-cache-disk' `
            -LogRoot $script:TempRoot `
            -SkipEventSourceRegistration

        $result.Status | Should -Be 'Success'
        $result.ExitCode | Should -Be 0
        $result.MountAttempted | Should -BeTrue
        Should -Invoke Mount-VHD -Times 1
        Test-Path -LiteralPath $result.Logs.Json | Should -BeTrue
    }

    It 'validates an already attached VHDX without remounting it' {
        Mock Get-VHD { [pscustomobject]@{ Path = $Path; Attached = $true; DiskNumber = 7; DiskIdentifier = 'disk-7' } }

        $result = Invoke-PersistentVHDXMount `
            -VhdPath $script:VhdPath `
            -ExpectedVolumeLabel 'cloud-cache-disk' `
            -ExpectedDriveLetter F `
            -ExpectedFileSystem NTFS `
            -TaskName 'AutoMount_VHDX_cloud-cache-disk' `
            -LogRoot $script:TempRoot `
            -SkipEventSourceRegistration

        $result.Status | Should -Be 'Success'
        $result.AlreadyAttached | Should -BeTrue
        Should -Invoke Mount-VHD -Times 0
    }

    It 'fails loudly when the VHDX file is missing' {
        Mock Get-VHD { throw 'Get-VHD should not be called for a missing file' }

        $result = Invoke-PersistentVHDXMount `
            -VhdPath (Join-Path $script:TempRoot 'missing.vhdx') `
            -TaskName 'AutoMount_VHDX_missing' `
            -LogRoot $script:TempRoot `
            -SkipEventSourceRegistration

        $result.Status | Should -Be 'Failed'
        $result.ExitCode | Should -Be 51
        $result.Errors[0] | Should -Match 'Missing VHDX file'
        Should -Invoke Get-VHD -Times 0
    }

    It 'fails when Hyper-V VHD commands are unavailable' {
        Mock Test-PersistentVHDXHyperVCommands { [pscustomobject]@{ GetVHD = $false; MountVHD = $false; Available = $false } }

        $result = Invoke-PersistentVHDXMount `
            -VhdPath $script:VhdPath `
            -TaskName 'AutoMount_VHDX_cloud-cache-disk' `
            -LogRoot $script:TempRoot `
            -SkipEventSourceRegistration

        $result.Status | Should -Be 'Failed'
        $result.ExitCode | Should -Be 52
        $result.Errors[0] | Should -Match 'Hyper-V PowerShell commands'
    }

    It 'fails when the expected drive letter is not present' {
        Mock Get-VHD { [pscustomobject]@{ Path = $Path; Attached = $true; DiskNumber = 7; DiskIdentifier = 'disk-7' } }

        $result = Invoke-PersistentVHDXMount `
            -VhdPath $script:VhdPath `
            -ExpectedVolumeLabel 'cloud-cache-disk' `
            -ExpectedDriveLetter W `
            -ExpectedFileSystem NTFS `
            -TaskName 'AutoMount_VHDX_cloud-cache-disk' `
            -LogRoot $script:TempRoot `
            -SkipEventSourceRegistration

        $result.Status | Should -Be 'Failed'
        $result.ExitCode | Should -Be 54
        ($result.Errors -join ';') | Should -Match 'Expected drive letter W'
    }

    It 'fails on mount timeout' {
        Mock Get-VHD { [pscustomobject]@{ Path = $Path; Attached = $false; DiskNumber = 7; DiskIdentifier = 'disk-7' } }

        $result = Invoke-PersistentVHDXMount `
            -VhdPath $script:VhdPath `
            -TaskName 'AutoMount_VHDX_cloud-cache-disk' `
            -MountTimeoutSeconds 0 `
            -LogRoot $script:TempRoot `
            -SkipEventSourceRegistration

        $result.Status | Should -Be 'Failed'
        $result.ExitCode | Should -Be 53
        ($result.Errors -join ';') | Should -Match 'Timed out'
    }

    It 'returns degraded when FilterManager Event ID 3 is present after mount' {
        Mock Get-VHD { [pscustomobject]@{ Path = $Path; Attached = $true; DiskNumber = 7; DiskIdentifier = 'disk-7' } }
        Mock Get-WinEvent {
            [pscustomobject]@{
                TimeCreated = Get-Date
                ProviderName = 'Microsoft-Windows-FilterManager'
                Id = 3
                LevelDisplayName = 'Error'
                Message = 'Filter attach failed for \Device\HarddiskVolume9 with status 0xC03A001C.'
            }
        }

        $result = Invoke-PersistentVHDXMount `
            -VhdPath $script:VhdPath `
            -ExpectedVolumeLabel 'cloud-cache-disk' `
            -ExpectedDriveLetter F `
            -ExpectedFileSystem NTFS `
            -TaskName 'AutoMount_VHDX_cloud-cache-disk' `
            -LogRoot $script:TempRoot `
            -SkipEventSourceRegistration

        $result.Status | Should -Be 'Degraded'
        $result.ExitCode | Should -Be 40
        $result.FilterManager.EventId3Count | Should -Be 1
    }

    It 'accepts share-ext4 as attached disk only without a Windows volume' {
        Mock Get-VHD { [pscustomobject]@{ Path = $Path; Attached = $true; DiskNumber = 6; DiskIdentifier = 'disk-6' } }
        Mock Get-Disk { [pscustomobject]@{ Number = 6; UniqueId = 'disk-6'; FriendlyName = 'Microsoft Virtual Disk' } }
        Mock Get-Partition { [pscustomobject]@{ DiskNumber = 6; PartitionNumber = 1; DriveLetter = $null } }
        Mock Get-Volume { throw 'No Windows volume' }

        $result = Invoke-PersistentVHDXMount `
            -VhdPath $script:VhdPath `
            -ExpectedState AttachedDiskOnly `
            -TaskName 'AutoMount_VHDX_share-ext4' `
            -LogRoot $script:TempRoot `
            -SkipEventSourceRegistration

        $result.Status | Should -Be 'Success'
        $result.ExitCode | Should -Be 0
        $result.Volumes.Count | Should -Be 0
    }

    It 'dry-runs a detached VHDX without calling Mount-VHD or writing events' {
        Mock Get-VHD { [pscustomobject]@{ Path = $Path; Attached = $false; DiskNumber = 7; DiskIdentifier = 'disk-7' } }

        $result = Invoke-PersistentVHDXMount `
            -VhdPath $script:VhdPath `
            -TaskName 'AutoMount_VHDX_cloud-cache-disk' `
            -LogRoot $script:TempRoot `
            -SkipEventSourceRegistration `
            -DryRun

        $result.Status | Should -Be 'DryRun'
        $result.ExitCode | Should -Be 0
        $result.MountAttempted | Should -BeFalse
        ($result.DegradedReasons -join ';') | Should -Match 'would be mounted'
        Should -Invoke Mount-VHD -Times 0
        Should -Invoke Write-PersistentVHDXEvent -Times 0
    }
}

Describe 'Register-PersistentVHDXTasks planner' -Tag 'Unit', 'Boot', 'VHD' {
    BeforeEach {
        Mock New-ScheduledTaskTrigger {
            [pscustomobject]@{ Delay = $null }
        }
        Mock New-ScheduledTaskSettingsSet {
            [pscustomobject]@{
                MultipleInstances = $MultipleInstances
                ExecutionTimeLimit = $ExecutionTimeLimit
                RestartCount = $RestartCount
                RestartInterval = $RestartInterval
            }
        }
        Mock New-ScheduledTaskPrincipal {
            [pscustomobject]@{ UserId = $UserId; RunLevel = $RunLevel }
        }
        Mock New-ScheduledTaskAction {
            [pscustomobject]@{ Execute = $Execute; Arguments = $Argument; WorkingDirectory = $WorkingDirectory }
        }
        Mock New-ScheduledTask {
            [pscustomobject]@{ Action = $Action; Trigger = $Trigger; Settings = $Settings; Principal = $Principal; Description = $Description }
        }
    }

    It 'generates wrapper-based startup tasks with staggered delays' {
        $plans = @(New-PersistentVHDXTaskPlan -ScriptPath $script:MountScript -LogRoot 'C:\Logs\VHDMount' -PowerShellExe 'pwsh.exe')

        $plans.TaskName | Should -Contain 'AutoMount_VHDX_cloud-cache-disk'
        $plans.TaskName | Should -Contain 'AutoMount_VHDX_shared-dev'
        $plans.TaskName | Should -Contain 'AutoMount_VHDX_share-ext4'
        ($plans | Where-Object TaskName -eq 'AutoMount_VHDX_cloud-cache-disk').Delay | Should -Be 'PT30S'
        ($plans | Where-Object TaskName -eq 'AutoMount_VHDX_shared-dev').Delay | Should -Be 'PT60S'
        ($plans | Where-Object TaskName -eq 'AutoMount_VHDX_share-ext4').Delay | Should -Be 'PT90S'
        ($plans | Where-Object TaskName -eq 'AutoMount_VHDX_share-ext4').ExpectedState | Should -Be 'AttachedDiskOnly'
    }

    It 'uses the maintained wrapper instead of hidden Mount-VHD one-liners' {
        $plans = @(New-PersistentVHDXTaskPlan -ScriptPath $script:MountScript -LogRoot 'C:\Logs\VHDMount' -PowerShellExe 'pwsh.exe')

        foreach ($plan in $plans) {
            $plan.Execute | Should -Be 'pwsh.exe'
            $plan.Argument | Should -Match ([regex]::Escape('Mount-PersistentVHDX.ps1'))
            $plan.Argument | Should -Match '-VhdPath'
            $plan.Argument | Should -Not -Match 'New-PersistentVHDX'
            $plan.Argument | Should -Not -Match 'Mount-VHD'
        }
    }

    It 'sets retry, execution limit, principal, and descriptions' {
        $plans = @(New-PersistentVHDXTaskPlan -ScriptPath $script:MountScript -LogRoot 'C:\Logs\VHDMount' -PowerShellExe 'pwsh.exe')

        foreach ($plan in $plans) {
            $plan.MultipleInstances | Should -Be 'IgnoreNew'
            $plan.RestartCount | Should -Be 3
            $plan.RestartInterval | Should -Be 'PT1M'
            $plan.ExecutionTimeLimit | Should -Be 'PT10M'
            $plan.Principal.UserId | Should -Be 'SYSTEM'
            $plan.Principal.RunLevel | Should -Be 'Highest'
            $plan.Description | Should -Match 'Mount'
        }
    }

    It 'only registers tasks when -Register is requested and uses Force for idempotency' {
        Mock Register-PersistentVHDXTaskEventSource {}
        Mock New-PersistentVHDXScheduledTaskDefinition { [pscustomobject]@{ Description = $Description } }
        Mock Register-PersistentVHDXScheduledTask {}
        Mock Test-Path { $true }

        Invoke-PersistentVHDXTaskRegistration -ScriptPath $script:MountScript -LogRoot 'C:\Logs\VHDMount' | Out-Null
        Should -Invoke Register-PersistentVHDXScheduledTask -Times 0

        Invoke-PersistentVHDXTaskRegistration -Register -ScriptPath $script:MountScript -LogRoot 'C:\Logs\VHDMount' | Out-Null
        Should -Invoke Register-PersistentVHDXScheduledTask -Times 3
    }

    It 'does not register tasks when DryRun is combined with Register' {
        Mock Register-PersistentVHDXTaskEventSource {}
        Mock New-PersistentVHDXScheduledTaskDefinition { [pscustomobject]@{ Description = $Description } }
        Mock Register-PersistentVHDXScheduledTask {}
        Mock Test-Path { $true }

        $plans = @(Invoke-PersistentVHDXTaskRegistration -Register -DryRun -ScriptPath $script:MountScript -LogRoot 'C:\Logs\VHDMount')

        $plans.Count | Should -Be 3
        Should -Invoke Register-PersistentVHDXScheduledTask -Times 0
    }
}
