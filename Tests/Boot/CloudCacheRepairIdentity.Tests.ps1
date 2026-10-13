#Requires -Modules Pester
BeforeAll {
    $script:RepairPath = Join-Path (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)) 'Tools/Repair-CloudCacheMount.ps1'
    . $script:RepairPath
    function Write-RepairFixture {
        param([string]$Status = 'Success', [int]$ExitCode = 0,
            [datetime]$StartedAt = (Get-Date), [datetime]$CompletedAt = (Get-Date),
            [string]$VhdPath = 'T:\vm\cloud-cache-disk.vhdx',
            [string]$Path = '\\?\Volume{c3e0daf0-3678-4842-8d9c-66aa3619b11a}\',
            [string]$DriveLetter = 'F', [string]$FileSystem = 'NTFS')
        $json = @{ StartedAt = $StartedAt.ToString('o'); CompletedAt = $CompletedAt.ToString('o');
            Status = $Status; ExitCode = $ExitCode; VhdPath = $VhdPath;
            Volumes = @(@{ Path = $Path; DriveLetter = $DriveLetter; FileSystem = $FileSystem; FileSystemLabel = 'cloud-cache-disk' })
        } | ConvertTo-Json -Depth 4
        [System.IO.File]::WriteAllText((Join-Path $script:ReceiptRoot 'current.result.json'), $json)
    }
}
Describe 'Cloud cache repair identity and receipt' -Tag 'Unit', 'Boot', 'VHD' {
    BeforeEach {
        Set-StrictMode -Version Latest
        $script:ReceiptRoot = Join-Path $TestDrive 'receipts'
        New-Item -ItemType Directory -Path $script:ReceiptRoot -Force | Out-Null
        Remove-Item (Join-Path $script:ReceiptRoot '*.json') -ErrorAction SilentlyContinue
        $script:StateArgs = @{ Label = 'cloud-cache-disk'; ExpectedVhd = 'T:\vm\cloud-cache-disk.vhdx'; ExpectedDriveLetter = 'F' }
        $script:RepairArgs = @{ MountWaitSeconds = 1; PollMilliseconds = 100; MountLogRoot = $script:ReceiptRoot; LogRoot = (Join-Path $TestDrive 'logs') }
        $script:Mounted = $true
        $script:Volume = [pscustomobject]@{ DriveLetter = 'F'; FileSystem = 'NTFS'; FileSystemLabel = 'cloud-cache-disk'; Path = '\\?\Volume{c3e0daf0-3678-4842-8d9c-66aa3619b11a}\' }
        $script:Associated = $script:Volume
        $script:Image = New-CimInstance -Namespace root/Microsoft/Windows/Storage -ClassName MSFT_DiskImage -ClientOnly -Property @{
            ImagePath = 'T:\vm\cloud-cache-disk.vhdx'; Attached = $true; Number = [uint32]4
        }
        $script:Disk = [pscustomobject]@{ Number = 4 }
        $script:PartitionFixture = New-CimInstance -Namespace root/Microsoft/Windows/Storage -ClassName MSFT_Partition -ClientOnly -Property @{
            DiskNumber = [uint32]4; PartitionNumber = [uint32]2; DriveLetter = [char]'F'
        }
        Mock Get-Volume { if ($script:Mounted) { $script:Volume } }
        Mock Get-Volume { $script:Associated } -ParameterFilter { $null -ne $Partition }
        Mock Get-DiskImage { $script:Image }
        Mock Get-Disk { $script:Disk }
        Mock Get-Partition { $script:PartitionFixture }
        Mock Test-Path { $true }
        Mock New-Item {}
        Mock Add-Content {}
        Mock Start-ScheduledTask {
            if ($TaskName -eq 'AutoMount_VHDX_cloud-cache-disk') { $script:Mounted = $true; Write-RepairFixture }
        }
        Mock Get-CimInstance { [pscustomobject]@{ UserName = 'user' } }
        Mock Get-Process { $null }
        Mock Start-Process {}
        Mock Mount-DiskImage {}
        Mock Dismount-DiskImage {}
    }
    It 'requires the real image association and canonical volume identity for healthy' {
        $state = Get-CacheVolumeState @script:StateArgs
        $state.State | Should -Be 'Healthy'
        $state.Identity | Should -Be 'c3e0daf0-3678-4842-8d9c-66aa3619b11a'
        Should -Invoke Get-DiskImage -Times 1 -ParameterFilter { $ImagePath -eq 'T:\vm\cloud-cache-disk.vhdx' }
        Should -Invoke Get-Partition -Times 1 -ParameterFilter { $DiskNumber -eq 4 }
        Should -Invoke Get-Volume -Times 1 -ParameterFilter { $null -ne $Partition }
    }
    It 'rejects labeled <Field> mismatch' -ForEach @(
        @{ Field = 'DriveLetter'; Value = 'G' }; @{ Field = 'FileSystem'; Value = 'exFAT' }
        @{ Field = 'Path'; Value = '' }; @{ Field = 'Path'; Value = '\\?\Volume{00000000-0000-0000-0000-000000000000}\' }
        @{ Field = 'Path'; Value = '\\?\Volume{zzzzzzzz-zzzz-zzzz-zzzz-zzzzzzzzzzzz}\' }
    ) {
        $script:Volume.$Field = $Value
        (Get-CacheVolumeState @script:StateArgs).State | Should -Be 'InvalidIdentity'
        Invoke-CloudCacheRepair @script:RepairArgs -DryRun | Should -Be 3
        Should -Invoke Start-ScheduledTask -Times 0
    }
    It 'rejects an ambiguous same-label volume' {
        Mock Get-Volume { @($script:Volume, $script:Volume) }
        (Get-CacheVolumeState @script:StateArgs).State | Should -Be 'InvalidIdentity'
    }
    It 'rejects conflicting canonical Path and UniqueId GUIDs' {
        $script:Volume | Add-Member UniqueId '\\?\Volume{23e0daf0-3678-4842-8d9c-66aa3619b11a}\'
        (Get-CacheVolumeState @script:StateArgs).State | Should -Be 'InvalidIdentity'
    }
    It 'accepts matching canonical UniqueId when Path is unavailable' {
        $script:Volume | Add-Member UniqueId $script:Volume.Path
        $script:Volume.PSObject.Properties.Remove('Path')
        (Get-CacheVolumeState @script:StateArgs).State | Should -Be 'Healthy'
    }
    It 'rejects an ambiguous F letter even with a unique expected label' {
        $script:OtherVolume = $script:Volume.PSObject.Copy()
        $script:OtherVolume.FileSystemLabel = 'foreign'
        Mock Get-Volume { @($script:Volume, $script:OtherVolume) }
        (Get-CacheVolumeState @script:StateArgs).State | Should -Be 'InvalidIdentity'
    }
    It 'does not repair another volume occupying F' {
        $script:Mounted = $false
        $script:Volume.FileSystemLabel = 'other'
        Mock Get-Volume { $script:Volume }
        Invoke-CloudCacheRepair @script:RepairArgs -DryRun | Should -Be 3
        Should -Invoke Start-ScheduledTask -Times 0
    }
    It 'rejects missing expected image' {
        Mock Get-DiskImage { $null }
        (Get-CacheVolumeState @script:StateArgs).State | Should -Be 'InvalidIdentity'
    }
    It 'rejects wrong image path' {
        $script:Image.ImagePath = 'T:\vm\another.vhdx'
        (Get-CacheVolumeState @script:StateArgs).State | Should -Be 'InvalidIdentity'
    }
    It 'rejects detached expected image' {
        $script:Image.Attached = $false
        (Get-CacheVolumeState @script:StateArgs).State | Should -Be 'InvalidIdentity'
        Should -Invoke Get-Disk -Times 0
    }
    It 'rejects malformed disk number <Number>' -ForEach @(
        @{ Number = $null }; @{ Number = -1 }; @{ Number = 'abc' }; @{ Number = '4.5' }; @{ Number = '4294967296' }
    ) {
        $script:Disk.Number = $Number
        (Get-CacheVolumeState @script:StateArgs).State | Should -Be 'InvalidIdentity'
        Should -Invoke Get-Partition -Times 0
    }
    It 'rejects missing or ambiguous associated disk <Count>' -ForEach @(@{ Count = 0 }; @{ Count = 2 }) {
        $script:DiskCount = $Count
        Mock Get-Disk { if ($script:DiskCount -eq 2) { @($script:Disk, $script:Disk) } }
        (Get-CacheVolumeState @script:StateArgs).State | Should -Be 'InvalidIdentity'
    }
    It 'rejects wrong partition disk association' {
        $script:PartitionFixture.DiskNumber = [uint32]9
        (Get-CacheVolumeState @script:StateArgs).State | Should -Be 'InvalidIdentity'
    }
    It 'rejects missing or malformed partition disk number <Number>' -ForEach @(@{ Number = $null }; @{ Number = 'bad' }) {
        $script:PartitionFixture = [pscustomobject]@{ DriveLetter = 'F'; DiskNumber = $Number }
        (Get-CacheVolumeState @script:StateArgs).State | Should -Be 'InvalidIdentity'
    }
    It 'rejects ambiguous matching partitions' {
        Mock Get-Partition { @($script:PartitionFixture, $script:PartitionFixture) }
        (Get-CacheVolumeState @script:StateArgs).State | Should -Be 'InvalidIdentity'
    }
    It 'rejects absent associated volume' {
        $script:Associated = $null
        (Get-CacheVolumeState @script:StateArgs).State | Should -Be 'InvalidIdentity'
    }
    It 'rejects wrong partition drive letter' {
        $script:PartitionFixture.DriveLetter = [char]'G'
        (Get-CacheVolumeState @script:StateArgs).State | Should -Be 'InvalidIdentity'
        Should -Invoke Get-Volume -Times 0 -ParameterFilter { $null -ne $Partition }
    }
    It 'rejects an association race changing <Field>' -ForEach @(
        @{ Field = 'DriveLetter'; Value = 'G' }; @{ Field = 'FileSystem'; Value = 'exFAT' }
        @{ Field = 'FileSystemLabel'; Value = 'other' }
        @{ Field = 'Path'; Value = '\\?\Volume{23e0daf0-3678-4842-8d9c-66aa3619b11a}\' }
    ) {
        $script:Associated = $script:Volume.PSObject.Copy()
        $script:Associated.$Field = $Value
        (Get-CacheVolumeState @script:StateArgs).State | Should -Be 'InvalidIdentity'
    }
    It 'reports external Storage errors instead of accepting identity' {
        Mock Get-Disk { throw 'Access denied' }
        $state = Get-CacheVolumeState @script:StateArgs
        $state.State | Should -Be 'InvalidIdentity'
        $state.Reason | Should -Match 'Access denied'
    }
    It 'leaves a healthy volume and deliberately quit clients alone' {
        Invoke-CloudCacheRepair @script:RepairArgs | Should -Be 0
        Should -Invoke Start-ScheduledTask -Times 0
        Should -Invoke Add-Content -ParameterFilter { $LiteralPath -eq (Join-Path $TestDrive 'logs/current.log') }
    }
    It 'validates a post-request receipt for the same GUID before requesting the launcher' {
        $script:Mounted = $false
        Invoke-CloudCacheRepair @script:RepairArgs | Should -Be 0
        Should -Invoke Start-ScheduledTask -Times 1 -Exactly -ParameterFilter { $TaskName -eq 'AutoMount_VHDX_cloud-cache-disk' }
        Should -Invoke Start-ScheduledTask -Times 1 -Exactly -ParameterFilter { $TaskName -eq 'CloudClients-AfterVHDX' }
    }
    It 'never starts clients in a headless session' {
        $script:Mounted = $false
        Mock Get-CimInstance { [pscustomobject]@{ UserName = $null } }
        Invoke-CloudCacheRepair @script:RepairArgs | Should -Be 0
        Should -Invoke Start-ScheduledTask -Times 0 -ParameterFilter { $TaskName -eq 'CloudClients-AfterVHDX' }
    }
    It 'waits for partial direct-write JSON to become complete' {
        $script:Mounted = $false
        Mock Start-ScheduledTask {
            if ($TaskName -eq 'AutoMount_VHDX_cloud-cache-disk') {
                $script:Mounted = $true
                [System.IO.File]::WriteAllText((Join-Path $script:ReceiptRoot 'current.result.json'), '{"StartedAt":')
            }
        }
        Mock Start-Sleep { Write-RepairFixture }
        Invoke-CloudCacheRepair @script:RepairArgs | Should -Be 0
        Should -Invoke Start-Sleep -Times 1 -Exactly
    }
    It 'waits for a Started receipt without a completion timestamp' {
        $script:Mounted = $false
        Mock Start-ScheduledTask {
            if ($TaskName -eq 'AutoMount_VHDX_cloud-cache-disk') {
                $script:Mounted = $true
                $json = @{ StartedAt = (Get-Date).ToString('o'); CompletedAt = $null; VhdPath = 'T:\vm\cloud-cache-disk.vhdx';
                    ExitCode = $null; Status = 'Started' } | ConvertTo-Json
                [System.IO.File]::WriteAllText((Join-Path $script:ReceiptRoot 'current.result.json'), $json)
            }
        }
        Mock Start-Sleep { Write-RepairFixture }
        Invoke-CloudCacheRepair @script:RepairArgs | Should -Be 0
        Should -Invoke Start-Sleep -Times 1 -Exactly
    }
    It 'refuses a post-request live <Field> race despite a successful receipt' -ForEach @(
        @{ Field = 'DriveLetter'; Value = 'G' }; @{ Field = 'FileSystem'; Value = 'exFAT' }
        @{ Field = 'Path'; Value = '\\?\Volume{00000000-0000-0000-0000-000000000000}\' }
    ) {
        $script:Mounted = $false
        $script:RaceField = $Field
        $script:RaceValue = $Value
        Mock Start-ScheduledTask {
            $script:Mounted = $true
            $script:Volume.($script:RaceField) = $script:RaceValue
            Write-RepairFixture
        }
        Invoke-CloudCacheRepair @script:RepairArgs | Should -Be 3
        Should -Invoke Start-ScheduledTask -Times 1 -Exactly -ParameterFilter { $TaskName -eq 'AutoMount_VHDX_cloud-cache-disk' }
        Should -Invoke Start-ScheduledTask -Times 0 -ParameterFilter { $TaskName -eq 'CloudClients-AfterVHDX' }
    }
    It 'waits for an absent or stale receipt <Kind>' -ForEach @(@{ Kind = 'Absent' }; @{ Kind = 'Stale' }) {
        $script:Mounted = $false
        $script:ReceiptKind = $Kind
        Mock Start-ScheduledTask {
            if ($TaskName -eq 'AutoMount_VHDX_cloud-cache-disk') {
                $script:Mounted = $true
                if ($script:ReceiptKind -eq 'Stale') { Write-RepairFixture -StartedAt (Get-Date).AddMinutes(-5) }
            }
        }
        Mock Start-Sleep { Write-RepairFixture }
        Invoke-CloudCacheRepair @script:RepairArgs | Should -Be 0
        Should -Invoke Start-Sleep -Times 1 -Exactly
    }
    It 'denies fresh receipt <Kind>' -ForEach @(
        @{ Kind = 'Failed' }; @{ Kind = 'Degraded' }; @{ Kind = 'WrongImage' }; @{ Kind = 'WrongGuid' }
        @{ Kind = 'WrongLetter' }; @{ Kind = 'WrongFS' }; @{ Kind = 'Future' }; @{ Kind = 'ExitCode' }
    ) {
        $script:Mounted = $false
        $script:ReceiptKind = $Kind
        Mock Start-ScheduledTask {
            if ($TaskName -eq 'AutoMount_VHDX_cloud-cache-disk') {
                $script:Mounted = $true
                switch ($script:ReceiptKind) {
                    'Failed' { Write-RepairFixture -Status Failed -ExitCode 51 }
                    'Degraded' { Write-RepairFixture -Status Degraded }
                    'WrongImage' { Write-RepairFixture -VhdPath 'T:\vm\other.vhdx' }
                    'WrongGuid' { Write-RepairFixture -Path '\\?\Volume{23e0daf0-3678-4842-8d9c-66aa3619b11a}\' }
                    'WrongLetter' { Write-RepairFixture -DriveLetter G }
                    'WrongFS' { Write-RepairFixture -FileSystem exFAT }
                    'Future' { Write-RepairFixture -StartedAt (Get-Date).AddMinutes(1) }
                    'ExitCode' { Write-RepairFixture -ExitCode 1 }
                }
            }
        }
        Invoke-CloudCacheRepair @script:RepairArgs | Should -Be 3
        Should -Invoke Start-ScheduledTask -Times 1 -Exactly -ParameterFilter { $TaskName -eq 'AutoMount_VHDX_cloud-cache-disk' }
        Should -Invoke Start-ScheduledTask -Times 0 -ParameterFilter { $TaskName -eq 'CloudClients-AfterVHDX' }
    }
    It 'times out on <Kind> within the original deadline without launching' -ForEach @(
        @{ Kind = 'Malformed' }; @{ Kind = 'Stale' }; @{ Kind = 'MissingVolume' }; @{ Kind = 'Absent' }
    ) {
        $script:Mounted = $false
        $script:ReceiptKind = $Kind
        Mock Start-ScheduledTask {
            if ($TaskName -eq 'AutoMount_VHDX_cloud-cache-disk') {
                $script:Mounted = $script:ReceiptKind -ne 'MissingVolume'
                switch ($script:ReceiptKind) {
                    'Malformed' { [System.IO.File]::WriteAllText((Join-Path $script:ReceiptRoot 'current.result.json'), '{bad') }
                    'Stale' { Write-RepairFixture -StartedAt (Get-Date).AddMinutes(-5) }
                    'MissingVolume' { Write-RepairFixture }
                }
            }
        }
        $timer = [System.Diagnostics.Stopwatch]::StartNew()
        Invoke-CloudCacheRepair @script:RepairArgs | Should -Be 1
        $timer.Elapsed.TotalSeconds | Should -BeGreaterThan 0.8
        $timer.Elapsed.TotalSeconds | Should -BeLessThan 3
        Should -Invoke Start-ScheduledTask -Times 0 -ParameterFilter { $TaskName -eq 'CloudClients-AfterVHDX' }
    }
    It 'keeps <Mode> free of filesystem writes and task, mount and process changes' -ForEach @(
        @{ Mode = 'DryRun' }; @{ Mode = 'LongDryRun' }; @{ Mode = 'WhatIf' }
    ) {
        $script:Mounted = $false
        switch ($Mode) {
            'DryRun' { Invoke-CloudCacheRepair @script:RepairArgs -DryRun | Should -Be 0 }
            'LongDryRun' { Invoke-CloudCacheRepair @script:RepairArgs -CliArgs '--DryRun' | Should -Be 0 }
            'WhatIf' { Invoke-CloudCacheRepair @script:RepairArgs -WhatIf | Should -Be 0 }
        }
        Should -Invoke New-Item -Times 0
        Should -Invoke Add-Content -Times 0
        Should -Invoke Start-ScheduledTask -Times 0
        Should -Invoke Mount-DiskImage -Times 0
        Should -Invoke Dismount-DiskImage -Times 0
        Should -Invoke Start-Process -Times 0
    }
    It 'reports missing backing file without requesting a mount' {
        $script:Mounted = $false
        Mock Test-Path { $false }
        Invoke-CloudCacheRepair @script:RepairArgs -DryRun | Should -Be 2
        Should -Invoke Start-ScheduledTask -Times 0
    }
    It 'reports a mount request error' {
        $script:Mounted = $false
        Mock Start-ScheduledTask { throw 'Task unavailable' }
        Invoke-CloudCacheRepair @script:RepairArgs | Should -Be 1
    }
    It 'reports a launcher request error after mount validation' {
        $script:Mounted = $false
        Mock Start-ScheduledTask { throw 'Launcher unavailable' } -ParameterFilter { $TaskName -eq 'CloudClients-AfterVHDX' }
        Invoke-CloudCacheRepair @script:RepairArgs | Should -Be 1
    }
    It 'rejects invalid <Parameter>' -ForEach @(
        @{ Parameter = 'MountWaitSeconds'; Value = 0 }; @{ Parameter = 'MountWaitSeconds'; Value = 3601 }
        @{ Parameter = 'PollMilliseconds'; Value = 0 }; @{ Parameter = 'ExpectedDriveLetter'; Value = 'F:' }
    ) {
        $invalid = @{ $Parameter = $Value }
        { Invoke-CloudCacheRepair @invalid } | Should -Throw
    }
}
