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
    It 'keeps an unavailable expected image association pending' {
        Mock Get-DiskImage { $null }
        (Get-CacheVolumeState @script:StateArgs).State | Should -Be 'Pending'
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
        @{ Number = -1 }; @{ Number = 'abc' }; @{ Number = '4.5' }; @{ Number = '4294967296' }
    ) {
        $script:Disk.Number = $Number
        (Get-CacheVolumeState @script:StateArgs).State | Should -Be 'InvalidIdentity'
        Should -Invoke Get-Partition -Times 0
    }
    It 'classifies incomplete and ambiguous associated disks <Count>' -ForEach @(@{ Count = 0; ExpectedState = 'Pending' }; @{ Count = 2; ExpectedState = 'InvalidIdentity' }) {
        $script:DiskCount = $Count
        Mock Get-Disk { if ($script:DiskCount -eq 2) { @($script:Disk, $script:Disk) } }
        (Get-CacheVolumeState @script:StateArgs).State | Should -Be $ExpectedState
    }
    It 'rejects wrong partition disk association' {
        $script:PartitionFixture.DiskNumber = [uint32]9
        (Get-CacheVolumeState @script:StateArgs).State | Should -Be 'InvalidIdentity'
    }
    It 'classifies incomplete and malformed partition disk numbers <Number>' -ForEach @(@{ Number = $null; ExpectedState = 'Pending' }; @{ Number = 'bad'; ExpectedState = 'InvalidIdentity' }) {
        $script:PartitionFixture = [pscustomobject]@{ DriveLetter = 'F'; DiskNumber = $Number }
        (Get-CacheVolumeState @script:StateArgs).State | Should -Be $ExpectedState
    }
    It 'rejects ambiguous matching partitions' {
        Mock Get-Partition { @($script:PartitionFixture, $script:PartitionFixture) }
        (Get-CacheVolumeState @script:StateArgs).State | Should -Be 'InvalidIdentity'
    }
    It 'keeps an absent associated volume pending' {
        $script:Associated = $null
        (Get-CacheVolumeState @script:StateArgs).State | Should -Be 'Pending'
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
        $state.State | Should -Be 'Pending'
        $state.Reason | Should -Match 'Access denied'
    }
    It 'never accepts partially emitted healthy query output followed by an external error' {
        Mock Get-Volume { $script:Volume; throw 'Storage enumeration failed after output' }
        $state = Get-CacheVolumeState @script:StateArgs
        $state.State | Should -Be 'Pending'
        $state.Identity | Should -BeNullOrEmpty
        $state.Reason | Should -Match 'Storage enumeration failed after output'
    }
    It 'does not hide definitive <Conflict> behind an incomplete labeled assignment' -ForEach @(
        @{ Conflict = 'WrongImage' }; @{ Conflict = 'DetachedImage' }; @{ Conflict = 'OtherDisk' }
    ) {
        $script:Volume.DriveLetter = [char]0
        switch ($Conflict) {
            'WrongImage' { $script:Image.ImagePath = 'T:\vm\other.vhdx' }
            'DetachedImage' { $script:Image.Attached = $false }
            'OtherDisk' { $script:PartitionFixture.DiskNumber = [uint32]9 }
        }
        Mock Start-Sleep {}
        Invoke-CloudCacheRepair @script:RepairArgs | Should -Be 3
        Should -Invoke Start-Sleep -Times 0
        Should -Invoke Start-ScheduledTask -Times 0
    }
    It 'keeps a primary <LabelKind> label nonadmitting until populated despite complete associated metadata' -ForEach @(
        @{ LabelKind = 'Empty'; Value = '' }; @{ LabelKind = 'Null'; Value = $null }
    ) {
        $script:Associated = $script:Volume.PSObject.Copy()
        $script:Volume.FileSystemLabel = $Value
        $state = Get-CacheVolumeState @script:StateArgs
        $state.State | Should -Be 'Pending'
        $state.Identity | Should -BeNullOrEmpty
        $state.Reason | Should -Match 'Primary volume label'
        Mock Start-Sleep { $script:Volume.FileSystemLabel = 'cloud-cache-disk' }
        Invoke-CloudCacheRepair @script:RepairArgs | Should -Be 0
        Should -Invoke Start-Sleep -Times 1 -Exactly
        Should -Invoke Start-ScheduledTask -Times 0
    }
    It 'fails fast on <Conflict> with a primary <LabelKind> label rather than hiding a conflict as pending' -ForEach @(
        @{ LabelKind = 'Empty'; Value = ''; Conflict = 'WrongImage' }
        @{ LabelKind = 'Empty'; Value = ''; Conflict = 'OtherDisk' }
        @{ LabelKind = 'Empty'; Value = ''; Conflict = 'WrongGuid' }
        @{ LabelKind = 'Empty'; Value = ''; Conflict = 'WrongFileSystem' }
        @{ LabelKind = 'Null'; Value = $null; Conflict = 'WrongImage' }
        @{ LabelKind = 'Null'; Value = $null; Conflict = 'OtherDisk' }
        @{ LabelKind = 'Null'; Value = $null; Conflict = 'WrongGuid' }
        @{ LabelKind = 'Null'; Value = $null; Conflict = 'WrongFileSystem' }
    ) {
        $script:Volume.FileSystemLabel = $Value
        switch ($Conflict) {
            'WrongImage' { $script:Image.ImagePath = 'T:\vm\other.vhdx' }
            'OtherDisk' { $script:PartitionFixture.DiskNumber = [uint32]9 }
            'WrongGuid' { $script:Associated = $script:Volume.PSObject.Copy(); $script:Associated.Path = '\\?\Volume{23e0daf0-3678-4842-8d9c-66aa3619b11a}\' }
            'WrongFileSystem' { $script:Volume.FileSystem = 'exFAT' }
        }
        Mock Start-Sleep {}
        (Get-CacheVolumeState @script:StateArgs).State | Should -Be 'InvalidIdentity'
        Invoke-CloudCacheRepair @script:RepairArgs | Should -Be 3
        Should -Invoke Start-Sleep -Times 0
        Should -Invoke Start-ScheduledTask -Times 0
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
    It 'waits for eventual <Kind> association before validating the fresh receipt' -ForEach @(
        @{ Kind = 'VolumeLetter' }; @{ Kind = 'VolumeFileSystem' }; @{ Kind = 'Image' }
        @{ Kind = 'VolumeLabelEmpty' }; @{ Kind = 'VolumeLabelNull' }
        @{ Kind = 'Disk' }; @{ Kind = 'DiskNumber' }; @{ Kind = 'Partition' }
        @{ Kind = 'PartitionLetter' }; @{ Kind = 'PartitionDiskNumber' }; @{ Kind = 'AssociatedVolume' }
        @{ Kind = 'AssociatedLetter' }; @{ Kind = 'AssociatedFileSystem' }; @{ Kind = 'AssociatedLabel' }
    ) {
        $script:Mounted = $false
        $script:PendingKind = $Kind
        $script:HealthyImage = $script:Image
        $script:HealthyPartition = $script:PartitionFixture
        Mock Start-ScheduledTask {
            if ($TaskName -eq 'AutoMount_VHDX_cloud-cache-disk') {
                $script:Mounted = $true
                Write-RepairFixture
                switch ($script:PendingKind) {
                    'VolumeLetter' { $script:Volume.DriveLetter = [char]0 }
                    'VolumeFileSystem' { $script:Volume.FileSystem = '' }
                    'VolumeLabelEmpty' { $script:Volume.FileSystemLabel = '' }
                    'VolumeLabelNull' { $script:Volume.FileSystemLabel = $null }
                    'Image' { $script:Image = $null }
                    'Disk' { $script:Disk = $null }
                    'DiskNumber' { $script:Disk.Number = $null }
                    'Partition' { $script:PartitionFixture = $null }
                    'PartitionLetter' { $script:PartitionFixture.DriveLetter = [char]0 }
                    'PartitionDiskNumber' { $script:PartitionFixture = [pscustomobject]@{ DriveLetter = 'F'; DiskNumber = $null } }
                    'AssociatedVolume' { $script:Associated = $null }
                    'AssociatedLetter' { $script:Associated = $script:Volume.PSObject.Copy(); $script:Associated.DriveLetter = '' }
                    'AssociatedFileSystem' { $script:Associated = $script:Volume.PSObject.Copy(); $script:Associated.FileSystem = '' }
                    'AssociatedLabel' { $script:Associated = $script:Volume.PSObject.Copy(); $script:Associated.FileSystemLabel = '' }
                }
            }
        }
        Mock Start-Sleep {
            $script:Volume.DriveLetter = 'F'; $script:Volume.FileSystem = 'NTFS'; $script:Volume.FileSystemLabel = 'cloud-cache-disk'
            $script:Image = $script:HealthyImage
            $script:Disk = [pscustomobject]@{ Number = 4 }
            $script:HealthyPartition.DriveLetter = [char]'F'
            $script:PartitionFixture = $script:HealthyPartition
            $script:Associated = $script:Volume
        }
        Invoke-CloudCacheRepair @script:RepairArgs | Should -Be 0
        Should -Invoke Start-Sleep -Times 1 -Exactly
        Should -Invoke Start-ScheduledTask -Times 1 -Exactly -ParameterFilter { $TaskName -eq 'AutoMount_VHDX_cloud-cache-disk' }
        Should -Invoke Start-ScheduledTask -Times 1 -Exactly -ParameterFilter { $TaskName -eq 'CloudClients-AfterVHDX' }
    }
    It 'recovers a transient <Query> query under the repair deadline without accepting the error as healthy' -ForEach @(
        @{ Query = 'Volume' }; @{ Query = 'Image' }; @{ Query = 'Disk' }; @{ Query = 'Partition' }; @{ Query = 'Association' }
    ) {
        $script:Mounted = $false
        $script:TransientQuery = $false
        switch ($Query) {
            'Volume' { Mock Get-Volume { if ($script:Mounted) { if ($script:TransientQuery) { throw 'Transient volume query' }; $script:Volume } } }
            'Image' { Mock Get-DiskImage { if ($script:TransientQuery) { throw 'Transient image query' }; $script:Image } }
            'Disk' { Mock Get-Disk { if ($script:TransientQuery) { throw 'Transient disk query' }; $script:Disk } }
            'Partition' { Mock Get-Partition { if ($script:TransientQuery) { throw 'Transient partition query' }; $script:PartitionFixture } }
            'Association' { Mock Get-Volume { if ($script:TransientQuery) { throw 'Transient association query' }; $script:Associated } -ParameterFilter { $null -ne $Partition } }
        }
        Mock Start-ScheduledTask {
            if ($TaskName -eq 'AutoMount_VHDX_cloud-cache-disk') { $script:Mounted = $true; $script:TransientQuery = $true; Write-RepairFixture }
        }
        Mock Start-Sleep { $script:TransientQuery = $false }
        Invoke-CloudCacheRepair @script:RepairArgs | Should -Be 0
        Should -Invoke Start-Sleep -Times 1 -Exactly
        Should -Invoke Start-ScheduledTask -Times 1 -Exactly -ParameterFilter { $TaskName -eq 'AutoMount_VHDX_cloud-cache-disk' }
        Should -Invoke Start-ScheduledTask -Times 1 -Exactly -ParameterFilter { $TaskName -eq 'CloudClients-AfterVHDX' }
    }
    It 'waits for both complete association and complete post-request receipt before launch' {
        $script:Mounted = $false
        $script:SleepCount = 0
        $script:LaunchCount = 0
        Mock Start-ScheduledTask {
            if ($TaskName -eq 'AutoMount_VHDX_cloud-cache-disk') {
                $script:Mounted = $true
                $script:Volume.DriveLetter = [char]0
                [IO.File]::WriteAllText((Join-Path $script:ReceiptRoot 'current.result.json'), '{"StartedAt":')
            } elseif ($TaskName -eq 'CloudClients-AfterVHDX') { $script:LaunchCount++ }
        }
        Mock Start-Sleep {
            $script:LaunchCount | Should -Be 0
            $script:SleepCount++
            if ($script:SleepCount -eq 1) { $script:Volume.DriveLetter = 'F' }
            if ($script:SleepCount -eq 2) { Write-RepairFixture }
        }
        Invoke-CloudCacheRepair @script:RepairArgs | Should -Be 0
        $script:SleepCount | Should -Be 2
        $script:LaunchCount | Should -Be 1
    }
    It 'passively waits for an initial <Kind> state and leaves intentionally quit clients alone' -ForEach @(
        @{ Kind = 'UnassignedLetter' }; @{ Kind = 'TransientQuery' }
    ) {
        if ($Kind -eq 'UnassignedLetter') { $script:Volume.DriveLetter = [char]0 } else {
            $script:TransientQuery = $true
            Mock Get-Disk { if ($script:TransientQuery) { throw 'Transient disk query' }; $script:Disk }
        }
        Mock Start-Sleep { $script:Volume.DriveLetter = 'F'; $script:TransientQuery = $false }
        Invoke-CloudCacheRepair @script:RepairArgs | Should -Be 0
        Should -Invoke Start-Sleep -Times 1 -Exactly
        Should -Invoke Start-ScheduledTask -Times 0
    }
    It 'keeps pending initial preview <Mode> free of changes and passive sleeps' -ForEach @(
        @{ Mode = 'DryRun' }; @{ Mode = 'LongDryRun' }; @{ Mode = 'WhatIf' }
    ) {
        $script:Volume.DriveLetter = [char]0
        Mock Start-Sleep {}
        switch ($Mode) {
            'DryRun' { Invoke-CloudCacheRepair @script:RepairArgs -DryRun | Should -Be 0 }
            'LongDryRun' { Invoke-CloudCacheRepair @script:RepairArgs -CliArgs '--DryRun' | Should -Be 0 }
            'WhatIf' { Invoke-CloudCacheRepair @script:RepairArgs -WhatIf | Should -Be 0 }
        }
        Should -Invoke Start-Sleep -Times 0
        Should -Invoke New-Item -Times 0
        Should -Invoke Add-Content -Times 0
        Should -Invoke Start-ScheduledTask -Times 0
        Should -Invoke Mount-DiskImage -Times 0
        Should -Invoke Start-Process -Times 0
    }
    It 'times out persistent <Kind> pending evidence without any client launch' -ForEach @(
        @{ Kind = 'InitialAssociation' }; @{ Kind = 'InitialQueryError' }
        @{ Kind = 'RequestedAssociation' }; @{ Kind = 'RequestedQueryError' }
    ) {
        switch ($Kind) {
            'InitialAssociation' { $script:Volume.DriveLetter = [char]0 }
            'InitialQueryError' { Mock Get-Disk { throw 'Persistent Storage query error' } }
            'RequestedAssociation' {
                $script:Mounted = $false
                Mock Start-ScheduledTask { $script:Mounted = $true; $script:PartitionFixture = $null; Write-RepairFixture }
            }
            'RequestedQueryError' {
                $script:Mounted = $false
                Mock Get-Disk { throw 'Persistent Storage query error' }
            }
        }
        $timer = [Diagnostics.Stopwatch]::StartNew()
        Invoke-CloudCacheRepair @script:RepairArgs | Should -Be 1
        $timer.Elapsed.TotalSeconds | Should -BeGreaterThan 0.8
        $timer.Elapsed.TotalSeconds | Should -BeLessThan 3
        Should -Invoke Start-ScheduledTask -Times 0 -ParameterFilter { $TaskName -eq 'CloudClients-AfterVHDX' }
        if ($Kind -like 'Initial*') { Should -Invoke Start-ScheduledTask -Times 0 }
        else { Should -Invoke Start-ScheduledTask -Times 1 -Exactly -ParameterFilter { $TaskName -eq 'AutoMount_VHDX_cloud-cache-disk' } }
    }
    It 'retains the original deadline when initial pending becomes missing and requests a mount' {
        $script:Volume.DriveLetter = [char]0
        Mock Start-Sleep { [Threading.Thread]::Sleep(700); $script:Mounted = $false }
        Mock Start-ScheduledTask {
            if ($TaskName -eq 'AutoMount_VHDX_cloud-cache-disk') {
                [Threading.Thread]::Sleep(400)
                $script:Mounted = $true; $script:Volume.DriveLetter = 'F'; Write-RepairFixture
            }
        }
        $timer = [Diagnostics.Stopwatch]::StartNew()
        Invoke-CloudCacheRepair @script:RepairArgs | Should -Be 1
        $timer.Elapsed.TotalSeconds | Should -BeGreaterThan 1
        $timer.Elapsed.TotalSeconds | Should -BeLessThan 3
        Should -Invoke Start-Sleep -Times 1 -Exactly
        Should -Invoke Start-ScheduledTask -Times 1 -Exactly -ParameterFilter { $TaskName -eq 'AutoMount_VHDX_cloud-cache-disk' }
        Should -Invoke Start-ScheduledTask -Times 0 -ParameterFilter { $TaskName -eq 'CloudClients-AfterVHDX' }
    }
    It 'fails fast on requested definitive <Kind> conflict despite a successful receipt' -ForEach @(
        @{ Kind = 'WrongDrive' }; @{ Kind = 'WrongFileSystem' }; @{ Kind = 'WrongGuid' }
        @{ Kind = 'WrongImage' }; @{ Kind = 'OtherDisk' }; @{ Kind = 'MalformedDisk' }
    ) {
        $script:Mounted = $false
        $script:ConflictKind = $Kind
        Mock Start-ScheduledTask {
            if ($TaskName -eq 'AutoMount_VHDX_cloud-cache-disk') {
                $script:Mounted = $true; Write-RepairFixture
                switch ($script:ConflictKind) {
                    'WrongDrive' { $script:Volume.DriveLetter = 'G' }
                    'WrongFileSystem' { $script:Volume.FileSystem = 'exFAT' }
                    'WrongGuid' { $script:Associated = $script:Volume.PSObject.Copy(); $script:Associated.Path = '\\?\Volume{23e0daf0-3678-4842-8d9c-66aa3619b11a}\' }
                    'WrongImage' { $script:Image.ImagePath = 'T:\vm\other.vhdx' }
                    'OtherDisk' { $script:PartitionFixture.DiskNumber = [uint32]9 }
                    'MalformedDisk' { $script:Disk.Number = 'bad' }
                }
            }
        }
        Mock Start-Sleep {}
        Invoke-CloudCacheRepair @script:RepairArgs | Should -Be 3
        Should -Invoke Start-Sleep -Times 0
        Should -Invoke Start-ScheduledTask -Times 0 -ParameterFilter { $TaskName -eq 'CloudClients-AfterVHDX' }
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
