#Requires -Modules Pester

BeforeAll {
    $script:GatePath = Join-Path (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)) 'Tools/Start-CloudClientsAfterVHDX.ps1'
    . $script:GatePath

    function Write-TestMountReceipt {
        param(
            [datetime]$StartedAt = (Get-Date).AddMinutes(-1),
            [int]$ExitCode = 0,
            [string]$Status = 'Success',
            [string]$VhdPath = 'T:\vm\cloud-cache-disk.vhdx'
        )
        $json = @{
            StartedAt = $StartedAt.ToString('o')
            ExitCode = $ExitCode
            Status = $Status
            VhdPath = $VhdPath
            Volumes = @(
                @{
                    DriveLetter = 'F'
                    FileSystemLabel = 'cloud-cache-disk'
                    FileSystem = 'NTFS'
                    Path = '\\?\Volume{c3e0daf0-3678-4842-8d9c-66aa3619b11a}\'
                }
            )
        } | ConvertTo-Json
        [System.IO.File]::WriteAllText((Join-Path $script:ReceiptRoot 'current.result.json'), $json)
    }
}

Describe 'Cloud client expected-image startup gate' -Tag 'Unit', 'Boot', 'VHD' {
    BeforeEach {
        $script:ReceiptRoot = Join-Path $TestDrive 'receipts'
        New-Item -ItemType Directory -Path $script:ReceiptRoot -Force | Out-Null
        Remove-Item -LiteralPath (Join-Path $script:ReceiptRoot 'current.result.json') -ErrorAction SilentlyContinue
        $script:OutputRoot = Join-Path $TestDrive 'client-logs'
        $script:GateArgs = @{
            AllowElevated = $true
            TimeoutSeconds = 2
            PollSeconds = 1
            MountLogRoot = $script:ReceiptRoot
            LogRoot = $script:OutputRoot
        }
        $script:ImageFixture = New-CimInstance -Namespace root/Microsoft/Windows/Storage -ClassName MSFT_DiskImage -ClientOnly -Property @{
            ImagePath = 'T:\vm\cloud-cache-disk.vhdx'
            Attached = $true
            Number = [uint32]4
        }
        $script:DiskFixture = New-CimInstance -Namespace root/Microsoft/Windows/Storage -ClassName MSFT_Disk -ClientOnly -Property @{
            Number = [uint32]4
        }
        $script:PartitionFixture = New-CimInstance -Namespace root/Microsoft/Windows/Storage -ClassName MSFT_Partition -ClientOnly -Property @{
            DiskNumber = [uint32]4
            PartitionNumber = [uint32]2
            DriveLetter = [char]'F'
        }
        $script:VolumeFixture = [pscustomobject]@{
            DriveLetter = 'F'
            FileSystemLabel = 'cloud-cache-disk'
            FileSystem = 'NTFS'
        }
        Mock Get-DiskImage { $script:ImageFixture }
        Mock Get-Disk { $script:DiskFixture }
        Mock Get-Partition { $script:PartitionFixture }
        Mock Get-Volume { $script:VolumeFixture }
        Mock Get-CimInstance { [pscustomobject]@{ LastBootUpTime = (Get-Date).AddHours(-1) } }
        $script:LiveVolumeFixture = [pscustomobject]@{
            DriveLetter = 'F:'
            Label = 'cloud-cache-disk'
            FileSystem = 'NTFS'
            DeviceID = '\\?\Volume{c3e0daf0-3678-4842-8d9c-66aa3619b11a}\'
        }
        Mock Get-CimInstance { $script:LiveVolumeFixture } -ParameterFilter { $ClassName -eq 'Win32_Volume' }
        Mock Test-Path { $true }
        Mock Get-Process { $null }
        Mock Start-Process {}
        Mock New-Item {}
        Mock Add-Content {}
        Mock Start-Sleep { [System.Threading.Thread]::Sleep($Milliseconds) }
    }

    It 'resolves a real Limited null image Number through canonical DevicePath' {
        $script:ImageFixture = New-CimInstance -Namespace root/Microsoft/Windows/Storage -ClassName MSFT_DiskImage -ClientOnly -Property @{
            ImagePath = 'T:\vm\cloud-cache-disk.vhdx'; Attached = $true; Number = [uint32]4; DevicePath = '\\.\PhysicalDrive4'
        }
        $script:ImageFixture.CimInstanceProperties['Number'].Value = $null
        Invoke-CloudClientGate @script:GateArgs -DryRun | Should -Be 0
        Should -Invoke Get-Disk -Times 1 -Exactly -ParameterFilter { $Number -eq 4 }
        Should -Invoke Get-Partition -Times 1 -Exactly -ParameterFilter { $DiskNumber -eq 4 }
        Should -Invoke Start-Process -Times 0 -Exactly
        Should -Invoke Add-Content -Times 0 -Exactly
    }
    It 'does not admit an image Number conflicting with its actual DevicePath' {
        $script:ImageFixture = New-CimInstance -Namespace root/Microsoft/Windows/Storage -ClassName MSFT_DiskImage -ClientOnly -Property @{
            ImagePath = 'T:\vm\cloud-cache-disk.vhdx'; Attached = $true; Number = [uint32]5; DevicePath = '\\.\PhysicalDrive4'
        }
        Write-TestMountReceipt
        Invoke-CloudClientGate @script:GateArgs -RequireMountLog -DryRun | Should -Be 2
        Should -Invoke Get-Disk -Times 0 -Exactly
        Should -Invoke Start-Process -Times 0 -Exactly
    }
    It 'refuses observed <Case> association even with a fresh matching receipt' -ForEach @(
        @{ Case = 'wrong returned disk'; Fault = 'WrongDisk' }
        @{ Case = 'ambiguous disks'; Fault = 'MultipleDisks' }
        @{ Case = 'wrong returned partition disk'; Fault = 'WrongPartition' }
        @{ Case = 'wrong disk in another returned partition'; Fault = 'OtherWrongPartition' }
        @{ Case = 'ambiguous expected partitions'; Fault = 'MultiplePartitions' }
        @{ Case = 'ambiguous images'; Fault = 'MultipleImages' }
    ) {
        Write-TestMountReceipt
        switch ($Fault) {
            'WrongDisk' { $script:DiskFixture.Number = [uint32]9 }
            'MultipleDisks' { Mock Get-Disk { @($script:DiskFixture, $script:DiskFixture) } }
            'WrongPartition' { $script:PartitionFixture.DiskNumber = [uint32]9 }
            'OtherWrongPartition' {
                Mock Get-Partition { @($script:PartitionFixture, [pscustomobject]@{ DiskNumber = [uint32]9; DriveLetter = 'G' }) }
            }
            'MultiplePartitions' { Mock Get-Partition { @($script:PartitionFixture, $script:PartitionFixture) } }
            'MultipleImages' { Mock Get-DiskImage { @($script:ImageFixture, $script:ImageFixture) } }
        }
        Invoke-CloudClientGate @script:GateArgs -RequireMountLog -DryRun | Should -Be 2
        Should -Invoke Get-CimInstance -Times 0 -Exactly -ParameterFilter { $ClassName -eq 'Win32_Volume' }
        Should -Invoke Start-Process -Times 0 -Exactly
        Should -Invoke Start-Sleep -Times 0 -Exactly
    }
    It 'retains exact receipt fallback for genuinely unavailable <Stage> Storage query' -ForEach @(
        @{ Stage = 'disk'; Fault = 'DiskDenied' }
        @{ Stage = 'partition'; Fault = 'PartitionDenied' }
        @{ Stage = 'empty disk result'; Fault = 'EmptyDisk' }
        @{ Stage = 'incomplete disk number'; Fault = 'NullDisk' }
    ) {
        Write-TestMountReceipt
        switch ($Fault) {
            'DiskDenied' { Mock Get-Disk { throw 'Access denied.' } }
            'PartitionDenied' { Mock Get-Partition { throw 'Access denied.' } }
            'EmptyDisk' { Mock Get-Disk { @() } }
            'NullDisk' { $script:DiskFixture.CimInstanceProperties['Number'].Value = $null }
        }
        Invoke-CloudClientGate @script:GateArgs -RequireMountLog -DryRun | Should -Be 0
        Should -Invoke Get-CimInstance -Times 1 -Exactly -ParameterFilter { $ClassName -eq 'Win32_Volume' }
        Should -Invoke Start-Process -Times 0 -Exactly
    }
    It 'starts both clients hidden after validating the expected image and fresh receipt' {
        Write-TestMountReceipt
        Invoke-CloudClientGate @script:GateArgs -RequireMountLog | Should -Be 0
        Should -Invoke Get-DiskImage -Times 1 -ParameterFilter { $ImagePath -eq 'T:\vm\cloud-cache-disk.vhdx' }
        Should -Invoke Start-Process -Times 1 -ParameterFilter {
            $FilePath -eq 'C:\Program Files (x86)\Dropbox\Client\Dropbox.exe' -and
            $ArgumentList -eq '/systemstartup' -and $WindowStyle -eq 'Hidden'
        }
        Should -Invoke Start-Process -Times 2 -Exactly -ParameterFilter { $WindowStyle -eq 'Hidden' }
        Should -Invoke Add-Content -ParameterFilter { $LiteralPath -eq (Join-Path $script:OutputRoot 'current.log') }
    }

    It 'keeps dry-run free of process and filesystem writes' {
        Write-TestMountReceipt
        Invoke-CloudClientGate @script:GateArgs -RequireMountLog -DryRun | Should -Be 0
        Should -Invoke Start-Process -Times 0
        Should -Invoke New-Item -Times 0
        Should -Invoke Add-Content -Times 0
    }

    It 'accepts the long dry-run CLI form with no writes' {
        Invoke-CloudClientGate @script:GateArgs -CliArgs '--DryRun' | Should -Be 0
        Should -Invoke Start-Process -Times 0
        Should -Invoke Add-Content -Times 0
    }

    It 'refuses <Field> mismatch' -ForEach @(
        @{ Field = 'DriveLetter'; Value = 'G' }
        @{ Field = 'FileSystem'; Value = 'exFAT' }
        @{ Field = 'FileSystemLabel'; Value = 'other-disk' }
    ) {
        $script:VolumeFixture.$Field = $Value
        Invoke-CloudClientGate @script:GateArgs -DryRun | Should -Be 2
        Should -Invoke Start-Process -Times 0
    }

    It 'times out when the expected image cannot be found, even with another F volume' {
        Mock Get-DiskImage { throw 'Image is missing.' }
        $script:GateArgs.TimeoutSeconds = 1
        Invoke-CloudClientGate @script:GateArgs -DryRun | Should -Be 2
        Should -Invoke Get-Volume -Times 0
        Should -Invoke Start-Process -Times 0
    }

    It 'refuses a detached image' {
        $script:ImageFixture.Attached = $false
        $script:GateArgs.TimeoutSeconds = 1
        Invoke-CloudClientGate @script:GateArgs -DryRun | Should -Be 2
        Should -Invoke Get-Disk -Times 0
        Should -Invoke Start-Process -Times 0
    }

    It 'does not substitute a same-label partition at another drive letter' {
        $script:PartitionFixture.DriveLetter = [char]'G'
        $script:GateArgs.TimeoutSeconds = 1
        Invoke-CloudClientGate @script:GateArgs -DryRun | Should -Be 2
        Should -Invoke Get-Volume -Times 0
        Should -Invoke Start-Process -Times 0
    }

    It 'does not accept a partition on a different disk' {
        $script:PartitionFixture.DiskNumber = [uint32]9
        $script:GateArgs.TimeoutSeconds = 1
        Invoke-CloudClientGate @script:GateArgs -DryRun | Should -Be 2
        Should -Invoke Get-Volume -Times 0
        Should -Invoke Start-Process -Times 0
    }

    It 'waits for a previous-boot receipt to be replaced by this boot success' {
        Write-TestMountReceipt -StartedAt (Get-Date).AddDays(-1)
        Mock Start-Sleep { Write-TestMountReceipt }
        Invoke-CloudClientGate @script:GateArgs -RequireMountLog -DryRun | Should -Be 0
        Should -Invoke Start-Sleep -Times 1 -Exactly
    }

    It 'waits for an initially absent receipt within the same deadline' {
        Mock Start-Sleep { Write-TestMountReceipt }
        Invoke-CloudClientGate @script:GateArgs -RequireMountLog -DryRun | Should -Be 0
        Should -Invoke Start-Sleep -Times 1 -Exactly
    }

    It 'times out if only a stale receipt remains' {
        Write-TestMountReceipt -StartedAt (Get-Date).AddDays(-1)
        $script:GateArgs.TimeoutSeconds = 1
        Invoke-CloudClientGate @script:GateArgs -RequireMountLog -DryRun | Should -Be 3
        Should -Invoke Start-Process -Times 0
    }

    It 'fails closed on a fresh failed mount receipt' {
        Write-TestMountReceipt -ExitCode 51 -Status Failed
        Invoke-CloudClientGate @script:GateArgs -RequireMountLog -DryRun | Should -Be 3
        Should -Invoke Start-Sleep -Times 0
        Should -Invoke Start-Process -Times 0
    }

    It 'fails closed on malformed receipt JSON' {
        Set-Content -LiteralPath (Join-Path $script:ReceiptRoot 'current.result.json') -Value '{invalid'
        Invoke-CloudClientGate @script:GateArgs -RequireMountLog -DryRun | Should -Be 3
        Should -Invoke Start-Process -Times 0
    }

    It 'waits for partially written JSON to become a complete successful receipt' {
        [System.IO.File]::WriteAllText((Join-Path $script:ReceiptRoot 'current.result.json'), '{"StartedAt":')
        Mock Start-Sleep { Write-TestMountReceipt }
        Invoke-CloudClientGate @script:GateArgs -RequireMountLog -DryRun | Should -Be 0
        Should -Invoke Start-Sleep -Times 1 -Exactly
    }

    It 'admits Limited-session Storage denial only with fresh receipt and exact live volume GUID' {
        Write-TestMountReceipt
        Mock Get-DiskImage { throw 'Access denied.' }
        Invoke-CloudClientGate @script:GateArgs -RequireMountLog -DryRun | Should -Be 0
        Should -Invoke Get-CimInstance -Times 1 -Exactly -ParameterFilter {
            $ClassName -eq 'Win32_Volume' -and $Filter -eq "DriveLetter='F:'"
        }
    }

    It 'never uses receipt fallback for another live volume GUID' {
        Write-TestMountReceipt
        Mock Get-DiskImage { throw 'Access denied.' }
        $script:LiveVolumeFixture.DeviceID = '\\?\Volume{aaaaaaaa-3678-4842-8d9c-66aa3619b11a}\'
        $script:GateArgs.TimeoutSeconds = 1
        Invoke-CloudClientGate @script:GateArgs -RequireMountLog -DryRun | Should -Be 2
        Should -Invoke Start-Process -Times 0
    }

    It 'never uses receipt fallback without RequireMountLog' {
        Write-TestMountReceipt
        Mock Get-DiskImage { throw 'Access denied.' }
        $script:GateArgs.TimeoutSeconds = 1
        Invoke-CloudClientGate @script:GateArgs -DryRun | Should -Be 2
        Should -Invoke Get-CimInstance -Times 0 -ParameterFilter { $ClassName -eq 'Win32_Volume' }
        Should -Invoke Start-Process -Times 0
    }

    It 'fails closed on a receipt for the wrong image' {
        Write-TestMountReceipt -VhdPath 'T:\vm\other.vhdx'
        Invoke-CloudClientGate @script:GateArgs -RequireMountLog -DryRun | Should -Be 3
        Should -Invoke Start-Process -Times 0
    }

    It 'fails closed when boot-time evidence is unavailable' {
        Mock Get-CimInstance { $null }
        Invoke-CloudClientGate @script:GateArgs -RequireMountLog -DryRun | Should -Be 3
        Should -Invoke Get-DiskImage -Times 0
        Should -Invoke Start-Process -Times 0
    }

    It 'reports process start failure as a nonzero result' {
        Mock Start-Process { throw 'Launch denied.' }
        Invoke-CloudClientGate @script:GateArgs | Should -Be 1
        Should -Invoke Start-Process -Times 2 -Exactly
    }

    It 'skips both clients when their processes already run' {
        Mock Get-Process { [pscustomobject]@{ Id = 123 } }
        Invoke-CloudClientGate @script:GateArgs | Should -Be 0
        Should -Invoke Start-Process -Times 0
    }

    It 'fails if a client executable is missing and no fallback exists' {
        Mock Test-Path { $false } -ParameterFilter { $Path -like '*GoogleDriveFS.exe' }
        Mock Get-ChildItem { $null } -ParameterFilter { $Path -like '*GoogleDriveFS.exe' }
        Invoke-CloudClientGate @script:GateArgs | Should -Be 1
        Should -Invoke Start-Process -Times 1 -Exactly -ParameterFilter { $FilePath -like '*Dropbox.exe' }
    }

    It 'uses the fallback executable when the versioned Google Drive path moved' {
        Mock Test-Path { $false } -ParameterFilter { $Path -like '*GoogleDriveFS.exe' }
        Mock Get-ChildItem {
            [pscustomobject]@{ FullName = 'C:\Program Files\Google\Drive File Stream\131.0.0.0\GoogleDriveFS.exe'; LastWriteTime = Get-Date }
        } -ParameterFilter { $Path -like '*GoogleDriveFS.exe' }
        Invoke-CloudClientGate @script:GateArgs | Should -Be 0
        Should -Invoke Start-Process -Times 1 -Exactly -ParameterFilter {
            $FilePath -eq 'C:\Program Files\Google\Drive File Stream\131.0.0.0\GoogleDriveFS.exe'
        }
    }

    It 'creates only the configured log directory on an actual launch' {
        Mock Test-Path { $false } -ParameterFilter { $Path -eq $script:OutputRoot }
        Invoke-CloudClientGate @script:GateArgs | Should -Be 0
        Should -Invoke New-Item -Times 1 -Exactly -ParameterFilter { $Path -eq $script:OutputRoot }
    }

    It 'returns help without probing the mount or launching clients' {
        Invoke-CloudClientGate @script:GateArgs -CliArgs '--help' | Should -Be 0
        Should -Invoke Get-DiskImage -Times 0
        Should -Invoke Start-Process -Times 0
        Should -Invoke New-Item -Times 0
        Should -Invoke Add-Content -Times 0
    }

    It 'times out if a current receipt never appears' {
        $script:GateArgs.TimeoutSeconds = 1
        Invoke-CloudClientGate @script:GateArgs -RequireMountLog -DryRun | Should -Be 3
        Should -Invoke Start-Process -Times 0
    }

    It 'refuses a receipt with a future start time' {
        Write-TestMountReceipt -StartedAt (Get-Date).AddHours(1)
        Invoke-CloudClientGate @script:GateArgs -RequireMountLog -DryRun | Should -Be 3
        Should -Invoke Start-Sleep -Times 0
        Should -Invoke Start-Process -Times 0
    }

    It 'refuses an image object whose path differs from the requested image' {
        $script:ImageFixture.ImagePath = 'T:\vm\other.vhdx'
        $script:GateArgs.TimeoutSeconds = 1
        Invoke-CloudClientGate @script:GateArgs -DryRun | Should -Be 2
        Should -Invoke Get-Disk -Times 0
        Should -Invoke Start-Process -Times 0
    }

    It 'never launches when validation finishes beyond the shared deadline' {
        Mock Get-Volume {
            [System.Threading.Thread]::Sleep(1100)
            $script:VolumeFixture
        }
        $script:GateArgs.TimeoutSeconds = 1
        Invoke-CloudClientGate @script:GateArgs -DryRun | Should -Be 2
        Should -Invoke Start-Sleep -Times 0
        Should -Invoke Start-Process -Times 0
    }

    It 'rejects invalid <Parameter> before probing the OS' -ForEach @(
        @{ Parameter = 'TimeoutSeconds'; Value = 0 }
        @{ Parameter = 'PollSeconds'; Value = 0 }
        @{ Parameter = 'ExpectedDriveLetter'; Value = 'F:' }
        @{ Parameter = 'ExpectedVhdPath'; Value = 'relative.vhdx' }
    ) {
        $badArgs = @{ AllowElevated = $true; DryRun = $true }
        $badArgs[$Parameter] = $Value
        { Invoke-CloudClientGate @badArgs } | Should -Throw
        Should -Invoke Get-DiskImage -Times 0
        Should -Invoke Start-Process -Times 0
    }
}
