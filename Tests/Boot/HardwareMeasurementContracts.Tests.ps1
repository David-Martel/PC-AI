#Requires -Version 7.0
BeforeAll {
    . (Join-Path $PSScriptRoot '..\..\Modules\PC-AI.Hardware\Public\Get-DiskHealth.ps1')
}
Describe 'Disk capability and measured health contracts' {
    BeforeEach {
        Mock Get-CimInstance { @() }
    }
    It 'does not trust a legacy native capability-only passing status' {
        Mock Get-HardwareDiskHealthNative {
            '[{"device_id":"disk0","model":"Fixture","status":"OK","severity":"OK","smart_capable":true,"smart_status_ok":true}]'
        }
        $disk = Get-DiskHealth
        $disk.Status | Should -Be 'Unknown'
        $disk.Severity | Should -Be 'Warning'
        $disk.IsSmartOK | Should -BeNullOrEmpty
        $disk.HealthDataAvailable | Should -BeFalse
    }
    It 'preserves unavailable measured health without turning it into a failure or pass' {
        Mock Get-HardwareDiskHealthNative {
            '[{"device_id":"disk0","model":"Fixture","status":"Unknown","severity":"Unknown","smart_capable":true,"smart_status_ok":null,"health_data_available":false,"health_source":"Capability only"}]'
        }
        $disk = Get-DiskHealth
        $disk.Status | Should -Be 'Unknown'
        $disk.IsSmartOK | Should -BeNullOrEmpty
        $disk.HealthSource | Should -Be 'Capability only'
    }
    It 'retains an explicitly measured failing SMART result' {
        Mock Get-HardwareDiskHealthNative {
            '[{"device_id":"disk0","model":"Fixture","status":"Pred Fail","severity":"Critical","smart_capable":true,"smart_status_ok":false,"health_data_available":true,"health_source":"Measured SMART fixture"}]'
        }
        $disk = Get-DiskHealth
        $disk.Status | Should -Be 'Pred Fail'
        $disk.Severity | Should -Be 'Critical'
        $disk.IsSmartOK | Should -BeFalse
        $disk.HealthDataAvailable | Should -BeTrue
    }
}
