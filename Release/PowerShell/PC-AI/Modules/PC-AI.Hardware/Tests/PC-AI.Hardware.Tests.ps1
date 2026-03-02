# PC-AI.Hardware Pester Tests

BeforeAll {
	$PC_AIRoot = Resolve-Path (Join-Path $PSScriptRoot '..\..\..')
	# Ensure PC-AI.Common and Acceleration are available
	Import-Module (Join-Path $PC_AIRoot 'Modules\PC-AI.Common') -Force
	Import-Module (Join-Path $PC_AIRoot 'Modules\PC-AI.Acceleration') -Force

	# Initialize native tools
	Initialize-PcaiNative -Force

	$ModulePath = Join-Path $PSScriptRoot '..'
	Import-Module (Join-Path $ModulePath 'PC-AI.Hardware.psd1') -Force
}

Describe 'PC-AI.Hardware Native Integration' {

	Context 'Get-DiskHealth' {
		It 'Should return disk health details from native module when available' {
			# Mock the native internal helper
			Mock Get-HardwareDiskHealthNative -MockWith { return '[{"device_id":"\\\\\\\\.\\\\\\\\PhysicalDrive0","model":"Mock SSD","serial_number":"MOCK123","status":"OK","smart_capable":true,"smart_status_ok":true,"severity":"OK"}]' } -ModuleName 'PC-AI.Hardware'

			$result = Get-DiskHealth
			$result | Should -Not -BeNullOrEmpty
			$result[0].Model | Should -Be 'Mock SSD'
			$result[0].Status | Should -Be 'OK'
		}

		It 'Should fallback to CIM when native module is unavailable' {
			# Mock native call to fail or be absent
			Mock Get-HardwareDiskHealthNative -MockWith { return $null } -ModuleName 'PC-AI.Hardware'

			$result = Get-DiskHealth
			$result | Should -Not -BeNullOrEmpty
			# Check for standard disk property
			$result[0].Model | Should -Not -BeNullOrEmpty
		}
	}

	Context 'Get-SystemEvents' {
		It 'Should return sampled hardware events from native module' {
			# Mock the native internal helper
			Mock Get-HardwareSystemEventsNative -MockWith { return '[{"time_created":"2026-01-31T08:00:00Z","provider_name":"disk","id":11,"level":2,"severity":"Error","message":"Mock error"}]' } -ModuleName 'PC-AI.Hardware'

			$result = Get-SystemEvents -Days 1
			$result | Should -Not -BeNullOrEmpty
			$result[0].ProviderName | Should -Be 'disk'
			$result[0].Id | Should -Be 11
		}
	}

	Context 'Get-DeviceErrors' {
		It 'Should return devices with errors from native PnP logic' {
			# Mock the native internal helper
			Mock Get-HardwarePnpDevicesNative -MockWith { return '[{"name":"Broken Device","class_name":"DiskDrive","manufacturer":"BadVendor","problem_code":43,"problem_description":"Windows has stopped this device because it has reported problems.","severity":"Error","status":"Error","device_id":"PCI\\123"}]' } -ModuleName 'PC-AI.Hardware'

			$result = Get-DeviceErrors
			$result | Should -Not -BeNullOrEmpty
			$result[0].Name | Should -Be 'Broken Device'
			$result[0].ErrorCode | Should -Be 43
		}
	}
}
