BeforeAll {
    $root = Join-Path $PSScriptRoot '../..'
    Import-Module (Join-Path $root 'Modules/PC-AI.Hardware/PC-AI.Hardware.psd1') -Force
}

Describe 'Native PnP device contract' {
    BeforeEach {
        Mock Get-HardwarePnpDevicesNative -ModuleName PC-AI.Hardware {
            '[{"name":"Healthy","pnp_class":"USB","config_error_code":0,"error_summary":"","status":"OK","device_id":"USB\\0"},{"name":"Broken","pnp_class":"Net","manufacturer":"Vendor","config_error_code":43,"error_summary":"Actual native error","status":"Error","device_id":"PCI\\1"}]'
        }
        Mock Get-CimInstance -ModuleName PC-AI.Hardware {
            [pscustomobject]@{ Name='CIM fallback'; PNPClass='USB'; Manufacturer='Vendor'; ConfigManagerErrorCode=14; Status='Error'; DeviceID='USB\fallback' }
        }
    }
    It 'returns only the actual native error and maps canonical fields' {
        $rows = @(Get-DeviceErrors)
        $rows.Count | Should -Be 1
        $rows[0].Name | Should -Be 'Broken'
        $rows[0].PNPClass | Should -Be 'Net'
        $rows[0].ErrorCode | Should -Be 43
        $rows[0].ErrorDescription | Should -Be 'Actual native error'
        $rows[0].Severity | Should -Not -BeNullOrEmpty
        Should -Invoke Get-CimInstance -ModuleName PC-AI.Hardware -Times 0
    }
    It 'includes healthy native rows only on explicit request' {
        $rows = @(Get-DeviceErrors -IncludeOK)
        $rows.Count | Should -Be 2
        @($rows | Where-Object ErrorCode -eq 0).Count | Should -Be 1
        Should -Invoke Get-CimInstance -ModuleName PC-AI.Hardware -Times 0
    }
    It 'applies class filtering to native results' {
        @(Get-DeviceErrors -Class USB).Count | Should -Be 0
        @(Get-DeviceErrors -IncludeOK -Class USB).Count | Should -Be 1
    }
    It 'accepts an empty native inventory without inventing a CIM error' {
        Mock Get-HardwarePnpDevicesNative -ModuleName PC-AI.Hardware { '[]' }
        @(Get-DeviceErrors).Count | Should -Be 0
        Should -Invoke Get-CimInstance -ModuleName PC-AI.Hardware -Times 0
    }
    It 'rejects an incompatible whole result and discards partial native rows: <Label>' -ForEach @(
        @{ Label='missing code'; Payload='[{"name":"Native partial","config_error_code":43},{"name":"Unknown"}]' }
        @{ Label='negative code'; Payload='[{"name":"Unknown","config_error_code":-1}]' }
        @{ Label='null code'; Payload='[{"name":"Unknown","config_error_code":null}]' }
        @{ Label='fractional code'; Payload='[{"name":"Unknown","config_error_code":0.5}]' }
        @{ Label='malformed JSON'; Payload='[' }
        @{ Label='null inventory'; Payload='null' }
        @{ Label='object inventory'; Payload='{}' }
        @{ Label='single object instead of array'; Payload='{"name":"Healthy","config_error_code":0}' }
        @{ Label='number inventory'; Payload='0' }
        @{ Label='boolean inventory'; Payload='false' }
        @{ Label='string inventory'; Payload='"[]"' }
    ) {
        $script:payload = $Payload
        Mock Get-HardwarePnpDevicesNative -ModuleName PC-AI.Hardware { $script:payload }
        $rows = @(Get-DeviceErrors)
        $rows.Count | Should -Be 1
        $rows[0].Name | Should -Be 'CIM fallback'
        $rows[0].ErrorCode | Should -Be 14
        Should -Invoke Get-CimInstance -ModuleName PC-AI.Hardware -Times 1
    }
    It 'uses CIM when the native call fails' {
        Mock Get-HardwarePnpDevicesNative -ModuleName PC-AI.Hardware { throw 'Native call failed' }
        @(Get-DeviceErrors)[0].Name | Should -Be 'CIM fallback'
    }
}
