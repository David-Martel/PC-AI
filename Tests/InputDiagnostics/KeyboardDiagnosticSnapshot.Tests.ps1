#Requires -Version 7.0
#Requires -Modules @{ ModuleName = 'Pester'; ModuleVersion = '5.0.0' }
BeforeAll {
    $script:SnapshotPath = Join-Path $PSScriptRoot '../../Tools/InputDiagnostics/Get-KeyboardDiagnosticSnapshot.ps1'
    . $script:SnapshotPath -DryRun | Out-Null
}

Describe 'Keyboard snapshot safety and contract' {
    BeforeEach {
        Mock Get-KeyboardSnapshotMachine { [pscustomobject]@{ Model = 'Test model'; OsUbr = 123 } }
        Mock Get-KeyboardSnapshotProcess { }
        Mock Get-KeyboardSnapshotDevice { }
        Mock Get-KeyboardSnapshotDriver { }
        Mock Get-KeyboardSnapshotFilter { }
        Mock Get-KeyboardSnapshotAccessibility { [pscustomobject]@{ StickyKeys = @{ Enabled = $false } } }
        Mock Get-KeyboardSnapshotPowerToys { throw [System.NotSupportedException]::new('private path') }
        Mock Get-KeyboardSnapshotEvent { }
    }
    It 'distinguishes successful data, empty results and unavailable probes' {
        $report = Get-KeyboardDiagnosticSnapshot
        $report.SchemaVersion | Should -Be '1.0'
        $report.Probes.Count | Should -Be 8
        ($report.Probes | Where-Object Name -EQ Machine).Status | Should -Be 'ok'
        ($report.Probes | Where-Object Name -EQ Processes).Status | Should -Be 'empty'
        ($report.Probes | Where-Object Name -EQ PowerToys).Status | Should -Be 'unavailable'
        @((Get-ChildItem -LiteralPath $TestDrive)).Count | Should -Be 0
    }
    It 'does not query probes or write during dry-run with an output path' {
        $path = Join-Path $TestDrive 'dry.json'
        (Get-KeyboardDiagnosticSnapshot -DryRun -OutputPath $path).Mode | Should -Be 'DryRun'
        Test-Path -LiteralPath $path | Should -BeFalse
        Should -Invoke Get-KeyboardSnapshotMachine -Times 0
        Should -Invoke Get-KeyboardSnapshotEvent -Times 0
    }
    It 'returns help without queries or writes' {
        Get-KeyboardDiagnosticSnapshot -Help -OutputPath (Join-Path $TestDrive 'help.json') | Should -Match 'DryRun'
        Should -Invoke Get-KeyboardSnapshotMachine -Times 0
        @((Get-ChildItem -LiteralPath $TestDrive)).Count | Should -Be 0
    }
    It 'accepts both CLI help aliases without performing CIM queries' {
        Mock Get-CimInstance { throw 'unexpected query' }
        & $script:SnapshotPath -h | Should -Match 'OutputPath'
        & $script:SnapshotPath --help | Should -Match 'OutputPath'
        Should -Invoke Get-CimInstance -Times 0
    }
    It 'preserves error and timeout status without exposing exception content' {
        Mock Get-KeyboardSnapshotMachine { throw 'SERIAL-SECRET typed-text-secret user-name-secret' }
        Mock Get-KeyboardSnapshotEvent { throw [System.TimeoutException]::new('typed-text-secret') }
        $report = Get-KeyboardDiagnosticSnapshot
        ($report.Probes | Where-Object Name -EQ Machine).Status | Should -Be 'error'
        ($report.Probes | Where-Object Name -EQ Events).Status | Should -Be 'timeout'
        ($report | ConvertTo-Json -Depth 12) | Should -Not -Match 'SERIAL-SECRET|typed-text-secret|user-name-secret|private path'
    }
    It 'writes only the explicitly requested JSON and returns the same report' {
        $path = Join-Path $TestDrive 'report.json'
        $report = Get-KeyboardDiagnosticSnapshot -OutputPath $path
        $saved = Get-Content -LiteralPath $path -Raw | ConvertFrom-Json
        ([datetime]$saved.CapturedAtUtc).ToUniversalTime() | Should -Be ([datetime]$report.CapturedAtUtc).ToUniversalTime()
        $saved.Probes.Count | Should -Be 8
        { Get-KeyboardDiagnosticSnapshot -OutputPath $path } | Should -Throw '*already exists*'
    }
    It 'rejects reserved Windows leaves before any probe' -ForEach @(
        @{ Leaf = 'NUL.json' }, @{ Leaf = 'COM1.json' }, @{ Leaf = 'AUX .json' }, @{ Leaf = '$null' }
    ) {
        { Get-KeyboardDiagnosticSnapshot -OutputPath (Join-Path $TestDrive $Leaf) } | Should -Throw '*non-reserved*'
        Should -Invoke Get-KeyboardSnapshotMachine -Times 0
    }
}

Describe 'Keyboard snapshot sanitization of raw sources' {
    BeforeAll { $script:EventJob = Start-ThreadJob { } }
    AfterAll { Remove-Job -Job $script:EventJob -Force }
    It 'projects machine properties without serial, computer name or owner' {
        Mock Get-CimInstance {
            [pscustomobject]@{ Manufacturer = 'Maker'; Model = 'Model'; SMBIOSBIOSVersion = '1'; ReleaseDate = $null; SerialNumber = 'SERIAL-SECRET'; Name = 'USER-SECRET' }
        }
        Mock Get-ItemProperty { [pscustomobject]@{ CurrentBuildNumber = '26000'; DisplayVersion = 'test'; UBR = 123; RegisteredOwner = 'USER-SECRET' } }
        $data = Get-KeyboardSnapshotMachine | ConvertTo-Json
        $data | Should -Match 'Model'
        $data | Should -Not -Match 'SERIAL-SECRET|USER-SECRET|SerialNumber|RegisteredOwner'
        Should -Invoke Get-CimInstance -Times 2 -ParameterFilter { $OperationTimeoutSec -eq 5 }
    }
    It 'projects process names and priorities without command lines' {
        Mock Get-Process { [pscustomobject]@{ ProcessName = 'ctfmon'; PriorityClass = 'Normal'; CommandLine = 'typed-text-secret'; Path = 'USER-SECRET' } }
        $data = Get-KeyboardSnapshotProcess | ConvertTo-Json
        $data | Should -Match 'ctfmon'
        $data | Should -Not -Match 'typed-text-secret|USER-SECRET|CommandLine'
    }
    It 'includes bounded Logitech and Lenovo input-related process names' {
        Mock Get-Process {
            foreach ($name in @('logioptionsplus_agent', 'logioptionsplus_updater', 'logioptionsplus_appbroker', 'Lenovo.Modern.ImController', 'LenovoGoCentral1', 'LenovoAccessoriesAndDisplayControlCenterService', 'unrelated-secret-process')) {
                [pscustomobject]@{ ProcessName = $name; PriorityClass = 'Normal' }
            }
        }
        $processes = @(Get-KeyboardSnapshotProcess)
        $processes.Count | Should -Be 6
        $processes.Name | Should -Not -Contain 'unrelated-secret-process'
    }
    It 'correlates keyboard, driver and filter rows using a non-raw stable token' {
        Mock Get-CimInstance { [pscustomobject]@{ PNPDeviceID = 'USB\SERIAL-SECRET'; PNPClass = 'Keyboard'; Status = 'OK'; Service = 'kbdhid'; ConfigManagerErrorCode = 0 } }
        Mock Get-ItemProperty {
            [pscustomobject]@{ Driver = '{4D36E96B-E325-11CE-BFC1-08002BE10318}\0001'; ProviderName = 'Test'; DriverVersion = '1'; DriverDate = 'test'; InfPath = 'keyboard.inf'; UpperFilters = @('kbdclass') }
        }
        $device = Get-KeyboardSnapshotDevice
        $driver = Get-KeyboardSnapshotDriver
        $filter = Get-KeyboardSnapshotFilter | Where-Object Scope -EQ Instance
        $device.DeviceToken | Should -Match '^kbd-[0-9a-f]{16}$'
        $driver.DeviceToken | Should -Be $device.DeviceToken
        $filter.DeviceToken | Should -Be $device.DeviceToken
        Get-KeyboardSnapshotDeviceToken 'usb\serial-secret' | Should -Be $device.DeviceToken
        (@($device, $driver, $filter) | ConvertTo-Json -Depth 8) | Should -Not -Match 'SERIAL-SECRET|PNPDeviceID'
    }
    It 'reads active device driver references without exposing instance identifiers' {
        Mock Get-CimInstance { [pscustomobject]@{ PNPDeviceID = 'USB\SERIAL-SECRET' } }
        Mock Get-ItemProperty {
            if ($LiteralPath -like '*\Enum\*') { [pscustomobject]@{ Driver = '{4D36E96B-E325-11CE-BFC1-08002BE10318}\0001' } }
            else { [pscustomobject]@{ ProviderName = 'Test provider'; DriverVersion = '1.2'; DriverDate = '2026-01-01'; InfPath = 'keyboard.inf'; SerialNumber = 'SERIAL-SECRET' } }
        }
        $data = Get-KeyboardSnapshotDriver
        $data.DriverVersion | Should -Be '1.2'
        $data.Status | Should -Be 'ok'
        $data.SignatureStatus | Should -Be 'not-queried'
        ($data | ConvertTo-Json) | Should -Not -Match 'SERIAL-SECRET|PNPDeviceID'
    }
    It 'counts PowerToys mappings without exposing remaps or text' {
        Mock Test-Path { $true }
        Mock Get-Content {
            if ($LiteralPath -match 'default.json$') {
                '{"remapKeys":{"inProcess":[{"originalKeys":"SERIAL-SECRET","newRemapKeys":"typed-text-secret"}]},"remapShortcuts":{"global":[{},{}]},"remapKeysToText":{"inProcess":[{"text":"USER-SECRET"}]}}'
            } elseif ($LiteralPath -match 'Keyboard Manager') { '{"properties":{"activeConfiguration":{"value":"default"}}}' }
            else { '{"enabled":{"Keyboard Manager":true}}' }
        }
        $data = Get-KeyboardSnapshotPowerToys
        $data.KeyRemapCount | Should -Be 1
        $data.ShortcutRemapCount | Should -Be 2
        $data.TextRemapCount | Should -Be 1
        ($data | ConvertTo-Json) | Should -Not -Match 'SERIAL-SECRET|typed-text-secret|USER-SECRET'
    }
    It 'reports unavailable profile counts as null instead of zero' {
        Mock Test-Path { $LiteralPath -notmatch 'default.json$' }
        Mock Get-Content { '{"enabled":{"Keyboard Manager":false}}' }
        $data = Get-KeyboardSnapshotPowerToys
        $data.RemapStatus | Should -Be 'unavailable'
        $data.KeyRemapCount | Should -BeNullOrEmpty
    }
    It 'reports unsupported mapping shape as unavailable: <Case>' -ForEach @(
        @{ Case = 'section is scalar'; ProfileJson = '{"remapKeys":"unexpected-schema"}' }
        @{ Case = 'section is array'; ProfileJson = '{"remapKeys":[]}' }
        @{ Case = 'child is object'; ProfileJson = '{"remapKeys":{"inProcess":{"originalKeys":"secret"}}}' }
        @{ Case = 'child is null'; ProfileJson = '{"remapKeys":{"inProcess":null}}' }
        @{ Case = 'mapping element is scalar'; ProfileJson = '{"remapKeys":{"inProcess":["secret"]}}' }
        @{ Case = 'mapping element is null'; ProfileJson = '{"remapKeys":{"inProcess":[null]}}' }
        @{ Case = 'malformed section follows valid section'; ProfileJson = '{"remapKeys":{"inProcess":[]},"remapShortcuts":{"global":{}}}' }
    ) {
        $script:ProfileFixture = $ProfileJson
        Mock Test-Path { $true }
        Mock Get-Content {
            if ($LiteralPath -match 'default.json$') { $script:ProfileFixture }
            elseif ($LiteralPath -match 'Keyboard Manager') { '{"properties":{"activeConfiguration":{"value":"default"}}}' }
            else { '{"enabled":{"Keyboard Manager":true}}' }
        }
        $probe = Invoke-KeyboardSnapshotProbe 'PowerToys' { Get-KeyboardSnapshotPowerToys }
        $probe.Status | Should -Be 'unavailable'
        $probe.Data.Count | Should -Be 0
        ($probe | ConvertTo-Json -Depth 8) | Should -Not -Match 'KeyRemapCount|ShortcutRemapCount|TextRemapCount|secret'
    }
    It 'accepts supported empty arrays with optional sections absent' {
        Mock Test-Path { $true }
        Mock Get-Content {
            if ($LiteralPath -match 'default.json$') { '{"remapKeys":{"inProcess":[]}}' }
            elseif ($LiteralPath -match 'Keyboard Manager') { '{"properties":{"activeConfiguration":{"value":"default"}}}' }
            else { '{"enabled":{"Keyboard Manager":true}}' }
        }
        $data = Get-KeyboardSnapshotPowerToys
        $data.RemapStatus | Should -Be 'ok'
        $data.KeyRemapCount | Should -Be 0
        $data.ShortcutRemapCount | Should -Be 0
        $data.TextRemapCount | Should -Be 0
    }
    It 'emits only event metadata and bounds the event query' {
        Mock Start-ThreadJob { $script:EventAction = $ScriptBlock; $script:EventJob }
        Mock Wait-Job { $true }
        Mock Receive-Job { & $script:EventAction 60 }
        Mock Stop-Job { }
        Mock Remove-Job { }
        Mock Get-WinEvent {
            [pscustomobject]@{ ProviderName = 'Display'; Id = 4101; Level = 3; TimeCreated = [datetime]::UtcNow; Message = 'typed-text-secret'; UserId = 'USER-SECRET' }
        }
        $data = Get-KeyboardSnapshotEvent -Minutes 60
        $data.Id | Should -Be 4101
        ($data | ConvertTo-Json) | Should -Not -Match 'typed-text-secret|USER-SECRET|Message|UserId'
        Should -Invoke Get-WinEvent -Times 1 -ParameterFilter { $MaxEvents -eq 100 }
        Should -Invoke Wait-Job -Times 1 -ParameterFilter { $Timeout -eq 8 }
    }
}
