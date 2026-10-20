#Requires -Version 7.0
#Requires -Modules @{ ModuleName = 'Pester'; ModuleVersion = '5.0.0' }

BeforeAll {
    $script:RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
    $script:Tool = Join-Path $script:RepoRoot 'Tools/Get-PcaiUsbDescriptorInventory.ps1'
    . $script:Tool -DryRun | Out-Null
    function New-UsbFixture {
        [pscustomobject]@{ HubIndex=1; PortCount=1; BusPowered=$false; Error=$null; Ports=@(
            [pscustomobject]@{ Port=1; ConnectionStatus=1; VendorId='413C'; ProductId='C015'
                DeclaredMaxBusCurrentMa=896; Error=$null; SpeedQueryError=$null; ConfigurationQueryError=$null }
        ) }
    }
}

Describe 'USB descriptor inventory wrapper safety' -Tag 'Unit' {
    It 'parses without syntax errors' {
        $tokens=$null; $errors=$null
        [Management.Automation.Language.Parser]::ParseFile($script:Tool,[ref]$tokens,[ref]$errors) | Out-Null
        $errors.Count | Should -Be 0
    }
    It 'does not load native code, query hardware or create folders on dry run' {
        Mock Add-Type { throw 'native loading prohibited' }
        Mock Read-PcaiUsbHubs { throw 'hardware query prohibited' }
        $folder = Join-Path $TestDrive 'dry-run-must-not-exist'
        $result = Invoke-PcaiUsbDescriptorInventory -DryRun -OutputPath (Join-Path $folder 'report.json')
        $result.status | Should -Be 'DryRun'
        $result.hardwareQueried | Should -BeFalse
        $result.nativeLoaded | Should -BeFalse
        $result.writesPerformed | Should -BeFalse
        Test-Path -LiteralPath $folder | Should -BeFalse
        Should -Invoke Add-Type -Times 0 -Exactly
        Should -Invoke Read-PcaiUsbHubs -Times 0 -Exactly
        (Invoke-PcaiUsbDescriptorInventory -DryRun -AsJson | ConvertFrom-Json).status | Should -Be 'DryRun'
    }
    It 'accepts help aliases without loading native code or writing output' {
        Mock Add-Type { throw 'help must not compile' }
        $path = Join-Path $TestDrive 'help.json'
        foreach ($alias in @('h','-help')) {
            $parameters = @{ OutputPath=$path; $alias=$true }
            (& $script:Tool @parameters | Out-String) | Should -Match 'declared bus-current limits'
        }
        Test-Path -LiteralPath $path | Should -BeFalse
        Should -Invoke Add-Type -Times 0 -Exactly
    }
    It 'rejects relative, reserved components and alternate data streams before querying' {
        Mock Read-PcaiUsbHubs { throw 'invalid path must not query' }
        { Invoke-PcaiUsbDescriptorInventory -OutputPath './relative.json' } | Should -Throw '*absolute*'
        foreach ($leaf in @('AUX.json','NUL','LPT1.','COM2.txt','$null','$null.json','CON.dir/report.json','COM¹.json','COM².json','COM³.json','LPT¹.json','LPT².json','LPT³.json')) {
            { Invoke-PcaiUsbDescriptorInventory -OutputPath (Join-Path $TestDrive $leaf) } | Should -Throw '*reserved*'
        }
        { Invoke-PcaiUsbDescriptorInventory -OutputPath (Join-Path $TestDrive 'safe.json:stream') } | Should -Throw '*stream*'
        { Invoke-PcaiUsbDescriptorInventory -OutputPath $TestDrive } | Should -Throw '*JSON file*'
        { Invoke-PcaiUsbDescriptorInventory -OutputPath (Join-Path $TestDrive 'report.txt') } | Should -Throw '*JSON file*'
        Should -Invoke Read-PcaiUsbHubs -Times 0 -Exactly
    }
    It 'rejects device namespace output syntax' {
        Mock Read-PcaiUsbHubs { throw 'device path must not query' }
        { Invoke-PcaiUsbDescriptorInventory -OutputPath '\\?\C:\Temp\usb.json' } | Should -Throw '*device namespace*'
        Should -Invoke Read-PcaiUsbHubs -Times 0 -Exactly
    }
    It 'preserves an existing file without Force and does not query hardware' {
        Mock Read-PcaiUsbHubs { throw 'existing evidence must be preserved' }
        $path = Join-Path $TestDrive 'existing.json'
        [IO.File]::WriteAllText($path,'original evidence')
        { Invoke-PcaiUsbDescriptorInventory -OutputPath $path } | Should -Throw '*already exists*'
        [IO.File]::ReadAllText($path) | Should -BeExactly 'original evidence'
        Should -Invoke Read-PcaiUsbHubs -Times 0 -Exactly
    }
    It 'propagates initialization/query failure without creating an output folder' {
        Mock Read-PcaiUsbHubs { throw 'fixture enumeration failure' }
        $folder = Join-Path $TestDrive 'failure-no-folder'
        { Invoke-PcaiUsbDescriptorInventory -OutputPath (Join-Path $folder 'usb.json') } | Should -Throw '*fixture enumeration failure*'
        Test-Path -LiteralPath $folder | Should -BeFalse
    }
    It 'does not truncate existing evidence when a forced query fails' {
        Mock Read-PcaiUsbHubs { throw 'fixture enumeration failure' }
        $path = Join-Path $TestDrive 'failed-force.json'
        [IO.File]::WriteAllText($path,'preserve on query failure')
        { Invoke-PcaiUsbDescriptorInventory -OutputPath $path -Force } | Should -Throw '*fixture enumeration failure*'
        [IO.File]::ReadAllText($path) | Should -BeExactly 'preserve on query failure'
    }
    It 'returns and writes source-labelled JSON with measured current unknown' {
        Mock Read-PcaiUsbHubs { New-UsbFixture }
        $path = Join-Path $TestDrive 'new-folder/usb.json'
        $json = Invoke-PcaiUsbDescriptorInventory -AsJson -OutputPath $path
        $report = $json | ConvertFrom-Json
        $report.hubCount | Should -Be 1
        $report.portCount | Should -Be 1
        $report.connectedPortCount | Should -Be 1
        $report.status | Should -Be 'Complete'
        $report.measuredBusCurrentMa | Should -BeNullOrEmpty
        $report.pcInputWatts | Should -BeNullOrEmpty
        $report.usbPdNegotiation | Should -Be 'Unknown'
        $report.interpretation | Should -Match 'not measured current'
        $report.hubs[0].HubIndex | Should -Be 1
        $report.hubs[0].PSObject.Properties.Name | Should -Not -Contain 'Path'
        [IO.File]::ReadAllText($path).Trim() | Should -BeExactly $json
    }
    It 'replaces only the explicit output when Force is supplied' {
        Mock Read-PcaiUsbHubs { New-UsbFixture }
        $path = Join-Path $TestDrive 'force.json'
        [IO.File]::WriteAllText($path,'old')
        Invoke-PcaiUsbDescriptorInventory -OutputPath $path -Force | Out-Null
        ([IO.File]::ReadAllText($path) | ConvertFrom-Json).reportType | Should -Be 'WindowsUsbDescriptorInventory'
    }
    It 'reports per-port failures as partial rather than a successful measurement' {
        Mock Read-PcaiUsbHubs {
            $fixture = New-UsbFixture
            $fixture.Ports[0].ConfigurationQueryError = 'Win32Error:31'
            $fixture.Ports[0].DeclaredMaxBusCurrentMa = $null
            $fixture
        }
        $report = Invoke-PcaiUsbDescriptorInventory
        $report.status | Should -Be 'Partial'
        $report.hubs[0].Ports[0].DeclaredMaxBusCurrentMa | Should -BeNullOrEmpty
    }
}

Describe 'USB configuration decoding without hardware' -Tag 'Unit' {
    BeforeAll {
        if (-not ('PcAi.Diagnostics.UsbInventory' -as [type])) {
            Add-Type -Path (Join-Path $script:RepoRoot 'Tools/Native/UsbHubInventory.cs')
        }
        function New-UsbConfiguration([byte]$Power=112, [byte]$Attributes=128) {
            [byte[]]@(9,2,18,0,1,1,0,$Attributes,$Power,9,4,0,0,0,14,1,0,0)
        }
    }
    It 'uses 8 mA units for active SuperSpeed and 2 mA units for active USB2' {
        foreach ($speed in @($true,$false)) {
            $port = [PcAi.Diagnostics.PortInfo]::new()
            $port.Configuration=1; $port.SuperSpeedOrHigher=$speed
            [PcAi.Diagnostics.UsbInventory]::DecodeConfiguration((New-UsbConfiguration),18,$port)
            $port.DeclaredMaxBusCurrentMa | Should -Be $(if ($speed) {896} else {224})
            $port.InterfaceClasses | Should -Contain '0E:01:00'
        }
    }
    It 'leaves the active budget unknown if speed is unknown or configuration differs' {
        foreach ($active in @(1,2)) {
            $port = [PcAi.Diagnostics.PortInfo]::new()
            $port.Configuration=$active
            if ($active -eq 2) { $port.SuperSpeedOrHigher=$true }
            [PcAi.Diagnostics.UsbInventory]::DecodeConfiguration((New-UsbConfiguration),18,$port)
            $port.MaxPowerRaw | Should -Be 112
            $port.DeclaredMaxBusCurrentMa | Should -BeNullOrEmpty
        }
    }
    It 'records self-power and remote-wake capability without claiming enabled state' {
        $port = [PcAi.Diagnostics.PortInfo]::new()
        $port.Configuration=1; $port.SuperSpeedOrHigher=$false
        [PcAi.Diagnostics.UsbInventory]::DecodeConfiguration((New-UsbConfiguration -Power 0 -Attributes 224),18,$port)
        $port.SelfPowered | Should -BeTrue
        $port.RemoteWakeCapable | Should -BeTrue
        $port.DeclaredMaxBusCurrentMa | Should -Be 0
        $port.PSObject.Properties.Name | Should -Not -Contain 'RemoteWakeEnabled'
    }
    It 'rejects descriptor configuration zero and retains unknown budget for an unconfigured device' {
        $port = [PcAi.Diagnostics.PortInfo]::new()
        $port.Configuration=0; $port.SuperSpeedOrHigher=$true
        [byte[]]$bytes = New-UsbConfiguration
        $bytes[5]=0
        { [PcAi.Diagnostics.UsbInventory]::DecodeConfiguration($bytes,18,$port) } | Should -Throw '*Malformed configuration header*'
        $port.DeclaredMaxBusCurrentMa | Should -BeNullOrEmpty
        [PcAi.Diagnostics.UsbInventory]::DecodeConfiguration((New-UsbConfiguration),18,$port)
        $port.MatchesActiveConfiguration | Should -BeFalse
        $port.DeclaredMaxBusCurrentMa | Should -BeNullOrEmpty
    }
    It 'rejects truncated, malformed and short interface descriptors before publishing fields' {
        $valid = New-UsbConfiguration
        foreach ($bytes in @(
            [byte[]]@(9,2,18),
            [byte[]]@(9,2,18,0,1,1,0,128,112,0,4,0,0,0,14,1,0,0),
            [byte[]]@(9,2,18,0,1,1,0,0,112,9,4,0,0,0,14,1,0,0),
            [byte[]]@(9,2,11,0,1,1,0,128,112,2,4)
        )) {
            $port = [PcAi.Diagnostics.PortInfo]::new()
            $port.Configuration=1; $port.SuperSpeedOrHigher=$true
            { [PcAi.Diagnostics.UsbInventory]::DecodeConfiguration($bytes,$bytes.Length,$port) } | Should -Throw
            $port.DeclaredMaxBusCurrentMa | Should -BeNullOrEmpty
            $port.DescriptorConfigurationValue | Should -BeNullOrEmpty
        }
        $valid.Length | Should -Be 18
    }
}
