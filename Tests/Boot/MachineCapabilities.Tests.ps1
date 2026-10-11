#Requires -Version 7.0
#Requires -Modules @{ ModuleName = 'Pester'; ModuleVersion = '5.0.0' }

BeforeAll {
    $script:RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
    $script:Tool = Join-Path $script:RepoRoot 'Tools\Get-PcaiMachineCapabilities.ps1'
    $script:ReportSchema = Join-Path $script:RepoRoot 'Config\machine-profiles\machine-capabilities.schema.json'
    $script:ProfileSchema = Join-Path $script:RepoRoot 'Config\machine-profiles\machine-profile.schema.json'
    function New-CapabilityFixtureProfile {
        param([string]$Path, [int]$Jobs = 4)
        [ordered]@{
            schemaVersion = 1; hostName = [environment]::MachineName.ToLowerInvariant()
            baseline = @{ manufacturer = $null; model = $null; processor = $null; source = 'Fixture' }
            expectedHardware = @(@{ name = 'NVIDIA A2000'; source = 'Fixture expectation'; lastReportedPresence = 'Unknown'; connection = 'Unknown' })
            statedPowerPath = @{ source = 'Fixture user statement'; maximumWatts = 100 }
            buildPolicy = @{ maxNativeBuildJobs = $Jobs; source = 'Fixture policy' }
        } | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $Path
    }
}

Describe 'Machine capability script safety' -Tag 'Boot', 'Unit' {
    It 'parses and contains no native imports, vendor process launches or mutations of Windows configuration' {
        $tokens = $null
        $parseErrors = $null
        $ast = [System.Management.Automation.Language.Parser]::ParseFile($script:Tool, [ref]$tokens, [ref]$parseErrors)
        $parseErrors.Count | Should -Be 0
        $commands = @($ast.FindAll({ param($node) $node -is [System.Management.Automation.Language.CommandAst] }, $true) |
            ForEach-Object { $_.GetCommandName() })
        foreach ($name in @('Import-Module', 'Add-Type', 'Start-Process', 'Register-ScheduledTask', 'Write-EventLog', 'Set-ItemProperty', 'Invoke-Expression', 'Remove-Item')) {
            $commands | Should -Not -Contain $name
        }
    }

    It 'performs no hardware queries, profile reads or writes during DryRun' -Skip:(-not $IsWindows) {
        Mock Get-CimInstance { throw 'DryRun query prohibited' }
        $folder = Join-Path $TestDrive 'must-not-exist'
        $path = Join-Path $folder 'capabilities.json'
        $result = & $script:Tool -DryRun -OutputPath $path -ProfilePath (Join-Path $TestDrive 'nonexistent-profile.json')
        $result.status | Should -Be 'DryRun'
        $result.hardwareQueried | Should -BeFalse
        $result.writesPerformed | Should -BeFalse
        Test-Path -LiteralPath $folder | Should -BeFalse
        Should -Invoke Get-CimInstance -Exactly -Times 0
        (& $script:Tool -DryRun -AsJson | ConvertFrom-Json).status | Should -Be 'DryRun'
    }

    It 'supports both help aliases without output writes' {
        $path = Join-Path $TestDrive 'help.json'
        foreach ($alias in @('h', '-help')) {
            $parameters = @{ OutputPath = $path; $alias = $true }
            (& $script:Tool @parameters | Out-String) | Should -Match 'observed Windows hardware capabilities'
        }
        Test-Path -LiteralPath $path | Should -BeFalse
    }

    It 'rejects relative, reserved and stream output paths before querying hardware' -Skip:(-not $IsWindows) {
        Mock Get-CimInstance { throw 'Invalid paths must not query' }
        { & $script:Tool -OutputPath '.\relative.json' } | Should -Throw '*absolute*'
        foreach ($leaf in @('AUX.json', 'NUL', 'LPT1.', 'COM2.txt', '$null')) {
            $path = Join-Path $TestDrive $leaf
            { & $script:Tool -OutputPath $path } | Should -Throw '*reserved*'
        }
        { & $script:Tool -OutputPath (Join-Path $TestDrive 'safe.json:stream') } | Should -Throw '*stream*'
        { & $script:Tool -OutputPath $TestDrive } | Should -Throw '*JSON file*'
        Should -Invoke Get-CimInstance -Exactly -Times 0
    }

    It 'preserves an existing output file unless explicitly forced' -Skip:(-not $IsWindows) {
        Mock Get-CimInstance { throw 'Existing output must be preserved without queries' }
        $path = Join-Path $TestDrive 'existing.json'
        Set-Content -LiteralPath $path -Value 'preserve this evidence'
        { & $script:Tool -OutputPath $path } | Should -Throw '*already exists*'
        (Get-Content -LiteralPath $path) | Should -Be 'preserve this evidence'
        Should -Invoke Get-CimInstance -Exactly -Times 0
    }

    It 'ships valid profiles with user expectations separated from unknown dtm-work identity and power' {
        foreach ($name in @('dtm-super-nuc', 'dtm-work')) {
            $path = Join-Path $script:RepoRoot "Config\machine-profiles\$name.json"
            Get-Content -LiteralPath $path -Raw | Test-Json -SchemaFile $script:ProfileSchema -ErrorAction Stop | Should -BeTrue
        }
        $work = Get-Content -LiteralPath (Join-Path $script:RepoRoot 'Config\machine-profiles\dtm-work.json') -Raw | ConvertFrom-Json
        $work.baseline.processor | Should -BeNullOrEmpty
        $work.baseline.model | Should -BeNullOrEmpty
        $work.statedPowerPath | Should -BeNullOrEmpty
        $work.expectedHardware[0].name | Should -Be 'NVIDIA A2000'
        $work.expectedHardware[0].lastReportedPresence | Should -Be 'Unknown'
        $nuc = Get-Content -LiteralPath (Join-Path $script:RepoRoot 'Config\machine-profiles\dtm-super-nuc.json') -Raw | ConvertFrom-Json
        $nuc.statedPowerPath.activeAdapter.maximumWatts | Should -Be 100
        $nuc.statedPowerPath.oemReference.maximumWatts | Should -Be 120
        $nuc.expectedHardware[1].lastReportedPresence | Should -Be 'Disconnected'
        $nuc.statedPowerPath.inputPowerMeasured | Should -BeFalse
    }
}

Describe 'Machine capability fixture behavior' -Tag 'Boot', 'Unit' -Skip:(-not $IsWindows) {
    BeforeEach {
        $script:FixtureProfile = Join-Path $TestDrive 'fixture-profile.json'
        New-CapabilityFixtureProfile $script:FixtureProfile
        Mock Get-CimInstance {
            param($ClassName)
            switch ($ClassName) {
                'Win32_ComputerSystem' { [pscustomobject]@{ Manufacturer = 'Fixture vendor'; Model = 'Fixture model'; TotalPhysicalMemory = 64GB; NumberOfLogicalProcessors = 20 }; break }
                'Win32_Processor' { [pscustomobject]@{ Name = 'Fixture CPU'; Manufacturer = 'Fixture vendor'; NumberOfCores = 14; NumberOfLogicalProcessors = 20 }; break }
                'Win32_OperatingSystem' { [pscustomobject]@{ Caption = 'Windows fixture'; Version = '10.0'; BuildNumber = 'fixture'; FreePhysicalMemory = 16GB / 1KB }; break }
                'Win32_PnPEntity' {
                    @(
                        [pscustomobject]@{ Name = 'Intel Iris Xe'; PNPClass = 'Display'; PNPDeviceID = 'PCI\VEN_8086&DEV_A7A0\PRIVATE_INSTANCE'; Present = $true; Status = 'OK'; ConfigManagerErrorCode = 0 }
                        [pscustomobject]@{ Name = 'NVIDIA A2000'; PNPClass = 'Display'; PNPDeviceID = 'PCI\VEN_10DE&DEV_25B6\PRIVATE_GPU_SERIAL'; Present = $true; Status = 'OK'; ConfigManagerErrorCode = 0 }
                        [pscustomobject]@{ Name = 'NVIDIA RTX 5060 Ti'; PNPClass = 'Display'; PNPDeviceID = 'PCI\VEN_10DE&DEV_2D04\DISCONNECTED_SECRET'; Present = $false; Status = 'OK'; ConfigManagerErrorCode = 45 }
                        [pscustomobject]@{ Name = 'USB camera'; PNPClass = 'Camera'; PNPDeviceID = 'USB\VID_046D&PID_0825\CAMERA_SERIAL_SECRET'; Present = $true; Status = 'OK'; ConfigManagerErrorCode = 0 }
                        [pscustomobject]@{ Name = 'Intel Thunderbolt controller'; PNPClass = 'System'; PNPDeviceID = 'PCI\VEN_8086&DEV_1137\TB_SECRET'; Present = $true; Status = 'OK'; ConfigManagerErrorCode = 0 }
                        [pscustomobject]@{ Name = 'USB NIC'; PNPClass = 'Net'; PNPDeviceID = 'USB\VID_0BDA&PID_8156\NIC_SERIAL_SECRET'; Present = $true; Status = 'OK'; ConfigManagerErrorCode = 0 }
                        [pscustomobject]@{ Name = 'USB audio'; PNPClass = 'Media'; PNPDeviceID = 'USB\VID_1234&PID_5678\AUDIO_SECRET'; Present = $null; Status = 'OK'; ConfigManagerErrorCode = $null }
                    ); break
                }
                'Win32_PnPSignedDriver' {
                    @(
                        [pscustomobject]@{ DeviceID = 'PCI\VEN_8086&DEV_A7A0\PRIVATE_INSTANCE'; DriverVersion = '32.0.fixture'; DriverProviderName = 'Intel'; IsSigned = $true }
                        [pscustomobject]@{ DeviceID = 'PCI\VEN_10DE&DEV_25B6\PRIVATE_GPU_SERIAL'; DriverVersion = 'fixture.nvidia'; DriverProviderName = 'NVIDIA'; IsSigned = $true }
                        [pscustomobject]@{ DeviceID = 'PCI\VEN_10DE&DEV_2D04\DISCONNECTED_SECRET'; DriverVersion = 'fixture.nvidia'; DriverProviderName = 'NVIDIA'; IsSigned = $true }
                    ); break
                }
                'Win32_VideoController' {
                    @(
                        [pscustomobject]@{ Name = 'Intel Iris Xe'; PNPDeviceID = 'PCI\VEN_8086&DEV_A7A0\PRIVATE_INSTANCE'; DriverVersion = '32.0.fixture' }
                        [pscustomobject]@{ Name = 'NVIDIA A2000'; PNPDeviceID = 'PCI\VEN_10DE&DEV_25B6\PRIVATE_GPU_SERIAL'; DriverVersion = 'fixture.nvidia' }
                    ); break
                }
                'Win32_NetworkAdapter' { [pscustomobject]@{ PNPDeviceID = 'USB\VID_0BDA&PID_8156\NIC_SERIAL_SECRET'; PhysicalAdapter = $true; NetConnectionStatus = 2; Speed = 2500000000 }; break }
                'Win32_PerfFormattedData_PowerMeterCounter_EnergyMeter' { [pscustomobject]@{ Name = 'RAPL_Package0_DRAM'; Energy = 0; Time = 1000; Power = 0 }; break }
                'Win32_PerfFormattedData_Counters_ProcessorInformation' { [pscustomobject]@{ Name = '_Total'; ProcessorFrequency = 1700; PerformanceLimitFlags = 2 }; break }
                'Win32_PerfFormattedData_PerfOS_Memory' { [pscustomobject]@{ AvailableBytes = 16GB; CommittedBytes = 32GB; CommitLimit = 64GB }; break }
                'Win32_PerfFormattedData_Counters_ThermalZoneInformation' { return @() }
                'MSAcpi_ThermalZoneTemperature' { throw [System.UnauthorizedAccessException]::new('Fixture provider denied') }
                default { throw "Unexpected fixture query: $ClassName" }
            }
        }
    }

    It 'classifies connected and disconnected GPUs independently of expected hardware and DLL availability' {
        $report = & $script:Tool -ProfilePath $script:FixtureProfile
        $report.gpuCapabilities.nvidiaDriverCandidateCount | Should -Be 1
        $report.gpuCapabilities.intelDriverCandidateCount | Should -Be 1
        $report.gpuCapabilities.cudaWorkloadValidation | Should -Be 'Unknown'
        $report.gpuCapabilities.gpuCoreTemperature | Should -Be 'Unknown'
        $disconnected = $report.devices | Where-Object name -eq 'NVIDIA RTX 5060 Ti'
        $disconnected.presence | Should -Be 'Disconnected'
        $disconnected.driver.state | Should -Be 'HistoricalOrDisconnected'
        $audio = $report.devices | Where-Object name -eq 'USB audio'
        $audio.presence | Should -Be 'Unknown'
        $audio.driver.state | Should -Be 'UnknownOrProblem'
        ($report.devices | Where-Object name -eq 'USB camera').categories | Should -Contain 'AudioVideo'
        ($report.devices | Where-Object name -eq 'USB NIC').network.speedBitsPerSecondRaw | Should -Be 2500000000
        ($report.devices | Where-Object name -eq 'Intel Thunderbolt controller').categories | Should -Contain 'ThunderboltUsb4'
        $report.powerObservation.wholePcInputWatts | Should -BeNullOrEmpty
        $report.powerObservation.inputVoltageVolts | Should -BeNullOrEmpty
        $report.statedPowerPath.maximumWatts | Should -Be 100
        $report.buildPolicy.recommendedNativeBuildJobs | Should -Be 4
        $report.buildPolicy.applied | Should -BeFalse
        Should -Invoke Get-CimInstance -Exactly -Times 12 -ParameterFilter { $OperationTimeoutSec -eq 3 }
    }

    It 'exports schema-valid JSON without unique hardware identifiers' {
        $path = Join-Path $TestDrive 'fixture-report.json'
        $json = & $script:Tool -ProfilePath $script:FixtureProfile -AsJson -OutputPath $path
        $json | Test-Json -SchemaFile $script:ReportSchema -ErrorAction Stop | Should -BeTrue
        (Get-Content -LiteralPath $path -Raw).Trim() | Should -Be $json.Trim()
        $json | Should -Not -Match 'PRIVATE_|SECRET|PNPDeviceID|DeviceID|MACAddress|SerialNumber|UUID'
        $json | Should -Match '"vendorId": "10DE"'
        $report = $json | ConvertFrom-Json
        ($report.providers | Where-Object source -eq 'MSAcpi_ThermalZoneTemperature').status | Should -Be 'Unknown'
        ($report.providers | Where-Object source -eq 'MSAcpi_ThermalZoneTemperature').error.exceptionType | Should -Be 'System.UnauthorizedAccessException'
        ($report.providers | Where-Object source -eq 'Win32_PerfFormattedData_Counters_ThermalZoneInformation').status | Should -Be 'SupportedNoInstances'
        ($report.providers | Where-Object source -eq 'Win32_PerfFormattedData_PowerMeterCounter_EnergyMeter').workloadValidation | Should -Be 'Unknown'
        & $script:Tool -ProfilePath $script:FixtureProfile -OutputPath $path -Force | Out-Null
        Get-Content -LiteralPath $path -Raw | Test-Json -SchemaFile $script:ReportSchema -ErrorAction Stop | Should -BeTrue
    }

    It 'never treats a video-only or mismatched-driver GPU as available compute' {
        Mock Get-CimInstance -ParameterFilter { $ClassName -eq 'Win32_VideoController' } {
            @(
                [pscustomobject]@{ Name = 'NVIDIA A2000'; PNPDeviceID = 'PCI\VEN_10DE&DEV_25B6\PRIVATE_GPU_SERIAL'; DriverVersion = 'wrong.version' }
                [pscustomobject]@{ Name = 'Ghost NVIDIA'; PNPDeviceID = 'PCI\VEN_10DE&DEV_FFFF\GHOST_SERIAL_SECRET'; DriverVersion = 'old.version'; Status = 'OK'; ConfigManagerErrorCode = 0 }
            )
        }
        $report = & $script:Tool -ProfilePath $script:FixtureProfile
        $report.gpuCapabilities.nvidiaDriverCandidateCount | Should -Be 0
        ($report.devices | Where-Object name -eq 'NVIDIA A2000').driver.state | Should -Be 'DriverVersionMismatch'
        ($report.devices | Where-Object name -eq 'Ghost NVIDIA').presence | Should -Be 'Unknown'
    }

    It 'keeps expected NVIDIA hardware separate when no GPU is observed and sensors fail' {
        Mock Get-CimInstance -ParameterFilter { $ClassName -in @('Win32_PnPEntity', 'Win32_VideoController', 'Win32_PnPSignedDriver') } { return @() }
        Mock Get-CimInstance -ParameterFilter { $ClassName -eq 'Win32_OperatingSystem' } { throw 'Unavailable OS fixture' }
        $report = & $script:Tool -ProfilePath $script:FixtureProfile
        $report.devices.Count | Should -Be 0
        $report.gpuCapabilities.nvidiaDriverCandidateCount | Should -Be 0
        $report.expectedHardware[0].name | Should -Be 'NVIDIA A2000'
        $report.buildPolicy.availablePhysicalBytesAtCapture | Should -BeNullOrEmpty
        $report.buildPolicy.recommendedNativeBuildJobs | Should -Be 1
        $report.powerObservation.wholePcInputWatts | Should -BeNullOrEmpty
    }

    It 'caps recommendations under observed memory pressure' {
        Mock Get-CimInstance -ParameterFilter { $ClassName -eq 'Win32_OperatingSystem' } {
            [pscustomobject]@{ Caption = 'Fixture'; Version = '10'; BuildNumber = 'fixture'; FreePhysicalMemory = 6GB / 1KB }
        }
        $report = & $script:Tool -ProfilePath $script:FixtureProfile
        $report.buildPolicy.recommendedNativeBuildJobs | Should -Be 1
    }

    It 'keeps build policy unknown when automatic profile discovery finds no machine profile' {
        # Run the real tool with a separate root that has no Config/machine-profiles/<host>.json.
        $isolatedTools = Join-Path $TestDrive 'unprofiled-machine\Tools'
        New-Item -ItemType Directory -Path $isolatedTools -Force | Out-Null
        $isolatedTool = Join-Path $isolatedTools 'Get-PcaiMachineCapabilities.ps1'
        Copy-Item -LiteralPath $script:Tool -Destination $isolatedTool
        $json = & $isolatedTool -AsJson
        $json | Test-Json -SchemaFile $script:ReportSchema -ErrorAction Stop | Should -BeTrue
        $report = $json | ConvertFrom-Json
        $report.profileSource | Should -BeNullOrEmpty
        $report.identity.logicalProcessors | Should -Be 20
        $report.buildPolicy.availablePhysicalBytesAtCapture | Should -Be 16GB
        $report.buildPolicy.configuredMaxNativeBuildJobs | Should -BeNullOrEmpty
        $report.buildPolicy.recommendedNativeBuildJobs | Should -BeNullOrEmpty
        $report.buildPolicy.source | Should -Be 'No explicit build policy'
        $report.buildPolicy.applied | Should -BeFalse
    }

    It 'keeps build policy unknown when the profile has no explicit cap: <Case>' -TestCases @(
        @{ Case = 'MissingPolicy' }, @{ Case = 'MissingCap' }, @{ Case = 'NullCap' }
    ) {
        param($Case)
        $profileFixture = Get-Content -LiteralPath $script:FixtureProfile -Raw | ConvertFrom-Json -AsHashtable
        switch ($Case) {
            'MissingPolicy' { $profileFixture.Remove('buildPolicy') | Out-Null }
            'MissingCap' { $profileFixture.buildPolicy = @{ source = 'Fixture without a cap' } }
            'NullCap' { $profileFixture.buildPolicy.maxNativeBuildJobs = $null }
        }
        $profileFixture | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $script:FixtureProfile
        $json = & $script:Tool -ProfilePath $script:FixtureProfile -AsJson
        $json | Test-Json -SchemaFile $script:ReportSchema -ErrorAction Stop | Should -BeTrue
        $report = $json | ConvertFrom-Json
        $report.profileSource | Should -Be 'fixture-profile.json'
        $report.buildPolicy.configuredMaxNativeBuildJobs | Should -BeNullOrEmpty
        $report.buildPolicy.recommendedNativeBuildJobs | Should -BeNullOrEmpty
        $report.buildPolicy.rule | Should -Match 'recommendation unknown'
        $report.buildPolicy.source | Should -Be 'No explicit build policy'
        $report.buildPolicy.applied | Should -BeFalse
    }

    It 'distinguishes an explicitly unsupported provider from unknown sensor health' {
        Mock Get-CimInstance -ParameterFilter { $ClassName -eq 'Win32_PerfFormattedData_PowerMeterCounter_EnergyMeter' } {
            $providerError = [Microsoft.Management.Infrastructure.CimException]::new('Fixture missing provider class')
            # Preserve the typed exception and attach the provider's status for
            # an offline fixture without deleting or disabling a real provider.
            $providerError | Add-Member -NotePropertyName StatusCode -NotePropertyValue 'InvalidClass' -Force
            throw $providerError
        }
        $report = & $script:Tool -ProfilePath $script:FixtureProfile
        $provider = $report.providers | Where-Object source -eq 'Win32_PerfFormattedData_PowerMeterCounter_EnergyMeter'
        $provider.status | Should -Be 'Unsupported'
        $provider.instanceCount | Should -BeNullOrEmpty
        $provider.error.statusCode | Should -Be 'InvalidClass'
        $provider.workloadValidation | Should -Be 'Unknown'
        $report.powerObservation.wholePcInputWatts | Should -BeNullOrEmpty
    }

    It 'rejects invalid schema, cross-host and unsafe build policies' {
        $path = Join-Path $TestDrive 'bad-profile.json'
        Set-Content -LiteralPath $path -Value '{"schemaVersion":99}'
        { & $script:Tool -ProfilePath $path } | Should -Throw '*schemaVersion*'
        Set-Content -LiteralPath $path -Value '{"schemaVersion":1,"hostName":"other-unrelated-host"}'
        { & $script:Tool -ProfilePath $path } | Should -Throw '*hostName*'
        New-CapabilityFixtureProfile $path -Jobs 65
        { & $script:Tool -ProfilePath $path } | Should -Throw '*maxNativeBuildJobs*'
    }
}

Describe 'Machine capability Windows smoke' -Tag 'Boot', 'Integration' -Skip:(-not $IsWindows) {
    It 'returns bounded live inventory matching the report schema without loading native modules' {
        $json = & $script:Tool -AsJson
        $json | Test-Json -SchemaFile $script:ReportSchema -ErrorAction Stop | Should -BeTrue
        $report = $json | ConvertFrom-Json
        $report.hostName | Should -Be ([environment]::MachineName.ToLowerInvariant())
        $report.queryProvenance.Count | Should -Be 12
        $report.identity.logicalProcessors | Should -BeGreaterThan 0
        if ($null -ne $report.buildPolicy.configuredMaxNativeBuildJobs) {
            $report.buildPolicy.recommendedNativeBuildJobs | Should -BeGreaterOrEqual 1
        }
        else { $report.buildPolicy.recommendedNativeBuildJobs | Should -BeNullOrEmpty }
        $report.powerObservation.inputVoltageVolts | Should -BeNullOrEmpty
        $report.gpuCapabilities.cudaWorkloadValidation | Should -Be 'Unknown'
    }
}
