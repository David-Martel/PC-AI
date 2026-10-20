#Requires -Version 7.0
#Requires -Modules @{ ModuleName = 'Pester'; ModuleVersion = '5.0.0' }

BeforeAll {
    $script:Collector = Join-Path (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)) 'Tools\Watch-CrashTelemetry.ps1'
}

Describe 'Crash telemetry safety' -Tag 'Boot', 'Unit' {
    It 'parses without syntax errors and imports no PC-AI/native modules' {
        $tokens = $null
        $parseErrors = $null
        $ast = [System.Management.Automation.Language.Parser]::ParseFile($script:Collector, [ref]$tokens, [ref]$parseErrors)
        $parseErrors.Count | Should -Be 0
        $commandNames = @($ast.FindAll({
            param($node)
            $node -is [System.Management.Automation.Language.CommandAst]
        }, $true) | ForEach-Object { $_.GetCommandName() })
        foreach ($command in @('Import-Module', 'Add-Type', 'Register-ScheduledTask', 'Set-ScheduledTask',
                'Unregister-ScheduledTask', 'Write-EventLog', 'Remove-Item', 'Set-ItemProperty', 'Stop-Process')) {
            $commandNames | Should -Not -Contain $command
        }
    }

    It 'returns DryRun without needing an output directory' {
        $result = & $script:Collector -DryRun
        $result.Status | Should -Be 'DryRun'
        $result.WritesPerformed | Should -BeFalse
        $result.SamplingPerformed | Should -BeFalse
    }

    It 'performs zero writes and zero CIM queries during DryRun' -Skip:(-not $IsWindows) {
        Mock -CommandName Get-CimInstance -MockWith { throw 'DryRun must not query providers.' }
        $captureFolder = Join-Path $TestDrive 'dry-run-must-not-exist'
        $before = @(Get-ChildItem -LiteralPath $TestDrive -Force -Recurse)
        $result = & $script:Collector -OutputDirectory $captureFolder -DryRun -Once
        $result.Status | Should -Be 'DryRun'
        Test-Path -LiteralPath $captureFolder | Should -BeFalse
        @(Get-ChildItem -LiteralPath $TestDrive -Force -Recurse).Count | Should -Be $before.Count
        Should -Invoke Get-CimInstance -Times 0 -Exactly
    }

    It 'supports help aliases without creating the capture directory' {
        $captureFolder = Join-Path $TestDrive 'help-must-not-exist'
        foreach ($alias in @('h', '-help')) {
            $parameters = @{ OutputDirectory = $captureFolder; $alias = $true }
            $helpText = & $script:Collector @parameters | Out-String
            $helpText | Should -Match 'freeze and restart'
        }
        Test-Path -LiteralPath $captureFolder | Should -BeFalse
    }

    It 'requires an explicit absolute dedicated output directory for capture' -Skip:(-not $IsWindows) {
        { & $script:Collector -Once } | Should -Throw '*OutputDirectory is required*'
        { & $script:Collector -OutputDirectory '.\relative' -Once } | Should -Throw '*absolute*'
        { & $script:Collector -OutputDirectory 'C:\' -Once } | Should -Throw '*dedicated*'
    }

    It 'rejects reserved device and literal null leaves before writes' -Skip:(-not $IsWindows) {
        foreach ($leaf in @('AUX', 'nul.txt', 'COM1.log', 'LPT9.', 'PRN ', '$null')) {
            $captureFolder = Join-Path $TestDrive $leaf
            { & $script:Collector -OutputDirectory $captureFolder -Once } | Should -Throw '*reserved*'
        }
    }
}

Describe 'Crash telemetry capture' -Tag 'Boot', 'Integration' -Skip:(-not $IsWindows) {
    It 'writes a real sample and a clean stop without importing profile integrations' {
        $captureFolder = Join-Path $TestDrive 'real-capture'
        $result = & $script:Collector -OutputDirectory $captureFolder -Once
        $result.Status | Should -Be 'OnceCompleted'
        $result.SamplesAttempted | Should -Be 1
        $files = @(Get-ChildItem -LiteralPath $captureFolder -File)
        $files.Count | Should -Be 1
        $rawRecords = @(Get-Content -LiteralPath $files[0].FullName)
        $records = @($rawRecords | ConvertFrom-Json)
        $records.Count | Should -Be 3
        $records[0].recordType | Should -Be 'start'
        $records[1].recordType | Should -Be 'sample'
        $records[2].recordType | Should -Be 'stop'
        foreach ($record in $records) { $record.schemaVersion | Should -Be 2 }
        $records[2].reason | Should -Be 'OnceCompleted'
        $sample = $records[1]
        # Newer ConvertFrom-Json versions parse UTC strings into local DateTime.
        # Assert the persisted wire format instead of that automatic conversion.
        $rawRecords[1] | Should -Match '"utc":"[^\"]+\+00:00"'
        { [datetimeoffset]::Parse($sample.local) } | Should -Not -Throw
        $sample.uptimeSeconds | Should -BeGreaterThan 0
        $sample.osMemory.totalPhysicalBytes | Should -BeGreaterThan 0
        $sample.osMemory.availablePhysicalBytes | Should -BeLessOrEqual $sample.osMemory.totalPhysicalBytes
        $sample.memoryCounters.commitLimitBytes | Should -BeGreaterThan 0
        $sample.memoryCounters.committedBytes | Should -BeGreaterThan 0
        $sample.cpu.totalProcessorTimePercent | Should -BeGreaterOrEqual 0
        $sample.topProcesses.Count | Should -BeGreaterThan 0
        $sample.topProcesses.Count | Should -BeLessOrEqual 5
        $sample.topProcesses[0].workingSetBytes | Should -BeGreaterThan 0
        $sample.thermalInterpretation | Should -Match 'not CPU core'
        $sample.energyMeter.scope | Should -Match 'no whole-PC input watts, voltage or USB-PD'
        foreach ($channel in $sample.energyMeter.channels) {
            $channel.name | Should -Not -Be '_Total'
            if ($channel.counters.Power -gt 0 -and $channel.counters.Time -gt 0) {
                $channel.watts | Should -Be ($channel.counters.Power / 1000.0)
            }
            else { $channel.watts | Should -BeNullOrEmpty }
        }
        if ($null -ne $sample.processorInformation) {
            $sample.processorInformation.name | Should -Be '_Total'
            $sample.processorInformation.actualFrequencyMHz | Should -BeGreaterOrEqual 0
            $sample.processorInformation.performanceLimitFlagsRaw | Should -BeGreaterOrEqual 0
        }
        foreach ($zone in $sample.performanceThermalZones) {
            $zone.rawHighPrecisionTemperatureTenthsKelvin | Should -BeGreaterOrEqual 2000
            $zone.rawHighPrecisionTemperatureTenthsKelvin | Should -BeLessOrEqual 4500
            $zone.kind | Should -Match 'not CPU core'
        }
        foreach ($zone in $sample.thermalZones) {
            $zone.rawTenthsKelvin | Should -BeGreaterOrEqual 2000
            $zone.rawTenthsKelvin | Should -BeLessOrEqual 4500
        }
        foreach ($queryError in $sample.errors) {
            $queryError.source | Should -Not -BeNullOrEmpty
            $queryError.exceptionType | Should -Not -BeNullOrEmpty
            $queryError.category | Should -Not -BeNullOrEmpty
        }
        # Appending a second session must preserve prior evidence.
        & $script:Collector -OutputDirectory $captureFolder -Once | Out-Null
        @(Get-Content -LiteralPath $files[0].FullName | ConvertFrom-Json).Count | Should -Be 6
    }

    It 'preserves unavailable provider fields as null while retaining observed zero and byte scaling' {
        Mock -CommandName Get-CimInstance -MockWith {
            param($ClassName)
            switch ($ClassName) {
                'Win32_OperatingSystem' {
                    return [pscustomobject]@{
                        LastBootUpTime = $null; TotalVisibleMemorySize = 0; FreePhysicalMemory = $null
                        TotalVirtualMemorySize = 2; FreeVirtualMemory = $null
                    }
                }
                'Win32_PerfFormattedData_PerfOS_Memory' {
                    return [pscustomobject]@{
                        AvailableBytes = 0; CommittedBytes = $null; CommitLimit = $null
                        PercentCommittedBytesInUse = $null; PagesPersec = $null
                        PageReadsPersec = $null; PageWritesPersec = $null
                    }
                }
                'Win32_PerfFormattedData_PerfOS_Processor' {
                    return [pscustomobject]@{
                        PercentProcessorTime = 0; PercentPrivilegedTime = $null
                        PercentInterruptTime = $null; PercentDPCTime = $null
                    }
                }
                'Win32_PerfFormattedData_Counters_ProcessorInformation' {
                    return [pscustomobject]@{
                        ActualFrequency = $null; ProcessorFrequency = 0; PercentofMaximumFrequency = $null
                        PercentProcessorPerformance = $null; PercentPerformanceLimit = $null
                        PerformanceLimitFlags = $null; ProcessorStateFlags = $null; ParkingStatus = 0
                    }
                }
                default { throw 'Fixture optional provider unavailable' }
            }
        }
        $captureFolder = Join-Path $TestDrive 'nullable-provider-capture'
        $result = & $script:Collector -OutputDirectory $captureFolder -Once
        $result.Status | Should -Be 'OnceCompleted'
        $records = @(Get-Content -LiteralPath $result.LastFile | ConvertFrom-Json)
        $sample = $records | Where-Object recordType -eq 'sample'
        $sample.uptimeSeconds | Should -BeNullOrEmpty
        $sample.osMemory.lastBootUtc | Should -BeNullOrEmpty
        $sample.osMemory.totalPhysicalBytes | Should -BeExactly 0
        $sample.osMemory.totalVirtualBytes | Should -BeExactly 2048
        $sample.osMemory.availablePhysicalBytes | Should -BeNullOrEmpty
        $sample.osMemory.availableVirtualBytes | Should -BeNullOrEmpty
        $sample.memoryCounters.availableBytes | Should -BeExactly 0
        foreach ($name in @('committedBytes', 'commitLimitBytes', 'percentCommittedBytesInUse', 'pagesPerSecond', 'pageReadsPerSecond', 'pageWritesPerSecond')) {
            $sample.memoryCounters.$name | Should -BeNullOrEmpty
        }
        $sample.cpu.totalProcessorTimePercent | Should -BeExactly 0
        foreach ($name in @('privilegedTimePercent', 'interruptTimePercent', 'dpcTimePercent')) {
            $sample.cpu.$name | Should -BeNullOrEmpty
        }
        $sample.processorInformation.processorFrequencyMHz | Should -BeExactly 0
        $sample.processorInformation.parkingStatusRaw | Should -BeExactly 0
        foreach ($name in @('actualFrequencyMHz', 'percentOfMaximumFrequency', 'percentProcessorPerformance', 'percentPerformanceLimit', 'performanceLimitFlagsRaw', 'processorStateFlagsRaw')) {
            $sample.processorInformation.$name | Should -BeNullOrEmpty
        }
        $records[-1].reason | Should -Be 'OnceCompleted'
        @($sample.errors | Where-Object source -in @('Win32_OperatingSystem', 'Win32_PerfFormattedData_PerfOS_Memory',
                'Win32_PerfFormattedData_PerfOS_Processor', 'Win32_PerfFormattedData_Counters_ProcessorInformation')).Count | Should -Be 0
    }

    It 'persists typed CIM failure details while continuing capture' {
        Mock -CommandName Get-CimInstance -MockWith { throw [System.InvalidOperationException]::new('Fixture provider failure') }
        $captureFolder = Join-Path $TestDrive 'provider-failure'
        $result = & $script:Collector -OutputDirectory $captureFolder -Once
        $result.Status | Should -Be 'OnceCompleted'
        $sample = @(Get-Content -LiteralPath $result.LastFile | ConvertFrom-Json)[1]
        $sample.errors.Count | Should -BeGreaterOrEqual 7
        @($sample.errors | Where-Object source -eq 'Win32_OperatingSystem').Count | Should -Be 1
        $sample.errors[0].exceptionType | Should -Be 'System.InvalidOperationException'
        $sample.errors[0].message | Should -Be 'Fixture provider failure'
        $sample.osMemory | Should -BeNullOrEmpty
        $sample.processorInformation | Should -BeNullOrEmpty
        $sample.energyMeter.status | Should -Be 'Unavailable'
        $sample.energyMeter.channels.Count | Should -Be 0
        $sample.performanceThermalZones.Count | Should -Be 0
        foreach ($source in @('Win32_PerfFormattedData_Counters_ProcessorInformation',
                'Win32_PerfFormattedData_PowerMeterCounter_EnergyMeter',
                'Win32_PerfFormattedData_Counters_ThermalZoneInformation')) {
            @($sample.errors | Where-Object source -eq $source).Count | Should -Be 1
        }
        $sample.topProcesses.Count | Should -BeGreaterThan 0
    }

    It 'retains channel units and raw values without summing package subdomains or fabricating zero watts' {
        Mock -CommandName Get-CimInstance -MockWith {
            param($ClassName)
            switch ($ClassName) {
                'Win32_PerfFormattedData_PowerMeterCounter_EnergyMeter' {
                    @(
                        [pscustomobject]@{ Name = 'RAPL_Package0_PKG'; Energy = [ulong]990000000001; Time = 1000; Power = 12500 }
                        [pscustomobject]@{ Name = 'RAPL_Package0_PP0'; Energy = 500; Time = 1000; Power = 6500 }
                        [pscustomobject]@{ Name = 'RAPL_Package0_PP1'; Energy = 500; Time = 1000; Power = 2000 }
                        [pscustomobject]@{ Name = 'RAPL_Package0_DRAM'; Energy = 0; Time = 1000; Power = 0 }
                        [pscustomobject]@{ Name = 'UnknownInterval'; Energy = 500; Time = 0; Power = 2500 }
                        [pscustomobject]@{ Name = '_Total'; Energy = 99999; Time = 1000; Power = 999000 }
                    )
                    break
                }
                'Win32_PerfFormattedData_Counters_ProcessorInformation' {
                    [pscustomobject]@{
                        ActualFrequency = 2400; ProcessorFrequency = 1700
                        PercentofMaximumFrequency = 110; PercentProcessorPerformance = 120
                        PercentPerformanceLimit = 80; PerformanceLimitFlags = 2
                        ProcessorStateFlags = 4; ParkingStatus = 0
                    }
                    break
                }
                'Win32_PerfFormattedData_Counters_ThermalZoneInformation' {
                    @(
                        [pscustomobject]@{ Name = 'PerfValid'; HighPrecisionTemperature = 3272; Temperature = 327; PercentPassiveLimit = 80; ThrottleReasons = 2 }
                        [pscustomobject]@{ Name = 'PerfInvalid'; HighPrecisionTemperature = 100; Temperature = 10; PercentPassiveLimit = 100; ThrottleReasons = 0 }
                    )
                    break
                }
                'MSAcpi_ThermalZoneTemperature' {
                    @(
                        [pscustomobject]@{ InstanceName = 'AcpiValid'; CurrentTemperature = 3200 }
                        [pscustomobject]@{ InstanceName = 'AcpiInvalid'; CurrentTemperature = 0 }
                    )
                    break
                }
                default { throw 'Unneeded provider in power fixture' }
            }
        }
        $captureFolder = Join-Path $TestDrive 'power-fixture'
        $result = & $script:Collector -OutputDirectory $captureFolder -Once
        $result.Status | Should -Be 'OnceCompleted'
        $sample = @(Get-Content -LiteralPath $result.LastFile | ConvertFrom-Json)[1]
        $power = $sample.energyMeter
        $power.status | Should -Be 'Available'
        $power.units.Energy | Should -Be 'picowatt-hours'
        $power.units.Time | Should -Be 'milliseconds'
        $power.units.Power | Should -Be 'milliwatts'
        $power.unitProvenance.stringResourceIds.Power | Should -Be 27
        $power.aggregation | Should -Match 'never sum channels or use _Total'
        $power.scope | Should -Match 'no whole-PC input watts, voltage or USB-PD'
        $power.channels.Count | Should -Be 5
        @($power.channels | Where-Object name -eq '_Total').Count | Should -Be 0
        $pkg = $power.channels | Where-Object name -eq 'RAPL_Package0_PKG'
        $pkg.watts | Should -Be 12.5
        $pkg.counters.Energy | Should -Be 990000000001
        $pkg.counters.Power | Should -Be 12500
        $pkg.domain | Should -Match 'includes overlapping PP0 and PP1'
        $dram = $power.channels | Where-Object name -eq 'RAPL_Package0_DRAM'
        $dram.counters.Power | Should -Be 0
        $dram.watts | Should -BeNullOrEmpty
        $dram.readingStatus | Should -Be 'ZeroOrUnsupported'
        $unknown = $power.channels | Where-Object name -eq 'UnknownInterval'
        $unknown.watts | Should -BeNullOrEmpty
        $unknown.readingStatus | Should -Be 'UnknownMeterInterval'
        $sample.processorInformation.actualFrequencyMHz | Should -Be 2400
        $sample.processorInformation.processorFrequencyMHz | Should -Be 1700
        $sample.processorInformation.percentOfMaximumFrequency | Should -Be 110
        $sample.processorInformation.percentProcessorPerformance | Should -Be 120
        $sample.processorInformation.percentPerformanceLimit | Should -Be 80
        $sample.processorInformation.performanceLimitFlagsRaw | Should -Be 2
        $sample.processorInformation.processorStateFlagsRaw | Should -Be 4
        $sample.performanceThermalZones.Count | Should -Be 1
        $sample.performanceThermalZones[0].source | Should -Be 'Win32_PerfFormattedData_Counters_ThermalZoneInformation'
        $sample.performanceThermalZones[0].celsius | Should -Be 54.05
        $sample.performanceThermalZones[0].throttleReasonsRaw | Should -Be 2
        $sample.rejectedPerformanceThermalReadings | Should -Be 1
        $sample.thermalZones.Count | Should -Be 1
        $sample.thermalZones[0].source | Should -Be 'MSAcpi_ThermalZoneTemperature'
        $sample.thermalZones[0].celsius | Should -Be 46.85
        $sample.rejectedThermalReadings | Should -Be 1
        Should -Invoke Get-CimInstance -Times 1 -Exactly -ParameterFilter {
            $ClassName -eq 'Win32_PerfFormattedData_Counters_ProcessorInformation' -and
            $Filter -eq "Name='_Total'" -and $OperationTimeoutSec -eq 3
        }
        foreach ($source in @('Win32_PerfFormattedData_PowerMeterCounter_EnergyMeter',
                'Win32_PerfFormattedData_Counters_ThermalZoneInformation')) {
            Should -Invoke Get-CimInstance -Times 1 -Exactly -ParameterFilter {
                $ClassName -eq $source -and $OperationTimeoutSec -eq 3
            }
        }
    }

    It 'preserves null and omitted energy meter fields separately from observed zero and full unsigned counters' {
        Mock Get-CimInstance {
            param($ClassName)
            if ($ClassName -eq 'Win32_PerfFormattedData_PowerMeterCounter_EnergyMeter') {
                return @(
                    [pscustomobject]@{ Name = 'AllNull'; Energy = $null; Time = $null; Power = $null }
                    [pscustomobject]@{ Name = 'AllOmitted' }
                    [pscustomobject]@{ Name = 'NullPower'; Energy = 0; Time = 1000; Power = $null }
                    [pscustomobject]@{ Name = 'OmittedPower'; Energy = 5; Time = 1000 }
                    [pscustomobject]@{ Name = 'NullTime'; Energy = 10; Time = $null; Power = 2500 }
                    [pscustomobject]@{ Name = 'OmittedTime'; Energy = 10; Power = 2500 }
                    [pscustomobject]@{ Name = 'NullEnergy'; Energy = $null; Time = 1000; Power = 6000 }
                    [pscustomobject]@{ Name = 'OmittedEnergy'; Time = 1000; Power = 5000 }
                    [pscustomobject]@{ Name = 'ObservedZero'; Energy = 0; Time = 0; Power = 0 }
                    [pscustomobject]@{ Name = 'FullUnsignedEnergy'; Energy = [ulong]::MaxValue; Time = 1000; Power = 12500 }
                )
            }
            throw 'Fixture optional provider unavailable'
        }
        $result = & $script:Collector -OutputDirectory (Join-Path $TestDrive 'nullable-energy-meter') -Once
        $result.Status | Should -Be 'OnceCompleted'
        $sample = @(Get-Content -LiteralPath $result.LastFile | ConvertFrom-Json)[1]
        $sample.energyMeter.status | Should -Be 'Available'
        $sample.energyMeter.channels.Count | Should -Be 10
        @($sample.errors | Where-Object source -eq 'Win32_PerfFormattedData_PowerMeterCounter_EnergyMeter').Count | Should -Be 0
        foreach ($name in @('AllNull', 'AllOmitted')) {
            $channel = $sample.energyMeter.channels | Where-Object name -eq $name
            foreach ($counter in @('Energy', 'Time', 'Power')) { $channel.counters.$counter | Should -BeNullOrEmpty }
            $channel.readingStatus | Should -Be 'UnknownPowerCounter'
            $channel.watts | Should -BeNullOrEmpty
        }
        foreach ($name in @('NullPower', 'OmittedPower')) {
            $channel = $sample.energyMeter.channels | Where-Object name -eq $name
            $channel.counters.Power | Should -BeNullOrEmpty
            $channel.counters.Time | Should -BeExactly 1000
            $channel.readingStatus | Should -Be 'UnknownPowerCounter'
            $channel.watts | Should -BeNullOrEmpty
        }
        foreach ($name in @('NullTime', 'OmittedTime')) {
            $channel = $sample.energyMeter.channels | Where-Object name -eq $name
            $channel.counters.Time | Should -BeNullOrEmpty
            $channel.counters.Power | Should -BeExactly 2500
            $channel.readingStatus | Should -Be 'UnknownMeterInterval'
            $channel.watts | Should -BeNullOrEmpty
        }
        foreach ($name in @('NullEnergy', 'OmittedEnergy')) {
            $channel = $sample.energyMeter.channels | Where-Object name -eq $name
            $channel.counters.Energy | Should -BeNullOrEmpty
            $channel.readingStatus | Should -Be 'MeasuredChannelAverage'
            $channel.watts | Should -Be ($channel.counters.Power / 1000.0)
        }
        $zero = $sample.energyMeter.channels | Where-Object name -eq 'ObservedZero'
        foreach ($counter in @('Energy', 'Time', 'Power')) { $zero.counters.$counter | Should -BeExactly 0 }
        $zero.readingStatus | Should -Be 'ZeroOrUnsupported'
        $zero.watts | Should -BeNullOrEmpty
        $unsigned = $sample.energyMeter.channels | Where-Object name -eq 'FullUnsignedEnergy'
        [string]$unsigned.counters.Energy | Should -Be ([ulong]::MaxValue).ToString()
        $unsigned.watts | Should -Be 12.5
    }

    It 'reports empty meter instances as unavailable channels rather than zero power' {
        Mock -CommandName Get-CimInstance -MockWith {
            param($ClassName)
            if ($ClassName -eq 'Win32_PerfFormattedData_PowerMeterCounter_EnergyMeter') { return @() }
            throw 'Fixture provider unavailable'
        }
        $captureFolder = Join-Path $TestDrive 'empty-meters'
        $result = & $script:Collector -OutputDirectory $captureFolder -Once
        $result.Status | Should -Be 'OnceCompleted'
        $sample = @(Get-Content -LiteralPath $result.LastFile | ConvertFrom-Json)[1]
        $sample.energyMeter.status | Should -Be 'NoChannelInstances'
        $sample.energyMeter.channels.Count | Should -Be 0
        $sample.processorInformation | Should -BeNullOrEmpty
        $sample.performanceThermalZones.Count | Should -Be 0
    }

    It 'counts existing logs toward its aggregate daily storage cap without deleting evidence' {
        $captureFolder = Join-Path $TestDrive 'full-day'
        [System.IO.Directory]::CreateDirectory($captureFolder) | Out-Null
        $day = [datetime]::UtcNow.ToString('yyyy-MM-dd')
        $fullLog = Join-Path $captureFolder "crash-$day-01.jsonl"
        [System.IO.File]::WriteAllBytes($fullLog, [byte[]]::new(65536))
        Mock -CommandName Get-CimInstance -MockWith { throw 'No sampling at full storage cap.' }
        $result = & $script:Collector -OutputDirectory $captureFolder -MaxDailyBytes 65536 -Once
        $result.Status | Should -Be 'DailyStorageLimit'
        (Get-Item -LiteralPath $fullLog).Length | Should -Be 65536
        @(Get-ChildItem -LiteralPath $captureFolder -File).Count | Should -Be 1
        Should -Invoke Get-CimInstance -Times 0 -Exactly
    }

    It 'rotates into a second segment instead of exceeding the first segment cap' {
        $captureFolder = Join-Path $TestDrive 'real-rotation'
        [System.IO.Directory]::CreateDirectory($captureFolder) | Out-Null
        $day = [datetime]::UtcNow.ToString('yyyy-MM-dd')
        $firstLog = Join-Path $captureFolder "crash-$day-01.jsonl"
        [System.IO.File]::WriteAllBytes($firstLog, [byte[]]::new(61000))
        $result = & $script:Collector -OutputDirectory $captureFolder -SegmentBytes 65536 -Once
        $result.Status | Should -Be 'OnceCompleted'
        $files = @(Get-ChildItem -LiteralPath $captureFolder -File | Sort-Object Name)
        $files.Count | Should -Be 2
        foreach ($file in $files) { $file.Length | Should -BeLessOrEqual 65536 }
        $secondRecords = @(Get-Content -LiteralPath $files[1].FullName | ConvertFrom-Json)
        $secondRecords[0].recordType | Should -Be 'sample'
        $secondRecords[-1].recordType | Should -Be 'stop'
    }

    It 'stops at two segments while preserving the stop marker' {
        $captureFolder = Join-Path $TestDrive 'segment-limit'
        [System.IO.Directory]::CreateDirectory($captureFolder) | Out-Null
        $day = [datetime]::UtcNow.ToString('yyyy-MM-dd')
        foreach ($number in @(1, 2)) {
            $log = Join-Path $captureFolder ('crash-{0}-{1:D2}.jsonl' -f $day, $number)
            [System.IO.File]::WriteAllBytes($log, [byte[]]::new(62000))
        }
        $result = & $script:Collector -OutputDirectory $captureFolder -SegmentBytes 65536 -Once
        $result.Status | Should -Be 'DailyStorageLimit'
        $files = @(Get-ChildItem -LiteralPath $captureFolder -File)
        $files.Count | Should -Be 2
        foreach ($file in $files) { $file.Length | Should -BeLessOrEqual 65536 }
        # The fixture is pre-filled with NUL bytes; only appended lines are JSON.
        $text = [System.IO.File]::ReadAllText(($files | Sort-Object Name)[1].FullName).TrimStart([char]0)
        $records = @($text -split '\r?\n' | Where-Object { $_ } | ConvertFrom-Json)
        $records[-1].recordType | Should -Be 'stop'
        $records[-1].reason | Should -Be 'DailyStorageLimit'
    }

    It 'prevents another process from capturing the same directory' {
        $captureFolder = Join-Path $TestDrive 'duplicate-must-not-exist'
        $readyFile = Join-Path $TestDrive 'mutex-ready'
        $holderScript = Join-Path $TestDrive 'Hold-Mutex.ps1'
        [System.IO.File]::WriteAllText($holderScript, @'
param([string]$CaptureFolder, [string]$ReadyFile)
$hash = [System.Security.Cryptography.SHA256]::Create()
try {
    $normalized = [System.IO.Path]::GetFullPath($CaptureFolder).TrimEnd('\', '/').ToUpperInvariant()
    $hex = [System.BitConverter]::ToString($hash.ComputeHash([System.Text.Encoding]::UTF8.GetBytes($normalized))).Replace('-', '')
}
finally { $hash.Dispose() }
$mutex = [System.Threading.Mutex]::new($false, ('Global\PC-AI-CrashTelemetry-{0}' -f $hex))
try {
    if (-not $mutex.WaitOne(0)) { throw 'Fixture mutex not acquired' }
    [System.IO.File]::WriteAllText($ReadyFile, 'ready')
    Start-Sleep -Seconds 30
}
finally { $mutex.ReleaseMutex(); $mutex.Dispose() }
'@)
        $childArguments = '-NoLogo -NoProfile -File "{0}" -CaptureFolder "{1}" -ReadyFile "{2}"' -f $holderScript, $captureFolder, $readyFile
        $child = Start-Process -FilePath (Join-Path $PSHOME 'pwsh.exe') -ArgumentList $childArguments -WindowStyle Hidden -PassThru
        try {
            $wait = [System.Diagnostics.Stopwatch]::StartNew()
            while (-not [System.IO.File]::Exists($readyFile) -and $wait.Elapsed.TotalSeconds -lt 15 -and -not $child.HasExited) {
                Start-Sleep -Milliseconds 100
            }
            [System.IO.File]::Exists($readyFile) | Should -BeTrue
            $result = & $script:Collector -OutputDirectory $captureFolder -Once
            $result.Status | Should -Be 'AlreadyRunning'
            Test-Path -LiteralPath $captureFolder | Should -BeFalse
        }
        finally {
            if (-not $child.HasExited) { $child.Kill(); $child.WaitForExit() }
            $child.Dispose()
        }
    }
}
