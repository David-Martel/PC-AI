<#
.SYNOPSIS
    Integration tests for PC_AI diagnostic report generation

.DESCRIPTION
    Tests end-to-end report generation workflow with all modules working together
#>

BeforeAll {
    # Import all modules
    $ModulesPath = Join-Path $PSScriptRoot '..\..\Modules'

    $Modules = @(
        'PC-AI.Hardware'
        'PC-AI.Virtualization'
        'PC-AI.USB'
        'PC-AI.Network'
        'PC-AI.Performance'
        'PC-AI.Cleanup'
        'PC-AI.LLM'
    )

    foreach ($module in $Modules) {
        $modulePath = Join-Path $ModulesPath "$module\$module.psd1"
        if (Test-Path $modulePath) {
            Import-Module $modulePath -Force -ErrorAction Stop
        }
    }

    # Import mock data
    $MockDataPath = Join-Path $PSScriptRoot '..\Fixtures\MockData.psm1'
    Import-Module $MockDataPath -Force -ErrorAction Stop

    # Create test output directory
    $script:TestOutputPath = Join-Path $TestDrive "IntegrationTests"
    New-Item -ItemType Directory -Path $script:TestOutputPath -Force | Out-Null
}

Describe "Full Diagnostic Report Generation" -Tag 'Integration', 'E2E', 'Slow' {
    Context "When generating a complete diagnostic report" {
        BeforeAll {
            # Get-DeviceErrors, Get-UsbStatus and Get-SystemEvents each try a NATIVE
            # probe first and only fall back to CIM/Get-WinEvent. Mocking just the
            # fallback leaves the result dependent on whether some other suite loaded
            # the acceleration modules, which is why this file passes alone and fails
            # in a full run. Force the fallback path deterministically.
            Mock Get-HardwarePnpDevicesNative { $null } -ModuleName PC-AI.Hardware
            Mock Get-HardwareSystemEventsNative { $null } -ModuleName PC-AI.Hardware
            Mock Get-HardwareDiskHealthNative { $null } -ModuleName PC-AI.Hardware

            # Mock all external commands
            Mock Get-CimInstance {
                param($ClassName)
                switch ($ClassName) {
                    'Win32_PnPEntity' { Get-MockDevicesWithErrors }
                    'Win32_NetworkAdapter' { Get-MockNetworkAdapters }
                    default { @() }
                }
            } -ModuleName PC-AI.Hardware

            Mock Invoke-Expression {
                param($Command)
                switch -Wildcard ($Command) {
                    "*wmic diskdrive*" { Get-MockDiskSmartOutput -Health Healthy }
                    "*wsl*" { Get-MockWSLOutput -Command Status }
                    default { "" }
                }
            } -ModuleName PC-AI.Hardware

            Mock Get-WinEvent {
                Get-MockDiskUsbEvents -ErrorType Mixed
            } -ModuleName PC-AI.Hardware

            $script:ReportPath = Join-Path $script:TestOutputPath "DiagnosticReport.txt"
        }

        It "Should generate report without errors" {
            { New-DiagnosticReport -OutputPath $script:ReportPath } | Should -Not -Throw
        }

        It "Should create report file" {
            New-DiagnosticReport -OutputPath $script:ReportPath
            Test-Path $script:ReportPath | Should -Be $true
        }

        It "Should include device error section" {
            New-DiagnosticReport -OutputPath $script:ReportPath
            $content = Get-Content $script:ReportPath -Raw
            $content | Should -Match "Device.*Error|Error.*Device"
        }

        It "Should include disk health section" {
            New-DiagnosticReport -OutputPath $script:ReportPath
            $content = Get-Content $script:ReportPath -Raw
            $content | Should -Match "Disk.*Health|SMART.*Status"
        }

        It "Should include system events section" {
            New-DiagnosticReport -OutputPath $script:ReportPath
            $content = Get-Content $script:ReportPath -Raw
            $content | Should -Match "System.*Event|Event.*Log|Recent.*System|System.*Error"
        }

        It "Should include USB status section" {
            New-DiagnosticReport -OutputPath $script:ReportPath
            $content = Get-Content $script:ReportPath -Raw
            $content | Should -Match "USB"
        }

        It "Should include network adapters section" {
            New-DiagnosticReport -OutputPath $script:ReportPath
            $content = Get-Content $script:ReportPath -Raw
            $content | Should -Match "Network.*Adapter|Adapter.*Status"
        }

        It "Should format report sections with headers" {
            New-DiagnosticReport -OutputPath $script:ReportPath
            $content = Get-Content $script:ReportPath -Raw
            $content | Should -Match "={3,}|#{2,}"  # Section headers
        }

        It "Should include timestamp" {
            New-DiagnosticReport -OutputPath $script:ReportPath
            $content = Get-Content $script:ReportPath -Raw
            $content | Should -Match "\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4}"
        }
    }

    Context "When report generation fails partially" {
        BeforeAll {
            # Get-DeviceErrors, Get-UsbStatus and Get-SystemEvents each try a NATIVE
            # probe first and only fall back to CIM/Get-WinEvent. Mocking just the
            # fallback leaves the result dependent on whether some other suite loaded
            # the acceleration modules, which is why this file passes alone and fails
            # in a full run. Force the fallback path deterministically.
            Mock Get-HardwarePnpDevicesNative { $null } -ModuleName PC-AI.Hardware
            Mock Get-HardwareSystemEventsNative { $null } -ModuleName PC-AI.Hardware
            Mock Get-HardwareDiskHealthNative { $null } -ModuleName PC-AI.Hardware

            Mock Get-CimInstance { Get-MockDevicesHealthy } -ModuleName PC-AI.Hardware
            Mock Invoke-Expression { throw "Command failed" } -ModuleName PC-AI.Hardware
            Mock Get-WinEvent { Get-MockDiskUsbEvents -ErrorType None } -ModuleName PC-AI.Hardware

            $script:PartialReportPath = Join-Path $script:TestOutputPath "PartialReport.txt"
        }

        It "Should generate partial report on non-critical failures" {
            { New-DiagnosticReport -OutputPath $script:PartialReportPath -ErrorAction SilentlyContinue } | Should -Not -Throw
        }

        It "Should include successful sections" {
            New-DiagnosticReport -OutputPath $script:PartialReportPath -ErrorAction SilentlyContinue
            Test-Path $script:PartialReportPath | Should -Be $true
        }

        It "Should note failed sections" {
            New-DiagnosticReport -OutputPath $script:PartialReportPath -ErrorAction SilentlyContinue
            $content = Get-Content $script:PartialReportPath -Raw
            $content | Should -Match "Error|Failed|Unavailable"
        }
    }
}

Describe "Report Analysis with LLM" -Tag 'Integration', 'E2E', 'Slow' {
    Context "When analyzing generated report with LLM" {
        BeforeAll {
            Mock Get-Content {
                @"
PC Diagnostic Report - $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')

=== Device Errors ===
USB Mass Storage Device - Error Code: 43
Intel USB 3.0 Controller - Error Code: 10

=== Disk Health ===
Samsung SSD 980 PRO: OK
WDC HDD: Pred Fail

=== System Events (Last 3 Days) ===
- Disk error: Bad block detected
- USB error: Device descriptor validation failed
"@
            } -ModuleName PC-AI.LLM

            Mock Send-OllamaRequest {
                Get-MockOllamaResponse -Type Success
            } -ModuleName PC-AI.LLM

            # Invoke-PCDiagnosis no longer calls Send-OllamaRequest. It walks the
            # configured provider fallback order, gating each provider on
            # Get-CachedProviderHealth and then calling Invoke-OllamaChat /
            # Invoke-OpenAIChat. Mocking only the old entry point left the health
            # gate to run for real, find nothing listening, and throw "No LLM
            # providers are reachable in the configured fallback order" -- so these
            # tests never reached the mock at all. Mock the current path too.
            Mock Get-CachedProviderHealth { $true } -ModuleName PC-AI.LLM
            Mock Invoke-OllamaChat { (Get-MockOllamaResponse -Type Success).response } -ModuleName PC-AI.LLM
            Mock Invoke-OpenAIChat { (Get-MockOllamaResponse -Type Success).response } -ModuleName PC-AI.LLM

            $script:StubReportPath = Join-Path $script:TestOutputPath 'StubReport.txt'
            # The tests below pass "TestDrive:\report.txt" but nothing ever
            # created it. Mocking Get-Content is not sufficient: -DiagnosticReportPath
            # carries a ValidateScript { Test-Path $_ -PathType Leaf }, and
            # parameter validation runs BEFORE the function body, so the call
            # was rejected at binding time with "the validation script ... did
            # not return a result of True" and never reached the mock at all.
            @'
=== Diagnostic Report (test fixture) ===
- Disk error: Bad block detected
'@ | Set-Content -LiteralPath $script:StubReportPath -Encoding utf8

            $script:AnalysisPath = Join-Path $script:TestOutputPath "Analysis.txt"
        }

        It "Should analyze report without errors" {
            { Invoke-PCDiagnosis -DiagnosticReportPath $script:StubReportPath -OutputPath $script:AnalysisPath } | Should -Not -Throw
        }

        It "Should create analysis output file" {
            Invoke-PCDiagnosis -DiagnosticReportPath $script:StubReportPath -OutputPath $script:AnalysisPath
            Should -Invoke Send-OllamaRequest -ModuleName PC-AI.LLM
        }

        It "Should include diagnostic data in prompt" {
            Invoke-PCDiagnosis -DiagnosticReportPath $script:StubReportPath

            Should -Invoke Send-OllamaRequest -ModuleName PC-AI.LLM -ParameterFilter {
                $Prompt -match "USB.*43|Error Code: 43"
            }
        }

        It "Should use appropriate system prompt" {
            Invoke-PCDiagnosis -DiagnosticReportPath $script:StubReportPath

            Should -Invoke Send-OllamaRequest -ModuleName PC-AI.LLM -ParameterFilter {
                $SystemMessage -match "diagnostic|hardware|analyze"
            }
        }
    }
}

Describe "Multi-Module Workflow" -Tag 'Integration', 'E2E', 'Slow' {
    Context "When executing complete diagnostics-to-analysis workflow" {
        BeforeAll {
            # Get-DeviceErrors, Get-UsbStatus and Get-SystemEvents each try a NATIVE
            # probe first and only fall back to CIM/Get-WinEvent. Mocking just the
            # fallback leaves the result dependent on whether some other suite loaded
            # the acceleration modules, which is why this file passes alone and fails
            # in a full run. Force the fallback path deterministically.
            Mock Get-HardwarePnpDevicesNative { $null } -ModuleName PC-AI.Hardware
            Mock Get-HardwareSystemEventsNative { $null } -ModuleName PC-AI.Hardware
            Mock Get-HardwareDiskHealthNative { $null } -ModuleName PC-AI.Hardware

            # Mock all dependent functions
            Mock Get-CimInstance { Get-MockDevicesWithErrors } -ModuleName PC-AI.Hardware
            Mock Invoke-Expression { Get-MockDiskSmartOutput -Health Warning } -ModuleName PC-AI.Hardware
            Mock Get-WinEvent { Get-MockDiskUsbEvents -ErrorType Mixed } -ModuleName PC-AI.Hardware
            Mock Get-PSDrive { Get-MockDiskSpace -Status LowSpace } -ModuleName PC-AI.Performance
            Mock Send-OllamaRequest { Get-MockOllamaResponse -Type Success } -ModuleName PC-AI.LLM

            # Invoke-PCDiagnosis no longer calls Send-OllamaRequest. It walks the
            # configured provider fallback order, gating each provider on
            # Get-CachedProviderHealth and then calling Invoke-OllamaChat /
            # Invoke-OpenAIChat. Mocking only the old entry point left the health
            # gate to run for real, find nothing listening, and throw "No LLM
            # providers are reachable in the configured fallback order" -- so these
            # tests never reached the mock at all. Mock the current path too.
            Mock Get-CachedProviderHealth { $true } -ModuleName PC-AI.LLM
            Mock Invoke-OllamaChat { (Get-MockOllamaResponse -Type Success).response } -ModuleName PC-AI.LLM
            Mock Invoke-OpenAIChat { (Get-MockOllamaResponse -Type Success).response } -ModuleName PC-AI.LLM

            $script:WorkflowReportPath = Join-Path $script:TestOutputPath "WorkflowReport.txt"
            $script:WorkflowAnalysisPath = Join-Path $script:TestOutputPath "WorkflowAnalysis.txt"
        }

        It "Step 1: Should generate diagnostic report" {
            { New-DiagnosticReport -OutputPath $script:WorkflowReportPath } | Should -Not -Throw
            Test-Path $script:WorkflowReportPath | Should -Be $true
        }

        It "Step 2: Should check disk space" {
            $diskSpace = Get-DiskSpace
            $diskSpace | Should -Not -BeNullOrEmpty
            $diskSpace[0].PercentFree | Should -BeLessThan 15  # Low space
        }

        It "Step 3: Should analyze PATH for duplicates" {
            Mock Get-ItemProperty {
                [PSCustomObject]@{
                    Path = "C:\Windows\System32;C:\Program Files\Git\cmd;C:\Windows\System32"
                }
            } -ModuleName PC-AI.Cleanup

            $pathDupes = Get-PathDuplicates
            $pathDupes | Should -Match "duplicate|C:\\Windows\\System32"
        }

        It "Step 4: Should send report to LLM for analysis" {
            Mock Get-Content {
                Get-Content $script:WorkflowReportPath
            } -ModuleName PC-AI.LLM

            { Invoke-PCDiagnosis -DiagnosticReportPath $script:WorkflowReportPath -OutputPath $script:WorkflowAnalysisPath } | Should -Not -Throw
        }

        It "Step 5: Should have complete workflow outputs" {
            Test-Path $script:WorkflowReportPath | Should -Be $true
            Should -Invoke Send-OllamaRequest -ModuleName PC-AI.LLM
        }
    }
}

Describe "Cross-Module Data Flow" -Tag 'Integration', 'DataFlow' {
    Context "When modules exchange data" {
        BeforeAll {
            # Get-DeviceErrors, Get-UsbStatus and Get-SystemEvents each try a NATIVE
            # probe first and only fall back to CIM/Get-WinEvent. Mocking just the
            # fallback leaves the result dependent on whether some other suite loaded
            # the acceleration modules, which is why this file passes alone and fails
            # in a full run. Force the fallback path deterministically.
            Mock Get-HardwarePnpDevicesNative { $null } -ModuleName PC-AI.Hardware
            Mock Get-HardwareSystemEventsNative { $null } -ModuleName PC-AI.Hardware
            Mock Get-HardwareDiskHealthNative { $null } -ModuleName PC-AI.Hardware

            Mock Get-CimInstance { Get-MockDevicesWithErrors } -ModuleName PC-AI.Hardware
            Mock Invoke-Expression { Get-MockWSLOutput -Command Status } -ModuleName PC-AI.Virtualization
        }

        It "Should pass device data between Hardware and USB modules" {
            $hwDevices = Get-DeviceErrors
            $hwDevices | Should -Not -BeNullOrEmpty

            # USB module should be able to filter this data
            $usbDevices = $hwDevices | Where-Object { $_.DeviceID -match '^USB\\' }
            $usbDevices | Should -Not -BeNullOrEmpty
        }

        It "Should correlate WSL status with network diagnostics" {
            Mock Invoke-Expression {
                param($Command)
                if ($Command -match "wsl") {
                    Get-MockWSLOutput -Command Status
                }
                else {
                    ""
                }
            } -ModuleName PC-AI.Virtualization

            $wslStatus = Get-WSLStatus
            $wslStatus | Should -Not -BeNullOrEmpty
            $wslStatus | Should -Match "WSL|Version"
        }
    }
}

AfterAll {
    # Clean up test output
    if (Test-Path $script:TestOutputPath) {
        Remove-Item $script:TestOutputPath -Recurse -Force -ErrorAction SilentlyContinue
    }

    # Remove modules
    @(
        'PC-AI.Hardware'
        'PC-AI.Virtualization'
        'PC-AI.USB'
        'PC-AI.Network'
        'PC-AI.Performance'
        'PC-AI.Cleanup'
        'PC-AI.LLM'
        'MockData'
    ) | ForEach-Object {
        Remove-Module $_ -Force -ErrorAction SilentlyContinue
    }
}
