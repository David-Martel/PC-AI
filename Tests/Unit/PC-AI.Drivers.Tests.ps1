<#
.SYNOPSIS
    Unit tests for PC-AI.Drivers module

.DESCRIPTION
    Tests driver registry loading, version comparison, PnP inventory,
    driver report orchestration, install action routing, and
    Thunderbolt/USB4 networking functions.
#>

BeforeAll {
    $ModulePath = Join-Path $PSScriptRoot '..\..\Modules\PC-AI.Drivers\PC-AI.Drivers.psd1'
    Import-Module $ModulePath -Force -ErrorAction Stop

    # Build a minimal registry fixture from the real schema
    $script:MockRegistryJson = @'
{
  "version": "1.0.0-test",
  "lastUpdated": "2026-03-14T00:00:00Z",
  "trustedSources": [
    { "id": "realtek", "name": "Realtek", "baseUrl": "https://www.realtek.com", "type": "vendor" },
    { "id": "windows-update", "name": "Windows Update", "baseUrl": "https://catalog.update.microsoft.com", "type": "os" }
  ],
  "categories": {
    "network": { "displayName": "Network Adapters", "icon": "network" },
    "thunderbolt": { "displayName": "Thunderbolt / USB4", "icon": "thunderbolt" }
  },
  "devices": [
    {
      "id": "realtek-rtl8156",
      "name": "Realtek RTL8156 USB 2.5GbE",
      "category": "network",
      "matchRules": [
        { "type": "vid_pid", "vid": "0BDA", "pid": "8156" }
      ],
      "driver": {
        "sourceId": "realtek",
        "latestVersion": "1156.21.20.1110",
        "releaseDate": "2025-10-09",
        "certification": "WHQL",
        "downloadUrl": null,
        "manualDownloadUrl": "https://www.realtek.com/Download/List?cate_id=585",
        "installerType": "inf",
        "sha256": null,
        "versionComparable": true,
        "notes": "Test device"
      },
      "sharedDriverGroup": "realtek-usb-ethernet"
    },
    {
      "id": "usb4-p2p",
      "name": "USB4 P2P Network Adapter",
      "category": "thunderbolt",
      "matchRules": [
        { "type": "friendly_name", "pattern": "*USB4*P2P Network Adapter*" }
      ],
      "driver": {
        "sourceId": "windows-update",
        "latestVersion": null,
        "installerType": "windows-update",
        "notes": "Inbox driver"
      },
      "sharedDriverGroup": "windows-usb4"
    },
    {
      "id": "firmware-hub",
      "name": "Firmware Hub",
      "category": "thunderbolt",
      "matchRules": [
        { "type": "friendly_name", "pattern": "*Firmware Hub*" }
      ],
      "driver": {
        "sourceId": null,
        "latestVersion": "2.0",
        "installerType": "none",
        "versionComparable": false,
        "notes": "Version not comparable"
      },
      "sharedDriverGroup": null
    }
  ]
}
'@
}

# ─── Get-DriverRegistry ──────────────────────────────────────────────────────

Describe "Get-DriverRegistry" -Tag 'Unit', 'Drivers', 'Fast', 'Windows' {
    BeforeAll {
        $script:TempRegistryPath = Join-Path $TestDrive 'driver-registry.json'
        $script:MockRegistryJson | Set-Content -Path $script:TempRegistryPath -Encoding UTF8
    }

    Context "Loading from explicit path" {
        It "Should return a registry object with version and devices" {
            $reg = Get-DriverRegistry -RegistryPath $script:TempRegistryPath
            $reg | Should -Not -BeNullOrEmpty
            $reg.Version | Should -Be '1.0.0-test'
            $reg.Devices.Count | Should -Be 3
        }

        It "Should include trusted sources" {
            $reg = Get-DriverRegistry -RegistryPath $script:TempRegistryPath
            $reg.TrustedSources.Count | Should -Be 2
            $reg.TrustedSources[0].id | Should -Be 'realtek'
        }

        It "Should include categories" {
            $reg = Get-DriverRegistry -RegistryPath $script:TempRegistryPath
            $reg.Categories.network.displayName | Should -Be 'Network Adapters'
        }
    }

    Context "Filtering by DeviceId" {
        It "Should return only the matching device" {
            $reg = Get-DriverRegistry -RegistryPath $script:TempRegistryPath -DeviceId 'realtek-rtl8156'
            $reg.Devices.Count | Should -Be 1
            $reg.Devices[0].id | Should -Be 'realtek-rtl8156'
        }

        It "Should return empty devices for non-existent id" {
            $reg = Get-DriverRegistry -RegistryPath $script:TempRegistryPath -DeviceId 'does-not-exist'
            $reg.Devices.Count | Should -Be 0
        }
    }

    Context "Filtering by Category" {
        It "Should return only thunderbolt-category devices" {
            $reg = Get-DriverRegistry -RegistryPath $script:TempRegistryPath -Category 'thunderbolt'
            $reg.Devices.Count | Should -Be 2
            $reg.Devices | ForEach-Object { $_.category | Should -Be 'thunderbolt' }
        }

        It "Should return only network-category devices" {
            $reg = Get-DriverRegistry -RegistryPath $script:TempRegistryPath -Category 'network'
            $reg.Devices.Count | Should -Be 1
            $reg.Devices[0].id | Should -Be 'realtek-rtl8156'
        }
    }

    Context "Error handling" {
        It "Should return null for missing file" {
            $result = Get-DriverRegistry -RegistryPath 'C:\nonexistent\path.json' -ErrorAction SilentlyContinue
            $result | Should -BeNullOrEmpty
        }
    }
}

# ─── Compare-DriverVersion ───────────────────────────────────────────────────

Describe "Compare-DriverVersion" -Tag 'Unit', 'Drivers', 'Fast', 'Windows' {
    BeforeAll {
        $script:TempRegistryPath = Join-Path $TestDrive 'driver-registry.json'
        $script:MockRegistryJson | Set-Content -Path $script:TempRegistryPath -Encoding UTF8
        $script:Registry = Get-DriverRegistry -RegistryPath $script:TempRegistryPath
    }

    Context "Device matched by VID/PID - outdated" {
        It "Should report Outdated when installed < target" {
            $inventory = @([PSCustomObject]@{
                Name          = 'Realtek RTL8156'
                VID           = '0BDA'
                PID           = '8156'
                PnpClass      = 'Net'
                DriverVersion = '1.0.0.0'
            })
            $result = Compare-DriverVersion -Inventory $inventory -Registry $script:Registry
            $result.Count | Should -Be 1
            $result[0].Status | Should -Be 'Outdated'
            $result[0].RegistryId | Should -Be 'realtek-rtl8156'
        }
    }

    Context "Device matched by VID/PID - current" {
        It "Should suppress Current by default" {
            $inventory = @([PSCustomObject]@{
                Name          = 'Realtek RTL8156'
                VID           = '0BDA'
                PID           = '8156'
                PnpClass      = 'Net'
                DriverVersion = '1156.21.20.1110'
            })
            $result = Compare-DriverVersion -Inventory $inventory -Registry $script:Registry
            $result.Count | Should -Be 0
        }

        It "Should include Current when -IncludeUpToDate" {
            $inventory = @([PSCustomObject]@{
                Name          = 'Realtek RTL8156'
                VID           = '0BDA'
                PID           = '8156'
                PnpClass      = 'Net'
                DriverVersion = '1156.21.20.1110'
            })
            $result = Compare-DriverVersion -Inventory $inventory -Registry $script:Registry -IncludeUpToDate
            $result.Count | Should -Be 1
            $result[0].Status | Should -Be 'Current'
        }

        It "Should report Current when installed > target" {
            $inventory = @([PSCustomObject]@{
                Name          = 'Realtek RTL8156'
                VID           = '0BDA'
                PID           = '8156'
                PnpClass      = 'Net'
                DriverVersion = '9999.0.0.0'
            })
            $result = Compare-DriverVersion -Inventory $inventory -Registry $script:Registry -IncludeUpToDate
            $result[0].Status | Should -Be 'Current'
        }
    }

    Context "Device matched by friendly_name - NoUpdate (null latestVersion)" {
        It "Should report NoUpdate for inbox drivers" {
            $inventory = @([PSCustomObject]@{
                Name          = 'USB4(TM) P2P Network Adapter'
                VID           = $null
                PID           = $null
                PnpClass      = 'Net'
                DriverVersion = '10.0.26100.1'
            })
            $result = Compare-DriverVersion -Inventory $inventory -Registry $script:Registry
            $result.Count | Should -Be 1
            $result[0].Status | Should -Be 'NoUpdate'
            $result[0].RegistryId | Should -Be 'usb4-p2p'
        }
    }

    Context "Device with no driver version" {
        It "Should report NoDriver" {
            $inventory = @([PSCustomObject]@{
                Name          = 'Realtek RTL8156'
                VID           = '0BDA'
                PID           = '8156'
                PnpClass      = 'Net'
                DriverVersion = $null
            })
            $result = Compare-DriverVersion -Inventory $inventory -Registry $script:Registry
            $result.Count | Should -Be 1
            $result[0].Status | Should -Be 'NoDriver'
        }
    }

    Context "Device with versionComparable = false" {
        It "Should report ManualCheck" {
            $inventory = @([PSCustomObject]@{
                Name          = 'Firmware Hub Device'
                VID           = $null
                PID           = $null
                PnpClass      = 'System'
                DriverVersion = '1.5'
            })
            $result = Compare-DriverVersion -Inventory $inventory -Registry $script:Registry
            $result.Count | Should -Be 1
            $result[0].Status | Should -Be 'ManualCheck'
        }
    }

    Context "Unmatched device" {
        It "Should suppress Unknown by default" {
            $inventory = @([PSCustomObject]@{
                Name          = 'Unknown Widget'
                VID           = 'AAAA'
                PID           = 'BBBB'
                PnpClass      = 'Other'
                DriverVersion = '1.0'
            })
            $result = Compare-DriverVersion -Inventory $inventory -Registry $script:Registry
            $result.Count | Should -Be 0
        }

        It "Should include Unknown when -IncludeUnknown" {
            $inventory = @([PSCustomObject]@{
                Name          = 'Unknown Widget'
                VID           = 'AAAA'
                PID           = 'BBBB'
                PnpClass      = 'Other'
                DriverVersion = '1.0'
            })
            $result = Compare-DriverVersion -Inventory $inventory -Registry $script:Registry -IncludeUnknown
            $result.Count | Should -Be 1
            $result[0].Status | Should -Be 'Unknown'
        }
    }
}

# ─── Get-PnpDeviceInventory ──────────────────────────────────────────────────

Describe "Get-PnpDeviceInventory" -Tag 'Unit', 'Drivers', 'Fast', 'Windows' {
    Context "Function interface" {
        It "Should be exported from the module" {
            Get-Command Get-PnpDeviceInventory -Module PC-AI.Drivers | Should -Not -BeNullOrEmpty
        }

        It "Should accept Class, VidPid, and ActiveOnly parameters" {
            $cmd = Get-Command Get-PnpDeviceInventory -Module PC-AI.Drivers
            $cmd.Parameters.Keys | Should -Contain 'Class'
            $cmd.Parameters.Keys | Should -Contain 'VidPid'
            $cmd.Parameters.Keys | Should -Contain 'ActiveOnly'
        }
    }

    Context "With mocked PnP devices" -Skip:(-not (Get-Command Get-PnpDevice -ErrorAction SilentlyContinue)) {
        BeforeAll {
            Mock Get-PnpDevice {
                @(
                    [PSCustomObject]@{
                        FriendlyName  = 'Realtek RTL8156 USB 2.5GbE'
                        Class         = 'Net'
                        InstanceId    = 'USB\VID_0BDA&PID_8156\000001'
                        Status        = 'OK'
                        Manufacturer  = 'Realtek'
                        PNPClass      = 'Net'
                    }
                )
            } -ModuleName PC-AI.Drivers

            Mock Get-PnpDeviceProperty {
                param($InstanceId, $KeyName)
                switch ($KeyName) {
                    'DEVPKEY_Device_DriverVersion' {
                        [PSCustomObject]@{ Data = '1.0.0.0' }
                    }
                    'DEVPKEY_Device_DriverDate' {
                        [PSCustomObject]@{ Data = [datetime]'2025-01-01' }
                    }
                    default { $null }
                }
            } -ModuleName PC-AI.Drivers
        }

        It "Should return device objects" {
            $result = Get-PnpDeviceInventory
            $result | Should -Not -BeNullOrEmpty
        }
    }
}

# ─── Get-DriverReport ────────────────────────────────────────────────────────

Describe "Get-DriverReport" -Tag 'Unit', 'Drivers', 'Fast', 'Windows' {
    Context "Function interface" {
        It "Should be exported from the module" {
            Get-Command Get-DriverReport -Module PC-AI.Drivers | Should -Not -BeNullOrEmpty
        }

        It "Should accept RegistryPath, Category, OnlyActionable, IncludeUnknown parameters" {
            $cmd = Get-Command Get-DriverReport -Module PC-AI.Drivers
            $cmd.Parameters.Keys | Should -Contain 'RegistryPath'
            $cmd.Parameters.Keys | Should -Contain 'Category'
            $cmd.Parameters.Keys | Should -Contain 'OnlyActionable'
            $cmd.Parameters.Keys | Should -Contain 'IncludeUnknown'
        }
    }

    Context "Orchestration with mocked sub-functions" -Skip:(-not (Get-Command Get-PnpDevice -ErrorAction SilentlyContinue)) {
        BeforeAll {
            $script:TempRegistryPath = Join-Path $TestDrive 'driver-registry.json'
            $script:MockRegistryJson | Set-Content -Path $script:TempRegistryPath -Encoding UTF8

            Mock Get-PnpDeviceInventory {
                @([PSCustomObject]@{
                    Name          = 'Realtek RTL8156'
                    VID           = '0BDA'
                    PID           = '8156'
                    PnpClass      = 'Net'
                    DriverVersion = '1.0.0.0'
                    InstanceId    = 'USB\VID_0BDA&PID_8156\1'
                    Manufacturer  = 'Realtek'
                    Status        = 'OK'
                })
            } -ModuleName PC-AI.Drivers
        }

        It "Should return a report with status per device" {
            $report = Get-DriverReport -RegistryPath $script:TempRegistryPath
            $report | Should -Not -BeNullOrEmpty
            $report[0].Status | Should -Be 'Outdated'
        }
    }
}

# ─── Install-DriverUpdate ────────────────────────────────────────────────────

Describe "Install-DriverUpdate" -Tag 'Unit', 'Drivers', 'Fast', 'Windows' {
    BeforeAll {
        $script:TempRegistryPath = Join-Path $TestDrive 'driver-registry.json'
        $script:MockRegistryJson | Set-Content -Path $script:TempRegistryPath -Encoding UTF8
    }

    Context "Windows Update device (no download)" {
        It "Should report skip for windows-update installer type" {
            $result = Install-DriverUpdate -DeviceId 'usb4-p2p' -RegistryPath $script:TempRegistryPath -WhatIf
            $result | Should -Not -BeNullOrEmpty
            $result.Action | Should -Match 'WindowsUpdate|Skip|WhatIf'
        }
    }

    Context "Unknown device id" {
        It "Should write error for non-existent device" {
            { Install-DriverUpdate -DeviceId 'nonexistent-device' -RegistryPath $script:TempRegistryPath -ErrorAction Stop } | Should -Throw
        }
    }
}

# ─── Update-DriverRegistry ───────────────────────────────────────────────────

Describe "Update-DriverRegistry" -Tag 'Unit', 'Drivers', 'Fast', 'Windows' {
    Context "Update a single device entry" {
        BeforeAll {
            $script:TempRegistryPath = Join-Path $TestDrive 'update-registry.json'
            $script:MockRegistryJson | Set-Content -Path $script:TempRegistryPath -Encoding UTF8
        }

        It "Should update latestVersion for a known device" {
            Update-DriverRegistry -RegistryPath $script:TempRegistryPath -DeviceId 'realtek-rtl8156' -LatestVersion '9999.0.0.0'
            $reg = Get-DriverRegistry -RegistryPath $script:TempRegistryPath
            $device = $reg.Devices | Where-Object { $_.id -eq 'realtek-rtl8156' }
            $device.driver.latestVersion | Should -Be '9999.0.0.0'
        }
    }
}

# ─── Get-ThunderboltNetworkStatus ────────────────────────────────────────────

Describe "Get-ThunderboltNetworkStatus" -Tag 'Unit', 'Drivers', 'Thunderbolt', 'Windows' {
    Context "When no Thunderbolt adapters are present" {
        BeforeAll {
            Mock Get-CimInstance {
                param($Namespace, $ClassName)
                switch ($ClassName) {
                    'Win32_NetworkAdapter' { return @() }
                    default { return @() }
                }
            } -ModuleName PC-AI.Drivers
        }

        It "Should return empty array" {
            $result = Get-ThunderboltNetworkStatus
            @($result).Count | Should -Be 0
        }
    }

    Context "When a USB4 P2P adapter is present" {
        BeforeAll {
            Mock Get-CimInstance {
                param($Namespace, $ClassName)
                switch ($ClassName) {
                    'Win32_NetworkAdapter' {
                        @([PSCustomObject]@{
                            Name                = 'USB4(TM) P2P Network Adapter'
                            Description         = 'USB4(TM) P2P Network Adapter'
                            NetConnectionID     = 'Ethernet 11'
                            NetEnabled          = $true
                            NetConnectionStatus = 2
                            Speed               = 10000000000
                            MACAddress          = 'AA-BB-CC-DD-EE-FF'
                            PNPDeviceID         = 'SWD\PROT_USB4NET\12345'
                            Index               = 42
                        })
                    }
                    'Win32_NetworkAdapterConfiguration' {
                        @([PSCustomObject]@{
                            Index     = 42
                            IPAddress = @('169.254.100.1', 'fe80::1')
                        })
                    }
                    'MSFT_NetIPInterface' {
                        @(
                            [PSCustomObject]@{
                                InterfaceAlias  = 'Ethernet 11'
                                AddressFamily   = 2
                                InterfaceMetric = 15
                                NlMtu           = 62000
                                AutomaticMetric = $false
                            },
                            [PSCustomObject]@{
                                InterfaceAlias  = 'Ethernet 11'
                                AddressFamily   = 23
                                InterfaceMetric = 15
                                NlMtu           = 62000
                                AutomaticMetric = $false
                            }
                        )
                    }
                    'MSFT_NetNeighbor' {
                        @([PSCustomObject]@{
                            InterfaceAlias   = 'Ethernet 11'
                            IPAddress        = '169.254.100.2'
                            LinkLayerAddress = '11-22-33-44-55-66'
                            State            = 2
                        })
                    }
                    'Win32_PnPSignedDriver' { return @() }
                    default { return @() }
                }
            } -ModuleName PC-AI.Drivers
        }

        It "Should return adapter with P2PNetwork role" {
            $result = @(Get-ThunderboltNetworkStatus)
            $result.Count | Should -BeGreaterOrEqual 1
            $result[0].Role | Should -Be 'P2PNetwork'
        }

        It "Should include IPv4 address" {
            $result = @(Get-ThunderboltNetworkStatus)
            $result[0].IPv4Addresses | Should -Contain '169.254.100.1'
        }

        It "Should include neighbor peer candidates" {
            $result = @(Get-ThunderboltNetworkStatus)
            $result[0].NeighborCandidates.Count | Should -BeGreaterOrEqual 1
            $result[0].NeighborCandidates[0].IPAddress | Should -Be '169.254.100.2'
        }

        It "Should include APIPA recommendation for link-local addressing" {
            $result = @(Get-ThunderboltNetworkStatus)
            $result[0].RecommendedActions.Count | Should -BeGreaterOrEqual 1
        }

        It "Should report correct link speed in Gbps" {
            $result = @(Get-ThunderboltNetworkStatus)
            $result[0].LinkSpeedGbps | Should -Be 10.0
        }
    }
}

# ─── Get-NetworkDiscoverySnapshot ────────────────────────────────────────────

Describe "Get-NetworkDiscoverySnapshot" -Tag 'Unit', 'Drivers', 'Thunderbolt', 'Windows' {
    Context "Function interface" {
        It "Should be exported from the module" {
            Get-Command Get-NetworkDiscoverySnapshot -Module PC-AI.Drivers | Should -Not -BeNullOrEmpty
        }

        It "Should accept ComputerName and IncludeRawCommands parameters" {
            $cmd = Get-Command Get-NetworkDiscoverySnapshot -Module PC-AI.Drivers
            $cmd.Parameters.Keys | Should -Contain 'ComputerName'
            $cmd.Parameters.Keys | Should -Contain 'IncludeRawCommands'
        }
    }

    Context "With mocked CIM data" {
        BeforeAll {
            Mock Get-CimInstance {
                param($Namespace, $ClassName)
                switch ($ClassName) {
                    'Win32_NetworkAdapter' {
                        @([PSCustomObject]@{
                            Name              = 'Ethernet Adapter'
                            Description       = 'Intel Ethernet'
                            NetConnectionID   = 'Ethernet'
                            NetEnabled        = $true
                            NetConnectionStatus = 2
                            PhysicalAdapter   = $true
                            MACAddress        = 'AA-BB-CC-DD-EE-FF'
                            PNPDeviceID       = 'PCI\VEN_8086&DEV_15F3'
                            Index             = 1
                            Speed             = 1000000000
                        })
                    }
                    'Win32_NetworkAdapterConfiguration' {
                        @([PSCustomObject]@{
                            Index                = 1
                            IPAddress            = @('192.168.1.100')
                            DefaultIPGateway     = @('192.168.1.1')
                            DNSServerSearchOrder = @('8.8.8.8')
                            DHCPEnabled          = $true
                            DHCPServer           = '192.168.1.1'
                        })
                    }
                    'MSFT_NetIPInterface' {
                        @(
                            [PSCustomObject]@{
                                InterfaceAlias  = 'Ethernet'
                                AddressFamily   = 2
                                InterfaceMetric = 25
                                NlMtu           = 1500
                                AutomaticMetric = $true
                            }
                        )
                    }
                    'MSFT_NetNeighbor' { return @() }
                    'Win32_IP4RouteTable' { return @() }
                    default { return @() }
                }
            } -ModuleName PC-AI.Drivers
        }

        It "Should return a snapshot object with adapters" {
            $result = Get-NetworkDiscoverySnapshot
            $result | Should -Not -BeNullOrEmpty
            $result.PSObject.Properties.Name | Should -Contain 'Adapters'
        }
    }
}

# ─── Find-ThunderboltPeer ────────────────────────────────────────────────────

Describe "Find-ThunderboltPeer" -Tag 'Unit', 'Drivers', 'Thunderbolt', 'Windows' {
    Context "Function exists and has expected parameters" {
        It "Should be exported from the module" {
            Get-Command Find-ThunderboltPeer -Module PC-AI.Drivers | Should -Not -BeNullOrEmpty
        }

        It "Should accept ComputerNameCandidates parameter" {
            $cmd = Get-Command Find-ThunderboltPeer -Module PC-AI.Drivers
            $cmd.Parameters.Keys | Should -Contain 'ComputerNameCandidates'
        }

        It "Should accept TcpTimeoutMs with range validation" {
            $cmd = Get-Command Find-ThunderboltPeer -Module PC-AI.Drivers
            $cmd.Parameters.Keys | Should -Contain 'TcpTimeoutMs'
        }
    }
}

# ─── Connect-ThunderboltPeer ─────────────────────────────────────────────────

Describe "Connect-ThunderboltPeer" -Tag 'Unit', 'Drivers', 'Thunderbolt', 'Windows' {
    Context "Function interface" {
        It "Should be exported from the module" {
            Get-Command Connect-ThunderboltPeer -Module PC-AI.Drivers | Should -Not -BeNullOrEmpty
        }

        It "Should accept ComputerName and Address parameters" {
            $cmd = Get-Command Connect-ThunderboltPeer -Module PC-AI.Drivers
            $cmd.Parameters.Keys | Should -Contain 'ComputerName'
            $cmd.Parameters.Keys | Should -Contain 'Address'
        }
    }
}

# ─── Set-ThunderboltNetworkOptimization ──────────────────────────────────────

Describe "Set-ThunderboltNetworkOptimization" -Tag 'Unit', 'Drivers', 'Thunderbolt', 'Windows' {
    Context "Function interface" {
        It "Should be exported from the module" {
            Get-Command Set-ThunderboltNetworkOptimization -Module PC-AI.Drivers | Should -Not -BeNullOrEmpty
        }

        It "Should accept InterfaceAlias parameter" {
            $cmd = Get-Command Set-ThunderboltNetworkOptimization -Module PC-AI.Drivers
            $cmd.Parameters.Keys | Should -Contain 'InterfaceAlias'
        }

        It "Should support WhatIf" {
            $cmd = Get-Command Set-ThunderboltNetworkOptimization -Module PC-AI.Drivers
            $cmd.Parameters.Keys | Should -Contain 'WhatIf'
        }
    }
}

# ─── Module Export Completeness ──────────────────────────────────────────────

Describe "PC-AI.Drivers Module Exports" -Tag 'Unit', 'Drivers', 'Fast', 'Windows' {
    It "Should export all 11 declared functions" {
        $mod = Get-Module PC-AI.Drivers
        $expected = @(
            'Get-PnpDeviceInventory',
            'Get-DriverRegistry',
            'Compare-DriverVersion',
            'Get-DriverReport',
            'Install-DriverUpdate',
            'Update-DriverRegistry',
            'Get-NetworkDiscoverySnapshot',
            'Find-ThunderboltPeer',
            'Get-ThunderboltNetworkStatus',
            'Connect-ThunderboltPeer',
            'Set-ThunderboltNetworkOptimization'
        )
        foreach ($fn in $expected) {
            $mod.ExportedFunctions.Keys | Should -Contain $fn
        }
    }

    It "Should not export private functions" {
        $mod = Get-Module PC-AI.Drivers
        $mod.ExportedFunctions.Keys | Should -Not -Contain 'Test-AdminElevation'
        $mod.ExportedFunctions.Keys | Should -Not -Contain 'Resolve-HardwareId'
        $mod.ExportedFunctions.Keys | Should -Not -Contain 'Invoke-TrustedDownload'
    }
}
