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
    # Evict any copy already loaded by an earlier suite before importing.
    # `Import-Module -Force` re-imports, but it does NOT remove a copy that
    # was loaded from a different path, so two modules of the same name can
    # coexist. Pester then refuses to mock into either -- "Multiple script or
    # manifest modules named 'X' are currently loaded" -- and every mocked
    # call falls through to the real cmdlet. That is why these files pass in
    # isolation and fail in a full run.
    Get-Module 'PC-AI.Drivers' -All | Remove-Module -Force -ErrorAction SilentlyContinue
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

Describe "Get-DriverRegistry" -Tag 'Unit', 'Drivers', 'Fast', 'Portable' {
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

Describe "Compare-DriverVersion" -Tag 'Unit', 'Drivers', 'Fast', 'Portable' {
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
        # Protects: default Current filtering and exact matched-version identity under StrictMode.
        # Detects: leaked Current rows, suppressed classifier errors, or an always-empty classifier.
        # Needs: private registry and same-input positive IncludeUpToDate control; no hardware or network.
        # Breadcrumb: Modules/PC-AI.Drivers/Public/Compare-DriverVersion.ps1.
        It "Should suppress Current by default" {
            $inventory = @([PSCustomObject]@{
                Name          = 'Realtek RTL8156'
                VID           = '0BDA'
                PID           = '8156'
                PnpClass      = 'Net'
                DriverVersion = '1156.21.20.1110'
            })
            $result = @(Compare-DriverVersion -Inventory $inventory -Registry $script:Registry -ErrorAction Stop)
            $result.Count | Should -Be 0
            $included = @(Compare-DriverVersion -Inventory $inventory -Registry $script:Registry -IncludeUpToDate -ErrorAction Stop)
            $included.Count | Should -Be 1
            $included[0].Status | Should -BeExactly 'Current'
            $included[0].DeviceName | Should -BeExactly 'Realtek RTL8156'
            $included[0].RegistryId | Should -BeExactly 'realtek-rtl8156'
            $included[0].InstalledVersion | Should -BeExactly '1156.21.20.1110'
            $included[0].TargetVersion | Should -BeExactly '1156.21.20.1110'
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
        # Protects: default Unknown filtering and exact unmatched-device identity under StrictMode.
        # Detects: leaked Unknown rows, suppressed classifier errors, or an always-empty classifier.
        # Needs: private registry and same-input positive IncludeUnknown control; no hardware or network.
        # Breadcrumb: Modules/PC-AI.Drivers/Public/Compare-DriverVersion.ps1.
        It "Should suppress Unknown by default" {
            $inventory = @([PSCustomObject]@{
                Name          = 'Unknown Widget'
                VID           = 'AAAA'
                PID           = 'BBBB'
                PnpClass      = 'Other'
                DriverVersion = '1.0'
            })
            $result = @(Compare-DriverVersion -Inventory $inventory -Registry $script:Registry -ErrorAction Stop)
            $result.Count | Should -Be 0
            $included = @(Compare-DriverVersion -Inventory $inventory -Registry $script:Registry -IncludeUnknown -ErrorAction Stop)
            $included.Count | Should -Be 1
            $included[0].Status | Should -BeExactly 'Unknown'
            $included[0].DeviceName | Should -BeExactly 'Unknown Widget'
            $included[0].InstalledVersion | Should -BeExactly '1.0'
            $included[0].RegistryId | Should -BeNullOrEmpty
            $included[0].TargetVersion | Should -BeNullOrEmpty
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

Describe "Get-PnpDeviceInventory" -Tag 'Unit', 'Drivers', 'Fast', 'Portable' {
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

Describe "Get-DriverReport" -Tag 'Unit', 'Drivers', 'Fast', 'Portable' {
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

            # Get-DriverReport calls Get-PnpDevice and Get-PnpDeviceProperty
            # DIRECTLY, despite its doc comment claiming it "Orchestrates
            # Get-PnpDeviceInventory". Mocking the inventory function had no effect
            # whatsoever -- the real PnP enumeration ran, and this assertion was
            # decided by whatever hardware the host happens to have. On this
            # workstation a real Realtek 0BDA:8156 adapter reports exactly the
            # registry's latestVersion, so Status came back 'Current' against an
            # expected 'Outdated'. Mock what the function actually calls.
            Mock Get-PnpDevice {
                @([PSCustomObject]@{
                        InstanceId   = 'USB\VID_0BDA&PID_8156\1'
                        FriendlyName = 'Realtek RTL8156'
                        Class        = 'Net'
                        Status       = 'OK'
                    })
            } -ModuleName PC-AI.Drivers

            Mock Get-PnpDeviceProperty {
                switch ($KeyName) {
                    'DEVPKEY_Device_DriverVersion' { [PSCustomObject]@{ Data = '1.0.0.0' } }
                    'DEVPKEY_Device_Manufacturer' { [PSCustomObject]@{ Data = 'Realtek' } }
                    default { $null }
                }
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

Describe "Install-DriverUpdate" -Tag 'Unit', 'Drivers', 'Fast', 'Portable' {
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

Describe 'Driver download WhatIf and existing-byte integrity contracts' -Tag 'Unit', 'Drivers', 'Portable', 'DriverDownloadContract' {
    BeforeAll {
        # SHA256 of the independent three-byte ASCII fixture "abc".
        $script:DownloadContractHash = 'BA7816BF8F01CFEA414140DE5DAE2223B00361A396177A9CB410FF61F20015AD'
    }

    BeforeEach {
        $script:DownloadContractRoot = Join-Path $TestDrive ([guid]::NewGuid().ToString('N'))
        $script:DownloadContractRegistry = Join-Path $TestDrive 'download-contract-registry.json'
        $registry = $script:MockRegistryJson | ConvertFrom-Json
        $registry.trustedSources[0].baseUrl = 'https://drivers.example.invalid'
        $registry.devices[0].driver.downloadUrl = 'https://drivers.example.invalid/driver.exe'
        $registry.devices[0].driver.installerType = 'exe'
        $registry.devices[0].driver.sha256 = $script:DownloadContractHash
        $registry.devices[0].sharedDriverGroup = $null
        $registry | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $script:DownloadContractRegistry -Encoding utf8
        Mock Test-AdminElevation { $true } -ModuleName PC-AI.Drivers
        Mock Invoke-WebRequest { throw 'Unmocked driver network access is forbidden.' } -ModuleName PC-AI.Drivers
        Mock Start-Process { throw 'Actual driver installer execution is forbidden.' } -ModuleName PC-AI.Drivers
        Mock Expand-Archive { throw 'Actual driver archive extraction is forbidden.' } -ModuleName PC-AI.Drivers
    }

    # Protects: both download and install WhatIf modes suppress all mutation, with absent or existing destinations.
    # Detects: premature directory creation, DownloadOnly bypass, network access, cache reads, or installer dispatch.
    # Needs: private fixture registry and bytes; all four DownloadOnly/Existing case combinations are inert.
    # Breadcrumb: Modules/PC-AI.Drivers/Public/Install-DriverUpdate.ps1; Private/Invoke-TrustedDownload.ps1.
    It 'performs no write or download in WhatIf with DownloadOnly=<DownloadOnly> Existing=<Existing>' -ForEach @(
        @{ DownloadOnly = $false; Existing = $false },
        @{ DownloadOnly = $true; Existing = $false },
        @{ DownloadOnly = $false; Existing = $true },
        @{ DownloadOnly = $true; Existing = $true }
    ) {
        $existingPath = Join-Path $script:DownloadContractRoot 'realtek-rtl8156.exe'
        if ($Existing) {
            [void][IO.Directory]::CreateDirectory($script:DownloadContractRoot)
            [IO.File]::WriteAllBytes($existingPath, [byte[]](0x78, 0x79, 0x7A))
        }
        $registryBefore = [IO.File]::ReadAllText($script:DownloadContractRegistry)
        Mock New-Item { throw 'WhatIf reached directory mutation.' } -ModuleName PC-AI.Drivers
        Mock Get-FileHash { throw 'WhatIf reached download acceptance.' } -ModuleName PC-AI.Drivers
        $result = Install-DriverUpdate -DeviceId 'realtek-rtl8156' -RegistryPath $script:DownloadContractRegistry -DownloadDir $script:DownloadContractRoot -DownloadOnly:$DownloadOnly -WhatIf
        $result.Action | Should -BeExactly 'WhatIf'
        $result.Success | Should -BeTrue
        $result.FilePath | Should -BeNullOrEmpty
        Should -Invoke New-Item -Times 0 -Exactly -ModuleName PC-AI.Drivers -Scope It
        Should -Invoke Invoke-WebRequest -Times 0 -Exactly -ModuleName PC-AI.Drivers -Scope It
        Should -Invoke Get-FileHash -Times 0 -Exactly -ModuleName PC-AI.Drivers -Scope It
        Should -Invoke Start-Process -Times 0 -Exactly -ModuleName PC-AI.Drivers -Scope It
        Should -Invoke Expand-Archive -Times 0 -Exactly -ModuleName PC-AI.Drivers -Scope It
        [IO.Directory]::Exists($script:DownloadContractRoot) | Should -Be $Existing
        [IO.File]::ReadAllText($script:DownloadContractRegistry) | Should -BeExactly $registryBefore
        if ($Existing) { [Convert]::ToBase64String([IO.File]::ReadAllBytes($existingPath)) | Should -BeExactly 'eHl6' }
    }

    # Protects: normal DownloadOnly still produces a verified private file without launching or extracting it.
    # Detects: a fix that blocks useful downloads, alters forwarding, or dispatches an installer.
    # Needs: actual public/private maintained functions, private registry, inert web-response bytes "abc".
    # Breadcrumb: Modules/PC-AI.Drivers/Public/Install-DriverUpdate.ps1; Private/Invoke-TrustedDownload.ps1.
    It 'downloads verified bytes normally without executing an installer' {
        Mock Invoke-WebRequest { param($Uri, $OutFile) [IO.File]::WriteAllBytes($OutFile, [byte[]](0x61, 0x62, 0x63)) } -ModuleName PC-AI.Drivers
        $result = Install-DriverUpdate -DeviceId 'realtek-rtl8156' -RegistryPath $script:DownloadContractRegistry -DownloadDir $script:DownloadContractRoot -DownloadOnly -Confirm:$false
        $result.Action | Should -BeExactly 'DownloadOnly'
        $result.Success | Should -BeTrue
        $result.FilePath | Should -BeExactly (Join-Path $script:DownloadContractRoot 'realtek-rtl8156.exe')
        [Convert]::ToBase64String([IO.File]::ReadAllBytes($result.FilePath)) | Should -BeExactly 'YWJj'
        Should -Invoke Invoke-WebRequest -Times 1 -Exactly -ModuleName PC-AI.Drivers -Scope It -ParameterFilter { $Uri -eq 'https://drivers.example.invalid/driver.exe' }
        Should -Invoke Start-Process -Times 0 -Exactly -ModuleName PC-AI.Drivers -Scope It
        Should -Invoke Expand-Archive -Times 0 -Exactly -ModuleName PC-AI.Drivers -Scope It
    }

    # Protects: valid existing download bytes remain reusable without network or installation.
    # Detects: a regression that refuses known-good cached data or overwrites it while validating.
    # Needs: private literal filename containing brackets and an independent known SHA256 digest.
    # Breadcrumb: Modules/PC-AI.Drivers/Private/Invoke-TrustedDownload.ps1.
    It 'accepts matching existing bytes using their literal path without network access' {
        [void][IO.Directory]::CreateDirectory($script:DownloadContractRoot)
        $path = Join-Path $script:DownloadContractRoot 'installer[1].exe'
        [IO.File]::WriteAllBytes($path, [byte[]](0x61, 0x62, 0x63))
        $result = & (Get-Module PC-AI.Drivers) { param($Path, $Hash) Invoke-TrustedDownload -Url 'https://drivers.example.invalid/driver.exe' -OutFile $Path -TrustedHosts 'drivers.example.invalid' -ExpectedSha256 $Hash } $path $script:DownloadContractHash.ToLowerInvariant()
        $result | Should -BeExactly $path
        [Convert]::ToBase64String([IO.File]::ReadAllBytes($path)) | Should -BeExactly 'YWJj'
        Should -Invoke Invoke-WebRequest -Times 0 -Exactly -ModuleName PC-AI.Drivers -Scope It
    }

    # Protects: a mismatched cached download is refused while the user's exact file is preserved.
    # Detects: accepting unverified bytes, implicit redownload, or deletion of preexisting user data.
    # Needs: private "xyz" bytes and the independent expected "abc" digest; no service or installer.
    # Breadcrumb: Modules/PC-AI.Drivers/Public/Install-DriverUpdate.ps1; Private/Invoke-TrustedDownload.ps1.
    It 'refuses a mismatched existing download and preserves it without redownloading' {
        [void][IO.Directory]::CreateDirectory($script:DownloadContractRoot)
        $path = Join-Path $script:DownloadContractRoot 'realtek-rtl8156.exe'
        [IO.File]::WriteAllBytes($path, [byte[]](0x78, 0x79, 0x7A))
        $result = Install-DriverUpdate -DeviceId 'realtek-rtl8156' -RegistryPath $script:DownloadContractRegistry -DownloadDir $script:DownloadContractRoot -DownloadOnly -Confirm:$false -ErrorAction SilentlyContinue
        $result.Action | Should -BeExactly 'Failed'
        $result.Success | Should -BeFalse
        [Convert]::ToBase64String([IO.File]::ReadAllBytes($path)) | Should -BeExactly 'eHl6'
        Should -Invoke Invoke-WebRequest -Times 0 -Exactly -ModuleName PC-AI.Drivers -Scope It
        Should -Invoke Start-Process -Times 0 -Exactly -ModuleName PC-AI.Drivers -Scope It
    }

    # Protects: verification failure never accepts, deletes, or replaces an existing download.
    # Detects: swallowing a hash read error as a cache hit or falling through to a network write.
    # Needs: private existing bytes and an inert fault at the file-hash boundary.
    # Breadcrumb: Modules/PC-AI.Drivers/Private/Invoke-TrustedDownload.ps1.
    It 'refuses an existing file whose hash cannot be read while retaining its bytes' {
        [void][IO.Directory]::CreateDirectory($script:DownloadContractRoot)
        $path = Join-Path $script:DownloadContractRoot 'installer.exe'
        [IO.File]::WriteAllBytes($path, [byte[]](0x61, 0x62, 0x63))
        Mock Get-FileHash { throw 'Synthetic hash read failure.' } -ModuleName PC-AI.Drivers
        $result = & (Get-Module PC-AI.Drivers) { param($Path, $Hash) Invoke-TrustedDownload -Url 'https://drivers.example.invalid/driver.exe' -OutFile $Path -TrustedHosts 'drivers.example.invalid' -ExpectedSha256 $Hash -ErrorAction SilentlyContinue } $path $script:DownloadContractHash
        $result | Should -BeNullOrEmpty
        [Convert]::ToBase64String([IO.File]::ReadAllBytes($path)) | Should -BeExactly 'YWJj'
        Should -Invoke Get-FileHash -Times 1 -Exactly -ModuleName PC-AI.Drivers -Scope It
        Should -Invoke Invoke-WebRequest -Times 0 -Exactly -ModuleName PC-AI.Drivers -Scope It
    }

    # Protects: rejecting wrong downloaded bytes removes only the exact private download leaf.
    # Detects: wildcard cleanup deleting an unrelated sibling while leaving the actual failed download behind.
    # Needs: inert web-response "xyz" bytes at '[probe].exe', private 'p.exe' sentinel and independent "abc" digest.
    # Breadcrumb: Modules/PC-AI.Drivers/Private/Invoke-TrustedDownload.ps1.
    It 'removes a wrong-digest downloaded bracket filename while preserving its wildcard-matching sibling' {
        [void][IO.Directory]::CreateDirectory($script:DownloadContractRoot)
        $path = Join-Path $script:DownloadContractRoot '[probe].exe'
        $sibling = Join-Path $script:DownloadContractRoot 'p.exe'
        [IO.File]::WriteAllBytes($sibling, [byte[]](0x53, 0x41, 0x46, 0x45))
        Mock Invoke-WebRequest { param($Uri, $OutFile) [IO.File]::WriteAllBytes($OutFile, [byte[]](0x78, 0x79, 0x7A)) } -ModuleName PC-AI.Drivers
        $result = & (Get-Module PC-AI.Drivers) { param($Path, $Hash) Invoke-TrustedDownload -Url 'https://drivers.example.invalid/driver.exe' -OutFile $Path -TrustedHosts 'drivers.example.invalid' -ExpectedSha256 $Hash -ErrorAction SilentlyContinue } $path $script:DownloadContractHash
        $result | Should -BeNullOrEmpty
        [IO.File]::Exists($path) | Should -BeFalse
        [Convert]::ToBase64String([IO.File]::ReadAllBytes($sibling)) | Should -BeExactly 'U0FGRQ=='
        Should -Invoke Invoke-WebRequest -Times 1 -Exactly -ModuleName PC-AI.Drivers -Scope It
        Should -Invoke Start-Process -Times 0 -Exactly -ModuleName PC-AI.Drivers -Scope It
    }

    # Protects: the existing optional-digest contract still permits explicitly unpinned cached files.
    # Detects: accidental new digest requirements or redundant reads/network calls for callers without a hash.
    # Needs: private existing bytes and explicit absence of ExpectedSha256; no service.
    # Breadcrumb: Modules/PC-AI.Drivers/Private/Invoke-TrustedDownload.ps1.
    It 'reuses existing bytes when no digest was requested without claiming verification' {
        [void][IO.Directory]::CreateDirectory($script:DownloadContractRoot)
        $path = Join-Path $script:DownloadContractRoot 'installer.exe'
        [IO.File]::WriteAllBytes($path, [byte[]](0x78, 0x79, 0x7A))
        Mock Get-FileHash { throw 'No digest was requested.' } -ModuleName PC-AI.Drivers
        $result = & (Get-Module PC-AI.Drivers) { param($Path) Invoke-TrustedDownload -Url 'https://drivers.example.invalid/driver.exe' -OutFile $Path -TrustedHosts 'drivers.example.invalid' } $path
        $result | Should -BeExactly $path
        Should -Invoke Get-FileHash -Times 0 -Exactly -ModuleName PC-AI.Drivers -Scope It
        Should -Invoke Invoke-WebRequest -Times 0 -Exactly -ModuleName PC-AI.Drivers -Scope It
    }
}

# ─── Update-DriverRegistry ───────────────────────────────────────────────────

Describe "Update-DriverRegistry" -Tag 'Unit', 'Drivers', 'Fast', 'Portable' {
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

Describe "Get-ThunderboltNetworkStatus" -Tag 'Unit', 'Drivers', 'Thunderbolt', 'Portable' {
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

Describe "Get-NetworkDiscoverySnapshot" -Tag 'Unit', 'Drivers', 'Thunderbolt', 'Portable' {
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

    # A disabled adapter (seen live: Cisco AnyConnect) has no MSFT_NetIPInterface
    # or MSFT_NetNeighbor rows, and a physical adapter can lack NetConnectionID.
    # @($table[$missingKey]) is a one-element array holding $null, so under
    # StrictMode the AddressFamily filter threw and NeighborCount reported 1;
    # a $null key threw "the array index evaluated to null" outright.
    Context "With adapters that have no IP interface, neighbors, or alias" {
        BeforeAll {
            Mock Get-CimInstance {
                param($Namespace, $ClassName)
                switch ($ClassName) {
                    'Win32_NetworkAdapter' {
                        @(
                            [PSCustomObject]@{
                                Name = 'Ethernet Adapter'; Description = 'Intel Ethernet'
                                NetConnectionID = 'Ethernet'; NetEnabled = $true
                                PhysicalAdapter = $true; MACAddress = 'AA-BB-CC-DD-EE-FF'
                                PNPDeviceID = 'PCI\VEN_8086&DEV_15F3'; Index = 1; Speed = 1000000000
                            },
                            [PSCustomObject]@{
                                Name = 'VPN Miniport'; Description = 'Disabled VPN adapter'
                                NetConnectionID = 'Ethernet 13'; NetEnabled = $false
                                PhysicalAdapter = $true; MACAddress = $null
                                PNPDeviceID = 'ROOT\NET\0000'; Index = 2; Speed = $null
                            },
                            [PSCustomObject]@{
                                Name = 'Unnamed NIC'; Description = 'Physical adapter without alias'
                                NetConnectionID = $null; NetEnabled = $false
                                PhysicalAdapter = $true; MACAddress = $null
                                PNPDeviceID = 'PCI\VEN_10EC&DEV_8168'; Index = 3; Speed = $null
                            }
                        )
                    }
                    'Win32_NetworkAdapterConfiguration' {
                        # Windows keeps a configuration row for every adapter index,
                        # including disabled ones, with the address fields empty.
                        @(
                            [PSCustomObject]@{
                                Index = 1; IPAddress = @('192.168.1.100'); DefaultIPGateway = @('192.168.1.1')
                                DNSServerSearchOrder = @('8.8.8.8'); DHCPEnabled = $true; DHCPServer = '192.168.1.1'
                            },
                            [PSCustomObject]@{
                                Index = 2; IPAddress = $null; DefaultIPGateway = $null
                                DNSServerSearchOrder = $null; DHCPEnabled = $true; DHCPServer = $null
                            },
                            [PSCustomObject]@{
                                Index = 3; IPAddress = $null; DefaultIPGateway = $null
                                DNSServerSearchOrder = $null; DHCPEnabled = $true; DHCPServer = $null
                            }
                        )
                    }
                    'MSFT_NetIPInterface' {
                        @([PSCustomObject]@{
                            InterfaceAlias = 'Ethernet'; AddressFamily = 2; InterfaceMetric = 25; NlMtu = 1500
                        })
                    }
                    'MSFT_NetNeighbor' {
                        @([PSCustomObject]@{
                            InterfaceAlias = 'Ethernet'; IPAddress = '192.168.1.1'
                            LinkLayerAddress = '11-22-33-44-55-66'; State = 'Reachable'
                        })
                    }
                    default { return @() }
                }
            } -ModuleName PC-AI.Drivers
        }

        It "Should complete without errors" {
            # Called directly, not inside a Should -Not -Throw block: that block is
            # a child scope, so -ErrorVariable would never reach this $errs. A throw
            # still fails the test.
            $script:snapshot = Get-NetworkDiscoverySnapshot -ErrorVariable errs
            $errs.Count | Should -Be 0
            $script:snapshot.AdapterCount | Should -Be 3
        }

        It "Should report zero neighbors and null metrics for an adapter with no IP interface" {
            $vpn = $script:snapshot.Adapters | Where-Object Name -eq 'VPN Miniport'
            $vpn.NeighborCount | Should -Be 0
            $vpn.IPv4Metric | Should -BeNullOrEmpty
            $vpn.IPv4Mtu | Should -BeNullOrEmpty
        }

        It "Should still report the healthy adapter's interface and neighbor data" {
            $eth = $script:snapshot.Adapters | Where-Object Name -eq 'Ethernet Adapter'
            $eth.IPv4Metric | Should -Be 25
            $eth.IPv4Mtu | Should -Be 1500
            $eth.NeighborCount | Should -Be 1
        }
    }
}

# ─── Find-ThunderboltPeer ────────────────────────────────────────────────────

Describe "Find-ThunderboltPeer" -Tag 'Unit', 'Drivers', 'Thunderbolt', 'Portable' {
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

Describe "Connect-ThunderboltPeer" -Tag 'Unit', 'Drivers', 'Thunderbolt', 'Portable' {
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

Describe "Set-ThunderboltNetworkOptimization" -Tag 'Unit', 'Drivers', 'Thunderbolt', 'Portable' {
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

Describe "PC-AI.Drivers Module Exports" -Tag 'Unit', 'Drivers', 'Fast', 'Portable' {
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
