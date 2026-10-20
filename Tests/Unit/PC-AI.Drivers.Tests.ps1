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

        It "Should accept ComputerNameCandidates and SshPort parameters" {
            $cmd = Get-Command Find-ThunderboltPeer -Module PC-AI.Drivers
            $cmd.Parameters.Keys | Should -Contain 'ComputerNameCandidates'
            $cmd.Parameters.Keys | Should -Contain 'SshPort'
        }

        It "Should accept TcpTimeoutMs with range validation" {
            $cmd = Get-Command Find-ThunderboltPeer -Module PC-AI.Drivers
            $cmd.Parameters.Keys | Should -Contain 'TcpTimeoutMs'
        }
    }

    Context "When Win32_NetworkAdapter.Index differs from TCP/IP InterfaceIndex" {
        BeforeAll {
            Mock Get-CimInstance {
                param($Namespace, $ClassName)
                switch ($ClassName) {
                    'Win32_NetworkAdapter' {
                        @([PSCustomObject]@{
                            Name           = 'USB4(TM) P2P Network Adapter #4'
                            Description    = 'USB4(TM) P2P Network Adapter #4'
                            NetConnectionID = 'Ethernet 18'
                            NetEnabled     = $true
                            MACAddress     = '34-4D-A1-C4-C4-22'
                            Speed          = 20000000000
                            Index          = 28
                            InterfaceIndex = 95
                            PNPDeviceID    = 'USB4\VEN_045E&DEV_0000083F&PROT_USB4NET\4&6E3A7BC&2&0'
                        })
                    }
                    'Win32_PnPEntity' {
                        @([PSCustomObject]@{
                            Name        = 'USB4(TM) Host Router (Microsoft)'
                            PNPDeviceID = 'PCI\VEN_8086&DEV_7EC2\0000'
                            Status      = 'OK'
                        })
                    }
                    'MSFT_NetIPAddress' {
                        @([PSCustomObject]@{
                            InterfaceIndex = 95
                            InterfaceAlias = 'Ethernet 18'
                            AddressFamily  = 2
                            IPAddress      = '169.254.43.202'
                            PrefixLength   = 16
                        })
                    }
                    'MSFT_NetNeighbor' {
                        @([PSCustomObject]@{
                            InterfaceIndex   = 95
                            InterfaceAlias   = 'Ethernet 18'
                            IPAddress        = '169.254.196.28'
                            LinkLayerAddress = '08-93-C1-7B-39-51'
                            State            = 'Reachable'
                        })
                    }
                    'MSFT_NetRoute' {
                        @(
                            [PSCustomObject]@{
                                InterfaceIndex    = 15
                                DestinationPrefix = '169.254.0.0/16'
                                RouteMetric       = 256
                            },
                            [PSCustomObject]@{
                                InterfaceIndex    = 95
                                DestinationPrefix = '169.254.0.0/16'
                                RouteMetric       = 256
                            }
                        )
                    }
                    'MSFT_NetIPInterface' {
                        @(
                            [PSCustomObject]@{
                                InterfaceIndex  = 15
                                InterfaceAlias  = 'vEthernet (WSL)'
                                AddressFamily   = 2
                                InterfaceMetric = 15
                            },
                            [PSCustomObject]@{
                                InterfaceIndex  = 95
                                InterfaceAlias  = 'Ethernet 18'
                                AddressFamily   = 2
                                InterfaceMetric = 15
                            }
                        )
                    }
                    default { return @() }
                }
            } -ModuleName PC-AI.Drivers
        }

        It "Should use InterfaceIndex (95) instead of WMI Index (28) to match addresses and neighbors" {
            $res = Find-ThunderboltPeer -ComputerNameCandidates @() -TcpTimeoutMs 250
            $res.AdapterCount | Should -Be 1
            $res.Adapters[0].ifIndex | Should -Be 95
            $res.Adapters[0].WmiIndex | Should -Be 28
            $res.LocalAddresses.Count | Should -Be 1
            $res.LocalAddresses[0].IPAddress | Should -Be '169.254.43.202'
            $res.Neighbors.Count | Should -Be 1
            $res.Neighbors[0].IPAddress | Should -Be '169.254.196.28'
            $res.Usb4Topology.Count | Should -Be 1
            $res.Usb4Topology[0].Role | Should -Be 'HostRouter'
        }

        It "Should detect competing 169.254.0.0/16 APIPA route conflicts on virtual adapters with lower ifIndex" {
            $res = Find-ThunderboltPeer -ComputerNameCandidates @() -TcpTimeoutMs 250
            $res.ApipaRouteConflicts.Count | Should -Be 1
            $res.ApipaRouteConflicts[0].ConflictingInterfaceAlias | Should -Be 'vEthernet (WSL)'
            $res.ApipaRouteConflicts[0].ConflictingInterfaceIndex | Should -Be 15
        }
    }
}

# ─── Connect-ThunderboltPeer ─────────────────────────────────────────────────

Describe "Connect-ThunderboltPeer" -Tag 'Unit', 'Drivers', 'Thunderbolt', 'Portable' {
    Context "Function interface" {
        It "Should be exported from the module" {
            Get-Command Connect-ThunderboltPeer -Module PC-AI.Drivers | Should -Not -BeNullOrEmpty
        }

        It "Should accept ComputerName, Address, and Protocol parameters" {
            $cmd = Get-Command Connect-ThunderboltPeer -Module PC-AI.Drivers
            $cmd.Parameters.Keys | Should -Contain 'ComputerName'
            $cmd.Parameters.Keys | Should -Contain 'Address'
            $cmd.Parameters.Keys | Should -Contain 'Protocol'
            $cmd.Parameters.Keys | Should -Contain 'SshUser'
            $cmd.Parameters.Keys | Should -Contain 'SshCommand'
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

# ─── Get-ThunderboltPeerIdentity ─────────────────────────────────────────────

Describe "Get-ThunderboltPeerIdentity" -Tag 'Unit', 'Drivers', 'Thunderbolt', 'Fast', 'Portable' {
    It "Should resolve canonical identity and short aliases for known fleet hosts (including dtm-supernuc alias)" {
        $carbon = Get-ThunderboltPeerIdentity -ComputerName 'dtm-carbon-two'
        $nuc    = Get-ThunderboltPeerIdentity -ComputerName 'dtm-supernuc'
        $carbon.ComputerName | Should -BeExactly 'dtm-carbon-two'
        $carbon.IsKnownFleetPeer | Should -BeTrue
        $carbon.PrimaryUla | Should -BeExactly 'fd00:7462:7032::c2'
        $carbon.TbHostAlias | Should -BeExactly 'dtm-carbon-two-tb'

        $nuc.ComputerName | Should -BeExactly 'dtm-super-nuc'
        $nuc.IsKnownFleetPeer | Should -BeTrue
        $nuc.PrimaryUla | Should -BeExactly 'fd00:7462:7032::50'
        $nuc.TbHostAlias | Should -BeExactly 'dtm-super-nuc-tb'
    }

    It "Should derive deterministic non-colliding IPv6 Link-Local and ULA addresses for arbitrary unknown hosts without registry entries" {
        $missingReg = Join-Path $TestDrive 'nonexistent-peers.json'
        $hostA = Get-ThunderboltPeerIdentity -ComputerName 'dtm-future-blade-01' -RegistryPath $missingReg
        $hostB = Get-ThunderboltPeerIdentity -ComputerName 'guest-thunderbolt-pc' -RegistryPath $missingReg

        $hostA.IsKnownFleetPeer | Should -BeFalse
        $hostA.DeterministicLinkLocal | Should -Match '^fe80::d700:[0-9a-f]{4}:[0-9a-f]{4}:[0-9a-f]{4}$'
        $hostA.DeterministicUla | Should -Match '^fd00:7462:7032:d700::[0-9a-f]{4}:[0-9a-f]{4}:[0-9a-f]{4}$'
        ([ipaddress]::Parse($hostA.DeterministicLinkLocal)).AddressFamily | Should -Be ([System.Net.Sockets.AddressFamily]::InterNetworkV6)
        ([ipaddress]::Parse($hostA.DeterministicUla)).AddressFamily | Should -Be ([System.Net.Sockets.AddressFamily]::InterNetworkV6)
        $hostA.DeterministicUla | Should -Not -Be $hostB.DeterministicUla
        $hostA.DeterministicLinkLocal | Should -Not -Be $hostB.DeterministicLinkLocal
    }

    It "Should compute commutative pairwise /30 IPv4 subnets for dtm-carbon-two and dtm-super-nuc and disjoint /30 subnets when attaching a 3rd peer" {
        $carbonToNuc = Get-ThunderboltPeerIdentity -ComputerName 'dtm-carbon-two' -PeerComputerName 'dtm-super-nuc'
        $nucToCarbon = Get-ThunderboltPeerIdentity -ComputerName 'dtm-super-nuc' -PeerComputerName 'dtm-carbon-two'

        $carbonToNuc.PairwiseIPv4.SubnetPrefix | Should -BeExactly $nucToCarbon.PairwiseIPv4.SubnetPrefix
        $carbonToNuc.PairwiseIPv4.LocalIPv4Address | Should -BeExactly $nucToCarbon.PairwiseIPv4.PeerIPv4Address
        $carbonToNuc.PairwiseIPv4.PeerIPv4Address | Should -BeExactly $nucToCarbon.PairwiseIPv4.LocalIPv4Address

        # Attaching a 3rd machine (dtm-carbon-two) to dtm-p1gen7 while dtm-p1gen9 is also connected
        $p1gen7ToP1gen9  = Get-ThunderboltPeerIdentity -ComputerName 'dtm-p1gen7' -PeerComputerName 'dtm-p1gen9'
        $p1gen7ToCarbon2 = Get-ThunderboltPeerIdentity -ComputerName 'dtm-p1gen7' -PeerComputerName 'dtm-carbon-two'
        $p1gen7ToP1gen9.PairwiseIPv4.SubnetPrefix | Should -Not -Be $p1gen7ToCarbon2.PairwiseIPv4.SubnetPrefix
    }
}

# ─── Register-ThunderboltPeer ────────────────────────────────────────────────

Describe "Register-ThunderboltPeer" -Tag 'Unit', 'Drivers', 'Thunderbolt', 'Fast', 'Portable' {
    Context "When multiple USB4 P2P adapters are active simultaneously on one host" {
        BeforeAll {
            Mock Get-ThunderboltNetworkStatus {
                @(
                    [PSCustomObject]@{
                        InterfaceAlias     = 'Ethernet 18'
                        InterfaceIndex     = 95
                        Name               = 'USB4(TM) P2P Network Adapter #4'
                        Description        = 'USB4(TM) P2P Network Adapter #4'
                        Role               = 'P2PNetwork'
                        NetEnabled         = $true
                        LinkSpeedGbps      = 20.0
                        MacAddress         = '34-4D-A1-C4-C4-22'
                        IPv4Addresses      = @('169.254.43.202')
                        IPv6Addresses      = @('fe80::364d:a1ff:fec4:c422')
                        NeighborCandidates = @()
                        RecommendedActions = @()
                    },
                    [PSCustomObject]@{
                        InterfaceAlias     = 'Ethernet 19'
                        InterfaceIndex     = 96
                        Name               = 'USB4(TM) P2P Network Adapter #5'
                        Description        = 'USB4(TM) P2P Network Adapter #5'
                        Role               = 'P2PNetwork'
                        NetEnabled         = $true
                        LinkSpeedGbps      = 40.0
                        MacAddress         = '34-4D-A1-C4-C4-23'
                        IPv4Addresses      = @('169.254.88.10')
                        IPv6Addresses      = @('fe80::364d:a1ff:fec4:c423')
                        NeighborCandidates = @()
                        RecommendedActions = @()
                    }
                )
            } -ModuleName PC-AI.Drivers
        }

        It "Should plan all active USB4 P2P adapters simultaneously without throwing a single-adapter error" {
            $plan = Register-ThunderboltPeer -ComputerName 'dtm-p1gen7' -UpdateHostsFile -EnableSmbDirect
            $plan.Applied | Should -BeFalse
            $plan.AdapterCount | Should -Be 2
            $plan.AdapterPlans.Count | Should -Be 2
            $plan.AdapterPlans[0].InterfaceIndex | Should -Be 95
            $plan.AdapterPlans[1].InterfaceIndex | Should -Be 96
            ($plan.PlannedActions | Where-Object Step -eq 'BindLocalUlaIPv6').Count | Should -BeGreaterOrEqual 2
        }
    }
}

# ─── Module Export Completeness ──────────────────────────────────────────────

Describe "PC-AI.Drivers Module Exports" -Tag 'Unit', 'Drivers', 'Fast', 'Portable' {
    It "Should export all 13 declared functions" {
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
            'Set-ThunderboltNetworkOptimization',
            'Get-ThunderboltPeerIdentity',
            'Register-ThunderboltPeer',
            'Invoke-ThunderboltPeerLifecycle'
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


# ─── Invoke-ThunderboltPeerLifecycle ─────────────────────────────────────────

Describe "Invoke-ThunderboltPeerLifecycle" -Tag 'Unit', 'Drivers', 'Thunderbolt', 'Fast', 'Portable' {
    Context "Additive power capabilities and lifecycle state on Get-ThunderboltNetworkStatus" {
        BeforeAll {
            Mock Get-CimInstance {
                param($Namespace, $ClassName)
                switch ($ClassName) {
                    'Win32_NetworkAdapter' {
                        @([PSCustomObject]@{
                            Name                = 'USB4(TM) P2P Network Adapter #2'
                            Description         = 'USB4(TM) P2P Network Adapter #2'
                            NetConnectionID     = 'Ethernet 4'
                            NetEnabled          = $true
                            NetConnectionStatus = 2
                            Speed               = 20000000000
                            MACAddress          = '08-93-C1-7B-39-51'
                            PNPDeviceID         = 'USB4\VEN_045E&DEV_0000083F&PROT_USB4NET\4&6E3A7BC&2&0'
                            Index               = 15
                            InterfaceIndex      = 22
                        })
                    }
                    'Win32_NetworkAdapterConfiguration' {
                        @([PSCustomObject]@{
                            Index     = 15
                            IPAddress = @('169.254.196.28', 'fe80::d700:9', 'fd00:7462:7032::9')
                        })
                    }
                    'MSFT_NetIPInterface' {
                        @(
                            [PSCustomObject]@{
                                InterfaceAlias  = 'Ethernet 4'
                                AddressFamily   = 2
                                InterfaceMetric = 5
                                NlMtu           = 62000
                                AutomaticMetric = $false
                            }
                        )
                    }
                    'MSFT_NetNeighbor' {
                        @([PSCustomObject]@{
                            InterfaceAlias   = 'Ethernet 4'
                            IPAddress        = 'fe80::d700:7'
                            LinkLayerAddress = '34-4D-A1-C4-C4-22'
                            State            = 'Reachable'
                        })
                    }
                    'MSFT_NetAdapterPowerManagementSettingData' {
                        @([PSCustomObject]@{
                            Name              = 'Ethernet 4'
                            SelectiveSuspend  = 2
                            WakeOnMagicPacket = 0
                            WakeOnPattern     = 0
                            ArpOffload        = 0
                            NSOffload         = 0
                        })
                    }
                    default { return @() }
                }
            } -ModuleName PC-AI.Drivers
        }

        It "Should report SelectiveSuspend=Enabled, Wake/Offload=Unsupported, AlwaysOnTransitCapable=false, and expiring evidence" {
            $status = @(Get-ThunderboltNetworkStatus -EvidenceTtlSeconds 300)
            $status.Count | Should -Be 1
            $status[0].SelectiveSuspend | Should -Be 'Enabled'
            $status[0].WakeOnMagicPacket | Should -Be 'Unsupported'
            $status[0].WakeOnPattern | Should -Be 'Unsupported'
            $status[0].ArpOffload | Should -Be 'Unsupported'
            $status[0].NSOffload | Should -Be 'Unsupported'
            $status[0].AlwaysOnTransitCapable | Should -BeFalse
            $status[0].TransitClaimProhibitionReasons.Count | Should -BeGreaterOrEqual 3
            $status[0].LifecycleState | Should -Be 'ready'
            $status[0].EvidenceTtlSeconds | Should -Be 300
            $status[0].EvidenceExpired | Should -BeFalse
        }
    }

    Context "When evaluating topology links, stale routes, and Modern Standby invalidation" {
        BeforeAll {
            Mock Get-ThunderboltNetworkStatus {
                @(
                    [PSCustomObject]@{
                        InterfaceAlias                 = 'Ethernet 4'
                        InterfaceIndex                 = 22
                        Name                           = 'USB4(TM) P2P Network Adapter #2'
                        Description                    = 'USB4(TM) P2P Network Adapter #2'
                        Role                           = 'P2PNetwork'
                        NetEnabled                     = $true
                        NetConnectionStatus            = 2
                        LinkSpeedGbps                  = 20.0
                        MacAddress                     = '08-93-C1-7B-39-51'
                        IPv4Addresses                  = @('169.254.196.28')
                        IPv6Addresses                  = @('fe80::d700:9', 'fd00:7462:7032::9')
                        SelectiveSuspend               = 'Enabled'
                        WakeOnMagicPacket              = 'Unsupported'
                        WakeOnPattern                  = 'Unsupported'
                        ArpOffload                     = 'Unsupported'
                        NSOffload                      = 'Unsupported'
                        AlwaysOnTransitCapable         = $false
                        TransitClaimProhibitionReasons = @('SelectiveSuspend=Enabled')
                        LifecycleState                 = 'ready'
                        NeighborCandidates             = @(
                            [PSCustomObject]@{
                                IPAddress        = 'fe80::d700:7'
                                LinkLayerAddress = '34-4D-A1-C4-C4-22'
                                State            = 'Reachable'
                                PeerHostname     = 'dtm-p1gen7'
                            }
                        )
                        RecommendedActions             = @()
                    }
                )
            } -ModuleName PC-AI.Drivers

            Mock Get-CimInstance {
                param($Namespace, $ClassName)
                if ($ClassName -eq 'MSFT_NetRoute') {
                    return @(
                        [PSCustomObject]@{
                            InterfaceIndex    = 22
                            DestinationPrefix = 'fd00:7462:7032::7/128'
                            RouteMetric       = 5
                        },
                        [PSCustomObject]@{
                            InterfaceIndex    = 22
                            DestinationPrefix = 'fd00:7462:7032::1/128'
                            RouteMetric       = 5
                        },
                        [PSCustomObject]@{
                            InterfaceIndex    = 22
                            DestinationPrefix = 'fd00:7462:7032::21/128'
                            RouteMetric       = 5
                        },
                        [PSCustomObject]@{
                            InterfaceIndex    = 22
                            DestinationPrefix = '172.31.251.14/32'
                            RouteMetric       = 5
                        }
                    )
                }
                return @()
            } -ModuleName PC-AI.Drivers

            Mock Get-WinEvent {
                $now = [datetime]::UtcNow
                @(
                    [PSCustomObject]@{
                        Id          = 507
                        TimeCreated = $now.AddSeconds(-30)
                        Message     = "The system is exiting connected standby `r`n`r`nReason: Input Touchpad."
                    },
                    [PSCustomObject]@{
                        Id          = 506
                        TimeCreated = $now.AddSeconds(-120)
                        Message     = "The system is entering connected standby `r`n`r`nReason: Idle Timeout."
                    }
                )
            } -ModuleName PC-AI.Drivers
        }

        It "Should mark gen7-gen9 ready, ms01-plugable disconnected, preserve Milly bridge, withdraw stale ms01/canary routes, and never claim always-on transit" {
            $report = Invoke-ThunderboltPeerLifecycle -ComputerName 'dtm-p1gen9' -EvidenceObservedAtUtc ([datetime]::UtcNow)
            $report.AnyAlwaysOnTransitClaim | Should -BeFalse
            $report.Ms01UplinkCapMbps | Should -Be 1000
            $report.ModernStandby.LastEnterReason | Should -Be 'Idle Timeout.'
            $report.ModernStandby.LastExitReason | Should -Be 'Input Touchpad.'

            $gen7Link = $report.Links | Where-Object LinkId -eq 'gen7-gen9-usb4-hub-chain'
            $gen7Link.State | Should -Be 'ready'
            $gen7Link.AlwaysOnTransitCapable | Should -BeFalse
            $gen7Link.PreserveExistingPair | Should -BeTrue
            $gen7Link.Canary.IdentityVerified | Should -BeTrue

            $plugableLink = $report.Links | Where-Object LinkId -eq 'ms01-plugable-dock-usb4-routed'
            $plugableLink.State | Should -Be 'disconnected'
            $plugableLink.DownstreamLinkStatus | Should -Be 'none'
            $plugableLink.WindowsXDomainPresent | Should -BeFalse

            $millyLink = $report.Links | Where-Object LinkId -eq 'ms01-milly-usb4-bridge'
            $millyLink.PreserveBridgeCustody | Should -BeTrue
            $millyLink.State | Should -Be 'ready'

            $gen7Route = $report.RouteActions | Where-Object DestinationPrefix -eq 'fd00:7462:7032::7/128'
            $gen7Route.Action | Should -Be 'PreserveActiveOrBridgeRoute'

            $millyRoute = $report.RouteActions | Where-Object DestinationPrefix -eq 'fd00:7462:7032::21/128'
            $millyRoute.Action | Should -Be 'PreserveActiveOrBridgeRoute'

            $ms01Route = $report.RouteActions | Where-Object DestinationPrefix -eq 'fd00:7462:7032::1/128'
            $ms01Route.Action | Should -Be 'WithdrawStaleOwnedRoute'

            $staleCanaryRoute = $report.RouteActions | Where-Object DestinationPrefix -eq '172.31.251.14/32'
            $staleCanaryRoute.Action | Should -Be 'WithdrawStaleOwnedRoute'
        }

        It "Should invalidate stale pre-standby evidence when Event 506 occurred after EvidenceObservedAtUtc and downgrade ready to asleep" {
            $preStandbyTime = [datetime]::UtcNow.AddMinutes(-10)
            $report = Invoke-ThunderboltPeerLifecycle -ComputerName 'dtm-p1gen9' -EvidenceObservedAtUtc $preStandbyTime
            $gen7Link = $report.Links | Where-Object LinkId -eq 'gen7-gen9-usb4-hub-chain'
            $gen7Link.Evidence.InvalidatedByModernStandby | Should -BeTrue
            $gen7Link.Evidence.Expired | Should -BeTrue
            $gen7Link.State | Should -Be 'asleep'
        }

        It "Should enforce bounded resume reprobe budget when ReprobeOnResume is requested" {
            $preStandbyTime = [datetime]::UtcNow.AddMinutes(-10)
            $report = Invoke-ThunderboltPeerLifecycle -ComputerName 'dtm-p1gen9' -EvidenceObservedAtUtc $preStandbyTime -ReprobeOnResume -MaxReprobeAttempts 2 -ReprobeTimeoutMs 100 -TotalReprobeBudgetMs 500
            $report.ResumeReprobe | Should -Not -BeNullOrEmpty
            $report.ResumeReprobe.MaxAttempts | Should -Be 2
            $report.ResumeReprobe.PerAttemptTimeoutMs | Should -Be 100
            $report.ResumeReprobe.TotalBudgetMs | Should -Be 500
            $report.ResumeReprobe.AttemptsExecuted | Should -BeLessOrEqual 2
        }
    }

    Context "Negative test: identity-bound canary rejects spoofing/unexpected neighbor MAC" {
        BeforeAll {
            Mock Get-ThunderboltNetworkStatus {
                @(
                    [PSCustomObject]@{
                        InterfaceAlias                 = 'Ethernet 4'
                        InterfaceIndex                 = 22
                        Name                           = 'USB4(TM) P2P Network Adapter #2'
                        Description                    = 'USB4(TM) P2P Network Adapter #2'
                        Role                           = 'P2PNetwork'
                        NetEnabled                     = $true
                        NetConnectionStatus            = 2
                        LinkSpeedGbps                  = 20.0
                        MacAddress                     = '08-93-C1-7B-39-51'
                        IPv4Addresses                  = @('169.254.196.28')
                        IPv6Addresses                  = @('fe80::d700:9')
                        SelectiveSuspend               = 'Enabled'
                        WakeOnMagicPacket              = 'Unsupported'
                        WakeOnPattern                  = 'Unsupported'
                        ArpOffload                     = 'Unsupported'
                        NSOffload                      = 'Unsupported'
                        AlwaysOnTransitCapable         = $false
                        TransitClaimProhibitionReasons = @('SelectiveSuspend=Enabled')
                        LifecycleState                 = 'probing'
                        NeighborCandidates             = @(
                            [PSCustomObject]@{
                                IPAddress        = 'fe80::d700:7'
                                LinkLayerAddress = 'DE-AD-BE-EF-00-01'
                                State            = 'Reachable'
                                PeerHostname     = 'unknown-rogue-host'
                            }
                        )
                        RecommendedActions             = @()
                    }
                )
            } -ModuleName PC-AI.Drivers

            Mock Get-WinEvent { return @() } -ModuleName PC-AI.Drivers
            Mock Get-CimInstance { return @() } -ModuleName PC-AI.Drivers
        }

        It "Should keep link in 'probing' and refuse 'ready' when neighbor identity/MAC does not match expected peer" {
            $report = Invoke-ThunderboltPeerLifecycle -ComputerName 'dtm-p1gen9' -ReprobeTimeoutMs 50
            $gen7Link = $report.Links | Where-Object LinkId -eq 'gen7-gen9-usb4-hub-chain'
            $gen7Link.Canary.Reachable | Should -BeTrue
            $gen7Link.Canary.IdentityVerified | Should -BeFalse
            $gen7Link.State | Should -Be 'probing'
        }
    }

    Context "5 USB4 Acceptance Cases and Lenovo 40BA Dock Firmware Inventory" {
        BeforeAll {
            Mock Get-WinEvent { return @() } -ModuleName PC-AI.Drivers
            Mock Get-CimInstance { return @() } -ModuleName PC-AI.Drivers
        }

        It "Case 1: Rejects InstanceId prefix-only classification when DEVPKEY_Device_Parent ancestry is missing" {
            Mock Get-ThunderboltNetworkStatus {
                @(
                    [PSCustomObject]@{
                        InterfaceAlias                 = 'Ethernet 4'
                        InterfaceIndex                 = 22
                        Name                           = 'USB4(TM) P2P Network Adapter #2'
                        Description                    = 'USB4(TM) P2P Network Adapter #2'
                        Role                           = 'P2PNetwork'
                        NetEnabled                     = $true
                        NetConnectionStatus            = 2
                        LinkSpeedGbps                  = 20.0
                        MacAddress                     = '08-93-C1-7B-39-51'
                        PnpDeviceId                    = 'USB4\VEN_045E&DEV_0000083F&PROT_USB4NET\4&6E3A7BC&2&0'
                        PnpAncestry                    = @('USB4\VEN_045E&DEV_0000083F&PROT_USB4NET\4&6E3A7BC&2&0')
                        RootRouterId                   = $null
                        IntermediateDockRouters        = @()
                        TopologyPathClass              = 'unknown'
                        IPv4Addresses                  = @('169.254.196.28')
                        IPv6Addresses                  = @('fe80::d700:9')
                        SelectiveSuspend               = 'Enabled'
                        WakeOnMagicPacket              = 'Unsupported'
                        WakeOnPattern                  = 'Unsupported'
                        ArpOffload                     = 'Unsupported'
                        NSOffload                      = 'Unsupported'
                        AlwaysOnTransitCapable         = $false
                        TransitClaimProhibitionReasons = @('SelectiveSuspend=Enabled')
                        LifecycleState                 = 'ready'
                        NeighborCandidates             = @(
                            [PSCustomObject]@{
                                IPAddress        = 'fe80::d700:7'
                                LinkLayerAddress = '34-4D-A1-C4-C4-22'
                                State            = 'Reachable'
                                PeerHostname     = 'dtm-p1gen7'
                            }
                        )
                        RecommendedActions             = @()
                    }
                )
            } -ModuleName PC-AI.Drivers

            $report = Invoke-ThunderboltPeerLifecycle -ComputerName 'dtm-p1gen9' -ReprobeTimeoutMs 50
            $gen7Link = $report.Links | Where-Object LinkId -eq 'gen7-gen9-usb4-hub-chain'
            $gen7Link.LinkAdmitted | Should -BeFalse
            $gen7Link.AdmissionReason | Should -Be 'pnp_ancestry_prefix_only_rejected'
            $gen7Link.RequiredTopologyQualified | Should -BeFalse
        }

        It "Case 2 & 3 & 4 & 5: Enforces no_observed_xdomain, distinguishes host_port_control vs required_two_hub, preserves ShareTBService/default routes, and codifies 20.0 Gbps nominal display + Lenovo 40BA dock firmware" {
            Mock Get-ThunderboltNetworkStatus {
                @(
                    [PSCustomObject]@{
                        InterfaceAlias                 = 'Ethernet 4'
                        InterfaceIndex                 = 22
                        Name                           = 'USB4(TM) P2P Network Adapter #2'
                        Description                    = 'USB4(TM) P2P Network Adapter #2'
                        Role                           = 'P2PNetwork'
                        NetEnabled                     = $true
                        NetConnectionStatus            = 2
                        LinkSpeedGbps                  = 20.0
                        WindowsReportedLinkSpeedBps    = [uint64]20000000000
                        LinkSpeedIsNominalDriverDisplay = $true
                        LinkSpeedIsHardwareMaximum     = $false
                        MacAddress                     = '08-93-C1-7B-39-51'
                        PnpDeviceId                    = 'USB4\VEN_045E&DEV_0000083F&PROT_USB4NET\4&6E3A7BC&2&0'
                        PnpAncestry                    = @(
                            'USB4\VEN_045E&DEV_0000083F&PROT_USB4NET\4&6E3A7BC&2&0',
                            'USB4\ROOT_DEVICE_ROUTER&VID_8086&PID_5781\0000',
                            'PCI\VEN_8086&DEV_5781&SUBSYS_234117AA&REV_01\4&36A9B02E&0&00E0'
                        )
                        RootRouterId                   = 'USB4\ROOT_DEVICE_ROUTER&VID_8086&PID_5781\0000'
                        IntermediateDockRouters        = @('USB4\VID_17EF&PID_312D\0000')
                        TopologyPathClass              = 'single_hub_chain'
                        IPv4Addresses                  = @('169.254.196.28')
                        IPv6Addresses                  = @('fe80::d700:9')
                        SelectiveSuspend               = 'Enabled'
                        WakeOnMagicPacket              = 'Unsupported'
                        WakeOnPattern                  = 'Unsupported'
                        ArpOffload                     = 'Unsupported'
                        NSOffload                      = 'Unsupported'
                        AlwaysOnTransitCapable         = $false
                        TransitClaimProhibitionReasons = @('SelectiveSuspend=Enabled')
                        LifecycleState                 = 'ready'
                        NeighborCandidates             = @(
                            [PSCustomObject]@{
                                IPAddress        = 'fe80::d700:7'
                                LinkLayerAddress = '34-4D-A1-C4-C4-22'
                                State            = 'Reachable'
                                PeerHostname     = 'dtm-p1gen7'
                            }
                        )
                        RecommendedActions             = @()
                    },
                    [PSCustomObject]@{
                        InterfaceAlias                 = 'Ethernet 5'
                        InterfaceIndex                 = 62
                        Name                           = 'USB4(TM) P2P Network Adapter'
                        Description                    = 'USB4(TM) P2P Network Adapter'
                        Role                           = 'P2PNetwork'
                        NetEnabled                     = $true
                        NetConnectionStatus            = 2
                        LinkSpeedGbps                  = 20.0
                        WindowsReportedLinkSpeedBps    = [uint64]20000000000
                        LinkSpeedIsNominalDriverDisplay = $true
                        LinkSpeedIsHardwareMaximum     = $false
                        MacAddress                     = 'BA-B3-23-70-0F-A5'
                        PnpDeviceId                    = 'USB4\VEN_1D6B&DEV_0001&SUBVEN_0000&SUBDEV_0000&PROT_USB4NET\4&3A9F4D01&0&1'
                        PnpAncestry                    = @(
                            'USB4\VEN_1D6B&DEV_0001&SUBVEN_0000&SUBDEV_0000&PROT_USB4NET\4&3A9F4D01&0&1',
                            'USB4\ROOT_DEVICE_ROUTER&VID_8086&PID_E333\0000',
                            'PCI\VEN_8086&DEV_E333&SUBSYS_234117AA&REV_00\3&11583659&0&3B'
                        )
                        RootRouterId                   = 'USB4\ROOT_DEVICE_ROUTER&VID_8086&PID_E333\0000'
                        IntermediateDockRouters        = @()
                        TopologyPathClass              = 'host_port_control'
                        IPv4Addresses                  = @('169.254.99.212')
                        IPv6Addresses                  = @('fe80::2d23:a515:9d68:5710')
                        SelectiveSuspend               = 'Enabled'
                        WakeOnMagicPacket              = 'Unsupported'
                        WakeOnPattern                  = 'Unsupported'
                        ArpOffload                     = 'Unsupported'
                        NSOffload                      = 'Unsupported'
                        AlwaysOnTransitCapable         = $false
                        TransitClaimProhibitionReasons = @('SelectiveSuspend=Enabled')
                        LifecycleState                 = 'ready'
                        NeighborCandidates             = @(
                            [PSCustomObject]@{
                                IPAddress        = 'fe80::90d7:68ff:fe33:c6c4'
                                LinkLayerAddress = '92-D7-68-33-C6-C4'
                                State            = 'Reachable'
                                PeerHostname     = 'ms01'
                            }
                        )
                        RecommendedActions             = @()
                    }
                )
            } -ModuleName PC-AI.Drivers

            $report = Invoke-ThunderboltPeerLifecycle -ComputerName 'dtm-p1gen9' -ReprobeTimeoutMs 50

            # Case 4: Preserve ShareTBService and default routes
            $report.ShareTbServiceRunning | Should -BeTrue
            $report.DefaultRoutesPreserved | Should -BeTrue

            # Case 1 & 5 + Dock Inventory on gen7-gen9-usb4-hub-chain
            $gen7Link = $report.Links | Where-Object LinkId -eq 'gen7-gen9-usb4-hub-chain'
            $gen7Link.LinkAdmitted | Should -BeTrue
            $gen7Link.AdmissionReason | Should -Be 'admitted_verified_xdomain_and_canary'
            $gen7Link.RequiredTopologyQualified | Should -BeTrue
            $gen7Link.RootRouterId | Should -Be 'USB4\ROOT_DEVICE_ROUTER&VID_8086&PID_5781\0000'
            $gen7Link.LocalRootRouterFw | Should -Be '67.67'
            $gen7Link.DockModel | Should -Be '40BA'
            $gen7Link.DockSerial | Should -Be '1S40BAYJ0019GD'
            $gen7Link.DockStatus | Should -Be 'Factory (0x0)'
            $gen7Link.DockFwPackage | Should -Be '1.0.07'
            $gen7Link.DockFwTarget | Should -Be '1.2.99'
            $gen7Link.DockPdFw | Should -Be '0.0.47'
            $gen7Link.DockTbt5Nvm | Should -Be '62.62'
            $gen7Link.DockDmcFw | Should -Be '0.6.5'
            $gen7Link.WindowsReportedLinkSpeedBps | Should -Be 20000000000
            $gen7Link.LinkSpeedIsNominalDriverDisplay | Should -BeTrue
            $gen7Link.LinkSpeedIsHardwareMaximum | Should -BeFalse
            $gen7Link.MeasuredPayloadGbpsSamples | Should -Be @(7.86, 12.90, 14.12)

            # Case 2: No observed XDomain on ms01-plugable-dock-usb4-routed -> no link admission
            $plugableTwoHub = $report.Links | Where-Object LinkId -eq 'ms01-plugable-dock-usb4-routed'
            $plugableTwoHub.WindowsXDomainPresent | Should -BeFalse
            $plugableTwoHub.LinkAdmitted | Should -BeFalse
            $plugableTwoHub.AdmissionReason | Should -Be 'no_observed_xdomain'
            $plugableTwoHub.RequiredTopologyQualified | Should -BeFalse

            # Case 3: host_port_control on Ethernet 5 (ifIndex=62, PID_E333 FW 11.7) cannot satisfy required_two_hub
            $hostPortControl = $report.Links | Where-Object LinkId -eq 'ms01-plugable-gen9-hostport-control'
            $hostPortControl.InterfaceIndex | Should -Be 62
            $hostPortControl.RootRouterId | Should -Be 'USB4\ROOT_DEVICE_ROUTER&VID_8086&PID_E333\0000'
            $hostPortControl.LocalRootRouterFw | Should -Be '11.7'
            $hostPortControl.RequiredTopologyClass | Should -Be 'required_two_hub'
            $hostPortControl.ObservedTopologyClass | Should -Be 'host_port_control'
            $hostPortControl.QualificationScope | Should -Be 'host_port_control_only'
            $hostPortControl.RequiredTopologyQualified | Should -BeFalse
        }
    }

    Context "Review Fix 1: Scoped route deletion preserves duplicate foreign-interface routes and verifies post-delete removal" {
        BeforeAll {
            Mock Get-WinEvent { return @() } -ModuleName PC-AI.Drivers
            Mock Get-ThunderboltNetworkStatus {
                @(
                    [PSCustomObject]@{
                        InterfaceAlias                 = 'Ethernet 4'
                        InterfaceIndex                 = 22
                        Name                           = 'USB4(TM) P2P Network Adapter #2'
                        Description                    = 'USB4(TM) P2P Network Adapter #2'
                        Role                           = 'P2PNetwork'
                        NetEnabled                     = $true
                        NetConnectionStatus            = 2
                        LinkSpeedGbps                  = 20.0
                        MacAddress                     = '08-93-C1-7B-39-51'
                        PnpDeviceId                    = 'USB4\VEN_045E&DEV_0000083F&PROT_USB4NET\4&6E3A7BC&2&0'
                        PnpAncestry                    = @(
                            'USB4\VEN_045E&DEV_0000083F&PROT_USB4NET\4&6E3A7BC&2&0',
                            'USB4\ROOT_DEVICE_ROUTER&VID_8086&PID_5781\0000'
                        )
                        RootRouterId                   = 'USB4\ROOT_DEVICE_ROUTER&VID_8086&PID_5781\0000'
                        IntermediateDockRouters        = @('USB4\VID_17EF&PID_312D\0000')
                        TopologyPathClass              = 'single_hub_chain'
                        IPv4Addresses                  = @('169.254.196.28')
                        IPv6Addresses                  = @('fe80::d700:9')
                        SelectiveSuspend               = 'Enabled'
                        WakeOnMagicPacket              = 'Unsupported'
                        WakeOnPattern                  = 'Unsupported'
                        ArpOffload                     = 'Unsupported'
                        NSOffload                      = 'Unsupported'
                        AlwaysOnTransitCapable         = $false
                        TransitClaimProhibitionReasons = @('SelectiveSuspend=Enabled')
                        LifecycleState                 = 'ready'
                        NeighborCandidates             = @()
                        RecommendedActions             = @()
                    }
                )
            } -ModuleName PC-AI.Drivers
        }

        It "Should scope route deletion strictly to evaluated USB4 InterfaceIndex (22), preserve duplicate route on foreign InterfaceIndex (99), and verify removal" {
            $script:deletedCalls = [System.Collections.Generic.List[string]]::new()
            $script:routeQueryCount = 0

            Mock Get-CimInstance {
                param($Namespace, $ClassName)
                if ($ClassName -eq 'MSFT_NetRoute') {
                    $script:routeQueryCount++
                    if ($script:routeQueryCount -eq 1) {
                        return @(
                            [PSCustomObject]@{
                                InterfaceIndex    = 22
                                DestinationPrefix = 'fd00:7462:7032::1/128'
                                RouteMetric       = 5
                            },
                            [PSCustomObject]@{
                                InterfaceIndex    = 99
                                DestinationPrefix = 'fd00:7462:7032::1/128'
                                RouteMetric       = 25
                            }
                        )
                    } else {
                        # After scoped deletion on InterfaceIndex 22, only foreign InterfaceIndex 99 remains
                        return @(
                            [PSCustomObject]@{
                                InterfaceIndex    = 99
                                DestinationPrefix = 'fd00:7462:7032::1/128'
                                RouteMetric       = 25
                            }
                        )
                    }
                }
                return @()
            } -ModuleName PC-AI.Drivers

            Mock Invoke-ScopedRouteDelete {
                param($DestinationPrefix, $InterfaceIndex)
                $script:deletedCalls.Add("${DestinationPrefix}@${InterfaceIndex}")
            } -ModuleName PC-AI.Drivers

            $report = Invoke-ThunderboltPeerLifecycle -ComputerName 'dtm-p1gen9' -WithdrawStaleRoutes -Apply -Confirm:$false -ReprobeTimeoutMs 50
            $script:deletedCalls.Count | Should -Be 1
            $script:deletedCalls[0] | Should -Be 'fd00:7462:7032::1/128@22'

            $usb4Action = $report.RouteActions | Where-Object { $_.DestinationPrefix -eq 'fd00:7462:7032::1/128' -and $_.InterfaceIndex -eq 22 }
            $usb4Action.Action | Should -Be 'WithdrawStaleOwnedRoute'
            $usb4Action.Executed | Should -BeTrue
            $usb4Action.Error | Should -BeNullOrEmpty

            $foreignAction = $report.RouteActions | Where-Object { $_.DestinationPrefix -eq 'fd00:7462:7032::1/128' -and $_.InterfaceIndex -eq 99 }
            $foreignAction.Action | Should -Be 'PreserveForeignInterfaceRoute'
            $foreignAction.Executed | Should -BeFalse
        }

        It "Should set Executed=false and record Error when route deletion fails or route remains present in MSFT_NetRoute" {
            Mock Get-CimInstance {
                param($Namespace, $ClassName)
                if ($ClassName -eq 'MSFT_NetRoute') {
                    return @(
                        [PSCustomObject]@{
                            InterfaceIndex    = 22
                            DestinationPrefix = '172.31.251.14/32'
                            RouteMetric       = 5
                        }
                    )
                }
                return @()
            } -ModuleName PC-AI.Drivers

            Mock Invoke-ScopedRouteDelete {
                param($DestinationPrefix, $InterfaceIndex)
                throw "Access is denied deleting $DestinationPrefix on ifIndex $InterfaceIndex"
            } -ModuleName PC-AI.Drivers

            $report = Invoke-ThunderboltPeerLifecycle -ComputerName 'dtm-p1gen9' -WithdrawStaleRoutes -Apply -Confirm:$false -ReprobeTimeoutMs 50
            $failedAction = $report.RouteActions | Where-Object { $_.DestinationPrefix -eq '172.31.251.14/32' -and $_.InterfaceIndex -eq 22 }
            $failedAction.Action | Should -Be 'WithdrawStaleOwnedRoute'
            $failedAction.Executed | Should -BeFalse
            $failedAction.Error | Should -Match 'Access is denied'
        }
    }

    Context "Review Fix 2: Scoped IPv6 Link-Local ScopeId preservation in TCP canary across multiple interfaces" {
        BeforeAll {
            Mock Get-WinEvent { return @() } -ModuleName PC-AI.Drivers
            Mock Get-CimInstance { return @() } -ModuleName PC-AI.Drivers
        }

        It "Should preserve IPv6 ScopeId (%62 vs %22) when connecting to link-local neighbor addresses and expose ScopeId on Canary result" {
            $script:observedScopeIds = [System.Collections.Generic.List[long]]::new()

            Mock Invoke-TcpCanaryConnect {
                param([System.Net.IPAddress]$IPAddress, [int]$Port, [int]$TimeoutMs)
                if ($IPAddress.AddressFamily -eq [System.Net.Sockets.AddressFamily]::InterNetworkV6) {
                    $script:observedScopeIds.Add($IPAddress.ScopeId)
                    if ($IPAddress.ScopeId -eq 62 -and $Port -eq 22) {
                        return [PSCustomObject]@{ Reachable = $true; LatencyMs = 2 }
                    }
                }
                return [PSCustomObject]@{ Reachable = $false; LatencyMs = $null }
            } -ModuleName PC-AI.Drivers

            Mock Get-ThunderboltNetworkStatus {
                @(
                    [PSCustomObject]@{
                        InterfaceAlias                 = 'Ethernet 5'
                        InterfaceIndex                 = 62
                        Name                           = 'USB4(TM) P2P Network Adapter'
                        Description                    = 'USB4(TM) P2P Network Adapter'
                        Role                           = 'P2PNetwork'
                        NetEnabled                     = $true
                        NetConnectionStatus            = 2
                        LinkSpeedGbps                  = 20.0
                        MacAddress                     = 'BA-B3-23-70-0F-A5'
                        PnpDeviceId                    = 'USB4\VEN_1D6B&DEV_0001&SUBVEN_0000&SUBDEV_0000&PROT_USB4NET\4&3A9F4D01&0&1'
                        PnpAncestry                    = @(
                            'USB4\VEN_1D6B&DEV_0001&SUBVEN_0000&SUBDEV_0000&PROT_USB4NET\4&3A9F4D01&0&1',
                            'USB4\ROOT_DEVICE_ROUTER&VID_8086&PID_E333\0000'
                        )
                        RootRouterId                   = 'USB4\ROOT_DEVICE_ROUTER&VID_8086&PID_E333\0000'
                        IntermediateDockRouters        = @()
                        TopologyPathClass              = 'host_port_control'
                        IPv4Addresses                  = @('169.254.99.212')
                        IPv6Addresses                  = @('fe80::2d23:a515:9d68:5710')
                        SelectiveSuspend               = 'Enabled'
                        WakeOnMagicPacket              = 'Unsupported'
                        WakeOnPattern                  = 'Unsupported'
                        ArpOffload                     = 'Unsupported'
                        NSOffload                      = 'Unsupported'
                        AlwaysOnTransitCapable         = $false
                        TransitClaimProhibitionReasons = @('SelectiveSuspend=Enabled')
                        LifecycleState                 = 'ready'
                        NeighborCandidates             = @(
                            [PSCustomObject]@{
                                IPAddress        = 'fe80::90d7:68ff:fe33:c6c4%62'
                                LinkLayerAddress = '92-D7-68-33-C6-C4'
                                State            = 'Reachable'
                                PeerHostname     = 'ms01'
                            }
                        )
                        RecommendedActions             = @()
                    }
                )
            } -ModuleName PC-AI.Drivers

            # Create a minimal registry where ms01 has no ULA so the canary probes the link-local neighbor fe80::90d7:68ff:fe33:c6c4%62
            $scopedRegPath = Join-Path $TestDrive 'scoped-ll-peers.json'
            @{
                schemaVersion = '1.2.0'
                fleetPeers    = @(
                    @{ name = 'dtm-p1gen9'; shortUla = 'fd00:7462:7032::9'; shortLinkLocal = 'fe80::d700:9' },
                    @{ name = 'ms01'; shortLinkLocal = 'fe80::90d7:68ff:fe33:c6c4' }
                )
                topologyLinks = @(
                    @{
                        id                    = 'ms01-plugable-gen9-hostport-control'
                        sourceHost            = 'ms01'
                        targetHost            = 'dtm-p1gen9'
                        targetInterface       = 'Ethernet 5'
                        requiredTopologyClass = 'required_two_hub'
                        observedTopologyClass = 'host_port_control'
                        qualificationScope    = 'host_port_control_only'
                        expectedPeerMacs      = @{ 'dtm-p1gen9' = 'BA-B3-23-70-0F-A5'; 'ms01' = '92-D7-68-33-C6-C4' }
                    }
                )
            } | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $scopedRegPath -Encoding UTF8

            $report = Invoke-ThunderboltPeerLifecycle -ComputerName 'dtm-p1gen9' -RegistryPath $scopedRegPath -ReprobeTimeoutMs 50
            $link = $report.Links | Where-Object LinkId -eq 'ms01-plugable-gen9-hostport-control'
            $link.Canary.ScopeId | Should -Be 62
            $link.Canary.Transport | Should -Be 'tcp/22'
            $link.Canary.Reachable | Should -BeTrue
            $script:observedScopeIds | Should -Contain 62
        }
    }

    Context "Review Fix 3: Observed S0 Low Power Idle capability query (Supported, Unsupported, Unknown)" {
        BeforeAll {
            Mock Get-ThunderboltNetworkStatus { return @() } -ModuleName PC-AI.Drivers
            Mock Get-CimInstance { return @() } -ModuleName PC-AI.Drivers
        }

        It "Should report S0LowPowerIdleState=Supported and S0LowPowerIdleConnectedSupported=true when powercfg reports S0 Network Connected" {
            Mock Get-WinEvent { return @() } -ModuleName PC-AI.Drivers
            Mock Get-ModernStandbyCapability {
                return @"
The following sleep states are available on this system:
    Standby (S0 Low Power Idle) Network Connected
    Hibernate

The following sleep states are not available on this system:
    Standby (S3)
"@
            } -ModuleName PC-AI.Drivers

            $report = Invoke-ThunderboltPeerLifecycle -ComputerName 'dtm-p1gen9'
            $report.ModernStandby.S0LowPowerIdleState | Should -Be 'Supported'
            $report.ModernStandby.S0LowPowerIdleConnectedSupported | Should -BeTrue
        }

        It "Should report S0LowPowerIdleState=Unsupported and S0LowPowerIdleConnectedSupported=false on S3 desktop hardware" {
            Mock Get-WinEvent { return @() } -ModuleName PC-AI.Drivers
            Mock Get-ModernStandbyCapability {
                return @"
The following sleep states are available on this system:
    Standby (S3)
    Hibernate

The following sleep states are not available on this system:
    Standby (S0 Low Power Idle)
        The system firmware does not support this standby state.
"@
            } -ModuleName PC-AI.Drivers

            $report = Invoke-ThunderboltPeerLifecycle -ComputerName 'dtm-p1gen9'
            $report.ModernStandby.S0LowPowerIdleState | Should -Be 'Unsupported'
            $report.ModernStandby.S0LowPowerIdleConnectedSupported | Should -BeFalse
        }

        It "Should report S0LowPowerIdleState=Unknown, S0LowPowerIdleConnectedSupported=null, and EventLogAvailable=false when queries are unavailable" {
            Mock Get-WinEvent { throw "Access denied reading System event log" } -ModuleName PC-AI.Drivers
            Mock Get-ModernStandbyCapability { return $null } -ModuleName PC-AI.Drivers

            $report = Invoke-ThunderboltPeerLifecycle -ComputerName 'dtm-p1gen9'
            $report.ModernStandby.S0LowPowerIdleState | Should -Be 'Unknown'
            $report.ModernStandby.S0LowPowerIdleConnectedSupported | Should -BeNullOrEmpty
            $report.ModernStandby.EventLogAvailable | Should -BeFalse
        }
    }
}

