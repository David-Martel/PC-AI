# Deep dive on I2C / HID filter chain / power management for the touchpad stack
$ErrorActionPreference = 'Continue'
$root = $PSScriptRoot
$out  = Join-Path $root '20-i2c-deep-dive.txt'
"# I2C / HID filter / power deep-dive $(Get-Date -Format o)" | Set-Content $out

# 1. Full HID filter driver chain enumeration
"`n## 1. HID filter chain (UpperFilters/LowerFilters across HID stack)" | Add-Content $out
$hidClasses = @(
    'HKLM:\SYSTEM\CurrentControlSet\Control\Class\{745a17a0-74d3-11d0-b6fe-00a0c90f57da}',  # HIDClass
    'HKLM:\SYSTEM\CurrentControlSet\Control\Class\{4d36e96f-e325-11ce-bfc1-08002be10318}',  # Mouse
    'HKLM:\SYSTEM\CurrentControlSet\Control\Class\{4d36e978-e325-11ce-bfc1-08002be10318}',  # Ports/I2C
    'HKLM:\SYSTEM\CurrentControlSet\Control\Class\{4d36e97d-e325-11ce-bfc1-08002be10318}'   # System (covers iaLPSS2)
)
foreach ($k in $hidClasses) {
    "`n### $k" | Add-Content $out
    if (Test-Path $k) {
        $props = Get-ItemProperty -Path $k -ErrorAction SilentlyContinue
        if ($props) {
            "Class : $($props.Class)" | Add-Content $out
            "UpperFilters: $($props.UpperFilters -join ',')" | Add-Content $out
            "LowerFilters: $($props.LowerFilters -join ',')" | Add-Content $out
        }
    }
}

# 2. Touchpad device tree - parents, children, related devices
"`n## 2. Touchpad device hierarchy (parent + siblings on same I2C bus)" | Add-Content $out
$tpDevices = Get-PnpDevice -ErrorAction SilentlyContinue | Where-Object {
    $_.InstanceId -like '*SNSL002D*' -or $_.InstanceId -like '*ETDHSA*' -or
    $_.Service -eq 'iaLPSS2_I2C_MTL' -or
    $_.InstanceId -match 'PCI\\VEN_8086&DEV_(7E78|7E50|7D03)'
}
foreach ($d in $tpDevices) {
    "`n### $($d.FriendlyName) ($($d.InstanceId))" | Add-Content $out
    "  Status: $($d.Status) / Class: $($d.Class) / Service: $($d.Service)" | Add-Content $out
    $props = Get-PnpDeviceProperty -InstanceId $d.InstanceId -KeyName `
        'DEVPKEY_Device_Parent','DEVPKEY_Device_Children','DEVPKEY_Device_Siblings',`
        'DEVPKEY_Device_BusReportedDeviceDesc','DEVPKEY_Device_LocationInfo',`
        'DEVPKEY_Device_PowerData','DEVPKEY_Device_PowerRelations',`
        'DEVPKEY_Device_DriverInfPath','DEVPKEY_Device_DriverVersion',`
        'DEVPKEY_Device_Service' -ErrorAction SilentlyContinue
    foreach ($p in $props) {
        $val = $p.Data
        if ($val -is [array]) { $val = $val -join '; ' }
        "  $($p.KeyName) = $val" | Add-Content $out
    }
}

# 3. Selective suspend / runtime power management on touchpad device
"`n## 3. Power management settings (selective suspend / D-state)" | Add-Content $out
$instances = @(
    'ACPI\SNSL002D\4&39979B3E&0',
    'HID\SNSL002D&COL01\5&14B88203&0&0000',
    'HID\SNSL002D&COL02\5&14B88203&0&0001',
    'HID\SNSL002D&COL03\5&14B88203&0&0002',
    'HID\SNSL002D&COL04\5&14B88203&0&0003'
)
foreach ($i in $instances) {
    "`n### $i" | Add-Content $out
    # Get the registry path for the device's parameters
    $devKey = "HKLM:\SYSTEM\CurrentControlSet\Enum\$i"
    if (Test-Path $devKey) {
        $devProps = Get-ItemProperty -Path $devKey -ErrorAction SilentlyContinue
        "  Service        : $($devProps.Service)" | Add-Content $out
        "  Driver         : $($devProps.Driver)" | Add-Content $out
        "  ConfigFlags    : $($devProps.ConfigFlags)" | Add-Content $out
        "  CapabilityFlags: $($devProps.CapabilityFlags)" | Add-Content $out

        # Check Device Parameters subkey for power settings
        $paramsKey = "$devKey\Device Parameters"
        if (Test-Path $paramsKey) {
            $params = Get-ItemProperty -Path $paramsKey -ErrorAction SilentlyContinue
            "  Device Parameters:" | Add-Content $out
            $params.PSObject.Properties | Where-Object {
                $_.Name -notmatch '^PS' -and $_.Name -notmatch 'IdleTime'
            } | ForEach-Object {
                "    $($_.Name) = $($_.Value)" | Add-Content $out
            }
        }
        # Check Power management
        $pmKey = "$devKey\Device Parameters\WDF"
        if (Test-Path $pmKey) {
            "  WDF Power:" | Add-Content $out
            (Get-ItemProperty -Path $pmKey -ErrorAction SilentlyContinue).PSObject.Properties |
                Where-Object Name -notmatch '^PS' |
                ForEach-Object { "    $($_.Name) = $($_.Value)" | Add-Content $out }
        }
    }
}

# 4. I2C controller power settings
"`n## 4. I2C Host Controller power settings" | Add-Content $out
$i2cControllers = Get-PnpDevice -ErrorAction SilentlyContinue | Where-Object { $_.Service -eq 'iaLPSS2_I2C_MTL' }
foreach ($c in $i2cControllers) {
    "`n### $($c.InstanceId)" | Add-Content $out
    $cKey = "HKLM:\SYSTEM\CurrentControlSet\Enum\$($c.InstanceId)"
    if (Test-Path $cKey) {
        $params = Get-ItemProperty -Path "$cKey\Device Parameters" -ErrorAction SilentlyContinue
        if ($params) {
            $params.PSObject.Properties | Where-Object Name -notmatch '^PS' |
                ForEach-Object { "  $($_.Name) = $($_.Value)" | Add-Content $out }
        }
    }
}

# 5. Look for I2C-specific event logs
"`n## 5. Available I2C / HID / WUDF analytic-debug log channels (often disabled)" | Add-Content $out
try {
    $logs = Get-WinEvent -ListLog * -ErrorAction SilentlyContinue | Where-Object {
        $_.LogName -match 'I2C|hidi2c|HidClass|WUDFHost|HidEventLog|UserModeDriverFramework|DriverFrameworks|DeviceSetup|DeviceManagement|InputCore'
    }
    $logs | Sort-Object LogName | Format-Table LogName, IsEnabled, RecordCount -AutoSize | Out-String | Add-Content $out
} catch {
    "ERROR: $($_.Exception.Message)" | Add-Content $out
}

# 6. WUDFHost log (analytic) - may have I2C/touchpad-specific entries
"`n## 6. UserModeDriverFramework Operational/Analytic events last 4h, filter HID/I2C/touch" | Add-Content $out
try {
    $cutoff = (Get-Date).AddHours(-4)
    $providers = @(
        'Microsoft-Windows-UserModeDriverFramework',
        'Microsoft-Windows-DriverFrameworks-UserMode',
        'Microsoft-Windows-WUDFx02000',
        'Microsoft-Windows-Kernel-PnP'
    )
    foreach ($prov in $providers) {
        try {
            $evts = Get-WinEvent -FilterHashtable @{ProviderName=$prov; StartTime=$cutoff} -ErrorAction SilentlyContinue -MaxEvents 500
            if ($evts) {
                $relevant = $evts | Where-Object {
                    $_.Message -match 'SNSL|Sensel|HID|I2C|touchpad|hidi2c|iaLPSS|ELAS|ACPI\\SNSL|WUDFRd' -and
                    $_.LevelDisplayName -in 'Error','Warning','Critical'
                }
                if ($relevant) {
                    "`n### $prov ($($relevant.Count) relevant)" | Add-Content $out
                    $relevant | Select-Object -First 50 | ForEach-Object {
                        $msg = ($_.Message -split [Environment]::NewLine | Select-Object -First 1)
                        "$($_.TimeCreated) [Id $($_.Id) Lvl $($_.LevelDisplayName)] $msg"
                    } | Add-Content $out
                }
            }
        } catch { }
    }
} catch {
    "ERROR: $($_.Exception.Message)" | Add-Content $out
}

# 7. Reliability monitor - has it caught anything related?
"`n## 7. Reliability Monitor entries last 7d (top failures)" | Add-Content $out
try {
    $rel = Get-WmiObject -Class Win32_ReliabilityRecords -ErrorAction SilentlyContinue |
        Where-Object {
            $msg = $_.Message
            $msg -match 'touchpad|HID|SNSL|Sensel|hidi2c|iaLPSS|input|mouse'
        }
    if ($rel) {
        $rel | Sort-Object TimeGenerated -Descending | Select-Object -First 20 |
            Format-Table TimeGenerated, EventIdentifier, SourceName, ProductName, @{N='Msg';E={$_.Message.Substring(0,[math]::Min(100,$_.Message.Length))}} -AutoSize -Wrap |
            Out-String | Add-Content $out
    } else {
        "(no touchpad/HID-related reliability events in last 7d)" | Add-Content $out
    }
} catch {
    "ERROR: $($_.Exception.Message)" | Add-Content $out
}

# 8. SNSL002D-specific events from any source
"`n## 8. ALL events mentioning SNSL/Sensel/hidi2c/iaLPSS in last 24h (any log)" | Add-Content $out
try {
    $cutoff = (Get-Date).AddHours(-24)
    $allLogs = Get-WinEvent -ListLog * -ErrorAction SilentlyContinue |
        Where-Object { $_.RecordCount -gt 0 -and $_.LogName -match 'System|Application|Microsoft-Windows-Kernel|Microsoft-Windows-DeviceSetup|Microsoft-Windows-DriverFrameworks|Microsoft-Windows-UserMode' } |
        Select-Object -ExpandProperty LogName

    $hits = @()
    foreach ($log in $allLogs) {
        try {
            $events = Get-WinEvent -LogName $log -ErrorAction SilentlyContinue -MaxEvents 2000 |
                Where-Object { $_.TimeCreated -gt $cutoff } |
                Where-Object { $_.Message -match 'SNSL|Sensel|hidi2c|iaLPSS|ELAS_B41A' }
            if ($events) { $hits += $events }
        } catch { }
    }
    "Total hits: $($hits.Count)" | Add-Content $out
    if ($hits) {
        $hits | Sort-Object TimeCreated | Select-Object -First 100 | ForEach-Object {
            $msg = ($_.Message -split [Environment]::NewLine | Select-Object -First 2) -join ' | '
            "$($_.TimeCreated) [$($_.LogName)] [$($_.Id)/$($_.LevelDisplayName)] $msg"
        } | Add-Content $out
    }
} catch {
    "ERROR: $($_.Exception.Message)" | Add-Content $out
}

# 9. Check whether touchpad's "Allow computer to turn off device to save power" is set
"`n## 9. EnhancedPowerManagementEnabled across HID stack" | Add-Content $out
$enumRoot = 'HKLM:\SYSTEM\CurrentControlSet\Enum'
foreach ($i in $instances) {
    $k = Join-Path $enumRoot $i
    if (Test-Path "$k\Device Parameters") {
        $epm = Get-ItemProperty -Path "$k\Device Parameters" -Name 'EnhancedPowerManagementEnabled' -ErrorAction SilentlyContinue
        if ($epm) {
            "$i : EnhancedPowerManagementEnabled = $($epm.EnhancedPowerManagementEnabled)" | Add-Content $out
        } else {
            "$i : (no EnhancedPowerManagementEnabled value set - inheriting default)" | Add-Content $out
        }
        # SelectiveSuspendEnabled lives elsewhere
        $ssEnabled = Get-ItemProperty -Path "$k\Device Parameters" -Name 'SelectiveSuspendEnabled' -ErrorAction SilentlyContinue
        if ($ssEnabled) {
            "$i : SelectiveSuspendEnabled = $($ssEnabled.SelectiveSuspendEnabled)" | Add-Content $out
        }
    }
}
# I2C controllers
foreach ($c in $i2cControllers) {
    $k = Join-Path $enumRoot $c.InstanceId
    if (Test-Path "$k\Device Parameters") {
        $epm = Get-ItemProperty -Path "$k\Device Parameters" -Name 'EnhancedPowerManagementEnabled' -ErrorAction SilentlyContinue
        if ($epm) {
            "$($c.InstanceId) : EnhancedPowerManagementEnabled = $($epm.EnhancedPowerManagementEnabled)" | Add-Content $out
        }
    }
}

# 10. Check kernel-level Storage / Disk I/O events that COULD also represent I2C bus errors
"`n## 10. Kernel-IO + Storage events last 24h, filter to common I2C/HID timing/transfer issues" | Add-Content $out
try {
    $cutoff = (Get-Date).AddHours(-24)
    $kioEvts = Get-WinEvent -FilterHashtable @{LogName='System'; ProviderName='Microsoft-Windows-Kernel-IO'; StartTime=$cutoff} -ErrorAction SilentlyContinue -MaxEvents 500
    if ($kioEvts) {
        "$($kioEvts.Count) Kernel-IO events found" | Add-Content $out
        $kioEvts | Group-Object Id | Sort-Object Count -Descending | Format-Table Count, Name -AutoSize | Out-String | Add-Content $out
    } else {
        "(none)" | Add-Content $out
    }
} catch {
    "(no Kernel-IO log channel: $($_.Exception.Message))" | Add-Content $out
}

# 11. Check for Synaptics filter driver SmiProDrv
"`n## 11. SmbCls / I2C bus filter / Synaptics SmiProDrv presence" | Add-Content $out
$drivers = Get-WindowsDriver -Online -ErrorAction SilentlyContinue | Where-Object {
    $_.OriginalFileName -match 'i2c|hid|smbcls|smbus|smibus|smipro|elan|sensel|trackpad|trackpoint' -or
    $_.Driver -match 'oem.*\.inf' -and $_.ClassName -match 'HIDClass|System|Mouse|Sensor'
}
if ($drivers) {
    $drivers | Format-Table Driver, OriginalFileName, ClassName, ProviderName, Date -AutoSize -Wrap | Out-String | Add-Content $out
}

# 12. Latency-sensitive: check disk I/O contention on the system disk specifically right now
"`n## 12. Current disk queue depth / latency (storport perf counters)" | Add-Content $out
try {
    $samples = Get-Counter -Counter '\PhysicalDisk(*)\Avg. Disk Queue Length','\PhysicalDisk(*)\Avg. Disk sec/Read','\PhysicalDisk(*)\Avg. Disk sec/Write' -SampleInterval 1 -MaxSamples 3 -ErrorAction SilentlyContinue
    foreach ($s in $samples) {
        $s.CounterSamples | Format-Table Path, CookedValue -AutoSize | Out-String | Add-Content $out
    }
} catch {
    "ERROR: $($_.Exception.Message)" | Add-Content $out
}

"`n# Done $(Get-Date -Format o)" | Add-Content $out
Write-Host "Wrote $out"
