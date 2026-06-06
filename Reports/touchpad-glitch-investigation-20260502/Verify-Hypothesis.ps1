$ErrorActionPreference = 'Continue'
$root = $PSScriptRoot
$out = Join-Path $root '15-hypothesis-verification.txt'
"# Hypothesis verification $(Get-Date -Format o)" | Set-Content $out

# Verification 1: I2C controller actual driver bind date
"`n## V1: I2C Host Controller — DriverDate / DriverVersion / DriverInfPath" | Add-Content $out
try {
    $i2c = Get-PnpDevice -Class System -ErrorAction SilentlyContinue | Where-Object {
        $_.FriendlyName -match 'I2C.*Host Controller|LPSS.*I2C|Serial IO I2C'
    }
    if (-not $i2c) {
        # Fallback: look by service name
        $i2c = Get-PnpDevice -ErrorAction SilentlyContinue | Where-Object { $_.Service -match 'iaLPSS2_I2C' }
    }
    if ($i2c) {
        foreach ($d in $i2c) {
            "InstanceId: $($d.InstanceId)" | Add-Content $out
            "Friendly  : $($d.FriendlyName)" | Add-Content $out
            "Service   : $($d.Service)" | Add-Content $out
            $props = Get-PnpDeviceProperty -InstanceId $d.InstanceId -KeyName 'DEVPKEY_Device_DriverDate','DEVPKEY_Device_DriverVersion','DEVPKEY_Device_DriverInfPath','DEVPKEY_Device_DriverInfSection' -ErrorAction SilentlyContinue
            foreach ($p in $props) {
                "  $($p.KeyName): $($p.Data)" | Add-Content $out
            }
            "" | Add-Content $out
        }
    } else {
        "(no I2C Host Controller found)" | Add-Content $out
    }
} catch {
    "ERROR: $($_.Exception.Message)" | Add-Content $out
}

# Verification 2: WUDFRd 219 events with extracted device IDs (Properties[2])
"`n## V2: WUDFRd 219 events last 7d — actual failing device IDs" | Add-Content $out
try {
    $cutoff = (Get-Date).AddDays(-7)
    $evts = Get-WinEvent -FilterHashtable @{LogName='System'; ProviderName='Microsoft-Windows-Kernel-PnP'; Id=219; StartTime=$cutoff} -ErrorAction Stop
    "Total: $($evts.Count) events" | Add-Content $out

    if ($evts) {
        "`n### Per-event detail (TimeCreated | Status | Device)" | Add-Content $out
        foreach ($e in ($evts | Sort-Object TimeCreated -Descending | Select-Object -First 30)) {
            $status = if ($e.Properties.Count -gt 1 -and $e.Properties[1].Value) { '0x{0:X8}' -f $e.Properties[1].Value } else { '-' }
            $device = if ($e.Properties.Count -gt 2 -and $e.Properties[2].Value) { [string]$e.Properties[2].Value } else { '-' }
            "$($e.TimeCreated) | $status | $device" | Add-Content $out
        }

        "`n### Group by device (count desc)" | Add-Content $out
        $evts | Group-Object {
            if ($_.Properties.Count -gt 2 -and $_.Properties[2].Value) { [string]$_.Properties[2].Value } else { '<unknown>' }
        } | Sort-Object Count -Descending |
        Select-Object Count, Name | Format-Table -AutoSize -Wrap | Out-String | Add-Content $out
    }
} catch {
    "ERROR: $($_.Exception.Message)" | Add-Content $out
}

# Verification 3: DeviceSetupManager activity in last 7d (what actually got installed)
"`n## V3: DeviceSetupManager Admin events last 7d (driver install/update activity)" | Add-Content $out
try {
    $cutoff = (Get-Date).AddDays(-7)
    $dsmLog = 'Microsoft-Windows-DeviceSetupManager/Admin'
    $evts = Get-WinEvent -LogName $dsmLog -MaxEvents 1000 -ErrorAction SilentlyContinue |
        Where-Object { $_.TimeCreated -gt $cutoff }
    if ($evts) {
        "Events found: $($evts.Count)" | Add-Content $out
        $evts | Sort-Object TimeCreated | Select-Object -First 100 |
            ForEach-Object {
                $msg = ($_.Message -split [Environment]::NewLine | Select-Object -First 1)
                "$($_.TimeCreated) [$($_.Id)] $msg"
            } | Add-Content $out
    } else {
        "(no DeviceSetupManager events in last 7d)" | Add-Content $out
    }
} catch {
    "ERROR: $($_.Exception.Message)" | Add-Content $out
}

# Verification 4: PnpInstaller log around 4/30 17:00-22:00
"`n## V4: Kernel-PnP / Drivers events 4/30 16:00 to 5/1 02:00 (regression window)" | Add-Content $out
try {
    $start = [datetime]'2026-04-30 16:00:00'
    $end   = [datetime]'2026-05-01 02:00:00'
    $providers = @('Microsoft-Windows-Kernel-PnP','Microsoft-Windows-DriverFrameworks-UserMode','Microsoft-Windows-WUDFHost')
    foreach ($prov in $providers) {
        "`n### Provider: $prov" | Add-Content $out
        try {
            $e = Get-WinEvent -FilterHashtable @{LogName='System'; ProviderName=$prov; StartTime=$start; EndTime=$end} -ErrorAction SilentlyContinue
            if ($e) {
                "Count: $($e.Count)" | Add-Content $out
                $e | Sort-Object TimeCreated | Select-Object -First 50 |
                    ForEach-Object {
                        $msg = ($_.Message -split [Environment]::NewLine | Select-Object -First 1)
                        "$($_.TimeCreated) [Id $($_.Id) Lvl $($_.LevelDisplayName)] $msg"
                    } | Add-Content $out
            } else {
                "(none)" | Add-Content $out
            }
        } catch {
            "(error: $($_.Exception.Message))" | Add-Content $out
        }
    }
} catch {
    "ERROR: $($_.Exception.Message)" | Add-Content $out
}

# Verification 5: All driver bind dates in the touchpad chain
"`n## V5: All touchpad-chain device DriverDates (sorted recency)" | Add-Content $out
try {
    $patterns = @('SNSL002D','ELAS','ETDHSA','iaLPSS2','VEN_8086.*DEV_(7E|7D|A8|A9)')
    $candidates = Get-PnpDevice -ErrorAction SilentlyContinue | Where-Object {
        $id = $_.InstanceId
        $patterns | ForEach-Object { if ($id -match $_) { return $true } }
    }
    $rows = foreach ($d in $candidates) {
        $props = Get-PnpDeviceProperty -InstanceId $d.InstanceId -KeyName 'DEVPKEY_Device_DriverDate','DEVPKEY_Device_DriverVersion','DEVPKEY_Device_DriverInfPath' -ErrorAction SilentlyContinue
        $date = ($props | Where-Object KeyName -eq 'DEVPKEY_Device_DriverDate').Data
        $ver  = ($props | Where-Object KeyName -eq 'DEVPKEY_Device_DriverVersion').Data
        $inf  = ($props | Where-Object KeyName -eq 'DEVPKEY_Device_DriverInfPath').Data
        [pscustomobject]@{
            Date = $date
            Version = $ver
            Inf = $inf
            Friendly = $d.FriendlyName
            InstanceId = $d.InstanceId
        }
    }
    $rows | Sort-Object Date -Descending | Format-Table -AutoSize -Wrap | Out-String | Add-Content $out
} catch {
    "ERROR: $($_.Exception.Message)" | Add-Content $out
}

"`n# Done $(Get-Date -Format o)" | Add-Content $out
Write-Host "Wrote $out"
