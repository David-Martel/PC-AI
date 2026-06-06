# Pre-plan verification checks
# Validates assumptions before writing remediation plan
$ErrorActionPreference = 'Continue'
$out = "$PSScriptRoot\14-preplan-checks.txt"
"# Pre-plan verification checks ($(Get-Date -Format o))" | Set-Content $out

# 14a: KB removability via dism
"`n## 14a: KB5083631 / recent KBs - removability check (dism Permanent flag)" | Add-Content $out
try {
    $packages = & dism /online /get-packages /format:table 2>&1 | Out-String
    $packages -split "`n" | Where-Object { $_ -match 'KB508|KB5083631|KB5082417|KB5088467' } | Add-Content $out

    "`n### Detailed: KB5083631" | Add-Content $out
    $detail = & dism /online /get-packages /format:list 2>&1 | Out-String
    $blocks = $detail -split '(?=Package Identity)'
    foreach ($b in $blocks) {
        if ($b -match 'KB5083631') {
            $b | Add-Content $out
        }
    }
} catch {
    "ERROR: $($_.Exception.Message)" | Add-Content $out
}

# 14b: existing restore points
"`n## 14b: Existing restore points" | Add-Content $out
try {
    $rps = Get-ComputerRestorePoint -ErrorAction Stop
    if ($rps) {
        $rps | Format-Table SequenceNumber, CreationTime, Description, RestorePointType -AutoSize | Out-String | Add-Content $out
    } else {
        "(no restore points found)" | Add-Content $out
    }
} catch {
    "ERROR: $($_.Exception.Message)" | Add-Content $out
}

# 14c: SystemRestorePointCreationFrequency
"`n## 14c: SystemRestorePointCreationFrequency (minutes between forced points)" | Add-Content $out
try {
    $key = 'HKLM:\Software\Microsoft\Windows NT\CurrentVersion\SystemRestore'
    $val = Get-ItemProperty -Path $key -Name SystemRestorePointCreationFrequency -ErrorAction SilentlyContinue
    if ($val) {
        "Frequency: $($val.SystemRestorePointCreationFrequency) minutes" | Add-Content $out
    } else {
        "(unset - default 1440 minutes / 24h applies)" | Add-Content $out
    }

    $sr = Get-ItemProperty -Path 'HKLM:\Software\Microsoft\Windows NT\CurrentVersion\SystemRestore' -ErrorAction SilentlyContinue
    "Other SystemRestore settings:" | Add-Content $out
    $sr | Format-List | Out-String | Add-Content $out
} catch {
    "ERROR: $($_.Exception.Message)" | Add-Content $out
}

# 14d: WUDFRd 0xC0000365 history - chronic vs new
"`n## 14d: WUDFRd 219 events last 30d - filter to touchpad-relevant (ELAS/SNSL/hidi2c/iaLPSS2)" | Add-Content $out
try {
    $cutoff = (Get-Date).AddDays(-30)
    $evts = Get-WinEvent -FilterHashtable @{LogName='System'; ProviderName='Microsoft-Windows-Kernel-PnP'; Id=219; StartTime=$cutoff} -ErrorAction Stop |
        Where-Object { $_.Message -match 'ELAS|SNSL|hidi2c|iaLPSS2|Sensel|HIDClass|MOUHID' }
    if ($evts) {
        "Touchpad-relevant WUDFRd 219 events in last 30d: $($evts.Count)" | Add-Content $out
        $evts | Group-Object { ($_.Message -split "`n")[0].Substring(0, [math]::Min(120, ($_.Message -split "`n")[0].Length)) } |
            Sort-Object Count -Descending |
            Select-Object Count, Name -First 20 |
            Format-Table -AutoSize -Wrap | Out-String | Add-Content $out

        "`n### Earliest 5 + latest 5 (to gauge chronicity)" | Add-Content $out
        $evts | Sort-Object TimeCreated | Select-Object -First 5 |
            ForEach-Object { "$($_.TimeCreated) | $(($_.Message -split [Environment]::NewLine)[0])" } |
            Add-Content $out
        "---" | Add-Content $out
        $evts | Sort-Object TimeCreated -Descending | Select-Object -First 5 |
            ForEach-Object { "$($_.TimeCreated) | $(($_.Message -split [Environment]::NewLine)[0])" } |
            Add-Content $out
    } else {
        "(no touchpad-relevant WUDFRd 219 events in last 30d - if these only appeared after 5/1 update they ARE signal)" | Add-Content $out
    }

    "`n### Total WUDFRd 219 events in last 30d (any device)" | Add-Content $out
    $allEvts = Get-WinEvent -FilterHashtable @{LogName='System'; ProviderName='Microsoft-Windows-Kernel-PnP'; Id=219; StartTime=$cutoff} -ErrorAction Stop
    "Total: $($allEvts.Count)" | Add-Content $out
} catch {
    "ERROR: $($_.Exception.Message)" | Add-Content $out
}

# 14e: setupapi.dev.log for 5/1 update window
"`n## 14e: setupapi.dev.log entries around 5/1 update window (touchpad/HID/I2C related)" | Add-Content $out
try {
    $logPath = 'C:\Windows\inf\setupapi.dev.log'
    if (Test-Path $logPath) {
        $matches = Select-String -Path $logPath -Pattern 'SNSL002D|hidi2c|iaLPSS2|ETDHSA|ELAS|input\.inf|mtconfig|Sensel|Synaptics' -SimpleMatch -ErrorAction SilentlyContinue |
            Where-Object {
                $line = $_.Line
                $line -match '2026[-/]05[-/]01' -or $line -match '2026[-/]05[-/]02' -or $line -match '05/01/2026' -or $line -match '05/02/2026'
            }

        if ($matches) {
            "Matches found: $($matches.Count)" | Add-Content $out
            $matches | Select-Object -First 80 | ForEach-Object {
                $ln = $_.LineNumber
                $tx = $_.Line
                "${ln}: ${tx}" | Add-Content $out
            }
        } else {
            "(no 5/1-5/2 matches - searching most recent 5000 lines for relevant patterns)" | Add-Content $out
            $tail = Get-Content $logPath -Tail 5000 -ErrorAction SilentlyContinue
            if ($tail) {
                $relevant = $tail | Select-String -Pattern 'SNSL002D|hidi2c|iaLPSS2|ETDHSA|input\.inf' -SimpleMatch
                "Relevant in last 5000 lines: $($relevant.Count)" | Add-Content $out
                $relevant | Select-Object -First 40 | ForEach-Object { $_.Line } | Add-Content $out
            }
        }
    } else {
        "(setupapi.dev.log not found at $logPath)" | Add-Content $out
    }
} catch {
    "ERROR: $($_.Exception.Message)" | Add-Content $out
}

# 14f: rollback feasibility for ETDHSA driver (the only "real" Lenovo HID driver in the chain)
"`n## 14f: ETDHSA driver rollback chain (multiple versions in DriverStore)" | Add-Content $out
try {
    $store = & pnputil /enum-drivers 2>&1 | Out-String
    $blocks = $store -split '(?=Published Name)'
    foreach ($b in $blocks) {
        if ($b -match 'etdhsa\.inf|n48et\.inf|iaLPSS2_I2C_MTL') {
            $b.Trim() | Add-Content $out
            "---" | Add-Content $out
        }
    }
} catch {
    "ERROR: $($_.Exception.Message)" | Add-Content $out
}

"`n# Done - $(Get-Date -Format o)" | Add-Content $out
Write-Host "Wrote: $out"
