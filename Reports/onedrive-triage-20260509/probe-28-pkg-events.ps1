$ErrorActionPreference = 'Continue'
$out = 'C:\codedev\PC_AI\Reports\onedrive-triage-20260509\28-pkg-events.txt'
"=== OneDrive AppX package + PushNotifications event log diagnostics at $(Get-Date -Format o) ===" | Out-File $out

"`n=== AppX package state ===" | Out-File $out -Append
Get-AppxPackage -Name 'Microsoft.OneDriveSync*' -ErrorAction SilentlyContinue |
    Format-List Name, PackageFullName, Architecture, Version, Publisher, PackageUserInformation, Status, IsBundle, IsFramework, IsResourcePackage, NonRemovable, SignatureKind, InstallLocation, ResourceId |
    Out-String | Out-File $out -Append

"`n=== AppX package full status detail ===" | Out-File $out -Append
$pkg = Get-AppxPackage -Name 'Microsoft.OneDriveSync*' -ErrorAction SilentlyContinue | Select-Object -First 1
if ($pkg) {
    "Status: $($pkg.Status)" | Out-File $out -Append
    if ($pkg.PSObject.Properties['Dependencies']) {
        "Dependencies: $($pkg.Dependencies | ForEach-Object { $_.PackageFullName })" | Out-File $out -Append
    }
    if ($pkg.PSObject.Properties['PackageUserInformation']) {
        "User information: $($pkg.PackageUserInformation | Out-String)" | Out-File $out -Append
    }
}

"`n=== PushNotifications-Platform event log (last 24h) ===" | Out-File $out -Append
$providers = @(
    'Microsoft-Windows-PushNotifications-Platform',
    'Microsoft-Windows-PushNotifications-InProc',
    'Microsoft-Windows-PushNotifications-Developer',
    'Microsoft-Windows-WNSCP-Server'
)
foreach ($pv in $providers) {
    "`n--- Provider: $pv ---" | Out-File $out -Append
    try {
        $events = Get-WinEvent -ProviderName $pv -MaxEvents 100 -ErrorAction Stop |
            Where-Object { $_.TimeCreated -gt (Get-Date).AddHours(-24) }
        "Total events 24h: $($events.Count)" | Out-File $out -Append
        $events | Select-Object -First 30 TimeCreated, Id, LevelDisplayName, @{n='Snippet';e={($_.Message -split [Environment]::NewLine)[0]}} |
            Format-Table -AutoSize -Wrap | Out-String -Width 240 | Out-File $out -Append
    } catch {
        "Provider not available: $($_.Exception.Message)" | Out-File $out -Append
    }
}

"`n=== Operational log: PushNotifications/Operational and Analytic ===" | Out-File $out -Append
$logs = @(
    'Microsoft-Windows-PushNotification-Platform/Operational',
    'Microsoft-Windows-PushNotifications-Platform/Operational',
    'Microsoft-Windows-PushNotifications-Platform/Admin'
)
foreach ($lg in $logs) {
    "`n--- Log: $lg ---" | Out-File $out -Append
    try {
        $events = Get-WinEvent -LogName $lg -MaxEvents 100 -ErrorAction Stop |
            Where-Object { $_.TimeCreated -gt (Get-Date).AddHours(-24) }
        "Total: $($events.Count)" | Out-File $out -Append
        $events | Select-Object -First 30 TimeCreated, Id, LevelDisplayName, ProviderName, @{n='Snippet';e={($_.Message -split [Environment]::NewLine)[0]}} |
            Format-Table -AutoSize -Wrap | Out-String -Width 240 | Out-File $out -Append
    } catch {
        "Log not enabled: $($_.Exception.Message)" | Out-File $out -Append
    }
}

"`n=== Anything mentioning OneDriveSync in any event log (last 24h) ===" | Out-File $out -Append
try {
    $logs = Get-WinEvent -ListLog * -ErrorAction SilentlyContinue | Where-Object { $_.RecordCount -gt 0 -and $_.LogName -match 'PushNotif|Wns|WNS' }
    foreach ($lg in $logs) {
        "Log: $($lg.LogName) (records: $($lg.RecordCount), last 24h check)" | Out-File $out -Append
        try {
            $matches = Get-WinEvent -LogName $lg.LogName -MaxEvents 200 -ErrorAction SilentlyContinue |
                Where-Object { ($_.TimeCreated -gt (Get-Date).AddHours(-24)) -and ($_.Message -match 'OneDrive|FileSync|SkyDrive|Microsoft\.OneDriveSync') }
            "  matched: $($matches.Count)" | Out-File $out -Append
            $matches | Select-Object -First 20 TimeCreated, Id, LevelDisplayName, ProviderName, @{n='Snippet';e={($_.Message -split [Environment]::NewLine)[0..2] -join ' | '}} |
                Format-Table -AutoSize -Wrap | Out-String -Width 240 | Out-File $out -Append
        } catch {}
    }
} catch {}

"`n=== System log (related to apps) for OneDrive in last 6h ===" | Out-File $out -Append
try {
    Get-WinEvent -FilterHashtable @{LogName='System'; StartTime=(Get-Date).AddHours(-6)} -ErrorAction Stop |
        Where-Object { $_.Message -match 'OneDrive|OneDriveSync|FileSync|SkyDrive|push notification|WPN|WNS' } |
        Select-Object -First 25 TimeCreated, Id, LevelDisplayName, ProviderName, @{n='Snippet';e={($_.Message -split [Environment]::NewLine)[0]}} |
        Format-Table -AutoSize -Wrap | Out-String -Width 240 | Out-File $out -Append
} catch {
    "System log err: $($_.Exception.Message)" | Out-File $out -Append
}

"`n=== Check if Operational log is enabled ===" | Out-File $out -Append
$logCfg = Get-WinEvent -ListLog '*PushNotif*' -ErrorAction SilentlyContinue
$logCfg | Select-Object LogName, IsEnabled, RecordCount, LogMode | Format-Table -AutoSize | Out-String | Out-File $out -Append

"`n=== Check WPN diagnostic registry ===" | Out-File $out -Append
Get-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Services\WpnService\Parameters' -ErrorAction SilentlyContinue |
    Select-Object * -ExcludeProperty PS* | Format-List | Out-String | Out-File $out -Append

Write-Host "Wrote $out"
