$ErrorActionPreference = 'Continue'
$out = 'C:\codedev\PC_AI\Reports\onedrive-triage-20260509\21-config-audit.txt'
"=== OneDrive configuration audit at $(Get-Date -Format o) ===" | Out-File $out

"`n--- HKLM Group Policy keys for OneDrive (could lock features) ---" | Out-File $out -Append
$pol = @(
    'HKLM:\SOFTWARE\Policies\Microsoft\OneDrive',
    'HKLM:\SOFTWARE\Microsoft\OneDrive',
    'HKCU:\Software\Policies\Microsoft\OneDrive'
)
foreach ($k in $pol) {
    "--- $k ---" | Out-File $out -Append
    Get-ItemProperty $k -ErrorAction SilentlyContinue | Select-Object * -ExcludeProperty PS* |
        Format-List | Out-String | Out-File $out -Append
}

"`n--- HKLM\SOFTWARE\Policies\Microsoft\Office (could affect FileCoAuth) ---" | Out-File $out -Append
Get-ItemProperty 'HKLM:\SOFTWARE\Policies\Microsoft\Office' -ErrorAction SilentlyContinue |
    Select-Object * -ExcludeProperty PS* | Format-List | Out-String | Out-File $out -Append

"`n--- Files-On-Demand state and storage sense ---" | Out-File $out -Append
$tenant = Get-ItemProperty 'HKCU:\Software\Microsoft\OneDrive\Accounts\Personal\Tenants\OneDrive' -ErrorAction SilentlyContinue
"Personal tenant root: $($tenant)" | Out-File $out -Append
$ods = Get-ItemProperty 'HKCU:\Software\Microsoft\OneDrive\Accounts\Personal' -ErrorAction SilentlyContinue
"FilesOnDemand: $($ods.FilesOnDemand)" | Out-File $out -Append
"placeholdersEnabled: (see global.ini SavedPlaceholdersEnabledState)" | Out-File $out -Append

"`n--- Storage Sense state ---" | Out-File $out -Append
$ssKey = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\StorageSense\Parameters\StoragePolicy'
Get-ItemProperty $ssKey -ErrorAction SilentlyContinue | Select-Object * -ExcludeProperty PS* |
    Format-List | Out-String | Out-File $out -Append

"`n--- KFM (Known Folder Move) status ---" | Out-File $out -Append
$kfm = Get-ItemProperty 'HKCU:\Software\Microsoft\OneDrive\Accounts\Personal' -ErrorAction SilentlyContinue
"KfmFoldersProtectedNow: $($kfm.KfmFoldersProtectedNow) (bitfield: 1=Desktop,2=Pictures,4=Documents,8=Music,16=Videos)" | Out-File $out -Append
"LastKFMOptInTime: $($kfm.LastKFMOptInTime)" | Out-File $out -Append

"`n--- KFM.db (Known Folder Move state) ---" | Out-File $out -Append
$kfmDb = "$env:LOCALAPPDATA\Microsoft\OneDrive\settings\Personal\KFM.db"
if (Test-Path $kfmDb) {
    $kfmCopy = 'C:\codedev\PC_AI\Reports\onedrive-triage-20260509\KFM.copy.db'
    Copy-Item $kfmDb $kfmCopy -Force -ErrorAction SilentlyContinue
    "KFM.db tables:" | Out-File $out -Append
    sqlite3.exe $kfmCopy '.tables' 2>&1 | Out-File $out -Append
    "KFM.db schema:" | Out-File $out -Append
    sqlite3.exe $kfmCopy '.schema' 2>&1 | Out-File $out -Append
    "KFM contents (first 30 rows per table):" | Out-File $out -Append
    $tables = (sqlite3.exe $kfmCopy "SELECT name FROM sqlite_master WHERE type='table';" 2>&1) -split "`n"
    foreach ($t in $tables) {
        $t = $t.Trim(); if (-not $t) { continue }
        "--- $t ---" | Out-File $out -Append
        sqlite3.exe $kfmCopy ".mode line" "SELECT * FROM `"$t`" LIMIT 30;" 2>&1 | Out-File $out -Append
    }
}

"`n--- Storage Provider Sync Roots ---" | Out-File $out -Append
$srKey = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Explorer\StorageProviders\SyncRootManager'
Get-ChildItem $srKey -ErrorAction SilentlyContinue | ForEach-Object {
    "Root: $($_.PSChildName)" | Out-File $out -Append
    Get-ItemProperty $_.PSPath | Select-Object DisplayNameResource, Description, ProviderName, IsBackedByEnterpriseCloud, ShellRoot |
        Format-List | Out-String | Out-File $out -Append
}

"`n--- File explorer sync overlay ---" | Out-File $out -Append
$shex = 'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\ShellIconOverlayIdentifiers'
Get-ChildItem $shex -ErrorAction SilentlyContinue | Where-Object { $_.PSChildName -match 'OneDrive|SkyDrive' } |
    Select-Object PSChildName, @{n='CLSID';e={(Get-ItemProperty $_.PSPath).'(default)'}} |
    Format-Table -AutoSize | Out-String | Out-File $out -Append

"`n--- OneDrive Quota / sub limits ---" | Out-File $out -Append
$rs = $tenant
if ($rs) {
    "Personal tenant 'OneDrive' value (hex bytes): $rs" | Out-File $out -Append
}

"`n--- Defender exclusions (per Microsoft OneDrive recommendation) ---" | Out-File $out -Append
$prefs = Get-MpPreference -ErrorAction SilentlyContinue
if ($prefs) {
    "ExclusionPath count: $($prefs.ExclusionPath.Count)" | Out-File $out -Append
    $odExcl = $prefs.ExclusionPath | Where-Object { $_ -match 'OneDrive' }
    "OneDrive-related exclusions ($($odExcl.Count)):" | Out-File $out -Append
    $odExcl | ForEach-Object { "  $_" } | Out-File $out -Append
    "ExclusionProcess count: $($prefs.ExclusionProcess.Count)" | Out-File $out -Append
    $odProc = $prefs.ExclusionProcess | Where-Object { $_ -match 'OneDrive|FileSync|FileCoAuth' }
    "OneDrive-related process exclusions ($($odProc.Count)):" | Out-File $out -Append
    $odProc | ForEach-Object { "  $_" } | Out-File $out -Append
}

"`n--- Cloud Files filter mini-status ---" | Out-File $out -Append
fltmc filters 2>&1 | Where-Object { $_ -match 'CldFlt|cldflt|bindflt' } | Out-File $out -Append

"`n--- WAM Account Manager state (relevant for OneDrive Personal sign-in) ---" | Out-File $out -Append
$wam = Get-ItemProperty 'HKCU:\Software\Microsoft\IdentityCRL\AcceptedDomains' -ErrorAction SilentlyContinue
"WAM accepted domains (top 5):" | Out-File $out -Append
$wam.PSObject.Properties | Where-Object { $_.Name -notmatch '^PS' } | Select-Object -First 10 |
    ForEach-Object { "  $($_.Name) = $($_.Value)" } | Out-File $out -Append

Write-Host "Wrote $out"
