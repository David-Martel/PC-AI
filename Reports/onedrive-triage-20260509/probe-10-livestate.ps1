$ErrorActionPreference = 'Continue'
$out = 'C:\codedev\PC_AI\Reports\onedrive-triage-20260509\10-live-state.txt'
"=== Live OneDrive state at $(Get-Date -Format o) ===" | Out-File $out

"`n--- Get-OneDriveSyncSettings (if module exposes it) ---" | Out-File $out -Append
# Try OneDrive PowerShell module if present
$module = Get-Module -ListAvailable -Name OneDrive -ErrorAction SilentlyContinue
"OneDrive PS module: $(if ($module) { 'present' } else { 'NOT installed' })" | Out-File $out -Append

"`n--- COM SyncEngine query (Microsoft.OneDriveCloud.Personal) ---" | Out-File $out -Append
try {
    # This invokes the OneDrive COM API used by File Explorer to query sync state for a path
    $shell = New-Object -ComObject Shell.Application -ErrorAction Stop
    $folder = $shell.Namespace("$env:USERPROFILE\OneDrive")
    if ($folder) {
        # Detail index 303 is "Sync status" in modern Windows
        $item = $folder.Self
        $statusIndex = 303
        $status = $folder.GetDetailsOf($item, $statusIndex)
        "Sync status of OneDrive folder (col $statusIndex): $status" | Out-File $out -Append
        # Try other indexes
        for ($i = 280; $i -lt 320; $i++) {
            $name = $folder.GetDetailsOf($null, $i)
            if ($name) { "  col${i}=$name" | Out-File $out -Append }
        }
    } else {
        "Could not open OneDrive folder via shell" | Out-File $out -Append
    }
} catch {
    "Shell COM error: $($_.Exception.Message)" | Out-File $out -Append
}

"`n--- SettingsDatabase.db schema (if accessible) ---" | Out-File $out -Append
$db = "$env:LOCALAPPDATA\Microsoft\OneDrive\settings\Personal\SettingsDatabase.db"
$dbCopy = "C:\codedev\PC_AI\Reports\onedrive-triage-20260509\SettingsDatabase.copy.db"
try {
    Copy-Item $db $dbCopy -Force -ErrorAction Stop
    "Copied DB to $dbCopy ($([math]::Round((Get-Item $dbCopy).Length/1KB,1)) KB)" | Out-File $out -Append
    # Try sqlite3 if on PATH
    $sqlite = Get-Command sqlite3.exe -ErrorAction SilentlyContinue
    if ($sqlite) {
        "Using sqlite3: $($sqlite.Source)" | Out-File $out -Append
        $cmd = $sqlite.Source
        "`n--- Tables ---" | Out-File $out -Append
        & $cmd $dbCopy '.tables' 2>&1 | Out-File $out -Append
        "`n--- Schema ---" | Out-File $out -Append
        & $cmd $dbCopy '.schema' 2>&1 | Out-File $out -Append
        "`n--- Row counts ---" | Out-File $out -Append
        $tables = & $cmd $dbCopy "SELECT name FROM sqlite_master WHERE type='table';" 2>&1
        foreach ($t in $tables) {
            try {
                $count = & $cmd $dbCopy "SELECT COUNT(*) FROM `"$t`";" 2>&1
                "  ${t}: $count" | Out-File $out -Append
            } catch {}
        }
        "`n--- key_value_table (top 200 rows) ---" | Out-File $out -Append
        & $cmd $dbCopy "SELECT * FROM key_value_table LIMIT 200;" 2>&1 | Out-File $out -Append
    } else {
        "sqlite3.exe not on PATH; copied DB only" | Out-File $out -Append
    }
} catch {
    "DB copy/parse failed: $($_.Exception.Message)" | Out-File $out -Append
}

"`n--- SyncEngineDatabase.db schema ---" | Out-File $out -Append
$sdb = "$env:LOCALAPPDATA\Microsoft\OneDrive\settings\Personal\SyncEngineDatabase.db"
$sdbCopy = "C:\codedev\PC_AI\Reports\onedrive-triage-20260509\SyncEngineDatabase.copy.db"
try {
    Copy-Item $sdb $sdbCopy -Force -ErrorAction Stop
    "Copied SyncEngine DB to $sdbCopy ($([math]::Round((Get-Item $sdbCopy).Length/1MB,1)) MB)" | Out-File $out -Append
    $sqlite = Get-Command sqlite3.exe -ErrorAction SilentlyContinue
    if ($sqlite) {
        $cmd = $sqlite.Source
        "`n--- Tables ---" | Out-File $out -Append
        & $cmd $sdbCopy '.tables' 2>&1 | Out-File $out -Append
        "`n--- Row counts ---" | Out-File $out -Append
        $tables = (& $cmd $sdbCopy "SELECT name FROM sqlite_master WHERE type='table';" 2>&1) -split "`n"
        foreach ($t in $tables) {
            $t = $t.Trim(); if (-not $t) { continue }
            try {
                $count = (& $cmd $sdbCopy "SELECT COUNT(*) FROM `"$t`";" 2>&1).Trim()
                "  ${t}: $count" | Out-File $out -Append
            } catch {}
        }
    }
} catch {
    "SyncEngine DB copy failed: $($_.Exception.Message)" | Out-File $out -Append
}

"`n--- HKCU Tenants subkey contents ---" | Out-File $out -Append
Get-ChildItem 'HKCU:\Software\Microsoft\OneDrive\Accounts\Personal\Tenants' -ErrorAction SilentlyContinue | ForEach-Object {
    $tn = Get-ItemProperty $_.PSPath
    "Tenant key: $($_.PSChildName)" | Out-File $out -Append
    $tn | Select-Object * -ExcludeProperty PS* | Format-List | Out-String | Out-File $out -Append
}

"`n--- HKCU AuthenticationURLs ---" | Out-File $out -Append
Get-ItemProperty 'HKCU:\Software\Microsoft\OneDrive\Accounts\Personal\AuthenticationURLs' -ErrorAction SilentlyContinue |
    Select-Object * -ExcludeProperty PS* | Format-List | Out-String | Out-File $out -Append

"`n--- Cloud Files filter ETL ---" | Out-File $out -Append
$cldflt = 'C:\WINDOWS\system32\LogFiles\CloudFiles\CldFlt2.etl'
if (Test-Path $cldflt) {
    $f = Get-Item $cldflt
    "ETL: $cldflt size=$([math]::Round($f.Length/1KB,1)) KB mtime=$($f.LastWriteTime)" | Out-File $out -Append
}

"`n--- ODL/ODLgz log file timestamps (last 10 sent + active) ---" | Out-File $out -Append
$logRoot = "$env:LOCALAPPDATA\Microsoft\OneDrive\logs\Personal"
"Newest 10 .odlsent:" | Out-File $out -Append
Get-ChildItem "$logRoot\*.odlsent" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending |
    Select-Object -First 10 Name, LastWriteTime, @{n='SizeKB';e={[math]::Round($_.Length/1KB,1)}} |
    Format-Table -AutoSize | Out-String | Out-File $out -Append
"Newest 5 .odlgz:" | Out-File $out -Append
Get-ChildItem "$logRoot\*.odlgz" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending |
    Select-Object -First 5 Name, LastWriteTime, @{n='SizeKB';e={[math]::Round($_.Length/1KB,1)}} |
    Format-Table -AutoSize | Out-String | Out-File $out -Append

Write-Host "Wrote $out"
