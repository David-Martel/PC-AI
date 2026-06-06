$ErrorActionPreference = 'Continue'
$out = 'C:\codedev\PC_AI\Reports\onedrive-triage-20260509\14-svchistory.txt'
"=== od_ServiceOperationHistory + OCSI.db query at $(Get-Date -Format o) ===" | Out-File $out

$sqlite = (Get-Command sqlite3.exe -ErrorAction SilentlyContinue).Source
if (-not $sqlite) { 'sqlite3.exe not on PATH' | Out-File $out -Append; exit 1 }

# Use the SyncEngineDatabase.copy.db we already copied
$sdbCopy = 'C:\codedev\PC_AI\Reports\onedrive-triage-20260509\SyncEngineDatabase.copy.db'
"Source DB: $sdbCopy" | Out-File $out -Append

"`n--- od_ServiceOperationHistory schema ---" | Out-File $out -Append
& $sqlite $sdbCopy ".schema od_ServiceOperationHistory" 2>&1 | Out-File $out -Append

"`n--- od_ServiceOperationHistory total rows ---" | Out-File $out -Append
& $sqlite $sdbCopy "SELECT COUNT(*) FROM od_ServiceOperationHistory;" 2>&1 | Out-File $out -Append

"`n--- Most recent 50 service ops ---" | Out-File $out -Append
& $sqlite $sdbCopy ".headers on" ".mode column" ".width 25 8 30 60" "SELECT * FROM od_ServiceOperationHistory ORDER BY rowid DESC LIMIT 50;" 2>&1 | Out-File $out -Append

"`n--- Distinct status codes / values ---" | Out-File $out -Append
$cols = (& $sqlite $sdbCopy "PRAGMA table_info(od_ServiceOperationHistory);" 2>&1) -split "`n"
$cols | Out-File $out -Append

"`n--- Failed/non-200 ops (broad scan) ---" | Out-File $out -Append
& $sqlite $sdbCopy "SELECT * FROM od_ServiceOperationHistory WHERE rowid IN (SELECT rowid FROM od_ServiceOperationHistory ORDER BY rowid DESC LIMIT 500);" 2>&1 |
    Select-String -Pattern '(?i)(error|fail|throttl|4\d\d|5\d\d|0x80|HRESULT)' |
    Select-Object -First 50 |
    Out-File $out -Append

"`n--- od_ThrottleHistory ---" | Out-File $out -Append
& $sqlite $sdbCopy "SELECT * FROM od_ThrottleHistory ORDER BY rowid DESC LIMIT 20;" 2>&1 | Out-File $out -Append

"`n--- od_GraphMetadata_LastWrite ---" | Out-File $out -Append
& $sqlite $sdbCopy "SELECT * FROM od_GraphMetadata_LastWrite;" 2>&1 | Out-File $out -Append

"`n--- od_ScopeInfo_Records (the drive scope) ---" | Out-File $out -Append
& $sqlite $sdbCopy ".mode line" "SELECT * FROM od_ScopeInfo_Records;" 2>&1 | Out-File $out -Append

# OCSI.db
"`n=== OCSI.db ===" | Out-File $out -Append
$ocsi = "$env:LOCALAPPDATA\Microsoft\OneDrive\settings\Personal\OCSI.db"
$ocsiCopy = 'C:\codedev\PC_AI\Reports\onedrive-triage-20260509\OCSI.copy.db'
try {
    Copy-Item $ocsi $ocsiCopy -Force -ErrorAction Stop
    "Copied OCSI.db ($([math]::Round((Get-Item $ocsiCopy).Length/1MB,1)) MB)" | Out-File $out -Append
    "`n--- OCSI tables ---" | Out-File $out -Append
    & $sqlite $ocsiCopy ".tables" 2>&1 | Out-File $out -Append
    "`n--- OCSI row counts ---" | Out-File $out -Append
    $tables = (& $sqlite $ocsiCopy "SELECT name FROM sqlite_master WHERE type='table';" 2>&1) -split "`n"
    foreach ($t in $tables) {
        $t = $t.Trim(); if (-not $t) { continue }
        try {
            $count = (& $sqlite $ocsiCopy "SELECT COUNT(*) FROM `"$t`";" 2>&1).Trim()
            "  ${t}: $count" | Out-File $out -Append
        } catch {}
    }
    "`n--- OCSI schema (focused on problem-like tables) ---" | Out-File $out -Append
    & $sqlite $ocsiCopy ".schema" 2>&1 | Select-String -Pattern '(?i)(problem|conflict|fail|error|warning|stuck|pending|sync)' | Out-File $out -Append
} catch {
    "OCSI copy failed: $($_.Exception.Message)" | Out-File $out -Append
}

Write-Host "Wrote $out"
