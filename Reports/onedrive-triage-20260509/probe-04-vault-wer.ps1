$ErrorActionPreference = 'Continue'
$out = 'C:\codedev\PC_AI\Reports\onedrive-triage-20260509\04-vault-and-wer.txt'
"=== Vault state + WER recheck at $(Get-Date -Format o) ===" | Out-File $out

"`n--- Personal Vault registry state ---" | Out-File $out -Append
$pv = Get-ItemProperty -Path 'HKCU:\Software\Microsoft\OneDrive\Accounts\Personal\PersonalVault' -ErrorAction SilentlyContinue
if ($pv) {
    $pv | Select-Object * -ExcludeProperty PS* | Format-List | Out-String | Out-File $out -Append
} else {
    'No PersonalVault key' | Out-File $out -Append
}

"`n--- OneDrive Account Personal registry root (selected keys) ---" | Out-File $out -Append
$acct = Get-ItemProperty -Path 'HKCU:\Software\Microsoft\OneDrive\Accounts\Personal' -ErrorAction SilentlyContinue
if ($acct) {
    $acct | Select-Object UserFolder, UserEmail, cid, ServiceEndpointUri, SPOResourceId, OneAuthAccountId, ConfiguredTenantId, LastSignInTime, IsLoggedIn, LastSyncTimeStamp, ClientFirstSignInTimestamp, FilesOnDemand |
        Format-List | Out-String | Out-File $out -Append
}

"`n--- OneDrive process snapshot ---" | Out-File $out -Append
Get-Process -Name OneDrive,FileSyncHelper,FileCoAuth,Microsoft.SharePoint -ErrorAction SilentlyContinue |
    Select-Object Name, Id, StartTime, Responding, @{n='Mem(MB)';e={[math]::Round($_.WorkingSet64/1MB,1)}}, @{n='Threads';e={$_.Threads.Count}} |
    Format-Table -AutoSize | Out-String | Out-File $out -Append

"`n--- WER ReportQueue (OneDrive entries) ---" | Out-File $out -Append
$werQ = Get-ChildItem 'C:\ProgramData\Microsoft\Windows\WER\ReportQueue' -Directory -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -match 'OneDrive|FileSync|FileCoAuth' }
"ReportQueue count: $($werQ.Count)" | Out-File $out -Append
$werQ | Sort-Object LastWriteTime -Descending | Select-Object -First 10 Name, LastWriteTime |
    Format-Table -AutoSize | Out-String | Out-File $out -Append

$werA = Get-ChildItem 'C:\ProgramData\Microsoft\Windows\WER\ReportArchive' -Directory -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -match 'OneDrive|FileSync|FileCoAuth' } |
    Sort-Object LastWriteTime -Descending
"`nReportArchive count: $($werA.Count)" | Out-File $out -Append
$werA | Select-Object -First 10 Name, LastWriteTime | Format-Table -AutoSize | Out-String | Out-File $out -Append

"`n--- Newest WER Report.wer (if any) ---" | Out-File $out -Append
$newest = ($werQ + $werA) | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($newest) {
    "Newest: $($newest.FullName)" | Out-File $out -Append
    $reportFile = Get-ChildItem $newest.FullName -Filter 'Report.wer' -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($reportFile) {
        Get-Content $reportFile.FullName -TotalCount 50 -ErrorAction SilentlyContinue | Out-File $out -Append
    }
}

"`n--- Application log Errors (last 24h, filtered to OneDrive ecosystem) ---" | Out-File $out -Append
try {
    $events = Get-WinEvent -FilterHashtable @{LogName='Application'; Level=2; StartTime=(Get-Date).AddHours(-24)} -MaxEvents 200 -ErrorAction Stop
    $hits = $events | Where-Object { $_.Message -match 'OneDrive|FileSync|FileCoAuth|Microsoft\.SharePoint' }
    "Total Level=2 events 24h: $($events.Count); OneDrive-related: $($hits.Count)" | Out-File $out -Append
    $hits | Select-Object -First 20 TimeCreated, Id, ProviderName, @{n='Snippet';e={($_.Message -split "`n")[0]}} |
        Format-Table -AutoSize -Wrap | Out-String -Width 240 | Out-File $out -Append
} catch {
    "GetEvent: $($_.Exception.Message)" | Out-File $out -Append
}

Write-Host "Wrote $out"
Get-Content $out
