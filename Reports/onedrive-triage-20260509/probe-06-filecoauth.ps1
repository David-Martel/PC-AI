$ErrorActionPreference = 'Continue'
$out = 'C:\codedev\PC_AI\Reports\onedrive-triage-20260509\06-filecoauth.txt'
"=== FileCoAuth diagnostic at $(Get-Date -Format o) ===" | Out-File $out

"`n--- OneDrive subdirs (versioned binary roots) ---" | Out-File $out -Append
$odRoot = "$env:LOCALAPPDATA\Microsoft\OneDrive"
Get-ChildItem $odRoot -Directory -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -match '^\d' } |
    Sort-Object LastWriteTime -Descending |
    Select-Object Name, LastWriteTime |
    Format-Table -AutoSize | Out-String | Out-File $out -Append

"`n--- Searching for FileCoAuth.exe ---" | Out-File $out -Append
$found = Get-ChildItem $odRoot -Filter 'FileCoAuth.exe' -Recurse -ErrorAction SilentlyContinue
foreach ($f in $found) {
    $v = $f.VersionInfo
    "$($f.FullName) | ver=$($v.FileVersion) | size=$([math]::Round($f.Length/1KB,1)) KB | written=$($f.LastWriteTime)" | Out-File $out -Append
}
"Found $($found.Count) FileCoAuth.exe instance(s)" | Out-File $out -Append

"`n--- Office ClickToRun install ---" | Out-File $out -Append
$c2r = Get-ItemProperty 'HKLM:\SOFTWARE\Microsoft\Office\ClickToRun\Configuration' -ErrorAction SilentlyContinue
if ($c2r) {
    $c2r | Select-Object ProductReleaseIds, VersionToReport, ClientCulture, UpdateChannel, InstallationPath, Platform |
        Format-List | Out-String | Out-File $out -Append
} else {
    'No ClickToRun key' | Out-File $out -Append
}

"`n--- Office app process snapshot ---" | Out-File $out -Append
Get-Process -Name WINWORD,EXCEL,POWERPNT,OUTLOOK,ONENOTE,TEAMS -ErrorAction SilentlyContinue |
    Select-Object Name, Id, StartTime, Responding |
    Format-Table -AutoSize | Out-String | Out-File $out -Append

"`n--- WinEvent Application log entries mentioning FileCoAuth (last 7 days) ---" | Out-File $out -Append
try {
    $events = Get-WinEvent -FilterHashtable @{LogName='Application'; StartTime=(Get-Date).AddDays(-7)} -ErrorAction Stop
    $hits = $events | Where-Object { $_.Message -match 'FileCoAuth' }
    "Total Application events 7d: $($events.Count); FileCoAuth-mentioning: $($hits.Count)" | Out-File $out -Append
    $hits | Select-Object -First 25 TimeCreated, Id, LevelDisplayName, ProviderName, @{n='Snippet';e={($_.Message -split "`n")[0]}} |
        Format-Table -AutoSize -Wrap | Out-String -Width 240 | Out-File $out -Append
} catch {
    "GetEvent: $($_.Exception.Message)" | Out-File $out -Append
}

"`n--- CIM process tree (FileCoAuth + likely parents) ---" | Out-File $out -Append
Get-CimInstance Win32_Process -Filter "Name='FileCoAuth.exe' OR Name='OneDrive.exe' OR Name='WINWORD.EXE' OR Name='EXCEL.EXE' OR Name='POWERPNT.EXE' OR Name='ONENOTE.EXE'" -ErrorAction SilentlyContinue |
    Select-Object Name, ProcessId, ParentProcessId, CreationDate |
    Format-Table -AutoSize | Out-String | Out-File $out -Append

Write-Host "Wrote $out"
Get-Content $out
