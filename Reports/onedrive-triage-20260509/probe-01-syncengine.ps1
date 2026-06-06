$ErrorActionPreference = 'Continue'
$out = 'C:\codedev\PC_AI\Reports\onedrive-triage-20260509\01-syncengine-inventory.txt'
$logRoot = "$env:LOCALAPPDATA\Microsoft\OneDrive\logs\Personal"
"=== SyncEngine log inventory at $(Get-Date -Format o) ===" | Out-File $out
"Log root: $logRoot" | Out-File $out -Append
$logs = Get-ChildItem "$logRoot\SyncEngine-*.aodl" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending
"Total .aodl files: $($logs.Count)" | Out-File $out -Append
$logs | Select-Object -First 10 Name, LastWriteTime, @{n='SizeKB';e={[math]::Round($_.Length/1KB,1)}} |
    Format-Table -AutoSize | Out-String | Out-File $out -Append
"`n=== Newest 3 paths ===" | Out-File $out -Append
$logs | Select-Object -First 3 -ExpandProperty FullName | Out-File $out -Append

# Also list other log types in the dir
"`n=== Other log types in Personal\ ===" | Out-File $out -Append
Get-ChildItem $logRoot -ErrorAction SilentlyContinue | Group-Object {
    if ($_.Name -match '^([A-Za-z]+)') { $matches[1] } else { 'OTHER' }
} | Select-Object Count, Name | Format-Table -AutoSize | Out-String | Out-File $out -Append

Write-Host "Wrote $out"
Get-Content $out -TotalCount 30
