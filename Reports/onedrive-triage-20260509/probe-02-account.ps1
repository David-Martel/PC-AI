$ErrorActionPreference = 'Continue'
$out = 'C:\codedev\PC_AI\Reports\onedrive-triage-20260509\02-account-settings.txt'
$settings = "$env:LOCALAPPDATA\Microsoft\OneDrive\settings\Personal"
"=== OneDrive Personal settings inventory at $(Get-Date -Format o) ===" | Out-File $out
"Settings dir: $settings" | Out-File $out -Append
Get-ChildItem $settings -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object Name, LastWriteTime, @{n='SizeKB';e={[math]::Round($_.Length/1KB,2)}} |
    Format-Table -AutoSize | Out-String | Out-File $out -Append

"`n=== ClientPolicy*.ini snippets ===" | Out-File $out -Append
Get-ChildItem "$settings\ClientPolicy*.ini" -ErrorAction SilentlyContinue | ForEach-Object {
    "--- $($_.Name) ($([math]::Round($_.Length/1KB,1)) KB) ---" | Out-File $out -Append
    Get-Content $_.FullName -TotalCount 80 -ErrorAction SilentlyContinue | Out-File $out -Append
    "" | Out-File $out -Append
}

"`n=== global.ini and personal.ini snippets ===" | Out-File $out -Append
Get-ChildItem $settings -Filter '*.ini' -ErrorAction SilentlyContinue | Where-Object { $_.Name -notmatch 'ClientPolicy' } | ForEach-Object {
    "--- $($_.Name) ---" | Out-File $out -Append
    Get-Content $_.FullName -ErrorAction SilentlyContinue |
        Where-Object { $_ -notmatch 'PrimaryEmail|email|Address|UserPrincipalName' } |
        Select-Object -First 80 |
        Out-File $out -Append
    "" | Out-File $out -Append
}

Write-Host "Wrote $out"
Get-Content $out -TotalCount 25
