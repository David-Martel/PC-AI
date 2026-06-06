$ErrorActionPreference = 'Continue'
$out = 'C:\codedev\PC_AI\Reports\onedrive-triage-20260509\24-wpn-channels.txt'
"=== WPN channel inspection at $(Get-Date -Format o) ===" | Out-File $out

$wpnCopy = 'C:\codedev\PC_AI\Reports\onedrive-triage-20260509\wpndatabase.copy.db'

"`n--- All NotificationHandlers ---" | Out-File $out -Append
sqlite3.exe $wpnCopy ".headers on" ".mode column" "SELECT RecordId, PrimaryId, WNSId, HandlerType, datetime(CreatedTime/10000-11644473600,'unixepoch') AS CreatedAt, datetime(ModifiedTime/10000-11644473600,'unixepoch') AS ModifiedAt FROM NotificationHandler;" 2>&1 | Out-File $out -Append

"`n--- WNSPushChannel table (current registered channels) ---" | Out-File $out -Append
sqlite3.exe $wpnCopy ".mode column" "SELECT HandlerId, ChannelId, datetime(CreatedTime/10000-11644473600,'unixepoch') AS CreatedAt, datetime(ExpiryTime/10000-11644473600,'unixepoch') AS Expires, substr(Uri,1,80) AS UriShort FROM WNSPushChannel;" 2>&1 | Out-File $out -Append

"`n--- OneDrive-related entries ---" | Out-File $out -Append
sqlite3.exe $wpnCopy ".mode column" "SELECT RecordId, PrimaryId FROM NotificationHandler WHERE PrimaryId LIKE '%OneDrive%' OR PrimaryId LIKE '%FileSync%' OR PrimaryId LIKE '%Microsoft.SkyDrive%';" 2>&1 | Out-File $out -Append

"`n--- All channel HandlerIds with Handler details ---" | Out-File $out -Append
sqlite3.exe $wpnCopy ".mode column" "SELECT c.HandlerId, h.PrimaryId AS Handler, datetime(c.ExpiryTime/10000-11644473600,'unixepoch') AS Expires FROM WNSPushChannel c LEFT JOIN NotificationHandler h ON c.HandlerId=h.RecordId ORDER BY c.HandlerId;" 2>&1 | Out-File $out -Append

Write-Host "Wrote $out"
Get-Content $out
