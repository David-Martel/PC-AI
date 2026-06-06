$ErrorActionPreference = 'Continue'
$out = 'C:\codedev\PC_AI\Reports\onedrive-triage-20260509\08-syncengine-strings.txt'
"=== SyncEngine .aodl strings extraction at $(Get-Date -Format o) ===" | Out-File $out

$logRoot = "$env:LOCALAPPDATA\Microsoft\OneDrive\logs\Personal"
$active = Get-ChildItem "$logRoot\SyncEngine-*.aodl" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (-not $active) {
    "No active .aodl found" | Out-File $out -Append
    exit 1
}
"Active log: $($active.FullName) ($([math]::Round($active.Length/1KB,1)) KB, mtime=$($active.LastWriteTime))" | Out-File $out -Append

# Read raw bytes; OneDrive .aodl format is binary with UTF-8 / UTF-16LE strings interleaved.
$bytes = [System.IO.File]::ReadAllBytes($active.FullName)
"Total bytes: $($bytes.Length)" | Out-File $out -Append

# Extract printable ASCII runs (>= 8 chars)
$ascii = [System.Text.Encoding]::ASCII.GetString($bytes)
$asciiRuns = [regex]::Matches($ascii, '[\x20-\x7E]{8,}') | ForEach-Object { $_.Value }
"ASCII runs >=8 chars: $($asciiRuns.Count)" | Out-File $out -Append

# Extract UTF-16LE string runs (>= 8 chars). UTF-16LE means every char has \x00 high byte for BMP latin.
$utf16 = [System.Text.Encoding]::Unicode.GetString($bytes)
$utf16Runs = [regex]::Matches($utf16, '[\x20-\x7E]{8,}') | ForEach-Object { $_.Value }
"UTF-16LE runs >=8 chars: $($utf16Runs.Count)" | Out-File $out -Append

$all = ($asciiRuns + $utf16Runs) | Sort-Object -Unique

# Filter to interesting patterns
$patterns = @(
    'error','Error','ERROR','fail','Fail','FAIL','exception','Exception',
    'throttle','Throttle','retry','Retry','timeout','Timeout',
    'unauthorized','forbidden','denied','expired','invalid_grant','invalid_token',
    'http\d{3}','HTTP/','status code','StatusCode','HRESULT','0x[0-9A-Fa-f]{8}',
    'login\.live','onedrive','docs\.live','microsoftpersonalcontent','sharepoint',
    'queue','Queue','pending','Pending','stalled','Stalled',
    'auth','Auth','token','Token','signed','SignIn','SignedOut','LogOut',
    'WAM','OneAuth','ADAL','MSAL'
)
$pattern = '(' + ($patterns -join '|') + ')'
$hits = $all | Where-Object { $_ -match $pattern }

"`n--- Filtered hits ($($hits.Count)) ---" | Out-File $out -Append
$hits | Select-Object -First 200 | Out-File $out -Append

"`n--- All HRESULT-looking 0x... codes ---" | Out-File $out -Append
$hex = $all | Where-Object { $_ -match '0x[0-9A-Fa-f]{8}' } | Sort-Object -Unique
$hex | Out-File $out -Append

"`n--- All log-level lines (containing words like ERROR/WARN/INFO and a word) ---" | Out-File $out -Append
$logLines = $all | Where-Object { $_ -match '(ERROR|WARNING|FATAL)' -and $_ -match '[a-zA-Z]{3,}' }
$logLines | Select-Object -First 100 | Out-File $out -Append

Write-Host "Wrote $out"
"Hit counts: ascii=$($asciiRuns.Count) utf16=$($utf16Runs.Count) filtered=$($hits.Count) hresult=$($hex.Count) loglines=$($logLines.Count)"
