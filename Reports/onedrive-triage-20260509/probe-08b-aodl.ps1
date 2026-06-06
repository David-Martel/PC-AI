$ErrorActionPreference = 'Continue'
$out = 'C:\codedev\PC_AI\Reports\onedrive-triage-20260509\08-syncengine-strings.txt'
"=== SyncEngine .aodl strings extraction at $(Get-Date -Format o) ===" | Out-File $out

$logRoot = "$env:LOCALAPPDATA\Microsoft\OneDrive\logs\Personal"
$active = Get-ChildItem "$logRoot\SyncEngine-*.aodl" -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (-not $active) { "No .aodl found" | Out-File $out -Append; exit 1 }
"Active log: $($active.FullName) (size=$($active.Length), mtime=$($active.LastWriteTime))" | Out-File $out -Append

# Open with FileShare.ReadWrite to bypass OneDrive's exclusive write lock
$fs = $null
try {
    $fs = [System.IO.File]::Open($active.FullName, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, [System.IO.FileShare]::ReadWrite)
    $ms = New-Object System.IO.MemoryStream
    $fs.CopyTo($ms)
    $bytes = $ms.ToArray()
} finally {
    if ($fs) { $fs.Dispose() }
}
"Read bytes: $($bytes.Length)" | Out-File $out -Append
if ($bytes.Length -lt 100) { "Too small to parse" | Out-File $out -Append; exit 1 }

# OneDrive .aodl format: each entry has UTF-16LE strings interleaved with binary
$ascii = [System.Text.Encoding]::ASCII.GetString($bytes)
$utf16 = [System.Text.Encoding]::Unicode.GetString($bytes)

$asciiRuns = [regex]::Matches($ascii, '[\x20-\x7E]{8,}') | ForEach-Object { $_.Value }
$utf16Runs = [regex]::Matches($utf16, '[\x20-\x7E]{8,}') | ForEach-Object { $_.Value }
"ASCII runs: $($asciiRuns.Count); UTF-16LE runs: $($utf16Runs.Count)" | Out-File $out -Append

$all = ($asciiRuns + $utf16Runs) | Sort-Object -Unique

# Save full string corpus for grep
$allFile = 'C:\codedev\PC_AI\Reports\onedrive-triage-20260509\08-aodl-strings-all.txt'
$all | Out-File $allFile

$patterns = @(
    'error','fail','exception','throttl','retry','timeout',
    'unauthorized','forbidden','denied','expired','invalid_grant','invalid_token',
    'http\d{3}','HTTP/','status','HRESULT','0x[0-9A-Fa-f]{8}',
    'login\.live','onedrive','docs\.live','microsoftpersonalcontent','sharepoint',
    'queue','pending','stalled',
    'auth','token','signed','SignIn','SignedOut','LogOut',
    'WAM','OneAuth','MSAL','reset','migrat',
    'upload','Upload','Sync','sync','scan','Scan'
)
$pattern = '(?i)(' + ($patterns -join '|') + ')'
$hits = $all | Where-Object { $_ -match $pattern }

"`n--- Filtered hits ($($hits.Count), max 300) ---" | Out-File $out -Append
$hits | Select-Object -First 300 | Out-File $out -Append

"`n--- HRESULT hex codes ---" | Out-File $out -Append
$all | Where-Object { $_ -match '0x[0-9A-Fa-f]{8}' } | Sort-Object -Unique | Out-File $out -Append

Write-Host "Wrote $out and $allFile (corpus)"
"Stats: total=$($all.Count) filtered=$($hits.Count)"
