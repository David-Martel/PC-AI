$ErrorActionPreference = 'Continue'
$out = 'C:\codedev\PC_AI\Reports\onedrive-triage-20260509\13-odlsent-strings.txt'
"=== .odlsent gunzip + strings at $(Get-Date -Format o) ===" | Out-File $out

$logRoot = "$env:LOCALAPPDATA\Microsoft\OneDrive\logs\Personal"
$workDir = 'C:\codedev\PC_AI\Reports\onedrive-triage-20260509\odlsent-work'
$null = New-Item -ItemType Directory -Path $workDir -Force

# Newest 5 .odlsent files
$logs = Get-ChildItem "$logRoot\*.odlsent" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 5
"Processing $($logs.Count) .odlsent files" | Out-File $out -Append

$keystorePath = "$logRoot\general.keystore"
"general.keystore exists: $(Test-Path $keystorePath)" | Out-File $out -Append

# OneDrive .odlsent format: gzip-compressed binary record stream.
# We attempt gzip first, then fall back to raw strings if not standard gzip.
$allStrings = @()
foreach ($log in $logs) {
    "`n=== $($log.Name) ($([math]::Round($log.Length/1KB,1)) KB) ===" | Out-File $out -Append

    $bytes = $null
    try {
        $fs = [System.IO.File]::Open($log.FullName, 'Open', 'Read', 'ReadWrite')
        $ms = New-Object System.IO.MemoryStream
        $fs.CopyTo($ms)
        $fs.Dispose()
        $bytes = $ms.ToArray()
    } catch {
        "  Read failed: $($_.Exception.Message)" | Out-File $out -Append
        continue
    }

    # Detect gzip magic 1f 8b
    $isGzip = ($bytes.Length -ge 2 -and $bytes[0] -eq 0x1F -and $bytes[1] -eq 0x8B)
    "  Gzip magic: $isGzip ; first 16 bytes: $((($bytes[0..15] | ForEach-Object { '{0:x2}' -f $_ }) -join ' '))" | Out-File $out -Append

    $decompressed = $null
    if ($isGzip) {
        try {
            $compressedStream = New-Object System.IO.MemoryStream(,$bytes)
            $gz = New-Object System.IO.Compression.GZipStream($compressedStream, [System.IO.Compression.CompressionMode]::Decompress)
            $outMs = New-Object System.IO.MemoryStream
            $gz.CopyTo($outMs)
            $gz.Dispose(); $compressedStream.Dispose()
            $decompressed = $outMs.ToArray()
            "  Decompressed bytes: $($decompressed.Length)" | Out-File $out -Append
        } catch {
            "  Gzip decompress failed: $($_.Exception.Message)" | Out-File $out -Append
        }
    }

    $target = if ($decompressed) { $decompressed } else { $bytes }
    $ascii = [System.Text.Encoding]::ASCII.GetString($target)
    $utf16 = [System.Text.Encoding]::Unicode.GetString($target)
    $a = [regex]::Matches($ascii, '[\x20-\x7E]{8,}') | ForEach-Object { $_.Value }
    $u = [regex]::Matches($utf16, '[\x20-\x7E]{8,}') | ForEach-Object { $_.Value }
    "  ASCII strings: $($a.Count); UTF-16 strings: $($u.Count)" | Out-File $out -Append
    $allStrings += $a + $u
}

$unique = $allStrings | Sort-Object -Unique
"`nTotal unique strings: $($unique.Count)" | Out-File $out -Append

# Save the corpus for later grep
$unique | Out-File "$workDir\strings-all.txt"

# Filter to interesting patterns
$patterns = @(
    'error','exception','fail','throttl','retry','timeout',
    'unauthorized','forbidden','denied','expired','invalid_grant','invalid_token',
    'http\d{3}','status code','HRESULT','0x[0-9A-Fa-f]{8}',
    'login\.live','onedrive','docs\.live','microsoftpersonalcontent','sharepoint',
    'queue','pending','stalled',
    'token','signed','SignedOut','LogOut',
    'reset','migrat','upload','sync','converge','quota','InvalidItem',
    'AuthorizationFailed','AuthenticationFailed','RequestThrottled',
    'StatusCode','HResult','HRESULT 0x','Server returned'
)
$pattern = '(?i)(' + ($patterns -join '|') + ')'
$hits = $unique | Where-Object { $_ -match $pattern }
"Filtered hits: $($hits.Count)" | Out-File $out -Append

"`n--- Top 200 filtered hits ---" | Out-File $out -Append
$hits | Select-Object -First 200 | Out-File $out -Append

"`n--- HRESULT / hex error codes ---" | Out-File $out -Append
$unique | Where-Object { $_ -match '0x[0-9A-Fa-f]{8}' } | Sort-Object -Unique | Out-File $out -Append

"`n--- HTTP status mentions ---" | Out-File $out -Append
$unique | Where-Object { $_ -match '\b(4\d\d|5\d\d)\b' -and $_ -match '(?i)(status|http|response|code|error)' } | Sort-Object -Unique | Out-File $out -Append

Write-Host "Wrote $out and corpus at $workDir\strings-all.txt"
"Stats: total=$($unique.Count) hits=$($hits.Count)"
