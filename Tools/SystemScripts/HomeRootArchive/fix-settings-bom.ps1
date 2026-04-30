# Fix Docker settings-store.json BOM issue
$settingsPath = "$env:APPDATA\Docker\settings-store.json"

Write-Host "Fixing settings file BOM issue..."

# Read as bytes
$bytes = [System.IO.File]::ReadAllBytes($settingsPath)
Write-Host "File size: $($bytes.Length) bytes"
Write-Host "First 3 bytes: $($bytes[0]), $($bytes[1]), $($bytes[2])"

# Check for UTF-8 BOM (EF BB BF)
$startIndex = 0
if ($bytes[0] -eq 0xEF -and $bytes[1] -eq 0xBB -and $bytes[2] -eq 0xBF) {
    Write-Host "Found UTF-8 BOM, removing..."
    $startIndex = 3
}

# Get content without BOM
$content = [System.Text.Encoding]::UTF8.GetString($bytes, $startIndex, $bytes.Length - $startIndex)

# Write back without BOM
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($settingsPath, $content, $utf8NoBom)

Write-Host "Settings file fixed!"

# Verify JSON is valid
try {
    $json = $content | ConvertFrom-Json
    Write-Host "JSON is valid!" -ForegroundColor Green
    Write-Host "CustomWslDistroDir: $($json.CustomWslDistroDir)"
} catch {
    Write-Host "JSON error: $_" -ForegroundColor Red
}
