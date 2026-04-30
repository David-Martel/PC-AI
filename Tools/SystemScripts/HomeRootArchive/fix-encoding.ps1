# Fix console encoding for WSL and Claude application
# This script sets UTF-8 encoding for proper text display

# Set console output encoding to UTF-8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::InputEncoding = [System.Text.Encoding]::UTF8

# Set PowerShell output encoding
$OutputEncoding = [System.Text.Encoding]::UTF8

# Set Windows code page to UTF-8
chcp 65001 | Out-Null

# Set environment variable for WSL
$env:WSL_UTF8 = "1"
$env:LANG = "en_US.UTF-8"

Write-Host "✅ Console encoding set to UTF-8" -ForegroundColor Green
Write-Host "✅ WSL output should now display correctly" -ForegroundColor Green