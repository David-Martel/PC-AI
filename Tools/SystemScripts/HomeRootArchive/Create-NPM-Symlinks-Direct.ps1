# Create NPM Symlinks - Direct Execution (Run as Administrator)
# Creates symlinks from .local\bin to npm global directory

$npmGlobalBin = "C:\Users\david\AppData\Roaming\npm"
$localBin = "C:\users\david\.local\bin"

Write-Host "=== Creating NPM Symlinks ===" -ForegroundColor Cyan
Write-Host ""

# Check if running as admin
$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "ERROR: This script must be run as Administrator to create symlinks" -ForegroundColor Red
    Write-Host "Please run: Start-Process powershell -ArgumentList '-ExecutionPolicy Bypass -File $PSCommandPath' -Verb RunAs" -ForegroundColor Yellow
    exit 1
}

$targets = @{
    "tsc.cmd" = "$npmGlobalBin\tsc.cmd"
    "tsserver.cmd" = "$npmGlobalBin\tsserver.cmd"
    "eslint.cmd" = "$npmGlobalBin\eslint.cmd"
    "prettier.cmd" = "$npmGlobalBin\prettier.cmd"
    "npx.cmd" = "$npmGlobalBin\npx.cmd"
}

foreach ($name in $targets.Keys) {
    $target = $targets[$name]
    $link = Join-Path $localBin $name

    if (-not (Test-Path $target)) {
        Write-Host "[SKIP] $name - target not found: $target" -ForegroundColor Gray
        continue
    }

    # Remove existing file/link
    if (Test-Path $link) {
        Remove-Item $link -Force
        Write-Host "[REMOVE] Removed existing $name" -ForegroundColor Yellow
    }

    # Create symlink
    try {
        New-Item -ItemType SymbolicLink -Path $link -Target $target -Force | Out-Null
        Write-Host "[CREATE] $name -> $target" -ForegroundColor Green
    } catch {
        Write-Host "[ERROR] Failed to create symlink for $name`: $_" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "=== Verification ===" -ForegroundColor Cyan
foreach ($name in $targets.Keys) {
    $link = Join-Path $localBin $name
    if (Test-Path $link) {
        $item = Get-Item $link
        if ($item.LinkType -eq 'SymbolicLink') {
            Write-Host "✓ $name (symlink to $($item.Target))" -ForegroundColor Green
        } else {
            Write-Host "✗ $name (not a symlink)" -ForegroundColor Red
        }
    }
}

Write-Host ""
Write-Host "Done! Test with: tsc --version, eslint --version, prettier --version" -ForegroundColor Cyan
